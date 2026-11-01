"""Full V1 tokenwise relay with real CPU AMX prefill and lazy GPU loading.

Run --role cpu with CPU Python, then --role gpu with GPU Python and the
same fresh --folder. CPU retains weight/KV ownership until folder/stop exists.
This tests the handoff; its sequential staging is not a cold-start benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['VLLM_USE_V1'] = '1'
PROMPT = [151643, 9707, 11, 358, 1097]


def install_tokenwise_kv_audit(worker):
    """Compare actual imported GPU KV bits with the live CPU AMX SHM shards."""
    import re
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1 import shm_connector as module
    from vllm.distributed import get_tensor_model_parallel_rank
    runner = worker.model_runner
    config = runner.vllm_config
    cpu_tp = runner.parallel_config.shm_tp_size
    gpu_tp = runner.parallel_config.tensor_parallel_size
    rank = get_tensor_model_parallel_rank()
    heads = runner.model_config.get_total_num_kv_heads()
    assert heads % cpu_tp == heads % gpu_tp == 0
    cpu_heads, gpu_heads = heads // cpu_tp, heads // gpu_tp
    assert max(cpu_tp, gpu_tp) % min(cpu_tp, gpu_tp) == 0
    source_ranks = (range(rank * (cpu_tp // gpu_tp), (rank + 1) * (cpu_tp // gpu_tp))
                    if gpu_tp < cpu_tp else [rank // (gpu_tp // cpu_tp)])
    layers = runner.model_config.hf_text_config.num_hidden_layers
    blocks = config.load_config.shm_num_blocks
    tokens = config.cache_config.block_size
    dims = runner.model_config.get_head_size()
    from pathlib import Path
    name = Path(runner.model_config.model).name
    sources = [torch.from_file(f'/dev/shm/{name}_kv_cache_rank_{source_rank}',
        shared=False, size=layers * 2 * blocks * cpu_heads * tokens * dims,
        dtype=torch.int16).reshape(layers, 2, blocks, cpu_heads, -1)
        for source_rank in source_ranks]
    t, d = torch.arange(tokens)[:, None], torch.arange(dims)[None, :]
    indices = (((t // 16 * 16 * (dims // 2) + t % 16 + d // 2 * 16) * 2 + d % 2).flatten(),
               (t // 2 * 32 + t % 2 + d // 16 * tokens * 16 + d % 16 * 2).flatten())
    pending = []
    worker._tokenwise_kv_audit_elements = 0
    native_load = module.load_kv_cache_to_gpu
    native_sync = module.load_kv_cache_to_gpu_sync
    def record(gpu_block_ids, cpu_block_ids, layer_ids):
        pending.append((list(gpu_block_ids), list(cpu_block_ids), list(layer_ids)))
        return native_load(gpu_block_ids, cpu_block_ids, layer_ids)
    def audit():
        native_sync()
        checked = 0
        for local, remote, layer_ids in pending:
            for layer in layer_ids:
                matching = [entry for key, entry in config.compilation_config.static_forward_context.items()
                            if re.match(rf'model\.layers\.{layer}\.self_attn\.attn$', key)]
                assert len(matching) == 1, (layer, len(matching))
                cache = matching[0].kv_cache[0]
                offset = (rank * gpu_heads) % cpu_heads
                for kind in range(2):
                    packed = torch.cat([source[layer, kind, remote] for source in sources], dim=1)
                    expected = packed[:, offset:offset + gpu_heads][..., indices[kind]].reshape(
                        len(remote), gpu_heads, tokens, dims)
                    actual = cache[kind, local].transpose(1, 2).cpu().view(torch.int16)
                    assert torch.equal(actual, expected), ('Tokenwise KV import mismatch', rank, layer, kind)
                    checked += expected.numel()
        pending.clear()
        if checked:
            worker._tokenwise_kv_audit_elements += checked
            print('TOKENWISE_KV_AUDIT ' + str(dict(rank=rank, exact_elements=checked)), flush=True)
    module.load_kv_cache_to_gpu = record
    module.load_kv_cache_to_gpu_sync = audit


def tokenwise_audit_count(worker):
    return worker._tokenwise_kv_audit_elements


def produce(args):
    from vllm import LLM, SamplingParams
    args.folder.mkdir(parents=True, exist_ok=True)
    for path in args.checkpoint.iterdir():
        destination = args.folder / path.name
        if path.is_file() and path.suffix in ('.json', '.safetensors'):
            if not destination.exists():
                destination.symlink_to(path.resolve())
    model = LLM(model=str(args.folder), tensor_parallel_size=args.cpu_tp,
                dtype='bfloat16', load_format='shm', enforce_eager=True,
                max_model_len=max(256, len(PROMPT) + args.max_tokens), max_num_seqs=2,
                max_num_batched_tokens=args.chunk_size, enable_chunked_prefill=True,
                block_size=32, skip_tokenizer_init=True,
                enable_prefix_caching=False,
                kv_transfer_config={'kv_connector': 'ShmConnector',
                                    'kv_role': 'kv_producer'})
    try:
        start = time.perf_counter()
        if args.overlap:
            info = model.llm_engine.collective_rpc('get_shm_kv_cache_info')[0]
            (args.folder / 'cpu-ready.json').write_text(json.dumps(info))
            while not (args.folder / 'start-relay').exists():
                time.sleep(0.001)
            start = time.perf_counter()
        cpu_prompt = PROMPT[:args.cpu_prefix_tokens] if args.cpu_prefix_tokens else PROMPT
        output = model.generate(
            [{'prompt_token_ids': cpu_prompt}],
            SamplingParams(temperature=0, max_tokens=1, extra_args={
                'kv_transfer_params': {'do_remote_decode': True}}),
            use_tqdm=False)[0]
        info = model.llm_engine.collective_rpc('get_shm_kv_cache_info')[0]
        result = dict(cpu_seconds=time.perf_counter() - start,
                      cpu_start=start, cpu_end=time.perf_counter(),
                      tokens=([] if len(cpu_prompt) < len(PROMPT)
                              else list(output.outputs[0].token_ids)),
                      kv_transfer_params=output.kv_transfer_params,
                      shm_info=info)
        assert result['kv_transfer_params']['num_computed_tokens'] == len(cpu_prompt)
        pending = args.folder / 'handoff.json.tmp'
        pending.write_text(json.dumps(result))
        pending.replace(args.folder / 'handoff.json')
        print('RELAY_CPU ' + json.dumps(result), flush=True)
        while not (args.folder / 'stop').exists():
            time.sleep(0.1)
    finally:
        model.llm_engine.engine_core.shutdown()


def consume(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    if args.overlap:
        info = json.loads((args.folder / 'cpu-ready.json').read_text())
    else:
        data = json.loads((args.folder / 'handoff.json').read_text())
        info = data['shm_info']
    size, blocks, block_len, cpu_tp = info
    layers = AutoConfig.from_pretrained(args.folder).num_hidden_layers
    start = time.perf_counter()
    model = LLM(model=str(args.folder), tensor_parallel_size=args.gpu_tp,
                shm_tp_size=cpu_tp, load_format='shm', lazy_load=True,
                load_async=True, load_method='tokenwise', dtype='bfloat16',
                enforce_eager=True, max_model_len=max(256, len(PROMPT) + args.max_tokens),
                max_num_seqs=2, max_num_batched_tokens=args.chunk_size,
                enable_chunked_prefill=True, block_size=32,
                gpu_memory_utilization=0.5, skip_tokenizer_init=True,
                enable_prefix_caching=False, shm_kv_cache_size=size,
                shm_num_blocks=blocks, shm_block_len=block_len,
                kv_transfer_config={'kv_connector': 'ShmConnector',
                                    'kv_role': 'kv_consumer'})
    warm = time.perf_counter()
    try:
        lazy_start = time.perf_counter()
        if args.overlap:
            (args.folder / 'start-relay').touch()
        model.llm_engine.lazy_init(layer_idxes_by_pp=[list(range(layers))],
                                   request_id='relay-test')
        ready = time.perf_counter()
        if args.overlap:
            while not (args.folder / 'handoff.json').exists():
                time.sleep(0.001)
            data = json.loads((args.folder / 'handoff.json').read_text())
        engine = model.llm_engine
        if args.audit_kv:
            engine.collective_rpc(install_tokenwise_kv_audit)
        if args.exercise_gate:
            # Let an ordinary request arrive before the designated relay.
            # The engine must keep accepting input while it waits for relay.
            engine.add_request('reference-test', {'prompt_token_ids': PROMPT},
                               SamplingParams(temperature=0, max_tokens=args.max_tokens))
            time.sleep(0.1)
        engine.add_request(
            'relay-test', {'prompt_token_ids': PROMPT + data['tokens']},
            SamplingParams(temperature=0, max_tokens=args.max_tokens - len(data['tokens']), extra_args={
                'kv_transfer_params': data['kv_transfer_params']}))
        finished = {}
        elapsed = None
        while engine.has_unfinished_requests():
            for item in engine.step():
                if item.finished:
                    finished[item.request_id] = item
                    if item.request_id == 'relay-test':
                        elapsed = time.perf_counter() - ready
        output = finished['relay-test']
        tokens = data['tokens'] + list(output.outputs[0].token_ids)
        if args.exercise_gate:
            reference = finished['reference-test']
        else:
            # A request without transfer recomputes the entire prompt.
            reference = model.generate(
                [{'prompt_token_ids': PROMPT}],
                SamplingParams(temperature=0, max_tokens=args.max_tokens), use_tqdm=False)[0]
        expected = list(reference.outputs[0].token_ids)
        result = dict(gpu_tp=args.gpu_tp, prewarm_seconds=warm - start,
                      lazy_init_seconds=ready - lazy_start,
                      lazy_start=lazy_start, lazy_end=ready,
                      cpu_start=data['cpu_start'], cpu_end=data['cpu_end'],
                      amx_lazy_overlap_seconds=max(0, min(ready, data['cpu_end'])
                          - max(lazy_start, data['cpu_start'])),
                      request_seconds=elapsed,
                      tokens=tokens, reference_tokens=expected,
                      num_cached_tokens=output.num_cached_tokens)
        if args.audit_kv:
            counts = engine.collective_rpc(tokenwise_audit_count)
            assert all(count > 0 for count in counts), counts
            result['kv_audit_exact_elements_by_rank'] = counts
        print('RELAY_HANDOFF ' + json.dumps(result), flush=True)
        assert tokens == expected, result
        assert output.num_cached_tokens == data['kv_transfer_params']['num_computed_tokens'], result
    finally:
        (args.folder / 'stop').touch()
        model.llm_engine.engine_core.shutdown()


def main():
    global PROMPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cpu-tp', type=int, default=2)
    parser.add_argument('--gpu-tp', type=int, default=2)
    parser.add_argument('--exercise-gate', action='store_true')
    parser.add_argument('--prompt-tokens', type=int, default=5)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=4)
    parser.add_argument('--audit-kv', action='store_true')
    parser.add_argument('--prompt-token-file', type=Path)
    parser.add_argument('--cpu-prefix-tokens', type=int, default=0,
                        help='Compute only this prompt prefix on CPU; discard its sampled token')
    parser.add_argument('--overlap', action='store_true',
                        help='Start CPU prefill concurrently with GPU lazy_init')
    args = parser.parse_args()
    if args.audit_kv:
        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    if args.max_tokens < 1:
        parser.error('--max-tokens must be positive')
    PROMPT = ([151643] + [9707, 11, 358, 1097]
              * ((args.prompt_tokens + 3) // 4))[:args.prompt_tokens]
    if args.prompt_token_file:
        PROMPT = json.loads(args.prompt_token_file.read_text())
        if (not isinstance(PROMPT, list) or not PROMPT
                or any(type(token) is not int or token < 0 for token in PROMPT)):
            parser.error('--prompt-token-file must contain a nonempty list of token IDs')
    if not 0 <= args.cpu_prefix_tokens <= len(PROMPT):
        parser.error('cpu-prefix-tokens must be within the prompt length')
    args.folder = args.folder.resolve()
    if args.role == 'cpu':
        if args.checkpoint is None:
            parser.error('--checkpoint is required for the CPU producer')
        produce(args)
    else:
        consume(args)


if __name__ == '__main__':
    main()
