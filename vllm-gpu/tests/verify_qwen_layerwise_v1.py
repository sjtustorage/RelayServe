"""Live V1 layerwise relay; run CPU and GPU roles in their own environments.

Use a fresh folder and an external timeout. The CPU keeps SHM alive until stop.
GPU compares relay tokens against an independent, complete GPU prefill.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['VLLM_USE_V1'] = '1'


def install_cpu_norm_experiment(worker, library):
    """Explicit diagnostic override; never installed by production engines."""
    import torch
    torch.ops.load_library(library)
    registration = torch.library.Library('_C', 'IMPL', 'CPU')
    registration.impl('rms_norm', torch.ops.relay_norm_experiment.wide.default)
    registration.impl('fused_add_rms_norm',
                      torch.ops.relay_norm_experiment.wide_fused.default)
    # Registrations are removed when their Library object is destroyed.
    worker._relay_norm_experiment_registration = registration


def install_kv_audit(worker, boundary_layer=2, boundary_request='second-relay'):
    """Test-only audit immediately after native import, before model forward."""
    import re
    import torch
    from vllm.v1.worker import gpu_model_runner as module
    from vllm.distributed import get_tensor_model_parallel_rank

    runner = worker.model_runner
    pending = []
    native_load = module.load_kv_cache_to_gpu
    def record_load(local, remote, layers):
        pending.append((list(local), list(remote), list(layers)))
        return native_load(local, remote, layers)
    module.load_kv_cache_to_gpu = record_load
    receive = runner.recv_shm_kv_cache
    rank = get_tensor_model_parallel_rank()
    gpu_tp = runner.parallel_config.tensor_parallel_size
    cpu_tp = runner.parallel_config.shm_tp_size
    assert max(gpu_tp, cpu_tp) % min(gpu_tp, cpu_tp) == 0
    heads = runner.model_config.get_total_num_kv_heads()
    cpu_heads, gpu_heads = heads // cpu_tp, heads // gpu_tp
    config = runner.vllm_config
    blocks = config.load_config.shm_num_blocks
    tokens = config.cache_config.block_size
    dims = runner.model_config.get_head_size()
    # SHM contains the full CPU model, independently of the GPU PP shard.
    layers_count = runner.model_config.hf_text_config.num_hidden_layers
    name = Path(runner.model_config.model).name
    if gpu_tp < cpu_tp:
        ratio = cpu_tp // gpu_tp
        source_ranks = range(rank * ratio, (rank + 1) * ratio)
    else:
        source_ranks = [rank // (gpu_tp // cpu_tp)]
    sources = [torch.from_file(
        f'/dev/shm/{name}_kv_cache_rank_{source_rank}', shared=False,
        size=layers_count * 2 * blocks * cpu_heads * tokens * dims,
        dtype=torch.int16).reshape(layers_count, 2, blocks, cpu_heads, -1)
        for source_rank in source_ranks]
    t = torch.arange(tokens)[:, None]
    d = torch.arange(dims)[None, :]
    key_indices = ((t // 16 * 16 * (dims // 2) + t % 16
                    + d // 2 * 16) * 2 + d % 2).flatten()
    value_indices = (t // 2 * 32 + t % 2 + d // 16 * tokens * 16
                     + d % 16 * 2).flatten()
    offset = (rank * gpu_heads) % cpu_heads
    cpu_boundary_state = {}
    boundary_prompt = []
    def audited_receive(receive=receive):
        pending.clear()
        receive()
        checked = 0
        for local, remote, layer_ids in pending:
            for layer_id in layer_ids:
                matching = [entry for layer_name, entry in
                    config.compilation_config.static_forward_context.items()
                    if re.match(rf'model\.layers\.{layer_id}\.self_attn\.attn$',
                                layer_name)]
                assert len(matching) == 1, (layer_id, len(matching))
                cache = matching[0].kv_cache[0]
                for kind, indices in enumerate((key_indices, value_indices)):
                    # Read each live shard now: do not snapshot SHM before
                    # CPU writes or accidentally audit only the first rank.
                    packed = torch.cat([source[layer_id, kind, remote]
                                        for source in sources], dim=1)
                    expected = packed[:, offset:offset + gpu_heads][..., indices].reshape(
                            len(remote), gpu_heads, tokens, dims)
                    actual = cache[kind, local].transpose(1, 2).cpu().view(
                        torch.int16)
                    assert torch.equal(actual, expected), (
                        'KV import mismatch', rank, layer_id, kind, local, remote)
                    checked += expected.numel()
        if checked:
            print(json.dumps(dict(kv_audit_rank=rank, exact_elements=checked,
                                  step=runner.step_cnt)), flush=True)
    runner.recv_shm_kv_cache = audited_receive
    finish = runner.finish_layerwise_kv_handoff
    # A fast background loader can transfer the entire CPU prefix through
    # final handoff, without ever contracting it in recv_shm_kv_cache.
    # Audit that path too; otherwise a successful hidden audit can conceal
    # that no actual KV import was inspected.
    runner.finish_layerwise_kv_handoff = lambda: audited_receive(finish)
    receive_hidden = runner.recv_shm_hidden_states
    hidden_source = torch.from_file(
        f'/dev/shm/{name}_hidden_states', shared=False,
        size=2 * runner.max_num_tokens * runner.hidden_size,
        dtype=torch.int16).reshape(2, runner.max_num_tokens, runner.hidden_size)
    def audited_hidden(num_input_tokens, scheduler_output):
        receive_hidden(num_input_tokens, scheduler_output)
        metadata = runner.shm_scheduler_output
        if metadata is None:
            return
        # Build the expected row mapping from request counts directly, rather
        # than reusing hidden_state_spans from the implementation under test.
        source_offset = 0
        source_rows = {}
        for request_id, count in metadata['req_to_num_scheduled_tokens'].items():
            source_rows[request_id] = (source_offset, count)
            source_offset += count
        destination_offset = 0
        for request_id in runner.input_batch.req_ids:
            count = scheduler_output.num_scheduled_tokens[request_id]
            start, remote_count = source_rows[request_id]
            assert count == remote_count
            for kind, target in enumerate((runner.hidden_states, runner.residual)):
                actual = target[destination_offset:destination_offset + count].cpu().view(
                    torch.int16)
                expected = hidden_source[kind, start:start + count]
                assert torch.equal(actual, expected), (
                    'Hidden import mismatch', rank, kind, request_id, runner.step_cnt)
                if (request_id == boundary_request and count > 1
                        and metadata['shm_computed_layers'] == boundary_layer):
                    cpu_boundary_state[kind] = expected.clone().view(torch.bfloat16)
                    boundary_prompt[:] = runner.input_ids.cpu[
                        destination_offset:destination_offset + count].tolist()
            destination_offset += count
        assert destination_offset == scheduler_output.total_num_scheduled_tokens
        print(json.dumps(dict(hidden_audit_rank=rank,
                              exact_elements=2 * destination_offset * runner.hidden_size,
                              step=runner.step_cnt)), flush=True)
    runner.recv_shm_hidden_states = audited_hidden
    def compare_boundary(layer, positional, keyword):
        if runner.shm_scheduler_output is not None or len(cpu_boundary_state) != 2:
            return
        # Match the independent GPU prefill by token IDs and positions;
        # never compare a decode step or another request with the snapshot.
        positions = keyword.get('positions', positional[0] if positional else None)
        count = len(boundary_prompt)
        if positions is None or positions.numel() != count:
            return
        if not torch.equal(positions.cpu(), torch.arange(count)):
            return
        if runner.input_ids.cpu[:count].tolist() != boundary_prompt:
            return
        hidden = keyword.get('hidden_states', positional[1] if len(positional) > 1 else None)
        residual = keyword.get('residual', positional[2] if len(positional) > 2 else None)
        diagnostic = dict(boundary_rank=rank, boundary_layer=boundary_layer,
                          prompt_tokens=count, tensors={})
        snapshots = {}
        for kind, label, actual in ((0, 'hidden', hidden), (1, 'residual', residual)):
            expected = cpu_boundary_state[kind]
            actual = actual.detach().cpu().clone()
            delta = actual.float() - expected.float()
            diagnostic['tensors'][label] = dict(
                total_elements=expected.numel(),
                unequal_elements=int((actual != expected).sum()),
                max_abs_difference=float(delta.abs().max()),
                rms_difference=float(delta.square().mean().sqrt()),
                gpu_rms=float(actual.float().square().mean().sqrt()),
                cpu_rms=float(expected.float().square().mean().sqrt()))
            snapshots[label] = dict(cpu=expected, gpu=actual)
        folder = Path(runner.model_config.model)
        (folder / f'boundary-rank-{rank}.json').write_text(json.dumps(diagnostic, indent=2))
        torch.save(snapshots, folder / f'boundary-rank-{rank}.pt')
        print(json.dumps(diagnostic), flush=True)
        cpu_boundary_state.clear()
    runner.model.model.layers[boundary_layer].register_forward_pre_hook(
        compare_boundary, with_kwargs=True)



def install_layer_capture(worker):
    """Record actual short-prompt layer outputs, once per worker and layer."""
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    runner = worker.model_runner
    role = 'cpu' if runner.device.type == 'cpu' else 'gpu'
    rank = get_tensor_model_parallel_rank()
    saved = set()
    def capture(layer_idx):
        def hook(module, positional, keyword, output):
            if layer_idx in saved:
                return
            if role == 'gpu' and runner.shm_scheduler_output is not None:
                return
            positions = keyword.get('positions', positional[0] if positional else None)
            if positions is None:
                return
            ids = runner.input_ids.cpu[:positions.numel()].tolist()
            for start in range(len(ids) - 4):
                if ids[start:start + 5] != [151643, 9906, 11, 358, 1097]:
                    continue
                if positions[start:start + 5].cpu().tolist() != list(range(5)):
                    continue
                hidden, residual = output
                torch.save(dict(hidden=hidden[start:start + 5].detach().cpu().clone(),
                                residual=residual[start:start + 5].detach().cpu().clone()),
                           Path(runner.model_config.model) /
                           f'layer-{layer_idx}-{role}-rank-{rank}.pt')
                saved.add(layer_idx)
                break
        return hook
    for index in (0, 1):
        runner.model.model.layers[index].register_forward_hook(
            capture(index), with_kwargs=True)
    selected_row = [None]
    operator_states = {}
    def select_short_prompt(module, positional, keyword):
        selected_row[0] = None
        if 'operators' in saved:
            return
        if role == 'gpu' and runner.shm_scheduler_output is not None:
            return
        positions = keyword.get('positions', positional[0] if positional else None)
        if positions is None:
            return
        ids = runner.input_ids.cpu[:positions.numel()].tolist()
        for start in range(len(ids) - 4):
            if (ids[start:start + 5] == [151643, 9906, 11, 358, 1097]
                    and positions[start:start + 5].cpu().tolist() == list(range(5))):
                selected_row[0] = start
                hidden = keyword.get('hidden_states', positional[1] if len(positional) > 1 else None)
                operator_states['input'] = hidden[start:start + 5].detach().cpu().clone()
                break
    def record_operator(label):
        def hook(module, positional, output):
            start = selected_row[0]
            if start is None:
                return
            values = output if isinstance(output, tuple) else (output,)
            for index, value in enumerate(values):
                if isinstance(value, torch.Tensor):
                    operator_states[f'{label}.{index}'] = value[start:start + 5].detach().cpu().clone()
        return hook
    def finish_operators(module, positional, output):
        if selected_row[0] is None:
            return
        torch.save(operator_states, Path(runner.model_config.model) /
                   f'operators-{role}-rank-{rank}.pt')
        saved.add('operators')
        selected_row[0] = None
    layer = runner.model.model.layers[0]
    layer.register_forward_pre_hook(select_short_prompt, with_kwargs=True)
    for label in ('input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'):
        getattr(layer, label).register_forward_hook(record_operator(label))
    layer.register_forward_hook(finish_operators)


def wait(path):
    deadline = time.monotonic() + 180
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--cpu-layers', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=40)
    parser.add_argument('--ordinary-first', action='store_true')
    parser.add_argument('--two-relays', action='store_true')
    parser.add_argument('--prompt-tokens', type=int, default=65)
    parser.add_argument('--prompt-token-file', type=Path,
                        help='JSON token IDs for a varied long-text prompt')
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--diagnose-logprobs', action='store_true')
    parser.add_argument('--audit-kv', action='store_true')
    parser.add_argument('--warmup-diagnostics', action='store_true',
                        help='GPU-only stream timing and CPU call profiling')
    parser.add_argument('--cpu-norm-experiment', type=Path,
                        help='Diagnostic CPU operator override from isolated benchmark')
    parser.add_argument('--gpu-pp', type=int, default=1)
    parser.add_argument('--cold-layer-counts', type=int, nargs='+')
    args = parser.parse_args()
    if args.audit_kv or args.cpu_norm_experiment:
        # Local test sends a callable through the frontend utility RPC.
        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    folder = args.folder.resolve()
    prompt = ([151643] + [9707, 11, 358, 1097]
              * ((args.prompt_tokens + 3) // 4))[:args.prompt_tokens]
    if args.prompt_token_file:
        prompt = json.loads(args.prompt_token_file.read_text())
        if (not isinstance(prompt, list) or not prompt
                or any(type(token) is not int or token < 0 for token in prompt)):
            parser.error('--prompt-token-file must contain a nonempty list of token IDs')
        args.prompt_tokens = len(prompt)
    common = dict(model=str(folder), tensor_parallel_size=args.tp,
                  load_format='shm', load_method='layerwise', lazy_load=True,
                  dtype='bfloat16', enforce_eager=True,
                  max_model_len=max(256, args.prompt_tokens + args.output_tokens),
                  max_num_seqs=2, max_num_batched_tokens=args.chunk_size,
                  enable_chunked_prefill=True, block_size=32,
                  skip_tokenizer_init=True, enable_prefix_caching=False)
    if args.role == 'cpu':
        if args.checkpoint is None:
            parser.error('--checkpoint required for CPU')
        folder.mkdir(parents=True, exist_ok=True)
        for path in args.checkpoint.iterdir():
            if path.is_file() and path.suffix in ('.json', '.safetensors'):
                (folder / path.name).symlink_to(path.resolve())
        model = LLM(**common)
    else:
        wait(folder / 'cpu-ready.json')
        if args.warmup_diagnostics:
            common['worker_cls'] = (
                'relay_warmup_diagnostic_worker.WarmupDiagnosticWorker')
        size, blocks, block_len, cpu_tp = json.loads(
            (folder / 'cpu-ready.json').read_text())
        model = LLM(**common, pipeline_parallel_size=args.gpu_pp,
                    shm_tp_size=cpu_tp, load_async=True,
                    shm_kv_cache_size=size, shm_num_blocks=blocks,
                    shm_block_len=block_len, gpu_memory_utilization=0.5)
    engine = model.llm_engine
    initialization_start = None
    try:
        if args.role == 'cpu':
            if args.gpu_pp > 1:
                engine.engine_core.call_utility('update_pp_cleanup', True)
            engine.collective_rpc('update_computing_layers', args=(args.cpu_layers,))
            if args.cpu_norm_experiment:
                engine.collective_rpc(install_cpu_norm_experiment,
                                      args=(str(args.cpu_norm_experiment.resolve()),))
            if args.audit_kv:
                engine.collective_rpc(install_layer_capture)
            info = engine.collective_rpc('get_shm_kv_cache_info')[0]
            (folder / 'cpu-ready.json').write_text(json.dumps(info))
            wait(folder / 'start-relay')
        else:
            layers = AutoConfig.from_pretrained(folder).num_hidden_layers
            counts = args.cold_layer_counts or [layers - args.cpu_layers]
            if len(counts) != args.gpu_pp or sum(counts) != layers - args.cpu_layers:
                raise ValueError('cold-layer-counts must partition GPU suffix')
            partition = []
            start = args.cpu_layers
            for count in counts:
                partition.append(list(range(start, start + count)))
                start += count
            # CPU starts its first AMX batch while GPU model-specific
            # initialization and background weight transfer are in progress.
            initialization_start = time.monotonic()
            (folder / 'start-relay').touch()
            print(json.dumps(dict(lazy_init_start=initialization_start)), flush=True)
            engine.lazy_init(partition,
                             request_id='layerwise-relay')
            print(json.dumps(dict(lazy_init_end=time.monotonic())), flush=True)
            if args.audit_kv:
                if args.gpu_pp != 1:
                    raise ValueError('KV audit currently requires PP=1')
                engine.collective_rpc(install_kv_audit,
                                      args=(args.cpu_layers, 'layerwise-relay'))
                engine.collective_rpc(install_layer_capture)
        if args.role == 'gpu' and args.ordinary_first:
            engine.add_request('ordinary-first', {'prompt_token_ids': prompt},
                               SamplingParams(temperature=0,
                                              max_tokens=args.output_tokens,
                                              ignore_eos=True))
        relay_prompts = {'layerwise-relay': prompt}
        if args.two_relays:
            relay_prompts['second-relay'] = [151643, 9906, 11, 358, 1097]
        for request_id, request_prompt in relay_prompts.items():
            engine.add_request(request_id, {'prompt_token_ids': request_prompt},
                               SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                              ignore_eos=True,
                                              logprobs=5 if args.diagnose_logprobs else None))
        if args.role == 'gpu':
            (folder / 'start-relay').touch()
        tokens = None
        ordinary_tokens = None
        relay_tokens = {}
        relay_logprobs = {}
        first_token_times = {}
        while engine.has_unfinished_requests() and not (
                args.role == 'cpu' and (folder / 'stop').exists()):
            for output in engine.step():
                if (args.role == 'gpu' and output.outputs
                        and output.outputs[0].token_ids
                        and output.request_id not in first_token_times):
                    observed = time.monotonic()
                    first_token_times[output.request_id] = observed
                    print(json.dumps(dict(
                        request_id=output.request_id,
                        first_token_time=observed,
                        model_creation_to_first_token_seconds=(
                            observed - initialization_start))), flush=True)
                if output.finished:
                    if output.request_id in relay_prompts:
                        relay_tokens[output.request_id] = list(output.outputs[0].token_ids)
                        relay_logprobs[output.request_id] = output.outputs[0].logprobs
                    if output.request_id == 'layerwise-relay':
                        tokens = list(output.outputs[0].token_ids)
                    elif output.request_id == 'ordinary-first':
                        ordinary_tokens = list(output.outputs[0].token_ids)
        if args.role == 'gpu':
            if args.gpu_pp > 1:
                transition_complete = engine.get_pp_translation_completed()
                print(json.dumps(dict(pp_transition_complete=transition_complete)),
                      flush=True)
                assert transition_complete, 'request ended before uniform PP recovery'
            reference = model.generate([{'prompt_token_ids': prompt}],
                SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True),
                use_tqdm=False)[0]
            expected = list(reference.outputs[0].token_ids)
            print(json.dumps(dict(tokens=tokens, reference=expected)), flush=True)
            assert tokens == expected
            if args.ordinary_first:
                assert ordinary_tokens == expected, ordinary_tokens
            if args.two_relays:
                second = model.generate(
                    [{'prompt_token_ids': relay_prompts['second-relay']}],
                    SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                   ignore_eos=True,
                                   logprobs=5 if args.diagnose_logprobs else None),
                    use_tqdm=False)[0]
                second_expected = list(second.outputs[0].token_ids)
                print(json.dumps(dict(request_id='second-relay',
                                      tokens=relay_tokens['second-relay'],
                                      reference=second_expected)), flush=True)
                if args.diagnose_logprobs:
                    mismatch = next((i for i, (a, b) in enumerate(zip(
                        relay_tokens['second-relay'], second_expected)) if a != b), None)
                    def probabilities(items):
                        return {token: value.logprob for token, value in items.items()}
                    if mismatch is not None:
                        print(json.dumps(dict(first_mismatch=mismatch,
                            relay_top=probabilities(relay_logprobs['second-relay'][mismatch]),
                            reference_top=probabilities(second.outputs[0].logprobs[mismatch]))),
                            flush=True)
                        batch_reference = model.generate(
                            [{'prompt_token_ids': p} for p in relay_prompts.values()],
                            SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                           ignore_eos=True, logprobs=5), use_tqdm=False)
                        batched_second = batch_reference[1].outputs[0]
                        print(json.dumps(dict(reference_batch_tokens=list(
                            batched_second.token_ids), reference_batch_top=probabilities(
                                batched_second.logprobs[mismatch]))), flush=True)
                        # Both executions have the same history before their
                        # first mismatch. Recompute that history in one GPU
                        # prefill to distinguish incremental-cache behavior
                        # from an independently reconstructed context. This
                        # is evidence, not a replacement for the exact check.
                        common_prefix = (relay_prompts['second-relay']
                                         + second_expected[:mismatch])
                        replay = model.generate(
                            [{'prompt_token_ids': common_prefix}],
                            SamplingParams(temperature=0, max_tokens=1,
                                           ignore_eos=True, logprobs=5),
                            use_tqdm=False)[0].outputs[0]
                        diagnostic = dict(
                            first_mismatch=mismatch,
                            context_tokens=len(common_prefix),
                            relay_top=probabilities(
                                relay_logprobs['second-relay'][mismatch]),
                            reference_top=probabilities(
                                second.outputs[0].logprobs[mismatch]),
                            reference_batch_top=probabilities(
                                batched_second.logprobs[mismatch]),
                            recomputed_token_ids=list(replay.token_ids),
                            recomputed_top=probabilities(replay.logprobs[0]))
                        (folder / 'divergence.json').write_text(
                            json.dumps(diagnostic, indent=2))
                        print(json.dumps(diagnostic), flush=True)
                assert relay_tokens['second-relay'] == second_expected
            (folder / 'stop').touch()
        else:
            wait(folder / 'stop')
    finally:
        if args.role == 'gpu':
            (folder / 'stop').touch()
        engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
