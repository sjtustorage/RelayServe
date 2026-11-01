"""Persistent CPU relay producer for the resident service.

The process owns one vLLM CPU engine and communicates through JSON lines.  It
is intentionally independent of Ray: the controller may keep this process
alive while GPU model slots are unloaded and recreated.  Commands are
serialized because a producer's SHM KV allocator is shared by its requests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _reply(stdout, sequence, *, result: Any = None, error: str | None = None,
           fatal: bool = False) -> None:
    payload = dict(sequence=sequence, ok=error is None, result=result, error=error)
    if fatal:
        payload['fatal'] = True
    print('CPU_PRODUCER_REPLY ' + json.dumps(payload, separators=(',', ':')),
          flush=True, file=stdout)


class CpuProducer:
    def __init__(self, command: dict[str, Any]):
        from vllm import LLM
        model = str(command['model'])
        tp = command['tensor_parallel_size']
        warmup = command.get('warmup', True)
        if type(warmup) is not bool:
            raise ValueError('warmup must be a boolean')
        self.model = LLM(
            model=model, tensor_parallel_size=tp, dtype='bfloat16',
            load_format='shm', enforce_eager=True, lazy_load=True,
            max_model_len=command.get('max_model_len', 40960),
            max_num_batched_tokens=command.get('max_num_batched_tokens', 4096),
            enable_chunked_prefill=True, block_size=command.get('block_size', 32),
            skip_tokenizer_init=True, enable_prefix_caching=False,
            kv_transfer_config=dict(kv_connector='ShmConnector',
                                    kv_role='kv_producer'))
        self.engine = self.model.llm_engine
        self.model_path = model
        self.tp = tp
        self.pending_request = None
        self.warmup_lengths = []
        self.warmup_seconds = 0.0
        if not warmup:
            print('CPU_PRODUCER_WARMUP ' + json.dumps(dict(
                lengths=[], seconds=0.0, samples=[])), flush=True)
            return
        # Run ordinary local requests after engine creation: unlike dummy
        # compilation warmup these exercise real attention and KV scheduling.
        # No remote-decode ownership is retained by these requests.
        from vllm import SamplingParams
        limit = min(4096, self.engine.model_config.max_model_len - 1,
                    command.get('max_num_batched_tokens', 4096))
        lengths = sorted({min(length, limit) for length in
                          (128, 256, 512, 1024, 2048, 4096) if limit > 0})
        warmup_started = time.perf_counter()
        warmup_samples = []
        for length in lengths:
            shape_started = time.perf_counter()
            self.model.generate([{'prompt_token_ids': [1] * length}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                use_tqdm=False)
            warmup_samples.append(dict(tokens=length,
                seconds=time.perf_counter() - shape_started))
        self.warmup_lengths = lengths
        self.warmup_seconds = time.perf_counter() - warmup_started
        print('CPU_PRODUCER_WARMUP ' + json.dumps(dict(
            lengths=lengths, seconds=self.warmup_seconds,
            samples=warmup_samples)), flush=True)

    def layerwise(self, command):
        """Run a fresh CPU prefix session until GPU takeover acknowledges it."""
        if self.pending_request is not None:
            raise ValueError('release the outstanding KV handoff before another request')
        prompt = command['prompt_token_ids']
        layers = command.get('cpu_layers')
        request_id = command.get('request_id')
        maximum = command.get('max_tokens', 8)
        config = self.engine.model_config
        if (type(layers) is not int or not 0 < layers < config.hf_config.num_hidden_layers
                or not isinstance(request_id, str) or not request_id
                or type(maximum) is not int or maximum < 1
                or len(prompt) + maximum > config.max_model_len
                or any(token >= config.get_vocab_size() for token in prompt)):
            raise ValueError('invalid layerwise partition, request or prompt')
        from vllm import SamplingParams
        # Leaving layerwise clears worker/session counters. Enter through cpu
        # even after a previous request finished before full GPU takeover.
        core = self.engine.engine_core
        core.call_utility('update_load_method', 'cpu')
        core.call_utility('update_load_method', 'layerwise')
        self.engine.collective_rpc('update_computing_layers', args=(layers,))
        started = time.perf_counter()
        self.engine.add_request(request_id, {'prompt_token_ids': prompt},
            SamplingParams(temperature=0, max_tokens=maximum, ignore_eos=True))
        tokens = []
        while self.engine.has_unfinished_requests():
            for output in self.engine.step():
                if output.request_id == request_id:
                    tokens = list(output.outputs[0].token_ids)
        finished = time.perf_counter()
        # Every iteration waits for GPU output. No speculative next CPU batch.
        core.call_utility('update_load_method', 'cpu')
        return dict(request_id=request_id, token_ids=tokens, seconds=finished - started,
                    started_at=started, finished_at=finished, cpu_layers=layers)

    def shm_info(self):
        return self.engine.collective_rpc('get_shm_kv_cache_info')[0]

    def generate(self, command):
        if self.pending_request is not None:
            raise ValueError('release the outstanding KV handoff before another request')
        from vllm import SamplingParams
        prompt = command['prompt_token_ids']
        maximum = command.get('max_tokens', 1)
        if (type(maximum) is not int or maximum < 1 or
                len(prompt) + maximum > self.engine.model_config.max_model_len or
                any(token >= self.engine.model_config.get_vocab_size() for token in prompt)):
            raise ValueError('invalid generation length or token vocabulary')
        params = SamplingParams(
            temperature=command.get('temperature', 0),
            max_tokens=command.get('max_tokens', 1),
            ignore_eos=True,
            extra_args={'kv_transfer_params': {
                'do_remote_decode': True,
            }})
        started = time.perf_counter()
        output = self.model.generate([{'prompt_token_ids': prompt}], params,
                                     use_tqdm=False)[0]
        finished = time.perf_counter()
        self.pending_request = output.request_id
        return dict(token_ids=list(output.outputs[0].token_ids),
                    request_id=output.request_id,
                    kv_transfer_params=output.kv_transfer_params,
                    shm_info=self.shm_info(),
                    seconds=finished - started,
                    started_at=started, finished_at=finished)

    def release(self, request_id):
        if request_id != self.pending_request or request_id is None:
            raise ValueError('stale or missing handoff request ID')
        # The output processor has already finished this request, but its
        # scheduler KV blocks remain reserved for remote decode. Address the
        # engine core directly so the completed request is actually freed.
        self.engine.engine_core.abort_requests([request_id])
        self.pending_request = None
        return dict(released=True)

    def close(self):
        self.engine.engine_core.shutdown()


def serve(stdin, stdout):
    producer = None
    expected = 0
    try:
        for line in stdin:
            sequence = None
            accepted = False
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError('command must be an object')
                sequence = command.get('sequence')
                if type(sequence) is not int or sequence != expected:
                    raise ValueError(f'expected sequence {expected}')
                accepted = True
                op = command.get('op')
                if op == 'start':
                    if producer is not None:
                        raise ValueError('producer is already started')
                    if (not isinstance(command.get('model'), str) or
                            type(command.get('tensor_parallel_size')) is not int or
                            command['tensor_parallel_size'] < 1):
                        raise ValueError('start requires model and positive tensor parallel size')
                    producer = CpuProducer(command)
                    result = dict(model=producer.model_path, tensor_parallel_size=producer.tp,
                                  shm_info=producer.shm_info())
                elif op in ('generate', 'layerwise'):
                    if producer is None:
                        raise ValueError('producer is not started')
                    prompt = command.get('prompt_token_ids')
                    if (not isinstance(prompt, list) or not prompt or
                            any(type(token) is not int or token < 0 for token in prompt)):
                        raise ValueError('prompt_token_ids must be a nonempty integer list')
                    result = (producer.layerwise(command) if op == 'layerwise'
                              else producer.generate(command))
                elif op == 'status':
                    result = None if producer is None else dict(
                        model=producer.model_path, tensor_parallel_size=producer.tp,
                        shm_info=producer.shm_info())
                elif op == 'release':
                    if producer is None:
                        raise ValueError('producer is not started')
                    result = producer.release(command.get('request_id'))
                elif op == 'shutdown':
                    expected += 1
                    _reply(stdout, sequence, result={'stopped': True})
                    return
                else:
                    raise ValueError(f'unknown producer operation: {op}')
                expected += 1
                _reply(stdout, sequence, result=result)
            except Exception as exc:
                if accepted:
                    expected += 1
                fatal = not isinstance(exc, (ValueError, KeyError, TypeError))
                _reply(stdout, sequence, error=f'{type(exc).__name__}: {exc}', fatal=fatal)
                if fatal:
                    # Device/engine failures can leave producer ownership
                    # uncertain. Never accept another command in that state.
                    return
    finally:
        if producer is not None:
            producer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        help='Optional model to start immediately')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault('VLLM_USE_V1', '1')
    if args.model:
        command = dict(sequence=0, op='start', model=str(args.model),
                       tensor_parallel_size=args.tensor_parallel_size)
        # Preserve the same protocol for eager startup, but do not print a
        # reply until the command loop has emitted it.
        import io
        input_stream = io.StringIO(json.dumps(command) + '\n')
        class Combined:
            def __iter__(self):
                yield from input_stream
                yield from sys.stdin
        serve(Combined(), sys.stdout)
    else:
        serve(sys.stdin, sys.stdout)


if __name__ == '__main__':
    main()
