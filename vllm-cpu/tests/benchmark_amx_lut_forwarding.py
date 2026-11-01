"""Paired AMX prefetch-distance benchmark using real checkpoint dimensions."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
import amx_gemm_bf16_ext as amx
from vllm._custom_ops import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cpu-ids', default='96-142')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    h, i = config['hidden_size'], config['intermediate_size']
    d = config.get('head_dim', h // config['num_attention_heads'])
    shapes = [('qkv', (config['num_attention_heads'] + 2 * config['num_key_value_heads']) * d // 2, h),
              ('o', h, config['num_attention_heads'] * d // 2),
              ('gate_up', i, h), ('down', h, i // 2)]
    amx.init_cpu_threads_env(args.cpu_ids)
    torch.manual_seed(17)
    randomizer = random.Random(17)
    report = dict(checkpoint=str(args.checkpoint), cpu_ids=args.cpu_ids, shapes=[])
    with torch.inference_mode():
        for name, n, k in shapes:
            a = torch.randn(4096, k, dtype=torch.bfloat16)
            weight = torch.randn(n, k, dtype=torch.bfloat16)
            amx.pack_weight_nk(weight)
            outputs = [torch.empty(4096, n, dtype=torch.bfloat16) for _ in range(2)]
            cfg = get_config(4096, n, k)
            common = {key: cfg[key] for key in ('TM', 'TN', 'TK', 'prefetch_a', 'prefetch_b', 'prefetch_c')}
            # Keep hardware prefetch enabled in both arms. MSR configuration
            # is a host-wide policy and must not change during this benchmark.
            tuned = {key: cfg[key] for key in ('pf_a_dist', 'pf_b_dist', 'pf_c_dist')}
            def run(arm):
                started = time.perf_counter()
                amx.gemm_bf16(a, weight.t(), outputs[arm], **common,
                              **(tuned if arm else {}))
                return time.perf_counter() - started
            for _ in range(5):
                run(0); run(1)
            assert torch.equal(outputs[0], outputs[1]), name
            timings = [[], []]
            for _ in range(args.repeats):
                order = [0, 1]; randomizer.shuffle(order)
                for arm in order:
                    timings[arm].append(run(arm))
            assert torch.equal(outputs[0], outputs[1]), name
            medians = [statistics.median(values) for values in timings]
            result = dict(name=name, m=4096, n=n, k=k, config=cfg,
                          exact=True, seconds=timings, median_seconds=medians,
                          tuned_over_default=medians[1] / medians[0])
            report['shapes'].append(result)
            print(json.dumps(result), flush=True)
            del a, weight, outputs
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
