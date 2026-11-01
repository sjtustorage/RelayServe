"""Run both real CPU/GPU roles and assert asymmetric PP recovery succeeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-python', required=True)
    parser.add_argument('--gpu-python', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--prompt-token-file', type=Path, required=True)
    parser.add_argument('--devices', default='0,1,4,5')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--cpu-layers', type=int, default=8)
    parser.add_argument('--cold-layer-counts', type=int, nargs='+', required=True)
    parser.add_argument('--output-tokens', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    if (args.tp < 1 or args.timeout <= 0 or args.output_tokens < 1
            or not 0 < args.cpu_layers < config['num_hidden_layers']
            or len(args.cold_layer_counts) < 2 or min(args.cold_layer_counts) < 1
            or sum(args.cold_layer_counts) != config['num_hidden_layers'] - args.cpu_layers
            or len(args.devices.split(',')) != args.tp * len(args.cold_layer_counts)):
        parser.error('Invalid CPU/GPU layer partition or visible devices')
    # Require fresh names so old manifests/signals cannot satisfy this run.
    args.folder.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    common = [str(Path(__file__).with_name('verify_qwen_layerwise_v1.py')),
        '--folder', str(args.folder.resolve()), '--tp', str(args.tp),
        '--gpu-pp', str(len(args.cold_layer_counts)), '--cpu-layers', str(args.cpu_layers),
        '--cold-layer-counts', *map(str, args.cold_layer_counts),
        '--output-tokens', str(args.output_tokens), '--prompt-token-file',
        str(args.prompt_token_file.resolve()), '--two-relays']
    cpu_env = dict(os.environ, PYTHONPATH=str(root / 'vllm-cpu'),
        VLLM_CPU_KVCACHE_SPACE='4', OMP_NUM_THREADS='48', KMP_AFFINITY='disabled', VLLM_USE_V1='1')
    gpu_env = dict(os.environ, PYTHONPATH=str(root / 'vllm-gpu'),
                   CUDA_VISIBLE_DEVICES=args.devices, VLLM_USE_V1='1')
    processes = []
    try:
        with (args.folder / 'cpu.log').open('w') as c, (args.folder / 'gpu.log').open('w') as g:
            cpu = subprocess.Popen([args.cpu_python, *common, '--role', 'cpu',
                '--checkpoint', str(args.checkpoint.resolve())], env=cpu_env, stdout=c, stderr=subprocess.STDOUT)
            processes.append(cpu)
            gpu = subprocess.Popen([args.gpu_python, *common, '--role', 'gpu'],
                env=gpu_env, stdout=g, stderr=subprocess.STDOUT)
            processes.append(gpu)
            deadline = time.monotonic() + args.timeout
            while gpu.poll() is None:
                if cpu.poll() is not None and cpu.returncode != 0:
                    raise RuntimeError('CPU role failed; inspect cpu.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('PP recovery exceeded timeout')
                time.sleep(.1)
            (args.folder / 'stop').touch()
            cpu.wait(timeout=40)
            assert gpu.returncode == cpu.returncode == 0, (gpu.returncode, cpu.returncode)
        records = []
        for line in (args.folder / 'gpu.log').read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        assert any(isinstance(r, dict) and r.get('pp_transition_complete') is True for r in records)
        outputs = [r for r in records if isinstance(r, dict) and 'tokens' in r and 'reference' in r]
        assert len(outputs) == 2 and all(len(r['tokens']) == args.output_tokens and r['tokens'] == r['reference'] for r in outputs)
        result = dict(cpu_tp=args.tp, gpu_tp=args.tp, gpu_pp=len(args.cold_layer_counts),
                      cpu_layers=args.cpu_layers, cold_layer_counts=args.cold_layer_counts,
                      pp_transition_complete=True, outputs=outputs)
        (args.folder / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        (args.folder / 'stop').touch()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
