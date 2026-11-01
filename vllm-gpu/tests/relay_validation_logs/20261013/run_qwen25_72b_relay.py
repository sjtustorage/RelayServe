import json
import os
from pathlib import Path
import subprocess
import time
import argparse
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline')
parser.add_argument('--prompt-tokens', type=int, default=128)
parser.add_argument('--arch-profile', action='store_true')
parser.add_argument('--model', choices=['Qwen2.5-72B', 'Qwen3-32B'],
                    default='Qwen2.5-72B')
parser.add_argument('--gpu-devices', default=None,
                    help='TP ranks ordered to match the two CPU NUMA shards')
args = parser.parse_args()

gpu_tp = 4 if args.model == 'Qwen2.5-72B' else 2
args.gpu_devices = args.gpu_devices or ('0,1,4,5' if gpu_tp == 4 else '0,4')
if len(args.gpu_devices.split(',')) != gpu_tp:
    parser.error('--gpu-devices must match model TP degree')
checkpoint = Path('/workspace/models') / args.model
deadline = time.monotonic() + 1800
while not (checkpoint / ('download-complete.json' if gpu_tp == 4 else
                         'model.safetensors.index.json')).exists():
    if time.monotonic() > deadline:
        raise TimeoutError(f'{args.model} checkpoint did not become ready')
    time.sleep(1)
print(f'{args.model} checkpoint ready; starting CPU TP2 producer', flush=True)
folder = Path(f'/tmp/relay-init{72 if gpu_tp == 4 else 32}-tp{gpu_tp}-{args.name}-20261013')
if folder.exists() or Path(str(folder) + '.commands.json').exists():
    parser.error('use a fresh --name; existing run artifacts must not be overwritten')
cpu_env = os.environ.copy()
cpu_env.update(VLLM_CPU_OMP_THREADS_BIND='0-47|48-95',
               VLLM_CPU_KVCACHE_SPACE='4' if args.prompt_tokens > 128 else '1', OMP_NUM_THREADS='48')
gpu_env = os.environ.copy()
# The shell preloads the CPU AMX OpenMP runtime. GPU processes use their
# own PyTorch runtime; inheriting libiomp pins their main threads to core 0.
preloads = gpu_env.get('LD_PRELOAD', '').replace(':', ' ').split()
gpu_env['LD_PRELOAD'] = ':'.join(
    item for item in preloads if not Path(item).name.startswith('libiomp'))
gpu_env.update(VLLM_PROFILE_RELAY='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN',
               VLLM_FLASH_ATTN_VERSION='3', CUDA_VISIBLE_DEVICES=args.gpu_devices)
if args.arch_profile:
    gpu_env['VLLM_PROFILE_SHM_ARCH'] = str(folder) + '.arch'
test = '/workspace/vllm-gpu/tests/verify_qwen_layerwise_v1.py'
common = ['--folder', str(folder), '--cpu-layers', '8',
          '--prompt-tokens', str(args.prompt_tokens), '--output-tokens', '8']
cpu_cmd = ['timeout', '900', '/workspace/vllm-cpu/.venv/bin/python', test,
           '--role', 'cpu', '--checkpoint', str(checkpoint), '--tp', '2', *common]
gpu_cmd = ['timeout', '600', '/workspace/vllm-gpu/.venv/bin/python', test,
           '--role', 'gpu', '--tp', str(gpu_tp), *common]
record = {'checkpoint': str(checkpoint), 'cpu_command': cpu_cmd,
          'gpu_command': gpu_cmd, 'cpu_environment': {k: cpu_env[k] for k in
          ('VLLM_CPU_OMP_THREADS_BIND', 'VLLM_CPU_KVCACHE_SPACE', 'OMP_NUM_THREADS')},
          'gpu_environment': {k: gpu_env[k] for k in
          ('VLLM_PROFILE_RELAY', 'VLLM_ATTENTION_BACKEND', 'VLLM_FLASH_ATTN_VERSION', 'CUDA_VISIBLE_DEVICES')}}
for role, environment in [('cpu', cpu_env), ('gpu', gpu_env)]:
    for key in ('LD_PRELOAD', 'KMP_AFFINITY', 'KMP_BLOCKTIME',
                'OMP_WAIT_POLICY', 'VLLM_LOGGING_LEVEL'):
        if key in environment:
            record[role + '_environment'][key] = environment[key]
if args.arch_profile:
    record['gpu_environment']['VLLM_PROFILE_SHM_ARCH'] = gpu_env['VLLM_PROFILE_SHM_ARCH']
for role, repo in [('cpu', '/workspace/vllm-cpu'), ('gpu', '/workspace/vllm-gpu')]:
    record[role + '_revision'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo)
    record[role + '_diff_sha256'] = hashlib.sha256(diff).hexdigest()
    Path(str(folder) + '.' + role + '.patch').write_bytes(diff)
Path(str(folder) + '.commands.json').write_text(json.dumps(record, indent=2))
with open(str(folder) + '.cpu.log', 'w') as cpu_log, open(str(folder) + '.gpu.log', 'w') as gpu_log:
    cpu = subprocess.Popen(cpu_cmd, cwd='/workspace/vllm-cpu', env=cpu_env,
                           stdout=cpu_log, stderr=subprocess.STDOUT)
    gpu = None
    try:
        ready_deadline = time.monotonic() + 870
        while not (folder / 'cpu-ready.json').exists():
            if cpu.poll() is not None:
                raise RuntimeError(f'CPU preparation failed: {cpu.returncode}')
            if time.monotonic() > ready_deadline:
                raise TimeoutError('CPU preparation timed out')
            time.sleep(1)
        print(f'CPU packed weights ready; starting GPU TP{gpu_tp} initialization test', flush=True)
        gpu = subprocess.Popen(gpu_cmd, cwd='/workspace/vllm-gpu', env=gpu_env,
                               stdout=gpu_log, stderr=subprocess.STDOUT)
        record['gpu_exit_code'] = gpu.wait()
        record['cpu_exit_code'] = cpu.wait(timeout=30)
        assert record['gpu_exit_code'] == record['cpu_exit_code'] == 0, record
        print(f'PASS: real {args.model} CPU TP2 -> GPU TP{gpu_tp} relay', flush=True)
    finally:
        if folder.exists():
            (folder / 'stop').touch()
        if gpu is not None and gpu.poll() is None:
            gpu.terminate()
        if cpu.poll() is None:
            cpu.terminate()
        Path(str(folder) + '.commands.json').write_text(json.dumps(record, indent=2))
