"""Manual cross-runtime test: run with GPU Python and --cpu-python PATH.

Uses real AMX packing, POSIX SHM, native CUDA transfer and CUDA unpacking.
The CPU repository defaults to the sibling vllm-cpu checkout.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def shard(weight, name, tp, rank):
    if "norm" in name:
        return weight.clone()
    dim = 1 if "o_proj" in name or "down_proj" in name else 0
    parts = weight.split([256, 128, 128], dim=0) if "qkv_proj" in name else (
        weight.chunk(2, dim=0) if "gate_up_proj" in name else [weight])
    return __import__("torch").cat([part.chunk(tp, dim=dim)[rank] for part in parts],
                                 dim=0).contiguous()


def produce(folder, cpu_tp):
    import torch
    from shm_manager.torch import share_cpu_state_dict
    from vllm.model_executor.layers.utils import ops
    weights = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            256, 256, generator=torch.Generator().manual_seed(42)).bfloat16(),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(
            256, dtype=torch.bfloat16),
    }
    generator = torch.Generator().manual_seed(43)
    for name, shape in [("self_attn.qkv_proj", (512, 256)),
                        ("mlp.gate_up_proj", (512, 256)),
                        ("mlp.down_proj", (256, 256))]:
        weights["model.layers.0." + name + ".weight"] = torch.randn(
            *shape, generator=generator).bfloat16()
    torch.save(weights, folder / "reference.pt")
    segments = {}
    index = {}
    offset = 0
    for name, weight in weights.items():
        size = weight.numel() * weight.element_size()
        index[name] = (offset, size, list(weight.shape), list(weight.stride()), str(weight.dtype))
        offset += size
    for rank in range(cpu_tp):
        local = {name: shard(weight, name, cpu_tp, rank) for name, weight in weights.items()}
        shared, _, path = share_cpu_state_dict(local, folder / f"rank_{rank}", alignment=1)
        for name, weight in shared.items():
            if "norm" not in name:
                ops.pack_weight_bf16(weight)
        stat = os.stat(path)
        segments[str(rank)] = {"inode": stat.st_ino, "size": stat.st_size}
    (folder / "shm_weights.json").write_text(json.dumps({
        "version": 1, "cpu_tp_size": cpu_tp, "tensor_index": index,
        "segments": segments,
    }))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-python")
    parser.add_argument("--cpu-repo", type=Path,
                        default=Path(__file__).resolve().parents[2] / "vllm-cpu")
    parser.add_argument("--produce", type=Path)
    parser.add_argument("--async-load", action="store_true")
    parser.add_argument("--cpu-tp", type=int, default=1)
    parser.add_argument("--gpu-tp", type=int, default=1)
    parser.add_argument("--gpu-rank", type=int, default=0)
    args = parser.parse_args()
    if args.produce:
        produce(args.produce, args.cpu_tp)
        return
    if not args.cpu_python:
        parser.error("--cpu-python is required")
    import torch
    from shm_manager import torch as shared
    with tempfile.TemporaryDirectory(prefix="relay-h2d-") as temporary:
        folder = Path(temporary)
        shm_paths = [Path("/dev/shm") / (folder.name + f"-rank_{rank}")
                     for rank in range(args.cpu_tp)]
        try:
            environment = dict(os.environ, PYTHONPATH=str(args.cpu_repo), OMP_NUM_THREADS="2")
            subprocess.run([args.cpu_python, str(Path(__file__).resolve()),
                            "--produce", str(folder), "--cpu-tp", str(args.cpu_tp)], env=environment,
                           cwd=temporary, check=True, timeout=120)
            torch.cuda.init()
            free_before, _ = torch.cuda.mem_get_info()
            # The fixture has Q=256, K=128 and V=128 rows. Use four
            # KV heads so TP=4 exercises partitioning rather than replication.
            shared.init_model_info(str(folder), args.gpu_rank, args.gpu_tp, args.cpu_tp, 8, 4, 32)
            weights = shared.init_cuda_memory_dict(torch.cuda.current_device(), [0])
            assert not shared.check_model_load_state([0]), "Unloaded layer reported ready"
            assert shared.model_info.manager.completed_layer_load_seconds() == []
            if args.async_load:
                shared.update_model_arch_init_state(True, bound_submissions=True)
                shared.load_model_to_gpu([0])
                shared.load_model_to_gpu_sync([0])
                shared.update_model_arch_init_state(False)
            else:
                shared.load_model_to_gpu_blocking([0])
            assert shared.check_model_load_state([0]), "Completed layer reported unready"
            samples = shared.model_info.manager.completed_layer_load_seconds()
            assert len(samples) == 1 and samples[0] > 0, samples
            # Exercise reuse of events and staging buffers through the blocking
            # entry point after the first load, including initialization waits.
            shared.update_module_loading_state(True)
            shared.load_model_to_gpu_blocking([0])
            shared.update_module_loading_state(False)
            assert shared.check_model_load_state([0])
            samples = shared.model_info.manager.completed_layer_load_seconds()
            assert len(samples) == 1 and samples[0] > 0, samples
            torch.cuda.synchronize()
            free_after, _ = torch.cuda.mem_get_info()
            assert free_before - free_after < 16 * 1024 * 1024, (
                "Tiny weights unexpectedly require large staging buffers", free_before - free_after)
            reference = torch.load(folder / "reference.pt", weights_only=True)
            assert weights.keys() == reference.keys()
            for name, weight in weights.items():
                print("CHECK",name, tuple(weight.shape), tuple(reference[name].shape))
                expected = shard(reference[name], name, args.gpu_tp, args.gpu_rank)
                torch.testing.assert_close(weight.cpu(), expected, rtol=0, atol=0)
            # A resident switch must fence a pending copy before detaching
            # host sources, preserve surviving tensors and keep producer SHM.
            manager = shared.model_info.manager
            shared.load_model_to_gpu([0])
            shared.reset_model_info()
            assert shared.model_info.manager is None
            assert folder.name not in Path('/proc/self/maps').read_text(), (
                'Consumer weight SHM mapping survived reset')
            for name, weight in weights.items():
                expected = shard(reference[name], name, args.gpu_tp, args.gpu_rank)
                torch.testing.assert_close(weight.cpu(), expected, rtol=0, atol=0)
            assert all(path.exists() for path in shm_paths)
            try:
                manager.load_layers_async([0])
            except RuntimeError as exc:
                assert 'closed' in str(exc)
            else:
                raise AssertionError('Closed manager accepted a new submission')
            # Reattach in the same process and repeat transfer after reset.
            shared.init_model_info(str(folder), args.gpu_rank, args.gpu_tp,
                                   args.cpu_tp, 8, 4, 32)
            reloaded = shared.init_cuda_memory_dict(torch.cuda.current_device(), [0])
            shared.load_model_to_gpu_blocking([0])
            for name, weight in reloaded.items():
                expected = shard(reference[name], name, args.gpu_tp, args.gpu_rank)
                torch.testing.assert_close(weight.cpu(), expected, rtol=0, atol=0)
            shared.reset_model_info()
            shared.reset_model_info()  # Empty reset must be idempotent.
            assert folder.name not in Path('/proc/self/maps').read_text()
            print("PASS: native SHM H2D and CUDA unpack match CPU weights exactly")
        finally:
            for shm_path in shm_paths:
                shm_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
