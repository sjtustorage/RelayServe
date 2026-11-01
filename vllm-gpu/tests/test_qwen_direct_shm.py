"""Manual full Qwen3 weight-path test using separate CPU/GPU runtimes.

Run with GPU Python and --cpu-python pointing to the CPU project's .venv.
The producer uses real distributed initialization and SharedMemoryLoader.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import json


def produce(folder, cpu_tp, checkpoint=None, load_only=False, keep_alive=False):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.model_loader import shared_memory_loader as loader

    rank = int(os.environ.get("RANK", "0"))
    torch.manual_seed(19)
    hf_config = (Qwen3Config.from_pretrained(checkpoint) if checkpoint else Qwen3Config(hidden_size=256, intermediate_size=256,
                           num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=64, vocab_size=256,
                           max_position_embeddings=128, tie_word_embeddings=False))
    init_distributed_environment(world_size=cpu_tp, rank=rank, local_rank=rank,
                                 distributed_init_method="file://" + str(folder / "rendezvous"),
                                 backend="gloo")
    if rank == 0:
        if checkpoint:
            for path in checkpoint.iterdir():
                if path.is_file() and (path.suffix == ".json" or path.suffix == ".safetensors"):
                    (folder / path.name).symlink_to(path.resolve())
        else:
            hf_model = Qwen3ForCausalLM(hf_config).to(torch.bfloat16)
            hf_model.save_pretrained(folder, safe_serialization=True)
            del hf_model
    torch.distributed.barrier()
    config = EngineArgs(model=str(folder), dtype="bfloat16", load_format="shm",
                        max_model_len=128, enforce_eager=True,
                        tensor_parallel_size=cpu_tp,
                        skip_tokenizer_init=True).create_engine_config()
    initialize_model_parallel(cpu_tp, 1, backend="gloo")
    postprocess = loader.process_weights_after_loading

    def capture(model, *args):
        capture_start = time.perf_counter()
        # Capture the logical tensors after real model-specific mapping and TP
        # loading, immediately before the real AMX packing changes their layout.
        reference = {}
        checkpoint_files = None
        if checkpoint and rank == 0:
            from safetensors import safe_open
            checkpoint_files = json.loads(
                (checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

            def original(name):
                with safe_open(str(checkpoint / checkpoint_files[name]), framework="pt") as stream:
                    return stream.get_tensor(name)
        for name, value in model.state_dict().items():
            parts = [torch.empty_like(value) for _ in range(cpu_tp)]
            torch.distributed.all_gather(parts, value.contiguous())
            if "norm" in name:
                reference[name] = parts[0]
            elif "qkv_proj" in name:
                split = [hf_config.num_attention_heads * hf_config.head_dim // cpu_tp,
                         hf_config.num_key_value_heads * hf_config.head_dim // cpu_tp,
                         hf_config.num_key_value_heads * hf_config.head_dim // cpu_tp]
                shards = [part.split(split, dim=0) for part in parts]
                reference[name] = torch.cat([torch.cat([part[i] for part in shards])
                                             for i in range(3)])
            elif "gate_up_proj" in name:
                shards = [part.chunk(2, dim=0) for part in parts]
                reference[name] = torch.cat([torch.cat([part[i] for part in shards])
                                             for i in range(2)])
            else:
                dim = 1 if "o_proj" in name or "down_proj" in name else 0
                reference[name] = torch.cat(parts, dim=dim)
            if checkpoint_files is not None:
                if "qkv_proj" in name:
                    expected = torch.cat([original(name.replace("qkv_proj", part))
                                          for part in ("q_proj", "k_proj", "v_proj")])
                elif "gate_up_proj" in name:
                    expected = torch.cat([original(name.replace("gate_up_proj", part))
                                          for part in ("gate_proj", "up_proj")])
                else:
                    expected = original(name)
                torch.testing.assert_close(reference[name], expected, rtol=0, atol=0,
                                           msg=lambda message: name + ": " + message)
        if rank == 0:
            torch.save(reference, folder / "reference.pt")
        print(f"rank {rank}: reference reconstruction/write took "
              f"{time.perf_counter() - capture_start:.3f}s", flush=True)
        pack_start = time.perf_counter()
        postprocess(model, *args)
        print(f"rank {rank}: actual weight postprocessing took "
              f"{time.perf_counter() - pack_start:.3f}s", flush=True)

    if not load_only:
        loader.process_weights_after_loading = capture
    load_start = time.perf_counter()
    model = loader.SharedMemoryLoader(config.load_config).load_model(vllm_config=config)
    print(f"rank {rank}: complete model load took {time.perf_counter() - load_start:.3f}s",
          flush=True)
    torch.distributed.barrier()
    if rank == 0:
        (folder / "ready").touch()
    while (not load_only or keep_alive) and not (folder / "stop").exists():
        time.sleep(0.1)  # Retain model ownership while GPU verifies the weights.
    assert model is not None
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-python")
    parser.add_argument("--produce", type=Path)
    parser.add_argument("--cpu-tp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu-tp", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--gpu-rank", type=int, default=0)
    parser.add_argument("--load-only", action="store_true",
                        help="CPU producer only: measure loading without reference generation")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Retain CPU SHM ownership after --load-only for engine inference")
    args = parser.parse_args()
    if args.produce:
        produce(args.produce, args.cpu_tp, args.checkpoint, args.load_only,
                args.keep_alive)
        return
    if not args.cpu_python:
        parser.error("--cpu-python is required")
    import torch
    from shm_manager import torch as shared

    with tempfile.TemporaryDirectory(prefix="relay-qwen-") as temporary:
        folder = Path(temporary)
        cpu_repo = Path(__file__).resolve().parents[2] / "vllm-cpu"
        environment = dict(os.environ, PYTHONPATH=str(cpu_repo), OMP_NUM_THREADS="2",
                           VLLM_CPU_USE_AMX_GEMM_BF16="1", VLLM_USE_V1="1",
                           VLLM_DIST_IDENT=folder.name)
        command = [args.cpu_python]
        if args.cpu_tp > 1:
            command += ["-m", "torch.distributed.run", "--standalone",
                        "--nproc-per-node", str(args.cpu_tp)]
        command += [str(Path(__file__).resolve()), "--produce", str(folder),
                    "--cpu-tp", str(args.cpu_tp)]
        if args.checkpoint:
            command += ["--checkpoint", str(args.checkpoint.resolve())]
        producer = subprocess.Popen(command, env=environment, cwd=temporary)
        try:
            deadline = time.monotonic() + 1800
            while not (folder / "ready").exists():
                if producer.poll() is not None:
                    raise RuntimeError(f"CPU producer exited with {producer.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("CPU model loading timed out")
                time.sleep(0.1)
            configuration = json.loads((folder / "config.json").read_text())
            layers = list(range(configuration["num_hidden_layers"]))
            shared.init_model_info(str(folder), args.gpu_rank, args.gpu_tp, args.cpu_tp,
                                   configuration["num_attention_heads"],
                                   configuration["num_key_value_heads"], configuration["head_dim"])
            weights = shared.init_cuda_memory_dict(torch.cuda.current_device(), layers)
            shared.load_model_to_gpu(layers)
            shared.load_model_to_gpu_sync(layers)
            # Repeated requests must wait for the newly submitted work rather
            # than treating an earlier layer-completion event as sufficient.
            for _ in range(1 if args.checkpoint else 8):
                shared.load_model_to_gpu(layers)
            shared.load_model_to_gpu_sync(layers)
            assert shared.check_model_load_state(layers)
            reference = torch.load(folder / "reference.pt", weights_only=True, mmap=True)
            assert weights.keys() == reference.keys()
            for name, weight in weights.items():
                expected = reference[name]
                if "norm" not in name and args.gpu_tp > 1:
                    if "qkv_proj" in name:
                        sizes = [configuration["num_attention_heads"] * configuration["head_dim"],
                                 configuration["num_key_value_heads"] * configuration["head_dim"]]
                        expected = torch.cat([part.chunk(args.gpu_tp, dim=0)[args.gpu_rank]
                                              for part in expected.split([sizes[0], sizes[1], sizes[1]], dim=0)])
                    elif "gate_up_proj" in name:
                        expected = torch.cat([part.chunk(args.gpu_tp, dim=0)[args.gpu_rank]
                                              for part in expected.chunk(2, dim=0)])
                    else:
                        dim = 1 if "o_proj" in name or "down_proj" in name else 0
                        expected = expected.chunk(args.gpu_tp, dim=dim)[args.gpu_rank]
                torch.testing.assert_close(weight.cpu(), expected, rtol=0, atol=0,
                                           msg=lambda message: name + ": " + message)
            print(f"PASS: all {len(weights)} complete Qwen3 model tensors match exactly")
        finally:
            if producer.poll() is None:
                (folder / "stop").touch()
                try:
                    producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    producer.wait(timeout=10)
            for rank in range(args.cpu_tp):
                (Path("/dev/shm") / (folder.name + f"-rank_{rank}")).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
