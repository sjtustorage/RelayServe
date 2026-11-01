"""Verify full V1 Qwen3-32B inference against a live CPU SHM producer.

Start test_qwen_direct_shm.py with --produce DIR --checkpoint CHECKPOINT
--load-only --keep-alive using CPU torchrun, then run this with GPU Python.
Expected tokens come from the standard GPU safetensors loader for this prompt.
Times report GPU engine initialization and the complete four-token request,
excluding CPU weight ingestion; they are not cold-start TTFT.
"""
import argparse
import json
import os
import time

os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cpu-tp", type=int, default=2)
    parser.add_argument("--expected-tokens", type=json.loads,
                        default=[360, 13, 358, 2776])
    args = parser.parse_args()
    start = time.perf_counter()
    model = LLM(model=args.model, tensor_parallel_size=args.tp,
                shm_tp_size=args.cpu_tp, load_format="shm", dtype="bfloat16",
                enforce_eager=True, max_model_len=256, max_num_seqs=2,
                gpu_memory_utilization=0.5, skip_tokenizer_init=True)
    ready = time.perf_counter()
    try:
        outputs = model.generate(
            [{"prompt_token_ids": [151643, 9707, 11, 358, 1097]}],
            SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
        tokens = outputs[0].outputs[0].token_ids
        print("RELAY_RESULT " + json.dumps({
            "gpu_tp": args.tp,
            "initialization_seconds": ready - start,
            "request_seconds": time.perf_counter() - ready,
            "tokens": tokens,
        }), flush=True)
        assert tokens == args.expected_tokens, tokens
    finally:
        model.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
