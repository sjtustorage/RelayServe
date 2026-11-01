"""Produce a cache fixture with the actual CPU AMX cache-writing operator."""
import argparse
import torch
from vllm import _custom_ops as ops

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output')
parser.add_argument('--blocks', type=int, default=3)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dims', type=int, default=64)
parser.add_argument('--cpu-tp', type=int, default=1)
args = parser.parse_args()
assert args.heads % args.cpu_tp == 0
torch.manual_seed(71)
blocks, heads, tokens, dims = args.blocks, args.heads, 32, args.dims
key = torch.randn(blocks * tokens, heads, dims).bfloat16()
value = torch.randn_like(key)
cache = torch.zeros(2, blocks, heads, tokens, dims, dtype=torch.bfloat16)
slots = torch.arange(blocks * tokens, dtype=torch.int64)
ops.cpu_attn_reshape_and_cache(key, value, cache[0], cache[1], slots, "amx")
reference = torch.stack([key.reshape(blocks, tokens, heads, dims).transpose(1, 2),
                         value.reshape(blocks, tokens, heads, dims).transpose(1, 2)])
query = torch.randn(1, heads, dims).bfloat16()
starts = torch.tensor([0, 1], dtype=torch.int32)
lengths = torch.tensor([blocks * tokens], dtype=torch.int32)
metadata = ops.cpu_attn_get_scheduler_metadata(
    1, heads, heads, dims, lengths, torch.bfloat16, starts, True, -1, "amx", False)
output = torch.empty_like(query)
ops.cpu_attention_with_kv_cache(query, cache[0], cache[1], output, starts,
                                lengths, dims ** -0.5, True, None, (-1, -1),
                                torch.arange(blocks, dtype=torch.int32).view(1, -1),
                                0.0, metadata, None)
expected = torch.nn.functional.scaled_dot_product_attention(
    query.transpose(0, 1).unsqueeze(0).float(),
    key.transpose(0, 1).unsqueeze(0).float(),
    value.transpose(0, 1).unsqueeze(0).float()).squeeze(0).transpose(0, 1)
torch.testing.assert_close(output.float(), expected, rtol=0.03, atol=0.01)
shards = []
for rank in range(args.cpu_tp):
    start = rank * (heads // args.cpu_tp)
    end = start + heads // args.cpu_tp
    shard = torch.zeros(2, blocks, heads // args.cpu_tp, tokens, dims,
                        dtype=torch.bfloat16)
    ops.cpu_attn_reshape_and_cache(key[:, start:end].contiguous(),
                                  value[:, start:end].contiguous(),
                                  shard[0], shard[1], slots, 'amx')
    shards.append(shard)
torch.save({"packed": cache, "reference": reference,
            "packed_shards": shards, "cpu_tp": args.cpu_tp,
            "query": query, "cpu_attention": output}, args.output)
print("PASS: CPU AMX attention matches logical KV reference")
