# Rejected KV index stream-allocation experiment

H20, real Qwen3-32B, CPU TP2/GPU TP2,13 CPU layers,128 tokens,
chunk4096, V1 eager lazy SHM, FA3. Two fresh pools per implementation,
each first model and reload. CPU weights/context/groups prepared before timer;
full warmup and AMX/transfer/init concurrency retained. All exact outputs match
independent original safetensors. Jobs ran serially.

Candidate replaces per-transfer index cudaMalloc/cudaMemcpy/cudaFree with
cudaMallocAsync/cudaMemcpyAsync/cudaFreeAsync on the KV copy stream and removes
the intermediate copy fence, keeping the final stream fence. Native KV tests
pass across TP, replicated KV heads and invalid mappings; three async reset
cycles pass with exact TP2->TP4 KV/hidden states and no consumer SHM mappings.

| Configuration | First creation-to-token samples ms | Reload samples ms |
| --- | --- | --- |
| Original | 809.6 / 793.8 | 769.2 / 766.3 |
| Stream allocation candidate | 898.6 / 1016.7 | 843.7 / 882.9 |

No end-to-end benefit was verified. Source and native binary restored to the
original implementation. This is not a retained production fallback. Small
sequential samples do not identify the mechanism of the regression. Raw data
and artifacts are preserved in fa3_kv_stream_allocation_results.json.
