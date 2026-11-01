import json
import os
from pathlib import Path

os.environ['VLLM_USE_V1'] = '1'
from vllm import LLM, SamplingParams

checkpoint = '/workspace/models/Qwen2.5-72B'
model = LLM(model=checkpoint, tensor_parallel_size=4,
            dtype='bfloat16', load_format='safetensors',
            enforce_eager=True, max_model_len=4104,
            max_num_batched_tokens=4096, max_num_seqs=2,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            gpu_memory_utilization=0.5, skip_tokenizer_init=True,
            block_size=32)
outputs = {}
for length in (128, 4096):
    prompt = ([151643] + [9707, 11, 358, 1097]
              * ((length + 3) // 4))[:length]
    output = model.generate([{'prompt_token_ids': prompt}],
                            SamplingParams(temperature=0, max_tokens=8,
                                           ignore_eos=True),
                            use_tqdm=False)[0].outputs[0]
    outputs[str(length)] = list(output.token_ids)
    print(json.dumps({'independent_safetensors_prompt_tokens': length,
                      'tokens': outputs[str(length)]}), flush=True)
Path('/tmp/qwen25-72b-independent-safetensors-reference.json').write_text(
    json.dumps(outputs, indent=2))
