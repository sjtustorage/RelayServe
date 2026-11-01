import inspect
import os

def _patch_hf_datasets_as_dataset_for_modelscope():
    """ModelScope forwards ``verification_mode`` to ``DatasetBuilder.as_dataset``; HF ``datasets`` 4.x no longer accepts it."""
    try:
        from datasets.builder import DatasetBuilder
    except ImportError:
        return
    _orig = DatasetBuilder.as_dataset
    if "verification_mode" in inspect.signature(_orig).parameters:
        return

    def as_dataset(self, split=None, in_memory=False, verification_mode=None, **kwargs):
        return _orig(self, split=split, in_memory=in_memory)

    DatasetBuilder.as_dataset = as_dataset


_patch_hf_datasets_as_dataset_for_modelscope()

from modelscope import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
if hasattr(tokenizer, "deprecation_warnings"):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

code_prompt = "You are a code assistant, specialized in helping with coding tasks. Your role is to provide concise, correct, and efficient code solutions. If the user has a specific coding problem, you should offer a solution in the relevant programming language, explain the approach briefly, and suggest improvements if necessary. If the user asks for clarification or has an incomplete problem statement, prompt them to provide more details. Prioritize fast, clear, and useful code snippets over lengthy explanations. Support a wide range of programming languages and concepts, and ensure your responses are easy to understand."
chat_prompt = "You are a chat assistant, designed to engage in friendly, helpful, and informative conversations. Your role is to understand the user's queries and respond in a natural, conversational manner. Provide relevant, concise, and accurate information based on the user's needs. Ensure that your responses are consistent and relevant to the ongoing discussion."
sum_prompt = "You are a text summarizer. Your role is to read the provided text and any additional reference materials, then generate a concise and accurate summary. The summary should capture the most important points and key insights from the text, ensuring that it is coherent and easy to understand. Use the auxiliary materials to enrich your summary and provide a more complete understanding of the content. If the text is long or complex, focus on the main ideas, themes, or arguments. Be sure to provide a well-organized and structured summary, highlighting any critical or notable aspects."

code_prompt_len = len(tokenizer(code_prompt)['input_ids']) + 4
chat_prompt_len = len(tokenizer(chat_prompt)['input_ids']) + 4
sum_prompt_len = len(tokenizer(sum_prompt)['input_ids']) + 4


def _truncate_enabled() -> bool:
    return os.environ.get("TRACE_NO_TRUNCATE", "").strip().lower() not in ("1", "true", "yes")


def _max_context_tokens() -> int:
    """Hard cap for total prompt (system + history + user) when truncation is on.

    Default 32k tokens. Override with ``TRACE_PROMPT_MAX_LEN``. Disable truncation with
    ``TRACE_NO_TRUNCATE=1`` (unbounded; can exceed GPU ``max_model_len``).
    """
    if not _truncate_enabled():
        return 2**30
    v = os.environ.get("TRACE_PROMPT_MAX_LEN", "").strip()
    if v:
        return max(32, int(v))
    return 32768


def get_prompts_from_ds(ds):
    prompts = []
    tot_len = 0
    max_model_len = _max_context_tokens()
    do_trunc = _truncate_enabled()

    for example in ds:
        if "prompt" in example:
            # humaneval (code)
            input = example['prompt']
            if do_trunc:
                input_ids = tokenizer(input, max_length=max_model_len-code_prompt_len-4, truncation=True)['input_ids']
            else:
                input_ids = tokenizer(input, truncation=False)['input_ids']
            prompt = tokenizer.decode(input_ids)
            tot_len += len(input_ids) + sum_prompt_len + 4
            prompts.append([{"role": "system", "content": code_prompt}, {"role": "user", "content": prompt}])
            continue
        if 'conversations' in example:
            # sharegpt (chat)
            conversation = example['conversations']
            history_openai_format = [{"role": "system", "content": chat_prompt}]
            content_tokens = chat_prompt_len
            for talk in conversation:
                role = talk['from']
                inst = talk['value']
                num_tokens_inst = len(tokenizer(inst)['input_ids']) + 4
                if role == 'human':
                    if do_trunc and content_tokens + num_tokens_inst >= max_model_len:
                        # truncate and break
                        inst_ids = tokenizer(inst, max_length=max_model_len-content_tokens-4, truncation=True)['input_ids']
                        inst = tokenizer.decode(inst_ids)
                        tot_len += len(inst_ids) + 4 + content_tokens
                        history_openai_format.append({"role": "user", "content": inst})
                        prompts.append(history_openai_format)
                        break
                    content_tokens += num_tokens_inst
                    history_openai_format.append({"role": "user", "content": inst})
                    tot_len += content_tokens
                    prompts.append(history_openai_format.copy())
                else:
                    content_tokens += num_tokens_inst
                    if do_trunc and content_tokens + 4 >= max_model_len:
                        break
                    history_openai_format.append({"role": "assistant", "content": inst})
        else:
            # longbench (summary)
            input = example['input'] + " \nAuxiliary materials: " + example['context']
            if do_trunc:
                input_ids = tokenizer(input, max_length=max_model_len-sum_prompt_len-4, truncation=True)['input_ids']
            else:
                input_ids = tokenizer(input, truncation=False)['input_ids']
            prompt = tokenizer.decode(input_ids)
            tot_len += len(input_ids) + sum_prompt_len + 4
            prompts.append([{"role": "system", "content": sum_prompt}, {"role": "user", "content": prompt}])

    print(f"average prompt length = {tot_len / len(prompts)}")

    return prompts

def get_prompts(dataset_name: str):
    if "LongBench" in dataset_name:
        subsets = ["gov_report", "qmsum", "vcsum"]
        split = 'test'
    elif "ShareGPT" in dataset_name:
        subsets = ["default"]
        split = 'train'
    else:
        subsets = ["openai_humaneval"]
        split = 'test'
    
    prompts = []
    for subset in subsets:
        ds = MsDataset.load(dataset_name, subset_name=subset, split=split, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS, trust_remote_code=True)
        new_prompts = get_prompts_from_ds(ds)
        for new_prompt in new_prompts:
            prompts.append(new_prompt)
    
    print(f"generated {len(prompts)} prompts")
    
    return prompts