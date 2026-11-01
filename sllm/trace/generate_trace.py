import os
import sys
import pickle
from prompts import get_prompts
from trace import Trace
import random
from random import sample

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from util import join_split_model_id

# How many types of instances of each model
model_list = {
    "chatbot": {
        "Qwen3-8B": 32,
        "Qwen3-32B": 8,
    },
    "code": {
        "Qwen3-8B": 32,
        "Qwen3-32B": 8,
    },
    "summarization": {
        "Qwen3-8B": 32,
        "Qwen3-32B": 8,
    }
}

if __name__ == '__main__':
    random.seed(998244353)
    trace_name = "azure_v2"
    trace_dir = sys.argv[1]
    trace = Trace(trace_name, trace_dir)

    expended_model_list = []
    cur_model_index = {}
    for task_type, model_dict in model_list.items():
        for model_id, model_num in model_dict.items():
            if model_id not in cur_model_index:
                cur_model_index[model_id] = 0
                cur_index = 0
            else:
                cur_index = cur_model_index[model_id]
            for index in range(model_num):
                expended_model_list.append(join_split_model_id(model_id, index + cur_index))
            cur_model_index[model_id] += model_num

    # Generate a 1-day trace with Gamma distribution
    replays = trace.replay(expended_model_list,
                           end_time="1.5.0",
                           model_mapping_strategy="round_robin",
                           arrival_distribution="gamma",
                           cv_scale_factor=8)

    all_prompts = {
        "chatbot": get_prompts('gliang1001/ShareGPT_V3_unfiltered_cleaned_split'),
        "code": get_prompts('modelscope/humaneval'),
        "summarization": get_prompts('ZhipuAI/LongBench')
    }

    requests = []
    cur_model_index = {}
    for task_type, model_dict in model_list.items():
        for model_id, model_num in model_dict.items():
            if model_id not in cur_model_index:
                cur_model_index[model_id] = 0
                cur_index = 0
            else:
                cur_index = cur_model_index[model_id]
            for index in range(model_num):
                model_name = join_split_model_id(model_id, cur_index + index)
                replay = replays[model_name]
                arrivals = replay.arrivals
                arrivals /= 14.0    # map 14-day trace into 1-day
                prompts = [sample(all_prompts[task_type], 1)[0] for _ in range(len(arrivals))]
                for arrival, prompt in zip(arrivals, prompts):
                    new_req = (arrival, model_name, prompt)
                    requests.append(new_req)
            cur_model_index[model_id] += model_num
    
    print(f"Generated {len(requests)} requests in total.")

    with open('trace.pkl', 'wb') as f:
        pickle.dump(requests, f)

