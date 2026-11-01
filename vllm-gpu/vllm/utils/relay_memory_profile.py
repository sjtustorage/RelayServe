"""Model/topology-specific offline SHM memory budgets.

The checkpoint directory names identify SHM producers, not model architecture.
Profiles therefore match checkpoint configuration content and execution settings.
"""
import hashlib
import json
import os
from pathlib import Path

import torch


def memory_profile_identity(config, device):
    model = config.model_config
    checkpoint_config = Path(model.model) / 'config.json'
    if not checkpoint_config.is_file():
        raise ValueError('Offline SHM memory profiles require a local config.json')
    data = json.loads(checkpoint_config.read_text())
    # These metadata fields do not change storage or inference operations.
    for name in ('_name_or_path', '_commit_hash', 'transformers_version'):
        data.pop(name, None)
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    properties = torch.cuda.get_device_properties(device)
    identity = dict(
        config_sha256=hashlib.sha256(canonical).hexdigest(),
        dtype=str(model.dtype), kv_dtype=config.cache_config.cache_dtype,
        block_size=config.cache_config.block_size,
        enable_prefix_caching=config.cache_config.enable_prefix_caching,
        quantization=getattr(model, 'quantization', None),
        lora_enabled=getattr(config, 'lora_config', None) is not None,
        speculative_enabled=getattr(config, 'speculative_config', None) is not None,
        attention_backend=os.environ.get('VLLM_ATTENTION_BACKEND'),
        tp=config.parallel_config.tensor_parallel_size,
        pp=config.parallel_config.pipeline_parallel_size,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        max_num_seqs=config.scheduler_config.max_num_seqs,
        max_model_len=model.max_model_len,
        enforce_eager=model.enforce_eager,
        device_name=properties.name,
        device_capability=[properties.major, properties.minor],
        device_total_memory=properties.total_memory,
        torch_version=torch.__version__,
        attention_version=os.environ.get('VLLM_FLASH_ATTN_VERSION'),
    )
    inventory = getattr(config.cache_config, '_relay_group_inventory', None)
    if config.parallel_config.pipeline_parallel_size > 1:
        # Different stage boundaries change both model and activation memory.
        # Preserve PP=1 profile identities while distinguishing PP layouts.
        identity['pp_layer_partition'] = os.environ.get('VLLM_PP_LAYER_PARTITION')
    if inventory is not None:
        identity['resident_group_inventory'] = inventory
    return identity


def read_non_kv_memory(profile_path, identity):
    """Reject missing/mismatched budgets rather than applying another TP's data."""
    path = Path(profile_path)
    document = json.loads(path.read_text())
    if (not isinstance(document, dict) or document.get('version') != 1
            or not isinstance(document.get('profiles'), list)):
        raise ValueError(f'Invalid offline SHM memory profile schema: {path}')
    matches = [entry for entry in document['profiles']
               if isinstance(entry, dict) and entry.get('identity') == identity]
    if len(matches) != 1:
        raise ValueError(f'Expected one matching offline SHM memory profile in {path}; '
                         f'found {len(matches)} for TP={identity["tp"]}, '
                         f'PP={identity["pp"]}, chunk={identity["max_num_batched_tokens"]}')
    value = matches[0].get('non_kv_memory_bytes')
    if type(value) is not int or value <= 0:
        raise ValueError(f'Invalid non-KV memory budget in {path}')
    return value


def find_memory_profile(model_path):
    explicit = os.environ.get('VLLM_SHM_MEMORY_PROFILE')
    if explicit:
        return Path(explicit)
    checkpoint_profile = Path(model_path) / 'relay_memory_profile.json'
    return checkpoint_profile if checkpoint_profile.is_file() else None
