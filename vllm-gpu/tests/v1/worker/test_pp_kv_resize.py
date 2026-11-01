from types import SimpleNamespace
import pytest
import torch
from vllm.v1.worker.cold_start_transition import resize_kv_cache_for_uniform


@pytest.mark.parametrize('new_blocks', [3, 8])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('strided', [False, True])
def test_resize_preserves_keys_values_and_shares_allocation(new_blocks, device, strided):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    old_blocks = 5
    page_bytes = 2 * 3 * 4 * 2  # K/V, heads, head size, BF16 bytes.
    def reshape(name, raw):
        if strided:
            return raw.view(torch.bfloat16).view(2, 3, -1, 4).permute(0, 2, 1, 3)
        return raw.view(torch.bfloat16).view(2, -1, 3, 4)
    raw = {f'model.layers.{layer}.self_attn':
           torch.empty(old_blocks * page_bytes, dtype=torch.int8, device=device)
           for layer in range(3)}
    caches = {name: reshape(name, value) for name, value in raw.items()}
    for layer, value in enumerate(caches.values()):
        value[0].fill_(layer + 1)
        value[1].fill_(layer + 10)
    config = SimpleNamespace(num_blocks=old_blocks,
        kv_cache_groups=[SimpleNamespace(layer_names=list(caches))])
    caches, raw = resize_kv_cache_for_uniform(caches, raw, config, [0, 1],
        old_blocks, new_blocks, page_bytes, reshape)
    assert len(caches) == 2
    assert len({value.untyped_storage().data_ptr() for value in raw.values()}) == 1
    for layer, value in enumerate(caches.values()):
        count = min(old_blocks, new_blocks)
        assert torch.all(value[0, :count] == layer + 1)
        assert torch.all(value[1, :count] == layer + 10)
    assert config.num_blocks == new_blocks
    assert len(config.kv_cache_groups[0].layer_names) == 2


def test_missing_uniform_layer_fails_before_mutation():
    caches = {'model.layers.0.self_attn': torch.zeros(2, 1, 1)}
    raw = {}
    with pytest.raises(ValueError, match='Missing uniform'):
        resize_kv_cache_for_uniform(caches, raw, SimpleNamespace(), [1],
                                   1, 2, 4, None)
    assert len(caches) == 1
