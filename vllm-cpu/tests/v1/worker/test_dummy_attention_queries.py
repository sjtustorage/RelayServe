"""Dummy attention must receive queries, including after a previous batch."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker import gpu_model_runner as module


@pytest.mark.parametrize('tokens,decode,expected', [
    (2, True, [0, 1, 2]),
    (3, False, [0, 1, 3]),
    (5, False, [0, 2, 5]),
])
@pytest.mark.parametrize('initial_offset', [0, 91])
def test_dummy_attention_receives_complete_query_ranges(monkeypatch, tokens,
                                                       decode, expected,
                                                       initial_offset):
    def buffer(size, dtype):
        return CpuGpuBuffer(size, dtype=dtype, device=torch.device('cpu'),
                            pin_memory=False)

    query = buffer(3, torch.int32)
    query.cpu.fill_(initial_offset)  # Fresh buffers and stale prior batches.
    seen = []

    def build(metadata):
        seen.append(metadata)
        np.testing.assert_array_equal(metadata.query_start_loc_cpu.numpy(),
                                      expected)
        assert metadata.query_start_loc.tolist() == expected
        assert metadata.num_actual_tokens == expected[-1]
        return metadata

    monkeypatch.setattr(module, 'get_pp_group', lambda: SimpleNamespace(
        is_first_rank=True))
    monkeypatch.setattr(module, 'set_forward_context',
                        lambda *args, **kwargs: nullcontext())
    runner = SimpleNamespace(
        get_dp_padding=lambda _: (0, None), uniform_decode_query_len=1,
        scheduler_config=SimpleNamespace(max_num_seqs=2,
                                         max_num_batched_tokens=8),
        query_start_loc=query, seq_lens=buffer(2, torch.int32),
        max_model_len=32,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(
            layer_names=['layer'])]),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.zeros(2, dtype=torch.int32),
            block_table=[SimpleNamespace(
                get_device_tensor=lambda: torch.zeros((2, 1), dtype=torch.int32),
                slot_mapping=torch.zeros(8, dtype=torch.int64))]),
        attn_groups=[[SimpleNamespace(metadata_builder=SimpleNamespace(
            build_for_cudagraph_capture=build))]],
        maybe_dummy_run_with_lora=lambda *args: nullcontext(), lora_config=None,
        supports_mm_inputs=False, input_ids=buffer(8, torch.int32),
        _init_model_kwargs=lambda _: {}, uses_mrope=False,
        positions=buffer(8, torch.int64), maybe_randomize_inputs=lambda _: nullcontext(),
        vllm_config=SimpleNamespace(),
        model=lambda **kwargs: torch.zeros((tokens, 4)),
        use_aux_hidden_state_outputs=False, speculative_config=None)
    output, last = module.GPUModelRunner._dummy_run(
        runner, tokens, force_attention=True, uniform_decode=decode,
        skip_eplb=True)
    assert len(seen) == 1
    assert output.shape == (tokens, 4)
    assert last.shape == (2, 4)
