import pytest
import torch
import amx_gemm_bf16_ext as amx


@pytest.mark.parametrize("n,k", [(32, 32), (64, 96), (4096, 5120),
                                  (12800, 5120), (5120, 12800)])
def test_panel_packing_matches_transposed_reference_and_preserves_storage(n, k):
    torch.set_num_threads(4)
    original = torch.randn(n, k, generator=torch.Generator().manual_seed(31)).bfloat16()
    expected = amx.pack_weight(original.t().contiguous()).flatten()
    # Odd storage offset covers SHM-like destinations without vector alignment.
    storage = torch.full((n * k + 2,), 17, dtype=torch.bfloat16)
    target = storage[1:-1].view(n, k)
    target.copy_(original)
    pointer = target.data_ptr()
    amx.pack_weight_nk(target)
    assert target.data_ptr() == pointer
    assert torch.equal(target.flatten(), expected)
    assert storage[0] == 17 and storage[-1] == 17


@pytest.mark.parametrize("shape", [(31, 32), (32, 33), (0, 32), (32,)])
def test_invalid_packing_shape_rejected_before_write(shape):
    target = torch.full(shape, 17, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        amx.pack_weight_nk(target)
    assert torch.all(target == 17)
