import tempfile
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from shm_manager.safetensors_reader import load_qwen_regions
from shm_manager._C import read_file_regions


def test_vocab_padding_and_rank_slices():
    config = SimpleNamespace(num_key_value_heads=2)
    embedding = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    norm = torch.arange(4).to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as directory:
        save_file({"model.embed_tokens.weight": embedding,
                   "model.norm.weight": norm}, str(Path(directory) / "model.safetensors"))
        for rank in range(2):
            destination = torch.full((4, 4), float("nan"), dtype=torch.bfloat16)
            norm_destination = torch.empty_like(norm)
            load_qwen_regions(directory, {"model.embed_tokens.weight": destination,
                                          "model.norm.weight": norm_destination},
                              config, rank, 2, 2)
            expected = torch.zeros_like(destination)
            values = embedding[rank * 4:(rank + 1) * 4]
            expected[:values.shape[0]] = values
            assert torch.equal(destination, expected)
            assert torch.equal(norm_destination, norm)


def test_reader_binding_preserves_caller_affinity():
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        import pytest
        pytest.skip("requires two permitted CPUs")
    original = torch.arange(1024, dtype=torch.int32)
    destination = torch.empty_like(original)
    with tempfile.TemporaryDirectory() as directory:
        filename = Path(directory) / "unaligned.bin"
        filename.write_bytes(b"header!" + original.numpy().tobytes())
        try:
            os.sched_setaffinity(0, {allowed[0]})
            read_file_regions([(str(filename), 7, destination.data_ptr(),
                                destination.numel() * destination.element_size())],
                              2, str(allowed[1]))
            assert os.sched_getaffinity(0) == {allowed[0]}
            assert torch.equal(destination, original)
        finally:
            os.sched_setaffinity(0, allowed)


def test_invalid_reader_binding_rejected_before_write():
    import pytest
    destination = torch.full((8,), 17, dtype=torch.int32)
    with pytest.raises(ValueError):
        read_file_regions([("/not-needed", 0, destination.data_ptr(), 32)],
                          4, "not-a-cpu")
    assert torch.all(destination == 17)
