"""Check the native diagnostic against known contiguous and strided pages."""
import torch

from shm_manager._C import ModelManager


def test_source_coverage_counts_strides_and_page_union(monkeypatch, capfd):
    monkeypatch.setenv("VLLM_PROFILE_SHM_ATTACH", "1")
    manager = ModelManager(2, 2, 0, 4, torch.cuda.current_device())
    manager.update_total_layers(0)
    manager.update_manager_info({0: 1 << 44}, {0: 1 << 48})
    # The first task occupies one page, repeated metadata must not double
    # count its union. The column task occupies two separated pages.
    for _ in range(2):
        manager.generate_tensor_metadata(0, 0, [128, 64], [32, 64],
                                         [0], 0, 0, -1, -1, True, "rows")
    manager.generate_tensor_metadata(16384, 8192, [64, 256], [64, 64],
                                     [0], 1, 0, -1, -1, True, "columns")
    try:
        manager.generate_copy_tasks()
        output = capfd.readouterr().err
        assert "raw_ranges=4 logical_bytes=16384 pages=3 merged_ranges=3" in output
        assert "gap_bytes=65536 covered_bytes=28672 ranges=1" in output
        assert manager.copy_source_ranges() == {0: [(0, 4096), (16384, 4096), (24576, 4096)]}
        assert manager.copy_source_ranges(65536) == {0: [(0, 28672)]}
    finally:
        manager.close()


def test_gap_coalescing_keeps_producer_ranks_separate():
    manager = ModelManager(2, 2, 0, 1, torch.cuda.current_device())
    manager.update_total_layers(0)
    # Adjacent source mappings must remain independently registerable even
    # when the selected gap threshold would join their absolute addresses.
    manager.update_manager_info({0: 1 << 44, 1: (1 << 44) + 4096}, {0: 1 << 48})
    manager.generate_tensor_metadata(0, 0, [64, 64], [64, 64],
                                     [0, 1], 0, 0, -1, -1, True, "rows")
    try:
        manager.generate_copy_tasks()
        assert manager.copy_source_ranges(1048576) == {0: [(0, 4096)], 1: [(0, 4096)]}
    finally:
        manager.close()
