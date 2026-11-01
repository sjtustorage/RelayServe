#include "gemm.hpp"

namespace amx_bf16 {

void LoadTileConfig() {
    if (!set_tiledata_use()) {
        std::cerr << "AMX init failed on thread " << omp_get_thread_num() << std::endl;
        return;
    }

    TileConfig tile_data;
    for (int i = 0; i < 4; ++i) {
        tile_data.set_row_col(i, MAX_ROWS, MAX_COLS_FP32 * sizeof(float));
    }
    for (int i = 4; i < 6; ++i) {
        tile_data.set_row_col(i, MAX_ROWS, MAX_COLS_BF16 * sizeof(bf16_t));
    }
    for (int i = 6; i < 8; ++i) {
        tile_data.set_row_col(i, MAX_COLS_BF16 / KPACK_BF16, MAX_COLS_FP32 * KPACK_BF16 * sizeof(bf16_t));
    }
    _tile_loadconfig(&tile_data);
}

void LoadMTailTileConfig(int top_rows, int bottom_rows) {
    TileConfig tile_data;
    tile_data.set_row_col(C00, top_rows, MAX_COLS_FP32 * sizeof(float));
    tile_data.set_row_col(C01, top_rows, MAX_COLS_FP32 * sizeof(float));
    tile_data.set_row_col(C10, bottom_rows,bottom_rows > 0 ? MAX_COLS_FP32 * sizeof(float) : 0);
    tile_data.set_row_col(C11, bottom_rows,bottom_rows > 0 ? MAX_COLS_FP32 * sizeof(float) : 0);
    tile_data.set_row_col(A0, top_rows, MAX_COLS_BF16 * sizeof(bf16_t));
    tile_data.set_row_col(A1, bottom_rows, bottom_rows > 0 ? MAX_COLS_BF16 * sizeof(bf16_t) : 0);
    for (int i = 6; i < 8; ++i) {
        tile_data.set_row_col(i, MAX_COLS_BF16 / KPACK_BF16, MAX_COLS_FP32 * KPACK_BF16 * sizeof(bf16_t));
    }
    _tile_loadconfig(&tile_data);
}

void RunAMX32x32(AMXBlockTask* task) {
    const int lda = task->lda;
    const int ldc = task->ldc;
    const int lda_bytes = lda * (int)sizeof(bf16_t);
    const int ldc_bytes = ldc * (int)sizeof(float);
    const int pf_a_dist = task->pf_a_dist;
    const int pf_b_dist = task->pf_b_dist;
    const int pf_c_dist = task->pf_c_dist;

    for (int i = 0; i < task->rows; i += M_STEP) {
        int next_a_i = i + M_STEP;
        bool has_next_a = next_a_i < task->rows;
        const int8_t* pf_A = nullptr;
        size_t pf_A_size = M_STEP * task->depth * sizeof(bf16_t);
        size_t current_A_size = 0;
        int prefetch_a_count = 0;
        int next_a_k = 0;
        if (task->prefetch_a && has_next_a)
            pf_A = reinterpret_cast<const int8_t*>(&task->a[OFFSET2D(next_a_i, next_a_k, lda)]);

        for (int j = 0; j < task->cols; j += N_STEP) {
            const bf16_t* packed_b_panel_ptr =
                task->packed_weight + (size_t)(j / N_STEP) * task->packed_weight_k * N_STEP;

            int next_b_j = j + N_STEP;
            bool has_next_b = next_b_j < task->cols;
            const int8_t* pf_B = nullptr;
            size_t pf_B_size = N_STEP * task->depth * sizeof(bf16_t);
            size_t current_B_size = 0;
            if (task->prefetch_b && has_next_b)
                pf_B = reinterpret_cast<const int8_t*>(task->packed_weight + (size_t)(j / N_STEP) * task->packed_weight_k * N_STEP);

            if (task->zero_accumulator) {
                clear_4_tileC();
            } else {
                load_4_tileC_l2(&task->c[OFFSET2D(i, j, ldc)], ldc);
            }

            int next_c_i = (j + N_STEP == task->cols) ? i + M_STEP : i;
            int next_c_j = (j + N_STEP == task->cols) ? 0 : j + N_STEP;
            bool has_next_c = (next_c_i < task->rows) && (next_c_j < task->cols);
            const int8_t* pf_C = nullptr;
            size_t pf_C_size = M_STEP * N_STEP * sizeof(float);
            size_t current_C_size = 0;
            if (task->prefetch_c && has_next_c)
                pf_C = reinterpret_cast<const int8_t*>(&task->c[OFFSET2D(next_c_i, next_c_j, ldc)]);
                            
            for (int k = 0; k < task->depth; k += K_STEP) {
                const bf16_t* A_base = &task->a[OFFSET2D(i, k, lda)];

                _tile_loadd(A0, A_base, lda_bytes);
                _tile_loadd(A1, A_base + MAX_ROWS * lda, lda_bytes);
                _tile_stream_loadd(B0, packed_b_panel_ptr, MIN_STRIDE);
                _tile_stream_loadd(B1, packed_b_panel_ptr + TILE_SIZE_BF16, MIN_STRIDE);

                if (pf_A && current_A_size < pf_A_size) {
                    for (int pa = 0; pa < pf_a_dist; pa++) {
                        _mm_prefetch(pf_A, _MM_HINT_T2);
                        pf_A += lda_bytes;
                        current_A_size += CACHELINE_SIZE;
                        prefetch_a_count++;
                    }
                    if (prefetch_a_count >= 2 * MAX_ROWS) {
                        next_a_k += K_STEP;
                        pf_A = reinterpret_cast<const int8_t*>(&task->a[OFFSET2D(next_a_i, next_a_k, lda)]);
                        prefetch_a_count = 0;
                    }
                }

                if (pf_B && current_B_size < pf_B_size) {
                    for (int pb = 0; pb < pf_b_dist; pb++) {
                        _mm_prefetch(pf_B, _MM_HINT_T2);
                        pf_B += TILE_SIZE_BF16;
                        current_B_size += TILE_SIZE_BF16;
                    }
                }

                _tile_dpbf16ps(C00, A0, B0);

                if (pf_B && current_B_size < pf_B_size) {
                    for (int pb = 0; pb < pf_b_dist; pb++) {
                        _mm_prefetch(pf_B, _MM_HINT_T2);
                        pf_B += TILE_SIZE_BF16;
                        current_B_size += TILE_SIZE_BF16;
                    }
                }

                if (pf_A && current_A_size < pf_A_size) {
                    for (int pa = 0; pa < pf_a_dist; pa++) {
                        _mm_prefetch(pf_A, _MM_HINT_T2);
                        pf_A += lda_bytes;
                        current_A_size += CACHELINE_SIZE;
                        prefetch_a_count++;
                    }
                    if (prefetch_a_count >= 2 * MAX_ROWS) {
                        next_a_k += K_STEP;
                        pf_A = reinterpret_cast<const int8_t*>(&task->a[OFFSET2D(next_a_i, next_a_k, lda)]);
                        prefetch_a_count = 0;
                    }
                }
                _tile_dpbf16ps(C01, A0, B1);

                if (pf_B && current_B_size < pf_B_size) {
                    for (int pb = 0; pb < pf_b_dist; pb++) {
                        _mm_prefetch(pf_B, _MM_HINT_T2);
                        pf_B += TILE_SIZE_BF16;
                        current_B_size += TILE_SIZE_BF16;
                    }
                }

                if (pf_C && current_C_size < pf_C_size) {
                    for (int pc = 0; pc < pf_c_dist; pc++) {
                        if (current_C_size >= pf_C_size) break;
                        _mm_prefetch(pf_C, _MM_HINT_T2);
                        _mm_prefetch(pf_C + CACHELINE_SIZE, _MM_HINT_T2);
                        pf_C += ldc_bytes;
                        current_C_size += 2 * CACHELINE_SIZE;
                    }
                }

                _tile_dpbf16ps(C10, A1, B0);

                if (pf_B && current_B_size < pf_B_size) {
                    for (int pb = 0; pb < pf_b_dist; pb++) {
                        _mm_prefetch(pf_B, _MM_HINT_T2);
                        pf_B += TILE_SIZE_BF16;
                        current_B_size += TILE_SIZE_BF16;
                    }
                }

                if (pf_C && current_C_size < pf_C_size) {
                    for (int pc = 0; pc < pf_c_dist; pc++) {
                        if (current_C_size >= pf_C_size) break;
                        _mm_prefetch(pf_C, _MM_HINT_T2);
                        _mm_prefetch(pf_C + CACHELINE_SIZE, _MM_HINT_T2);
                        pf_C += ldc_bytes;
                        current_C_size += 2 * CACHELINE_SIZE;
                    }
                }

                _tile_dpbf16ps(C11, A1, B1);

                packed_b_panel_ptr += 2 * TILE_SIZE_BF16;
            }

            store_4_tileC_l1(&task->c[OFFSET2D(i, j, ldc)], ldc);
        }
    }
}

KernelProfileCounters ProfileAMX32x32(AMXBlockTask* task) {
    KernelProfileCounters counters;
    counters.tsc_overhead_cycles = calibrate_tsc_overhead();

    const int lda = task->lda;
    const int ldc = task->ldc;
    const int lda_bytes = lda * (int)sizeof(bf16_t);
    const int ldc_bytes = ldc * (int)sizeof(float);
    const int pf_a_dist = task->pf_a_dist;
    const int pf_b_dist = task->pf_b_dist;
    const int pf_c_dist = task->pf_c_dist;

    uint64_t tile_loadd_samples = 0;
    uint64_t tile_stream_samples = 0;
    uint64_t tile_dpbf16ps_samples = 0;
    uint64_t prefetch_samples = 0;

    for (int i = 0; i < task->rows; i += M_STEP) {
        const int next_a_i = i + M_STEP;
        const bool has_next_a = next_a_i < task->rows;
        const int8_t* pf_A = nullptr;
        const size_t pf_A_size = M_STEP * task->depth * sizeof(bf16_t);
        size_t current_A_size = 0;
        int prefetch_a_count = 0;
        int next_a_k = 0;
        if (task->prefetch_a && has_next_a) {
            pf_A = reinterpret_cast<const int8_t*>(&task->a[OFFSET2D(next_a_i, next_a_k, lda)]);
        }

        for (int j = 0; j < task->cols; j += N_STEP) {
            const bf16_t* packed_b_panel_ptr =
                task->packed_weight + (size_t)(j / N_STEP) * task->packed_weight_k * N_STEP;

            if (task->zero_accumulator) {
                clear_4_tileC();
            } else {
                const uint64_t t0 = read_tsc_begin();
                load_4_tileC_l2(&task->c[OFFSET2D(i, j, ldc)], ldc);
                const uint64_t t1 = read_tsc_end();
                counters.tile_stream_loadd_cycles += (t1 - t0);
                counters.tile_stream_loadd_count += 4;
                ++tile_stream_samples;
            }

            const int next_c_i = (j + N_STEP == task->cols) ? i + M_STEP : i;
            const int next_c_j = (j + N_STEP == task->cols) ? 0 : j + N_STEP;
            const bool has_next_c = (next_c_i < task->rows) && (next_c_j < task->cols);
            const int8_t* pf_C = nullptr;
            const size_t pf_C_size = M_STEP * N_STEP * sizeof(float);
            size_t current_C_size = 0;
            if (task->prefetch_c && has_next_c) {
                pf_C = reinterpret_cast<const int8_t*>(&task->c[OFFSET2D(next_c_i, next_c_j, ldc)]);
            }

            for (int k = 0; k < task->depth; k += K_STEP) {
                ++counters.k_iterations;
                const bf16_t* A_base = &task->a[OFFSET2D(i, k, lda)];

                {
                    const uint64_t t0 = read_tsc_begin();
                    _tile_loadd(A0, A_base, lda_bytes);
                    _tile_loadd(A1, A_base + MAX_ROWS * lda, lda_bytes);
                    const uint64_t t1 = read_tsc_end();
                    counters.tile_loadd_cycles += (t1 - t0);
                    counters.tile_loadd_count += 2;
                    ++tile_loadd_samples;
                }

                {
                    const uint64_t t0 = read_tsc_begin();
                    _tile_stream_loadd(B0, packed_b_panel_ptr, MIN_STRIDE);
                    _tile_stream_loadd(B1, packed_b_panel_ptr + TILE_SIZE_BF16, MIN_STRIDE);
                    const uint64_t t1 = read_tsc_end();
                    counters.tile_stream_loadd_cycles += (t1 - t0);
                    counters.tile_stream_loadd_count += 2;
                    ++tile_stream_samples;
                }

                if (pf_A && current_A_size < pf_A_size) {
                    const uint64_t t0 = read_tsc_begin();
                    for (int u = 0; u < pf_a_dist; ++u) {
                        _mm_prefetch(pf_A, _MM_HINT_T2);
                        pf_A += lda_bytes;
                        current_A_size += CACHELINE_SIZE;
                        ++prefetch_a_count;
                    }
                    const uint64_t t1 = read_tsc_end();
                    counters.prefetch_cycles += (t1 - t0);
                    counters.prefetch_count += pf_a_dist;
                    ++prefetch_samples;

                    if (prefetch_a_count >= 2 * MAX_ROWS) {
                        next_a_k += K_STEP;
                        pf_A = reinterpret_cast<const int8_t*>(&task->a[OFFSET2D(next_a_i, next_a_k, lda)]);
                        prefetch_a_count = 0;
                    }
                }

                {
                    const uint64_t t0 = read_tsc_begin();
                    _tile_dpbf16ps(C00, A0, B0);
                    const uint64_t t1 = read_tsc_end();
                    counters.tile_dpbf16ps_cycles += (t1 - t0);
                    ++counters.tile_dpbf16ps_count;
                    ++tile_dpbf16ps_samples;
                }

                if (pf_A && current_A_size < pf_A_size) {
                    const uint64_t t0 = read_tsc_begin();
                    for (int u = 0; u < pf_a_dist; ++u) {
                        _mm_prefetch(pf_A, _MM_HINT_T2);
                        pf_A += lda_bytes;
                        current_A_size += CACHELINE_SIZE;
                        ++prefetch_a_count;
                    }
                    const uint64_t t1 = read_tsc_end();
                    counters.prefetch_cycles += (t1 - t0);
                    counters.prefetch_count += pf_a_dist;
                    ++prefetch_samples;

                    if (prefetch_a_count >= 2 * MAX_ROWS) {
                        next_a_k += K_STEP;
                        pf_A = reinterpret_cast<const int8_t*>(&task->a[OFFSET2D(next_a_i, next_a_k, lda)]);
                        prefetch_a_count = 0;
                    }
                }

                {
                    const uint64_t t0 = read_tsc_begin();
                    _tile_dpbf16ps(C01, A0, B1);
                    const uint64_t t1 = read_tsc_end();
                    counters.tile_dpbf16ps_cycles += (t1 - t0);
                    ++counters.tile_dpbf16ps_count;
                    ++tile_dpbf16ps_samples;
                }

                if (pf_C && current_C_size < pf_C_size) {
                    const uint64_t t0 = read_tsc_begin();
                    for (int u = 0; u < pf_c_dist; ++u) {
                        if (current_C_size >= pf_C_size) break;
                        _mm_prefetch(pf_C, _MM_HINT_T2);
                        _mm_prefetch(pf_C + CACHELINE_SIZE, _MM_HINT_T2);
                        pf_C += ldc_bytes;
                        current_C_size += 2 * CACHELINE_SIZE;
                    }
                    const uint64_t t1 = read_tsc_end();
                    counters.prefetch_cycles += (t1 - t0);
                    counters.prefetch_count += 2 * pf_c_dist;
                    ++prefetch_samples;
                }

                {
                    const uint64_t t0 = read_tsc_begin();
                    _tile_dpbf16ps(C10, A1, B0);
                    const uint64_t t1 = read_tsc_end();
                    counters.tile_dpbf16ps_cycles += (t1 - t0);
                    ++counters.tile_dpbf16ps_count;
                    ++tile_dpbf16ps_samples;
                }

                if (pf_C && current_C_size < pf_C_size) {
                    const uint64_t t0 = read_tsc_begin();
                    for (int u = 0; u < pf_c_dist; ++u) {
                        if (current_C_size >= pf_C_size) break;
                        _mm_prefetch(pf_C, _MM_HINT_T2);
                        _mm_prefetch(pf_C + CACHELINE_SIZE, _MM_HINT_T2);
                        pf_C += ldc_bytes;
                        current_C_size += 2 * CACHELINE_SIZE;
                    }
                    const uint64_t t1 = read_tsc_end();
                    counters.prefetch_cycles += (t1 - t0);
                    counters.prefetch_count += 2 * pf_c_dist;
                    ++prefetch_samples;
                }

                {
                    const uint64_t t0 = read_tsc_begin();
                    _tile_dpbf16ps(C11, A1, B1);
                    const uint64_t t1 = read_tsc_end();
                    counters.tile_dpbf16ps_cycles += (t1 - t0);
                    ++counters.tile_dpbf16ps_count;
                    ++tile_dpbf16ps_samples;
                }

                packed_b_panel_ptr += 2 * TILE_SIZE_BF16;
            }

            store_4_tileC_l1(&task->c[OFFSET2D(i, j, ldc)], ldc);
        }
    }

    counters.tile_loadd_cycles =
        subtract_tsc_overhead(counters.tile_loadd_cycles, tile_loadd_samples, counters.tsc_overhead_cycles);
    counters.tile_stream_loadd_cycles =
        subtract_tsc_overhead(counters.tile_stream_loadd_cycles, tile_stream_samples, counters.tsc_overhead_cycles);
    counters.tile_dpbf16ps_cycles =
        subtract_tsc_overhead(counters.tile_dpbf16ps_cycles, tile_dpbf16ps_samples, counters.tsc_overhead_cycles);
    counters.prefetch_cycles =
        subtract_tsc_overhead(counters.prefetch_cycles, prefetch_samples, counters.tsc_overhead_cycles);

    return counters;
}

void RunAMXMTail(AMXBlockTask* task) {
    TORCH_CHECK(task->rows > 0 && task->rows < M_STEP,
                "RunAMXMTail expects 1 <= M < ", M_STEP, ", got ", task->rows);

    const int top_rows = MIN(task->rows, MAX_ROWS);
    const int bottom_rows = task->rows - top_rows;
    const bool has_bottom = bottom_rows > 0;
    const int lda = task->lda;
    const int ldc = task->ldc;
    const int lda_bytes = lda * (int)sizeof(bf16_t);

    LoadMTailTileConfig(top_rows, bottom_rows);

    for (int j = 0; j < task->cols; j += N_STEP) {
        const bf16_t* packed_b_panel_ptr =
            task->packed_weight + (size_t)(j / N_STEP) * task->packed_weight_k * N_STEP;

        if (task->zero_accumulator) {
            _tile_zero(C00);
            _tile_zero(C01);
            if (has_bottom) {
                _tile_zero(C10);
                _tile_zero(C11);
            }
        } else {
            load_tail_tile_c_l2(&task->c[OFFSET2D(0, j, ldc)], ldc, has_bottom);
        }

        for (int k = 0; k < task->depth; k += K_STEP) {
            const bf16_t* A_base = &task->a[OFFSET2D(0, k, lda)];
            _tile_loadd(A0, A_base, lda_bytes);
            _tile_stream_loadd(B0, packed_b_panel_ptr, MIN_STRIDE);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(B1, packed_b_panel_ptr + TILE_SIZE_BF16, MIN_STRIDE);
            _tile_dpbf16ps(C01, A0, B1);

            if (has_bottom) {
                _tile_loadd(A1, A_base + MAX_ROWS * lda, lda_bytes);
                _tile_dpbf16ps(C10, A1, B0);
                _tile_dpbf16ps(C11, A1, B1);
            }

            packed_b_panel_ptr += 2 * TILE_SIZE_BF16;
        }

        store_tail_tile_c_l1(&task->c[OFFSET2D(0, j, ldc)], ldc, has_bottom);
    }

    // Restore full-tile configuration for following AMX32x32 work on this thread.
    LoadTileConfig();
}

}  // namespace amx_bf16
