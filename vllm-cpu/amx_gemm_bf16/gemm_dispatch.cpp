#include "gemm.hpp"
#include <cstdlib>
#include <fcntl.h>

#define MSR_MISC_FEATURE_CONTROL 0x1A4

namespace amx_bf16 {

static int g_hw_prefetch_state = -1; // -1 = unknown, 0 = disabled, 1 = enabled

void set_hw_prefetch(bool enable) {
    int target = enable ? 1 : 0;
    if (g_hw_prefetch_state == target) return;

    uint64_t val = enable ? 0x0 : 0xF;
    #pragma omp parallel
    {
        int cpu_id = sched_getcpu();
        char path[64];
        snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu_id);
        int fd = open(path, O_WRONLY);
        if (fd >= 0) {
            pwrite(fd, &val, sizeof(val), MSR_MISC_FEATURE_CONTROL);
            close(fd);
        } else {
            #pragma omp single
            {
                fprintf(stderr,
                    "[warn] set_hw_prefetch: cannot open %s "
                    "(load msr module: modprobe msr; grant access: chmod o+rw /dev/cpu/*/msr)\n",
                    path);
            }
        }
    }
    g_hw_prefetch_state = target;
}

inline KernelProfileCounters DispatchBlockProfile(
        const bf16_t* a_block,
        const bf16_t* packed_weight_base,
        float* c_block,
        int block_rows,
        int block_cols,
        int full_k,
        int lda,
        int c_ldc,
        int k_block,
        int n_start,
        bool prefetch_a,
        bool prefetch_b,
        bool prefetch_c,
        int tid,
        int pf_a_dist = 2,
        int pf_b_dist = 2,
        int pf_c_dist = 1) {
    (void)prefetch_b;
    KernelProfileCounters counters;
    const BlockPartition row_partition = SplitRowBlock(block_rows);

    for (int k_slice_start = 0; k_slice_start < full_k; k_slice_start += k_block) {
        const int current_k_span = MIN(k_block, full_k - k_slice_start);
        const bf16_t* packed_weight_k_slice = GetPackedWeightPanelKSlice(
            packed_weight_base, full_k, k_slice_start, n_start);

        if (row_partition.full_rows > 0) {
            AMXBlockTask full_task = MakeAMXBlockTask(
                &a_block[OFFSET2D(0, k_slice_start, lda)],
                packed_weight_k_slice,
                &c_block[OFFSET2D(0, 0, c_ldc)],
                row_partition.full_rows,
                block_cols,
                current_k_span,
                full_k,
                lda,
                c_ldc,
                k_slice_start == 0,
                prefetch_a,
                prefetch_b,
                prefetch_c,
                tid,
                pf_a_dist,
                pf_b_dist,
                pf_c_dist);
            counters.MergeFrom(ProfileAMX32x32(&full_task));
        }

        if (row_partition.tail_rows > 0) {
            AMXBlockTask tail_task = MakeAMXBlockTask(
                &a_block[OFFSET2D(row_partition.full_rows, k_slice_start, lda)],
                packed_weight_k_slice,
                &c_block[OFFSET2D(row_partition.full_rows, 0, c_ldc)],
                row_partition.tail_rows,
                block_cols,
                current_k_span,
                full_k,
                lda,
                c_ldc,
                k_slice_start == 0,
                prefetch_a,
                prefetch_b,
                prefetch_c,
                tid,
                pf_a_dist,
                pf_b_dist,
                pf_c_dist);
            RunAMXMTail(&tail_task);
        }
    }
    return counters;
}


py::dict MakeEmptyProfileOutput() {
    py::dict out;
    out["tsc_overhead_cycles"] = py::int_(0);
    out["tile_loadd_cycles"] = py::int_(0);
    out["tile_stream_loadd_cycles"] = py::int_(0);
    out["tile_dpbf16ps_cycles"] = py::int_(0);
    out["prefetch_cycles"] = py::int_(0);
    out["tile_loadd_count"] = py::int_(0);
    out["tile_stream_loadd_count"] = py::int_(0);
    out["tile_dpbf16ps_count"] = py::int_(0);
    out["prefetch_count"] = py::int_(0);
    out["k_iterations"] = py::int_(0);
    out["avg_tile_loadd_cycles"] = py::float_(0.0);
    out["avg_tile_stream_loadd_cycles"] = py::float_(0.0);
    out["avg_tile_dpbf16ps_cycles"] = py::float_(0.0);
    out["avg_prefetch_cycles"] = py::float_(0.0);
    return out;
}

py::dict BuildProfileOutput(const KernelProfileCounters& counters) {
    py::dict out;
    out["tsc_overhead_cycles"] = py::int_(counters.tsc_overhead_cycles);
    out["tile_loadd_cycles"] = py::int_(counters.tile_loadd_cycles);
    out["tile_stream_loadd_cycles"] = py::int_(counters.tile_stream_loadd_cycles);
    out["tile_dpbf16ps_cycles"] = py::int_(counters.tile_dpbf16ps_cycles);
    out["prefetch_cycles"] = py::int_(counters.prefetch_cycles);
    out["tile_loadd_count"] = py::int_(counters.tile_loadd_count);
    out["tile_stream_loadd_count"] = py::int_(counters.tile_stream_loadd_count);
    out["tile_dpbf16ps_count"] = py::int_(counters.tile_dpbf16ps_count);
    out["prefetch_count"] = py::int_(counters.prefetch_count);
    out["k_iterations"] = py::int_(counters.k_iterations);
    out["avg_tile_loadd_cycles"] = py::float_(
        counters.tile_loadd_count > 0 ? (double)counters.tile_loadd_cycles / (double)counters.tile_loadd_count : 0.0);
    out["avg_tile_stream_loadd_cycles"] = py::float_(
        counters.tile_stream_loadd_count > 0 ? (double)counters.tile_stream_loadd_cycles / (double)counters.tile_stream_loadd_count : 0.0);
    out["avg_tile_dpbf16ps_cycles"] = py::float_(
        counters.tile_dpbf16ps_count > 0 ? (double)counters.tile_dpbf16ps_cycles / (double)counters.tile_dpbf16ps_count : 0.0);
    out["avg_prefetch_cycles"] = py::float_(
        counters.prefetch_count > 0 ? (double)counters.prefetch_cycles / (double)counters.prefetch_count : 0.0);
    return out;
}

void DispatchBlockCompute(
        const bf16_t* a_block,
        const bf16_t* packed_weight_base,
        float* c_block,
        int block_rows,
        int block_cols,
        int packed_weight_cols,
        int full_k,
        int lda,
        int c_ldc,
        int k_block,
        int n_start,
        bool prefetch_a,
        bool prefetch_b,
        bool prefetch_c,
        int tid,
        int pf_a_dist = 2,
        int pf_b_dist = 2,
        int pf_c_dist = 1) {
    const BlockPartition row_partition = SplitRowBlock(block_rows);

    for (int k_slice_start = 0; k_slice_start < full_k; k_slice_start += k_block) {
        const int current_k_span = MIN(k_block, full_k - k_slice_start);

        const bf16_t* packed_weight_k_slice = GetPackedWeightPanelKSlice(
            packed_weight_base, full_k, k_slice_start, n_start);

        if (row_partition.full_rows > 0) {
            AMXBlockTask full_task = MakeAMXBlockTask(
                &a_block[OFFSET2D(0, k_slice_start, lda)],
                packed_weight_k_slice,
                &c_block[OFFSET2D(0, 0, c_ldc)],
                row_partition.full_rows,
                block_cols,
                current_k_span,
                full_k,
                lda,
                c_ldc,
                k_slice_start == 0,
                prefetch_a,
                prefetch_b,
                prefetch_c,
                tid,
                pf_a_dist,
                pf_b_dist,
                pf_c_dist);
            RunAMX32x32(&full_task);
        }

        if (row_partition.tail_rows > 0) {
            AMXBlockTask tail_task = MakeAMXBlockTask(
                &a_block[OFFSET2D(row_partition.full_rows, k_slice_start, lda)],
                packed_weight_k_slice,
                &c_block[OFFSET2D(row_partition.full_rows, 0, c_ldc)],
                row_partition.tail_rows,
                block_cols,
                current_k_span,
                full_k,
                lda,
                c_ldc,
                k_slice_start == 0,
                prefetch_a,
                prefetch_b,
                prefetch_c,
                tid,
                pf_a_dist,
                pf_b_dist,
                pf_c_dist);
            RunAMXMTail(&tail_task);
        }
    }
}

void GemmBF16(torch::Tensor A,
              torch::Tensor packed_weight,
              torch::Tensor C_bf16,
              torch::Tensor bias,
              int TM_in,
              int TN_in,
              int TK,
              bool prefetch_a,
              bool prefetch_b,
              bool prefetch_c,
              int pf_a_dist,
              int pf_b_dist,
              int pf_c_dist,
              bool hw_prefetch) {
    set_hw_prefetch(hw_prefetch);

    static const bool kProfilePerThread =
        (std::getenv("AMX_GEMM_PROFILE_THREADS") != nullptr);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = C_bf16.size(1);
    const int lda = K;
    const int ldc = N;

    TORCH_CHECK((N % N_STEP) == 0, "gemm_bf16 requires N multiple of ", N_STEP, ", got ", N);
    TORCH_CHECK((K % K_STEP) == 0, "gemm_bf16 requires K multiple of ", K_STEP, ", got ", K);
    TORCH_CHECK(TK > 0, "gemm_bf16 requires TK > 0, got ", TK);
    TORCH_CHECK((TK % K_STEP) == 0, "gemm_bf16 requires TK multiple of ", K_STEP, ", got ", TK);
    TORCH_CHECK(TN_in <= 0 || (TN_in % N_STEP) == 0,
                "gemm_bf16 requires TN multiple of ", N_STEP, " when TN is set, got ", TN_in);

    const bool has_bias = bias.defined() && bias.numel() > 0;
    float* bias_fp32 = nullptr;
    if (has_bias) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1-D, got ", bias.dim(), "-D");
        TORCH_CHECK(bias.size(0) == N, "bias length (", bias.size(0), ") must equal N (", N, ")");
        bias_fp32 = (float*)_mm_malloc((size_t)N * sizeof(float), 64);
        TORCH_CHECK(bias_fp32 != nullptr, "failed to allocate bias fp32 buffer");
        const bf16_t* bias_bf16 = (const bf16_t*)bias.data_ptr<c10::BFloat16>();
        for (int i = 0; i < N; ++i) {
            bias_fp32[i] = static_cast<float>(bias_bf16[i]);
        }
    }

    const int num_threads = omp_get_max_threads();

    const int TM = (TM_in > 0) ? TM_in : ROUNDUP(M, M_STEP);
    const int TN = (TN_in > 0) ? TN_in : MIN(416, ROUNDUP(CEIL(N, num_threads), N_STEP));
    const int blocks_m = CEIL(M, TM);
    const int blocks_n = CEIL(N, TN);
    const int total_blocks = blocks_m * blocks_n;

    const bf16_t* a_ptr_base = (const bf16_t*)A.data_ptr<c10::BFloat16>();
    const bf16_t* packed_weight_base = (const bf16_t*)packed_weight.data_ptr<c10::BFloat16>();
    bf16_t* c_bf16_base = (bf16_t*)C_bf16.data_ptr<c10::BFloat16>();

    #pragma omp parallel num_threads(num_threads)
    {
        auto time0 = std::chrono::high_resolution_clock::now();
        const int tid = omp_get_thread_num();
        LoadTileConfig();
        auto time1 = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(time1 - time0).count();
        if (kProfilePerThread) {
            printf("tid %d, Time load tile config = %f ms\n", tid, elapsed_seconds * 1000);
        }

        auto time2 = std::chrono::high_resolution_clock::now();
        int blocks_done = 0;
        const int scratch_ldc = TN;
        const size_t scratch_bytes = (size_t)TM * scratch_ldc * sizeof(float);
        float* scratch = (float*)_mm_malloc(scratch_bytes, 64);
        TORCH_CHECK(scratch != nullptr, "gemm_bf16 failed to allocate per-thread scratch buffer");

        for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
            const int bm = (block_id / blocks_n) * TM;
            const int bn = (block_id % blocks_n) * TN;
            const int current_M = MIN(TM, M - bm);
            const int current_N = MIN(TN, N - bn);
            const bf16_t* a_ptr = &a_ptr_base[OFFSET2D(bm, 0, lda)];

            DispatchBlockCompute(
                a_ptr,
                packed_weight_base,
                scratch,
                current_M,
                current_N,
                N,
                K,
                lda,
                scratch_ldc,
                TK,
                bn,
                prefetch_a,
                prefetch_b,
                prefetch_c,
                sched_getcpu(),
                pf_a_dist,
                pf_b_dist,
                pf_c_dist);

            bf16_t* c_out = &c_bf16_base[OFFSET2D(bm, bn, ldc)];
            if (has_bias) {
                fp32_add_bias_to_bf16_fast(scratch, scratch_ldc, c_out, ldc,
                                           current_M, current_N, bias_fp32 + bn);
            } else {
                fp32_matrix_to_bf16_fast(scratch, scratch_ldc, c_out, ldc, current_M, current_N);
            }
            ++blocks_done;
        }
        _mm_free(scratch);
        auto time3 = std::chrono::high_resolution_clock::now();
        elapsed_seconds = std::chrono::duration<double>(time3 - time2).count();
        if (kProfilePerThread) {
            printf("tid %d, blocks=%d, Time dispatch block compute = %f ms\n",
                   tid, blocks_done, elapsed_seconds * 1000);
        }
    }

    if (bias_fp32) {
        _mm_free(bias_fp32);
    }
}

py::dict GemmProfile(torch::Tensor A,
                     torch::Tensor packed_weight,
                     torch::Tensor C,
                     int TM_in,
                     int TN_in,
                     int TK,
                     bool prefetch_a,
                     bool prefetch_b,
                     bool prefetch_c,
                     int pf_a_dist,
                     int pf_b_dist,
                     int pf_c_dist) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = C.size(1);
    const int lda = K;
    const int ldc = N;

    TORCH_CHECK((N % N_STEP) == 0, "gemm_bf16 requires N multiple of ", N_STEP, ", got ", N);
    TORCH_CHECK((K % K_STEP) == 0, "gemm_bf16 requires K multiple of ", K_STEP, ", got ", K);
    TORCH_CHECK(TK > 0, "gemm_bf16 requires TK > 0, got ", TK);
    TORCH_CHECK((TK % K_STEP) == 0, "gemm_bf16 requires TK multiple of ", K_STEP, ", got ", TK);
    TORCH_CHECK(TN_in <= 0 || (TN_in % N_STEP) == 0,
                "gemm_bf16 requires TN multiple of ", N_STEP, " when TN is set, got ", TN_in);

    const int num_threads = omp_get_max_threads();
    const int TM = (TM_in > 0) ? TM_in : ROUNDUP(M, M_STEP);
    const int TN = (TN_in > 0) ? TN_in : MIN(416, ROUNDUP(CEIL(N, num_threads), N_STEP));
    const int blocks_m = CEIL(M, TM);
    const int blocks_n = CEIL(N, TN);
    const int total_blocks = blocks_m * blocks_n;

    if (total_blocks <= 0) {
        return MakeEmptyProfileOutput();
    }

    const bf16_t* a_ptr_base = (const bf16_t*)A.data_ptr<c10::BFloat16>();
    const bf16_t* packed_weight_base = (const bf16_t*)packed_weight.data_ptr<c10::BFloat16>();
    bf16_t* c_bf16_base = (bf16_t*)C.data_ptr<c10::BFloat16>();

    KernelProfileCounters total_counters;

    #pragma omp parallel num_threads(num_threads)
    {
        LoadTileConfig();
        const int tid = omp_get_thread_num();
        const int scratch_ldc = TN;
        const size_t scratch_bytes = (size_t)TM * scratch_ldc * sizeof(float);
        float* scratch = (float*)_mm_malloc(scratch_bytes, 64);
        TORCH_CHECK(scratch != nullptr, "gemm_profile failed to allocate per-thread scratch buffer");

        KernelProfileCounters local_counters;
        for (int block_id = tid; block_id < total_blocks; block_id += num_threads) {
            const int bm = (block_id / blocks_n) * TM;
            const int bn = (block_id % blocks_n) * TN;
            const int current_M = MIN(TM, M - bm);
            const int current_N = MIN(TN, N - bn);
            const bf16_t* a_ptr = &a_ptr_base[OFFSET2D(bm, 0, lda)];

            KernelProfileCounters block_counters = DispatchBlockProfile(
                a_ptr,
                packed_weight_base,
                scratch,
                current_M,
                current_N,
                K,
                lda,
                scratch_ldc,
                TK,
                bn,
                prefetch_a,
                prefetch_b,
                prefetch_c,
                tid,
                pf_a_dist,
                pf_b_dist,
                pf_c_dist);
            local_counters.MergeFrom(block_counters);

            bf16_t* c_out = &c_bf16_base[OFFSET2D(bm, bn, ldc)];
            fp32_matrix_to_bf16_fast(scratch, scratch_ldc, c_out, ldc, current_M, current_N);
        }

        _mm_free(scratch);

        #pragma omp critical
        {
            total_counters.MergeFrom(local_counters);
            total_counters.tsc_overhead_cycles = local_counters.tsc_overhead_cycles;
        }
    }

    return BuildProfileOutput(total_counters);
}

}
