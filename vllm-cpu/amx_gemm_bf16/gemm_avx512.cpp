#include "gemm.hpp"

namespace amx_bf16 {

namespace {

union M512BF16Cast {
    __m512i i;
    __m512bh bh;
};

inline __m512bh load_m512bh(const bf16_t* ptr) {
    M512BF16Cast cast{};
    cast.i = _mm512_load_si512((const void*)ptr);
    return cast.bh;
}

inline __m512bh broadcast_bf16_pair(const bf16_t* ptr) {
    uint32_t pair_bits = 0;
    std::memcpy(&pair_bits, ptr, sizeof(pair_bits));
    M512BF16Cast cast{};
    cast.i = _mm512_set1_epi32((int)pair_bits);
    return cast.bh;
}

}  // namespace

void RunAVX512Kernel(AVX512BlockTask* task) {
    const bf16_t* A = task->a;
    const bf16_t* packed_weight = task->packed_weight;
    float* c_fp32 = task->c_fp32;
    bf16_t* c_bf16 = task->c_bf16;
    int rows = task->rows;
    int local_cols = task->local_cols;
    int packed_cols = task->packed_cols;
    int depth = task->depth;
    int lda = task->lda;
    int ldc = task->ldc;
    int tk_block = task->tk_block;
    int n_offset = task->n_offset;
    int bn_begin = task->bn_begin;
    int bn_end = task->bn_end;

    TORCH_CHECK(rows > 0 && rows <= 4, "AVX512 small-M path expects 1 <= M <= 4, got ", rows);
    TORCH_CHECK((local_cols % N_STEP) == 0, "AVX512 small-M path requires N multiple of ", N_STEP, ", got ", local_cols);
    TORCH_CHECK((depth % K_STEP) == 0, "AVX512 small-M path requires K multiple of ", K_STEP, ", got ", depth);

    for (int bn = bn_begin; bn < bn_end; bn += N_STEP) {
        __m512 acc0[4];
        __m512 acc1[4];
        for (int m = 0; m < rows; ++m) {
            acc0[m] = _mm512_setzero_ps();
            acc1[m] = _mm512_setzero_ps();
        }

        for (int tk = 0; tk < depth; tk += tk_block) {
            const int current_k_span = MIN(tk_block, depth - tk);
            TORCH_CHECK((current_k_span % K_STEP) == 0,
                        "AVX512 small-M path requires TK-aligned packed B; current_TK=",
                        current_k_span, ", K_STEP=", K_STEP);

            const int global_n_start = n_offset + bn;
            const bf16_t* packed_b_k_slice =
                packed_weight + (size_t)(global_n_start / N_STEP) * depth * N_STEP + tk * N_STEP;
            for (int kk = 0; kk < current_k_span; kk += K_STEP) {
                const bf16_t* packed_b_tile_pair = packed_b_k_slice + kk * N_STEP;
                const bf16_t* b0 = packed_b_tile_pair;
                const bf16_t* b1 = packed_b_tile_pair + TILE_SIZE_BF16;

                for (int pair = 0; pair < MAX_COLS_BF16 / KPACK_BF16; ++pair) {
                    const __m512bh vb0 = load_m512bh(b0 + pair * MAX_COLS_BF16);
                    const __m512bh vb1 = load_m512bh(b1 + pair * MAX_COLS_BF16);
                    const int k_pair = tk + kk + pair * KPACK_BF16;

                    for (int m = 0; m < rows; ++m) {
                        const __m512bh va = broadcast_bf16_pair(&A[OFFSET2D(m, k_pair, lda)]);
                        acc0[m] = _mm512_dpbf16_ps(acc0[m], va, vb0);
                        acc1[m] = _mm512_dpbf16_ps(acc1[m], va, vb1);
                    }
                }
            }
        }

        alignas(64) float tmp0[MAX_COLS_FP32];
        alignas(64) float tmp1[MAX_COLS_FP32];
        for (int m = 0; m < rows; ++m) {
            _mm512_store_ps(tmp0, acc0[m]);
            _mm512_store_ps(tmp1, acc1[m]);
            if (c_fp32 != nullptr) {
                std::memcpy(&c_fp32[OFFSET2D(m, bn, ldc)], tmp0, sizeof(tmp0));
                std::memcpy(&c_fp32[OFFSET2D(m, bn + MAX_COLS_FP32, ldc)], tmp1, sizeof(tmp1));
            } else {
                fp32_to_bf16_fast(tmp0, &c_bf16[OFFSET2D(m, bn, ldc)], MAX_COLS_FP32);
                fp32_to_bf16_fast(tmp1, &c_bf16[OFFSET2D(m, bn + MAX_COLS_FP32, ldc)], MAX_COLS_FP32);
            }
        }
    }
}

void RunAVX512(AVX512BlockTask* task) {
    const int num_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(num_threads)
    for (int bn = task->bn_begin; bn < task->bn_end; bn += N_STEP) {
        RunAVX512Kernel(task);
    }
}

}  // namespace amx_bf16
