#include "gemm.hpp"

namespace amx_bf16 {

namespace {

void pack_rows_2_vnni_bf16(const bf16_t* src, bf16_t* dst, int ldb) {
    __m256i r0 = _mm256_loadu_si256((const __m256i*)(src));
    __m256i r1 = _mm256_loadu_si256((const __m256i*)(src + ldb));
    __m256i lo = _mm256_unpacklo_epi16(r0, r1);
    __m256i hi = _mm256_unpackhi_epi16(r0, r1);
    __m512i zmm_out = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    const __m512i perm_idx = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
    zmm_out = _mm512_permutexvar_epi64(perm_idx, zmm_out);
    _mm512_store_si512(dst, zmm_out);
}

void pack_tile_b(const bf16_t* src, bf16_t* dst, int ldb) {
    for (int r = 0; r < MAX_COLS_BF16 / KPACK_BF16; ++r) {
        pack_rows_2_vnni_bf16(src, dst, ldb);
        src += KPACK_BF16 * ldb;
        dst += MAX_COLS_FP32 * KPACK_BF16;
    }
}

void pack_rows_2_vnni_bf16_from_nk(
        const bf16_t* src_base,
        bf16_t* dst,
        int ldb,
        int n0,
        int k0) {
    const __m512i row_offsets = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i base_rows = _mm512_add_epi32(row_offsets, _mm512_set1_epi32(n0));
    const __m512i gather_index = _mm512_add_epi32(
        _mm512_mullo_epi32(base_rows, _mm512_set1_epi32(ldb)),
        _mm512_set1_epi32(k0));
    const __m512i packed = _mm512_i32gather_epi32(gather_index, src_base, sizeof(bf16_t));
    _mm512_storeu_si512((__m512i*)dst, packed);
}

void pack_tile_b_from_nk(
        const bf16_t* src_base,
        bf16_t* dst,
        int ldb,
        int n0,
        int k0) {
    for (int r = 0; r < MAX_COLS_BF16 / KPACK_BF16; ++r) {
        pack_rows_2_vnni_bf16_from_nk(src_base, dst, ldb, n0, k0 + r * KPACK_BF16);
        dst += MAX_COLS_FP32 * KPACK_BF16;
    }
}

}  // namespace

torch::Tensor PackWeight(torch::Tensor weight) {
    TORCH_CHECK(weight.dim() == 2, "PackWeight expects a 2D tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "PackWeight expects bf16 input");
    const int K = weight.size(0);
    const int N = weight.size(1);
    TORCH_CHECK((K % K_STEP) == 0, "PackWeight expects K multiple of ", K_STEP, ", got ", K);
    TORCH_CHECK((N % N_STEP) == 0, "PackWeight expects N multiple of ", N_STEP, ", got ", N);
    const int ldb = N;
    const bf16_t* src_base = (const bf16_t*)weight.data_ptr<c10::BFloat16>();

    auto packed_b = torch::empty_like(weight);
    bf16_t* dst_base = (bf16_t*)packed_b.data_ptr<c10::BFloat16>();

    const int num_threads = omp_get_max_threads();
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int n_panel_start = 0; n_panel_start < N; n_panel_start += N_STEP) {
        bf16_t* packed_panel_base =
            dst_base + (size_t)(n_panel_start / N_STEP) * K * N_STEP;
        for (int k_tile_start = 0; k_tile_start < K; k_tile_start += K_STEP) {
            pack_tile_b(&src_base[OFFSET2D(k_tile_start, n_panel_start, ldb)], packed_panel_base, ldb);
            packed_panel_base += TILE_SIZE_BF16;
            pack_tile_b(&src_base[OFFSET2D(k_tile_start, n_panel_start + MAX_COLS_FP32, ldb)], packed_panel_base, ldb);
            packed_panel_base += TILE_SIZE_BF16;
        }
    }

    return packed_b;
}

void PackWeightNK(torch::Tensor weight) {
    TORCH_CHECK(weight.device().is_cpu(), "PackWeightNK expects CPU storage");
    TORCH_CHECK(weight.dim() == 2, "PackWeightNK expects a 2D tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "Expect bf16");
    const int N = weight.size(0);
    const int K = weight.size(1);
    TORCH_CHECK(K > 0 && K % K_STEP == 0, "PackWeightNK expects positive K multiple of ", K_STEP);
    TORCH_CHECK(N > 0 && N % N_STEP == 0, "PackWeightNK expects positive N multiple of ", N_STEP);
    bf16_t* original_ptr = (bf16_t*)weight.data_ptr<c10::BFloat16>();

    // A packed N_STEP panel occupies exactly the same contiguous region as
    // its original N_STEP rows. Panels are independent, so each thread can
    // stage one panel and write it back without overwriting another's input.
    // This replaces a full-matrix temporary and serial full-matrix memcpy.
    const int num_threads = std::min(omp_get_max_threads(), N / N_STEP);
    auto scratch = torch::empty({num_threads, K * N_STEP}, weight.options());
    bf16_t* scratch_ptr = (bf16_t*)scratch.data_ptr<c10::BFloat16>();
    const size_t panel_elements = (size_t)K * N_STEP;
    #pragma omp parallel num_threads(num_threads)
    {
        bf16_t* panel = scratch_ptr + omp_get_thread_num() * panel_elements;
        #pragma omp for schedule(static)
        for (int n_panel_start = 0; n_panel_start < N; n_panel_start += N_STEP) {
            bf16_t* output = panel;
            for (int k_tile_start = 0; k_tile_start < K; k_tile_start += K_STEP) {
                pack_tile_b_from_nk(original_ptr, output, K, n_panel_start, k_tile_start);
                output += TILE_SIZE_BF16;
                pack_tile_b_from_nk(original_ptr, output, K,
                                    n_panel_start + MAX_COLS_FP32, k_tile_start);
                output += TILE_SIZE_BF16;
            }
            std::memcpy(original_ptr + (size_t)n_panel_start * K, panel,
                        panel_elements * sizeof(bf16_t));
        }
    }
}
}  // namespace amx_bf16
