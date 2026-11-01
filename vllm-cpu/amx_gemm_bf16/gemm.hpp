#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <sys/syscall.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include "utils.hpp"

#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define FLOOR(x, y) ((x) / (y))
#define ROUNDUP(x, y) (CEIL(x, y) * (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define MAX_ROWS       16
#define MAX_COLS_BF16  32   
#define MAX_COLS_FP32  16   
#define TILE_SIZE_BF16 (MAX_ROWS * MAX_COLS_BF16)  
#define TILE_SIZE_FP32 (MAX_ROWS * MAX_COLS_FP32)  
#define MIN_STRIDE 64 
#define CACHELINE_SIZE 64
#define KPACK_BF16 2  
#define M_STEP (MAX_ROWS * 2)      
#define N_STEP (MAX_COLS_FP32 * 2) 
#define K_STEP MAX_COLS_BF16       
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// 寄存器别名
#define C00 0
#define C01 1
#define C10 2
#define C11 3
#define A0  4
#define A1  5
#define B0  6
#define B1  7

using bf16_t = c10::BFloat16;
namespace py = pybind11;

namespace amx_bf16 {

// ---------------------------------------------------------------------------
// Public entrypoints
// ---------------------------------------------------------------------------

// Public C++ entrypoints used by the Python bindings.
torch::Tensor PackWeight(torch::Tensor weight);
void PackWeightNK(torch::Tensor weight);

void GemmBF16(torch::Tensor A,
              torch::Tensor packed_weight,
              torch::Tensor C_bf16,
              torch::Tensor bias = {},
              int TM = -1,
              int TN = -1,
              int TK = 1376,
              bool prefetch_a = false,  
              bool prefetch_b = false,
              bool prefetch_c = false,
              int pf_a_dist = 2,
              int pf_b_dist = 2,
              int pf_c_dist = 1,
              bool hw_prefetch = true);

void set_hw_prefetch(bool enable);

// Experimental: packed_weight is VNNI-converted but stored in K-tile-major order
// (i.e., tiles for a fixed N panel are NOT contiguous across K).
void GemmBF16VNNIOnly(torch::Tensor A,
                      torch::Tensor packed_weight_vnni_only,
                      torch::Tensor C_bf16,
                      int TM = -1,
                      int TN = -1,
                      int TK = 1376,
                      bool prefetch_a = false,
                      bool prefetch_b = false,
                      bool prefetch_c = false);

py::dict GemmProfile(torch::Tensor A,
                     torch::Tensor packed_weight,
                     torch::Tensor C,
                     int TM = -1,
                     int TN = -1,
                     int TK = 1376,
                     bool prefetch_a = false,
                     bool prefetch_b = false,
                     bool prefetch_c = false,
                     int pf_a_dist = 2,
                     int pf_b_dist = 2,
                     int pf_c_dist = 1);


// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

inline bf16_t float_to_bf16_scalar(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFF;
    return static_cast<bf16_t>((u + rounding_bias) >> 16);
}

inline void fp32_to_bf16_fast(const float* src, bf16_t* dst, int count) {
    int i = 0;

#if defined(__AVX512BF16__)
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __m256bh packed = _mm512_cvtneps_pbh(v);
        std::memcpy(dst + i, &packed, sizeof(packed));
    }
#endif

    for (; i < count; ++i) {
        dst[i] = float_to_bf16_scalar(src[i]);
    }
}

inline void fp32_matrix_to_bf16_fast(
        const float* src,
        int ld_src,
        bf16_t* dst,
        int ld_dst,
        int rows,
        int cols) {
    for (int r = 0; r < rows; ++r) {
        fp32_to_bf16_fast(src + r * ld_src, dst + r * ld_dst, cols);
    }
}

inline void fp32_add_bias_to_bf16_fast(
        const float* src,
        int ld_src,
        bf16_t* dst,
        int ld_dst,
        int rows,
        int cols,
        const float* bias) {
    for (int r = 0; r < rows; ++r) {
        const float* src_row = src + r * ld_src;
        bf16_t* dst_row = dst + r * ld_dst;
        int i = 0;
#if defined(__AVX512BF16__)
        for (; i + 16 <= cols; i += 16) {
            __m512 v = _mm512_loadu_ps(src_row + i);
            __m512 b = _mm512_loadu_ps(bias + i);
            v = _mm512_add_ps(v, b);
            __m256bh packed = _mm512_cvtneps_pbh(v);
            std::memcpy(dst_row + i, &packed, sizeof(packed));
        }
#endif
        for (; i < cols; ++i) {
            dst_row[i] = float_to_bf16_scalar(src_row[i] + bias[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Tile helpers
// ---------------------------------------------------------------------------

static inline bool set_tiledata_use() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        return false;
    }
    return true;
}

static inline void load_tail_tile_c_l2(const float* c, int ldc, bool has_bottom) {
    _tile_stream_loadd(C00, c, ldc * sizeof(float));
    _tile_stream_loadd(C01, c + MAX_COLS_FP32, ldc * sizeof(float));
    if (has_bottom) {
        _tile_stream_loadd(C10, c + MAX_ROWS * ldc, ldc * sizeof(float));
        _tile_stream_loadd(C11, c + MAX_ROWS * ldc + MAX_COLS_FP32, ldc * sizeof(float));
    }
}

static inline void store_tail_tile_c_l1(float* c, int ldc, bool has_bottom) {
    _tile_stored(C00, c, ldc * sizeof(float));
    _tile_stored(C01, c + MAX_COLS_FP32, ldc * sizeof(float));
    if (has_bottom) {
        _tile_stored(C10, c + MAX_ROWS * ldc, ldc * sizeof(float));
        _tile_stored(C11, c + MAX_ROWS * ldc + MAX_COLS_FP32, ldc * sizeof(float));
    }
}

// tilezero 4 tileC
static inline void clear_4_tileC() {
    _tile_zero(C00);
    _tile_zero(C01);
    _tile_zero(C10);
    _tile_zero(C11);
}

// tileloadd C0,C1,C2,C3
static inline void load_4_tileC_l1(const float* c, int ldc) {
    _tile_loadd(C00, c, ldc * sizeof(float));
    _tile_loadd(C01, c + MAX_COLS_FP32, ldc * sizeof(float));
    _tile_loadd(C10, c + MAX_ROWS * ldc, ldc * sizeof(float));
    _tile_loadd(C11, c + MAX_ROWS * ldc + MAX_COLS_FP32, ldc * sizeof(float));
}

// tilestored C0,C1,C2,C3
static inline void store_4_tileC_l1(float* c, int ldc) {
    _tile_stored(C00, c, ldc * sizeof(float));
    _tile_stored(C01, c + MAX_COLS_FP32, ldc * sizeof(float));
    _tile_stored(C10, c + MAX_ROWS * ldc, ldc * sizeof(float));
    _tile_stored(C11, c + MAX_ROWS * ldc + MAX_COLS_FP32, ldc * sizeof(float));
}


// tileloaddt1 C0,C1,C2,C3
static inline void load_4_tileC_l2(const float* c, int ldc) {
    _tile_stream_loadd(C00, c, ldc * sizeof(float));
    _tile_stream_loadd(C01, c + MAX_COLS_FP32, ldc * sizeof(float));
    _tile_stream_loadd(C10, c + MAX_ROWS * ldc, ldc * sizeof(float));
    _tile_stream_loadd(C11, c + MAX_ROWS * ldc + MAX_COLS_FP32, ldc * sizeof(float));
}

// ---------------------------------------------------------------------------
// Internal helpers shared by the split implementation files
// ---------------------------------------------------------------------------

struct alignas(64) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[8];
    uint8_t reserved_1[16];
    uint8_t rows[8];
    uint8_t reserved_2[8];

    TileConfig() {
        memset(this, 0, sizeof(TileConfig));
        palette_id = 1;
        start_row = 0;
    }

    void set_row_col(int i, uint8_t row, uint16_t col) {
        colsb[i] = col;
        rows[i] = row;
    }
};


struct AMXBlockTask {
    const bf16_t* a;
    const bf16_t* packed_weight;
    float* c;
    int rows;
    int cols;
    int depth;
    int packed_weight_k;
    int lda;
    int ldc;
    bool zero_accumulator;
    bool prefetch_a;
    bool prefetch_b;
    bool prefetch_c;
    int tid;
    int pf_a_dist;
    int pf_b_dist;
    int pf_c_dist;
};

struct AVX512BlockTask {
    const bf16_t* a;
    const bf16_t* packed_weight;
    float* c_fp32;
    bf16_t* c_bf16;
    int rows;
    int local_cols;
    int packed_cols;
    int depth;
    int lda;
    int ldc;
    int tk_block;
    int n_offset;
    int bn_begin;
    int bn_end;
};

struct BlockPartition {
    int full_rows;
    int tail_rows;
};

struct KernelProfileCounters {
    uint64_t tsc_overhead_cycles = 0;
    uint64_t tile_loadd_cycles = 0;
    uint64_t tile_stream_loadd_cycles = 0;
    uint64_t tile_dpbf16ps_cycles = 0;
    uint64_t tile_loadd_count = 0;
    uint64_t tile_stream_loadd_count = 0;
    uint64_t tile_dpbf16ps_count = 0;
    uint64_t k_iterations = 0;
    uint64_t prefetch_cycles = 0;
    uint64_t prefetch_count = 0;

    void MergeFrom(const KernelProfileCounters& other) {
        tsc_overhead_cycles = other.tsc_overhead_cycles;
        tile_loadd_cycles += other.tile_loadd_cycles;
        tile_stream_loadd_cycles += other.tile_stream_loadd_cycles;
        tile_dpbf16ps_cycles += other.tile_dpbf16ps_cycles;
        tile_loadd_count += other.tile_loadd_count;
        tile_stream_loadd_count += other.tile_stream_loadd_count;
        tile_dpbf16ps_count += other.tile_dpbf16ps_count;
        k_iterations += other.k_iterations;
        prefetch_cycles += other.prefetch_cycles;
        prefetch_count += other.prefetch_count;
    }
};

static inline uint64_t read_tsc_begin() {
    _mm_lfence();
    asm volatile("" ::: "memory");
    return __rdtsc();
}

static inline uint64_t read_tsc_end() {
    unsigned int aux = 0;
    uint64_t tsc = __rdtscp(&aux);
    asm volatile("" ::: "memory");
    _mm_lfence();
    return tsc;
}

inline uint64_t calibrate_tsc_overhead() {
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < 256; ++i) {
        const uint64_t t0 = read_tsc_begin();
        const uint64_t t1 = read_tsc_end();
        best = std::min(best, t1 - t0);
    }
    return best;
}

static inline uint64_t subtract_tsc_overhead(
        uint64_t cycles,
        uint64_t count,
        uint64_t overhead_per_sample) {
    const uint64_t overhead = count * overhead_per_sample;
    return cycles > overhead ? (cycles - overhead) : 0;
}

static inline BlockPartition SplitRowBlock(int rows) {
    return {
        .full_rows = (rows / M_STEP) * M_STEP,
        .tail_rows = rows % M_STEP,
    };
}

static inline bool UseAVX512(int rows) {
    // 禁用 AVX512 tail 路径，统一走 AMX kernel。
    // 逻辑上等价于将 M 维 padding 到 M_STEP 的倍数再用 AMX 计算。
    return rows > 0 && rows <= 4;
}

static inline bool UseAMXMTail(int rows) {
    // 只要有 tail 行（1 ~ M_STEP-1），都走 AMX tail kernel
    return rows > 0 && rows < M_STEP;
}

// Packed B layout:
//   [N panel 0: full K][N panel 1: full K]...
// Each N panel spans N_STEP columns and stores the whole K dimension contiguously.
static inline const bf16_t* GetPackedWeightPanelKSlice(
        const bf16_t* packed_weight_base,
        int full_k,
        int k_slice_start,
        int n_start) {
    const int n_panel_index = n_start / N_STEP;
    return packed_weight_base
        + (size_t)n_panel_index * full_k * N_STEP
        + (size_t)k_slice_start * N_STEP;
}

static inline AMXBlockTask MakeAMXBlockTask(
        const bf16_t* a,
        const bf16_t* packed_weight,
        float* c,
        int rows,
        int cols,
        int depth,
        int packed_weight_k,
        int lda,
        int ldc,
        bool zero_accumulator,
        bool prefetch_a,
        bool prefetch_b,
        bool prefetch_c,
        int tid,
        int pf_a_dist = 2,
        int pf_b_dist = 2,
        int pf_c_dist = 1) {
    return AMXBlockTask{
        .a = a,
        .packed_weight = packed_weight,
        .c = c,
        .rows = rows,
        .cols = cols,
        .depth = depth,
        .packed_weight_k = packed_weight_k,
        .lda = lda,
        .ldc = ldc,
        .zero_accumulator = zero_accumulator,
        .prefetch_a = prefetch_a,
        .prefetch_b = prefetch_b,
        .prefetch_c = prefetch_c,
        .tid = tid,
        .pf_a_dist = pf_a_dist,
        .pf_b_dist = pf_b_dist,
        .pf_c_dist = pf_c_dist,
    };
}

static inline AVX512BlockTask MakeAVX512BlockTask(
        const bf16_t* a,
        const bf16_t* packed_weight,
        float* c_fp32,
        bf16_t* c_bf16,
        int rows,
        int local_cols,
        int packed_cols,
        int depth,
        int lda,
        int ldc,
        int tk_block,
        int n_offset,
        int bn_begin,
        int bn_end) {
    return AVX512BlockTask{
        .a = a,
        .packed_weight = packed_weight,
        .c_fp32 = c_fp32,
        .c_bf16 = c_bf16,
        .rows = rows,
        .local_cols = local_cols,
        .packed_cols = packed_cols,
        .depth = depth,
        .lda = lda,
        .ldc = ldc,
        .tk_block = tk_block,
        .n_offset = n_offset,
        .bn_begin = bn_begin,
        .bn_end = bn_end,
    };
}

py::dict MakeEmptyProfileOutput();
py::dict BuildProfileOutput(const KernelProfileCounters& counters);

void LoadTileConfig();
void LoadMTailTileConfig(int top_rows, int bottom_rows);

void RunAMX32x32(AMXBlockTask* task);
void RunAMXMTail(AMXBlockTask* task);
KernelProfileCounters ProfileAMX32x32(AMXBlockTask* task);

void RunAVX512Kernel(AVX512BlockTask* task);

void RunAVX512(AVX512BlockTask* task);

} // namespace amx_bf16
