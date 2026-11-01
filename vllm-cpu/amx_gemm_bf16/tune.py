#!/usr/bin/env python3
"""
AMX GEMM 自动调参脚本
针对不同输入长度 M 和 (K,N) 组合，网格搜索最佳的 (TM, TN, TK) 以及预取配置。

两阶段搜索:
  阶段1: 使用默认预取设置，搜索最佳 (TM, TN, TK)
  阶段2: 固定最佳 (TM, TN, TK)，搜索最佳预取参数组合

用法:
    # 单个 (K,N):
    numactl -C 0-29 -m 0 python tune.py --shapes 4096x12288 --M-list 1024,2048,4096
    # 多个 (K,N) 组合 (典型 LLaMA 层):
    numactl -C 0-29 -m 0 python tune.py --shapes 4096x12288,4096x4096,12288x4096
    # 跳过预取搜索:
    numactl -C 0-29 -m 0 python tune.py --shapes 4096x12288 --skip-prefetch
"""

import torch
import amx_gemm_bf16_ext
import time
import argparse
import math
import json
from dataclasses import dataclass, field, asdict

# ===================== 常量 =====================
M_STEP = 32   # tile M 步长
N_STEP = 32   # tile N 步长
K_STEP = 32   # tile K 步长

BENCH_ITERS  = 1

# L3 cache flush buffer (SPR 30-core ≈ 45MB L3, 用 64MB 确保冲掉)
FLUSH_SIZE = 80 * 1024 * 1024  # 64 MB
_flush_buf = None

def get_flush_buf():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(FLUSH_SIZE // 4, dtype=torch.float32)
    return _flush_buf

def flush_cache():
    buf = get_flush_buf()
    buf.fill_(1.0)
    _ = buf.sum().item()

# ===================== 工具函数 =====================

def roundup(x, align):
    return ((x + align - 1) // align) * align

def ceil_div(a, b):
    return (a + b - 1) // b

def gen_TM_candidates(M):
    candidates = set()
    if M <= 2048:
        candidates.add(M)
        return sorted(c for c in candidates if c <= M)
    for v in [512, 768, 1024, 1536, 2048]:
        if v <= M and M % v == 0:
            candidates.add(v)
        if v <= M:
            candidates.add(v)
    candidates.add(roundup(M, M_STEP) if M % M_STEP != 0 else M)
    for k in range(1, min(9, M // M_STEP + 1)):
        tm = roundup(ceil_div(M, k), M_STEP)
        if tm <= M:
            candidates.add(tm)
    tm_half = roundup(ceil_div(M, 2), M_STEP)
    if tm_half <= M:
        candidates.add(tm_half)
    return sorted(c for c in candidates if c >= M_STEP and c <= M)

def gen_TN_candidates(N, num_threads):
    candidates = set()
    for v in [32, 64, 96, 128, 192, 256, 320, 384, 416, 448, 512, 640, 768, 1024]:
        if v <= N:
            candidates.add(v)
    tn_base = ceil_div(N, num_threads)
    for delta in range(-2, 3):
        v = roundup(max(N_STEP, tn_base + delta * N_STEP), N_STEP)
        if N_STEP <= v <= N:
            candidates.add(v)
    return sorted(c for c in candidates if c >= N_STEP and c <= N)

def gen_TK_candidates(K):
    candidates = set()
    for v in [512, 640, 768, 896, 1024, 1152, 1280, 1376, 1408, 1536, 2048]:
        if v <= K:
            candidates.add(v)
    candidates.add(K if K % K_STEP == 0 else roundup(K, K_STEP))
    return sorted(c for c in candidates if c >= K_STEP and c <= K)

def gen_prefetch_candidates():
    """生成预取参数候选列表。
    
    每个候选是一个 dict:
      prefetch_a, prefetch_b, prefetch_c: bool
      pf_a_dist, pf_b_dist, pf_c_dist:   int (仅在对应 prefetch 为 True 时有效)
      hw_prefetch: bool  (是否开启CPU硬件预取)
    """
    candidates = []
    dist_a_vals = [1, 2, 3, 4]
    dist_b_vals = [1, 2, 3, 4]
    dist_c_vals = [1, 2, 3, 4]

    for pfa in [False, True]:
        for pfb in [False, True]:
            for pfc in [False, True]:
                if not pfa and not pfb and not pfc:
                    continue
                a_dists = dist_a_vals if pfa else [0]
                b_dists = dist_b_vals if pfb else [0]
                c_dists = dist_c_vals if pfc else [0]
                for ad in a_dists:
                    for bd in b_dists:
                        for cd in c_dists:
                            candidates.append(dict(
                                prefetch_a=pfa, prefetch_b=pfb, prefetch_c=pfc,
                                pf_a_dist=ad, pf_b_dist=bd, pf_c_dist=cd,
                                hw_prefetch=True))
    return candidates

def gen_prefetch_candidates_quick():
    """快速模式：只测试最常见的预取组合，含硬件预取开/关。"""
    base = [
        dict(prefetch_a=False, prefetch_b=False, prefetch_c=False, pf_a_dist=2, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=False, pf_a_dist=2, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=True,  pf_a_dist=2, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=True,  prefetch_c=True,  pf_a_dist=2, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=True,  pf_a_dist=1, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=True,  pf_a_dist=3, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=True,  pf_a_dist=4, pf_b_dist=2, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=True,  pf_a_dist=2, pf_b_dist=2, pf_c_dist=2),
        dict(prefetch_a=True,  prefetch_b=False, prefetch_c=True,  pf_a_dist=2, pf_b_dist=2, pf_c_dist=3),
        dict(prefetch_a=True,  prefetch_b=True,  prefetch_c=True,  pf_a_dist=2, pf_b_dist=1, pf_c_dist=1),
        dict(prefetch_a=True,  prefetch_b=True,  prefetch_c=True,  pf_a_dist=2, pf_b_dist=3, pf_c_dist=1),
    ]
    candidates = []
    for hw_pf in [True, False]:
        for cfg in base:
            c = cfg.copy()
            c['hw_prefetch'] = hw_pf
            candidates.append(c)
    return candidates

# ===================== 基准测试 =====================

def benchmark_one(A, packed_B, M, N, TM, TN, TK,
                  prefetch_a=False, prefetch_b=False, prefetch_c=False,
                  pf_a_dist=0, pf_b_dist=0, pf_c_dist=0,
                  hw_prefetch=True):
    """冷启动基准测试: 每次调用前刷缓存, 模拟 vLLM 逐层调用场景"""
    gemm_kwargs = dict(TK=TK, TM=TM, TN=TN,
                       prefetch_a=prefetch_a, prefetch_b=prefetch_b, prefetch_c=prefetch_c,
                       pf_a_dist=pf_a_dist, pf_b_dist=pf_b_dist, pf_c_dist=pf_c_dist,
                       hw_prefetch=hw_prefetch)

    C = torch.empty((M, N), dtype=torch.bfloat16)
    amx_gemm_bf16_ext.gemm_bf16(A, packed_B, C, **gemm_kwargs)
    
    times = []
    for _ in range(BENCH_ITERS):
        flush_cache()
        t0 = time.perf_counter()
        C = torch.empty((M, N), dtype=torch.bfloat16)
        amx_gemm_bf16_ext.gemm_bf16(A, packed_B, C, **gemm_kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    times.sort()
    trim = max(1, len(times) // 5)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    avg = sum(trimmed) / len(trimmed)
    return avg

def compute_gflops(M, N, K, time_ms):
    flops = 2.0 * M * N * K
    return flops / (time_ms / 1000) / 1e9

# ===================== 搜索主逻辑 =====================

@dataclass
class SearchResult:
    M: int
    K: int
    N: int
    best_TM: int
    best_TN: int
    best_TK: int
    best_pf_time_ms: float
    best_no_pf_time_ms: float
    best_gflops: float
    total_configs: int
    search_time_s: float
    # 预取参数
    prefetch_a: bool = False
    prefetch_b: bool = False
    prefetch_c: bool = False
    pf_a_dist: int = 0
    pf_b_dist: int = 0
    pf_c_dist: int = 0
    hw_prefetch: bool = True
    prefetch_configs_searched: int = 0

def pf_tag(cfg):
    """预取配置的短字符串表示"""
    parts = []
    if cfg['prefetch_a']: parts.append(f"A({cfg['pf_a_dist']})")
    if cfg['prefetch_b']: parts.append(f"B({cfg['pf_b_dist']})")
    if cfg['prefetch_c']: parts.append(f"C({cfg['pf_c_dist']})")
    hw = "on" if cfg.get('hw_prefetch', True) else "off"
    sw = ",".join(parts) if parts else "none"
    return f"sw[{sw}],hw={hw}"

def search_for_M(M, K, N, num_threads, packed_weight, quick=False, skip_prefetch=False):
    """对给定 M，两阶段搜索最佳 (TM, TN, TK) 和预取配置"""
    
    tm_candidates = gen_TM_candidates(M)
    tn_candidates = gen_TN_candidates(N, num_threads)
    tk_candidates = gen_TK_candidates(K)
    
    if quick:
        tm_candidates = tm_candidates[:6]
        tn_candidates = tn_candidates[:6]
        tk_candidates = tk_candidates[:4]
    
    total_configs = len(tm_candidates) * len(tn_candidates) * len(tk_candidates)
    print(f"\n{'='*60}")
    print(f"  M={M}, K={K}, N={N}, threads={num_threads}")
    print(f"  TM candidates ({len(tm_candidates)}): {tm_candidates}")
    print(f"  TN candidates ({len(tn_candidates)}): {tn_candidates}")
    print(f"  TK candidates ({len(tk_candidates)}): {tk_candidates}")
    print(f"  Total tile configs to search: {total_configs}")
    print(f"{'='*60}")
    
    A = torch.randn(M, K, dtype=torch.bfloat16)
    
    best_time = float('inf')
    best_config = (0, 0, 0)
    best_gflops = 0.0
    
    search_start = time.time()
    
    # ---- 阶段 1: 搜索 (TM, TN, TK)，使用默认预取 (pfa=True, pfc=True) ----
    print(f"\n  --- Phase 1: Tile search (prefetch_a=True, prefetch_c=True, default dist) ---")
    for tk_idx, TK in enumerate(tk_candidates):
        print(f"\n  [TK={TK}] ({tk_idx+1}/{len(tk_candidates)}) searching TM x TN...")
        
        group_best_time = float('inf')
        group_best_config = (0, 0)
        
        for TM in tm_candidates:
            for TN in tn_candidates:
                try:
                    elapsed = benchmark_one(A, packed_weight, M, N, TM, TN, TK)
                    gflops = compute_gflops(M, N, K, elapsed)
                    print(f"      TM={TM}, TN={TN}, TK={TK} -> {elapsed:.3f} ms")
                    
                    if elapsed < group_best_time:
                        group_best_time = elapsed
                        group_best_config = (TM, TN)
                    
                    if elapsed < best_time:
                        best_time = elapsed
                        best_config = (TM, TN, TK)
                        best_gflops = gflops
                        
                except Exception as e:
                    print(f"    SKIP TM={TM}, TN={TN}, TK={TK}: {e}")
                    continue
        
        print(f"    TK={TK} best: TM={group_best_config[0]}, TN={group_best_config[1]}, "
              f"time={group_best_time:.3f} ms, "
              f"GFLOPS={compute_gflops(M, N, K, group_best_time):.1f}")
    
    phase1_time = time.time() - search_start
    print(f"\n  >>> Phase 1 BEST: TM={best_config[0]}, TN={best_config[1]}, TK={best_config[2]}")
    print(f"      time={best_time:.3f} ms, GFLOPS={best_gflops:.1f}, search took {phase1_time:.1f}s")

    # 默认预取结果
    best_pf = dict(prefetch_a=False, prefetch_b=False, prefetch_c=False,
                   pf_a_dist=0, pf_b_dist=0, pf_c_dist=0, hw_prefetch=True)
    pf_configs_searched = 0

    # ---- 阶段 2: 固定最佳 tile，搜索预取参数 ----
    if not skip_prefetch:
        pf_candidates = gen_prefetch_candidates_quick() if quick else gen_prefetch_candidates()
        pf_configs_searched = len(pf_candidates)
        
        print(f"\n  --- Phase 2: Prefetch search ({pf_configs_searched} configs) ---")
        print(f"      Fixed: TM={best_config[0]}, TN={best_config[1]}, TK={best_config[2]}")
        
        best_pf_time = best_time
        best_pf_cfg = best_pf.copy()
        best_pf_gflops = -1.0
        
        for pi, pf_cfg in enumerate(pf_candidates):
            try:
                elapsed = benchmark_one(
                    A, packed_weight, M, N,
                    best_config[0], best_config[1], best_config[2],
                    **pf_cfg)
                tag = pf_tag(pf_cfg)
                print(f"      [{pi+1}/{pf_configs_searched}] {tag} -> {elapsed:.3f} ms")
                
                if elapsed < best_pf_time:
                    best_pf_time = elapsed
                    best_pf_cfg = pf_cfg.copy()
                    best_pf_gflops = compute_gflops(M, N, K, best_pf_time)
                    
            except Exception as e:
                print(f"      [{pi+1}/{pf_configs_searched}] SKIP: {e}")
                continue
                
        best_pf = best_pf_cfg if best_pf_gflops > 0 else best_pf
        best_gflops = max(best_gflops, best_pf_gflops)
        print(f"\n  >>> Phase 2 BEST: {pf_tag(best_pf)}")
        print(f"      best pf time={best_pf_time:.3f} ms, best no pf time={best_time:.3f} ms, GFLOPS={best_gflops:.1f}")
    
    search_time = time.time() - search_start
    
    result = SearchResult(
        M=M, K=K, N=N,
        best_TM=best_config[0],
        best_TN=best_config[1],
        best_TK=best_config[2],
        best_pf_time_ms=best_pf_time,
        best_no_pf_time_ms=best_time,
        best_gflops=best_gflops,
        total_configs=total_configs + pf_configs_searched,
        search_time_s=search_time,
        prefetch_a=best_pf['prefetch_a'],
        prefetch_b=best_pf['prefetch_b'],
        prefetch_c=best_pf['prefetch_c'],
        pf_a_dist=best_pf['pf_a_dist'],
        pf_b_dist=best_pf['pf_b_dist'],
        pf_c_dist=best_pf['pf_c_dist'],
        hw_prefetch=best_pf.get('hw_prefetch', True),
        prefetch_configs_searched=pf_configs_searched,
    )
    
    print(f"\n  >>> FINAL for M={M}: TM={best_config[0]}, TN={best_config[1]}, TK={best_config[2]}")
    print(f"      {pf_tag(best_pf)}")
    print(f"      best pf time={best_pf_time:.3f} ms, best no pf time={best_time:.3f} ms, GFLOPS={best_gflops:.1f}")
    print(f"      search took {search_time:.1f}s ({total_configs} tile + {pf_configs_searched} prefetch configs)")
    
    return result


def get_config(M, N, K, configs):
    """查找 (M, K, N) 对应的最优 GEMM 配置。

    查找策略:
      1. 精确匹配 (M, K, N) -> O(1) dict 查找
      2. 同 (K, N) 下对 M 做最近匹配 (bisect, O(log n)):
         - 优先选 >= M 的最小条目 (向上取整, 避免 tile 不足)
         - 若不存在, 选最大的条目 (用最大 M 的配置兜底)
      3. (K, N) 完全未调优 -> 返回 DEFAULT_CONFIG

    Args:
        M: 行数
        N: 输出列数
        K: 内维
        configs: dict, 包含 GEMM_CONFIGS, _KN_INDEX, DEFAULT_CONFIG
                 (即 json_to_py 生成的 py 模块, 或直接 import 后的模块属性)

    Returns:
        dict with keys: TM, TN, TK, prefetch_a/b/c, pf_a/b/c_dist, hw_prefetch
    """
    from bisect import bisect_left

    gemm_configs = configs["GEMM_CONFIGS"]
    kn_index = configs["_KN_INDEX"]
    default_config = configs["DEFAULT_CONFIG"]

    key = (M, K, N)
    if key in gemm_configs:
        return gemm_configs[key]

    m_list = kn_index.get((K, N))
    if m_list is None:
        return default_config

    idx = bisect_left(m_list, M)
    if idx < len(m_list):
        chosen_m = m_list[idx]
    else:
        chosen_m = m_list[-1]

    return gemm_configs[(chosen_m, K, N)]


def json_to_py(json_path, py_path):
    """将 tune_results.json 转换为可直接 import 的 .py 配置文件。

    生成的 py 文件包含:
      - GEMM_CONFIGS: dict, key = (M, K, N), value = 参数 dict
      - _KN_INDEX:    按 (K,N) 分组的辅助索引 (M 升序), 用于快速最近匹配
      - DEFAULT_CONFIG: 默认参数 (未找到 key 时使用)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    meta = data.get("config", {})

    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["K"], r["N"])].append(r)

    lines = []
    lines.append('"""')
    lines.append(f"Auto-generated by tune.py from {json_path}")
    lines.append(f"Cores: {meta.get('cores', '?')}, Threads: {meta.get('num_threads', '?')}")
    lines.append("")
    lines.append("Usage:")
    lines.append("    import tune_config")
    lines.append("    from tune import get_config")
    lines.append("")
    lines.append("    configs = {")
    lines.append('        "GEMM_CONFIGS": tune_config.GEMM_CONFIGS,')
    lines.append('        "_KN_INDEX": tune_config._KN_INDEX,')
    lines.append('        "DEFAULT_CONFIG": tune_config.DEFAULT_CONFIG,')
    lines.append("    }")
    lines.append("    cfg = get_config(M=1024, N=12288, K=4096, configs=configs)")
    lines.append("    amx_gemm_bf16_ext.gemm_bf16(")
    lines.append("        A, packed_B, C,")
    lines.append('        TM=cfg["TM"], TN=cfg["TN"], TK=cfg["TK"],')
    lines.append('        prefetch_a=cfg["prefetch_a"], prefetch_b=cfg["prefetch_b"],')
    lines.append('        prefetch_c=cfg["prefetch_c"],')
    lines.append('        pf_a_dist=cfg["pf_a_dist"], pf_b_dist=cfg["pf_b_dist"],')
    lines.append('        pf_c_dist=cfg["pf_c_dist"],')
    lines.append('        hw_prefetch=cfg["hw_prefetch"],')
    lines.append("    )")
    lines.append('"""')
    lines.append("")

    lines.append("DEFAULT_CONFIG = {")
    lines.append('    "TM": -1, "TN": -1, "TK": 1376,')
    lines.append('    "prefetch_a": True, "prefetch_b": False, "prefetch_c": True,')
    lines.append('    "pf_a_dist": 2, "pf_b_dist": 2, "pf_c_dist": 1,')
    lines.append('    "hw_prefetch": True,')
    lines.append("}")
    lines.append("")

    lines.append("GEMM_CONFIGS = {")
    for (K, N), entries in sorted(groups.items()):
        lines.append(f"    # K={K}, N={N}")
        for r in sorted(entries, key=lambda x: x["M"]):
            M = r["M"]
            lines.append(
                f"    ({M}, {K}, {N}): {{"
                f'"TM": {r["best_TM"]}, "TN": {r["best_TN"]}, "TK": {r["best_TK"]}, '
                f'"prefetch_a": {r["prefetch_a"]}, "prefetch_b": {r["prefetch_b"]}, '
                f'"prefetch_c": {r["prefetch_c"]}, '
                f'"pf_a_dist": {r["pf_a_dist"]}, "pf_b_dist": {r["pf_b_dist"]}, '
                f'"pf_c_dist": {r["pf_c_dist"]}, '
                f'"hw_prefetch": {r.get("hw_prefetch", True)}'
                f"}},  # {r['best_gflops']:.1f} GFLOPS, pf:{r['best_pf_time_ms']:.3f} ms, nopf:{r['best_no_pf_time_ms']:.3f} ms"
            )
    lines.append("}")
    lines.append("")

    # 生成 _KN_INDEX: {(K,N): [M1, M2, ...]} (已排序)
    lines.append("# (K,N) -> sorted list of M values for bisect lookup")
    lines.append("_KN_INDEX = {")
    for (K, N), entries in sorted(groups.items()):
        m_list = sorted(r["M"] for r in entries)
        lines.append(f"    ({K}, {N}): {m_list},")
    lines.append("}")
    lines.append("")

    with open(py_path, "w") as f:
        f.write("\n".join(lines))

    print(f"配置已导出到 {py_path} ({len(results)} 条)")


def parse_shapes(s):
    pairs = []
    for item in s.split(","):
        item = item.strip()
        # 允许 × (U+00D7)、* 等，统一成 ASCII x
        item = item.replace("×", "x").replace("*", "x")
        if "x" in item:
            k, n = item.split("x", 1)
            pairs.append((int(k.strip()), int(n.strip())))
        else:
            raise ValueError(f"格式错误: '{item}', 期望 KxN 如 4096x12288")
    return pairs


def main():
    global BENCH_ITERS
    parser = argparse.ArgumentParser(description="AMX GEMM 自动调参 (tile + prefetch)")
    parser.add_argument("--shapes", type=str, default="4096x3072,2048x4096,4096x12288,6144x4096,5120x3584,2560x5120,5120x17408,8704x5120,8192x5120,4096x8192,8192x29568,14784x8192",
                        help="(K,N) 组合列表, 格式 KxN, 逗号分隔")
    parser.add_argument("--cores", type=str, default="98-143", help="CPU 核心绑定 (传给 init_cpu_threads_env)")
    parser.add_argument("--quick", action="store_true", help="快速模式: 缩减搜索空间")
    parser.add_argument("--output", type=str, default="tune_results.json", help="结果输出文件")
    parser.add_argument("--iters", type=int, default=1, help="基准测试迭代次数")
    parser.add_argument("--skip-prefetch", action="store_true", help="跳过阶段2预取参数搜索")
    parser.add_argument("--M-list", type=str, default=None,
                        help="自定义 M 值列表, 逗号分隔. 默认 range(64, 4097, 64)")
    parser.add_argument("--py-output", type=str, default="tune_config.py",
                        help="生成的 Python 配置文件路径 (可直接 import)")
    args = parser.parse_args()
    
    BENCH_ITERS = args.iters
    
    amx_gemm_bf16_ext.init_cpu_threads_env(args.cores)
    
    num_threads = len(args.cores.split(",")) if "," in args.cores else 1
    if "-" in args.cores:
        parts = args.cores.split("-")
        num_threads = int(parts[1]) - int(parts[0]) + 1
    
    if args.M_list:
        M_list = [int(x.strip()) for x in args.M_list.split(",")]
    else:
        M_list = [x for x in range(1, 17)] + [x for x in range(32, 4097, 32)]

    shapes = parse_shapes(args.shapes)
    
    print(f"AMX GEMM 自动调参 (tile + prefetch)")
    print(f"  (K,N) shapes: {shapes}")
    print(f"  cores={args.cores} ({num_threads} threads)")
    print(f"  M values: {M_list}")
    print(f"  Iters={BENCH_ITERS}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Skip prefetch: {args.skip_prefetch}")
    
    results = []
    
    for K, N in shapes:
        print(f"\n\n{'#'*60}")
        print(f"  Shape: K={K}, N={N}")
        print(f"{'#'*60}")
        
        weight = torch.randn(N, K, dtype=torch.bfloat16)
        amx_gemm_bf16_ext.pack_weight_nk(weight)
        print(f"  Weight packed: ({N}, {K}) -> VNNI layout")
        
        for M in M_list:
            result = search_for_M(M, K, N, num_threads, weight,
                                  quick=args.quick, skip_prefetch=args.skip_prefetch)
            results.append(asdict(result))
    
    amx_gemm_bf16_ext.set_hw_prefetch(True)

    # 输出汇总表
    print(f"\n\n{'='*155}")
    header = (f"{'K':>6} | {'N':>6} | {'M':>6} | {'TM':>6} | {'TN':>6} | {'TK':>6} | "
              f"{'pf_a':>5} | {'pf_b':>5} | {'pf_c':>5} | "
              f"{'a_d':>4} | {'b_d':>4} | {'c_d':>4} | {'hw_pf':>5} | "
              f"{'pf time(ms)':>10} | {'no pf time(ms)':>10} | {'GFLOPS':>10} | {'configs':>8}")
    print(header)
    print(f"{'-'*155}")
    for r in results:
        print(f"{r['K']:>6} | {r['N']:>6} | {r['M']:>6} | {r['best_TM']:>6} | {r['best_TN']:>6} | {r['best_TK']:>6} | "
              f"{'Y' if r['prefetch_a'] else 'N':>5} | {'Y' if r['prefetch_b'] else 'N':>5} | {'Y' if r['prefetch_c'] else 'N':>5} | "
              f"{r['pf_a_dist']:>4} | {r['pf_b_dist']:>4} | {r['pf_c_dist']:>4} | "
              f"{'Y' if r.get('hw_prefetch', True) else 'N':>5} | "
              f"{r['best_pf_time_ms']:>10.3f} | {r['best_no_pf_time_ms']:>10.3f} | {r['best_gflops']:>10.1f} | {r['total_configs']:>8}")
    print(f"{'='*155}")
    
    with open(args.output, "w") as f:
        json.dump({
            "config": {"shapes": [f"{k}x{n}" for k, n in shapes],
                       "cores": args.cores, "num_threads": num_threads,
                       "skip_prefetch": args.skip_prefetch},
            "results": results,
        }, f, indent=2)
    print(f"\nJSON 结果已保存到 {args.output}")

    json_to_py(args.output, args.py_output)

    print(f"\n// 推荐配置 (可粘贴到 C++ 或 Python):")
    print(f"// {num_threads} threads")
    for r in results:
        pfa_s = "true" if r['prefetch_a'] else "false"
        pfb_s = "true" if r['prefetch_b'] else "false"
        pfc_s = "true" if r['prefetch_c'] else "false"
        hwpf_s = "true" if r.get('hw_prefetch', True) else "false"
        print(f"// K={r['K']:>5}, N={r['N']:>5}, M={r['M']:>5}: "
              f"TM={r['best_TM']:>5}, TN={r['best_TN']:>5}, TK={r['best_TK']:>5}, "
              f"pfa={pfa_s}, pfb={pfb_s}, pfc={pfc_s}, "
              f"a_dist={r['pf_a_dist']}, b_dist={r['pf_b_dist']}, c_dist={r['pf_c_dist']}, "
              f"hw_pf={hwpf_s}  "
              f"// {r['best_gflops']:.1f} GFLOPS, pf:{r['best_pf_time_ms']:.3f} ms, nopf:{r['best_no_pf_time_ms']:.3f} ms")


if __name__ == "__main__":
    main()
