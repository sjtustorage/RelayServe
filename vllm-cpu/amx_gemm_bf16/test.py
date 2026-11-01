"""
perf/硬件计数器按进程统计，无法只对某几行 Python 做硬件级切片。
用 --perf-stat 时增大 --bench-loops，使 GEMM 主导计数；LLC MPKI 由 perf 输出自行计算：
  MPKI ≈ (LLC-load-misses + LLC-store-misses) / instructions * 1000
"""
import torch
import amx_gemm_bf16_ext  # 编译后的扩展名
import time
import argparse
import os
import sys

class AMXLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, use_bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        self.weight_ref = self.weight.detach().clone()

        if use_bias:
            self.bias = torch.randn(out_features, dtype=torch.bfloat16)
        else:
            self.bias = None
        
        start_time = time.perf_counter()
        amx_gemm_bf16_ext.pack_weight_nk(self.weight)
        end_time = time.perf_counter()
        print(f"Pack time: {end_time - start_time:.4f} seconds")
        
    def forward(self, input, output,
                prefetch_a=False, prefetch_b=False, prefetch_c=False,
                pf_a_dist=2, pf_b_dist=2, pf_c_dist=1,
                hw_prefetch=True):
        amx_gemm_bf16_ext.gemm_bf16(
            input,
            self.weight,
            output,
            TM=512,
            TN=224,
            TK=1280,
            prefetch_a=prefetch_a,
            prefetch_b=prefetch_b,
            prefetch_c=prefetch_c,
            pf_a_dist=pf_a_dist,
            pf_b_dist=pf_b_dist,
            pf_c_dist=pf_c_dist,
            bias=self.bias,
            hw_prefetch=hw_prefetch,
        )

def verify_correctness(M, K, N, rtol=1e-2, atol=1e-1):
    """将 AMX GEMM 结果与 torch.matmul 标准矩阵乘进行对比 (无 bias)"""
    model = AMXLinear(K, N, use_bias=False)
    inp = torch.randn(M, K, dtype=torch.bfloat16)
    output = torch.empty((M, N), dtype=torch.bfloat16)

    model(inp, output)

    ref = torch.matmul(inp, model.weight_ref.T)

    close = torch.allclose(output, ref, rtol=rtol, atol=atol)
    max_diff = (output - ref).abs().max().item()
    mean_diff = (output - ref).abs().mean().item()

    print(f"[GEMM {M}x{K} @ {K}x{N}]  match={close}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    return close

def verify_correctness_with_bias(M, K, N, rtol=1e-2, atol=1e-1):
    """将 AMX GEMM+bias 结果与 torch.matmul+bias 对比"""
    model = AMXLinear(K, N, use_bias=True)
    inp = torch.randn(M, K, dtype=torch.bfloat16)
    output = torch.empty((M, N), dtype=torch.bfloat16)

    model(inp, output)

    ref = torch.matmul(inp, model.weight_ref.T) + model.bias.unsqueeze(0)

    close = torch.allclose(output, ref, rtol=rtol, atol=atol)
    max_diff = (output - ref).abs().max().item()
    mean_diff = (output - ref).abs().mean().item()

    print(f"[GEMM+bias {M}x{K} @ {K}x{N}]  match={close}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    if not close:
        print("  output sample:", output[0, :8])
        print("  ref    sample:", ref[0, :8])
    return close

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-N", type=int, default=12288)
    parser.add_argument("-cores", type=str, default="98-143", help="CPU 核心绑定")
    parser.add_argument("-verify", action="store_true", help="是否执行正确性验证")
    parser.add_argument("-verify-bias", action="store_true", help="是否执行 bias 正确性验证")
    parser.add_argument("-pfa", action="store_true", help="是否在性能测试中启用预取A")
    parser.add_argument("-pfb", action="store_true", help="是否在性能测试中启用预取B")
    parser.add_argument("-pfc", action="store_true", help="是否在性能测试中启用预取C")
    parser.add_argument("--hw-prefetch-off", action="store_true", help="关闭CPU硬件预取 (MSR 0x1A4)")
    parser.add_argument("--pf-a-dist", type=int, default=2, help="A 预取每次的行数 (default: 2)")
    parser.add_argument("--pf-b-dist", type=int, default=2, help="B 预取每次的块数 (default: 2)")
    parser.add_argument("--pf-c-dist", type=int, default=1, help="C 预取每次的行数 (default: 1)")
    parser.add_argument("--use-bias", action="store_true", help="性能测试中使用 bias")
    parser.add_argument(
        "--bench-loops",
        type=int,
        default=100,
        help="性能测试中 forward 循环次数；配合 perf 时建议调大以摊薄解释器/初始化噪声",
    )
    parser.add_argument(
        "--perf-stat",
        action="store_true",
        help="用 perf stat 统计 cycles、instructions、LLC-load/store-misses（整进程；见文件头说明算 MPKI）",
    )
    args = parser.parse_args()

    if args.perf_stat and os.environ.get("AMX_GEMM_PERF_WRAPPED") != "1":
        os.environ["AMX_GEMM_PERF_WRAPPED"] = "1"
        perf_argv = [
            "perf",
            "stat",
            "-e",
            "cycles,instructions,LLC-load-misses,LLC-store-misses",
            "--",
            sys.executable,
            __file__,
        ] + [a for a in sys.argv[1:] if a != "--perf-stat"]
        os.execvp("perf", perf_argv)

    amx_gemm_bf16_ext.init_cpu_threads_env(args.cores)

    hw_pf = not args.hw_prefetch_off

    if args.verify:
        verify_correctness(args.M, args.K, args.N)

    if args.verify_bias:
        verify_correctness_with_bias(args.M, args.K, args.N)

    # 性能测试
    print("\n=== Performance Benchmark ===")
    model = AMXLinear(args.K, args.N, use_bias=args.use_bias)
    input_tensor = torch.randn(args.M, args.K, dtype=torch.bfloat16)
    output = torch.empty((args.M, args.N), dtype=torch.bfloat16)

    # warmup
    model(input_tensor, output,
          prefetch_a=args.pfa, prefetch_b=args.pfb, prefetch_c=args.pfc,
          pf_a_dist=args.pf_a_dist, pf_b_dist=args.pf_b_dist, pf_c_dist=args.pf_c_dist,
          hw_prefetch=hw_pf)


    times = []
    for i in range(args.bench_loops):
        flush_cache()
        time0 = time.time()
        model(input_tensor, output,
              prefetch_a=args.pfa, prefetch_b=args.pfb, prefetch_c=args.pfc,
              pf_a_dist=args.pf_a_dist, pf_b_dist=args.pf_b_dist, pf_c_dist=args.pf_c_dist,
              hw_prefetch=hw_pf)
        time1 = time.time()
        times.append(time1-time0)
        print(f"time: {(time1-time0)*1000}ms")

    ms_per_iter = sum(times) * 1000 / args.bench_loops
    flops = 2.0 * args.M * args.N * args.K
    gflops = flops / (ms_per_iter / 1000) / 1e9
    print(f"Time: {ms_per_iter:.3f} ms per iter (over {args.bench_loops} iters), {gflops:.1f} GFLOPS")
    print(f"  sw prefetch: A={args.pfa}(dist={args.pf_a_dist}) B={args.pfb}(dist={args.pf_b_dist}) C={args.pfc}(dist={args.pf_c_dist})")
    print(f"  hw prefetch: {'off' if args.hw_prefetch_off else 'on'}")
    if args.use_bias:
        print(f"  bias: enabled")
