"""
Run every kernel benchmark and save results/benchmarks/results/latest.json.

Usage (from repo root):
  python benchmarks/run_all.py
  python benchmarks/run_all.py --save benchmarks/results/my_run.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import bench_ms, get_device_info, report_bandwidth, report_flops


def _load_kernel(filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sep(title: str = ""):
    w = 72
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * w)


def _peak_mem_delta_mb(fn, *args) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    fn(*args)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def _bw_pct(n_bytes: int, ms: float, peak_gbs: float) -> float:
    achieved = n_bytes / (ms * 1e-3) / 1e9
    return achieved / peak_gbs * 100


def _tflops_pct(flops: int, ms: float, peak_tf: float) -> float:
    achieved = flops / (ms * 1e-3) / 1e12
    return achieved / peak_tf * 100


def run(save_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    k1 = _load_kernel("01_vector_add.py")
    k2 = _load_kernel("02_matmul.py")
    k3 = _load_kernel("03_fused_softmax.py")
    k4 = _load_kernel("04_online_softmax.py")
    k5 = _load_kernel("05_layer_norm.py")
    k6 = _load_kernel("06_rope_embeddings.py")
    k7 = _load_kernel("07_flash_attention.py")

    gpu_name, peak_bw, peak_tf = get_device_info()
    out: dict = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_name,
            "peak_hbm_gbs": peak_bw,
            "peak_fp16_tflops": peak_tf,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "kernels": {},
    }

    print("=" * 72)
    print("  Triton Kernels for LLM Inference: Benchmark Suite")
    print("=" * 72)
    print(f"  GPU              : {gpu_name}")
    print(f"  Peak HBM         : {peak_bw:,} GB/s")
    print(f"  Peak fp16 Tensor : {peak_tf:,} TFLOP/s")
    print("=" * 72)

    # ── 01 Vector add ─────────────────────────────────────────────────────
    _sep("01 · Vector Add")
    N = 100_000_000
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    n_bytes = 3 * N * a.element_size()
    ms_torch = bench_ms(lambda: a + b)
    ms_triton = bench_ms(k1.vector_add_autotuned, a, b)
    report_bandwidth("PyTorch", ms_torch, n_bytes, peak_bw)
    report_bandwidth("Triton autotuned", ms_triton, n_bytes, peak_bw)
    out["kernels"]["vector_add"] = {
        "n_elements": N,
        "pytorch_ms": ms_torch,
        "triton_ms": ms_triton,
        "hbm_pct": _bw_pct(n_bytes, ms_triton, peak_bw),
    }

    # ── 02 GEMM ───────────────────────────────────────────────────────────
    _sep("02 · Tiled GEMM")
    size = 4096
    A = torch.randn(size, size, device="cuda", dtype=torch.float16)
    B = torch.randn(size, size, device="cuda", dtype=torch.float16)
    flops = 2 * size**3
    ms_cublas = bench_ms(lambda: torch.mm(A, B))
    ms_triton = bench_ms(k2.matmul, A, B)
    report_flops("cuBLAS", ms_cublas, flops, peak_tf)
    report_flops("Triton grouped", ms_triton, flops, peak_tf)
    cublas_tflops = flops / (ms_cublas * 1e-3) / 1e12
    triton_tflops = flops / (ms_triton * 1e-3) / 1e12
    print(f"  → Triton is {triton_tflops / cublas_tflops * 100:.0f}% of cuBLAS throughput")
    out["kernels"]["gemm"] = {
        "size": size,
        "cublas_ms": ms_cublas,
        "triton_ms": ms_triton,
        "cublas_tflops": cublas_tflops,
        "triton_tflops": triton_tflops,
        "pct_of_cublas": triton_tflops / cublas_tflops * 100,
        "pct_peak_fp16": _tflops_pct(flops, ms_triton, peak_tf),
    }

    # ── 03 Fused softmax ──────────────────────────────────────────────────
    _sep("03 · Fused Softmax")
    M, N_soft = 4096, 4096
    x = torch.randn(M, N_soft, device="cuda", dtype=torch.float32)
    n_bytes = 2 * x.numel() * x.element_size()
    ms_naive = bench_ms(k3.softmax_naive, x)
    ms_fused = bench_ms(k3.softmax_fused, x)
    report_bandwidth("Naive (5 kernels)", ms_naive, n_bytes, peak_bw)
    report_bandwidth("Triton fused", ms_fused, n_bytes, peak_bw)
    out["kernels"]["fused_softmax"] = {
        "shape": [M, N_soft],
        "naive_ms": ms_naive,
        "fused_ms": ms_fused,
        "fusion_speedup": ms_naive / ms_fused,
        "hbm_pct": _bw_pct(n_bytes, ms_fused, peak_bw),
    }

    # ── 04 Online softmax (two-pass; bridge to FA) ────────────────────────
    _sep("04 · Online Softmax (two-pass)")
    ms_online = bench_ms(k4.online_softmax_triton, x)
    report_bandwidth("Triton online (2-pass)", ms_online, n_bytes, peak_bw)
    out["kernels"]["online_softmax"] = {
        "shape": [M, N_soft],
        "online_ms": ms_online,
        "hbm_pct": _bw_pct(n_bytes, ms_online, peak_bw),
        "vs_fused_speedup": ms_online / ms_fused,
    }

    # ── 05 RMSNorm ────────────────────────────────────────────────────────
    _sep("05 · RMSNorm")
    w = torch.ones(N_soft, device="cuda", dtype=torch.float32)
    n_bytes_rms = (2 * M * N_soft + N_soft) * x.element_size()
    ms_rms_torch = bench_ms(k5.rmsnorm_torch, x, w)
    ms_rms_triton = bench_ms(lambda: k5.rmsnorm_fwd(x, w))
    report_bandwidth("PyTorch", ms_rms_torch, n_bytes_rms, peak_bw)
    report_bandwidth("Triton fused", ms_rms_triton, n_bytes_rms, peak_bw)
    out["kernels"]["rmsnorm"] = {
        "shape": [M, N_soft],
        "pytorch_ms": ms_rms_torch,
        "triton_ms": ms_rms_triton,
        "hbm_pct": _bw_pct(n_bytes_rms, ms_rms_triton, peak_bw),
    }

    # ── 06 RoPE ───────────────────────────────────────────────────────────
    _sep("06 · RoPE")
    B, H, S, D = 4, 32, 4096, 128
    x_rope = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    cos, sin = k6.precompute_freqs(S, D, device="cuda")
    n_bytes_rope = 2 * x_rope.numel() * x_rope.element_size()
    ms_rope_torch = bench_ms(k6.apply_rope_torch, x_rope, cos, sin)
    ms_rope_triton = bench_ms(k6.apply_rope_triton, x_rope, pos)
    report_bandwidth("PyTorch (+freq table)", ms_rope_torch, n_bytes_rope, peak_bw)
    report_bandwidth("Triton fused", ms_rope_triton, n_bytes_rope, peak_bw)
    out["kernels"]["rope"] = {
        "shape": [B, H, S, D],
        "pytorch_ms": ms_rope_torch,
        "triton_ms": ms_rope_triton,
        "hbm_pct": _bw_pct(n_bytes_rope, ms_rope_triton, peak_bw),
    }

    # ── 07 FlashAttention ─────────────────────────────────────────────────
    _sep("07 · FlashAttention-2 forward")
    B_fa, H_fa, D_fa = 1, 8, 64
    fa_sweep = []
    print(f"  {'S':>6}  {'Naive ms':>10}  {'FA ms':>10}  {'Speedup':>9}  "
          f"{'Naive MB':>10}  {'FA MB':>8}  {'Mem×':>8}")
    print("  " + "─" * 62)
    for S in [512, 1024, 2048, 4096, 8192, 16384]:
        Q = torch.randn(B_fa, H_fa, S, D_fa, device="cuda", dtype=torch.float16)
        K = torch.randn(B_fa, H_fa, S, D_fa, device="cuda", dtype=torch.float16)
        V = torch.randn(B_fa, H_fa, S, D_fa, device="cuda", dtype=torch.float16)
        try:
            ms_naive = bench_ms(k7.naive_attention, Q, K, V, is_causal=True)
            ms_fa = bench_ms(k7.flash_attention_forward, Q, K, V, is_causal=True)
            mem_naive = _peak_mem_delta_mb(
                lambda: k7.naive_attention(Q, K, V, is_causal=True)
            )
            mem_fa = _peak_mem_delta_mb(
                lambda: k7.flash_attention_forward(Q, K, V, is_causal=True)
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {S:>6}  OOM, skipping longer sequences")
            torch.cuda.empty_cache()
            break
        speedup = ms_naive / ms_fa
        mem_ratio = mem_naive / mem_fa if mem_fa > 0.5 else None
        mem_str = f"{mem_ratio:>7.1f}×" if mem_ratio is not None else "    n/a"
        print(
            f"  {S:>6}  {ms_naive:>10.3f}  {ms_fa:>10.3f}  {speedup:>8.2f}×  "
            f"{mem_naive:>10.1f}  {mem_fa:>8.1f}  {mem_str}"
        )
        fa_sweep.append(
            {
                "seq_len": S,
                "naive_ms": ms_naive,
                "fa_ms": ms_fa,
                "speedup": speedup,
                "naive_mem_mb": mem_naive,
                "fa_mem_mb": mem_fa,
                "mem_ratio": mem_ratio,
            }
        )

    out["kernels"]["flash_attention"] = {
        "config": {"batch": B_fa, "heads": H_fa, "head_dim": D_fa, "causal": True},
        "sweep": fa_sweep,
    }
    if fa_sweep:
        at_8k = next((r for r in fa_sweep if r["seq_len"] == 8192), fa_sweep[-1])
        out["kernels"]["flash_attention"]["headline"] = {
            "seq_len": at_8k["seq_len"],
            "speedup": at_8k["speedup"],
            "mem_ratio": at_8k["mem_ratio"],
        }

    # ── Summary ───────────────────────────────────────────────────────────
    _sep("Summary")
    bw_kernels = ["vector_add", "fused_softmax", "rmsnorm", "rope"]
    for key in bw_kernels:
        pct = out["kernels"][key]["hbm_pct"]
        print(f"  {key:<20}  {pct:5.1f}% peak HBM")
    gemm = out["kernels"]["gemm"]
    print(f"  {'gemm':<20}  {gemm['pct_peak_fp16']:5.1f}% peak fp16  ({gemm['pct_of_cublas']:.0f}% of cuBLAS)")
    if fa_sweep:
        hl = out["kernels"]["flash_attention"].get("headline", {})
        print(
            f"  {'flash_attention':<20}  {hl.get('speedup', 0):.2f}× speedup @ S={hl.get('seq_len')}  "
            f"mem {hl.get('mem_ratio', 0):.1f}×"
        )

    _sep()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved → {save_path}")
    print("Plots → python benchmarks/plot_results.py")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("benchmarks/results/latest.json"),
        help="JSON output path (relative to repo root)",
    )
    args = parser.parse_args()
    save = args.save if args.save.is_absolute() else ROOT / args.save
    run(save)


if __name__ == "__main__":
    main()
