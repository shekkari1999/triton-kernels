"""
Shared benchmark utilities for all Triton kernel files.

Provides:
  - GPU property detection (name, peak HBM bandwidth, peak FLOP/s)
  - Consistent timing via triton.testing.do_bench
  - Bandwidth and FLOP/s reporters that show % of theoretical peak
  - Peak GPU memory measurement via torch.cuda.max_memory_allocated
"""

import torch
import triton

# ── Hardware tables ───────────────────────────────────────────────────────────

# Peak HBM bandwidth in GB/s (from spec sheets)
_PEAK_HBM = {
    "A100 SXM": 2039,
    "A100":     2039,
    "H100 SXM": 3350,
    "H100":     3350,
    "H200":     4800,
    "V100 SXM": 900,
    "V100":     900,
    "RTX 4090": 1008,
    "RTX 4080": 717,
    "RTX 3090": 936,
    "RTX 3080": 760,
    "RTX 2080": 448,
    "A10":      600,
    "A30":      933,
    "A40":      696,
    "L40":      864,
    "L4":       300,
}

# Peak fp16 tensor-core TFLOP/s (from spec sheets)
_PEAK_TFLOPS = {
    "A100 SXM": 312,
    "A100":     312,
    "H100 SXM": 989,
    "H100":     989,
    "H200":    1979,
    "V100 SXM": 112,
    "V100":     112,
    "RTX 4090": 165,
    "RTX 4080":  97,
    "RTX 3090":  71,
    "RTX 3080":  68,
    "A10":       62,
    "A30":      165,
    "A40":      149,
    "L40":      181,
    "L4":        30,
}


def get_device_info() -> tuple[str, float, float]:
    """
    Returns (device_name, peak_hbm_gbs, peak_fp16_tflops).
    Falls back to conservative estimates if the GPU is not in the table.
    """
    if not torch.cuda.is_available():
        return "CPU (no CUDA)", 50.0, 1.0

    name = torch.cuda.get_device_properties(0).name

    peak_bw = 500.0    # conservative fallback
    for key, bw in _PEAK_HBM.items():
        if key in name:
            peak_bw = float(bw)
            break

    peak_tf = 50.0
    for key, tf in _PEAK_TFLOPS.items():
        if key in name:
            peak_tf = float(tf)
            break

    return name, peak_bw, peak_tf


def bench_ms(fn, *args, warmup: int = 25, rep: int = 100) -> float:
    """
    Returns median wall-time in milliseconds.
    Uses triton.testing.do_bench which warms up the kernel and
    handles CUDA synchronisation correctly.
    """
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


def report_header():
    name, bw, tf = get_device_info()
    print(f"GPU : {name}")
    print(f"Peak HBM bandwidth : {bw:,} GB/s")
    print(f"Peak fp16 TFLOP/s  : {tf:,} TFLOP/s")
    print()
    return bw, tf


def report_bandwidth(label: str, ms: float, n_bytes: int, peak_gbs: float):
    """
    Print one benchmark row for a bandwidth-bound kernel.
    n_bytes = total bytes read + written.
    """
    achieved_gbs = n_bytes / (ms * 1e-3) / 1e9
    pct = achieved_gbs / peak_gbs * 100
    status = "✓" if pct >= 75 else "~"
    print(f"  {label:<38} {ms:>7.3f} ms   "
          f"{achieved_gbs:>8.1f} GB/s   "
          f"({pct:5.1f}% of peak)  {status}")


def report_flops(label: str, ms: float, flops: int, peak_tflops: float):
    """
    Print one benchmark row for a compute-bound kernel.
    """
    achieved = flops / (ms * 1e-3) / 1e12
    pct = achieved / peak_tflops * 100
    print(f"  {label:<38} {ms:>7.3f} ms   "
          f"{achieved:>8.1f} TFLOP/s  "
          f"({pct:5.1f}% of peak)")


def measure_peak_memory_mb(fn, *args) -> float:
    """
    Returns the peak GPU memory (MB) allocated during fn(*args).
    Resets peak stats before the call so the result is isolated.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6
