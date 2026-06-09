"""
Kernel 3: Fused Softmax

Problem with naive (unfused) softmax:
  Each PyTorch op (max, subtract, exp, sum, divide) triggers a separate
  GPU kernel → 5 full passes over the input tensor → 5× HBM round trips.

Fused softmax:
  We do everything in one kernel launch:
    1. Load an entire row into SRAM
    2. Compute max for numerical stability (subtract before exp)
    3. Compute exp of every element
    4. Sum the exps
    5. Divide every element by the sum
    6. Write output

  Result: 1 read + 1 write → near-peak bandwidth utilization.

  One kernel block ↔ one row of the matrix.

Limitation:
  BLOCK_SIZE must cover the entire row in a single block.
  If N > hardware max thread count per block (usually 1024 threads × 4 bytes),
  Triton loops internally — you don't need to change anything.
"""

import time
import pytest
import torch
import triton
import triton.language as tl

from utils import bench_ms, report_bandwidth, report_header


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PyTorch naive (5 separate kernel launches)
# ──────────────────────────────────────────────────────────────────────────────

def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    row_max = x.max(dim=1, keepdim=True).values       # kernel 1
    x_shifted = x - row_max                            # kernel 2
    x_exp = x_shifted.exp()                            # kernel 3
    row_sum = x_exp.sum(dim=1, keepdim=True)           # kernel 4
    return x_exp / row_sum                             # kernel 5


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Triton fused kernel
# ──────────────────────────────────────────────────────────────────────────────

def _num_warps(block_size: int) -> int:
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


@triton.heuristics({"num_warps": lambda args: _num_warps(args["BLOCK_SIZE"])})
@triton.jit
def _softmax_fwd_kernel(
    out_ptr,
    inp_ptr,
    inp_row_stride,   # how many elements to jump to get to next row
    out_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: one program per row.
    Each program:
      ① loads the row, masking cols >= n_cols with -inf (so max/sum ignore them)
      ② computes stable softmax in registers
      ③ writes the normalised row back to HBM
    """
    row = tl.program_id(0)

    # Pointer to the start of our row
    row_inp_ptr = inp_ptr + row * inp_row_stride
    row_out_ptr = out_ptr + row * out_row_stride

    # Column offsets for this block; BLOCK_SIZE >= n_cols (next power of 2)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load — invalid lanes get -inf so they don't affect max / sum
    x = tl.load(row_inp_ptr + cols, mask=mask, other=-float("inf"))

    # Stable softmax: subtract row max before exp to avoid overflow
    x_shifted = x - tl.max(x, axis=0)
    x_exp     = tl.exp(x_shifted)
    x_norm    = x_exp / tl.sum(x_exp, axis=0)

    # Cast to output dtype (handles float32→float16 etc.)
    x_norm = x_norm.to(out_ptr.dtype.element_ty)

    tl.store(row_out_ptr + cols, x_norm, mask=mask)


def softmax_fused(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape

    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)   # must cover the whole row

    grid = (M,)
    _softmax_fwd_kernel[grid](
        out, x,
        x.stride(0), out.stride(0),
        n_cols=N, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", [
    (64, 64), (128, 128), (256, 512), (512, 256),
    (1024, 1024), (2048, 2048),
    (128, 100), (256, 777),    # non-power-of-2 columns
])
def test_softmax_fused(M, N):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref  = softmax_naive(x)
    ours = softmax_fused(x)
    assert torch.allclose(ref, ours, atol=1e-5), \
        f"FAILED M={M} N={N}  max_diff={( ref-ours).abs().max():.2e}"
    print(f"  PASSED  M={M:5d}  N={N:5d}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _time_ms(fn, x, warmup=20, iters=200):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _bw_gbs(ms, x):
    # 2 × tensor bytes: one read, one write
    return 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9


if __name__ == "__main__":
    import pytest, pathlib
    print("Running correctness tests …")
    pytest.main([f"{pathlib.Path(__file__)}::test_softmax_fused", "-v", "-s"])

    # ── Bandwidth benchmark (claim: ~75-85% of peak HBM) ─────────
    peak_bw, _ = report_header()
    M = 4096

    print("─" * 72)
    print(f"  {'Kernel (M=4096)':<38} {'Time':>9}   {'Bandwidth':>11}   {'% Peak':>8}")
    print("─" * 72)
    for N in [256, 512, 1024, 2048, 4096]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        n_bytes = 2 * x.numel() * x.element_size()  # 1 read + 1 write

        ms_naive  = bench_ms(softmax_naive, x)
        ms_torch  = bench_ms(lambda t: torch.softmax(t, dim=1), x)
        ms_triton = bench_ms(softmax_fused, x)

        report_bandwidth(f"Naive (unfused)    N={N}", ms_naive,  n_bytes, peak_bw)
        report_bandwidth(f"torch.softmax      N={N}", ms_torch,  n_bytes, peak_bw)
        report_bandwidth(f"Triton fused       N={N}", ms_triton, n_bytes, peak_bw)
        print()
    print("─" * 72)
    print("Target: fused Triton ≥75% of peak HBM; unfused naive should be lower")
