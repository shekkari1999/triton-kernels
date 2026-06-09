"""
Kernel 5: Fused RMSNorm  (Root Mean Square Layer Normalization)

RMSNorm is used by Llama, Qwen, Mistral, and most modern LLMs instead
of LayerNorm because it is cheaper (no mean subtraction) and equally
effective in practice.

Formula:
    y = x / RMS(x) * γ
    where  RMS(x) = sqrt( mean(x²) + ε )
                  = sqrt( (1/N) Σ x_i² + ε )
    γ  is a learnable scale parameter (same shape as the feature dimension).

Naive PyTorch launches 4 kernels: square → mean → sqrt+add → divide → scale.

Fused Triton version: one pass reads x, computes RMS, then normalises and
scales in the same pass.  A second kernel handles the backward (dX, dγ).

Shapes:
    x:  (M, N)   M rows, N features
    γ:  (N,)
    y:  (M, N)
"""

import time
import pytest
import torch
import triton
import triton.language as tl

from utils import bench_ms, report_bandwidth, report_header


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PyTorch reference
# ──────────────────────────────────────────────────────────────────────────────

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Fused forward kernel
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rstd_ptr,           # stores 1/RMS per row (needed for backward)
    x_row_stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row.
    Accumulates sum-of-squares in float32 for numerical precision
    regardless of input dtype.
    """
    row = tl.program_id(0)
    x_start = x_ptr + row * x_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_start + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS = sqrt(mean(x^2) + eps)
    sq_sum = tl.sum(x * x, axis=0)
    rms    = tl.sqrt(sq_sum / N + eps)
    rstd   = 1.0 / rms

    # Normalise and scale
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd) * w

    # Write output
    tl.store(out_ptr + row * x_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
    # Store reciprocal std for the backward pass
    tl.store(rstd_ptr + row, rstd)


def rmsnorm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert x.is_cuda and weight.is_cuda
    M, N = x.shape
    assert weight.shape == (N,)

    out  = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](
        x, weight, out, rstd,
        x.stride(0), N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out, rstd


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Backward kernel (dX and dγ)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rmsnorm_bwd_kernel(
    dx_ptr,
    dw_ptr,           # accumulates dL/dγ across rows
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    x_row_stride,
    dy_row_stride,
    dx_row_stride,
    M, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dX derivation (one row):
        y   = x_hat * w   where  x_hat = x * rstd
        dy  = upstream gradient

        dL/dx_hat = dy * w
        dL/dx = rstd * [ dL/dx_hat − (1/N) * x_hat * sum(dL/dx_hat * x_hat) ]
              = rstd * [ c1 − (c2/N) * x_hat ]
        where
          c1 = dy * w
          c2 = sum(dy * w * x_hat)  (scalar per row)
    """
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x    = tl.load(x_ptr  + row * x_row_stride  + cols, mask=mask, other=0.0).to(tl.float32)
    dy   = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w    = tl.load(w_ptr  + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    x_hat = x * rstd
    c1    = dy * w
    c2    = tl.sum(c1 * x_hat, axis=0)

    dx = rstd * (c1 - (c2 / N) * x_hat)
    tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

    # Accumulate dγ: sum across rows in dw_ptr atomically
    tl.atomic_add(dw_ptr + cols, (dy * x_hat).to(dw_ptr.dtype.element_ty), mask=mask)


def rmsnorm_bwd(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor):
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight)  # accumulates atomically
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _rmsnorm_bwd_kernel[grid](
        dx, dw, dy, x, weight, rstd,
        x.stride(0), dy.stride(0), dx.stride(0),
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx, dw


# ──────────────────────────────────────────────────────────────────────────────
# 4.  nn.Module wrapper with autograd
# ──────────────────────────────────────────────────────────────────────────────

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        out, rstd = rmsnorm_fwd(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_bwd(dy, x, weight, rstd)
        return dx, dw, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", [
    (64, 128), (128, 512), (256, 1024), (512, 4096),
    (64, 300), (128, 777),   # non-power-of-2
])
def test_rmsnorm_fwd(M, N):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    w = torch.randn(N,    device="cuda", dtype=torch.float32)

    ref  = rmsnorm_torch(x, w)
    ours, _ = rmsnorm_fwd(x, w)

    assert torch.allclose(ref, ours, atol=1e-4), \
        f"FAILED  M={M} N={N}  max_diff={(ref - ours).abs().max():.2e}"
    print(f"  PASSED  M={M:5d}  N={N:5d}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _time_ms(fn, *args, warmup=20, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    import pytest, pathlib
    print("Running correctness tests …")
    pytest.main([f"{pathlib.Path(__file__)}::test_rmsnorm_fwd", "-v", "-s"])

    # ── Bandwidth benchmark (claim: ~75-85% of peak HBM) ─────────
    peak_bw, _ = report_header()
    M = 4096

    print("─" * 72)
    print(f"  {'Kernel (M=4096)':<38} {'Time':>9}   {'Bandwidth':>11}   {'% Peak':>8}")
    print("─" * 72)
    for N in [512, 1024, 2048, 4096, 8192]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        w = torch.ones(N, device="cuda", dtype=torch.float32)
        # read x + read w + write out  (3 tensor accesses)
        n_bytes = (2 * M * N + N) * x.element_size()

        ms_torch  = bench_ms(rmsnorm_torch, x, w)
        ms_triton = bench_ms(lambda: rmsnorm_fwd(x, w))

        report_bandwidth(f"PyTorch RMSNorm    N={N}", ms_torch,  n_bytes, peak_bw)
        report_bandwidth(f"Triton fused       N={N}", ms_triton, n_bytes, peak_bw)
        print()
    print("─" * 72)
    print("Target: fused Triton ≥75% of peak HBM bandwidth")
