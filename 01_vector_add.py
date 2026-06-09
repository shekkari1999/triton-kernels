"""
Kernel 1: Vector Addition  (element-wise ops warm-up)

This is the "hello world" of GPU kernels.  It demonstrates the two core ideas
that every subsequent kernel builds on:

  1. PID mapping  — each parallel program ID owns a slice of the output tensor.
  2. Blocked processing + masking  — each PID handles BLOCK_SIZE elements at
     once, and out-of-bounds lanes are masked so we never write garbage.

Since this kernel does nothing but read two arrays and write one, it is
100 % memory-bound.  A well-written implementation should saturate HBM
bandwidth (roughly 900 GB/s on an A100).

Progression in this file:
  Step 1: one element per PID  (naive, huge scheduling overhead at scale)
  Step 2: BLOCK_SIZE elements per PID  (production style)
"""

import math
import time

import torch
import triton
import triton.language as tl

from utils import bench_ms, report_bandwidth, report_header


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CPU Pseudocode — understand the index math before touching Triton
# ──────────────────────────────────────────────────────────────────────────────

def vector_add_pseudocode(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Serial simulation of what Triton runs in parallel.

    Key insight: every PID is assigned a contiguous block of indices.
    pid=0 → [0 .. BLOCK_SIZE-1], pid=1 → [BLOCK_SIZE .. 2*BLOCK_SIZE-1], …

    The mask prevents writing beyond the end of the vector when n_elements
    is not a multiple of BLOCK_SIZE.
    """
    n = a.numel()
    BLOCK_SIZE = 1024
    output = torch.empty_like(a)

    for pid in range(math.ceil(n / BLOCK_SIZE)):
        offsets = pid * BLOCK_SIZE + torch.arange(BLOCK_SIZE)
        mask = offsets < n
        valid = offsets[mask]
        output[valid] = a[valid] + b[valid]

    return output


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Triton kernel — blocked vector add (BLOCK_SIZE elements per PID)
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _vector_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,   # constexpr so Triton can unroll loops & specialize
):
    """
    One kernel instance handles [pid*BLOCK_SIZE, (pid+1)*BLOCK_SIZE).

    tl.arange produces a vector of BLOCK_SIZE consecutive integers.
    The mask ensures we never read/write past the end of the arrays.
    Both tl.load and tl.store accept the same mask argument.
    """
    pid = tl.program_id(axis=0)

    # Compute the indices this PID is responsible for
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Tensors must be on GPU"
    assert a.shape == b.shape

    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    # ceil(n / BLOCK_SIZE) programs; each handles BLOCK_SIZE elements
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _vector_add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Bonus: autotuned version  (Triton searches over BLOCK_SIZE configs)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048, 4096]
        for nw in [4, 8]
    ],
    key=["n_elements"],
)
@triton.jit
def _vector_add_autotuned(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def vector_add_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    n = a.numel()
    grid = lambda args: (triton.cdiv(n, args["BLOCK_SIZE"]),)
    _vector_add_autotuned[grid](a, b, out, n)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main: correctness + benchmark
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Correctness ──────────────────────────────────────────────
    a = torch.randn(10_007, device="cuda")
    b = torch.randn(10_007, device="cuda")
    assert torch.allclose(vector_add_pseudocode(a.cpu(), b.cpu()).cuda(), a + b), "Pseudocode mismatch"
    assert torch.allclose(vector_add(a, b), a + b), "Triton mismatch"
    print("Correctness: PASSED")

    # ── HBM bandwidth benchmark ───────────────────────────────────
    # Claim: bandwidth-bound kernels reach ~75-85% of peak HBM
    # 3 tensor accesses: read a, read b, write out
    peak_bw, _ = report_header()[0], None

    print("─" * 72)
    print(f"  {'Kernel':<38} {'Time':>9}   {'Bandwidth':>11}   {'% Peak':>8}")
    print("─" * 72)
    for N in [10_000_000, 50_000_000, 100_000_000]:
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        n_bytes = 3 * N * a.element_size()

        ms_torch  = bench_ms(lambda: a + b)
        ms_triton = bench_ms(vector_add, a, b)
        ms_tuned  = bench_ms(vector_add_autotuned, a, b)

        report_bandwidth(f"PyTorch          (N={N//1_000_000}M)", ms_torch,  n_bytes, peak_bw)
        report_bandwidth(f"Triton fixed     (N={N//1_000_000}M)", ms_triton, n_bytes, peak_bw)
        report_bandwidth(f"Triton autotuned (N={N//1_000_000}M)", ms_tuned,  n_bytes, peak_bw)
        print()
    print("─" * 72)
    print("Target: ≥75% of peak HBM bandwidth (memory-bound kernel)")
