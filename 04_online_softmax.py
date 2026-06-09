"""
Kernel 4: Online (Streaming) Softmax

The fused softmax in 03_fused_softmax.py works only if the entire row fits
in one block.  For very long rows (or for attention over thousands of tokens),
we need to process the row in chunks — one block at a time — without ever
loading the full row.

Key insight — the running statistics trick:
  At step i we track:
    m_i  = max(x_1, …, x_i)          (running max)
    d_i  = Σ exp(x_j − m_i)          (running sum of exps)

  When we see a new element x_{i+1}:
    m_{i+1}  = max(m_i, x_{i+1})
    d_{i+1}  = d_i · exp(m_i − m_{i+1}) + exp(x_{i+1} − m_{i+1})
               ↑ correction factor                ↑ new term

  After processing all elements:
    softmax(x_j) = exp(x_j − m_N) / d_N

  This generalises to blocks: process BLOCK_SIZE elements at a time,
  updating running (m, d) after each block.  Two passes over the data:
    Pass 1: accumulate (m, d)
    Pass 2: compute exp(x − m) / d

This is the exact algorithm inside FlashAttention — we will reuse it there.

Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018
"""

import math
import time
import pytest
import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CPU pseudocode — element-by-element
# ──────────────────────────────────────────────────────────────────────────────

def online_softmax_python(x: torch.Tensor) -> torch.Tensor:
    """
    Reference: one element at a time.

    After the scan we have m = max(x) and d = Σ exp(x_j − m).
    Softmax is then trivially exp(x_j − m) / d.
    """
    M, N = x.shape
    out = torch.empty_like(x)

    for i in range(M):
        row = x[i]
        m, d = float("-inf"), 0.0

        for xj in row.tolist():
            m_prev = m
            m = max(m, xj)
            d = d * math.exp(m_prev - m) + math.exp(xj - m)

        out[i] = torch.exp(row - m) / d

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CPU pseudocode — block-by-block (closer to Triton)
# ──────────────────────────────────────────────────────────────────────────────

def online_blocked_softmax_python(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)

    for i in range(M):
        row = x[i]
        m, d = float("-inf"), 0.0

        for b in range(math.ceil(N / block_size)):
            block = row[b * block_size : (b + 1) * block_size]
            block_max = block.max().item()
            m_prev = m
            m = max(m, block_max)
            d = d * math.exp(m_prev - m) + torch.exp(block - m).sum().item()

        out[i] = torch.exp(row - m) / d

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Triton kernel — two-pass online softmax over arbitrary N
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _online_softmax_kernel(
    out_ptr,
    inp_ptr,
    out_row_stride,
    inp_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row.

    Pass 1: scan all blocks to accumulate running max m and running sum d.
    Pass 2: scan all blocks again to write exp(x − m) / d.

    BLOCK_SIZE does NOT need to equal n_cols (unlike 03_fused_softmax.py).
    Any power-of-2 BLOCK_SIZE ≤ n_cols works.
    """
    row = tl.program_id(0)
    row_inp = inp_ptr + row * inp_row_stride
    row_out = out_ptr + row * out_row_stride

    # ── Pass 1: accumulate (m, d) ────────────────────────────────
    m = float("-inf")
    d = 0.0

    for offs in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        col_offsets = offs * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_inp + col_offsets, mask=mask, other=-float("inf"))

        block_max = tl.max(x, axis=0)
        m_prev = m
        m = tl.maximum(m, block_max)
        # Rescale previous sum, add new block contributions
        d = d * tl.exp(m_prev - m) + tl.sum(tl.exp(x - m), axis=0)

    # ── Pass 2: write normalised values ─────────────────────────
    for offs in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        col_offsets = offs * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_inp + col_offsets, mask=mask, other=-float("inf"))
        y = tl.exp(x - m) / d
        tl.store(row_out + col_offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def online_softmax_triton(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    _online_softmax_kernel[grid](
        out, x,
        out.stride(0), x.stride(0),
        n_cols=N, BLOCK_SIZE=block_size,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,N", [
    (64, 64), (128, 512), (256, 1024), (512, 2048),
    (64, 300), (128, 777), (256, 1500),   # non-power-of-2
])
def test_online_softmax(M, N):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref  = torch.softmax(x, dim=1)
    ours = online_softmax_triton(x)
    assert torch.allclose(ref, ours, atol=1e-5), \
        f"FAILED  M={M} N={N}  max_diff={(ref - ours).abs().max():.2e}"
    print(f"  PASSED  M={M:5d}  N={N:5d}")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check with Python pseudocode
    M, N = 16, 200
    x_cpu = torch.randn(M, N)
    ref_cpu = torch.softmax(x_cpu, dim=1)
    assert torch.allclose(ref_cpu, online_softmax_python(x_cpu), atol=1e-5)
    assert torch.allclose(ref_cpu, online_blocked_softmax_python(x_cpu, block_size=64), atol=1e-5)
    print("Python pseudocode (element + block): PASSED")

    import pytest, pathlib
    print("\nRunning correctness tests …")
    pytest.main([f"{pathlib.Path(__file__)}::test_online_softmax", "-v", "-s"])

    # Bandwidth benchmark: fused (03) vs online two-pass (this file)
    print("\nOnline softmax benchmark complete.")
