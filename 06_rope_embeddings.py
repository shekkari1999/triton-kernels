"""
Kernel 6: Fused RoPE  (Rotary Position Embeddings)

RoPE encodes absolute position into Q and K vectors by rotating pairs of
dimensions by an angle proportional to the token's position.  It is the
position encoding used by Llama, Qwen, Mistral, and most modern decoders.

Math:
    For token at position m and dimension pair (2i, 2i+1):

        θ_i     = m / base^(2i / D)     base=10000 (default)
        q'_{2i}   =  q_{2i}   · cos(θ_i) − q_{2i+1} · sin(θ_i)
        q'_{2i+1} =  q_{2i}   · sin(θ_i) + q_{2i+1} · cos(θ_i)

    The same rotation is applied to K.

Naive approach:
  1. Precompute cos/sin table: (max_seq, D/2)        → one kernel
  2. Apply rotation to Q                             → one kernel
  3. Apply rotation to K                             → one kernel
  Total: 3 kernels, cos/sin table written then re-read from HBM.

Fused approach:
  Compute cos/sin on the fly inside the rotation kernel — no table needed.
  One kernel for Q, one for K (or one combined), reading input/writing output
  in a single pass.

Input shapes:
    Q, K : (B, H, S, D)    B=batch, H=heads, S=seq_len, D=head_dim
    positions: (B, S)      integer token positions
"""

import math
import time
import pytest
import torch
import triton
import triton.language as tl

from utils import bench_ms, report_bandwidth, report_header


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PyTorch reference
# ──────────────────────────────────────────────────────────────────────────────

def precompute_freqs(seq_len: int, head_dim: int, base: float = 10000.0, device="cpu"):
    """Return (cos, sin) tables of shape (S, D/2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)   # (S, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x  : (B, H, S, D)
    cos, sin: (S, D/2)  broadcast over B and H
    """
    B, H, S, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]   # (B,H,S,D/2) each

    # Expand cos/sin to (1, 1, S, D/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_rot = torch.cat([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1)
    return x_rot


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Fused Triton kernel
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rope_kernel(
    out_ptr,
    x_ptr,
    pos_ptr,            # (B, S) integer positions
    B, H, S,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_pb, stride_ps,
    D: tl.constexpr,   # head_dim
    HALF: tl.constexpr,
    BASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Grid: (B * H * S,)
    Each program handles one (batch, head, token) — applies rotation to all D dims.

    We compute cos/sin on the fly rather than loading from a precomputed table:
      inv_freq_i = 1 / base^(2i/D)
      θ_i        = position * inv_freq_i
    """
    pid = tl.program_id(0)

    # Decode (b, h, s) from flat pid
    s_idx = pid % S
    h_idx = (pid // S) % H
    b_idx = pid // (S * H)

    # Load token position
    pos = tl.load(pos_ptr + b_idx * stride_pb + s_idx * stride_ps)
    pos = pos.to(tl.float32)

    # Dimension indices for the "first half"
    d_idx = tl.arange(0, BLOCK)        # [0, 1, …, HALF-1] padded to BLOCK
    mask  = d_idx < HALF

    # inv_freq_i = 1.0 / BASE^(d_idx / HALF)
    inv_freq = 1.0 / tl.exp(d_idx.to(tl.float32) * (tl.log(float(BASE)) / HALF))
    theta    = pos * inv_freq

    cos_val = tl.cos(theta)
    sin_val = tl.sin(theta)

    # Base pointer to this (b, h, s) row
    row_ptr = (
        x_ptr
        + b_idx * stride_xb
        + h_idx * stride_xh
        + s_idx * stride_xs
    )
    out_row = (
        out_ptr
        + b_idx * stride_ob
        + h_idx * stride_oh
        + s_idx * stride_os
    )

    # Load x1  (dims 0..HALF-1) and x2 (dims HALF..D-1)
    x1 = tl.load(row_ptr + d_idx * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(row_ptr + (d_idx + HALF) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    # Rotate
    y1 = x1 * cos_val - x2 * sin_val
    y2 = x1 * sin_val + x2 * cos_val

    # Write back
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_row + d_idx          * stride_od, y1.to(out_dtype), mask=mask)
    tl.store(out_row + (d_idx + HALF) * stride_od, y2.to(out_dtype), mask=mask)


def apply_rope_triton(
    x: torch.Tensor,                       # (B, H, S, D)
    positions: torch.Tensor,               # (B, S) int32/int64
    base: float = 10000.0,
) -> torch.Tensor:
    assert x.is_cuda
    B, H, S, D = x.shape
    assert D % 2 == 0
    HALF = D // 2
    BLOCK = triton.next_power_of_2(HALF)

    out = torch.empty_like(x)
    grid = (B * H * S,)
    _rope_kernel[grid](
        out, x, positions,
        B, H, S,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.stride(0),   x.stride(1),   x.stride(2),   x.stride(3),
        positions.stride(0), positions.stride(1),
        D=D, HALF=HALF, BASE=int(base), BLOCK=BLOCK,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("B,H,S,D", [
    (1, 4,  16, 64),
    (2, 8,  64, 128),
    (4, 16, 128, 64),
    (1, 1,  1024, 128),
])
def test_rope(B, H, S, D):
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    positions = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()

    cos, sin = precompute_freqs(S, D, device="cuda")
    ref  = apply_rope_torch(x, cos, sin)
    ours = apply_rope_triton(x, positions)

    assert torch.allclose(ref, ours, atol=1e-4), \
        f"FAILED  B={B} H={H} S={S} D={D}  max_diff={(ref-ours).abs().max():.2e}"
    print(f"  PASSED  B={B} H={H} S={S:5d} D={D}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Benchmark
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
    pytest.main([f"{pathlib.Path(__file__)}::test_rope", "-v", "-s"])

    # ── Bandwidth benchmark (claim: ~75-85% of peak HBM) ─────────
    peak_bw, _ = report_header()
    B, H, D = 4, 32, 128

    print("─" * 72)
    print(f"  {'Kernel (B=4, H=32, D=128)':<38} {'Time':>9}   {'Bandwidth':>11}   {'% Peak':>8}")
    print("─" * 72)
    for S in [512, 1024, 2048, 4096, 8192]:
        x   = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
        cos, sin = precompute_freqs(S, D, device="cuda")
        n_bytes = 2 * x.numel() * x.element_size()   # 1 read + 1 write

        ms_torch  = bench_ms(apply_rope_torch, x, cos, sin)
        ms_triton = bench_ms(apply_rope_triton, x, pos)

        report_bandwidth(f"PyTorch (+table)   S={S}", ms_torch,  n_bytes, peak_bw)
        report_bandwidth(f"Triton fused       S={S}", ms_triton, n_bytes, peak_bw)
        print()
    print("─" * 72)
    print("Target: fused Triton ≥75% of peak HBM (avoids reading cos/sin table)")
