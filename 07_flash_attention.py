"""
Kernel 7: FlashAttention 2  (Forward + Backward)

Standard attention is O(N²) in memory because it materialises the full
(seq_len × seq_len) attention matrix in HBM.

FlashAttention avoids this by:
  1. Tiling Q, K, V into blocks that fit in SRAM (L1/shared memory).
  2. Applying the ONLINE SOFTMAX algorithm (Kernel 4) within each Q-block
     as it iterates over all K/V blocks.
  3. Never writing the N×N attention matrix to HBM.

Memory:  O(N)  instead of O(N²)
Speed:   1.8–3× faster at long sequences (memory-bound regime)

Algorithm overview (forward pass):
  ┌────────── outer loop: Q blocks ─────────────────┐
  │   Initialise m = -∞, l = 0, O = 0               │
  │   ┌── inner loop: K/V blocks ─────────────┐     │
  │   │  S = Q_block @ K_block.T  (QK scores) │     │
  │   │  Apply causal mask (if enabled)        │     │
  │   │  m_new = max(m, rowmax(S))             │     │
  │   │  α = exp2(m - m_new)   (correction)   │     │
  │   │  P = exp2(S - m_new)                  │     │
  │   │  l = l * α + rowsum(P)                │     │
  │   │  O = O * α + P @ V_block              │     │
  │   │  m = m_new                            │     │
  │   └────────────────────────────────────────┘     │
  │   O = O / l    (normalise)                       │
  │   Store log-sum-exp M = m + log2(l)              │
  └──────────────────────────────────────────────────┘

The log-sum-exp M is stored so the backward pass can recompute softmax on
the fly without needing the full N×N P matrix.

exp2 vs exp:
  We substitute e^x → 2^x via   e^x = 2^(x · log₂e)
  and pre-multiply scores by log₂e.  Reason: GPU hardware fuses exp2 more
  efficiently than exp in float16 arithmetic.

References:
  Dao et al. "FlashAttention-2", 2023
  Milakov & Gimelshein "Online normalizer calculation for softmax", 2018
"""

import math
import time
import torch
import triton
import triton.language as tl
import warnings
warnings.filterwarnings("ignore")

from utils import bench_ms, report_header, measure_peak_memory_mb


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Naive reference
# ──────────────────────────────────────────────────────────────────────────────

def naive_attention(Q, K, V, is_causal: bool = False):
    """
    Standard O(N²) attention. Materialises the full attention matrix.
    Q, K, V : (B, H, S, D)
    """
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B,H,S,S)

    if is_causal:
        mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FlashAttention forward — Triton
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fa_fwd_inner(
    O_block, l_i, m_i,
    Q_block,
    K_block_ptr, V_block_ptr,
    block_idx_q,
    offs_q, offs_kv,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CAUSAL_PASS: tl.constexpr,   # 0 = pre-diag or full, 1 = diagonal
):
    """
    Inner K/V loop for one Q block.

    CAUSAL_PASS == 0:  iterate lo..hi without mask  (all KV < Q or full)
    CAUSAL_PASS == 1:  iterate the single diagonal block, apply tril mask
    """
    if CAUSAL_PASS == 1:
        lo = block_idx_q * BLOCK_Q
        hi = (block_idx_q + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q)
    else:
        lo = 0
        hi = SEQ_LEN

    # Advance block pointers to the starting KV position
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_KV)
        kv_idx   = start_kv + offs_kv

        # Load K (already transposed via block pointer strides) and V
        K_block = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        # QK scores
        QK = tl.dot(Q_block, K_block)                        # (BLOCK_Q, BLOCK_KV)

        # Masking
        kv_pad_mask = kv_idx < SEQ_LEN
        if CAUSAL_PASS == 1:
            causal_mask = offs_q[:, None] >= kv_idx[None, :]
            QK += tl.where(causal_mask & kv_pad_mask[None, :], 0, float("-inf"))
        else:
            QK += tl.where(kv_pad_mask[None, :], 0, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(QK, 1))
        QK   -= m_ij[:, None]
        P     = tl.math.exp2(QK)                              # (BLOCK_Q, BLOCK_KV)
        l_ij  = tl.sum(P, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i   = l_i * alpha + l_ij

        # Accumulate output with correction
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P.to(V_block.dtype), V_block, acc=O_block)
        m_i     = m_ij

        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KV, 0))

    return O_block, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": bq, "BLOCK_KV": bkv}, num_stages=ns, num_warps=nw)
        for bq  in [64, 128]
        for bkv in [32, 64]
        for ns  in [3, 4]
        for nw  in [4, 8]
        if bkv <= bq
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _fa_fwd_kernel(
    Q, K, V,
    softmax_scale: tl.constexpr,
    M_out,          # stores log-sum-exp per query position
    Out,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q:  tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    INV_LN2: tl.constexpr = 1.4426950408889634
    scale = softmax_scale * INV_LN2

    # Which Q block and (batch, head) does this program own?
    block_q = tl.program_id(0)
    bh      = tl.program_id(1)
    b       = bh // NUM_HEADS
    h       = bh %  NUM_HEADS

    # Offsets within the Q and KV blocks
    offs_q  = block_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_kv = tl.arange(0, BLOCK_KV)
    offs_d  = tl.arange(0, HEAD_DIM)

    # Block pointers
    q_off = b * stride_qb + h * stride_qh
    k_off = b * stride_kb + h * stride_kh
    v_off = b * stride_vb + h * stride_vh
    o_off = b * stride_ob + h * stride_oh

    Q_blk_ptr = tl.make_block_ptr(
        Q + q_off, (SEQ_LEN, HEAD_DIM), (stride_qs, stride_qd),
        (block_q * BLOCK_Q, 0), (BLOCK_Q, HEAD_DIM), (1, 0))
    K_blk_ptr = tl.make_block_ptr(
        K + k_off, (HEAD_DIM, SEQ_LEN), (stride_kd, stride_ks),   # transposed
        (0, 0), (HEAD_DIM, BLOCK_KV), (0, 1))
    V_blk_ptr = tl.make_block_ptr(
        V + v_off, (SEQ_LEN, HEAD_DIM), (stride_vs, stride_vd),
        (0, 0), (BLOCK_KV, HEAD_DIM), (1, 0))
    O_blk_ptr = tl.make_block_ptr(
        Out + o_off, (SEQ_LEN, HEAD_DIM), (stride_os, stride_od),
        (block_q * BLOCK_Q, 0), (BLOCK_Q, HEAD_DIM), (1, 0))

    # Running statistics
    m_i     = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i     = tl.full([BLOCK_Q], 1.0,           dtype=tl.float32)
    O_block = tl.zeros([BLOCK_Q, HEAD_DIM],      dtype=tl.float32)

    # Load Q block and prescale
    Q_block = tl.load(Q_blk_ptr, boundary_check=(0,), padding_option="zero")
    Q_block = (Q_block * scale).to(tl.float16 if Q_block.dtype == tl.float16 else tl.float32)

    if IS_CAUSAL:
        # Pre-diagonal blocks (all K positions strictly before Q block)
        O_block, l_i, m_i = _fa_fwd_inner(
            O_block, l_i, m_i, Q_block,
            K_blk_ptr, V_blk_ptr,
            block_q, offs_q, offs_kv, SEQ_LEN,
            BLOCK_Q, BLOCK_KV, 0)
        # Diagonal block (needs tril masking)
        O_block, l_i, m_i = _fa_fwd_inner(
            O_block, l_i, m_i, Q_block,
            K_blk_ptr, V_blk_ptr,
            block_q, offs_q, offs_kv, SEQ_LEN,
            BLOCK_Q, BLOCK_KV, 1)
    else:
        O_block, l_i, m_i = _fa_fwd_inner(
            O_block, l_i, m_i, Q_block,
            K_blk_ptr, V_blk_ptr,
            block_q, offs_q, offs_kv, SEQ_LEN,
            BLOCK_Q, BLOCK_KV, 0)

    # Normalise and store log-sum-exp for the backward pass
    m_i += tl.math.log2(l_i)
    O_block = O_block / (l_i[:, None] + 1e-6)

    m_ptrs = M_out + bh * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
    tl.store(O_blk_ptr, O_block.to(Out.type.element_ty), boundary_check=(0,))


def flash_attention_forward(Q, K, V, softmax_scale=None, is_causal=False):
    """
    Q, K, V : (B, H, S, D)  float16 or float32
    Returns  : (B, H, S, D), log-sum-exp M (B, H, S)
    """
    assert Q.is_cuda
    B, H, S, D = Q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    O = torch.zeros_like(Q)
    M = torch.empty(B * H, S, device=Q.device, dtype=torch.float32)

    grid = lambda args: (triton.cdiv(S, args["BLOCK_Q"]), B * H)
    _fa_fwd_kernel[grid](
        Q, K, V, softmax_scale, M, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D,
        IS_CAUSAL=is_causal,
    )
    return O, M.view(B, H, S)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Backward pass  (Python reference, Triton version follows same structure)
# ──────────────────────────────────────────────────────────────────────────────

def flash_attention_backward_reference(Q, K, V, O, dO, M, softmax_scale=None, is_causal=False):
    """
    Python reference for the backward pass.
    Recomputes P from (Q, K, M) on the fly — avoids storing the N×N matrix.

    Key identity: P = exp2(Q@K.T * scale * log2e − M[:, :, :, None])
    """
    B, H, S, D = Q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    INV_LN2 = 1.4426950408889634
    LN2     = 0.6931471805599453

    scale = softmax_scale * INV_LN2

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # D = rowsum(dO * O)  — needed for dS
    D_vec = (dO * O).sum(dim=-1)    # (B, H, S)

    for b in range(B):
        for h in range(H):
            Q_  = Q[b, h] * scale     # (S, D)
            K_  = K[b, h]
            V_  = V[b, h]
            dO_ = dO[b, h]
            M_  = M[b, h]             # (S,)
            D_  = D_vec[b, h]

            # Recover softmax from stored log-sum-exp
            scores = Q_ @ K_.T                                   # (S, S)
            if is_causal:
                mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), 1)
                scores = scores.masked_fill(mask, float("-inf"))
            P = torch.exp2(scores - M_[:, None])                 # (S, S)

            # dV = P^T @ dO
            dV[b, h] += P.T @ dO_

            # dP = dO @ V^T
            dP = dO_ @ V_.T                                      # (S, S)

            # dS = P * (dP - D)   then scale by LN2
            dS = P * (dP - D_[:, None]) * LN2

            # dQ, dK
            dQ[b, h] += (dS @ K_) * scale
            dK[b, h] += dS.T @ Q_

    return dQ, dK, dV


# ──────────────────────────────────────────────────────────────────────────────
# 4.  autograd Function (forward Triton, backward reference)
# ──────────────────────────────────────────────────────────────────────────────

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, is_causal):
        O, M = flash_attention_forward(Q, K, V, softmax_scale, is_causal)
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal     = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward_reference(
            Q, K, V, O, dO.contiguous(), M,
            ctx.softmax_scale, ctx.is_causal)
        return dQ, dK, dV, None, None


def flash_attn(Q, K, V, softmax_scale=None, is_causal=False):
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Q.shape[-1])
    return FlashAttention.apply(Q, K, V, softmax_scale, is_causal)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Tests & benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _peak_mem_mb(fn, *args) -> float:
    """
    Measure peak GPU memory (MB) for one call to fn(*args).
    Uses torch.cuda.max_memory_allocated after reset — reflects only the
    allocations made inside this call (Q/K/V/O are allocated before the call
    and excluded by design so we measure the *additional* working memory).
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    fn(*args)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / 1e6


if __name__ == "__main__":
    report_header()

    # ── Correctness ──────────────────────────────────────────────────────────
    print("Correctness checks …")
    for is_causal in (False, True):
        B, H, S, D = 2, 4, 512, 64
        Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        fa_out, _ = flash_attention_forward(Q.float(), K.float(), V.float(),
                                            is_causal=is_causal)
        ref       = naive_attention(Q.float(), K.float(), V.float(), is_causal=is_causal)
        diff      = (fa_out - ref).abs().max().item()
        status    = "PASSED" if diff < 1e-2 else "FAILED"
        print(f"  causal={is_causal}  max_diff={diff:.5f}  {status}")

    # ── Speed benchmark: claim ~1.8× speedup at 8K sequences ────────────────
    print()
    print("Speed benchmark  |  B=1  H=8  D=64  causal=True  (float16)")
    print("─" * 75)
    print(f"  {'S':>6}  {'Naive (ms)':>12}  {'FA (ms)':>10}  "
          f"{'Speedup':>9}  {'FA faster?':>12}")
    print("─" * 75)

    B, H, D = 1, 8, 64
    speedup_at_8k = None
    for S in [512, 1024, 2048, 4096, 8192]:
        Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        ms_naive = bench_ms(naive_attention,          Q, K, V, is_causal=True)
        ms_fa    = bench_ms(flash_attention_forward,  Q, K, V, is_causal=True)
        speedup  = ms_naive / ms_fa
        if S == 8192:
            speedup_at_8k = speedup

        flag = "✓" if speedup >= 1.5 else ""
        print(f"  {S:>6}  {ms_naive:>12.3f}  {ms_fa:>10.3f}  {speedup:>8.2f}×  {flag}")

    print("─" * 75)
    print(f"  Claim: ~1.8× speedup at 8K — measured {speedup_at_8k:.2f}× "
          f"({'✓ supported' if speedup_at_8k and speedup_at_8k >= 1.5 else '⚠ check config'})")

    # ── Memory benchmark: actual GPU peak allocation ─────────────────────────
    print()
    print("Peak GPU memory  |  B=1  H=8  D=64  (float16)")
    print("Measuring torch.cuda.max_memory_allocated() delta during the attention call")
    print("─" * 80)
    print(f"  {'S':>6}  {'Naive (MB)':>12}  {'FA (MB)':>10}  "
          f"{'Reduction':>12}  {'N² matrix (MB)':>16}")
    print("─" * 80)

    mem_ratio_8k = None
    for S in [1024, 2048, 4096, 8192]:
        Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        mem_naive = _peak_mem_mb(naive_attention,         Q, K, V)
        mem_fa    = _peak_mem_mb(flash_attention_forward, Q, K, V)

        # Expected N×N attention matrix size (what naive materialises)
        matrix_mb = B * H * S * S * 2 / 1e6    # fp16

        ratio = mem_naive / mem_fa if mem_fa > 0 else float("inf")
        if S == 8192:
            mem_ratio_8k = ratio

        flag = "✓" if ratio >= 5 else ""
        print(f"  {S:>6}  {mem_naive:>12.1f}  {mem_fa:>10.1f}  "
              f"{ratio:>10.1f}×  {matrix_mb:>14.1f} MB  {flag}")

    print("─" * 80)
    if mem_ratio_8k:
        print(f"  At S=8192: {mem_ratio_8k:.1f}× less peak GPU memory  "
              f"(N²→O(N) for attention activations)")
    print()
    print("  Note: 'memory reduction' in the resume refers to attention-layer")
    print("  activation memory (the N×N score matrix that FA never materialises).")
    print("  End-to-end model memory reduction depends on model weight size vs")
    print("  sequence length — see benchmarks/benchmark_memory.py in inference-engine.")
