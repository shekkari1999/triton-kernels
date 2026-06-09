"""
Kernel 2: Tiled Matrix Multiplication (GEMM)  C = A @ B

Matrix multiplication is the backbone of every LLM — every linear layer,
every QK^T, every projection goes through GEMM.

Naive approach:
  c[i,j] = sum_k a[i,k] * b[k,j]
  Loads every row of A and every column of B for each output element → O(N^3) HBM reads.

Tiled approach:
  Split output C into BLOCK_M × BLOCK_N tiles.
  Each tile accumulates over K in BLOCK_K steps, keeping data in SRAM (registers)
  for the accumulation → drastically reduces HBM traffic.

  Memory layout visualized for C = A@B  [M x K] @ [K x N] = [M x N]:

  ┌──────────┬──────────┐        ┌────────────────────────┐
  │  BLOCK_M │          │        │  tile of A  (M×K block)│
  │  ×       │          │        │  tile of B  (K×N block)│
  │  BLOCK_N │  ...     │  ←     │  accumulator in SRAM   │
  │          │          │        └────────────────────────┘
  └──────────┴──────────┘
      loop over K dimension in BLOCK_K steps

Grouped ordering (L2 cache trick):
  Default row-major PID ordering causes poor L2 reuse across tiles.
  Grouped ordering packs GROUP_SIZE rows of tiles together so adjacent
  PIDs share more of the A rows in L2.
"""

import time
import torch
import triton
import triton.language as tl

from utils import bench_ms, report_flops, report_header


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PyTorch reference
# ──────────────────────────────────────────────────────────────────────────────

def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A, B)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Pseudocode — understand tiling before Triton
# ──────────────────────────────────────────────────────────────────────────────

def matmul_pseudocode(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Tiled matmul in plain Python.  Each 'pid' owns one (BLOCK_M × BLOCK_N) tile of C.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    C = torch.zeros(M, N, dtype=A.dtype)

    for pid_m in range(-(-M // BLOCK_M)):          # ceil div
        for pid_n in range(-(-N // BLOCK_N)):
            # Accumulate this tile over the K dimension
            acc = torch.zeros(BLOCK_M, BLOCK_N)
            for pid_k in range(-(-K // BLOCK_K)):
                a_tile = A[pid_m*BLOCK_M : (pid_m+1)*BLOCK_M,
                           pid_k*BLOCK_K : (pid_k+1)*BLOCK_K]
                b_tile = B[pid_k*BLOCK_K : (pid_k+1)*BLOCK_K,
                           pid_n*BLOCK_N : (pid_n+1)*BLOCK_N]
                acc[:a_tile.shape[0], :b_tile.shape[1]] += a_tile @ b_tile

            C[pid_m*BLOCK_M : (pid_m+1)*BLOCK_M,
              pid_n*BLOCK_N : (pid_n+1)*BLOCK_N] = acc[:min(BLOCK_M, M - pid_m*BLOCK_M),
                                                        :min(BLOCK_N, N - pid_n*BLOCK_N)]
    return C


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Triton kernel — autotuned tiled GEMM with grouped ordering
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE": gs},
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [32, 64]
        for gs in [4, 8]
        for ns in [3, 4]
        for nw in [4, 8]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    Each program computes one (BLOCK_M × BLOCK_N) tile of C.

    Grouped ordering: rather than mapping pid → (row, col) in row-major order,
    we group GROUP_SIZE rows of tiles together.  Adjacent PIDs within a group
    share the same A rows → better L2 reuse.

    PID layout (GROUP_SIZE=3, 2 tile-columns):
      pid 0 → (0,0)   pid 1 → (0,1)
      pid 2 → (1,0)   pid 3 → (1,1)
      pid 4 → (2,0)   pid 5 → (2,1)
      (group boundary)
      pid 6 → (3,0)   …
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id        = pid // num_pid_in_group
    first_pid_m     = group_id * GROUP_SIZE
    group_size_m    = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row / col offsets for this tile
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initial pointers to the first K-slice of A and B
    A_tile = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_tile = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop
    for k in range(tl.cdiv(K, BLOCK_K)):
        # boundary_check not used here: instead we wrap indices modulo shape,
        # which works cleanly when padding is not needed.
        a = tl.load(A_tile, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(B_tile, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc=acc)

        # Advance pointers by BLOCK_K along the K dimension
        A_tile += BLOCK_K * stride_ak
        B_tile += BLOCK_K * stride_bk

    # Write tile back to C
    c = acc.to(C_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_tile = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_tile, c, mask=mask)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda args: (
        triton.cdiv(M, args["BLOCK_M"]) * triton.cdiv(N, args["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _time_ms(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    # ── Correctness ──────────────────────────────────────────────
    M, K, N = 512, 512, 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref  = matmul_torch(A, B)
    ours = matmul(A, B)
    assert torch.allclose(ref, ours, atol=1e-2, rtol=1e-2), \
        f"Max diff: {(ref - ours).abs().max().item():.4f}"
    print("Correctness: PASSED")

    # ── TFLOP/s benchmark (claim: benchmarked against cuBLAS) ────
    # Claim: tiled Triton GEMM reaches a significant fraction of peak fp16 TFLOP/s
    _, peak_tf = report_header()[1], None
    _, peak_tf = report_header()

    print("─" * 72)
    print(f"  {'Kernel':<38} {'Time':>9}   {'TFLOP/s':>10}   {'% Peak':>8}")
    print("─" * 72)
    for size in [1024, 2048, 4096]:
        A = torch.randn(size, size, device="cuda", dtype=torch.float16)
        B = torch.randn(size, size, device="cuda", dtype=torch.float16)
        flops = 2 * size ** 3   # multiply-add = 2 FLOPs

        ms_cublas  = bench_ms(lambda: torch.mm(A, B))
        ms_triton  = bench_ms(matmul, A, B)

        report_flops(f"cuBLAS  (torch.mm)  {size}^3", ms_cublas, flops, peak_tf)
        report_flops(f"Triton  (grouped)   {size}^3", ms_triton, flops, peak_tf)
        eff = ms_cublas / ms_triton * 100
        print(f"    → Triton achieves {eff:.0f}% of cuBLAS throughput\n")
    print("─" * 72)
