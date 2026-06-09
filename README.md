# Triton Kernels for LLM Inference

Triton implementations for transformer inference workloads: vector add, tiled GEMM, fused softmax, RMSNorm, RoPE, and FlashAttention-2 forward.

One file per kernel. Each includes a PyTorch baseline, Triton kernel, tests, and benchmarks. Bandwidth-bound ops report % of peak HBM; GEMM reports TFLOP/s vs cuBLAS.

```
01_vector_add.py       PID mapping, masking, autotune
02_matmul.py           Tiled GEMM, grouped program IDs
03_fused_softmax.py    Fused softmax (5 PyTorch ops -> 1 kernel)
04_online_softmax.py   Two-pass online softmax
05_layer_norm.py       Fused RMSNorm
06_rope_embeddings.py  Fused RoPE
07_flash_attention.py  FlashAttention-2 forward (backward: Python reference)
```

```bash
pip install -r requirements.txt
python benchmarks/run_all.py
python benchmarks/plot_results.py
```

---

## How benchmarks are measured

All numbers come from `python benchmarks/run_all.py`. Implementation is in `utils.py` and `benchmarks/run_all.py`.

### Timing

- `triton.testing.do_bench`: 25 warmup iterations, 100 timed runs, **median** latency in ms.
- CUDA sync before/after each timed region.
- Triton kernels are compared against a PyTorch baseline in the same file (e.g. 5-kernel naive softmax vs fused Triton, `torch.mm` vs tiled GEMM).

### % peak HBM (vector add, softmax, RMSNorm, RoPE)

```
achieved_GB/s = bytes_moved / time_s / 1e9
% peak HBM    = achieved_GB/s / peak_HBM * 100
```

`peak_HBM` is looked up from GPU spec tables in `utils.py` (H100: 3350 GB/s). Bytes moved:

| Kernel | Bytes counted |
|--------|----------------|
| Vector add | 3N (read `a`, read `b`, write `out`) |
| Softmax | 2 × numel (read input, write output) |
| RMSNorm | 2MN + N (read `x`, write `out`, read weight `w`) |
| RoPE | 2 × numel (read `x`, write `out`) |

RoPE also runs `cos`/`sin` on the fly, so it is more compute-bound than softmax/RMSNorm. A lower % peak HBM there does not mean the kernel is wrong.

### GEMM TFLOP/s

- Problem: 4096³ fp16 matmul (`C = A @ B`).
- FLOPs: `2 × 4096³` (one multiply-add = 2 FLOPs).
- Baseline: `torch.mm` (cuBLAS).
- Reported: achieved TFLOP/s and Triton throughput as % of cuBLAS.

### FlashAttention speedup

- Config: `B=1`, `H=8`, `D=64`, fp16, **causal=True`.
- Sweep `S` ∈ {512, 1024, 2048, 4096, 8192, 16384}.
- **Naive**: `Q @ K^T` → softmax → `@ V`, materializing the full `S × S` score matrix.
- **Triton FA**: tiled forward kernel in `07_flash_attention.py`, no `S × S` matrix in HBM.
- **Speedup** = `naive_ms / fa_ms` from `do_bench`.

### FlashAttention memory

- `Q`, `K`, `V` are allocated **before** the timed region.
- Reset `torch.cuda.max_memory_allocated()`, run one forward pass, take the delta from current `memory_allocated()`.
- This measures **extra** peak memory during the forward pass, not model weights.
- **Mem ratio** = `naive_delta_MB / fa_delta_MB`.

At large `S`, naive attention allocates the `S × S` attention matrix; FA stays near O(S) for activations, so the ratio grows with sequence length.

---

## Results

H100 80GB HBM3, June 2026. Full sweep in `benchmarks/results/latest.json`.

| Kernel | Metric | Measured | GPU | Date |
|--------|--------|----------|-----|------|
| Fused softmax | % peak HBM | 79.4% | H100 80GB | 2026-06 |
| RMSNorm | % peak HBM | 79.9% | H100 80GB | 2026-06 |
| RoPE | % peak HBM | 32.9% | H100 80GB | 2026-06 |
| GEMM 4096³ | % of cuBLAS | 85% | H100 80GB | 2026-06 |
| FlashAttention | speedup @ S=8192 | 16.9× | H100 80GB | 2026-06 |
| FlashAttention | memory ratio @ S=8192 | 257× | H100 80GB | 2026-06 |

| S | Naive (ms) | FA (ms) | Speedup | Mem ratio |
|---|------------|---------|---------|-----------|
| 512 | 0.043 | 0.015 | 2.9× | 17× |
| 1024 | 0.091 | 0.021 | 4.4× | 33× |
| 2048 | 0.381 | 0.039 | 9.8× | 65× |
| 4096 | 1.523 | 0.108 | 14.1× | 129× |
| 8192 | 6.465 | 0.382 | 16.9× | 257× |
| 16384 | 20.868 | 1.521 | 13.7× | 513× |

![HBM bandwidth](benchmarks/results/figures/hbm_bandwidth.png)
![FlashAttention](benchmarks/results/figures/flash_attention.png)
![GEMM](benchmarks/results/figures/gemm_tflops.png)

---

## Future scope

- Nsight Compute write-up for fused softmax or FlashAttention
- Improve RoPE kernel bandwidth
- Triton backward for FlashAttention (optional)
