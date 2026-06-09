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

## Results

| Kernel | Metric | Measured | GPU | Date |
|--------|--------|----------|-----|------|
| Fused softmax | % peak HBM | | | |
| RMSNorm | % peak HBM | | | |
| RoPE | % peak HBM | | | |
| GEMM 4096³ | % of cuBLAS | | | |
| FlashAttention | speedup @ S=8192 | | | |
| FlashAttention | memory ratio @ S=8192 | | | |

Charts: `benchmarks/results/figures/` (after `plot_results.py`)

---

## Future scope

- Triton backward for FlashAttention
- Fused attention + RMSNorm epilogue
- Nsight Compute profiling per kernel
- INT8 / FP8 quantized GEMM
