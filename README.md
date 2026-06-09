# Triton Kernels for LLM Inference

Educational Triton implementations from vector add → tiled GEMM → fused transformer ops → **FlashAttention-2 forward**.

One idea per file: pseudocode first, then a Triton kernel, correctness checks, and benchmarks with the right roofline metric (memory-bound vs compute-bound).

Related serving work (paged KV, batching): [mini-vllm](https://github.com/shekkari1999/mini-vllm).

---

## Mental model

| Bound | Ask | Metric |
|-------|-----|--------|
| **Memory** | How many bytes touch HBM? | GB/s, **% of peak HBM** |
| **Compute** | How many FLOPs per byte? | TFLOP/s, **% of peak fp16** |

- **Fusion** (softmax, RMSNorm, RoPE): fewer HBM round-trips → higher bandwidth %.
- **Tiling** (GEMM, FlashAttention): keep hot data in SRAM → higher TFLOP/s or lower memory.

---

## Curriculum

| File | Teaches | LLM use |
|------|---------|---------|
| `01_vector_add.py` | PID mapping, masking, `@triton.autotune` | Baseline GPU kernel |
| `02_matmul.py` | Tiled GEMM, `tl.dot`, grouped program IDs | Every linear layer |
| `03_fused_softmax.py` | Kernel fusion (5 launches → 1) | Attention softmax |
| `04_online_softmax.py` | Streaming max/sum (two-pass) | Bridge to FlashAttention |
| `05_layer_norm.py` | Fused RMSNorm + backward | Llama / Qwen blocks |
| `06_rope_embeddings.py` | Fused RoPE | Position encoding |
| `07_flash_attention.py` | Tiled FA-2 forward, online softmax, causal mask | Long-context attention |

FlashAttention backward is a **Python reference** in `07_flash_attention.py` (forward is Triton).

---

## Repo layout

```
triton-kernels/
  01_vector_add.py … 07_flash_attention.py   # one kernel per file
  utils.py                                   # timing, % peak HBM / TFLOP/s
  benchmarks/
    run_all.py                               # run everything → JSON
    plot_results.py                          # JSON → PNG charts
    results/
      latest.json                            # after GPU run
      figures/                               # after plot_results.py
  requirements.txt
```

---

## GPU to rent

For resume-grade numbers that top labs recognize, rent:

### First choice: **NVIDIA H100 80GB (SXM or PCIe)**

- Best tensor-core throughput; your `utils.py` table already has H100 peaks.
- Comfortably runs FA sweep through **S=16K** and leaves headroom.
- Providers: **Lambda Labs**, **CoreWeave**, **RunPod**, **Vast.ai** (filter: H100 80GB).

### Solid alternative: **NVIDIA A100 80GB**

- Industry-standard baseline; interviewers know these numbers cold.
- Same providers; often easier to book than H100.

### Minimum: **A100 40GB**

- Enough for the full suite through **S=8192** (headline resume point).
- Skip S=16384 if OOM.

**Avoid for final numbers:** consumer GPUs (4090, etc.) — fine for development, weaker signal on a systems resume.

### Suggested rent session (~2–3 hours)

1. Spin up instance (Ubuntu 22.04, CUDA 12.x).
2. Install deps, run benchmarks, run plots, commit `latest.json` + figures.
3. Tear down.

---

## Setup (on the rented GPU)

```bash
git clone https://github.com/shekkari1999/triton-kernels.git && cd triton-kernels

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# sanity: CUDA visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

PyTorch + Triton wheels must match the instance CUDA driver. If `pip install torch` fails, use [pytorch.org](https://pytorch.org/get-started/locally/) cu124 wheel for CUDA 12.4+.

---

## Run benchmarks

### All kernels (one command)

```bash
python benchmarks/run_all.py
```

Writes `benchmarks/results/latest.json` and prints a **resume line** at the end.

### Plots for README / portfolio

```bash
python benchmarks/plot_results.py
```

Creates:

- `benchmarks/results/figures/hbm_bandwidth.png`
- `benchmarks/results/figures/flash_attention.png`
- `benchmarks/results/figures/gemm_tflops.png`

### Single kernel (while learning)

```bash
python 01_vector_add.py
python 07_flash_attention.py
```

### Correctness tests

```bash
pytest 03_fused_softmax.py 04_online_softmax.py 05_layer_norm.py 06_rope_embeddings.py -q
```

---

## What we always benchmark

| Kernel type | Baseline | Triton | Always report |
|-------------|----------|--------|----------------|
| Bandwidth (01, 03–06) | PyTorch or naive multi-kernel | Fused kernel | ms, GB/s, **% peak HBM** |
| GEMM (02) | `torch.mm` (cuBLAS) | Tiled matmul | TFLOP/s, **% of cuBLAS** |
| FlashAttention (07) | Naive `QK^T` + softmax | FA forward | speedup vs S, **peak memory ratio** |

Targets (guidelines, not guarantees on every GPU):

- Fused bandwidth ops: **≥75%** peak HBM
- GEMM: **≥40%** peak fp16 and **≥50%** of cuBLAS
- FA @ S=8192 causal: **≥1.5×** speedup, **≥2×** memory reduction vs naive

Paste measured values into the table below after your run.

---

## Results

*Fill this in after `python benchmarks/run_all.py` on H100/A100.*

| Kernel | Metric | Measured | GPU | Date |
|--------|--------|----------|-----|------|
| Fused softmax | % peak HBM | — | — | — |
| RMSNorm | % peak HBM | — | — | — |
| RoPE | % peak HBM | — | — | — |
| GEMM 4096³ | % of cuBLAS | — | — | — |
| FlashAttention | speedup @ S=8192 | — | — | — |
| FlashAttention | memory ratio @ S=8192 | — | — | — |

Charts: `benchmarks/results/figures/`

---

## Resume (template)

> **Triton Kernels for LLM Inference** — Progressive Triton suite (fused softmax/RMSNorm/RoPE, tiled GEMM, FlashAttention-2 forward). FA **{X}×** faster than naive attention at S=8192 with **{Y}×** lower peak memory; fused ops reach **{Z}%** of peak HBM on **{GPU}**.  
> GitHub: `github.com/shekkari1999/triton-kernels`

---

## References

- Milakov & Gimelshein, [Online softmax](https://arxiv.org/abs/1805.02867) (2018)
- [Triton documentation](https://triton-lang.org/)
