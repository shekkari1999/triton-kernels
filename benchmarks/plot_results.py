"""
Plot benchmark JSON → benchmarks/results/figures/*.png

Usage:
  python benchmarks/plot_results.py
  python benchmarks/plot_results.py benchmarks/results/latest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "benchmarks" / "results" / "figures"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"No results at {path}. Run: python benchmarks/run_all.py")
    return json.loads(path.read_text())


def plot_bandwidth_bars(data: dict, out_dir: Path):
    kernels = data["kernels"]
    labels = []
    pcts = []
    for key, label in [
        ("vector_add", "Vector add"),
        ("fused_softmax", "Fused softmax"),
        ("rmsnorm", "RMSNorm"),
        ("rope", "RoPE"),
    ]:
        if key in kernels:
            labels.append(label)
            pcts.append(kernels[key]["hbm_pct"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, pcts, color="#4C72B0")
    ax.axhline(75, color="#C44E52", linestyle="--", linewidth=1, label="75% target")
    ax.set_ylabel("% of peak HBM bandwidth")
    ax.set_title(f"Memory-bound kernels — {data['metadata']['gpu']}")
    ax.set_ylim(0, max(max(pcts) * 1.15, 80))
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "hbm_bandwidth.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_flash_attention(data: dict, out_dir: Path):
    fa = data["kernels"].get("flash_attention", {})
    sweep = fa.get("sweep", [])
    if not sweep:
        return

    seqs = [r["seq_len"] for r in sweep]
    speedups = [r["speedup"] for r in sweep]
    mem_ratios = [r["mem_ratio"] for r in sweep if r["mem_ratio"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(seqs, speedups, "o-", color="#55A868", linewidth=2, markersize=6)
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Speedup (naive / FA)")
    ax1.set_title("FlashAttention latency")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    if mem_ratios:
        ax2.plot(seqs[: len(mem_ratios)], mem_ratios, "s-", color="#C44E52", linewidth=2, markersize=6)
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Peak memory ratio (naive / FA)")
    ax2.set_title("FlashAttention memory")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"FlashAttention-2 forward — {data['metadata']['gpu']}", fontsize=11)
    fig.tight_layout()
    path = out_dir / "flash_attention.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_gemm(data: dict, out_dir: Path):
    g = data["kernels"].get("gemm")
    if not g:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["cuBLAS", "Triton"]
    tflops = [g["cublas_tflops"], g["triton_tflops"]]
    ax.bar(labels, tflops, color=["#8172B2", "#4C72B0"])
    ax.set_ylabel("TFLOP/s")
    ax.set_title(f"GEMM {g['size']}³ — {data['metadata']['gpu']}")
    for i, v in enumerate(tflops):
        ax.text(i, v + max(tflops) * 0.02, f"{v:.0f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = out_dir / "gemm_tflops.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmarks/results/latest.json"
    if not json_path.is_absolute():
        json_path = ROOT / json_path

    data = _load(json_path)
    out_dir = FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_bandwidth_bars(data, out_dir)
    plot_flash_attention(data, out_dir)
    plot_gemm(data, out_dir)


if __name__ == "__main__":
    main()
