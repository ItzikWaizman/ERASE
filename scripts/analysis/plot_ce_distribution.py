"""Plot the per-sample CE (avg_loss) distribution from a TOFU_EVAL.json file.

Usage:
    python scripts/analysis/plot_ce_distribution.py <path_to_TOFU_EVAL.json>

Produces a histogram saved as `ce_distribution.png` alongside the JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_ce_values(path: Path) -> list[float]:
    with open(path) as f:
        data = json.load(f)
    values_by_idx = data["forget_Q_A_Prob"]["value_by_index"]
    return [float(v["avg_loss"]) for v in values_by_idx.values()]


def plot_distribution(ce_values: list[float], out_path: Path, title: str = ""):
    arr = np.array(ce_values)
    mean, std, median = arr.mean(), arr.std(), np.median(arr)
    in_band = np.sum((arr >= 2.0) & (arr <= 4.0))
    under = np.sum(arr < 2.0)
    over = np.sum(arr > 4.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(15, arr.max() + 1), 60)
    ax.hist(arr, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")

    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean={mean:.2f}")
    ax.axvline(median, color="orange", linestyle="-.", linewidth=2, label=f"Median={median:.2f}")

    ax.axvspan(2.0, 4.0, alpha=0.1, color="green", label="Target band [2, 4]")

    ax.set_xlabel("Per-sample CE (avg_loss)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or f"CE Distribution (n={len(arr)})", fontsize=13)

    stats_text = (
        f"Mean: {mean:.3f}\n"
        f"Std:  {std:.3f}\n"
        f"Median: {median:.3f}\n"
        f"In band [2,4]: {in_band}/{len(arr)}\n"
        f"Under (<2): {under}\n"
        f"Over (>4): {over}"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontfamily="monospace",
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  n={len(arr)}, mean={mean:.3f}, std={std:.3f}, median={median:.3f}")
    print(f"  In band [2,4]: {in_band}, Under: {under}, Over: {over}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <TOFU_EVAL.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    ce_values = load_ce_values(path)
    out_path = path.parent / "ce_distribution.png"
    title = path.parent.parent.name if path.parent.name == "evals" else path.parent.name
    plot_distribution(ce_values, out_path, title=title)


if __name__ == "__main__":
    main()
