"""
Build paper-grade figures for the interference-reduction experiment.

Loads metrics.json + per_component_layer{L}.npz produced by
run_interference_reduction.py and writes:

- FIG_interference_reduction.png  (headline bar plot of 1 - R^l per layer)
- FIG_per_component.png           (top-50 lambda_k * M_kk before/after P, log y)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics_json",
        default="saves/diagnostics/geometric/interference_reduction/metrics.json",
    )
    p.add_argument(
        "--per_component_npz",
        default="saves/diagnostics/geometric/interference_reduction/per_component_layer0.npz",
    )
    p.add_argument(
        "--output_dir",
        default="saves/diagnostics/geometric/interference_reduction",
    )
    p.add_argument(
        "--filtered_only",
        action="store_true",
        default=True,
        help="Plot only layers actually filtered in production. The metric "
        "is a per-layer self-comparison vs. P=I (R=1), so unfiltered "
        "layers do not contribute to the question being asked.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.metrics_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cfg = data["config"]
    metrics = sorted(data["metrics"], key=lambda m: m["layer"])
    if args.filtered_only:
        metrics = [m for m in metrics if m["is_filtered"]]
    alpha = cfg["alpha"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # =======================================================================
    # Figure 1: interference-reduction headline bar plot
    # =======================================================================
    layers = [m["layer"] for m in metrics]
    R = np.asarray([m["R"] for m in metrics])
    is_filt = np.asarray([m["is_filtered"] for m in metrics])
    pct_reduction = (1.0 - R) * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    color_filt = "#2b6cb0"      # blue
    x = np.arange(len(layers))
    ax.bar(x, pct_reduction, color=color_filt, width=0.65,
           edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{ell}" for ell in layers])
    ax.set_ylabel(r"Interference reduction $(1-R^{\ell})\,\times\,100\%$")
    ax.set_xlabel(r"Transformer layer  $\ell$")
    ax.set_title(
        rf"Per-layer self-comparison: $P^{{\ell}}=(I+\alpha C_{{\mathrm{{retain}}}}^{{\ell}})^{{-1}}$,"
        rf" $\alpha={alpha:g}$"
    )
    ax.axhline(100, color="black", linewidth=0.5, linestyle=":")
    # Reference line for the no-filter baseline (P = I  =>  R = 1, reduction = 0).
    ax.axhline(0, color="#4a5568", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(
        len(layers) - 0.5, 1.5,
        r"baseline: no filter ($P{=}I$, $R^{\ell}{=}1$)",
        ha="right", va="bottom", fontsize=8, color="#4a5568",
    )

    # Annotate exact R above each bar (with appropriate precision).
    for xi, ri, pi in zip(x, R, pct_reduction):
        label = f"R={ri:.3f}" if ri >= 0.001 else r"$R\!<\!10^{-3}$"
        ax.text(
            xi, pi + 1.5,
            label, ha="center", va="bottom", fontsize=8.5,
            color="#1a202c"
        )

    fig.subplots_adjust(bottom=0.14, top=0.88, left=0.11, right=0.97)
    out1 = os.path.join(args.output_dir, "FIG_interference_reduction.png")
    fig.savefig(out1, bbox_inches="tight")
    print(f"Wrote {out1}", flush=True)
    plt.close(fig)

    # =======================================================================
    # Figure 2: per-component (lambda_k * M_kk) before vs. after, log y
    # =======================================================================
    pc_arr = np.load(args.per_component_npz)
    eigvals = pc_arr["eigvals"]
    M = pc_arr["M"]
    w_off = pc_arr["weights_off"]
    w_on = pc_arr["weights_on"]
    pc_layer = cfg.get("per_component_layer", 0)
    K = len(eigvals)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    width = 0.42
    xs = np.arange(K)
    eps = 1e-12
    ax.bar(xs - width/2, np.maximum(w_off, eps),
           width, color="#cbd5e0", label=r"before P:  $\lambda_k\,M_{kk}$",
           edgecolor="white", linewidth=0.4)
    ax.bar(xs + width/2, np.maximum(w_on, eps),
           width, color="#2b6cb0", label=r"after P:   $\lambda_k\,M_{kk}\,(1+\alpha\lambda_k)^{-2}$",
           edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel(r"Eigen-component index $k$ of $C_{\mathrm{retain}}^{\ell}$ (sorted by $\lambda_k$)")
    ax.set_ylabel(r"Per-component contribution to $\mathrm{trace}(\cdot)$  (log)")
    R_l = next(m["R"] for m in metrics if m["layer"] == pc_layer)
    ax.set_title(
        rf"Layer L{pc_layer}: top-{K} eigendirections.   $\sum$ ratio = $R^{{{pc_layer}}}={R_l:.3g}$"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-1, K)

    fig.subplots_adjust(bottom=0.16, top=0.90, left=0.11, right=0.97)
    out2 = os.path.join(args.output_dir, "FIG_per_component.png")
    fig.savefig(out2, bbox_inches="tight")
    print(f"Wrote {out2}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
