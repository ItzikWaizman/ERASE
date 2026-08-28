"""
Build a polished, paper-ready figure for the activation-projection geometric
validation. Loads the JSON metrics produced by run_activation_projection.py
and emits a single 2-panel figure plus a focused 3-layer cumulative curve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math

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
        default="saves/diagnostics/geometric/activation_projection/activation_metrics.json",
    )
    p.add_argument(
        "--output_dir",
        default="saves/diagnostics/geometric/activation_projection",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.metrics_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cfg = data["config"]
    metrics = data["metrics"]
    alpha = cfg["alpha"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 220,
    })

    layers = [m["layer"] for m in metrics]
    is_filt = [m["is_filtered"] for m in metrics]
    rr = [m["norm_red_retain"] for m in metrics]
    rd = [m["norm_red_delta"] for m in metrics]
    rdn = [m["norm_red_delta_null"] for m in metrics]
    # Bootstrap means and 95% CIs for D and D_null (panel B)
    D = [m["boot_D_signal_mean"] for m in metrics]
    D_null = [m["boot_D_null_mean"] for m in metrics]
    D_lo = [m["boot_D_signal_ci_lo"] for m in metrics]
    D_hi = [m["boot_D_signal_ci_hi"] for m in metrics]
    Dn_lo = [m["boot_D_null_ci_lo"] for m in metrics]
    Dn_hi = [m["boot_D_null_ci_hi"] for m in metrics]
    n = len(metrics)

    # === Figure 1: 2-panel overview =========================================
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel A: norm preservation (retain bulk vs delta signal vs delta null)
    ax = axes[0]
    x = list(range(n))
    w = 0.27
    ax.bar([xi - w for xi in x], rr, width=w, color="#666666",
           label=r"retain bulk $\|P X_r\|/\|X_r\|$")
    ax.bar(x, rdn, width=w, color="#bdbdbd", edgecolor="#666666",
           label=r"null $\|P\delta_\mathrm{null}\|/\|\delta_\mathrm{null}\|$")
    ax.bar([xi + w for xi in x], rd, width=w, color="#1f77b4",
           label=r"signal $\|P\delta\|/\|\delta\|,\;\delta=\mu_f-\mu_r$")
    for i, f in enumerate(is_filt):
        if f:
            ax.axvspan(i - 0.5, i + 0.5, color="#fff5cc", alpha=0.55, zorder=0)
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel(r"norm preservation under $P$")
    ax.set_ylim(0, 1.08)
    ax.set_title(r"(a) $P=(I+\alpha\,C_\text{retain})^{-1}$ preserves the "
                 r"signal more than the bulk")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: D_signal vs D_null (selectivity ratio) with bootstrap 95% CIs.
    # Layer 1 has a near-rank-one spectrum and a huge bootstrap CI; we cap its
    # plotted height/CI at the y-axis ceiling and mark it as off-scale.
    ax = axes[1]
    w = 0.4
    y_cap = 4.0
    D_plot = [min(d, y_cap) for d in D]
    Dn_plot = [min(d, y_cap) for d in D_null]
    D_hi_plot = [min(h, y_cap) for h in D_hi]
    Dn_hi_plot = [min(h, y_cap) for h in Dn_hi]
    Dn_err_lo = [max(Dn_plot[i] - Dn_lo[i], 0.0) for i in range(n)]
    Dn_err_hi = [max(Dn_hi_plot[i] - Dn_plot[i], 0.0) for i in range(n)]
    D_err_lo = [max(D_plot[i] - D_lo[i], 0.0) for i in range(n)]
    D_err_hi = [max(D_hi_plot[i] - D_plot[i], 0.0) for i in range(n)]
    ax.bar([xi - w / 2 for xi in x], Dn_plot, width=w, color="#bdbdbd",
           edgecolor="#666666",
           yerr=[Dn_err_lo, Dn_err_hi],
           ecolor="#444444",
           error_kw={"elinewidth": 0.9, "capsize": 2.5},
           label=r"$D_\mathrm{null}^\ell$ (bootstrap mean$\,\pm\,95\%$ CI)")
    ax.bar([xi + w / 2 for xi in x], D_plot, width=w, color="#d62728",
           yerr=[D_err_lo, D_err_hi],
           ecolor="#7a1414",
           error_kw={"elinewidth": 0.9, "capsize": 2.5},
           label=r"$D^\ell$ (bootstrap mean$\,\pm\,95\%$ CI)")
    # mark off-scale layers (L1)
    for i in range(n):
        if D[i] > y_cap or D_hi[i] > y_cap or D_null[i] > y_cap or Dn_hi[i] > y_cap:
            ax.text(i, y_cap * 0.98, r"$\uparrow$", color="black",
                    ha="center", va="top", fontsize=11, fontweight="bold")
    for i, f in enumerate(is_filt):
        if f:
            ax.axvspan(i - 0.5, i + 0.5, color="#fff5cc", alpha=0.55, zorder=0)
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel(r"$D^\ell = \|P\delta\|/\|\delta\|\ \ /\ \ \|PX_r\|/\|X_r\|$")
    ax.set_title("(b) Differential preservation, bootstrap 95% intervals")
    ax.set_ylim(0, y_cap)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=7.0)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        r"Geometric validation: spectral filter $P$ preserves the forget"
        r"$\,\!$-vs-retain differential signal selectively  "
        rf"($\alpha={alpha}$, yellow stripe = filtered layer)",
        y=1.02, fontsize=10.5,
    )
    fig.tight_layout()
    out = os.path.join(args.output_dir, "FIG_geometric_validation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)
    out_pdf = os.path.join(args.output_dir, "FIG_geometric_validation.pdf")
    fig2 = plt.figure(figsize=(8.5, 3.4))
    plt.close(fig2)  # no-op; we already saved png. Now also produce pdf.
    # Re-render to pdf
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    ax = axes[0]
    ax.bar([xi - w / 2 for xi in x], rr, width=w * 0.65, color="#666666",
           label=r"retain bulk")  # smaller widths for pdf
    # (For brevity we just save the png as canonical; pdf optional.)
    plt.close(fig)

    # === Figure 2: cumulative energy on three illustrative layers ==========
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8), sharey=True)

    def get_metric(layer_id):
        return next(m for m in metrics if m["layer"] == layer_id)

    # L0: filtered, strong P effect, clean asymmetry.
    # L8: control, weak P effect, no asymmetry (null line overlaps retain).
    # L15: control with strong P effect; same mechanism would emerge.
    show = [(0, "L0 (filtered, $\\lambda_\\mathrm{max}=5.5$)"),
            (8, "L8 (control, $\\lambda_\\mathrm{max}=1.8$)"),
            (15, "L15 (control, $\\lambda_\\mathrm{max}=447$)")]
    for ax, (layer_id, title) in zip(axes, show):
        m = get_metric(layer_id)
        ax.plot(m["cum_grid"], m["cum_energy_retain"], color="#666666",
                lw=1.6, label=r"retain $X_r$")
        ax.plot(m["cum_grid"], m["cum_energy_forget"], color="#1f77b4",
                lw=1.6, label=r"forget $X_f$")
        ax.plot(m["cum_grid"], m["cum_energy_delta_null"], color="#bdbdbd",
                lw=1.4, ls=":", label=r"$\delta_\mathrm{null}$ (random retain split)")
        ax.plot(m["cum_grid"], m["cum_energy_delta"], color="#d62728",
                lw=1.7, ls="--",
                label=r"$\delta=\mu_f-\mu_r$ (forget-vs-retain)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"top-$K$ eigenvectors of $C_\text{retain}$")
        if ax is axes[0]:
            ax.set_ylabel("cumulative energy fraction")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(loc="center right", fontsize=7.6, framealpha=0.95)
    fig.suptitle(
        r"Cumulative energy in $C_\text{retain}$ eigenbasis: "
        r"the forget-vs-retain differential $\delta$ lives in the tail "
        r"only where P contracts ($\lambda_\mathrm{max}$ large)",
        y=1.02, fontsize=10.5,
    )
    fig.tight_layout()
    out = os.path.join(args.output_dir, "FIG_cumulative_energy.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()
