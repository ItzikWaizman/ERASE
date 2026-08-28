"""
Plot training curves (loss, entropy) from trainer_state.json for ERASE runs.

Usage:
    python scripts/analysis/plot_training_curves.py                       # all runs in saves/unlearn/
    python scripts/analysis/plot_training_curves.py --filter "COV_LOSS"   # only matching dirs
    python scripts/analysis/plot_training_curves.py --dir saves/unlearn/MyRun  # single run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_trainer_state(task_dir: Path) -> list[dict] | None:
    ts = task_dir / "trainer_state.json"
    if not ts.is_file():
        return None
    try:
        data = json.loads(ts.read_text(encoding="utf-8"))
        return data.get("log_history", [])
    except (json.JSONDecodeError, KeyError):
        return None


def plot_curves(task_dir: Path, log_history: list[dict]) -> None:
    import matplotlib.pyplot as plt

    def extract(key: str) -> tuple[list, list]:
        return (
            [e["epoch"] for e in log_history if key in e],
            [e[key] for e in log_history if key in e],
        )

    ep_loss, losses = extract("loss")
    ep_ce, ce_losses = extract("forget_ce_loss")
    ep_ce_raw, ce_raw_losses = extract("forget_ce_raw")
    ep_rl, retain_losses = extract("retain_loss")
    ep_ml, mmlu_losses = extract("mmlu_loss")
    ep_crl, cov_retain_losses = extract("cov_retain_loss")
    ep_cfl, cov_forget_losses = extract("cov_forget_loss")
    ep_re, retain_ent = extract("retain_entropy")
    ep_fe, forget_ent = extract("forget_entropy")

    if not ep_loss:
        print(f"  Skip {task_dir.name}: no loss entries", file=sys.stderr)
        return

    # Special-case: when raw CE differs from clamped CE (cap is active), draw
    # both on the same panel so the cap is visually obvious. Otherwise just plot
    # the single CE curve. We pass an "overlay" tuple in the panel spec.
    ce_overlay = None
    if ce_raw_losses and ce_losses and any(
        abs(r - c) > 1e-6 for r, c in zip(ce_raw_losses, ce_losses)
    ):
        ce_overlay = (ep_ce_raw, ce_raw_losses, "lightgray", "Forget CE (raw)")

    panels = [
        ("loss", ep_loss, losses, "b-", "Total Loss", "Total Training Loss", True, None),
    ]
    if ce_losses:
        panels.append(("ce", ep_ce, ce_losses, "m-", "Forget CE Loss",
                       "Forget CE (post-clamp solid, raw shaded)" if ce_overlay
                       else "Forget Cross-Entropy Loss",
                       False, ce_overlay))
    if retain_losses:
        panels.append(("rl", ep_rl, retain_losses, "c-", "Retain Loss (KL)",
                       "Retain KL Divergence Loss", False, None))
    if mmlu_losses:
        panels.append(("ml", ep_ml, mmlu_losses, "y-", "MMLU Loss",
                       "MMLU NLL Loss", False, None))
    if cov_retain_losses:
        panels.append(("crl", ep_crl, cov_retain_losses, "tab:orange",
                       "Cov Retain Loss", "Covariance Retain (Option A)", False, None))
    if cov_forget_losses:
        panels.append(("cfl", ep_cfl, cov_forget_losses, "tab:brown",
                       "Cov Forget Loss", "Covariance Forget (Option B)", False, None))
    if retain_ent:
        panels.append(("retain_ent", ep_re, retain_ent, "r-",
                       "Retain Entropy", "Retain-Set Avg Entropy", False, None))
    if forget_ent:
        panels.append(("forget_ent", ep_fe, forget_ent, "g-",
                       "Forget Entropy", "Forget-Set Avg Entropy", False, None))

    n = len(panels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(axes.flatten())

    for ax, (key, x, y, color, ylabel, title, show_zero, overlay) in zip(axes_flat, panels):
        if overlay is not None:
            ox, oy, ocolor, olabel = overlay
            ax.plot(ox, oy, color=ocolor, linewidth=1.0, alpha=0.7, label=olabel)
        ax.plot(x, y, color, linewidth=1.2, alpha=0.9,
                label=ylabel if overlay is not None else None)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if show_zero:
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        if overlay is not None:
            ax.legend(loc="upper left", fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(task_dir.name, fontsize=10, y=1.01)
    fig.tight_layout()
    dest = task_dir / "training_curves.png"
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {dest}")


def main() -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        sys.exit("Install matplotlib: pip install matplotlib")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=None,
                    help="Single task directory to plot")
    ap.add_argument("--filter", type=str, default=None,
                    help="Substring filter for directory names")
    ap.add_argument("--unlearn-root", type=Path,
                    default=REPO_ROOT / "saves" / "unlearn",
                    help="Root directory containing task dirs")
    args = ap.parse_args()

    if args.dir:
        dirs = [args.dir.resolve()]
    else:
        root = args.unlearn_root
        if not root.is_dir():
            sys.exit(f"Not a directory: {root}")
        dirs = sorted(d for d in root.iterdir() if d.is_dir())
        if args.filter:
            dirs = [d for d in dirs if args.filter in d.name]

    if not dirs:
        sys.exit("No matching directories found.")

    plotted = 0
    for task_dir in dirs:
        log_history = load_trainer_state(task_dir)
        if log_history is None:
            continue
        plot_curves(task_dir, log_history)
        plotted += 1

    print(f"\nPlotted {plotted} run(s).")


if __name__ == "__main__":
    main()
