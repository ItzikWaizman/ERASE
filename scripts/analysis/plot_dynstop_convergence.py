"""Plot dynamic-stopping convergence: samples remaining vs training step.

Usage:
    python scripts/analysis/plot_dynstop_convergence.py <path_to_trainer_state.json>

Produces `dynstop_convergence.png` alongside the JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <trainer_state.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        print("No log_history found in trainer_state.json")
        sys.exit(1)

    steps = []
    n_done = []
    for entry in log_history:
        if "dynstop_n_done_total" in entry and "step" in entry:
            steps.append(entry["step"])
            n_done.append(entry["dynstop_n_done_total"])

    if not steps:
        print("No dynstop_n_done_total entries found — dynamic stopping was not active.")
        sys.exit(0)

    steps = np.array(steps)
    n_done = np.array(n_done)

    total_samples = int(n_done.max()) if n_done.max() > 0 else 400
    # Try to infer total from the run (done can't exceed total)
    # Use 400 as fallback (TOFU forget10 split size)
    if total_samples < 10:
        total_samples = 400
    n_remaining = total_samples - n_done

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, n_remaining, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Samples Remaining (not yet in band)", fontsize=12)
    ax.set_title(
        f"Dynamic Stopping Convergence (total={total_samples})",
        fontsize=13,
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Annotate final state
    final_done = int(n_done[-1])
    final_step = int(steps[-1])
    ax.axhline(0, color="green", linestyle="--", alpha=0.5, label="All done")
    stats_text = (
        f"Final step: {final_step}\n"
        f"Done: {final_done}/{total_samples}\n"
        f"Remaining: {total_samples - final_done}"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontfamily="monospace",
    )
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_path = path.parent / "dynstop_convergence.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  Total steps: {final_step}, Done: {final_done}/{total_samples}")


if __name__ == "__main__":
    main()
