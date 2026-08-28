"""Plot the per-sample MIA-score distribution (forget vs holdout) from MUSE
MUSE_EVAL.json files -- the MUSE analog of the TOFU ce_distribution plot.

PrivLeak in MUSE is driven by how separable the forget-set MIA scores are from
the holdout set (a never-trained reference). The retrain oracle should have
forget ~ holdout (overlapping -> AUC ~ 0.5 -> PrivLeak ~ 0); a model that still
"remembers" has the forget distribution shifted (AUC ~ 1 -> PrivLeak very
negative). This overlays forget (red) vs holdout (blue) per model so the shift
is visible.

Usage:
    python scripts/analysis/plot_muse_mia_distribution.py \
        --metric mia_min_k --out results/remote_runs_muse/muse_mia_distribution.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "results" / "remote_runs_muse"

# (label, run_dir) -> the MUSE_EVAL.json is found at <dir>/MUSE_EVAL.json or
# <dir>/checkpoint-*/evals/MUSE_EVAL.json.
MODELS = [
    ("target", "muse_Llama-2-7b-hf_News_target"),
    ("retrain (oracle)", "muse_Llama-2-7b-hf_News_retrain"),
    ("ERASE tau=1.5", "MUSE_news_7b_erase_lratau_ner_fg_lr0.04_a2_tau1.5"),
    ("ERASE tau=1.75", "MUSE_news_7b_erase_lratau_ner_fg_lr0.04_a2_tau1.75"),
    ("ERASE tau=2.0", "MUSE_news_7b_erase_lratau_ner_fg_lr0.04_a2_tau2.0"),
]


def find_eval(run_dir: Path) -> Path | None:
    direct = run_dir / "MUSE_EVAL.json"
    if direct.is_file():
        return direct
    hits = sorted(run_dir.glob("checkpoint-*/evals/MUSE_EVAL.json"))
    return hits[-1] if hits else None


def scores(eval_json: dict, metric: str, split: str) -> np.ndarray:
    vbi = eval_json[metric][split]["value_by_index"]
    out = []
    for _, v in vbi.items():
        out.append(v["score"] if isinstance(v, dict) else float(v))
    return np.asarray(out, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="mia_min_k",
                   help="MIA metric to plot (mia_min_k drives PrivLeak).")
    p.add_argument("--out", default=str(RUNS_DIR / "muse_mia_distribution.png"))
    args = p.parse_args()

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(8, 2.1 * len(MODELS)),
                             sharex=True)
    for ax, (label, sub) in zip(axes, MODELS):
        ep = find_eval(RUNS_DIR / sub)
        if ep is None:
            ax.set_title(f"{label}: MUSE_EVAL.json not found")
            continue
        d = json.loads(ep.read_text(encoding="utf-8"))
        f = scores(d, args.metric, "forget")
        h = scores(d, args.metric, "holdout")
        auc = d[args.metric].get("auc")
        priv = d["privleak"]["agg_value"] if isinstance(d["privleak"], dict) else d["privleak"]
        lo = min(f.min(), h.min())
        hi = max(f.max(), h.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(h, bins=bins, alpha=0.55, color="tab:blue", label="holdout (never trained)", density=True)
        ax.hist(f, bins=bins, alpha=0.55, color="tab:red", label="forget", density=True)
        ax.axvline(f.mean(), color="tab:red", ls="--", lw=1)
        ax.axvline(h.mean(), color="tab:blue", ls="--", lw=1)
        ax.set_title(f"{label}   |   {args.metric} AUC={auc:.3f}   PrivLeak={priv:.1f}   "
                     f"(forget mean {f.mean():.2f} vs holdout {h.mean():.2f})",
                     fontsize=9)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel(f"{args.metric} per-sample score, nats  (LOW = confident/memorized = 'member';  "
                        f"HIGH = surprised = 'non-member/never-seen')")
    fig.suptitle(f"MUSE-News forget vs holdout MIA-score distributions ({args.metric})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
