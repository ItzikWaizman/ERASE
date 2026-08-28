"""
Load TOFU_SUMMARY.json for baseline unlearning runs + one ERASE run, compare to oracle.

1) Pareto scatter + front: model_utility (max) vs extraction_strength (min)
2) Pareto scatter + front: model_utility (max) vs forget_truth_ratio (min)
3) Per-metric bar charts: scores normalized so oracle = 1.0   — Higher-is-better: score = value / oracle
   — Lower-is-better: score = oracle / value

Outputs under results/plots/ (PNG).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_REL = Path("saves") / "eval" / "tofu_llama-1b_oracle_retain90" / "TOFU_SUMMARY.json"

# (display label, subdir under saves/unlearn, or list of candidate subdirs)
METHOD_RUNS: list[tuple[str, str | list[str]]] = [
    ("GradAscent", "baseline_train_GradAscent_default"),
    ("GradDiff", "baseline_train_GradDiff_hf_hparams"),
    ("NPO", "baseline_train_NPO_hf_hparams"),
    ("RMU", "baseline_train_RMU_hf_hparams"),
    ("SimNPO", "baseline_train_SimNPO_hf_hparams"),
    ("UNDIAL", "baseline_train_UNDIAL_hf_hparams"),
    ("ERASE (prev)", "expD_10ep_wiki_lr0.077_b0_a0.31_k10_rw0.32"),
    ("ERASE+CovRetain", "New_Loss_Exp_10ep_wiki_lr0.08_b0_a0.05_k10_rw0.05_crw0.001_L1234"),
    ("ERASE+CovForget", "COV_LOSS_ABLATION_10ep_wiki_lr0.08_b0_a0.05_k10_rw0.05_cfw0.005_L1234"),
]

# Metrics where lower raw value is better (normalize with oracle/value)
LOWER_IS_BETTER = frozenset(
    {
        "extraction_strength",
        "forget_truth_ratio",
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "forget_quality",
    }
)

# Oracle TOFU_SUMMARY has no consistent "direction" for this; skip normalized bar.
SKIP_NORM_METRICS = frozenset({"privleak"})

# Dropped from oracle-normalized bar charts only (still included in Pareto plots).
NORM_SKIP_METHOD_LABELS = frozenset({"GradAscent"})


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def summary_path(repo: Path, unlearn_subdir: str) -> Path:
    return repo / "saves" / "unlearn" / unlearn_subdir / "evals" / "TOFU_SUMMARY.json"


def resolve_method_paths(repo: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for label, spec in METHOD_RUNS:
        if isinstance(spec, str):
            candidates = [spec]
        else:
            candidates = spec
        chosen: Path | None = None
        for sub in candidates:
            p = summary_path(repo, sub)
            if p.is_file():
                chosen = p
                break
        if chosen is None:
            tried = ", ".join(str(summary_path(repo, s)) for s in candidates)
            print(f"Warning: missing {label} — tried: {tried}", file=sys.stderr)
            continue
        out.append((label, chosen))
    return out


def pareto_max_min_maximize_first(
    primary: list[float], secondary: list[float]
) -> list[bool]:
    """Non-dominated when maximizing primary[·] and minimizing secondary[·]."""
    n = len(primary)
    nd = [True] * n
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if primary[j] >= primary[i] and secondary[j] <= secondary[i]:
                if primary[j] > primary[i] or secondary[j] < secondary[i]:
                    nd[i] = False
                    break
    return nd


def plot_pareto(
    repo: Path,
    out_dir: Path,
    x_key: str,
    y_key: str,
    x_minimize: bool,
    labels: list[str],
    rows: list[dict],
    oracle: dict,
    fname: str,
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    xs: list[float] = []
    ys: list[float] = []
    labs: list[str] = []
    for lab, m in zip(labels, rows):
        xv = m.get(x_key)
        yv = m.get(y_key)
        if xv is None or yv is None:
            continue
        xs.append(float(xv))
        ys.append(float(yv))
        labs.append(lab)

    if len(xs) < 1:
        print(f"Skip Pareto {fname}: no points", file=sys.stderr)
        return

    if x_minimize:
        sec = xs
        prim = ys
    else:
        prim = xs
        sec = ys

    nd = pareto_max_min_maximize_first(prim, sec)

    fig, ax = plt.subplots(figsize=(10, 7))
    o_x = oracle.get(x_key)
    o_y = oracle.get(y_key)
    if o_x is not None and o_y is not None:
        ax.scatter(
            [float(o_x)],
            [float(o_y)],
            s=160,
            c="gold",
            edgecolors="black",
            zorder=3,
            marker="*",
            label="Oracle (retain90)",
        )

    texts = []
    for i, (lab, x, y, on) in enumerate(zip(labs, xs, ys, nd)):
        ax.scatter(
            x,
            y,
            s=90,
            c="tab:green" if on else "lightgray",
            edgecolors="black",
            zorder=2 if on else 1,
        )
        texts.append((lab, x, y))

    label_offsets = {
        "ERASE (prev)":     (8, -18),
        "ERASE+CovRetain":  (-5, 10),
        "ERASE+CovForget":  (8, -20),
        "RMU":            (-8, 8),
        "UNDIAL":         (6, -12),
        "GradAscent":     (6, 8),
        "NPO":            (6, 8),
        "SimNPO":         (6, 8),
        "GradDiff":       (6, 8),
    }
    for lab, x, y in texts:
        dx, dy = label_offsets.get(lab, (6, 6))
        ax.annotate(
            lab, (x, y), fontsize=8.5, fontweight="bold" if "ERASE" in lab else "normal",
            xytext=(dx, dy), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, shrinkA=0, shrinkB=3),
        )

    pf_x = [xs[i] for i in range(len(xs)) if nd[i]]
    pf_y = [ys[i] for i in range(len(ys)) if nd[i]]
    if x_minimize:
        order = sorted(range(len(pf_x)), key=lambda k: pf_x[k])
    else:
        order = sorted(range(len(pf_x)), key=lambda k: pf_x[k])
    pf_xs = [pf_x[i] for i in order]
    pf_ys = [pf_y[i] for i in order]
    if len(pf_xs) >= 2:
        ax.plot(pf_xs, pf_ys, "g--", alpha=0.65, linewidth=2, label="Pareto front")
    elif len(pf_xs) == 1:
        ax.scatter(pf_xs, pf_ys, s=120, facecolors="none", edgecolors="green", linewidths=2)

    ax.set_xlabel(
        f"{x_key} (lower better)" if x_minimize else f"{x_key} (higher better)",
        fontsize=11,
    )
    ax.set_ylabel(f"{y_key} (higher better)", fontsize=11)
    ax.set_title(f"Pareto: {y_key} vs {x_key} {title_suffix}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    dest = out_dir / fname
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print("Wrote", dest)

    table = {
        "x_key": x_key,
        "y_key": y_key,
        "labels": labs,
        x_key: xs,
        y_key: ys,
        "on_pareto_front": nd,
    }
    (out_dir / fname.replace(".png", ".json")).write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )


def normalized_score(metric: str, value: float, oracle_val: float) -> float | None:
    if oracle_val == 0 and metric in LOWER_IS_BETTER:
        return None
    if oracle_val == 0 and metric not in LOWER_IS_BETTER:
        return None
    if metric in LOWER_IS_BETTER:
        if value == 0:
            return None
        return oracle_val / value
    return value / oracle_val


def plot_normalized_metrics(
    repo: Path,
    out_dir: Path,
    labels: list[str],
    rows: list[dict],
    oracle: dict,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metric_keys = sorted(
        k
        for k in oracle
        if k not in SKIP_NORM_METRICS
        and isinstance(oracle.get(k), (int, float))
        and not isinstance(oracle.get(k), bool)
    )

    for metric in metric_keys:
        o = oracle.get(metric)
        if o is None:
            continue
        o = float(o)
        vals: list[float] = []
        use_labels: list[str] = []
        for lab, m in zip(labels, rows):
            if lab in NORM_SKIP_METHOD_LABELS:
                continue
            v = m.get(metric)
            if v is None:
                continue
            s = normalized_score(metric, float(v), o)
            if s is None:
                continue
            vals.append(s)
            use_labels.append(lab)

        if not vals:
            continue

        fig, ax = plt.subplots(figsize=(max(7.5, len(vals) * 0.45), 4.8))
        x = np.arange(len(vals))
        ax.bar(x, vals, color="steelblue")
        ax.axhline(1.0, color="orange", linestyle="--", linewidth=1.5, label="Oracle (=1)")
        ax.set_xticks(x)
        ax.set_xticklabels(use_labels, rotation=35, ha="right")
        direction = "lower raw better → higher score" if metric in LOWER_IS_BETTER else "higher raw better"
        ax.set_ylabel(f"Score vs oracle ({direction})")
        ax.set_title(f"{metric} (oracle-normalized)")
        ax.legend()
        fig.tight_layout()
        safe = metric.replace(" ", "_")
        dest = out_dir / f"norm_oracle_{safe}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        print("Wrote", dest)


def main() -> None:
    try:
        import matplotlib.pyplot as plt # noqa: F401
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    out_dir = repo / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_path = repo / ORACLE_REL
    oracle = load_json(oracle_path)
    if not oracle:
        print(f"Oracle not found: {oracle_path}", file=sys.stderr)
        sys.exit(1)

    resolved = resolve_method_paths(repo)
    if not resolved:
        print("No method summaries found.", file=sys.stderr)
        sys.exit(1)

    labels = [t[0] for t in resolved]
    rows = []
    for _, p in resolved:
        m = load_json(p)
        if not m:
            print(f"Empty or invalid: {p}", file=sys.stderr)
            sys.exit(1)
        rows.append(m)

    plot_pareto(
        repo,
        out_dir,
        x_key="extraction_strength",
        y_key="model_utility",
        x_minimize=True,
        labels=labels,
        rows=rows,
        oracle=oracle,
        fname="pareto_model_utility_vs_extraction_strength.png",
        title_suffix="(methods)",
    )
    plot_pareto(
        repo,
        out_dir,
        x_key="forget_truth_ratio",
        y_key="model_utility",
        x_minimize=True,
        labels=labels,
        rows=rows,
        oracle=oracle,
        fname="pareto_model_utility_vs_forget_truth_ratio.png",
        title_suffix="(methods)",
    )
    plot_normalized_metrics(repo, out_dir, labels, rows, oracle)


if __name__ == "__main__":
    main()
