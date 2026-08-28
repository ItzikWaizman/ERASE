"""CLI wrapper for src/analysis/paper_aggregates.py.

Computes Memorization / Utility / Aggregate scores (OpenUnlearning paper
Table 3 style) from a TOFU_EVAL.json and writes PAPER_AGGREGATES.json +
PAPER_AGGREGATES.md next to it.

Usage:
  # single run dir (anything that resolves to .../<task>/evals or .../<task>/)
  python scripts/analysis/compute_paper_aggregates.py --dir saves/unlearn/<task>

  # batch over all tasks under saves/unlearn that have a TOFU_EVAL.json
  python scripts/analysis/compute_paper_aggregates.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.paper_aggregates import aggregate_eval_dir  # noqa: E402


def _resolve_eval_dir(path: Path) -> Path | None:
    """Accept either a task dir (.../<task>/) or its evals subdir."""
    path = path.resolve()
    if (path / "TOFU_EVAL.json").is_file():
        return path
    sub = path / "evals"
    if (sub / "TOFU_EVAL.json").is_file():
        return sub
    return None


def _maybe_dir(p: Path | None) -> Path | None:
    if p is None:
        return None
    if p.is_file():
        return p
    if p.is_dir():
        return _resolve_eval_dir(p)
    return None


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=None,
                    help="Single task dir (or its evals subdir) to aggregate.")
    ap.add_argument("--all", action="store_true",
                    help="Aggregate all tasks under saves/unlearn that have a TOFU_EVAL.json.")
    ap.add_argument("--unlearn-root", type=Path,
                    default=REPO_ROOT / "saves" / "unlearn")
    ap.add_argument("--init-eval", type=Path, default=None,
                    help="Path to init-finetuned eval dir or TOFU_EVAL.json. "
                         "Enables Util normalization (paper Table 3 style).")
    ap.add_argument("--retain-eval", type=Path, default=None,
                    help="Path to retain-oracle eval dir or TOFU_EVAL.json. "
                         "Enables Privacy aggregate computation.")
    args = ap.parse_args()

    if not args.dir and not args.all:
        sys.exit("Specify --dir <path> or --all.")

    init_resolved = _maybe_dir(args.init_eval)
    retain_resolved = _maybe_dir(args.retain_eval)
    if args.init_eval and init_resolved is None:
        sys.exit(f"--init-eval: no TOFU_EVAL.json found at {args.init_eval}")
    if args.retain_eval and retain_resolved is None:
        sys.exit(f"--retain-eval: no TOFU_EVAL.json found at {args.retain_eval}")

    targets: list[Path] = []
    if args.dir:
        ed = _resolve_eval_dir(args.dir)
        if ed is None:
            sys.exit(f"No TOFU_EVAL.json found under {args.dir}")
        targets.append(ed)

    if args.all:
        if not args.unlearn_root.is_dir():
            sys.exit(f"Not a directory: {args.unlearn_root}")
        for task in sorted(args.unlearn_root.iterdir()):
            if not task.is_dir():
                continue
            ed = _resolve_eval_dir(task)
            if ed is not None:
                targets.append(ed)

    if not targets:
        sys.exit("No eval directories with TOFU_EVAL.json found.")

    n_ok = 0
    for ed in targets:
        payload = aggregate_eval_dir(ed, init_resolved, retain_resolved)
        if payload is None:
            print(f"  skip (bad json): {ed}")
            continue
        a = payload["aggregates"]
        run = ed.parent.name if ed.name == "evals" else ed.name
        print(
            f"  ok: {run:55s}  "
            f"Mem={_fmt(a['memorization'])}  "
            f"Util={_fmt(a['utility'])}  "
            f"Priv={_fmt(a['privacy'])}  "
            f"Agg={_fmt(a['aggregate'])}"
        )
        n_ok += 1
    print(f"\nWrote PAPER_AGGREGATES.{{json,md}} for {n_ok}/{len(targets)} runs.")


if __name__ == "__main__":
    main()
