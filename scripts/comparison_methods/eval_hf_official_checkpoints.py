"""
Evaluate published Open Unlearning HF checkpoints (TOFU Llama-3.2-1B, forget10).

Uses eval/tofu/llama1b/default and (by default) open-unlearning oracle TOFU_EVAL.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from baseline_registry import HF_CHECKPOINT_EVAL_RUNS

SCRIPT_DIR = Path(__file__).resolve().parent
# Dir is comparison_methods: parents[2] == research-unlearning (parents[0]==self)
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
import _common  # noqa: E402

EVAL = REPO_ROOT / "src" / "eval.py"
PYTHON = _common.PYTHON


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--open-unlearning-root",
        type=Path,
        default=None,
        help="If set, retain_logs and default --eval-saves-dir use this repo.",
    )
    p.add_argument(
        "--eval-saves-dir",
        type=Path,
        default=None,
        help="Parent for hf_official_* output dirs (default: <OU>/saves/eval if OU set).",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Aggregate JSON (default: <eval-saves-dir>/../results/... or REPO results).",
    )
    args = p.parse_args()

    ou = args.open_unlearning_root.expanduser().resolve() if args.open_unlearning_root else None
    if ou is not None:
        os.environ["OPEN_UNLEARNING_ROOT"] = str(ou)
    rl = _common.oracle_eval_path()
    if not rl.is_file():
        raise SystemExit(f"Missing oracle TOFU_EVAL.json: {rl}")

    eval_root = args.eval_saves_dir
    if eval_root is None:
        if ou is not None:
            eval_root = ou / "saves" / "eval"
        else:
            eval_root = REPO_ROOT / "saves" / "eval"
    eval_root = eval_root.resolve()
    eval_root.mkdir(parents=True, exist_ok=True)

    summary_out = args.summary_out
    if summary_out is None:
        summary_out = (
            ou / "results" / "comparison" / "HF_OFFICIAL_CHECKPOINTS_SUMMARY.json"
            if ou is not None
            else REPO_ROOT / "results" / "comparison" / "HF_OFFICIAL_CHECKPOINTS_SUMMARY.json"
        )
    summary_out = summary_out.resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for task_name, model_id in HF_CHECKPOINT_EVAL_RUNS:
        out_dir = eval_root / task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            "-W",
            "ignore",
            str(EVAL),
            "--config-name=eval.yaml",
            f"experiment={_common.EVAL_EXPERIMENT}",
            f"task_name={task_name}",
            "forget_split=forget10",
            "holdout_split=holdout10",
            f"retain_logs_path={rl.as_posix()}",
            f"model.model_args.pretrained_model_name_or_path={model_id}",
            "model.model_args.attn_implementation=eager",
            f"paths.output_dir={out_dir.as_posix()}",
        ]
        print("Running:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            results[task_name] = {"error": "eval_failed", "model_id": model_id}
            continue
        sp = out_dir / "TOFU_SUMMARY.json"
        if sp.is_file():
            results[task_name] = {
                "model_id": model_id,
                "metrics": json.loads(sp.read_text(encoding="utf-8")),
            }
        else:
            results[task_name] = {"error": "no TOFU_SUMMARY", "model_id": model_id}

    summary_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {summary_out}", flush=True)


if __name__ == "__main__":
    main()
