"""Run train+eval for every comparison method at 5 epochs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import _common
from baseline_registry_5ep import (
    TASK_TRAIN_SIMNPO_5EP, train_overrides_simnpo_5ep,
    TASK_TRAIN_RMU_5EP, train_overrides_rmu_5ep,
    TASK_TRAIN_NPO_5EP, train_overrides_npo_5ep,
    TASK_TRAIN_UNDIAL_5EP, train_overrides_undial_5ep,
    TASK_TRAIN_GRADDIFF_5EP, train_overrides_graddiff_5ep,
    TASK_TRAIN_GRAD_ASCENT_5EP, train_overrides_grad_ascent_5ep,
)

REPO_ROOT = _common.REPO_ROOT
EVAL_BATCH_SIZE = 8

METHODS = [
    ("SimNPO", TASK_TRAIN_SIMNPO_5EP, train_overrides_simnpo_5ep),
    ("RMU", TASK_TRAIN_RMU_5EP, train_overrides_rmu_5ep),
    ("NPO", TASK_TRAIN_NPO_5EP, train_overrides_npo_5ep),
    ("UNDIAL", TASK_TRAIN_UNDIAL_5EP, train_overrides_undial_5ep),
    ("GradDiff", TASK_TRAIN_GRADDIFF_5EP, train_overrides_graddiff_5ep),
    ("GradAscent", TASK_TRAIN_GRAD_ASCENT_5EP, train_overrides_grad_ascent_5ep),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--open-unlearning-root", type=Path, default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    args = p.parse_args()

    if args.open_unlearning_root is not None:
        os.environ["OPEN_UNLEARNING_ROOT"] = str(
            args.open_unlearning_root.expanduser().resolve()
        )
    _common.require_oracle()

    results = {}
    for trainer_name, task, overrides_fn in METHODS:
        summary_path = REPO_ROOT / "saves" / "unlearn" / task / "evals" / "TOFU_SUMMARY.json"
        if summary_path.is_file():
            metrics = json.loads(summary_path.read_text(encoding="utf-8"))
            results[trainer_name] = {"task": task, "metrics": metrics}
            print(f"\nSKIP {trainer_name} (already has TOFU_SUMMARY): {metrics}", flush=True)
            continue

        print(f"\n{'='*60}\n5-EPOCH BASELINE: {trainer_name}\n{'='*60}", flush=True)

        ckpt_dir = REPO_ROOT / "saves" / "unlearn" / task
        ckpt_exists = (ckpt_dir / "model.safetensors").is_file()

        if not args.skip_train and not ckpt_exists:
            rc = _common.train_local(trainer_name, task, overrides_fn())
            if rc != 0:
                print(f"TRAIN FAILED for {trainer_name}", flush=True)
                results[trainer_name] = {"error": "train_failed"}
                continue
            if args.train_only:
                continue
        elif ckpt_exists:
            print(f"  Checkpoint exists, skipping training", flush=True)

        rc = _common.eval_local_checkpoint(task, eval_batch_size=args.eval_batch_size)
        if rc != 0:
            print(f"EVAL FAILED for {trainer_name}", flush=True)
            results[trainer_name] = {"error": "eval_failed"}
            continue

        if summary_path.is_file():
            metrics = json.loads(summary_path.read_text(encoding="utf-8"))
            results[trainer_name] = {"task": task, "metrics": metrics}
            print(f"  {trainer_name}: {metrics}", flush=True)
        else:
            results[trainer_name] = {"task": task, "error": "no_summary"}

    out_dir = REPO_ROOT / "results" / "erase_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "baselines_5ep.json"
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}", flush=True)


if __name__ == "__main__":
    main()
