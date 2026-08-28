"""Run train+eval for every comparison method (hub-matched hyperparameters)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import _common

REPO_ROOT = _common.REPO_ROOT
PYTHON = _common.PYTHON
HERE = Path(__file__).resolve().parent

SCRIPTS = (
    #"run_simnpo_baseline.py",
    #"run_rmu_baseline.py",
    #"run_npo_baseline.py",
    "run_undial_baseline.py",
    "run_graddiff_baseline.py",
    "run_grad_ascent_baseline.py",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--open-unlearning-root",
        type=Path,
        default=None,
        help="Sets OPEN_UNLEARNING_ROOT for oracle retain_logs_path (train+eval).",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only run eval for each method (expects checkpoints already under saves/unlearn/).",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Only train each method (skip eval after training).",
    )
    args = p.parse_args()
    if args.open_unlearning_root is not None:
        os.environ["OPEN_UNLEARNING_ROOT"] = str(
            args.open_unlearning_root.expanduser().resolve()
        )
    extra: list[str] = []
    if args.skip_train:
        extra.append("--skip-train")
    if args.train_only:
        extra.append("--train-only")

    for name in SCRIPTS:
        script = HERE / name
        cmd = [PYTHON, "-W", "ignore", str(script), *extra]
        print("RUN:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
