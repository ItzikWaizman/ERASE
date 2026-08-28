"""Train UNDIAL with hub-matched hyperparameters, then eval (TOFU forget10, Llama 1B)."""

from __future__ import annotations

import argparse
import sys

import _common
from baseline_registry import TASK_TRAIN_UNDIAL, train_overrides_undial


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-only", action="store_true")
    args = p.parse_args()
    _common.require_oracle()
    if not args.skip_train:
        if _common.train_local("UNDIAL", TASK_TRAIN_UNDIAL, train_overrides_undial()) != 0:
            sys.exit(1)
        if args.train_only:
            return
    if _common.eval_local_checkpoint(TASK_TRAIN_UNDIAL) != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
