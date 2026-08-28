"""Train naive gradient ascent (GradAscent) using unlearn/tofu/default hyperparameters, then eval."""

from __future__ import annotations

import argparse
import sys

import _common
from baseline_registry import TASK_TRAIN_GRAD_ASCENT, train_overrides_grad_ascent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-only", action="store_true")
    args = p.parse_args()
    _common.require_oracle()
    if not args.skip_train:
        if _common.train_local(
            "GradAscent", TASK_TRAIN_GRAD_ASCENT, train_overrides_grad_ascent()
        ) != 0:
            sys.exit(1)
        if args.train_only:
            return
    if _common.eval_local_checkpoint(TASK_TRAIN_GRAD_ASCENT) != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
