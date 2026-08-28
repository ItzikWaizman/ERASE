"""Build the comprehensive ERASE hyperparameter grid as N shard JSON files.

This is the **hyperparameter grid search** (not the ablation study, which
toggles components on/off and lives elsewhere).

Design (see DEPLOY.md "The grid search" section for the rationale):

Block A -- 6-D Cartesian (720 runs)::

    cov_retain_weight       (4)  [0.005, 0.01, 0.05, 0.1]
    forget_loss_per_token_cap (5) [2, 2.5, 3, 3.5, 4]
    mmlu_loss_weight        (2)  [0, 0.003]
    retain_weight           (3)  [0.001, 0.05, 0.1]
    tr_loss_weight          (3)  [0, 0.001, 0.01]
    epochs                  (2)  [10, 15]

Block B -- 1-D OAT extras at the grid center (9 runs)::

    cov_retain_weight = 0.001                         (1 run)
    mmlu_loss_weight  = 0.0005, 0.001, 0.005          (3 runs)
    tr_loss_weight    = 1e-5, 0.05                    (2 runs)
    epochs            = 12, 13, 14                    (3 runs)

Center cell::

    cov_retain_weight=0.005, forget_loss_per_token_cap=3,
    mmlu_loss_weight=0.003, retain_weight=0.05,
    tr_loss_weight=0, epochs=15.

Common to all runs (overrides DEFAULT_RUN in run_erase.py):

    cov_forget_weight = 0       (we sweep WITHOUT cov_forget term)
    tr_loss_target_logtr = 0    (TR loss target = no-info logit ratio)
    task_prefix = "GRID"

Total: 729 runs.

Default partitioning targets **6 GPUs x 2 processes per GPU = 12 shards**
(~60 runs per process).

Usage::

    python scripts/erase/build_grid_search.py              # 12 shards (default)
    python scripts/erase/build_grid_search.py --shards 24  # different fan-out
    python scripts/erase/build_grid_search.py --out-dir /tmp/foo

Output::

    configs/runs/grid_search/all_runs.json    (full list of 729 runs)
    configs/runs/grid_search/shard_00.json    (one shard per process)
    configs/runs/grid_search/shard_01.json
    ...
    configs/runs/grid_search/shard_11.json
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "configs" / "runs" / "grid_search"


# ---------------------------------------------------------------------------
# Knobs that are common to every run in the grid
# ---------------------------------------------------------------------------
COMMON: dict = {
    "task_prefix": "GRID",
    "cov_forget_weight": 0,
    "tr_loss_target_logtr": 0,
}


# ---------------------------------------------------------------------------
# Block A: 6-D Cartesian
# ---------------------------------------------------------------------------
COV_RETAIN_VALUES = [0.005, 0.01, 0.05, 0.1]
PT_VALUES = [2, 2.5, 3, 3.5, 4]
MMLU_VALUES = [0, 0.003]
RETAIN_VALUES = [0.001, 0.05, 0.1]
TR_VALUES = [0, 0.001, 0.01]
EPOCHS_VALUES = [10, 15]


def build_block_a() -> list[dict]:
    runs: list[dict] = []
    for cov, pt, mm, rw, tr, ep in product(
        COV_RETAIN_VALUES, PT_VALUES, MMLU_VALUES,
        RETAIN_VALUES, TR_VALUES, EPOCHS_VALUES,
    ):
        runs.append({
            **COMMON,
            "cov_retain_weight": cov,
            "forget_loss_per_token_cap": pt,
            "mmlu_loss_weight": mm,
            "retain_weight": rw,
            "tr_loss_weight": tr,
            "epochs": ep,
        })
    return runs


# ---------------------------------------------------------------------------
# Block B: 1-D OAT extras at the grid center
# ---------------------------------------------------------------------------
CENTER: dict = {
    "cov_retain_weight": 0.005,
    "forget_loss_per_token_cap": 3,
    "mmlu_loss_weight": 0.003,
    "retain_weight": 0.05,
    "tr_loss_weight": 0,
    "epochs": 15,
}


def _at_center(**override) -> dict:
    return {**COMMON, **CENTER, **override}


def build_block_b() -> list[dict]:
    runs: list[dict] = []
    runs.append(_at_center(cov_retain_weight=0.001))
    runs.append(_at_center(mmlu_loss_weight=0.0005))
    runs.append(_at_center(mmlu_loss_weight=0.001))
    runs.append(_at_center(mmlu_loss_weight=0.005))
    runs.append(_at_center(tr_loss_weight=1e-5))
    runs.append(_at_center(tr_loss_weight=0.05))
    runs.append(_at_center(epochs=12))
    runs.append(_at_center(epochs=13))
    runs.append(_at_center(epochs=14))
    return runs


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def shard_round_robin(runs: list[dict], n_shards: int) -> list[list[dict]]:
    """Round-robin so every shard sees every axis early (good for monitoring)."""
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, r in enumerate(runs):
        shards[i % n_shards].append(r)
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--shards", type=int, default=12,
        help="Number of shard files (= parallel processes). Default 12 "
        "(2 per GPU x 6 GPUs).",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Where to write the JSON shard files.",
    )
    args = ap.parse_args()

    block_a = build_block_a()
    block_b = build_block_b()
    runs = block_a + block_b
    print(
        f"Block A (6-D Cartesian): {len(block_a)} runs\n"
        f"Block B (OAT extras):    {len(block_b)} runs\n"
        f"TOTAL:                   {len(runs)} runs"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.out_dir / "all_runs.json"
    all_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nFull list -> {all_path}")

    shards = shard_round_robin(runs, args.shards)
    width = max(2, len(str(args.shards - 1)))
    print(f"\nSplitting into {args.shards} shard(s):")
    for i, sh in enumerate(shards):
        path = args.out_dir / f"shard_{i:0{width}d}.json"
        path.write_text(json.dumps(sh, indent=2), encoding="utf-8")
        print(f"  shard {i:0{width}d}: {len(sh):>3} runs -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
