"""
Drive one ERASE-on-MUSE VJP variant across the full tau sweep, in a single job.

Reads a sweep spec (configs/runs/MUSE_news_7b_erase/sweep.json) that defines:
  - the Hydra experiment to run,
  - the list of CET targets (tau_sweep) + band half-width,
  - common method overrides, and
  - per-method overrides for the 3 VJP variants {raw, ner, qa}.

For the chosen --method it loops over every tau, launching src/train.py with the
right Hydra overrides (tau drives the CET target, band, and dynamic-stop
thresholds). Each run trains + evaluates MUSE in-process; the 13GB checkpoint
weights are deleted afterwards unless --keep-weights is given.

Usage (one GPU job per method):
    python scripts/erase/run_muse_sweep.py \
        configs/runs/MUSE_news_7b_erase/sweep.json --method raw
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_json")
    p.add_argument("--method", required=True, choices=["raw", "ner", "qa"])
    p.add_argument("--config-name", default="unlearn.yaml")
    p.add_argument("--keep-weights", action="store_true")
    p.add_argument(
        "--tag", default="",
        help="Optional run tag inserted into task_name/output dir so a rerun "
             "lands in fresh saves/unlearn dirs instead of overwriting a prior "
             "sweep (e.g. --tag gstop -> MUSE_news_7b_erase_gstop_ner_tau1.5).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Validate the full pipeline fast: run ONLY the first tau, capped to "
             "a few optimizer steps + one epoch, then the normal MUSE eval. Use "
             "this once before the real sweep to catch config/model/eval errors "
             "in minutes instead of after every tau's model load.",
    )
    p.add_argument(
        "--grid", action="store_true",
        help="Convergence search: instead of the tau sweep, fix tau=probe_tau and "
             "sweep lr_sweep x alpha_sweep (from the sweep JSON), with a "
             "constant-with-warmup schedule (cosine-to-zero starved late epochs). "
             "Use this to find an LR/alpha that actually moves the weights.",
    )
    p.add_argument(
        "--full-grid", action="store_true",
        help="Full 3-D search over lr_grid x alpha_grid x tau_grid (from the sweep "
             "JSON), constant schedule. Combine with --shard/--nshards to split the "
             "cells across parallel jobs (e.g. a SLURM --array).",
    )
    p.add_argument(
        "--shard", type=int, default=0,
        help="0-based shard index for --full-grid (defaults to SLURM_ARRAY_TASK_ID "
             "if set). Runs cells[shard::nshards].",
    )
    p.add_argument(
        "--nshards", type=int, default=0,
        help="Total number of shards for --full-grid (defaults to "
             "SLURM_ARRAY_TASK_COUNT if set, else 1).",
    )
    return p.parse_args()


def _run_one(task, experiment, config_name, base, extra_overrides, hw, tau,
             n_chunks, keep_weights):
    """Launch one src/train.py run + post-hoc monitor; returns rc."""
    overrides = base + extra_overrides + [f"task_name={task}"]
    cmd = [
        sys.executable, "src/train.py",
        f"--config-name={config_name}",
        f"experiment={experiment}",
    ] + overrides
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return rc
    out_dir = Path("saves/unlearn") / task
    try:
        subprocess.run([
            sys.executable, "scripts/erase/analyze_muse_run.py",
            "--run_dir", str(out_dir),
            "--target_tau", str(tau),
            "--band", str(hw),
            "--n_chunks", str(n_chunks),
        ], check=False)
    except Exception as e:  # noqa: BLE001
        print(f"[run_muse_sweep] analyze {task} skipped ({e}).", flush=True)
    if not keep_weights:
        for pat in ("*.safetensors", "pytorch_model*.bin", "optimizer.pt"):
            for f in out_dir.rglob(pat):
                try:
                    f.unlink()
                except OSError:
                    pass
    return rc


def main():
    args = parse_args()
    spec = json.loads(Path(args.sweep_json).read_text(encoding="utf-8"))

    experiment = spec["experiment"]
    taus = spec["tau_sweep"]
    hw = float(spec.get("band_halfwidth", 0.5))
    prefix = spec.get("task_prefix", "MUSE_news_7b_erase")
    common = spec.get("common_overrides", {})
    method_over = spec["methods"][args.method]
    n_chunks = int(spec.get("n_chunks", 407))

    base = [f"{k}={_fmt(v)}" for k, v in {**common, **method_over}.items()]
    tag = f"_{args.tag}" if args.tag else ""

    # -------- lr x alpha convergence grid (fixed tau) --------
    if args.grid:
        probe_tau = float(spec.get("probe_tau", taus[0]))
        lr_sweep = spec.get("lr_sweep", [0.1])
        alpha_sweep = spec.get("alpha_sweep", [1.0])
        sched = spec.get("grid_scheduler", "constant_with_warmup")
        bl = round(probe_tau - hw, 4)
        bu = round(probe_tau + hw, 4)
        combos = [(lr, a) for lr in lr_sweep for a in alpha_sweep]
        print(f"[run_muse_sweep] GRID MODE: tau={probe_tau} "
              f"sched={sched} {len(combos)} (lr,alpha) runs.", flush=True)
        failures = []
        for lr, a in combos:
            task = (f"{prefix}{tag}_{args.method}_grid"
                    f"_lr{lr}_a{a}_tau{probe_tau}")
            extra = [
                f"trainer.args.learning_rate={lr}",
                # Flat LR (no cosine decay-to-zero, which starved late epochs in
                # the prior run). No warmup: it isn't required, and if a high-LR
                # cell diverges the grad_norm / weight_delta / NaN diagnostics
                # will flag it so we can add warmup with cause.
                f"trainer.args.lr_scheduler_type={sched}",
                f"trainer.method_args.alpha={a}",
                f"trainer.method_args.forget_loss_answer_target={probe_tau}",
                f"trainer.method_args.forget_loss_band_lower={bl}",
                f"trainer.method_args.forget_loss_band_upper={bu}",
                f"trainer.method_args.dynamic_stop_loss_threshold={bl}",
                f"trainer.method_args.dynamic_stop_log_upper={bu}",
            ]
            print(f"\n===== [{args.method}] GRID lr={lr} alpha={a} "
                  f"tau={probe_tau} =====", flush=True)
            rc = _run_one(task, experiment, args.config_name, base, extra,
                          hw, probe_tau, n_chunks, args.keep_weights)
            if rc != 0:
                print(f"[run_muse_sweep] lr={lr} alpha={a} FAILED rc={rc}; "
                      f"continuing.", flush=True)
                failures.append((lr, a))
        print(f"\n===== [{args.method}] grid done. failures={failures} =====",
              flush=True)
        if failures and len(failures) == len(combos):
            sys.exit(1)
        return

    # -------- full 3-D lr x alpha x tau search (shardable) --------
    if args.full_grid:
        sched = spec.get("fullgrid_scheduler", spec.get("grid_scheduler", "constant"))
        lr_grid = spec.get("lr_grid", spec.get("lr_sweep", [0.04, 0.06]))
        alpha_grid = spec.get("alpha_grid", spec.get("alpha_sweep", [2, 5, 7, 10]))
        tau_grid = spec.get("tau_grid", taus)
        # Shard selection: CLI flag wins, else fall back to the SLURM array env.
        shard = args.shard if args.shard else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        nshards = args.nshards if args.nshards else int(
            os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        nshards = max(1, nshards)
        cells = [(lr, a, tau)
                 for lr in lr_grid for a in alpha_grid for tau in tau_grid]
        mine = cells[shard::nshards]
        print(f"[run_muse_sweep] FULL-GRID: {len(cells)} cells "
              f"(lr={lr_grid} x alpha={alpha_grid} x tau={tau_grid}), sched={sched}; "
              f"shard {shard}/{nshards} -> {len(mine)} cells.", flush=True)
        failures = []
        for lr, a, tau in mine:
            bl = round(float(tau) - hw, 4)
            bu = round(float(tau) + hw, 4)
            task = (f"{prefix}{tag}_{args.method}_fg"
                    f"_lr{lr}_a{a}_tau{tau}")
            extra = [
                f"trainer.args.learning_rate={lr}",
                f"trainer.args.lr_scheduler_type={sched}",
                f"trainer.method_args.alpha={a}",
                f"trainer.method_args.forget_loss_answer_target={tau}",
                f"trainer.method_args.forget_loss_band_lower={bl}",
                f"trainer.method_args.forget_loss_band_upper={bu}",
                f"trainer.method_args.dynamic_stop_loss_threshold={bl}",
                f"trainer.method_args.dynamic_stop_log_upper={bu}",
            ]
            print(f"\n===== [{args.method}] FG lr={lr} alpha={a} tau={tau} "
                  f"band=[{bl},{bu}] =====", flush=True)
            rc = _run_one(task, experiment, args.config_name, base, extra,
                          hw, tau, n_chunks, args.keep_weights)
            if rc != 0:
                print(f"[run_muse_sweep] lr={lr} alpha={a} tau={tau} FAILED "
                      f"rc={rc}; continuing.", flush=True)
                failures.append((lr, a, tau))
        print(f"\n===== [{args.method}] full-grid shard {shard}/{nshards} done. "
              f"failures={failures} =====", flush=True)
        if failures and len(failures) == len(mine):
            sys.exit(1)
        return

    if args.smoke:
        taus = taus[:1]
        print("[run_muse_sweep] SMOKE MODE: first tau only, capped steps.", flush=True)

    failures = []
    for tau in taus:
        bl = round(float(tau) - hw, 4)
        bu = round(float(tau) + hw, 4)
        suffix = "_smoke" if args.smoke else ""
        tag = f"_{args.tag}" if args.tag else ""
        task = f"{prefix}{tag}_{args.method}_tau{tau}{suffix}"
        overrides = base + [
            f"task_name={task}",
            f"trainer.method_args.forget_loss_answer_target={tau}",
            f"trainer.method_args.forget_loss_band_lower={bl}",
            f"trainer.method_args.forget_loss_band_upper={bu}",
            f"trainer.method_args.dynamic_stop_loss_threshold={bl}",
            f"trainer.method_args.dynamic_stop_log_upper={bu}",
        ]
        if args.smoke:
            # Cap training to a handful of steps; keep the normal eval so the
            # MUSE eval + PrivLeak reference path is exercised too.
            overrides += [
                "trainer.args.max_steps=8",
                "trainer.args.num_train_epochs=1",
            ]
        cmd = [
            sys.executable, "src/train.py",
            f"--config-name={args.config_name}",
            f"experiment={experiment}",
        ] + overrides

        print(f"\n===== [{args.method}] tau={tau} band=[{bl},{bu}] =====", flush=True)
        print("RUN:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[run_muse_sweep] tau={tau} FAILED rc={rc}; continuing.", flush=True)
            failures.append(tau)
            continue

        out_dir = Path("saves/unlearn") / task

        # Monitoring: loss-vs-epoch curve + diagnostic stats. Best-effort; reads
        # trainer_state.json (JSON, not weights) so it must run before cleanup.
        try:
            subprocess.run([
                sys.executable, "scripts/erase/analyze_muse_run.py",
                "--run_dir", str(out_dir),
                "--target_tau", str(tau),
                "--band", str(hw),
                "--n_chunks", str(n_chunks),
            ], check=False)
        except Exception as e:  # noqa: BLE001
            print(f"[run_muse_sweep] analyze tau={tau} skipped ({e}).", flush=True)

        if not args.keep_weights:
            for pat in ("*.safetensors", "pytorch_model*.bin", "optimizer.pt"):
                for f in out_dir.rglob(pat):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    print(f"\n===== [{args.method}] sweep done. failures={failures} =====", flush=True)
    if failures and len(failures) == len(taus):
        sys.exit(1)


if __name__ == "__main__":
    main()
