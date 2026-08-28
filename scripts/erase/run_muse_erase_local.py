"""
Local ERASE (ERASE) runner for MUSE-News on the finetuned Llama-3.2-1B target.

This is the LOCAL / single-GPU (RTX 5080, 16 GB) analog of the cluster MUSE
sweep (scripts/erase/run_muse_sweep.py + configs/runs/MUSE_news_7b_erase). It is
structured like scripts/erase/run_exp_d_topk_vjp.py (an explicit RUNS list, a
train -> analyze loop, optional weight cleanup) but is MUSE/1B-specific:

  * experiment   = unlearn/muse/erase_1b   (target/oracle/cov paths baked in)
  * each run trains AND evaluates MUSE in-process (the experiment sets
    do_eval: true), writing MUSE_EVAL.json + MUSE_SUMMARY.json into the run dir;
    analyze_muse_run.py then writes the loss-vs-epoch curve from trainer_state.
  * we START WITH THE NER VARIANT (selective VJP over spaCy entity spans), as
    requested -- author_only_vjp + author_mask_mode=span + the 1B span cache.

tau calibration
---------------
The CET target tau should sit near the RETRAIN ORACLE's per-chunk CE (the
distribution ERASE matches). After prep_local.sh writes
saves/precompute/muse_news_1b/ce_{target,retrain}.json, this script prints the
target floor + oracle center and a suggested tau grid. Edit TAU_GRID below to
match, then launch for real.

Usage (from repo root, local Windows Python):
    python scripts/erase/run_muse_erase_local.py            # run the RUNS grid
    python scripts/erase/run_muse_erase_local.py --smoke    # 1 cell, few steps
    python scripts/erase/run_muse_erase_local.py --print-tau-only
    python scripts/erase/run_muse_erase_local.py --keep-weights
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train.py"
ANALYZE = ROOT / "scripts" / "erase" / "analyze_muse_run.py"
PYTHON = sys.executable

# --- experiment + on-disk prerequisites (all relative to repo root) ----------
EXPERIMENT = "unlearn/muse/erase_1b"
PREP_DIR = ROOT / "saves" / "precompute" / "muse_news_1b"
COV_DIR = PREP_DIR / "wikipedia_covariance"
SPANS_NER = PREP_DIR / "spans_ner.json"
CE_TARGET = PREP_DIR / "ce_target.json"
CE_RETRAIN = PREP_DIR / "ce_retrain.json"
RETAIN_LOGS = ROOT / "saves" / "eval" / "muse_news_1b_retain_eval" / "MUSE_EVAL.json"

TASK_PREFIX = "MUSE_news_1b_erase"
BAND_HALFWIDTH = 0.5      # band = [tau - hw, tau + hw]; dynstop uses the same
N_CHUNKS = 889            # forget-split raw chunks (for analyze_muse_run.py)

# --- sweep grid --------------------------------------------------------------
# Start with the NER variant. lr/alpha are seeded from where the 7B grid landed
# (lr 0.02 did ~nothing, 0.1 destroyed the model, alpha had a strong effect);
# the 1B is a different model so treat these as a starting point and refine.
# TAU_GRID should be centered on the oracle CE -- see _suggest_tau() / the
# printout at launch and adjust before the real sweep.
LR_GRID = [0.04, 0.06]
ALPHA_GRID = [2, 5]
# Oracle-centered (ce_retrain.json: mean per-chunk CE = 5.11, p10-p90 ~ 4.7-5.5;
# the memorized target floor is 0.12). ERASE steers forget CE up toward the
# oracle, so center tau on ~5.1; +/- the band half-width brackets the oracle
# spread. (The 1B sits higher than the 7B's ~2.8 center because the smaller
# model is intrinsically more perplexed by the unseen forget news text.)
TAU_GRID = [4.6, 5.1, 5.6]

EPOCHS = 100              # dynamic stopping ends most samples earlier
LR_SCHEDULER = "constant"  # the 7B grid showed cosine decayed lr to ~0 mid-run

# Dynamic per-sample stopping (mirrors the 7B erase defaults; band-driven knobs
# get the per-run tau band, the rest are fixed here).
DYNSTOP = {
    "dynamic_stop_max_active_steps_per_sample": 10,
    "dynamic_stop_longtail_threshold": 20,
    "dynamic_stop_done_sample_prob": 0.2,
    "dynamic_stop_decode_threshold": 3,
}

# NER selective-VJP knobs (the "ner" method block from the 7B sweep.json),
# pointed at the locally-built 1B span cache.
NER_OVERRIDES = {
    "trainer.method_args.author_only_vjp": "true",
    "trainer.method_args.author_mask_mode": "span",
    "trainer.method_args.answer_only_down_proj_grad": "false",
    "trainer.method_args.vjp_renormalize": "true",
    "trainer.method_args.forget_span_cache": SPANS_NER.as_posix(),
}

TARGET_LAYERS = [0, 1, 2, 3, 4]


def _fmt_layers(layers: list[int]) -> str:
    return "[" + ",".join(str(x) for x in layers) + "]"


def _read_ce(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _suggest_tau() -> None:
    """Print the target floor + oracle center and a suggested tau grid."""
    tgt = _read_ce(CE_TARGET)
    ret = _read_ce(CE_RETRAIN)
    print("\n=== tau calibration (per-chunk CE, nats/token) ===")
    if tgt:
        print(f"  TARGET  (memorized floor): mean={tgt['mean_per_chunk_ce']:.3f} "
              f"p10={tgt['p10']:.3f} median={tgt['median']:.3f} p90={tgt['p90']:.3f}")
    else:
        print(f"  TARGET  : {CE_TARGET} not found (run prep_local.sh).")
    if ret:
        print(f"  RETRAIN (oracle center)  : mean={ret['mean_per_chunk_ce']:.3f} "
              f"p10={ret['p10']:.3f} median={ret['median']:.3f} p90={ret['p90']:.3f}")
        c = ret["mean_per_chunk_ce"]
        grid = [round(c - 0.5, 2), round(c, 2), round(c + 0.5, 2), round(c + 1.0, 2)]
        print(f"  -> suggested TAU_GRID centered on oracle mean: {grid}")
    else:
        print(f"  RETRAIN : {CE_RETRAIN} not found (run prep_local.sh).")
    print(f"  (current TAU_GRID in this file: {TAU_GRID})\n", flush=True)


def _check_prereqs() -> None:
    missing = []
    if not COV_DIR.is_dir() or not any(COV_DIR.glob("C_retain_layer_*.pt")):
        missing.append(f"covariance dir {COV_DIR}")
    if not SPANS_NER.is_file():
        missing.append(f"NER span cache {SPANS_NER}")
    if not RETAIN_LOGS.is_file():
        missing.append(f"oracle eval logs {RETAIN_LOGS}")
    if missing:
        sys.exit("[run_muse_erase_local] missing prerequisites:\n  - "
                 + "\n  - ".join(missing)
                 + "\nRun configs/runs/MUSE_news_1b/prep_local.sh first.")


def _env() -> dict:
    """Local-friendly env: cap CPU threads (the box thrashed at full count),
    unbuffered output, eager attention is already set in the config."""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "6")
    env.setdefault("MKL_NUM_THREADS", "6")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def train_run(task: str, lr, alpha, tau, smoke: bool) -> int:
    bl = round(float(tau) - BAND_HALFWIDTH, 4)
    bu = round(float(tau) + BAND_HALFWIDTH, 4)
    overrides = [
        f"task_name={task}",
        f"trainer.args.learning_rate={lr}",
        f"trainer.args.lr_scheduler_type={LR_SCHEDULER}",
        f"trainer.method_args.alpha={alpha}",
        f"trainer.method_args.target_layers={_fmt_layers(TARGET_LAYERS)}",
        f"trainer.method_args.forget_loss_answer_target={tau}",
        f"trainer.method_args.forget_loss_band_lower={bl}",
        f"trainer.method_args.forget_loss_band_upper={bu}",
        f"trainer.method_args.dynamic_stop_loss_threshold={bl}",
        f"trainer.method_args.dynamic_stop_log_upper={bu}",
        f"trainer.args.num_train_epochs={EPOCHS}",
    ]
    overrides += [f"trainer.method_args.{k}={v}" for k, v in DYNSTOP.items()]
    overrides += [f"{k}={v}" for k, v in NER_OVERRIDES.items()]
    if smoke:
        # max_steps isn't in the unlearn trainer schema -> needs the + (append)
        # prefix; num_train_epochs IS in the schema so it's a plain override.
        overrides += ["+trainer.args.max_steps=8", "trainer.args.num_train_epochs=1"]
    cmd = [
        PYTHON, "-W", "ignore", str(TRAIN),
        "--config-name=unlearn.yaml",
        f"experiment={EXPERIMENT}",
    ] + overrides
    print(f"\n=== TRAIN {task}  (lr={lr} alpha={alpha} tau={tau} "
          f"band=[{bl},{bu}]) ===", flush=True)
    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), env=_env()).returncode


def analyze_run(task: str, tau) -> None:
    out_dir = ROOT / "saves" / "unlearn" / task
    try:
        subprocess.run([
            PYTHON, str(ANALYZE),
            "--run_dir", str(out_dir),
            "--target_tau", str(tau),
            "--band", str(BAND_HALFWIDTH),
            "--n_chunks", str(N_CHUNKS),
        ], cwd=str(ROOT), check=False)
    except Exception as e:  # noqa: BLE001
        print(f"[run_muse_erase_local] analyze {task} skipped ({e}).", flush=True)


def cleanup_weights(task: str) -> None:
    out_dir = ROOT / "saves" / "unlearn" / task
    for pat in ("*.safetensors", "pytorch_model*.bin", "optimizer.pt"):
        for f in out_dir.rglob(pat):
            try:
                f.unlink()
            except OSError:
                pass


def read_metrics(task: str) -> dict | None:
    sp = ROOT / "saves" / "unlearn" / task / "evals" / "MUSE_SUMMARY.json"
    if not sp.is_file():
        sp = ROOT / "saves" / "unlearn" / task / "MUSE_SUMMARY.json"
    if sp.is_file():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Run only the first (lr,alpha,tau) cell, capped to a few "
                        "steps + 1 epoch, with the normal MUSE eval. Catches "
                        "config/model/eval errors fast before the real sweep.")
    p.add_argument("--keep-weights", action="store_true",
                   help="Keep the *.safetensors after each run (default: delete "
                        "to save disk; eval JSON + curves are always kept).")
    p.add_argument("--print-tau-only", action="store_true",
                   help="Print the tau calibration suggestion and exit.")
    args = p.parse_args()

    _suggest_tau()
    if args.print_tau_only:
        return

    _check_prereqs()

    cells = [(lr, a, tau) for lr in LR_GRID for a in ALPHA_GRID for tau in TAU_GRID]
    if args.smoke:
        cells = cells[:1]
        print(f"[run_muse_erase_local] SMOKE MODE: 1 cell {cells}, capped steps.",
              flush=True)
    print(f"[run_muse_erase_local] NER variant, {len(cells)} cell(s): "
          f"lr={LR_GRID} x alpha={ALPHA_GRID} x tau={TAU_GRID}", flush=True)

    rows, failures = [], []
    for lr, a, tau in cells:
        suffix = "_smoke" if args.smoke else ""
        task = f"{TASK_PREFIX}_ner_lr{lr}_a{a}_tau{tau}{suffix}"
        rc = train_run(task, lr, a, tau, args.smoke)
        if rc != 0:
            print(f"[run_muse_erase_local] lr={lr} a={a} tau={tau} FAILED rc={rc}; "
                  f"continuing.", flush=True)
            failures.append((lr, a, tau))
            continue
        analyze_run(task, tau)
        rows.append({"task": task, "lr": lr, "alpha": a, "tau": tau,
                     "metrics": read_metrics(task)})
        if not args.keep_weights:
            cleanup_weights(task)

    out = ROOT / "results" / "muse_1b_erase_ner.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": rows, "failures": failures}, indent=2),
                   encoding="utf-8")
    print(f"\n===== NER sweep done. {len(rows)} ok, {len(failures)} failed. "
          f"Wrote {out} =====", flush=True)
    if failures and len(failures) == len(cells):
        sys.exit(1)


if __name__ == "__main__":
    main()
