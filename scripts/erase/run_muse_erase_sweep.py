"""General MUSE-News ERASE sweep runner with explicit per-job run-lists.

Unlike run_muse_sweep.py (which only sweeps lr x alpha x tau via --full-grid),
this runner reads a job JSON whose "runs" list can vary ANY axis (top-k vs NER,
per-token vs mean CET, alpha, clipping, retain weight, layers, dynamic-stop,
...). Every non-swept knob falls back to a single shared BASELINE so each job
isolates exactly one variable.

Per run (sequential, single GPU):
  1. train + eval (experiment=unlearn/muse/erase sets do_eval + the full MUSE suite)
  2. analyze_muse_run.py (loss-vs-epoch curve + diagnostics)
  3. per-token forget-CE telemetry: dump 5 forget chunks and plot vs the oracle
     (results/token_ce/oracle.json, precomputed once) -- shows per-token mean,
     variance and oracle-similarity, as requested.
  4. delete weights (unless --keep-weights)

Job JSON format (configs/runs/MUSE_news_7b_erase_<knob>/set.json):
  {
    "task_prefix": "MUSE_news_7b_erase_alpha",
    "base": { ...optional per-job overrides of BASELINE... },
    "runs": [ {"alpha": 2, "tau": 2.0}, {"alpha": 5, "tau": 2.5}, ... ]
  }

Run-dict keys (all optional; missing -> BASELINE):
  lr, epochs, scheduler, max_grad_norm, alpha, retain_weight, answer_mode,
  layers (list), max_active_steps, tau, method ("ner"|"topk"|"raw"), topk (count)

Usage:
    python scripts/erase/run_muse_erase_sweep.py configs/runs/MUSE_news_7b_erase_alpha/set.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train.py"
ANALYZE = ROOT / "scripts" / "erase" / "analyze_muse_run.py"
DUMP = ROOT / "scripts" / "analysis" / "dump_forget_token_ce.py"
PLOT = ROOT / "scripts" / "analysis" / "plot_forget_token_ce.py"
PY = sys.executable

EXPERIMENT = "unlearn/muse/erase"
TOKENIZER = "NousResearch/Llama-2-7b-hf"
SPANS_NER = "saves/precompute/muse_news_7b/spans_ner.json"
ORACLE_TOKEN_CE = ROOT / "results" / "token_ce" / "oracle.json"
# Band half-width around tau. The retrain oracle's per-chunk forget CE is
# 1.875 (p10-p90 = 1.69-2.06; measure_muse_ce on MUSE-News_retrain), so a
# +/-0.25 band [tau-0.25, tau+0.25] brackets the oracle spread and the
# dynamic-stop threshold (tau-hw) lands chunks near the oracle, not far below.
BAND_HW = 0.25
N_CHUNKS = 407

# Shared baseline B: lr 0.05, tau 1.9 (~ oracle CE 1.875), alpha 3, 35 epochs,
# cosine, NER VJP, mean CET, no clip, retain off, layers 0-4, default
# dynamic-stop. Each job overrides exactly one axis (via its set.json runs/base).
BASELINE = {
    "lr": 0.05,
    "epochs": 40,
    "scheduler": "cosine",
    "max_grad_norm": 0.0,
    "alpha": 3,
    "retain_weight": 0.0,
    "answer_mode": "mse",
    "layers": [0, 1, 2, 3, 4],
    "max_active_steps": 10,
    "tau": 1.9,
    "method": "ner",
    "retain_ce_stop_threshold": 0.0,
    "retain_ce_monitor_interval": 25,
    # --- signal monitor + automatic stop (deliverable early-stop) ---
    "monitor_interval": 0,          # >0 enables the dual-probe monitor + JSONL
    "probe_size": 4,
    "stop_forget_ce": 0.0,          # auto-stop when forget probe CE >= this
    "stop_forget_ce_auto": False,   # True -> stop_forget_ce = tau (per run)
    "stop_retain_rise": 0.0,        # collapse guard: retain CE rise (nats) above baseline
    "stop_warmup": 0,
    "stop_patience": 2,
    # checkpoint-only mode: stop rules save checkpoint-{step}-earlystop-{kind}
    # + eval and CONTINUE training (signal validation without censoring).
    "stop_ckpt_only": False,
    # deep-collapse bail-out (REAL stop even in ckpt-only mode). 0 = off.
    "hard_stop_retain_rise": 0.0,
    # long-tail patience width (remaining-samples threshold). 0 = keep default.
    "longtail_threshold": 0,
    # --- dense step eval (telemetry only; 0 -> use EVAL_AT_EPOCHS) ---
    "eval_every_steps": 0,
    "eval_every_steps_warmup": 0,
    # --- full-forget-set per-token CE dump every N epochs (0 = off) ---
    "token_ce_dump_epochs": 0.0,
    "token_ce_dump_max_samples": 0,
    # --- std-narrowing / loss-shape knobs ---
    "target_std": 0.0,              # forget_loss_answer_target_std (mse jitter)
    "target_onesided": False,       # right-skewed jitter
    "token_ce_ceiling": 0.0,        # exclude tokens already above this CE
    "token_onesided": False,        # mse_token: penalise only tokens below tau
}

EVAL_AT_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40]


def method_overrides(run: dict) -> list[str]:
    m = run.get("method", "ner")
    if m == "ner":
        return [
            "trainer.method_args.author_only_vjp=true",
            "trainer.method_args.author_mask_mode=span",
            "trainer.method_args.answer_only_down_proj_grad=false",
            "trainer.method_args.vjp_renormalize=true",
            "trainer.method_args.topk_vjp_count=0",
            f"trainer.method_args.forget_span_cache={SPANS_NER}",
        ]
    if m == "topk":
        # top-K highest-norm VJP positions; K is an absolute token count.
        # MUSE chunks are ~2048 supervised tokens, so K=102/205/410 ~= 5/10/20%.
        return [
            "trainer.method_args.author_only_vjp=false",
            "trainer.method_args.author_mask_mode=off",
            "trainer.method_args.answer_only_down_proj_grad=false",
            "trainer.method_args.vjp_renormalize=true",
            f"trainer.method_args.topk_vjp_count={int(run.get('topk', 205))}",
        ]
    if m == "raw":
        return [
            "trainer.method_args.author_only_vjp=false",
            "trainer.method_args.author_mask_mode=off",
            "trainer.method_args.answer_only_down_proj_grad=true",
            "trainer.method_args.topk_vjp_count=0",
        ]
    raise SystemExit(f"unknown method {m!r}")


def run_overrides(run: dict) -> list[str]:
    ov = [
        f"trainer.args.learning_rate={run['lr']}",
        f"trainer.args.num_train_epochs={run['epochs']}",
        f"trainer.args.lr_scheduler_type={run['scheduler']}",
        f"trainer.method_args.alpha={run['alpha']}",
        f"trainer.method_args.forget_loss_answer_mode={run['answer_mode']}",
        f"trainer.method_args.target_layers=[{','.join(str(x) for x in run['layers'])}]",
        f"trainer.method_args.dynamic_stop_max_active_steps_per_sample={run['max_active_steps']}",
    ]
    if float(run.get("max_grad_norm", 0.0)) > 0:
        ov.append(f"trainer.args.max_grad_norm={run['max_grad_norm']}")
    # Retain-CE early-stop gate
    rcs = float(run.get("retain_ce_stop_threshold", 0.0))
    if rcs > 0:
        ov.append(f"trainer.method_args.retain_ce_stop_threshold={rcs}")
        ov.append(f"trainer.method_args.retain_ce_monitor_interval="
                  f"{int(run.get('retain_ce_monitor_interval', 25))}")
    tau = float(run["tau"])
    bl = round(tau - BAND_HW, 4)
    bu = round(tau + BAND_HW, 4)
    ov += [
        f"trainer.method_args.forget_loss_answer_target={tau}",
        f"trainer.method_args.forget_loss_band_lower={bl}",
        f"trainer.method_args.forget_loss_band_upper={bu}",
        f"trainer.method_args.dynamic_stop_loss_threshold={bl}",
        f"trainer.method_args.dynamic_stop_log_upper={bu}",
    ]
    # --- std-narrowing / loss-shape knobs ---
    tstd = float(run.get("target_std", 0.0))
    if tstd > 0:
        ov.append(f"trainer.method_args.forget_loss_answer_target_std={tstd}")
    tcc = float(run.get("token_ce_ceiling", 0.0))
    if tcc > 0:
        ov.append(f"trainer.method_args.forget_loss_token_ce_ceiling={tcc}")
    # --- signal monitor + automatic stop ---
    mon = int(run.get("monitor_interval", 0))
    if mon > 0:
        ov += [
            f"trainer.method_args.signal_monitor_interval={mon}",
            f"trainer.method_args.signal_monitor_probe_size={int(run.get('probe_size', 4))}",
            f"trainer.method_args.stop_monitor_warmup_steps={int(run.get('stop_warmup', 0))}",
            f"trainer.method_args.stop_patience={int(run.get('stop_patience', 2))}",
        ]
        sf = float(run.get("stop_forget_ce", 0.0))
        if sf <= 0 and bool(run.get("stop_forget_ce_auto", False)):
            sf = tau  # natural-level stop: forget probe CE reaches the CET target
        if sf > 0:
            ov.append(f"trainer.method_args.stop_forget_ce_target={sf}")
        sr = float(run.get("stop_retain_rise", 0.0))
        if sr > 0:
            ov.append(f"trainer.method_args.stop_retain_ce_rise={sr}")
        if bool(run.get("stop_ckpt_only", False)):
            ov.append("trainer.method_args.stop_signal_checkpoint_only=true")
        hs = float(run.get("hard_stop_retain_rise", 0.0))
        if hs > 0:
            ov.append(f"trainer.method_args.hard_stop_retain_ce_rise={hs}")
    lt = int(run.get("longtail_threshold", 0))
    if lt > 0:
        ov.append(f"trainer.method_args.dynamic_stop_longtail_threshold={lt}")
    # --- full-forget-set per-token CE dump (offline stop-signal mining) ---
    tcd = float(run.get("token_ce_dump_epochs", 0.0))
    if tcd > 0:
        ov.append(f"trainer.method_args.token_ce_dump_epochs={tcd}")
        tcm = int(run.get("token_ce_dump_max_samples", 0))
        if tcm > 0:
            ov.append(f"trainer.method_args.token_ce_dump_max_samples={tcm}")
    ov += method_overrides(run)
    # Mid-training eval cadence. Default: MUSE eval every K epochs. When
    # eval_every_steps > 0, switch to dense STEP-based eval (telemetry to
    # validate the auto-stop signal); the reported model is still the
    # auto-stopped final model.
    evs = int(run.get("eval_every_steps", 0))
    if evs > 0:
        ov.append(f"+trainer.args.eval_every_steps={evs}")
        ov.append(
            f"+trainer.args.eval_every_steps_warmup="
            f"{int(run.get('eval_every_steps_warmup', 0))}"
        )
    else:
        ep_list = ",".join(str(e) for e in EVAL_AT_EPOCHS)
        ov.append(f"+trainer.args.eval_at_epochs=[{ep_list}]")
    return ov


def task_suffix(run: dict, base: dict) -> str:
    """Encode only the keys that differ from BASELINE+base into the task name."""
    ref = {**BASELINE, **base}
    parts = []
    for k in ("method", "topk", "answer_mode", "alpha", "tau", "lr", "epochs",
              "max_grad_norm", "retain_weight", "max_active_steps", "scheduler",
              "retain_ce_stop_threshold", "monitor_interval", "stop_forget_ce",
              "stop_retain_rise", "target_std", "target_onesided",
              "token_ce_ceiling", "token_onesided", "eval_every_steps"):
        if k in run and run[k] != ref.get(k):
            v = run[k]
            parts.append(f"{k}{v}")
    if "layers" in run and run["layers"] != ref.get("layers"):
        parts.append("L" + "".join(str(x) for x in run["layers"]))
    return "_".join(parts) if parts else "baseline"


def select_best_checkpoint(task: str) -> dict | None:
    """TELEMETRY ONLY -- NOT the reported result. Scans
    checkpoint-*/evals/MUSE_SUMMARY.json and reports the best by
    retain_knowmem * (1 - forget_verbmem) so we can validate, offline, that the
    automatic stop (signal_monitor) landed at the sweet spot. The REPORTED model
    is always the auto-stopped final model (no post-hoc checkpoint picking), so
    the comparison to SimNPO/NPO/UNDIAL is apples-to-apples."""
    out_dir = ROOT / "saves" / "unlearn" / task
    best, best_score, best_path = None, -1.0, None
    for cp in sorted(out_dir.glob("checkpoint-*/evals/MUSE_SUMMARY.json")):
        try:
            d = json.loads(cp.read_text(encoding="utf-8"))
            rk = d.get("retain_knowmem_ROUGE", 0)
            fv = d.get("forget_verbmem_ROUGE", 1)
            score = rk * (1 - fv)
            if score > best_score:
                best_score, best, best_path = score, d, cp
        except (json.JSONDecodeError, OSError):
            continue
    # Also check the final (end-of-training) eval
    for p in [out_dir / "MUSE_SUMMARY.json", out_dir / "evals" / "MUSE_SUMMARY.json"]:
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                rk = d.get("retain_knowmem_ROUGE", 0)
                fv = d.get("forget_verbmem_ROUGE", 1)
                score = rk * (1 - fv)
                if score > best_score:
                    best_score, best, best_path = score, d, p
            except (json.JSONDecodeError, OSError):
                pass
    if best is not None:
        best["_best_checkpoint"] = str(best_path)
        best["_best_score"] = round(best_score, 4)
        # Write best summary to a known location for easy comparison
        dest = out_dir / "BEST_CHECKPOINT.json"
        dest.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"[best_checkpoint] {task}: score={best_score:.4f} from {best_path.parent.name}",
              flush=True)
    return best


def token_ce_telemetry(task: str) -> None:
    out_dir = ROOT / "saves" / "unlearn" / task / "evals"
    dump = out_dir / "token_ce.json"
    try:
        rc = subprocess.run([
            PY, str(DUMP),
            "--model_name", str(ROOT / "saves" / "unlearn" / task),
            "--tokenizer_name", TOKENIZER,
            "--n_chunks", "5", "--max_length", "2048",
            "--output", str(dump),
        ], cwd=str(ROOT)).returncode
        if rc != 0 or not dump.is_file():
            print(f"[token_ce] dump failed for {task}", flush=True)
            return
        plot_args = [PY, str(PLOT), "--dump", f"run={dump}",
                     "--out-prefix", str(out_dir / "forget_token_ce"),
                     "--detail-chunks", "0", "1", "2"]
        if ORACLE_TOKEN_CE.is_file():
            plot_args += ["--dump", f"oracle={ORACLE_TOKEN_CE}"]
        subprocess.run(plot_args, cwd=str(ROOT))
    except Exception as e:  # noqa: BLE001
        print(f"[token_ce] telemetry skipped for {task} ({e})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_json")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--no-token-ce", action="store_true",
                    help="Skip the per-token CE telemetry dump+plot.")
    ap.add_argument("--index", type=int, default=-1,
                    help="Run only the run at this index of the job's run list "
                         "(SLURM array task). -1 = run all sequentially.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the train commands without executing (smoke).")
    args = ap.parse_args()

    spec = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    prefix = spec["task_prefix"]
    base = spec.get("base", {})
    runs = spec["runs"]
    if args.index >= 0:
        if args.index >= len(runs):
            print(f"[run_muse_erase_sweep] index {args.index} out of range "
                  f"(0..{len(runs)-1}); nothing to do.", flush=True)
            return 0
        runs = [runs[args.index]]
    retain_logs = spec.get("retain_logs_path",
                           "saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json")

    print(f"=== MUSE ERASE sweep: {prefix} ({len(runs)} runs) ===", flush=True)
    failures = []
    for i, r in enumerate(runs):
        run = {**BASELINE, **base, **r}
        task = f"{prefix}_{task_suffix(r, base)}"
        out_dir = ROOT / "saves" / "unlearn" / task
        cmd = [
            PY, "-W", "ignore", str(TRAIN),
            "--config-name=unlearn.yaml",
            f"experiment={EXPERIMENT}",
            f"task_name={task}",
            f"retain_logs_path={retain_logs}",
        ] + run_overrides(run)
        print(f"\n[{i+1}/{len(runs)}] {task}\nRUN: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"[run_muse_erase_sweep] {task} FAILED rc={rc}; continuing.", flush=True)
            failures.append(task)
            continue
        try:
            subprocess.run([
                PY, str(ANALYZE), "--run_dir", str(out_dir),
                "--target_tau", str(run["tau"]), "--band", str(BAND_HW),
                "--n_chunks", str(N_CHUNKS),
            ], cwd=str(ROOT), check=False)
        except Exception as e:  # noqa: BLE001
            print(f"[analyze] skipped {task} ({e})", flush=True)
        select_best_checkpoint(task)
        if not args.no_token_ce:
            token_ce_telemetry(task)
        if not args.keep_weights:
            for pat in ("*.safetensors", "pytorch_model*.bin", "optimizer.pt"):
                for f in out_dir.rglob(pat):
                    try:
                        f.unlink()
                    except OSError:
                        pass
    print(f"\n=== {prefix} done. failures={failures} ===", flush=True)
    if failures and len(failures) == len(runs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
