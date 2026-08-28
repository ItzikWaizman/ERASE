"""Train + eval baseline unlearning methods on MUSE-News with hyperparameter sweeps.

MUSE sibling of run_baseline_sweep.py (which is TOFU-hardwired: forget10/retain90,
TOFU_SUMMARY, paper aggregates, TOFU AltPO data). This version drives the
MUSE-News 7B baselines (NPO / SimNPO / UNDIAL) for head-to-head comparison with
ERASE, reusing the MUSE unlearn experiment configs/experiment/unlearn/muse/baseline.yaml.

Per run (sequential, single GPU):
  1. train + eval in ONE src/train.py call (baseline.yaml sets do_eval=true with the
     full MUSE metric suite + the retrain reference for PrivLeak) -> writes
     saves/unlearn/<task>/MUSE_EVAL.json + MUSE_SUMMARY.json
  2. record MUSE_SUMMARY
  3. delete model weights (baselines aren't kept)

JSON format (configs/runs/MUSE_news_7b_<method>/set.json)
---------------------------------------------------------
{
  "model_tag":       "llama2_7b",
  "method":          "NPO",            # NPO | SimNPO | UNDIAL
  "experiment":      "unlearn/muse/baseline",
  "retain_logs":     "saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json",
  "train_batch_size": 1,
  "grad_accum":       32,
  "runs": [
    {"lr": 1e-5, "epochs": 10, "beta": 0.05, "alpha": 1.0, "gamma": 1.0},
    ...
  ]
}

Method-specific run keys (mapped to trainer.method_args.* -- see src/trainer/unlearn):
  NPO    : beta (NPO temperature), alpha (retain wt), gamma (forget wt)
  SimNPO : beta (temperature), delta (SimPO margin), alpha (retain wt), gamma (forget wt)
  UNDIAL : beta (logit-penalty strength), alpha (retain wt), gamma (forget wt)
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
SAVES = ROOT / "saves" / "unlearn"

# method -> the run-dict keys that map to trainer.method_args.*
METHOD_KEYS = {
    "NPO": ("beta", "alpha", "gamma", "retain_loss_type"),
    "SimNPO": ("beta", "delta", "alpha", "gamma", "retain_loss_type"),
    "UNDIAL": ("beta", "alpha", "gamma", "retain_loss_type"),
}


def _python() -> str:
    return os.environ.get("RESEARCH_UNLEARNING_PYTHON") or sys.executable


def _run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===\n  $ {' '.join(cmd)}\n", flush=True)
    env = os.environ.copy()
    # Single visible GPU so HF accelerate doesn't auto-wrap in DataParallel
    # (DP master GPU OOMs on 7B + ref_model during backward when N_GPUs>1).
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def _clear_cuda():
    snippet = (
        "import gc, torch; gc.collect();"
        " torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    )
    subprocess.run([_python(), "-c", snippet], cwd=str(ROOT), capture_output=True)


def build_task_name(method: str, model_tag: str, run: dict) -> str:
    """`{model_tag}_{method}/{swept-hparams}` -- one group dir per (model, method)."""
    group = f"muse_news_{model_tag}_{method}"
    parts = [f"lr{run['lr']:g}", f"ep{run['epochs']}"]
    for key in sorted(run.keys()):
        if key in ("lr", "epochs"):
            continue
        v = run[key]
        parts.append(f"{key}{v:g}" if isinstance(v, float) else f"{key}{v}")
    return f"{group}/{'_'.join(parts)}"


def method_overrides(method: str, run: dict) -> list[str]:
    ov = [
        f"trainer.args.learning_rate={run['lr']}",
        f"trainer.args.num_train_epochs={run['epochs']}",
    ]
    for key in METHOD_KEYS.get(method, ()):
        if key in run:
            ov.append(f"trainer.method_args.{key}={run[key]}")
    return ov


def train_eval_one(task: str, cfg: dict, run: dict) -> bool:
    method = cfg["method"]
    retain_logs = ROOT / cfg["retain_logs"]
    cmd = [
        _python(), "-W", "ignore", str(TRAIN),
        "--config-name=unlearn.yaml",
        f"experiment={cfg['experiment']}",
        f"trainer={method}",
        f"task_name={task}",
        f"retain_logs_path={retain_logs.as_posix()}",
        f"trainer.args.per_device_train_batch_size={cfg.get('train_batch_size', 1)}",
        f"trainer.args.gradient_accumulation_steps={cfg.get('grad_accum', 32)}",
    ] + method_overrides(method, run)
    return _run(cmd, f"TRAIN+EVAL {task}") == 0


def read_summary(task: str) -> dict | None:
    for p in (SAVES / task / "MUSE_SUMMARY.json",
              SAVES / task / "evals" / "MUSE_SUMMARY.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return None


def delete_weights(task: str) -> None:
    task_dir = SAVES / task
    files = list(task_dir.glob("*.safetensors")) + list(task_dir.glob("pytorch_model*.bin"))
    if not files:
        return
    gb = sum(p.stat().st_size for p in files) / (1024 ** 3)
    for p in files:
        try:
            p.unlink()
        except OSError as e:
            print(f"   could not delete {p}: {e}", flush=True)
    print(f"=== DELETE weights {task} ({len(files)} files, {gb:.2f} GB) ===", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, help="JSON sweep config file")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "muse_baseline_sweeps")
    p.add_argument("--keep-weights", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    runs = cfg["runs"]
    method = cfg["method"]
    model_tag = cfg["model_tag"]
    if method not in METHOD_KEYS:
        sys.exit(f"unknown method {method!r}; allowed: {sorted(METHOD_KEYS)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"muse_news_{model_tag}_{method}_sweep.json"

    print(f"=== MUSE-News baseline sweep: {method} on {model_tag} "
          f"({len(runs)} runs) ===", flush=True)
    rows: list[dict] = []
    _clear_cuda()

    for i, run in enumerate(runs):
        task = build_task_name(method, model_tag, run)
        cached = SAVES / task / "MUSE_SUMMARY.json"
        cached_evals = SAVES / task / "evals" / "MUSE_SUMMARY.json"
        row_base = {"run_index": i, "task": task, "run": run}

        if (cached.is_file() or cached_evals.is_file()) and not args.force:
            print(f"[{i+1}/{len(runs)}] skip {task} (cached)", flush=True)
            rows.append({**row_base, "metrics": read_summary(task), "skipped": True})
            summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            continue

        print(f"\n[{i+1}/{len(runs)}] {task}", flush=True)
        if not train_eval_one(task, cfg, run):
            rows.append({**row_base, "error": "train_eval_failed"})
            _clear_cuda()
            summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            continue
        _clear_cuda()

        rows.append({**row_base, "metrics": read_summary(task)})
        if not args.keep_weights:
            delete_weights(task)
        summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
