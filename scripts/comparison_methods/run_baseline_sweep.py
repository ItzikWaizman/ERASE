"""Train + eval baseline unlearning methods with hyperparameter sweeps.

Designed for Slurm: reads a JSON config, runs each configuration
sequentially on the single GPU it sees. Pipeline per run:
  1. train (src/train.py)
  2. eval  (src/eval.py)  -- paper-grade TOFU
  3. compute_paper_aggregates
  4. delete model weights (always -- baselines don't get highlighted)

JSON format
-----------
{
  "model_tag":       "llama2_7b",
  "experiment":      "unlearn/tofu/llama2_7b/default",
  "eval_experiment": "eval/tofu/llama2_7b/paper",
  "retain_logs":     "saves/eval/llama2_7b/retain_oracle/TOFU_EVAL.json",
  "init_ref_dir":    "saves/eval/llama2_7b/init_finetuned",
  "retain_ref_dir":  "saves/eval/llama2_7b/retain_oracle",
  "eval_batch_size":  4,
  "train_batch_size": 4,
  "grad_accum":       2,
  "method":          "SimNPO",
  "forget_split":    "forget10",   # optional, default forget10
  "retain_split":    "retain90",   # optional, default retain90
  "holdout_split":   "holdout10",  # optional, default holdout10
  "runs": [
    {"lr": 2e-5, "epochs": 10, "beta": 4.5, "delta": 0, "gamma": 0.125},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train.py"
EVAL = ROOT / "src" / "eval.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "analysis" / "compute_paper_aggregates.py"
SAVES = ROOT / "saves" / "unlearn"

_NO_BUILTIN_EVAL = [
    "trainer.args.eval_strategy=no",
    "trainer.args.do_eval=False",
    "trainer.args.eval_on_start=False",
]


def _python() -> str:
    return os.environ.get("RESEARCH_UNLEARNING_PYTHON") or sys.executable


def _run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===\n  $ {' '.join(cmd)}\n", flush=True)
    env = os.environ.copy()
    # Force a single visible GPU so HF accelerate doesn't auto-wrap models in
    # DataParallel (DP master GPU OOMs on 7B during backward when N_GPUs>1).
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def _clear_cuda():
    snippet = (
        "import gc, torch; gc.collect();"
        " torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    )
    subprocess.run([_python(), "-c", snippet], cwd=str(ROOT), capture_output=True)


def build_task_name(cfg: dict, method: str, model_tag: str, run: dict) -> str:
    """Build a path-like task identifier of the form ``{group}/{subname}``.

    Runs sharing the same ``(model_tag, method)`` pair are collected under a
    shared ``{model_tag}_{method}`` group dir, so ``saves/unlearn`` doesn't
    become a flat soup of cross-method run folders. The ``subname`` portion
    encodes only the swept hyperparameters. Hydra resolves
    ``saves/${mode}/${task_name}`` from ``paths/default.yaml`` and is happy
    with the embedded ``/``.

    Skips ``trainable_params_regex`` because the layer info is already encoded
    in ``module_regex`` and the regex characters (parens, pipes, backslashes)
    yield ugly + non-portable directory names.
    """
    split = cfg.get("forget_split", "forget10")
    group = f"{model_tag}_{method}" if split == "forget10" else f"{model_tag}_{method}_{split}"
    parts = [f"lr{run['lr']}", f"ep{run['epochs']}"]
    for key in sorted(run.keys()):
        if key in ("lr", "epochs", "trainable_params_regex"):
            continue
        v = run[key]
        short = key.replace("steering_coeff", "sc").replace("module_regex", "layer")
        if isinstance(v, float):
            parts.append(f"{short}{v:g}")
        elif isinstance(v, list):
            parts.append(f"{short}list{len(v)}")
        else:
            parts.append(f"{short}{v}")
    subname = "_".join(parts)
    return f"{group}/{subname}"


def build_method_overrides(method: str, run: dict) -> list[str]:
    """Convert run dict to Hydra trainer overrides."""
    overrides = [
        f"trainer.args.learning_rate={run['lr']}",
        f"trainer.args.num_train_epochs={run['epochs']}",
    ]
    if "weight_decay" in run:
        overrides.append(f"trainer.args.weight_decay={run['weight_decay']}")
    if "warmup_epochs" in run:
        overrides.append(f"trainer.args.warmup_epochs={run['warmup_epochs']}")
    if "seed" in run:
        # Multi-seed significance runs (R2): trainer default is seed 0.
        overrides.append(f"trainer.args.seed={run['seed']}")

    # AltPO uses DPO trainer internally
    trainer_method = "DPO" if method == "AltPO" else method
    method_keys = {
        "SimNPO": ("beta", "delta", "alpha", "gamma", "retain_loss_type"),
        "NPO":    ("beta", "alpha", "gamma", "retain_loss_type"),
        "RMU":    ("gamma", "steering_coeff", "alpha", "module_regex",
                   "trainable_params_regex", "retain_loss_type"),
        "GradAscent": (),
        "GradDiff": ("gamma", "alpha", "retain_loss_type"),
        "UNDIAL":   ("gamma", "alpha", "beta", "retain_loss_type"),
        "DPO":    ("beta", "alpha", "gamma", "retain_loss_type"),
        "AltPO":  ("beta", "alpha", "gamma", "retain_loss_type"),
    }
    for key in method_keys.get(method, ()):
        if key in run:
            val = run[key]
            if key == "module_regex":
                overrides.append(f"trainer.method_args.module_regex={val}")
            elif key == "trainable_params_regex":
                # Hydra list-of-strings override; accept either a list or a
                # single regex string. The trainer expects a list. Each item
                # is wrapped in single quotes because raw regex contains
                # parentheses and pipes that the OmegaConf override grammar
                # treats as special tokens.
                if isinstance(val, str):
                    val = [val]
                joined = ",".join(f"'{item}'" for item in val)
                overrides.append(
                    f"trainer.method_args.trainable_params_regex=[{joined}]"
                )
            else:
                overrides.append(f"trainer.method_args.{key}={val}")
    return overrides


def _altpo_data_overrides(cfg: dict) -> list[str]:
    """Return Hydra overrides for AltPO alternate-response data loading."""
    data_file = cfg["altpo_data_file"]
    return [
        "data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset",
        "~data.forget.TOFU_QA_forget.args.hf_args.name",
        "data.forget.TOFU_QA_forget.args.hf_args.path=json",
        f"+data.forget.TOFU_QA_forget.args.hf_args.data_files={data_file}",
        "data.forget.TOFU_QA_forget.args.hf_args.split=train",
        "+data.forget.TOFU_QA_forget.args.alternate_key=alternate",
        "+data.forget.TOFU_QA_forget.args.return_original=True",
    ]


def train_one(task: str, cfg: dict, run: dict) -> bool:
    method = cfg["method"]
    trainer = "DPO" if method == "AltPO" else method
    retain_logs = ROOT / cfg["retain_logs"]
    cmd = [
        _python(), "-W", "ignore", str(TRAIN),
        "--config-name=unlearn.yaml",
        f"experiment={cfg['experiment']}",
        f"trainer={trainer}",
        f"task_name={task}",
        f"forget_split={cfg.get('forget_split', 'forget10')}",
        f"retain_split={cfg.get('retain_split', 'retain90')}",
        f"holdout_split={cfg.get('holdout_split', 'holdout10')}",
        f"retain_logs_path={retain_logs.as_posix()}",
        f"trainer.args.per_device_train_batch_size={cfg.get('train_batch_size', 4)}",
        f"trainer.args.gradient_accumulation_steps={cfg.get('grad_accum', 2)}",
        "model.model_args.attn_implementation=eager",
        f"trainer.args.gradient_checkpointing={cfg.get('gradient_checkpointing', True)}",
        # Default trainer optim is paged_adamw_32bit which tries to pin ~84GB of
        # host RAM for fp32 states. Slurm jobs typically request <128GB and this
        # falls back to allocating states on the GPU, OOMing 7B+ref_model setups.
        # adamw_bnb_8bit keeps 8-bit states on GPU (~14GB) with zero host pinning.
        # Small models (1B) can override via cfg "optim" (e.g. adamw_torch for
        # exact fp32 AdamW states, matching released-checkpoint training).
        f"trainer.args.optim={cfg.get('optim', 'adamw_bnb_8bit')}",
    ]
    cmd += build_method_overrides(method, run) + _NO_BUILTIN_EVAL
    if method == "AltPO":
        cmd += _altpo_data_overrides(cfg)
    return _run(cmd, f"TRAIN {task}") == 0


def eval_one(task: str, cfg: dict) -> bool:
    task_dir = SAVES / task
    out = task_dir / "evals"
    retain_logs = ROOT / cfg["retain_logs"]
    cmd = [
        _python(), "-W", "ignore", str(EVAL),
        "--config-name=eval.yaml",
        f"experiment={cfg['eval_experiment']}",
        f"task_name={task}_eval",
        f"forget_split={cfg.get('forget_split', 'forget10')}",
        f"holdout_split={cfg.get('holdout_split', 'holdout10')}",
        f"retain_logs_path={retain_logs.as_posix()}",
        f"model.model_args.pretrained_model_name_or_path={task_dir.as_posix()}",
        "model.model_args.attn_implementation=eager",
        f"paths.output_dir={out.as_posix()}",
        f"eval.tofu.batch_size={cfg.get('eval_batch_size', 4)}",
    ]
    return _run(cmd, f"EVAL {task}") == 0


def aggregate_one(task: str, cfg: dict) -> dict | None:
    task_dir = SAVES / task
    init_ref = ROOT / cfg["init_ref_dir"]
    retain_ref = ROOT / cfg["retain_ref_dir"]
    cmd = [_python(), str(AGGREGATE_SCRIPT), "--dir", str(task_dir)]
    if (init_ref / "TOFU_EVAL.json").is_file():
        cmd += ["--init-eval", str(init_ref)]
    if (retain_ref / "TOFU_EVAL.json").is_file():
        cmd += ["--retain-eval", str(retain_ref)]
    _run(cmd, f"AGGREGATE {task}")
    pa = task_dir / "evals" / "PAPER_AGGREGATES.json"
    if not pa.is_file():
        return None
    try:
        return json.loads(pa.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def delete_weights(task: str) -> None:
    task_dir = SAVES / task
    weight_files = list(task_dir.glob("*.safetensors"))
    if not weight_files:
        return
    total = sum(p.stat().st_size for p in weight_files)
    for p in weight_files:
        try:
            p.unlink()
        except OSError as e:
            print(f"   could not delete {p}: {e}", flush=True)
    gb = total / (1024 ** 3)
    print(f"=== DELETE weights {task} ({len(weight_files)} files, {gb:.2f} GB) ===",
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, help="JSON sweep config file")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "baseline_sweeps")
    p.add_argument("--force", action="store_true")
    p.add_argument("--index", type=int, default=-1,
                    help="Run only the run at this index of the config's run "
                         "list (SLURM array task: pass $SLURM_ARRAY_TASK_ID). "
                         "-1 = run all sequentially (default).")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    all_runs = cfg["runs"]
    method = cfg["method"]
    model_tag = cfg["model_tag"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.index >= 0:
        if args.index >= len(all_runs):
            print(f"ERROR: --index {args.index} out of range (0..{len(all_runs)-1})",
                  file=sys.stderr, flush=True)
            return 1
        runs = [all_runs[args.index]]
        start_idx = args.index
        # Array-task-unique summary file so concurrent tasks never race on
        # the same JSON (each array task owns exactly one file).
        summary_path = args.output_dir / f"{model_tag}_{method}_sweep_idx{args.index}.json"
    else:
        runs = all_runs
        start_idx = 0
        summary_path = args.output_dir / f"{model_tag}_{method}_sweep.json"

    print(f"=== Baseline sweep: {method} on {model_tag} ({len(runs)}/{len(all_runs)} runs"
          f"{f', index={args.index}' if args.index >= 0 else ''}) ===", flush=True)
    rows: list[dict] = []
    _clear_cuda()

    for offset, run in enumerate(runs):
        i = start_idx + offset
        task = build_task_name(cfg, method, model_tag, run)
        cached = SAVES / task / "evals" / "TOFU_SUMMARY.json"
        row_base = {"run_index": i, "task": task, "run": run}

        if cached.is_file() and not args.force:
            print(f"[{i+1}/{len(runs)}] skip {task} (cached)", flush=True)
            metrics = json.loads(cached.read_text(encoding="utf-8"))
            rows.append({**row_base, "metrics": metrics, "skipped": True})
            summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            continue

        print(f"\n[{i+1}/{len(runs)}] {task}", flush=True)

        if not train_one(task, cfg, run):
            rows.append({**row_base, "error": "train_failed"})
            _clear_cuda()
            summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            continue
        _clear_cuda()

        if not eval_one(task, cfg):
            rows.append({**row_base, "error": "eval_failed"})
            _clear_cuda()
            delete_weights(task)
            summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            continue
        _clear_cuda()

        sp = SAVES / task / "evals" / "TOFU_SUMMARY.json"
        metrics = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
        row = {**row_base, "metrics": metrics}

        agg = aggregate_one(task, cfg)
        if agg:
            row["paper_aggregates"] = agg.get("aggregates", {})

        delete_weights(task)
        rows.append(row)
        summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
