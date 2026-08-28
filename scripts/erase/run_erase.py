"""Clean ERASE experiment runner for remote / Slurm clusters.

Designed to be small, self-contained, and easy to launch in parallel via
``sbatch``. Each invocation runs a list of ERASE configurations sequentially
on the single GPU it sees (set ``CUDA_VISIBLE_DEVICES`` per Slurm job).
Multi-GPU parallelism is intentionally NOT handled inside one process --
launch one process per GPU instead.

Pipeline per run
----------------
1. ``src/train.py``  -- ERASE training under ``saves/unlearn/<task>/``
2. ``src/eval.py``   -- paper-grade TOFU eval into ``saves/unlearn/<task>/evals/``
3. ``scripts/analysis/compute_paper_aggregates.py``
                     -- Mem / Util / Priv / Agg into ``PAPER_AGGREGATES.json``
4. Disk hygiene based on the aggregate scores:
     * Agg > AGG_HIGHLIGHT_MIN AND Util >= UTIL_HIGHLIGHT_MIN
         -> the task folder is moved to ``saves/highlights/<task>/`` and
            its weights are kept.
     * Otherwise the ``*.safetensors`` files are deleted to save disk
       space; eval artefacts (TOFU_EVAL.json, PAPER_AGGREGATES.{json,md},
       trainer_state.json, .hydra) are always preserved.

Default winner config
---------------------
``DEFAULT_RUN`` matches ``configs/runs/winner_default.json`` (keep them in
sync). Partial run JSON files merge on top of this baseline.

Custom RUNS list
----------------
Pass ``--runs path/to/runs.json`` to override the default list. The file
must be a JSON array of objects, each object being a dict of overrides
applied on top of ``DEFAULT_RUN`` (so partial entries are fine).

Examples
--------
Run the default winner config on the local GPU::

    python scripts/erase/run_erase.py

Run a custom RUNS file and write the summary JSON to a per-job dir::

    python scripts/erase/run_erase.py \\
        --runs configs/runs/winner_default.json \\
        --output-dir results/erase_remote/job_${SLURM_ARRAY_TASK_ID} \\
        --task-prefix ERASE_${SLURM_JOB_ID}

Parallel sbatch flow
--------------------
Each Slurm job gets a separate ``--runs`` file and ``--output-dir`` so jobs
write their summaries to non-conflicting paths. Task names already encode
hyperparameters, so jobs with disjoint hyperparameters never collide on
``saves/unlearn/<task>/`` either. See DEPLOY.md for a sample sbatch script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "erase"))
from _task_name import build_rich_task_name  # noqa: E402


# ---------------------------------------------------------------------------
# Config: defaults + thresholds
# ---------------------------------------------------------------------------

# Must match configs/runs/winner_default.json (single object) — update both
# when changing the default recipe. Everything else falls back to the trainer
# config (configs/trainer/ERASE.yaml).
DEFAULT_RUN: dict = {
    "task_prefix": "ERASE_DEFAULT",
    "cov_source": "wikipedia",
    "model_tag": "llama1b",
    "forget_split": "forget10",
    "retain_split": "retain90",
    "holdout_split": "holdout10",
    "epochs": 10,
    "lr": 0.04,
    "alpha": 0.5,
    "topk": 10,
    "layers": [1, 2, 3, 4, 5],
    "forget_loss_max_ce": 20,
    "forget_loss_per_token_cap": 5,
    "forget_loss_weight": 0.9,
    "forget_loss_type": "ce",
    "lr_scheduler": "cosine",
    "author_only_vjp": True,
    "author_mask_mode": "span",
    "vjp_renormalize": True,
}


# Disk-hygiene thresholds. Weights survive iff BOTH conditions hold;
# otherwise the *.safetensors files in the task folder are deleted.
AGG_HIGHLIGHT_MIN: float = 0.71
UTIL_HIGHLIGHT_MIN: float = 0.90

# Subprocess paths.
TRAIN: Path = ROOT / "src" / "train.py"
EVAL: Path = ROOT / "src" / "eval.py"
AGGREGATE_SCRIPT: Path = ROOT / "scripts" / "analysis" / "compute_paper_aggregates.py"
PLOT_SCRIPT: Path = ROOT / "scripts" / "analysis" / "plot_training_curves.py"
CE_DIST_SCRIPT: Path = ROOT / "scripts" / "analysis" / "plot_ce_distribution.py"
DYNSTOP_SCRIPT: Path = ROOT / "scripts" / "analysis" / "plot_dynstop_convergence.py"
HIGHLIGHTS_ROOT: Path = ROOT / "saves" / "highlights"

# ---------------------------------------------------------------------------
# Per-model profiles
# ---------------------------------------------------------------------------
# Each RUN may set a "model_tag" field selecting one of these profiles. It
# determines which Hydra experiment (unlearn + paper-eval) is invoked, where
# the precomputed covariances live, and which TOFU reference evals the
# aggregator normalizes against. Defaults to "llama1b" so existing 1B sweeps
# keep working unchanged.
DEFAULT_MODEL_TAG: str = "llama1b"

MODEL_PROFILES: dict[str, dict] = {
    "llama1b": {
        "unlearn_experiment": "unlearn/tofu/llama1b/erase",
        "eval_experiment":    "eval/tofu/llama1b/paper",
        "cov_dir_wikipedia":  ROOT / "saves" / "precompute" / "llama1b" / "wikipedia_covariance",
        "cov_dir_tofu":       ROOT / "saves" / "precompute" / "llama1b" / "covariances",
        # Same oracle retain logs used by the local 1B runner
        # (run_exp_d_topk_vjp.py) and recorded in the 1B winner's hydra cfg.
        "retain_logs":        ROOT / "saves" / "eval" / "tofu_llama-1b_oracle_retain90" / "TOFU_EVAL.json",
        "init_ref_dir":       ROOT / "saves" / "eval" / "init_finetuned",
        "retain_ref_dir":     ROOT / "saves" / "eval" / "retain_oracle",
    },
    "llama2_7b": {
        "unlearn_experiment": "unlearn/tofu/llama2_7b/erase",
        "eval_experiment":    "eval/tofu/llama2_7b/paper",
        "cov_dir_wikipedia":  ROOT / "saves" / "precompute" / "llama2_7b" / "wikipedia_covariance",
        "cov_dir_tofu":       ROOT / "saves" / "precompute" / "llama2_7b" / "covariances",
        "retain_logs":        ROOT / "saves" / "eval" / "llama2_7b" / "retain_oracle" / "TOFU_EVAL.json",
        "init_ref_dir":       ROOT / "saves" / "eval" / "llama2_7b" / "init_finetuned",
        "retain_ref_dir":     ROOT / "saves" / "eval" / "llama2_7b" / "retain_oracle",
    },
    # Qwen2.5-3B-Instruct, targeting the v2 init_finetuned model trained by
    # configs/runs/A3_qwen_finetune_v2_full. The author-name token-id cache
    # is automatically per-tokenizer (see ERASE._tokenizer_cache_tag in
    # src/trainer/unlearn/erase.py, commit b42333a) so no extra adjustment is
    # required when switching from Llama to Qwen -- the first run rebuilds
    # the cache file under a Qwen-specific filename.
    "qwen3b": {
        "unlearn_experiment": "unlearn/tofu/qwen3b/erase",
        "eval_experiment":    "eval/tofu/qwen3b/paper",
        "cov_dir_wikipedia":  ROOT / "saves" / "precompute" / "qwen3b" / "wikipedia_covariance",
        "cov_dir_tofu":       ROOT / "saves" / "precompute" / "qwen3b" / "covariances",
        "retain_logs":        ROOT / "saves" / "eval" / "qwen3b" / "retain_oracle_v2" / "TOFU_EVAL.json",
        "init_ref_dir":       ROOT / "saves" / "eval" / "qwen3b" / "init_finetuned_v2",
        "retain_ref_dir":     ROOT / "saves" / "eval" / "qwen3b" / "retain_oracle_v2",
    },
    # Phi-3.5-mini-instruct (3.8B), targeting the v2 init_finetuned model
    # trained by configs/runs/A3_phi35_finetune_v2_full. Same per-tokenizer
    # cache handling as the qwen3b profile -- no extra adjustment needed.
    "phi35": {
        "unlearn_experiment": "unlearn/tofu/phi35/erase",
        "eval_experiment":    "eval/tofu/phi35/paper",
        "cov_dir_wikipedia":  ROOT / "saves" / "precompute" / "phi35" / "wikipedia_covariance",
        "cov_dir_tofu":       ROOT / "saves" / "precompute" / "phi35" / "covariances",
        "retain_logs":        ROOT / "saves" / "eval" / "phi35" / "retain_oracle_v2" / "TOFU_EVAL.json",
        "init_ref_dir":       ROOT / "saves" / "eval" / "phi35" / "init_finetuned_v2",
        "retain_ref_dir":     ROOT / "saves" / "eval" / "phi35" / "retain_oracle_v2",
    },
}


def _profile(cfg: dict | None = None) -> dict:
    tag = (cfg or {}).get("model_tag", DEFAULT_MODEL_TAG)
    if tag not in MODEL_PROFILES:
        raise SystemExit(
            f"unknown model_tag={tag!r}; allowed: {sorted(MODEL_PROFILES)}"
        )
    return MODEL_PROFILES[tag]


# ---------------------------------------------------------------------------
# Task-name builder
# ---------------------------------------------------------------------------

def _layers_hydra(layers) -> str:
    return "[" + ",".join(str(x) for x in layers) + "]"


def build_task_name(cfg: dict) -> str:
    """Build a self-documenting on-disk task name for a RUN dict.

    Delegates to the shared :func:`_task_name.build_rich_task_name` so that
    folders produced by this remote runner are byte-identical in format to
    those produced by the local ``run_exp_d_topk_vjp.py`` runner. Every
    hyperparameter that meaningfully changes training appears in the name
    (lr, alpha, cap, ptcap, scheduler, layers, ebs, determinism flag, etc.),
    so no two distinct configs collide on disk.
    """
    cov = cfg.get("cov_source", "wikipedia")
    cov_tag = "wiki" if cov == "wikipedia" else "tofu"
    return build_rich_task_name(
        cfg,
        cov_tag,
        default_epochs=cfg.get("epochs", 10),
        default_prefix=cfg.get("task_prefix", "ERASE"),
    )


# ---------------------------------------------------------------------------
# Hydra-override builder (RUN dict -> list of trainer.method_args overrides)
# ---------------------------------------------------------------------------

# Keys that map straight to ``trainer.method_args.<key>`` with no special
# handling. Add new method_args here as the trainer grows.
_METHOD_ARG_KEYS = (
    "alpha",
    "forget_loss_weight", "forget_loss_max_ce",
    "forget_loss_per_token_cap",
    "forget_loss_answer_target", "forget_loss_answer_target_std",
    "forget_loss_token_ce_ceiling",
    "forget_loss_answer_mode", "forget_loss_band_lower", "forget_loss_band_upper",
    "forget_loss_type",
    "forget_entropy_reg_weight", "forget_entropy_reg_target",
    "signal_monitor_interval", "signal_monitor_probe_size",
    "token_ce_dump_epochs", "token_ce_dump_max_samples",
    "author_mask_mode", "answer_only_down_proj_grad",
    "train_scope",
    "dynamic_stop_loss_threshold", "dynamic_stop_log_upper",
    "dynamic_stop_max_active_steps_per_sample", "dynamic_stop_decode_threshold",
    "dynamic_stop_longtail_threshold",
    "dynamic_stop_done_sample_prob",
)


def build_train_overrides(cfg: dict, cov_dir: Path) -> list[str]:
    ev_ep = cfg.get("eval_at_epochs") or []
    # When eval_at_epochs is set we flip do_eval=True so the EvalAtEpochsCallback
    # (wired in src/trainer/__init__.py) runs the full TOFU eval at end of each
    # chosen epoch. Results land under saves/unlearn/<task>/checkpoint-{step}/evals/.
    do_eval_flag = "True" if ev_ep else "False"
    extra: list[str] = [
        f"trainer.args.learning_rate={cfg.get('lr', 0.04)}",
        f"trainer.args.num_train_epochs={cfg.get('epochs', 10)}",
        "trainer.args.eval_strategy=no",
        f"trainer.args.do_eval={do_eval_flag}",
        "trainer.args.eval_on_start=False",
        f"trainer.method_args.covariance_dir={cov_dir.as_posix()}",
        f"trainer.method_args.target_layers={_layers_hydra(cfg.get('layers', [1, 2, 3, 4, 5]))}",
        f"trainer.method_args.topk_vjp_count={cfg.get('topk', 10)}",
    ]
    if ev_ep:
        extra.append(
            "+trainer.args.eval_at_epochs=["
            + ",".join(str(int(x)) for x in ev_ep)
            + "]"
        )
        # Pass init-finetuned + retain-oracle reference dirs through to the
        # EvalAtEpochsCallback so it can write PAPER_AGGREGATES.{json,md}
        # (Mem / Util / Priv / Agg) at every in-training checkpoint -- same
        # roll-up as the post-train aggregator. Each ref is optional and is
        # only forwarded when its TOFU_EVAL.json actually exists.
        prof = _profile(cfg)
        init_ref = prof["init_ref_dir"]
        retain_ref = prof["retain_ref_dir"]
        if (init_ref / "TOFU_EVAL.json").is_file():
            extra.append(
                f"+trainer.args.eval_at_epochs_init_eval={init_ref.as_posix()}"
            )
        if (retain_ref / "TOFU_EVAL.json").is_file():
            extra.append(
                f"+trainer.args.eval_at_epochs_retain_eval={retain_ref.as_posix()}"
            )
    extra.append(
        "trainer.method_args.author_only_vjp="
        f"{str(cfg.get('author_only_vjp', False)).lower()}"
    )
    extra.append(
        "trainer.method_args.vjp_renormalize="
        f"{str(cfg.get('vjp_renormalize', False)).lower()}"
    )

    for key in _METHOD_ARG_KEYS:
        if key in cfg:
            val = cfg[key]
            if isinstance(val, bool):
                val = str(val).lower()
            extra.append(f"trainer.method_args.{key}={val}")

    use_adam = str(cfg.get("optim", "sgd")).lower() == "adam"
    if use_adam:
        extra += [
            "trainer.args.optim=adamw_torch",
            "+trainer.args.adam_beta1=0.9",
            "+trainer.args.adam_beta2=0.95",
            "trainer.args.weight_decay=0.0",
        ]
        if "lr_scheduler" not in cfg:
            cfg.setdefault("lr_scheduler", "cosine")
        if "warmup_ratio" not in cfg:
            cfg.setdefault("warmup_ratio", 0.05)
    if "seed" in cfg:
        # Multi-seed significance runs (R2): trainer default is seed 0.
        # NOTE: encode the seed in task_prefix too -- build_rich_task_name
        # does not include it, so identical configs with different seeds
        # would otherwise collide on the task dir.
        extra.append(f"trainer.args.seed={cfg['seed']}")
    if cfg.get("lr_scheduler"):
        extra.append(f"+trainer.args.lr_scheduler_type={cfg['lr_scheduler']}")
    if "warmup_ratio" in cfg:
        extra.append(f"+trainer.args.warmup_ratio={cfg['warmup_ratio']}")
    if "max_grad_norm" in cfg:
        extra.append(f"+trainer.args.max_grad_norm={cfg['max_grad_norm']}")
    if "max_steps" in cfg:
        extra.append(f"+trainer.args.max_steps={cfg['max_steps']}")
    if "per_device_train_batch_size" in cfg:
        extra.append(
            f"trainer.args.per_device_train_batch_size={cfg['per_device_train_batch_size']}"
        )
    if "gradient_accumulation_steps" in cfg:
        extra.append(
            f"trainer.args.gradient_accumulation_steps={cfg['gradient_accumulation_steps']}"
        )
    if cfg.get("deterministic", False):
        # HF's full_determinism + no-worker / no-pin dataloader + eager attn.
        # CUBLAS_WORKSPACE_CONFIG is set via _det_env() on the subprocess.
        # dataloader_pin_memory / dataloader_num_workers are NOT present in
        # the composed trainer.args struct (trainer/finetune.yaml + ERASE.yaml),
        # so Hydra requires the `+` prefix to ADD them rather than override.
        extra += [
            "+trainer.args.full_determinism=true",
            "+trainer.args.dataloader_pin_memory=false",
            "+trainer.args.dataloader_num_workers=0",
            "model.model_args.attn_implementation=eager",
        ]
    return extra


def _det_env() -> dict:
    """Subprocess env with determinism vars layered on top of os.environ.

    CUBLAS_WORKSPACE_CONFIG MUST be set before torch is imported by the child
    process, otherwise torch.use_deterministic_algorithms() can't enable
    deterministic cuBLAS GEMMs.
    """
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


# ---------------------------------------------------------------------------
# Subprocess wrappers
# ---------------------------------------------------------------------------

def _python() -> str:
    return os.environ.get("RESEARCH_UNLEARNING_PYTHON") or sys.executable


def _run(cmd: list[str], label: str, env: dict | None = None) -> int:
    print(f"\n=== {label} ===\n  $ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def train_run(task: str, cfg: dict, cov_dir: Path) -> bool:
    prof = _profile(cfg)
    cmd = [
        _python(), "-W", "ignore", str(TRAIN),
        "--config-name=unlearn.yaml",
        f"experiment={prof['unlearn_experiment']}",
        "trainer=ERASE",
        f"task_name={task}",
        f"forget_split={cfg.get('forget_split', 'forget10')}",
        f"retain_split={cfg.get('retain_split', 'retain90')}",
        f"holdout_split={cfg.get('holdout_split', 'holdout10')}",
        f"retain_logs_path={cfg.get('retain_logs_path', prof['retain_logs'].as_posix())}",
    ] + build_train_overrides(cfg, cov_dir)
    env = _det_env() if cfg.get("deterministic", False) else None
    return _run(cmd, f"TRAIN {task}", env=env) == 0


def eval_run(task: str, cfg: dict | None = None) -> bool:
    prof = _profile(cfg)
    out = ROOT / "saves" / "unlearn" / task / "evals"
    cmd = [
        _python(), "-W", "ignore", str(EVAL),
        "--config-name=eval.yaml",
        f"experiment={prof['eval_experiment']}",
        f"task_name={task}_eval",
        f"forget_split={(cfg or {}).get('forget_split', 'forget10')}",
        f"holdout_split={(cfg or {}).get('holdout_split', 'holdout10')}",
        f"retain_logs_path={(cfg or {}).get('retain_logs_path', prof['retain_logs'].as_posix())}",
        f"model.model_args.pretrained_model_name_or_path={ROOT.as_posix()}/saves/unlearn/{task}",
        "model.model_args.attn_implementation=eager",
        f"paths.output_dir={out.as_posix()}",
        "eval.tofu.batch_size=8",
    ]
    env = _det_env() if (cfg is not None and cfg.get("deterministic", False)) else None
    return _run(cmd, f"EVAL {task}", env=env) == 0


def aggregate_run(task: str, cfg: dict | None = None) -> dict | None:
    prof = _profile(cfg)
    task_dir = ROOT / "saves" / "unlearn" / task
    init_ref = Path((cfg or {}).get("init_ref_dir", prof["init_ref_dir"]))
    retain_ref = Path((cfg or {}).get("retain_ref_dir", prof["retain_ref_dir"]))
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


def plot_run(task: str) -> None:
    if not PLOT_SCRIPT.is_file():
        return
    task_dir = ROOT / "saves" / "unlearn" / task
    _run([_python(), str(PLOT_SCRIPT), "--dir", str(task_dir)], f"PLOT {task}")
    tofu_eval = task_dir / "evals" / "TOFU_EVAL.json"
    if CE_DIST_SCRIPT.is_file() and tofu_eval.is_file():
        _run([_python(), str(CE_DIST_SCRIPT), str(tofu_eval)], f"CE_DIST {task}")
    trainer_state = task_dir / "trainer_state.json"
    if DYNSTOP_SCRIPT.is_file() and trainer_state.is_file():
        _run([_python(), str(DYNSTOP_SCRIPT), str(trainer_state)], f"DYNSTOP {task}")


def clear_cuda_cache() -> None:
    """Spawn a tiny subprocess that empties the CUDA caching allocator."""
    snippet = (
        "import gc, torch; gc.collect();"
        " torch.cuda.empty_cache() if torch.cuda.is_available() else None;"
        " torch.cuda.ipc_collect() if torch.cuda.is_available() else None"
    )
    subprocess.run([_python(), "-c", snippet], cwd=str(ROOT), capture_output=True)


# ---------------------------------------------------------------------------
# Disk-hygiene: HIGHLIGHT (move + keep) vs DELETE-WEIGHTS.
# ---------------------------------------------------------------------------

def _move_to_highlights(task_dir: Path) -> Path | None:
    HIGHLIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    dest = HIGHLIGHTS_ROOT / task_dir.name
    if dest.exists():
        n = 1
        while True:
            cand = HIGHLIGHTS_ROOT / f"{task_dir.name}__dup{n}"
            if not cand.exists():
                dest = cand
                break
            n += 1
    try:
        shutil.move(str(task_dir), str(dest))
        return dest
    except OSError as e:
        print(f"   could not move {task_dir} -> {dest}: {e}", flush=True)
        return None


def _delete_weights(task_dir: Path, reason: str) -> None:
    weight_files = list(task_dir.glob("*.safetensors"))
    if not weight_files:
        return
    total_bytes = sum(p.stat().st_size for p in weight_files)
    for p in weight_files:
        try:
            p.unlink()
        except OSError as e:
            print(f"   could not delete {p}: {e}", flush=True)
    gb = total_bytes / (1024 ** 3)
    print(
        f"=== DELETE weights for {task_dir.name} ({len(weight_files)} file(s), "
        f"{gb:.2f} GB) -- {reason} ===",
        flush=True,
    )


def cleanup_weights(
    task: str,
    agg: float | None,
    util: float | None,
    *,
    always_delete: bool = False,
) -> str:
    """Apply HIGHLIGHT / DELETE based on aggregate AND utility."""
    task_dir = ROOT / "saves" / "unlearn" / task
    if not task_dir.is_dir():
        return "missing"

    if always_delete:
        _delete_weights(task_dir, reason="--always-delete-weights")
        return "deleted"

    if agg is None:
        print(f"=== KEEP weights for {task}: paper Agg unavailable ===", flush=True)
        return "kept"

    util_ok = util is not None and util >= UTIL_HIGHLIGHT_MIN
    agg_ok = agg > AGG_HIGHLIGHT_MIN
    if agg_ok and util_ok:
        moved = _move_to_highlights(task_dir)
        if moved is not None:
            print(
                f"=== HIGHLIGHT {task} -> {moved}: "
                f"Agg={agg:.4f} > {AGG_HIGHLIGHT_MIN} AND "
                f"Util={util:.4f} >= {UTIL_HIGHLIGHT_MIN} ===",
                flush=True,
            )
            return "highlighted"
        return "kept"
    util_str = f"{util:.4f}" if util is not None else "n/a"
    _delete_weights(
        task_dir,
        reason=f"Agg={agg:.4f} util={util_str} below "
        f"({AGG_HIGHLIGHT_MIN}, {UTIL_HIGHLIGHT_MIN}) bar",
    )
    return "deleted"


# ---------------------------------------------------------------------------
# RUNS loading
# ---------------------------------------------------------------------------

def load_runs(runs_path: Path | None) -> list[dict]:
    """Load and normalize the list of runs.

    Each entry is layered on top of ``DEFAULT_RUN`` so partial JSON files
    (specifying only the fields you want to override) work as expected.
    """
    if runs_path is None:
        raw_runs = [{}]  # one default run
    else:
        raw = json.loads(runs_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_runs = [raw]
        elif isinstance(raw, list):
            raw_runs = raw
        else:
            raise SystemExit(
                f"--runs file must be a JSON list or object, got {type(raw).__name__}"
            )
    runs: list[dict] = []
    for item in raw_runs:
        if not isinstance(item, dict):
            raise SystemExit(f"each RUN entry must be a JSON object, got {type(item).__name__}")
        merged = dict(DEFAULT_RUN)
        merged.update(item)
        runs.append(merged)
    return runs


def _cov_dir_for(source: str, cfg: dict | None = None) -> Path:
    p = _profile(cfg)
    return p["cov_dir_wikipedia"] if source == "wikipedia" else p["cov_dir_tofu"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "runs_positional",
        nargs="?",
        type=Path,
        help="Optional positional RUNS JSON path. Equivalent to --runs FILE.",
    )
    p.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Path to a JSON file with the RUNS list (object or list of objects). "
        "Each entry overrides DEFAULT_RUN. Defaults to a single winner run.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "erase_remote",
        help="Where to write the per-job summary JSON. Use a unique value per "
        "Slurm job to avoid collisions.",
    )
    p.add_argument(
        "--task-prefix",
        type=str,
        default=None,
        help="Override the 'task_prefix' field of every RUN. Set this per "
        "Slurm job (e.g. ERASE_${SLURM_JOB_ID}) to namespace the runs.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run tasks even if a cached eval summary already exists.",
    )
    p.add_argument(
        "--always-delete-weights",
        action="store_true",
        help="Always delete model weights after eval, even for highlight-worthy "
        "runs. Use when disk quota is limited (e.g. 7B models).",
    )
    p.add_argument(
        "--summary-name",
        type=str,
        default="erase_runs.json",
        help="Filename written under --output-dir with the run summaries.",
    )
    args = p.parse_args()

    if args.runs is not None and args.runs_positional is not None:
        raise SystemExit("Pass the RUNS file either positionally or via --runs, not both.")
    runs_path = args.runs if args.runs is not None else args.runs_positional

    runs = load_runs(runs_path)
    if args.task_prefix:
        for cfg in runs:
            cfg["task_prefix"] = args.task_prefix

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / args.summary_name

    print(f"Loaded {len(runs)} run(s); output dir: {args.output_dir}", flush=True)
    rows: list[dict] = []
    clear_cuda_cache()

    for i, cfg in enumerate(runs):
        cov_source = cfg.get("cov_source", "wikipedia")
        if cov_source not in ("wikipedia", "tofu"):
            print(f"[run {i}] invalid cov_source={cov_source!r}; skipping", flush=True)
            continue
        cov_dir = _cov_dir_for(cov_source, cfg)
        if not cov_dir.is_dir():
            print(
                f"[run {i}] missing covariance dir: {cov_dir}\n"
                "  -> compute it first (see DEPLOY.md / scripts/setup_remote.py)",
                flush=True,
            )
            continue

        task = build_task_name(cfg)
        cached = ROOT / "saves" / "unlearn" / task / "evals" / "TOFU_SUMMARY.json"
        cached_highlight = HIGHLIGHTS_ROOT / task / "evals" / "TOFU_SUMMARY.json"
        row_base = {"run_index": i, "task": task, **cfg}

        if cached.is_file() and not args.force:
            metrics = json.loads(cached.read_text(encoding="utf-8"))
            rows.append({**row_base, "metrics": metrics, "skipped": True})
            print(f"[run {i}] skip {task} (cached)", flush=True)
            continue
        if cached_highlight.is_file() and not args.force:
            metrics = json.loads(cached_highlight.read_text(encoding="utf-8"))
            rows.append({**row_base, "metrics": metrics, "skipped": True, "highlight": True})
            print(f"[run {i}] skip {task} (highlighted earlier)", flush=True)
            continue

        if not train_run(task, cfg, cov_dir):
            rows.append({**row_base, "error": "train_failed"})
            clear_cuda_cache()
            continue
        clear_cuda_cache()

        if not eval_run(task, cfg):
            rows.append({**row_base, "error": "eval_failed"})
            clear_cuda_cache()
            continue
        clear_cuda_cache()
        plot_run(task)

        sp = ROOT / "saves" / "unlearn" / task / "evals" / "TOFU_SUMMARY.json"
        metrics = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
        row = {**row_base, "metrics": metrics}

        agg_payload = aggregate_run(task, cfg)
        agg_value: float | None = None
        util_value: float | None = None
        if agg_payload is not None:
            paper_aggs = agg_payload.get("aggregates") or {}
            row["paper_aggregates"] = paper_aggs
            agg_value = paper_aggs.get("aggregate")
            util_value = paper_aggs.get("utility")

        outcome = cleanup_weights(
            task, agg_value, util_value,
            always_delete=args.always_delete_weights,
        )
        row["outcome"] = outcome
        rows.append(row)
        # Stream the summary after every run so a crash doesn't lose progress.
        summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
