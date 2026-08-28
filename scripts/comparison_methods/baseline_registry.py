"""
HF checkpoint IDs and hub-matched training overrides for TOFU Llama-3.2-1B-Instruct (forget10).

Primary benchmark path: train locally with `run_*_baseline.py` (uses TASK_TRAIN_* and
train_overrides_*). That produces `saves/unlearn/baseline_train_*/evals/TOFU_SUMMARY.json`.

`HF_CHECKPOINT_EVAL_RUNS` is only for optional sanity checks via `eval_hf_official_checkpoints.py`
(evaluate published weights without training).

GradAscent: no matching public checkpoint in that naming scheme; use TRAIN_GRAD_ASCENT_DEFAULTS
from configs/experiment/unlearn/tofu/default.yaml.
"""

from __future__ import annotations

# --- Hugging Face model IDs (evaluate downloaded weights as-is) ---

HF_SIMNPO = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
    "SimNPO_lr2e-05_b4.5_a1_d0_g0.125_ep10"
)
HF_RMU = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
    "RMU_lr5e-05_layer10_scoeff10_epoch10"
)
HF_NPO = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
    "NPO_lr1e-05_beta0.5_alpha1_epoch10"
)
HF_GRADDIFF = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
    "GradDiff_lr1e-05_alpha5_epoch10"
)
# Matches UNDIAL trainer defaults in configs/trainer/UNDIAL.yaml (lr 1e-4, beta 10); alpha=1 from hub name.
HF_UNDIAL = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
    "UNDIAL_lr0.0001_beta10_alpha1_epoch10"
)

HF_CHECKPOINT_EVAL_RUNS: list[tuple[str, str]] = [
    ("hf_official_SimNPO", HF_SIMNPO),
    ("hf_official_RMU", HF_RMU),
    ("hf_official_NPO", HF_NPO),
    ("hf_official_GradDiff", HF_GRADDIFF),
    ("hf_official_UNDIAL", HF_UNDIAL),
]

# --- Local training task names (saves/unlearn/<task>/) ---

TASK_TRAIN_SIMNPO = "baseline_train_SimNPO_hf_hparams"
TASK_TRAIN_RMU = "baseline_train_RMU_hf_hparams"
TASK_TRAIN_NPO = "baseline_train_NPO_hf_hparams"
TASK_TRAIN_UNDIAL = "baseline_train_UNDIAL_hf_hparams"
TASK_TRAIN_GRADDIFF = "baseline_train_GradDiff_hf_hparams"
TASK_TRAIN_GRAD_ASCENT = "baseline_train_GradAscent_default"


def train_overrides_simnpo() -> list[str]:
    return [
        "trainer.args.learning_rate=2e-5",
        "trainer.args.num_train_epochs=10",
        "trainer.method_args.delta=0",
        "trainer.method_args.beta=4.5",
        "trainer.method_args.alpha=1",
        "trainer.method_args.gamma=0.125",
    ]


def train_overrides_rmu() -> list[str]:
    return [
        "trainer.args.learning_rate=5e-5",
        "trainer.args.num_train_epochs=10",
        r"trainer.method_args.module_regex=model\.layers\.10",
        "trainer.method_args.steering_coeff=10",
    ]


def train_overrides_npo() -> list[str]:
    return [
        "trainer.args.learning_rate=1e-5",
        "trainer.args.num_train_epochs=10",
        "trainer.method_args.beta=0.5",
        "trainer.method_args.alpha=1",
        "trainer.method_args.gamma=1",
    ]


def train_overrides_undial() -> list[str]:
    return [
        "trainer.args.learning_rate=1e-4",
        "trainer.args.num_train_epochs=10",
        "trainer.method_args.beta=10",
        "trainer.method_args.alpha=1",
        "trainer.method_args.gamma=1",
    ]


def train_overrides_graddiff() -> list[str]:
    return [
        "trainer.args.learning_rate=1e-5",
        "trainer.args.num_train_epochs=10",
        "trainer.method_args.gamma=1",
        "trainer.method_args.alpha=5",
    ]


def train_overrides_grad_ascent() -> list[str]:
    return [
        "trainer.args.learning_rate=1e-5",
        "trainer.args.weight_decay=0.01",
        "trainer.args.num_train_epochs=10",
        "trainer.args.warmup_epochs=1.0",
    ]
