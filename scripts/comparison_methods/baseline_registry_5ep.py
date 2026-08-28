"""5-epoch variants of baseline training overrides and task names."""

from __future__ import annotations

from baseline_registry import (
    train_overrides_simnpo,
    train_overrides_rmu,
    train_overrides_npo,
    train_overrides_undial,
    train_overrides_graddiff,
    train_overrides_grad_ascent,
)

TASK_TRAIN_SIMNPO_5EP = "baseline_train_SimNPO_hf_hparams_5ep"
TASK_TRAIN_RMU_5EP = "baseline_train_RMU_hf_hparams_5ep"
TASK_TRAIN_NPO_5EP = "baseline_train_NPO_hf_hparams_5ep"
TASK_TRAIN_UNDIAL_5EP = "baseline_train_UNDIAL_hf_hparams_5ep"
TASK_TRAIN_GRADDIFF_5EP = "baseline_train_GradDiff_hf_hparams_5ep"
TASK_TRAIN_GRAD_ASCENT_5EP = "baseline_train_GradAscent_default_5ep"


def _override_epochs(overrides: list[str], epochs: int = 5) -> list[str]:
    return [
        f"trainer.args.num_train_epochs={epochs}" if o.startswith("trainer.args.num_train_epochs=") else o
        for o in overrides
    ]


def train_overrides_simnpo_5ep() -> list[str]:
    return _override_epochs(train_overrides_simnpo())

def train_overrides_rmu_5ep() -> list[str]:
    return _override_epochs(train_overrides_rmu())

def train_overrides_npo_5ep() -> list[str]:
    return _override_epochs(train_overrides_npo())

def train_overrides_undial_5ep() -> list[str]:
    return _override_epochs(train_overrides_undial())

def train_overrides_graddiff_5ep() -> list[str]:
    return _override_epochs(train_overrides_graddiff())

def train_overrides_grad_ascent_5ep() -> list[str]:
    return _override_epochs(train_overrides_grad_ascent())
