import gc
import logging
import os
from pathlib import Path
from typing import Sequence

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EvalAtEpochsCallback(TrainerCallback):
    """Run ``FinetuneTrainer.evaluate()`` (TOFU evaluators, etc.) at end of chosen epochs.

    Epochs are **1-based counters** matching ``TrainerState.epoch`` at ``on_epoch_end``
    (after the first full pass ``state.epoch == 1.0``, after the 10th ``== 10.0``, …).

    Outputs go to ``{output_dir}/checkpoint-{global_step}/evals/`` (same as a manual
    ``trainer.evaluate()`` call).

    When ``init_eval_dir`` and/or ``retain_eval_dir`` are supplied, the OpenUnlearning
    paper aggregator (Mem / Util / Priv / Agg, harmonic-mean style) is run on the
    freshly-written ``TOFU_EVAL.json`` and ``PAPER_AGGREGATES.{json,md}`` is written
    next to it -- mirroring what ``scripts/analysis/compute_paper_aggregates.py``
    does for the post-train eval, so every per-epoch checkpoint gets the same
    paper-grade roll-up.

    Aggregator failures are logged and swallowed: a missing ``init_eval_dir`` or
    malformed eval file must not abort training.

    Keep ``eval_strategy=no`` so the Trainer does not also run eval every epoch.
    """

    def __init__(
        self,
        trainer,
        epochs: Sequence[int],
        init_eval_dir: "os.PathLike | str | None" = None,
        retain_eval_dir: "os.PathLike | str | None" = None,
    ):
        self._trainer = trainer
        self._epochs = {int(e) for e in epochs if int(e) > 0}
        if not self._epochs:
            raise ValueError("eval_at_epochs must include at least one positive integer")
        self._init_eval_dir = Path(init_eval_dir) if init_eval_dir else None
        self._retain_eval_dir = Path(retain_eval_dir) if retain_eval_dir else None

    def _aggregate_paper(self, run_dir: str, global_step: int, completed_epoch: int) -> None:
        """Compute PAPER_AGGREGATES.{json,md} for the just-written eval dir.

        Mirrors ``aggregate_run`` in ``scripts/erase/run_exp_d_topk_vjp.py`` but is
        called in-process via ``analysis.paper_aggregates.aggregate_eval_dir``.
        Any failure is logged and swallowed -- this is opportunistic enrichment,
        never a training blocker.
        """
        try:
            from analysis.paper_aggregates import aggregate_eval_dir
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "eval_at_epochs: paper-aggregator import failed (%s); "
                "skipping PAPER_AGGREGATES at epoch %s.",
                e,
                completed_epoch,
            )
            return
        eval_dir = Path(run_dir) / f"checkpoint-{global_step}" / "evals"
        if not (eval_dir / "TOFU_EVAL.json").is_file():
            logger.warning(
                "eval_at_epochs: TOFU_EVAL.json not found at %s after evaluate(); "
                "skipping PAPER_AGGREGATES.",
                eval_dir,
            )
            return
        try:
            payload = aggregate_eval_dir(
                eval_dir,
                self._init_eval_dir,
                self._retain_eval_dir,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "eval_at_epochs: paper-aggregator raised at epoch %s (non-fatal): %s",
                completed_epoch,
                e,
            )
            return
        if payload is None:
            logger.warning(
                "eval_at_epochs: paper-aggregator returned None at epoch %s "
                "(malformed TOFU_EVAL.json?); PAPER_AGGREGATES not written.",
                completed_epoch,
            )
            return
        agg = payload.get("aggregates", {}) or {}

        def _f(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

        logger.info(
            "PAPER_AGGREGATES @ epoch %s (step %s): "
            "Mem=%s  Util=%s  Priv=%s  Agg=%s",
            completed_epoch,
            global_step,
            _f(agg.get("memorization")),
            _f(agg.get("utility")),
            _f(agg.get("privacy")),
            _f(agg.get("aggregate")),
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        completed = int(round(float(state.epoch)))
        if completed not in self._epochs:
            return control
        if not getattr(self._trainer, "evaluators", None):
            logger.warning(
                "eval_at_epochs=%s: no evaluators on trainer; skipping eval at epoch %s",
                sorted(self._epochs),
                completed,
            )
            return control
        logger.info(
            "Scheduled TOFU eval at end of epoch %s (global_step=%s)",
            completed,
            state.global_step,
        )
        gc.collect()
        torch.cuda.empty_cache()
        self._trainer.evaluate()
        try:
            run_dir = self._trainer._get_output_dir(trial=None)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "eval_at_epochs: could not resolve trainer output_dir (%s); "
                "skipping PAPER_AGGREGATES at epoch %s.",
                e,
                completed,
            )
            return control
        self._aggregate_paper(run_dir, int(state.global_step), completed)
        return control


class EvalAtStepsCallback(TrainerCallback):
    """Run ``trainer.evaluate()`` every ``every_steps`` optimiser steps.

    This is TELEMETRY for designing the automatic stop signal: it densely
    samples the true (privleak / retain_knowmem) trajectory so we can validate,
    offline, which cheap forget/retain signal best marks the sweet spot. The
    reported model is still the auto-stopped final model -- this callback only
    writes per-step ``checkpoint-{global_step}/evals/`` summaries, not the
    reported number.

    Keep ``eval_strategy=no`` so the Trainer does not also eval on its own.
    """

    def __init__(self, trainer, every_steps: int, warmup_steps: int = 0):
        self._trainer = trainer
        self._every = int(every_steps)
        self._warmup = int(warmup_steps)
        if self._every <= 0:
            raise ValueError("eval_every_steps must be a positive integer")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        step = int(state.global_step)
        if step <= 0 or step < self._warmup or step % self._every != 0:
            return control
        if not getattr(self._trainer, "evaluators", None):
            return control
        logger.info("Scheduled step eval at global_step=%s", step)
        gc.collect()
        torch.cuda.empty_cache()
        self._trainer.evaluate()
        return control
