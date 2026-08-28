import torch
from typing import Dict, Any
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer
from trainer.callbacks import EvalAtEpochsCallback, EvalAtStepsCallback
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU
from trainer.unlearn.erase import ERASE


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    eval_at_epochs = trainer_args.pop("eval_at_epochs", None)
    # Optional reference paths used by EvalAtEpochsCallback to compute
    # PAPER_AGGREGATES.{json,md} (Mem / Util / Priv / Agg) at every per-epoch
    # eval. Either / both can be None: missing init -> Util cannot be
    # normalized; missing retain -> Privacy aggregate not computed. Mem is
    # always available either way.
    eval_at_epochs_init_eval = trainer_args.pop("eval_at_epochs_init_eval", None)
    eval_at_epochs_retain_eval = trainer_args.pop("eval_at_epochs_retain_eval", None)
    eval_every_steps = trainer_args.pop("eval_every_steps", None)
    eval_every_steps_warmup = trainer_args.pop("eval_every_steps_warmup", None)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return (
        trainer_args,
        eval_at_epochs,
        eval_at_epochs_init_eval,
        eval_at_epochs_retain_eval,
        eval_every_steps,
        eval_every_steps_warmup,
    )


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    (
        trainer_args,
        eval_at_epochs,
        eval_at_epochs_init_eval,
        eval_at_epochs_retain_eval,
        eval_every_steps,
        eval_every_steps_warmup,
    ) = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    if eval_at_epochs:
        epochs_list = sorted({int(x) for x in eval_at_epochs if int(x) > 0})
        if epochs_list:
            if evaluators:
                trainer.add_callback(
                    EvalAtEpochsCallback(
                        trainer,
                        epochs_list,
                        init_eval_dir=eval_at_epochs_init_eval,
                        retain_eval_dir=eval_at_epochs_retain_eval,
                    )
                )
                _refs_msg = []
                if eval_at_epochs_init_eval:
                    _refs_msg.append(f"init_eval={eval_at_epochs_init_eval}")
                if eval_at_epochs_retain_eval:
                    _refs_msg.append(f"retain_eval={eval_at_epochs_retain_eval}")
                logger.info(
                    "eval_at_epochs enabled: will run TOFU eval after epochs %s%s",
                    epochs_list,
                    f"; PAPER_AGGREGATES with {', '.join(_refs_msg)}" if _refs_msg
                    else " (no init/retain refs -> partial PAPER_AGGREGATES only)",
                )
            else:
                logger.warning(
                    "eval_at_epochs=%s ignored: no evaluators configured for this run",
                    epochs_list,
                )
        else:
            logger.warning(
                "eval_at_epochs=%s contains no positive integers; ignored",
                list(eval_at_epochs),
            )
    if eval_every_steps and int(eval_every_steps) > 0:
        if evaluators:
            trainer.add_callback(
                EvalAtStepsCallback(
                    trainer,
                    int(eval_every_steps),
                    warmup_steps=int(eval_every_steps_warmup or 0),
                )
            )
            logger.info(
                "eval_every_steps enabled: dense eval every %s steps "
                "(warmup %s) -- TELEMETRY only; reported model is the "
                "auto-stopped final model.",
                int(eval_every_steps),
                int(eval_every_steps_warmup or 0),
            )
        else:
            logger.warning(
                "eval_every_steps=%s ignored: no evaluators configured",
                eval_every_steps,
            )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)
_register_trainer(ERASE)
