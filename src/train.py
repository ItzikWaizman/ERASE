import os

# Must be set BEFORE `import torch` so cuBLAS picks it up when the CUDA
# context is created. Required for `torch.use_deterministic_algorithms(True)`
# to work with matmuls on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import logging

import torch
import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# #region agent log
def _train_startup_log() -> None:
    import json, time
    payload = {
        "sessionId": "46cfae",
        "location": "src/train.py:startup",
        "message": "GPU state at python startup, before any model load",
        "hypothesisId": "H6,H7,H8",
        "data": {
            "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "env_SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "<unset>"),
            "env_SLURMD_NODENAME": os.environ.get("SLURMD_NODENAME", "<unset>"),
            "env_SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"),
            "torch_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "timestamp": int(time.time() * 1000),
    }
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        payload["data"]["cuda_free_gb"] = round(free_b / (1024**3), 2)
        payload["data"]["cuda_total_gb"] = round(total_b / (1024**3), 2)
        payload["data"]["other_tenants_gb"] = round((total_b - free_b) / (1024**3), 2)
        try:
            payload["data"]["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    print(f"[MEM_DEBUG] {payload['location']} h={payload['hypothesisId']} {payload['message']} | {payload['data']}", flush=True)


_train_startup_log()
# #endregion

logger = logging.getLogger(__name__)


def _maybe_merge_lora_before_save(trainer) -> None:
    """If the trainer holds a PEFT-wrapped model, merge LoRA adapters into the
    base weights and replace trainer.model with the plain HF model BEFORE
    saving. This keeps on-disk checkpoints plain HF models, so the existing
    eval pipeline (which does AutoModelForCausalLM.from_pretrained) works
    unchanged. Mid-training evaluate() calls (EvalAtEpochsCallback) happen on
    the in-memory PEFT model and are unaffected.
    """
    model = getattr(trainer, "model", None)
    if model is None:
        return
    if not (
        hasattr(model, "merge_and_unload")
        and model.__class__.__name__.startswith("Peft")
    ):
        return
    try:
        merged = model.merge_and_unload()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "merge_and_unload() failed (%s); saving PEFT adapters as-is.", e
        )
        return
    trainer.model = merged
    if hasattr(trainer, "model_wrapped"):
        trainer.model_wrapped = merged
    logger.info("Merged LoRA adapters into base weights before save.")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        _maybe_merge_lora_before_save(trainer)
        trainer.save_model(trainer_args.output_dir)
    else:
        logger.info(
            "do_train=False; saving loaded model to %s (no optimization steps).",
            trainer_args.output_dir,
        )
        _maybe_merge_lora_before_save(trainer)
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
