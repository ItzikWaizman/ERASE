import os

# Must be set BEFORE `import torch` so cuBLAS picks it up when the CUDA
# context is created. Required for `torch.use_deterministic_algorithms(True)`
# to work with matmuls on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators

if torch.cuda.is_available():
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
