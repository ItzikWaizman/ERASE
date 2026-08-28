import logging
from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator
from evals.muse import MUSEEvaluator

# LMEvalEvaluator pulls in lm_eval -> sacrebleu -> lxml.etree, which on
# locked-down Windows machines (WDAC / Application Control policies) can fail
# at import time with "DLL load failed while importing etree". TOFU and MUSE
# evaluators don't need any of that, so import LMEvalEvaluator lazily and
# tolerate failures: only fail if the user actually asks for the LMEval
# handler (then get_evaluator() will raise a clear NotImplementedError).
try:
    from evals.lm_eval import LMEvalEvaluator  # noqa: F401
    _LMEVAL_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # noqa: BLE001
    LMEvalEvaluator = None  # type: ignore[assignment]
    _LMEVAL_IMPORT_ERROR = _e
    logging.getLogger("evaluator").warning(
        "LMEvalEvaluator could not be imported (%s: %s). "
        "TOFU/MUSE evaluators still work; only the LMEval handler is unavailable.",
        type(_e).__name__,
        _e,
    )

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        if evaluator_handler_name == "LMEvalEvaluator" and _LMEVAL_IMPORT_ERROR is not None:
            raise NotImplementedError(
                f"LMEvalEvaluator was disabled at import time due to: "
                f"{type(_LMEVAL_IMPORT_ERROR).__name__}: {_LMEVAL_IMPORT_ERROR}"
            )
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
if LMEvalEvaluator is not None:
    _register_evaluator(LMEvalEvaluator)
