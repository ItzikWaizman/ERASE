from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict, OmegaConf
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def _flash_attn_2_supported() -> bool:
    """flash_attention_2 needs the flash_attn package AND an Ampere+ GPU (sm>=80).
    V100 (sm=70) is unsupported even if the package is present."""
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 8


def _resolve_attn_chain(requested) -> list:
    """Ordered, deduped list of attn implementations to try.

    - requested is None/"auto": prefer flash_attention_2 only when the node
      actually supports it, else sdpa; then fall back sdpa -> eager.
    - requested is explicit (e.g. "flash_attention_2"): honor it first, but
      still append sdpa -> eager so a node that can't run it degrades instead
      of crashing.
    """
    chain = []
    if requested and requested != "auto":
        chain.append(requested)
    else:
        chain.append("flash_attention_2" if _flash_attn_2_supported() else "sdpa")
    for fb in ("sdpa", "eager"):
        if fb not in chain:
            chain.append(fb)
    return chain


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
        requested_attn = model_args.pop("attn_implementation", None)
    base_kwargs = OmegaConf.to_container(model_args, resolve=True) or {}

    # Robust attention selection: try the best supported impl for this node and
    # degrade gracefully (flash_attention_2 -> sdpa -> eager) on any load error.
    attn_chain = _resolve_attn_chain(requested_attn)
    model = None
    last_err = None
    for attn in attn_chain:
        try:
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn,
                **base_kwargs,
                cache_dir=hf_home,
            )
            if attn != requested_attn:
                logger.warning(
                    f"attn_implementation: using '{attn}' "
                    f"(requested={requested_attn!r}, tried order={attn_chain})."
                )
            break
        except Exception as e:
            last_err = e
            logger.warning(
                f"from_pretrained failed with attn_implementation='{attn}': "
                f"{type(e).__name__}: {e}. Falling back to next option."
            )
    if model is None:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {last_err} while fetching model using "
            f"{model_handler}.from_pretrained() (tried attn {attn_chain})."
        )
    tokenizer = get_tokenizer(tokenizer_args)

    # Optional LoRA wrapping (PEFT). Controlled by an optional `model.lora`
    # block in the model config. When absent or disabled, behaviour is
    # unchanged and peft is not imported at all.
    lora_cfg = model_cfg.get("lora", None)
    if lora_cfg is not None and bool(lora_cfg.get("enabled", True)):
        model = _wrap_with_lora(model, lora_cfg)

    return model, tokenizer


def _wrap_with_lora(model, lora_cfg: DictConfig):
    """Wrap `model` with a LoRA adapter using the provided config.

    Expected fields on `lora_cfg` (all optional except `target_modules`):
        enabled: bool           (default True; if False this helper is not called)
        r: int                  (LoRA rank; default 16)
        alpha: int|float        (LoRA alpha / scaling; default 2*r)
        dropout: float          (default 0.0)
        bias: str               (none|all|lora_only; default none)
        target_modules: list[str]  (e.g. [q_proj, k_proj, v_proj, o_proj])
        layers_to_transform: list[int] | int | null (default null = all layers)
    """
    from peft import LoraConfig, get_peft_model  # lazy import
    from omegaconf import OmegaConf

    r = int(lora_cfg.get("r", 16))
    alpha = float(lora_cfg.get("alpha", 2 * r))
    dropout = float(lora_cfg.get("dropout", 0.0))
    bias = str(lora_cfg.get("bias", "none"))

    target_modules = lora_cfg.get("target_modules", None)
    if target_modules is None:
        raise ValueError(
            "model.lora.target_modules must be set (e.g. [q_proj,k_proj,v_proj,o_proj])."
        )
    target_modules = list(OmegaConf.to_container(target_modules, resolve=True))

    layers_to_transform = lora_cfg.get("layers_to_transform", None)
    if layers_to_transform is not None:
        lt = OmegaConf.to_container(layers_to_transform, resolve=True)
        if isinstance(lt, list):
            layers_to_transform = [int(x) for x in lt]
        else:
            layers_to_transform = int(lt)

    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    logger.info(
        "Wrapped model with LoRA: r=%s, alpha=%s, dropout=%s, bias=%s, "
        "target_modules=%s, layers_to_transform=%s",
        r, alpha, dropout, bias, target_modules, layers_to_transform,
    )
    try:
        model.print_trainable_parameters()
    except Exception as _e:  # noqa: BLE001
        logger.warning("print_trainable_parameters() failed: %s", _e)
    return model


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
