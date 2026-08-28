"""Shared on-disk task-name builder for the ERASE runners.

``scripts/erase/run_erase.py`` calls :func:`build_rich_task_name` to compute
the ``task_name`` used as the on-disk folder under ``saves/unlearn/``.

The name encodes every hyperparameter that meaningfully affects an ERASE run
(the retained knobs of ``configs/trainer/ERASE.yaml``), so that two RUN dicts
producing different metrics never share a folder and the folder names stay
self-documenting at a glance.
"""
from __future__ import annotations


def build_rich_task_name(
    cfg: dict,
    cov_tag: str,
    *,
    default_epochs: int = 10,
    default_prefix: str = "ERASE",
) -> str:
    """Compose a self-documenting on-disk task name for a RUN dict.

    Only the hyperparameters used by ERASE are encoded; unset/default knobs
    contribute nothing so names stay compact.
    """
    epochs = cfg.get("epochs", default_epochs)
    prefix = cfg.get("task_prefix", default_prefix)

    layers_tag = "_L" + "".join(str(x) for x in cfg["layers"]) if "layers" in cfg else ""
    sched_tag = f"_{cfg['lr_scheduler']}" if "lr_scheduler" in cfg else ""
    optim_tag = f"_{cfg['optim'].lower()}" if cfg.get("optim", "sgd").lower() != "sgd" else ""

    flw_tag = (
        f"_flw{cfg['forget_loss_weight']}"
        if "forget_loss_weight" in cfg and cfg["forget_loss_weight"] != 1.0 else ""
    )
    cap_tag = f"_cap{cfg['forget_loss_max_ce']}" if cfg.get("forget_loss_max_ce", 0) else ""
    ptcap_tag = (
        f"_ptcap{cfg['forget_loss_per_token_cap']}"
        if cfg.get("forget_loss_per_token_cap", 0) else ""
    )
    atgt_tag = (
        f"_atgt{cfg['forget_loss_answer_target']}"
        if cfg.get("forget_loss_answer_target", 0) else ""
    )

    author_tag = "_authoronly" if cfg.get("author_only_vjp", False) else ""
    _amm = cfg.get("author_mask_mode", "token_set")
    amm_tag = f"_amm{_amm}" if _amm != "token_set" else ""
    renorm_tag = "_renorm" if cfg.get("vjp_renormalize", False) else ""

    _bs = cfg.get("per_device_train_batch_size", 4)
    _gas = cfg.get("gradient_accumulation_steps", 2)
    ebs_tag = (
        f"_ebs{_bs * _gas}"
        if "per_device_train_batch_size" in cfg or "gradient_accumulation_steps" in cfg
        else ""
    )
    det_tag = "_det" if cfg.get("deterministic", False) else ""
    seed_tag = f"_seed{cfg['seed']}" if "seed" in cfg else ""

    _ts = str(cfg.get("train_scope", "down_proj_only")).lower()
    scope_tag = f"_scope{_ts}" if _ts != "down_proj_only" else ""

    dyn_tag = ""
    _dyn_thr = cfg.get("dynamic_stop_loss_threshold", 0.0)
    if _dyn_thr and float(_dyn_thr) > 0:
        dyn_tag = f"_dyn{_dyn_thr}"
        _dyn_up = cfg.get("dynamic_stop_log_upper", 0.0)
        if _dyn_up and float(_dyn_up) > 0:
            dyn_tag += f"U{_dyn_up}"

    steps_tag = f"_s{cfg['max_steps']}" if "max_steps" in cfg else ""
    gnorm_tag = f"_gn{cfg['max_grad_norm']}" if "max_grad_norm" in cfg else ""

    return (
        f"{prefix}_{epochs}ep_{cov_tag}_lr{cfg['lr']}"
        f"_a{cfg['alpha']}_k{cfg['topk']}{flw_tag}{cap_tag}{ptcap_tag}{atgt_tag}"
        f"{sched_tag}{layers_tag}{steps_tag}{gnorm_tag}{optim_tag}"
        f"{author_tag}{amm_tag}{renorm_tag}{ebs_tag}{det_tag}{scope_tag}{dyn_tag}{seed_tag}"
    )
