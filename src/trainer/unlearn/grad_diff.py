import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


# #region agent log
_MEM_DEBUG_FIRST_CALL = True


def _mem_debug_log(location: str, message: str, data: dict, hypothesis: str = "") -> None:
    import json, os, time, torch
    global _MEM_DEBUG_FIRST_CALL
    payload = {
        "sessionId": "46cfae",
        "location": location,
        "message": message,
        "hypothesisId": hypothesis,
        "data": dict(data),
        "timestamp": int(time.time() * 1000),
    }
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = round(free_b / (1024**3), 2)
        total_gb = round(total_b / (1024**3), 2)
        alloc_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
        reserved_gb = round(torch.cuda.memory_reserved() / (1024**3), 2)
        # "other" memory on the GPU = total - free - what this process owns
        other_gb = round(max(total_gb - free_gb - reserved_gb, 0), 2)
        payload["data"]["cuda_alloc_gb"] = alloc_gb
        payload["data"]["cuda_reserved_gb"] = reserved_gb
        payload["data"]["cuda_max_alloc_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        payload["data"]["cuda_free_gb"] = free_gb
        payload["data"]["cuda_total_gb"] = total_gb
        payload["data"]["other_tenants_gb"] = other_gb
        if _MEM_DEBUG_FIRST_CALL:
            payload["data"]["env_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            payload["data"]["env_SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID", "<unset>")
            payload["data"]["env_SLURMD_NODENAME"] = os.environ.get("SLURMD_NODENAME", "<unset>")
            payload["data"]["torch_device_count"] = torch.cuda.device_count()
            try:
                payload["data"]["device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
            _MEM_DEBUG_FIRST_CALL = False
    print(f"[MEM_DEBUG] {location} h={hypothesis} {message} | {payload['data']}", flush=True)


def _summarize_module(m) -> dict:
    n_params = sum(p.numel() for p in m.parameters())
    n_grad_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    dtypes = {}
    for p in m.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    return {
        "type": type(m).__name__,
        "n_params_M": round(n_params / 1e6, 1),
        "n_grad_params_M": round(n_grad_params / 1e6, 1),
        "param_dtypes": {k: round(v / 1e6, 1) for k, v in dtypes.items()},
    }
# #endregion


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        # #region agent log
        _mem_debug_log(
            "grad_diff.__init__:after_super",
            "main model ready before ref_model",
            {"model_summary": _summarize_module(self.model)},
            hypothesis="H4,H5",
        )
        # #endregion
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        # #region agent log
        _mem_debug_log(
            "grad_diff._prepare_ref_model:start",
            "before deepcopy of ref_model",
            {},
            hypothesis="H1,H5",
        )
        # #endregion
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        # #region agent log
        _mem_debug_log(
            "grad_diff._prepare_ref_model:end",
            "after ref_model preparation",
            {"ref_model_summary": _summarize_module(ref_model)},
            hypothesis="H1,H5",
        )
        # #endregion
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
