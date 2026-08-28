import torch.nn.functional as F

from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff, _mem_debug_log


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]

        # #region agent log
        retain_in = inputs["retain"]
        _mem_debug_log(
            "simnpo.compute_loss:entry",
            "before any forward pass",
            {
                "forget_shape": list(forget_inputs["input_ids"].shape),
                "retain_shape": list(retain_in["input_ids"].shape),
                "wrapped_model_type": type(model).__name__,
            },
            hypothesis="H2,H3,H4",
        )
        # #endregion

        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)

        # #region agent log
        _mem_debug_log(
            "simnpo.compute_loss:after_forget_fwd",
            "after forward pass on forget",
            {},
            hypothesis="H2",
        )
        # #endregion

        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # #region agent log
        _mem_debug_log(
            "simnpo.compute_loss:after_retain_fwd",
            "after forward pass on retain (before backward)",
            {},
            hypothesis="H2,H3",
        )
        # #endregion

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
