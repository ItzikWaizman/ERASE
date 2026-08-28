from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff, _mem_debug_log


class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]

        # #region agent log
        retain_in = inputs["retain"]
        _mem_debug_log(
            "npo.compute_loss:entry",
            "before any forward pass",
            {
                "forget_shape": list(forget_inputs["input_ids"].shape),
                "retain_shape": list(retain_in["input_ids"].shape),
                "wrapped_model_type": type(model).__name__,
                "ref_model_type": type(self.ref_model).__name__,
            },
            hypothesis="H1,H2,H3,H4",
        )
        # #endregion

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        # #region agent log
        _mem_debug_log(
            "npo.compute_loss:after_dpo_loss",
            "after DPO loss (model + ref_model fwd on forget)",
            {},
            hypothesis="H1,H2",
        )
        # #endregion

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # #region agent log
        _mem_debug_log(
            "npo.compute_loss:after_retain_fwd",
            "after model fwd on retain (before backward)",
            {},
            hypothesis="H2,H3",
        )
        # #endregion

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
