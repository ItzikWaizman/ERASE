"""
ERASE: Precise Gradient Ascent for machine unlearning (Llama MLP down_proj).

Gradient projection:
    ∇W_proj = ∇W @ P.T   with   P = (I + alpha * C_retain)^{-1}.

C matrices are precomputed per-layer (see scripts/erase/compute_covariances.py).
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import logging
import torch
import torch.nn.functional as F
from torch.optim import SGD
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


def build_projection_matrix(c_retain: torch.Tensor, alpha: float) -> torch.Tensor:
    d = c_retain.shape[0]
    I = torch.eye(d, dtype=c_retain.dtype, device=c_retain.device)
    A = I + alpha * c_retain
    return torch.linalg.solve(A, I)


class ERASE(UnlearnTrainer):
    def __init__(
        self,
        covariance_dir: str,
        target_layers: list,
        alpha: float = 1.0,
        forget_loss_weight: float = 1.0,
        forget_loss_max_ce: float = 0.0,
        forget_loss_per_token_cap: float = 0.0,
        forget_loss_answer_target: float = 0.0,
        # Per-sample target JITTER (std). When > 0, each forget sample gets its
        # OWN MSE target drawn (stable, keyed) from N(forget_loss_answer_target,
        # this_std), clamped to +/- 2.5 std. This injects the oracle's natural
        # spread so the unlearned forget-CE distribution matches N(mean, std)
        # instead of collapsing to a single-tau spike (fixes the narrow-variance
        # privacy gap). Active only for forget_loss_answer_mode == "mse".
        forget_loss_answer_target_std: float = 0.0,
        # #1 Per-token CE CEILING: tokens whose (detached) CE already exceeds
        # this are EXCLUDED from the forget push (mask out of the loss), so the
        # hardest/most-surprising tokens stop being driven past ~retain level.
        # Absolute proxy for "as surprising as retain" (we can't see retain
        # per-token at train time). 0 = off. Targets the MinK++ over-forget.
        forget_loss_token_ce_ceiling: float = 0.0,
        forget_loss_answer_mode: str = "sigmoid",   # "sigmoid" | "mse"
        forget_loss_band_lower: float = 2.0,
        forget_loss_band_upper: float = 4.0,
        forget_loss_type: str = "ce",          # "ce" (CE-max)
        # Auxiliary entropy regularizer (composable with the CE-up forget path).
        # Adds + reg_weight * (H_target - H_forget)^2 to the loss using the
        # mean per-token entropy of the model's distribution at FORGET answer
        # positions. Pushes forget distributions toward retain-like spread to
        # avoid the "confidently wrong" attractor that hurts MIN-K++ alignment.
        # 0 = off. Active only when forget_loss_type == "ce".
        forget_entropy_reg_weight: float = 0.0,
        forget_entropy_reg_target: float = 0.0,
        topk_vjp_count: int = 0,
        author_only_vjp: bool = False,
        # Restrict the down_proj VJP reconstruction to answer (labelled)
        # positions only, i.e. positions where the forget batch has
        # labels != -100. Used by the MUSE recipe ("no VJP entity filtering":
        # the whole supervised chunk is the answer). Ignored when
        # author_only_vjp or topk_vjp_count > 0 take priority. False = the
        # full autograd down_proj gradient (all positions) is projected by P.
        answer_only_down_proj_grad: bool = False,
        author_mask_mode: str = "token_set",
        forget_span_cache: str = "",
        vjp_renormalize: bool = False,
        # Per-sample dynamic stopping. When the LENGTH-NORMALIZED mean CE
        # on a sample's answer tokens reaches `dynamic_stop_loss_threshold`,
        # the sample contributes 0 to the forget loss (its gradient is
        # masked out, no further ascent). This replaces the batch-mean
        # `forget_loss_max_ce` and per-position `forget_loss_per_token_cap`
        # clamps with a per-sample "stop when forgotten" rule. Stateless:
        # recomputed every step from the current per-sample CE, so a
        # sample can re-enter the active pool if its CE drifts below the
        # threshold (rare during ascent). 0.0 = disabled (legacy
        # bit-identical). When > 0, both legacy caps are IGNORED for the
        # forget loss (still applied to logging only).
        dynamic_stop_loss_threshold: float = 0.0,
        # Informational-only upper threshold for the dynamic-stop logger.
        # Per step we count samples whose mean-CE exceeds this; useful for
        # monitoring whether the stop threshold keeps samples inside a
        # target band. Has NO effect on gradients. 0.0 = no logging.
        dynamic_stop_log_upper: float = 0.0,
        # Long-tail patience: when ``remaining < dynamic_stop_longtail_threshold``
        # AND a single active sample has appeared in MORE THAN this many
        # OPTIMISER STEPS (deduplicated by global_step, so the same key
        # cannot tick the counter multiple times in one optimiser step
        # even if the sampler-level filter has stuffed several copies into
        # the same micro-batch), the WHOLE run is stopped. 0 = disabled.
        dynamic_stop_max_active_steps_per_sample: int = 0,
        # The "<N remaining" gate that activates long-tail patience.
        # Patience is suppressed whenever ``remaining >= this value`` so
        # the rule only fires on the resistant tail, not during normal
        # mid-training where many samples are still active. 0 = patience
        # always active (legacy behaviour). Default 10 matches the user
        # spec ("if we have <10 remaining ...").
        dynamic_stop_longtail_threshold: int = 10,
        # When the remaining-sample count drops to this value or below, the
        # trainer decodes and logs each stuck sample's input + target so
        # we can see WHAT is resistant. Decoded once per sample (cached).
        # 0 = no decoding. Default 3 surfaces the long-tail samples without
        # spamming the log on early-training batches.
        dynamic_stop_decode_threshold: int = 3,
        # Probability [0,1] of re-sampling a "done" sample. When > 0, done
        # samples are NOT permanently removed from the sampler; instead they
        # remain available with reduced probability. Training stops once ALL
        # forget samples have entered the band at least once (i.e. the "done"
        # set reaches full size). 0.0 = legacy behavior (remove permanently,
        # never re-sample). Typical value: 0.2 (20% chance per draw).
        dynamic_stop_done_sample_prob: float = 0.0,
        # Retain-CE monitor: cheap early-stop gate. Every
        # `retain_ce_monitor_interval` optimiser steps, compute the mean CE
        # on a small cached retain mini-batch (no grad, ~2s). If it exceeds
        # `retain_ce_stop_threshold`, halt training. The threshold is
        # ABSOLUTE CE (e.g. 2.2 nats); set it to the target model's initial
        # retain CE + a delta (measured empirically). 0 = disabled.
        retain_ce_stop_threshold: float = 0.0,
        retain_ce_monitor_interval: int = 25,
        # --- Signal monitor + automatic stop (the DELIVERABLE early-stop) ---
        # Every `signal_monitor_interval` optimiser steps (0 = off), evaluate a
        # FIXED forget probe and a FIXED retain probe (teacher-forced CE, no
        # grad, no generation) and append a rich record of *candidate* trigger
        # signals to a JSONL (telemetry, used offline to design better stops).
        # The stop itself uses ONLY forget/retain (never MIA/oracle/holdout):
        #   * stop_forget_ce_target > 0: stop once the forget-probe mean CE has
        #     risen to the "natural" level (>= target) for `stop_patience`
        #     consecutive monitor hits -> forget text now looks un-memorised.
        #   * stop_retain_ce_rise > 0: COLLAPSE GUARD -- stop the instant the
        #     retain-probe mean CE rises `stop_retain_ce_rise` nats above its
        #     step-0 baseline (retain starting to degrade; fires before the
        #     ROUGE cliff because CE is continuous).
        # When a stop fires, the final saved model IS the reported result (no
        # post-hoc best-checkpoint picking), so the comparison to baselines is
        # apples-to-apples.
        signal_monitor_interval: int = 0,
        signal_monitor_probe_size: int = 4,
        signal_monitor_log: str = "",
        # --- Full-forget-set per-token CE dump (offline stop-signal mining) ---
        # Every `token_ce_dump_epochs` epochs (0 = off), run a no-grad forward
        # over the ENTIRE forget set and save each sample's per-token CE
        # (fp16) to <output_dir>/token_ce_dumps/ep..._step....npz. A dump also
        # fires at epoch ~0 (baseline anchor). On MUSE (407 x 2048-token
        # chunks) one dump is ~1-2 min on an A100; every 2 epochs over 40
        # epochs adds ~30-40 min total. The offline analysis correlates the
        # evolving token-CE distributions with the checkpoint evals to find
        # the collapse precursor.
        token_ce_dump_epochs: float = 0.0,
        token_ce_dump_max_samples: int = 0,  # 0 = all forget samples
        stop_forget_ce_target: float = 0.0,
        stop_retain_ce_rise: float = 0.0,
        stop_monitor_warmup_steps: int = 0,
        stop_patience: int = 2,
        # --- Checkpoint-only stop mode (signal validation without censoring) ---
        # When True, a firing stop rule does NOT terminate training. Instead the
        # trainer (once per rule kind) saves the current weights to
        # checkpoint-{step}-earlystop-{kind} with an EARLYSTOP_SIGNAL.json
        # marker and runs an immediate eval at that exact step, then training
        # continues to the budget. This lets us verify offline whether the
        # signal landed on the sweet spot (fair-comparison candidate) while
        # still observing the full post-signal trajectory.
        stop_signal_checkpoint_only: bool = False,
        # Deep-collapse bail-out: a REAL stop (even in checkpoint-only mode)
        # once the retain-probe CE has risen this many nats above baseline.
        # Set well past the collapse-guard signal level (e.g. 3.0 vs 1.5) so
        # it only fires when the model is unrecoverable and further compute
        # is wasted. 0 = off.
        hard_stop_retain_ce_rise: float = 0.0,
        train_scope: str = "down_proj_only",
        *args,
        **kwargs,
    ):
        """
        Args:
            covariance_dir: Directory with C_retain .pt files per layer.
            target_layers: Layer indices (Llama model.layers[i].mlp.down_proj).
            alpha: Projection hyperparameter.
            topk_vjp_count: If > 0, select top-K VJPs by norm (used when
                author_only_vjp is False).
            forget_loss_per_token_cap: Per-position upper bound on the forget CE
                BEFORE averaging. 0 (default) = OFF (preserves all existing
                behaviour). When > 0, each per-token CE is clamped to this cap;
                clamp's gradient is 0 above the cap, so once a position is
                "forgotten enough" no further gradient flows to it. Prevents the
                runaway in which a few subject positions accumulate 30+ nats and
                crash the model into degenerate output (Mode-2 collapse, e.g. the
                "spell spell..." failure on Author_block_13). Diagnostic data
                showed GOOD authors had per-token Δ CE < 18 nats while collapsed
                authors had every subject position above 30; a cap in [10, 20]
                preserves GOOD-author dynamics while halting the collapse.
            forget_loss_answer_target: Per-question BOUNDED forget objective.
                0 (default) = OFF. When > 0, replaces the standard "ascend CE"
                loss with a saturating sigmoid push toward a target per-token
                NLL. Specifically, for each sample we compute the length-
                normalized mean CE over all answer positions (after any
                per-token cap), then the loss is
                    L = - forget_loss_weight * mean_batch[ sigmoid(
                          mean_answer_CE_per_sample - forget_loss_answer_target
                        )]
                This gives us a BOUNDED objective (L in [-weight, 0]) that
                naturally converges: once mean_answer_CE passes the target,
                sigmoid saturates to 1 and gradient -> 0 per sample. The
                target is in units of "nats per answer token" (length-
                invariant thanks to the normalization), so typical values
                are in [2, 5] depending on how deep you want to forget.
                Composable with forget_loss_per_token_cap (per-token cap is
                applied BEFORE the mean). With
                forget_loss_answer_mode="mse", the sigmoid gate is replaced
                by the two-sided squared pull (CET loss)
                    L = - forget_loss_weight * mean_batch[
                          (mean_answer_CE_per_sample - target)^2 ]
            vjp_renormalize: When True, rescale the VJP-filtered gradient so
                its magnitude is independent of how many positions were
                selected.  The CE loss is mean-reduced over *all* m answer
                tokens, so each δ_i carries a 1/m factor.  When only |S|
                positions survive the author/topk filter, the raw sum
                Σ_{i∈S} δ_i x_i^T is implicitly scaled by |S|/m compared
                to using all answer tokens.  This flag multiplies
                filtered_forget by m/|S| to undo that coupling, making
                every question contribute equally regardless of the
                number of author tokens matched.  Default False preserves
                all pre-existing behaviour.
        """
        super().__init__(*args, **kwargs)
        self.target_layers = list(target_layers)
        self.alpha = alpha
        self.topk_vjp_count = int(topk_vjp_count)
        self.author_only_vjp = bool(author_only_vjp)
        self.answer_only_down_proj_grad = bool(answer_only_down_proj_grad)
        self.vjp_renormalize = bool(vjp_renormalize)
        # "token_set" = legacy flat token-ID membership (backward compat),
        # "span"      = contiguous span matching (precise, no subword leaks),
        # "off"       = disable author mask entirely.
        self.author_mask_mode = str(author_mask_mode).lower()
        if self.author_mask_mode not in ("token_set", "span", "off"):
            raise ValueError(
                f"author_mask_mode must be 'token_set', 'span', or 'off', "
                f"got {author_mask_mode!r}"
            )
        # Path to a precomputed forget-span cache (JSON, same shape as the TOFU
        # author spans: {"spans": {key: [token_id_seq, ...]}}). When set AND
        # author_mask_mode="span", the VJP mask is built from this cache instead
        # of the TOFU author extractor -- this is how NER / QA-keyword VJP
        # localization is enabled on raw-text benchmarks (MUSE). Coverage is
        # logged and the "zero author matches" hard-fail guard is disabled in
        # this mode.
        self.forget_span_cache = str(forget_span_cache)
        self._forget_span_first_tok_index: dict | None = None
        self._hooks_registered = False
        self._hook_x: dict[int, torch.Tensor] = {}
        self._hook_delta: dict[int, torch.Tensor] = {}
        # Per-step answer-position mask (labels != -100) for the forget batch,
        # built in training_step when answer_only_down_proj_grad is active.
        self._answer_mask: torch.Tensor | None = None
        # Legacy flat token-ID set (mode="token_set"). Lazy-built.
        self._author_token_ids: set[int] | None = None
        # Span-based author name sequences (mode="span"). Lazy-built.
        self._author_name_spans: dict[str, list[list[int]]] | None = None

        self._needs_hooks = (
            self.topk_vjp_count > 0
            or self.author_only_vjp
            or self.answer_only_down_proj_grad
        )

        self.forget_loss_weight = float(forget_loss_weight)
        # Cap on per-batch forget CE loss. <= 0 disables the cap. When the
        # batch-MEAN CE exceeds the cap, we clamp it (gradient becomes 0
        # beyond the cap), which prevents runaway gradient ascent on already-
        # forgotten examples and tames the noisy CE curve.
        self.forget_loss_max_ce = float(forget_loss_max_ce)
        # Per-position CE clamp applied BEFORE averaging. Independent of
        # forget_loss_max_ce (which acts on the post-average scalar). 0 = off.
        self.forget_loss_per_token_cap = float(forget_loss_per_token_cap)
        # Per-question BOUNDED forget objective via a length-normalized
        # sigmoid push: L = -w * mean_batch[sigmoid(mean_answer_CE - target)].
        # 0 = off (default). See docstring above for full behaviour.
        self.forget_loss_answer_target = float(forget_loss_answer_target)
        self.forget_loss_answer_target_std = float(forget_loss_answer_target_std)
        self.forget_loss_token_ce_ceiling = float(forget_loss_token_ce_ceiling)
        # Per-sample jittered target cache: stable key -> tau_i (deterministic).
        self._dyn_sample_targets: dict = {}
        self.forget_loss_answer_mode = forget_loss_answer_mode
        self.forget_loss_band_lower = float(forget_loss_band_lower)
        self.forget_loss_band_upper = float(forget_loss_band_upper)
        # Forget loss formulation:
        #   "ce" -> maximize cross-entropy (push p(target) -> 0).
        #           Bounded by forget_loss_max_ce when set.
        self.forget_loss_type = str(forget_loss_type).lower()
        if self.forget_loss_type != "ce":
            raise ValueError(
                f"forget_loss_type must be 'ce', got {forget_loss_type!r}"
            )
        # Auxiliary entropy regularizer (composable with CE-up). When weight > 0
        # and target > 0, adds w * (target - mean_t H(p_t))^2 on FORGET answer
        # positions, where p_t is the per-step model distribution. This pulls
        # the full softmax toward retain-like entropy, fighting the "peaked
        # wrong" attractor (one runner-up takes all displaced mass) that
        # CE-up + cap alone do not prevent.
        self.forget_entropy_reg_weight = float(forget_entropy_reg_weight)
        self.forget_entropy_reg_target = float(forget_entropy_reg_target)
        self._last_forget_z_p10: float | None = None
        self._last_forget_z_mean: float | None = None

        self.p_matrices = {}
        self._load_p_matrices(covariance_dir)

        self._last_retain_entropy: float | None = None
        self._last_forget_entropy: float | None = None
        self._last_forget_ce_loss: float | None = None
        self._last_forget_ce_raw: float | None = None

        # --- "why aren't the weights moving?" diagnostics (long-sequence aware) ---
        # Per-TOKEN forget-CE distribution (vs the per-sample MEAN the loss uses),
        # the fraction of answer tokens below/inside/above the target band, the
        # mean supervised sequence length T (the 1/T gradient-dilution factor on
        # MUSE's 2048-tok chunks), and how far the trainable weights have actually
        # moved from init. All are best-effort and never affect the loss.
        self._last_tok_ce_p10: float | None = None
        self._last_tok_ce_p50: float | None = None
        self._last_tok_ce_p90: float | None = None
        self._last_tok_frac_below: float | None = None   # < band_lower
        self._last_tok_frac_in: float | None = None       # within [lower, upper]
        self._last_tok_frac_above: float | None = None    # > band_upper
        self._last_mean_seq_len: float | None = None       # mean answer tokens / sample
        self._last_weight_delta: float | None = None       # ||W - W0|| over trainable
        self._last_weight_rel_delta: float | None = None   # ||W - W0|| / ||W0||
        self._w0_norm_sq_cache: float | None = None        # ||W0||^2 (lazy)
        self._w0_cache: dict | None = None                 # name -> W0 (lazy clone)

        self.dynamic_stop_loss_threshold = float(dynamic_stop_loss_threshold)
        self.dynamic_stop_log_upper = float(dynamic_stop_log_upper)
        self.dynamic_stop_max_active_steps_per_sample = int(
            dynamic_stop_max_active_steps_per_sample
        )
        self.dynamic_stop_decode_threshold = int(dynamic_stop_decode_threshold)
        self.dynamic_stop_done_sample_prob = float(
            dynamic_stop_done_sample_prob
        )
        self.retain_ce_stop_threshold = float(retain_ce_stop_threshold)
        self.retain_ce_monitor_interval = int(retain_ce_monitor_interval)
        self._retain_ce_monitor_cache = None  # cached retain mini-batch (set on first use)
        self._retain_ce_baseline = None       # initial retain CE at step 0 (for logging)
        self._last_retain_ce_monitor = float("nan")

        # --- Signal monitor + automatic stop state ---
        self.signal_monitor_interval = int(signal_monitor_interval)
        self.signal_monitor_probe_size = int(signal_monitor_probe_size)
        self.signal_monitor_log = str(signal_monitor_log)
        # --- Full-forget-set token-CE dump state ---
        self.token_ce_dump_epochs = float(token_ce_dump_epochs)
        self.token_ce_dump_max_samples = int(token_ce_dump_max_samples)
        self._token_ce_last_dump_epoch: float = float("-inf")
        # Baselines for the drift/agreement signals (cached at first monitor):
        # per-token CE and top-1 argmax of each probe at ~step 0.
        self._probe_base: dict = {}  # {"forget"/"retain": {"tok_ce": [...], "argmax": [...]}}
        self.stop_forget_ce_target = float(stop_forget_ce_target)
        self.stop_retain_ce_rise = float(stop_retain_ce_rise)
        self.stop_monitor_warmup_steps = int(stop_monitor_warmup_steps)
        self.stop_patience = max(1, int(stop_patience))
        self.stop_signal_checkpoint_only = bool(stop_signal_checkpoint_only)
        self.hard_stop_retain_ce_rise = float(hard_stop_retain_ce_rise)
        self._signal_ckpt_fired: set = set()   # rule kinds already checkpointed
        self._pending_signal_ckpt = None       # (kind, reason) deferred to a safe point
        # Fixed probes (accumulated over the first few monitor calls, then frozen)
        self._probe_forget: list = []          # list of {input_ids, attention_mask, labels} (CPU)
        self._probe_retain: list = []
        self._probe_frozen: bool = False
        self._probe_forget_baseline: float | None = None   # step-~0 forget probe mean CE
        self._probe_retain_baseline: float | None = None   # step-~0 retain probe mean CE
        self._monitor_forget_hits: int = 0     # consecutive "forget reached target" hits
        self._signal_log_path = None           # resolved lazily (needs output_dir)
        self._last_grad_norm: float | None = None
        self._grad_norm_ema: float | None = None
        self._last_signal_record: dict | None = None
        self.dynamic_stop_longtail_threshold = int(
            dynamic_stop_longtail_threshold
        )
        # Per-key map of "last optimiser step at which we ticked the active
        # counter". Used to dedupe duplicate appearances of the same key
        # within a single optimiser step (the swap-based sampler filter
        # routinely places 4 copies of the lone active sample in one
        # micro-batch x 2 micro-batches per optimiser step).
        self._dyn_last_active_step: dict = {}
        # Stateful per-sample "done" set. Each element is a hashable key
        # built from (input_ids, labels) of a forget sample (padding
        # stripped via attention_mask). Once a sample's mean-CE crosses
        # `dynamic_stop_loss_threshold` it is added here and excluded from
        # the forget loss FOR THE REST OF TRAINING -- even if its CE later
        # drifts back below the threshold.
        self._dyn_done_keys: set = set()
        # Per-sample count of "active" gradient steps (sample in batch and
        # still not in done set). Used by the patience knob.
        self._dyn_active_steps: dict = {}
        # Per-sample decoded-log dedup: keys we've already printed once.
        self._dyn_decoded_keys: set = set()
        # SAMPLER-LEVEL filtering plumbing (the user-facing semantic: a
        # done sample is *removed from the dataloader*, not just masked
        # in compute_loss). Populated lazily on the first compute_loss
        # call so that init has no extra cost when dynamic_stop is off.
        # _dyn_idx_to_key[i]   = key derived from forget_dataset[i]
        # _dyn_key_to_idx[k]   = inverse map (one entry per unique key)
        # _dyn_active_indices  = SHARED set of still-active forget indices,
        #                        also handed to ForgetRetainDataset so its
        #                        __getitem__ swaps out done samples for
        #                        random active ones on the fly. None until
        #                        bootstrapped.
        self._dyn_idx_to_key = None
        self._dyn_key_to_idx = None
        self._dyn_active_indices = None
        self._dyn_n_total: int = -1   # lazy-resolved on first batch
        self._dyn_early_stop_logged: bool = False
        # Rolling per-step counters for TB. -1/NaN = "off".
        self._last_n_active = -1
        self._last_n_done = -1                # done IN THIS BATCH
        self._last_n_done_total = -1          # cumulative across all batches
        self._last_n_overshoot = -1
        self._last_max_per_sample_ce = float("nan")
        # VJP masking diagnostics (-1 = off / not a masked variant). These make
        # the ner/qa "did the entity mask actually match, or did we silently
        # fall back to the whole-chunk mask?" signal visible in trainer_state.json.
        self._last_span_coverage = -1.0       # frac of rows with >=1 matched span
        self._last_vjp_selected_pct = float("nan")  # % of positions kept for VJP
        # Scope of the trainable parameter set. The spectral filter P
        # (and any VJP filtering: author_only_vjp / topk_vjp_count) is
        # ALWAYS applied only to the down_proj.weight gradient at
        # target_layers.
        #
        #   "down_proj_only"     (default; legacy): only down_proj at
        #                        target_layers is trainable. All other
        #                        params frozen. Bit-identical to all
        #                        pre-existing ERASE runs.
        valid_scopes = ("down_proj_only",)
        if str(train_scope).lower() not in valid_scopes:
            raise ValueError(
                f"train_scope must be one of {valid_scopes}, got {train_scope!r}"
            )
        self.train_scope = str(train_scope).lower()

        logger.info(
            f"ERASE (Llama down_proj): layers={self.target_layers}, alpha={alpha}, "
            f"topk_vjp={self.topk_vjp_count}, author_mask_mode={self.author_mask_mode}, "
            f"vjp_renormalize={self.vjp_renormalize}, "
            f"forget_loss_per_token_cap={self.forget_loss_per_token_cap}, "
            f"forget_loss_answer_target={self.forget_loss_answer_target}, "
            f"forget_loss_answer_mode={self.forget_loss_answer_mode}, "
            f"forget_loss_band=[{self.forget_loss_band_lower}, {self.forget_loss_band_upper}], "
            f"forget_entropy_reg(weight={self.forget_entropy_reg_weight}, "
            f"target={self.forget_entropy_reg_target}), "
            f"train_scope={self.train_scope}"
        )

    def _load_p_matrices(self, covariance_dir: str):
        for layer_idx in self.target_layers:
            c_retain = torch.load(
                os.path.join(covariance_dir, f"C_retain_layer_{layer_idx}.pt"),
                map_location="cpu",
                weights_only=True,
            )
            self.p_matrices[layer_idx] = build_projection_matrix(
                c_retain, self.alpha
            )
        logger.info(
            f"P matrices: {len(self.p_matrices)} layers (alpha={self.alpha})"
        )

    def _unwrap_model(self, model=None):
        m = model if model is not None else self.model
        while hasattr(m, "module"):
            m = m.module
        return m

    def _get_down_proj(self, model, layer_idx):
        """Return the Llama MLP down_proj module at ``layer_idx``."""
        return model.model.layers[layer_idx].mlp.down_proj

    def _ensure_hooks(self, model):
        if self._hooks_registered:
            return
        base = self._unwrap_model(model)
        for layer_idx in self.target_layers:
            mod = self._get_down_proj(base, layer_idx)
            li = int(layer_idx)

            def _fw_hook(module, inp, out, li=li):
                if li in self._hook_x:
                    return
                if inp[0] is not None:
                    self._hook_x[li] = inp[0].detach()
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    def _tensor_bw(grad, _li=li):
                        self._hook_delta[_li] = grad.detach()
                    out.register_hook(_tensor_bw)

            mod.register_forward_hook(_fw_hook)
        self._hooks_registered = True
        logger.info(
            "ERASE: forward hooks on down_proj (forget VJP capture; backward hooks on outputs)"
        )

    def create_optimizer(self):
        # Always start from a fully-frozen model and selectively unfreeze
        # by `train_scope`. The spectral filter P is independent of this:
        # it always applies only to the down_proj.weight gradient at
        # `target_layers` (in training_step).
        for param in self.model.parameters():
            param.requires_grad = False

        if self.train_scope == "down_proj_only":
            for layer_idx in self.target_layers:
                self._get_down_proj(self.model, layer_idx).weight.requires_grad = True
            scope_msg = "down_proj weights only"
        else:
            # Defensive: __init__ already validated, but keep parity.
            raise ValueError(f"unknown train_scope={self.train_scope!r}")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"ERASE: train_scope={self.train_scope} -> {scope_msg}; "
            f"{len(trainable)} trainable tensors, "
            f"{n_trainable:,} / {n_total:,} params "
            f"({100.0 * n_trainable / max(1, n_total):.1f}%). "
            f"Spectral filter P applied to down_proj at layers "
            f"{self.target_layers}."
        )

        wd = getattr(self.args, "weight_decay", 0.0)
        lr = self.args.learning_rate
        self.optimizer = SGD(
            trainable,
            lr=lr,
            momentum=0.0,
            weight_decay=wd,
        )
        return self.optimizer

    def _tokenizer_cache_tag(self) -> str:
        """Filename-safe tag identifying the active tokenizer.

        Author-name token-ID caches MUST be tokenizer-specific: the same
        author string tokenizes to completely different IDs across model
        families (Llama-2 vocab 32k vs Llama-3 vocab 128k), and a stale
        1B-built cache silently reused on 7B masks zero tokens, zeroing
        the projected forget gradient. We key the cache file by
        (vocab_size, basename(name_or_path)) which is unique per tokenizer
        in practice while staying stable across runs of the same model.
        """
        tok = self.processing_class
        name = getattr(tok, "name_or_path", "") or ""
        # Take the last path component and sanitize for use as a filename.
        base = name.replace("\\", "/").rstrip("/").split("/")[-1] or "tokenizer"
        safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in base)
        return f"v{int(getattr(tok, 'vocab_size', 0))}_{safe}"

    def _build_author_name_spans(self) -> dict[str, list[list[int]]]:
        """Extract TOFU forget10 author names and build span-based token seqs.

        Instead of a flat set of token-IDs (which causes massive false-positive
        hits on common subwords like ',', '.', 'a', 's'), we store the full
        token-ID *sequence* for each name variant.  At mask-build time we do a
        sliding-window match so only contiguous spans that form a real author
        name are marked True.

        Returns ``{name_str: [seq1, seq2, …]}`` where each ``seq`` is a
        ``list[int]`` of token-IDs for one contextual variant of the name.
        """
        import json as _json
        import re as _re
        from pathlib import Path as _Path

        cache_dir = _Path(__file__).resolve().parents[3] / "saves" / "precompute"
        tag = self._tokenizer_cache_tag()
        cache_path = cache_dir / f"forget10_author_name_spans__{tag}.json"
        if cache_path.is_file():
            try:
                cached = _json.loads(cache_path.read_text(encoding="utf-8"))
                spans: dict[str, list[list[int]]] = cached["spans"]
                total_seqs = sum(len(v) for v in spans.values())
                logger.info(
                    f"Loaded author name spans from cache for tokenizer "
                    f"'{tag}': {len(spans)} names, {total_seqs} token sequences"
                )
                return spans
            except (KeyError, _json.JSONDecodeError):
                pass

        try:
            from datasets import load_dataset
            ds = load_dataset("locuslab/TOFU", "forget10", split="train")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load TOFU forget10 for author extraction: {e}")
            return {}

        # TOFU forget10: 20 authors × 20 Qs = 400 rows.  The first Q in
        # each block of 20 always asks "What is the full name of …?" and
        # the answer contains the author's name.  We use multiple regex
        # patterns to cover the varied answer formats, then fall back to
        # frequency-based extraction with a blocklist of common non-name
        # phrases that the regex would otherwise pick up.
        names: set[str] = set()
        _CAP_NAME = r"([A-Z][\w\-'\.]*(?:\s+[A-Z][\w\-'\.]*){0,4})"
        _name_patterns = [
            r"(?:author'?s?\s+)?(?:full\s+)?name\s+is\s+" + _CAP_NAME,
            r"^" + _CAP_NAME + r"\s+is\s+the\s+(?:author|fictitious)",
            r"is\s+(?:named|called)\s+" + _CAP_NAME,
            r"is\s+" + _CAP_NAME + r"[,.]",
            r"^" + _CAP_NAME + r",\s+(?:known|a\s+)",
        ]
        for i in range(0, len(ds), 20):
            ans = ds[i]["answer"]
            for pat in _name_patterns:
                for m in _re.finditer(pat, ans):
                    nm = m.group(1).strip().rstrip(".,;:")
                    if nm:
                        names.add(nm)

        if len(names) < 10:
            from collections import Counter
            counter = Counter()
            cap_pat = _re.compile(r"\b([A-Z][\w\-'\.]+(?:\s+[A-Z][\w\-'\.]+){1,3})\b")
            for ex in ds:
                for m in cap_pat.finditer(ex["answer"]):
                    counter[m.group(1).strip().rstrip(".,;:")] += 1
            for nm, cnt in counter.items():
                if cnt >= 5:
                    names.add(nm)

        # Remove common non-person-name phrases that the fallback regex
        # picks up from TOFU answer text.
        _NOT_AUTHOR_NAMES = {
            "African American", "South Africa", "South Korea",
            "Middle Eastern", "Contemporary Romance", "Love Inspired",
            "Star Wars", "In Night's Silence", "Stars Will Be Our",
            "Ababa", "Addis Ababa", "New York", "Baghdad",
            "Kuwait City", "Cape Town", "Santiago", "Manama",
            "Tehran", "Dhaka", "Tokyo", "Taipei", "Seoul",
            "Karachi", "Beijing", "Beirut", "Astana", "Baku",
        }
        names -= _NOT_AUTHOR_NAMES
        # Also strip possessive forms — we'll match the base name and its
        # possessive separately so the mask covers "Nakamura's" too.
        base_names = {n.rstrip("'s") if n.endswith("'s") else n for n in names}
        names = set()
        for n in base_names:
            names.add(n)
            names.add(n + "'s")

        # Also add individual name parts (first, middle, last) so we match
        # partial references like "Nakamura" or "Behrouz" used alone.
        all_parts: set[str] = set()
        for name in sorted(names):
            if name.endswith("'s"):
                continue
            for part in name.split():
                if len(part) >= 3:
                    all_parts.add(part)
                    all_parts.add(part + "'s")

        tok = self.processing_class
        spans: dict[str, list[list[int]]] = {}
        all_variants = sorted(names | all_parts)
        for name in all_variants:
            is_part = name in all_parts
            seqs: list[list[int]] = []
            seen: set[tuple[int, ...]] = set()
            for variant in (name, " " + name, "\n" + name):
                ids = tok(variant, add_special_tokens=False).get("input_ids", [])
                ids = [int(t) for t in ids]
                # Full names require len>=2 (avoids punctuation);
                # individual parts allow len>=1 so single-token names
                # like " Park" (token 5657) are matched.
                min_len = 1 if is_part else 2
                if len(ids) >= min_len:
                    key = tuple(ids)
                    if key not in seen:
                        seen.add(key)
                        seqs.append(ids)
            if seqs:
                spans[name] = seqs

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps({
            "author_names": sorted(names),
            "spans": spans,
        }, indent=2), encoding="utf-8")
        total_seqs = sum(len(v) for v in spans.values())
        logger.info(
            f"Built author name spans: {len(spans)} names, {total_seqs} token sequences "
            f"(cached at {cache_path})"
        )
        return spans

    # ------------------------------------------------------------------
    # Legacy flat token-ID set (author_mask_mode="token_set")
    # ------------------------------------------------------------------
    def _build_author_token_ids(self) -> set[int]:
        """Legacy: flat set of token-IDs for author-name subwords.

        Kept for backward compatibility (``author_mask_mode='token_set'``).
        Matches ~27% of forget tokens due to common subword collisions —
        prefer ``author_mask_mode='span'`` for precise matching.
        """
        import json as _json
        import re as _re
        from pathlib import Path as _Path

        cache_dir = _Path(__file__).resolve().parents[3] / "saves" / "precompute"
        tag = self._tokenizer_cache_tag()
        cache_path = cache_dir / f"forget10_author_token_ids__{tag}.json"
        if cache_path.is_file():
            try:
                cached = _json.loads(cache_path.read_text(encoding="utf-8"))
                logger.info(
                    f"Loaded legacy author tokens from cache for tokenizer "
                    f"'{tag}': {len(cached['author_names'])} names -> "
                    f"{len(cached['token_ids'])} unique token-ids"
                )
                return set(cached["token_ids"])
            except (KeyError, _json.JSONDecodeError):
                pass

        try:
            from datasets import load_dataset
            ds = load_dataset("locuslab/TOFU", "forget10", split="train")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load TOFU forget10 for author extraction: {e}")
            return set()

        names: set[str] = set()
        patterns = [
            r"author's full name is\s+([A-Z][\w\-'\.]*(?:\s+[A-Z][\w\-'\.]*){0,4})",
            r"^([A-Z][\w\-'\.]*(?:\s+[A-Z][\w\-'\.]*){0,4})\s+is\s+the author",
        ]
        for i in range(0, len(ds), 20):
            ans = ds[i]["answer"]
            for pat in patterns:
                for m in _re.finditer(pat, ans):
                    nm = m.group(1).strip().rstrip(".,;:")
                    if nm:
                        names.add(nm)

        if len(names) < 10:
            from collections import Counter
            counter = Counter()
            cap_pat = _re.compile(
                r"\b([A-Z][\w\-'\.]+(?:\s+[A-Z][\w\-'\.]+){1,3})\b"
            )
            for ex in ds:
                for m in cap_pat.finditer(ex["answer"]):
                    counter[m.group(1).strip().rstrip(".,;:")] += 1
            for nm, cnt in counter.items():
                if cnt >= 5:
                    names.add(nm)

        tok = self.processing_class
        token_ids: set[int] = set()
        for name in names:
            for variant in (name, " " + name, "\n" + name, name + ".", name + ","):
                ids = tok(variant, add_special_tokens=False).get("input_ids", [])
                token_ids.update(int(t) for t in ids)
            for piece in name.split():
                for variant in (piece, " " + piece, piece + ".", piece + ","):
                    ids = tok(variant, add_special_tokens=False).get("input_ids", [])
                    token_ids.update(int(t) for t in ids)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps({
            "author_names": sorted(names),
            "token_ids": sorted(token_ids),
        }, indent=2), encoding="utf-8")
        logger.info(
            f"Built legacy author token-id set: {len(names)} names -> "
            f"{len(token_ids)} unique token-ids (cached at {cache_path})"
        )
        return token_ids

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def _author_mask_for(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a [B, T] boolean mask marking author-name positions.

        Dispatches based on ``self.author_mask_mode``:
          - ``"token_set"``: legacy flat token-ID membership (~27% match rate).
          - ``"span"``:      contiguous span matching (precise, ~5-8% match rate).
          - ``"off"``:       all-False mask (effectively disables author mask).

        Returns a bool tensor on ``input_ids.device``.
        """
        if self.author_mask_mode == "off":
            return torch.zeros_like(input_ids, dtype=torch.bool)

        if self.author_mask_mode == "token_set":
            return self._author_mask_token_set(input_ids)

        return self._author_mask_span(input_ids)

    def _author_mask_token_set(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Legacy: True where token-id is in the flat author-token set."""
        if self._author_token_ids is None:
            self._author_token_ids = self._build_author_token_ids()
        token_id_set = self._author_token_ids
        flat = input_ids.detach().cpu().reshape(-1).tolist()
        mask = torch.tensor(
            [tid in token_id_set for tid in flat],
            dtype=torch.bool,
            device=input_ids.device,
        ).view_as(input_ids)
        return mask

    def _load_forget_span_cache(self) -> dict[str, list[list[int]]]:
        """Load precomputed forget spans from ``self.forget_span_cache``.

        Expected JSON: ``{"spans": {key: [[tok, ...], ...], ...}, ...}`` -- the
        same shape produced by ``_build_author_name_spans`` and by the offline
        ``scripts/erase/build_muse_spans.py`` (NER / QA-keyword) builder.
        """
        import json as _json
        from pathlib import Path as _Path

        p = _Path(self.forget_span_cache)
        if not p.is_absolute():
            p = _Path(__file__).resolve().parents[3] / p
        data = _json.loads(p.read_text(encoding="utf-8"))
        raw_spans = data.get("spans", {})
        spans = {
            k: [[int(t) for t in seq] for seq in v if seq]
            for k, v in raw_spans.items()
        }
        spans = {k: v for k, v in spans.items() if v}
        n_seqs = sum(len(v) for v in spans.values())
        logger.info(
            f"[forget_span_cache] loaded {len(spans)} keys / {n_seqs} token "
            f"sequences from {p}"
        )
        return spans

    def _build_span_first_tok_index(self) -> None:
        """Index span token-sequences by their first token id for fast match.

        MUSE NER/QA caches can hold thousands of entity sequences; a naive
        sliding-window over every sequence at every position is too slow on
        2048-token chunks. Bucketing by the first token id means each position
        only checks the (usually few) sequences that could start there.
        """
        index: dict[int, list[list[int]]] = {}
        for seqs in (self._author_name_spans or {}).values():
            for seq in seqs:
                if seq:
                    index.setdefault(seq[0], []).append(seq)
        self._forget_span_first_tok_index = index

    def _author_mask_span(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Span-based: True only for contiguous name/entity token spans.

        Uses ``self.forget_span_cache`` when set (MUSE NER / QA-keyword),
        otherwise the TOFU author extractor. Matching is accelerated with a
        first-token index so large entity caches stay tractable.
        """
        if self._author_name_spans is None:
            if self.forget_span_cache:
                self._author_name_spans = self._load_forget_span_cache()
            else:
                self._author_name_spans = self._build_author_name_spans()
            self._build_span_first_tok_index()

        B, T = input_ids.shape
        ids_cpu = input_ids.detach().cpu().tolist()
        mask_flat = [[False] * T for _ in range(B)]
        index = self._forget_span_first_tok_index or {}

        for b in range(B):
            row = ids_cpu[b]
            for t in range(T):
                seqs = index.get(row[t])
                if not seqs:
                    continue
                for seq in seqs:
                    slen = len(seq)
                    if t + slen <= T and row[t:t + slen] == seq:
                        for k in range(slen):
                            mask_flat[b][t + k] = True

        return torch.tensor(mask_flat, dtype=torch.bool, device=input_ids.device)

    def log(self, logs: dict, start_time: float | None = None, **kwargs) -> None:
        if self._last_retain_entropy is not None and "loss" in logs:
            logs["retain_entropy"] = self._last_retain_entropy
        if self._last_forget_entropy is not None and "loss" in logs:
            logs["forget_entropy"] = self._last_forget_entropy
        if self._last_forget_z_p10 is not None and "loss" in logs:
            logs["forget_z_p10"] = self._last_forget_z_p10
            logs["forget_z_mean"] = self._last_forget_z_mean
        if not math.isnan(self._last_retain_ce_monitor) and "loss" in logs:
            logs["retain_ce_monitor"] = self._last_retain_ce_monitor
        if self._last_forget_ce_loss is not None and "loss" in logs:
            logs["forget_ce_loss"] = self._last_forget_ce_loss
        if self._last_forget_ce_raw is not None and "loss" in logs:
            logs["forget_ce_raw"] = self._last_forget_ce_raw
        # Per-sample dynamic-stop diagnostics. Only surfaced when the gate
        # is active (counters initialised to -1 / NaN otherwise).
        if (
            self.dynamic_stop_loss_threshold > 0
            and self._last_n_active >= 0
            and "loss" in logs
        ):
            logs["dynstop_n_active"] = self._last_n_active        # active in batch
            logs["dynstop_n_done_batch"] = self._last_n_done       # done in batch
            logs["dynstop_n_done_total"] = self._last_n_done_total # cumulative
            logs["dynstop_max_per_sample_ce"] = self._last_max_per_sample_ce
            if self._last_n_overshoot >= 0:
                logs["dynstop_n_overshoot"] = self._last_n_overshoot
        if self._last_span_coverage >= 0.0 and "loss" in logs:
            logs["vjp_span_coverage"] = self._last_span_coverage
            logs["vjp_selected_pct"] = self._last_vjp_selected_pct
        # Per-token CE distribution + band fractions + mean seq length.
        if self._last_tok_ce_p50 is not None and "loss" in logs:
            logs["tok_ce_p10"] = self._last_tok_ce_p10
            logs["tok_ce_p50"] = self._last_tok_ce_p50
            logs["tok_ce_p90"] = self._last_tok_ce_p90
            logs["tok_frac_below_band"] = self._last_tok_frac_below
            logs["tok_frac_in_band"] = self._last_tok_frac_in
            logs["tok_frac_above_band"] = self._last_tok_frac_above
            logs["mean_seq_len"] = self._last_mean_seq_len
        # How far have the trainable weights actually moved from init?
        if "loss" in logs:
            self._update_weight_movement()
            if self._last_weight_delta is not None:
                logs["weight_delta"] = self._last_weight_delta
                logs["weight_rel_delta"] = self._last_weight_rel_delta
        # Capture grad norm (+ EMA) so the signal monitor can use it as a
        # candidate trigger. HF logs it as "grad_norm" on optimiser steps.
        gn = logs.get("grad_norm")
        if gn is not None:
            try:
                gn = float(gn)
                self._last_grad_norm = gn
                self._grad_norm_ema = (
                    gn if self._grad_norm_ema is None
                    else 0.9 * self._grad_norm_ema + 0.1 * gn
                )
            except (TypeError, ValueError):
                pass
        # Surface the latest monitor signals on the normal log line too.
        if self._last_signal_record is not None and "loss" in logs:
            for k in ("forget_ce_probe", "retain_ce_probe",
                      "forget_ce_probe_std", "retain_ce_rise"):
                v = self._last_signal_record.get(k)
                if v is not None:
                    logs[f"mon_{k}"] = v
        super().log(logs, start_time=start_time, **kwargs)

    def _update_weight_movement(self) -> None:
        """Best-effort ||W - W0|| over the (few) trainable tensors, plus the
        relative change ||W - W0|| / ||W0||. Caches W0 on first call. Used to
        answer "are the weights moving at all?" -- if this stays ~0 while grad
        norm is non-zero, the spectral filter / LR / scope is over-constraining
        the update. Never raises."""
        try:
            model = getattr(self, "model", None)
            if model is None:
                return
            named = [
                (n, p) for n, p in model.named_parameters()
                if p.requires_grad
            ]
            if not named:
                return
            if self._w0_cache is None:
                self._w0_cache = {
                    n: p.detach().clone() for n, p in named
                }
                ssq = 0.0
                for _, w0 in self._w0_cache.items():
                    ssq += float((w0.float() ** 2).sum())
                self._w0_norm_sq_cache = max(ssq, 1e-12)
            delta_sq = 0.0
            for n, p in named:
                w0 = self._w0_cache.get(n)
                if w0 is not None and w0.shape == p.shape:
                    delta_sq += float(((p.detach() - w0).float() ** 2).sum())
            self._last_weight_delta = delta_sq ** 0.5
            self._last_weight_rel_delta = (
                self._last_weight_delta / (self._w0_norm_sq_cache ** 0.5)
            )
        except Exception:  # noqa: BLE001 - diagnostics must never crash a run
            pass

    # ------------------------------------------------------------------
    # Dynamic-stop SAMPLER-LEVEL plumbing.
    # ------------------------------------------------------------------
    # See ``__init__`` and the dynstop branch in ``compute_loss`` for the
    # high-level semantics. The two helpers below are the only places we
    # touch the live ``_dyn_active_indices`` set; everything else just
    # reads ``_dyn_done_keys``.

    def _bootstrap_dynamic_stop_active_set(self) -> None:
        """One-shot precompute: build (key -> forget index) map, hand a
        SHARED active-index set to ForgetRetainDataset so its __getitem__
        starts swapping done samples for random active ones on the fly.

        The keys MUST be byte-identical to the keys built in compute_loss
        (padding-stripped via attention_mask). Any drift here makes the
        sampler-level filter silently no-op for that sample.
        """
        try:
            forget_ds = self.train_dataset.forget  # type: ignore[attr-defined]
        except AttributeError:
            logger.warning(
                "[dynamic_stop] train_dataset has no .forget; sampler-level "
                "filtering disabled (loss-mask still active)."
            )
            self._dyn_idx_to_key = []
            self._dyn_key_to_idx = {}
            return

        n = len(forget_ds)
        idx_to_key: list = [None] * n
        key_to_idx: dict = {}

        # CRITICAL: replicate the collator's attention-mask logic.
        # `DataCollatorForSupervisedDataset.__call__` discards each item's
        # original attention_mask and rebuilds it as
        # ``input_ids.ne(tokenizer.pad_token_id)``. For Llama-3.2-Instruct
        # the pad_token_id == 128009 (`<|eot_id|>`), which is *also* the
        # genuine end-of-turn token at the end of every real example. So
        # at runtime that real eot is treated as padding and stripped out
        # of the runtime key. If we used the dataset's own all-ones
        # attention_mask at init, our init keys would still contain the
        # eot and never match -- which is exactly the bug we observed
        # (sample_keys_in_init_map == 0 for 1506 dynstop calls).
        tok = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )
        pad_id = getattr(tok, "pad_token_id", None) if tok is not None else None

        def _build_key(item):
            ids = item["input_ids"]
            lbls = item["labels"]
            if isinstance(ids, torch.Tensor):
                if pad_id is not None:
                    m = (ids != pad_id)
                else:
                    am = item["attention_mask"]
                    m = am.bool() if isinstance(am, torch.Tensor) else torch.tensor(
                        [bool(a) for a in am]
                    )
                ids_t = tuple(ids[m].tolist())
                lbl_t = (
                    tuple(lbls[m].tolist())
                    if isinstance(lbls, torch.Tensor)
                    else tuple(int(t) for t, keep in zip(lbls, m.tolist()) if keep)
                )
            else:
                if pad_id is not None:
                    keep_list = [t != pad_id for t in ids]
                else:
                    am = item["attention_mask"]
                    keep_list = [bool(a) for a in (
                        am.tolist() if isinstance(am, torch.Tensor) else am
                    )]
                ids_t = tuple(int(t) for t, k in zip(ids, keep_list) if k)
                lbl_t = tuple(int(t) for t, k in zip(lbls, keep_list) if k)
            return (ids_t, lbl_t)

        collisions = 0
        for i in range(n):
            k = _build_key(forget_ds[i])
            idx_to_key[i] = k
            if k in key_to_idx:
                collisions += 1
            else:
                key_to_idx[k] = i

        self._dyn_idx_to_key = idx_to_key
        self._dyn_key_to_idx = key_to_idx
        self._dyn_active_indices = set(range(n))
        # Resolve the global early-stop total here, in the SAME key-space the
        # done-counter uses (``len(self._dyn_done_keys)``). Using the unique-key
        # count fixes datasets (e.g. MUSE PretrainingDataset) that have no
        # ``.forget.data`` attribute for the lazy fallback below to introspect.
        # For TOFU this equals len(forget.data), so behaviour is unchanged.
        self._dyn_n_total = len(key_to_idx)
        if hasattr(self.train_dataset, "set_dyn_active_indices"):
            self.train_dataset.set_dyn_active_indices(
                self._dyn_active_indices
            )
        if hasattr(self.train_dataset, "set_done_sample_prob"):
            self.train_dataset.set_done_sample_prob(
                self.dynamic_stop_done_sample_prob
            )
        sample_lens = [len(k[0]) for k in list(key_to_idx.keys())[:3]]
        global_step_now = (
            int(self.state.global_step)
            if hasattr(self, "state") and self.state is not None
            else -1
        )
        msg = (
            f"[dynamic_stop] sampler-level filter ARMED. "
            f"forget_set_size={n} unique_keys={len(key_to_idx)} "
            f"pad_id={pad_id} sample_ids_len={sample_lens} "
            f"global_step_at_bootstrap={global_step_now}"
        )
        if collisions:
            msg += f" duplicate_keys={collisions} (expected for tied tokenisations)"
        logger.info(msg)

    def _dyn_drop_active_index(self, key) -> None:
        """Remove the (single) forget index matching ``key`` from the
        live active set so the dataloader stops drawing it on the next
        ``__getitem__``. Silent no-op if either the key map hasn't been
        bootstrapped yet or the key was already removed.
        """
        if self._dyn_active_indices is None or self._dyn_key_to_idx is None:
            return
        idx = self._dyn_key_to_idx.get(key)
        if idx is not None:
            self._dyn_active_indices.discard(idx)

    def _per_token_ce(
        self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token CE for causal LMs (mirrors HF's standard label shift).

        Returns
        -------
        per_tok_ce : [B, T-1]
            CE at each predicting position. 0 at ignored positions
            (because of ``ignore_index``).
        shift_labels : [B, T-1]
            Labels at predicting positions; equals ``labels[..., 1:]``.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        b, t_m1, v = shift_logits.shape
        flat_ce = F.cross_entropy(
            shift_logits.view(-1, v),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        )
        return flat_ce.view(b, t_m1), shift_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        forget_mask = (forget_inputs["labels"] != -100)

        # Are any of the new per-position / per-question caps active?
        # Each behaves independently. If neither is set, the original
        # batch-mean code path runs unchanged (bit-identical numerics).
        per_token_cap_active = (
            self.forget_loss_type == "ce"
            and self.forget_loss_per_token_cap > 0
        )
        answer_target_active = (
            self.forget_loss_type == "ce"
            and (
                self.forget_loss_answer_target > 0
                or self.forget_loss_answer_mode == "mse"
            )
        )
        new_caps_active = per_token_cap_active or answer_target_active

        # Forget forward.
        outputs = model(
            input_ids=forget_inputs["input_ids"],
            attention_mask=forget_inputs["attention_mask"],
            labels=forget_inputs["labels"],
        )

        # Always log the raw CE so the curve plot is comparable across loss
        # types, even when CE isn't actually being optimized.
        forget_ce_raw = outputs.loss
        self._last_forget_ce_raw = float(forget_ce_raw.detach())

        if self.forget_loss_type == "ce" and self.dynamic_stop_loss_threshold > 0:
            # DYNAMIC PER-SAMPLE STOP (STATEFUL + GLOBAL EARLY-STOP).
            #
            # Per-sample analogue of forget_loss_max_ce. Each sample is
            # identified by a stable (input_ids, labels) key (padding
            # stripped via attention_mask, so the key is invariant to
            # dynamic-padding differences across batches). The instant a
            # sample's LENGTH-NORMALIZED mean CE on its answer tokens
            # reaches `dynamic_stop_loss_threshold`, its key is added to
            # ``self._dyn_done_keys`` and the sample contributes 0 to the
            # forget loss FOR THE REST OF TRAINING (no gradient ascent),
            # even if its CE later drifts back below the threshold.
            #
            # Once every sample in the forget set has been added to the
            # done set, ``self.control.should_training_stop`` is set so
            # the HF Trainer terminates the run cleanly at the end of the
            # current step -- realising the user's "2 epochs for one,
            # 10 for another" objective.
            #
            # Both forget_loss_max_ce and forget_loss_per_token_cap are
            # IGNORED in this branch (the per-sample gate subsumes them).
            # Lazy-bootstrap the sampler-level filtering plumbing on the
            # FIRST dynstop call (cheap; one full pass over the forget
            # set tokenising in-cache). After this, ``_dyn_active_indices``
            # is a SHARED mutable set that ForgetRetainDataset consults on
            # every __getitem__ so done samples are swapped out for random
            # active ones -- realising the user-visible semantic: a "done"
            # sample is removed from the dataloader, not just masked from
            # the loss.
            if (
                self._dyn_idx_to_key is None
                and hasattr(self.train_dataset, "forget")
            ):
                self._bootstrap_dynamic_stop_active_set()

            per_tok_ce, shift_labels = self._per_token_ce(
                outputs.logits, forget_inputs["labels"], ignore_index=-100,
            )
            shift_answer_mask = (shift_labels != -100)
            ans_f = shift_answer_mask.to(per_tok_ce.dtype)
            ans_per_sample = ans_f.sum(dim=-1).clamp(min=1.0)              # [B]
            mean_ce_per_sample = (
                (per_tok_ce * ans_f).sum(dim=-1) / ans_per_sample
            )                                                              # [B]

            # --- per-TOKEN CE distribution diagnostics (long-sequence aware) ---
            # The loss steers the per-sample MEAN; this exposes the underlying
            # per-token spread so we can see (a) the strongly-memorised low-CE
            # tail that drags the mean down on 2048-tok chunks, and (b) what
            # fraction of tokens sit below / inside / above the target band --
            # i.e. whether up-pushed and down-pushed tokens are fighting. Cheap,
            # best-effort, no grad; never alters the loss.
            try:
                with torch.no_grad():
                    tok_vals = per_tok_ce[shift_answer_mask]
                    if tok_vals.numel() > 0:
                        tv = tok_vals.float()
                        q = torch.quantile(
                            tv, torch.tensor([0.1, 0.5, 0.9], device=tv.device)
                        )
                        self._last_tok_ce_p10 = float(q[0])
                        self._last_tok_ce_p50 = float(q[1])
                        self._last_tok_ce_p90 = float(q[2])
                        bl = self.forget_loss_band_lower
                        bu = self.forget_loss_band_upper
                        n = float(tv.numel())
                        self._last_tok_frac_below = float((tv < bl).sum()) / n
                        self._last_tok_frac_above = float((tv > bu).sum()) / n
                        self._last_tok_frac_in = max(
                            0.0,
                            1.0 - self._last_tok_frac_below - self._last_tok_frac_above,
                        )
                        self._last_mean_seq_len = float(ans_per_sample.float().mean())
            except Exception:  # noqa: BLE001 - diagnostics must never crash a run
                pass

            # Build per-row stable keys. Mask on the original device (GPU)
            # first, then sync-copy the tiny masked tensor to CPU and
            # convert to tuple. NEVER use non_blocking=True before indexing
            # -- a known PyTorch race triggers the
            # "out_ptr == out_accessor[..]" internal assertion when the
            # copy hasn't completed before mask-indexing kicks in.
            with torch.no_grad():
                am = forget_inputs["attention_mask"]
                ii = forget_inputs["input_ids"]
                lb = forget_inputs["labels"]
                bsz = ii.shape[0]
                keys = []
                for r in range(bsz):
                    m = am[r].bool()
                    ids_t = tuple(ii[r][m].cpu().tolist())
                    lab_t = tuple(lb[r][m].cpu().tolist())
                    keys.append((ids_t, lab_t))
                ce_cpu = mean_ce_per_sample.detach().cpu()

                # Stateful active mask: "active" iff key NOT yet in the done
                # set. New crossings this step are added to the done set
                # AFTER the active mask is built, so they still contribute
                # to THIS step's gradient (the very step that crossed the
                # threshold counts; subsequent steps do not).
                # When done_sample_prob > 0, done samples that still
                # appear in the batch (via the reduced-prob re-draw)
                # remain active so band/MSE keeps correcting drift.
                if self.dynamic_stop_done_sample_prob > 0:
                    active_list = [1.0] * bsz
                else:
                    active_list = [
                        1.0 if k not in self._dyn_done_keys else 0.0
                        for k in keys
                    ]
                # Add new crossings to the persistent done set; also bump
                # per-sample active-step counters (deduped per optimiser
                # step) so the long-tail patience rule has a clean
                # measurement to compare against.
                newly_done = 0
                gs = (
                    int(self.state.global_step)
                    if hasattr(self, "state") and self.state is not None
                    else -1
                )
                for r, k in enumerate(keys):
                    if active_list[r] != 1.0:
                        continue
                    last_seen = self._dyn_last_active_step.get(k, -1)
                    if gs > last_seen:
                        self._dyn_active_steps[k] = (
                            self._dyn_active_steps.get(k, 0) + 1
                        )
                        self._dyn_last_active_step[k] = gs
                    ce_val = float(ce_cpu[r].item())
                    if (
                        self.forget_loss_answer_mode == "mse"
                        and self.forget_loss_answer_target_std > 0
                    ):
                        # Per-sample jitter: freeze each sample once it reaches
                        # ITS OWN target (within a small margin), so the frozen
                        # CE distribution = {tau_i} ~ N(tau, std) = oracle shape.
                        tau_i = self._get_sample_target(k)
                        in_band = ce_val >= (tau_i - 0.15)
                    elif self.forget_loss_answer_mode == "mse":
                        # Two-sided: only stop if sample landed INSIDE the band
                        # (used as the "target zone" for mse mode). If it
                        # overshot above the upper bound, keep training so
                        # the answer-mode loss can pull it back.
                        in_band = (
                            ce_val >= self.forget_loss_band_lower
                            and ce_val <= self.forget_loss_band_upper
                        )
                    else:
                        in_band = ce_val >= self.dynamic_stop_loss_threshold
                    if in_band:
                        self._dyn_done_keys.add(k)
                        self._dyn_drop_active_index(k)
                        newly_done += 1

                # Periodic stdout summary (every 50 optimiser steps) so the
                # SLURM log shows progress without the noise of per-step writes.
                try:
                    if gs >= 0 and gs % 50 == 0:
                        n_active_dbg = int(sum(active_list))
                        n_total = (
                            len(self._dyn_active_indices)
                            if self._dyn_active_indices is not None else -1
                        )
                        n_done = len(self._dyn_done_keys)
                        logger.info(
                            f"[dynamic_stop] step={gs} "
                            f"done={n_done}/{n_total + n_done} "
                            f"active_in_batch={n_active_dbg}/{len(keys)} "
                            f"max_ce={float(ce_cpu.max().item()):.3f} "
                            f"min_ce={float(ce_cpu.min().item()):.3f} "
                            f"newly_done={newly_done}"
                        )
                except Exception:
                    pass

                # ONE-SHOT: log whether runtime keys match init keys. Critical
                # diagnostic if dynstop ever silently breaks again (the H1 bug
                # we hit before: collator stripping eot_id from runtime keys
                # only). If `match_ids_only_count` is 0 here, the sampler
                # filter is silently no-op.
                if not getattr(self, "_dyn_key_diff_logged", False):
                    self._dyn_key_diff_logged = True
                    try:
                        rt0_ids, rt0_lbls = keys[0]
                        match_ids_only = sum(
                            1 for k in self._dyn_idx_to_key or []
                            if k is not None and k[0] == rt0_ids
                        )
                        match_labels_only = sum(
                            1 for k in self._dyn_idx_to_key or []
                            if k is not None and k[1] == rt0_lbls
                        )
                        rt_ids_lens = [len(k[0]) for k in keys[:2]]
                        init_ids_lens = (
                            [len(self._dyn_idx_to_key[i][0])
                             for i in range(min(2, len(self._dyn_idx_to_key or [])))]
                            if self._dyn_idx_to_key else []
                        )
                        logger.info(
                            f"[dynamic_stop] first_batch_key_match: "
                            f"rt_ids_lens={rt_ids_lens} "
                            f"init_ids_lens={init_ids_lens} "
                            f"match_ids_only={match_ids_only}/"
                            f"{len(self._dyn_idx_to_key or [])} "
                            f"match_labels_only={match_labels_only}/"
                            f"{len(self._dyn_idx_to_key or [])}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[dynamic_stop] key_diff log failed: {e}"
                        )

            active_mask = torch.tensor(
                active_list,
                dtype=per_tok_ce.dtype,
                device=per_tok_ce.device,
            )                                                               # [B]
            n_active = active_mask.sum().clamp(min=1.0)

            # Choose loss function: MSE uses a two-sided objective that both
            # pushes CE up AND pulls it back (the CET loss). Raw ascent (-CE)
            # is the legacy default when no answer mode is set.
            # #1 Per-token CE ceiling: drop tokens already above the ceiling from
            # the forget push (effective answer mask). When off, ans_f_eff==ans_f.
            if self.forget_loss_token_ce_ceiling > 0:
                tok_keep = (
                    per_tok_ce.detach() < self.forget_loss_token_ce_ceiling
                ).to(per_tok_ce.dtype)
                ans_f_eff = ans_f * tok_keep
                ans_per_sample_eff = ans_f_eff.sum(dim=-1).clamp(min=1.0)
                mean_ce_eff = (
                    (per_tok_ce * ans_f_eff).sum(dim=-1) / ans_per_sample_eff
                )
            else:
                ans_f_eff = ans_f
                ans_per_sample_eff = ans_per_sample
                mean_ce_eff = mean_ce_per_sample

            if self.forget_loss_answer_mode == "mse":
                if self.forget_loss_answer_target_std > 0:
                    # Per-sample jittered target -> reproduces oracle spread.
                    tau_vec = torch.tensor(
                        [self._get_sample_target(k) for k in keys],
                        device=mean_ce_eff.device,
                        dtype=mean_ce_eff.dtype,
                    )
                    mse_per_sample = (mean_ce_eff - tau_vec) ** 2
                else:
                    mse_per_sample = (
                        mean_ce_eff - self.forget_loss_answer_target
                    ) ** 2
                forget_ce = -(mse_per_sample * active_mask).sum() / n_active
            else:
                forget_ce = (mean_ce_per_sample * active_mask).sum() / n_active

            loss = -self.forget_loss_weight * forget_ce
            self._last_forget_ce_loss = float(forget_ce.detach())

            # Per-step diagnostics for TB.
            with torch.no_grad():
                self._last_n_active = int(active_mask.sum().item())
                self._last_n_done = bsz - self._last_n_active
                self._last_n_done_total = len(self._dyn_done_keys)
                if self.dynamic_stop_log_upper > 0:
                    self._last_n_overshoot = int(
                        (mean_ce_per_sample > self.dynamic_stop_log_upper)
                        .sum().item()
                    )
                else:
                    self._last_n_overshoot = -1
                self._last_max_per_sample_ce = float(
                    mean_ce_per_sample.max().item()
                )

            # Lazy-resolve the total forget-set size once (used for global
            # early stop). Normally this is already set by the sampler-level
            # bootstrap (= number of unique forget keys). This block only runs
            # as a fallback if that never armed; falls back to a sentinel if
            # introspection fails.
            if self._dyn_n_total < 0:
                try:
                    self._dyn_n_total = len(
                        self.train_dataset.forget.data  # type: ignore[attr-defined]
                    )
                    logger.info(
                        f"[dynamic_stop] forget set size = {self._dyn_n_total}; "
                        f"will signal training stop when all are done."
                    )
                except Exception:  # noqa: BLE001
                    # Last-ditch: use the dataset's own __len__ if available.
                    try:
                        self._dyn_n_total = len(self.train_dataset.forget)  # type: ignore[attr-defined]
                        logger.info(
                            f"[dynamic_stop] forget set size = {self._dyn_n_total} "
                            f"(via __len__); will signal stop when all are done."
                        )
                    except Exception:  # noqa: BLE001
                        self._dyn_n_total = 0  # disable global early-stop
                        logger.warning(
                            "[dynamic_stop] could not introspect forget set size; "
                            "global early-stop disabled (per-sample gate still active)."
                        )

            # Per-step progress line: prints EVERY gradient step so the user
            # can watch the "samples left" curve live in stdout. Cheap.
            _step = getattr(self.state, "global_step", -1) if hasattr(self, "state") else -1
            _remaining = (
                max(0, self._dyn_n_total - self._last_n_done_total)
                if self._dyn_n_total > 0 else -1
            )
            logger.info(
                f"[dynstop] step={_step} done={self._last_n_done_total}/"
                f"{self._dyn_n_total if self._dyn_n_total > 0 else '?'} "
                f"remaining={_remaining if _remaining >= 0 else '?'} "
                f"newly_done={newly_done} "
                f"batch_active={self._last_n_active}/{bsz} "
                f"max_ce={self._last_max_per_sample_ce:.3f}"
            )

            # Stuck-sample decoded log: when the long tail is small (<=
            # dynamic_stop_decode_threshold) we surface each still-active
            # sample's question + GT once so the user can see WHAT is
            # resistant. Decoded once per sample (cached in
            # self._dyn_decoded_keys). Uses self.processing_class
            # (HF >= 4.46) or falls back to self.tokenizer.
            decode_thr = self.dynamic_stop_decode_threshold
            if (
                decode_thr > 0
                and 0 <= _remaining <= decode_thr
            ):
                tok = (
                    getattr(self, "processing_class", None)
                    or getattr(self, "tokenizer", None)
                )
                if tok is not None:
                    for r, k in enumerate(keys):
                        if k in self._dyn_done_keys:
                            continue
                        if k in self._dyn_decoded_keys:
                            continue
                        try:
                            ids_row = forget_inputs["input_ids"][r]
                            lbl_row = forget_inputs["labels"][r]
                            am_row = forget_inputs["attention_mask"][r].bool()
                            non_pad_ids = ids_row[am_row].tolist()
                            ans_mask = (lbl_row != -100) & am_row
                            ans_ids = ids_row[ans_mask].tolist()
                            full_text = tok.decode(
                                non_pad_ids, skip_special_tokens=True
                            )
                            ans_text = tok.decode(
                                ans_ids, skip_special_tokens=True
                            )
                            ce_val = float(ce_cpu[r].item())
                            astep = self._dyn_active_steps.get(k, 0)
                            logger.info(
                                f"[dynstop][stuck] mean_ce={ce_val:.3f} "
                                f"active_steps={astep}\n"
                                f"  prompt+answer: {full_text!r}\n"
                                f"  GT answer:    {ans_text!r}"
                            )
                            self._dyn_decoded_keys.add(k)
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                f"[dynstop] decode failed for stuck sample: {e}"
                            )

            # Global early-stop: when all forget samples have crossed the
            # threshold, signal the Trainer to terminate at end of step.
            if (
                self._dyn_n_total > 0
                and self._last_n_done_total >= self._dyn_n_total
                and not self._dyn_early_stop_logged
            ):
                logger.info(
                    f"[dynamic_stop] All {self._dyn_n_total} forget samples "
                    f"have reached mean-CE >= {self.dynamic_stop_loss_threshold}; "
                    f"signalling training stop."
                )
                self._dyn_early_stop_logged = True
                if hasattr(self, "control") and self.control is not None:
                    self.control.should_training_stop = True

            # Long-tail patience: ONLY fires when we are deep in the
            # long tail (remaining < dynamic_stop_longtail_threshold)
            # and SOME active sample has been receiving gradient for
            # more optimiser steps than the patience budget. Stops the
            # whole run -- the typical situation is "1-3 mathematically
            # resistant samples remain and ascending forever won't cross
            # them under this geometry, so let's bail and evaluate".
            patience = self.dynamic_stop_max_active_steps_per_sample
            longtail_thr = self.dynamic_stop_longtail_threshold
            in_longtail = (
                self._dyn_n_total > 0
                and 0 < _remaining < longtail_thr
            )
            longtail_trigger_steps = -1
            longtail_trigger_ce = -1.0
            if (
                patience > 0
                and in_longtail
                and not self._dyn_early_stop_logged
            ):
                # Find the still-active sample(s) in the current batch
                # whose active-step count has exceeded the patience.
                for r, k in enumerate(keys):
                    if k in self._dyn_done_keys:
                        continue
                    cnt = self._dyn_active_steps.get(k, 0)
                    if cnt > patience:
                        longtail_trigger_steps = cnt
                        longtail_trigger_ce = float(ce_cpu[r].item())
                        logger.info(
                            f"[dynamic_stop] LONG-TAIL PATIENCE "
                            f"(remaining={_remaining} < "
                            f"{longtail_thr}): a sample has appeared in "
                            f"{cnt} optimiser steps with mean_ce="
                            f"{longtail_trigger_ce:.3f} (still < threshold "
                            f"{self.dynamic_stop_loss_threshold}); "
                            f"signalling training stop."
                        )
                        self._dyn_early_stop_logged = True
                        if hasattr(self, "control") and self.control is not None:
                            self.control.should_training_stop = True
                        break

        else:
            if not new_caps_active:
                # ORIGINAL path: batch-mean CE (optionally clamped via the
                # legacy mean cap). Bit-identical to all pre-cap runs.
                forget_ce = forget_ce_raw
                if self.forget_loss_max_ce > 0:
                    # torch.clamp passes gradient through where ce <= cap and
                    # zeros it out where ce > cap, stopping ascent on already-
                    # forgotten batches and taming the noisy CE curve.
                    forget_ce = torch.clamp(forget_ce, max=self.forget_loss_max_ce)
                loss = -self.forget_loss_weight * forget_ce
                self._last_forget_ce_loss = float(forget_ce.detach())
            else:
                # NEW path: compute per-position CE so we can apply position-
                # level / per-question caps before averaging.
                #
                # Step 1. per-token CE [B, T-1] (zero at ignored positions).
                per_tok_ce, shift_labels = self._per_token_ce(
                    outputs.logits, forget_inputs["labels"], ignore_index=-100,
                )
                shift_answer_mask = (shift_labels != -100)

                # Step 2. per-position cap (Mode-2 collapse fix).
                if per_token_cap_active:
                    per_tok_ce = torch.clamp(
                        per_tok_ce, max=self.forget_loss_per_token_cap,
                    )

                # Step 3. per-question aggregation — two mutually-exclusive
                # paths: (a) bounded per-question target objective,
                # (b) plain batch-mean over all answer tokens.
                ans_count = shift_answer_mask.float().sum().clamp(min=1.0)
                if answer_target_active:
                    # (a) Variant A: bounded per-question objective. For each
                    # sample we take the LENGTH-NORMALIZED mean per-token CE
                    # over its answer positions (length-invariant so a fixed
                    # target has the same semantics across short/long answers),
                    # then apply a sigmoid gate centred at the target. The
                    # scalar loss is the batch-mean of the gate, bounded in
                    # [0, 1]. Gradient naturally vanishes once each sample's
                    # mean CE passes the target -> no unbounded ascent, no
                    # need for forget_loss_max_ce, no need to hand-tune the
                    # stopping point.
                    ans_f = shift_answer_mask.to(per_tok_ce.dtype)
                    ans_per_sample = ans_f.sum(dim=-1).clamp(min=1.0)            # [B]
                    mean_ce_per_sample = (
                        (per_tok_ce * ans_f).sum(dim=-1) / ans_per_sample
                    )                                                            # [B]

                    if self.forget_loss_answer_mode == "mse":
                        target = self.forget_loss_answer_target
                        mse_per_sample = (mean_ce_per_sample - target) ** 2  # [B]
                        forget_ce = -mse_per_sample.mean()
                    else:
                        dist = mean_ce_per_sample - self.forget_loss_answer_target  # [B]
                        gate_per_sample = torch.sigmoid(dist)                    # [B] in (0, 1)
                        forget_ce = gate_per_sample.mean()
                else:
                    # (b) Plain batch-mean over answer tokens.
                    ans_f = shift_answer_mask.to(per_tok_ce.dtype)
                    forget_ce = (per_tok_ce * ans_f).sum() / ans_count

                # Step 4. legacy mean cap (only meaningful when the scalar
                # "forget_ce" is still in CE-nats units; the sigmoid path is
                # already bounded in [0, 1] so an additional cap is a no-op).
                if self.forget_loss_max_ce > 0 and not answer_target_active:
                    forget_ce = torch.clamp(forget_ce, max=self.forget_loss_max_ce)

                loss = -self.forget_loss_weight * forget_ce
                self._last_forget_ce_loss = float(forget_ce.detach())

            # Auxiliary entropy regularizer (composable with CE-up + cap).
            # Pulls mean per-token entropy on forget answer positions toward
            # `forget_entropy_reg_target`. Acts on the FULL softmax shape, so
            # it directly fights the "peaked wrong" attractor that hurts
            # MIN-K++ alignment. Zero = off (preserves existing numerics).
            if (
                self.forget_entropy_reg_weight > 0
                and self.forget_entropy_reg_target > 0
            ):
                lp_f = F.log_softmax(outputs.logits, dim=-1)
                p_f_aux = lp_f.exp()
                per_tok_H_aux = -(p_f_aux * lp_f).sum(dim=-1)  # [B, T]
                # Align entropy mask with the supervised next-token positions
                # used by the CE term: forget_mask is on `labels` (already in
                # next-token indexing in this codepath), but per-token quantities
                # from logits are also pre-shift here, so apply forget_mask
                # directly (same convention as the diagnostic block below).
                m_aux = forget_mask.to(
                    device=per_tok_H_aux.device, dtype=per_tok_H_aux.dtype
                )
                denom_aux = m_aux.sum().clamp(min=1.0)
                forget_H_aux = (per_tok_H_aux * m_aux).sum() / denom_aux
                ent_reg = (self.forget_entropy_reg_target - forget_H_aux) ** 2
                loss = loss + self.forget_entropy_reg_weight * ent_reg
                self._last_forget_entropy_reg = float(ent_reg.detach())
                self._last_forget_entropy_reg_H = float(forget_H_aux.detach())

        with torch.no_grad():
            log_p_f = F.log_softmax(outputs.logits, dim=-1)
            p_f = log_p_f.exp()
            per_tok_ent_f = -(p_f * log_p_f).sum(dim=-1)
            m_f = forget_mask.to(device=per_tok_ent_f.device, dtype=per_tok_ent_f.dtype)
            self._last_forget_entropy = float((per_tok_ent_f * m_f).sum() / m_f.sum().clamp(min=1.0))
            # MIN-K++ core statistic telemetry (always on; ~free given the
            # softmax above). z = (logp(target) - mu) / sigma per answer
            # token, with the proper next-token shift. mu = E_p[logp] is just
            # -H (already computed); only the second moment is extra.
            try:
                lbl_d = forget_inputs["labels"][:, 1:]
                mz_d = lbl_d != -100
                if bool(mz_d.any()):
                    lp_d = log_p_f[:, :-1, :]
                    logp_tgt_d = lp_d.gather(
                        -1, lbl_d.clamp(min=0).unsqueeze(-1)
                    ).squeeze(-1)                                  # [B, T-1]
                    mu_d = -per_tok_ent_f[:, :-1]
                    m2_d = (p_f[:, :-1, :] * lp_d.square()).sum(-1)
                    sigma_d = (m2_d - mu_d.square()).clamp(min=1e-6).sqrt()
                    z_d = ((logp_tgt_d - mu_d) / sigma_d)[mz_d].float()
                    self._last_forget_z_p10 = float(z_d.quantile(0.1))
                    self._last_forget_z_mean = float(z_d.mean())
            except Exception:  # noqa: BLE001 - diagnostics must never crash a run
                pass

        # --- Retain-CE monitor (no-grad, no loss contribution, early-stop only) ---
        if (
            self.retain_ce_stop_threshold > 0
            and self.state.global_step > 0
            and self.state.global_step % self.retain_ce_monitor_interval == 0
        ):
            self._check_retain_ce_monitor(model, inputs)

        # --- Signal monitor + automatic stop (the deliverable early-stop) ---
        if self.signal_monitor_interval > 0:
            # Accumulate the fixed probes early (before freezing) on every step
            # so the baselines are measured near step 0.
            if not self._probe_frozen:
                self._accumulate_probes(inputs)
            if (
                self.state.global_step > 0
                and self.state.global_step % self.signal_monitor_interval == 0
            ):
                self._run_signal_monitor(model)

        # --- Full-forget-set per-token CE dump (offline stop-signal mining) ---
        if self.token_ce_dump_epochs > 0:
            ep_now = float(self.state.epoch or 0.0)
            if ep_now - self._token_ce_last_dump_epoch >= self.token_ce_dump_epochs:
                self._token_ce_last_dump_epoch = ep_now
                self._dump_forget_token_ce(model)

        return (loss, outputs) if return_outputs else loss

    def _get_sample_target(self, key) -> float:
        """Per-sample MSE target. If jitter is off, the global tau; else a
        stable per-key draw from N(tau, std) clamped to +/- 2.5 std, so the
        forget-CE distribution reproduces the oracle's spread."""
        tau = self.forget_loss_answer_target
        std = self.forget_loss_answer_target_std
        if std <= 0:
            return tau
        t = self._dyn_sample_targets.get(key)
        if t is None:
            r = random.Random(hash(key) & 0xFFFFFFFF)
            t = r.gauss(tau, std)
            lo, hi = tau - 2.5 * std, tau + 2.5 * std
            t = max(lo, min(hi, t))
            self._dyn_sample_targets[key] = t
        return t

    def _check_retain_ce_monitor(self, model, inputs) -> None:
        """Compute mean CE on a cached retain mini-batch; halt if above threshold."""
        import torch.nn.functional as _F
        with torch.no_grad():
            if self._retain_ce_monitor_cache is None:
                retain = inputs.get("retain")
                if retain is None:
                    return
                self._retain_ce_monitor_cache = {
                    "input_ids": retain["input_ids"][:1].clone(),
                    "attention_mask": retain["attention_mask"][:1].clone(),
                    "labels": retain["labels"][:1].clone(),
                }
            cache = self._retain_ce_monitor_cache
            out = model(
                input_ids=cache["input_ids"],
                attention_mask=cache["attention_mask"],
            )
            logits = out.logits[:, :-1, :]
            labels = cache["labels"][:, 1:]
            mask = (labels != -100)
            ce = _F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
            )
            ce_masked = ce[mask.reshape(-1)]
            mean_ce = float(ce_masked.mean()) if ce_masked.numel() > 0 else 0.0
            self._last_retain_ce_monitor = mean_ce
            if self._retain_ce_baseline is None:
                self._retain_ce_baseline = mean_ce
                logger.info(
                    f"[retain_ce_monitor] baseline retain CE = {mean_ce:.4f} "
                    f"(threshold = {self.retain_ce_stop_threshold:.4f})"
                )
            if mean_ce > self.retain_ce_stop_threshold:
                logger.info(
                    f"[retain_ce_monitor] STOPPING: retain CE {mean_ce:.4f} > "
                    f"threshold {self.retain_ce_stop_threshold:.4f} "
                    f"(baseline was {self._retain_ce_baseline:.4f}, "
                    f"step {self.state.global_step})"
                )
                if hasattr(self, "control") and self.control is not None:
                    self.control.should_training_stop = True

    # ------------------------------------------------------------------ #
    #  Signal monitor + automatic stop                                    #
    # ------------------------------------------------------------------ #
    def _accumulate_probes(self, inputs) -> None:
        """Collect a FIXED forget + retain probe (CPU clones) over the first
        few steps, then freeze. Probes are held constant for the whole run so
        the CE signal is low-variance (unlike the batch-1 per-step forget CE,
        which swings 0.3-1.3 step-to-step on MUSE)."""
        n = self.signal_monitor_probe_size
        try:
            fg = inputs.get("forget")
            rt = inputs.get("retain")
        except AttributeError:
            return

        def _take(batch, store):
            if batch is None or len(store) >= n:
                return
            ids = batch.get("input_ids")
            if ids is None:
                return
            rows = ids.shape[0]
            for r in range(rows):
                if len(store) >= n:
                    break
                store.append({
                    "input_ids": ids[r:r + 1].detach().cpu().clone(),
                    "attention_mask": batch["attention_mask"][r:r + 1].detach().cpu().clone(),
                    "labels": batch["labels"][r:r + 1].detach().cpu().clone(),
                })

        _take(fg, self._probe_forget)
        _take(rt, self._probe_retain)
        if len(self._probe_forget) >= n and len(self._probe_retain) >= n:
            self._probe_frozen = True

    def _probe_ce(self, model, probe: list) -> dict | None:
        """Teacher-forced mean/percentile CE over a fixed probe (no grad)."""
        import torch.nn.functional as _F
        if not probe:
            return None
        device = next(model.parameters()).device
        per_sample = []
        tok_all = []
        tok_list = []      # per-sample per-token CE (fp32 cpu), supervised positions only
        argmax_list = []   # per-sample top-1 predicted ids at supervised positions
        with torch.no_grad():
            for ex in probe:
                ids = ex["input_ids"].to(device)
                am = ex["attention_mask"].to(device)
                out = model(input_ids=ids, attention_mask=am)
                logits = out.logits[:, :-1, :]
                labels = ex["labels"].to(device)[:, 1:]
                mask = (labels != -100)
                if mask.sum() == 0:
                    continue
                ce = _F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    labels.reshape(-1),
                    reduction="none",
                )
                ce = ce[mask.reshape(-1)]
                per_sample.append(float(ce.mean()))
                ce_cpu = ce.detach().float().cpu()
                tok_all.append(ce_cpu)
                tok_list.append(ce_cpu)
                am_pos = logits.argmax(dim=-1).reshape(-1)[mask.reshape(-1)]
                argmax_list.append(am_pos.detach().cpu())
        if not per_sample:
            return None
        ps = torch.tensor(per_sample)
        toks = torch.cat(tok_all) if tok_all else torch.tensor([0.0])
        q = torch.quantile(toks, torch.tensor([0.1, 0.5, 0.9]))
        return {
            "mean": float(ps.mean()),
            "std": float(ps.std(unbiased=False)) if ps.numel() > 1 else 0.0,
            "tok_std": float(toks.std(unbiased=False)) if toks.numel() > 1 else 0.0,
            "tok_p10": float(q[0]), "tok_p50": float(q[1]), "tok_p90": float(q[2]),
            "tok_max": float(toks.max()),
            "n": int(ps.numel()),
            "tok_list": tok_list,
            "argmax_list": argmax_list,
        }

    def _run_signal_monitor(self, model) -> None:
        """Compute candidate trigger signals on the fixed probes, append a JSONL
        record (telemetry), and apply the automatic stop rule (forget/retain
        only -- nothing the method is graded on)."""
        if not self._probe_frozen and not (self._probe_forget and self._probe_retain):
            return
        was_training = model.training
        model.eval()
        try:
            fg = self._probe_ce(model, self._probe_forget)
            rt = self._probe_ce(model, self._probe_retain)
        finally:
            if was_training:
                model.train()
        if fg is None or rt is None:
            return
        if self._probe_forget_baseline is None:
            self._probe_forget_baseline = fg["mean"]
            self._probe_retain_baseline = rt["mean"]
            logger.info(
                f"[signal_monitor] baselines: forget_CE={fg['mean']:.4f} "
                f"retain_CE={rt['mean']:.4f} (step {self.state.global_step})"
            )
        # Cache per-token baselines once (drift / top-1 agreement signals).
        if not self._probe_base:
            self._probe_base = {
                "forget": {"tok_ce": fg["tok_list"], "argmax": fg["argmax_list"]},
                "retain": {"tok_ce": rt["tok_list"], "argmax": rt["argmax_list"]},
            }

        def _drift_agree(now: dict, base: dict) -> tuple:
            """(mean |tok_ce_now - tok_ce_base|, top-1 agreement fraction).
            Top-1 agreement on the FORGET probe is a cheap teacher-forced proxy
            for verbatim reproduction (verbmem ROUGE): it decays as the model
            escapes rote memorisation. On the RETAIN probe, a falling agreement
            is a generation-drift precursor that shows up BEFORE the mean
            retain CE rises (the collapse guard's blind spot)."""
            drifts, agrees, n_tok = [], 0, 0
            for ce_now, ce_base, am_now, am_base in zip(
                now["tok_list"], base["tok_ce"],
                now["argmax_list"], base["argmax"],
            ):
                m = min(ce_now.numel(), ce_base.numel())
                if m == 0:
                    continue
                drifts.append((ce_now[:m] - ce_base[:m]).abs().mean())
                agrees += int((am_now[:m] == am_base[:m]).sum())
                n_tok += m
            if not drifts:
                return None, None
            return (
                float(torch.stack(drifts).mean()),
                agrees / max(n_tok, 1),
            )

        fg_drift, fg_agree = _drift_agree(fg, self._probe_base["forget"])
        rt_drift, rt_agree = _drift_agree(rt, self._probe_base["retain"])
        # Fraction of forget-probe tokens already above the CET target tau
        # (token-level overshoot -- the min-k++/PrivLeak driver).
        tau_ref = self.forget_loss_answer_target or 0.0
        if tau_ref > 0:
            all_tok = torch.cat(fg["tok_list"])
            fg_frac_above_tau = float((all_tok > tau_ref).float().mean())
        else:
            fg_frac_above_tau = None

        retain_rise = rt["mean"] - (self._probe_retain_baseline or 0.0)
        forget_rise = fg["mean"] - (self._probe_forget_baseline or 0.0)
        done_total = len(self._dyn_done_keys) if hasattr(self, "_dyn_done_keys") else None

        rec = {
            "step": int(self.state.global_step),
            "epoch": round(float(self.state.epoch or 0.0), 4),
            "lr": float(self._last_lr()) if hasattr(self, "_last_lr") else None,
            # forget probe (rises 0.64 -> tau as text becomes un-memorised)
            "forget_ce_probe": round(fg["mean"], 5),
            "forget_ce_probe_std": round(fg["std"], 5),
            "forget_ce_rise": round(forget_rise, 5),
            "forget_tok_std": round(fg["tok_std"], 5),
            "forget_tok_p10": round(fg["tok_p10"], 5),
            "forget_tok_p50": round(fg["tok_p50"], 5),
            "forget_tok_p90": round(fg["tok_p90"], 5),
            "forget_tok_max": round(fg["tok_max"], 5),
            "forget_frac_tok_above_tau": (round(fg_frac_above_tau, 5)
                                          if fg_frac_above_tau is not None else None),
            # top-1 agreement vs step-0 self: forget ~= verbatim-reproduction
            # proxy (escape tracker); retain ~= generation-drift precursor.
            "forget_top1_agree": round(fg_agree, 5) if fg_agree is not None else None,
            "retain_top1_agree": round(rt_agree, 5) if rt_agree is not None else None,
            "forget_tok_ce_drift": round(fg_drift, 5) if fg_drift is not None else None,
            "retain_tok_ce_drift": round(rt_drift, 5) if rt_drift is not None else None,
            # retain probe (collapse guard: should stay flat)
            "retain_ce_probe": round(rt["mean"], 5),
            "retain_ce_probe_std": round(rt["std"], 5),
            "retain_ce_rise": round(retain_rise, 5),
            # optimiser / drift candidate signals
            "grad_norm": self._last_grad_norm,
            "grad_norm_ema": (round(self._grad_norm_ema, 4)
                              if self._grad_norm_ema is not None else None),
            "weight_rel_delta": self._last_weight_rel_delta,
            # forget loss-shape candidate signals
            "forget_entropy": self._last_forget_entropy,
            "retain_entropy": self._last_retain_entropy,
            # MIN-K++ core statistic (z of target token under own softmax)
            "forget_z_p10": self._last_forget_z_p10,
            "forget_z_mean": self._last_forget_z_mean,
            "dynstop_done_total": done_total,
            "tok_frac_below_band": self._last_tok_frac_below,
            "tok_frac_in_band": self._last_tok_frac_in,
            "tok_frac_above_band": self._last_tok_frac_above,
        }
        self._last_signal_record = rec
        self._write_signal_record(rec)

        # ----- automatic stop rule (forget/retain only) -----
        if self.state.global_step < self.stop_monitor_warmup_steps:
            return

        # Deep-collapse bail-out: REAL stop even in checkpoint-only mode --
        # past this point the model is unrecoverable, further compute wasted.
        if (
            self.hard_stop_retain_ce_rise > 0
            and retain_rise >= self.hard_stop_retain_ce_rise
        ):
            reason = (
                f"HARD STOP (deep collapse): retain CE rose {retain_rise:.4f} >= "
                f"{self.hard_stop_retain_ce_rise:.4f} nats above baseline "
                f"{self._probe_retain_baseline:.4f}"
            )
            logger.info(
                f"[signal_monitor] STOPPING at step {self.state.global_step}: {reason}"
            )
            rec["stop_reason"] = reason
            self._write_signal_record({"step": int(self.state.global_step),
                                       "event": "hard_stop", "reason": reason})
            if hasattr(self, "control") and self.control is not None:
                self.control.should_training_stop = True
            return

        stop_reason, stop_kind = None, None
        if self.stop_retain_ce_rise > 0 and retain_rise >= self.stop_retain_ce_rise:
            stop_kind = "collapse"
            stop_reason = (
                f"COLLAPSE GUARD: retain CE rose {retain_rise:.4f} >= "
                f"{self.stop_retain_ce_rise:.4f} nats above baseline "
                f"{self._probe_retain_baseline:.4f}"
            )
        if stop_reason is None and self.stop_forget_ce_target > 0:
            if fg["mean"] >= self.stop_forget_ce_target:
                self._monitor_forget_hits += 1
                if self._monitor_forget_hits >= self.stop_patience:
                    stop_kind = "forget-target"
                    stop_reason = (
                        f"forget CE {fg['mean']:.4f} >= target "
                        f"{self.stop_forget_ce_target:.4f} for "
                        f"{self._monitor_forget_hits} consecutive hits"
                    )
            else:
                self._monitor_forget_hits = 0
        if stop_reason is not None:
            if self.stop_signal_checkpoint_only:
                # Fire once per rule kind: save a signal checkpoint + eval at
                # a safe point (after the optimiser step), then keep training.
                if stop_kind not in self._signal_ckpt_fired:
                    self._signal_ckpt_fired.add(stop_kind)
                    logger.info(
                        f"[signal_monitor] SIGNAL FIRED (checkpoint-only) at step "
                        f"{self.state.global_step}: {stop_reason}"
                    )
                    rec["stop_signal"] = stop_reason
                    self._write_signal_record({
                        "step": int(self.state.global_step),
                        "event": "stop_signal", "kind": stop_kind,
                        "reason": stop_reason,
                    })
                    self._pending_signal_ckpt = (stop_kind, stop_reason)
            else:
                logger.info(
                    f"[signal_monitor] STOPPING at step {self.state.global_step}: "
                    f"{stop_reason}"
                )
                rec["stop_reason"] = stop_reason
                self._write_signal_record({"step": int(self.state.global_step),
                                           "event": "stop", "reason": stop_reason})
                if hasattr(self, "control") and self.control is not None:
                    self.control.should_training_stop = True

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """After the optimiser step (outside compute_loss, activations freed),
        materialise any pending earlystop-signal checkpoint."""
        out = super()._maybe_log_save_evaluate(*args, **kwargs)
        if self._pending_signal_ckpt is not None:
            kind, reason = self._pending_signal_ckpt
            self._pending_signal_ckpt = None
            self._do_signal_checkpoint(kind, reason)
        return out

    def _do_signal_checkpoint(self, kind: str, reason: str) -> None:
        """Log the early-stop signal, write a marker JSON, and run an immediate
        eval (model is already in memory). No model weights are saved to disk.
        Best-effort: failures must never abort training."""
        step = int(self.state.global_step)
        out = os.path.join(
            self.args.output_dir, f"checkpoint-{step}-earlystop-{kind}"
        )
        try:
            os.makedirs(out, exist_ok=True)
            marker = {
                "step": step,
                "epoch": round(float(self.state.epoch or 0.0), 4),
                "kind": kind,
                "reason": reason,
            }
            with open(os.path.join(out, "EARLYSTOP_SIGNAL.json"), "w",
                      encoding="utf-8") as f:
                json.dump(marker, f, indent=2)
            logger.info(f"[signal_monitor] signal fired at step {step}: {out}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[signal_monitor] signal marker write failed: {e}")
        try:
            if getattr(self, "evaluators", None):
                logger.info(
                    f"[signal_monitor] running eval at signal step {step}"
                )
                self.evaluate()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[signal_monitor] signal-step eval failed: {e}")

    def _last_lr(self) -> float:
        try:
            return float(self.lr_scheduler.get_last_lr()[0])
        except Exception:  # noqa: BLE001
            return float("nan")

    def _dump_forget_token_ce(self, model) -> None:
        """No-grad forward over the ENTIRE forget set; save each sample's
        per-token CE (fp16, supervised positions only) to
        <output_dir>/token_ce_dumps/ep<epoch>_step<gs>.npz. Keyed s0000..sNNNN
        in dataset order, so dumps are comparable across epochs sample-by-
        sample and token-by-token (offline collapse-precursor mining)."""
        import numpy as _np
        import torch.nn.functional as _F
        try:
            forget_ds = self.train_dataset.forget  # type: ignore[attr-defined]
        except AttributeError:
            logger.warning("[token_ce_dump] train_dataset has no .forget; skipping")
            self.token_ce_dump_epochs = 0.0  # don't retry every step
            return
        n = len(forget_ds)
        if self.token_ce_dump_max_samples > 0:
            n = min(n, self.token_ce_dump_max_samples)
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        t0 = time.time()
        arrays: dict = {}
        try:
            with torch.no_grad():
                for i in range(n):
                    item = forget_ds[i]
                    # MUSE completion items / TOFU QA items may be nested one
                    # level (e.g. {"original": ..., "alternate": ...}).
                    if "input_ids" not in item and "original" in item:
                        item = item["original"]
                    ids = item["input_ids"]
                    lbls = item["labels"]
                    if not isinstance(ids, torch.Tensor):
                        ids = torch.tensor(ids)
                    if not isinstance(lbls, torch.Tensor):
                        lbls = torch.tensor(lbls)
                    if ids.dim() == 1:
                        ids = ids.unsqueeze(0)
                        lbls = lbls.unsqueeze(0)
                    am = item.get("attention_mask")
                    if am is None:
                        am = torch.ones_like(ids)
                    elif not isinstance(am, torch.Tensor):
                        am = torch.tensor(am)
                    if am.dim() == 1:
                        am = am.unsqueeze(0)
                    out = model(input_ids=ids.to(device),
                                attention_mask=am.to(device))
                    logits = out.logits[:, :-1, :]
                    labels = lbls.to(device)[:, 1:]
                    mask = (labels != -100)
                    if mask.sum() == 0:
                        arrays[f"s{i:04d}"] = _np.zeros(0, dtype=_np.float16)
                        continue
                    ce = _F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        labels.reshape(-1),
                        reduction="none",
                    )[mask.reshape(-1)]
                    arrays[f"s{i:04d}"] = (
                        ce.detach().cpu().numpy().astype(_np.float16)
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[token_ce_dump] failed at sample {len(arrays)}: {e}")
            if was_training:
                model.train()
            return
        finally:
            if was_training:
                model.train()
        out_dir = os.path.join(
            getattr(self.args, "output_dir", ".") or ".", "token_ce_dumps"
        )
        try:
            os.makedirs(out_dir, exist_ok=True)
            ep = float(self.state.epoch or 0.0)
            gs = int(self.state.global_step)
            path = os.path.join(out_dir, f"ep{ep:07.2f}_step{gs:06d}.npz")
            _np.savez_compressed(
                path,
                _meta=_np.array([ep, float(gs)]),
                **arrays,
            )
            logger.info(
                f"[token_ce_dump] wrote {len(arrays)} samples to {path} "
                f"({time.time() - t0:.1f}s)"
            )
        except OSError as e:
            logger.warning(f"[token_ce_dump] could not write dump: {e}")

    def _write_signal_record(self, rec: dict) -> None:
        import json as _json
        if self._signal_log_path is None:
            path = self.signal_monitor_log
            if not path:
                out = getattr(self.args, "output_dir", ".") or "."
                path = os.path.join(out, "signal_monitor.jsonl")
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            except OSError:
                pass
            self._signal_log_path = path
        try:
            with open(self._signal_log_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
        except OSError as e:  # noqa: BLE001
            logger.warning(f"[signal_monitor] could not write log: {e}")

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        self._answer_mask = None
        if self._needs_hooks:
            self._ensure_hooks(model)
            if self.answer_only_down_proj_grad and self.topk_vjp_count <= 0:
                forget = inputs.get("forget")
                if forget is not None and forget.get("labels") is not None:
                    self._answer_mask = forget["labels"] != -100
            self._hook_x.clear()
            self._hook_delta.clear()

        loss = super().training_step(model, inputs, num_items_in_batch)

        with torch.no_grad():
            for layer_idx in self.target_layers:
                w2 = self._get_down_proj(model, layer_idx)

                if w2.weight.grad is None:
                    continue
                d_out, d_in = w2.weight.shape

                x = self._hook_x.get(layer_idx)
                delta = self._hook_delta.get(layer_idx)

                need_filter = (
                    (self.topk_vjp_count > 0
                     or self.author_only_vjp
                     or (self.answer_only_down_proj_grad and self._answer_mask is not None))
                    and x is not None and delta is not None
                )

                filtered_forget = None
                if need_filter:
                    if self.author_only_vjp:
                        # author_only_vjp takes priority: keep only positions whose token-id
                        # belongs to the precomputed forget name/entity spans.
                        forget_input_ids = inputs["forget"]["input_ids"]  # [B, T]
                        author_mask_bt = self._author_mask_for(forget_input_ids).to(x.device)
                        n_rows = author_mask_bt.shape[0]
                        n_cov = n_rows
                        mask_flat = author_mask_bt.reshape(-1)
                        n_selected = int(mask_flat.sum())
                        # Debug log: how many positions contributed this step
                        if layer_idx == self.target_layers[0]:
                            self._last_span_coverage = n_cov / max(1, n_rows)
                            self._last_vjp_selected_pct = (
                                100.0 * n_selected / max(1, mask_flat.numel())
                            )
                            logger.info(
                                f"[author_vjp] step={self.state.global_step} "
                                f"layer={layer_idx} n_author_tokens={n_selected} "
                                f"of {mask_flat.numel()} "
                                f"({self._last_vjp_selected_pct:.2f}%) "
                                f"span_coverage={n_cov}/{n_rows} rows"
                            )
                            # Hard-fail guard: if author masking is on but no
                            # author tokens are matched in the early batches,
                            # the cached author-name spans almost certainly
                            # belong to a different tokenizer (the original
                            # bug that produced no-op 7B ERASE runs). Raising
                            # immediately is much safer than silently training
                            # on a zero gradient for hours. Skipped in
                            # forget_span_cache mode, where empty matches are
                            # expected.
                            self._author_zero_streak = (
                                getattr(self, "_author_zero_streak", 0) + 1
                                if n_selected == 0
                                else 0
                            )
                            if (
                                self._author_zero_streak >= 5
                                and not self.forget_span_cache
                            ):
                                raise RuntimeError(
                                    "ERASE author_only_vjp: 5 consecutive batches "
                                    "produced zero author-token matches. The cached "
                                    "author spans were likely built with a different "
                                    "tokenizer than the active model. Delete "
                                    "saves/precompute/forget10_author_name_spans__*.json "
                                    "and rerun, or set author_mask_mode='off' / "
                                    "author_only_vjp=False if intentional."
                                )
                        xf = x.reshape(-1, d_in)[mask_flat]
                        gf = delta.reshape(-1, d_out)[mask_flat]
                        if xf.shape[0] > 0:
                            filtered_forget = gf.T @ xf.to(dtype=gf.dtype)
                        else:
                            filtered_forget = torch.zeros(
                                d_out, d_in, device=w2.weight.device, dtype=delta.dtype,
                            )
                    elif self.topk_vjp_count > 0:
                        delta_flat = delta.reshape(-1, d_out)
                        x_flat = x.reshape(-1, d_in)
                        norms = delta_flat.norm(dim=-1)
                        k = min(self.topk_vjp_count, norms.shape[0])
                        topk_idx = norms.topk(k).indices
                        xf = x_flat[topk_idx]
                        gf = delta_flat[topk_idx]
                        filtered_forget = gf.T @ xf.to(dtype=gf.dtype)
                        n_selected = k
                    else:
                        # answer_only_down_proj_grad: sum the VJP outer product
                        # over answer (labelled) positions only. For MUSE this
                        # is the whole supervised chunk.
                        m = self._answer_mask.to(device=x.device)
                        mask_flat = m.reshape(-1)
                        xf = x.reshape(-1, d_in)[mask_flat]
                        gf = delta.reshape(-1, d_out)[mask_flat]
                        if xf.shape[0] > 0:
                            filtered_forget = gf.T @ xf.to(dtype=gf.dtype)
                        else:
                            filtered_forget = torch.zeros(
                                d_out, d_in, device=w2.weight.device, dtype=delta.dtype,
                            )
                        n_selected = 0  # no renorm for answer_only

                    # VJP renormalization: undo the 1/m factor from mean-reduced
                    # CE and replace with 1/|S| so every question contributes
                    # equally regardless of the number of selected positions.
                    if (self.vjp_renormalize
                            and filtered_forget is not None
                            and n_selected > 0):
                        forget_labels = inputs["forget"].get("labels")
                        if forget_labels is not None:
                            n_answer = int((forget_labels != -100).sum())
                        else:
                            n_answer = n_selected
                        if n_answer > 0 and n_answer != n_selected:
                            renorm_factor = n_answer / n_selected
                            filtered_forget = filtered_forget * renorm_factor
                            if layer_idx == self.target_layers[0]:
                                logger.info(
                                    f"[vjp_renorm] step={self.state.global_step} "
                                    f"n_answer={n_answer}, n_selected={n_selected}, "
                                    f"factor={renorm_factor:.3f}"
                                )

                P = self.p_matrices[layer_idx].to(
                    dtype=w2.weight.grad.dtype, device=w2.weight.grad.device
                )

                if filtered_forget is not None:
                    w2.weight.grad = filtered_forget.to(dtype=w2.weight.grad.dtype)
                w2.weight.grad = w2.weight.grad @ P.T

            if self._needs_hooks:
                self._hook_x.clear()
                self._hook_delta.clear()

        return loss
