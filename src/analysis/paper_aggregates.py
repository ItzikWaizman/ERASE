"""Paper-style aggregate scores for OpenUnlearning Table 3.

Computes Memorization (Mem.), Utility, and Aggregate (Agg.) scores from a
TOFU_EVAL.json produced by the standard eval pipeline.

Source: OpenUnlearning paper (arXiv:2506.12618), Appendix F.1
"Aggregating metric scores".

    Mem  = HM(1-ES, 1-EM, 1-Para_Prob, 1-Truth_Ratio_paper)
    Util = HM(MU/MU_init, fluency/fluency_init)                 # paper Table 3 style
    Priv = HM(s_LOSS, s_ZLib, s_MinK, s_MinK++)
           where s_MIA = exp(-alpha * |MIA_method - MIA_retain|)
           alpha = ln(10) / 0.6  ~ 3.84  (calibrated so Priv(init) ~ 0.10)
    Agg  = HM(Mem, Priv, Util)   (Priv dropped if no retain ref provided)

IMPORTANT: the metric named `forget_Q_A_gibberish` in TOFU_EVAL.json is
actually the probability of class 0 of the gibberish detector, which is
"clean" (not gibberish). So we treat that field directly as the FLUENCY
score; do NOT do `1 - x`. See classifier id2label on the HF model card
(class 0 = clean, 1 = mild_gibberish, 2 = noise, 3 = word_salad).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


# Mapping from paper-aggregate inputs to the metric_name keys in TOFU_EVAL.json.
MEM_INPUT_METRICS = {
    "ES": "extraction_strength",
    "EM": "exact_memorization",
    "PARA_PROB": "forget_Q_A_PARA_Prob",
    "TRUTH_RATIO": "forget_truth_ratio_paper",  # paper aggregator: correct/(correct+wrong)
}
# Fallback: if the paper-flavored key isn't present, fall back to the original
# TOFU one. Documented so users can see which one fed the aggregate.
MEM_INPUT_FALLBACKS = {
    "TRUTH_RATIO": "forget_truth_ratio",
}

UTIL_INPUT_METRICS = {
    "MU": "model_utility",
    # NOTE: the field named `forget_Q_A_gibberish` is actually P(class_0) where
    # class 0 of the underlying classifier is "clean", so this value IS the
    # fluency score directly (NOT 1 - fluency). Don't invert it.
    "FLUENCY": "forget_Q_A_gibberish",
}

PRIV_INPUT_METRICS = {
    "MIA_LOSS": "mia_loss",
    "MIA_ZLIB": "mia_zlib",
    "MIA_MINK": "mia_min_k",
    "MIA_MINKPP": "mia_min_k_plus_plus",
}


def _safe_hm(values: Iterable[float]) -> float | None:
    """Harmonic mean with the paper's convention: returns 0 if any value is
    <= 0 or NaN, returns None if no usable values are present.
    """
    vs = [float(v) for v in values if v is not None]
    if not vs:
        return None
    if any((not math.isfinite(v)) for v in vs):
        return 0.0
    if any(v <= 0 for v in vs):
        return 0.0
    return len(vs) / sum(1.0 / v for v in vs)


def _agg_value(eval_dict: dict, metric_key: str) -> float | None:
    entry = eval_dict.get(metric_key)
    if not isinstance(entry, dict):
        return None
    v = entry.get("agg_value")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Exponential decay rate for the per-MIA alignment formula. Calibrated so
# that Priv(init_finetuned) ~ 0.10 at the empirical |Δ_MIA| ~ 0.6:
#   exp(-alpha * 0.6) = 0.10  =>  alpha = ln(10) / 0.6  ~  3.838
# Exponential is preferred over the previous linear (1 - C*|Δ|) form because
# it never clips to 0, so a single misaligned MIA can't collapse the HM.
_PRIV_ALIGNMENT_ALPHA = math.log(10.0) / 0.6  # ~ 3.838


def _priv_alignment_score(mia_method: float | None, mia_retain: float | None) -> float | None:
    """Per-MIA alignment-with-retain score s_MIA in (0, 1].

    The OpenUnlearning paper says s_MIA measures how closely the unlearned
    model's MIA value aligns with the retain oracle's, but does NOT give a
    closed-form formula in the appendix excerpt available. We use:

        s_MIA = exp(-alpha * |MIA_method - MIA_retain|)

    Boundary conditions:
      - retain vs retain: |0|     -> s = 1.0   (Table 3: Retain Priv = 1.00) ok
      - init  vs retain: |~0.6|   -> s ~= 0.10 -> Priv ~= 0.10 (matches paper)
    """
    if mia_method is None or mia_retain is None:
        return None
    diff = abs(float(mia_method) - float(mia_retain))
    return float(math.exp(-_PRIV_ALIGNMENT_ALPHA * diff))


def compute_aggregates(
    eval_dict: dict,
    init_eval_dict: dict | None = None,
    retain_eval_dict: dict | None = None,
) -> dict:
    """Compute paper-style Mem / Util / Priv / Agg from a TOFU_EVAL dict.

    Args:
        eval_dict: TOFU_EVAL.json contents for the method being evaluated.
        init_eval_dict: TOFU_EVAL.json for the init-finetuned model. Required
            for Util normalization (Util = HM(MU/MU_init, fluency/fluency_init)).
            Without it, Util reduces to HM(MU, fluency) -- which is mostly only
            useful as a sanity check, not for paper-comparable numbers.
        retain_eval_dict: TOFU_EVAL.json for the retain oracle. Required for
            Privacy: s_MIA = exp(-alpha * |MIA_method - MIA_retain|), then
            Priv = HM of the four s_MIA values.
    """
    # ---- Memorization (no normalization; metrics already in [0,1]) ----
    mem_raw = {}
    mem_source = {}
    for k, primary in MEM_INPUT_METRICS.items():
        v = _agg_value(eval_dict, primary)
        if v is None and k in MEM_INPUT_FALLBACKS:
            fb = MEM_INPUT_FALLBACKS[k]
            v = _agg_value(eval_dict, fb)
            mem_source[k] = f"fallback:{fb}"
        else:
            mem_source[k] = primary if v is not None else None
        mem_raw[k] = v
    mem_inverted = {k: (1.0 - v) if v is not None else None for k, v in mem_raw.items()}
    mem_score = _safe_hm(v for v in mem_inverted.values() if v is not None) if all(
        v is not None for v in mem_inverted.values()
    ) else None

    # ---- Utility (paper-style: HM of normalized MU and normalized fluency) ----
    # The TOFU_EVAL field "forget_Q_A_gibberish" is misleadingly named: it's the
    # probability of class 0 of the gibberish detector, and class 0 is "clean",
    # so the value IS the fluency directly. Do NOT compute 1 - gibberish.
    mu = _agg_value(eval_dict, UTIL_INPUT_METRICS["MU"])
    fluency = _agg_value(eval_dict, UTIL_INPUT_METRICS["FLUENCY"])

    util_normalizer = None
    mu_norm = mu
    fluency_norm = fluency
    if init_eval_dict is not None:
        mu_init = _agg_value(init_eval_dict, UTIL_INPUT_METRICS["MU"])
        fluency_init = _agg_value(init_eval_dict, UTIL_INPUT_METRICS["FLUENCY"])
        util_normalizer = {
            "model_utility_init": mu_init,
            "fluency_init": fluency_init,
        }
        if mu is not None and mu_init is not None and mu_init > 0:
            # Cap at 1.0: a slightly-better-than-init method (numerical noise)
            # shouldn't push the normalized score above the reference.
            mu_norm = min(mu / mu_init, 1.0)
        if fluency is not None and fluency_init is not None and fluency_init > 0:
            fluency_norm = min(fluency / fluency_init, 1.0)
    util_score = (
        _safe_hm([mu_norm, fluency_norm])
        if (mu_norm is not None and fluency_norm is not None)
        else None
    )

    util_inputs = {
        "model_utility_raw": mu,
        "fluency_raw": fluency,
        "model_utility_normalized": mu_norm if init_eval_dict is not None else None,
        "fluency_normalized": fluency_norm if init_eval_dict is not None else None,
        "normalizer": util_normalizer,
    }

    # ---- Privacy (computed iff retain oracle eval provided) ----
    priv_raw = {k: _agg_value(eval_dict, m) for k, m in PRIV_INPUT_METRICS.items()}
    priv_alignment = None
    priv_score = None
    priv_retain_raw = None
    if retain_eval_dict is not None:
        priv_retain_raw = {
            k: _agg_value(retain_eval_dict, m)
            for k, m in PRIV_INPUT_METRICS.items()
        }
        priv_alignment = {
            k: _priv_alignment_score(priv_raw[k], priv_retain_raw[k])
            for k in PRIV_INPUT_METRICS
        }
        usable = [v for v in priv_alignment.values() if v is not None]
        if usable:
            priv_score = _safe_hm(usable)

    # ---- Aggregate ----
    parts = [v for v in (mem_score, priv_score, util_score) if v is not None]
    agg_score = _safe_hm(parts) if parts else None

    return {
        "aggregates": {
            "memorization": mem_score,
            "utility": util_score,
            "privacy": priv_score,
            "aggregate": agg_score,
        },
        "inputs": {
            "memorization_raw": mem_raw,
            "memorization_inverted": mem_inverted,
            "memorization_source_keys": mem_source,
            "utility": util_inputs,
            "privacy_raw_mia": priv_raw,
            "privacy_alignment_scores": priv_alignment,
            "privacy_retain_reference": priv_retain_raw,
        },
        "notes": {
            "utility_formula": (
                "Util = HM(MU/MU_init, fluency/fluency_init). The "
                "`forget_Q_A_gibberish` field is actually P(class_0='clean') "
                "and is used directly as fluency (NOT 1 - x)."
            ),
            "privacy_formula": (
                "s_MIA = exp(-alpha * |MIA_method - MIA_retain|), "
                f"alpha = ln(10)/0.6 ~ {_PRIV_ALIGNMENT_ALPHA:.3f}; "
                "Priv = HM(s_LOSS, s_ZLib, s_MinK, s_MinK++). "
                "Calibrated so Priv(retain)=1.0 (boundary) and "
                "Priv(init_finetuned) ~ 0.10 (paper Table 3 boundary)."
            ),
            "aggregate_formula": (
                "Agg = HM(Mem, Priv, Util) (paper Section 5). When the retain "
                "ref isn't provided Priv is dropped and Agg = HM(Mem, Util)."
            ),
        },
    }


def render_markdown_table(aggregates_payload: dict, run_name: str) -> str:
    a = aggregates_payload["aggregates"]
    inp = aggregates_payload["inputs"]
    util_in = inp["utility"]
    norm = util_in.get("normalizer")

    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [f"# Paper Aggregates: `{run_name}`\n"]

    lines.append("## Aggregate scores (OpenUnlearning Table 3 style)\n")
    lines.append("| Score | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Memorization | {fmt(a['memorization'])} |")
    lines.append(f"| Utility | {fmt(a['utility'])} |")
    lines.append(f"| Privacy | {fmt(a['privacy'])} |")
    lines.append(f"| **Aggregate** | **{fmt(a['aggregate'])}** |")

    lines.append("\n## Memorization inputs (raw and 1 - x)\n")
    lines.append("| Metric | Raw | 1 - Raw (used in HM) |")
    lines.append("| --- | ---: | ---: |")
    for k in MEM_INPUT_METRICS:
        raw = inp["memorization_raw"].get(k)
        inv = inp["memorization_inverted"].get(k)
        lines.append(f"| {k} | {fmt(raw)} | {fmt(inv)} |")

    lines.append("\n## Utility inputs (raw vs normalized by init-finetuned)\n")
    lines.append("| Field | Raw | Normalized (used in HM) |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| model_utility | {fmt(util_in.get('model_utility_raw'))} | "
                 f"{fmt(util_in.get('model_utility_normalized'))} |")
    lines.append(f"| fluency (= P(clean)) | {fmt(util_in.get('fluency_raw'))} | "
                 f"{fmt(util_in.get('fluency_normalized'))} |")
    if norm is not None:
        lines.append(f"\n_Init-finetuned reference: MU={fmt(norm.get('model_utility_init'))}, "
                     f"fluency={fmt(norm.get('fluency_init'))}_")

    lines.append("\n## Privacy: per-MIA alignment with retain oracle\n")
    lines.append("s_MIA = exp(-alpha * |MIA_method - MIA_retain|), alpha = "
                 f"{_PRIV_ALIGNMENT_ALPHA:.3f}. Priv = HM of the four s_MIAs.\n")
    lines.append("| MIA | Method | Retain ref | s_MIA |")
    lines.append("| --- | ---: | ---: | ---: |")
    align = inp.get("privacy_alignment_scores") or {}
    ref = inp.get("privacy_retain_reference") or {}
    for k in PRIV_INPUT_METRICS:
        method_v = inp["privacy_raw_mia"].get(k)
        ref_v = ref.get(k) if isinstance(ref, dict) else None
        s = align.get(k) if isinstance(align, dict) else None
        lines.append(f"| {k} | {fmt(method_v)} | {fmt(ref_v)} | {fmt(s)} |")

    return "\n".join(lines) + "\n"


def _load_eval_dict(eval_dir: Path | None) -> dict | None:
    if eval_dir is None:
        return None
    p = Path(eval_dir)
    cand = p / "TOFU_EVAL.json" if p.is_dir() else p
    if not cand.is_file():
        return None
    try:
        return json.loads(cand.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def aggregate_eval_dir(
    eval_dir: Path,
    init_eval_dir: Path | None = None,
    retain_eval_dir: Path | None = None,
) -> dict | None:
    """Read TOFU_EVAL.json from `eval_dir`, write PAPER_AGGREGATES.{json,md}.

    Optional args `init_eval_dir` and `retain_eval_dir` enable normalization
    (Util) and Privacy computation respectively. They can be either the run
    directory (containing TOFU_EVAL.json) or the JSON file path itself.

    Returns the aggregates payload on success, or None if TOFU_EVAL.json is
    missing / malformed.
    """
    eval_dir = Path(eval_dir)
    eval_dict = _load_eval_dict(eval_dir)
    if eval_dict is None:
        return None
    init_dict = _load_eval_dict(init_eval_dir)
    retain_dict = _load_eval_dict(retain_eval_dir)

    payload = compute_aggregates(eval_dict, init_dict, retain_dict)
    out_json = eval_dir / "PAPER_AGGREGATES.json"
    out_md = eval_dir / "PAPER_AGGREGATES.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run_name = eval_dir.parent.name if eval_dir.name == "evals" else eval_dir.name
    out_md.write_text(render_markdown_table(payload, run_name), encoding="utf-8")
    return payload
