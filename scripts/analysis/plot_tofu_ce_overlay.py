"""Overlay forget-set CE (avg_loss) distributions: unlearned run vs oracle, for
Phi-3.5 (problem case) and Llama-1B (good case). Diagnoses why Phi privacy lags.

The TOFU privacy/forget_quality signal is how closely the unlearned model's
forget-set loss distribution matches the RETRAIN ORACLE's. If our run's forget
CE is shifted/peaked vs the oracle, MIA can separate them -> bad privacy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "phi_run": ROOT / "results/remote_phi/A18P_b1.85_3.85_tgt2.85_L0_5_a2.5_lr0.045_dyn1.5U5_300ep_wiki_lr0.045_b0_a2.5_k10_cfauth_flw0.9_atgt2.85_cosine_L012345_authoronly_ammspan_ebs8_det_dyn1.5U5.0/evals/TOFU_EVAL.json",
    "phi_oracle": ROOT / "saves/eval/phi35/retain_oracle_v2/TOFU_EVAL.json",
    "1b_run": ROOT / "saves/highlights/1B_WNINER_MSE/evals/TOFU_EVAL.json",
    "1b_oracle": ROOT / "saves/eval/tofu_llama-1b_oracle_retain90/TOFU_EVAL.json",
}


def forget_ce(d):
    vbi = d["forget_Q_A_Prob"]["value_by_index"]
    return np.array([float(v["avg_loss"]) for v in vbi.values()])


def agg(d, key):
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("agg_value", v.get("auc"))
    return v


def main():
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in MODELS.items() if p.is_file()}
    for k, p in MODELS.items():
        if not p.is_file():
            print(f"[MISSING] {k}: {p}")

    # Metrics table
    def _f(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) else "n/a".rjust(len(format(0, fmt)))
    print(f"\n{'model':<12} {'fq':>10} {'privleak':>9} {'mutil':>7} {'mia_mk':>7} {'mia_loss':>8} {'ce_mean':>8} {'ce_std':>7}")
    print("-" * 80)
    for k, d in data.items():
        ce = forget_ce(d)
        print(f"{k:<12} {_f(agg(d,'forget_quality'),'10.2e')} {_f(agg(d,'privleak'),'9.1f')} "
              f"{_f(agg(d,'model_utility'),'7.3f')} {_f(agg(d,'mia_min_k'),'7.3f')} "
              f"{_f(agg(d,'mia_loss'),'8.3f')} {ce.mean():>8.3f} {ce.std():>7.3f}")

    # Overlay plot: 2 panels (Phi, 1B), each: run (red) vs oracle (green)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairs = [("Phi-3.5 (problem)", "phi_run", "phi_oracle", axes[0]),
             ("Llama-1B (good)", "1b_run", "1b_oracle", axes[1])]
    for title, run_k, ora_k, ax in pairs:
        if run_k not in data or ora_k not in data:
            ax.set_title(f"{title}: missing data")
            continue
        run_ce = forget_ce(data[run_k])
        ora_ce = forget_ce(data[ora_k])
        lo = min(run_ce.min(), ora_ce.min())
        hi = max(run_ce.max(), ora_ce.max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(ora_ce, bins=bins, alpha=0.55, color="tab:green",
                density=True, label=f"oracle (mean {ora_ce.mean():.2f}, std {ora_ce.std():.2f})")
        ax.hist(run_ce, bins=bins, alpha=0.55, color="tab:red",
                density=True, label=f"our run (mean {run_ce.mean():.2f}, std {run_ce.std():.2f})")
        ax.axvline(ora_ce.mean(), color="tab:green", ls="--", lw=1.5)
        ax.axvline(run_ce.mean(), color="tab:red", ls="--", lw=1.5)
        fq = agg(data[run_k], "forget_quality")
        pl = agg(data[run_k], "privleak")
        fq_s = f"{fq:.2e}" if isinstance(fq, (int, float)) else "n/a"
        pl_s = f"{pl:.1f}" if isinstance(pl, (int, float)) else "n/a"
        ax.set_title(f"{title}\nforget_quality={fq_s}  privleak={pl_s}", fontsize=10)
        ax.set_xlabel("forget per-sample CE (avg_loss, nats)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Forget-set CE distribution: unlearned run vs retrain oracle", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "results/remote_phi/phi_vs_1b_ce_overlay.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
