"""R2 rebuttal summary: seed variance of ERASE vs AltPO on TOFU forget10 / Llama-3.2-1B.

Collects PAPER_AGGREGATES.json from the 5-seed ERASE runs (task prefixes
R2_ERASE_s0..s4, searched in saves/unlearn AND saves/highlights since the ERASE
runner moves highlight-worthy runs) and the 5-seed AltPO retraining runs
(saves/unlearn/llama1b_AltPO/lr2e-05_ep10_alpha2_beta0.05_gamma1_seed{n}).

Writes results/EMBLPRebuttal/r2_seed_variance.md with mean +/- std per pillar
and Welch t-tests, plus the raw per-seed table. Safe to re-run any time; it
summarizes whatever runs have finished so far.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "results" / "EMBLPRebuttal" / "r2_seed_variance.md"

SEEDS = [0, 1, 2, 3, 4]
PILLARS = ["memorization", "privacy", "utility", "aggregate"]

# Paper single-seed point estimates (for the reproduction check).
PAPER_ERASE_AGG = 0.786
PAPER_ALTPO_AGG = 0.766


def _find_erase_dir(seed: int) -> Path | None:
    prefix = f"R2_ERASE_s{seed}_"
    for base in (ROOT / "saves" / "unlearn", ROOT / "saves" / "highlights"):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if d.name.startswith(prefix) and (d / "evals" / "PAPER_AGGREGATES.json").is_file():
                return d
    return None


def _find_altpo_dir(seed: int) -> Path | None:
    d = (ROOT / "saves" / "unlearn" / "llama1b_AltPO"
         / f"lr2e-05_ep10_alpha2_beta0.05_gamma1_seed{seed}")
    return d if (d / "evals" / "PAPER_AGGREGATES.json").is_file() else None


def _load_aggregates(run_dir: Path) -> dict[str, float]:
    payload = json.loads(
        (run_dir / "evals" / "PAPER_AGGREGATES.json").read_text(encoding="utf-8")
    )
    return payload["aggregates"]


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t-test (two-sided). Falls back to a manual computation if scipy
    is unavailable."""
    try:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except ImportError:
        ma, mb = statistics.mean(a), statistics.mean(b)
        va, vb = statistics.variance(a), statistics.variance(b)
        na, nb = len(a), len(b)
        se = math.sqrt(va / na + vb / nb)
        if se == 0:
            return float("inf") if ma != mb else 0.0, 0.0 if ma != mb else 1.0
        t = (ma - mb) / se
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        )
        # Normal approximation to the t CDF (df >= ~4 makes this close enough
        # for a fallback path; scipy is the primary path).
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return t, p


def main() -> int:
    rows: dict[str, dict[int, dict[str, float] | None]] = {"ERASE": {}, "AltPO": {}}
    dirs: dict[str, dict[int, Path | None]] = {"ERASE": {}, "AltPO": {}}
    for seed in SEEDS:
        for method, finder in (("ERASE", _find_erase_dir), ("AltPO", _find_altpo_dir)):
            d = finder(seed)
            dirs[method][seed] = d
            rows[method][seed] = _load_aggregates(d) if d else None

    lines: list[str] = []
    lines.append("# R2 — Seed Variance: ERASE vs AltPO (TOFU forget10, Llama-3.2-1B)")
    lines.append("")
    lines.append(f"Generated {date.today().isoformat()} by "
                 "`scripts/analysis/summarize_r2_seed_variance.py`.")
    lines.append("")
    lines.append("**Protocol.** Full re-unlearning (not re-evaluation): TOFU eval is "
                 "deterministic under greedy decoding, so the variance a reviewer asks "
                 "about is training stochasticity. Both methods were re-run 5 times "
                 "(seeds 0–4) from their exact winner hyperparameters, each followed by "
                 "the full paper eval + aggregation. ERASE: lr 0.025, α 4.0, layers 0–5, "
                 "topk 10, τ 2.0 (MSE), dynstop 2.0/U4.0, 100 ep. AltPO: lr 2e-5, "
                 "β 0.05, α 2, γ 1, 10 ep, DPO trainer, alt5_seed_0.json alternates "
                 "(held fixed to isolate training-seed variance; the released winner "
                 "checkpoint used this same file).")
    lines.append("")

    # Per-seed tables.
    for method in ("ERASE", "AltPO"):
        lines.append(f"## {method} per-seed results")
        lines.append("")
        lines.append("| seed | memorization | privacy | utility | aggregate | run dir |")
        lines.append("|---|---|---|---|---|---|")
        for seed in SEEDS:
            r = rows[method][seed]
            d = dirs[method][seed]
            if r is None:
                lines.append(f"| {seed} | — | — | — | — | (not finished) |")
            else:
                rel = d.relative_to(ROOT).as_posix()
                lines.append(
                    f"| {seed} | {r['memorization']:.4f} | {r['privacy']:.4f} | "
                    f"{r['utility']:.4f} | {r['aggregate']:.4f} | `{rel}` |"
                )
        lines.append("")

    # Mean +/- std + Welch t-tests where both sides have >= 2 seeds.
    lines.append("## Summary (mean ± std) and Welch t-tests")
    lines.append("")
    lines.append("| pillar | ERASE (mean ± std, n) | AltPO (mean ± std, n) | Welch t | p (two-sided) |")
    lines.append("|---|---|---|---|---|")
    for pillar in PILLARS:
        vals = {
            m: [rows[m][s][pillar] for s in SEEDS if rows[m][s] is not None]
            for m in ("ERASE", "AltPO")
        }
        cells = {}
        for m in ("ERASE", "AltPO"):
            v = vals[m]
            if len(v) == 0:
                cells[m] = "—"
            elif len(v) == 1:
                cells[m] = f"{v[0]:.4f} (n=1)"
            else:
                cells[m] = f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f} (n={len(v)})"
        if len(vals["ERASE"]) >= 2 and len(vals["AltPO"]) >= 2:
            t, p = _welch_t(vals["ERASE"], vals["AltPO"])
            t_cell, p_cell = f"{t:.3f}", f"{p:.4f}"
        else:
            t_cell = p_cell = "—"
        lines.append(f"| {pillar} | {cells['ERASE']} | {cells['AltPO']} | {t_cell} | {p_cell} |")
    lines.append("")

    # Reproduction check.
    lines.append("## Reproduction check (seed 0 vs paper point estimates)")
    lines.append("")
    e0, a0 = rows["ERASE"][0], rows["AltPO"][0]
    if e0:
        lines.append(f"- ERASE seed-0 aggregate: **{e0['aggregate']:.4f}** "
                     f"(paper: {PAPER_ERASE_AGG}).")
    if a0:
        lines.append(f"- AltPO seed-0 aggregate: **{a0['aggregate']:.4f}** "
                     f"(released HF checkpoint `AltPO_lr2e-05_beta0.05_alpha2_epoch10`: "
                     f"{PAPER_ALTPO_AGG}). Exact-match is not expected — the released "
                     "checkpoint was trained on different hardware/library versions — "
                     "but it should land within the seed spread.")
    if not (e0 or a0):
        lines.append("- (no seed-0 runs finished yet)")
    lines.append("")

    n_done = sum(1 for m in rows for s in SEEDS if rows[m][s] is not None)
    lines.append(f"_Runs summarized: {n_done}/10._")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD} ({n_done}/10 runs summarized)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
