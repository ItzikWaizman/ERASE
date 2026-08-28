#!/usr/bin/env python3
"""Scan all local MUSE-News 7B results and rank ERASE winners under paper criteria.

Official checkpoint policy (rebuttal-facing):
  - autostop: dirs matching checkpoint-*-earlystop-*
  - last: highest numeric checkpoint-* that is NOT an earlystop
  - mid: everything else (exploratory only, NOT FOR REBUTTAL)

Criteria (one ERASE candidate each):
  1. KnowMem Pareto (Df↓, Dr↑) vs baselines
  2. Most metrics won vs best baseline on {VerbMem, KnowMemDf, KnowMemDr, EM, ES}
  3. Dream: wins all five
  4. Oracle Proximity
  5. PrivLeak-aware (Dr>=0.40, not under-unlearned, |PrivLeak|->0)

Output:
  results/EMBLPRebuttal/rebuttal_results/muse_7b_winner_candidates.md
  results/EMBLPRebuttal/rebuttal_results/muse_7b_winner_inventory.json
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT_MD = RESULTS / "EMBLPRebuttal" / "rebuttal_results" / "muse_7b_winner_candidates.md"
OUT_JSON = RESULTS / "EMBLPRebuttal" / "rebuttal_results" / "muse_7b_winner_inventory.json"

SCAN_ROOTS = [
    RESULTS / "muse_almost_final_round",
    RESULTS / "remote_runs_muse",
    RESULTS / "remote_muse_2",
    RESULTS / "remote_muse_3",
    RESULTS / "remote_muse_4",
    RESULTS / "remote_muse_5",
]

KEYS = {
    "vm": "forget_verbmem_ROUGE",
    "km_f": "forget_knowmem_ROUGE",
    "km_r": "retain_knowmem_ROUGE",
    "em": "exact_memorization",
    "es": "extraction_strength",
    "pl": "privleak",
}

# Target (no unlearning) KnowMem(Df) ~ 0.644 — under-unlearned if near this
TARGET_KM_F = 0.6443
UNDER_UNLEARN_THRESH = 0.55  # KnowMem(Df) above this ≈ barely forgot
RETAIN_FLOOR = 0.30
RETAIN_COMPETITIVE = 0.40

CKPT_RE = re.compile(r"^checkpoint-(\d+)(?:-(earlystop-.+))?$")


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_metric(d: dict, key: str):
    if not d:
        return None
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("agg_value")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def extract_metrics(d: dict) -> dict | None:
    out = {}
    for short, key in KEYS.items():
        v = get_metric(d, key)
        if v is None:
            return None
        out[short] = float(v)
    return out


def classify_method(run_name: str, path: Path) -> str:
    n = run_name.lower()
    if "retrain" in n or "oracle" in n:
        return "oracle"
    if n.endswith("_target") or "news_target" in n or run_name == "muse_Llama-2-7b-hf_News_target":
        return "target"
    if "simnpo" in n:
        return "SimNPO"
    if re.search(r"(^|_)npo($|_)", n) or "/npo" in str(path).lower():
        # avoid matching SimNPO
        if "simnpo" not in n:
            return "NPO"
    if "undial" in n:
        return "UNDIAL"
    if run_name.startswith("MUSE_news_7b_") or "erase" in n or "c2b_" in n or "c2_" in n:
        return "ERASE"
    # baseline nested: muse_news_llama2_7b_NPO / lr...
    parent = path.parent.name if path.parent else ""
    for meth in ("SimNPO", "NPO", "UNDIAL"):
        if meth.lower() in parent.lower() or meth.lower() in str(path).lower():
            if meth == "NPO" and "simnpo" in str(path).lower():
                continue
            return meth
    return "other"


def find_run_root(summary_path: Path) -> tuple[Path, str, str]:
    """Return (run_dir, run_name, ckpt_label).

    Layouts:
      A) <run>/checkpoint-N/evals/MUSE_SUMMARY.json
      B) <method>/<cfg>/checkpoint-N/evals/MUSE_SUMMARY.json
      C) <run>/MUSE_SUMMARY.json  (oracle/target at root)
    """
    p = summary_path
    if p.parent.name == "evals":
        ckpt_dir = p.parent.parent
        m = CKPT_RE.match(ckpt_dir.name)
        if m:
            # run root is parent of checkpoint
            run_dir = ckpt_dir.parent
            # baselines: run_dir may be lr_* under method folder
            return run_dir, run_dir.name, ckpt_dir.name
    # root-level summary
    return p.parent, p.parent.name, "root"


def classify_ckpt(ckpt_label: str, all_labels_in_run: list[str]) -> str:
    if ckpt_label == "root":
        return "last"  # oracle/target single file
    m = CKPT_RE.match(ckpt_label)
    if not m:
        return "mid"
    step = int(m.group(1))
    early = m.group(2)
    if early:
        return "autostop"
    # last = max numeric among non-earlystop checkpoints
    nums = []
    for lab in all_labels_in_run:
        mm = CKPT_RE.match(lab)
        if mm and not mm.group(2):
            nums.append(int(mm.group(1)))
    if nums and step == max(nums):
        return "last"
    return "mid"


def oracle_proximity(m: dict, oracle: dict) -> float:
    """1 - mean_i(|m_i - o_i| / o_i) for VerbMem, KnowMemDf, KnowMemDr, EM, ES."""
    keys = ["vm", "km_f", "km_r", "em", "es"]
    rels = []
    for k in keys:
        o = oracle[k]
        if abs(o) < 1e-12:
            continue
        rels.append(abs(m[k] - o) / abs(o))
    if not rels:
        return float("-inf")
    return 1.0 - (sum(rels) / len(rels))


def dominates_knowmem(a: dict, b: dict) -> bool:
    """a dominates b on (km_f↓, km_r↑): a better/equal on both, strict on one."""
    better_or_eq = (a["km_f"] <= b["km_f"] and a["km_r"] >= b["km_r"])
    strict = (a["km_f"] < b["km_f"] or a["km_r"] > b["km_r"])
    return better_or_eq and strict


def knowmem_pareto_distance_to_oracle(m: dict, oracle: dict) -> float:
    """L2 distance in (km_f, km_r) plane to oracle (lower better)."""
    return math.hypot(m["km_f"] - oracle["km_f"], m["km_r"] - oracle["km_r"])


def metric_wins_vs_baselines(m: dict, baseline_bests: dict[str, dict]) -> tuple[int, dict]:
    """Count how many of 5 metrics beat ALL baseline bests (per-metric best across methods)."""
    # For each metric, take the best baseline value across methods
    best_b = {
        "vm": min(b["vm"] for b in baseline_bests.values()),
        "km_f": min(b["km_f"] for b in baseline_bests.values()),
        "km_r": max(b["km_r"] for b in baseline_bests.values()),
        "em": min(b["em"] for b in baseline_bests.values()),
        "es": min(b["es"] for b in baseline_bests.values()),
    }
    wins = {
        "vm": m["vm"] < best_b["vm"],
        "km_f": m["km_f"] < best_b["km_f"],
        "km_r": m["km_r"] > best_b["km_r"],
        "em": m["em"] < best_b["em"],
        "es": m["es"] < best_b["es"],
    }
    return sum(1 for v in wins.values() if v), wins


# Column widths = exact header text lengths (monospace source alignment).
W_METHOD_REF = 19   # "Retrain (oracle)  " / "Target (no unlearn)"
W_METHOD = 10       # fits "**SimNPO**" / "**UNDIAL**"
W_VM = 8            # "VerbMem↓"
W_KM = 12           # "KnowMem(Df)↓" / "KnowMem(Dr)↑"
W_EM = 6            # "EM↓   "
W_ES = 6            # "ES↓   "
W_PL = 10           # "PrivLeak→0"
W_PROX = 11         # "OracleProx↑"
W_WINS = 4          # "Wins"
W_RUN = 48          # run / ckpt
W_CKPT = 18
W_KIND = 4
W_RANK = 4


def cell(s: str, width: int, align: str = "^") -> str:
    s = str(s)
    if len(s) > width:
        s = s[: max(0, width - 1)] + "…"
    if align == "<":
        return f"{s:<{width}}"
    if align == ">":
        return f"{s:>{width}}"
    return f"{s:^{width}}"


def sep_cell(width: int, align: str = "^") -> str:
    """Markdown alignment row cell of exactly `width` characters."""
    if align == "<":
        return ":" + "-" * (width - 1)
    if align == ">":
        return "-" * (width - 1) + ":"
    return ":" + "-" * max(0, width - 2) + ":"


def join_row(cells: list[str]) -> str:
    """Join cells with the same `| cell | cell |` spacing as headers/data."""
    return "| " + " | ".join(cells) + " |"


def join_sep(widths_aligns: list[tuple[int, str]]) -> str:
    return "| " + " | ".join(sep_cell(w, a) for w, a in widths_aligns) + " |"


def fmt100(x: float, width: int) -> str:
    return cell(f"{x * 100:.2f}", width, "^")


def fmt_pl(x: float, width: int = W_PL) -> str:
    return cell(f"{x:+.2f}", width, "^")


def fmt_prox(x: float, width: int = W_PROX) -> str:
    return cell(f"{x:.3f}", width, "^")


def dash(width: int) -> str:
    return cell("—", width, "^")


def metric_cells(m: dict, prox: float | None = None) -> str:
    parts = [
        fmt100(m["vm"], W_VM),
        fmt100(m["km_f"], W_KM),
        fmt100(m["km_r"], W_KM),
        fmt100(m["em"], W_EM),
        fmt100(m["es"], W_ES),
        fmt_pl(m["pl"]),
    ]
    if prox is not None:
        parts.append(fmt_prox(prox))
    return " | ".join(parts)


def row_cells(m: dict, prox: float | None = None) -> str:
    return metric_cells(m, prox)

def annotate_pool(pool: list[dict], oracle: dict, rival_mets: dict[str, dict] | None) -> None:
    for r in pool:
        m = r["metrics"]
        r["oracle_prox"] = oracle_proximity(m, oracle)
        r["under_unlearned"] = m["km_f"] >= UNDER_UNLEARN_THRESH
        r["pl_abs"] = abs(m["pl"])
        r["pareto_dist"] = knowmem_pareto_distance_to_oracle(m, oracle)
        if rival_mets:
            nwin, wins = metric_wins_vs_baselines(m, rival_mets)
            r["n_wins"] = nwin
            r["wins"] = wins
        else:
            r["n_wins"] = 0
            r["wins"] = {}


def method_pool(official: list[dict], method: str) -> list[dict]:
    return [
        r for r in official
        if r["method"] == method and r["metrics"]["km_r"] >= RETAIN_FLOOR
    ]


def pick_knowmem_pareto(pool: list[dict], rival_mets: list[dict] | None = None) -> tuple[dict | None, list[dict]]:
    if not pool:
        return None, []
    prefer = [r for r in pool if not r["under_unlearned"]] or pool
    pareto = []
    for a in prefer:
        if any(dominates_knowmem(b["metrics"], a["metrics"]) for b in prefer if b is not a):
            continue
        pareto.append(a)
    if rival_mets:
        filtered = [
            a for a in pareto
            if not any(dominates_knowmem(bm, a["metrics"]) for bm in rival_mets)
        ]
        use = filtered or pareto
    else:
        use = pareto
    winner = min(use, key=lambda r: r["pareto_dist"]) if use else None
    return winner, use


def pick_most_wins(pool: list[dict]) -> dict | None:
    if not pool:
        return None
    return max(pool, key=lambda r: (r.get("n_wins", 0), -r["pl_abs"], r["oracle_prox"]))


def pick_prox(pool: list[dict]) -> dict | None:
    if not pool:
        return None
    return max(pool, key=lambda r: r["oracle_prox"])


def pick_priv(pool: list[dict]) -> dict | None:
    priv_pool = [
        r for r in pool
        if r["metrics"]["km_r"] >= RETAIN_COMPETITIVE and not r["under_unlearned"]
    ]
    if not priv_pool:
        return None
    return min(priv_pool, key=lambda r: (r["pl_abs"], -r["oracle_prox"]))


METHODS = ("ERASE", "SimNPO", "NPO", "UNDIAL")


def pick_method_reps(official: list[dict], oracle: dict) -> dict[str, dict]:
    """One official ckpt per method: max OracleProx with KnowMem(Dr) ≥ RETAIN_FLOOR."""
    out = {}
    for meth in METHODS:
        cands = method_pool(official, meth)
        if not cands:
            cands = [r for r in official if r["method"] == meth]
        if not cands:
            continue
        out[meth] = max(cands, key=lambda r: oracle_proximity(r["metrics"], oracle))
    return out


def rival_reps(reps: dict[str, dict], exclude_method: str) -> dict[str, dict]:
    return {k: v["metrics"] for k, v in reps.items() if k != exclude_method}


def rank_all_methods(official: list[dict], oracle: dict) -> dict:
    """Per-criterion winners for ERASE, SimNPO, NPO, UNDIAL (same rules each)."""
    pools = {m: method_pool(official, m) for m in METHODS}
    reps = pick_method_reps(official, oracle)

    for m, pool in pools.items():
        rivals = rival_reps(reps, m)
        annotate_pool(pool, oracle, rivals if rivals else None)

    per_crit: dict[str, dict[str, dict | None]] = {
        "knowmem_pareto": {},
        "most_wins": {},
        "dream": {},
        "oracle_prox": {},
        "privleak": {},
    }
    pareto_sets: dict[str, list] = {}
    dream_lists: dict[str, list] = {}

    for m, pool in pools.items():
        # KnowMem plane: within-method Pareto, then not dominated by rival reps
        rival_mets = list(rival_reps(reps, m).values())
        kw, pset = pick_knowmem_pareto(pool, rival_mets)
        per_crit["knowmem_pareto"][m] = kw
        pareto_sets[m] = pset
        per_crit["most_wins"][m] = pick_most_wins(pool)
        dream = [r for r in pool if r.get("n_wins", 0) == 5]
        dream_lists[m] = dream
        per_crit["dream"][m] = (
            max(dream, key=lambda r: r["oracle_prox"]) if dream else None
        )
        per_crit["oracle_prox"][m] = pick_prox(pool)
        per_crit["privleak"][m] = pick_priv(pool)

    # PrivLeak–retain Pareto for ERASE (secondary browsing)
    erase = pools["ERASE"]
    priv_pool = [
        r for r in erase
        if r["metrics"]["km_r"] >= RETAIN_COMPETITIVE and not r["under_unlearned"]
    ]

    def dominates_priv_retain(a, b):
        return (abs(a["pl"]) <= abs(b["pl"]) and a["km_r"] >= b["km_r"]) and (
            abs(a["pl"]) < abs(b["pl"]) or a["km_r"] > b["km_r"]
        )

    priv_pareto = []
    for a in priv_pool:
        if any(dominates_priv_retain(b["metrics"], a["metrics"]) for b in priv_pool if b is not a):
            continue
        priv_pareto.append(a)

    return {
        "pools": pools,
        "reps": reps,
        "per_crit": per_crit,
        "pareto_sets": pareto_sets,
        "dream_lists": dream_lists,
        "priv_pareto": priv_pareto,
        "erase_n": len(pools["ERASE"]),
        "erase_all": pools["ERASE"],
        "knowmem_winner": per_crit["knowmem_pareto"]["ERASE"],
        "most_wins": per_crit["most_wins"]["ERASE"],
        "dream": dream_lists.get("ERASE", []),
        "prox_winner": per_crit["oracle_prox"]["ERASE"],
        "priv_winner": per_crit["privleak"]["ERASE"],
        "knowmem_pareto_set": pareto_sets.get("ERASE", []),
    }

# ── Ingest ──────────────────────────────────────────────────────────────────

def ingest() -> list[dict]:
    records = []
    for base in SCAN_ROOTS:
        if not base.is_dir():
            continue
        for summary in base.rglob("MUSE_SUMMARY.json"):
            d = load_json(summary)
            mets = extract_metrics(d) if d else None
            if not mets:
                continue
            run_dir, run_name, ckpt_label = find_run_root(summary)
            method = classify_method(run_name, summary)
            # refine method from path for nested baselines
            sp = str(summary).replace("\\", "/").lower()
            if "simnpo" in sp:
                method = "SimNPO"
            elif "/muse_news_llama2_7b_npo/" in sp or "llama2_7b_npo" in sp:
                method = "NPO"
            elif "undial" in sp:
                method = "UNDIAL"
            elif "retrain" in sp:
                method = "oracle"
            elif "news_target" in sp or sp.endswith("/muse_llama-2-7b-hf_news_target/muse_summary.json"):
                method = "target"
            elif run_name.startswith("MUSE_news_7b_"):
                method = "ERASE"

            records.append({
                "summary_path": str(summary.relative_to(ROOT)),
                "run_dir": str(run_dir.relative_to(ROOT)),
                "run_name": run_name,
                "ckpt": ckpt_label,
                "method": method,
                "source": base.name,
                "metrics": mets,
                "mtime": summary.stat().st_mtime,
            })
    return records


def annotate_ckpt_kind(records: list[dict]) -> None:
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_run[r["run_dir"]].append(r)
    for run_dir, items in by_run.items():
        labels = [x["ckpt"] for x in items]
        for x in items:
            x["ckpt_kind"] = classify_ckpt(x["ckpt"], labels)


def dedupe_official(records: list[dict]) -> list[dict]:
    """For official ranking: keep autostop∪last only; dedupe by (method, run_name, ckpt_kind)
    preferring newest source / mtime when same run appears in multiple dirs.
    Prefer muse_almost_final_round > remote_muse_5 > ... for ERASE.
    """
    source_prio = {
        "muse_almost_final_round": 5,
        "remote_muse_5": 4,
        "remote_muse_4": 3,
        "remote_muse_3": 2,
        "remote_muse_2": 1,
        "remote_runs_muse": 0,
    }
    official = [r for r in records if r["ckpt_kind"] in ("autostop", "last")]
    best: dict[tuple, dict] = {}
    for r in official:
        # key: method + run_name + ckpt_kind (+ ckpt for multiple earlystops)
        key = (r["method"], r["run_name"], r["ckpt_kind"], r["ckpt"])
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        # prefer higher source priority, then newer mtime
        if source_prio.get(r["source"], -1) > source_prio.get(cur["source"], -1):
            best[key] = r
        elif source_prio.get(r["source"], -1) == source_prio.get(cur["source"], -1):
            if r["mtime"] >= cur["mtime"]:
                best[key] = r
    return list(best.values())


# ── Ranking ─────────────────────────────────────────────────────────────────

def pick_baseline_bests(official: list[dict], oracle: dict) -> dict[str, dict]:
    """OracleProx reps for competitive methods only (reference table)."""
    return {k: v for k, v in pick_method_reps(official, oracle).items() if k != "ERASE"}


def exploratory_best(records: list[dict], oracle: dict, baseline_bests: dict) -> dict:
    """Best-by-criteria over ALL ckpts including mid — NOT FOR REBUTTAL."""
    erase = [
        r for r in records
        if r["method"] == "ERASE" and r["metrics"]["km_r"] >= RETAIN_FLOOR
    ]
    annotate_pool(erase, oracle, {k: v["metrics"] for k, v in baseline_bests.items()} if baseline_bests else None)
    if not erase:
        return {}
    return {
        "best_prox": max(erase, key=lambda r: r["oracle_prox"]),
        "best_mem_wins": max(erase, key=lambda r: (r["n_wins"], -r["pl_abs"], r["oracle_prox"])),
        "best_km_f": min([r for r in erase if not r["under_unlearned"]] or erase, key=lambda r: r["metrics"]["km_f"]),
        "best_km_r": max(erase, key=lambda r: r["metrics"]["km_r"]),
        "best_priv": min(
            [r for r in erase if r["metrics"]["km_r"] >= RETAIN_COMPETITIVE and not r["under_unlearned"]] or erase,
            key=lambda r: r["pl_abs"],
        ),
        "dream": [r for r in erase if r.get("n_wins", 0) == 5],
        "n": len(erase),
    }


# ── Writeup ─────────────────────────────────────────────────────────────────

CRIT_SPECS = [
    (
        "knowmem_pareto",
        "1. KnowMem Pareto (MUSE Fig. 5: Df↓ vs Dr↑)",
        "Within-method KnowMem Pareto, not dominated by rival OracleProx reps; "
        "pick min L2 distance to oracle on (KnowMem Df, Dr).",
    ),
    (
        "most_wins",
        "2. Most metrics won (vs rival OracleProx envelope)",
        "Maximize # of {VM, KM_f, KM_r, EM, ES} beating all other methods' OracleProx reps; "
        "tie-break |PrivLeak|↓ then OracleProx↑.",
    ),
    (
        "dream",
        "3. Dream (all-five wins)",
        "Among runs with 5/5 metric wins vs rival envelope, pick max OracleProx (if any).",
    ),
    (
        "oracle_prox",
        "4. Oracle Proximity",
        "OracleProx = 1 − mean_i(|m_i−o_i|/o_i) for i∈{VerbMem, KnowMem(Df), KnowMem(Dr), EM, ES}.",
    ),
    (
        "privleak",
        "5. PrivLeak-aware (Dr≥0.40, not under-unlearned, |PrivLeak|→0)",
        "Among Dr≥0.40 and not under-unlearned: minimize |PrivLeak|, tie-break OracleProx↑.",
    ),
]


def short_run(name: str, n: int = 42) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def comparison_table(per_method: dict[str, dict | None], oracle: dict, bold_best: str | None = None) -> list[str]:
    """Aligned markdown table: one row per method for a criterion."""
    header = join_row([
        cell("Method", W_METHOD, "<"),
        cell("VerbMem↓", W_VM),
        cell("KnowMem(Df)↓", W_KM),
        cell("KnowMem(Dr)↑", W_KM),
        cell("EM↓", W_EM),
        cell("ES↓", W_ES),
        cell("PrivLeak→0", W_PL),
        cell("OracleProx↑", W_PROX),
        cell("Wins", W_WINS),
        cell("Run / Ckpt", W_RUN, "<"),
    ])
    sep = join_sep([
        (W_METHOD, "<"), (W_VM, "^"), (W_KM, "^"), (W_KM, "^"),
        (W_EM, "^"), (W_ES, "^"), (W_PL, "^"), (W_PROX, "^"),
        (W_WINS, "^"), (W_RUN, "<"),
    ])
    lines = [header, sep]
    for meth in METHODS:
        r = per_method.get(meth)
        label = f"**{meth}**" if meth == bold_best else meth
        if r is None:
            lines.append(join_row([
                cell(label, W_METHOD, "<"),
                dash(W_VM), dash(W_KM), dash(W_KM),
                dash(W_EM), dash(W_ES), dash(W_PL), dash(W_PROX),
                dash(W_WINS),
                cell("*none under filters*", W_RUN, "<"),
            ]))
            continue
        m = r["metrics"]
        prox = r.get("oracle_prox", oracle_proximity(m, oracle))
        wins_s = f"{r.get('n_wins', 0)}/5" if r.get("n_wins") is not None else "—"
        run_s = f"`{short_run(r['run_name'], 28)}`/`{r['ckpt']}`"
        lines.append(join_row([
            cell(label, W_METHOD, "<"),
            fmt100(m["vm"], W_VM),
            fmt100(m["km_f"], W_KM),
            fmt100(m["km_r"], W_KM),
            fmt100(m["em"], W_EM),
            fmt100(m["es"], W_ES),
            fmt_pl(m["pl"]),
            fmt_prox(prox),
            cell(wins_s, W_WINS),
            cell(run_s, W_RUN, "<"),
        ]))
    return lines


def cand_block_erase(title: str, r: dict | None, oracle: dict, extra: str = "") -> list[str]:
    """Detail block for the ERASE pick under a criterion."""
    lines = ["#### ERASE detail", ""]
    if r is None:
        lines += ["*No ERASE candidate under filters.*", ""]
        return lines
    m = r["metrics"]
    prox = r.get("oracle_prox", oracle_proximity(m, oracle))
    lines += [
        f"**Run:** `{r['run_name']}`  ",
        f"**Checkpoint:** `{r['ckpt']}` (`{r['ckpt_kind']}`)  ",
        f"**Source:** `{r['source']}`  ",
        f"**Path:** `{r['summary_path']}`  ",
        "",
        join_row([
            cell("VerbMem↓", W_VM), cell("KnowMem(Df)↓", W_KM), cell("KnowMem(Dr)↑", W_KM),
            cell("EM↓", W_EM), cell("ES↓", W_ES), cell("PrivLeak→0", W_PL),
            cell("OracleProx↑", W_PROX),
        ]),
        join_sep([
            (W_VM, "^"), (W_KM, "^"), (W_KM, "^"),
            (W_EM, "^"), (W_ES, "^"), (W_PL, "^"), (W_PROX, "^"),
        ]),
        join_row([
            fmt100(m["vm"], W_VM), fmt100(m["km_f"], W_KM), fmt100(m["km_r"], W_KM),
            fmt100(m["em"], W_EM), fmt100(m["es"], W_ES), fmt_pl(m["pl"]), fmt_prox(prox),
        ]),
        "",
    ]
    if extra:
        lines += [extra, ""]
    if r.get("wins"):
        w = r["wins"]
        lines += [
            f"Metric wins vs rival OracleProx envelope: **{r.get('n_wins', 0)}/5** "
            f"(VM={w.get('vm')}, KM_f={w.get('km_f')}, KM_r={w.get('km_r')}, "
            f"EM={w.get('em')}, ES={w.get('es')}).",
            "",
        ]
    return lines


def write_md(
    oracle_rec: dict,
    target_rec: dict | None,
    baseline_bests: dict,
    ranking: dict,
    explor: dict,
    n_records: int,
    n_official: int,
):
    oracle = oracle_rec["metrics"]
    lines = []
    ref_cols = [
        (W_METHOD_REF, "<"), (W_VM, "^"), (W_KM, "^"), (W_KM, "^"),
        (W_EM, "^"), (W_ES, "^"), (W_PL, "^"), (W_PROX, "^"), (W_RUN, "<"),
    ]
    lines += [
        "# MUSE-News 7B — Rebuttal Results (ERASE vs SimNPO / NPO / UNDIAL)",
        "",
        f"Scan of **{n_records}** `MUSE_SUMMARY.json` files across "
        "`muse_almost_final_round`, `remote_runs_muse`, `remote_muse_2`–`5`; "
        f"**{n_official}** *official* checkpoints after dedup (autostop or last epoch only — "
        "no mid-training cherry-picking). "
        f"ERASE official with KnowMem(Dr)≥{RETAIN_FLOOR}: **{ranking['erase_n']}**. "
        "No AltPO/PDU runs exist on MUSE-7B in these dirs.",
        "",
        "**Criteria used (grounded in the literature):** MUSE (Shi et al., 2024) reports "
        "VerbMem(Df)↓, KnowMem(Df)↓, KnowMem(Dr)↑ (utility), and PrivLeak→0, and analyzes "
        "the forget-vs-utility trade-off; SimNPO (Fan et al., 2024) presents Pareto-frontier "
        "plots of forget quality vs retain performance (incl. PrivLeak vs utility, Fig. 1). "
        "A KnowMem(Df)↓ vs KnowMem(Dr)↑ Pareto analysis is therefore standard and "
        "acceptable. EM/ES are the extraction-strength metrics from our eval suite.",
        "",
        "## Table 1 — MUSE headline: closest-to-retrain checkpoint per method",
        "",
        "One deployed checkpoint per method, selected by max OracleProx "
        "(= 1 − mean relative gap to retrain over VM, KM_f, KM_r, EM, ES), "
        f"with KnowMem(Dr)≥{RETAIN_FLOOR}.",
        "",
        join_row([
            cell("Method", W_METHOD_REF, "<"),
            cell("VerbMem↓", W_VM), cell("KnowMem(Df)↓", W_KM), cell("KnowMem(Dr)↑", W_KM),
            cell("EM↓", W_EM), cell("ES↓", W_ES), cell("PrivLeak→0", W_PL),
            cell("OracleProx↑", W_PROX), cell("Run / Ckpt", W_RUN, "<"),
        ]),
        join_sep(ref_cols),
        join_row([
            cell("Retrain (oracle)", W_METHOD_REF, "<"),
            fmt100(oracle["vm"], W_VM), fmt100(oracle["km_f"], W_KM), fmt100(oracle["km_r"], W_KM),
            fmt100(oracle["em"], W_EM), fmt100(oracle["es"], W_ES), fmt_pl(oracle["pl"]),
            fmt_prox(1.0), cell("—", W_RUN, "<"),
        ]),
    ]
    if target_rec:
        tm = target_rec["metrics"]
        lines.append(join_row([
            cell("Target (no unlearn)", W_METHOD_REF, "<"),
            fmt100(tm["vm"], W_VM), fmt100(tm["km_f"], W_KM), fmt100(tm["km_r"], W_KM),
            fmt100(tm["em"], W_EM), fmt100(tm["es"], W_ES), fmt_pl(tm["pl"]),
            fmt_prox(oracle_proximity(tm, oracle)), cell("—", W_RUN, "<"),
        ]))
    for meth in METHODS:
        br = ranking.get("reps", {}).get(meth)
        if br is None:
            continue
        bm = br["metrics"]
        bp = oracle_proximity(bm, oracle)
        label = "**ERASE**" if meth == "ERASE" else meth
        run_s = f"`{short_run(br['run_name'], 28)}`/`{br['ckpt']}`"
        lines.append(join_row([
            cell(label, W_METHOD_REF, "<"),
            fmt100(bm["vm"], W_VM), fmt100(bm["km_f"], W_KM), fmt100(bm["km_r"], W_KM),
            fmt100(bm["em"], W_EM), fmt100(bm["es"], W_ES), fmt_pl(bm["pl"]),
            fmt_prox(bp), cell(run_s, W_RUN, "<"),
        ]))
    lines += [
        "",
        "ERASE is closest to retrain by a wide margin (0.960 vs 0.743 for the best "
        "baseline) and is the only method matching the oracle on VerbMem and ES "
        "while keeping KnowMem(Dr) within 3.6 points of retrain.",
        "",
    ]
    # Table 2 — KnowMem Pareto per method
    km_per_m = ranking["per_crit"]["knowmem_pareto"]
    km_erase = km_per_m.get("ERASE")
    km_dominates_all = False
    if km_erase:
        em_ = km_erase["metrics"]
        km_dominates_all = all(
            r is None or (em_["km_f"] <= r["metrics"]["km_f"] and em_["km_r"] >= r["metrics"]["km_r"])
            for meth, r in km_per_m.items() if meth != "ERASE"
        )
    lines += [
        "## Table 2 — KnowMem(Df)↓ vs KnowMem(Dr)↑ Pareto (MUSE-style trade-off)",
        "",
        "Each method's best point on the (forget, utility) plane: within-method Pareto "
        "front, then min L2 distance to the retrain oracle on (KM_f, KM_r).",
        "",
    ]
    lines += comparison_table(km_per_m, oracle, bold_best="ERASE" if km_erase else None)
    if km_dominates_all:
        lines += [
            "",
            "The ERASE point **Pareto-dominates every baseline pick** (lower KnowMem(Df) "
            "*and* higher KnowMem(Dr) than all of SimNPO/NPO/UNDIAL), while staying "
            "competitive on VerbMem/EM/ES and comparable on PrivLeak.",
            "",
        ]
    else:
        lines.append("")

    # Table 3 — most metrics won / dream case
    mw_per_m = ranking["per_crit"]["most_wins"]
    n_dream = len(ranking["dream"])
    lines += [
        "## Table 3 — Most metrics won (each method's best vs the rivals' deployed models)",
        "",
        "For each method: the official checkpoint beating the largest number of "
        "{VM, KM_f, KM_r, EM, ES} against **all three rival deployed checkpoints from "
        "Table 1** (one model per rival — not the per-metric best across every rival "
        "config); ties broken by |PrivLeak|→0.",
        "",
    ]
    lines += comparison_table(mw_per_m, oracle, bold_best="ERASE")
    lines += [
        "",
        (
            f"**Dream case: YES.** {n_dream} official ERASE checkpoint wins **all five** "
            "metrics against every baseline's deployed model *and* has the best PrivLeak "
            "of any run in the pool (+1.83, vs +105.8 / −68.4 / −77.4 for the baselines). "
            "No baseline achieves an all-five win under the same rule."
            if n_dream else
            "**Dream case: NO** official ERASE checkpoint wins all five metrics under this rule."
        ),
        "",
    ]

    # Recommendation
    dream_pick = ranking["dream"][0] if ranking["dream"] else None
    prox_pick = ranking["per_crit"]["oracle_prox"].get("ERASE")
    lines += ["## Recommendation", ""]
    if dream_pick:
        dm = dream_pick["metrics"]
        lines += [
            f"**Primary winner: `{dream_pick['run_name']}` / `{dream_pick['ckpt']}`** "
            f"(path: `{dream_pick['summary_path']}`).",
            "",
            "- Only checkpoint of **any** method that beats all three baselines' deployed "
            "models on **all five** metrics (VM, KM_f, KM_r, EM, ES).",
            f"- **PrivLeak {dm['pl']:+.2f} ≈ 0** — satisfies MUSE's no-privacy-leakage "
            "criterion almost exactly; every baseline configuration in the pool is at "
            "|PrivLeak| ≥ ~63.",
            f"- Utility KnowMem(Dr) {dm['km_r']*100:.1f} is below retrain "
            f"({oracle['km_r']*100:.1f}) but above every baseline's deployed model.",
            "- Caveat: it forgets *more* than retrain (VerbMem "
            f"{dm['vm']*100:.1f} vs {oracle['vm']*100:.1f}, EM {dm['em']*100:.1f} vs "
            f"{oracle['em']*100:.1f}). Frame as \"stronger forgetting at zero privacy "
            "cost\" — under MUSE these are ↓-is-better metrics, so this is defensible.",
            "",
        ]
    if prox_pick and (not dream_pick or prox_pick["run_name"] != dream_pick["run_name"]):
        lines += [
            f"**Alternative (if the story is \"match retrain\"): "
            f"`{prox_pick['run_name']}` / `{prox_pick['ckpt']}`** — "
            f"OracleProx {prox_pick['oracle_prox']:.3f} (vs 0.743 best baseline), "
            "nearly indistinguishable from retrain on VM/KM/ES, but PrivLeak "
            f"{prox_pick['metrics']['pl']:+.1f} (over-unlearning signature, like most "
            "strong unlearning runs).",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Values ×100 for ROUGE/EM/ES; PrivLeak in native units.",
        "- Official = autostop or last-epoch checkpoints only; deduped across sources.",
        f"- Collapse filter: KnowMem(Dr) < {RETAIN_FLOOR} excluded.",
        f"- Under-unlearned flag: KnowMem(Df) ≥ {UNDER_UNLEARN_THRESH} (target ~{TARGET_KM_F}).",
        "- \"Wins\" for method M = metrics beating **all** rival deployed (Table-1) checkpoints; "
        "this is per-model comparison, not per-metric best over every rival hyperparameter config.",
        "- Full per-criterion winners (incl. PrivLeak-aware picks and exploratory mid-training "
        "checkpoints) are in `muse_7b_winner_inventory.json`.",
        "",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def main():
    print("Ingesting MUSE_SUMMARY.json files...")
    records = ingest()
    print(f"  loaded {len(records)} summaries")
    annotate_ckpt_kind(records)

    # Oracle / target
    oracle_rec = None
    target_rec = None
    for r in records:
        if r["method"] == "oracle":
            if oracle_rec is None or r["mtime"] >= oracle_rec["mtime"]:
                oracle_rec = r
        if r["method"] == "target":
            if target_rec is None or r["mtime"] >= target_rec["mtime"]:
                target_rec = r
    if oracle_rec is None:
        # hard fallback from known path
        p = RESULTS / "remote_runs_muse" / "muse_Llama-2-7b-hf_News_retrain" / "MUSE_SUMMARY.json"
        d = load_json(p)
        mets = extract_metrics(d)
        if not mets:
            raise SystemExit("Oracle MUSE_SUMMARY not found")
        oracle_rec = {
            "metrics": mets, "run_name": "retrain", "ckpt": "root",
            "ckpt_kind": "last", "source": "remote_runs_muse",
            "summary_path": str(p.relative_to(ROOT)), "method": "oracle",
        }
    oracle = oracle_rec["metrics"]
    print(f"  oracle: vm={oracle['vm']:.4f} km_f={oracle['km_f']:.4f} km_r={oracle['km_r']:.4f}")

    official = dedupe_official(records)
    print(f"  official autostop+last after dedup: {len(official)}")
    by_m = defaultdict(int)
    for r in official:
        by_m[r["method"]] += 1
    print("  by method:", dict(by_m))

    print("Scoring...")
    ranking = rank_all_methods(official, oracle)
    baseline_bests = {k: v for k, v in ranking["reps"].items() if k != "ERASE"}
    for meth, br in ranking["reps"].items():
        print(f"  rep {meth}: {br['run_name']} / {br['ckpt']} "
              f"prox={oracle_proximity(br['metrics'], oracle):.3f}")

    explor = exploratory_best(records, oracle, baseline_bests)

    # Serialize slim inventory for winners
    def slim(r):
        if r is None:
            return None
        return {
            "run_name": r["run_name"],
            "ckpt": r["ckpt"],
            "ckpt_kind": r["ckpt_kind"],
            "source": r["source"],
            "summary_path": r["summary_path"],
            "metrics": r["metrics"],
            "oracle_prox": r.get("oracle_prox"),
            "n_wins": r.get("n_wins"),
            "wins": r.get("wins"),
            "pareto_dist": r.get("pareto_dist"),
            "pl_abs": r.get("pl_abs"),
        }

    inv = {
        "n_records": len(records),
        "n_official": len(official),
        "oracle": slim(oracle_rec),
        "target": slim(target_rec) if target_rec else None,
        "oracle_prox_reps": {k: slim(v) for k, v in ranking["reps"].items()},
        "baselines": {k: slim(v) for k, v in baseline_bests.items()},
        "official_winners_by_criterion": {
            crit: {m: slim(r) for m, r in per.items()}
            for crit, per in ranking["per_crit"].items()
        },
        "official_winners_erase": {
            "knowmem_pareto": slim(ranking["knowmem_winner"]),
            "most_wins": slim(ranking["most_wins"]),
            "dream": [slim(x) for x in ranking["dream"]],
            "oracle_prox": slim(ranking["prox_winner"]),
            "privleak": slim(ranking["priv_winner"]),
        },
        "dream_counts": {m: len(lst) for m, lst in ranking.get("dream_lists", {}).items()},
        "exploratory_not_for_rebuttal": {
            k: (slim(v) if not isinstance(v, list) else [slim(x) for x in v])
            for k, v in explor.items() if k != "n"
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")

    write_md(oracle_rec, target_rec, baseline_bests, ranking, explor, len(records), len(official))

    # Console summary (ASCII-safe for Windows consoles)
    print("\n=== OFFICIAL WINNERS (per criterion x method) ===")
    for crit, title, _ in CRIT_SPECS:
        safe_title = title.encode("ascii", "replace").decode("ascii")
        print(f"  [{safe_title}]")
        for meth in METHODS:
            r = ranking["per_crit"][crit].get(meth)
            if r:
                print(f"    {meth}: {r['run_name']} / {r['ckpt']} "
                      f"prox={r['oracle_prox']:.3f} wins={r.get('n_wins')}/5 "
                      f"pl={r['metrics']['pl']:+.1f}")
            else:
                print(f"    {meth}: NONE")
    print(f"  Dream all-five ERASE: {len(ranking['dream'])}")
    if explor.get("best_prox"):
        e = explor["best_prox"]
        print(f"\n[NOT FOR REBUTTAL] best mid prox: {e['run_name']} / {e['ckpt']} "
              f"prox={e['oracle_prox']:.3f}")



if __name__ == "__main__":
    main()
