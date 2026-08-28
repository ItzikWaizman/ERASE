"""Evaluate every published OpenUnlearning checkpoint for NPO, GradDiff,
UNDIAL, AltPO (and optionally IdkNLL) on TOFU forget10 / Llama-3.2-1B-Instruct.

Same structure as eval_hf_simnpo_grid.py / eval_hf_rmu_grid.py: paper-grade
eval, incremental summary, skip-if-cached. Each method gets its own output
directory under saves/eval_grid_<method>/.

After all evals are done, the script also:
  * runs ``compute_paper_aggregates`` for every checkpoint folder (including
    eval_grid_simnpo / eval_grid_rmu which were populated by older scripts);
  * builds per-method top-3 highlights under ``saves/baseline_highlights/1B/``;
  * writes a ``MAIN_COMPARISON_1B.md`` summary table over the 6 baseline
    methods (SimNPO, NPO, RMU, GradDiff, UNDIAL, AltPO) plus the two ERASE
    champions, with columns Agg | Util | Priv | FQ.

Usage:
    python scripts/analysis/eval_hf_other_methods_grid.py
    python scripts/analysis/eval_hf_other_methods_grid.py --only npo
    python scripts/analysis/eval_hf_other_methods_grid.py --limit 3
    python scripts/analysis/eval_hf_other_methods_grid.py --force
    python scripts/analysis/eval_hf_other_methods_grid.py --skip-eval
    python scripts/analysis/eval_hf_other_methods_grid.py --only-aggregate
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "comparison_methods"))
sys.path.insert(0, str(ROOT / "src"))

import shutil

import _common  # type: ignore  # noqa: E402

PYTHON = _common.PYTHON
EVAL_SCRIPT = ROOT / "src" / "eval.py"
PAPER_EVAL_EXPERIMENT = "eval/tofu/llama1b/paper"

REQUIRED_METRIC_KEYS = (
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_PARA_Prob",
    "forget_truth_ratio_paper",
    "model_utility",
    "forget_Q_A_gibberish",
    "mia_loss",
    "mia_zlib",
    "mia_min_k",
    "mia_min_k_plus_plus",
)

# ---------------------------------------------------------------------------
# Checkpoint registries (pulled from HF API 2026-04-27)
# ---------------------------------------------------------------------------

_P = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"

HF_NPO: list[str] = [
    f"{_P}NPO_lr{lr}_beta{b}_alpha{a}_epoch{ep}"
    for lr in ("1e-05", "2e-05", "5e-05")
    for b in ("0.05", "0.1", "0.5")
    for a in ("1", "2", "5")
    for ep in ("5", "10")
]

HF_GRADDIFF: list[str] = [
    f"{_P}GradDiff_lr{lr}_alpha{a}_epoch{ep}"
    for lr in ("1e-05", "2e-05", "3e-05", "4e-05", "5e-05")
    for a in ("1", "2", "5", "10")
    for ep in ("5", "10")
]

HF_UNDIAL: list[str] = [
    f"{_P}UNDIAL_lr{lr}_beta{b}_alpha{a}_epoch{ep}"
    for lr in ("0.0001", "1e-05", "5e-05")
    for b in ("10", "3", "30")
    for a in ("1", "2", "5")
    for ep in ("5", "10")
]

HF_ALTPO: list[str] = [
    f"{_P}AltPO_lr{lr}_beta{b}_alpha{a}_epoch{ep}"
    for lr in ("1e-05", "2e-05", "5e-05")
    for b in ("0.05", "0.1", "0.5")
    for a in ("1", "2", "5")
    for ep in ("5", "10")
]

HF_IDKNLL: list[str] = [
    f"{_P}IdkNLL_lr{lr}_alpha{a}_epoch{ep}"
    for lr in ("1e-05", "2e-05", "3e-05", "4e-05", "5e-05")
    for a in ("1", "2", "5", "10")
    for ep in ("5", "10")
]

METHOD_GRIDS: dict[str, tuple[list[str], Path, str]] = {
    "npo":      (HF_NPO,      ROOT / "saves" / "eval_grid_npo",      "NPO"),
    "graddiff": (HF_GRADDIFF,  ROOT / "saves" / "eval_grid_graddiff", "GradDiff"),
    "undial":   (HF_UNDIAL,    ROOT / "saves" / "eval_grid_undial",   "UNDIAL"),
    "altpo":    (HF_ALTPO,     ROOT / "saves" / "eval_grid_altpo",    "AltPO"),
    "idknll":   (HF_IDKNLL,    ROOT / "saves" / "eval_grid_idknll",   "IdkNLL"),
}

# ---------------------------------------------------------------------------
# Aggregation / highlights config
# ---------------------------------------------------------------------------
# Methods to include in MAIN_COMPARISON_1B.md and baseline_highlights/1B/.
# Folders may have been populated by this script OR by older grid scripts
# (eval_grid_simnpo, eval_grid_rmu). Aggregation walks the folder either way.
AGGREGATE_METHODS: list[tuple[str, Path]] = [
    ("SimNPO",    ROOT / "saves" / "eval_grid_simnpo"),
    ("NPO",       ROOT / "saves" / "eval_grid_npo"),
    ("RMU",       ROOT / "saves" / "eval_grid_rmu"),
    ("GradDiff",  ROOT / "saves" / "eval_grid_graddiff"),
    ("UNDIAL",    ROOT / "saves" / "eval_grid_undial"),
    ("AltPO",     ROOT / "saves" / "eval_grid_altpo"),
]

# ERASE champions (rows added to MAIN_COMPARISON_1B.md alongside baselines).
ERASE_HIGHLIGHTS: list[tuple[str, Path]] = [
    (
        "ERASE (Anchored)",
        ROOT / "saves" / "highlights"
        / "ITER7_ALPHA_BOOST_13ep_wiki_lr0.04_b0_a2.5_k10_rw0.1_crw0.005_cfauth_flw0.9_cap20_ptcap3.0_cosine_L012345_authoronly_ammspan_ebs8_det",
    ),
    (
        "ERASE (Pure-α)",
        ROOT / "saves" / "highlights"
        / "L05_PTCAP_SWEEP_28ep_wiki_lr0.025_b0_a4.0_k10_cfauth_flw0.9_cap20_ptcap2.7_cosine_L012345_authoronly_ammspan_ebs8_det",
    ),
]

INIT_REF_DIR = ROOT / "saves" / "eval" / "init_finetuned"
RETAIN_REF_DIR = ROOT / "saves" / "eval" / "retain_oracle"
HIGHLIGHTS_OUT = ROOT / "saves" / "baseline_highlights" / "1B"
AGGREGATE_SCRIPT = ROOT / "scripts" / "analysis" / "compute_paper_aggregates.py"
HIGHLIGHT_FILES_TO_COPY = (
    "TOFU_SUMMARY.json",
    "PAPER_AGGREGATES.json",
    "PAPER_AGGREGATES.md",
)


def _short_name(model_id: str) -> str:
    m = re.match(
        r"open-unlearning/unlearn_tofu_Llama-3\.2-1B-Instruct_forget10_(?P<rest>.+)",
        model_id,
    )
    return m.group("rest") if m else model_id.replace("/", "__")


def clear_cuda_cache() -> None:
    snippet = (
        "import gc, torch; gc.collect();"
        " torch.cuda.empty_cache() if torch.cuda.is_available() else None;"
        " torch.cuda.ipc_collect() if torch.cuda.is_available() else None"
    )
    subprocess.run([PYTHON, "-c", snippet], cwd=str(ROOT), capture_output=True)


def _eval_complete(out_dir: Path) -> bool:
    sp = out_dir / "TOFU_SUMMARY.json"
    if not sp.is_file():
        return False
    try:
        d = json.loads(sp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(k in d for k in REQUIRED_METRIC_KEYS)


def _purge_hf_cache(model_id: str) -> None:
    """Remove a model's cached download from the HF hub cache to free disk."""
    from huggingface_hub import scan_cache_dir
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                for rev in repo.revisions:
                    strategy = cache_info.delete_revisions(rev.commit_hash)
                    strategy.execute()
                print(f"   purged HF cache for {model_id}", flush=True)
                return
    except Exception as e:
        print(f"   cache purge warning: {e}", flush=True)


def eval_one(name: str, model_id: str, out_root: Path) -> bool:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rl = _common.oracle_eval_path()
    if not rl.is_file():
        sys.exit(f"Missing oracle TOFU_EVAL.json: {rl}")
    cmd = [
        PYTHON, "-W", "ignore", str(EVAL_SCRIPT),
        "--config-name=eval.yaml",
        f"experiment={PAPER_EVAL_EXPERIMENT}",
        f"task_name={name}",
        "forget_split=forget10", "holdout_split=holdout10",
        f"retain_logs_path={rl.as_posix()}",
        f"model.model_args.pretrained_model_name_or_path={model_id}",
        "model.model_args.attn_implementation=eager",
        f"paths.output_dir={out_dir.as_posix()}",
        "eval.tofu.batch_size=8",
    ]
    print(f"\n=== EVAL [{name}] ===", flush=True)
    print(f"   model: {model_id}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc == 0 or _eval_complete(out_dir):
        return True
    print(f"   subprocess rc={rc} and TOFU_SUMMARY missing required metrics", flush=True)
    return False


def write_summary(
    method_label: str,
    checkpoints: list[str],
    rows: list[dict],
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    sj = out_root / f"{method_label.upper()}_GRID_RAW_SUMMARY.json"
    sm = out_root / f"{method_label.upper()}_GRID_RAW_SUMMARY.md"
    sj.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    cols = [
        ("MU",         "model_utility"),
        ("ES",         "extraction_strength"),
        ("EM",         "exact_memorization"),
        ("Para",       "forget_Q_A_PARA_Prob"),
        ("TR_paper",   "forget_truth_ratio_paper"),
        ("Gibberish",  "forget_Q_A_gibberish"),
        ("MIA_LOSS",   "mia_loss"),
        ("MIA_ZLib",   "mia_zlib"),
        ("MIA_MinK",   "mia_min_k"),
        ("MIA_MinK++", "mia_min_k_plus_plus"),
        ("ForgetQ",    "forget_quality"),
    ]
    header = "| #   | Short name" + "".join(f" | {h:>10s}" for h, _ in cols) + " |"
    divider = "| --- | ---" + "".join(" | ---:" for _ in cols) + " |"
    body: list[str] = []
    for i, r in enumerate(rows):
        m = r.get("metrics") or {}
        body.append(
            f"| {i:<3d} | `{r.get('short_name','')}`"
            + "".join(f" | {fmt(m.get(k)):>10s}" for _, k in cols)
            + " |"
        )

    completed = sum(1 for r in rows if r.get("metrics"))
    failed = sum(1 for r in rows if r.get("error"))
    lines = [
        f"# {method_label} HF grid: raw eval metrics\n",
        f"Total checkpoints: {len(checkpoints)}; completed: {completed}; failed: {failed}.\n",
        "## Raw metrics per checkpoint\n",
        header, divider, *body,
    ]
    sm.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {sj}", flush=True)
    print(f"Wrote {sm}", flush=True)


def run_method(
    key: str,
    limit: int | None,
    force: bool,
) -> None:
    checkpoints, out_root, label = METHOD_GRIDS[key]
    targets = checkpoints[:limit] if limit is not None else list(checkpoints)
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}: {len(targets)} checkpoints -> {out_root}", flush=True)
    print(f"{'='*60}", flush=True)

    rows: list[dict] = []
    clear_cuda_cache()

    for i, model_id in enumerate(targets):
        name = _short_name(model_id)
        out_dir = out_root / name
        row_base = {"i": i, "short_name": name, "model_id": model_id}

        if _eval_complete(out_dir) and not force:
            print(f"== {name}: cached, skipping eval ==", flush=True)
        else:
            ok = eval_one(name, model_id, out_root)
            if not ok:
                rows.append({**row_base, "error": "eval_failed"})
                clear_cuda_cache()
                _purge_hf_cache(model_id)
                write_summary(label, checkpoints, rows, out_root)
                continue
            clear_cuda_cache()
            _purge_hf_cache(model_id)

        sp = out_dir / "TOFU_SUMMARY.json"
        metrics = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
        rows.append({**row_base, "metrics": metrics})
        write_summary(label, checkpoints, rows, out_root)

    write_summary(label, checkpoints, rows, out_root)


# ---------------------------------------------------------------------------
# Aggregation pass: ensure every <grid>/<short_name>/ has PAPER_AGGREGATES.json
# ---------------------------------------------------------------------------

def _has_summary(d: Path) -> bool:
    return (d / "TOFU_SUMMARY.json").is_file()


def _has_aggregates(d: Path) -> bool:
    return (d / "PAPER_AGGREGATES.json").is_file()


def _resolve_eval_dir_for_aggregate(d: Path) -> Path | None:
    """Accept either a flat dir holding TOFU_EVAL.json or one with an evals/
    subfolder (ERASE highlight layout). Returns the dir to pass to --dir."""
    if (d / "TOFU_EVAL.json").is_file():
        return d
    sub = d / "evals"
    if (sub / "TOFU_EVAL.json").is_file():
        return sub
    return None


def _aggregate_one(eval_dir: Path) -> bool:
    """Run compute_paper_aggregates on a single dir. Returns True iff
    PAPER_AGGREGATES.json exists afterwards.
    """
    cmd = [
        PYTHON, str(AGGREGATE_SCRIPT),
        "--dir", str(eval_dir),
        "--init-eval", str(INIT_REF_DIR),
        "--retain-eval", str(RETAIN_REF_DIR),
    ]
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    return _has_aggregates(eval_dir) and rc == 0


def aggregate_all_existing(force: bool = False) -> None:
    """Walk every (method_label, grid_root) and run compute_paper_aggregates
    on each subfolder that has TOFU_SUMMARY.json. Idempotent: skips folders
    that already have PAPER_AGGREGATES.json unless ``force=True``.
    Also runs on the two ERASE champion highlight dirs (their evals/ subfolder).
    """
    print(f"\n{'='*60}\n  AGGREGATE PASS\n{'='*60}", flush=True)
    if not (INIT_REF_DIR / "TOFU_EVAL.json").is_file():
        print(
            f"   missing INIT reference: {INIT_REF_DIR / 'TOFU_EVAL.json'} -- abort", flush=True
        )
        sys.exit(1)
    if not (RETAIN_REF_DIR / "TOFU_EVAL.json").is_file():
        print(
            f"   missing RETAIN reference: {RETAIN_REF_DIR / 'TOFU_EVAL.json'} -- abort",
            flush=True,
        )
        sys.exit(1)

    targets: list[tuple[str, Path]] = []
    for label, root in AGGREGATE_METHODS:
        if not root.is_dir():
            print(f"   skip {label}: {root} does not exist", flush=True)
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            ed = _resolve_eval_dir_for_aggregate(d)
            if ed is None:
                continue
            targets.append((label, ed))

    for label, hdir in ERASE_HIGHLIGHTS:
        ed = _resolve_eval_dir_for_aggregate(hdir)
        if ed is not None:
            targets.append((label, ed))
        else:
            print(f"   skip ERASE {label}: no TOFU_EVAL.json under {hdir}", flush=True)

    print(f"   {len(targets)} eval dirs found across all groups", flush=True)
    n_done = 0
    n_skipped = 0
    n_failed = 0
    for label, ed in targets:
        if _has_aggregates(ed) and not force:
            n_skipped += 1
            continue
        ok = _aggregate_one(ed)
        if ok:
            n_done += 1
        else:
            n_failed += 1
            print(f"   AGGREGATE FAILED: [{label}] {ed}", flush=True)
    print(
        f"\n   aggregate pass: {n_done} new, {n_skipped} skipped (cached), "
        f"{n_failed} failed",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Highlights builder: per-method top-3 + main comparison
# ---------------------------------------------------------------------------


def _read_aggregates(eval_dir: Path) -> dict | None:
    pa = eval_dir / "PAPER_AGGREGATES.json"
    if not pa.is_file():
        return None
    try:
        return json.loads(pa.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_summary(eval_dir: Path) -> dict | None:
    sp = eval_dir / "TOFU_SUMMARY.json"
    if not sp.is_file():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _row_from_eval_dir(eval_dir: Path) -> dict | None:
    """Build a flat record {short_name, agg, mem, util, priv, fq, eval_dir, parent_dir}.

    Mem is the paper-aggregated memorization HM (PAPER_AGGREGATES.aggregates.memorization),
    NOT a single raw metric. FQ is the raw forget_quality (KS-test p-value) from
    TOFU_SUMMARY.json. Both are kept because Mem is the headline OU column and FQ is the
    classical TOFU number.

    For grid dirs the eval_dir IS the parent (TOFU_EVAL.json is at top level).
    For ERASE highlights, eval_dir is .../<task>/evals and parent is .../<task>.
    Returns None if either aggregates or summary is unreadable.
    """
    pa = _read_aggregates(eval_dir)
    su = _read_summary(eval_dir)
    if pa is None or su is None:
        return None
    aggs = pa.get("aggregates") or {}
    parent = eval_dir.parent if eval_dir.name == "evals" else eval_dir
    return {
        "short_name": parent.name,
        "agg": aggs.get("aggregate"),
        "mem": aggs.get("memorization"),
        "util": aggs.get("utility"),
        "priv": aggs.get("privacy"),
        "fq": su.get("forget_quality"),
        "eval_dir": eval_dir,
        "parent_dir": parent,
    }


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _select_top3(rows: list[dict]) -> list[tuple[dict, str]]:
    """Top 2 by aggregate plus 1 by forget_quality (if not in top-2). Returns
    list of (row, reason) tuples preserving the chosen order.
    """
    have_agg = [r for r in rows if r.get("agg") is not None]
    have_fq = [r for r in rows if r.get("fq") is not None]
    if not have_agg and not have_fq:
        return []
    by_agg = sorted(have_agg, key=lambda r: r["agg"], reverse=True)
    chosen: list[tuple[dict, str]] = []
    seen: set[str] = set()
    for r in by_agg[:2]:
        chosen.append((r, "best-Agg"))
        seen.add(r["short_name"])
    if have_fq:
        by_fq = sorted(have_fq, key=lambda r: r["fq"], reverse=True)
        for r in by_fq:
            if r["short_name"] not in seen:
                chosen.append((r, "best-FQ"))
                break
    return chosen


def _copy_highlight_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fn in HIGHLIGHT_FILES_TO_COPY:
        src = src_dir / fn
        if src.is_file():
            shutil.copy2(src, dst_dir / fn)


def _write_method_highlights(label: str, picks: list[tuple[dict, str]]) -> None:
    method_dir = HIGHLIGHTS_OUT / label
    method_dir.mkdir(parents=True, exist_ok=True)
    for row, _reason in picks:
        _copy_highlight_files(row["eval_dir"], method_dir / row["short_name"])

    md_lines = [
        f"# {label} top-3 (1B forget10)",
        "",
        "Selected from `saves/eval_grid_*` after running compute_paper_aggregates.",
        "Columns: Agg / Mem / Util / Priv come from PAPER_AGGREGATES.json (paper HMs);",
        "FQ is the raw TOFU forget_quality (KS-test p-value).",
        "",
        "| Rank | Reason     | Config                                                                            |     Agg |     Mem |    Util |    Priv |      FQ |",
        "| ---- | ---------- | --------------------------------------------------------------------------------- | ------: | ------: | ------: | ------: | ------: |",
    ]
    for i, (row, reason) in enumerate(picks, 1):
        md_lines.append(
            f"| {i}    | {reason:<10s} | `{row['short_name']}` | "
            f"{_fmt(row['agg'])} | {_fmt(row['mem'])} | {_fmt(row['util'])} | "
            f"{_fmt(row['priv'])} | {_fmt(row['fq'])} |"
        )
    (method_dir / "HIGHLIGHTS.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _winner_for_main_table(rows: list[dict]) -> dict | None:
    have_agg = [r for r in rows if r.get("agg") is not None]
    if not have_agg:
        return None
    return max(have_agg, key=lambda r: r["agg"])


def build_per_method_highlights_and_main_table() -> None:
    print(f"\n{'='*60}\n  HIGHLIGHTS + MAIN_COMPARISON\n{'='*60}", flush=True)
    HIGHLIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    main_rows: list[tuple[str, dict | None]] = []

    for label, root in AGGREGATE_METHODS:
        if not root.is_dir():
            print(f"   skip {label}: {root} does not exist", flush=True)
            main_rows.append((label, None))
            continue
        rows: list[dict] = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            ed = _resolve_eval_dir_for_aggregate(d)
            if ed is None:
                continue
            r = _row_from_eval_dir(ed)
            if r is not None:
                rows.append(r)
        if not rows:
            print(f"   {label}: 0 evaluable rows -> skipping", flush=True)
            main_rows.append((label, None))
            continue
        picks = _select_top3(rows)
        _write_method_highlights(label, picks)
        winner = _winner_for_main_table(rows)
        main_rows.append((label, winner))
        print(f"   {label}: {len(rows)} rows, top-3 -> {HIGHLIGHTS_OUT / label}", flush=True)

    # ERASE champions: each has its own evals/ subdir
    for label, hdir in ERASE_HIGHLIGHTS:
        ed = _resolve_eval_dir_for_aggregate(hdir)
        if ed is None:
            print(f"   skip {label}: {hdir} not aggregable", flush=True)
            main_rows.append((label, None))
            continue
        r = _row_from_eval_dir(ed)
        main_rows.append((label, r))

    # Write MAIN_COMPARISON_1B.md
    md = [
        "# 1B forget10 -- best-Aggregate per method",
        "",
        "Baselines: SimNPO / NPO / RMU / GradDiff / UNDIAL / AltPO. Each row is the",
        "best checkpoint by paper Aggregate within the corresponding `eval_grid_*`",
        "folder. ERASE champions are listed for direct comparison.",
        "",
        "Columns: Agg / Mem / Util / Priv are the OpenUnlearning paper HMs",
        "(PAPER_AGGREGATES.json). FQ is the raw TOFU forget_quality (KS-test p-value)",
        "from TOFU_SUMMARY.json -- complementary to Mem, not interchangeable.",
        "",
        "| Method            |     Agg |     Mem |    Util |    Priv |      FQ | Best config |",
        "| ----------------- | ------: | ------: | ------: | ------: | ------: | ----------- |",
    ]
    for label, row in main_rows:
        if row is None:
            md.append(f"| {label:<17s} |   n/a   |   n/a   |   n/a   |   n/a   |   n/a   | -- |")
            continue
        md.append(
            f"| {label:<17s} | {_fmt(row['agg'])} | {_fmt(row['mem'])} | "
            f"{_fmt(row['util'])} | {_fmt(row['priv'])} | {_fmt(row['fq'])} | "
            f"`{row['short_name']}` |"
        )
    out = HIGHLIGHTS_OUT / "MAIN_COMPARISON_1B.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n   wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=list(METHOD_GRIDS.keys()), default=None,
                   help="Run only one method (default: all five).")
    p.add_argument("--limit", type=int, default=None,
                   help="Eval only the first N checkpoints per method.")
    p.add_argument("--force", action="store_true",
                   help="Re-eval even if cached.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip every method's eval loop; only run aggregate + highlights.")
    p.add_argument("--only-aggregate", action="store_true",
                   help="Alias for --skip-eval.")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip the post-eval aggregate + highlights pass.")
    p.add_argument("--skip-idknll", action="store_true", default=True,
                   help="Skip the IdkNLL grid (40 ckpts not in the comparison). On by default.")
    p.add_argument("--include-idknll", dest="skip_idknll", action="store_false",
                   help="Include IdkNLL in the eval loop (off by default).")
    p.add_argument("--force-aggregate", action="store_true",
                   help="Re-run compute_paper_aggregates even when PAPER_AGGREGATES.json exists.")
    args = p.parse_args()

    skip_eval = args.skip_eval or args.only_aggregate
    if not skip_eval:
        if args.only:
            methods = [args.only]
        else:
            methods = [k for k in METHOD_GRIDS.keys() if not (args.skip_idknll and k == "idknll")]
        print(f"Methods to evaluate: {methods}", flush=True)
        for key in methods:
            run_method(key, args.limit, args.force)
    else:
        print("Skipping eval loop (--skip-eval / --only-aggregate).", flush=True)

    if args.skip_aggregate:
        print("Skipping aggregate + highlights pass (--skip-aggregate).", flush=True)
    else:
        aggregate_all_existing(force=args.force_aggregate)
        build_per_method_highlights_and_main_table()

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
