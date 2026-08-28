import json
import os
import re
from pathlib import Path


def parse_dir_name(dirname):
    info = {}
    info["dirname"] = dirname
    m = re.match(r"^(.+?)_(\d+)ep_", dirname)
    if m:
        info["prefix"] = m.group(1)
        info["epochs"] = int(m.group(2))
    else:
        info["prefix"] = dirname[:20]
        info["epochs"] = 0

    m = re.search(r"_a(\d+\.?\d*)_", dirname)
    if m:
        info["alpha"] = float(m.group(1))
    else:
        info["alpha"] = None

    m = re.search(r"_lr(\d+\.?\d*)_", dirname)
    if m:
        info["lr"] = float(m.group(1))
    else:
        info["lr"] = None

    m = re.search(r"_(L[\d]+)_", dirname)
    if m:
        info["layers"] = m.group(1)
    else:
        info["layers"] = None

    m = re.search(r"_rw(\d+\.?\d*)", dirname)
    if m:
        info["rw"] = float(m.group(1))
    else:
        info["rw"] = None

    m = re.search(r"_ptcap(\d+\.?\d*)", dirname)
    if m:
        info["ptcap"] = float(m.group(1))
    else:
        info["ptcap"] = None

    info["smooth"] = "smooth" in dirname.lower()
    return info


def load_metrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    keys = [
        "exact_memorization",
        "extraction_strength",
        "forget_Q_A_gibberish",
        "model_utility",
        "forget_quality",
        "mia_min_k_plus_plus",
        "forget_truth_ratio",
    ]
    metrics = {}
    for k in keys:
        metrics[k] = data.get(k, None)
    return metrics


def find_summaries(base_dir):
    results = []
    base = Path(base_dir)
    if not base.exists():
        return results
    for root, dirs, files in os.walk(base):
        if "TOFU_SUMMARY.json" in files:
            json_path = os.path.join(root, "TOFU_SUMMARY.json")
            # The run directory is the parent of evals/
            run_dir = os.path.basename(os.path.dirname(root))
            if run_dir == "evals":
                run_dir = os.path.basename(
                    os.path.dirname(os.path.dirname(json_path))
                )
            info = parse_dir_name(run_dir)
            info["source"] = (
                "highlights" if "highlights" in str(root) else "unlearn"
            )
            try:
                metrics = load_metrics(json_path)
                info.update(metrics)
                results.append(info)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
    return results


def fmt(v, width=8):
    if v is None:
        return "-".center(width)
    if isinstance(v, float):
        return f"{v:.4f}".rjust(width)
    return str(v).rjust(width)


def print_table(rows, title):
    sep = "=" * 140
    dash = "-" * 140
    if not rows:
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        print("  (no runs found)")
        return
    print(f"\n{sep}")
    n = len(rows)
    print(f"  {title}  ({n} runs)")
    print(sep)
    hdr = (
        f"{'PREFIX':>20s} {'ALPHA':>7s} {'EP':>4s} {'LR':>7s} {'RW':>6s} "
        f"{'PTCAP':>6s} {'LAYERS':>10s} {'SMTH':>6s} "
        f"{'GIBBRSH':>8s} {'EX_MEM':>8s} {'EXT_STR':>8s} "
        f"{'M_UTIL':>8s} {'FQ':>8s} {'MIA_MK':>8s} {'FTR':>8s}"
    )
    print(hdr)
    print(dash)
    for r in rows:
        alpha_s = f"{r['alpha']:.1f}" if r["alpha"] is not None else "-"
        lr_s = f"{r['lr']}" if r["lr"] is not None else "-"
        rw_s = f"{r['rw']}" if r["rw"] is not None else "-"
        ptcap_s = f"{r['ptcap']}" if r["ptcap"] is not None else "-"
        layers_s = r["layers"] if r["layers"] else "-"
        smooth_s = "Y" if r["smooth"] else "-"
        line = (
            f"{r['prefix']:>20s} {alpha_s:>7s} {r['epochs']:>4d} {lr_s:>7s} {rw_s:>6s} "
            f"{ptcap_s:>6s} {layers_s:>10s} {smooth_s:>6s} "
            f"{fmt(r.get('forget_Q_A_gibberish'))} {fmt(r.get('exact_memorization'))} {fmt(r.get('extraction_strength'))} "
            f"{fmt(r.get('model_utility'))} {fmt(r.get('forget_quality'))} {fmt(r.get('mia_min_k_plus_plus'))} {fmt(r.get('forget_truth_ratio'))}"
        )
        print(line)


def main():
    base1 = os.path.join("saves", "unlearn")
    base2 = os.path.join("saves", "highlights")
    all_runs = find_summaries(base1) + find_summaries(base2)
    print(f"Total runs found: {len(all_runs)}")

    l06 = [
        r
        for r in all_runs
        if r["prefix"].startswith("L06_ALPHA") and r["source"] == "unlearn"
    ]
    l05 = [
        r
        for r in all_runs
        if r["prefix"].startswith("L05_ALPHA") and r["source"] == "unlearn"
    ]
    l04 = [
        r
        for r in all_runs
        if r["prefix"].startswith("L04_ALPHA") and r["source"] == "unlearn"
    ]
    highlights = [r for r in all_runs if r["source"] == "highlights"]
    used = set()
    for r in l06 + l05 + l04 + highlights:
        used.add(id(r))
    iter_prefixes = ("ITER7", "ITER5", "ITER6")
    others = [
        r
        for r in all_runs
        if id(r) not in used and r["prefix"].startswith(iter_prefixes)
    ]
    remaining = [
        r
        for r in all_runs
        if id(r) not in used and not r["prefix"].startswith(iter_prefixes)
    ]

    l06.sort(key=lambda r: (r["alpha"] or 0, r["epochs"]))
    l05.sort(key=lambda r: (r["alpha"] or 0, r["epochs"]))
    l04.sort(key=lambda r: (r["alpha"] or 0, r["epochs"]))
    highlights.sort(key=lambda r: -(r.get("forget_Q_A_gibberish") or 0))
    others.sort(key=lambda r: -(r.get("forget_Q_A_gibberish") or 0))

    print_table(l06, "SECTION 1: All L06_ALPHA runs (sorted by alpha, then epochs)")
    print_table(l05, "SECTION 2: All L05_ALPHA runs (sorted by alpha, then epochs)")
    print_table(l04, "SECTION 3: All L04_ALPHA runs (sorted by alpha, then epochs)")
    print_table(
        highlights, "SECTION 4: All runs in highlights/ (sorted by gibberish desc)"
    )
    print_table(
        others,
        "SECTION 5: Other recent runs - ITER7/ITER5/ITER6 (sorted by gibberish desc)",
    )
    if remaining:
        remaining.sort(key=lambda r: -(r.get("forget_Q_A_gibberish") or 0))
        print_table(remaining, "SECTION 6: All other runs not in above sections")


if __name__ == "__main__":
    main()
