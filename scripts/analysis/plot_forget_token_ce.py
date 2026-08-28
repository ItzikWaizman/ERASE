"""Plot per-token forget-CE from dump_forget_token_ce.py dumps: an unlearned
model vs the oracle (and target). Two views:
  (1) pooled per-token CE distribution per model (overall forget-CE shift), and
  (2) per-token CE traces for a few example chunks, with NER-entity tokens
      shaded so you can see whether the entity tokens (the ones the VJP mask
      targets) actually rose in CE relative to the oracle.

Run locally AFTER copying the cluster dumps into results/token_ce/.
    python scripts/analysis/plot_forget_token_ce.py --detail-chunks 0 1 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = ROOT / "results" / "token_ce"
SPANS = ROOT / "results" / "remote_runs_muse" / "spans_ner.json"

# label -> dump file (only those that exist are plotted)
MODELS = [
    ("oracle", "oracle.json"),
    ("target", "target.json"),
    ("tau1.5", "tau1.5.json"),
    ("tau1.75", "tau1.75.json"),
    ("tau2.0", "tau2.0.json"),
]
COLORS = {"oracle": "tab:green", "target": "black", "run": "tab:red",
          "tau1.5": "tab:blue", "tau1.75": "tab:orange", "tau2.0": "tab:red"}


def load_dumps():
    out = []
    for label, fn in MODELS:
        p = DUMP_DIR / fn
        if p.is_file():
            out.append((label, json.loads(p.read_text(encoding="utf-8"))))
        else:
            print(f"[skip] {p} not found")
    return out


def load_span_token_seqs():
    if not SPANS.is_file():
        return []
    d = json.loads(SPANS.read_text(encoding="utf-8"))
    seqs = []
    for s in d["spans"].values():
        seqs.extend(tuple(x) for x in s)
    return seqs


def entity_mask(token_ids, span_seqs):
    """Mark positions covered by any span token-id subsequence (trainer-style)."""
    mask = np.zeros(len(token_ids), dtype=bool)
    by_first = {}
    for seq in span_seqs:
        by_first.setdefault(seq[0], []).append(seq)
    i = 0
    n = len(token_ids)
    while i < n:
        hit = False
        for seq in by_first.get(token_ids[i], []):
            L = len(seq)
            if i + L <= n and tuple(token_ids[i:i + L]) == seq:
                mask[i:i + L] = True
                i += L
                hit = True
                break
        if not hit:
            i += 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-chunks", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--out-prefix", default=str(ROOT / "results" / "token_ce" / "forget_token_ce"))
    ap.add_argument("--dump", action="append", default=[],
                    help="Explicit label=path dump(s); repeatable. Overrides the "
                         "default MODELS list. Used by the per-run telemetry "
                         "(e.g. --dump run=.../token_ce.json --dump oracle=.../oracle.json).")
    args = ap.parse_args()

    if args.dump:
        dumps = []
        for spec in args.dump:
            label, _, path = spec.partition("=")
            p = Path(path)
            if p.is_file():
                dumps.append((label, json.loads(p.read_text(encoding="utf-8"))))
            else:
                print(f"[skip] {p} not found")
    else:
        dumps = load_dumps()
    if not dumps:
        raise SystemExit(f"No dumps found (DUMP_DIR={DUMP_DIR} or --dump). "
                         f"Run dump_forget_token_ce.py first.")
    span_seqs = load_span_token_seqs()

    # --- (1) pooled per-token CE distribution ---
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 12, 60)
    for label, d in dumps:
        all_ce = np.array([c for ch in d["chunks"] for c in ch["token_ce"]])
        ax.hist(all_ce, bins=bins, density=True, histtype="step", linewidth=2,
                color=COLORS.get(label), label=f"{label} (mean {all_ce.mean():.2f})")
    ax.set_xlabel("per-token forget CE (nats)")
    ax.set_ylabel("density")
    ax.set_title(f"MUSE-News per-token forget CE distribution "
                 f"({len(dumps[0][1]['chunks'])} chunks pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_distribution.png", dpi=130)
    print(f"Saved -> {args.out_prefix}_distribution.png")

    # --- (2) per-token traces for example chunks (entity tokens shaded) ---
    for ci in args.detail_chunks:
        fig, ax = plt.subplots(figsize=(14, 4))
        ref = dumps[0][1]["chunks"]
        if ci >= len(ref):
            continue
        tok_ids = ref[ci]["token_ids"]
        if span_seqs:
            mask = entity_mask(tok_ids, span_seqs)
            for j in np.where(mask)[0]:
                ax.axvspan(j - 0.5, j + 0.5, color="gold", alpha=0.25, lw=0)
        for label, d in dumps:
            if ci < len(d["chunks"]):
                ce = d["chunks"][ci]["token_ce"]
                ax.plot(range(len(ce)), ce, lw=0.9, color=COLORS.get(label), label=label)
        ax.set_xlabel(f"token position (chunk {ci}; gold = NER-entity tokens)")
        ax.set_ylabel("per-token CE (nats)")
        ax.set_title(f"Per-token forget CE, chunk {ci}: unlearned vs oracle")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_chunk{ci}.png", dpi=130)
        print(f"Saved -> {args.out_prefix}_chunk{ci}.png")


if __name__ == "__main__":
    main()
