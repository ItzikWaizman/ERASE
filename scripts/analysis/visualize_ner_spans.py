"""Visualize which forget tokens the NER span cache marks as 'entity' (the
positions ERASE's selective VJP localizes the forget gradient to). No model /
GPU needed -- just the span cache + the forget text + the Llama-2 tokenizer.

Replicates the trainer's chunking (concat forget texts with \\n\\n, tokenize
w/o special tokens, split into max_length chunks) and the span subsequence
match, then writes:
  - an HTML file with entity tokens highlighted (open in a browser), and
  - a per-chunk coverage printout + an ASCII preview of chunk 0.

Usage:
    python scripts/analysis/visualize_ner_spans.py --n_chunks 5 --preview_tokens 160
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
SPANS = ROOT / "results" / "remote_runs_muse" / "spans_ner.json"
# Local tokenizer (shipped inside a downloaded run folder -> no HF download).
TOK_DIR = ROOT / "results" / "remote_runs_muse" / "MUSE_news_7b_erase_lratau_ner_fg_lr0.04_a2_tau1.5"
OUT_HTML = ROOT / "results" / "remote_runs_muse" / "ner_spans_preview.html"


def chunk_raw_text(texts, tok, max_length):
    full = tok("\n\n".join(texts), add_special_tokens=False)["input_ids"]
    return [full[i * max_length:(i + 1) * max_length]
            for i in range(len(full) // max_length + 1)
            if len(full[i * max_length:(i + 1) * max_length]) >= 2]


def entity_mask(token_ids, by_first):
    mask = [False] * len(token_ids)
    i, n = 0, len(token_ids)
    while i < n:
        hit = False
        for seq in by_first.get(token_ids[i], []):
            L = len(seq)
            if i + L <= n and token_ids[i:i + L] == list(seq):
                for j in range(i, i + L):
                    mask[j] = True
                i += L
                hit = True
                break
        if not hit:
            i += 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_chunks", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--preview_tokens", type=int, default=160)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    spans = json.loads(SPANS.read_text(encoding="utf-8"))["spans"]
    by_first = {}
    for seqs in spans.values():
        for s in seqs:
            by_first.setdefault(int(s[0]), []).append([int(x) for x in s])

    ds = load_dataset("muse-bench/MUSE-News", "raw", split="forget")
    chunks = chunk_raw_text(list(ds["text"]), tok, args.max_length)[: args.n_chunks]

    total_tok = total_ent = 0
    html_parts = ["<html><head><meta charset='utf-8'><style>"
                  "body{font-family:sans-serif;line-height:1.8;max-width:1000px;margin:20px auto}"
                  "mark{background:gold;padding:0 1px;border-radius:2px}"
                  "h3{margin-top:28px}</style></head><body>"
                  "<h2>NER entity tokens on MUSE-News forget chunks "
                  "(gold = VJP-targeted entity token)</h2>"]
    preview_ascii = ""
    for ci, ids in enumerate(chunks):
        mask = entity_mask(ids, by_first)
        nent = sum(mask)
        total_tok += len(ids); total_ent += nent
        toks = tok.convert_ids_to_tokens(ids)
        html_parts.append(f"<h3>chunk {ci}: {nent}/{len(ids)} tokens entity "
                          f"({100*nent/len(ids):.1f}%)</h3><p>")
        for t, m in zip(toks, mask):
            piece = html.escape(t.replace("\u2581", " "))
            html_parts.append(f"<mark>{piece}</mark>" if m else piece)
        html_parts.append("</p>")
        if ci == 0:
            buf = []
            for t, m in zip(toks[:args.preview_tokens], mask[:args.preview_tokens]):
                p = t.replace("\u2581", " ")
                buf.append(f"[{p.strip()}]" if m else p)
            preview_ascii = "".join(buf).encode("ascii", "replace").decode("ascii")
    html_parts.append("</body></html>")
    OUT_HTML.write_text("".join(html_parts), encoding="utf-8")

    print(f"=== NER span coverage over first {len(chunks)} forget chunks ===")
    print(f"entity tokens: {total_ent}/{total_tok} = {100*total_ent/total_tok:.1f}%")
    print(f"HTML -> {OUT_HTML}")
    print(f"\n--- chunk 0 ASCII preview (first {args.preview_tokens} tokens; "
          f"[..] = entity) ---\n{preview_ascii}")


if __name__ == "__main__":
    main()
