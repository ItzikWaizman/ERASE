"""
Build a forget-span cache for ERASE's selective-VJP on MUSE (raw text).

ERASE localizes the forget gradient to subject-entity token spans. On TOFU
those spans come from author names; MUSE raw text has no labels, so we extract
salient entities with spaCy NER and emit a token-span cache in the SAME format
the ERASE trainer already consumes for TOFU author spans:

    {"spans": {entity_str: [[tok_id, ...], ...], ...}, "entities": [...], ...}

Two modes (the two VJP variants):
  --mode ner : NER over the RAW forget passages (muse-bench/MUSE-<split>, raw,
               split=forget). Entities found in the text the model trained on.
  --mode qa  : NER over the forget knowledge-probe Q+A (config=knowmem,
               split=forget_qa). Entities that define the *facts* being tested,
               then matched back into the raw forget chunks at train time.

The trainer (src/trainer/unlearn/erase.py) loads this via method_arg
`forget_span_cache` with `author_mask_mode="span"`, and falls back to the full
supervised-token VJP for any chunk with no matched span.

Requires spaCy + an English model:
    pip install spacy && python -m spacy download en_core_web_sm

Usage:
    python scripts/erase/build_muse_spans.py --mode ner \
        --tokenizer_name meta-llama/Llama-2-7b-hf \
        --data_path muse-bench/MUSE-News \
        --output saves/precompute/muse_news_7b/spans_ner.json
"""

import os
import json
import argparse

from transformers import AutoTokenizer
from datasets import load_dataset

DEFAULT_ENT_TYPES = ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "NORP"]


def load_spacy(model="en_core_web_sm"):
    try:
        import spacy
    except ImportError as e:
        raise SystemExit(
            "spaCy is required. Install with:\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm"
        ) from e
    try:
        # Only the NER pipeline is needed; disabling the rest speeds it up.
        return spacy.load(model, disable=["lemmatizer", "tagger", "parser"])
    except OSError as e:
        raise SystemExit(
            f"spaCy model '{model}' not found. Install it with:\n"
            f"  python -m spacy download {model}"
        ) from e


def collect_entities(texts, nlp, ent_types, min_chars=3, max_words=6):
    keep = set(ent_types)
    ents: set[str] = set()
    for doc in nlp.pipe((t for t in texts if t), batch_size=64):
        for ent in doc.ents:
            if ent.label_ not in keep:
                continue
            s = ent.text.strip().strip(".,;:'\"()[]")
            if len(s) < min_chars or len(s.split()) > max_words:
                continue
            ents.add(s)
    return sorted(ents)


def tokenize_spans(entities, tokenizer):
    """Token-id sequences for each entity (with leading-space/newline variants).

    Mirrors _build_author_name_spans: multi-token entities only (len>=2) to
    avoid matching a single common subword everywhere.
    """
    spans: dict[str, list[list[int]]] = {}
    for ent in entities:
        seqs: list[list[int]] = []
        seen: set[tuple] = set()
        for variant in (ent, " " + ent, "\n" + ent):
            ids = tokenizer(variant, add_special_tokens=False).get("input_ids", [])
            ids = [int(t) for t in ids]
            if len(ids) >= 2:
                key = tuple(ids)
                if key not in seen:
                    seen.add(key)
                    seqs.append(ids)
        if seqs:
            spans[ent] = seqs
    return spans


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ner", "qa"], required=True)
    p.add_argument("--tokenizer_name", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--data_path", default="muse-bench/MUSE-News")
    p.add_argument("--forget_split", default="forget")
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--ent_types", nargs="*", default=DEFAULT_ENT_TYPES)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    nlp = load_spacy(args.spacy_model)

    if args.mode == "ner":
        ds = load_dataset(args.data_path, "raw", split=args.forget_split)
        texts = list(ds["text"])
        source = f"{args.data_path}/raw/{args.forget_split}"
    else:  # qa
        ds = load_dataset(args.data_path, "knowmem", split="forget_qa")
        texts = [
            f"{row.get('question', '')} {row.get('answer', '')}".strip()
            for row in ds
        ]
        source = f"{args.data_path}/knowmem/forget_qa"

    print(f"[{args.mode}] {len(texts)} texts from {source}")
    entities = collect_entities(texts, nlp, args.ent_types)
    print(f"[{args.mode}] {len(entities)} unique entities ({args.ent_types})")

    spans = tokenize_spans(entities, tokenizer)
    n_seqs = sum(len(v) for v in spans.values())
    print(f"[{args.mode}] {len(spans)} entities -> {n_seqs} token sequences")

    out = {
        "spans": spans,
        "entities": sorted(spans.keys()),
        "meta": {
            "mode": args.mode,
            "source": source,
            "tokenizer": args.tokenizer_name,
            "spacy_model": args.spacy_model,
            "ent_types": args.ent_types,
            "n_entities": len(spans),
            "n_token_sequences": n_seqs,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
