"""
Calibrate the ERASE CET target tau for MUSE.

Replicates OpenUnlearning's PretrainingDataset chunking exactly
(src/data/pretraining.py): concatenate all forget articles with "\n\n",
tokenize WITHOUT special tokens, and split into fixed-length chunks. For each
chunk we measure the model's mean per-token cross-entropy under teacher
forcing -- i.e. the same per-sample quantity the CET loss steers.

Run this on BOTH:
  * the MUSE target  (muse-bench/MUSE-<split>_target)  -> "memorized" CE floor
  * the MUSE retrain (muse-bench/MUSE-<split>_retrain)  -> oracle "forgotten" CE

tau should be centered near the RETRAIN oracle's per-chunk CE (that is the
distribution ERASE is trying to match), with the 10-value sweep spanning
roughly [target_floor, oracle + a margin].

Usage:
    python scripts/erase/measure_muse_ce.py \
        --model_name muse-bench/MUSE-News_target \
        --tokenizer_name meta-llama/Llama-2-7b-hf \
        --data_path muse-bench/MUSE-News --name raw --split forget \
        --max_length 2048 \
        --output saves/precompute/muse_news_7b/ce_target.json
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


def chunk_raw_text(texts, tokenizer, max_length):
    """Mirror PretrainingDataset._chunk_raw_text (src/data/pretraining.py)."""
    raw_text = "\n\n".join(texts)
    full = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
    num_chunks = len(full) // max_length + 1
    chunks = []
    for i in range(num_chunks):
        seg = full[i * max_length : (i + 1) * max_length]
        if len(seg) >= 2:  # need >=2 tokens to form one (input, label) pair
            chunks.append(seg)
    return chunks


@torch.no_grad()
def chunk_mean_ce(model, device, chunk_ids):
    input_ids = torch.tensor([chunk_ids], device=device)
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="mean",
    )
    return float(ce)


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--tokenizer_name", type=str, default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--data_path", type=str, default="muse-bench/MUSE-News")
    p.add_argument("--name", type=str, default="raw")
    p.add_argument("--split", type=str, default="forget")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print(f"Loading {args.data_path} [{args.name}/{args.split}] ...")
    data = load_dataset(args.data_path, args.name, split=args.split)
    chunks = chunk_raw_text(data["text"], tokenizer, args.max_length)
    print(f"Chunks @ max_length={args.max_length}: {len(chunks)}")

    ces = [chunk_mean_ce(model, device, c) for c in tqdm(chunks, desc="per-chunk CE")]
    ces_sorted = sorted(ces)
    n = len(ces)
    mean = sum(ces) / n
    var = sum((x - mean) ** 2 for x in ces) / n

    stats = {
        "model_name": args.model_name,
        "data": f"{args.data_path}/{args.name}/{args.split}",
        "max_length": args.max_length,
        "n_chunks": n,
        "mean_per_chunk_ce": mean,
        "std_per_chunk_ce": var ** 0.5,
        "median": percentile(ces_sorted, 50),
        "p10": percentile(ces_sorted, 10),
        "p25": percentile(ces_sorted, 25),
        "p75": percentile(ces_sorted, 75),
        "p90": percentile(ces_sorted, 90),
        "min": ces_sorted[0],
        "max": ces_sorted[-1],
    }

    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== per-chunk CE (nats/token) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {args.output}")
    print(
        "\ntau hint: center the 10-value sweep near the RETRAIN mean "
        "(oracle), spanning ~[target_mean, retrain_mean + ~1.5]."
    )


if __name__ == "__main__":
    main()
