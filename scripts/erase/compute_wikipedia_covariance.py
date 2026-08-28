"""
Precompute per-layer activation covariances from Wikipedia for ERASE (Llama MLP down_proj).

Mimics PFT's covariance collection: gather E[x x^T] from general text (Wikipedia)
so the projection P = (I + alpha * C_wiki)^{-1} smooths activations away from
the general text manifold.

Saves C_retain_layer_{i}.pt (named for compatibility with ERASE loader) and metadata.pt.

Optional --forget_split additionally accumulates C_forget on the given TOFU
forget split (instead of writing dummy zeros), so a single output_dir can serve
both sides of ERASE's projection P = (I + alpha*C_retain)^-1 * (I + beta*C_forget).

Run from repository root:
    python scripts/erase/compute_wikipedia_covariance.py \
        --model_name open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --output_dir saves/precompute/llama1b/wikipedia_covariance \
        --num_samples 100000 --batch_size 4 --forget_split forget10
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


class WikipediaDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_length=256):
        self.data = hf_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class TofuQADataset(Dataset):
    """TOFU Q&A pairs formatted as plain text, mirroring compute_covariances.py."""

    def __init__(self, hf_data, tokenizer, max_length=256,
                 question_key="question", answer_key="answer"):
        self.data = hf_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = f"Question: {row[self.qk]}\nAnswer: {row[self.ak]}\n\n"
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def pad_collate(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def accumulate_covariances(model, dataloader, device, layer_indices, d_in, desc=""):
    """Accumulate E[x x^T] at down_proj input for the given layer indices.

    Returns ({layer_idx: covariance_tensor_cpu}, total_tokens).
    """
    covariances = {
        i: torch.zeros(d_in, d_in, dtype=torch.float32, device=device)
        for i in layer_indices
    }
    total_tokens = 0
    layer_acts = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_acts[layer_idx] = input[0].detach().float()
        return hook

    hooks = []
    for i in layer_indices:
        proj = model.model.layers[i].mlp.down_proj
        hooks.append(proj.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

            non_pad = attention_mask.bool().reshape(-1)
            for i in layer_indices:
                x = layer_acts[i].reshape(-1, d_in)
                x_valid = x[non_pad]
                covariances[i].addmm_(x_valid.T, x_valid)

            total_tokens += non_pad.sum().item()
            layer_acts.clear()

    for h in hooks:
        h.remove()

    result = {i: c.cpu() for i, c in covariances.items()}
    del covariances
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, total_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        type=str,
        default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    )
    p.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "Tokenizer to load. Defaults to --model_name. Set explicitly when "
            "the model repo ships no tokenizer (e.g. muse-bench/MUSE-*_target, "
            "whose tokenizer is meta-llama/Llama-2-7b-hf)."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="saves/precompute/llama1b/wikipedia_covariance",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of Wikipedia articles to use.",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument(
        "--forget_split",
        type=str,
        default=None,
        help=(
            "If set (e.g. 'forget10'), compute a real C_forget from this TOFU "
            "split alongside the Wikipedia C_retain. If unset, writes zero "
            "C_forget tensors (legacy behavior, kept for backward compatibility)."
        ),
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Restrict to an inclusive layer range 'LO-HI' (e.g. '20-31'). Only "
            "these C_retain/C_forget files are written; existing files for "
            "other layers are untouched. Metadata goes to "
            "metadata_layers_{LO}_{HI}.pt so the original metadata.pt is never "
            "clobbered. Uses the same shuffle seed + sample count as a full "
            "run, so the token stream (and hence the statistics) matches the "
            "previously computed layers."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name} (Llama down_proj)")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    d_in = model.config.intermediate_size
    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        if not (0 <= lo <= hi < num_layers):
            raise SystemExit(f"--layers {args.layers} out of range 0..{num_layers-1}")
        layer_indices = list(range(lo, hi + 1))
    else:
        layer_indices = list(range(num_layers))
    print(f"Layers: {num_layers} (computing {layer_indices[0]}..{layer_indices[-1]}) "
          f"| intermediate_size: {d_in}")

    print(f"Loading Wikipedia...")
    try:
        raw_ds = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    except RuntimeError:
        raw_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", trust_remote_code=True)
    raw_ds = raw_ds.shuffle(seed=42)
    if args.num_samples > 0 and args.num_samples < len(raw_ds):
        raw_ds = raw_ds.select(range(args.num_samples))
    print(f"Wikipedia samples: {len(raw_ds)}")

    wiki_ds = WikipediaDataset(raw_ds, tokenizer, args.max_length)
    wiki_loader = DataLoader(
        wiki_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
    )

    print(f"\nComputing C_wiki over {len(wiki_ds)} Wikipedia samples...")
    c_wiki_list, n_tokens = accumulate_covariances(
        model, wiki_loader, device, layer_indices, d_in, desc="C_wiki"
    )
    print(f"  Total tokens: {n_tokens:,}")

    c_forget_list = None
    n_forget_tokens = 0
    forget_samples = 0
    if args.forget_split:
        print(f"\nLoading TOFU forget split: {args.forget_split}")
        forget_data = load_dataset("locuslab/TOFU", args.forget_split, split="train")
        forget_ds = TofuQADataset(forget_data, tokenizer, args.max_length)
        forget_samples = len(forget_ds)
        print(f"Forget samples: {forget_samples}")
        forget_loader = DataLoader(
            forget_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=0,
        )
        print(f"\nComputing C_forget over {forget_samples} TOFU samples...")
        c_forget_list, n_forget_tokens = accumulate_covariances(
            model, forget_loader, device, layer_indices, d_in, desc="C_forget"
        )
        print(f"  Forget tokens: {n_forget_tokens:,}")

    print(f"\nSaving to {args.output_dir}")
    for i in layer_indices:
        c_wiki = c_wiki_list[i] / n_tokens
        torch.save(c_wiki, os.path.join(args.output_dir, f"C_retain_layer_{i}.pt"))
        if c_forget_list is not None:
            c_forget = c_forget_list[i] / n_forget_tokens
            torch.save(c_forget, os.path.join(args.output_dir, f"C_forget_layer_{i}.pt"))
        else:
            # Backward compat: dummy zeros so ERASE's loader doesn't fail when beta=0.
            c_dummy = torch.zeros_like(c_wiki)
            torch.save(c_dummy, os.path.join(args.output_dir, f"C_forget_layer_{i}.pt"))
        if i % 4 == 0:
            msg = f"  Layer {i:2d}: C_wiki diag_mean={c_wiki.diag().mean():.6f}"
            if c_forget_list is not None:
                cf = c_forget_list[i] / n_forget_tokens
                msg += f" | C_forget diag_mean={cf.diag().mean():.6f}"
            print(msg)

    meta = {
        "model_name": args.model_name,
        "model_type": "llama",
        "mlp_proj_name": "down_proj",
        "source": "wikipedia_20220301.en",
        "num_layers": num_layers,
        "layer_indices": layer_indices,
        "d_in": d_in,
        "n_tokens": n_tokens,
        "num_samples": len(wiki_ds),
        "max_length": args.max_length,
        "forget_split": args.forget_split,
        "n_forget_tokens": n_forget_tokens,
        "forget_samples": forget_samples,
    }
    meta_name = (
        f"metadata_layers_{layer_indices[0]}_{layer_indices[-1]}.pt"
        if args.layers else "metadata.pt"
    )
    torch.save(meta, os.path.join(args.output_dir, meta_name))
    print("Done.", meta)


if __name__ == "__main__":
    main()
