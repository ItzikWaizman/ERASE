"""
Precompute per-layer activation covariances for ERASE (Llama MLP down_proj).

Builds C_retain from TOFU retain split + optional MMLU, C_forget from TOFU forget split.
Writes C_retain_layer_{i}.pt, C_forget_layer_{i}.pt, metadata.pt under --output_dir.

Run from repository root:
    python scripts/erase/compute_covariances.py \\
        --model_name open-unlearning/tofu_Llama-3.2-1B-Instruct_full \\
        --forget_split forget10 --retain_split retain90 \\
        --output_dir saves/precompute/llama1b/covariances \\
        --mmlu_samples 5000 --batch_size 4
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


class QATextDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_length=256, question_key="question", answer_key="answer"):
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


class MMLUDataset(Dataset):
    def __init__(self, hf_data, tokenizer, max_length=256):
        self.data = hf_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.choices = ["A", "B", "C", "D"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        choices_str = "\n".join(
            f"({c}) {row['choices'][i]}"
            for i, c in enumerate(self.choices)
            if i < len(row["choices"])
        )
        text = (
            f"Question: {row['question']}\n{choices_str}\n"
            f"Answer: ({self.choices[row['answer']]})\n\n"
        )
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


def accumulate_covariances(model, dataloader, device, num_layers, d_in, desc=""):
    covariances = [
        torch.zeros(d_in, d_in, dtype=torch.float32, device=device)
        for _ in range(num_layers)
    ]
    total_tokens = 0
    layer_acts = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_acts[layer_idx] = input[0].detach().float()

        return hook

    hooks = []
    for i in range(num_layers):
        proj = model.model.layers[i].mlp.down_proj
        hooks.append(proj.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

            non_pad = attention_mask.bool().reshape(-1)
            for i in range(num_layers):
                x = layer_acts[i].reshape(-1, d_in)
                x_valid = x[non_pad]
                covariances[i].addmm_(x_valid.T, x_valid)

            total_tokens += non_pad.sum().item()
            layer_acts.clear()

    for h in hooks:
        h.remove()

    result = [c.cpu() for c in covariances]
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
    p.add_argument("--forget_split", type=str, default="forget10")
    p.add_argument("--retain_split", type=str, default="retain90")
    p.add_argument(
        "--output_dir",
        type=str,
        default="saves/precompute/llama1b/covariances",
    )
    p.add_argument(
        "--mmlu_samples",
        type=int,
        default=5000,
        help="MMLU test rows to add to C_retain. 0 disables. -1 uses all.",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name} (Llama down_proj)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    d_in = model.config.intermediate_size
    print(f"Layers: {num_layers} | intermediate_size: {d_in}")

    retain_data = load_dataset("locuslab/TOFU", args.retain_split, split="train")
    retain_ds = QATextDataset(retain_data, tokenizer, args.max_length)
    print(f"Retain QA samples: {len(retain_ds)}")

    mmlu_ds = None
    parts = [retain_ds]
    if args.mmlu_samples != 0:
        mmlu_data = load_dataset("cais/mmlu", "all", split="test")
        if args.mmlu_samples > 0 and args.mmlu_samples < len(mmlu_data):
            mmlu_data = mmlu_data.select(range(args.mmlu_samples))
        mmlu_ds = MMLUDataset(mmlu_data, tokenizer, args.max_length)
        print(f"MMLU samples in C_retain: {len(mmlu_ds)}")
        parts.append(mmlu_ds)
    else:
        print("MMLU disabled (--mmlu_samples 0)")

    combined_retain = ConcatDataset(parts)
    retain_loader = DataLoader(
        combined_retain,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
    )

    forget_data = load_dataset("locuslab/TOFU", args.forget_split, split="train")
    forget_ds = QATextDataset(forget_data, tokenizer, args.max_length)
    print(f"Forget QA samples: {len(forget_ds)}")
    forget_loader = DataLoader(
        forget_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
    )

    mmlu_count = len(mmlu_ds) if mmlu_ds is not None else 0
    print(f"\nC_retain over {len(combined_retain)} sequences (retain={len(retain_ds)}, MMLU={mmlu_count})")
    c_retain_list, n_retain = accumulate_covariances(
        model, retain_loader, device, num_layers, d_in, desc="C_retain"
    )
    print(f"  Retain tokens: {n_retain:,}")

    print(f"\nC_forget over {len(forget_ds)} samples...")
    c_forget_list, n_forget = accumulate_covariances(
        model, forget_loader, device, num_layers, d_in, desc="C_forget"
    )
    print(f"  Forget tokens: {n_forget:,}")

    print(f"\nSaving to {args.output_dir}")
    for i in range(num_layers):
        c_retain = c_retain_list[i] / n_retain
        c_forget = c_forget_list[i] / n_forget
        torch.save(c_retain, os.path.join(args.output_dir, f"C_retain_layer_{i}.pt"))
        torch.save(c_forget, os.path.join(args.output_dir, f"C_forget_layer_{i}.pt"))
        if i % 4 == 0:
            print(
                f"  Layer {i:2d}: C_retain diag_mean={c_retain.diag().mean():.4f} | "
                f"C_forget diag_mean={c_forget.diag().mean():.4f}"
            )

    meta = {
        "model_name": args.model_name,
        "model_type": "llama",
        "mlp_proj_name": "down_proj",
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "num_layers": num_layers,
        "d_in": d_in,
        "n_retain_tokens": n_retain,
        "n_forget_tokens": n_forget,
        "retain_samples": len(retain_ds),
        "mmlu_samples": mmlu_count,
        "forget_samples": len(forget_ds),
        "max_length": args.max_length,
    }
    torch.save(meta, os.path.join(args.output_dir, "metadata.pt"))
    print("Done.", meta)


if __name__ == "__main__":
    main()
