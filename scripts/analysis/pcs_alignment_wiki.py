"""
Experiment E: PC alignment between C_forget (TOFU) and C_wiki (Wikipedia).

Generates eigenvalue decay + alignment heatmap for each layer,
same layout as pcs_alignment.py but substituting C_wiki for C_retain.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

LAYERS = [0, 1, 2]
TOP_K = 100
FORGET_DIR = _REPO_ROOT / "saves" / "precompute" / "llama1b" / "covariances"
WIKI_DIR = _REPO_ROOT / "saves" / "precompute" / "llama1b" / "wikipedia_covariance"
DEVICE = "cpu"


def load_and_decompose(path):
    print(f"Loading {path.name} from {path.parent.name}/...")
    C = torch.load(path, map_location=DEVICE).to(torch.float32)
    evals, evecs = torch.linalg.eigh(C)
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    return evals.cpu(), evecs[:, :TOP_K].cpu()


def analyze_layer(layer: int):
    forget_path = FORGET_DIR / f"C_forget_layer_{layer}.pt"
    wiki_path = WIKI_DIR / f"C_retain_layer_{layer}.pt"

    if not forget_path.exists():
        print(f"Missing {forget_path}")
        return
    if not wiki_path.exists():
        print(f"Missing {wiki_path}")
        return

    evals_forget, pcs_forget = load_and_decompose(forget_path)
    evals_wiki, pcs_wiki = load_and_decompose(wiki_path)

    print(f"\nLayer {layer} - Top 10 eigenvalues:")
    print(f"  Forget: {evals_forget[:10].numpy()}")
    print(f"  Wiki:   {evals_wiki[:10].numpy()}")

    similarity = torch.abs(torch.matmul(pcs_forget.T, pcs_wiki)).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    indices = np.arange(1, len(evals_forget) + 1)
    ax1.plot(indices[:500], evals_forget[:500].numpy(), label="Forget", alpha=0.8, linewidth=2)
    ax1.plot(indices[:500], evals_wiki[:500].numpy(), label="Wiki", alpha=0.8, linestyle="--", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_title(f"Eigenvalue Decay (Layer {layer})", fontsize=14)
    ax1.set_xlabel("PC Index")
    ax1.set_ylabel("Eigenvalue (Log Scale)")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    sns.heatmap(similarity, ax=ax2, cmap="magma", vmin=0, vmax=1)
    ax2.set_title(f"PC Alignment: Forget vs Wiki (Top {TOP_K})", fontsize=14)
    ax2.set_xlabel("Wiki PCs")
    ax2.set_ylabel("Forget PCs")

    plt.tight_layout()
    out_path = _SCRIPT_DIR / f"analysis_wiki_layer_{layer}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for layer in LAYERS:
        analyze_layer(layer)
    print("\nDone.")


if __name__ == "__main__":
    main()
