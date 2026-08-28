import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
LAYER = 0  # Change this to analyze different layers
TOP_K = 100
# Matches scripts/erase/compute_covariances.py default --output_dir
DATA_DIR = _REPO_ROOT / "saves" / "precompute" / "llama1b" / "covariances"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def load_and_decompose(path):
    print(f"Loading {path.name}...")
    # Load and ensure float32 for eigenvalue stability
    C = torch.load(path, map_location=DEVICE).to(torch.float32)
    
    # eigh returns eigenvalues in ascending order
    evals, evecs = torch.linalg.eigh(C)
    
    # Flip to get descending order (largest first)
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    
    return evals.cpu(), evecs[:, :TOP_K].cpu()

def main():
    forget_path = DATA_DIR / f"C_forget_layer_{LAYER}.pt"
    # Pipeline saves C_retain_layer_*.pt (see compute_covariances.py), not C_retrain_*.
    retain_path = DATA_DIR / f"C_retain_layer_{LAYER}.pt"

    if not forget_path.exists() or not retain_path.exists():
        print(f"Error: Files for layer {LAYER} not found in {DATA_DIR}")
        print(f"  Expected: {forget_path.name} and {retain_path.name}")
        print("  Run: python scripts/erase/compute_covariances.py (with your model args).")
        return

    # 1. Decompose both matrices
    evals_forget, pcs_forget = load_and_decompose(forget_path)
    evals_retain, pcs_retain = load_and_decompose(retain_path)

    print("\nTop 10 eigenvalues (Forget):", evals_forget[:10].numpy())
    print("Top 10 eigenvalues (Retain):", evals_retain[:10].numpy())

    # 2. Compute PC Similarity (Top 100)
    # Using absolute value because PC direction (+/-) is arbitrary
    similarity = torch.abs(torch.matmul(pcs_forget.T, pcs_retain)).numpy()

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot A: Eigenvalue Scree Plot (Log Scale)
    indices = np.arange(1, len(evals_forget) + 1)
    ax1.plot(indices[:500], evals_forget[:500].numpy(), label='Forget', alpha=0.8, linewidth=2)
    ax1.plot(indices[:500], evals_retain[:500].numpy(), label='Retain', alpha=0.8, linestyle='--', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title(f"Eigenvalue Decay (Layer {LAYER})", fontsize=14)
    ax1.set_xlabel("PC Index")
    ax1.set_ylabel("Eigenvalue (Log Scale)")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Plot B: Similarity Heatmap
    sns.heatmap(similarity, ax=ax2, cmap='magma', vmin=0, vmax=1)
    ax2.set_title(f"PC Alignment (Top {TOP_K})", fontsize=14)
    ax2.set_xlabel("Retain PCs")
    ax2.set_ylabel("Forget PCs")

    plt.tight_layout()
    out_path = _SCRIPT_DIR / f"analysis_layer_{LAYER}.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nAnalysis complete! Plot saved as: {out_path}")

if __name__ == "__main__":
    main()