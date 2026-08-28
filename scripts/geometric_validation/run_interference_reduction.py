"""
Geometric validation v2: interference-reduction ratio R^l per layer.

Tests whether the spectral filter

    P^l = (I + alpha * C_retain^l)^{-1}

reduces the second-moment overlap between the forget-side activation
distribution and the retain (Wikipedia) one, at each transformer layer.

Population metric (no model inference required):

    R^l = trace(P^l C_retain^l P^l C_forget^l)
        / trace(C_retain^l C_forget^l)

In the eigenbasis of C_retain^l = U Lambda U^T, with
M_kk := u_k^T C_forget^l u_k:

    R^l = sum_k (lambda_k * M_kk) * (1 + alpha*lambda_k)^{-2}
        / sum_k (lambda_k * M_kk).

Interpretation:
- R^l = 1   : P^l does nothing, no reduction of forget/retain interference.
- R^l --> 0 : P^l fully suppresses the directions in which the forget
              gradient could damage retain (Wikipedia) knowledge.

Inputs:
- saves/precompute/llama1b/wikipedia_covariance/C_retain_layer_{l}.pt
- saves/precompute/llama1b/wikipedia_covariance/C_forget_layer_{l}.pt

Outputs (under --output_dir):
- metrics.json
- REPORT.md
- per_component_layer{L0}.npz   raw arrays for the per-component figure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cov_dir",
        default="saves/precompute/llama1b/wikipedia_covariance",
        help="Directory with C_retain_layer_{l}.pt and C_forget_layer_{l}.pt.",
    )
    p.add_argument(
        "--filtered_layers",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Layers where P^l is applied in production (Pure-alpha winner).",
    )
    p.add_argument(
        "--control_layers",
        nargs="+",
        type=int,
        default=[8, 12, 15],
        help="Layers NOT filtered in production; we still compute R^l there "
        "to show the spectral filter would have less to do (interference "
        "between forget and retain is structurally smaller).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="alpha used by the Pure-alpha winner.",
    )
    p.add_argument(
        "--per_component_layer",
        type=int,
        default=0,
        help="Layer to dump per-component arrays for (figure 2).",
    )
    p.add_argument(
        "--per_component_top_k",
        type=int,
        default=50,
        help="How many leading components to dump for the per-component figure.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the eigendecomposition.",
    )
    p.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Precision for eig/matmuls. float64 is safer for ill-conditioned C.",
    )
    p.add_argument(
        "--output_dir",
        default="saves/diagnostics/geometric/interference_reduction",
    )
    return p.parse_args()


def _load_cov(cov_dir: str, layer: int, kind: str, dtype: torch.dtype) -> torch.Tensor:
    path = os.path.join(cov_dir, f"C_{kind}_layer_{layer}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing covariance file: {path}")
    C = torch.load(path, map_location="cpu", weights_only=True)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise RuntimeError(f"{path} is not a square matrix; got {tuple(C.shape)}")
    # Symmetrize against tiny floating drift (cov is symmetric in expectation).
    C = 0.5 * (C + C.T)
    return C.to(dtype=dtype)


def _layer_metrics(
    Cr: torch.Tensor,
    Cf: torch.Tensor,
    alpha: float,
    device: str,
) -> dict:
    """Eigendecompose C_retain, then evaluate R^l in that basis.

    All return values are CPU floats / numpy arrays, suitable for JSON.
    """
    d = Cr.shape[0]
    Cr_dev = Cr.to(device)
    Cf_dev = Cf.to(device)

    eigvals, U = torch.linalg.eigh(Cr_dev)
    # eigh returns ascending eigenvalues; flip to descending for readability.
    eigvals = torch.flip(eigvals, dims=[0])
    U = torch.flip(U, dims=[1])
    # Numerical floor (tiny negatives from finite precision).
    eigvals = torch.clamp(eigvals, min=0.0)

    # M_kk = (U^T C_f U)_kk -- the per-component forget energy in retain's
    # eigenbasis. Computed as elementwise sum over (Cf @ U) * U.
    CfU = Cf_dev @ U
    M = (U * CfU).sum(dim=0)
    M = torch.clamp(M, min=0.0)  # small negatives possible with float32

    weights_off = eigvals * M  # numerator integrand BEFORE filter
    suppression = (1.0 + alpha * eigvals).pow(-2)  # filter contraction per dim
    weights_on = weights_off * suppression  # numerator integrand AFTER filter

    num = weights_on.sum().item()
    den = weights_off.sum().item()
    R = num / den if den > 0 else float("nan")

    # For figures / sanity:
    cumulative_off = torch.cumsum(weights_off, dim=0) / weights_off.sum()
    cumulative_on = torch.cumsum(weights_on, dim=0) / weights_off.sum()
    # Effective rank of C_retain (entropy of normalized eigenvalues).
    p = eigvals / eigvals.sum().clamp(min=1e-30)
    eff_rank = torch.exp(-(p * (p.clamp(min=1e-30)).log()).sum()).item()

    # Direct trace check (via single matmul) to cross-validate the eig formula.
    # P = U diag(1/(1+a*l)) U^T -- but using U at fp32 is not necessary here.
    # We compare the population metric.
    # trace(P Cr P Cf) = sum_k l_k/(1+a l_k)^2 * M_kk
    # trace(Cr Cf)    = sum_k l_k * M_kk
    direct_num = (eigvals / (1.0 + alpha * eigvals).pow(2) * M).sum().item()
    direct_den = (eigvals * M).sum().item()

    return {
        "dim": int(d),
        "alpha": float(alpha),
        "R": R,
        "R_direct": direct_num / direct_den if direct_den > 0 else float("nan"),
        "trace_Cr_Cf": float(direct_den),
        "trace_PCrPCf": float(direct_num),
        "effective_rank": eff_rank,
        "lambda_top1": float(eigvals[0].item()),
        "lambda_top10_sum_frac": float(
            eigvals[:10].sum().item() / eigvals.sum().clamp(min=1e-30).item()
        ),
        "M_top1": float(M[0].item()),
        # Light arrays (top-200 only) for downstream plots / report.
        "lambda_top": eigvals[:200].cpu().numpy().tolist(),
        "M_top": M[:200].cpu().numpy().tolist(),
        "weights_off_top": weights_off[:200].cpu().numpy().tolist(),
        "weights_on_top": weights_on[:200].cpu().numpy().tolist(),
        "cumulative_off_top": cumulative_off[:200].cpu().numpy().tolist(),
        "cumulative_on_top": cumulative_on[:200].cpu().numpy().tolist(),
    }


def main():
    args = parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device
    print(f"device={device}  dtype={dtype}", flush=True)

    layers = sorted(set(list(args.filtered_layers) + list(args.control_layers)))
    print(f"layers: {layers}", flush=True)

    metrics = []
    for ell in layers:
        t0 = time.time()
        Cr = _load_cov(args.cov_dir, ell, "retain", dtype)
        Cf = _load_cov(args.cov_dir, ell, "forget", dtype)
        m = _layer_metrics(Cr, Cf, args.alpha, device)
        m["layer"] = int(ell)
        m["is_filtered"] = ell in set(args.filtered_layers)
        m["wall_time_s"] = round(time.time() - t0, 2)
        print(
            f"layer {ell:>2}  "
            f"R={m['R']:.4f}  R_direct={m['R_direct']:.4f}  "
            f"trace(Cr Cf)={m['trace_Cr_Cf']:.3e}  "
            f"eff_rank={m['effective_rank']:.1f}  "
            f"lambda1={m['lambda_top1']:.3e}  "
            f"M1={m['M_top1']:.3e}  "
            f"({m['wall_time_s']}s)",
            flush=True,
        )
        metrics.append(m)

    payload = {
        "config": {
            "cov_dir": args.cov_dir,
            "alpha": args.alpha,
            "filtered_layers": list(args.filtered_layers),
            "control_layers": list(args.control_layers),
            "dtype": args.dtype,
            "device": device,
            "per_component_layer": args.per_component_layer,
            "per_component_top_k": args.per_component_top_k,
        },
        "metrics": metrics,
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {metrics_path}", flush=True)

    # Per-component dump for the headline supporting figure.
    pc = next(m for m in metrics if m["layer"] == args.per_component_layer)
    pc_path = os.path.join(
        out_dir, f"per_component_layer{args.per_component_layer}.npz"
    )
    K = args.per_component_top_k
    np.savez(
        pc_path,
        eigvals=np.asarray(pc["lambda_top"][:K]),
        M=np.asarray(pc["M_top"][:K]),
        weights_off=np.asarray(pc["weights_off_top"][:K]),
        weights_on=np.asarray(pc["weights_on_top"][:K]),
    )
    print(f"Wrote {pc_path}", flush=True)

    # Brief markdown report.
    report = ["# Geometric validation v2 -- interference-reduction ratio", ""]
    report.append(f"alpha = {args.alpha}, dtype = {args.dtype}")
    report.append("")
    report.append("| layer | filtered? | R | trace(Cr Cf) | eff. rank | lambda_1 |")
    report.append("|------:|:---------:|--:|-------------:|----------:|---------:|")
    for m in metrics:
        flag = "filtered" if m["is_filtered"] else "control"
        report.append(
            f"| {m['layer']:>2} | {flag:>9} | {m['R']:.4f} | "
            f"{m['trace_Cr_Cf']:.3e} | {m['effective_rank']:.1f} | "
            f"{m['lambda_top1']:.3e} |"
        )
    report.append("")
    filt_R = [m["R"] for m in metrics if m["is_filtered"]]
    ctrl_R = [m["R"] for m in metrics if not m["is_filtered"]]
    report.append("## Aggregate")
    report.append(
        f"- filtered layers ({sorted(args.filtered_layers)}): "
        f"mean R = {np.mean(filt_R):.4f}, "
        f"median = {np.median(filt_R):.4f}, "
        f"max = {np.max(filt_R):.4f}"
    )
    report.append(
        f"- control layers ({sorted(args.control_layers)}): "
        f"mean R = {np.mean(ctrl_R):.4f}, "
        f"median = {np.median(ctrl_R):.4f}, "
        f"min = {np.min(ctrl_R):.4f}"
    )
    if filt_R and ctrl_R:
        report.append(
            f"- filtered/control ratio of mean R = "
            f"{np.mean(filt_R) / np.mean(ctrl_R):.3f} (smaller is better for our claim)"
        )
    report.append("")
    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
