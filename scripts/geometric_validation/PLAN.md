# Geometric Validation Experiment v2 — agreed plan

**Trigger phrase from user**: "execute the geometric experiment"

When the user issues that phrase, run this plan as-is. Do not re-discuss.

## What we are testing

Does the spectral filter `P^ℓ = (I + α C_retain^ℓ)^-1` reduce the
expected damage that a forget gradient at layer ℓ would inflict on the
model's general (Wikipedia) knowledge?

Old experiment (TOFU forget vs. TOFU retain through `C_retain` from
Wikipedia) was the wrong frame: both TOFU subsets are statistically
similar w.r.t. the Wikipedia covariance. Section was deleted from
`main.tex`.

## The metric (per layer ℓ)

```
R^ℓ = trace(P^ℓ C_retain^ℓ P^ℓ C_forget^ℓ)
       /  trace(C_retain^ℓ C_forget^ℓ)
```

- `R^ℓ = 1` → P does nothing.
- `R^ℓ → 0` → P fully removes interference between forget gradient and
  Wikipedia knowledge.

In eigenbasis of `C_retain^ℓ = U Λ U^T`, with
`M_kk = E_f[(u_k^T x_f)²]` (per-component forget energy):

```
R^ℓ = Σ_k λ_k · M_kk · (1+αλ_k)^-2  /  Σ_k λ_k · M_kk
```

Both `λ_k` and `M_kk` are already in the existing JSON
(`saves/diagnostics/geometric/activation_projection/activation_metrics.json`)
under `eigvals_top_k` and `energy_forget`. **No re-run of the model
needed for the population-level metric.** Truncate sum to top-K (K=200)
eigenvectors; verify with full-rank computation if data is available.

## Layers to include

Same as before:
- Filtered: 0, 1, 2, 3, 4, 5 (Pure-α winner config)
- Controls (unfiltered in production): 8, 12, 15

## Sanity check (additional, agreed)

Collect a held-out Wikipedia activation sample `Y_w` (~12 K tokens, do
NOT reuse the C_retain construction split — pick a different Wiki
slice). At each layer compute the **empirical pair-wise interference
reduction**:

```
R_emp^ℓ = ‖(P X_f) Y_w^T‖_F² / ‖X_f Y_w^T‖_F²
```

Show `R_emp^ℓ ≈ R^ℓ`. This is purely a sanity check that the
covariance-based population metric matches what real Wikipedia tokens
would give.

Cost: ~25 minutes of CPU forward pass on Wikipedia + a tiny matmul per
layer.

## Figures to produce

1. **One bar per layer**: `1 − R^ℓ` = % interference reduction, for all
   9 layers. Yellow stripe on filtered (0–5). Headline figure.
2. **Per-component decomposition for layer 0**: bars of
   `λ_k · M_kk` (before P) vs. `λ_k · M_kk · (1+αλ_k)^-2` (after P)
   for top-50 eigenvectors. Annotation showing total reduction.
3. (Optional, if `R_emp^ℓ` matches `R^ℓ`): scatter of
   `R^ℓ vs. R_emp^ℓ` to demonstrate the population metric is well
   estimated by Wiki samples.

## File outputs

- `saves/diagnostics/geometric/interference_reduction/REPORT.md`
- `saves/diagnostics/geometric/interference_reduction/FIG_interference_reduction.png`
- `saves/diagnostics/geometric/interference_reduction/FIG_interference_per_component.png`
- `saves/diagnostics/geometric/interference_reduction/metrics.json`

## What goes in main.tex on success

Re-introduce a short subsection (~half page) in the same slot in
`sec:analysis`:

```
\subsection{Geometric Validation: Interference Reduction}
```

Lead with `R^ℓ` numbers + Fig. 1, end with the per-component panel as
the visual mechanism.

Do not lead with a long mathematical derivation; the derivation
belongs in a one-paragraph "Setup" only. Explain in plain English: P
shrinks the forget activation along the directions Wikipedia knowledge
uses, so the gradient update has less to bite into.
