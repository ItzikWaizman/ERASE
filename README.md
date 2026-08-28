# ERASE: Precise Machine Unlearning via Geometry-Aware Gradient Shaping and Adaptive Loss Control

Official code for the paper **"ERASE: Precise Machine Unlearning via Geometry-Aware Gradient Shaping and Adaptive Loss Control"** (Findings of EMNLP 2026).

ERASE is a geometry-aware unlearning framework that combines three complementary mechanisms:

1. a **covariance-based spectral filter** that attenuates gradient updates along principal directions of retained model knowledge;
2. **selective error-signal (VJP) construction** that restricts the forget gradient to subject-entity token positions (benchmark-provided entities on TOFU, automatic NER spans on document-level corpora such as MUSE-News);
3. an **adaptive Cross-Entropy Target (CET) loss with dynamic per-sample stopping** that steers each forget sample to a target cross-entropy and halts once it converges.

ERASE requires **no reference model, no retain-set loss, and no auxiliary task supervision** during training.

This repository is built on the excellent [OpenUnlearning](https://github.com/locuslab/open-unlearning) framework (MIT license); all evaluation follows its standardized pipeline. The ERASE-specific additions are the trainer (`src/trainer/unlearn/erase.py`), its configs, the span-construction and covariance scripts under `scripts/erase/`, and small patches to the shared data/train/eval plumbing (dynamic-stop sampling in `src/data/unlearn.py`, deterministic seeding in `src/trainer/utils.py` and `src/eval.py`).

## Installation

```bash
conda create -n erase python=3.11
conda activate erase
pip install -e .
pip install -r requirements.txt

# Only needed for MUSE-News (automatic NER span extraction):
python -m spacy download en_core_web_sm
```

Fine-tuned TOFU target and retain-oracle checkpoints for Llama-3.2-1B are downloaded from the [OpenUnlearning Hugging Face hub](https://huggingface.co/open-unlearning); see `setup_data.py` and `docs/`. For Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct we fine-tune the TOFU target/oracle models ourselves with the recipes in `configs/experiment/finetune/tofu/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/trainer/unlearn/erase.py` | The ERASE trainer (spectral filter, selective VJP, CET loss, dynamic stopping) |
| `configs/trainer/ERASE.yaml` | Hydra defaults for the trainer |
| `configs/winners/` | The exact winning configurations behind every table in the paper |
| `scripts/erase/` | Covariance precompute, NER span builder, run/sweep drivers |
| `scripts/analysis/` | Paper aggregates, tau probe, plotting |
| `scripts/geometric_validation/` | Geometric-validation figures (weight-drift projections) |
| `configs/experiment/` | Hydra experiment recipes (TOFU per-model, MUSE, finetuning) |

## Reproducing the paper results

**One-time precompute** (per model): the Wikipedia activation covariance used by the spectral filter:

```bash
python scripts/erase/compute_wikipedia_covariance.py --help
```

**TOFU Forget10, all four model families (Table 1):**

```bash
python scripts/erase/run_erase.py configs/winners/erase_tofu_forget10_llama1b.json
python scripts/erase/run_erase.py configs/winners/erase_tofu_forget10_llama2_7b.json
python scripts/erase/run_erase.py configs/winners/erase_tofu_forget10_qwen3b.json
python scripts/erase/run_erase.py configs/winners/erase_tofu_forget10_phi35.json
```

**TOFU forget05 (Table 3):**

```bash
python scripts/erase/run_erase.py configs/winners/erase_tofu_forget05_llama1b.json
```

**MUSE-News on Llama-2-7B (Table 2):** first build the NER span cache, then run the winner:

```bash
python scripts/erase/build_muse_spans.py --mode ner --output saves/precompute/muse_news_7b/spans_ner.json
python scripts/erase/run_muse_erase_sweep.py configs/winners/erase_muse_news_llama2_7b.json
```

**Multi-seed stability (Appendix):**

```bash
python scripts/erase/run_erase.py configs/winners/erase_seeds_llama2_7b.json
python scripts/erase/run_erase.py configs/winners/erase_seeds_qwen3b.json
python scripts/analysis/summarize_r2_seed_variance.py
```

**Oracle-free CET-target probe (Appendix):**

```bash
python scripts/analysis/tau_fictitious_probe.py
```

**Aggregate scores** (Memorization / Utility / Privacy / Aggregate as defined in the paper) are computed with `scripts/analysis/compute_paper_aggregates.py` on any evaluated checkpoint.

Baseline sweeps (GradAscent, GradDiff, NPO, SimNPO, AltPO, RMU, UNDIAL) use the OpenUnlearning implementations; the grids and winning configurations are listed in the paper's appendix, with driver scripts under `scripts/comparison_methods/`.

## Citation

```bibtex
@inproceedings{erase2026,
  title={{ERASE}: Precise Machine Unlearning via Geometry-Aware Gradient Shaping and Adaptive Loss Control},
  author={Waizman, Itzik and Katz, Shahar and Wolf, Lior},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026}
}
```

Please also cite [OpenUnlearning](https://github.com/locuslab/open-unlearning) (Dorna et al., 2025), whose framework this repository builds on, and the TOFU and MUSE benchmarks.

## License

MIT (inherited from OpenUnlearning; see `LICENSE`).
