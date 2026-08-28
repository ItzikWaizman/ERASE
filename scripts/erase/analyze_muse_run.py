"""
Post-hoc training+eval monitor for an ERASE-on-MUSE run.

Reads `<run_dir>/trainer_state.json` (HF Trainer log history; survives the
checkpoint deletion done by run_muse_sweep.py) and the MUSE eval summary
(`MUSE_SUMMARY.json` / `MUSE_EVAL.json`, if present) and emits:

  - <run_dir>/train_curve.png  : loss / forget-CE / grad-norm / dynamic-stop /
                                 LR / VJP-coverage curves vs epoch.
  - <run_dir>/train_stats.json : machine-readable summary + auto diagnostics
                                 for "did it converge to the target, and is the
                                 LR / #epochs / VJP mask behaving?".

Everything is judged against the *target tau band* you are steering toward
([tau-band, tau+band]); we deliberately do NOT reference any oracle CE, since
in a real deployment the target is a chosen direction, not a known quantity.

It also prints a short human-readable verdict to stdout so you can eyeball it
straight from the SLURM log. The whole thing is best-effort: missing
matplotlib / missing fields / missing eval never raise, so it is safe to call
automatically at the end of every sweep run.

Usage:
    python scripts/erase/analyze_muse_run.py --run_dir saves/unlearn/<task> \
        --target_tau 1.9 --band 0.4 [--epoch_cap 20 --n_chunks 407]
"""

import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="saves/unlearn/<task> containing trainer_state.json")
    p.add_argument("--target_tau", type=float, default=None,
                   help="CET target tau this run steered toward.")
    p.add_argument("--band", type=float, default=0.4,
                   help="band half-width around tau ([tau-band, tau+band]).")
    p.add_argument("--epoch_cap", type=float, default=20.0,
                   help="num_train_epochs; used to flag if the run hit the cap.")
    p.add_argument("--n_chunks", type=int, default=407,
                   help="forget-set size; used for the %done diagnostic.")
    return p.parse_args()


def _series(rows, key):
    """(epoch, value) pairs where both exist and are finite numbers."""
    xs, ys = [], []
    for r in rows:
        e, v = r.get("epoch"), r.get(key)
        if isinstance(e, (int, float)) and isinstance(v, (int, float)):
            xs.append(e)
            ys.append(v)
    return xs, ys


def _last(rows, key):
    for r in reversed(rows):
        v = r.get(key)
        if isinstance(v, (int, float)):
            return v
    return None


def _load_eval_summary(run_dir):
    """Return (summary_dict_or_None, source_path_or_None)."""
    for name in ("MUSE_SUMMARY.json", "MUSE_EVAL.json"):
        p = run_dir / name
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8")), str(p)
            except Exception:  # noqa: BLE001
                return None, str(p)
    return None, None


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    state_path = run_dir / "trainer_state.json"
    if not state_path.is_file():
        print(f"[analyze] no trainer_state.json at {state_path}; skipping.")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    hist = state.get("log_history", [])
    train_rows = [h for h in hist if "loss" in h]  # train logs (not eval rows)
    if not train_rows:
        print(f"[analyze] no training rows in {state_path}; skipping.")
        return

    # ---- pull series ----
    ep_loss, loss = _series(train_rows, "loss")
    ep_ce, ce = _series(train_rows, "forget_ce_raw")
    ep_gn, gnorm = _series(train_rows, "grad_norm")
    ep_done, ndone = _series(train_rows, "dynstop_n_done_total")
    ep_active, nactive = _series(train_rows, "dynstop_n_active")
    ep_maxce, maxce = _series(train_rows, "dynstop_max_per_sample_ce")
    ep_lr, lr = _series(train_rows, "learning_rate")
    ep_cov, cov = _series(train_rows, "vjp_span_coverage")
    ep_sel, sel = _series(train_rows, "vjp_selected_pct")
    # "why aren't weights moving?" diagnostics
    ep_wd, wd = _series(train_rows, "weight_rel_delta")
    ep_p10, p10 = _series(train_rows, "tok_ce_p10")
    ep_p50, p50 = _series(train_rows, "tok_ce_p50")
    ep_p90, p90 = _series(train_rows, "tok_ce_p90")
    ep_fb, fbelow = _series(train_rows, "tok_frac_below_band")
    ep_fi, fin = _series(train_rows, "tok_frac_in_band")
    ep_fa, fabove = _series(train_rows, "tok_frac_above_band")
    ep_sl, seqlen = _series(train_rows, "mean_seq_len")

    max_epoch = max([e for e in (ep_loss or [0])])
    total_steps = state.get("global_step", len(train_rows))
    final_ce = ce[-1] if ce else _last(train_rows, "forget_ce_raw")
    final_done = ndone[-1] if ndone else _last(train_rows, "dynstop_n_done_total")
    final_cov = cov[-1] if cov else None
    final_wd = wd[-1] if wd else _last(train_rows, "weight_rel_delta")
    final_seqlen = seqlen[-1] if seqlen else _last(train_rows, "mean_seq_len")
    final_fbelow = fbelow[-1] if fbelow else _last(train_rows, "tok_frac_below_band")
    final_fin = fin[-1] if fin else _last(train_rows, "tok_frac_in_band")
    final_fabove = fabove[-1] if fabove else _last(train_rows, "tok_frac_above_band")
    lr_start = lr[0] if lr else None
    lr_end = lr[-1] if lr else None
    has_nan = any((isinstance(v, float) and v != v) for v in loss + ce)  # NaN check

    tau = args.target_tau
    lo = (tau - args.band) if tau is not None else None
    hi = (tau + args.band) if tau is not None else None

    # ---- diagnostics ----
    diags = []
    hit_cap = max_epoch >= (args.epoch_cap - 0.5)
    pct_done = (100.0 * final_done / args.n_chunks) if (final_done is not None) else None

    if has_nan:
        diags.append("NaN/Inf appeared in loss or forget CE -> DIVERGED "
                     "(lower LR / check grad norm).")

    if hit_cap and pct_done is not None and pct_done < 80.0:
        diags.append(
            f"Hit the {int(args.epoch_cap)}-epoch cap with only {pct_done:.0f}% of "
            f"chunks 'done' -> UNDER-CONVERGED: raise num_train_epochs and/or LR.")
    elif hit_cap:
        diags.append(
            f"Hit the {int(args.epoch_cap)}-epoch cap (dynamic-stop did not end the "
            f"run early). Consider a higher LR or more epochs if CE has not settled.")
    else:
        diags.append(
            f"Stopped early at epoch {max_epoch:.1f} via dynamic-stop "
            f"(the CET target band was reached before the cap).")

    if final_ce is not None and tau is not None:
        if lo <= final_ce <= hi:
            diags.append(
                f"Final forget CE {final_ce:.2f} is inside the target band "
                f"[{lo:.2f}, {hi:.2f}] (on-target).")
        elif final_ce < lo:
            diags.append(
                f"Final forget CE {final_ce:.2f} is BELOW the target band "
                f"[{lo:.2f}, {hi:.2f}] -> did not reach tau: train longer / raise LR.")
        else:
            diags.append(
                f"Final forget CE {final_ce:.2f} is ABOVE the target band "
                f"[{lo:.2f}, {hi:.2f}] -> overshoot: lower LR (dynamic-stop should "
                f"have caught it).")

    # crude LR-too-low signal: CE barely moved over the first ~quarter of training
    if len(ce) >= 8:
        q = max(1, len(ce) // 4)
        early_delta = ce[q] - ce[0]
        if abs(early_delta) < 0.05:
            diags.append(
                f"Forget CE moved only {early_delta:+.3f} over the first quarter "
                f"-> LR may be too low (slow start).")

    # Are the weights actually moving? (the core "nothing happens" check)
    if final_wd is not None:
        if final_wd < 1e-4:
            diags.append(
                f"Trainable weights moved only {final_wd:.2e} (rel) from init -> "
                f"update is being CRUSHED: raise LR, lower alpha (weaker spectral "
                f"filter), or widen target_layers/train_scope.")
        elif final_wd < 1e-3:
            diags.append(
                f"Trainable weights moved {final_wd:.2e} (rel) from init -> small; "
                f"likely under-powered LR/alpha for these long chunks.")
        else:
            diags.append(
                f"Trainable weights moved {final_wd:.2e} (rel) from init (non-trivial).")

    # Per-token band occupancy: is it under-push, or up/down tokens fighting?
    if final_fbelow is not None and final_fabove is not None:
        diags.append(
            f"Answer-token CE band occupancy: below={100*final_fbelow:.0f}%  "
            f"in={100*(final_fin or 0):.0f}%  above={100*final_fabove:.0f}% "
            f"(loss steers the per-SAMPLE mean, so within a sample tokens share "
            f"one sign; cross-sample/over-vs-under cancellation matters only if "
            f"'above' is also large).")
        if final_fbelow > 0.6 and final_fabove < 0.1:
            diags.append(
                "Most answer tokens are still BELOW the band with almost none above "
                "-> not gradient cancellation; simply not enough push (LR/alpha/epochs).")

    # Long-sequence gradient dilution (the TOFU->MUSE difference).
    if final_seqlen is not None:
        diags.append(
            f"Mean supervised length T ~= {final_seqlen:.0f} tokens. The per-sample "
            f"MEAN objective dilutes each token's gradient by ~1/T, so the per-token "
            f"push here is ~{final_seqlen/30.0:.0f}x weaker than a ~30-tok TOFU answer "
            f"at equal LR.")

    # VJP mask health (ner/qa only; -1/None for raw)
    if final_cov is not None:
        if final_cov < 0.5:
            diags.append(
                f"VJP span coverage {100*final_cov:.0f}% of rows -> most chunks fell "
                f"back to the whole-chunk mask; the entity mask is barely active "
                f"(check the span cache / tokenizer).")
        else:
            diags.append(f"VJP span coverage {100*final_cov:.0f}% of rows (mask active).")

    # ---- eval summary ----
    eval_summary, eval_src = _load_eval_summary(run_dir)
    eval_metrics = None
    if eval_summary is None:
        diags.append(
            "No MUSE eval summary found in run_dir -> eval may have failed or "
            "wrote elsewhere (check paths.output_dir / the SLURM log for a "
            "traceback or OOM; PrivLeak needs the retrain MUSE_EVAL.json).")
    else:
        # SUMMARY is {metric: agg_value}; EVAL is {metric: {agg_value:..}}.
        eval_metrics = {}
        for k, v in eval_summary.items():
            if isinstance(v, dict) and "agg_value" in v:
                eval_metrics[k] = v["agg_value"]
            elif isinstance(v, (int, float)):
                eval_metrics[k] = v
        diags.append(f"MUSE eval present ({Path(eval_src).name}): "
                     f"{', '.join(eval_metrics) if eval_metrics else 'no scalar metrics'}.")

    stats = {
        "run_dir": str(run_dir),
        "task": run_dir.name,
        "total_steps": total_steps,
        "max_epoch": round(max_epoch, 3),
        "epoch_cap": args.epoch_cap,
        "hit_epoch_cap": bool(hit_cap),
        "diverged_nan": bool(has_nan),
        "target_tau": tau,
        "target_band": ([round(lo, 3), round(hi, 3)] if tau is not None else None),
        "final_forget_ce_raw": final_ce,
        "in_target_band": (bool(lo <= final_ce <= hi)
                           if (final_ce is not None and tau is not None) else None),
        "final_n_done_total": final_done,
        "n_chunks": args.n_chunks,
        "pct_chunks_done": (round(pct_done, 1) if pct_done is not None else None),
        "final_vjp_span_coverage": (round(final_cov, 3) if final_cov is not None else None),
        "final_weight_rel_delta": (final_wd if final_wd is not None else None),
        "mean_seq_len": (round(final_seqlen, 1) if final_seqlen is not None else None),
        "tok_frac_below_band": (round(final_fbelow, 3) if final_fbelow is not None else None),
        "tok_frac_in_band": (round(final_fin, 3) if final_fin is not None else None),
        "tok_frac_above_band": (round(final_fabove, 3) if final_fabove is not None else None),
        "grad_norm_max": (round(max(gnorm), 4) if gnorm else None),
        "grad_norm_last": (round(gnorm[-1], 4) if gnorm else None),
        "lr_start": lr_start,
        "lr_end": lr_end,
        "eval_metrics": eval_metrics,
        "diagnostics": diags,
    }
    (run_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[analyze] {run_dir.name}")
    print(f"  steps={total_steps}  epochs={max_epoch:.1f}/{int(args.epoch_cap)}"
          f"  final_CE={final_ce}  done={final_done}/{args.n_chunks}"
          f"  cov={final_cov}")
    if eval_metrics:
        print(f"  eval: {eval_metrics}")
    for d in diags:
        print(f"  - {d}")

    # ---- figure (best-effort) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle(run_dir.name, fontsize=11)

        if ep_loss:
            ax[0, 0].plot(ep_loss, loss, lw=1.2)
        ax[0, 0].set_title("total training loss")
        ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel("loss")
        ax[0, 0].grid(alpha=0.3)

        if ep_ce:
            ax[0, 1].plot(ep_ce, ce, color="tab:red", lw=1.2, label="mean forget CE")
        if ep_maxce:
            ax[0, 1].plot(ep_maxce, maxce, color="tab:orange", lw=0.9,
                          alpha=0.7, label="max per-sample CE")
        if tau is not None:
            ax[0, 1].axhspan(lo, hi, color="tab:green", alpha=0.12,
                             label=f"target band tau={tau}")
            ax[0, 1].axhline(tau, color="tab:green", ls="--", lw=1.0)
        ax[0, 1].set_title("forget cross-entropy")
        ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("CE (nats)")
        ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=7)

        if ep_gn:
            ax[0, 2].plot(ep_gn, gnorm, color="tab:brown", lw=1.0)
        ax[0, 2].set_title("grad norm (divergence check)")
        ax[0, 2].set_xlabel("epoch"); ax[0, 2].set_ylabel("||g||")
        ax[0, 2].grid(alpha=0.3)

        if ep_done:
            ax[1, 0].plot(ep_done, ndone, color="tab:green", lw=1.2,
                          label="cumulative done")
        if ep_active:
            ax[1, 0].plot(ep_active, nactive, color="tab:blue", lw=0.9,
                          alpha=0.7, label="active in batch")
        ax[1, 0].axhline(args.n_chunks, color="k", ls=":", lw=0.9,
                         label=f"n_chunks={args.n_chunks}")
        ax[1, 0].set_title("dynamic per-sample stopping")
        ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("# chunks")
        ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=7)

        if ep_lr:
            ax[1, 1].plot(ep_lr, lr, color="tab:purple", lw=1.2)
        ax[1, 1].set_title("learning rate (cosine)")
        ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("lr")
        ax[1, 1].grid(alpha=0.3)

        if ep_cov:
            ax[1, 2].plot(ep_cov, [100 * c for c in cov], color="tab:cyan",
                          lw=1.2, label="span coverage % rows")
        if ep_sel:
            ax[1, 2].plot(ep_sel, sel, color="tab:gray", lw=0.9, alpha=0.7,
                          label="positions kept %")
        ax[1, 2].set_title("VJP mask coverage (ner/qa)")
        ax[1, 2].set_xlabel("epoch"); ax[1, 2].set_ylabel("%")
        ax[1, 2].set_ylim(0, 105); ax[1, 2].grid(alpha=0.3)
        ax[1, 2].legend(fontsize=7)

        # --- row 3: "why aren't the weights moving?" debug ---
        if ep_p50:
            ax[2, 0].plot(ep_p50, p50, color="tab:red", lw=1.2, label="median tok CE")
        if ep_p10:
            ax[2, 0].plot(ep_p10, p10, color="tab:blue", lw=0.8, alpha=0.7, label="p10")
        if ep_p90:
            ax[2, 0].plot(ep_p90, p90, color="tab:orange", lw=0.8, alpha=0.7, label="p90")
        if tau is not None:
            ax[2, 0].axhspan(lo, hi, color="tab:green", alpha=0.12)
        ax[2, 0].set_title("per-TOKEN forget CE spread")
        ax[2, 0].set_xlabel("epoch"); ax[2, 0].set_ylabel("CE (nats)")
        ax[2, 0].grid(alpha=0.3); ax[2, 0].legend(fontsize=7)

        if ep_fb:
            ax[2, 1].plot(ep_fb, [100 * f for f in fbelow], color="tab:blue",
                          lw=1.0, label="% below band")
        if ep_fi:
            ax[2, 1].plot(ep_fi, [100 * f for f in fin], color="tab:green",
                          lw=1.0, label="% in band")
        if ep_fa:
            ax[2, 1].plot(ep_fa, [100 * f for f in fabove], color="tab:red",
                          lw=1.0, label="% above band")
        ax[2, 1].set_title("answer-token band occupancy")
        ax[2, 1].set_xlabel("epoch"); ax[2, 1].set_ylabel("%")
        ax[2, 1].set_ylim(0, 105); ax[2, 1].grid(alpha=0.3); ax[2, 1].legend(fontsize=7)

        if ep_wd:
            ax[2, 2].plot(ep_wd, wd, color="tab:brown", lw=1.2)
        ax[2, 2].set_title("trainable weight movement ||W-W0||/||W0||")
        ax[2, 2].set_xlabel("epoch"); ax[2, 2].set_ylabel("rel. delta")
        ax[2, 2].grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(run_dir / "train_curve.png", dpi=120)
        plt.close(fig)
        print(f"  -> wrote {run_dir/'train_curve.png'}")
    except Exception as e:  # noqa: BLE001 - monitoring must never crash a run
        print(f"  [analyze] plot skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
