#!/usr/bin/env python3
"""drugs_diagnostic.py — is all_drugs' high MAE a magnitude effect or a real miss?

all_drugs OOF MAE ~4.0 dwarfs all_stimulants ~1.0. Two very different causes:
  (A) all_drugs simply has ~4x larger rates, so 4x MAE is EXPECTED and the model is
      doing fine proportionally  -> bottom-up composition is the only lever.
  (B) all_drugs is mispredicted DISPROPORTIONATELY (worse than its size explains)
      -> a targeted fix (transform, bias, outlier jurisdiction) exists before
      reaching for composition.

This computes, per scored category:
  - raw MAE
  - mean true rate (magnitude)
  - normalized MAE = MAE / mean_rate  (apples-to-apples across categories)
  - mean signed error (bias: are we systematically over/under-predicting?)
  - which jurisdictions drive all_drugs error

Reads reports/oof_predictions.csv + joins truth. Run from repo root.
    python drugs_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

SCORED = ["all_drugs", "all_opioids", "all_stimulants"]
KEY = ["jurisdiction", "overdose_category", "period_id"]
TARGET = "rate_per_10000_ed_visits"


def load_truth() -> pd.DataFrame:
    for fn in ["train/dose_sys_train.csv", "train/target_train.csv", "train/target.csv"]:
        p = DATA / fn
        if p.exists():
            t = pd.read_csv(p)
            if TARGET in t.columns and all(k in t.columns for k in KEY):
                return t[KEY + [TARGET]].rename(columns={TARGET: "y_true"})
    sys.exit("No train target file found with key+target.")


def main() -> None:
    oof = pd.read_csv(REPORTS / "oof_predictions.csv")
    pred_col = next((c for c in ["predicted_target", "y_pred"] if c in oof.columns), None)
    if pred_col is None:
        sys.exit(f"No prediction column in OOF: {list(oof.columns)}")
    oof = oof.merge(load_truth(), on=KEY, how="left")
    oof = oof[oof["y_true"].notna()].copy()
    oof["err"] = oof[pred_col] - oof["y_true"]
    oof["abs_err"] = oof["err"].abs()

    print(f"{'category':<16}{'MAE':>8}{'mean_rate':>11}{'norm_MAE':>10}"
          f"{'bias':>9}{'n':>7}")
    print("-" * 62)
    rows = []
    for cat in SCORED:
        sub = oof[oof["overdose_category"] == cat]
        if not len(sub):
            print(f"{cat:<16}  (no rows)")
            continue
        mae = sub["abs_err"].mean()
        mean_rate = sub["y_true"].mean()
        norm = mae / mean_rate if mean_rate else float("nan")
        bias = sub["err"].mean()   # + = over-predicting, - = under
        rows.append((cat, norm))
        print(f"{cat:<16}{mae:>8.3f}{mean_rate:>11.3f}{norm:>10.3f}"
              f"{bias:>+9.3f}{len(sub):>7}")
    print("-" * 62)
    print("norm_MAE = MAE / mean_rate. If all_drugs' norm_MAE is similar to the")
    print("others, its big MAE is just MAGNITUDE (-> bottom-up composition is the")
    print("lever). If all_drugs' norm_MAE is much HIGHER, it's a real disproportionate")
    print("miss (-> targeted fix first). bias != 0 means systematic over/under-predict")
    print("(a log1p back-transform artifact would show as consistent one-sided bias).")

    # Verdict
    norms = dict(rows)
    if "all_drugs" in norms and len(norms) > 1:
        others = np.mean([v for k, v in norms.items() if k != "all_drugs"])
        ad = norms["all_drugs"]
        print("-" * 62)
        if ad > 1.5 * others:
            print(f"VERDICT: all_drugs norm_MAE ({ad:.3f}) >> others ({others:.3f}) — "
                  f"DISPROPORTIONATE miss. Look for a targeted cause (bias/transform/"
                  f"outlier jurisdictions below) BEFORE composition.")
        elif ad < 1.2 * others:
            print(f"VERDICT: all_drugs norm_MAE ({ad:.3f}) ~ others ({others:.3f}) — "
                  f"its big MAE is MAGNITUDE, not mis-prediction. Model is fine "
                  f"proportionally; BOTTOM-UP COMPOSITION is the main lever.")
        else:
            print(f"VERDICT: all_drugs norm_MAE ({ad:.3f}) moderately above others "
                  f"({others:.3f}) — mix of magnitude and some real miss.")

    # Jurisdictions driving all_drugs error — MAGNITUDE vs genuine MISS
    ad_rows = oof[oof["overdose_category"] == "all_drugs"].copy()
    if len(ad_rows):
        print("\nTop 12 jurisdictions by all_drugs NORMALIZED error "
              "(norm = MAE/mean_rate — controls for state size):")
        g = ad_rows.groupby("jurisdiction").agg(
            mae=("abs_err", "mean"),
            mean_rate=("y_true", "mean"),
            bias=("err", "mean"),
            n=("abs_err", "count"),
        )
        g["norm_mae"] = g["mae"] / g["mean_rate"].replace(0, np.nan)
        cat_norm = ad_rows["abs_err"].mean() / ad_rows["y_true"].mean()
        print(f"   (category-wide all_drugs norm_MAE = {cat_norm:.3f} — compare each state to this)")
        print(f"   {'state':<6}{'MAE':>9}{'mean_rate':>11}{'norm_MAE':>10}{'bias':>9}{'n':>6}")
        gg = g.sort_values("mae", ascending=False).head(12)
        for juris, r in gg.iterrows():
            flag = ""
            if r["norm_mae"] > 1.5 * cat_norm:
                flag = "  <-- genuine MISS (norm >> avg)"
            elif r["norm_mae"] < 0.8 * cat_norm:
                flag = "  (just magnitude)"
            print(f"   {juris:<6}{r['mae']:>9.3f}{r['mean_rate']:>11.3f}"
                  f"{r['norm_mae']:>10.3f}{r['bias']:>+9.3f}{int(r['n']):>6}{flag}")

        # How concentrated is the error? What share of total all_drugs abs-error
        # comes from the worst 5 states?
        total_abs = ad_rows["abs_err"].sum()
        worst5 = g.sort_values("mae", ascending=False).head(5).index
        share = ad_rows[ad_rows["jurisdiction"].isin(worst5)]["abs_err"].sum() / total_abs
        n_states = ad_rows["jurisdiction"].nunique()
        print(f"\n   Worst 5 of {n_states} states carry {share:.0%} of total all_drugs error.")

        # Verdict for the jurisdiction question
        worst_norm = g.loc[gg.index[0], "norm_mae"]
        print("-" * 62)
        if worst_norm > 1.5 * cat_norm:
            print(f"VERDICT: top states have norm_MAE WELL above the category avg "
                  f"({worst_norm:.3f} vs {cat_norm:.3f}) — they are GENUINELY mispredicted, "
                  f"not just large. A targeted high-volatility-state approach may beat "
                  f"composition. Check the 'genuine MISS' flags above.")
        else:
            print(f"VERDICT: top states' norm_MAE (~{worst_norm:.3f}) is close to the "
                  f"category avg ({cat_norm:.3f}) — their big MAE is MAGNITUDE (huge rates), "
                  f"not disproportionate miss. No cheap per-state fix; BOTTOM-UP "
                  f"COMPOSITION remains the lever.")
        # consistent-sign bias among worst states = systematic, possibly fixable
        worst_bias = g.loc[gg.index[:5], "bias"]
        if (worst_bias > 0).all() or (worst_bias < 0).all():
            direction = "UNDER" if worst_bias.mean() < 0 else "OVER"
            print(f"NOTE: worst states all bias one direction ({direction}-predicting, "
                  f"mean {worst_bias.mean():+.2f}) — a systematic, potentially correctable skew.")


if __name__ == "__main__":
    main()