#!/usr/bin/env python3
"""transform_bias_diagnostic.py — is log1p causing high-rate under-prediction?

WV (true rate ~57) is under-predicted by ~14 (bias -13.8, ~88% of its MAE). The
hypothesis: log1p compresses the top of the distribution, so a small log-space
miss back-transforms into a large raw under-prediction at high magnitudes — and
regularization shrinks the extreme cell toward the mean on top of that.

This does NOT retrain. It uses the EXISTING OOF predictions and asks three things:
  1. Does bias grow with the true rate? (the log1p/shrinkage signature)
  2. Is the bias multiplicative (constant in log-space) rather than additive?
     If so, predictions are ~a constant FRACTION too low — classic log back-transform.
  3. What would a simple per-decile bias correction recover?

Reads reports/oof_predictions.csv + truth. Run from repo root.
    python transform_bias_diagnostic.py
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
    sys.exit("No train target file found.")


def main() -> None:
    oof = pd.read_csv(REPORTS / "oof_predictions.csv")
    pred_col = next((c for c in ["predicted_target", "y_pred"] if c in oof.columns), None)
    oof = oof.merge(load_truth(), on=KEY, how="left")
    oof = oof[oof["y_true"].notna()].copy()
    oof = oof[oof["overdose_category"].isin(SCORED)].copy()   # scored rows only

    yt = oof["y_true"].values
    yp = oof[pred_col].clip(lower=0).values
    oof["err"] = yp - yt                         # signed raw error (+over, -under)
    # log-space residual (the space the model actually optimized)
    oof["log_resid"] = np.log1p(yp) - np.log1p(yt)
    # multiplicative ratio: how the prediction compares as a fraction of truth
    oof["ratio"] = (yp + 1e-9) / (yt + 1e-9)

    print("=== Bias across the true-rate distribution (scored rows) ===")
    print("If bias gets more NEGATIVE as rate rises -> high rates systematically")
    print("under-predicted (the log1p/shrinkage signature).\n")
    oof["rate_decile"] = pd.qcut(oof["y_true"], 10, labels=False, duplicates="drop")
    print(f"{'decile':>7}{'rate_lo':>9}{'rate_hi':>9}{'mean_rate':>10}"
          f"{'raw_bias':>10}{'log_resid':>11}{'mean_ratio':>11}{'n':>6}")
    for d, sub in oof.groupby("rate_decile"):
        print(f"{int(d):>7}{sub['y_true'].min():>9.1f}{sub['y_true'].max():>9.1f}"
              f"{sub['y_true'].mean():>10.2f}{sub['err'].mean():>+10.3f}"
              f"{sub['log_resid'].mean():>+11.4f}{sub['ratio'].mean():>11.3f}{len(sub):>6}")

    # Diagnosis: is the bias ADDITIVE (constant raw offset) or MULTIPLICATIVE
    # (constant fraction)? log back-transform bias is multiplicative.
    top = oof[oof["rate_decile"] == oof["rate_decile"].max()]
    bot = oof[oof["rate_decile"] <= 1]
    print("-" * 64)
    print(f"Top decile  (rate~{top['y_true'].mean():.1f}): raw_bias={top['err'].mean():+.2f}  "
          f"log_resid={top['log_resid'].mean():+.4f}  ratio={top['ratio'].mean():.3f}")
    print(f"Bottom 2 deciles (rate~{bot['y_true'].mean():.1f}): raw_bias={bot['err'].mean():+.2f}  "
          f"log_resid={bot['log_resid'].mean():+.4f}  ratio={bot['ratio'].mean():.3f}")
    print()
    log_resid_flat = abs(top["log_resid"].mean() - bot["log_resid"].mean()) < 0.03
    raw_bias_grows = top["err"].mean() < bot["err"].mean() - 1.0
    if raw_bias_grows and log_resid_flat:
        print("VERDICT: raw under-prediction GROWS with rate while log-space residual is")
        print("roughly FLAT -> the bias is MULTIPLICATIVE: the model is ~equally accurate")
        print("in log space, but log1p back-transform + shrinkage turns that into a large")
        print("RAW under-prediction at high rates. FIX = address the transform (predict raw,")
        print("or correct the back-transform / Jensen bias), NOT add features.")
    elif raw_bias_grows:
        print("VERDICT: raw under-prediction GROWS with rate AND log-space residual also")
        print("grows -> high rates are genuinely harder/shrunk in log space too. Transform")
        print("change helps partially; magnitude up-weighting or per-group calibration also needed.")
    else:
        print("VERDICT: bias does NOT grow with rate -> log1p is not the high-rate culprit;")
        print("look elsewhere (per-group shrinkage, features).")

    # What a simple per-group (jurisdiction×category) bias correction would recover,
    # estimated honestly: leave-one-fold-out style using the 'fold' column if present.
    print("-" * 64)
    if "fold" in oof.columns:
        # For each row, bias-correct using the mean residual of its group computed
        # on OTHER folds (avoids using the row's own fold -> not leak-optimistic).
        oof["grp"] = oof["jurisdiction"] + "|" + oof["overdose_category"]
        corrected = oof[pred_col].clip(lower=0).copy().astype(float)
        for grp, gidx in oof.groupby("grp").groups.items():
            grows = oof.loc[gidx]
            for f in grows["fold"].unique():
                this = grows[grows["fold"] == f].index
                other = grows[grows["fold"] != f]
                if len(other) >= 2:
                    corr = other["err"].mean()       # mean over-pred on other folds
                    corrected.loc[this] = (oof.loc[this, pred_col].clip(lower=0) - corr)
        corrected = corrected.clip(lower=0)
        base_mae = oof["err"].abs().mean()
        corr_mae = (corrected.values - yt).__abs__().mean()
        print(f"Per-group out-of-fold bias correction (honest estimate):")
        print(f"   scored MAE now          : {base_mae:.4f}")
        print(f"   scored MAE bias-corrected: {corr_mae:.4f}   "
              f"({(base_mae-corr_mae)/base_mae*100:+.1f}%)")
        print("   (If this is a big drop, a per-group calibration step — fit per-fold —")
        print("    is a cheap, leak-safe lever. If small, the transform fix matters more.)")
    else:
        print("No 'fold' column in OOF — skipping the bias-correction estimate.")


if __name__ == "__main__":
    main()