#!/usr/bin/env python3
"""recent_vs_full_distance.py — sliding vs expanding, without a saturating classifier.

The adversarial CLASSIFIER pins at AUC 1.0 on this data (val is a narrow future
window: seasonal phase + variance compression separate it perfectly regardless of
drift). A delta between two 1.0s is meaningless, so AUC cannot decide sliding vs
expanding here.

This instead measures, PER FEATURE, how far train is from val using a metric that
does not saturate, and asks: does that distance SHRINK when train is restricted to
recent periods? If yes -> old data is less representative -> sliding helps. If the
distance is unchanged -> the gap is structural (val is just "the future") ->
sliding discards history for nothing -> expanding is the better baseline.

Distance per feature = standardized mean shift |mean_train - mean_val| / std_pooled
                       (a.k.a. effect size), plus KS for shape. Neither saturates.

    python recent_vs_full_distance.py

Reads data/features_train.parquet + val + reports/profile.json. Run AFTER the
parquet text-leak fix (state_doh_release must be gone).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# Excluded from the distance scan: these encode window-membership, not covariate
# drift. Seasonal phase and time index differ between any past/future split by
# construction, so including them would just re-measure "val is the future".
WINDOW_FEATURES = {
    "period_id_ord", "period_id_trend", "period_id_of_cycle", "horizon",
    "period_id_sin", "period_id_cos", "period_id_sin2", "period_id_cos2",
    "period_id_quarter", "period_id_month", "month_of_year",
    "quarter_of_year", "is_quarter_start",
}
NON_FEATURES = {"adversarial_weights"}
RECENT_PERIODS = 14


def rank_of(df: pd.DataFrame, profile: dict) -> pd.Series:
    tcol = profile.get("time_col")
    info = profile.get("period_rank_info") or {}
    id_to_rank = info.get("id_to_rank")
    if id_to_rank:
        return df[tcol].map(id_to_rank)
    return pd.to_numeric(df[tcol], errors="coerce")


def std_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    """|mean(a) - mean(b)| / pooled_std. Effect size; 0 = identical, grows w/ shift."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    sp = np.sqrt((np.var(a) + np.var(b)) / 2.0)
    if sp == 0:
        return 0.0
    return abs(np.mean(a) - np.mean(b)) / sp


def main() -> None:
    profile = json.loads((REPORTS / "profile.json").read_text())
    tr = pd.read_parquet(DATA / "features_train.parquet")
    vl = pd.read_parquet(DATA / "features_val.parquet")

    if "state_doh_release" in tr.columns:
        print("[!] state_doh_release still in parquet — run the parquet fix first.\n")

    cols = [c for c in tr.columns if c in vl.columns
            and pd.api.types.is_numeric_dtype(tr[c])
            and c not in WINDOW_FEATURES and c not in NON_FEATURES]

    tr_rank = rank_of(tr, profile)
    rank_max = int(np.nanmax(tr_rank.values))
    cutoff = rank_max - RECENT_PERIODS + 1
    recent = tr_rank >= cutoff
    print(f"features scanned (window/seasonal excluded): {len(cols)}")
    print(f"train ranks 0–{rank_max} | recent = {cutoff}–{rank_max} "
          f"({int(recent.sum())} rows) | val = {len(vl)} rows")
    print("=" * 72)

    rows = []
    for c in cols:
        d_full = std_mean_shift(tr[c].values.astype(float), vl[c].values.astype(float))
        d_recent = std_mean_shift(tr.loc[recent, c].values.astype(float),
                                  vl[c].values.astype(float))
        if np.isnan(d_full) or np.isnan(d_recent):
            continue
        rows.append((c, d_full, d_recent, d_full - d_recent))

    df = pd.DataFrame(rows, columns=["feature", "dist_full", "dist_recent", "improvement"])
    df = df.sort_values("dist_full", ascending=False)

    print("Top 15 features by full-train distance to val "
          "(improvement = how much closer RECENT train is):")
    print(f"{'feature':<38}{'full':>8}{'recent':>9}{'improve':>10}")
    for _, r in df.head(15).iterrows():
        print(f"{r.feature:<38}{r.dist_full:>8.3f}{r.dist_recent:>9.3f}{r.improvement:>+10.3f}")
    print("=" * 72)

    mean_full = df["dist_full"].mean()
    mean_recent = df["dist_recent"].mean()
    mean_impr = df["improvement"].mean()
    frac_improved = float((df["improvement"] > 0.05).mean())
    print(f"mean distance  full-train -> val : {mean_full:.3f}")
    print(f"mean distance  recent-train -> val: {mean_recent:.3f}")
    print(f"mean improvement (full - recent) : {mean_impr:+.3f}")
    print(f"fraction of features meaningfully closer when recent: {frac_improved:.0%}")
    print("-" * 72)

    rel = mean_impr / mean_full if mean_full > 0 else 0.0
    if rel >= 0.25 and frac_improved >= 0.40:
        print(f"VERDICT: recent train is substantially closer to val ({rel:.0%} mean "
              f"distance reduction, {frac_improved:.0%} of features improve). Old data "
              f"IS less representative — SLIDING is justified. Tune window_size near "
              f"the recent span.")
    elif rel >= 0.10:
        print(f"VERDICT: recent train is modestly closer ({rel:.0%} reduction). Mild "
              f"drift. Sliding may help marginally; decide on CV-MAE once the modeler "
              f"runs. Not a strong mandate.")
    else:
        print(f"VERDICT: recent train is NOT closer to val ({rel:+.0%} change, only "
              f"{frac_improved:.0%} of features improve). The gap is STRUCTURAL — val "
              f"is a narrow future window, not drifted covariates. SLIDING discards "
              f"~{100*(1-RECENT_PERIODS/(rank_max+1)):.0f}% of history for no gain — "
              f"EXPANDING is the better baseline. Flip cv_plan back to expanding "
              f"before training.")


if __name__ == "__main__":
    main()