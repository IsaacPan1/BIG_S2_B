#!/usr/bin/env python3
"""recent_vs_full_adv.py — does sliding-window CV actually buy anything?

The gross train-vs-val adversarial AUC is always high for a forecasting holdout,
because val is a narrow FUTURE window (seasonal phase + variance compression make
it separable regardless of drift). That number cannot decide expanding vs sliding.

The question sliding answers is narrower: is RECENT training data more val-like
than OLD training data? If yes, old data is less representative and discarding it
(sliding) helps. If recent and full train separate from val equally, the gap is
structural (val is just "the future"), sliding discards history for nothing, and
expanding is the better baseline.

This runs the adversarial classifier twice with TIME-INDEX + SEASONALITY features
EXCLUDED (so it measures covariate drift, not window membership):
  full   train  vs val
  recent train  vs val   (most recent N periods only)
and compares.

    python recent_vs_full_adv.py

Reads data/features_train.parquet + features_val.parquet and reports/profile.json
(for the rank time axis). Run AFTER the parquet text-leak fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# Excluded from the adversarial INPUT: these encode window-membership, not drift.
# (Same spirit as adv_auc_check.py's TIME_INDEX, plus seasonal-phase features.)
WINDOW_FEATURES = {
    "period_id_ord", "period_id_trend", "period_id_of_cycle", "horizon",
    "period_id_sin", "period_id_cos", "period_id_sin2", "period_id_cos2",
    "period_id_quarter", "period_id_month", "month_of_year",
    "quarter_of_year", "is_quarter_start",
}
NON_FEATURES = {"adversarial_weights"}

# How many of the most recent train periods count as "recent". Default ~= 2x the
# val width; overridable. Too small = noisy AUC; too large = not really "recent".
RECENT_PERIODS = 14


def rank_of(df: pd.DataFrame, profile: dict) -> pd.Series:
    """Map period_id -> integer rank via profile codebook; fall back to raw if numeric."""
    tcol = profile.get("time_col")
    info = profile.get("period_rank_info") or {}
    id_to_rank = info.get("id_to_rank")
    if id_to_rank:
        return df[tcol].map(id_to_rank)
    # no codebook: assume the time column is already orderable
    return pd.to_numeric(df[tcol], errors="coerce")


def numeric_feature_matrix(tr: pd.DataFrame, vl: pd.DataFrame) -> list[str]:
    common = [c for c in tr.columns if c in vl.columns]
    cols = [c for c in common
            if pd.api.types.is_numeric_dtype(tr[c])
            and c not in WINDOW_FEATURES
            and c not in NON_FEATURES]
    return cols


def adv_auc(Xtr: pd.DataFrame, Xvl: pd.DataFrame, seed: int = 42) -> float:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    try:
        from catboost import CatBoostClassifier
        def mk():
            return CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                                      loss_function="Logloss", verbose=False,
                                      allow_writing_files=False, random_seed=seed)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        def mk():
            return GradientBoostingClassifier(random_state=seed)
    X = pd.concat([Xtr, Xvl], ignore_index=True)
    y = np.concatenate([np.ones(len(Xtr), dtype=np.int8),
                        np.zeros(len(Xvl), dtype=np.int8)])
    X = X.fillna(X.median(numeric_only=True))
    if y.sum() < 20 or (y == 0).sum() < 20:
        return float("nan")
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tri, vai in skf.split(X, y):
        clf = mk()
        clf.fit(X.iloc[tri].values, y[tri])
        oof[vai] = clf.predict_proba(X.iloc[vai].values)[:, 1]
    return float(roc_auc_score(y, oof))


def main() -> None:
    profile = json.loads((REPORTS / "profile.json").read_text())
    tr = pd.read_parquet(DATA / "features_train.parquet")
    vl = pd.read_parquet(DATA / "features_val.parquet")

    cols = numeric_feature_matrix(tr, vl)
    print(f"adversarial feature cols (window/seasonal features EXCLUDED): {len(cols)}")
    if "state_doh_release" in tr.columns:
        print("[!] state_doh_release still in parquet — run the parquet text-leak fix first.")
    print("-" * 64)

    tr_rank = rank_of(tr, profile)
    rank_max = int(np.nanmax(tr_rank.values))
    cutoff = rank_max - RECENT_PERIODS + 1
    recent_mask = tr_rank >= cutoff
    print(f"train rank range: 0–{rank_max}   recent window: ranks {cutoff}–{rank_max} "
          f"({RECENT_PERIODS} periods, {int(recent_mask.sum())} rows)")
    print("-" * 64)

    auc_full = adv_auc(tr[cols], vl[cols])
    auc_recent = adv_auc(tr.loc[recent_mask, cols], vl[cols])

    print(f"full-train  vs val  AUC : {auc_full:.4f}")
    print(f"recent-train vs val AUC : {auc_recent:.4f}")
    print(f"delta (full - recent)   : {auc_full - auc_recent:+.4f}")
    print("-" * 64)

    drop = auc_full - auc_recent
    if np.isnan(auc_full) or np.isnan(auc_recent):
        print("VERDICT: insufficient rows to judge (recent window too small?). "
              "Increase RECENT_PERIODS.")
    elif drop >= 0.10:
        print(f"VERDICT: recent train is meaningfully MORE val-like (AUC drops "
              f"{drop:.3f} when restricted to recent periods). Old data IS less "
              f"representative — SLIDING is justified; discarding old history reduces "
              f"the train/val gap. Tune window_size around the recent span.")
    elif drop >= 0.03:
        print(f"VERDICT: recent train is SOMEWHAT more val-like (AUC drops {drop:.3f}). "
              f"Mild drift. Sliding may help marginally; not a strong mandate over "
              f"expanding. Decide on CV-MAE once the modeler exists.")
    else:
        print(f"VERDICT: recent train is NOT more val-like (AUC barely moves: "
              f"{drop:+.3f}). The train/val gap is STRUCTURAL (val is a future window, "
              f"not drifted covariates). SLIDING discards ~{100*(1-RECENT_PERIODS/(rank_max+1)):.0f}% "
              f"of history for no representativeness gain — EXPANDING is the better "
              f"baseline. Revisit the sliding decision.")


if __name__ == "__main__":
    main()