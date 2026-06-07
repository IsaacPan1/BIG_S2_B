#!/usr/bin/env python3
"""adv_auc_check.py — is the 0.9999 adversarial AUC a time-index artifact?

Runs the train-vs-val adversarial classifier three ways:
  (A) all numeric features            -> reproduce the ~0.9999 the profiler saw
  (B) drop time-index features        -> if AUC collapses, the 0.9999 was the
                                          time axis acting as a split label
  (C) per-feature AUC ranking         -> name the columns doing the separating

A row is "train" (1) or "val" (0); a classifier that can tell them apart near-
perfectly is keying on something that encodes the split. Real covariate shift
sits ~0.7-0.95; ~1.0 means a feature IS the train/val label (here: rank 0-76 vs
77-83). Run from repo root after a feature-engineering run.

    python adv_auc_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Time-index / ordinal features: monotonic in period, so train (early ranks) and
# val (late ranks) are separable by construction.
TIME_INDEX = {
    "period_id_ord", "period_id_trend", "horizon",
    "period_id_of_cycle",  # ramps within cycle; include to be safe
}

# Direct label leak: adversarial_weights exists only for train rows (val carries
# the default), so it encodes the train/val split itself. It is a sample weight,
# never a feature, and must be excluded from BOTH the adversarial check and the
# model matrix.
LABEL_LEAK = {"adversarial_weights"}

# Free-text / high-cardinality string columns. Raw text is near-unique per row,
# so a classifier memorizes it as the split label (perfect separator), AND it
# upcasts the model matrix to object dtype (the silent CatBoost hang). These must
# never enter X as raw values; any signal must be extracted into derived numerics.
TEXT_LEAK = {"state_doh_release"}


def load_xy() -> tuple[pd.DataFrame, np.ndarray]:
    tr = pd.read_parquet(DATA / "features_train.parquet")
    vl = pd.read_parquet(DATA / "features_val.parquet")
    common = [c for c in tr.columns if c in vl.columns]
    # Report non-numeric columns instead of silently dropping them — a text column
    # hidden here is exactly what masqueraded as "shift".
    non_numeric = [c for c in common if not pd.api.types.is_numeric_dtype(tr[c])]
    if non_numeric:
        print(f"[non-numeric columns excluded from numeric matrix]: {non_numeric}")
    num = [c for c in common if pd.api.types.is_numeric_dtype(tr[c])]
    X = pd.concat([tr[num], vl[num]], ignore_index=True)
    y = np.concatenate([np.ones(len(tr), dtype=np.int8),
                        np.zeros(len(vl), dtype=np.int8)])
    # impute so a NaN pattern can't itself leak the split
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def adv_auc(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> float:
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
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tri, vai in skf.split(X, y):
        clf = mk()
        clf.fit(X.iloc[tri].values, y[tri])
        oof[vai] = clf.predict_proba(X.iloc[vai].values)[:, 1]
    return float(roc_auc_score(y, oof))


def per_feature_auc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Univariate separability: |AUC-0.5|*2 per column. ~1.0 = perfect splitter."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for c in X.columns:
        v = X[c].values.astype(float)
        if np.nanstd(v) == 0:
            out[c] = 0.0
            continue
        try:
            a = roc_auc_score(y, v)
        except Exception:
            out[c] = 0.0
            continue
        out[c] = abs(a - 0.5) * 2.0
    return pd.Series(out).sort_values(ascending=False)


def main() -> None:
    X, y = load_xy()
    print(f"rows: train={int(y.sum())}  val={int((y == 0).sum())}  features={X.shape[1]}")
    print("-" * 64)

    present_time = [c for c in TIME_INDEX if c in X.columns]
    present_leak = [c for c in LABEL_LEAK if c in X.columns]
    present_text = [c for c in TEXT_LEAK if c in X.columns]
    print(f"time-index features present : {present_time}")
    print(f"label-leak features present : {present_leak}")
    print(f"text-leak features present  : {present_text}  (raw text never belongs in X)")
    print("-" * 64)

    auc_all = adv_auc(X, y)
    print(f"(A) AUC, all numeric features          : {auc_all:.4f}")

    X_no_time = X.drop(columns=present_time)
    auc_no_time = adv_auc(X_no_time, y)
    print(f"(B) AUC, time-index dropped            : {auc_no_time:.4f}")

    # The number that matters: every known leak channel removed (time index,
    # label-leak weight, and any text column that slipped into the numeric matrix).
    X_clean = X.drop(columns=present_time + present_leak + present_text)
    auc_clean = adv_auc(X_clean, y)
    print(f"(C) AUC, all known leaks dropped       : {auc_clean:.4f}   <-- TRUE SHIFT")
    print("-" * 64)

    pf = per_feature_auc(X, y)
    print("(D) top 12 single-feature separability (1.0 = perfect train/val splitter):")
    for c, s in pf.head(12).items():
        tag = ""
        if c in present_time:
            tag = "  <-- TIME INDEX (drop from adversarial set, keep for model)"
        elif c in present_leak:
            tag = "  <-- LABEL LEAK (drop from model matrix entirely)"
        print(f"     {s:5.3f}  {c}{tag}")
    print("-" * 64)

    # Verdict keyed on the CLEAN auc, not on whether dropping time-index alone moved it.
    if present_leak:
        print(f"[!] {present_leak} is in the feature matrix and separates train/val at "
              f"~{pf.get(present_leak[0], float('nan')):.3f}. This is a direct label "
              f"leak — it must go to fit(sample_weight=...), never into X.")
    if auc_clean >= 0.95:
        print(f"VERDICT: even with ALL leak channels removed, AUC is {auc_clean:.4f} — "
              f"SEVERE genuine covariate shift (see gtrends/naloxone block in D). Sliding "
              f"CV is justified; per-fold adversarial weighting may help, but build it "
              f"inside the CV loop, not as a global column.")
    elif auc_clean >= 0.80:
        print(f"VERDICT: true-shift AUC {auc_clean:.4f} — MODERATE real shift once leaks "
              f"are removed. The 1.0 was mostly over-determined separation from leak "
              f"channels. Sliding is defensible; weighting is optional, not urgent.")
    else:
        print(f"VERDICT: true-shift AUC {auc_clean:.4f} — shift is MILD once leaks are "
              f"removed. The original ~1.0 was almost entirely time-index + label-leak "
              f"artifacts, not distributional shift. Reconsider whether sliding/weighting "
              f"buys anything over expanding.")


if __name__ == "__main__":
    main()