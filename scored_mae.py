#!/usr/bin/env python3
"""scored_mae.py — recompute MAE filtered to the SCORED overdose categories.

The competition scores only a subset of overdose_category values (you identified
drug / stimulant / opioid). Every MAE the modeler currently prints averages over
ALL categories, so it measures a different target than the leaderboard. This script
recomputes the key MAEs on the scored subset and prints them next to the all-category
numbers, so you can see how much dilution was happening.

EDIT SCORED_CATEGORIES below to the EXACT strings from your data once confirmed via:
    python -c "import pandas as pd; print(sorted(pd.read_csv('data/val/covariates.csv')['overdose_category'].unique()))"

Run from repo root after a modeler run (needs reports/oof_predictions.csv).
    python scored_mae.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

# ---- EDIT THIS to the exact overdose_category strings the competition scores ----
SCORED_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]
# ---------------------------------------------------------------------------------

GROUP_COL = "overdose_category"
CAT_COL_CANDIDATES = [GROUP_COL, "overdose_category", "category"]


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def find_cat_col(df: pd.DataFrame) -> str | None:
    for c in CAT_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def report(df: pd.DataFrame, true_col: str, pred_col: str, label: str) -> None:
    cat_col = find_cat_col(df)
    all_mae = mae(df[true_col], df[pred_col])
    print(f"\n{label}")
    print(f"  all-category MAE : {all_mae:.4f}   (n={len(df)})")
    if cat_col is None:
        print(f"  [!] no category column found in this file (cols: {list(df.columns)[:8]}...) "
              f"— cannot compute scored-only MAE here")
        return
    present = set(df[cat_col].unique())
    missing = [c for c in SCORED_CATEGORIES if c not in present]
    if missing:
        print(f"  [!] SCORED_CATEGORIES not all present in {cat_col}: missing {missing}")
        print(f"      categories present: {sorted(present)}")
        print(f"      -> fix SCORED_CATEGORIES to match exact labels before trusting the scored MAE")
    scored = df[df[cat_col].isin(SCORED_CATEGORIES)]
    if len(scored) == 0:
        print(f"  scored-category MAE: N/A (no rows match {SCORED_CATEGORIES})")
        return
    scored_mae = mae(scored[true_col], scored[pred_col])
    print(f"  scored-only  MAE : {scored_mae:.4f}   (n={len(scored)}, "
          f"{len(scored)/len(df)*100:.0f}% of rows)")
    # per scored category, so you see which one drives error
    print(f"  per scored category:")
    for c in SCORED_CATEGORIES:
        sub = scored[scored[cat_col] == c]
        if len(sub):
            print(f"      {c:<12} MAE={mae(sub[true_col], sub[pred_col]):.4f}  (n={len(sub)})")


def load_truth() -> pd.DataFrame:
    """Load ground-truth target keyed on (jurisdiction, overdose_category, period_id)."""
    target_col = "rate_per_10000_ed_visits"
    key = ["jurisdiction", "overdose_category", "period_id"]
    for fn in ["train/dose_sys_train.csv", "train/target_train.csv", "train/target.csv"]:
        p = DATA / fn
        if p.exists():
            t = pd.read_csv(p)
            if target_col in t.columns and all(k in t.columns for k in key):
                return t[key + [target_col]].rename(columns={target_col: "y_true"})
    sys.exit("Could not find a train target file with key+target columns; "
             "edit load_truth() with the right filename.")


def main() -> None:
    oof_path = REPORTS / "oof_predictions.csv"
    if not oof_path.exists():
        sys.exit(f"{oof_path} not found — run the modeler first")
    oof = pd.read_csv(oof_path)
    print("OOF columns:", list(oof.columns))

    pred_col = next((c for c in ["predicted_target", "y_pred", "prediction", "pred"]
                     if c in oof.columns), None)
    if pred_col is None:
        sys.exit(f"No prediction column in OOF (have: {list(oof.columns)}).")

    key = ["jurisdiction", "overdose_category", "period_id"]
    if "y_true" not in oof.columns:
        if not all(k in oof.columns for k in key):
            sys.exit(f"OOF lacks join keys {key}; have {list(oof.columns)}.")
        truth = load_truth()
        before = len(oof)
        oof = oof.merge(truth, on=key, how="left")
        matched = int(oof["y_true"].notna().sum())
        print(f"[join] matched {matched}/{before} OOF rows to ground truth")
        if matched == 0:
            sys.exit("Join produced zero matches — key mismatch between OOF and target file.")
        oof = oof[oof["y_true"].notna()]

    report(oof, "y_true", pred_col, "=== Nested-CV OOF (5 folds) ===")

    print("\n" + "=" * 60)
    print("Compare the two MAEs above. If scored-only differs materially from")
    print("all-category, your model selection has been optimizing the wrong target.")
    print("The scored-only number is the one that matches the leaderboard.")


if __name__ == "__main__":
    main()