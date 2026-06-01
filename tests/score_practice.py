#!/usr/bin/env python3
"""Score a submission.csv against the hidden validation truth for a practice dataset.

Usage
-----
    python tests/score_practice.py --submission submission.csv --dataset retail_sales
    python tests/score_practice.py --submission submission.csv --dataset energy_load
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Key columns and target column for each dataset
DATASET_CONFIG: dict[str, dict] = {
    "retail_sales": {
        "truth_path": "practice_data/retail_sales/_truth/val_truth.csv",
        "key_cols": ["store_id", "product_id", "week"],
        "target_col": "weekly_sales",
    },
    "energy_load": {
        "truth_path": "practice_data/energy_load/_truth/val_truth.csv",
        "key_cols": ["region_id", "timestamp"],
        "target_col": "load_mw",
    },
    "medical_imaging": {
        "truth_path": "practice_data/medical_imaging/_truth/val_truth.csv",
        "key_cols": ["patient_id"],
        "target_col": "hospitalization_days",
    },
}


def score(submission_path: str, dataset: str) -> dict:
    cfg = DATASET_CONFIG[dataset]
    truth_path = Path(cfg["truth_path"])

    if not truth_path.exists():
        sys.exit(f"ERROR: truth file not found at {truth_path}")
    if not Path(submission_path).exists():
        sys.exit(f"ERROR: submission file not found at {submission_path}")

    truth = pd.read_csv(truth_path)
    submission = pd.read_csv(submission_path)

    key_cols = cfg["key_cols"]
    target_col = cfg["target_col"]

    # Check submission has required columns
    missing_keys = [c for c in key_cols if c not in submission.columns]
    if missing_keys:
        sys.exit(f"ERROR: submission is missing key columns: {missing_keys}")
    if target_col not in submission.columns:
        sys.exit(f"ERROR: submission is missing target column: '{target_col}'")

    # Merge
    merged = truth.merge(
        submission[key_cols + [target_col]],
        on=key_cols,
        suffixes=("_truth", "_pred"),
    )

    n_expected = len(truth)
    n_scored = len(merged)
    n_missing = n_expected - n_scored

    truth_vals = merged[f"{target_col}_truth"].to_numpy(dtype=float)
    pred_vals = merged[f"{target_col}_pred"].to_numpy(dtype=float)

    nan_preds = np.isnan(pred_vals).sum()
    if nan_preds:
        print(f"WARNING: {nan_preds} predictions are NaN — treated as 0 for scoring.")
        pred_vals = np.nan_to_num(pred_vals, nan=0.0)

    mae = float(np.mean(np.abs(truth_vals - pred_vals)))
    rmse = float(np.sqrt(np.mean((truth_vals - pred_vals) ** 2)))

    nonzero = truth_vals != 0
    mape = (
        float(np.mean(np.abs((truth_vals[nonzero] - pred_vals[nonzero]) / truth_vals[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )

    baseline_mae = float(np.mean(np.abs(truth_vals - truth_vals.mean())))
    skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else float("nan")

    return {
        "dataset": dataset,
        "n_expected": n_expected,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 2),
        "skill_vs_mean": round(skill, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a submission.csv against practice-data ground truth."
    )
    parser.add_argument("--submission", required=True, help="Path to your submission.csv")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIG.keys()),
        help="Which practice dataset to score against",
    )
    args = parser.parse_args()

    results = score(args.submission, args.dataset)

    width = 50
    print(f"\n{'=' * width}")
    print(f"  Scoring: {results['dataset']}")
    print(f"{'=' * width}")
    for k, v in results.items():
        if k == "dataset":
            continue
        print(f"  {k:<20s}: {v}")
    print(f"{'=' * width}\n")

    if results["n_missing"] > 0:
        print(f"WARNING: {results['n_missing']} rows were expected but not found in submission.\n")


if __name__ == "__main__":
    main()
