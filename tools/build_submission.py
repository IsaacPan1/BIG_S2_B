"""Validate predictions and write submission.csv.

Reads reports/profile.json for the target column name, id columns, and file
paths.  Works for any dataset convention — combined train file (e.g.
energy_load's train.csv) or split covariate + target files (e.g. retail_sales).

Usage (from repo root):
    python tools/build_submission.py [--repo-root PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ── Repo root ────────────────────────────────────────────────────────────────
def _repo_root_from_args() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", default=None)
    args, _ = parser.parse_known_args()
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parent.parent

REPO_ROOT   = _repo_root_from_args()
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR    = REPO_ROOT / "data"


# ── Helper: load training target values ──────────────────────────────────────

def _load_train_target(data_dir: Path, fp: dict, target_col: str) -> pd.Series | None:
    """Return the training target Series, using file_paths or convention detection."""
    # Try file_paths first (most reliable)
    for key in ("train_target", "train_data"):
        fname = fp.get(key)
        if fname:
            path = data_dir / fname
            if path.exists():
                df = pd.read_csv(path)
                if target_col in df.columns:
                    return df[target_col]

    # Convention detection fallback (no file_paths in older profiles)
    for fname in ("target_train.csv", "train.csv"):
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            if target_col in df.columns:
                return df[target_col]

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load profile
    profile_path = REPORTS_DIR / "profile.json"
    if not profile_path.exists():
        sys.exit(f"ERROR: {profile_path} not found — run profile_data.py first")

    with open(profile_path) as f:
        profile = json.load(f)

    target_col = profile.get("target_col", "predicted_target")
    fp         = profile.get("file_paths", {})

    # Load predictions from modeler
    pred_path = REPORTS_DIR / "predictions.csv"
    if not pred_path.exists():
        sys.exit(f"ERROR: {pred_path} not found — run modeler first")
    pred = pd.read_csv(pred_path)
    print(f"Loaded predictions: {pred.shape}, columns: {list(pred.columns)}")

    # Load sample_submission
    sample_path = DATA_DIR / "sample_submission.csv"
    if not sample_path.exists():
        sys.exit(f"ERROR: {sample_path} not found")
    sample = pd.read_csv(sample_path)
    print(f"Loaded sample_submission: {sample.shape}, columns: {list(sample.columns)}")

    # Derive id_cols from sample_submission (everything except the target column)
    id_cols = [c for c in sample.columns if c != target_col]
    print(f"id_cols (from sample_submission): {id_cols}")
    print(f"target_col: {target_col}")

    # Build submission DataFrame
    if "predicted_target" not in pred.columns:
        sys.exit("ERROR: predictions.csv missing 'predicted_target' column")

    # Determine join keys: prefer composite key (excluding row_id) when the
    # prediction row_ids don't map to the same composite keys as the sample.
    # This happens when predictions cover more rows/periods than the submission
    # template (e.g. predictions has 2856 rows for 7 periods, sample has 918
    # rows for 6 periods — row_id=0 points to different rows in each file).
    join_cols = id_cols
    if "row_id" in id_cols and "row_id" in pred.columns:
        composite_id_cols = [c for c in id_cols if c != "row_id"]
        if composite_id_cols:
            # Check if row_id is a reliable join key by seeing whether
            # a few shared row_ids agree on their composite key values.
            shared_ids = list(set(sample["row_id"]) & set(pred["row_id"]))[:5]
            if shared_ids:
                sample_keys = (
                    sample[sample["row_id"].isin(shared_ids)]
                    .set_index("row_id")[composite_id_cols]
                    .sort_index()
                )
                pred_keys = (
                    pred[pred["row_id"].isin(shared_ids)]
                    .set_index("row_id")[composite_id_cols]
                    .sort_index()
                )
                # If any composite key differs, row_id is not a safe join key
                try:
                    aligned = sample_keys.eq(pred_keys)
                    if not aligned.all(axis=None):
                        join_cols = composite_id_cols
                        print(
                            f"row_id maps to different composite keys in predictions vs sample; "
                            f"joining on composite key: {join_cols}"
                        )
                except Exception:
                    join_cols = composite_id_cols
                    print(f"row_id alignment check failed; joining on composite key: {join_cols}")

    pred_aligned = pred[join_cols + ["predicted_target"]].copy()
    pred_aligned = pred_aligned.rename(columns={"predicted_target": target_col})

    submission = sample[id_cols].copy()
    submission = submission.merge(pred_aligned, on=join_cols, how="left")
    print(f"After merge: {submission.shape}, columns: {list(submission.columns)}")

    # Validation checks
    warnings_list: list[str] = []
    checks_passed = True

    if len(submission) != len(sample):
        msg = f"Row count mismatch: expected {len(sample)}, got {len(submission)}"
        print(f"ERROR: {msg}")
        checks_passed = False
        warnings_list.append(msg)
    else:
        print(f"Row count OK: {len(submission)}")

    for col in sample.columns:
        if col not in submission.columns:
            msg = f"Missing required column: {col}"
            print(f"ERROR: {msg}")
            checks_passed = False
            warnings_list.append(msg)

    n_nan = int(submission[target_col].isna().sum())
    if n_nan > 0:
        msg = f"{n_nan} NaN values in target column '{target_col}'"
        print(f"WARNING: {msg}")
        warnings_list.append(msg)
        train_vals = _load_train_target(DATA_DIR, fp, target_col)
        if train_vals is not None:
            fallback_mean = float(train_vals.mean())
            submission[target_col] = submission[target_col].fillna(fallback_mean)
            warnings_list.append(
                f"Filled {n_nan} NaN values with training mean {fallback_mean:.4f}"
            )
            print(f"Filled NaN with training mean: {fallback_mean:.4f}")
        else:
            warnings_list.append(f"Could not compute training mean — {n_nan} NaNs remain")
    else:
        print("NaN check OK: 0 NaN values")

    n_negative = int((submission[target_col] < 0).sum())
    if n_negative > 0:
        msg = f"{n_negative} negative predictions clipped to 0"
        print(f"WARNING: {msg}")
        warnings_list.append(msg)
        submission[target_col] = submission[target_col].clip(lower=0)
    else:
        print("Non-negative check OK: 0 negative values")

    try:
        train_vals = _load_train_target(DATA_DIR, fp, target_col)
        if train_vals is not None:
            train_max = float(train_vals.max())
            train_min = float(train_vals.min())
            pred_max  = float(submission[target_col].max())
            pred_min  = float(submission[target_col].min())
            if pred_max > train_max * 10:
                msg = f"Prediction max {pred_max:.2f} is >10x training max {train_max:.2f}"
                warnings_list.append(msg)
                print(f"WARNING: {msg}")
            if train_min > 0 and pred_min < train_min / 10:
                msg = f"Prediction min {pred_min:.2f} is <0.1x training min {train_min:.2f}"
                warnings_list.append(msg)
                print(f"WARNING: {msg}")
            print(
                f"Range check: train [{train_min:.2f}, {train_max:.2f}], "
                f"pred [{pred_min:.2f}, {pred_max:.2f}]"
            )
    except Exception as exc:
        warnings_list.append(f"Could not perform range check: {exc}")

    submission = submission[list(sample.columns)]
    print(f"Final submission columns: {list(submission.columns)}")

    sub_path = REPO_ROOT / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Written submission.csv to {sub_path} ({len(submission)} rows)")

    # Write submission_summary.json
    stats = {
        "min":        float(submission[target_col].min()),
        "max":        float(submission[target_col].max()),
        "mean":       float(submission[target_col].mean()),
        "std":        float(submission[target_col].std()),
        "n_nan":      int(submission[target_col].isna().sum()),
        "n_negative": int((submission[target_col] < 0).sum()),
    }
    summary = {
        "row_count":                len(submission),
        "columns":                  list(submission.columns),
        "target_column":            target_col,
        "prediction_stats":         stats,
        "validation_checks_passed": checks_passed,
        "warnings":                 warnings_list,
    }
    summary_path = REPORTS_DIR / "submission_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Written submission_summary.json")

    marker_path = REPORTS_DIR / "submission_writer_was_here.txt"
    with open(marker_path, "w") as f:
        f.write("submission_writer completed successfully\n")
        f.write(f"submission.csv written to: {sub_path}\n")
        f.write(f"row_count: {len(submission)}\n")
        f.write(f"target_column: {target_col}\n")
        f.write(f"validation_checks_passed: {checks_passed}\n")
    print("Written submission_writer_was_here.txt")

    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
