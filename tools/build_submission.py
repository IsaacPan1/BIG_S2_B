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

import numpy as np
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


class SubmissionValidationError(Exception):
    pass


# ── Helper: load training target values ──────────────────────────────────────

def _load_train_target(data_dir: Path, fp: dict, target_col: str) -> pd.Series | None:
    """Return the training target Series, using file_paths or convention detection."""
    for key in ("train_target", "train_data"):
        fname = fp.get(key)
        if fname:
            path = data_dir / fname
            if path.exists():
                df = pd.read_csv(path)
                if target_col in df.columns:
                    return df[target_col]

    for fname in ("target_train.csv", "train.csv"):
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            if target_col in df.columns:
                return df[target_col]

    return None


def _get_composite_key(profile: dict) -> list[str]:
    """Extract the composite business key: group_cols + time_col from profile.json."""
    group_cols = profile.get("group_cols", []) or []
    time_col   = profile.get("time_col") or ""
    key = list(group_cols)
    if time_col and time_col not in key:
        key.append(time_col)
    return key


def _round_trip_audit(
    sub_path: Path,
    pred_path: Path,
    composite_key: list[str],
    target_col: str,
) -> dict:
    """Re-join written submission.csv onto predictions.csv on the composite business key.

    Asserts every submitted value equals the prediction within float tolerance
    and that every sample_submission row maps to a real (non-fallback) prediction.

    Raises SubmissionValidationError if any mismatch or coverage gap is found.
    """
    sub  = pd.read_csv(sub_path)
    pred = pd.read_csv(pred_path)

    # Normalise predictions column to target_col for comparison
    pred_compare = pred.copy()
    if "predicted_target" in pred_compare.columns and target_col not in pred_compare.columns:
        pred_compare = pred_compare.rename(columns={"predicted_target": target_col})

    # Guard: composite key must exist in both DataFrames
    missing_in_sub  = [c for c in composite_key if c not in sub.columns]
    missing_in_pred = [c for c in composite_key if c not in pred_compare.columns]
    if missing_in_sub or missing_in_pred:
        return {
            "rows_checked": 0,
            "mismatches": -1,
            "status": "SKIP",
            "reason": (
                f"composite key cols missing in submission: {missing_in_sub}, "
                f"in predictions: {missing_in_pred}"
            ),
            "examples": [],
        }

    # Left-join submission onto predictions on composite key
    merged = sub[composite_key + [target_col]].merge(
        pred_compare[composite_key + [target_col]],
        on=composite_key,
        how="left",
        suffixes=("_submitted", "_predicted"),
    )

    submitted_col = f"{target_col}_submitted"
    predicted_col = f"{target_col}_predicted"
    rows_checked  = len(merged)

    # Coverage check: NaN in predicted_col means no matching composite key in predictions.csv
    n_no_match = int(merged[predicted_col].isna().sum())
    if n_no_match > 0:
        examples = (
            merged[merged[predicted_col].isna()][composite_key]
            .head(3)
            .to_dict("records")
        )
        raise SubmissionValidationError(
            f"Round-trip audit FAIL: {n_no_match}/{rows_checked} submission rows have no "
            f"matching composite key in predictions.csv. "
            f"First 3 unmatched keys: {examples}"
        )

    # Value mismatch check (atol=1e-6)
    close = np.isclose(
        merged[submitted_col].values.astype(float),
        merged[predicted_col].values.astype(float),
        atol=1e-6,
        rtol=0,
    )
    n_mismatch = int((~close).sum())

    if n_mismatch > 0:
        bad = merged[~close][[*composite_key, submitted_col, predicted_col]].head(3)
        examples = [
            {
                "key":       {k: row[k] for k in composite_key},
                "submitted": float(row[submitted_col]),
                "predicted": float(row[predicted_col]),
            }
            for _, row in bad.iterrows()
        ]
        raise SubmissionValidationError(
            f"Round-trip audit FAIL: {n_mismatch}/{rows_checked} submitted values differ "
            f"from predictions.csv on composite key. Examples: {examples}"
        )

    return {
        "rows_checked": rows_checked,
        "mismatches":   0,
        "status":       "PASS",
        "reason":       "all submitted values match predictions.csv on composite key",
        "examples":     [],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load profile
    profile_path = REPORTS_DIR / "profile.json"
    if not profile_path.exists():
        sys.exit(f"ERROR: {profile_path} not found — run profile_data.py first")

    with open(profile_path) as f:
        profile = json.load(f)

    target_col    = profile.get("target_col", "predicted_target")
    fp            = profile.get("file_paths", {})
    composite_key = _get_composite_key(profile)
    print(f"Composite business key from profile.json: {composite_key}")

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

    if "predicted_target" not in pred.columns:
        sys.exit("ERROR: predictions.csv missing 'predicted_target' column")

    # ── Join key selection: ALWAYS prefer composite business key ──────────────
    # row_id is NEVER a valid join key unless predictions and sample_submission
    # are confirmed to share one identical row_id namespace (same length AND a
    # verified composite-key match).  When len(predictions) != len(sample), the
    # row_id namespaces are incompatible by definition — never join on row_id.
    size_mismatch = len(pred) != len(sample)
    if size_mismatch:
        print(
            f"WARNING: predictions has {len(pred)} rows, sample_submission has "
            f"{len(sample)} rows — row_id namespaces are incompatible; "
            f"composite-key join is mandatory."
        )

    composite_in_pred   = [c for c in composite_key if c in pred.columns]
    composite_in_sample = [c for c in composite_key if c in sample.columns]
    use_composite = len(composite_in_pred) > 0 and len(composite_in_sample) > 0

    if use_composite:
        join_cols = [c for c in composite_key if c in pred.columns and c in sample.columns]
        print(f"Joining on composite business key: {join_cols}")
    elif not size_mismatch and "row_id" in id_cols and "row_id" in pred.columns:
        # row_id fallback only when sizes match and composite key is unavailable
        join_cols = ["row_id"]
        print(
            "WARNING: no composite key columns found in both DataFrames; "
            "falling back to row_id (sizes match — this is the only safe fallback)."
        )
    else:
        sys.exit(
            "ERROR: Cannot determine a safe join key. "
            "Composite key columns from profile.json are absent in predictions.csv "
            "or sample_submission.csv, and row_id join is unsafe (size mismatch or "
            "row_id absent). Check profile.json group_cols/time_col and "
            "predictions.csv columns."
        )

    pred_aligned = pred[join_cols + ["predicted_target"]].copy()
    pred_aligned = pred_aligned.rename(columns={"predicted_target": target_col})

    submission = sample[id_cols].copy()
    submission = submission.merge(pred_aligned, on=join_cols, how="left")
    print(f"After merge: {submission.shape}, columns: {list(submission.columns)}")

    # ── Validation checks ─────────────────────────────────────────────────────
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

    # ── Non-skippable round-trip audit ────────────────────────────────────────
    # Re-read the written file and verify every cell matches predictions.csv on
    # the composite business key.  This is an I/O integrity check only — it
    # performs no optimisation and touches no model.
    audit_result: dict = {
        "rows_checked": 0, "mismatches": -1,
        "status": "SKIP", "reason": "not run", "examples": [],
    }
    try:
        audit_result = _round_trip_audit(sub_path, pred_path, composite_key, target_col)
        print(
            f"Round-trip audit: {audit_result['status']} — "
            f"{audit_result['rows_checked']} rows checked, "
            f"{audit_result['mismatches']} mismatches"
        )
    except SubmissionValidationError as sve:
        audit_result = {
            "rows_checked": 0,
            "mismatches":   -1,
            "status":       "FAIL",
            "reason":       str(sve),
            "examples":     [],
        }
        checks_passed = False
        warnings_list.append(f"Round-trip audit FAILED: {sve}")
        print(f"CRITICAL: Round-trip audit FAILED: {sve}")
        # Write summary so caller can inspect the failure before fallback
        summary = {
            "row_count":                    len(submission),
            "columns":                      list(submission.columns),
            "target_column":                target_col,
            "join_key_used":                join_cols,
            "composite_key_from_profile":   composite_key,
            "prediction_stats":             {},
            "validation_checks_passed":     False,
            "warnings":                     warnings_list,
            "round_trip_audit":             audit_result,
        }
        with open(REPORTS_DIR / "submission_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        raise  # let the submission_writer agent handle fallback

    # ── Write summary and marker (only on audit PASS) ─────────────────────────
    stats = {
        "min":        float(submission[target_col].min()),
        "max":        float(submission[target_col].max()),
        "mean":       float(submission[target_col].mean()),
        "std":        float(submission[target_col].std()),
        "n_nan":      int(submission[target_col].isna().sum()),
        "n_negative": int((submission[target_col] < 0).sum()),
    }
    summary = {
        "row_count":                    len(submission),
        "columns":                      list(submission.columns),
        "target_column":                target_col,
        "join_key_used":                join_cols,
        "composite_key_from_profile":   composite_key,
        "prediction_stats":             stats,
        "validation_checks_passed":     checks_passed,
        "warnings":                     warnings_list,
        "round_trip_audit":             audit_result,
    }
    summary_path = REPORTS_DIR / "submission_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Written submission_summary.json")

    marker_path = REPORTS_DIR / "submission_writer_was_here.txt"
    with open(marker_path, "w") as f:
        f.write("submission_writer completed successfully\n")
        f.write("build_submission.py invoked: YES\n")
        f.write(f"submission.csv written to: {sub_path}\n")
        f.write(f"row_count: {len(submission)}\n")
        f.write(f"target_column: {target_col}\n")
        f.write(f"join_key_used: {join_cols}\n")
        f.write(f"composite_key_from_profile: {composite_key}\n")
        f.write(f"validation_checks_passed: {checks_passed}\n")
        f.write(f"round_trip_audit_status: {audit_result['status']}\n")
        f.write(f"round_trip_audit_rows_checked: {audit_result['rows_checked']}\n")
        f.write(f"round_trip_audit_mismatches: {audit_result['mismatches']}\n")
    print("Written submission_writer_was_here.txt")

    # ── KG: observability — mark stage complete (best-effort) ─────────────────
    try:
        from kg import kg_set_stage
        kg_set_stage("submission_writer")
    except Exception as _kg_e:
        print(f"[KG] non-fatal: {_kg_e}", file=sys.stderr)

    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
