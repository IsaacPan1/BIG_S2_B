import pandas as pd
import json
import os
import sys

REPO_ROOT = r"C:\Users\isaac\OneDrive\Desktop\award_B"
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

# ---- Load predictions ----
pred_path = os.path.join(REPORTS_DIR, "predictions.csv")
pred = pd.read_csv(pred_path)
print(f"Loaded predictions: {pred.shape}, columns: {list(pred.columns)}")

# ---- Load sample_submission ----
sample_path = os.path.join(REPO_ROOT, "data", "sample_submission.csv")
sample = pd.read_csv(sample_path)
print(f"Loaded sample_submission: {sample.shape}, columns: {list(sample.columns)}")

# ---- Determine target column ----
target_col = "weekly_sales"
id_cols = ["store_id", "product_id", "week"]

# ---- Build submission DataFrame ----
pred_renamed = pred[id_cols + ["predicted_target"]].copy()
pred_renamed = pred_renamed.rename(columns={"predicted_target": target_col})

submission = sample[id_cols].copy()
submission = submission.merge(pred_renamed, on=id_cols, how="left")
print(f"After merge: {submission.shape}, columns: {list(submission.columns)}")

# ---- Validation checks ----
warnings = []
checks_passed = True

expected_rows = len(sample)
actual_rows = len(submission)
if actual_rows != expected_rows:
    msg = f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
    print(f"ERROR: {msg}")
    checks_passed = False
    warnings.append(msg)
else:
    print(f"Row count OK: {actual_rows}")

required_cols = list(sample.columns)
for col in required_cols:
    if col not in submission.columns:
        msg = f"Missing required column: {col}"
        print(f"ERROR: {msg}")
        checks_passed = False
        warnings.append(msg)

n_nan = int(submission[target_col].isna().sum())
if n_nan > 0:
    msg = f"{n_nan} NaN values in target column {target_col}"
    print(f"WARNING: {msg}")
    warnings.append(msg)
    train_target = pd.read_csv(os.path.join(REPO_ROOT, "data", "target_train.csv"))
    fallback_mean = train_target[target_col].mean()
    submission[target_col] = submission[target_col].fillna(fallback_mean)
    warnings.append(f"Filled {n_nan} NaN values with training mean {fallback_mean:.4f}")
    print(f"Filled NaN with training mean: {fallback_mean:.4f}")
else:
    print(f"NaN check OK: 0 NaN values")

n_negative = int((submission[target_col] < 0).sum())
if n_negative > 0:
    msg = f"{n_negative} negative predictions clipped to 0"
    print(f"WARNING: {msg}")
    warnings.append(msg)
    submission[target_col] = submission[target_col].clip(lower=0)
else:
    print(f"Non-negative check OK: 0 negative values")

try:
    train_target = pd.read_csv(os.path.join(REPO_ROOT, "data", "target_train.csv"))
    train_max = train_target[target_col].max()
    train_min = train_target[target_col].min()
    pred_max = submission[target_col].max()
    pred_min = submission[target_col].min()
    if pred_max > train_max * 10:
        msg = f"Prediction max {pred_max:.2f} is >10x training max {train_max:.2f}"
        warnings.append(msg)
        print(f"WARNING: {msg}")
    if pred_min < train_min / 10 and train_min > 0:
        msg = f"Prediction min {pred_min:.2f} is <0.1x training min {train_min:.2f}"
        warnings.append(msg)
        print(f"WARNING: {msg}")
    print(f"Range check: train [{train_min:.2f}, {train_max:.2f}], pred [{pred_min:.2f}, {pred_max:.2f}]")
except Exception as e:
    warnings.append(f"Could not perform range check: {e}")

submission = submission[required_cols]
print(f"Final submission columns: {list(submission.columns)}")

sub_path = os.path.join(REPO_ROOT, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"Written submission.csv to {sub_path} ({len(submission)} rows)")

stats = {
    "min": float(submission[target_col].min()),
    "max": float(submission[target_col].max()),
    "mean": float(submission[target_col].mean()),
    "std": float(submission[target_col].std()),
    "n_nan": int(submission[target_col].isna().sum()),
    "n_negative": int((submission[target_col] < 0).sum())
}

summary = {
    "row_count": len(submission),
    "columns": list(submission.columns),
    "target_column": target_col,
    "prediction_stats": stats,
    "validation_checks_passed": checks_passed,
    "warnings": warnings
}

summary_path = os.path.join(REPORTS_DIR, "submission_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Written submission_summary.json")

marker_path = os.path.join(REPORTS_DIR, "submission_writer_was_here.txt")
with open(marker_path, "w") as f:
    f.write("submission_writer completed successfully\n")
    f.write(f"submission.csv written to: {sub_path}\n")
    f.write(f"row_count: {len(submission)}\n")
    f.write(f"target_column: {target_col}\n")
    f.write(f"validation_checks_passed: {checks_passed}\n")
print(f"Written submission_writer_was_here.txt")

print("\n=== DONE ===")
print(json.dumps(summary, indent=2))
