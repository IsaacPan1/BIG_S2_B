import pandas as pd
import numpy as np

# Load files
preds = pd.read_csv("reports/predictions.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")

print("Predictions shape:", preds.shape)
print("Predictions columns:", list(preds.columns))
print("Sample submission shape:", sample_sub.shape)
print("Sample submission columns:", list(sample_sub.columns))

# Validation checks
assert preds.shape[0] == sample_sub.shape[0], "Row count mismatch: {} vs {}".format(preds.shape[0], sample_sub.shape[0])
assert list(preds.columns) == list(sample_sub.columns), "Column mismatch: {} vs {}".format(list(preds.columns), list(sample_sub.columns))
assert preds["weekly_sales"].isna().sum() == 0, "NaN predictions found"
assert (preds["weekly_sales"] >= 0).all(), "Negative predictions found"

print("All validation checks passed!")
print("Prediction range: {:.2f} - {:.2f}".format(preds["weekly_sales"].min(), preds["weekly_sales"].max()))
print("Prediction mean: {:.2f}".format(preds["weekly_sales"].mean()))

# Write submission.csv at repo root
preds.to_csv("submission.csv", index=False)
print("Written submission.csv")

# Verify it was written correctly
verify = pd.read_csv("submission.csv")
assert verify.shape == preds.shape
assert verify["weekly_sales"].isna().sum() == 0
print("submission.csv verified:", verify.shape)
print(verify.head())
