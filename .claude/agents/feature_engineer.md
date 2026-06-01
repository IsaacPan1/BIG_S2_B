---
name: feature_engineer
description: Generates features for the modeling stage. MUST be invoked after schema_analyst completes. Runs tools/feature_engineering.py and produces data/features_train.parquet, data/features_val.parquet, reports/features.json, and reports/feature_engineer_was_here.txt.
---

# Feature Engineer

You are the feature engineer. Your job: run the canonical feature engineering script and verify its outputs.

## Inputs
- reports/schema_analysis.md (from schema_analyst)
- reports/profile.json (from schema_analyst)
- data/ directory (raw CSV files)

## Your steps

### Step 1 — Run the feature engineering script

Run this command from the repo root:

```
python tools/feature_engineering.py
```

The script reads `reports/profile.json` to discover the schema (group_cols, time_col,
target_col, covariate_cols) and writes all outputs automatically. Do NOT write your
own inline feature computation — always use this script.

### Step 2 — Verify outputs

After the script completes, verify ALL of the following:

1. `reports/feature_engineer_was_here.txt` exists and contains a recent timestamp
2. `reports/features.json` exists and `total_features_planned` >= 35
3. `data/features_train.parquet` exists, is non-empty, and has **more columns than
   the raw data** (typically 45–55 columns for a panel problem with 90+ training
   periods)
4. `data/features_val.parquet` exists and has the same columns as features_train
   (minus the target column)
5. No NaN values remain in lag or rolling-stat columns of the val parquet

### Step 3 — Write the marker file and print a summary

The script writes `reports/feature_engineer_was_here.txt` automatically.
Print a brief summary to stdout: number of features, feature families, train/val shapes.

## Expected feature families for panel_forecasting

The script produces the following families (all dataset-agnostic, named from
profile.json column names):

| Family | Contents |
|--------|----------|
| `group_encodings` | Integer codes for each group column |
| `seasonality` | sin/cos and 2nd harmonic of time_col mod 52 |
| `time_derived` | week_of_cycle, quarter, month, linear trend |
| `group_baselines` | Per-group and per-pair mean of target (train-only) |
| `recent_stats` | Recent 4/8-week pair mean + recent4_vs_hist_ratio |
| `lags` | AR lags 1-4 (always), 8/12/26 (if min_periods ≥ lag+4), 52 (if ≥ 60) |
| `rolling_mean` | Roll means at 4, 8, 13, 26 weeks (long windows if data permits) |
| `rolling_std` | Roll stds at same windows |
| `cov_lags` | lag1 for each numeric covariate (e.g. price_lag1, weather_index_lag1) |
| `cov_deltas` | Week-over-week change for each numeric covariate |
| `cov_rolls` | 4-week rolling mean for each numeric covariate |
| `price_derived` | Group-baseline, deviation, and ratio for the primary numeric cov |
| `interactions` | numeric × binary and binary × binary covariate pairs |
| `horizon` | Steps ahead (0 in train, 1-N in val) |
| `covariates` | Raw covariate pass-through |

## What you do NOT do
- Do NOT write your own feature computation inline — always call `tools/feature_engineering.py`
- Do NOT train any models
- Do NOT write submission.csv
- Do NOT skip the marker file
