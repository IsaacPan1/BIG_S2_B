"""
Feature engineering for panel forecasting — dataset-agnostic.

Reads reports/profile.json for schema (group_cols, time_col, target_col,
covariate_cols) and file_paths (train_data, train_target, val_features).
Works with both the combined-file convention (e.g. energy_load: train.csv)
and the split-file convention (e.g. retail_sales: covariates_train.csv +
target_train.csv).

Writes:
  data/features_train.parquet
  data/features_val.parquet
  reports/features.json
  reports/feature_engineer_was_here.txt
"""
import pandas as pd
import numpy as np
import json
import datetime
import os
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
REPORT_DIR = REPO_ROOT / "reports"

print("=== Feature Engineer ===")

# ── Load profile ──────────────────────────────────────────────────────────────
profile    = json.load(open(REPORT_DIR / "profile.json"))
target_col = profile["target_col"]
group_cols = profile["group_cols"]
time_col   = profile["time_col"]
covar_cols = profile.get("covariate_cols", [])
fp         = profile.get("file_paths", {})

print(f"Target: {target_col}, Groups: {group_cols}, Time: {time_col}")

# ── Load train and val data ───────────────────────────────────────────────────
def _load_csv(fname: str | None, fallback_names: list[str]) -> pd.DataFrame:
    """Load a named file or fall back to convention-based detection."""
    if fname:
        path = DATA_DIR / fname
        if path.exists():
            return pd.read_csv(path)
    for name in fallback_names:
        path = DATA_DIR / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find data file (tried: {fname!r}, {fallback_names})"
    )

combined = fp.get("train_target_in_combined_file", True)
train_data_file   = fp.get("train_data")
train_target_file = fp.get("train_target")
val_features_file = fp.get("val_features")

if combined or train_data_file == train_target_file:
    # Combined convention: single file has features + target
    train = _load_csv(train_data_file, ["train.csv"])
    cov_train = train
    tgt_train = train
else:
    # Split convention: separate covariate and target files
    cov_train = _load_csv(train_data_file,   ["covariates_train.csv"])
    tgt_train = _load_csv(train_target_file, ["target_train.csv"])

cov_val = _load_csv(val_features_file, ["val_features.csv", "covariates_val.csv"])

# ── Merge train covariates + target (only needed for split convention) ────────
if combined or train_data_file == train_target_file:
    train = cov_train.copy()
else:
    join_cols = [c for c in group_cols + [time_col]
                 if c in cov_train.columns and c in tgt_train.columns]
    train = cov_train.merge(tgt_train[join_cols + [target_col]],
                            on=join_cols, how="left")

val = cov_val.copy()
print(f"Train shape: {train.shape}  Val shape: {val.shape}")

# ── Sort for lag computation ───────────────────────────────────────────────────
train = train.sort_values(group_cols + [time_col]).reset_index(drop=True)
val   = val.sort_values(group_cols + [time_col]).reset_index(drop=True)

# ── Combine train+val for shift-based features ────────────────────────────────
val[target_col] = np.nan
combined_df = pd.concat([train, val], ignore_index=True)
combined_df = combined_df.sort_values(group_cols + [time_col]).reset_index(drop=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def group_shift(df, col, periods):
    return df.groupby(group_cols)[col].shift(periods)

def group_rolling_mean(df, col, window):
    return df.groupby(group_cols)[col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

def group_rolling_std(df, col, window):
    return df.groupby(group_cols)[col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).std()
    )

# ── Lag features ──────────────────────────────────────────────────────────────
LAG_PERIODS = [1, 2, 3, 4, 8, 10, 12]
for lag in LAG_PERIODS:
    combined_df[f"lag_{lag}"] = group_shift(combined_df, target_col, lag)
print("Lag features done.")

# ── Rolling mean features ──────────────────────────────────────────────────────
ROLLING_WINDOWS = [4, 8, 12]
for w in ROLLING_WINDOWS:
    combined_df[f"roll_mean_{w}"] = group_rolling_mean(combined_df, target_col, w)
print("Rolling mean features done.")

# ── Rolling std features ───────────────────────────────────────────────────────
ROLLING_STD_WINDOWS = [4, 8]
for w in ROLLING_STD_WINDOWS:
    combined_df[f"roll_std_{w}"] = group_rolling_std(combined_df, target_col, w)
print("Rolling std features done.")

# ── Group baselines (computed from train only) ─────────────────────────────────
for g in group_cols:
    col = f"{g}_mean_{target_col}"
    combined_df[col] = combined_df[g].map(train.groupby(g)[target_col].mean())

if len(group_cols) >= 2:
    pair_key = "_".join(group_cols)
    sp_mean = train.groupby(group_cols)[target_col].mean().rename(f"{pair_key}_mean")
    sp_std  = train.groupby(group_cols)[target_col].std().rename(f"{pair_key}_std")
    combined_df = combined_df.merge(sp_mean, on=group_cols, how="left")
    combined_df = combined_df.merge(sp_std,  on=group_cols, how="left")
print("Group baselines done.")

# ── Seasonality features ───────────────────────────────────────────────────────
PERIOD = 52
combined_df[f"{time_col}_sin"]  = np.sin(2 * np.pi * combined_df[time_col] / PERIOD)
combined_df[f"{time_col}_cos"]  = np.cos(2 * np.pi * combined_df[time_col] / PERIOD)
combined_df[f"{time_col}_mod4"] = combined_df[time_col] % 4
combined_df[f"{time_col}_mod13"]= combined_df[time_col] % 13
print("Seasonality features done.")

# ── Group integer encodings ────────────────────────────────────────────────────
for g in group_cols:
    combined_df[f"{g}_enc"] = combined_df[g].astype("category").cat.codes
print("ID encodings done.")

# ── Covariate features: deviation/ratio for primary numeric covariate ──────────
numeric_covars = [c for c in covar_cols
                  if pd.api.types.is_numeric_dtype(train.get(c, pd.Series(dtype=float)))]
if numeric_covars and len(group_cols) >= 1:
    primary_cov = numeric_covars[0]
    ref_group   = group_cols[-1]
    grp_cov_mean = train.groupby(ref_group)[primary_cov].mean().rename(f"{ref_group}_mean_{primary_cov}")
    combined_df = combined_df.merge(grp_cov_mean, on=ref_group, how="left")
    combined_df[f"{primary_cov}_deviation"] = (
        combined_df[primary_cov] - combined_df[f"{ref_group}_mean_{primary_cov}"]
    )
print("Covariate features done.")

# ── Split back into train / val ────────────────────────────────────────────────
train_times = sorted(train[time_col].unique())
val_times   = sorted(val[time_col].unique())

features_train = combined_df[combined_df[time_col].isin(train_times)].copy()
features_val   = combined_df[combined_df[time_col].isin(val_times)].copy()

print(f"\nFeatures train shape: {features_train.shape}")
print(f"Features val shape:   {features_val.shape}")

# ── Write parquet ──────────────────────────────────────────────────────────────
features_train.to_parquet(DATA_DIR / "features_train.parquet", index=False)
features_val.to_parquet(  DATA_DIR / "features_val.parquet",   index=False)
print("\nParquet files written.")

# ── Write features.json ───────────────────────────────────────────────────────
id_and_target = set(group_cols + [time_col, target_col])
feature_cols  = [c for c in features_train.columns if c not in id_and_target]

features_json = {
    "problem_type":           profile.get("problem_type"),
    "target_col":             target_col,
    "group_cols":             group_cols,
    "time_col":               time_col,
    "feature_families": {
        "lags":          [f"lag_{k}" for k in LAG_PERIODS],
        "rolling_means": [f"roll_mean_{w}" for w in ROLLING_WINDOWS],
        "rolling_stds":  [f"roll_std_{w}"  for w in ROLLING_STD_WINDOWS],
        "group_baselines": (
            [f"{g}_mean_{target_col}" for g in group_cols]
            + ([f"{'_'.join(group_cols)}_mean", f"{'_'.join(group_cols)}_std"]
               if len(group_cols) >= 2 else [])
        ),
        "covariates":    covar_cols,
        "seasonality":   [f"{time_col}_sin", f"{time_col}_cos",
                          f"{time_col}_mod4", f"{time_col}_mod13"],
        "encodings":     [f"{g}_enc" for g in group_cols],
    },
    "feature_columns":        feature_cols,
    "total_features_planned": len(feature_cols),
    "train_shape":            list(features_train.shape),
    "val_shape":              list(features_val.shape),
}
with open(REPORT_DIR / "features.json", "w") as f:
    json.dump(features_json, f, indent=2)
print("reports/features.json written.")

# ── Marker file ───────────────────────────────────────────────────────────────
with open(REPORT_DIR / "feature_engineer_was_here.txt", "w") as f:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    f.write(f"feature_engineer sub-agent executed at {ts}\n")
print("reports/feature_engineer_was_here.txt written.")

print("\n=== Feature engineering complete ===")
print(f"Total features: {len(feature_cols)}")
