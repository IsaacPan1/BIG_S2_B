"""
Feature engineering for panel forecasting.
Reads: data/covariates_train.csv, data/target_train.csv, data/covariates_val.csv, reports/profile.json
Writes: data/features_train.parquet, data/features_val.parquet, reports/features.json, reports/feature_engineer_was_here.txt
"""
import pandas as pd
import numpy as np
import json
import datetime
import os

print("=== Feature Engineer ===")

# ── Load data ────────────────────────────────────────────────────────────────
cov_train = pd.read_csv("data/covariates_train.csv")
tgt_train  = pd.read_csv("data/target_train.csv")
cov_val    = pd.read_csv("data/covariates_val.csv")
profile    = json.load(open("reports/profile.json"))

target_col  = profile["target_col"]       # weekly_sales
group_cols  = profile["group_cols"]       # [store_id, product_id]
time_col    = profile["time_col"]         # week
covar_cols  = profile["covariate_cols"]   # [price, promotion_active, holiday_flag, weather_index]

print(f"Target: {target_col}, Groups: {group_cols}, Time: {time_col}")

# ── Merge train covariates + target ──────────────────────────────────────────
train = cov_train.merge(tgt_train[group_cols + [time_col, target_col]],
                        on=group_cols + [time_col], how="left")
val   = cov_val.copy()

print(f"Train shape after merge: {train.shape}")
print(f"Val shape: {val.shape}")

# ── Sort for lag computation ──────────────────────────────────────────────────
train = train.sort_values(group_cols + [time_col]).reset_index(drop=True)
val   = val.sort_values(group_cols + [time_col]).reset_index(drop=True)

# ── Combine train+val for feature computation (val target = NaN) ──────────────
val[target_col] = np.nan
combined = pd.concat([train, val], ignore_index=True)
combined = combined.sort_values(group_cols + [time_col]).reset_index(drop=True)

# ── Helper: group-level shift ─────────────────────────────────────────────────
def group_shift(df, col, periods, group_cols):
    return df.groupby(group_cols)[col].shift(periods)

def group_rolling_mean(df, col, window, group_cols):
    return df.groupby(group_cols)[col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

def group_rolling_std(df, col, window, group_cols):
    return df.groupby(group_cols)[col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).std()
    )

# ── Lag features (weeks 1–4, 8, 10, 12 from current time) ────────────────────
LAG_PERIODS = [1, 2, 3, 4, 8, 10, 12]
for lag in LAG_PERIODS:
    combined[f"lag_{lag}"] = group_shift(combined, target_col, lag, group_cols)
print("Lag features done.")

# ── Rolling mean features ─────────────────────────────────────────────────────
ROLLING_WINDOWS = [4, 8, 12]
for w in ROLLING_WINDOWS:
    combined[f"roll_mean_{w}"] = group_rolling_mean(combined, target_col, w, group_cols)
print("Rolling mean features done.")

# ── Rolling std features ──────────────────────────────────────────────────────
ROLLING_STD_WINDOWS = [4, 8]
for w in ROLLING_STD_WINDOWS:
    combined[f"roll_std_{w}"] = group_rolling_std(combined, target_col, w, group_cols)
print("Rolling std features done.")

# ── Group baselines (store mean, product mean, store-product mean) ─────────────
# Compute from TRAIN only to avoid leakage
store_mean    = train.groupby("store_id")[target_col].mean().rename("store_mean_sales")
product_mean  = train.groupby("product_id")[target_col].mean().rename("product_mean_sales")
sp_mean       = train.groupby(["store_id","product_id"])[target_col].mean().rename("sp_mean_sales")
sp_std        = train.groupby(["store_id","product_id"])[target_col].std().rename("sp_std_sales")

combined = combined.merge(store_mean,   on="store_id",   how="left")
combined = combined.merge(product_mean, on="product_id", how="left")
combined = combined.merge(sp_mean,      on=["store_id","product_id"], how="left")
combined = combined.merge(sp_std,       on=["store_id","product_id"], how="left")
print("Group baselines done.")

# ── Seasonality features (from time column) ────────────────────────────────────
combined["week_sin"]  = np.sin(2 * np.pi * combined[time_col] / 52)
combined["week_cos"]  = np.cos(2 * np.pi * combined[time_col] / 52)
combined["week_mod4"] = combined[time_col] % 4   # quarterly cycle
combined["week_mod13"]= combined[time_col] % 13  # quarter of year
print("Seasonality features done.")

# ── Store / product integer encodings ────────────────────────────────────────
combined["store_id_enc"]   = combined["store_id"].str.extract(r'(\d+)').astype(int)
combined["product_id_enc"] = combined["product_id"].str.extract(r'(\d+)').astype(int)
print("ID encodings done.")

# ── Price deviation from product mean ────────────────────────────────────────
product_price_mean = train.groupby("product_id")["price"].mean().rename("product_mean_price")
combined = combined.merge(product_price_mean, on="product_id", how="left")
combined["price_deviation"] = (combined["price"] - combined["product_mean_price"]) / combined["product_mean_price"]
print("Price deviation feature done.")

# ── Split back into train / val ───────────────────────────────────────────────
train_times = sorted(train[time_col].unique())
val_times   = sorted(val[time_col].unique())

features_train = combined[combined[time_col].isin(train_times)].copy()
features_val   = combined[combined[time_col].isin(val_times)].copy()

# ── Drop target from val (it's NaN anyway, but be explicit) ──────────────────
# Keep target in train for the modeler

print(f"\nFeatures train shape: {features_train.shape}")
print(f"Features val shape:   {features_val.shape}")
print(f"Feature columns: {list(features_train.columns)}")

# ── Write parquet ─────────────────────────────────────────────────────────────
features_train.to_parquet("data/features_train.parquet", index=False)
features_val.to_parquet("data/features_val.parquet", index=False)
print("\nParquet files written.")

# ── Write features.json ───────────────────────────────────────────────────────
feature_cols = [c for c in features_train.columns
                if c not in group_cols + [time_col, target_col]]

features_json = {
    "problem_type": profile["problem_type"],
    "target_col": target_col,
    "group_cols": group_cols,
    "time_col": time_col,
    "feature_families": {
        "lags": LAG_PERIODS,
        "rolling_means": ROLLING_WINDOWS,
        "rolling_stds": ROLLING_STD_WINDOWS,
        "group_baselines": ["store_mean_sales", "product_mean_sales", "sp_mean_sales", "sp_std_sales"],
        "covariates": covar_cols,
        "seasonality": {
            "week_sin": "sin(2pi*week/52)",
            "week_cos": "cos(2pi*week/52)",
            "week_mod4": "week % 4",
            "week_mod13": "week % 13"
        },
        "id_encodings": ["store_id_enc", "product_id_enc"],
        "price_features": ["price_deviation", "product_mean_price"]
    },
    "feature_columns": feature_cols,
    "total_features_planned": len(feature_cols),
    "train_shape": list(features_train.shape),
    "val_shape": list(features_val.shape)
}
with open("reports/features.json", "w") as f:
    json.dump(features_json, f, indent=2)
print("reports/features.json written.")

# ── Write marker file ─────────────────────────────────────────────────────────
with open("reports/feature_engineer_was_here.txt", "w") as f:
    f.write(f"feature_engineer sub-agent executed at {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z\n")
print("reports/feature_engineer_was_here.txt written.")

print("\n=== Feature engineering complete ===")
print(f"Total features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")
