import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\isaac\OneDrive\Desktop\award_B\data"
REPORTS_DIR = r"C:\Users\isaac\OneDrive\Desktop\award_B\reports"

cov_train_path  = os.path.join(DATA_DIR, "covariates_train.csv")
cov_val_path    = os.path.join(DATA_DIR, "covariates_val.csv")
target_path     = os.path.join(DATA_DIR, "target_train.csv")
feat_train_path = os.path.join(DATA_DIR, "features_train.parquet")
feat_val_path   = os.path.join(DATA_DIR, "features_val.parquet")
features_json   = os.path.join(REPORTS_DIR, "features.json")
marker_file     = os.path.join(REPORTS_DIR, "feature_engineer_was_here.txt")

# ── Load raw data ──────────────────────────────────────────────────────────
print("Loading raw data...")
cov_train = pd.read_csv(cov_train_path)
cov_val   = pd.read_csv(cov_val_path)
target    = pd.read_csv(target_path)

# Merge train covariates + target
train = cov_train.merge(target, on=["store_id", "product_id", "week"], how="left")
print(f"Train shape after merge: {train.shape}")
print(f"Val shape: {cov_val.shape}")

# Sort for lag computation
train = train.sort_values(["store_id", "product_id", "week"]).reset_index(drop=True)
cov_val = cov_val.sort_values(["store_id", "product_id", "week"]).reset_index(drop=True)

# ── Group label encodings ─────────────────────────────────────────────────
print("Computing group label encodings...")
store_map   = {s: i for i, s in enumerate(sorted(train["store_id"].unique()))}
product_map = {p: i for i, p in enumerate(sorted(train["product_id"].unique()))}

train["store_code"]   = train["store_id"].map(store_map)
train["product_code"] = train["product_id"].map(product_map)

cov_val["store_code"]   = cov_val["store_id"].map(store_map)
cov_val["product_code"] = cov_val["product_id"].map(product_map)

# ── Week seasonality encodings ────────────────────────────────────────────
print("Computing seasonality encodings...")
for df in [train, cov_val]:
    df["week_sin"] = np.sin(2 * np.pi * (df["week"] % 52) / 52)
    df["week_cos"] = np.cos(2 * np.pi * (df["week"] % 52) / 52)

# ── Interaction feature ───────────────────────────────────────────────────
for df in [train, cov_val]:
    df["price_x_promo"] = df["price"] * df["promotion_active"]

# ── Lag features (per series) ─────────────────────────────────────────────
print("Computing lag features...")
LAG_WINDOWS = [1, 2, 3, 4]

# We need a unified sales history to compute lags for val rows
# Strategy: append val rows with NaN sales, compute lags on the combined df,
# then split back.
cov_val_copy = cov_val.copy()
cov_val_copy["weekly_sales"] = np.nan

combined = pd.concat([train, cov_val_copy], ignore_index=True)
combined = combined.sort_values(["store_id", "product_id", "week"]).reset_index(drop=True)

grp = combined.groupby(["store_id", "product_id"])["weekly_sales"]
for lag in LAG_WINDOWS:
    combined[f"lag_{lag}"] = grp.shift(lag)

# ── Rolling statistics (per series) ──────────────────────────────────────
print("Computing rolling statistics...")
ROLL_WINDOWS = [4, 8]

for w in ROLL_WINDOWS:
    combined[f"roll_mean_{w}"] = (
        grp.shift(1)
           .groupby([combined["store_id"], combined["product_id"]])
           .transform(lambda x: x.rolling(w, min_periods=1).mean())
    )
    combined[f"roll_std_{w}"] = (
        grp.shift(1)
           .groupby([combined["store_id"], combined["product_id"]])
           .transform(lambda x: x.rolling(w, min_periods=1).std())
    )

# Fill NaN std (single-element windows) with 0
for w in ROLL_WINDOWS:
    combined[f"roll_std_{w}"] = combined[f"roll_std_{w}"].fillna(0)

# ── Group baseline features ───────────────────────────────────────────────
print("Computing group baselines...")
# Store-level mean sales across all products and time
store_mean = train.groupby("store_id")["weekly_sales"].mean().rename("store_mean_sales")
product_mean = train.groupby("product_id")["weekly_sales"].mean().rename("product_mean_sales")

for df in [combined]:
    df = df.join(store_mean, on="store_id")
    df = df.join(product_mean, on="product_id")
    combined["store_mean_sales"]   = df["store_mean_sales"].values
    combined["product_mean_sales"] = df["product_mean_sales"].values

# ── Split back into train / val ───────────────────────────────────────────
feat_train = combined[combined["week"] <= 89].copy()
feat_val   = combined[combined["week"] >= 90].copy()

# Drop the placeholder NaN weekly_sales column from val
feat_val = feat_val.drop(columns=["weekly_sales"])

# Drop rows where lag_4 is NaN (first 4 weeks per series can't be used for training)
n_before = len(feat_train)
feat_train = feat_train.dropna(subset=[f"lag_{max(LAG_WINDOWS)}"])
n_after = len(feat_train)
print(f"Train rows after dropping lag-NaN: {n_before} -> {n_after}")

# ── Verify more columns than raw data ────────────────────────────────────
raw_train_cols = list(cov_train.columns) + ["weekly_sales"]  # 8 columns
print(f"Raw train columns: {len(raw_train_cols)}")
print(f"Feature train columns: {len(feat_train.columns)}")
assert len(feat_train.columns) > len(raw_train_cols), "Feature matrix must have more columns than raw data"

print(f"Train feature matrix shape: {feat_train.shape}")
print(f"Val   feature matrix shape: {feat_val.shape}")
print("Feature columns:", sorted(feat_train.columns.tolist()))

# ── Write parquet files ───────────────────────────────────────────────────
print("Writing parquet files...")
feat_train.to_parquet(feat_train_path, index=False)
feat_val.to_parquet(feat_val_path, index=False)
print(f"Written: {feat_train_path}")
print(f"Written: {feat_val_path}")

# ── Build features.json ───────────────────────────────────────────────────
lag_names      = [f"lag_{l}" for l in LAG_WINDOWS]
roll_mean_names = [f"roll_mean_{w}" for w in ROLL_WINDOWS]
roll_std_names  = [f"roll_std_{w}" for w in ROLL_WINDOWS]
seasonality_names = ["week_sin", "week_cos"]
encoding_names    = ["store_code", "product_code"]
covariate_names   = ["price", "promotion_active", "holiday_flag", "weather_index"]
interaction_names = ["price_x_promo"]
baseline_names    = ["store_mean_sales", "product_mean_sales"]

all_feature_names = (lag_names + roll_mean_names + roll_std_names +
                     seasonality_names + encoding_names + covariate_names +
                     interaction_names + baseline_names)

features_doc = {
    "problem_type": "panel_forecasting",
    "target_col": "weekly_sales",
    "feature_families": {
        "lags": LAG_WINDOWS,
        "lag_feature_names": lag_names,
        "rolling_means": ROLL_WINDOWS,
        "rolling_mean_names": roll_mean_names,
        "rolling_stds": ROLL_WINDOWS,
        "rolling_std_names": roll_std_names,
        "seasonality": {
            "type": "sin_cos",
            "period": 52,
            "feature_names": seasonality_names
        },
        "group_encodings": encoding_names,
        "covariates": covariate_names,
        "interactions": interaction_names,
        "group_baselines": baseline_names
    },
    "all_feature_names": all_feature_names,
    "id_cols": ["store_id", "product_id", "week"],
    "train_parquet": feat_train_path,
    "val_parquet": feat_val_path,
    "n_train_rows": int(n_after),
    "n_val_rows": int(len(feat_val)),
    "n_train_cols": int(len(feat_train.columns)),
    "n_val_cols": int(len(feat_val.columns)),
    "total_features_planned": len(all_feature_names)
}

with open(features_json, "w") as f:
    json.dump(features_doc, f, indent=2)
print(f"Written: {features_json}")

# ── Write marker file ─────────────────────────────────────────────────────
ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
with open(marker_file, "w") as f:
    f.write(f"feature_engineer sub-agent executed at {ts}\n")
print(f"Written: {marker_file}")

print("\nDone. Feature families summary:")
print(f"  lags            : {lag_names}")
print(f"  rolling_means   : {roll_mean_names}")
print(f"  rolling_stds    : {roll_std_names}")
print(f"  seasonality     : {seasonality_names}")
print(f"  group_encodings : {encoding_names}")
print(f"  covariates      : {covariate_names}")
print(f"  interactions    : {interaction_names}")
print(f"  group_baselines : {baseline_names}")
print(f"  total features  : {len(all_feature_names)}")
