"""
Feature engineering for panel forecasting — dataset-agnostic.

Reads reports/profile.json for schema and file_paths so it works with any
dataset convention (combined train.csv or split covariates_train + target_train).

Outputs:
  data/features_train.parquet
  data/features_val.parquet
  reports/features.json
  reports/feature_engineer_was_here.txt
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
DATA    = BASE / "data"
REPORTS = BASE / "reports"

# ── 0. Load profile ───────────────────────────────────────────────────────────
with open(REPORTS / "profile.json") as _f:
    _profile = json.load(_f)

_target_col = _profile["target_col"]
_group_cols = _profile["group_cols"]
_time_col   = _profile["time_col"]
_fp         = _profile.get("file_paths", {})
_combined   = _fp.get("train_target_in_combined_file", None)

def _load_csv(fname, fallbacks):
    for n in ([fname] if fname else []) + fallbacks:
        p = DATA / n
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Data file not found (tried {[fname]+fallbacks})")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
_train_data_file   = _fp.get("train_data")
_train_target_file = _fp.get("train_target")
_val_file          = _fp.get("val_features")

if _combined is True or _train_data_file == _train_target_file:
    cov_train = _load_csv(_train_data_file, ["train.csv"])
    tgt_train = cov_train
else:
    cov_train = _load_csv(_train_data_file,   ["covariates_train.csv"])
    tgt_train = _load_csv(_train_target_file, ["target_train.csv"])

cov_val = _load_csv(_val_file, ["val_features.csv", "covariates_val.csv"])
print(f"  cov_train {cov_train.shape}, cov_val {cov_val.shape}")

# ── 2. Merge train; mark val ──────────────────────────────────────────────────
if _combined is True or _train_data_file == _train_target_file:
    train = cov_train.copy()
else:
    _join = [c for c in _group_cols + [_time_col]
             if c in cov_train.columns and c in tgt_train.columns]
    train = cov_train.merge(tgt_train[_join + [_target_col]], on=_join, how="left")
val   = cov_val.copy()
val[_target_col] = np.nan

# Keep backward-compat aliases used by the rest of the script
_TARGET      = _target_col
_GROUP_COLS  = _group_cols
_TIME_COL    = _time_col

# Delegate to the canonical dataset-agnostic feature engineering script.
# This script's header already validated file loading (above); the canonical
# script re-reads profile.json and handles all panel + cross-sectional paths.
import subprocess, sys as _sys
print("Delegating to tools/feature_engineering.py ...")
_result = subprocess.run(
    [_sys.executable, str(BASE / "tools" / "feature_engineering.py")],
    cwd=str(BASE),
)
_sys.exit(_result.returncode)

# (dead code below this point — kept for reference only; sys.exit above fires first)

# ── 4. Trig seasonality (week mod 52) ─────────────────────────────────────────
print("Trig seasonality...")
full["week_sin"]  = np.sin(2 * np.pi * (full["week"] % 52) / 52)
full["week_cos"]  = np.cos(2 * np.pi * (full["week"] % 52) / 52)
full["week_sin2"] = np.sin(4 * np.pi * (full["week"] % 52) / 52)  # 2nd harmonic
full["week_cos2"] = np.cos(4 * np.pi * (full["week"] % 52) / 52)

# ── 5. Group baselines (computed strictly from train rows) ────────────────────
print("Computing group baselines...")
train_mask = full["week"] <= 89

sp_mean  = (full.loc[train_mask].groupby(["store_id", "product_id"])["weekly_sales"]
            .mean().rename("store_prod_mean"))
s_mean   = (full.loc[train_mask].groupby("store_id")["weekly_sales"]
            .mean().rename("store_mean"))
p_mean   = (full.loc[train_mask].groupby("product_id")["weekly_sales"]
            .mean().rename("product_mean"))

# Recent baselines (last 4 / 8 train weeks)
max_tw = int(full.loc[train_mask, "week"].max())
sp_r4  = (full.loc[train_mask & (full["week"] >= max_tw - 3)]
          .groupby(["store_id", "product_id"])["weekly_sales"]
          .mean().rename("store_prod_recent4_mean"))
sp_r8  = (full.loc[train_mask & (full["week"] >= max_tw - 7)]
          .groupby(["store_id", "product_id"])["weekly_sales"]
          .mean().rename("store_prod_recent8_mean"))

full = (full
        .merge(sp_mean, on=["store_id", "product_id"], how="left")
        .merge(s_mean,  on="store_id",                  how="left")
        .merge(p_mean,  on="product_id",                how="left")
        .merge(sp_r4,   on=["store_id", "product_id"], how="left")
        .merge(sp_r8,   on=["store_id", "product_id"], how="left"))

full["recent4_vs_hist_ratio"] = full["store_prod_recent4_mean"] / (full["store_prod_mean"] + 1e-9)

# ── 6. Price features ─────────────────────────────────────────────────────────
print("Price features...")
price_prod_mean = (full.loc[train_mask].groupby("product_id")["price"]
                   .mean().rename("price_prod_mean"))
full = full.merge(price_prod_mean, on="product_id", how="left")
full["price_ratio"]     = full["price"] / (full["price_prod_mean"] + 1e-9)
full["price_deviation"] = full["price"] - full["price_prod_mean"]

# ── 7. Interaction features ───────────────────────────────────────────────────
print("Interaction features...")
full["price_x_promo"]     = full["price_deviation"] * full["promotion_active"]
full["weather_x_holiday"] = full["weather_index"]   * full["holiday_flag"]
full["promo_x_holiday"]   = full["promotion_active"] * full["holiday_flag"]

# ── 8. Horizon indicator (0 in train, 1–10 in val) ────────────────────────────
full["horizon"] = np.where(full["week"] <= 89, 0, full["week"] - 89).astype(np.int8)

# ── 9. Lag features (using pandas groupby shift — fast) ───────────────────────
print("Lag features...")
LAG_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 26, 52]

grp_sales = full.groupby(["store_id", "product_id"])["weekly_sales"]
for lag in LAG_PERIODS:
    full[f"lag_{lag}"] = grp_sales.shift(lag)
    print(f"  lag_{lag}: {full[f'lag_{lag}'].isna().sum()} NaN")

# ── 10. Rolling stats (shift-then-roll via transform, fast) ───────────────────
print("Rolling statistics (transform)...")
WINDOWS = [4, 8, 13, 26]

for w in WINDOWS:
    full[f"roll_mean_{w}"] = (
        full.groupby(["store_id", "product_id"])["weekly_sales"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    )
    full[f"roll_std_{w}"] = (
        full.groupby(["store_id", "product_id"])["weekly_sales"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
    )
    print(f"  roll_mean_{w}: {full[f'roll_mean_{w}'].isna().sum()} NaN | "
          f"roll_std_{w}: {full[f'roll_std_{w}'].isna().sum()} NaN")

# ── 11. Week-level aggregate feature (train only) ─────────────────────────────
print("Week-level aggregates...")
week_mean = (full.loc[train_mask].groupby("week")["weekly_sales"]
             .mean().rename("week_mean_sales"))
full = full.merge(week_mean, on="week", how="left")

# ── 12. Fill remaining NaNs in val lag/rolling features ───────────────────────
# Val rows whose lags reach beyond week 89 produce NaN because those positions
# also have NaN weekly_sales in the stacked frame.  Fill with the most recent
# training sales value for each group (last-known value, much better than 0).
print("Filling val NaN lag/rolling with last training value per group...")
last_train = (full.loc[full["week"] == max_tw, ["store_id", "product_id", "weekly_sales"]]
              .rename(columns={"weekly_sales": "_last_train"})
              .set_index(["store_id", "product_id"]))
full = full.join(last_train, on=["store_id", "product_id"])

val_mask = full["week"] >= 90
lag_cols  = [f"lag_{lag}" for lag in LAG_PERIODS]
mean_cols = [f"roll_mean_{w}" for w in WINDOWS]
std_cols  = [f"roll_std_{w}"  for w in WINDOWS]

for col in lag_cols + mean_cols:
    nmask = val_mask & full[col].isna()
    if nmask.sum():
        full.loc[nmask, col] = full.loc[nmask, "_last_train"]
        print(f"  {col}: filled {int(nmask.sum())} NaN")

for col in std_cols:
    nmask = val_mask & full[col].isna()
    if nmask.sum():
        full.loc[nmask, col] = 0.0
        print(f"  {col}: filled {int(nmask.sum())} NaN with 0")

full.drop(columns=["_last_train"], inplace=True)

# ── 13. Split and save ────────────────────────────────────────────────────────
print("Splitting and saving...")
features_train = full[full["week"] <= 89].copy()
features_val   = full[full["week"] >= 90].copy()

assert features_val["weekly_sales"].isna().all(), "Val target should be NaN"
assert features_train["weekly_sales"].notna().all(), "Train target has unexpected NaN"

features_train.to_parquet(DATA / "features_train.parquet", index=False)
features_val.to_parquet(  DATA / "features_val.parquet",   index=False)
print(f"  features_train: {features_train.shape}")
print(f"  features_val:   {features_val.shape}")

# ── 14. features.json ─────────────────────────────────────────────────────────
raw_base  = {"store_id", "product_id", "week", "price", "promotion_active",
             "holiday_flag", "weather_index", "weekly_sales"}
all_cols  = set(features_train.columns)
engineered = sorted(all_cols - raw_base)

feature_families = {
    "lags":           [c for c in engineered if c.startswith("lag_")],
    "rolling_mean":   [c for c in engineered if c.startswith("roll_mean_")],
    "rolling_std":    [c for c in engineered if c.startswith("roll_std_")],
    "group_baselines":[c for c in engineered if any(k in c for k in
                        ["store_prod_mean", "store_mean", "product_mean",
                         "recent4_mean", "recent8_mean", "recent4_vs"])],
    "price_features": [c for c in engineered if "price" in c and "mean_sales" not in c],
    "seasonality":    [c for c in engineered if "sin" in c or "cos" in c],
    "interactions":   [c for c in engineered if "x_" in c],
    "week_aggregates":[c for c in engineered if c.startswith("week_") and
                        "sin" not in c and "cos" not in c],
    "encodings":      ["store_enc", "product_enc"],
    "horizon":        ["horizon"],
}

total_features = len(all_cols - {"store_id", "product_id", "week", "weekly_sales"})

features_meta = {
    "problem_type": "panel_forecasting",
    "target_col":   "weekly_sales",
    "group_cols":   ["store_id", "product_id"],
    "time_col":     "week",
    "feature_families": feature_families,
    "total_features_planned": len(engineered),
    "total_features_incl_raw": total_features,
    "raw_covariate_cols": ["price", "promotion_active", "holiday_flag", "weather_index"],
    "train_shape": list(features_train.shape),
    "val_shape":   list(features_val.shape),
    "lag_periods":     LAG_PERIODS,
    "rolling_windows": WINDOWS,
    "distribution_shift_warnings": [
        {
            "column": "weather_index",
            "ks_statistic": 0.6667,
            "p_value": 0.0,
            "note": ("Val weather is in a strongly negative regime relative to train. "
                     "Trig week encodings + raw week index provide alternative seasonal "
                     "signal that is not subject to this shift.")
        }
    ],
    "notes": [
        "Group baselines computed from train only (weeks 0-89), no leakage.",
        "Rolling stats use shift(1) before rolling to prevent target leakage.",
        "Val NaN lag/rolling features filled with last-known training sales per group.",
        "Recommended target transform: log1p(weekly_sales), invert with expm1.",
        "price_prod_mean computed per product from train rows only.",
    ]
}

with open(REPORTS / "features.json", "w") as fh:
    json.dump(features_meta, fh, indent=2)
print("  Saved reports/features.json")

# ── 15. Marker file ───────────────────────────────────────────────────────────
ts = datetime.utcnow().isoformat()
with open(REPORTS / "feature_engineer_was_here.txt", "w") as fh:
    fh.write(f"feature_engineer sub-agent executed at {ts}\n")
    fh.write(f"train shape: {list(features_train.shape)}\n")
    fh.write(f"val shape:   {list(features_val.shape)}\n")
    fh.write(f"engineered features ({len(engineered)}):\n")
    for col in engineered:
        fh.write(f"  {col}\n")
print("  Saved reports/feature_engineer_was_here.txt")

# ── 16. Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETE")
print("="*60)
print(f"  Train shape : {features_train.shape}")
print(f"  Val shape   : {features_val.shape}")
print(f"  Engineered features : {len(engineered)}")
print(f"\nFeature families:")
for family, cols in feature_families.items():
    print(f"  {family:22s}: {len(cols)} features — {cols[:3]}{'...' if len(cols)>3 else ''}")
