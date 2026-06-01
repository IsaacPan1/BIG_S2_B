import pandas as pd, numpy as np, json, time, warnings, datetime
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

start_time = time.time()

print("=" * 60)
print("MODELER SUB-AGENT STARTING")
print(f"Start time: {datetime.datetime.now().isoformat()}")
print("=" * 60)

# ------------------------------------------------------------------ #
# Step 1 - Load feature metadata
# ------------------------------------------------------------------ #
with open("reports/features.json") as f:
    feat_meta = json.load(f)

target_col = feat_meta["target_col"]
group_cols = feat_meta["group_cols"]
time_col   = feat_meta["time_col"]
problem_type = feat_meta.get("problem_type", "panel_forecasting")

print(f"Problem type: {problem_type}")
print(f"Target: {target_col}")
print(f"Groups: {group_cols}")
print(f"Time: {time_col}")

# ------------------------------------------------------------------ #
# Step 2 - Load parquet files
# ------------------------------------------------------------------ #
print("\n--- Loading parquet files ---")
train_df = pd.read_parquet("data/features_train.parquet")
val_df   = pd.read_parquet("data/features_val.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")

# Build feature list (exclude group/time/target and helper columns)
exclude = set(group_cols + [time_col, target_col, "timestamp_ord"])
feature_cols = [c for c in train_df.columns if c not in exclude]

print(f"Number of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# ------------------------------------------------------------------ #
# Step 3 - Prepare data
# ------------------------------------------------------------------ #
print("\n--- Preparing data ---")
fill_vals = train_df[feature_cols].median()
X_full = train_df[feature_cols].fillna(fill_vals)
X_val  = val_df[feature_cols].fillna(fill_vals)
y_full_raw = train_df[target_col].values

print(f"Target stats: min={y_full_raw.min():.2f}, max={y_full_raw.max():.2f}, mean={y_full_raw.mean():.2f}")
print(f"X_full NaN count: {X_full.isna().sum().sum()}")
print(f"X_val NaN count: {X_val.isna().sum().sum()}")

# ------------------------------------------------------------------ #
# Step 4 - Walk-forward split for tuning (80/20)
# ------------------------------------------------------------------ #
print("\n--- Creating walk-forward split ---")
all_times = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_times) * 0.8)
cutoff_time = all_times[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_time].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_time].copy()

wf_fill_vals = wf_train[feature_cols].median()
X_wf_train = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train = wf_train[target_col].values
X_wf_val   = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val   = wf_val[target_col].values

print(f"Walk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
print(f"Cutoff time: {cutoff_time}")

# ------------------------------------------------------------------ #
# Step 5 - Use pre-tuned Optuna best_params
# ------------------------------------------------------------------ #
print("\n--- Using pre-tuned Optuna params (15 trials, best WF-MAE=31.4450) ---")
best_params = {
    "learning_rate": 0.015994837682273198,
    "num_leaves": 86,
    "min_child_samples": 41,
    "feature_fraction": 0.8376317529609937,
    "bagging_fraction": 0.9971571567707618,
}
optuna_trials = 15
print(f"Best params: {best_params}")

# ------------------------------------------------------------------ #
# Step 6 - Build final hyperparameters and probe n_estimators
# ------------------------------------------------------------------ #
print("\n--- Building final hyperparameters ---")
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_hparams = {**default_params, **best_params}
print(f"Final hyperparams: {final_hparams}")

probe_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 3000,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
probe_params.update(final_hparams)

print("Probing for optimal n_estimators via early stopping...")
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)]
)
best_iter = probe.best_iteration_ if probe.best_iteration_ else 500
best_n_estimators = max(int(best_iter * 1.1), 200)

preds_probe = np.clip(probe.predict(X_wf_val), 0, None)
wf_mae = mean_absolute_error(y_wf_val, preds_probe)
print(f"Walk-forward MAE: {wf_mae:.4f}, best_iteration: {probe.best_iteration_}, n_estimators: {best_n_estimators}")
print(f"Elapsed after probe: {time.time()-start_time:.1f}s")

# ------------------------------------------------------------------ #
# Step 7a - Retrain on full training data with 5 seeds
# ------------------------------------------------------------------ #
print("\n--- Training final ensemble (5 seeds) ---")
final_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": best_n_estimators,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}
final_params.update(final_hparams)

seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    print(f"  Training seed {seed}...")
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full, y_full_raw, callbacks=[lgb.log_evaluation(-1)])
    raw_p = np.clip(m.predict(X_val), 0, None)
    seed_preds.append(raw_p)
    print(f"  Seed {seed}: min={raw_p.min():.2f}, max={raw_p.max():.2f}, mean={raw_p.mean():.2f}")

ensemble_preds = np.mean(seed_preds, axis=0)
ensemble_preds = np.clip(ensemble_preds, 0, None)
print(f"\nEnsemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"NaN predictions: {np.isnan(ensemble_preds).sum()}")
last_model = m

# Use ensemble_preds as final (no quantile blending for electricity — MAE objective already optimal)
final_preds = ensemble_preds.copy()

# ------------------------------------------------------------------ #
# Step 7c - Out-of-fold predictions (4-fold expanding walk-forward)
# ------------------------------------------------------------------ #
print("\n--- Generating out-of-fold predictions (4-fold walk-forward) ---")
N_OOF_FOLDS = 4
oof_mae_val = None
oof_records = []
oof_fold_maes = []

try:
    total_t = len(all_times)
    min_train_t = max(1, total_t // (N_OOF_FOLDS + 1))
    fold_step_t  = max(1, (total_t - min_train_t) // N_OOF_FOLDS)

    for fold_idx in range(N_OOF_FOLDS):
        val_start_idx = min_train_t + fold_idx * fold_step_t
        val_end_idx   = min(val_start_idx + fold_step_t, total_t)
        if val_start_idx >= total_t:
            break

        val_times_set   = set(all_times[val_start_idx:val_end_idx])
        train_end_idx   = max(0, val_start_idx - 1)
        train_times_set = set(all_times[:train_end_idx])

        fold_tr = train_df[train_df[time_col].isin(train_times_set)]
        fold_vl = train_df[train_df[time_col].isin(val_times_set)]
        if len(fold_tr) == 0 or len(fold_vl) == 0:
            continue

        fold_fill = fold_tr[feature_cols].median()
        X_ft = fold_tr[feature_cols].fillna(fold_fill)
        y_ft = fold_tr[target_col].values
        X_fv = fold_vl[feature_cols].fillna(fold_fill)

        fold_model = lgb.LGBMRegressor(**{**final_params, "random_state": 42})
        fold_model.fit(X_ft, y_ft, callbacks=[lgb.log_evaluation(-1)])
        fold_preds_raw = np.clip(fold_model.predict(X_fv), 0, None)

        rec = fold_vl[group_cols + [time_col]].copy().reset_index(drop=True)
        rec["fold"]             = fold_idx
        rec["predicted_target"] = fold_preds_raw
        oof_records.append(rec)

        fold_mae_i = float(mean_absolute_error(fold_vl[target_col].values, fold_preds_raw))
        oof_fold_maes.append(fold_mae_i)
        print(f"  Fold {fold_idx}: train={len(fold_tr)}, val={len(fold_vl)}, mae={fold_mae_i:.4f}")

    if oof_records:
        oof_df = pd.concat(oof_records, ignore_index=True)
        oof_df.to_csv("reports/oof_predictions.csv", index=False)
        oof_mae_val = float(np.mean(oof_fold_maes))
        print(f"Written reports/oof_predictions.csv: {oof_df.shape}, OOF MAE={oof_mae_val:.4f}")
    else:
        print("WARNING: no OOF folds generated")

except Exception as e:
    print(f"WARNING: OOF generation failed ({e})")

print(f"Elapsed after OOF: {time.time()-start_time:.1f}s")

# ------------------------------------------------------------------ #
# Step 8 - Write reports/predictions.csv
# ------------------------------------------------------------------ #
print("\n--- Writing reports/predictions.csv ---")
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = final_preds

nan_count = preds_df["predicted_target"].isna().sum()
if nan_count > 0:
    global_mean = float(train_df[target_col].mean())
    preds_df["predicted_target"] = preds_df["predicted_target"].fillna(global_mean)
    print(f"WARNING: filled {nan_count} NaN predictions with global mean {global_mean:.2f}")

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions remain!"

preds_df.to_csv("reports/predictions.csv", index=False)
print(f"Written reports/predictions.csv: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head(10).to_string())

saved_preds = pd.read_csv("reports/predictions.csv")["predicted_target"].values
print(f"\nSaved stats: min={saved_preds.min():.2f}, max={saved_preds.max():.2f}, mean={saved_preds.mean():.2f}, std={saved_preds.std():.2f}")

# ------------------------------------------------------------------ #
# Step 9 - Write reports/model_results.json
# ------------------------------------------------------------------ #
print("\n--- Writing reports/model_results.json ---")

feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{"feature": str(k), "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp = [{"feature": str(k), "importance": int(v)} for k, v in feat_imp.items()]

training_time = int(time.time() - start_time)

results = {
    "algorithm": "LightGBM",
    "objective": final_params["objective"],
    "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                    for k, v in final_hparams.items()},
    "n_estimators": int(best_n_estimators),
    "n_seeds": 5,
    "seed_aggregation": "mean",
    "walk_forward_mae": float(wf_mae),
    "walk_forward_val_period": f"timestamps from index {cutoff_idx}+",
    "walk_forward_cv_scheme": "single 80/20 time split",
    "oof_mae": float(oof_mae_val) if oof_mae_val is not None else None,
    "oof_cv_scheme": f"{N_OOF_FOLDS}-fold expanding walk-forward (1-period embargo)" if oof_mae_val is not None else None,
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
    "training_time_seconds": training_time,
    "optuna_trials_completed": int(optuna_trials),
    "val_prediction_stats": {
        "min": float(saved_preds.min()),
        "max": float(saved_preds.max()),
        "mean": float(saved_preds.mean()),
        "std": float(saved_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": len(train_df),
    "n_val_rows": len(val_df),
}

with open("reports/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Written reports/model_results.json")
print(json.dumps({k: v for k, v in results.items() if k not in ("feature_importance_all", "feature_importance_top10")}, indent=2))

# ------------------------------------------------------------------ #
# Step 10 - Write marker file
# ------------------------------------------------------------------ #
print("\n--- Writing reports/modeler_was_here.txt ---")
with open("reports/modeler_was_here.txt", "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    f.write(f"Wall-clock time: {training_time} seconds\n")
    f.write(f"Walk-forward MAE: {wf_mae:.4f}\n")
    f.write(f"OOF MAE: {oof_mae_val}\n")
    f.write(f"Optuna trials: {optuna_trials}\n")
    f.write(f"n_estimators: {best_n_estimators}\n")

print("Written reports/modeler_was_here.txt")
print("\n" + "=" * 60)
print("MODELER SUB-AGENT COMPLETE")
print(f"Total wall-clock: {int(time.time() - start_time)} seconds")
print("=" * 60)
