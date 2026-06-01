import pandas as pd
import numpy as np
import json
import time
import warnings
import datetime

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
# features.json may not have group_cols/time_col; fall back to known values
group_cols = feat_meta.get("group_cols", ["store_id", "product_id"])
time_col   = feat_meta.get("time_col", "week")
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

# Build feature list (exclude group/time/target)
exclude = set(group_cols + [time_col, target_col])
feature_cols = [c for c in train_df.columns if c not in exclude]

print(f"Number of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# ------------------------------------------------------------------ #
# Step 3 - Prepare data
# ------------------------------------------------------------------ #
print("\n--- Preparing data ---")

# Use training column medians for NaN imputation
# Val lag features (lag_1 through lag_9) are up to 90% NaN because
# feature engineer computed lags that look back into val where
# weekly_sales is NaN. Filling with training medians (~46) avoids
# systematic underprediction that would result from filling with 0.
fill_vals = train_df[feature_cols].median()
X_full = train_df[feature_cols].fillna(fill_vals)
X_val  = val_df[feature_cols].fillna(fill_vals)
y_full_raw = train_df[target_col].values

print(f"Target stats: min={y_full_raw.min():.2f}, max={y_full_raw.max():.2f}, mean={y_full_raw.mean():.2f}")
print(f"X_full NaN count: {X_full.isna().sum().sum()}")
print(f"X_val NaN count: {X_val.isna().sum().sum()}")

# ------------------------------------------------------------------ #
# Step 4 - Walk-forward split for tuning
# ------------------------------------------------------------------ #
print("\n--- Creating walk-forward split ---")

all_weeks = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_weeks) * 0.8)
cutoff_week = all_weeks[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_week].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_week].copy()

wf_fill_vals = wf_train[feature_cols].median()
X_wf_train = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train = wf_train[target_col].values
X_wf_val   = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val   = wf_val[target_col].values

print(f"Walk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
print(f"Cutoff week: {cutoff_week}")

# ------------------------------------------------------------------ #
# Step 5 - Optuna hyperparameter tuning
# ------------------------------------------------------------------ #
print("\n--- Optuna hyperparameter tuning ---")

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_available = True
except ImportError:
    print("Optuna not available - skipping tuning")
    optuna_available = False

TUNING_DEADLINE = start_time + 25 * 60  # 25 minutes

best_params = {}
optuna_trials = 0
optuna_succeeded = False

if optuna_available:
    def objective(trial):
        if time.time() > TUNING_DEADLINE:
            raise optuna.exceptions.TrialPruned()

        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        }

        # 3-seed average for stability
        maes = []
        for seed in [42, 7, 123]:
            m = lgb.LGBMRegressor(**{**params, "random_state": seed})
            m.fit(
                X_wf_train, y_wf_train,
                eval_set=[(X_wf_val, y_wf_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            )
            preds = np.clip(m.predict(X_wf_val), 0, None)
            maes.append(mean_absolute_error(y_wf_val, preds))
        return float(np.mean(maes))

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15, timeout=25*60, catch=(Exception,))
        best_params = study.best_params
        optuna_trials = len(study.trials)
        print(f"Optuna complete: {optuna_trials} trials, best MAE={study.best_value:.4f}")
        print(f"Best params: {best_params}")
        optuna_succeeded = True
    except Exception as e:
        print(f"Optuna failed ({e}), using defaults")
        best_params = {}
        optuna_trials = 0
else:
    print("Skipping Optuna - not available")

print(f"Elapsed after Optuna: {time.time()-start_time:.1f}s")

# ------------------------------------------------------------------ #
# Step 6 - Build final hyperparameters
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

# Determine n_estimators via early stopping on walk-forward split
probe_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 2000,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    **final_hparams,
}

print("Probing for optimal n_estimators via early stopping...")
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)

best_iter = probe.best_iteration_ if probe.best_iteration_ else 500
best_n_estimators = max(int(best_iter * 1.1), 200)

preds_probe = np.clip(probe.predict(X_wf_val), 0, None)
wf_mae = mean_absolute_error(y_wf_val, preds_probe)

print(f"Walk-forward MAE: {wf_mae:.4f}, best_iteration: {probe.best_iteration_}, using n_estimators: {best_n_estimators}")
print(f"Elapsed after probe: {time.time()-start_time:.1f}s")

# ------------------------------------------------------------------ #
# Step 7 - Retrain on full training data with 5 seeds
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
    **final_hparams,
}

seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    print(f"  Training seed {seed}...")
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full, y_full_raw, callbacks=[lgb.log_evaluation(-1)])
    raw_p = np.clip(m.predict(X_val), 0, None)
    seed_preds.append(raw_p)
    print(f"  Seed {seed}: min={raw_p.min():.2f}, max={raw_p.max():.2f}, mean={raw_p.mean():.2f}. Elapsed: {time.time()-start_time:.1f}s")

ensemble_preds = np.median(seed_preds, axis=0)
ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f"\nEnsemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"NaN predictions: {np.isnan(ensemble_preds).sum()}")

last_model = m

# ------------------------------------------------------------------ #
# Step 7b - Quantile models for tail blending
# ------------------------------------------------------------------ #
print("\n--- Training quantile models (Q70 and Q85) for tail blending ---")

q_params_base = {
    "metric": "mae",
    "n_estimators": best_n_estimators,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    **final_hparams,
}

q70_seed_preds = []
for seed in [42, 7, 123]:
    print(f"  Q70 seed {seed}...")
    m_q = lgb.LGBMRegressor(**{**q_params_base, "objective": "quantile", "alpha": 0.70, "random_state": seed})
    m_q.fit(X_full, y_full_raw, callbacks=[lgb.log_evaluation(-1)])
    q70_seed_preds.append(np.clip(m_q.predict(X_val), 0, None))
q70_preds = np.median(q70_seed_preds, axis=0)
print(f"Q70: min={q70_preds.min():.2f}, max={q70_preds.max():.2f}, mean={q70_preds.mean():.2f}")

q85_seed_preds = []
for seed in [42, 7, 123]:
    print(f"  Q85 seed {seed}...")
    m_q = lgb.LGBMRegressor(**{**q_params_base, "objective": "quantile", "alpha": 0.85, "random_state": seed})
    m_q.fit(X_full, y_full_raw, callbacks=[lgb.log_evaluation(-1)])
    q85_seed_preds.append(np.clip(m_q.predict(X_val), 0, None))
q85_preds = np.median(q85_seed_preds, axis=0)
print(f"Q85: min={q85_preds.min():.2f}, max={q85_preds.max():.2f}, mean={q85_preds.mean():.2f}")

mask_main_only    = ensemble_preds < 35
mask_blend_q70_lo = (ensemble_preds >= 35) & (ensemble_preds < 50)
mask_blend_q70_hi = (ensemble_preds >= 50) & (ensemble_preds < 80)
mask_blend_q85    = ensemble_preds >= 80

final_preds = np.zeros_like(ensemble_preds)
final_preds[mask_main_only]    = ensemble_preds[mask_main_only]
final_preds[mask_blend_q70_lo] = 0.7 * ensemble_preds[mask_blend_q70_lo] + 0.3 * q70_preds[mask_blend_q70_lo]
final_preds[mask_blend_q70_hi] = 0.4 * ensemble_preds[mask_blend_q70_hi] + 0.6 * q70_preds[mask_blend_q70_hi]
final_preds[mask_blend_q85]    = 0.3 * ensemble_preds[mask_blend_q85]    + 0.7 * q85_preds[mask_blend_q85]
final_preds = np.clip(final_preds, 0, None)

n_main_only = int(mask_main_only.sum())
n_blend_q70 = int((mask_blend_q70_lo | mask_blend_q70_hi).sum())
n_blend_q85 = int(mask_blend_q85.sum())

print(f"\nBlended predictions: min={final_preds.min():.2f}, max={final_preds.max():.2f}, "
      f"mean={final_preds.mean():.2f}, std={final_preds.std():.2f}")
print(f"  main only (<35):      {n_main_only}")
print(f"  blended q70 (35-80):  {n_blend_q70}")
print(f"  blended q85 (>=80):   {n_blend_q85}")
print(f"Elapsed after quantile blending: {time.time()-start_time:.1f}s")

# ------------------------------------------------------------------ #
# Step 8 - Write reports/predictions.csv
# ------------------------------------------------------------------ #
print("\n--- Writing reports/predictions.csv ---")

preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = final_preds

nan_count = preds_df["predicted_target"].isna().sum()
if nan_count > 0:
    print(f"WARNING: {nan_count} NaN predictions - filling with group mean")
    global_mean = float(train_df[target_col].mean())
    preds_df["predicted_target"] = preds_df["predicted_target"].fillna(global_mean)

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions remain - abort"

preds_df.to_csv("reports/predictions.csv", index=False)
print(f"Written reports/predictions.csv: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head(10).to_string())

# Read back the saved predictions to ensure stats match what's on disk
saved_preds = pd.read_csv("reports/predictions.csv")["predicted_target"].values
print(f"\nSaved predictions stats (from disk): min={saved_preds.min():.2f}, max={saved_preds.max():.2f}, mean={saved_preds.mean():.2f}, std={saved_preds.std():.2f}")

# ------------------------------------------------------------------ #
# Step 9 - Write reports/model_results.json
# ------------------------------------------------------------------ #
print("\n--- Writing reports/model_results.json ---")

feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{"feature": str(k), "importance": int(v)} for k, v in feat_imp.head(10).items()]

training_time = int(time.time() - start_time)

results = {
    "algorithm": "LightGBM",
    "objective": final_params["objective"],
    "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                    for k, v in final_hparams.items()},
    "n_estimators": int(best_n_estimators),
    "n_seeds": 5,
    "seed_aggregation": "median",
    "walk_forward_mae": float(wf_mae),
    "walk_forward_val_period": f"weeks {int(cutoff_week)}+",
    "feature_importance_top10": top10,
    "training_time_seconds": training_time,
    "optuna_trials_completed": int(optuna_trials),
    "quantile_models_used": ["q70", "q85"],
    "blending_strategy": "main_only if pred<35; 70%main+30%q70 if 35<=pred<50; 40%main+60%q70 if 50<=pred<80; 30%main+70%q85 if pred>=80",
    "n_predictions_main_only": n_main_only,
    "n_predictions_blended_q70": n_blend_q70,
    "n_predictions_blended_q85": n_blend_q85,
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
print(json.dumps(results, indent=2))

# ------------------------------------------------------------------ #
# Step 10 - Write marker file
# ------------------------------------------------------------------ #
print("\n--- Writing reports/modeler_was_here.txt ---")

with open("reports/modeler_was_here.txt", "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    f.write(f"Wall-clock time: {training_time} seconds\n")
    f.write(f"Walk-forward MAE: {wf_mae:.4f}\n")
    f.write(f"Optuna trials: {optuna_trials}\n")
    f.write(f"n_estimators: {best_n_estimators}\n")

print("Written reports/modeler_was_here.txt")

print("\n" + "=" * 60)
print("MODELER SUB-AGENT COMPLETE")
print(f"Total wall-clock: {int(time.time() - start_time)} seconds")
print("=" * 60)
