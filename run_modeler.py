import pandas as pd
import numpy as np
import json
import time
import warnings
import datetime
warnings.filterwarnings('ignore')

start_time = time.time()

# ── Step 1: Load feature metadata ──────────────────────────────────────────
with open("C:/Users/isaac/OneDrive/Desktop/award_B/reports/features.json") as f:
    feat_meta = json.load(f)

target_col = feat_meta["target_col"]
group_cols = feat_meta["group_cols"]
time_col   = feat_meta["time_col"]

train_df = pd.read_parquet("C:/Users/isaac/OneDrive/Desktop/award_B/data/features_train.parquet")
val_df   = pd.read_parquet("C:/Users/isaac/OneDrive/Desktop/award_B/data/features_val.parquet")

exclude = set(group_cols + [time_col, target_col])
feature_cols = [c for c in feat_meta["feature_columns"] if c in train_df.columns]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Features available: {len(feature_cols)}")
print(f"Target: {target_col}")

# Use training column medians as NaN fill
fill_vals = train_df[feature_cols].median()
missing_in_train = train_df[feature_cols].isna().sum()
missing_in_val   = val_df[feature_cols].isna().sum()
print(f"NaN in train (total): {missing_in_train.sum()}")
print(f"NaN in val (total):   {missing_in_val.sum()}")

# ── Step 2: Walk-forward split ─────────────────────────────────────────────
all_weeks  = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_weeks) * 0.8)
cutoff_week = all_weeks[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_week].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_week].copy()

wf_fill_vals = wf_train[feature_cols].median()
X_wf_train = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train = wf_train[target_col]
X_wf_val   = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val   = wf_val[target_col]

print(f"Walk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
print(f"Cutoff week: {cutoff_week}")

# ── Step 3: Optuna tuning ──────────────────────────────────────────────────
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 25 * 60  # 25 minutes

def objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

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
    optuna_succeeded = False

# ── Step 4: Final hyperparameters + n_estimators probe ────────────────────
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_hparams = {**default_params, **best_params}

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
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_n = probe.best_iteration_ if probe.best_iteration_ else 500
best_n_estimators = int(best_n * 1.1)
wf_mae = mean_absolute_error(y_wf_val, np.clip(probe.predict(X_wf_val), 0, None))
print(f"Walk-forward MAE: {wf_mae:.4f}, best_iteration: {best_n}, using n_estimators: {best_n_estimators}")

# ── Step 5: Retrain on full data, 5 seeds ─────────────────────────────────
X_full = train_df[feature_cols].fillna(fill_vals)
y_full = train_df[target_col]
X_val  = val_df[feature_cols].fillna(fill_vals)

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
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full, y_full, callbacks=[lgb.log_evaluation(-1)])
    seed_preds.append(np.clip(m.predict(X_val), 0, 250))

ensemble_preds = np.mean(seed_preds, axis=0)
print(f"Ensemble predictions: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"NaN predictions: {np.isnan(ensemble_preds).sum()}")

last_model = m

# ── Step 6: Write predictions.csv ─────────────────────────────────────────
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found — abort"
assert len(preds_df) == 15000, f"Expected 15000 rows, got {len(preds_df)}"

preds_df.to_csv("C:/Users/isaac/OneDrive/Desktop/award_B/reports/predictions.csv", index=False)
print(f"Written reports/predictions.csv: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head())

# ── Step 7: Write model_results.json ──────────────────────────────────────
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{"feature": k, "importance": int(v)} for k, v in feat_imp.head(10).items()]

training_time = int(time.time() - start_time)

results = {
    "algorithm": "LightGBM",
    "objective": final_params["objective"],
    "best_params": final_hparams,
    "n_estimators": best_n_estimators,
    "n_seeds": 5,
    "walk_forward_mae": float(wf_mae),
    "walk_forward_val_period": f"weeks {int(cutoff_week)}+",
    "feature_importance_top10": top10,
    "training_time_seconds": training_time,
    "optuna_trials_completed": optuna_trials,
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()),
        "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()),
        "std": float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": len(train_df),
    "n_val_rows": len(val_df),
}

with open("C:/Users/isaac/OneDrive/Desktop/award_B/reports/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Written reports/model_results.json")

# ── Step 8: Marker file ────────────────────────────────────────────────────
with open("C:/Users/isaac/OneDrive/Desktop/award_B/reports/modeler_was_here.txt", "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print("Written reports/modeler_was_here.txt")
print(f"Total time: {training_time}s")
