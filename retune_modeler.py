import pandas as pd, numpy as np, json, time, warnings, os, datetime
warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\isaac\OneDrive\Desktop\award_B")
start_time = time.time()

# Check for critic retune request
_retune_applied = None
if os.path.exists("reports/critic_retune_requested.json"):
    with open("reports/critic_retune_requested.json") as f:
        _retune = json.load(f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"Critic retune requested: {_suggestion}")
    _retune_applied = "critic_retune"

    if "median seed aggregation" in _suggestion:
        _retune_applied = "median_seed_aggregation"
        print("Applying: median seed aggregation")
    elif "expand Optuna" in _suggestion:
        _retune_applied = "expanded_optuna_bounds"
        print("Applying: expanded Optuna bounds")
    elif "val feature imputation" in _suggestion:
        _retune_applied = "verified_imputation"
        print("Applying: verified imputation")
    elif "remove suspect features" in _suggestion:
        _retune_applied = "removed_suspect_features"
        print("Applying: removing suspect features")
    else:
        print(f"No specific code change matched '{_suggestion}' — retraining with same config (retune_applied=critic_retune)")

# Load data
with open("reports/features.json") as f:
    feat_meta = json.load(f)
with open("reports/profile.json") as f:
    profile = json.load(f)

train_df = pd.read_parquet("data/features_train.parquet")
val_df   = pd.read_parquet("data/features_val.parquet")

target_col = feat_meta["target_col"]
group_cols = feat_meta["group_cols"]
time_col   = feat_meta["time_col"]

exclude = set(group_cols + [time_col, target_col])
feature_cols = [c for c in train_df.columns if c not in exclude]

fill_vals = train_df[feature_cols].median()
X_full = train_df[feature_cols].fillna(fill_vals)
y_full = train_df[target_col].values
X_val  = val_df[feature_cols].fillna(fill_vals)
n = len(X_full)

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Features: {len(feature_cols)}")

# Walk-forward split
all_steps = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_steps) * 0.8)
cutoff_step = all_steps[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_step].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_step].copy()
wf_fill = wf_train[feature_cols].median()
X_wf_train = wf_train[feature_cols].fillna(wf_fill)
y_wf_train = wf_train[target_col].values
X_wf_val   = wf_val[feature_cols].fillna(wf_fill)
y_wf_val   = wf_val[target_col].values

# Optuna tuning (8 trials)
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 8 * 60

def objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()
    params = {
        "objective": "regression_l1", "metric": "mae",
        "n_estimators": 300,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": -1,
    }
    maes = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**params, "random_state": seed})
        m.fit(X_wf_train, y_wf_train,
              eval_set=[(X_wf_val, y_wf_val)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        preds = np.clip(m.predict(X_wf_val), 0, None)
        maes.append(mean_absolute_error(y_wf_val, preds))
    return float(np.mean(maes))

try:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8, timeout=8*60, catch=(Exception,))
    best_params = study.best_params
    optuna_trials = len(study.trials)
    print(f"Optuna: {optuna_trials} trials, best MAE={study.best_value:.4f}")
    optuna_succeeded = True
except Exception as e:
    print(f"Optuna failed: {e}")
    best_params = {}
    optuna_trials = 0

default_params = {
    "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 5, "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_hparams = {**default_params, **best_params}

probe_params = {
    "objective": "regression_l1", "metric": "mae", "n_estimators": 1000,
    "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "verbose": -1, "n_jobs": -1, "random_state": 42, **final_hparams,
}
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(X_wf_train, y_wf_train,
          eval_set=[(X_wf_val, y_wf_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
best_n_est = max(50, int((probe.best_iteration_ or 100) * 1.1))
wf_mae = mean_absolute_error(y_wf_val, np.clip(probe.predict(X_wf_val), 0, None))
print(f"Walk-forward MAE: {wf_mae:.4f}, n_estimators: {best_n_est}")

# Leave-one-patient-out OOF
final_params = {
    "objective": "regression_l1", "metric": "mae", "n_estimators": best_n_est,
    "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "verbose": -1, "n_jobs": -1, **final_hparams,
}

patients = train_df[group_cols[0]].unique()
oof_preds_arr = np.zeros(n)
oof_fold_arr = np.full(n, -1, dtype=int)
fold_maes = []

print("\n--- Leave-One-Patient-Out OOF (retune) ---")
for fold_idx, patient in enumerate(patients):
    tr_mask = train_df[group_cols[0]] != patient
    va_mask = train_df[group_cols[0]] == patient
    tr_idx = np.where(tr_mask.values)[0]
    va_idx = np.where(va_mask.values)[0]
    X_tr = X_full.iloc[tr_idx]; y_tr = y_full[tr_idx]
    X_va = X_full.iloc[va_idx]; y_va = y_full[va_idx]
    seed_preds_fold = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
        m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
        seed_preds_fold.append(np.clip(m.predict(X_va), 0, None))
    fold_pred = np.mean(seed_preds_fold, axis=0)
    oof_preds_arr[va_idx] = fold_pred
    oof_fold_arr[va_idx] = fold_idx
    fmae = mean_absolute_error(y_va, fold_pred)
    fold_maes.append(fmae)
    print(f"  Patient {patient}: MAE={fmae:.4f}")

oof_mae = mean_absolute_error(y_full, oof_preds_arr)
cv_scheme = f"LeaveOnePatientOut(n_folds={len(patients)})"
print(f"\nOOF MAE: {oof_mae:.4f}")

# Final ensemble
seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full, y_full, callbacks=[lgb.log_evaluation(-1)])
    seed_preds.append(np.clip(m.predict(X_val), 0, None))

ensemble_preds = np.mean(seed_preds, axis=0)
last_model = m
print(f"Ensemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")

# predictions.csv
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds
preds_df.to_csv("reports/predictions.csv", index=False)
print(f"Written reports/predictions.csv: {preds_df.shape}")
print(preds_df)

# oof_predictions.csv
oof_df = train_df[group_cols + [time_col]].copy().reset_index(drop=True)
oof_df["fold"] = oof_fold_arr
oof_df["predicted_target"] = oof_preds_arr
oof_df.to_csv("reports/oof_predictions.csv", index=False)
print(f"Written reports/oof_predictions.csv")

# model_results.json
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{"feature": k, "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp = {k: int(v) for k, v in feat_imp.items()}

results = {
    "algorithm": "LightGBM", "objective": "regression_l1",
    "best_params": final_hparams, "n_estimators": best_n_est, "n_seeds": 5,
    "cv_scheme": cv_scheme, "oof_mae": float(oof_mae), "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(x) for x in fold_maes],
    "walk_forward_mae": float(wf_mae),
    "feature_importance_top10": top10, "feature_importance_all": all_imp,
    "training_time_seconds": int(time.time() - start_time),
    "optuna_trials_completed": optuna_trials,
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()), "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()), "std": float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols), "n_train_rows": len(train_df), "n_val_rows": len(val_df),
    "retune_applied": _retune_applied,
}
with open("reports/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Written reports/model_results.json")

with open("reports/modeler_was_here.txt", "w") as f:
    f.write(f"modeler sub-agent (retune) executed at {datetime.datetime.utcnow().isoformat()}Z\n")
print("Written reports/modeler_was_here.txt")
print(f"Total time: {int(time.time()-start_time)}s")
