"""
Modeler script: LightGBM with problem-type-aware CV + Optuna tuning.
CV splitter is chosen from profile.json:
  tabular_regression + group_cols  → GroupKFold(n_splits=min(5, n_unique_groups))
  tabular_regression, no groups    → RepeatedKFold(n_splits=5, n_repeats=3)
  classification                   → StratifiedKFold(n_splits=5)
  panel_forecasting / other        → KFold(n_splits=5, shuffle=True)
"""
import pandas as pd
import numpy as np
import json
import time
import warnings
import datetime
import os

import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

BASE_DIR = r"C:\Users\isaac\OneDrive\Desktop\award_B"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")

start_time = time.time()

print("=" * 60)
print("MODELER SUB-AGENT STARTING")
print(f"Start time: {datetime.datetime.now().isoformat()}")
print("=" * 60)

# ── Step 1: Load feature metadata + profile ───────────────────────────────────
feat_path = os.path.join(REPORTS_DIR, "features.json")
with open(feat_path) as f:
    feat_meta = json.load(f)

profile_path = os.path.join(REPORTS_DIR, "profile.json")
with open(profile_path) as f:
    profile = json.load(f)

target_col   = feat_meta["target_col"]
# profile.json is authoritative for problem_type and group_cols (set by schema_analyst)
problem_type = profile.get("problem_type") or feat_meta.get("problem_type", "tabular_regression")
group_cols   = profile.get("group_cols") or feat_meta.get("group_cols", [])
time_col     = profile.get("time_col") or feat_meta.get("time_col")

print(f"Problem type : {problem_type}")
print(f"Target       : {target_col}")
print(f"Groups       : {group_cols}")
print(f"Time col     : {time_col}")

# ── Step 2: Load parquet files ─────────────────────────────────────────────────
train_df = pd.read_parquet(os.path.join(DATA_DIR, "features_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, "features_val.parquet"))

print(f"\nTrain shape : {train_df.shape}")
print(f"Val shape   : {val_df.shape}")
print(f"Train cols  : {list(train_df.columns)}")

# ── Step 3: Define feature columns ────────────────────────────────────────────
# Exclude identifiers, target, group/time cols, and 'horizon' (panel artifact)
ALWAYS_EXCLUDE = {"horizon", "timestamp_ord"}
exclude_set = ALWAYS_EXCLUDE | set(group_cols) | {target_col}
if time_col:
    exclude_set.add(time_col)

# patient_id is an id column — keep it for output but not for features
id_cols_in_train = [c for c in ["patient_id"] if c in train_df.columns]
exclude_set.update(id_cols_in_train)

feature_cols = [c for c in train_df.columns if c not in exclude_set]
print(f"\nFeature cols ({len(feature_cols)}): {feature_cols}")

# NaN fill with training medians
fill_vals = train_df[feature_cols].median()

# ── Step 4: Build CV splits based on problem_type ────────────────────────────

def build_cv_splits(problem_type, group_cols, X, y, df, n_splits=5):
    """Return (splits, cv_scheme).

    splits: list of (train_indices, val_indices) positional arrays (iloc-compatible).
    Rows may appear in multiple val sets for RepeatedKFold; average predictions accordingly.
    """
    if problem_type == "tabular_regression":
        if group_cols:
            gc = group_cols[0]
            if gc in df.columns:
                groups = df[gc].values
                n_unique = len(np.unique(groups))
                actual_splits = max(2, min(n_splits, n_unique))
                gkf = GroupKFold(n_splits=actual_splits)
                splits = list(gkf.split(X, y, groups=groups))
                return splits, f"GroupKFold(n_splits={actual_splits}, group_col='{gc}')"
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
        splits = list(rkf.split(X, y))
        return splits, f"RepeatedKFold(n_splits={n_splits}, n_repeats=3)"
    elif problem_type == "classification":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        return splits, f"StratifiedKFold(n_splits={n_splits})"
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X, y))
        return splits, f"KFold(n_splits={n_splits}, shuffle=True)"

n = len(train_df)
X_full = train_df[feature_cols].fillna(fill_vals)
y_full = train_df[target_col].values
X_val  = val_df[feature_cols].fillna(fill_vals)

print(f"\nX_full NaN : {X_full.isna().sum().sum()}")
print(f"X_val NaN  : {X_val.isna().sum().sum()}")
print(f"y_full     : min={y_full.min()}, max={y_full.max()}, mean={y_full.mean():.3f}")

cv_splits, cv_scheme = build_cv_splits(problem_type, group_cols, X_full, y_full, train_df.reset_index(drop=True))
print(f"\nCV scheme : {cv_scheme}")
print(f"Folds     : {len(cv_splits)}")

# ── Step 5: 80/20 random split for Optuna probe (fast) ───────────────────────
np.random.seed(42)
perm = np.random.permutation(n)
split = int(n * 0.8)
probe_tr_idx, probe_va_idx = perm[:split], perm[split:]

X_ptr = X_full.iloc[probe_tr_idx].reset_index(drop=True)
y_ptr = y_full[probe_tr_idx]
X_pva = X_full.iloc[probe_va_idx].reset_index(drop=True)
y_pva = y_full[probe_va_idx]
# Re-fill on probe subset
ptr_fill = X_ptr.median()
X_ptr = X_ptr.fillna(ptr_fill)
X_pva = X_pva.fillna(ptr_fill)

print(f"\nProbe train: {X_ptr.shape}, probe val: {X_pva.shape}")

# ── Step 6: Optuna hyperparameter tuning (up to 30 trials, 20-min deadline) ───
TUNING_DEADLINE = start_time + 20 * 60

def objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "verbose": -1,
        "n_jobs": -1,
    }
    maes = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**params, "random_state": seed})
        m.fit(
            X_ptr, y_ptr,
            eval_set=[(X_pva, y_pva)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        preds = np.clip(m.predict(X_pva), 0, None)
        maes.append(mean_absolute_error(y_pva, preds))
    return float(np.mean(maes))

try:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, timeout=20 * 60, catch=(Exception,))
    best_params = study.best_params
    optuna_trials = len(study.trials)
    optuna_succeeded = True
    print(f"\nOptuna: {optuna_trials} trials, best MAE={study.best_value:.4f}")
    print(f"Best params: {best_params}")
except Exception as e:
    print(f"Optuna failed ({e}), using defaults")
    best_params = {}
    optuna_trials = 0
    optuna_succeeded = False

# ── Step 6: Merge tuned + fixed params, probe n_estimators ───────────────────
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
final_hparams = {**default_params, **best_params}

probe_fit_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 3000,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    **final_hparams,
}
probe_model = lgb.LGBMRegressor(**probe_fit_params)
probe_model.fit(
    X_ptr, y_ptr,
    eval_set=[(X_pva, y_pva)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
)
best_iter = probe_model.best_iteration_ if probe_model.best_iteration_ else 300
best_n_estimators = max(int(best_iter * 1.1), 200)
probe_mae = mean_absolute_error(y_pva, np.clip(probe_model.predict(X_pva), 0, None))
print(f"\nProbe MAE: {probe_mae:.4f}, best_iter={best_iter}, n_estimators={best_n_estimators}")

# ── Step 7: CV with problem-type-aware splitter for OOF predictions ───────────
final_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": best_n_estimators,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    **final_hparams,
}

# Accumulate: rows may appear in multiple val folds (RepeatedKFold)
oof_accum = np.zeros(n)
oof_count = np.zeros(n, dtype=int)
oof_folds = np.full(n, -1, dtype=int)  # last fold assignment (for oof_predictions.csv)
fold_maes = []

print(f"\n--- {cv_scheme} ---")
for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
    X_tr = X_full.iloc[tr_idx]
    y_tr = y_full[tr_idx]
    X_va = X_full.iloc[va_idx]
    y_va = y_full[va_idx]

    fold_fill = X_tr.median()
    X_tr = X_tr.fillna(fold_fill)
    X_va = X_va.fillna(fold_fill)

    fold_seed_preds = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
        m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
        fold_seed_preds.append(np.clip(m.predict(X_va), 0, None))

    fold_pred = np.mean(fold_seed_preds, axis=0)
    oof_accum[va_idx] += fold_pred
    oof_count[va_idx] += 1
    oof_folds[va_idx] = fold_idx  # last assignment wins for oof_predictions.csv

    fmae = mean_absolute_error(y_va, fold_pred)
    fold_maes.append(fmae)
    print(f"  Fold {fold_idx+1}: MAE={fmae:.4f}")

# Average across repeats (handles RepeatedKFold where count > 1 per row)
oof_preds = np.where(oof_count > 0, oof_accum / oof_count, float(y_full.mean()))
oof_mae = mean_absolute_error(y_full, oof_preds)
print(f"\nOOF MAE ({cv_scheme}): {oof_mae:.4f}")
print(f"Per-fold MAEs: {[round(m, 4) for m in fold_maes]}")

# ── Step 8: Retrain on full data with 5 seeds ─────────────────────────────────
print("\n--- Training on full data (5 seeds) ---")
seed_val_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full, y_full, callbacks=[lgb.log_evaluation(-1)])
    p = np.clip(m.predict(X_val), 0, None)
    seed_val_preds.append(p)
    print(f"  Seed {seed}: mean={p.mean():.3f}")

ensemble_preds = np.mean(seed_val_preds, axis=0)
print(f"\nEnsemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"NaN count: {np.isnan(ensemble_preds).sum()}")
last_model = m

# ── Step 9: Write reports/predictions.csv ────────────────────────────────────
# Identifier columns: the id col + any group/time cols present in val_df
id_candidates = ([profile.get("id_col")] if profile.get("id_col") else []) + group_cols + ([time_col] if time_col else [])
id_output = [c for c in id_candidates if c and c in val_df.columns]
if not id_output:  # fallback: first non-feature column
    id_output = [c for c in val_df.columns if c not in feature_cols][:1]

preds_df = val_df[id_output].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

nan_count = preds_df["predicted_target"].isna().sum()
if nan_count > 0:
    gm = float(y_full.mean())
    preds_df["predicted_target"] = preds_df["predicted_target"].fillna(gm)
    print(f"WARNING: filled {nan_count} NaN predictions with global mean {gm:.2f}")

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions remain!"

pred_path = os.path.join(REPORTS_DIR, "predictions.csv")
preds_df.to_csv(pred_path, index=False)
print(f"\nWritten {pred_path}: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head(5).to_string())

# ── Step 10: Write reports/oof_predictions.csv ───────────────────────────────
oof_id_cols = [c for c in id_output if c in train_df.columns]
oof_df = train_df[oof_id_cols].copy().reset_index(drop=True)
oof_df["fold"] = oof_folds
oof_df["predicted_target"] = oof_preds

oof_path = os.path.join(REPORTS_DIR, "oof_predictions.csv")
oof_df.to_csv(oof_path, index=False)
print(f"\nWritten {oof_path}: {oof_df.shape}")
print(f"OOF MAE: {oof_mae:.4f}")

# ── Step 11: Write reports/model_results.json ─────────────────────────────────
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10    = [{"feature": str(k), "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp  = [{"feature": str(k), "importance": int(v)} for k, v in feat_imp.items()]

training_time = int(time.time() - start_time)

results = {
    "algorithm": "LightGBM",
    "objective": final_params["objective"],
    "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                    for k, v in final_hparams.items()},
    "n_estimators": int(best_n_estimators),
    "n_seeds": 5,
    "cv_scheme": cv_scheme,
    "oof_mae": float(oof_mae),
    "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(m) for m in fold_maes],
    "probe_mae_80_20": float(probe_mae),
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
    "training_time_seconds": training_time,
    "optuna_trials_completed": int(optuna_trials),
    "optuna_succeeded": bool(optuna_succeeded),
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()),
        "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()),
        "std": float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": int(len(train_df)),
    "n_val_rows": int(len(val_df)),
}

results_path = os.path.join(REPORTS_DIR, "model_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWritten {results_path}")

# Pretty-print summary (no giant lists)
summary = {k: v for k, v in results.items() if k not in ("feature_importance_all", "feature_importance_top10")}
print(json.dumps(summary, indent=2))

# ── Step 12: Write marker file ────────────────────────────────────────────────
marker_path = os.path.join(REPORTS_DIR, "modeler_was_here.txt")
with open(marker_path, "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    f.write(f"OOF MAE ({cv_scheme}): {oof_mae:.4f}\n")
    f.write(f"Probe MAE (80/20): {probe_mae:.4f}\n")
    f.write(f"Training time: {training_time}s\n")
    f.write(f"Optuna trials: {optuna_trials}\n")
    f.write(f"n_estimators: {best_n_estimators}\n")
print(f"Written {marker_path}")

print("\n" + "=" * 60)
print("MODELER COMPLETE")
print(f"OOF MAE : {oof_mae:.4f}")
print(f"Val mean: {ensemble_preds.mean():.3f}")
print(f"Time    : {training_time}s")
print("=" * 60)
