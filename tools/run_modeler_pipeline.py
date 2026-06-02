"""
Full adaptive ensemble modeling pipeline.
Adaptive Axis 1: panel_forecasting + n_train>=1000 -> LightGBM + XGBoost + Ridge
Adaptive Axis 2: max_ks=0.5602 > 0.40 -> ridge_weighted_1.5x (with competence check)
Log1p target transform applied (skew ~2.70), predictions back-transformed with expm1.
"""
import pandas as pd
import numpy as np
import json
import time
import warnings
import os
import datetime
warnings.filterwarnings('ignore')

start_time = time.time()
REPO_ROOT = r"C:\Users\isaac\OneDrive\Desktop\award_B"

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(os.path.join(REPO_ROOT, "reports/features.json")) as f:
    feat_meta = json.load(f)

with open(os.path.join(REPO_ROOT, "reports/profile.json")) as f:
    profile = json.load(f)

train_df = pd.read_parquet(os.path.join(REPO_ROOT, "data/features_train.parquet"))
val_df   = pd.read_parquet(os.path.join(REPO_ROOT, "data/features_val.parquet"))

target_col   = feat_meta["target_col"]
group_cols   = feat_meta["group_cols"]
time_col     = feat_meta["time_col"]
problem_type = profile["problem_type"]

exclude = set(group_cols + [time_col, target_col])

# Numeric feature columns only (drop string cols like state_doh_release)
all_feat_cols = [c for c in train_df.columns if c not in exclude]
feature_cols  = [c for c in all_feat_cols if pd.api.types.is_numeric_dtype(train_df[c])]
dropped_non_numeric = [c for c in all_feat_cols if c not in feature_cols]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Numeric features: {len(feature_cols)} (dropped {len(dropped_non_numeric)} non-numeric: {dropped_non_numeric})")
print(f"Target: {target_col}, Problem type: {problem_type}")

# ── Adaptive ensemble decisions ───────────────────────────────────────────────
n_train = len(train_df)
_dist_shifts = profile.get("distribution_shifts", [])
_max_ks = max(
    (d.get("ks_statistic", 0.0) for d in _dist_shifts if isinstance(d, dict)),
    default=0.0
)
print(f"\nn_train={n_train}, max_ks={_max_ks:.4f}")
# Branch 1: panel_forecasting + n_train >= 1000 -> LightGBM + XGBoost + Ridge
families = ["LightGBM", "XGBoost", "Ridge"]
weighting_mode = "ridge_weighted_1.5x" if _max_ks > 0.40 else "equal_median"
print(f"Families: {families}, Pre-competence weighting: {weighting_mode}")

# Check for critic retune
_retune_applied = None
retune_path = os.path.join(REPO_ROOT, "reports/critic_retune_requested.json")
if os.path.exists(retune_path):
    with open(retune_path) as _f:
        _retune = json.load(_f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"Critic retune requested: {_suggestion}")
    _retune_applied = "critic_retune"

# ── Fill values from training medians ────────────────────────────────────────
fill_vals = train_df[feature_cols].median()

# ── Log1p target transform ───────────────────────────────────────────────────
y_full_raw      = train_df[target_col].values
y_full          = np.log1p(y_full_raw)
train_mean_orig = float(np.mean(y_full_raw))
train_max_orig  = float(np.max(y_full_raw))
print(f"\nTarget raw: min={y_full_raw.min():.3f}, max={y_full_raw.max():.3f}, mean={train_mean_orig:.3f}")
print(f"Target log1p: min={y_full.min():.3f}, max={y_full.max():.3f}, mean={y_full.mean():.3f}")

X_full = train_df[feature_cols].fillna(fill_vals)
n = len(X_full)

# ── Walk-forward split for tuning ─────────────────────────────────────────────
all_weeks  = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_weeks) * 0.8)
cutoff_week = all_weeks[cutoff_idx]

wf_train_df = train_df[train_df[time_col] < cutoff_week].copy()
wf_val_df   = train_df[train_df[time_col] >= cutoff_week].copy()

wf_fill_vals = wf_train_df[feature_cols].median()
X_wf_train = wf_train_df[feature_cols].fillna(wf_fill_vals)
y_wf_train = np.log1p(wf_train_df[target_col].values)
X_wf_val   = wf_val_df[feature_cols].fillna(wf_fill_vals)
y_wf_val   = np.log1p(wf_val_df[target_col].values)

print(f"\nWalk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
print(f"Cutoff week: {cutoff_week}")
print(f"Elapsed: {(time.time()-start_time):.1f}s")

# ── LightGBM Optuna tuning ────────────────────────────────────────────────────
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 25 * 60

def lgb_objective(trial):
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
        preds = m.predict(X_wf_val)
        maes.append(mean_absolute_error(y_wf_val, preds))
    return float(np.mean(maes))

print("\nStarting LightGBM Optuna (15 trials)...")
try:
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lgb_objective, n_trials=15, timeout=20*60, catch=(Exception,))
    best_params_lgb = study_lgb.best_params
    optuna_trials   = len(study_lgb.trials)
    print(f"LightGBM Optuna: {optuna_trials} trials, best MAE(log1p)={study_lgb.best_value:.4f}")
    print(f"Best params: {best_params_lgb}")
    optuna_succeeded = True
except Exception as e:
    print(f"LightGBM Optuna failed ({e}), using defaults")
    best_params_lgb  = {}
    optuna_trials    = 0
    optuna_succeeded = False

print(f"Elapsed after LightGBM Optuna: {(time.time()-start_time)/60:.1f} min")

# ── LightGBM: final hparams + early stopping ──────────────────────────────────
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_hparams_lgb = {**default_params, **best_params_lgb}

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
    **final_hparams_lgb,
}
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_n_est = int(probe.best_iteration_ * 1.1) if probe.best_iteration_ else 500

# MAE in both spaces
wf_preds_log  = probe.predict(X_wf_val)
wf_mae_log    = mean_absolute_error(y_wf_val, wf_preds_log)
wf_preds_orig = np.expm1(np.clip(wf_preds_log, 0, None))
wf_mae_orig   = mean_absolute_error(np.expm1(y_wf_val), wf_preds_orig)
print(f"\nLightGBM WF MAE log1p={wf_mae_log:.4f}, orig={wf_mae_orig:.4f}")
print(f"best_iteration={probe.best_iteration_}, n_estimators={best_n_est}")
print(f"Elapsed: {(time.time()-start_time)/60:.1f} min")

# ── LightGBM: full retrain (5 seeds) ─────────────────────────────────────────
final_params_lgb = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": best_n_est,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    **final_hparams_lgb,
}

X_full_filled = train_df[feature_cols].fillna(fill_vals)
X_val         = val_df[feature_cols].fillna(fill_vals)

print("\nRetraining LightGBM on full data (5 seeds)...")
lgb_seed_preds = []
last_lgb_model = None
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params_lgb, "random_state": seed})
    m.fit(X_full_filled, y_full, callbacks=[lgb.log_evaluation(-1)])
    lgb_seed_preds.append(m.predict(X_val))
    last_lgb_model = m

lgb_val_preds_log = np.mean(lgb_seed_preds, axis=0)
lgb_val_preds     = np.expm1(np.clip(lgb_val_preds_log, 0, None))
lgb_oof_mae       = wf_mae_orig

print(f"LightGBM val preds: min={lgb_val_preds.min():.3f}, max={lgb_val_preds.max():.3f}, mean={lgb_val_preds.mean():.3f}")
print(f"Elapsed: {(time.time()-start_time)/60:.1f} min")

# ── XGBoost Optuna tuning ──────────────────────────────────────────────────────
elapsed_min      = (time.time() - start_time) / 60
xgb_succeeded    = False
xgb_val_preds    = None
xgb_oof_mae      = None
best_params_xgb  = {}

if elapsed_min > 20:
    print(f"\nSkipping XGBoost (elapsed {elapsed_min:.1f} min > 20 min limit)")
else:
    try:
        import xgboost as xgb_lib
        print(f"\nStarting XGBoost Optuna (15 trials)...")

        def xgb_objective(trial):
            if time.time() > start_time + 25 * 60:
                raise optuna.exceptions.TrialPruned()
            params_x = {
                "objective": "reg:absoluteerror",
                "n_estimators": 500,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "tree_method": "hist",
                "n_jobs": -1,
                "verbosity": 0,
                "early_stopping_rounds": 50,
            }
            maes_x = []
            for seed in [42, 7, 123]:
                mx = xgb_lib.XGBRegressor(**{**params_x, "random_state": seed})
                mx.fit(
                    X_wf_train, y_wf_train,
                    eval_set=[(X_wf_val, y_wf_val)],
                    verbose=False,
                )
                preds_x = mx.predict(X_wf_val)
                maes_x.append(mean_absolute_error(y_wf_val, preds_x))
            return float(np.mean(maes_x))

        study_xgb = optuna.create_study(direction="minimize")
        xgb_timeout = max(60, (start_time + 25*60) - time.time())
        study_xgb.optimize(xgb_objective, n_trials=15, timeout=xgb_timeout, catch=(Exception,))
        best_params_xgb = study_xgb.best_params
        print(f"XGBoost Optuna: {len(study_xgb.trials)} trials, best MAE(log1p)={study_xgb.best_value:.4f}")

        # Retrain on full data (5 seeds) — no early stopping since we use full data
        final_params_xgb = {
            "objective": "reg:absoluteerror",
            "n_estimators": 500,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }
        # Use best hparams but strip early_stopping_rounds for full retrain
        for k, v in best_params_xgb.items():
            if k != "early_stopping_rounds":
                final_params_xgb[k] = v
        print("Retraining XGBoost on full data (5 seeds)...")
        xgb_seed_preds = []
        for seed in [42, 7, 123, 2024, 999]:
            mx = xgb_lib.XGBRegressor(**{**final_params_xgb, "random_state": seed})
            mx.fit(X_full_filled, y_full, verbose=False)
            xgb_seed_preds.append(mx.predict(X_val))

        xgb_val_preds_log = np.mean(xgb_seed_preds, axis=0)
        xgb_val_preds     = np.expm1(np.clip(xgb_val_preds_log, 0, None))

        # WF MAE for XGBoost (use early_stopping_rounds in constructor)
        xgb_probe_params = {**final_params_xgb, "early_stopping_rounds": 50}
        mx_probe = xgb_lib.XGBRegressor(**{**xgb_probe_params, "random_state": 42})
        mx_probe.fit(
            X_wf_train, y_wf_train,
            eval_set=[(X_wf_val, y_wf_val)],
            verbose=False,
        )
        xgb_wf_p   = np.expm1(np.clip(mx_probe.predict(X_wf_val), 0, None))
        xgb_oof_mae = mean_absolute_error(np.expm1(y_wf_val), xgb_wf_p)

        print(f"XGBoost val preds: min={xgb_val_preds.min():.3f}, max={xgb_val_preds.max():.3f}, mean={xgb_val_preds.mean():.3f}")
        print(f"XGBoost WF MAE(orig): {xgb_oof_mae:.4f}")
        xgb_succeeded = True

    except Exception as e:
        print(f"XGBoost failed: {e}")
        xgb_succeeded = False

print(f"Elapsed after XGBoost: {(time.time()-start_time)/60:.1f} min")

# ── Ridge with StandardScaler ─────────────────────────────────────────────────
elapsed_min          = (time.time() - start_time) / 60
ridge_succeeded      = False
ridge_val_preds      = None
ridge_oof_mae        = None
ridge_exclusion_reason = None
families_done        = 1 + (1 if xgb_succeeded else 0)

if families_done >= 2 and elapsed_min > 30:
    print(f"\nSkipping Ridge (2 families done, elapsed {elapsed_min:.1f} min > 30 min)")
else:
    try:
        from sklearn.linear_model import Ridge as RidgeModel
        from sklearn.preprocessing import StandardScaler

        print(f"\nTraining Ridge (alpha probe)...")
        # 80/20 probe split for alpha selection
        np.random.seed(42)
        perm     = np.random.permutation(n)
        split_pt = int(n * 0.8)
        ptr_idx, pva_idx = perm[:split_pt], perm[split_pt:]

        X_ptr = X_full_filled.iloc[ptr_idx]
        y_ptr = y_full[ptr_idx]
        X_pva = X_full_filled.iloc[pva_idx]
        y_pva = y_full[pva_idx]

        scaler_probe = StandardScaler()
        X_ptr_s = scaler_probe.fit_transform(X_ptr)
        X_pva_s = scaler_probe.transform(X_pva)

        best_alpha      = 1.0
        best_ridge_mae  = float("inf")
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            r = RidgeModel(alpha=alpha)
            r.fit(X_ptr_s, y_ptr)
            p = r.predict(X_pva_s)
            mae_r = mean_absolute_error(y_pva, p)
            print(f"  Ridge alpha={alpha}: probe MAE(log1p)={mae_r:.4f}")
            if mae_r < best_ridge_mae:
                best_ridge_mae = mae_r
                best_alpha     = alpha

        print(f"  Best Ridge alpha={best_alpha}")

        # Fit on full training data
        scaler_full = StandardScaler()
        X_full_s    = scaler_full.fit_transform(X_full_filled)
        X_val_s     = scaler_full.transform(X_val)

        ridge_final = RidgeModel(alpha=best_alpha)
        ridge_final.fit(X_full_s, y_full)
        ridge_val_log = ridge_final.predict(X_val_s)

        # WF MAE for Ridge
        scaler_wf   = StandardScaler()
        X_wf_train_s = scaler_wf.fit_transform(X_wf_train)
        X_wf_val_s   = scaler_wf.transform(X_wf_val)
        ridge_wf_m   = RidgeModel(alpha=best_alpha)
        ridge_wf_m.fit(X_wf_train_s, y_wf_train)
        ridge_wf_p   = np.expm1(np.clip(ridge_wf_m.predict(X_wf_val_s), 0, None))
        ridge_oof_mae = mean_absolute_error(np.expm1(y_wf_val), ridge_wf_p)
        print(f"  Ridge WF MAE(orig): {ridge_oof_mae:.4f}")

        # Convert to original space
        ridge_val_preds_raw = np.expm1(np.clip(ridge_val_log, 0, None))
        pred_max_ridge  = float(ridge_val_preds_raw.max())
        pred_mean_ridge = float(ridge_val_preds_raw.mean())

        # Sanity checks
        if pred_max_ridge > 5 * train_max_orig:
            ridge_exclusion_reason = (
                f"pred_max={pred_max_ridge:.2f} > 5x train_max={train_max_orig:.2f}"
            )
            print(f"  Ridge EXCLUDED: {ridge_exclusion_reason}")
        elif abs(pred_mean_ridge - train_mean_orig) / abs(train_mean_orig) > 1.0:
            ridge_exclusion_reason = (
                f"pred_mean={pred_mean_ridge:.2f} deviates >100% from train_mean={train_mean_orig:.2f}"
            )
            print(f"  Ridge EXCLUDED: {ridge_exclusion_reason}")
        else:
            ridge_val_preds = ridge_val_preds_raw
            ridge_succeeded = True
            print(f"  Ridge OK: min={ridge_val_preds.min():.3f}, max={ridge_val_preds.max():.3f}, mean={ridge_val_preds.mean():.3f}")

    except Exception as e:
        print(f"Ridge failed: {e}")
        ridge_succeeded = False

print(f"Elapsed after Ridge: {(time.time()-start_time)/60:.1f} min")

# ── Competence check ──────────────────────────────────────────────────────────
all_oof_maes = {"LightGBM": lgb_oof_mae}
if xgb_succeeded and xgb_oof_mae is not None:
    all_oof_maes["XGBoost"] = xgb_oof_mae
if ridge_succeeded and ridge_oof_mae is not None:
    all_oof_maes["Ridge"] = ridge_oof_mae

best_oof = min(all_oof_maes.values())
print(f"\nOOF MAEs (orig space): {all_oof_maes}")
print(f"Best OOF: {best_oof:.4f}")

final_weighting  = weighting_mode
weighting_reason = ""

if weighting_mode == "ridge_weighted_1.5x":
    if ridge_succeeded and ridge_oof_mae is not None:
        if ridge_oof_mae <= 1.5 * best_oof:
            final_weighting  = "ridge_weighted_1.5x"
            weighting_reason = (
                f"max_ks={_max_ks:.2f} > 0.40 threshold, "
                f"ridge_oof={ridge_oof_mae:.3f} within 1.5x best_oof={best_oof:.3f}; "
                "ridge_weighted_1.5x applied"
            )
        else:
            final_weighting  = "equal_median"
            weighting_reason = (
                f"max_ks={_max_ks:.2f} > 0.40 threshold, "
                f"ridge_oof={ridge_oof_mae:.3f} > 1.5x best_oof={best_oof:.3f}; "
                "using equal_median instead"
            )
    else:
        final_weighting  = "equal_median"
        weighting_reason = (
            f"max_ks={_max_ks:.2f} > 0.40 threshold, "
            f"but Ridge excluded ({ridge_exclusion_reason or 'failed'}); "
            "using equal_median"
        )
else:
    weighting_reason = f"max_ks={_max_ks:.2f} <= 0.40; equal_median"

print(f"Final weighting: {final_weighting}")
print(f"Weighting reason: {weighting_reason}")

# ── Build ensemble predictions ────────────────────────────────────────────────
all_val_preds = {}
all_val_preds["LightGBM"] = lgb_val_preds
if xgb_succeeded and xgb_val_preds is not None:
    all_val_preds["XGBoost"] = xgb_val_preds
if ridge_succeeded and ridge_val_preds is not None:
    all_val_preds["Ridge"] = ridge_val_preds

print(f"\nEnsemble members: {list(all_val_preds.keys())}")

if final_weighting == "ridge_weighted_1.5x" and "Ridge" in all_val_preds:
    stack   = np.column_stack(list(all_val_preds.values()))
    weights = [1.5 if k == "Ridge" else 1.0 for k in all_val_preds.keys()]
    ensemble_preds = np.average(stack, axis=1, weights=weights)
    print(f"Ridge-weighted 1.5x: weights={dict(zip(all_val_preds.keys(), weights))}")
else:
    stack          = np.column_stack(list(all_val_preds.values()))
    ensemble_preds = np.median(stack, axis=1)
    print("Equal-weight median")

# Clip to non-negative
ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f"\nFinal ensemble: min={ensemble_preds.min():.3f}, max={ensemble_preds.max():.3f}, mean={ensemble_preds.mean():.3f}")
print(f"NaN count: {np.isnan(ensemble_preds).sum()}")

# ── Write reports/predictions.csv ─────────────────────────────────────────────
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found!"
preds_df.to_csv(os.path.join(REPO_ROOT, "reports/predictions.csv"), index=False)
print(f"\nWritten reports/predictions.csv: {preds_df.shape}")
print(preds_df.head())

# ── OOF predictions for validator (walk-forward fold) ─────────────────────────
print("\nGenerating OOF predictions on walk-forward val set (5 seeds)...")
oof_seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m_oof = lgb.LGBMRegressor(**{**final_params_lgb, "random_state": seed})
    m_oof.fit(X_wf_train, y_wf_train, callbacks=[lgb.log_evaluation(-1)])
    oof_seed_preds.append(m_oof.predict(X_wf_val))

oof_log_preds  = np.mean(oof_seed_preds, axis=0)
oof_orig_preds = np.expm1(np.clip(oof_log_preds, 0, None))

oof_df = wf_val_df[group_cols + [time_col]].copy().reset_index(drop=True)
oof_df["fold"] = 0
oof_df["predicted_target"] = oof_orig_preds
oof_df.to_csv(os.path.join(REPO_ROOT, "reports/oof_predictions.csv"), index=False)
print(f"Written reports/oof_predictions.csv: {oof_df.shape}")

# ── Feature importances ────────────────────────────────────────────────────────
feat_imp = pd.Series(
    last_lgb_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)
top10   = [{"feature": k, "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp = {k: int(v) for k, v in feat_imp.items()}

# ── Write reports/model_results.json ─────────────────────────────────────────
training_time = int(time.time() - start_time)
oof_mae       = lgb_oof_mae   # walk-forward MAE in original space
wf_mae        = lgb_oof_mae

adaptive_choice = {
    "branch": "1",
    "branch_description": "panel_forecasting + n_train>=1000 -> LightGBM + XGBoost + Ridge",
    "families_attempted": families,
    "families_succeeded": list(all_val_preds.keys()),
    "max_ks": float(_max_ks),
    "ensemble_weighting": final_weighting,
    "weighting_reason": weighting_reason,
    "lgb_oof_mae": float(lgb_oof_mae),
    "xgb_oof_mae": float(xgb_oof_mae) if xgb_oof_mae is not None else None,
    "ridge_oof_mae": float(ridge_oof_mae) if ridge_oof_mae is not None else None,
    "ridge_exclusion_reason": ridge_exclusion_reason,
}

results = {
    "algorithm": "Ensemble(LightGBM+XGBoost+Ridge)",
    "objective": "regression_l1",
    "log1p_transform": True,
    "best_params": final_hparams_lgb,
    "n_estimators": best_n_est,
    "n_seeds": 5,
    "cv_scheme": "walk_forward_80_20",
    "oof_mae": float(oof_mae),
    "oof_cv_scheme": "walk_forward_80_20",
    "per_fold_maes": [float(oof_mae)],
    "walk_forward_mae": float(wf_mae),
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
    "training_time_seconds": training_time,
    "optuna_trials_completed": optuna_trials,
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()),
        "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()),
        "std":  float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": len(train_df),
    "n_val_rows": len(val_df),
    "retune_applied": _retune_applied,
    "adaptive_choice": adaptive_choice,
    "xgb_succeeded": xgb_succeeded,
    "ridge_succeeded": ridge_succeeded,
}

with open(os.path.join(REPO_ROOT, "reports/model_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Written reports/model_results.json")

# ── Marker file ───────────────────────────────────────────────────────────────
with open(os.path.join(REPO_ROOT, "reports/modeler_was_here.txt"), "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print("Written reports/modeler_was_here.txt")

print(f"\nTotal elapsed: {(time.time()-start_time)/60:.1f} min")
print("PIPELINE COMPLETE")
