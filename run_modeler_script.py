"""
Modeler pipeline for award_B panel_forecasting problem.
Target: weekly_sales (non-negative, mean=46.3, std=23.9)
Groups: store_id x product_id (1500 groups), week (0-89 train, 90-99 val)
n_train=135000 >= 1000 -> Branch 1: LightGBM + XGBoost + CatBoost + Ridge
max_ks=0.6667 > 0.40 -> ridge_weighted_1.5x (subject to competence check)
No log1p: skew moderate (mean=46.3, std=23.9) — target is non-negative, clip to 0.
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
pipeline_start_time = start_time

REPO_ROOT = "C:/Users/isaac/OneDrive/Desktop/award_B"

print(f"=== Modeler pipeline starting at {datetime.datetime.now().isoformat()} ===")

# ── Step 1: Load context ──────────────────────────────────────────────────────
with open(f"{REPO_ROOT}/reports/features.json") as f:
    feat_meta = json.load(f)

with open(f"{REPO_ROOT}/reports/profile.json") as f:
    profile = json.load(f)

target_col   = feat_meta["target_col"]    # weekly_sales
group_cols   = feat_meta["group_cols"]    # [store_id, product_id]
time_col     = feat_meta["time_col"]      # week
problem_type = profile.get("problem_type", "panel_forecasting")
n_train      = profile.get("n_train_rows", 135000)
n_val        = profile.get("n_val_rows", 15000)

print(f"problem_type={problem_type}, target={target_col}, n_train={n_train}, n_val={n_val}")

# ── Step 2: Load features ─────────────────────────────────────────────────────
train_df = pd.read_parquet(f"{REPO_ROOT}/data/features_train.parquet")
val_df   = pd.read_parquet(f"{REPO_ROOT}/data/features_val.parquet")

exclude = set(group_cols + ([time_col] if time_col else []) + [target_col, "adversarial_weights"])
feature_cols = [c for c in train_df.columns if c not in exclude]

# Drop non-numeric columns that can't be used directly by tree models
non_numeric = [c for c in feature_cols
               if not pd.api.types.is_numeric_dtype(train_df[c])]
if non_numeric:
    print(f"Dropping {len(non_numeric)} non-numeric feature cols: {non_numeric[:5]}")
    feature_cols = [c for c in feature_cols if c not in non_numeric]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Features (numeric): {len(feature_cols)}")

# Training-median fill for NaN defense
fill_vals = train_df[feature_cols].median()

# ── Step 2b: Adversarial weights ──────────────────────────────────────────────
_av_info = feat_meta.get("adversarial_validation", {})
_adv_weights = None
if _av_info.get("weights_applied", False) and "adversarial_weights" in train_df.columns:
    _adv_weights = train_df["adversarial_weights"].fillna(1.0).values
    print(f"Adversarial weights loaded: min={_adv_weights.min():.3f}, "
          f"max={_adv_weights.max():.3f}, mean={_adv_weights.mean():.3f}")
else:
    print("No adversarial weights -- training with uniform sample weights")

# ── Check for critic retune request ───────────────────────────────────────────
_retune_applied = None
if os.path.exists(f"{REPO_ROOT}/reports/critic_retune_requested.json"):
    with open(f"{REPO_ROOT}/reports/critic_retune_requested.json") as _f:
        _retune = json.load(_f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"\nCritic retune requested: {_suggestion}")
    if "median seed aggregation" in _suggestion:
        _retune_applied = "median_seed_aggregation"
    if "expand Optuna" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+expanded_optuna_bounds"
    if "val feature imputation" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+verified_imputation"
    if "np.clip applied after seed aggregation" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+clip_after_ensemble"
    if "remove suspect features" in _suggestion:
        try:
            with open(f"{REPO_ROOT}/reports/validator_review.json") as _vf:
                _vreview = json.load(_vf)
            _suspect = _vreview.get("feature_suspicion", [])
            if _suspect:
                feature_cols = [c for c in feature_cols if c not in _suspect]
                print(f"Removed {len(_suspect)} suspect features")
                _retune_applied = (_retune_applied or "") + f"+removed_{len(_suspect)}_features"
        except Exception as _ve:
            print(f"Could not read feature_suspicion: {_ve}")

_use_median_agg = _retune_applied is not None and "median_seed_aggregation" in _retune_applied
print(f"Seed aggregation: {'median' if _use_median_agg else 'mean'}")

# ── Adaptive Axis 2: distribution shift ───────────────────────────────────────
_dist_shifts = profile.get("distribution_shifts", [])
_max_ks = max(
    (d.get("ks_statistic", 0.0) for d in _dist_shifts if isinstance(d, dict)),
    default=0.0
)
print(f"Max KS statistic: {_max_ks:.4f} (threshold=0.40)")
_use_ridge_weighting = _max_ks > 0.40
print(f"Ridge weighting target: {'ridge_weighted_1.5x' if _use_ridge_weighting else 'equal_median'} "
      f"(before competence check)")

# ── Full arrays ───────────────────────────────────────────────────────────────
X_full_filled = train_df[feature_cols].fillna(fill_vals)
y_full        = train_df[target_col].values
X_val         = val_df[feature_cols].fillna(fill_vals)
n             = len(train_df)
train_mean_orig = float(np.mean(y_full))
train_max_orig  = float(np.max(y_full))

print(f"\nTarget stats: min={y_full.min():.2f}, max={y_full.max():.2f}, "
      f"mean={train_mean_orig:.2f}, std={y_full.std():.2f}")

# ── Step 4: Walk-forward split ─────────────────────────────────────────────────
all_weeks  = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_weeks) * 0.8)
cutoff_week = all_weeks[cutoff_idx]
print(f"\nWalk-forward cutoff: week {cutoff_week} (idx {cutoff_idx}/{len(all_weeks)})")

wf_train = train_df[train_df[time_col] < cutoff_week].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_week].copy()

wf_fill_vals = wf_train[feature_cols].median()
X_wf_train   = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train   = wf_train[target_col].values
X_wf_val     = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val     = wf_val[target_col].values

print(f"WF train: {X_wf_train.shape}, WF val: {X_wf_val.shape}")

# Sample weights for walk-forward train rows
_wf_sw = _adv_weights[wf_train.index.values] if _adv_weights is not None else None

from sklearn.metrics import mean_absolute_error

TUNING_DEADLINE = start_time + 25 * 60  # 25 min cap

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 1: LightGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== FAMILY 1: LightGBM ===")
print("="*60)

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

lgb_succeeded       = False
lgb_oof_mae         = float('inf')
lgb_val_preds       = None
lgb_model_result    = {"succeeded": False, "oof_mae": None, "exclusion_reason": None}
lgb_optuna_trials   = 0
best_lgb_params     = {}
last_lgb_model      = None
lgb_oof_preds_wf    = None   # WF val preds for oof_predictions.csv

try:
    _num_leaves_max = 255 if _retune_applied and "expanded_optuna_bounds" in _retune_applied else 127
    _min_child_min  = 3   if _retune_applied and "expanded_optuna_bounds" in _retune_applied else 5

    def lgb_objective(trial):
        if time.time() > TUNING_DEADLINE:
            raise optuna.exceptions.TrialPruned()
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, _num_leaves_max),
            "min_child_samples": trial.suggest_int("min_child_samples", _min_child_min, 60),
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
                sample_weight=_wf_sw,
                eval_set=[(X_wf_val, y_wf_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            )
            preds = np.clip(m.predict(X_wf_val), 0, None)
            maes.append(mean_absolute_error(y_wf_val, preds))
        return float(np.mean(maes))

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lgb_objective, n_trials=15, timeout=22*60, catch=(Exception,))
    best_lgb_params = study_lgb.best_params
    lgb_optuna_trials = len(study_lgb.trials)
    print(f"LGB Optuna: {lgb_optuna_trials} trials, best MAE={study_lgb.best_value:.4f}")
    print(f"Best LGB params: {best_lgb_params}")

except Exception as e_lgb_opt:
    print(f"LGB Optuna failed ({e_lgb_opt}), using defaults")

default_lgb_params = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_lgb_hparams = {**default_lgb_params, **best_lgb_params}

# Probe for best n_estimators via early stopping
probe_lgb_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 2000,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    **final_lgb_hparams,
}
probe_lgb = lgb.LGBMRegressor(**probe_lgb_params)
probe_lgb.fit(
    X_wf_train, y_wf_train,
    sample_weight=_wf_sw,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_lgb_n = int(probe_lgb.best_iteration_ * 1.1) if probe_lgb.best_iteration_ else 500
lgb_wf_preds = np.clip(probe_lgb.predict(X_wf_val), 0, None)
wf_mae_lgb = mean_absolute_error(y_wf_val, lgb_wf_preds)
print(f"LGB WF MAE: {wf_mae_lgb:.4f}, best_iter={probe_lgb.best_iteration_}, n_est={best_lgb_n}")

lgb_oof_mae      = wf_mae_lgb
wf_mae           = wf_mae_lgb
lgb_oof_preds_wf = lgb_wf_preds   # for oof_predictions.csv

# Final LGB hyperparams
final_lgb_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": best_lgb_n,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    **final_lgb_hparams,
}

# Retrain on full data with 5 seeds
lgb_seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_lgb_params, "random_state": seed})
    m.fit(X_full_filled, y_full, sample_weight=_adv_weights, callbacks=[lgb.log_evaluation(-1)])
    lgb_seed_preds.append(np.clip(m.predict(X_val), 0, None))

if _use_median_agg:
    lgb_val_preds = np.median(lgb_seed_preds, axis=0)
else:
    lgb_val_preds = np.mean(lgb_seed_preds, axis=0)

last_lgb_model = m
lgb_succeeded  = True
lgb_model_result = {
    "succeeded": True,
    "oof_mae": float(lgb_oof_mae),
    "n_estimators": best_lgb_n,
    "best_params": final_lgb_hparams,
    "optuna_trials": lgb_optuna_trials,
}
print(f"LGB val preds: min={lgb_val_preds.min():.2f}, max={lgb_val_preds.max():.2f}, "
      f"mean={lgb_val_preds.mean():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 2: XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== FAMILY 2: XGBoost ===")
print("="*60)

xgb_succeeded    = False
xgb_oof_mae      = float('inf')
xgb_val_preds    = None
xgb_model_result = {"succeeded": False, "oof_mae": None, "exclusion_reason": None}

elapsed_before_xgb = (time.time() - pipeline_start_time) / 60
print(f"Elapsed before XGBoost: {elapsed_before_xgb:.1f}m")

if elapsed_before_xgb > 20:
    print(f"XGBoost SKIPPED: elapsed={elapsed_before_xgb:.1f}m > 20m time guard")
    xgb_model_result["exclusion_reason"] = f"skipped_time_guard: elapsed={elapsed_before_xgb:.1f}m"
else:
    try:
        import xgboost as xgb_lib

        # Test if reg:absoluteerror is available
        try:
            _test_xgb = xgb_lib.XGBRegressor(objective="reg:absoluteerror", n_estimators=1, verbosity=0)
            _test_xgb.fit(X_wf_train.iloc[:10], y_wf_train[:10])
            _xgb_obj = "reg:absoluteerror"
        except Exception:
            _xgb_obj = "reg:squarederror"
        print(f"XGBoost objective: {_xgb_obj}")

        XGB_DEADLINE = pipeline_start_time + 40 * 60

        def xgb_objective(trial):
            if time.time() > XGB_DEADLINE:
                raise optuna.exceptions.TrialPruned()
            params = {
                "objective": _xgb_obj,
                "n_estimators": 400,
                "early_stopping_rounds": 50,  # XGBoost 3.x: must be in constructor
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "n_jobs": -1,
                "verbosity": 0,
            }
            maes = []
            for seed in [42, 7, 123]:
                m_x = xgb_lib.XGBRegressor(**{**params, "random_state": seed})
                m_x.fit(
                    X_wf_train, y_wf_train,
                    sample_weight=_wf_sw,
                    eval_set=[(X_wf_val, y_wf_val)],
                    verbose=False,
                )
                preds = np.clip(m_x.predict(X_wf_val), 0, None)
                maes.append(mean_absolute_error(y_wf_val, preds))
            return float(np.mean(maes))

        study_xgb = optuna.create_study(direction="minimize")
        study_xgb.optimize(xgb_objective, n_trials=15,
                           timeout=max(1, int(XGB_DEADLINE - time.time())),
                           catch=(Exception,))
        best_xgb_params    = study_xgb.best_params
        xgb_optuna_trials  = len(study_xgb.trials)
        print(f"XGB Optuna: {xgb_optuna_trials} trials, best MAE={study_xgb.best_value:.4f}")
        print(f"Best XGB params: {best_xgb_params}")

        # Probe for n_estimators with early stopping (constructor arg for XGBoost 3.x)
        probe_xgb_params = {
            "objective": _xgb_obj,
            "n_estimators": 800,
            "early_stopping_rounds": 50,
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": 42,
            **best_xgb_params,
        }
        probe_xgb = xgb_lib.XGBRegressor(**probe_xgb_params)
        probe_xgb.fit(
            X_wf_train, y_wf_train,
            sample_weight=_wf_sw,
            eval_set=[(X_wf_val, y_wf_val)],
            verbose=False,
        )
        best_xgb_n  = int(probe_xgb.best_iteration * 1.1) if probe_xgb.best_iteration else 400
        xgb_wf_preds = np.clip(probe_xgb.predict(X_wf_val), 0, None)
        xgb_oof_mae  = float(mean_absolute_error(y_wf_val, xgb_wf_preds))
        print(f"XGB WF MAE: {xgb_oof_mae:.4f}, n_est={best_xgb_n}")

        # Full retrain with 5 seeds
        final_xgb_params = {
            "objective": _xgb_obj,
            "n_estimators": best_xgb_n,
            "n_jobs": -1,
            "verbosity": 0,
            **best_xgb_params,
        }
        xgb_seed_preds = []
        for seed in [42, 7, 123, 2024, 999]:
            m_xgb = xgb_lib.XGBRegressor(**{**final_xgb_params, "random_state": seed})
            m_xgb.fit(X_full_filled, y_full, sample_weight=_adv_weights, verbose=False)
            xgb_seed_preds.append(np.clip(m_xgb.predict(X_val), 0, None))

        if _use_median_agg:
            xgb_val_preds = np.median(xgb_seed_preds, axis=0)
        else:
            xgb_val_preds = np.mean(xgb_seed_preds, axis=0)

        xgb_succeeded  = True
        xgb_model_result = {
            "succeeded": True,
            "oof_mae": float(xgb_oof_mae),
            "n_estimators": best_xgb_n,
            "best_params": best_xgb_params,
            "optuna_trials": xgb_optuna_trials,
        }
        print(f"XGB val preds: min={xgb_val_preds.min():.2f}, max={xgb_val_preds.max():.2f}, "
              f"mean={xgb_val_preds.mean():.2f}")

    except ImportError:
        print("XGBoost not available -- skipping")
        xgb_model_result["exclusion_reason"] = "import_error"
    except Exception as e_xgb:
        print(f"XGBoost training failed: {e_xgb}")
        import traceback; traceback.print_exc()
        xgb_model_result["exclusion_reason"] = f"training_error: {e_xgb}"

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 3: CatBoost (Axis 3 conditional)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== FAMILY 3: CatBoost (Axis 3 conditional) ===")
print("="*60)

try:
    import catboost as _cb_module
    _catboost_available = True
except ImportError:
    _catboost_available = False

_elapsed_before_cb = (time.time() - pipeline_start_time) / 60
_cb_result = {
    "attempted": False,
    "skip_reason": None,
    "succeeded": None,
    "oof_mae": None,
    "included_in_ensemble": None,
    "excluded_reason": None,
    "training_time_seconds": None,
    "elapsed_minutes_at_decision": float(_elapsed_before_cb),
}
_cb_decision = {
    "evaluated_for_inclusion": True,
    "elapsed_time_check_passed": _elapsed_before_cb < 40,
    "data_size_check_passed": n_train >= 500,
    "competence_check_result": None,
}

_should_run_cb = (
    _catboost_available
    and n_train >= 500
    and _elapsed_before_cb < 40
    and problem_type in ("panel_forecasting", "tabular_regression", "classification")
)

if not _catboost_available:
    _cb_result["skip_reason"] = "skipped_import_error"
    _cb_decision["competence_check_result"] = "skipped_import_error"
    print("CatBoost: skipping -- not installed")
elif n_train < 500:
    _cb_result["skip_reason"] = "skipped_data_too_small"
    _cb_decision["competence_check_result"] = "skipped_data_too_small"
    print(f"CatBoost: skipping -- n_train={n_train} < 500")
elif _elapsed_before_cb >= 40:
    _cb_result["skip_reason"] = "skipped_no_time"
    _cb_decision["competence_check_result"] = "skipped_no_time"
    print(f"CatBoost: skipping -- elapsed={_elapsed_before_cb:.1f}m >= 40m")
else:
    print(f"CatBoost: conditions met (elapsed={_elapsed_before_cb:.1f}m, n_train={n_train})")

cb_val_preds = None
_cb_start = time.time()

if _should_run_cb:
    _cb_result["attempted"] = True
    try:
        # Identify categorical columns (usually none in this encoded dataset)
        _cat_cols    = [c for c in feature_cols
                        if str(train_df[c].dtype) in ("object", "category")]
        _cat_indices = [feature_cols.index(c) for c in _cat_cols]
        _cb_loss = "MAE"

        def _cb_objective(trial):
            if (time.time() - pipeline_start_time) / 60 >= 50:
                raise optuna.exceptions.TrialPruned()
            _p = {
                "iterations": 400,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "loss_function": _cb_loss,
                "eval_metric": _cb_loss,
                "verbose": False,
                "allow_writing_files": False,
            }
            if _cat_indices:
                _p["cat_features"] = _cat_indices
            _cb_maes = []
            for _seed in [42, 7]:
                _p_s = {**_p, "random_seed": _seed}
                _m = _cb_module.CatBoostRegressor(**_p_s)
                _wf_sw_cb = (_adv_weights[wf_train.index.values]
                             if _adv_weights is not None else None)
                _m.fit(X_wf_train.values, y_wf_train,
                       sample_weight=_wf_sw_cb, verbose=False)
                preds = np.clip(_m.predict(X_wf_val.values), 0, None)
                _cb_maes.append(mean_absolute_error(y_wf_val, preds))
            return float(np.mean(_cb_maes))

        _cb_study = optuna.create_study(direction="minimize")
        _cb_study.optimize(_cb_objective, n_trials=10, timeout=10*60, catch=(Exception,))
        _cb_best = _cb_study.best_params
        print(f"CatBoost Optuna: {len(_cb_study.trials)} trials, best MAE={_cb_study.best_value:.4f}")

        # WF OOF for competence check
        _cb_final_params = {
            "iterations": 400,
            "learning_rate": _cb_best.get("learning_rate", 0.05),
            "depth": _cb_best.get("depth", 6),
            "l2_leaf_reg": _cb_best.get("l2_leaf_reg", 3.0),
            "loss_function": _cb_loss,
            "eval_metric": _cb_loss,
            "verbose": False,
            "allow_writing_files": False,
        }
        if _cat_indices:
            _cb_final_params["cat_features"] = _cat_indices

        _cb_wf_list = []
        for _seed in [42, 7, 123]:
            _pf = {**_cb_final_params, "random_seed": _seed}
            _m = _cb_module.CatBoostRegressor(**_pf)
            _wf_sw_cb = (_adv_weights[wf_train.index.values]
                         if _adv_weights is not None else None)
            _m.fit(X_wf_train.values, y_wf_train,
                   sample_weight=_wf_sw_cb, verbose=False)
            _cb_wf_list.append(np.clip(_m.predict(X_wf_val.values), 0, None))

        _cb_oof_mae = float(mean_absolute_error(y_wf_val, np.median(_cb_wf_list, axis=0)))
        print(f"CatBoost WF OOF MAE: {_cb_oof_mae:.4f}")
        _cb_result["oof_mae"] = _cb_oof_mae

        # Competence check vs best tree OOF
        _tree_oofs = []
        if lgb_succeeded:
            _tree_oofs.append(lgb_oof_mae)
        if xgb_succeeded:
            _tree_oofs.append(xgb_oof_mae)
        _best_tree_oof = min(_tree_oofs) if _tree_oofs else float('inf')
        _cb_passes    = _cb_oof_mae <= 1.5 * _best_tree_oof

        if not _cb_passes:
            _cb_result["included_in_ensemble"] = False
            _cb_result["excluded_reason"] = (
                f"excluded_too_weak: cb_oof={_cb_oof_mae:.4f} > "
                f"1.5x best_tree_oof={_best_tree_oof:.4f}"
            )
            _cb_decision["competence_check_result"] = "excluded_too_weak"
            print(f"CatBoost excluded: {_cb_result['excluded_reason']}")
        else:
            # Full retrain 5 seeds
            _cb_seed_preds = []
            for _seed in [42, 7, 123, 2024, 999]:
                _pf = {**_cb_final_params, "random_seed": _seed}
                _m  = _cb_module.CatBoostRegressor(**_pf)
                _m.fit(X_full_filled.values, y_full,
                       sample_weight=_adv_weights, verbose=False)
                _cb_seed_preds.append(np.clip(_m.predict(X_val.values), 0, None))

            cb_raw = np.clip(np.median(_cb_seed_preds, axis=0), 0, None)

            # Sanity checks
            _cb_pred_max  = float(np.max(cb_raw))
            _cb_pred_mean = float(np.mean(cb_raw))
            _sanity_ok    = True

            if _cb_pred_max > 5 * train_max_orig:
                _cb_result["included_in_ensemble"] = False
                _cb_result["excluded_reason"] = (
                    f"sanity_fail: pred_max={_cb_pred_max:.2f} > 5x train_max={train_max_orig:.2f}"
                )
                _cb_decision["competence_check_result"] = "excluded_sanity"
                _sanity_ok = False
                print(f"CatBoost excluded (sanity): {_cb_result['excluded_reason']}")
            elif (abs(train_mean_orig) > 0 and
                  abs(_cb_pred_mean - train_mean_orig) / abs(train_mean_orig) > 1.0):
                _cb_result["included_in_ensemble"] = False
                _cb_result["excluded_reason"] = (
                    f"sanity_fail: pred_mean={_cb_pred_mean:.2f} deviates >100% "
                    f"from train_mean={train_mean_orig:.2f}"
                )
                _cb_decision["competence_check_result"] = "excluded_sanity"
                _sanity_ok = False
                print(f"CatBoost excluded (sanity): {_cb_result['excluded_reason']}")

            if _sanity_ok:
                cb_val_preds = cb_raw
                _cb_result["included_in_ensemble"] = True
                _cb_result["succeeded"] = True
                _cb_decision["competence_check_result"] = "included"
                print(f"CatBoost INCLUDED: OOF={_cb_oof_mae:.4f}, within 1.5x best={_best_tree_oof:.4f}")
                print(f"  CB val: min={cb_val_preds.min():.2f}, max={cb_val_preds.max():.2f}, "
                      f"mean={cb_val_preds.mean():.2f}")
            else:
                _cb_result["succeeded"] = True  # trained OK; excluded on sanity

    except Exception as _cb_exc:
        _cb_result["succeeded"] = False
        _cb_result["skip_reason"] = f"training_error: {_cb_exc}"
        _cb_decision["competence_check_result"] = "training_error"
        print(f"CatBoost failed: {_cb_exc}")
        import traceback; traceback.print_exc()

    _cb_result["training_time_seconds"] = float(time.time() - _cb_start)
    print(f"CatBoost block done in {_cb_result['training_time_seconds']:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 4: Ridge
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== FAMILY 4: Ridge ===")
print("="*60)

ridge_succeeded    = False
ridge_oof_mae      = float('inf')
ridge_val_preds    = None
ridge_model_result = {"succeeded": False, "oof_mae": None, "exclusion_reason": None}

elapsed_for_ridge = (time.time() - pipeline_start_time) / 60
families_done     = sum([lgb_succeeded, xgb_succeeded,
                         bool(cb_val_preds is not None)])
print(f"Elapsed before Ridge: {elapsed_for_ridge:.1f}m, families done: {families_done}")

if families_done >= 2 and elapsed_for_ridge > 50:
    print(f"Ridge SKIPPED: {families_done} families, elapsed={elapsed_for_ridge:.1f}m > 50m")
    ridge_model_result["exclusion_reason"] = f"skipped_time_guard: elapsed={elapsed_for_ridge:.1f}m"
else:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # Alpha selection via 80/20 probe split
        np.random.seed(42)
        perm     = np.random.permutation(n)
        split_pt = int(n * 0.8)
        ptr_idx, pva_idx = perm[:split_pt], perm[split_pt:]

        X_ptr = X_full_filled.iloc[ptr_idx].fillna(fill_vals).values
        y_ptr = y_full[ptr_idx]
        X_pva = X_full_filled.iloc[pva_idx].fillna(fill_vals).values
        y_pva = y_full[pva_idx]

        scaler_probe = StandardScaler()
        X_ptr_s = scaler_probe.fit_transform(X_ptr)
        X_pva_s = scaler_probe.transform(X_pva)

        best_alpha          = 1.0
        best_ridge_mae_probe = float('inf')
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            _sw_probe = _adv_weights[ptr_idx] if _adv_weights is not None else None
            r = Ridge(alpha=alpha)
            r.fit(X_ptr_s, y_ptr, sample_weight=_sw_probe)
            preds = np.clip(r.predict(X_pva_s), 0, None)
            mae_v = mean_absolute_error(y_pva, preds)
            print(f"  Ridge alpha={alpha}: probe MAE={mae_v:.4f}")
            if mae_v < best_ridge_mae_probe:
                best_ridge_mae_probe = mae_v
                best_alpha = alpha

        print(f"Best Ridge alpha: {best_alpha}, probe MAE={best_ridge_mae_probe:.4f}")

        # WF OOF for competence check
        scaler_wf = StandardScaler()
        X_wf_tr_s = scaler_wf.fit_transform(X_wf_train.fillna(wf_fill_vals).values)
        X_wf_va_s = scaler_wf.transform(X_wf_val.fillna(wf_fill_vals).values)
        _sw_wf_r  = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
        ridge_wf  = Ridge(alpha=best_alpha)
        ridge_wf.fit(X_wf_tr_s, y_wf_train, sample_weight=_sw_wf_r)
        ridge_wf_preds = np.clip(ridge_wf.predict(X_wf_va_s), 0, None)
        ridge_oof_mae  = float(mean_absolute_error(y_wf_val, ridge_wf_preds))
        print(f"Ridge WF OOF MAE: {ridge_oof_mae:.4f}")

        # Competence check vs best tree OOF
        _tree_oofs_r = []
        if lgb_succeeded:
            _tree_oofs_r.append(lgb_oof_mae)
        if xgb_succeeded:
            _tree_oofs_r.append(xgb_oof_mae)
        _best_tree_r = min(_tree_oofs_r) if _tree_oofs_r else float('inf')

        ridge_competent = ridge_oof_mae <= 1.5 * _best_tree_r
        if not ridge_competent:
            print(f"Ridge EXCLUDED (competence): ridge_oof={ridge_oof_mae:.4f} > "
                  f"1.5x best_tree={_best_tree_r:.4f} (threshold={1.5*_best_tree_r:.4f})")
            ridge_model_result["exclusion_reason"] = (
                f"ridge_oof > 1.5x best_family_oof: "
                f"{ridge_oof_mae:.4f} > {1.5*_best_tree_r:.4f}"
            )
        else:
            # Full retrain
            scaler_full = StandardScaler()
            X_full_s   = scaler_full.fit_transform(X_full_filled.fillna(fill_vals).values)
            X_val_s    = scaler_full.transform(X_val.fillna(fill_vals).values)
            ridge_full = Ridge(alpha=best_alpha)
            ridge_full.fit(X_full_s, y_full, sample_weight=_adv_weights)
            ridge_raw  = np.clip(ridge_full.predict(X_val_s), 0, None)

            _ridge_pred_max  = float(np.max(ridge_raw))
            _ridge_pred_mean = float(np.mean(ridge_raw))
            _ridge_sanity_ok = True
            if _ridge_pred_max > 5 * train_max_orig:
                ridge_model_result["exclusion_reason"] = (
                    f"sanity_fail: pred_max={_ridge_pred_max:.2f} > "
                    f"5x train_max={train_max_orig:.2f}"
                )
                _ridge_sanity_ok = False
                print(f"Ridge excluded (sanity): {ridge_model_result['exclusion_reason']}")
            elif (abs(train_mean_orig) > 0 and
                  abs(_ridge_pred_mean - train_mean_orig) / abs(train_mean_orig) > 1.0):
                ridge_model_result["exclusion_reason"] = (
                    f"sanity_fail: pred_mean={_ridge_pred_mean:.2f} deviates >100% "
                    f"from train_mean={train_mean_orig:.2f}"
                )
                _ridge_sanity_ok = False
                print(f"Ridge excluded (sanity): {ridge_model_result['exclusion_reason']}")

            if _ridge_sanity_ok:
                ridge_val_preds = ridge_raw
                ridge_succeeded = True
                ridge_model_result = {
                    "succeeded": True,
                    "oof_mae": float(ridge_oof_mae),
                    "alpha": best_alpha,
                    "exclusion_reason": None,
                }
                print(f"Ridge INCLUDED: val preds: min={ridge_val_preds.min():.2f}, "
                      f"max={ridge_val_preds.max():.2f}, mean={ridge_val_preds.mean():.2f}")

    except Exception as e_ridge:
        print(f"Ridge training failed: {e_ridge}")
        import traceback; traceback.print_exc()
        ridge_model_result["exclusion_reason"] = f"training_error: {e_ridge}"

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== Ensemble Assembly ===")
print("="*60)

all_val_preds = {}
if lgb_succeeded and lgb_val_preds is not None:
    all_val_preds["lgb"] = lgb_val_preds
if xgb_succeeded and xgb_val_preds is not None:
    all_val_preds["xgb"] = xgb_val_preds
if cb_val_preds is not None and _cb_result.get("included_in_ensemble"):
    all_val_preds["catboost"] = cb_val_preds
if ridge_succeeded and ridge_val_preds is not None:
    all_val_preds["ridge"] = ridge_val_preds

print(f"Families in ensemble: {list(all_val_preds.keys())}")

# Axis 2: Ridge weighting with competence check
_ridge_in_ensemble = "ridge" in all_val_preds
ensemble_weighting = "equal_median"
_weighting_reason  = ""

if _use_ridge_weighting and _ridge_in_ensemble:
    _tree_oofs_ens = []
    if lgb_succeeded:
        _tree_oofs_ens.append(lgb_oof_mae)
    if xgb_succeeded:
        _tree_oofs_ens.append(xgb_oof_mae)
    _best_oof_ens = min(_tree_oofs_ens) if _tree_oofs_ens else float('inf')

    if ridge_oof_mae <= 1.5 * _best_oof_ens:
        ensemble_weighting = "ridge_weighted_1.5x"
        _weighting_reason  = (
            f"max_ks={_max_ks:.2f} > 0.40 threshold, "
            f"ridge_oof={ridge_oof_mae:.4f} within 1.5x best_oof={_best_oof_ens:.4f}; "
            f"ridge_weighted_1.5x applied"
        )
    else:
        _weighting_reason = (
            f"max_ks={_max_ks:.2f} > 0.40 threshold, "
            f"ridge_oof={ridge_oof_mae:.4f} > 1.5x best_oof={_best_oof_ens:.4f}; "
            f"using equal_median instead"
        )
elif _use_ridge_weighting and not _ridge_in_ensemble:
    _weighting_reason = (
        f"max_ks={_max_ks:.2f} > 0.40 but Ridge not in ensemble; equal_median"
    )
else:
    _weighting_reason = f"max_ks={_max_ks:.2f} <= 0.40; equal_median"

print(f"Ensemble weighting: {ensemble_weighting}")
print(f"Weighting reason: {_weighting_reason}")

stack        = np.stack(list(all_val_preds.values()), axis=0)
family_names = list(all_val_preds.keys())

if ensemble_weighting == "ridge_weighted_1.5x":
    weights       = np.array([1.5 if k == "ridge" else 1.0 for k in family_names])
    ensemble_preds = np.average(stack, axis=0, weights=weights)
    print(f"Weighted average: weights={dict(zip(family_names, weights))}")
else:
    ensemble_preds = np.median(stack, axis=0)
    print("Equal-weight median applied.")

# Clip to non-negative
ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f"\nFinal ensemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, "
      f"mean={ensemble_preds.mean():.2f}, std={ensemble_preds.std():.2f}")
print(f"NaN count: {np.isnan(ensemble_preds).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# OOF predictions
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Writing OOF predictions ===")
oof_df = wf_val[group_cols + [time_col]].copy().reset_index(drop=True)
oof_df["fold"] = 0
oof_df["predicted_target"] = lgb_oof_preds_wf if lgb_oof_preds_wf is not None else 0.0
oof_df.to_csv(f"{REPO_ROOT}/reports/oof_predictions.csv", index=False)
print(f"Written oof_predictions.csv: {oof_df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# predictions.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Writing predictions.csv ===")
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found!"
preds_df.to_csv(f"{REPO_ROOT}/reports/predictions.csv", index=False)
print(f"Written predictions.csv: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head(5).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# model_results.json
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Writing model_results.json ===")

if last_lgb_model is not None:
    feat_imp_series = pd.Series(
        last_lgb_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    top10   = [{"feature": k, "importance": int(v)}
               for k, v in feat_imp_series.head(10).items()]
    all_imp = {k: int(v) for k, v in feat_imp_series.items()}
else:
    top10   = []
    all_imp = {}

training_time = int(time.time() - start_time)
cv_scheme     = "walk_forward_0.8_train_0.2_val"

results = {
    "algorithm": "LightGBM+XGBoost+CatBoost+Ridge (adaptive ensemble)",
    "log1p_transform_applied": False,
    "objective": "regression_l1",
    "best_params": final_lgb_hparams,
    "n_estimators": best_lgb_n,
    "n_seeds": 5,
    "cv_scheme": cv_scheme,
    "oof_mae": float(lgb_oof_mae),
    "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(lgb_oof_mae)],
    "walk_forward_mae": float(wf_mae),
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
    "training_time_seconds": training_time,
    "optuna_trials_completed": lgb_optuna_trials,
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()),
        "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()),
        "std": float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": len(train_df),
    "n_val_rows": len(val_df),
    "retune_applied": _retune_applied,
    "family_results": {
        "lightgbm": lgb_model_result,
        "xgboost": xgb_model_result,
        "catboost": _cb_result,
        "ridge": ridge_model_result,
    },
    "adaptive_choice": {
        "problem_type": problem_type,
        "n_train": n_train,
        "ensemble_branch": "Branch1_panel_n>=1000_LGB+XGB+CatBoost+Ridge",
        "ensemble_weighting": ensemble_weighting,
        "weighting_reason": _weighting_reason,
        "max_ks_statistic": float(_max_ks),
        "ridge_excluded_reason": ridge_model_result.get("exclusion_reason"),
        "families_in_ensemble": family_names,
        "adversarial_validation": {
            "used_weights": _adv_weights is not None,
            "auc_from_feature_engineer": _av_info.get("auc_train_vs_val"),
            "weight_range_used": None,
        },
        "catboost_decision": _cb_decision,
    },
}

with open(f"{REPO_ROOT}/reports/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Written model_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# Marker file
# ══════════════════════════════════════════════════════════════════════════════
with open(f"{REPO_ROOT}/reports/modeler_was_here.txt", "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    f.write(f"algorithm: LightGBM+XGBoost+CatBoost+Ridge (adaptive ensemble)\n")
    f.write(f"log1p_transform: False\n")
    f.write(f"lgb_oof_mae: {lgb_oof_mae:.4f}\n")
    xgb_str = f"{xgb_oof_mae:.4f}" if xgb_succeeded else "N/A"
    f.write(f"xgb_oof_mae: {xgb_str}\n")
    cb_str = f"{_cb_result.get('oof_mae'):.4f}" if _cb_result.get("oof_mae") else "N/A"
    f.write(f"catboost_oof_mae: {cb_str}\n")
    ridge_str = f"{ridge_oof_mae:.4f}" if ridge_succeeded else "N/A"
    f.write(f"ridge_oof_mae: {ridge_str}\n")
    f.write(f"families_in_ensemble: {family_names}\n")
    f.write(f"ensemble_weighting: {ensemble_weighting}\n")
    f.write(f"total_training_time_seconds: {training_time}\n")

print("Written modeler_was_here.txt")
print(f"\n=== Modeler pipeline complete in {training_time}s ({training_time/60:.1f}m) ===")
print(f"  LGB OOF MAE:  {lgb_oof_mae:.4f}")
if xgb_succeeded:
    print(f"  XGB OOF MAE:  {xgb_oof_mae:.4f}")
if _cb_result.get("oof_mae"):
    print(f"  CB  OOF MAE:  {_cb_result['oof_mae']:.4f}")
if ridge_succeeded:
    print(f"  Ridge OOF MAE:{ridge_oof_mae:.4f}")
print(f"  Ensemble ({ensemble_weighting}): {len(family_names)} families: {family_names}")
print(f"  Val preds: mean={ensemble_preds.mean():.2f}, n={len(ensemble_preds)}")
