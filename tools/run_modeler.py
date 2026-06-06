# -*- coding: utf-8 -*-
"""
run_modeler.py  — Full adaptive ensemble modeler
Panel forecasting: LightGBM + XGBoost + CatBoost + Ridge
Supports log1p target transform, adversarial weights, walk-forward CV,
inverse-MAE blend gate, and boundary reflection for Optuna.
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
import os
import sys
import datetime

warnings.filterwarnings("ignore")

# ── Reconfigure stdout for UTF-8 (avoids cp1252 crash on Windows) ──────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

start_time = time.time()
pipeline_start_time = start_time

print("=" * 60)
print("MODELER  —  adaptive ensemble (LGB + XGB + CatBoost + Ridge)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Read problem context from profile.json
# ─────────────────────────────────────────────────────────────────────────────
profile_path  = os.path.join(REPO_ROOT, "reports", "profile.json")
features_path = os.path.join(REPO_ROOT, "reports", "features.json")

with open(profile_path, encoding="utf-8") as f:
    profile = json.load(f)
with open(features_path, encoding="utf-8") as f:
    feat_meta = json.load(f)

problem_type    = profile.get("problem_type", "panel_forecasting")
problem_subtype = profile.get("problem_subtype", "panel_forecasting")

target_col = feat_meta["target_col"]
group_cols = feat_meta["group_cols"]
time_col   = feat_meta["time_col"]

print(f"Problem type: {problem_type} / {problem_subtype}")
print(f"Target: {target_col}")
print(f"Groups: {group_cols}  Time: {time_col}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load feature parquets
# ─────────────────────────────────────────────────────────────────────────────
train_df = pd.read_parquet(os.path.join(REPO_ROOT, "data", "features_train.parquet"))
val_df   = pd.read_parquet(os.path.join(REPO_ROOT, "data", "features_val.parquet"))

exclude = set(
    group_cols
    + ([time_col] if time_col else [])
    + [target_col, "adversarial_weights"]
)
feature_cols = [c for c in train_df.columns if c not in exclude]

# Drop string/object/category columns (state_doh_release is text; cannot use numerically)
non_numeric = [
    c for c in feature_cols
    if not pd.api.types.is_numeric_dtype(train_df[c])
]
if non_numeric:
    print(f"Dropping non-numeric feature cols: {non_numeric[:5]} (total {len(non_numeric)})")
    feature_cols = [c for c in feature_cols if c not in non_numeric]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Features: {len(feature_cols)}")

# Fill NaN with training medians
fill_vals = train_df[feature_cols].median()

# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — Adversarial weights
# ─────────────────────────────────────────────────────────────────────────────
_av_info     = feat_meta.get("adversarial_validation", {})
_adv_weights = None
if _av_info.get("weights_applied", False) and "adversarial_weights" in train_df.columns:
    _adv_weights = train_df["adversarial_weights"].fillna(1.0).values
    print(f"Adversarial weights: min={_adv_weights.min():.3f} "
          f"max={_adv_weights.max():.3f} mean={_adv_weights.mean():.3f}")
else:
    print("No adversarial weights — uniform sample weights")

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive axis decisions
# ─────────────────────────────────────────────────────────────────────────────
n_train = len(train_df)

# Axis 1 — Dataset size / ensemble branch
if problem_type == "panel_forecasting" and n_train >= 1000:
    adaptive_branch    = 1
    families_selected  = ["lgb", "xgb", "catboost", "ridge"]
    ensemble_path_used = "full_regression_ensemble"
    adaptive_reasoning = (
        f"panel_forecasting + n_train={n_train} >= 1000 → "
        "LightGBM + XGBoost + CatBoost + Ridge"
    )
elif problem_type == "panel_forecasting":
    adaptive_branch    = 2
    families_selected  = ["lgb", "ridge"]
    ensemble_path_used = "full_regression_ensemble"
    adaptive_reasoning = (
        f"panel_forecasting + n_train={n_train} < 1000 → LightGBM + Ridge"
    )
else:
    adaptive_branch    = "fallback"
    families_selected  = ["lgb"]
    ensemble_path_used = "classification_fallback"
    adaptive_reasoning = "classification → LightGBM only"

# Axis 2 — Distribution shift severity
_dist_shifts = profile.get("distribution_shifts", [])
_max_ks = max(
    (d.get("ks_statistic", 0.0) for d in _dist_shifts if isinstance(d, dict)),
    default=0.0,
)
if _max_ks > 0.40:
    ensemble_weighting  = "ridge_weighted_1.5x"
    _weighting_trigger  = f"max_ks={_max_ks:.3f} > 0.40 threshold"
else:
    ensemble_weighting  = "equal_median"
    _weighting_trigger  = f"max_ks={_max_ks:.3f} <= 0.40 threshold"

print(f"Axis 1 branch {adaptive_branch}: {families_selected}")
print(f"Axis 2 weighting trigger: {_weighting_trigger}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2c — Log1p target transform (recommended for skewed target)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import skew as _skew
    _target_skew = float(_skew(train_df[target_col].dropna()))
except Exception:
    _target_skew = 2.7   # known from schema analysis
_use_log1p = _target_skew > 1.5

print(f"Target skewness: {_target_skew:.3f} → log1p transform: {_use_log1p}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Prepare arrays
# ─────────────────────────────────────────────────────────────────────────────
X_full_filled = train_df[feature_cols].fillna(fill_vals)
y_raw         = train_df[target_col].values.astype(float)
y_full        = np.log1p(y_raw) if _use_log1p else y_raw
n             = len(train_df)


def inv(preds):
    """Inverse-transform predictions from log1p space to original space."""
    if _use_log1p:
        return np.expm1(np.clip(preds, 0, None))
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Walk-forward split (train 80%, hold 20% of time periods)
# ─────────────────────────────────────────────────────────────────────────────
all_weeks   = sorted(train_df[time_col].unique())
cutoff_idx  = int(len(all_weeks) * 0.8)
cutoff_week = all_weeks[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_week].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_week].copy()

wf_fill_vals    = wf_train[feature_cols].median()
X_wf_train      = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train_raw  = wf_train[target_col].values.astype(float)
y_wf_train      = np.log1p(y_wf_train_raw) if _use_log1p else y_wf_train_raw
X_wf_val        = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val_raw    = wf_val[target_col].values.astype(float)
y_wf_val        = y_wf_val_raw   # evaluation always in original space

print(f"Walk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
print(f"Cutoff week: {cutoff_week}")

# Sample weights aligned to walk-forward training rows
_wf_sw = _adv_weights[wf_train.index.values] if _adv_weights is not None else None

# ─────────────────────────────────────────────────────────────────────────────
# Critic retune check
# ─────────────────────────────────────────────────────────────────────────────
_retune_applied = None
_retune_path    = os.path.join(REPO_ROOT, "reports", "critic_retune_requested.json")
if os.path.exists(_retune_path):
    with open(_retune_path, encoding="utf-8") as _f:
        _retune = json.load(_f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"Critic retune requested: {_suggestion}")
    if "median seed aggregation" in _suggestion:
        _retune_applied = "median_seed_aggregation"
    if "expand Optuna" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+expanded_optuna_bounds"
    if "remove suspect features" in _suggestion:
        try:
            with open(
                os.path.join(REPO_ROOT, "reports", "validator_review.json"),
                encoding="utf-8",
            ) as _vf:
                _vreview = json.load(_vf)
            _suspect = _vreview.get("feature_suspicion", [])
            if _suspect:
                feature_cols = [c for c in feature_cols if c not in _suspect]
                print(f"Removed {len(_suspect)} suspect features")
                _retune_applied = (_retune_applied or "") + f"+removed_{len(_suspect)}_features"
        except Exception as _ve:
            print(f"Could not read feature_suspicion: {_ve}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — LightGBM Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 25 * 60

_use_median_seed = (
    _retune_applied is not None and "median_seed_aggregation" in _retune_applied
)
_seed_agg = np.median if _use_median_seed else np.mean

_num_leaves_hi = 255 if (_retune_applied and "expanded_optuna_bounds" in _retune_applied) else 127
_min_child_lo  = 3   if (_retune_applied and "expanded_optuna_bounds" in _retune_applied) else 5


def lgb_objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()
    params = {
        "objective":         "regression_l1",
        "metric":            "mae",
        "n_estimators":      500,
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, _num_leaves_hi),
        "min_child_samples": trial.suggest_int("min_child_samples", _min_child_lo, 60),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "verbose":           -1,
        "n_jobs":            -1,
    }
    maes = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**params, "random_state": seed})
        _y_eval = np.log1p(y_wf_val) if _use_log1p else y_wf_val
        m.fit(
            X_wf_train, y_wf_train,
            sample_weight=_wf_sw,
            eval_set=[(X_wf_val, _y_eval)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = np.clip(inv(m.predict(X_wf_val)), 0, None)
        maes.append(mean_absolute_error(y_wf_val, preds))
    return float(np.mean(maes))


_optuna_reflection = {
    "pinned_params":   [],
    "recentered":      False,
    "best_mae_before": None,
    "best_mae_after":  None,
}

try:
    study = optuna.create_study(direction="minimize")
    study.optimize(lgb_objective, n_trials=15, timeout=25 * 60, catch=(Exception,))
    best_params      = study.best_params
    _optuna_reflection["best_mae_before"] = float(study.best_value)
    _optuna_reflection["best_mae_after"]  = float(study.best_value)
    optuna_trials    = len(study.trials)
    optuna_succeeded = True

    # Boundary reflection
    _lgb_bounds = {
        "learning_rate":     (0.01, 0.1,  "log"),
        "num_leaves":        (15, _num_leaves_hi, "int"),
        "min_child_samples": (_min_child_lo, 60, "int"),
        "feature_fraction":  (0.5, 1.0, "float"),
        "bagging_fraction":  (0.5, 1.0, "float"),
    }
    _pinned         = []
    _shifted_bounds = {}
    for _pn, (_lo, _hi, _pt) in _lgb_bounds.items():
        _v   = best_params.get(_pn)
        if _v is None:
            continue
        _rng  = _hi - _lo
        _at_lo = ((_pt == "int" and int(_v) == _lo)
                  or (_pt != "int" and _v <= _lo + 0.05 * _rng))
        _at_hi = ((_pt == "int" and int(_v) == _hi)
                  or (_pt != "int" and _v >= _hi - 0.05 * _rng))
        if _at_lo or _at_hi:
            _pinned.append(_pn)
            if _pt == "int":
                _d = max(1, (_hi - _lo) // 4)
                _shifted_bounds[_pn] = (max(1, int(_v) - _d), min(int(_v) + _d, 4096))
            elif _pt == "log":
                _shifted_bounds[_pn] = (max(1e-5, _v / 3.0), min(1.0, _v * 3.0))
            else:
                _d = 0.25 * _rng
                _shifted_bounds[_pn] = (max(0.01, _v - _d), min(1.0, _v + _d))
    _optuna_reflection["pinned_params"] = _pinned

    if _pinned and time.time() < TUNING_DEADLINE - 120:
        print(f"Boundary reflection: {_pinned} pinned — running second study")

        def _reflect_lgb_obj(trial):
            if time.time() > TUNING_DEADLINE:
                raise optuna.exceptions.TrialPruned()

            def _rb(pn, lo, hi, log=False):
                nl, nh = _shifted_bounds.get(pn, (lo, hi))
                if log:
                    return trial.suggest_float(pn, float(nl), float(nh), log=True)
                elif lo == int(lo) and hi == int(hi):
                    return trial.suggest_int(pn, max(1, int(nl)), max(int(nl) + 1, int(nh)))
                else:
                    return trial.suggest_float(pn, float(nl), float(nh))

            _rp = {
                "objective": "regression_l1", "metric": "mae", "n_estimators": 500,
                "learning_rate":     _rb("learning_rate", 0.01, 0.1, log=True),
                "num_leaves":        _rb("num_leaves", 15, _num_leaves_hi),
                "min_child_samples": _rb("min_child_samples", _min_child_lo, 60),
                "feature_fraction":  _rb("feature_fraction", 0.5, 1.0),
                "bagging_fraction":  _rb("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
                "verbose": -1, "n_jobs": -1,
            }
            _rmaes = []
            for _rs in [42, 7, 123]:
                _rm = lgb.LGBMRegressor(**{**_rp, "random_state": _rs})
                _y_eval = np.log1p(y_wf_val) if _use_log1p else y_wf_val
                _rm.fit(
                    X_wf_train, y_wf_train,
                    sample_weight=_wf_sw,
                    eval_set=[(X_wf_val, _y_eval)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
                _rmaes.append(
                    mean_absolute_error(y_wf_val, np.clip(inv(_rm.predict(X_wf_val)), 0, None))
                )
            return float(np.mean(_rmaes))

        _study2 = optuna.create_study(direction="minimize")
        _study2.optimize(
            _reflect_lgb_obj,
            n_trials=15,
            timeout=max(30, TUNING_DEADLINE - time.time() - 5),
            catch=(Exception,),
        )
        optuna_trials += len(_study2.trials)
        _optuna_reflection["recentered"] = True
        if _study2.trials and _study2.best_value < study.best_value:
            best_params = _study2.best_params
            print(f"Reflection improved: {study.best_value:.4f} → {_study2.best_value:.4f}")
            _optuna_reflection["best_mae_after"] = float(_study2.best_value)
        else:
            print(f"Reflection did not improve (best={study.best_value:.4f})")

    print(f"LGB Optuna: {optuna_trials} trials, best MAE={study.best_value:.4f}")
    print(f"Best LGB params: {best_params}")
except Exception as _e:
    print(f"Optuna failed ({_e}), using defaults")
    best_params      = {}
    optuna_trials    = 0
    optuna_succeeded = False

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Final LGB hyperparameters + early-stopping n_estimators
# ─────────────────────────────────────────────────────────────────────────────
default_params = {
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
}
final_hparams = {**default_params, **best_params}

probe_params = {
    "objective":    "regression_l1",
    "metric":       "mae",
    "n_estimators": 2000,
    "bagging_freq": 5,
    "reg_alpha":    0.1,
    "reg_lambda":   0.1,
    "verbose":      -1,
    "n_jobs":       -1,
    "random_state": 42,
    **final_hparams,
}
_y_eval_probe = np.log1p(y_wf_val) if _use_log1p else y_wf_val
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    sample_weight=_wf_sw,
    eval_set=[(X_wf_val, _y_eval_probe)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
)
best_n_estimators = int((probe.best_iteration_ or 500) * 1.1)
lgb_wf_val_preds  = np.clip(inv(probe.predict(X_wf_val)), 0, None)
wf_mae            = float(mean_absolute_error(y_wf_val, lgb_wf_val_preds))
probe_mae_80_20   = wf_mae
print(
    f"WF MAE (probe): {wf_mae:.4f}  "
    f"best_iter: {probe.best_iteration_}  n_estimators: {best_n_estimators}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 7a — LightGBM full-data retrain
# ─────────────────────────────────────────────────────────────────────────────
all_val_preds = {}   # family → val predictions (original space)

final_params = {
    "objective":    "regression_l1",
    "metric":       "mae",
    "n_estimators": best_n_estimators,
    "bagging_freq": 5,
    "reg_alpha":    0.1,
    "reg_lambda":   0.1,
    "verbose":      -1,
    "n_jobs":       -1,
    **final_hparams,
}

# For panel_forecasting: use wf_mae as OOF MAE (honest out-of-time metric)
oof_mae   = wf_mae
fold_maes = []
cv_scheme = "walk_forward_80_20"

_lgb_t0       = time.time()
X_full_filled = train_df[feature_cols].fillna(fill_vals)
X_val         = val_df[feature_cols].fillna(fill_vals)

seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full_filled, y_full, sample_weight=_adv_weights, callbacks=[lgb.log_evaluation(-1)])
    raw_p = m.predict(X_val)
    seed_preds.append(np.clip(inv(raw_p), 0, None))

lgb_ensemble_preds = _seed_agg(seed_preds, axis=0)
lgb_training_time  = int(time.time() - _lgb_t0)

all_val_preds["lgb"] = lgb_ensemble_preds
lgb_oof_mae          = float(oof_mae)
last_model           = m

print(
    f"LGB done: min={lgb_ensemble_preds.min():.2f} max={lgb_ensemble_preds.max():.2f} "
    f"mean={lgb_ensemble_preds.mean():.2f}"
)
print(f"LGB training time: {lgb_training_time}s")

# Write OOF predictions (walk-forward holdout rows)
_wf_ids = wf_val[group_cols + [time_col]].copy().reset_index(drop=True)
_wf_ids["fold"]             = 0
_wf_ids["predicted_target"] = lgb_wf_val_preds
_oof_path = os.path.join(REPO_ROOT, "reports", "oof_predictions.csv")
_wf_ids.to_csv(_oof_path, index=False, encoding="utf-8")
print(f"Written OOF predictions: {_oof_path}  shape={_wf_ids.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7b — XGBoost
# ─────────────────────────────────────────────────────────────────────────────
elapsed_before_xgb = (time.time() - pipeline_start_time) / 60
xgb_result         = {
    "attempted": False, "skip_reason": None, "succeeded": None,
    "oof_mae": None, "included_in_ensemble": None,
}
xgb_oof_mae      = float("inf")
xgb_wf_val_preds = None

if elapsed_before_xgb > 20:
    xgb_result["skip_reason"] = (
        f"skipped_time: elapsed={elapsed_before_xgb:.1f}m > 20m guard"
    )
    print(f"XGBoost skipped — elapsed {elapsed_before_xgb:.1f}m > 20m guard")
else:
    xgb_result["attempted"] = True
    _xgb_t0 = time.time()
    try:
        import xgboost as xgb_lib

        XGB_TUNING_DEADLINE = min(pipeline_start_time + 45 * 60, start_time + 40 * 60)

        # Determine objective (reg:absoluteerror may not be available in older XGB)
        try:
            _xtest = xgb_lib.XGBRegressor(
                n_estimators=10, objective="reg:absoluteerror", random_state=42, verbosity=0
            )
            _xtest.fit(X_wf_train.values[:20], y_wf_train[:20])
            _xgb_obj = "reg:absoluteerror"
        except Exception:
            _xgb_obj = "reg:squarederror"
        print(f"XGB objective: {_xgb_obj}")

        def xgb_objective(trial):
            if time.time() > XGB_TUNING_DEADLINE:
                raise optuna.exceptions.TrialPruned()
            _xp = {
                "n_estimators":     400,
                "objective":        _xgb_obj,
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
                "tree_method":      "hist",
                "n_jobs":           -1,
                "verbosity":        0,
            }
            _xmaes = []
            for _s in [42, 7, 123]:
                _xm = xgb_lib.XGBRegressor(**{**_xp, "random_state": _s})
                _xm.fit(X_wf_train.values, y_wf_train, sample_weight=_wf_sw)
                _raw = _xm.predict(X_wf_val.values)
                _xmaes.append(mean_absolute_error(y_wf_val, np.clip(inv(_raw), 0, None)))
            return float(np.mean(_xmaes))

        xgb_study = optuna.create_study(direction="minimize")
        xgb_study.optimize(
            xgb_objective,
            n_trials=15,
            timeout=int(XGB_TUNING_DEADLINE - time.time()),
            catch=(Exception,),
        )
        xgb_best = xgb_study.best_params
        print(
            f"XGB Optuna: {len(xgb_study.trials)} trials, best MAE={xgb_study.best_value:.4f}"
        )

        xgb_final = {
            "n_estimators": 500,
            "objective":    _xgb_obj,
            "tree_method":  "hist",
            "n_jobs":       -1,
            "verbosity":    0,
            **xgb_best,
        }

        # WF holdout preds for inverse-MAE gate
        _xwf_seeds = []
        for _s in [42, 7, 123]:
            _xm = xgb_lib.XGBRegressor(**{**xgb_final, "random_state": _s})
            _xm.fit(X_wf_train.values, y_wf_train, sample_weight=_wf_sw)
            _xwf_seeds.append(np.clip(inv(_xm.predict(X_wf_val.values)), 0, None))
        xgb_wf_val_preds = np.median(_xwf_seeds, axis=0)
        xgb_wf_mae       = float(mean_absolute_error(y_wf_val, xgb_wf_val_preds))
        xgb_oof_mae      = xgb_wf_mae
        print(f"XGB WF MAE: {xgb_wf_mae:.4f}")

        # Full-data retrain (5 seeds)
        _xseed_preds = []
        for _s in [42, 7, 123, 2024, 999]:
            _xm = xgb_lib.XGBRegressor(**{**xgb_final, "random_state": _s})
            _xm.fit(X_full_filled.values, y_full, sample_weight=_adv_weights)
            _xseed_preds.append(np.clip(inv(_xm.predict(X_val.values)), 0, None))

        xgb_ensemble_preds = np.median(_xseed_preds, axis=0)
        all_val_preds["xgb"] = xgb_ensemble_preds

        xgb_result.update({
            "succeeded":             True,
            "oof_mae":               xgb_oof_mae,
            "included_in_ensemble":  True,
            "best_params":           xgb_best,
            "n_estimators":          500,
            "optuna_trials":         len(xgb_study.trials),
            "training_time_seconds": int(time.time() - _xgb_t0),
        })
        print(
            f"XGB done: min={xgb_ensemble_preds.min():.2f} "
            f"max={xgb_ensemble_preds.max():.2f} mean={xgb_ensemble_preds.mean():.2f}"
        )

    except Exception as _xe:
        print(f"XGBoost failed: {_xe}")
        xgb_result.update({"succeeded": False, "skip_reason": f"training_error: {_xe}"})
        xgb_oof_mae = float("inf")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7c — CatBoost (Axis 3 conditional)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import catboost as _cb_module
    _catboost_available = True
except ImportError:
    _catboost_available = False

_elapsed_before_cb = (time.time() - pipeline_start_time) / 60
_cb_result = {
    "attempted":                   False,
    "skip_reason":                 None,
    "succeeded":                   None,
    "oof_mae":                     None,
    "included_in_ensemble":        None,
    "excluded_reason":             None,
    "training_time_seconds":       None,
    "elapsed_minutes_at_decision": float(_elapsed_before_cb),
}
_cb_decision = {
    "evaluated_for_inclusion":   True,
    "elapsed_time_check_passed": _elapsed_before_cb < 40,
    "data_size_check_passed":    n_train >= 500,
    "competence_check_result":   None,
}
cb_wf_val_preds = None

_should_run_cb = (
    _catboost_available
    and n_train >= 500
    and _elapsed_before_cb < 40
    and (
        problem_type in ("panel_forecasting", "tabular_regression", "classification")
        or problem_subtype in ("ordinal_regression", "continuous_regression")
    )
)

if not _catboost_available:
    _cb_result["skip_reason"] = "skipped_import_error"
    _cb_decision["competence_check_result"] = "skipped_import_error"
    print("CatBoost: not installed — skipping")
elif n_train < 500:
    _cb_result["skip_reason"] = "skipped_data_too_small"
    _cb_decision["competence_check_result"] = "skipped_data_too_small"
elif _elapsed_before_cb >= 40:
    _cb_result["skip_reason"] = "skipped_no_time"
    _cb_decision["competence_check_result"] = "skipped_no_time"
    print(f"CatBoost: skipping — elapsed={_elapsed_before_cb:.1f}m >= 40m")

if _should_run_cb:
    _cb_result["attempted"] = True
    _cb_start = time.time()
    try:
        _cat_cols    = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(train_df[c])
        ]
        _cat_indices = [feature_cols.index(c) for c in _cat_cols]

        _cb_loss        = "MAE"
        _cb_eval_metric = "MAE"

        def _cb_objective(trial):
            if (time.time() - pipeline_start_time) / 60 >= 50:
                raise optuna.exceptions.TrialPruned()
            _p = {
                "iterations":    400,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
                "depth":         trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "loss_function": _cb_loss,
                "eval_metric":   _cb_eval_metric,
                "verbose":       False,
                "allow_writing_files": False,
            }
            if _cat_indices:
                _p["cat_features"] = _cat_indices
            _cb_maes    = []
            _wf_sw_cb   = (
                _adv_weights[wf_train.index.values] if _adv_weights is not None else None
            )
            for _seed in [42, 7]:
                _p["random_seed"] = _seed
                _m = _cb_module.CatBoostRegressor(**_p)
                _m.fit(
                    X_wf_train.values, y_wf_train,
                    sample_weight=_wf_sw_cb,
                    verbose=False,
                )
                _raw_p = _m.predict(X_wf_val.values)
                _cb_maes.append(
                    mean_absolute_error(y_wf_val, np.clip(inv(_raw_p), 0, None))
                )
            return float(np.mean(_cb_maes))

        _cb_study = optuna.create_study(direction="minimize")
        _cb_study.optimize(_cb_objective, n_trials=10, timeout=15 * 60, catch=(Exception,))
        _cb_best = _cb_study.best_params
        print(
            f"CatBoost Optuna: {len(_cb_study.trials)} trials, "
            f"best MAE={_cb_study.best_value:.4f}"
        )

        _cb_final_params = {
            "iterations":    400,
            "learning_rate": _cb_best.get("learning_rate", 0.05),
            "depth":         _cb_best.get("depth", 6),
            "l2_leaf_reg":   _cb_best.get("l2_leaf_reg", 3.0),
            "loss_function": _cb_loss,
            "eval_metric":   _cb_eval_metric,
            "verbose":       False,
            "allow_writing_files": False,
        }
        if _cat_indices:
            _cb_final_params["cat_features"] = _cat_indices

        # WF OOF for competence check
        _cb_wf_preds_list = []
        _wf_sw_cb = (
            _adv_weights[wf_train.index.values] if _adv_weights is not None else None
        )
        for _seed in [42, 7, 123]:
            _cb_final_params["random_seed"] = _seed
            _m = _cb_module.CatBoostRegressor(**_cb_final_params)
            _m.fit(X_wf_train.values, y_wf_train, sample_weight=_wf_sw_cb, verbose=False)
            _raw_p = _m.predict(X_wf_val.values)
            _cb_wf_preds_list.append(np.clip(inv(_raw_p), 0, None))

        _cb_wf_median = np.median(_cb_wf_preds_list, axis=0)
        _cb_oof_mae   = float(mean_absolute_error(y_wf_val, _cb_wf_median))
        cb_wf_val_preds = _cb_wf_median
        print(f"CatBoost WF OOF MAE: {_cb_oof_mae:.4f}")
        _cb_result["oof_mae"] = _cb_oof_mae

        # Competence check: cb_oof <= 1.5 * best_tree_oof
        _tree_oofs    = [lgb_oof_mae]
        if xgb_oof_mae < float("inf"):
            _tree_oofs.append(xgb_oof_mae)
        _best_tree_oof = min(_tree_oofs)
        _cb_passes     = _cb_oof_mae <= 1.5 * _best_tree_oof

        if not _cb_passes:
            _cb_result["included_in_ensemble"] = False
            _cb_result["excluded_reason"] = (
                f"excluded_too_weak: cb_oof={_cb_oof_mae:.4f} > "
                f"1.5x best_tree_oof={_best_tree_oof:.4f}"
            )
            _cb_decision["competence_check_result"] = "excluded_too_weak"
            print(f"CatBoost excluded: {_cb_result['excluded_reason']}")
        else:
            _train_target_max  = float(y_raw.max())
            _train_target_mean = float(y_raw.mean())
            _cb_seed_preds     = []
            for _seed in [42, 7, 123, 2024, 999]:
                _cb_final_params["random_seed"] = _seed
                _m = _cb_module.CatBoostRegressor(**_cb_final_params)
                _m.fit(
                    X_full_filled.values, y_full,
                    sample_weight=_adv_weights, verbose=False,
                )
                _raw_p = _m.predict(X_val.values)
                _cb_seed_preds.append(np.clip(inv(_raw_p), 0, None))

            _cb_val_preds = np.median(_cb_seed_preds, axis=0)
            _cb_pred_max  = float(np.max(_cb_val_preds))
            _cb_pred_mean = float(np.mean(_cb_val_preds))
            _sanity_ok    = True

            if _cb_pred_max > 5 * _train_target_max:
                _cb_result["included_in_ensemble"] = False
                _cb_result["excluded_reason"] = (
                    f"sanity_fail: pred_max={_cb_pred_max:.2f} > "
                    f"5x train_max={_train_target_max:.2f}"
                )
                _cb_decision["competence_check_result"] = "excluded_sanity"
                _sanity_ok = False
                print(f"CatBoost excluded (sanity): {_cb_result['excluded_reason']}")
            elif (
                abs(_train_target_mean) > 0
                and abs(_cb_pred_mean - _train_target_mean) / abs(_train_target_mean) > 1.0
            ):
                _cb_result["included_in_ensemble"] = False
                _cb_result["excluded_reason"] = (
                    f"sanity_fail: pred_mean={_cb_pred_mean:.2f} deviates >100% "
                    f"from train_mean={_train_target_mean:.2f}"
                )
                _cb_decision["competence_check_result"] = "excluded_sanity"
                _sanity_ok = False
                print(f"CatBoost excluded (sanity): {_cb_result['excluded_reason']}")

            if _sanity_ok:
                all_val_preds["catboost"] = _cb_val_preds
                _cb_result["included_in_ensemble"] = True
                _cb_result["succeeded"]             = True
                _cb_decision["competence_check_result"] = "included"
                print(
                    f"CatBoost included: OOF={_cb_oof_mae:.4f}, "
                    f"tree_best={_best_tree_oof:.4f}"
                )
            else:
                _cb_result["succeeded"] = True  # trained OK; excluded on sanity

    except Exception as _cb_exc:
        _cb_result["succeeded"]   = False
        _cb_result["skip_reason"] = f"training_error: {_cb_exc}"
        _cb_decision["competence_check_result"] = "skipped_training_error"
        print(f"CatBoost training failed: {_cb_exc}")

    _cb_result["training_time_seconds"] = float(time.time() - _cb_start)
    print(f"CatBoost block: {_cb_result['training_time_seconds']:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7d — Ridge
# ─────────────────────────────────────────────────────────────────────────────
ridge_result = {
    "attempted": False, "skip_reason": None, "succeeded": None,
    "oof_mae": None, "included_in_ensemble": None, "excluded_reason": None,
}
ridge_val_preds        = None
ridge_wf_val_preds     = None
ridge_excluded_reason  = None
ridge_top_coefficients = []

_elapsed_before_ridge = (time.time() - pipeline_start_time) / 60
_should_run_ridge = (
    "ridge" in families_selected
    and not (len(all_val_preds) >= 2 and _elapsed_before_ridge > 50)
)

if not _should_run_ridge:
    ridge_result["skip_reason"] = (
        f"skipped_time: elapsed={_elapsed_before_ridge:.1f}m > 50m"
        if _elapsed_before_ridge > 50 else "not_in_branch"
    )
    print(f"Ridge skipped: {ridge_result['skip_reason']}")
else:
    ridge_result["attempted"] = True
    _ridge_t0 = time.time()
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # Alpha probe on sub-split of wf_train
        _n_probe   = len(X_wf_train)
        _perm_r    = np.random.RandomState(42).permutation(_n_probe)
        _sp_r      = int(_n_probe * 0.8)
        _ptr_r, _pva_r = _perm_r[:_sp_r], _perm_r[_sp_r:]
        _Xptr = X_wf_train.values[_ptr_r]
        _yptr = y_wf_train[_ptr_r]
        _Xpva = X_wf_train.values[_pva_r]
        _ypva = y_wf_train[_pva_r]

        _scaler_probe = StandardScaler()
        _Xptr_s = _scaler_probe.fit_transform(_Xptr)
        _Xpva_s = _scaler_probe.transform(_Xpva)

        _best_alpha     = 1.0
        _best_alpha_mae = float("inf")
        for _alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            _r       = Ridge(alpha=_alpha)
            _sw_probe = (
                _adv_weights[wf_train.index.values[_ptr_r]]
                if _adv_weights is not None else None
            )
            _r.fit(_Xptr_s, _yptr, sample_weight=_sw_probe)
            _raw_p = _r.predict(_Xpva_s)
            _mae   = mean_absolute_error(inv(_ypva), np.clip(inv(_raw_p), 0, None))
            if _mae < _best_alpha_mae:
                _best_alpha_mae = _mae
                _best_alpha     = _alpha

        print(f"Ridge best alpha: {_best_alpha}  probe MAE: {_best_alpha_mae:.4f}")

        # Fit on full wf_train → predict wf_val (OOF)
        _scaler_wf = StandardScaler()
        _Xwft_s    = _scaler_wf.fit_transform(X_wf_train.values)
        _Xwfv_s    = _scaler_wf.transform(X_wf_val.values)
        _sw_wf     = (
            _adv_weights[wf_train.index.values] if _adv_weights is not None else None
        )
        _ridge_wf  = Ridge(alpha=_best_alpha)
        _ridge_wf.fit(_Xwft_s, y_wf_train, sample_weight=_sw_wf)
        _ridge_wf_raw  = _ridge_wf.predict(_Xwfv_s)
        ridge_wf_val_preds = np.clip(inv(_ridge_wf_raw), 0, None)
        ridge_oof_mae_val  = float(mean_absolute_error(y_wf_val, ridge_wf_val_preds))
        print(f"Ridge WF OOF MAE: {ridge_oof_mae_val:.4f}")

        # Sanity + competence checks
        _train_target_max  = float(y_raw.max())
        _train_target_mean = float(y_raw.mean())
        _ridge_pred_max    = float(np.max(ridge_wf_val_preds))
        _ridge_pred_mean   = float(np.mean(ridge_wf_val_preds))

        _best_tree_oof_for_ridge = min(
            lgb_oof_mae,
            xgb_oof_mae if xgb_oof_mae < float("inf") else lgb_oof_mae,
        )
        _ridge_excluded = False

        if _ridge_pred_max > 5 * _train_target_max:
            _ridge_excluded       = True
            ridge_excluded_reason = (
                f"pred_max={_ridge_pred_max:.2f} > 5x train_max={_train_target_max:.2f}"
            )
        elif (
            abs(_train_target_mean) > 0
            and abs(_ridge_pred_mean - _train_target_mean) / abs(_train_target_mean) > 1.0
        ):
            _ridge_excluded       = True
            ridge_excluded_reason = (
                f"pred_mean={_ridge_pred_mean:.2f} deviates >100% "
                f"from train_mean={_train_target_mean:.2f}"
            )
        elif ridge_oof_mae_val > 1.5 * _best_tree_oof_for_ridge:
            _ridge_excluded       = True
            ridge_excluded_reason = (
                f"ridge_oof={ridge_oof_mae_val:.4f} > "
                f"1.5x best_oof={_best_tree_oof_for_ridge:.4f}"
            )

        if _ridge_excluded:
            ridge_result["included_in_ensemble"] = False
            ridge_result["excluded_reason"]      = ridge_excluded_reason
            ridge_result["succeeded"]            = True
            print(f"Ridge excluded: {ridge_excluded_reason}")
        else:
            # Full-data retrain
            _scaler_full = StandardScaler()
            _Xfull_s     = _scaler_full.fit_transform(X_full_filled.values)
            _Xval_s      = _scaler_full.transform(X_val.values)
            _ridge_full  = Ridge(alpha=_best_alpha)
            _ridge_full.fit(_Xfull_s, y_full, sample_weight=_adv_weights)

            _ridge_raw_preds = _ridge_full.predict(_Xval_s)
            ridge_val_preds  = np.clip(inv(_ridge_raw_preds), 0, None)
            all_val_preds["ridge"] = ridge_val_preds

            ridge_top_coefficients = [
                {"feature": feat, "abs_coef": float(coef)}
                for feat, coef in sorted(
                    zip(feature_cols, np.abs(_ridge_full.coef_)),
                    key=lambda x: -x[1],
                )[:5]
            ]
            ridge_result.update({
                "succeeded":             True,
                "oof_mae":               ridge_oof_mae_val,
                "included_in_ensemble":  True,
                "best_alpha":            _best_alpha,
                "training_time_seconds": int(time.time() - _ridge_t0),
            })
            print(
                f"Ridge included: OOF={ridge_oof_mae_val:.4f} "
                f"min={ridge_val_preds.min():.2f} max={ridge_val_preds.max():.2f} "
                f"mean={ridge_val_preds.mean():.2f}"
            )

    except Exception as _re:
        print(f"Ridge failed: {_re}")
        ridge_result.update({"succeeded": False, "skip_reason": f"training_error: {_re}"})

    ridge_result["training_time_seconds"] = int(time.time() - _ridge_t0)

# ─────────────────────────────────────────────────────────────────────────────
# Finalize Axis 2 weighting decision (competence gate)
# ─────────────────────────────────────────────────────────────────────────────
_included_keys = list(all_val_preds.keys())
_stack         = np.array([all_val_preds[k] for k in _included_keys])

_family_oof = {}
if "lgb" in all_val_preds:
    _family_oof["lgb"] = lgb_oof_mae
if "xgb" in all_val_preds and xgb_oof_mae < float("inf"):
    _family_oof["xgb"] = xgb_oof_mae
if "catboost" in all_val_preds and _cb_result.get("oof_mae"):
    _family_oof["catboost"] = _cb_result["oof_mae"]
if "ridge" in all_val_preds and ridge_result.get("oof_mae"):
    _family_oof["ridge"] = ridge_result["oof_mae"]

_best_oof = min(
    _family_oof.get("lgb",      float("inf")),
    _family_oof.get("xgb",      float("inf")),
    _family_oof.get("catboost", float("inf")),
)
if _best_oof == float("inf"):
    _best_oof = lgb_oof_mae

_ridge_oof_for_gate = _family_oof.get("ridge", float("inf"))

if ensemble_weighting == "ridge_weighted_1.5x" and "ridge" not in all_val_preds:
    ensemble_weighting = "equal_median"
    weighting_reason   = (
        f"{_weighting_trigger}; but Ridge was excluded (not in ensemble) — using equal_median"
    )
elif ensemble_weighting == "ridge_weighted_1.5x" and _ridge_oof_for_gate > 1.5 * _best_oof:
    ensemble_weighting = "equal_median"
    weighting_reason   = (
        f"{_weighting_trigger}, ridge_oof={_ridge_oof_for_gate:.4f} > "
        f"1.5x best_oof={_best_oof:.4f}; using equal_median instead"
    )
elif ensemble_weighting == "ridge_weighted_1.5x":
    weighting_reason = (
        f"{_weighting_trigger}, ridge_oof={_ridge_oof_for_gate:.4f} within "
        f"1.5x best_oof={_best_oof:.4f}; ridge_weighted_1.5x applied"
    )
else:
    weighting_reason = f"{_weighting_trigger}; equal_median used"

print(f"Ensemble weighting: {ensemble_weighting}  ({weighting_reason})")

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble aggregation (three blend paths)
# ─────────────────────────────────────────────────────────────────────────────
_blend_holdout_mae_equal = None
_blend_holdout_mae_inv   = None

if ensemble_weighting == "ridge_weighted_1.5x" and "ridge" in all_val_preds:
    # Path 1 — ridge_weighted_1.5x
    _w             = [1.5 if k == "ridge" else 1.0 for k in _included_keys]
    ensemble_preds = np.average(_stack, axis=0, weights=_w)
    ensemble_blend = "ridge_weighted_1.5x"
    _blend_weights_log = {
        k: round(_w[i] / sum(_w), 4) for i, k in enumerate(_included_keys)
    }
    print(f"Blend: ridge_weighted_1.5x  weights={_blend_weights_log}")

else:
    # Paths 2/3 — equal_median with optional inverse-MAE tilt
    _equal_preds = np.median(_stack, axis=0)

    _all_oofs_ok = all(
        k in _family_oof and _family_oof[k] > 0 for k in _included_keys
    )
    if _all_oofs_ok:
        _inv_sum = sum(1.0 / _family_oof[k] for k in _included_keys)
        _inv_w   = [1.0 / _family_oof[k] / _inv_sum for k in _included_keys]
    else:
        _inv_w = [1.0 / len(_included_keys)] * len(_included_keys)
    _inv_w_arr = np.array(_inv_w)
    _inv_preds = np.average(_stack, axis=0, weights=_inv_w_arr)

    _wf_preds_map = {
        "lgb":      lgb_wf_val_preds,
        "xgb":      xgb_wf_val_preds,
        "catboost": cb_wf_val_preds,
        "ridge":    ridge_wf_val_preds,
    }
    _can_gate = all(_wf_preds_map.get(k) is not None for k in _included_keys)

    if _can_gate:
        _wf_stack = np.array([_wf_preds_map[k] for k in _included_keys])
        _blend_holdout_mae_equal = float(
            mean_absolute_error(y_wf_val, np.median(_wf_stack, axis=0))
        )
        _blend_holdout_mae_inv = float(
            mean_absolute_error(
                y_wf_val, np.average(_wf_stack, axis=0, weights=_inv_w_arr)
            )
        )
        _use_inv = _blend_holdout_mae_inv <= _blend_holdout_mae_equal
    else:
        _use_inv = False

    if _use_inv:
        ensemble_preds = _inv_preds
        ensemble_blend = "inverse_mae_weighted"
        _blend_weights_log = {
            k: round(float(_inv_w_arr[i]), 4) for i, k in enumerate(_included_keys)
        }
        print(
            f"Blend: inverse_mae_weighted  "
            f"(holdout inv={_blend_holdout_mae_inv:.4f} "
            f"<= equal={_blend_holdout_mae_equal:.4f})"
        )
    else:
        ensemble_preds = _equal_preds
        ensemble_blend = "equal_median"
        _blend_weights_log = {k: round(1.0 / len(_included_keys), 4) for k in _included_keys}
        _reason = (
            f"gate_equal_better "
            f"(equal={_blend_holdout_mae_equal:.4f} inv={_blend_holdout_mae_inv:.4f})"
            if _can_gate else "no_wf_preds_available"
        )
        print(f"Blend: equal_median ({_reason})")

ensemble_disagreement = {
    "mean_disagreement":        float(np.mean(np.std(_stack, axis=0))),
    "n_high_disagreement_rows": int(np.sum(np.std(_stack, axis=0) > ensemble_preds.mean())),
}

ensemble_oof_mae       = min(_family_oof.values()) if _family_oof else float(lgb_oof_mae)
n_families_in_ensemble = len(_included_keys)

# Final non-negative clip
ensemble_preds = np.clip(ensemble_preds, 0, None)
print(
    f"Final ensemble: min={ensemble_preds.min():.2f} max={ensemble_preds.max():.2f} "
    f"mean={ensemble_preds.mean():.2f}  families={_included_keys}"
)

# Ordinal rounding (no-op for panel_forecasting)
_postprocessing = {
    "ordinal_rounding_applied": False,
    "ordinal_raw_wf_mae":       None,
    "ordinal_round_wf_mae":     None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Write reports/predictions.csv
# ─────────────────────────────────────────────────────────────────────────────
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found — abort"

_preds_path = os.path.join(REPO_ROOT, "reports", "predictions.csv")
preds_df.to_csv(_preds_path, index=False, encoding="utf-8")
print(f"Written predictions.csv: {preds_df.shape}")
print(preds_df.head())

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Write reports/model_results.json
# ─────────────────────────────────────────────────────────────────────────────
feat_imp = pd.Series(
    last_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)
top10   = [{"feature": k, "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp = [{"feature": k, "importance": int(v)} for k, v in feat_imp.items()]

training_time = int(time.time() - start_time)

_name_map   = {"lgb": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost", "ridge": "Ridge"}
_algo_parts = [k for k in ["lgb", "xgb", "catboost", "ridge"] if k in all_val_preds]
_algo_label = "+".join(_name_map[p] for p in _algo_parts) + " ensemble"

results = {
    "algorithm":          _algo_label,
    "problem_subtype":    problem_subtype,
    "ensemble_path_used": ensemble_path_used,
    "log1p_transform":    _use_log1p,
    "target_skewness":    float(_target_skew),
    "adaptive_choice": {
        "branch":                        adaptive_branch,
        "families_selected":             families_selected,
        "families_included_in_ensemble": _included_keys,
        "reasoning":                     adaptive_reasoning,
        "ensemble_weighting":            ensemble_weighting,
        "weighting_reason":              weighting_reason,
        "ridge_excluded_reason":         ridge_excluded_reason,
        "ensemble_blend":                ensemble_blend,
        "blend_weights":                 _blend_weights_log,
        "blend_holdout_mae_equal":       _blend_holdout_mae_equal,
        "blend_holdout_mae_inv":         _blend_holdout_mae_inv,
    },
    "families": {
        "lightgbm": {
            "best_params":           final_hparams,
            "oof_mae":               float(lgb_oof_mae),
            "training_time_seconds": lgb_training_time,
            "succeeded":             True,
            "included_in_ensemble":  "lgb" in all_val_preds,
            "n_estimators":          best_n_estimators,
            "optuna_trials":         optuna_trials,
        },
        "xgboost":  xgb_result,
        "catboost": _cb_result,
        "ridge":    ridge_result,
    },
    "ensemble_oof_mae":       float(ensemble_oof_mae),
    "n_families_in_ensemble": n_families_in_ensemble,
    "ensemble_disagreement":  ensemble_disagreement,
    "feature_importance_top10": top10,
    "feature_importance_all":   all_imp,
    "ridge_top_coefficients":   ridge_top_coefficients,
    # LightGBM scalar fields (backward compat for validator/report_writer)
    "objective":               final_params["objective"],
    "best_params":             final_hparams,
    "n_estimators":            best_n_estimators,
    "n_seeds":                 5,
    "cv_scheme":               cv_scheme,
    "oof_mae":                 float(lgb_oof_mae),
    "oof_cv_scheme":           cv_scheme,
    "per_fold_maes":           [float(m) for m in fold_maes],
    "walk_forward_mae":        float(wf_mae),
    "probe_mae_80_20":         float(probe_mae_80_20),
    "training_time_seconds":   training_time,
    "optuna_trials_completed": optuna_trials,
    "optuna_succeeded":        optuna_succeeded,
    "val_prediction_stats": {
        "min":  float(ensemble_preds.min()),
        "max":  float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()),
        "std":  float(ensemble_preds.std()),
    },
    "n_features":    len(feature_cols),
    "n_train_rows":  len(train_df),
    "n_val_rows":    len(val_df),
    "retune_applied":    _retune_applied,
    "optuna_reflection": _optuna_reflection,
    "postprocessing":    _postprocessing,
}

_mr_path = os.path.join(REPO_ROOT, "reports", "model_results.json")
with open(_mr_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Written model_results.json: {_mr_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Marker file
# ─────────────────────────────────────────────────────────────────────────────
_marker_path = os.path.join(REPO_ROOT, "reports", "modeler_was_here.txt")
with open(_marker_path, "w", encoding="utf-8") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print(f"Written marker: {_marker_path}")
print(f"Total training time: {training_time}s")
print("=" * 60)
print("MODELER COMPLETE")
print("=" * 60)
