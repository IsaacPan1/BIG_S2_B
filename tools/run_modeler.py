# -*- coding: utf-8 -*-
"""
run_modeler.py  —  CatBoost-only modeler with Ridge diagnostic
Panel forecasting: CatBoost is the sole predictor. Ridge is trained on the
walk-forward holdout for diagnostic reporting (OOF MAE + top linear
coefficients) but its predictions never enter the submission.

Preserves:
  - log1p target transform for skewed targets
  - adversarial sample weights
  - walk-forward 80/20 split for honest OOF MAE
  - Optuna tuning with boundary reflection
  - critic retune support
  - recursive multi-step forecasting (CatBoost ensemble)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

start_time = time.time()
pipeline_start_time = start_time

print("=" * 60)
print("MODELER  —  CatBoost predictor + Ridge diagnostic")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Read problem context
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

non_numeric = [
    c for c in feature_cols
    if not pd.api.types.is_numeric_dtype(train_df[c])
]
if non_numeric:
    print(f"Dropping non-numeric feature cols: {non_numeric[:5]} (total {len(non_numeric)})")
    feature_cols = [c for c in feature_cols if c not in non_numeric]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Features: {len(feature_cols)}")

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
# Step 2c — Log1p target transform (skewed target heuristic)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import skew as _skew
    _target_skew = float(_skew(train_df[target_col].dropna()))
except Exception:
    _target_skew = 0.0
_use_log1p = _target_skew > 1.5

print(f"Target skewness: {_target_skew:.3f} → log1p transform: {_use_log1p}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Prepare arrays
# ─────────────────────────────────────────────────────────────────────────────
X_full_filled = train_df[feature_cols].fillna(fill_vals)
y_raw         = train_df[target_col].values.astype(float)
y_full        = np.log1p(y_raw) if _use_log1p else y_raw
n             = len(train_df)
n_train       = n


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

_wf_sw = _adv_weights[wf_train.index.values] if _adv_weights is not None else None

# ─────────────────────────────────────────────────────────────────────────────
# Step 4b — Adaptive branch logging (CatBoost is always the predictor now)
# ─────────────────────────────────────────────────────────────────────────────
adaptive_branch    = 1
families_selected  = ["catboost", "ridge_diagnostic"]
ensemble_path_used = "catboost_only"
adaptive_reasoning = (
    f"CatBoost is the sole predictor; Ridge runs as a diagnostic baseline "
    f"(not in submission). n_train={n_train}."
)
ensemble_weighting = "single_model"
_weighting_trigger = "single-model pipeline"
weighting_reason   = "single CatBoost predictor; no ensemble weighting required"

print(f"Branch {adaptive_branch}: {families_selected}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Critic retune check
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
                _retune_applied = (
                    (_retune_applied or "") + f"+removed_{len(_suspect)}_features"
                )
        except Exception as _ve:
            print(f"Could not read feature_suspicion: {_ve}")

_use_median_seed = (
    _retune_applied is not None and "median_seed_aggregation" in _retune_applied
)
_seed_agg = np.median if _use_median_seed else np.mean

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — CatBoost Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────
import catboost as _cb_module
from sklearn.metrics import mean_absolute_error
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 25 * 60

_expanded = _retune_applied is not None and "expanded_optuna_bounds" in _retune_applied
_depth_hi = 10 if _expanded else 8
_lr_lo    = 0.005 if _expanded else 0.02
_lr_hi    = 0.15  if _expanded else 0.10

_cb_loss        = "MAE"
_cb_eval_metric = "MAE"


def cb_objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()
    params = {
        "iterations":    400,
        "learning_rate": trial.suggest_float("learning_rate", _lr_lo, _lr_hi, log=True),
        "depth":         trial.suggest_int("depth", 4, _depth_hi),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": _cb_loss,
        "eval_metric":   _cb_eval_metric,
        "verbose":       False,
        "allow_writing_files": False,
    }
    maes = []
    for seed in [42, 7, 123]:
        params["random_seed"] = seed
        m = _cb_module.CatBoostRegressor(**params)
        m.fit(X_wf_train.values, y_wf_train, sample_weight=_wf_sw, verbose=False)
        preds = np.clip(inv(m.predict(X_wf_val.values)), 0, None)
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
    study.optimize(cb_objective, n_trials=15, timeout=25 * 60, catch=(Exception,))
    best_params      = study.best_params
    _optuna_reflection["best_mae_before"] = float(study.best_value)
    _optuna_reflection["best_mae_after"]  = float(study.best_value)
    optuna_trials    = len(study.trials)
    optuna_succeeded = True

    # Boundary reflection
    _cb_bounds = {
        "learning_rate": (_lr_lo, _lr_hi, "log"),
        "depth":         (4, _depth_hi, "int"),
        "l2_leaf_reg":   (1.0, 10.0, "float"),
    }
    _pinned         = []
    _shifted_bounds = {}
    for _pn, (_lo, _hi, _pt) in _cb_bounds.items():
        _v = best_params.get(_pn)
        if _v is None:
            continue
        _rng = _hi - _lo
        _at_lo = ((_pt == "int" and int(_v) == _lo)
                  or (_pt != "int" and _v <= _lo + 0.05 * _rng))
        _at_hi = ((_pt == "int" and int(_v) == _hi)
                  or (_pt != "int" and _v >= _hi - 0.05 * _rng))
        if _at_lo or _at_hi:
            _pinned.append(_pn)
            if _pt == "int":
                _d = max(1, (_hi - _lo) // 4)
                _shifted_bounds[_pn] = (max(1, int(_v) - _d), min(int(_v) + _d, 16))
            elif _pt == "log":
                _shifted_bounds[_pn] = (max(1e-5, _v / 3.0), min(1.0, _v * 3.0))
            else:
                _d = 0.25 * _rng
                _shifted_bounds[_pn] = (max(0.1, _v - _d), min(20.0, _v + _d))
    _optuna_reflection["pinned_params"] = _pinned

    if _pinned and time.time() < TUNING_DEADLINE - 120:
        print(f"Boundary reflection: {_pinned} pinned — running second study")

        def _reflect_cb_obj(trial):
            if time.time() > TUNING_DEADLINE:
                raise optuna.exceptions.TrialPruned()

            def _rb(pn, lo, hi, log=False, is_int=False):
                nl, nh = _shifted_bounds.get(pn, (lo, hi))
                if log:
                    return trial.suggest_float(pn, float(nl), float(nh), log=True)
                elif is_int:
                    return trial.suggest_int(pn, max(1, int(nl)), max(int(nl) + 1, int(nh)))
                else:
                    return trial.suggest_float(pn, float(nl), float(nh))

            _rp = {
                "iterations":    400,
                "learning_rate": _rb("learning_rate", _lr_lo, _lr_hi, log=True),
                "depth":         _rb("depth", 4, _depth_hi, is_int=True),
                "l2_leaf_reg":   _rb("l2_leaf_reg", 1.0, 10.0),
                "loss_function": _cb_loss,
                "eval_metric":   _cb_eval_metric,
                "verbose":       False,
                "allow_writing_files": False,
            }
            _rmaes = []
            for _rs in [42, 7, 123]:
                _rp["random_seed"] = _rs
                _rm = _cb_module.CatBoostRegressor(**_rp)
                _rm.fit(X_wf_train.values, y_wf_train,
                        sample_weight=_wf_sw, verbose=False)
                _rmaes.append(
                    mean_absolute_error(y_wf_val,
                                        np.clip(inv(_rm.predict(X_wf_val.values)), 0, None))
                )
            return float(np.mean(_rmaes))

        _study2 = optuna.create_study(direction="minimize")
        _study2.optimize(
            _reflect_cb_obj,
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

    print(f"CatBoost Optuna: {optuna_trials} trials, best MAE={study.best_value:.4f}")
    print(f"Best CatBoost params: {best_params}")
except Exception as _e:
    print(f"Optuna failed ({_e}), using defaults")
    best_params      = {}
    optuna_trials    = 0
    optuna_succeeded = False

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — CatBoost final hyperparameters + WF probe for iterations
# ─────────────────────────────────────────────────────────────────────────────
default_params = {
    "learning_rate": 0.05,
    "depth":         6,
    "l2_leaf_reg":   3.0,
}
final_hparams = {**default_params, **best_params}

probe_params = {
    "iterations":    2000,
    "loss_function": _cb_loss,
    "eval_metric":   _cb_eval_metric,
    "verbose":       False,
    "allow_writing_files": False,
    "random_seed":   42,
    "od_type":       "Iter",
    "od_wait":       100,
    **final_hparams,
}

probe = _cb_module.CatBoostRegressor(**probe_params)
probe.fit(
    X_wf_train.values, y_wf_train,
    sample_weight=_wf_sw,
    eval_set=(X_wf_val.values, np.log1p(y_wf_val) if _use_log1p else y_wf_val),
    verbose=False,
)
_best_iter        = probe.get_best_iteration() or 500
best_n_estimators = int(_best_iter * 1.1)
cb_wf_val_preds   = np.clip(inv(probe.predict(X_wf_val.values)), 0, None)
wf_mae            = float(mean_absolute_error(y_wf_val, cb_wf_val_preds))
probe_mae_80_20   = wf_mae
print(
    f"WF MAE (probe): {wf_mae:.4f}  best_iter: {_best_iter}  "
    f"n_estimators: {best_n_estimators}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 7b — CatBoost full-data retrain (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────
all_val_preds = {}   # family → val predictions (original space) — predictor families only

final_params = {
    "iterations":    best_n_estimators,
    "loss_function": _cb_loss,
    "eval_metric":   _cb_eval_metric,
    "verbose":       False,
    "allow_writing_files": False,
    **final_hparams,
}

oof_mae   = wf_mae
fold_maes = []
cv_scheme = "walk_forward_80_20"

_cb_t0 = time.time()
X_full_filled = train_df[feature_cols].fillna(fill_vals)
X_val         = val_df[feature_cols].fillna(fill_vals)

_cb_trained_models = []
_cb_seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    params_s = {**final_params, "random_seed": seed}
    m = _cb_module.CatBoostRegressor(**params_s)
    m.fit(X_full_filled.values, y_full, sample_weight=_adv_weights, verbose=False)
    _cb_trained_models.append(m)
    raw_p = m.predict(X_val.values)
    _cb_seed_preds.append(np.clip(inv(raw_p), 0, None))

cb_ensemble_preds = _seed_agg(_cb_seed_preds, axis=0)
cb_training_time  = int(time.time() - _cb_t0)

all_val_preds["catboost"] = cb_ensemble_preds
cb_oof_mae                = float(oof_mae)
last_model                = m

print(
    f"CatBoost done: min={cb_ensemble_preds.min():.2f} "
    f"max={cb_ensemble_preds.max():.2f} mean={cb_ensemble_preds.mean():.2f}"
)
print(f"CatBoost training time: {cb_training_time}s")

# Write OOF predictions (walk-forward holdout rows)
_wf_ids = wf_val[group_cols + [time_col]].copy().reset_index(drop=True)
_wf_ids["fold"]             = 0
_wf_ids["predicted_target"] = cb_wf_val_preds
_oof_path = os.path.join(REPO_ROOT, "reports", "oof_predictions.csv")
_wf_ids.to_csv(_oof_path, index=False, encoding="utf-8")
print(f"Written OOF predictions: {_oof_path}  shape={_wf_ids.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Ridge diagnostic (NOT in submission)
# ─────────────────────────────────────────────────────────────────────────────
# Ridge is trained on the walk-forward holdout for:
#   - a linear-baseline OOF MAE to compare against CatBoost
#   - top absolute coefficients (interpretability signal)
# Its predictions are NOT written to predictions.csv and NOT blended.
ridge_diagnostic = {
    "attempted":          False,
    "succeeded":          None,
    "oof_mae":            None,
    "best_alpha":         None,
    "top_coefficients":   [],
    "training_time_seconds": None,
    "role":               "diagnostic_only",
    "skip_reason":        None,
}
ridge_top_coefficients = []

_ridge_t0 = time.time()
try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ridge_diagnostic["attempted"] = True

    # Alpha probe on sub-split of wf_train
    _n_probe = len(X_wf_train)
    _perm_r  = np.random.RandomState(42).permutation(_n_probe)
    _sp_r    = int(_n_probe * 0.8)
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
        _r = Ridge(alpha=_alpha)
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

    # Fit on full wf_train → predict wf_val for diagnostic OOF MAE
    _scaler_wf = StandardScaler()
    _Xwft_s = _scaler_wf.fit_transform(X_wf_train.values)
    _Xwfv_s = _scaler_wf.transform(X_wf_val.values)
    _sw_wf  = (
        _adv_weights[wf_train.index.values] if _adv_weights is not None else None
    )
    _ridge_wf = Ridge(alpha=_best_alpha)
    _ridge_wf.fit(_Xwft_s, y_wf_train, sample_weight=_sw_wf)
    _ridge_wf_raw  = _ridge_wf.predict(_Xwfv_s)
    _ridge_wf_pred = np.clip(inv(_ridge_wf_raw), 0, None)
    ridge_oof_mae  = float(mean_absolute_error(y_wf_val, _ridge_wf_pred))
    print(f"Ridge diagnostic WF OOF MAE: {ridge_oof_mae:.4f}")

    ridge_top_coefficients = [
        {"feature": feat, "abs_coef": float(coef)}
        for feat, coef in sorted(
            zip(feature_cols, np.abs(_ridge_wf.coef_)),
            key=lambda x: -x[1],
        )[:10]
    ]

    ridge_diagnostic.update({
        "succeeded":             True,
        "oof_mae":               ridge_oof_mae,
        "best_alpha":            _best_alpha,
        "top_coefficients":      ridge_top_coefficients,
        "training_time_seconds": int(time.time() - _ridge_t0),
    })
except Exception as _re:
    print(f"Ridge diagnostic failed: {_re}")
    ridge_diagnostic.update({
        "succeeded":             False,
        "skip_reason":           f"training_error: {_re}",
        "training_time_seconds": int(time.time() - _ridge_t0),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Final predictions (CatBoost only)
# ─────────────────────────────────────────────────────────────────────────────
_included_keys     = ["catboost"]
ensemble_preds     = np.clip(cb_ensemble_preds, 0, None)
ensemble_blend     = "single_catboost"
_blend_weights_log = {"catboost": 1.0}
_blend_holdout_mae_equal = None
_blend_holdout_mae_inv   = None

# Disagreement: trivial for single-model — std across seed predictions instead.
_seed_stack = np.array(_cb_seed_preds)
ensemble_disagreement = {
    "mean_disagreement":        float(np.mean(np.std(_seed_stack, axis=0))),
    "n_high_disagreement_rows": int(
        np.sum(np.std(_seed_stack, axis=0) > ensemble_preds.mean())
    ),
}

ensemble_oof_mae       = float(cb_oof_mae)
n_families_in_ensemble = 1

print(
    f"Final CatBoost preds: min={ensemble_preds.min():.2f} "
    f"max={ensemble_preds.max():.2f} mean={ensemble_preds.mean():.2f}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Recursive multi-step forecasting (CatBoost only)
# ─────────────────────────────────────────────────────────────────────────────
_lag_forecasting = {
    "method_used":             "imputation",
    "imputation_holdout_mae":  None,
    "recursive_holdout_mae":   None,
    "per_step_mae_imputation": [],
    "per_step_mae_recursive":  [],
    "notes":                   "not_attempted",
}

try:
    _rf_ord_col = "period_id_ord"
    _has_ord    = (
        problem_type == "panel_forecasting"
        and group_cols
        and time_col
        and _rf_ord_col in train_df.columns
        and _rf_ord_col in val_df.columns
    )

    if not _has_ord:
        _lag_forecasting["notes"] = (
            f"skipped: problem_type={problem_type}, "
            f"ord_col_present={_rf_ord_col in train_df.columns}"
        )
    else:
        _rf_lag_periods  = feat_meta.get("lag_periods",    [1, 2, 3, 4])
        _rf_roll_windows = feat_meta.get("rolling_windows", [4, 8])
        _rf_lag_cols   = [f"lag_{k}"       for k in _rf_lag_periods  if f"lag_{k}"       in feature_cols]
        _rf_rmean_cols = [f"roll_mean_{w}" for w in _rf_roll_windows if f"roll_mean_{w}" in feature_cols]
        _rf_rstd_cols  = [f"roll_std_{w}"  for w in _rf_roll_windows if f"roll_std_{w}"  in feature_cols]
        _rf_ceiling      = float(train_df[target_col].max()) * 10.0
        _rf_ceiling_hits = 0

        _fc_idx = {c: i for i, c in enumerate(feature_cols)}
        _fv_arr = np.array([fill_vals.get(c, 0.0) for c in feature_cols],
                           dtype=np.float64)

        def _rf_predict(X_df: pd.DataFrame) -> np.ndarray:
            """Median-seed CatBoost prediction in original space."""
            if not _cb_trained_models:
                return np.zeros(len(X_df))
            _sp = [
                np.clip(inv(m.predict(X_df.values)), 0, None)
                for m in _cb_trained_models
            ]
            return np.clip(_seed_agg(_sp, axis=0), 0, None)

        def _inject_target_feats(feat_arr: np.ndarray, local_i: int,
                                 hist: list) -> None:
            """Overwrite lag/rolling columns in feat_arr[local_i] from hist."""
            for k in _rf_lag_periods:
                ci = _fc_idx.get(f"lag_{k}")
                if ci is not None:
                    feat_arr[local_i, ci] = (
                        float(hist[-k]) if len(hist) >= k
                        else (float(hist[-1]) if hist else 0.0)
                    )
            for w in _rf_roll_windows:
                win = hist[-w:] if len(hist) >= w else hist
                ci_m = _fc_idx.get(f"roll_mean_{w}")
                ci_s = _fc_idx.get(f"roll_std_{w}")
                if ci_m is not None:
                    feat_arr[local_i, ci_m] = float(np.mean(win)) if win else 0.0
                if ci_s is not None:
                    feat_arr[local_i, ci_s] = (
                        float(np.std(win, ddof=1)) if len(win) >= 2 else 0.0
                    )

        def _build_gkeys(df: pd.DataFrame) -> list:
            if len(group_cols) == 1:
                return list(df[group_cols[0]])
            return [tuple(r) for r in df[group_cols].values.tolist()]

        _wf_v = wf_val.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
        _wf_t = wf_train.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
        _wf_train_max_ord = int(_wf_t[_rf_ord_col].max())
        _wf_val_periods   = sorted(_wf_v[_rf_ord_col].unique().tolist())
        _n_h_steps        = len(_wf_val_periods)
        _wf_step_nums     = (_wf_v[_rf_ord_col].values - _wf_train_max_ord).astype(int)
        _y_wf_truth       = _wf_v[target_col].values

        _hist_seed: dict = {}
        _gb_cols = group_cols if len(group_cols) > 1 else group_cols[0]
        for _gk, _gdf in _wf_t.groupby(_gb_cols):
            _hist_seed[_gk] = list(_gdf[target_col].values)

        _wf_gkeys = _build_gkeys(_wf_v)

        _wf_base = _wf_v[feature_cols].values.astype(np.float64)
        _nm = np.isnan(_wf_base)
        _wf_base[_nm] = np.take(_fv_arr, np.where(_nm)[1])

        # (A) Imputation holdout
        print("Recursive forecasting: scoring imputation holdout…")
        _last_known: dict = {
            _gk: float(v[-1]) if v else 0.0 for _gk, v in _hist_seed.items()
        }
        _imp_base = _wf_base.copy()
        for _col in _rf_lag_cols + _rf_rmean_cols:
            ci = _fc_idx.get(_col)
            if ci is None:
                continue
            for _ri, _gk in enumerate(_wf_gkeys):
                _imp_base[_ri, ci] = _last_known.get(_gk, _fv_arr[ci])
        for _col in _rf_rstd_cols:
            ci = _fc_idx.get(_col)
            if ci is not None:
                _imp_base[:, ci] = 0.0
        _imp_preds_h  = _rf_predict(pd.DataFrame(_imp_base, columns=feature_cols))
        _imp_hold_mae = float(mean_absolute_error(_y_wf_truth, _imp_preds_h))
        _per_step_imp = [
            float(mean_absolute_error(_y_wf_truth[_wf_step_nums == s],
                                      _imp_preds_h[_wf_step_nums == s]))
            for s in range(1, _n_h_steps + 1)
            if (_wf_step_nums == s).any()
        ]
        print(f"  Imputation holdout MAE: {_imp_hold_mae:.4f}")
        print(f"  Per-step (imp): {[round(v,3) for v in _per_step_imp]}")

        # (B) Recursive holdout
        print("Recursive forecasting: scoring recursive holdout…")
        _rec_preds_h = np.zeros(len(_wf_v))
        _rec_hist: dict = {_gk: list(v) for _gk, v in _hist_seed.items()}

        for _pord in _wf_val_periods:
            _pm    = (_wf_v[_rf_ord_col] == _pord).values
            _pidxs = np.where(_pm)[0]
            _sf    = _wf_base[_pidxs].copy()
            for _li, _ri in enumerate(_pidxs):
                _inject_target_feats(_sf, _li, _rec_hist.get(_wf_gkeys[_ri], []))
            _sp = _rf_predict(pd.DataFrame(_sf, columns=feature_cols))
            _cf = _sp > _rf_ceiling
            if _cf.any():
                _rf_ceiling_hits += int(_cf.sum())
                _sp = np.clip(_sp, 0, _rf_ceiling)
            _rec_preds_h[_pidxs] = _sp
            for _li, _ri in enumerate(_pidxs):
                _gk = _wf_gkeys[_ri]
                _rec_hist.setdefault(_gk, []).append(float(_sp[_li]))

        _rec_hold_mae = float(mean_absolute_error(_y_wf_truth, _rec_preds_h))
        _per_step_rec = [
            float(mean_absolute_error(_y_wf_truth[_wf_step_nums == s],
                                      _rec_preds_h[_wf_step_nums == s]))
            for s in range(1, _n_h_steps + 1)
            if (_wf_step_nums == s).any()
        ]
        print(f"  Recursive holdout MAE: {_rec_hold_mae:.4f}")
        print(f"  Per-step (rec): {[round(v,3) for v in _per_step_rec]}")
        if _rf_ceiling_hits:
            print(f"  Ceiling triggered {_rf_ceiling_hits} time(s) on holdout")

        _rec_wins = _rec_hold_mae <= _imp_hold_mae
        print(f"  Winner: {'RECURSIVE' if _rec_wins else 'IMPUTATION'} "
              f"(Δ={abs(_rec_hold_mae - _imp_hold_mae):.4f})")

        _method_used   = "imputation"
        _n_val_steps   = int(val_df[_rf_ord_col].nunique())
        _val_ceil_hits = 0

        if _rec_wins:
            print("Recursive forecasting: generating recursive val predictions…")
            _val_s = val_df.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
            _val_orig_positions = (
                val_df.sort_values(group_cols + [_rf_ord_col]).index.tolist()
            )
            _val_periods = sorted(_val_s[_rf_ord_col].unique().tolist())
            _val_gkeys   = _build_gkeys(_val_s)

            _train_s2 = train_df.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
            _full_hist: dict = {}
            for _gk, _gdf in _train_s2.groupby(_gb_cols):
                _full_hist[_gk] = list(_gdf[target_col].values)

            _val_base = _val_s[feature_cols].values.astype(np.float64)
            _nm2 = np.isnan(_val_base)
            _val_base[_nm2] = np.take(_fv_arr, np.where(_nm2)[1])

            _rec_val_preds  = np.zeros(len(_val_s))
            _val_hist: dict = {_gk: list(v) for _gk, v in _full_hist.items()}

            for _pord in _val_periods:
                _pm    = (_val_s[_rf_ord_col] == _pord).values
                _pidxs = np.where(_pm)[0]
                _sf    = _val_base[_pidxs].copy()
                for _li, _ri in enumerate(_pidxs):
                    _inject_target_feats(_sf, _li, _val_hist.get(_val_gkeys[_ri], []))
                _sp = _rf_predict(pd.DataFrame(_sf, columns=feature_cols))
                _cf = _sp > _rf_ceiling
                if _cf.any():
                    _val_ceil_hits += int(_cf.sum())
                    _sp = np.clip(_sp, 0, _rf_ceiling)
                _rec_val_preds[_pidxs] = _sp
                for _li, _ri in enumerate(_pidxs):
                    _gk = _val_gkeys[_ri]
                    _val_hist.setdefault(_gk, []).append(float(_sp[_li]))

            _aligned = np.zeros(len(val_df))
            for _new_pos, _orig_idx in enumerate(_val_orig_positions):
                _aligned[_orig_idx] = _rec_val_preds[_new_pos]

            if np.isnan(_aligned).any() or (np.array(_aligned) < 0).any():
                print("  WARNING: invalid recursive val preds — keeping imputation")
            else:
                ensemble_preds = np.clip(_aligned, 0, None)
                _method_used   = "recursive"
                print(f"  Recursive val: min={ensemble_preds.min():.2f} "
                      f"max={ensemble_preds.max():.2f} mean={ensemble_preds.mean():.2f}")
                if _val_ceil_hits:
                    print(f"  WARNING: val ceiling triggered {_val_ceil_hits} time(s)")

        _lag_forecasting = {
            "method_used":             _method_used,
            "imputation_holdout_mae":  _imp_hold_mae,
            "recursive_holdout_mae":   _rec_hold_mae,
            "per_step_mae_imputation": _per_step_imp,
            "per_step_mae_recursive":  _per_step_rec,
            "n_holdout_steps":         _n_h_steps,
            "n_val_steps":             _n_val_steps,
            "ceiling_hits_holdout":    _rf_ceiling_hits,
            "ceiling_hits_val":        _val_ceil_hits,
            "ceiling_value":           _rf_ceiling,
            "notes": (
                f"recursive won by {_imp_hold_mae - _rec_hold_mae:.4f}"
                if _rec_wins
                else f"imputation won by {_rec_hold_mae - _imp_hold_mae:.4f}"
            ),
        }

except Exception as _rf_exc:
    import traceback as _tb
    print(f"Recursive forecasting failed ({_rf_exc}) — using imputation path")
    _tb.print_exc()
    _lag_forecasting["notes"] = f"error: {str(_rf_exc)[:300]}"

_postprocessing = {
    "ordinal_rounding_applied": False,
    "ordinal_raw_wf_mae":       None,
    "ordinal_round_wf_mae":     None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — Write reports/predictions.csv
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
# Step 12 — Write reports/model_results.json
# ─────────────────────────────────────────────────────────────────────────────
_imp_arr = last_model.get_feature_importance()
feat_imp = pd.Series(_imp_arr, index=feature_cols).sort_values(ascending=False)
top10   = [{"feature": k, "importance": float(v)} for k, v in feat_imp.head(10).items()]
all_imp = [{"feature": k, "importance": float(v)} for k, v in feat_imp.items()]

training_time = int(time.time() - start_time)

results = {
    "algorithm":          "CatBoost (Ridge diagnostic)",
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
        "ridge_excluded_reason":         "ridge_is_diagnostic_only",
        "ensemble_blend":                ensemble_blend,
        "blend_weights":                 _blend_weights_log,
        "blend_holdout_mae_equal":       _blend_holdout_mae_equal,
        "blend_holdout_mae_inv":         _blend_holdout_mae_inv,
    },
    "families": {
        "catboost": {
            "best_params":           final_hparams,
            "oof_mae":               float(cb_oof_mae),
            "training_time_seconds": cb_training_time,
            "succeeded":             True,
            "included_in_ensemble":  True,
            "n_estimators":          best_n_estimators,
            "optuna_trials":         optuna_trials,
        },
        "ridge": {
            **ridge_diagnostic,
            "included_in_ensemble":  False,
        },
    },
    "ensemble_oof_mae":         float(ensemble_oof_mae),
    "n_families_in_ensemble":   n_families_in_ensemble,
    "ensemble_disagreement":    ensemble_disagreement,
    "feature_importance_top10": top10,
    "feature_importance_all":   all_imp,
    "ridge_top_coefficients":   ridge_top_coefficients,
    # Scalar fields kept for downstream (validator/report_writer) compatibility
    "objective":               _cb_loss,
    "best_params":             final_hparams,
    "n_estimators":            best_n_estimators,
    "n_seeds":                 5,
    "cv_scheme":               cv_scheme,
    "oof_mae":                 float(cb_oof_mae),
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
    "lag_forecasting":   _lag_forecasting,
}

_mr_path = os.path.join(REPO_ROOT, "reports", "model_results.json")
with open(_mr_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Written model_results.json: {_mr_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 13 — Marker file
# ─────────────────────────────────────────────────────────────────────────────
_marker_path = os.path.join(REPO_ROOT, "reports", "modeler_was_here.txt")
with open(_marker_path, "w", encoding="utf-8") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print(f"Written marker: {_marker_path}")
print(f"Total training time: {training_time}s")
print("=" * 60)
print("MODELER COMPLETE")
print("=" * 60)
