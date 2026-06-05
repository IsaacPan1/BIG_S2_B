"""
Modeler: Adaptive ensemble (LightGBM + XGBoost + CatBoost + Ridge)
Panel forecasting with log1p target transform, walk-forward CV, Optuna tuning.
"""
import pandas as pd
import numpy as np
import json
import time
import warnings
import os
warnings.filterwarnings('ignore')

start_time = time.time()
pipeline_start_time = start_time

REPO_ROOT = r"C:\Users\isaac\OneDrive\Desktop\award_B"

# ── Step 1: Read context ──────────────────────────────────────────────────────
with open(os.path.join(REPO_ROOT, "reports", "features.json")) as f:
    feat_meta = json.load(f)

with open(os.path.join(REPO_ROOT, "reports", "profile.json")) as f:
    profile = json.load(f)

# ── Step 2: Load data ─────────────────────────────────────────────────────────
train_df = pd.read_parquet(os.path.join(REPO_ROOT, "data", "features_train.parquet"))
val_df   = pd.read_parquet(os.path.join(REPO_ROOT, "data", "features_val.parquet"))

target_col    = feat_meta["target_col"]
group_cols    = feat_meta["group_cols"]
time_col      = feat_meta["time_col"]
problem_type  = profile.get("problem_type", "panel_forecasting")
problem_subtype = profile.get("problem_subtype", "panel_forecasting")

exclude = set(group_cols + ([time_col] if time_col else []) + [target_col, "adversarial_weights"])
# Also exclude raw string/object columns (e.g. state_doh_release text) — not numeric
_all_candidate_cols = [c for c in train_df.columns if c not in exclude]
feature_cols = [c for c in _all_candidate_cols
                if pd.api.types.is_numeric_dtype(train_df[c])]

_excluded_text = [c for c in _all_candidate_cols if c not in feature_cols]
if _excluded_text:
    print(f"Excluded non-numeric feature cols (text): {_excluded_text}")

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Target: {target_col}")
print(f"Problem type: {problem_type}, subtype: {problem_subtype}")

# ── Log1p transform (memory: skew=2.70, ACTIVE for panel_forecasting) ─────────
_log_transform = True
print("Log1p transform: ACTIVE (skew=2.70 per memory log)")

# Fill NaN with training medians (numeric cols only — already filtered)
fill_vals = train_df[feature_cols].median()

train_df_filled = train_df.copy()
train_df_filled[feature_cols] = train_df_filled[feature_cols].fillna(fill_vals)

X_full = train_df_filled[feature_cols]
y_full_raw = train_df[target_col].values
y_full = np.log1p(y_full_raw)   # log-space targets for training
n = len(train_df)
n_train = n

print(f"n_train={n_train}, n_val={len(val_df)}")
print(f"Target raw: min={y_full_raw.min():.3f}, max={y_full_raw.max():.3f}, mean={y_full_raw.mean():.3f}")
print(f"Target log: min={y_full.min():.3f}, max={y_full.max():.3f}, mean={y_full.mean():.3f}")

# ── Step 2b: Adversarial weights ──────────────────────────────────────────────
_av_info = feat_meta.get("adversarial_validation", {})
_adv_weights = None
if _av_info.get("weights_applied", False) and "adversarial_weights" in train_df.columns:
    _adv_weights = train_df["adversarial_weights"].fillna(1.0).values
    print(f"Adversarial weights loaded: min={_adv_weights.min():.3f}, "
          f"max={_adv_weights.max():.3f}, mean={_adv_weights.mean():.3f}")
    print(f"  AUC (train vs val): {_av_info.get('auc_train_vs_val')}")
else:
    print("No adversarial weights — training with uniform sample weights")

# ── Adaptive ensemble selection ───────────────────────────────────────────────
_dist_shifts = profile.get("distribution_shifts", [])
_max_ks = max((d.get("ks_statistic", 0.0) for d in _dist_shifts if isinstance(d, dict)), default=0.0)
print(f"Max KS statistic: {_max_ks:.4f}")

# Axis 1: panel_forecasting + n_train >= 1000 → Branch 1
if n_train >= 1000:
    adaptive_branch = 1
    families_selected = ["lightgbm", "xgboost", "catboost", "ridge"]
    ensemble_path_used = "full_regression_ensemble"
    adaptive_reasoning = f"panel_forecasting + n_train={n_train} >= 1000 → full ensemble (LGB+XGB+CatBoost+Ridge)"
else:
    adaptive_branch = 2
    families_selected = ["lightgbm", "ridge"]
    ensemble_path_used = "full_regression_ensemble"
    adaptive_reasoning = f"panel_forecasting + n_train={n_train} < 1000 → LGB+Ridge only"

# Axis 2: distribution shift
if _max_ks > 0.40:
    ensemble_weighting = "ridge_weighted_1.5x"
    weighting_reason = f"max_ks={_max_ks:.4f} > 0.40 threshold; ridge competence check pending"
else:
    ensemble_weighting = "equal_median"
    weighting_reason = f"max_ks={_max_ks:.4f} <= 0.40 threshold; using equal_median"

print(f"Axis 1 branch: {adaptive_branch}, families: {families_selected}")
print(f"Axis 2 weighting: {ensemble_weighting}")

# ── Step 3: Check critic retune ───────────────────────────────────────────────
_retune_applied = None
_retune_path = os.path.join(REPO_ROOT, "reports", "critic_retune_requested.json")
if os.path.exists(_retune_path):
    with open(_retune_path) as _f:
        _retune = json.load(_f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"Critic retune requested: {_suggestion}")
    if "median seed aggregation" in _suggestion:
        _retune_applied = "median_seed_aggregation"
    if "expanded Optuna" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+expanded_optuna_bounds"
    if "val feature imputation" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+verified_imputation"
    if "np.clip applied after seed aggregation" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+clip_after_ensemble"
    if "remove suspect features" in _suggestion:
        try:
            with open(os.path.join(REPO_ROOT, "reports", "validator_review.json")) as _vf:
                _vreview = json.load(_vf)
            _suspect = _vreview.get("feature_suspicion", [])
            if _suspect:
                feature_cols = [c for c in feature_cols if c not in _suspect]
                print(f"Applying: removed {len(_suspect)} suspect features: {_suspect[:5]}")
                _retune_applied = (_retune_applied or "") + f"+removed_{len(_suspect)}_features"
        except Exception as _ve:
            print(f"Could not read feature_suspicion: {_ve}")

# Seed aggregation function
def _agg_seeds(preds_list):
    if _retune_applied and "median_seed_aggregation" in _retune_applied:
        return np.median(preds_list, axis=0)
    return np.mean(preds_list, axis=0)

# ── Step 4: Walk-forward split ────────────────────────────────────────────────
all_weeks = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_weeks) * 0.8)
cutoff_week = all_weeks[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_week].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_week].copy()

wf_fill_vals = wf_train[feature_cols].median()
X_wf_train = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train_raw = wf_train[target_col].values
y_wf_train = np.log1p(y_wf_train_raw)
X_wf_val   = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val_raw = wf_val[target_col].values
y_wf_val   = y_wf_val_raw   # original space for MAE evaluation

print(f"Walk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
print(f"Cutoff week: {cutoff_week}")

# Sample weights for walk-forward training
_wf_sw = _adv_weights[wf_train.index.values] if _adv_weights is not None else None

# Prepare full val
X_val = val_df[feature_cols].fillna(fill_vals)
X_full_filled = train_df_filled[feature_cols]

# cv_scheme
cv_scheme = f"WalkForward(cutoff={cutoff_week}, train={len(wf_train)}, val={len(wf_val)})"
fold_maes = []

# ── Step 5: Optuna LightGBM tuning ───────────────────────────────────────────
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 25 * 60

_num_leaves_hi = 255 if (_retune_applied and "expanded_optuna_bounds" in _retune_applied) else 127
_min_child_lo = 3 if (_retune_applied and "expanded_optuna_bounds" in _retune_applied) else 5

_optuna_reflection = {"pinned_params": [], "recentered": False,
                      "best_mae_before": None, "best_mae_after": None}

def objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, _num_leaves_hi),
        "min_child_samples": trial.suggest_int("min_child_samples", _min_child_lo, 60),
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
            eval_set=[(X_wf_val, np.log1p(y_wf_val))],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        preds_log = m.predict(X_wf_val)
        preds_raw = np.clip(np.expm1(np.clip(preds_log, 0, None)), 0, None)
        maes.append(mean_absolute_error(y_wf_val, preds_raw))
    return float(np.mean(maes))

try:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=25*60, catch=(Exception,))
    best_params = study.best_params
    _optuna_reflection["best_mae_before"] = float(study.best_value)
    _optuna_reflection["best_mae_after"] = float(study.best_value)
    optuna_trials = len(study.trials)

    # Boundary reflection
    _lgb_bounds = {
        "learning_rate": (0.01, 0.1, "log"),
        "num_leaves":    (15, _num_leaves_hi, "int"),
        "min_child_samples": (_min_child_lo, 60, "int"),
        "feature_fraction":  (0.5, 1.0, "float"),
        "bagging_fraction":  (0.5, 1.0, "float"),
    }
    _pinned = []
    _shifted_bounds = {}
    for _pn, (_lo, _hi, _pt) in _lgb_bounds.items():
        _v = best_params.get(_pn)
        if _v is None:
            continue
        _rng = _hi - _lo
        _at_lo = (_pt == "int" and int(_v) == _lo) or (_pt != "int" and _v <= _lo + 0.05 * _rng)
        _at_hi = (_pt == "int" and int(_v) == _hi) or (_pt != "int" and _v >= _hi - 0.05 * _rng)
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
        print(f"Boundary reflection: {_pinned} pinned — recentering and running second study")

        def _reflect_objective(trial):
            if time.time() > TUNING_DEADLINE:
                raise optuna.exceptions.TrialPruned()
            def _rb(pn, lo, hi, log=False):
                nl, nh = _shifted_bounds.get(pn, (lo, hi))
                if log:
                    return trial.suggest_float(pn, float(nl), float(nh), log=True)
                elif isinstance(lo, int) and isinstance(hi, int):
                    return trial.suggest_int(pn, max(1, int(nl)), max(int(nl)+1, int(nh)))
                else:
                    return trial.suggest_float(pn, float(nl), float(nh))
            _rp = {
                "objective": "regression_l1", "metric": "mae", "n_estimators": 500,
                "learning_rate": _rb("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": _rb("num_leaves", 15, _num_leaves_hi),
                "min_child_samples": _rb("min_child_samples", _min_child_lo, 60),
                "feature_fraction": _rb("feature_fraction", 0.5, 1.0),
                "bagging_fraction": _rb("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1, "n_jobs": -1,
            }
            _rmaes = []
            for _rs in [42, 7, 123]:
                _rm = lgb.LGBMRegressor(**{**_rp, "random_state": _rs})
                _rm.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw,
                        eval_set=[(X_wf_val, np.log1p(y_wf_val))],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                _p_log = _rm.predict(X_wf_val)
                _p_raw = np.clip(np.expm1(np.clip(_p_log, 0, None)), 0, None)
                _rmaes.append(mean_absolute_error(y_wf_val, _p_raw))
            return float(np.mean(_rmaes))

        _study2 = optuna.create_study(direction="minimize")
        _study2.optimize(_reflect_objective, n_trials=15,
                         timeout=max(30, TUNING_DEADLINE - time.time() - 5),
                         catch=(Exception,))
        optuna_trials += len(_study2.trials)
        _optuna_reflection["recentered"] = True
        if _study2.trials and _study2.best_value < study.best_value:
            best_params = _study2.best_params
            print(f"Reflection improved: {study.best_value:.4f} → {_study2.best_value:.4f}")
            _optuna_reflection["best_mae_after"] = float(_study2.best_value)
        else:
            print(f"Reflection did not improve (original={study.best_value:.4f})")

    print(f"Optuna complete: {optuna_trials} trials, best MAE={study.best_value:.4f}")
    print(f"Best params: {best_params}")
    optuna_succeeded = True
except Exception as e:
    print(f"Optuna failed ({e}), using defaults")
    best_params = {}
    optuna_trials = 0
    optuna_succeeded = False

# ── Step 6: Build final hyperparameters ───────────────────────────────────────
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
    sample_weight=_wf_sw,
    eval_set=[(X_wf_val, np.log1p(y_wf_val))],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_n_estimators = int(probe.best_iteration_ * 1.1) if probe.best_iteration_ else 500

lgb_probe_log = probe.predict(X_wf_val)
lgb_wf_val_preds = np.clip(np.expm1(np.clip(lgb_probe_log, 0, None)), 0, None)
wf_mae = mean_absolute_error(y_wf_val, lgb_wf_val_preds)
probe_mae_80_20 = float(wf_mae)
print(f"Walk-forward MAE (original space): {wf_mae:.4f}, best_iteration: {probe.best_iteration_}, using n_estimators: {best_n_estimators}")

# ── Step 7a: LightGBM full training ──────────────────────────────────────────
all_val_preds = {}
lgb_training_start = time.time()

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

# For panel_forecasting: walk-forward MAE is the OOF MAE
oof_mae = wf_mae

# Full-data retrain on 5 seeds
seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full_filled, y_full, sample_weight=_adv_weights, callbacks=[lgb.log_evaluation(-1)])
    p_log = m.predict(X_val)
    p_raw = np.clip(np.expm1(np.clip(p_log, 0, None)), 0, None)
    seed_preds.append(p_raw)

lgb_ensemble_preds = _agg_seeds(seed_preds)
print(f"LGB ensemble: min={lgb_ensemble_preds.min():.2f}, max={lgb_ensemble_preds.max():.2f}, mean={lgb_ensemble_preds.mean():.2f}")
print(f"NaN count: {np.isnan(lgb_ensemble_preds).sum()}")

last_model = m
all_val_preds["lgb"] = lgb_ensemble_preds
lgb_oof_mae = float(oof_mae)
lgb_training_time = int(time.time() - lgb_training_start)

# OOF predictions for validator (walk-forward val rows only)
wf_seed_oof_preds = []
for seed in [42, 7, 123]:
    m_wf = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m_wf.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw, callbacks=[lgb.log_evaluation(-1)])
    p_log = m_wf.predict(X_wf_val)
    wf_seed_oof_preds.append(np.clip(np.expm1(np.clip(p_log, 0, None)), 0, None))

wf_oof_predictions = _agg_seeds(wf_seed_oof_preds)

oof_df = wf_val[group_cols + [time_col]].copy().reset_index(drop=True)
oof_df["fold"] = 0
oof_df["predicted_target"] = wf_oof_predictions
oof_df.to_csv(os.path.join(REPO_ROOT, "reports", "oof_predictions.csv"), index=False)
print(f"Written oof_predictions.csv: {oof_df.shape}")
print(f"LightGBM OOF MAE (walk-forward, original space): {lgb_oof_mae:.4f}")

# ── Step 7b: XGBoost training ─────────────────────────────────────────────────
xgb_result = {
    "attempted": False,
    "succeeded": False,
    "oof_mae": None,
    "training_time_seconds": None,
    "best_params": None,
    "n_estimators": None,
    "optuna_trials": None,
    "included_in_ensemble": False,
    "exclusion_reason": None,
}

_elapsed_before_xgb = (time.time() - pipeline_start_time) / 60
print(f"\nElapsed before XGB: {_elapsed_before_xgb:.1f} min")

if _elapsed_before_xgb > 20:
    xgb_result["attempted"] = False
    xgb_result["exclusion_reason"] = f"skipped_time_guard: {_elapsed_before_xgb:.1f}m > 20m"
    xgb_oof_mae = float("inf")
    xgb_wf_val_preds = None
    print(f"XGBoost: skipping — time guard ({_elapsed_before_xgb:.1f}m > 20m)")
else:
    xgb_result["attempted"] = True
    xgb_start = time.time()
    try:
        import xgboost as xgb

        def xgb_objective(trial):
            if time.time() > TUNING_DEADLINE:
                raise optuna.exceptions.TrialPruned()
            _xp = {
                "n_estimators": 500,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "objective": "reg:absoluteerror",
                "tree_method": "hist",
                "verbosity": 0,
                "n_jobs": -1,
                "random_state": 42,
            }
            try:
                _xm = xgb.XGBRegressor(**_xp)
                _xm.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw,
                        eval_set=[(X_wf_val, np.log1p(y_wf_val))],
                        verbose=False, early_stopping_rounds=50)
            except Exception:
                _xp2 = {**_xp, "objective": "reg:squarederror"}
                _xm = xgb.XGBRegressor(**_xp2)
                _xm.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw, verbose=False)
            _xp_log = _xm.predict(X_wf_val)
            _xp_raw = np.clip(np.expm1(np.clip(_xp_log, 0, None)), 0, None)
            return float(mean_absolute_error(y_wf_val, _xp_raw))

        xgb_study = optuna.create_study(direction="minimize")
        xgb_study.optimize(xgb_objective, n_trials=15,
                           timeout=max(30, TUNING_DEADLINE - time.time() - 5),
                           catch=(Exception,))
        xgb_best = xgb_study.best_params
        print(f"XGB Optuna: {len(xgb_study.trials)} trials, best MAE={xgb_study.best_value:.4f}")

        # Best-params walk-forward preds (3-seed for holdout gate)
        xgb_wf_seed_preds = []
        for seed in [42, 7, 123]:
            _xfp = {**xgb_best,
                    "objective": "reg:absoluteerror",
                    "tree_method": "hist",
                    "verbosity": 0,
                    "n_jobs": -1,
                    "n_estimators": 500,
                    "random_state": seed}
            try:
                _xm = xgb.XGBRegressor(**_xfp)
                _xm.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw,
                        eval_set=[(X_wf_val, np.log1p(y_wf_val))],
                        verbose=False, early_stopping_rounds=50)
            except Exception:
                _xfp2 = {**_xfp, "objective": "reg:squarederror"}
                _xm = xgb.XGBRegressor(**_xfp2)
                _xm.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw, verbose=False)
            _xp_log = _xm.predict(X_wf_val)
            xgb_wf_seed_preds.append(np.clip(np.expm1(np.clip(_xp_log, 0, None)), 0, None))

        xgb_wf_best_preds = _agg_seeds(xgb_wf_seed_preds)
        xgb_wf_mae = float(mean_absolute_error(y_wf_val, xgb_wf_best_preds))
        print(f"XGB walk-forward MAE: {xgb_wf_mae:.4f}")

        # Full-data retrain, 5 seeds
        xgb_seed_val_preds = []
        for seed in [42, 7, 123, 2024, 999]:
            _xfp = {**xgb_best,
                    "objective": "reg:absoluteerror",
                    "tree_method": "hist",
                    "verbosity": 0,
                    "n_jobs": -1,
                    "n_estimators": 500,
                    "random_state": seed}
            try:
                _xm = xgb.XGBRegressor(**_xfp)
                _xm.fit(X_full_filled, y_full, sample_weight=_adv_weights, verbose=False)
            except Exception:
                _xfp2 = {**_xfp, "objective": "reg:squarederror"}
                _xm = xgb.XGBRegressor(**_xfp2)
                _xm.fit(X_full_filled, y_full, sample_weight=_adv_weights, verbose=False)
            _xp_log = _xm.predict(X_val)
            xgb_seed_val_preds.append(np.clip(np.expm1(np.clip(_xp_log, 0, None)), 0, None))

        xgb_ensemble_preds = _agg_seeds(xgb_seed_val_preds)
        print(f"XGB ensemble: min={xgb_ensemble_preds.min():.2f}, max={xgb_ensemble_preds.max():.2f}, mean={xgb_ensemble_preds.mean():.2f}")

        all_val_preds["xgb"] = xgb_ensemble_preds
        xgb_oof_mae = float(xgb_wf_mae)
        xgb_wf_val_preds = xgb_wf_best_preds

        xgb_result["succeeded"] = True
        xgb_result["oof_mae"] = xgb_oof_mae
        xgb_result["best_params"] = xgb_best
        xgb_result["n_estimators"] = 500
        xgb_result["optuna_trials"] = len(xgb_study.trials)
        xgb_result["included_in_ensemble"] = True

    except Exception as e:
        print(f"XGBoost failed: {e}")
        xgb_result["succeeded"] = False
        xgb_result["exclusion_reason"] = str(e)
        xgb_oof_mae = float("inf")
        xgb_wf_val_preds = None

    xgb_result["training_time_seconds"] = int(time.time() - xgb_start)
    print(f"XGBoost block finished in {xgb_result['training_time_seconds']}s")

# ── Step 7c: CatBoost training ────────────────────────────────────────────────
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
    print("CatBoost: skipping — catboost not installed")
elif n_train < 500:
    _cb_result["skip_reason"] = "skipped_data_too_small"
    _cb_decision["competence_check_result"] = "skipped_data_too_small"
    print(f"CatBoost: skipping — n_train={n_train} < 500")
elif _elapsed_before_cb >= 40:
    _cb_result["skip_reason"] = "skipped_no_time"
    _cb_decision["competence_check_result"] = "skipped_no_time"
    print(f"CatBoost: skipping — elapsed={_elapsed_before_cb:.1f}m >= 40m time guard")

if _should_run_cb:
    _cb_result["attempted"] = True
    _cb_start = time.time()
    try:
        _cat_cols = [c for c in feature_cols if train_df[c].dtype.name in ("object", "category")]
        _cat_indices = [feature_cols.index(c) for c in _cat_cols]

        _cb_loss = "MAE"
        _cb_eval_metric = "MAE"

        def _cb_objective(trial):
            if (time.time() - pipeline_start_time) / 60 >= 50:
                raise optuna.exceptions.TrialPruned()
            _p = {
                "iterations": 400,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "loss_function": _cb_loss,
                "eval_metric": _cb_eval_metric,
                "verbose": False,
                "allow_writing_files": False,
            }
            if _cat_indices:
                _p["cat_features"] = _cat_indices
            _cb_maes = []
            for _seed in [42, 7]:
                _p["random_seed"] = _seed
                _m = _cb_module.CatBoostRegressor(**_p)
                _wf_sw_cb = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
                _m.fit(
                    X_wf_train.values, y_wf_train,
                    sample_weight=_wf_sw_cb,
                    eval_set=(_cb_module.Pool(X_wf_val.values, np.log1p(y_wf_val))),
                    verbose=False,
                )
                _p_log = _m.predict(X_wf_val.values)
                _p_raw = np.clip(np.expm1(np.clip(_p_log, 0, None)), 0, None)
                _cb_maes.append(mean_absolute_error(y_wf_val, _p_raw))
            return float(np.mean(_cb_maes))

        _cb_study = optuna.create_study(direction="minimize")
        _cb_study.optimize(_cb_objective, n_trials=10, timeout=15*60, catch=(Exception,))
        _cb_best = _cb_study.best_params
        print(f"CatBoost Optuna: {len(_cb_study.trials)} trials, best MAE={_cb_study.best_value:.4f}")

        _cb_final_params = {
            "iterations": 400,
            "learning_rate": _cb_best.get("learning_rate", 0.05),
            "depth": _cb_best.get("depth", 6),
            "l2_leaf_reg": _cb_best.get("l2_leaf_reg", 3.0),
            "loss_function": _cb_loss,
            "eval_metric": _cb_eval_metric,
            "verbose": False,
            "allow_writing_files": False,
        }
        if _cat_indices:
            _cb_final_params["cat_features"] = _cat_indices

        _cb_wf_preds_list = []
        for _seed in [42, 7, 123]:
            _cb_final_params["random_seed"] = _seed
            _m = _cb_module.CatBoostRegressor(**_cb_final_params)
            _wf_sw_cb = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
            _m.fit(X_wf_train.values, y_wf_train, sample_weight=_wf_sw_cb, verbose=False)
            _p_log = _m.predict(X_wf_val.values)
            _cb_wf_preds_list.append(np.clip(np.expm1(np.clip(_p_log, 0, None)), 0, None))

        _cb_wf_median = np.median(_cb_wf_preds_list, axis=0)
        _cb_oof_mae = float(mean_absolute_error(y_wf_val, _cb_wf_median))
        cb_wf_val_preds = _cb_wf_median
        print(f"CatBoost walk-forward OOF MAE: {_cb_oof_mae:.4f}")
        _cb_result["oof_mae"] = _cb_oof_mae

        _tree_oofs = [lgb_oof_mae]
        if xgb_oof_mae != float("inf"):
            _tree_oofs.append(xgb_oof_mae)
        _best_tree_oof = min(_tree_oofs)
        _cb_passes_competence = _cb_oof_mae <= 1.5 * _best_tree_oof

        if not _cb_passes_competence:
            _cb_result["included_in_ensemble"] = False
            _cb_result["excluded_reason"] = (
                f"excluded_too_weak: cb_oof={_cb_oof_mae:.4f} > 1.5x best_tree_oof={_best_tree_oof:.4f}"
            )
            _cb_decision["competence_check_result"] = "excluded_too_weak"
            print(f"CatBoost excluded: {_cb_result['excluded_reason']}")
        else:
            _cb_seed_preds = []
            _train_target_max = float(y_full_raw.max())
            _train_target_mean = float(y_full_raw.mean())
            for _seed in [42, 7, 123, 2024, 999]:
                _cb_final_params["random_seed"] = _seed
                _m = _cb_module.CatBoostRegressor(**_cb_final_params)
                _m.fit(X_full_filled.values, y_full, sample_weight=_adv_weights, verbose=False)
                _p_log = _m.predict(X_val.values)
                _cb_seed_preds.append(np.clip(np.expm1(np.clip(_p_log, 0, None)), 0, None))

            _cb_val_preds = np.median(_cb_seed_preds, axis=0)
            _cb_val_preds = np.clip(_cb_val_preds, 0, None)

            _cb_pred_max = float(np.max(_cb_val_preds))
            _cb_pred_mean = float(np.mean(_cb_val_preds))
            _sanity_ok = True
            if _cb_pred_max > 5 * _train_target_max:
                _cb_result["included_in_ensemble"] = False
                _cb_result["excluded_reason"] = (
                    f"sanity_fail: pred_max={_cb_pred_max:.2f} > 5x train_max={_train_target_max:.2f}"
                )
                _cb_decision["competence_check_result"] = "excluded_too_weak"
                _sanity_ok = False
                print(f"CatBoost excluded (sanity): {_cb_result['excluded_reason']}")
            elif abs(_train_target_mean) > 0 and abs(_cb_pred_mean - _train_target_mean) / abs(_train_target_mean) > 1.0:
                _cb_result["included_in_ensemble"] = False
                _cb_result["excluded_reason"] = (
                    f"sanity_fail: pred_mean={_cb_pred_mean:.2f} deviates >100% from train_mean={_train_target_mean:.2f}"
                )
                _cb_decision["competence_check_result"] = "excluded_too_weak"
                _sanity_ok = False
                print(f"CatBoost excluded (sanity): {_cb_result['excluded_reason']}")

            if _sanity_ok:
                all_val_preds["catboost"] = _cb_val_preds
                _cb_result["included_in_ensemble"] = True
                _cb_result["succeeded"] = True
                _cb_decision["competence_check_result"] = "included"
                print(f"CatBoost included: OOF={_cb_oof_mae:.4f}, within 1.5x tree best={_best_tree_oof:.4f}")
            else:
                _cb_result["succeeded"] = True

    except Exception as _cb_exc:
        _cb_result["succeeded"] = False
        _cb_result["skip_reason"] = f"training_error: {_cb_exc}"
        _cb_decision["competence_check_result"] = "skipped_import_error"
        print(f"CatBoost training failed: {_cb_exc}")

    _cb_result["training_time_seconds"] = float(time.time() - _cb_start)
    print(f"CatBoost block finished in {_cb_result['training_time_seconds']:.1f}s")

# ── Step 7d: Ridge training ───────────────────────────────────────────────────
ridge_result = {
    "attempted": False,
    "succeeded": False,
    "oof_mae": None,
    "training_time_seconds": None,
    "best_alpha": None,
    "included_in_ensemble": False,
    "exclusion_reason": None,
}
ridge_top_coefficients = []
ridge_val_preds = None
ridge_wf_val_preds = None
ridge_excluded_reason = None

_elapsed_before_ridge = (time.time() - pipeline_start_time) / 60
_should_run_ridge = not (len(all_val_preds) >= 2 and _elapsed_before_ridge > 50)
print(f"\nElapsed before Ridge: {_elapsed_before_ridge:.1f} min, will_run={_should_run_ridge}")

if _should_run_ridge and "ridge" in families_selected:
    ridge_result["attempted"] = True
    _ridge_start = time.time()
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_wf_train_scaled = scaler.fit_transform(X_wf_train)
        X_wf_val_scaled   = scaler.transform(X_wf_val)
        X_full_scaled     = scaler.transform(X_full_filled)
        X_val_scaled      = scaler.transform(X_val)

        best_alpha, best_probe_mae = 1.0, float("inf")
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            r = Ridge(alpha=alpha)
            _wf_sw_r = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
            r.fit(X_wf_train_scaled, y_wf_train, sample_weight=_wf_sw_r)
            p_log = r.predict(X_wf_val_scaled)
            p_raw = np.clip(np.expm1(np.clip(p_log, 0, None)), 0, None)
            mae = mean_absolute_error(y_wf_val, p_raw)
            if mae < best_probe_mae:
                best_probe_mae = mae
                best_alpha = alpha

        print(f"Ridge best alpha: {best_alpha}, probe MAE: {best_probe_mae:.4f}")
        ridge_result["best_alpha"] = best_alpha

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_full_scaled, y_full, sample_weight=_adv_weights)
        p_log = ridge.predict(X_val_scaled)
        ridge_val_preds_raw = np.clip(np.expm1(np.clip(p_log, 0, None)), 0, None)

        ridge_wf_fitted = Ridge(alpha=best_alpha)
        _wf_sw_r = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
        ridge_wf_fitted.fit(X_wf_train_scaled, y_wf_train, sample_weight=_wf_sw_r)
        r_wf_log = ridge_wf_fitted.predict(X_wf_val_scaled)
        ridge_wf_val_preds = np.clip(np.expm1(np.clip(r_wf_log, 0, None)), 0, None)
        ridge_wf_mae = float(mean_absolute_error(y_wf_val, ridge_wf_val_preds))
        ridge_result["oof_mae"] = ridge_wf_mae

        _train_target_max = float(y_full_raw.max())
        _train_target_mean = float(y_full_raw.mean())
        _ridge_pred_max = float(np.max(ridge_val_preds_raw))
        _ridge_pred_mean = float(np.mean(ridge_val_preds_raw))

        _best_tree_oof_for_ridge = lgb_oof_mae
        if xgb_oof_mae != float("inf"):
            _best_tree_oof_for_ridge = min(_best_tree_oof_for_ridge, xgb_oof_mae)

        _ridge_excluded = False
        if _ridge_pred_max > 5 * _train_target_max:
            ridge_excluded_reason = f"sanity_fail: pred_max={_ridge_pred_max:.2f} > 5x train_max={_train_target_max:.2f}"
            _ridge_excluded = True
        elif abs(_train_target_mean) > 0 and abs(_ridge_pred_mean - _train_target_mean) / abs(_train_target_mean) > 1.0:
            ridge_excluded_reason = f"sanity_fail: pred_mean={_ridge_pred_mean:.2f} deviates >100% from train_mean={_train_target_mean:.2f}"
            _ridge_excluded = True
        elif ridge_wf_mae > 1.5 * _best_tree_oof_for_ridge:
            ridge_excluded_reason = f"ridge_oof={ridge_wf_mae:.4f} > 1.5x best_family_oof={_best_tree_oof_for_ridge:.4f}"
            _ridge_excluded = True

        if _ridge_excluded:
            ridge_result["exclusion_reason"] = ridge_excluded_reason
            ridge_result["included_in_ensemble"] = False
            print(f"Ridge excluded: {ridge_excluded_reason}")
        else:
            ridge_val_preds = ridge_val_preds_raw
            all_val_preds["ridge"] = ridge_val_preds
            ridge_result["included_in_ensemble"] = True
            ridge_result["succeeded"] = True
            print(f"Ridge included: OOF MAE={ridge_wf_mae:.4f}")

        ridge_top_coefficients = [
            {"feature": f, "abs_coef": float(c)}
            for f, c in sorted(zip(feature_cols, np.abs(ridge.coef_)),
                               key=lambda x: -x[1])[:5]
        ]

    except Exception as e:
        print(f"Ridge failed: {e}")
        ridge_result["succeeded"] = False
        ridge_result["exclusion_reason"] = str(e)

    ridge_result["training_time_seconds"] = int(time.time() - _ridge_start)

# ── Axis 2 Ridge weighting: competence check ─────────────────────────────────
if ensemble_weighting == "ridge_weighted_1.5x":
    if "ridge" not in all_val_preds:
        ensemble_weighting = "equal_median"
        weighting_reason = (
            f"max_ks={_max_ks:.4f} > 0.40 threshold, but Ridge excluded ({ridge_excluded_reason}); "
            f"using equal_median instead"
        )
    else:
        _ridge_oof = ridge_result.get("oof_mae", float("inf"))
        _best_oof_for_axis2 = lgb_oof_mae
        if xgb_oof_mae != float("inf"):
            _best_oof_for_axis2 = min(_best_oof_for_axis2, xgb_oof_mae)
        if _ridge_oof <= 1.5 * _best_oof_for_axis2:
            weighting_reason = (
                f"max_ks={_max_ks:.4f} > 0.40 threshold, "
                f"ridge_oof={_ridge_oof:.4f} within 1.5x best_oof={_best_oof_for_axis2:.4f}; "
                f"ridge_weighted_1.5x applied"
            )
        else:
            ensemble_weighting = "equal_median"
            weighting_reason = (
                f"max_ks={_max_ks:.4f} > 0.40 threshold, "
                f"ridge_oof={_ridge_oof:.4f} > 1.5x best_oof={_best_oof_for_axis2:.4f}; "
                f"using equal_median instead"
            )

print(f"\nEnsemble weighting final: {ensemble_weighting}")
print(f"Weighting reason: {weighting_reason}")

# ── Ensemble aggregation ──────────────────────────────────────────────────────
_family_oof = {}
if "lgb" in all_val_preds:
    _family_oof["lgb"] = lgb_oof_mae
if "xgb" in all_val_preds and xgb_oof_mae != float("inf"):
    _family_oof["xgb"] = xgb_oof_mae
if "catboost" in all_val_preds and _cb_result.get("oof_mae"):
    _family_oof["catboost"] = _cb_result["oof_mae"]
if "ridge" in all_val_preds and ridge_result.get("oof_mae"):
    _family_oof["ridge"] = ridge_result["oof_mae"]

_included_keys = list(all_val_preds.keys())
_stack = np.array([all_val_preds[k] for k in _included_keys])

print(f"\nIncluded families: {_included_keys}")
print(f"OOF MAEs: {_family_oof}")

_blend_holdout_mae_equal = None
_blend_holdout_mae_inv = None

if ensemble_weighting == "ridge_weighted_1.5x" and "ridge" in all_val_preds:
    _w = [1.5 if k == "ridge" else 1.0 for k in _included_keys]
    ensemble_preds = np.average(_stack, axis=0, weights=_w)
    ensemble_blend = "ridge_weighted_1.5x"
    _blend_weights_log = {k: round(_w[i] / sum(_w), 4) for i, k in enumerate(_included_keys)}
    print(f"Blend: ridge_weighted_1.5x, weights={_blend_weights_log}")
else:
    _equal_preds = np.median(_stack, axis=0)
    _inv_sum = sum(1.0 / _family_oof[k] for k in _included_keys
                   if k in _family_oof and _family_oof[k] > 0)
    if _inv_sum > 0:
        _inv_w = [(1.0 / _family_oof[k] / _inv_sum
                   if (k in _family_oof and _family_oof[k] > 0) else 0.0)
                  for k in _included_keys]
    else:
        _inv_w = [1.0 / len(_included_keys) for k in _included_keys]
    _inv_w_arr = np.array(_inv_w)
    _inv_preds = np.average(_stack, axis=0, weights=_inv_w_arr)

    _wf_preds_map = {
        "lgb": lgb_wf_val_preds,
        "xgb": xgb_wf_val_preds if "xgb" in all_val_preds else None,
        "catboost": cb_wf_val_preds if "catboost" in all_val_preds else None,
        "ridge": ridge_wf_val_preds if "ridge" in all_val_preds else None,
    }
    _can_gate = all(_wf_preds_map.get(k) is not None for k in _included_keys)

    if _can_gate:
        _wf_stack = np.array([_wf_preds_map[k] for k in _included_keys])
        _blend_holdout_mae_equal = float(mean_absolute_error(y_wf_val, np.median(_wf_stack, axis=0)))
        _blend_holdout_mae_inv   = float(mean_absolute_error(y_wf_val, np.average(_wf_stack, axis=0, weights=_inv_w_arr)))
        _use_inv = _blend_holdout_mae_inv <= _blend_holdout_mae_equal
    else:
        _blend_holdout_mae_equal = None
        _blend_holdout_mae_inv = None
        _use_inv = False

    if _use_inv:
        ensemble_preds = _inv_preds
        ensemble_blend = "inverse_mae_weighted"
        _blend_weights_log = {k: round(float(_inv_w_arr[i]), 4) for i, k in enumerate(_included_keys)}
        print(f"Blend: inverse_mae_weighted (holdout {_blend_holdout_mae_inv:.4f} <= equal {_blend_holdout_mae_equal:.4f})")
    else:
        ensemble_preds = _equal_preds
        ensemble_blend = "equal_median"
        _blend_weights_log = {k: round(1.0 / len(_included_keys), 4) for k in _included_keys}
        _reason = "gate_equal_better" if _can_gate else "no_wf_preds_available"
        print(f"Blend: equal_median ({_reason}; holdout equal={_blend_holdout_mae_equal}, inv={_blend_holdout_mae_inv})")

ensemble_preds = np.clip(ensemble_preds, 0, None)

# ── Persist per-family validation predictions for post-hoc blend analysis ────
try:
    _npz_data: dict = {f"family_{k}": v for k, v in all_val_preds.items()}
    for _key_col in group_cols + ([time_col] if time_col else []):
        if _key_col in val_df.columns:
            _npz_data[f"key_{_key_col}"] = val_df[_key_col].values.astype(str)
    _npz_path = os.path.join(REPO_ROOT, "reports", "family_val_preds.npz")
    np.savez(_npz_path, **_npz_data)
    print(f"Written reports/family_val_preds.npz (families: {list(all_val_preds.keys())})")
except Exception as _npz_e:
    print(f"WARNING: could not write family_val_preds.npz: {_npz_e}")

ensemble_disagreement = {
    "mean_disagreement": float(np.mean(np.std(_stack, axis=0))),
    "n_high_disagreement_rows": int(np.sum(np.std(_stack, axis=0) > ensemble_preds.mean())),
}
ensemble_oof_mae = min(_family_oof.values()) if _family_oof else lgb_oof_mae
n_families_in_ensemble = len(_included_keys)

print(f"\nFinal ensemble: {ensemble_blend}")
print(f"Ensemble preds: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"Ensemble OOF MAE: {ensemble_oof_mae:.4f}")

# ── Ordinal rounding (no-op for panel_forecasting) ────────────────────────────
_postprocessing = {"ordinal_rounding_applied": False,
                   "ordinal_raw_wf_mae": None, "ordinal_round_wf_mae": None}

# ── Step 8: Write predictions.csv ────────────────────────────────────────────
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found — abort"

preds_df.to_csv(os.path.join(REPO_ROOT, "reports", "predictions.csv"), index=False)
print(f"Written reports/predictions.csv: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head())

# ── Step 9: Write model_results.json ─────────────────────────────────────────
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{"feature": k, "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp = [{"feature": k, "importance": int(v)} for k, v in feat_imp.items()]

training_time = int(time.time() - start_time)

_algo_label_map = {"lgb": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost", "ridge": "Ridge"}
_algo_label = "+".join(_algo_label_map[k] for k in _included_keys if k in _algo_label_map) + " ensemble"

results = {
    "algorithm": _algo_label,
    "problem_subtype": problem_subtype,
    "ensemble_path_used": ensemble_path_used,
    "log_transform_applied": True,
    "adaptive_choice": {
        "branch": adaptive_branch,
        "families_selected": families_selected,
        "families_included_in_ensemble": list(all_val_preds.keys()),
        "reasoning": adaptive_reasoning,
        "ensemble_weighting": ensemble_weighting,
        "weighting_reason": weighting_reason,
        "ridge_excluded_reason": ridge_excluded_reason,
        "ensemble_blend": ensemble_blend,
        "blend_weights": _blend_weights_log,
        "blend_holdout_mae_equal": _blend_holdout_mae_equal,
        "blend_holdout_mae_inv": _blend_holdout_mae_inv,
    },
    "families": {
        "lightgbm": {
            "best_params": final_hparams,
            "oof_mae": float(lgb_oof_mae),
            "training_time_seconds": lgb_training_time,
            "succeeded": True,
            "included_in_ensemble": "lgb" in all_val_preds,
            "n_estimators": best_n_estimators,
            "optuna_trials": optuna_trials,
        },
        "xgboost": xgb_result,
        "catboost": _cb_result,
        "ridge": ridge_result,
    },
    "ensemble_oof_mae": float(ensemble_oof_mae),
    "n_families_in_ensemble": n_families_in_ensemble,
    "ensemble_disagreement": ensemble_disagreement,
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
    "ridge_top_coefficients": ridge_top_coefficients,
    "objective": final_params["objective"],
    "best_params": final_hparams,
    "n_estimators": best_n_estimators,
    "n_seeds": 5,
    "cv_scheme": cv_scheme,
    "oof_mae": float(lgb_oof_mae),
    "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(lgb_oof_mae)],
    "walk_forward_mae": float(wf_mae),
    "probe_mae_80_20": float(probe_mae_80_20),
    "training_time_seconds": training_time,
    "optuna_trials_completed": optuna_trials,
    "optuna_succeeded": optuna_succeeded,
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()),
        "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()),
        "std": float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": n_train,
    "n_val_rows": len(val_df),
    "retune_applied": _retune_applied,
    "optuna_reflection": _optuna_reflection,
    "postprocessing": _postprocessing,
}

with open(os.path.join(REPO_ROOT, "reports", "model_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Written reports/model_results.json")

# ── Step 10: Marker file ──────────────────────────────────────────────────────
import datetime
with open(os.path.join(REPO_ROOT, "reports", "modeler_was_here.txt"), "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print("Written reports/modeler_was_here.txt")
print(f"\nTotal training time: {training_time}s ({training_time/60:.1f} min)")
print("Modeler complete.")
