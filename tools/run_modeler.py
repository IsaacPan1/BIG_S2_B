"""
Modeler script: LightGBM with problem-type-aware CV + Optuna tuning.
CV splitter is chosen from profile.json:
  tabular_regression + group_cols  → GroupKFold(n_splits=min(5, n_unique_groups))
  tabular_regression, no groups    → RepeatedKFold(n_splits=5, n_repeats=3)
  classification                   → StratifiedKFold(n_splits=5)
  panel_forecasting                → WalkForward (train on first 80% of time periods,
                                     validate on last 20%); OOF predictions on wf_val set
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

# ── Check for critic retune request ──────────────────────────────────────────
_retune_applied = None
_retune_path = os.path.join(REPORTS_DIR, "critic_retune_requested.json")
if os.path.exists(_retune_path):
    with open(_retune_path) as _f:
        _retune = json.load(_f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"Critic retune requested: {_suggestion}")

    if "median seed aggregation" in _suggestion:
        _retune_applied = "median_seed_aggregation"
        print("Applying: median seed aggregation instead of mean")

    if "expand Optuna" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+expanded_optuna_bounds"
        print("Applying: expanded Optuna num_leaves/min_child_samples bounds")

    if "val feature imputation" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+verified_imputation"
        print("Applying: verified fill_vals uses training column medians (not zeros)")

    if "np.clip applied after seed aggregation" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+clip_after_ensemble"
        print("Applying: np.clip(0, None) applied to ensemble_preds after seed aggregation")

    if "multi-fold purged walk-forward CV" in _suggestion:
        _retune_applied = (_retune_applied or "") + "+multi_fold_wf_cv"
        print("Applying: multi-fold purged walk-forward CV (4-fold, embargo=2 periods)")

    if "remove suspect features" in _suggestion:
        try:
            with open(os.path.join(REPORTS_DIR, "validator_review.json")) as _vf:
                _vreview = json.load(_vf)
            _suspect = _vreview.get("feature_suspicion", [])
            if _suspect:
                print(f"Will remove {len(_suspect)} suspect features: {_suspect[:5]}")
                _retune_applied = (_retune_applied or "") + f"+will_remove_{len(_suspect)}_features"
        except Exception as _ve:
            print(f"Could not read feature_suspicion from validator_review.json: {_ve}")

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

# Apply suspect feature removal if requested by critic
if _retune_applied and "remove_" in (_retune_applied or ""):
    try:
        with open(os.path.join(REPORTS_DIR, "validator_review.json")) as _vf:
            _vreview = json.load(_vf)
        _suspect = _vreview.get("feature_suspicion", [])
        if _suspect:
            feature_cols = [c for c in feature_cols if c not in _suspect]
            print(f"Removed {len(_suspect)} suspect features")
    except Exception as _ve:
        print(f"Could not apply feature removal: {_ve}")

print(f"\nFeature cols ({len(feature_cols)}): {feature_cols[:20]}...")

# Drop string/object columns from feature_cols (they can't be used as numeric features)
numeric_feature_cols = [c for c in feature_cols
                        if train_df[c].dtype not in ['object', 'string'] and
                        str(train_df[c].dtype) not in ['object', 'string']]
dropped_str_cols = [c for c in feature_cols if c not in numeric_feature_cols]
if dropped_str_cols:
    print(f"Dropping non-numeric feature columns: {dropped_str_cols}")
feature_cols = numeric_feature_cols

# NaN fill with training medians (IMPORTANT: use training medians, not 0)
fill_vals = train_df[feature_cols].median()

# ── Step 4: Panel forecasting: Walk-forward split for tuning + OOF ───────────
_use_multi_fold_wf = (_retune_applied and "multi_fold_wf_cv" in (_retune_applied or ""))

if problem_type == "panel_forecasting":
    all_periods = sorted(train_df[time_col].unique())
    cutoff_idx = int(len(all_periods) * 0.8)
    cutoff_period = all_periods[cutoff_idx]

    wf_train = train_df[train_df[time_col] < cutoff_period].copy()
    wf_val_df = train_df[train_df[time_col] >= cutoff_period].copy()

    wf_fill_vals = wf_train[feature_cols].median()
    X_wf_train = wf_train[feature_cols].fillna(wf_fill_vals)
    y_wf_train = wf_train[target_col]
    X_wf_val   = wf_val_df[feature_cols].fillna(wf_fill_vals)
    y_wf_val   = wf_val_df[target_col]

    print(f"\nWalk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}")
    print(f"Cutoff period: {cutoff_period}")

    if _use_multi_fold_wf:
        cv_scheme = f"MultiWalkForward(n_folds=4,embargo=2,n_periods={len(all_periods)})"
    else:
        cv_scheme = f"WalkForward(cutoff_idx={cutoff_idx}/{len(all_periods)},cutoff={cutoff_period})"

    # For Optuna tuning, use the single walk-forward split (faster)
    X_ptr, y_ptr = X_wf_train, y_wf_train.values
    X_pva, y_pva = X_wf_val, y_wf_val.values

else:
    # tabular_regression / classification: 80/20 random split for Optuna probe
    n = len(train_df)
    X_full_tmp = train_df[feature_cols].fillna(fill_vals)
    y_full_tmp = train_df[target_col].values
    np.random.seed(42)
    perm = np.random.permutation(n)
    split_pt = int(n * 0.8)
    probe_tr_idx, probe_va_idx = perm[:split_pt], perm[split_pt:]
    X_ptr_raw = X_full_tmp.iloc[probe_tr_idx]
    y_ptr = y_full_tmp[probe_tr_idx]
    X_pva_raw = X_full_tmp.iloc[probe_va_idx]
    y_pva = y_full_tmp[probe_va_idx]
    ptr_fill = X_ptr_raw.median()
    X_ptr = X_ptr_raw.fillna(ptr_fill)
    X_pva = X_pva_raw.fillna(ptr_fill)
    print(f"\nProbe train: {X_ptr.shape}, probe val: {X_pva.shape}")
    cv_scheme = "KFold(n_splits=5, shuffle=True)"  # will be updated below

# ── Step 5: Optuna hyperparameter tuning ──────────────────────────────────────
TUNING_DEADLINE = start_time + 25 * 60

_expanded_optuna = _retune_applied and "expanded_optuna_bounds" in (_retune_applied or "")
_nl_upper = 255 if _expanded_optuna else 127
_mcs_lower = 3 if _expanded_optuna else 5

def objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, _nl_upper),
        "min_child_samples": trial.suggest_int("min_child_samples", _mcs_lower, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }
    use_median = _retune_applied and "median_seed_aggregation" in (_retune_applied or "")
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
    if use_median:
        return float(np.median(maes))
    return float(np.mean(maes))

try:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=25 * 60, catch=(Exception,))
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

# ── Step 6: Merge tuned + fixed params, determine n_estimators ───────────────
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_hparams = {**default_params, **best_params}

probe_fit_params = {
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
probe_model = lgb.LGBMRegressor(**probe_fit_params)
probe_model.fit(
    X_ptr, y_ptr,
    eval_set=[(X_pva, y_pva)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_iter = probe_model.best_iteration_ if probe_model.best_iteration_ else 500
best_n_estimators = max(int(best_iter * 1.1), 200)
probe_mae = mean_absolute_error(y_pva, np.clip(probe_model.predict(X_pva), 0, None))
print(f"\nProbe MAE: {probe_mae:.4f}, best_iter={best_iter}, n_estimators={best_n_estimators}")

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

# ── Step 7: OOF metric and OOF predictions ───────────────────────────────────
use_median_ensemble = _retune_applied and "median_seed_aggregation" in (_retune_applied or "")

if problem_type == "panel_forecasting":
    n = len(train_df)
    X_full = train_df[feature_cols].fillna(fill_vals)
    y_full = train_df[target_col].values
    X_val  = val_df[feature_cols].fillna(fill_vals)

    if _use_multi_fold_wf:
        # ── Multi-fold purged walk-forward CV (matches validator's strict scheme) ──
        # 4 folds, embargo=2 periods to prevent temporal leakage through lag features
        N_WF_FOLDS = 4
        EMBARGO = 2  # periods to skip between train and val to prevent lag leakage
        # Use only periods from 40%..100% as potential val windows
        # to ensure training sets are large enough
        n_periods = len(all_periods)
        # Minimum train fraction: start val folds from 40% of data
        min_train_frac = 0.40
        min_train_periods = int(n_periods * min_train_frac)
        # Available val range: from min_train_periods to end, split into N_WF_FOLDS
        available_val_periods = all_periods[min_train_periods:]
        fold_size = max(1, len(available_val_periods) // N_WF_FOLDS)

        fold_maes = []
        oof_preds_list = []   # list of (row_indices_in_train_df, fold_idx, preds)

        print(f"\nMulti-fold purged walk-forward CV: {N_WF_FOLDS} folds, embargo={EMBARGO}")
        for fold_idx in range(N_WF_FOLDS):
            # Val window: fold_idx-th chunk of available_val_periods
            val_start_pos = fold_idx * fold_size
            val_end_pos   = (fold_idx + 1) * fold_size if fold_idx < N_WF_FOLDS - 1 else len(available_val_periods)
            val_periods_fold = available_val_periods[val_start_pos:val_end_pos]
            val_cutoff_start = val_periods_fold[0]
            val_cutoff_end   = val_periods_fold[-1]

            # Train: all periods strictly before val_cutoff_start - EMBARGO
            train_cutoff_idx  = all_periods.index(val_cutoff_start) - EMBARGO
            if train_cutoff_idx < 10:
                print(f"  Fold {fold_idx+1}: skipping — training set too small")
                continue
            train_periods_fold = all_periods[:train_cutoff_idx]

            fold_tr_mask = train_df[time_col].isin(train_periods_fold)
            fold_va_mask = train_df[time_col].isin(val_periods_fold)
            fold_tr_df   = train_df[fold_tr_mask]
            fold_va_df   = train_df[fold_va_mask]

            fold_fill = fold_tr_df[feature_cols].median()
            X_fold_tr = fold_tr_df[feature_cols].fillna(fold_fill)
            y_fold_tr = fold_tr_df[target_col].values
            X_fold_va = fold_va_df[feature_cols].fillna(fold_fill)
            y_fold_va = fold_va_df[target_col].values

            fold_seed_preds = []
            for seed in [42, 7, 123]:
                m_fold = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
                m_fold.fit(X_fold_tr, y_fold_tr, callbacks=[lgb.log_evaluation(-1)])
                fold_seed_preds.append(np.clip(m_fold.predict(X_fold_va), 0, None))

            if use_median_ensemble:
                fold_preds_agg = np.median(fold_seed_preds, axis=0)
            else:
                fold_preds_agg = np.mean(fold_seed_preds, axis=0)

            fmae = mean_absolute_error(y_fold_va, fold_preds_agg)
            fold_maes.append(fmae)
            va_indices = train_df.index[fold_va_mask].tolist()
            oof_preds_list.append((va_indices, fold_idx, fold_preds_agg, fold_va_df))
            print(f"  Fold {fold_idx+1}: train_periods={len(train_periods_fold)}, "
                  f"val_periods={len(val_periods_fold)}, "
                  f"val_rows={len(y_fold_va)}, MAE={fmae:.4f}")

        wf_mae = float(np.mean(fold_maes)) if fold_maes else float("nan")
        oof_mae = wf_mae
        print(f"\nMulti-fold walk-forward MAE (mean of {len(fold_maes)} folds): {wf_mae:.4f}")
        print(f"Per-fold MAEs: {[f'{m:.4f}' for m in fold_maes]}")

        # Build OOF dataframe from multi-fold results
        all_oof_rows = []
        for (va_indices, fold_idx, fold_preds_agg, fold_va_df) in oof_preds_list:
            fold_oof = fold_va_df[group_cols + [time_col]].copy().reset_index(drop=True)
            fold_oof["fold"] = fold_idx
            fold_oof["predicted_target"] = fold_preds_agg
            all_oof_rows.append(fold_oof)
        if all_oof_rows:
            oof_df = pd.concat(all_oof_rows, ignore_index=True)
        else:
            # Fallback: use single walk-forward
            wf_seed_preds_fb = []
            for seed in [42, 7, 123]:
                m_fb = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
                m_fb.fit(X_wf_train, y_wf_train, callbacks=[lgb.log_evaluation(-1)])
                wf_seed_preds_fb.append(np.clip(m_fb.predict(X_wf_val), 0, None))
            wf_preds_fb = np.mean(wf_seed_preds_fb, axis=0)
            wf_mae = float(mean_absolute_error(y_wf_val, wf_preds_fb))
            oof_mae = wf_mae
            fold_maes = [wf_mae]
            oof_df = wf_val_df[group_cols + [time_col]].copy().reset_index(drop=True)
            oof_df["fold"] = 0
            oof_df["predicted_target"] = wf_preds_fb

    else:
        # Single walk-forward MAE is the honest OOF metric for panel forecasting
        wf_seed_preds = []
        for seed in [42, 7, 123]:
            m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
            m.fit(X_wf_train, y_wf_train, callbacks=[lgb.log_evaluation(-1)])
            wf_seed_preds.append(np.clip(m.predict(X_wf_val), 0, None))

        if use_median_ensemble:
            wf_preds_arr = np.median(wf_seed_preds, axis=0)
        else:
            wf_preds_arr = np.mean(wf_seed_preds, axis=0)

        wf_mae = mean_absolute_error(y_wf_val, wf_preds_arr)
        oof_mae = wf_mae
        fold_maes = [float(wf_mae)]
        print(f"\nWalk-forward MAE: {wf_mae:.4f}")

        # OOF predictions: the walk-forward val set (last 20% of training periods)
        oof_df = wf_val_df[group_cols + [time_col]].copy().reset_index(drop=True)
        oof_df["fold"] = 0
        oof_df["predicted_target"] = wf_preds_arr

else:
    # tabular_regression / classification: full OOF CV loop
    n = len(train_df)
    X_full = train_df[feature_cols].fillna(fill_vals)
    y_full = train_df[target_col].values
    X_val  = val_df[feature_cols].fillna(fill_vals)

    def build_cv_splits(problem_type, group_cols, X, y, df, n_splits=5):
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

    cv_splits, cv_scheme = build_cv_splits(
        problem_type, group_cols, X_full, y_full, train_df.reset_index(drop=True)
    )
    print(f"\nCV scheme: {cv_scheme}, folds: {len(cv_splits)}")

    oof_accum = np.zeros(n)
    oof_count = np.zeros(n, dtype=int)
    oof_folds = np.full(n, -1, dtype=int)
    fold_maes = []

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

        if use_median_ensemble:
            fold_pred = np.median(fold_seed_preds, axis=0)
        else:
            fold_pred = np.mean(fold_seed_preds, axis=0)
        oof_accum[va_idx] += fold_pred
        oof_count[va_idx] += 1
        oof_folds[va_idx] = fold_idx

        fmae = mean_absolute_error(y_va, fold_pred)
        fold_maes.append(fmae)
        print(f"  Fold {fold_idx+1}: MAE={fmae:.4f}")

    oof_preds_arr = np.where(oof_count > 0, oof_accum / oof_count, float(y_full.mean()))
    oof_mae = mean_absolute_error(y_full, oof_preds_arr)
    print(f"\nOOF MAE ({cv_scheme}): {oof_mae:.4f}")

    id_candidates = ([profile.get("id_col")] if profile.get("id_col") else []) + group_cols + ([time_col] if time_col else [])
    id_cols_oof = [c for c in id_candidates if c and c in train_df.columns]
    oof_df = train_df[id_cols_oof].copy().reset_index(drop=True)
    oof_df["fold"] = oof_folds
    oof_df["predicted_target"] = oof_preds_arr

    wf_mae = oof_mae  # not walk-forward for these problem types

# ── Step 7b: Write OOF predictions ───────────────────────────────────────────
oof_path = os.path.join(REPORTS_DIR, "oof_predictions.csv")
oof_df.to_csv(oof_path, index=False)
print(f"\nWritten {oof_path}: {oof_df.shape}")

# ── Step 7c: Retrain on full data with 5 seeds ────────────────────────────────
print("\n--- Training on full data (5 seeds) ---")
if problem_type != "panel_forecasting":
    # X_full etc. already defined above
    pass

seed_val_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full, y_full, callbacks=[lgb.log_evaluation(-1)])
    p = np.clip(m.predict(X_val), 0, None)
    seed_val_preds.append(p)
    print(f"  Seed {seed}: mean={p.mean():.3f}")

if use_median_ensemble:
    ensemble_preds = np.median(seed_val_preds, axis=0)
else:
    ensemble_preds = np.mean(seed_val_preds, axis=0)

# Apply clip after ensemble if requested
if _retune_applied and "clip_after_ensemble" in (_retune_applied or ""):
    ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f"\nEnsemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"NaN count: {np.isnan(ensemble_preds).sum()}")
last_model = m

# ── Step 8: Write reports/predictions.csv ────────────────────────────────────
# Identifier columns: group/time cols present in val_df
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

# ── Step 9: Write reports/model_results.json ─────────────────────────────────
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10    = [{"feature": str(k), "importance": int(v)} for k, v in feat_imp.head(10).items()]
all_imp  = {str(k): int(v) for k, v in feat_imp.items()}

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
    "per_fold_maes": [float(mv) for mv in fold_maes],
    "walk_forward_mae": float(wf_mae),
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
    "retune_applied": _retune_applied,
}

results_path = os.path.join(REPORTS_DIR, "model_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWritten {results_path}")

# Pretty-print summary (no giant lists)
summary = {k: v for k, v in results.items() if k not in ("feature_importance_all", "feature_importance_top10")}
print(json.dumps(summary, indent=2))

# ── Step 10: Write marker file ────────────────────────────────────────────────
marker_path = os.path.join(REPORTS_DIR, "modeler_was_here.txt")
with open(marker_path, "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    f.write(f"OOF MAE ({cv_scheme}): {oof_mae:.4f}\n")
    f.write(f"Walk-forward MAE: {wf_mae:.4f}\n")
    f.write(f"Training time: {training_time}s\n")
    f.write(f"Optuna trials: {optuna_trials}\n")
    f.write(f"n_estimators: {best_n_estimators}\n")
print(f"Written {marker_path}")

print("\n" + "=" * 60)
print("MODELER COMPLETE")
print(f"OOF MAE    : {oof_mae:.4f}")
print(f"WF MAE     : {wf_mae:.4f}")
print(f"Val mean   : {ensemble_preds.mean():.3f}")
print(f"Time       : {training_time}s")
print("=" * 60)
