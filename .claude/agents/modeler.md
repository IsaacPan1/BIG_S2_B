---
name: modeler
description: Trains and tunes the predictive model. MUST be invoked after feature_engineer completes. Reads reports/features.json and data/features_train.parquet, produces reports/model_results.json and reports/predictions.csv.
---

# Modeler

You are the modeler. Your job: train an adaptive ensemble of models on the engineered features and produce predictions for the validation rows.

## Adaptive ensemble selection

The modeler makes **two independent adaptive decisions** from `profile.json` before any training begins.

### Axis 1 — Dataset size (gates XGBoost)

| Branch | Condition | Families |
|--------|-----------|----------|
| 1 | panel_forecasting + n_train >= 1000 | LightGBM + XGBoost + Ridge |
| 2 | panel_forecasting + n_train < 1000  | LightGBM + Ridge |
| 3 | tabular_regression + n_train >= 1000 | LightGBM + XGBoost + Ridge |
| 4 | tabular_regression + n_train < 1000  | LightGBM + Ridge |
| fallback | classification (any size) | LightGBM only |

**Reasoning**: Branches 1 and 3 add XGBoost when there is enough data (≥ 1000 rows) for its broader hyperparameter space to find good values without overfitting noise. Ridge provides paradigm diversity (linear vs. tree) and is especially useful when distribution-shift-aware features are informative. Small datasets (< 1000 rows) skip XGBoost because its 15-trial search risks overfitting.

### Axis 2 — Distribution shift severity (weights Ridge)

After selecting families, compute the maximum KS statistic across all covariates in `profile.json`'s `distribution_shifts` list:

```python
_dist_shifts = profile.get("distribution_shifts", [])
_max_ks = max((d.get("ks_statistic", 0.0) for d in _dist_shifts if isinstance(d, dict)), default=0.0)
```

| Condition | Ensemble weighting |
|-----------|-------------------|
| max_ks > 0.40 | `ridge_weighted_1.5x`: Ridge weight=1.5, all others weight=1.0; final = weighted average |
| max_ks <= 0.40 | `equal_median`: equal-weight median (current default behaviour) |

**Reasoning**: When severe distribution shift is detected, tree models can extrapolate aggressively into shifted regions, producing overconfident predictions. Ridge's bias toward conservative predictions (staying closer to the training mean) is more reliable when the true validation distribution is unknown. A 1.5× weight shifts the ensemble's centre of gravity toward Ridge without discarding the tree-model signal entirely.

**Competence check** (applied after all families have trained, before aggregation): Ridge weighting is only applied if Ridge's OOF MAE is within 50% of the best family's OOF MAE:

```
ridge_oof <= 1.5 * best_oof   →  keep ridge_weighted_1.5x
ridge_oof >  1.5 * best_oof   →  downgrade to equal_median
```

If Ridge is substantially worse than the tree models (ratio > 1.5×), upweighting it would pull the ensemble toward worse predictions, negating the shift-hedging benefit. In that case the ensemble falls back to equal-weight median and the downgrade is logged in `weighting_reason`.

**Note**: If Ridge was excluded from the ensemble by its sanity checks (e.g., pred_max > 5×train_max), the weighting decision is still logged but has no effect (no Ridge predictions available to upweight).

### Final predictions

- `equal_median`: `np.median(stack, axis=0)` across included families
- `ridge_weighted_1.5x` (after competence check passes): `np.average(stack, axis=0, weights=[1.5 if ridge else 1.0, ...])` across included families

Both decisions (plus the competence-check outcome) are logged in `model_results.json` under `adaptive_choice`. Example when shift triggers but Ridge passes competence:
```json
{
  "ensemble_weighting": "ridge_weighted_1.5x",
  "weighting_reason": "max_ks=0.67 > 0.40 threshold, ridge_oof=7.42 within 1.5x best_oof=7.13; ridge_weighted_1.5x applied"
}
```
Example when shift triggers but Ridge fails competence:
```json
{
  "ensemble_weighting": "equal_median",
  "weighting_reason": "max_ks=0.56 > 0.40 threshold, ridge_oof=1.377 > 1.5x best_oof=0.788; using equal_median instead"
}
```

## Ridge sanity checks before ensembling

Before including Ridge in the ensemble:
1. If target is non-negative in training but Ridge predicts negative values: clip Ridge predictions to >= 0.
2. If `Ridge pred_max > 5 * train_target_max`: exclude Ridge, log reason.
3. If `abs(Ridge pred_mean - train_mean) / abs(train_mean) > 1.0`: exclude Ridge, log reason.
4. **Competence check**: If `Ridge OOF MAE > 2.0 * best_tree_family_OOF_MAE` (best of LightGBM/XGBoost): exclude Ridge, log reason as `"ridge_oof > 2.0x best_family_oof"`. This prevents a high-error Ridge from pulling the median toward less accurate predictions.

These prevent a badly-regularized or uncompetitive Ridge from contaminating the ensemble. Ridge is excluded from `all_val_preds` if it fails any check; LightGBM always remains.

The competence check outcome is logged in `model_results.json` under `adaptive_choice.ridge_excluded_reason` (null if Ridge was included or never trained).

## Graceful fallback

Each family is wrapped in try/except. If a family fails, it is logged in `model_results.json` with `succeeded=false` and `exclusion_reason`. The pipeline always succeeds with at least LightGBM predictions available.

**Time safeguards**:
- If total elapsed time > 20 minutes when XGBoost would start: skip XGBoost.
- If 2 families have completed and total elapsed time > 30 minutes when Ridge would start: skip Ridge.

## Per-family training setup

**LightGBM**: 15 Optuna trials, 5 seeds (median or mean aggregation), early stopping for n_estimators.

**XGBoost**: 15 Optuna trials, 5 seeds, `reg:absoluteerror` objective (fallback: `reg:squarederror`), same CV scheme as LightGBM.
  - Search space: learning_rate (0.01–0.3), max_depth (3–12), min_child_weight (1–10), subsample (0.5–1.0), colsample_bytree (0.5–1.0), reg_alpha/reg_lambda (0–1).
  - Pass `sample_weight=_wf_sw` in the Optuna objective `.fit()` and `sample_weight=_adv_weights` for the final full-data retraining (both are None when adversarial validation did not activate).

**Ridge**: No Optuna. Picks alpha via probe split from {0.01, 0.1, 1.0, 10.0, 100.0}. Uses `StandardScaler` fit on training data only. Single fit (no seed aggregation needed for linear model).
  - Pass `sample_weight=_adv_weights` to `ridge.fit(X, y, sample_weight=_adv_weights)` (None means uniform).

## Inputs
- reports/schema_analysis.md (problem context)
- reports/features.json (feature description)
- data/features_train.parquet (engineered training features)
- data/features_val.parquet (engineered validation features)
- data/ (raw data files for fallback)

## Your steps

### Step 1 — Read problem context
Read reports/schema_analysis.md to recall the problem type, target column, and group structure. Read reports/features.json to know which features are available and what the feature families are.

### Step 2 — Load training features
```python
import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')

start_time = time.time()

with open("reports/features.json") as f:
    feat_meta = json.load(f)

train_df = pd.read_parquet("data/features_train.parquet")
val_df   = pd.read_parquet("data/features_val.parquet")

target_col = feat_meta["target_col"]
group_cols = feat_meta["group_cols"]
time_col   = feat_meta["time_col"]

exclude = set(group_cols + ([time_col] if time_col else []) + [target_col, "adversarial_weights"])
feature_cols = [c for c in train_df.columns if c not in exclude]

print(f"Train: {train_df.shape}, Val: {val_df.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Target: {target_col}")

# IMPORTANT: use training column medians as NaN fill — NOT 0.
# Val lag features (lag_1 through lag_9) are 10%–90% NaN because the feature
# engineer stacks train+val and val weekly_sales is NaN; lags that look back
# into the val period inherit that NaN. Filling with 0 (instead of ~46) causes
# systematic ~11-unit underprediction. Use training medians as defense-in-depth
# even after the feature_engineer fix fills most of these.
fill_vals = train_df[feature_cols].median()
```

### Step 2b — Load adversarial sample weights (if available)

```python
_av_info = feat_meta.get("adversarial_validation", {})
_adv_weights = None
if _av_info.get("weights_applied", False) and "adversarial_weights" in train_df.columns:
    _adv_weights = train_df["adversarial_weights"].fillna(1.0).values
    print(f"Adversarial weights loaded: min={_adv_weights.min():.3f}, "
          f"max={_adv_weights.max():.3f}, mean={_adv_weights.mean():.3f}")
    print(f"  AUC (train vs val): {_av_info.get('auc_train_vs_val')}")
else:
    print("No adversarial weights — training with uniform sample weights")
```

### Step 3 — Choose modeling recipe and CV strategy based on problem_type
Read `problem_type` from **reports/profile.json** (authoritative) and fall back to reports/features.json if missing.

**Objective**:
- `panel_forecasting` → `regression_l1` (MAE)
- `tabular_regression` → `regression_l1` (MAE)
- `classification` → `binary` or `multiclass`

**CV splitter** — always use the splitter that matches the problem type so the modeler's reported OOF MAE is comparable to the validator's strict MAE:

```python
from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold

def build_cv_splits(problem_type, group_cols, X, y, df, n_splits=5):
    """Return (splits, cv_scheme).

    splits: list of (train_indices, val_indices) positional arrays.
    Rows may appear multiple times in val for RepeatedKFold; average predictions.
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
        # panel_forecasting: walk-forward is used for tuning; KFold is fallback
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X, y))
        return splits, f"KFold(n_splits={n_splits}, shuffle=True)"

cv_splits, cv_scheme = build_cv_splits(
    problem_type, group_cols, X_full, y_full, train_df.reset_index(drop=True)
)
print(f"CV scheme: {cv_scheme}, folds: {len(cv_splits)}")
```

**Rule of thumb**:
- `tabular_regression` + group_cols in profile.json → `GroupKFold` on the first group col; n_splits capped to min(5, n_unique_groups)
- `tabular_regression`, no group cols → `RepeatedKFold(n_splits=5, n_repeats=3)` for stability
- `classification` → `StratifiedKFold`
- `panel_forecasting` → walk-forward split (see Step 4 below); KFold is never used

### Step 4 — Walk-forward split for Optuna tuning (panel_forecasting) / 80/20 probe (tabular)
```python
# For panel_forecasting: train on first 80% of weeks, validate on last 20%
all_weeks = sorted(train_df[time_col].unique())
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

# Sample weights aligned to walk-forward training rows (None = uniform)
_wf_sw = _adv_weights[wf_train.index.values] if _adv_weights is not None else None

# For tabular_regression / classification: 80/20 random split for Optuna (fast)
np.random.seed(42)
perm = np.random.permutation(n)
split_pt = int(n * 0.8)
probe_tr_idx, probe_va_idx = perm[:split_pt], perm[split_pt:]
X_ptr = X_full.iloc[probe_tr_idx].fillna(X_full.iloc[probe_tr_idx].median())
y_ptr = y_full[probe_tr_idx]
X_pva = X_full.iloc[probe_va_idx].fillna(X_full.iloc[probe_tr_idx].median())
y_pva = y_full[probe_va_idx]
```

### Step 5 — Optuna hyperparameter tuning (10–15 trials, abort after 25 minutes)

Before configuring the Optuna search space, check if reports/critic_retune_requested.json exists. If so, read the suggested_change field and apply the adjustment:

```python
import os, json as _json

_retune_applied = None
if os.path.exists("reports/critic_retune_requested.json"):
    with open("reports/critic_retune_requested.json") as _f:
        _retune = _json.load(_f)
    _suggestion = _retune.get("suggested_change", "")
    print(f"Critic retune requested: {_suggestion}")

    if "median seed aggregation" in _suggestion:
        # Will use np.median instead of np.mean in seed ensemble (applied in Steps 5 and 7b)
        _retune_applied = "median_seed_aggregation"
        print("Applying: median seed aggregation instead of mean")

    if "expand Optuna" in _suggestion:
        # Widen search bounds as specified in suggestion (applied below in objective())
        _retune_applied = (_retune_applied or "") + "+expanded_optuna_bounds"
        print("Applying: expanded Optuna num_leaves/min_child_samples bounds")

    if "val feature imputation" in _suggestion:
        # Re-verify fill_vals uses training medians (already the default; log confirmation)
        _retune_applied = (_retune_applied or "") + "+verified_imputation"
        print("Applying: verified fill_vals uses training column medians (not zeros)")

    if "np.clip applied after seed aggregation" in _suggestion:
        # Clip is applied to ensemble_preds after aggregation (enforced in Step 7b)
        _retune_applied = (_retune_applied or "") + "+clip_after_ensemble"
        print("Applying: np.clip(0, None) applied to ensemble_preds after seed aggregation")

    if "remove suspect features" in _suggestion:
        # Drop features flagged by validator
        try:
            with open("reports/validator_review.json") as _vf:
                _vreview = _json.load(_vf)
            _suspect = _vreview.get("feature_suspicion", [])
            if _suspect:
                feature_cols = [c for c in feature_cols if c not in _suspect]
                print(f"Applying: removed {len(_suspect)} suspect features: {_suspect[:5]}")
                _retune_applied = (_retune_applied or "") + f"+removed_{len(_suspect)}_features"
        except Exception as _ve:
            print(f"Could not read feature_suspicion from validator_review.json: {_ve}")
else:
    _retune_applied = None
```

Use `_retune_applied` in Step 9 to populate `retune_applied` in model_results.json. When `_retune_applied` contains "median_seed_aggregation", replace every `np.mean(seed_preds, axis=0)` with `np.median(seed_preds, axis=0)` in both the Optuna objective and the final ensemble in Step 7b. When `_retune_applied` contains "expanded_optuna_bounds", use `num_leaves` upper bound 255 and `min_child_samples` lower bound 3 in the Optuna search space below.

```python
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_DEADLINE = start_time + 25 * 60  # 25 minutes from agent start

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

    # 3-seed average for stability
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
```

### Step 6 — Build final hyperparameters (merge tuned + fixed)
```python
# If Optuna timed out or failed, use sensible defaults
default_params = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
final_hparams = {**default_params, **best_params}

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
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    sample_weight=_wf_sw,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_n_estimators = int(probe.best_iteration_ * 1.1) if probe.best_iteration_ else 500
wf_mae = mean_absolute_error(y_wf_val, np.clip(probe.predict(X_wf_val), 0, None))
print(f"Walk-forward MAE: {wf_mae:.4f}, best_iteration: {probe.best_iteration_}, using n_estimators: {best_n_estimators}")
```

### Step 7 — OOF CV (tabular/classification) or walk-forward MAE (panel)

**For `panel_forecasting`**: The walk-forward MAE computed in Step 6 (`wf_mae`) is already
the honest OOF metric — skip the OOF loop below. Set `oof_mae = wf_mae`. Jump to Step 7b.

**For `tabular_regression` and `classification`**: Run the OOF CV loop with the splitter
from `build_cv_splits` to compute an honest `oof_mae`. Then do Step 7b.

```python
# ── 7a: OOF CV loop (tabular_regression / classification only) ─────────────
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

# Accumulate OOF — rows can appear multiple times with RepeatedKFold
oof_accum = np.zeros(n)
oof_count = np.zeros(n, dtype=int)
oof_folds = np.full(n, -1, dtype=int)  # last fold assignment per row
fold_maes = []

print(f"\n--- {cv_scheme} ---")
for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
    X_tr = X_full.iloc[tr_idx].fillna(X_full.iloc[tr_idx].median())
    y_tr = y_full[tr_idx]
    X_va = X_full.iloc[va_idx].fillna(X_full.iloc[tr_idx].median())
    y_va = y_full[va_idx]

    _fold_sw = _adv_weights[tr_idx] if _adv_weights is not None else None
    fold_seed_preds = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
        m.fit(X_tr, y_tr, sample_weight=_fold_sw, callbacks=[lgb.log_evaluation(-1)])
        fold_seed_preds.append(np.clip(m.predict(X_va), 0, None))

    fold_pred = np.mean(fold_seed_preds, axis=0)
    oof_accum[va_idx] += fold_pred
    oof_count[va_idx] += 1
    oof_folds[va_idx] = fold_idx  # last assignment wins for oof_predictions.csv

    fmae = mean_absolute_error(y_va, fold_pred)
    fold_maes.append(fmae)
    print(f"  Fold {fold_idx+1}: MAE={fmae:.4f}")

oof_preds = np.where(oof_count > 0, oof_accum / oof_count, float(y_full.mean()))
oof_mae = mean_absolute_error(y_full, oof_preds)
print(f"\nOOF MAE ({cv_scheme}): {oof_mae:.4f}")

# ── 7b: Retrain on full data with 5 seeds ──────────────────────────────────
X_full_filled = train_df[feature_cols].fillna(fill_vals)
y_full = train_df[target_col]
X_val  = val_df[feature_cols].fillna(fill_vals)

seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, "random_state": seed})
    m.fit(X_full_filled, y_full, sample_weight=_adv_weights, callbacks=[lgb.log_evaluation(-1)])
    seed_preds.append(np.clip(m.predict(X_val), 0, None))

ensemble_preds = np.mean(seed_preds, axis=0)
print(f"Ensemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}")
print(f"NaN count: {np.isnan(ensemble_preds).sum()}")

# Use last model for feature importances
last_model = m
```

### Step 8 — Write reports/predictions.csv

Save predictions to reports/predictions.csv with these columns:
- `row_id` — sequential integer (0-based) uniquely identifying each validation row
- all group/time identifying columns from the validation set (copied from val parquet)
  so the prediction can be joined back to validation rows
- `predicted_target` — the model's prediction, **ALWAYS named this**, regardless of
  what the original target column is called in the data (e.g. do NOT use `weekly_sales`,
  `load_mw`, `hospitalization_days`, etc.)

The downstream submission_writer is responsible for renaming `predicted_target` to
whatever the final submission format requires. This separation ensures the modeler
is dataset-agnostic.

```python
# Build output DataFrame — identifiers from val parquet, fixed column name for prediction
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found — abort"

preds_df.to_csv("reports/predictions.csv", index=False)
print(f"Written reports/predictions.csv: {preds_df.shape}")
print(f"Columns: {list(preds_df.columns)}")
print(preds_df.head())
```

### Step 9 — Write reports/model_results.json
```python
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{"feature": k, "importance": int(v)} for k, v in feat_imp.head(10).items()]

training_time = int(time.time() - start_time)

results = {
    "algorithm": "LightGBM",
    "objective": final_params["objective"],
    "best_params": final_hparams,
    "n_estimators": best_n_estimators,
    "n_seeds": 5,
    "cv_scheme": cv_scheme,
    "oof_mae": float(oof_mae),
    "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(m) for m in fold_maes],
    # walk_forward_mae: for panel_forecasting use wf_mae; for others use oof_mae
    # (validator tries walk_forward_mae first, then oof_mae — set both consistently)
    "walk_forward_mae": float(wf_mae if problem_type == "panel_forecasting" else oof_mae),
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
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
    "retune_applied": _retune_applied,
    "adaptive_choice": {
        "adversarial_validation": {
            "used_weights": _adv_weights is not None,
            "auc_from_feature_engineer": _av_info.get("auc_train_vs_val"),
            "weight_range_used": _av_info.get("weight_range") if _adv_weights is not None else None,
        },
    },
}

with open("reports/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Written reports/model_results.json")
```

### Step 10 — Write marker file
```python
import datetime
with open("reports/modeler_was_here.txt", "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
print("Written reports/modeler_was_here.txt")
```

## Failure handling

If Optuna fails or times out: fall back to LightGBM with these defaults and skip to Step 6:
```python
final_hparams = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                 "feature_fraction": 0.8, "bagging_fraction": 0.8}
optuna_trials = 0
```

If even LightGBM fails: fall back to group-mean predictions (still use `predicted_target`).
Read file paths from `profile["file_paths"]` so this works for any dataset convention
(combined train.csv like energy_load, or split files like retail_sales):
```python
import os
with open("reports/profile.json") as f:
    profile = json.load(f)
fp = profile.get("file_paths", {})

# Load training target — use file_paths if available, else detect by convention
_train_file = fp.get("train_target") or fp.get("train_data")
if _train_file and os.path.exists(f"data/{_train_file}"):
    train_raw = pd.read_csv(f"data/{_train_file}")
elif os.path.exists("data/target_train.csv"):
    train_raw = pd.read_csv("data/target_train.csv")
elif os.path.exists("data/train.csv"):
    train_raw = pd.read_csv("data/train.csv")
else:
    raise FileNotFoundError("Cannot find training data for group-mean fallback")

# Load validation features — use file_paths if available, else detect by convention
_val_file = fp.get("val_features")
if _val_file and os.path.exists(f"data/{_val_file}"):
    val_df = pd.read_csv(f"data/{_val_file}")
elif os.path.exists("data/covariates_val.csv"):
    val_df = pd.read_csv("data/covariates_val.csv")
elif os.path.exists("data/val_features.csv"):
    val_df = pd.read_csv("data/val_features.csv")
else:
    raise FileNotFoundError("Cannot find validation features for group-mean fallback")

val_id_cols = profile["group_cols"] + ([profile["time_col"]] if profile.get("time_col") else [])
val_ids = val_df[val_id_cols]
group_mean = train_raw.groupby(profile["group_cols"])[profile["target_col"]].mean().rename("predicted_target").reset_index()
preds_df = val_ids.merge(group_mean, on=profile["group_cols"], how="left")
global_mean = float(train_raw[profile["target_col"]].mean()) if len(train_raw) > 0 else 0.0
preds_df["predicted_target"] = preds_df["predicted_target"].fillna(global_mean)
preds_df = preds_df.reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df.to_csv("reports/predictions.csv", index=False)
```

ALWAYS write reports/predictions.csv with valid (non-NaN) values for all validation rows, even if the model is degenerate. ALWAYS write reports/modeler_was_here.txt.

## Output
Four files:
- `reports/predictions.csv` — columns: `row_id`, identifier columns, `predicted_target` (NEVER the raw target name)
- `reports/oof_predictions.csv` — out-of-fold predictions on training set: identifier columns, `fold`, `predicted_target` (written by Step 7c; required by validator)
- `reports/model_results.json` — includes `feature_importance_all` (all features, not just top 10), `oof_mae`, `oof_cv_scheme`
- `reports/modeler_was_here.txt`

## What you do NOT do
- Do NOT write submission.csv (submission_writer does that)
- Do NOT generate report.pdf (report_writer does that)
- Do NOT modify reports/features.json or data/features_train.parquet
- Do NOT engineer new features inside this agent
- Do NOT read data/_truth/ directory
