---
name: modeler
description: Trains and tunes the predictive model. MUST be invoked after feature_engineer completes. Reads reports/features.json and data/features_train.parquet, produces reports/model_results.json and reports/predictions.csv.
---

# Modeler

You are the modeler. Your job: train an adaptive ensemble of models on the engineered features and produce predictions for the validation rows.

## Adaptive ensemble selection

The modeler makes **two independent adaptive decisions** from `profile.json` before any training begins.

### Axis 1 — Dataset size (gates XGBoost)

The ensemble path is determined by `problem_subtype` (from `profile.json`), not just `problem_type`.
`ordinal_regression` is treated **identically to `continuous_regression`** — it gets the full ensemble.

| Branch | Condition | Families | `ensemble_path_used` |
|--------|-----------|----------|---------------------|
| 1 | panel_forecasting + n_train >= 1000 | LightGBM + XGBoost + CatBoost + Ridge | `full_regression_ensemble` |
| 2 | panel_forecasting + n_train < 1000  | LightGBM + Ridge | `full_regression_ensemble` |
| 3 | continuous_regression or ordinal_regression + n_train >= 1000 | LightGBM + XGBoost + CatBoost + Ridge | `full_regression_ensemble` |
| 4 | continuous_regression or ordinal_regression + n_train < 1000  | LightGBM + Ridge | `full_regression_ensemble` |
| fallback | binary_classification or multiclass_classification (any size) | LightGBM only | `classification_fallback` |

**Reasoning**: Branches 1 and 3 add XGBoost when there is enough data (≥ 1000 rows) for its broader hyperparameter space to find good values without overfitting noise. Ridge provides paradigm diversity (linear vs. tree) and is especially useful when distribution-shift-aware features are informative. Small datasets (< 1000 rows) skip XGBoost because its 15-trial search risks overfitting.

`ordinal_regression` (e.g. hospitalization_days 0–14, severity score 1–5, visit_count 0+) gets the full regression ensemble because: (a) values have natural ordering, (b) distance matters for MAE evaluation, and (c) the full ensemble's MAE-optimised objective handles integer targets correctly without any modification. Classification path (`binary_classification`, `multiclass_classification`) uses LightGBM only — multi-family probability aggregation is a known limitation not yet implemented.

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

### Axis 3 — CatBoost time-gated conditional addition

After XGBoost training completes, evaluate whether to add CatBoost as a fourth tree family. CatBoost runs only when ALL four conditions are met:

1. `problem_type` ∈ {`panel_forecasting`, `tabular_regression`, `classification`}
2. `n_train >= 500` — ordered boosting requires sufficient data
3. `catboost` can be imported without error
4. `elapsed_minutes < 40` where `elapsed_minutes = (time.time() - pipeline_start_time) / 60`

If ANY condition fails: skip CatBoost silently, log `skip_reason`, continue to Ridge.

**Rationale**: CatBoost's symmetric oblivious trees and ordered boosting give it different inductive biases from LightGBM and XGBoost. Even when LGB and XGB produce similar OOF predictions, CatBoost may capture patterns neither tree family found. The 40-minute guard preserves budget for Ridge, validator, critic, submission_writer, and report_writer.

**Import guard** — wrap at the start of the CatBoost block:
```python
try:
    import catboost as cb
    _catboost_available = True
except ImportError:
    _catboost_available = False
```

**Conditional check** before training:
```python
elapsed_minutes = (time.time() - pipeline_start_time) / 60
_should_run_cb = (
    _catboost_available
    and n_train >= 500
    and elapsed_minutes < 40
    and (
        problem_type in ("panel_forecasting", "tabular_regression", "classification")
        or problem_subtype in ("ordinal_regression", "continuous_regression")
    )
)
```

**CatBoost Optuna search** (10 trials, fewer than LGB/XGB to manage compute):
- `learning_rate`: 0.02–0.10 log-uniform
- `depth`: 4–8
- `l2_leaf_reg`: 1–10
- `iterations`: 400 (fixed)
- `loss_function`: `'MAE'` for regression, `'Logloss'` for classification
- `cat_features`: pass categorical column indices explicitly
- `verbose=False`, `allow_writing_files=False`
- 5-seed multi-seed aggregation for final predictions
- `sample_weight=_adv_weights` if adversarial weights are available

**CatBoost competence check**: After CatBoost trains, compute its walk-forward OOF MAE and apply the 1.5× rule:
- `cb_oof <= 1.5 * best_tree_oof` → include CatBoost in ensemble (`best_tree_oof = min(lgb_oof, xgb_oof)`)
- `cb_oof > 1.5 * best_tree_oof` → exclude, log `excluded_too_weak`

Unlike the OOF similarity check (used for LGB/XGB redundancy), CatBoost uses only the competence check. Its architectural differences mean OOF correlation is not a reliable proxy for ensemble diversity.

**CatBoost sanity checks** (same thresholds as Ridge):
1. If target is non-negative in training but CatBoost predicts negatives: clip to >= 0.
2. If `cb_pred_max > 5 * train_target_max`: exclude, log reason.
3. If `abs(cb_pred_mean − train_mean) / abs(train_mean) > 1.0`: exclude, log reason.

If any sanity check fails or training errors: set `catboost.succeeded = False`, continue with 3-family ensemble.

### Final predictions

Three blend paths in priority order:

1. **`ridge_weighted_1.5x`** (Axis 2 shift-hedge; takes precedence when triggered): `np.average(stack, axis=0, weights=[1.5 if ridge else 1.0, ...])` across included families. Only active when max_ks > 0.40 AND Ridge passes competence.
2. **`inverse_mae_weighted`** (OOF tilt; applied on the equal_median path only): weights w_i = (1/oof_i) / Σ(1/oof_j) over included families. Gated by a walk-forward holdout comparison — used only when its holdout MAE ≤ equal-median holdout MAE, so it can never be worse than equal_median on the measured fold.
3. **`equal_median`** (default): `np.median(stack, axis=0)` across included families (2–4 families).

The active blend, per-family weights, and holdout MAE comparison are logged in `model_results.json` under `adaptive_choice.ensemble_blend`, `adaptive_choice.blend_weights`, `adaptive_choice.blend_holdout_mae_equal`, and `adaptive_choice.blend_holdout_mae_inv`.

Both Axis 2 decisions (plus the competence-check outcome) are also logged in `model_results.json` under `adaptive_choice`. Example when shift triggers but Ridge passes competence:
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
4. **Competence check**: If `Ridge OOF MAE > 1.5 * best_tree_family_OOF_MAE` (best of LightGBM/XGBoost): exclude Ridge, log reason as `"ridge_oof > 1.5x best_family_oof"`. This prevents a high-error Ridge from pulling the median toward less accurate predictions.

These prevent a badly-regularized or uncompetitive Ridge from contaminating the ensemble. Ridge is excluded from `all_val_preds` if it fails any check; LightGBM always remains.

The competence check outcome is logged in `model_results.json` under `adaptive_choice.ridge_excluded_reason` (null if Ridge was included or never trained).

## Graceful fallback

Each family is wrapped in try/except. If a family fails, it is logged in `model_results.json` with `succeeded=false` and `exclusion_reason`. The pipeline always succeeds with at least LightGBM predictions available.

**Time safeguards**:
- If total elapsed time > 20 minutes when XGBoost would start: skip XGBoost.
- Evaluate `elapsed_minutes = (time.time() - pipeline_start_time) / 60` after XGBoost. If `elapsed_minutes >= 40`: skip CatBoost (logged as `skipped_no_time`).
- If 2+ families have completed (or been attempted) and total elapsed time > 50 minutes when Ridge would start: skip Ridge.

## Per-family training setup

**LightGBM**: 15 Optuna trials, 5 seeds (median or mean aggregation), early stopping for n_estimators.
  - After training, store the walk-forward OOF MAE as **`lgb_oof_mae`** — required by the CatBoost competence check in Step 7c.

**XGBoost**: 15 Optuna trials, 5 seeds, `reg:absoluteerror` objective (fallback: `reg:squarederror`), same CV scheme as LightGBM.
  - Search space: learning_rate (0.01–0.3), max_depth (3–12), min_child_weight (1–10), subsample (0.5–1.0), colsample_bytree (0.5–1.0), reg_alpha/reg_lambda (0–1).
  - Pass `sample_weight=_wf_sw` in the Optuna objective `.fit()` and `sample_weight=_adv_weights` for the final full-data retraining (both are None when adversarial validation did not activate).
  - After training, store the walk-forward OOF MAE as **`xgb_oof_mae`** — required by the CatBoost competence check in Step 7c.

**CatBoost**: 10 Optuna trials (conditional on Axis 3 check), 5 seeds, `MAE` loss for regression / `Logloss` for classification.
  - Search space: `learning_rate` (0.02–0.10 log-uniform), `depth` (4–8), `l2_leaf_reg` (1–10), `iterations=400` (fixed).
  - Pass `cat_features` (list of categorical column indices) explicitly; set `verbose=False`, `allow_writing_files=False`.
  - Pass `sample_weight=_adv_weights` if adversarial weights are available.
  - Activation conditions: `_catboost_available AND n_train >= 500 AND elapsed_minutes < 40`.
  - Competence check: `cb_oof <= 1.5 * best_tree_oof` (best of LGB/XGB OOF MAE).
  - Sanity checks: pred_max ≤ 5×train_max; |pred_mean − train_mean| / |train_mean| ≤ 1.0.

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
pipeline_start_time = start_time  # aliased for CatBoost Axis 3 activation check

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

_optuna_reflection = {"pinned_params": [], "recentered": False,
                      "best_mae_before": None, "best_mae_after": None}

try:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=25*60, catch=(Exception,))
    best_params = study.best_params
    _optuna_reflection["best_mae_before"] = float(study.best_value)
    _optuna_reflection["best_mae_after"] = float(study.best_value)
    optuna_trials = len(study.trials)

    # ── Boundary reflection: intra-run only; NEVER writes critic_retune_attempted.txt ──
    # Detect params pinned at/near search bounds: ints exactly at bound; floats within 5% of range.
    _lgb_bounds = {
        "learning_rate": (0.01, 0.1, "log"),
        "num_leaves":    (15, 127, "int"),
        "min_child_samples": (5, 60, "int"),
        "feature_fraction":  (0.5, 1.0, "float"),
        "bagging_fraction":  (0.5, 1.0, "float"),
    }
    _pinned = []
    _shifted_bounds = {}  # param → (new_lo, new_hi)
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
                # Recenter geometrically around best value
                _shifted_bounds[_pn] = (max(1e-5, _v / 3.0), min(1.0, _v * 3.0))
            else:
                _d = 0.25 * _rng
                _shifted_bounds[_pn] = (max(0.01, _v - _d), min(1.0, _v + _d))
    _optuna_reflection["pinned_params"] = _pinned

    if _pinned and time.time() < TUNING_DEADLINE - 120:   # need ≥ 2 min remaining
        print(f"Boundary reflection: {_pinned} pinned — recentering and running second 15-trial study")

        def _reflect_objective(trial):
            if time.time() > TUNING_DEADLINE:
                raise optuna.exceptions.TrialPruned()
            def _rb(pn, lo, hi, log=False):
                nl, nh = _shifted_bounds.get(pn, (lo, hi))
                if log:
                    return trial.suggest_float(pn, float(nl), float(nh), log=True)
                elif lo == int(lo) and hi == int(hi):
                    return trial.suggest_int(pn, max(1, int(nl)), max(int(nl)+1, int(nh)))
                else:
                    return trial.suggest_float(pn, float(nl), float(nh))
            _rp = {
                "objective": "regression_l1", "metric": "mae", "n_estimators": 500,
                "learning_rate": _rb("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": _rb("num_leaves", 15, 127),
                "min_child_samples": _rb("min_child_samples", 5, 60),
                "feature_fraction": _rb("feature_fraction", 0.5, 1.0),
                "bagging_fraction": _rb("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1, "n_jobs": -1,
            }
            _rmaes = []
            for _rs in [42, 7, 123]:
                _rm = lgb.LGBMRegressor(**{**_rp, "random_state": _rs})
                _rm.fit(X_wf_train, y_wf_train, sample_weight=_wf_sw,
                        eval_set=[(X_wf_val, y_wf_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                _rmaes.append(mean_absolute_error(y_wf_val, np.clip(_rm.predict(X_wf_val), 0, None)))
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
            print(f"Reflection did not improve (original={study.best_value:.4f}, reflected={_study2.best_value if _study2.trials else 'N/A'})")
    # ── End boundary reflection ───────────────────────────────────────────────────────

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
lgb_wf_val_preds = np.clip(probe.predict(X_wf_val), 0, None)  # stored for inverse-MAE gate in Step 7d
wf_mae = mean_absolute_error(y_wf_val, lgb_wf_val_preds)
probe_mae_80_20 = float(wf_mae)   # snapshot of Optuna probe MAE before any update
print(f"Walk-forward MAE: {wf_mae:.4f}, best_iteration: {probe.best_iteration_}, using n_estimators: {best_n_estimators}")
```

### Step 7a — LightGBM training: OOF CV and full-data retrain

Initialize the shared prediction collector before any family trains:

```python
all_val_preds = {}  # keyed by family name; populated by Steps 7a–7d
```

**For `panel_forecasting`**: The walk-forward MAE computed in Step 6 (`wf_mae`) is already
the honest OOF metric — skip the OOF loop below. Set `oof_mae = wf_mae` and proceed to
the full-data retrain section below.

**For `tabular_regression` and `classification`**: Run the OOF CV loop with the splitter
from `build_cv_splits` to compute an honest `oof_mae`, then proceed to the full-data retrain.

```python
# ── OOF CV loop (tabular_regression / classification only) ─────────────────
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

# ── Full-data retrain on all 5 seeds ───────────────────────────────────────
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

# Register LightGBM in the shared collector; lgb_oof_mae used by Step 7c competence check
all_val_preds["lgb"] = ensemble_preds
lgb_oof_mae = float(oof_mae)
```

### Step 7b — XGBoost training

Train XGBoost following the walk-forward Optuna (15 trials) + 5-seed full-data retrain
pattern, using the XGBoost-specific setup from the **Per-family training setup** section.

Key requirements:
- **Time guard**: if `elapsed_minutes > 20` at the start of this step, skip XGBoost entirely
- Objective: `reg:absoluteerror` (fallback `reg:squarederror`)
- Pass `_wf_sw` to Optuna `.fit()`; pass `_adv_weights` to final full-data retrain
- 15 Optuna trials with the XGBoost search space

After XGBoost training completes, register its predictions, store its OOF MAE, and
store its walk-forward holdout predictions for the inverse-MAE gate in Step 7d:

```python
all_val_preds["xgb"] = xgb_ensemble_preds   # 5-seed median of val predictions
xgb_oof_mae = float(xgb_wf_mae)             # walk-forward OOF; used by Step 7c competence check
# Walk-forward holdout predictions: best-params 3-seed XGB ensemble predicted on X_wf_val
# (trained on wf_train only — same holdout used for Optuna tuning; out-of-sample).
xgb_wf_val_preds = xgb_wf_best_preds        # stored for inverse-MAE gate in Step 7d
```

If XGBoost is skipped or raises an exception: set `xgb_result["succeeded"] = False`,
`xgb_oof_mae = float("inf")` so Step 7c's competence check falls back to `lgb_oof_mae` only,
and leave `xgb_wf_val_preds` undefined (the gate in Step 7d handles missing entries via `None`).

### Step 7c — CatBoost training (conditional, Axis 3)

After XGBoost has trained (or been skipped) and before Ridge, evaluate the Axis 3 conditions and train CatBoost if all pass. Record timing and outcome in `_cb_result` so Step 9 can log it.

```python
# ── CatBoost import guard ──────────────────────────────────────────────────
try:
    import catboost as _cb_module
    _catboost_available = True
except ImportError:
    _catboost_available = False

# ── Axis 3 activation check ───────────────────────────────────────────────
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
        # Identify categorical feature indices for CatBoost
        _cat_cols = [c for c in feature_cols if train_df[c].dtype.name in ("object", "category")]
        _cat_indices = [feature_cols.index(c) for c in _cat_cols]

        _cb_loss = "MAE" if problem_type in ("panel_forecasting", "tabular_regression") else "Logloss"
        _cb_eval_metric = "MAE" if _cb_loss == "MAE" else "Logloss"

        # ── Optuna tuning (10 trials) ─────────────────────────────────────
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
                "random_seed": 42,
            }
            if _cat_indices:
                _p["cat_features"] = _cat_indices
            _cb_maes = []
            for _seed in [42, 7]:
                _p["random_seed"] = _seed
                _m = _cb_module.CatBoostRegressor(**_p) if _cb_loss == "MAE" else _cb_module.CatBoostClassifier(**_p)
                _wf_sw_cb = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
                _m.fit(
                    X_wf_train.values, y_wf_train.values,
                    sample_weight=_wf_sw_cb,
                    eval_set=(_cb_module.Pool(X_wf_val.values, y_wf_val.values)),
                    verbose=False,
                )
                _cb_maes.append(mean_absolute_error(y_wf_val, _m.predict(X_wf_val.values)))
            return float(np.mean(_cb_maes))

        _cb_study = optuna.create_study(direction="minimize")
        _cb_study.optimize(_cb_objective, n_trials=10, timeout=15*60, catch=(Exception,))
        _cb_best = _cb_study.best_params
        print(f"CatBoost Optuna: {len(_cb_study.trials)} trials, best MAE={_cb_study.best_value:.4f}")

        # ── Walk-forward OOF for competence check ─────────────────────────
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
            _m = (_cb_module.CatBoostRegressor(**_cb_final_params)
                  if _cb_loss == "MAE" else _cb_module.CatBoostClassifier(**_cb_final_params))
            _wf_sw_cb = _adv_weights[wf_train.index.values] if _adv_weights is not None else None
            _m.fit(X_wf_train.values, y_wf_train.values, sample_weight=_wf_sw_cb, verbose=False)
            _cb_wf_preds_list.append(_m.predict(X_wf_val.values))
        _cb_wf_median = np.median(_cb_wf_preds_list, axis=0)
        _cb_oof_mae = float(mean_absolute_error(y_wf_val, _cb_wf_median))
        cb_wf_val_preds = _cb_wf_median   # stored for inverse-MAE gate in Step 7d
        print(f"CatBoost walk-forward OOF MAE: {_cb_oof_mae:.4f}")
        _cb_result["oof_mae"] = _cb_oof_mae

        # ── Competence check (1.5x best tree OOF) ─────────────────────────
        # lgb_oof_mae and xgb_oof_mae must be set during LGB/XGB training above
        _tree_oofs = []
        try:
            _tree_oofs.append(lgb_oof_mae)
        except NameError:
            pass
        try:
            _tree_oofs.append(xgb_oof_mae)
        except NameError:
            pass
        _best_tree_oof = min(_tree_oofs) if _tree_oofs else float('inf')

        _cb_passes_competence = _cb_oof_mae <= 1.5 * _best_tree_oof

        if not _cb_passes_competence:
            _cb_result["included_in_ensemble"] = False
            _cb_result["excluded_reason"] = (
                f"excluded_too_weak: cb_oof={_cb_oof_mae:.4f} > 1.5x best_tree_oof={_best_tree_oof:.4f}"
            )
            _cb_decision["competence_check_result"] = "excluded_too_weak"
            print(f"CatBoost excluded: {_cb_result['excluded_reason']}")
        else:
            # ── Full-data retrain with 5 seeds ─────────────────────────────
            _cb_seed_preds = []
            _train_target_max = float(y_full.max()) if len(y_full) > 0 else 1e9
            _train_target_mean = float(y_full.mean())
            for _seed in [42, 7, 123, 2024, 999]:
                _cb_final_params["random_seed"] = _seed
                _m = (_cb_module.CatBoostRegressor(**_cb_final_params)
                      if _cb_loss == "MAE" else _cb_module.CatBoostClassifier(**_cb_final_params))
                _m.fit(X_full_filled.values, y_full.values, sample_weight=_adv_weights, verbose=False)
                _cb_seed_preds.append(_m.predict(X_val.values))

            _cb_val_preds = np.median(_cb_seed_preds, axis=0)

            # Non-negative clip if target is non-negative
            if _train_target_mean >= 0:
                _cb_val_preds = np.clip(_cb_val_preds, 0, None)

            # Sanity checks
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
                print(f"CatBoost included in ensemble: OOF={_cb_oof_mae:.4f}, within 1.5x tree best={_best_tree_oof:.4f}")
            else:
                _cb_result["succeeded"] = True  # trained OK; excluded on sanity

    except Exception as _cb_exc:
        _cb_result["succeeded"] = False
        _cb_result["skip_reason"] = f"training_error: {_cb_exc}"
        _cb_decision["competence_check_result"] = "skipped_import_error"
        print(f"CatBoost training failed: {_cb_exc} — continuing with existing ensemble")

    _cb_result["training_time_seconds"] = float(time.time() - _cb_start)
    print(f"CatBoost block finished in {_cb_result['training_time_seconds']:.1f}s")
```

After this block, `all_val_preds` may now contain `"catboost"` as a fourth key alongside `"lgb"`, `"xgb"` (if trained), and `"ridge"` (added in Step 7d below). Step 7d combines all included families into `ensemble_preds`.

### Step 7d — Ridge training and ensemble aggregation

**Ridge training**

Ridge is included when the Axis 1 branch selected it (n_train ≥ 1,000). Time guard:
skip Ridge if `len(all_val_preds) >= 2` and `elapsed_minutes > 50`. Alpha is selected
via probe split from {0.01, 0.1, 1.0, 10.0, 100.0}; `StandardScaler` is fit on
training data only. Apply the Ridge sanity checks and competence check from the
**Ridge sanity checks before ensembling** section above.

```python
# Log top-5 Ridge coefficients for model_results.json
ridge_top_coefficients = [
    {"feature": f, "abs_coef": float(c)}
    for f, c in sorted(zip(feature_cols, np.abs(ridge.coef_)),
                       key=lambda x: -x[1])[:5]
]
# Walk-forward holdout predictions (fit Ridge on wf_train with best alpha, predict on wf_val)
# — out-of-sample; same holdout used for alpha selection; stored for inverse-MAE gate in Step 7d.
ridge_wf_val_preds = np.clip(ridge_wf_scaler.transform(X_wf_val) @ ridge_wf_fitted.coef_
                             + ridge_wf_fitted.intercept_, 0, None)
# If Ridge passes all checks, register it in the shared collector
all_val_preds["ridge"] = ridge_val_preds
```

**Ensemble aggregation**

After all families have been attempted, combine `all_val_preds` into the final
`ensemble_preds` used in Step 8. Three paths in priority order; the active blend,
per-family weights, and holdout gate MAEs are logged in `adaptive_choice`:

```python
# Per-family OOF MAEs for included families only
_family_oof = {}
if "lgb" in all_val_preds:
    _family_oof["lgb"] = lgb_oof_mae
if "xgb" in all_val_preds:
    _family_oof["xgb"] = xgb_oof_mae
if "catboost" in all_val_preds and _cb_result.get("oof_mae"):
    _family_oof["catboost"] = _cb_result["oof_mae"]
if "ridge" in all_val_preds and ridge_result.get("oof_mae"):
    _family_oof["ridge"] = ridge_result["oof_mae"]

_included_keys = list(all_val_preds.keys())   # e.g. ["lgb", "xgb", "catboost"]
_stack = np.array([all_val_preds[k] for k in _included_keys])

# ── Path 1: ridge_weighted_1.5x (Axis 2 shift-hedge; takes precedence) ──────
if ensemble_weighting == "ridge_weighted_1.5x" and "ridge" in all_val_preds:
    _w = [1.5 if k == "ridge" else 1.0 for k in _included_keys]
    ensemble_preds = np.average(_stack, axis=0, weights=_w)
    ensemble_blend = "ridge_weighted_1.5x"
    _blend_weights_log = {k: round(_w[i] / sum(_w), 4) for i, k in enumerate(_included_keys)}
    _blend_holdout_mae_equal = None
    _blend_holdout_mae_inv = None

# ── Paths 2/3: equal_median default with optional inverse-MAE tilt ───────────
else:
    _equal_preds = np.median(_stack, axis=0)

    # Inverse-MAE weights: w_i = (1/oof_i) / Σ_j(1/oof_j) over included families
    _inv_sum = sum(1.0 / _family_oof[k] for k in _included_keys
                   if k in _family_oof and _family_oof[k] > 0)
    _inv_w = [(1.0 / _family_oof[k] / _inv_sum
               if (k in _family_oof and _family_oof[k] > 0) else 0.0)
              for k in _included_keys]
    _inv_w_arr = np.array(_inv_w)
    _inv_preds = np.average(_stack, axis=0, weights=_inv_w_arr)

    # Gate: compare both blends on walk-forward holdout (out-of-sample for each family)
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
        _use_inv = False   # can't verify; fall back to equal_median

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

ensemble_disagreement = {
    "mean_disagreement": float(np.mean(np.std(_stack, axis=0))),
    "n_high_disagreement_rows": int(np.sum(np.std(_stack, axis=0) > ensemble_preds.mean())),
}
ensemble_oof_mae = min(_family_oof.values()) if _family_oof else float(lgb_oof_mae)
n_families_in_ensemble = len(_included_keys)
```

**Ordinal rounding (Change 3 — portability; silent no-op for any non-ordinal subtype)**

Only when `problem_subtype == "ordinal_regression"`: compare walk-forward MAE of raw
continuous preds vs round-then-clip to the integer target range; apply the lower-MAE
transform to the final val preds; tie → raw. For `continuous_regression`, `panel_forecasting`,
or any other subtype, this block is a silent no-op and `_postprocessing` stays as initialized.

```python
_postprocessing = {"ordinal_rounding_applied": False,
                   "ordinal_raw_wf_mae": None, "ordinal_round_wf_mae": None}

if problem_subtype == "ordinal_regression":
    _t_min = int(y_full.min())
    _t_max = int(y_full.max())
    _rounded_preds = np.clip(np.round(ensemble_preds), _t_min, _t_max)

    # Gate on LGB walk-forward holdout (always available; proxy for full-ensemble rounding benefit)
    _ord_raw_wf_mae   = float(mean_absolute_error(y_wf_val, lgb_wf_val_preds))
    _ord_round_wf_mae = float(mean_absolute_error(
        y_wf_val, np.clip(np.round(lgb_wf_val_preds), _t_min, _t_max)))
    _postprocessing["ordinal_raw_wf_mae"]   = _ord_raw_wf_mae
    _postprocessing["ordinal_round_wf_mae"] = _ord_round_wf_mae

    if _ord_round_wf_mae < _ord_raw_wf_mae:   # strict improvement required; tie → raw
        ensemble_preds = _rounded_preds
        _postprocessing["ordinal_rounding_applied"] = True
        print(f"Ordinal rounding applied: wf MAE {_ord_raw_wf_mae:.4f} → {_ord_round_wf_mae:.4f}")
    else:
        print(f"Ordinal rounding skipped: raw wf MAE {_ord_raw_wf_mae:.4f} <= rounded {_ord_round_wf_mae:.4f}")
# Any other subtype: silent no-op. _postprocessing stays initialized with False/None above.
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

# Build algorithm label from included families
_algo_parts = [k for k in ["lightgbm", "xgboost", "catboost", "ridge"] if k in all_val_preds]
_algo_label = "+".join(p.capitalize() for p in _algo_parts) + " ensemble"

results = {
    "algorithm": _algo_label,                          # e.g. "Lightgbm+Xgboost+Catboost ensemble"
    "problem_subtype": problem_subtype,                 # from profile.json (e.g. "panel_forecasting")
    "ensemble_path_used": ensemble_path_used,           # e.g. "full_regression_ensemble"
    "adaptive_choice": {
        "branch": adaptive_branch,                      # Axis 1 branch number (1–4 or fallback)
        "families_selected": families_selected,         # list of families Axis 1 chose
        "families_included_in_ensemble": list(all_val_preds.keys()),
        "reasoning": adaptive_reasoning,                # human-readable branch explanation
        "ensemble_weighting": ensemble_weighting,       # Axis 2 decision: "equal_median" or "ridge_weighted_1.5x"
        "weighting_reason": weighting_reason,           # explains max_ks and competence check
        "ridge_excluded_reason": ridge_excluded_reason, # null if included or never trained
        # Blend selection (Change 1) — always present
        "ensemble_blend": ensemble_blend,               # "equal_median", "inverse_mae_weighted", or "ridge_weighted_1.5x"
        "blend_weights": _blend_weights_log,            # per-family normalized weights used
        "blend_holdout_mae_equal": _blend_holdout_mae_equal,  # None when ridge_weighted_1.5x path taken
        "blend_holdout_mae_inv":   _blend_holdout_mae_inv,    # None when ridge_weighted_1.5x path taken
    },
    "families": {
        # All four family result dicts, populated during Steps 7a–7d.
        # LightGBM is always present; XGBoost/CatBoost/Ridge present when attempted.
        "lightgbm": {
            "best_params": final_hparams,
            "oof_mae": float(lgb_oof_mae),
            "training_time_seconds": lgb_training_time,
            "succeeded": True,
            "included_in_ensemble": "lgb" in all_val_preds,
            "n_estimators": best_n_estimators,
            "optuna_trials": optuna_trials,
        },
        "xgboost": xgb_result,       # dict built during Step 7b; absent if not attempted
        "catboost": _cb_result,      # dict built during Step 7c; attempted=False when skipped
        "ridge": ridge_result,       # dict built during Step 7d
    },
    "ensemble_oof_mae": float(ensemble_oof_mae),
    "n_families_in_ensemble": n_families_in_ensemble,
    "ensemble_disagreement": ensemble_disagreement,
    "feature_importance_top10": top10,
    "feature_importance_all": all_imp,
    "ridge_top_coefficients": ridge_top_coefficients,   # top-5 by abs coef; [] if Ridge excluded
    # LightGBM scalar fields retained for backward compat with validator/report_writer
    "objective": final_params["objective"],
    "best_params": final_hparams,
    "n_estimators": best_n_estimators,
    "n_seeds": 5,
    "cv_scheme": cv_scheme,
    "oof_mae": float(lgb_oof_mae),
    "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(m) for m in fold_maes],
    # walk_forward_mae: validator tries walk_forward_mae first, then oof_mae
    "walk_forward_mae": float(wf_mae if problem_type == "panel_forecasting" else oof_mae),
    "probe_mae_80_20": float(probe_mae_80_20),          # Optuna walk-forward probe MAE
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
    "n_train_rows": len(train_df),
    "n_val_rows": len(val_df),
    "retune_applied": _retune_applied,
    "optuna_reflection": _optuna_reflection,    # boundary reflection metadata; always present
    "postprocessing": _postprocessing,          # ordinal rounding metadata; always present
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
- `reports/oof_predictions.csv` — out-of-fold predictions on training set: identifier columns, `fold`, `predicted_target` (written by Step 7a; required by validator)
- `reports/model_results.json` — includes `feature_importance_all` (all features, not just top 10), `oof_mae`, `oof_cv_scheme`
- `reports/modeler_was_here.txt`

## What you do NOT do
- Do NOT write submission.csv (submission_writer does that)
- Do NOT generate report.pdf (report_writer does that)
- Do NOT modify reports/features.json or data/features_train.parquet
- Do NOT engineer new features inside this agent
- Do NOT read data/_truth/ directory
