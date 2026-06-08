# -*- coding: utf-8 -*-
"""
run_modeler.py — CatBoost predictor + Ridge diagnostic, true nested CV.

Reads pipeline_config.json to discover Expert features (deterministic, already
present in features_train.parquet) and Adaptive steps (impute_missing,
scale_features, target_encode) that must be fit on the training fold only.

Architecture
------------
- Outer CV: purged walk-forward (panel_forecasting) or KFold/GroupKFold
  (tabular/classification). Produces honest OOF predictions.
- Inner CV per outer fold: same scheme on the outer-train rows; drives Optuna
  hyperparameter search (8 trials per outer fold).
- Per-fold Pipeline: built fresh inside every fold via
  tools.adaptive_pipeline.build_pipeline() and fit on the fold's training rows
  only. This makes the impute, scale, and target_encode steps leak-proof.
- CatBoost is the sole predictor (target_encode columns are passed as
  cat_features, not Pipeline-encoded). Ridge runs as a linear diagnostic
  baseline (Pipeline-encoded target_encode columns + scaled numerics) and
  never enters predictions.csv.
- Recursive multi-step forecasting (panel only) uses the final 5-seed CatBoost
  ensemble + the production Pipeline fit on the full training set.
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
import os
import sys
import datetime
import argparse

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from adaptive_pipeline import build_pipeline, load_config  # noqa: E402


if __name__ == "__main__":
    # ─── argparse: --debug toggles reduced-compute knobs (default OFF) ──────
    parser = argparse.ArgumentParser(
        description="Modeler — CatBoost + Ridge diagnostic with nested CV."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Fast dev iteration: cap iterations at 200, 2 Optuna trials, "
             "1-seed ensemble, 60s tuning cap. OOF is NOT a valid score.",
    )
    _cli_args = parser.parse_args()
    DEBUG = _cli_args.debug

    if DEBUG:
        print("*** DEBUG MODE — reduced compute; OOF is NOT a valid score ***")

    # Compute knobs gated by --debug. Default OFF = current full behavior.
    DEFAULT_ITERS          = 200 if DEBUG else 400
    PROBE_ITERS            = 200 if DEBUG else 2000
    OPTUNA_N_TRIALS        = 1   if DEBUG else 8
    FINAL_RETRAIN_SEEDS    = [42] if DEBUG else [42, 7, 123, 2024, 999]
    OUTER_FOLD_FINAL_SEEDS = (42,) if DEBUG else (42, 7, 123)
    TUNING_BUDGET_SECONDS  = 60  if DEBUG else 25 * 60

    start_time = time.time()
    pipeline_start_time = start_time

    print("=" * 60)
    print("MODELER  —  CatBoost + Ridge diagnostic (nested CV + Pipeline)")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Read contract, profile, and features metadata
    # ─────────────────────────────────────────────────────────────────────────
    profile_path  = os.path.join(REPO_ROOT, "reports", "profile.json")
    features_path = os.path.join(REPO_ROOT, "reports", "features.json")
    config_path   = os.path.join(REPO_ROOT, "pipeline_config.json")
    cv_plan_path  = os.path.join(REPO_ROOT, "reports", "cv_plan.json")

    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    with open(features_path, encoding="utf-8") as f:
        feat_meta = json.load(f)
    with open(cv_plan_path, encoding="utf-8") as f:
        cv_plan = json.load(f)
    config = load_config(config_path)

    problem_type    = profile.get("problem_type", "panel_forecasting")
    problem_subtype = profile.get("problem_subtype", "panel_forecasting")

    target_col = feat_meta["target_col"]
    group_cols = feat_meta["group_cols"]
    time_col   = feat_meta["time_col"]

    adaptive_steps  = config.get("adaptive_steps", [])
    model_settings  = config.get("model_settings", {})
    cb_objective    = model_settings.get("objective", "MAE")

    # CatBoost loss_function from config.model_settings.objective. Accept either a
    # raw CatBoost loss name ("MAE", "RMSE", "Huber:delta=1.0") or "MAE"/"RMSE".
    _cb_loss        = cb_objective
    _cb_eval_metric = "MAE" if cb_objective.startswith("MAE") else cb_objective.split(":")[0]

    print(f"Problem type: {problem_type} / {problem_subtype}")
    print(f"Target: {target_col}   Groups: {group_cols}   Time: {time_col}")
    print(f"CatBoost loss: {_cb_loss}   eval_metric: {_cb_eval_metric}")
    print(f"Adaptive steps: {[s.get('name') for s in adaptive_steps]}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Load feature parquets
    # ─────────────────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(os.path.join(REPO_ROOT, "data", "features_train.parquet"))
    val_df   = pd.read_parquet(os.path.join(REPO_ROOT, "data", "features_val.parquet"))

    # Identifier columns to NEVER hand to the model.
    _exclude_ids = set([time_col] if time_col else []) | {target_col, "adversarial_weights"}

    # All feature columns the modeler can use, including group columns (CatBoost
    # treats them as cat_features; Ridge target-encodes them via the Pipeline).
    all_feature_cols = [c for c in train_df.columns if c not in _exclude_ids]

    # target_encode columns flow into CatBoost as cat_features directly
    _te_step = next((s for s in adaptive_steps if s.get("name") == "target_encode"), {})
    cat_feature_cols = [c for c in _te_step.get("targets", []) if c in all_feature_cols]

    # Defensive net (belt & suspenders for the upstream dtype+cardinality gate in
    # tools/feature_engineering.py): drop two classes of columns that must never
    # reach the model.
    #   (a) Non-numeric columns that are not declared as cat_features — CatBoost
    #       rejects raw strings unless listed as cat_features, and a raw text
    #       column would upcast the matrix to object dtype.
    #   (b) Numeric ID-LIKE columns that are not declared as cat_features — a
    #       hidden row_id / hash / timestamp would individually identify the row
    #       and act as a perfect (overfit) feature. Detected by UNIQUENESS RATIO
    #       only — n_unique == n_rows OR n_unique / n_rows > ID_RATIO_MAX — never
    #       by absolute count. Engineered numerics (lags, rolling stats, ratios)
    #       routinely have thousands of unique values and must NOT be flagged.
    #       This threshold is intentionally separate from CARD_MAX (the
    #       categorical ceiling used by the upstream gate) — they answer
    #       different questions and must not share a knob.
    ID_RATIO_MAX = 0.99

    _drop_nonnumeric = [
        c for c in all_feature_cols
        if not pd.api.types.is_numeric_dtype(train_df[c])
        and c not in cat_feature_cols
    ]
    _n_train_rows = len(train_df)
    _drop_idlike: list[tuple[str, int]] = []  # (name, n_unique)
    for c in all_feature_cols:
        if c in _drop_nonnumeric or c in cat_feature_cols:
            continue
        _nu = int(train_df[c].nunique(dropna=True))
        if _nu == _n_train_rows or (_n_train_rows > 0
                                    and _nu / _n_train_rows > ID_RATIO_MAX):
            _drop_idlike.append((c, _nu))
    if _drop_nonnumeric:
        print(f"Dropping non-numeric, non-cat_feature columns: {_drop_nonnumeric[:5]} "
              f"(total {len(_drop_nonnumeric)})")
        all_feature_cols = [c for c in all_feature_cols if c not in _drop_nonnumeric]
    if _drop_idlike:
        print(f"Dropping numeric id-like columns (uniqueness ratio > {ID_RATIO_MAX}, "
              f"n_rows={_n_train_rows}):")
        for _c, _nu in _drop_idlike:
            print(f"  - {_c}  n_unique={_nu}  ratio={_nu/_n_train_rows:.4f}")
        _drop_idlike_names = {c for c, _ in _drop_idlike}
        all_feature_cols = [c for c in all_feature_cols if c not in _drop_idlike_names]

    print(f"Train: {train_df.shape}, Val: {val_df.shape}")
    print(f"Features handed to model: {len(all_feature_cols)} "
          f"(cat_features: {len(cat_feature_cols)})")

    # Adversarial weights are aligned by row index into train_df.
    _av_info     = feat_meta.get("adversarial_validation", {})
    _adv_weights = None
    if _av_info.get("weights_applied", False) and "adversarial_weights" in train_df.columns:
        _adv_weights = train_df["adversarial_weights"].fillna(1.0).values
        print(f"Adversarial weights: min={_adv_weights.min():.3f} "
              f"max={_adv_weights.max():.3f} mean={_adv_weights.mean():.3f}")
    else:
        print("No adversarial weights — uniform sample weights")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Log1p target transform (skewed target heuristic)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from scipy.stats import skew as _skew
        _target_skew = float(_skew(train_df[target_col].dropna()))
    except Exception:
        _target_skew = 0.0
    _use_log1p = _target_skew > 1.5

    print(f"Target skewness: {_target_skew:.3f} → log1p transform: {_use_log1p}")


    def inv(preds):
        """Inverse-transform predictions from log1p space to original space."""
        if _use_log1p:
            return np.expm1(np.clip(preds, 0, None))
        return np.array(preds)


    y_raw  = train_df[target_col].values.astype(float)
    y_full = np.log1p(y_raw) if _use_log1p else y_raw
    n_train = len(train_df)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Critic retune check (preserved from prior architecture)
    # ─────────────────────────────────────────────────────────────────────────
    _retune_applied = None
    _retune_path = os.path.join(REPO_ROOT, "reports", "critic_retune_requested.json")
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
                with open(os.path.join(REPO_ROOT, "reports", "validator_review.json"),
                          encoding="utf-8") as _vf:
                    _vreview = json.load(_vf)
                _suspect = [s["feature"] for s in _vreview.get("feature_suspicion", [])
                            if isinstance(s, dict) and "feature" in s]
                if _suspect:
                    all_feature_cols = [c for c in all_feature_cols if c not in _suspect]
                    cat_feature_cols = [c for c in cat_feature_cols if c not in _suspect]
                    print(f"Removed {len(_suspect)} suspect features")
                    _retune_applied = (_retune_applied or "") + f"+removed_{len(_suspect)}_features"
            except Exception as _ve:
                print(f"Could not read feature_suspicion: {_ve}")

    _use_median_seed = _retune_applied is not None and "median_seed_aggregation" in _retune_applied
    _seed_agg = np.median if _use_median_seed else np.mean
    _expanded = _retune_applied is not None and "expanded_optuna_bounds" in _retune_applied
    _depth_hi = 10 if _expanded else 8
    _lr_lo    = 0.005 if _expanded else 0.02
    _lr_hi    = 0.15  if _expanded else 0.10

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Nested CV split builders
    # ─────────────────────────────────────────────────────────────────────────
    N_OUTER_FOLDS  = 3
    N_INNER_FOLDS  = 3
    EMBARGO_PERIODS = 2


    def _panel_purged_splits(df, time_col, n_folds, embargo):
        """Return list of (train_idx, val_idx) for purged walk-forward CV."""
        periods = sorted(df[time_col].unique())
        total = len(periods)
        min_train = max(1, total // (n_folds + 1))
        step = max(1, (total - min_train) // n_folds)
        splits = []
        for k in range(n_folds):
            vs = min_train + k * step
            ve = min(vs + step, total)
            if vs >= total:
                break
            val_periods = set(periods[vs:ve])
            train_end = max(0, vs - embargo)
            train_periods = set(periods[:train_end])
            tr_idx = df.index[df[time_col].isin(train_periods)]
            vl_idx = df.index[df[time_col].isin(val_periods)]
            if len(tr_idx) == 0 or len(vl_idx) == 0:
                continue
            splits.append((np.asarray(tr_idx), np.asarray(vl_idx)))
        return splits


    def _generic_splits(df, n_folds):
        """KFold or GroupKFold fallback for tabular/classification problems."""
        from sklearn.model_selection import KFold, GroupKFold
        if group_cols and group_cols[0] in df.columns:
            groups = df[group_cols[0]].values
            n_unique = len(np.unique(groups))
            actual = max(2, min(n_folds, n_unique))
            gkf = GroupKFold(n_splits=actual)
            return [(df.index[tr].to_numpy(), df.index[va].to_numpy())
                    for tr, va in gkf.split(df, groups=groups)]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        return [(df.index[tr].to_numpy(), df.index[va].to_numpy())
                for tr, va in kf.split(df)]


    def build_outer_splits(df):
        if problem_type == "panel_forecasting" and time_col:
            return _panel_purged_splits(df, time_col, N_OUTER_FOLDS, EMBARGO_PERIODS), "purged_walk_forward"
        return _generic_splits(df, N_OUTER_FOLDS), "kfold_or_groupkfold"


    def build_inner_splits(df):
        if problem_type == "panel_forecasting" and time_col:
            return _panel_purged_splits(df, time_col, N_INNER_FOLDS, EMBARGO_PERIODS)
        return _generic_splits(df, N_INNER_FOLDS)


    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 — Helpers: per-fold Pipeline + CatBoost training
    # ─────────────────────────────────────────────────────────────────────────
    import catboost as _cb_module
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import Ridge
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)


    def _fit_transform_fold(X_tr_df, y_tr, X_va_df, for_model):
        """Build a fresh Pipeline from config, fit on the training fold ONLY,
        and transform train + val rows."""
        bp = build_pipeline(adaptive_steps, for_model=for_model)
        # Sample weights aren't passed to Pipeline — only y for target encoder
        bp.pipeline.fit(X_tr_df, y_tr)
        X_tr_t = bp.pipeline.transform(X_tr_df)
        X_va_t = bp.pipeline.transform(X_va_df)
        return X_tr_t, X_va_t, bp


    def _train_catboost_fold(X_tr_df, y_tr, X_va_df, y_va_orig, hparams,
                              sample_weight, seeds=(42,)):
        """Train CatBoost (multi-seed) on a fold-fit Pipeline; return val preds
        in original target space and the trained models."""
        X_tr_t, X_va_t, bp = _fit_transform_fold(X_tr_df, y_tr, X_va_df,
                                                  for_model="catboost")
        # cat_features for CatBoost are the target_encode targets that survived
        cf_idx = [X_tr_t.columns.get_loc(c) for c in cat_feature_cols if c in X_tr_t.columns]
        seed_preds, models = [], []
        base_params = {
            "iterations":    hparams.get("iterations", DEFAULT_ITERS),
            "learning_rate": hparams.get("learning_rate", 0.05),
            "depth":         hparams.get("depth", 6),
            "l2_leaf_reg":   hparams.get("l2_leaf_reg", 3.0),
            "loss_function": _cb_loss,
            "eval_metric":   _cb_eval_metric,
            "verbose":       False,
            "allow_writing_files": False,
            "cat_features":  cf_idx,
        }
        for s in seeds:
            params = {**base_params, "random_seed": s}
            m = _cb_module.CatBoostRegressor(**params)
            m.fit(X_tr_t, y_tr, sample_weight=sample_weight, verbose=False)
            seed_preds.append(np.clip(inv(m.predict(X_va_t)), 0, None))
            models.append(m)
        preds = _seed_agg(seed_preds, axis=0)
        return preds, models, bp


    def _cb_objective_factory(X_outer_tr_df, y_outer_tr, sw_outer_tr,
                              inner_splits_local):
        """Return an Optuna objective closure that runs inner-CV on the
        outer-training rows."""
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", _lr_lo, _lr_hi, log=True),
                "depth":         trial.suggest_int("depth", 4, _depth_hi),
                "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "iterations":    DEFAULT_ITERS,
            }
            inner_maes = []
            for inner_tr_idx, inner_va_idx in inner_splits_local:
                # Indices are positional into X_outer_tr_df / y_outer_tr
                X_in_tr = X_outer_tr_df.iloc[inner_tr_idx]
                X_in_va = X_outer_tr_df.iloc[inner_va_idx]
                y_in_tr = y_outer_tr[inner_tr_idx]
                y_in_va_raw = y_outer_tr[inner_va_idx]
                y_in_va_orig = np.expm1(y_in_va_raw) if _use_log1p else y_in_va_raw
                sw_in = sw_outer_tr[inner_tr_idx] if sw_outer_tr is not None else None
                try:
                    preds, _, _ = _train_catboost_fold(
                        X_in_tr, y_in_tr, X_in_va, y_in_va_orig, params,
                        sample_weight=sw_in, seeds=(42,))
                    inner_maes.append(mean_absolute_error(y_in_va_orig, preds))
                except Exception as _e:
                    # Failed trial returns inf; Optuna skips it
                    return float("inf")
            return float(np.mean(inner_maes)) if inner_maes else float("inf")
        return objective


    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 — Outer CV with nested Optuna
    # ─────────────────────────────────────────────────────────────────────────
    from cv_engine import CVEngine
    assert train_df.index.equals(pd.RangeIndex(len(train_df))), \
        "CVEngine returns positional indices; train_df must be RangeIndex"
    engine = CVEngine(cv_plan, train_df, profile=profile)
    outer_splits   = engine.split()
    outer_scheme   = cv_plan["cv"]["cv_type"]
    N_OUTER_FOLDS  = len(outer_splits)
    print(f"Outer CV: {len(outer_splits)} folds ({outer_scheme})")

    oof_preds = np.full(len(train_df), np.nan)
    outer_fold_maes  = []
    outer_fold_best_params  = []
    outer_fold_sizes        = []
    total_optuna_trials     = 0

    TUNING_DEADLINE = start_time + TUNING_BUDGET_SECONDS  # cap on the nested-CV phase

    for of_i, (outer_tr_idx, outer_va_idx) in enumerate(outer_splits):
        # HEARTBEAT — printed at the START of every fold (debug AND full modes),
        # BEFORE Optuna search. Fires for budget-exceeded folds too — those still
        # train with defaults and score.
        print(f"Outer fold {of_i + 1}/{len(outer_splits)}: training "
              f"(n_train={len(outer_tr_idx)})...")

        # In DEBUG, bypass the deadline entirely (n_trials=1 + iterations=200 keeps
        # every fold cheap). In full mode, a hit deadline means SKIP-OPTUNA, NOT
        # skip-fold: the fold must still fit CatBoost with defaults and contribute
        # its MAE to the OOF estimate.
        elapsed = time.time() - start_time
        budget_exceeded = (not DEBUG) and (elapsed > TUNING_DEADLINE - start_time)

        X_outer_tr_df = train_df[all_feature_cols].iloc[outer_tr_idx]
        y_outer_tr    = y_full[outer_tr_idx]
        X_outer_va_df = train_df[all_feature_cols].iloc[outer_va_idx]
        y_outer_va_orig = y_raw[outer_va_idx]
        sw_outer_tr   = _adv_weights[outer_tr_idx] if _adv_weights is not None else None
        outer_fold_sizes.append(len(outer_tr_idx))

        if budget_exceeded:
            print(f"  Outer fold {of_i}: tuning budget exceeded — fitting CatBoost "
                  f"with default hparams (no Optuna), still scoring fold")
            best_params = {}
        else:
            # Inner splits use positional indices into the outer-train slice.
            # Re-derive on a fresh DataFrame whose RangeIndex matches outer_tr_idx order.
            _inner_df = train_df.iloc[outer_tr_idx].reset_index(drop=True)
            inner_splits = build_inner_splits(_inner_df)
            # Convert label-index splits to positional indices
            inner_splits_pos = [
                (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
                for tr, va in inner_splits
            ]

            objective = _cb_objective_factory(
                X_outer_tr_df.reset_index(drop=True),
                y_outer_tr, sw_outer_tr, inner_splits_pos,
            )
            study = optuna.create_study(direction="minimize")
            if DEBUG:
                _per_fold_timeout = None  # debug: no per-trial wall-clock pressure
            else:
                _per_fold_timeout = max(60, int((TUNING_DEADLINE - time.time())
                                                / max(1, len(outer_splits) - of_i)))
            study.optimize(objective, n_trials=OPTUNA_N_TRIALS,
                           timeout=_per_fold_timeout, catch=(Exception,))
            best_params = study.best_params if study.best_trial is not None else {}
            total_optuna_trials += len(study.trials)
            print(f"  Outer fold {of_i}: Optuna {len(study.trials)} trials, "
                  f"best inner MAE = {study.best_value:.4f}  params={best_params}")

        # Train final outer-fold model with best params (or defaults); predict on outer-val
        try:
            preds, _, _ = _train_catboost_fold(
                X_outer_tr_df, y_outer_tr, X_outer_va_df, y_outer_va_orig,
                best_params, sample_weight=sw_outer_tr, seeds=OUTER_FOLD_FINAL_SEEDS)
        except Exception as _e:
            print(f"  Outer fold {of_i}: final fit failed ({_e}); skipping fold")
            continue
        oof_preds[outer_va_idx] = preds
        fmae = float(mean_absolute_error(y_outer_va_orig, preds))
        outer_fold_maes.append(fmae)
        outer_fold_best_params.append(best_params)
        print(f"  Outer fold {of_i}: MAE = {fmae:.4f}")

    # Honest OOF MAE on rows covered by at least one outer fold
    _covered = ~np.isnan(oof_preds)
    if _covered.any():
        oof_mae = float(mean_absolute_error(y_raw[_covered], oof_preds[_covered]))
    else:
        oof_mae = float("inf")
    print(f"Nested-CV OOF MAE: {oof_mae:.4f}  (covered {_covered.sum()}/{len(train_df)} rows)")
    print(f"Per-fold MAEs: {[round(m, 4) for m in outer_fold_maes]}")

    # Aggregate best hyperparameters across outer folds (median per param)
    def _aggregate_params(list_of_dicts):
        if not list_of_dicts:
            return {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0}
        keys = set().union(*[d.keys() for d in list_of_dicts])
        agg = {}
        for k in keys:
            vals = [d[k] for d in list_of_dicts if k in d]
            agg[k] = (int(np.median(vals)) if isinstance(vals[0], int)
                      else float(np.median(vals)))
        return agg


    final_hparams = _aggregate_params(outer_fold_best_params)
    print(f"Final aggregated hyperparameters: {final_hparams}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8 — Walk-forward 80/20 holdout (for n_estimators probe + Ridge baseline)
    # ─────────────────────────────────────────────────────────────────────────
    if time_col and problem_type == "panel_forecasting":
        all_weeks   = sorted(train_df[time_col].unique())
        cutoff_idx  = int(len(all_weeks) * 0.8)
        cutoff_week = all_weeks[cutoff_idx]
        wf_train_mask = train_df[time_col] < cutoff_week
        wf_val_mask   = train_df[time_col] >= cutoff_week
    else:
        # Fallback: 80/20 random
        _perm = np.random.RandomState(42).permutation(n_train)
        _sp   = int(n_train * 0.8)
        wf_train_mask = pd.Series(False, index=train_df.index)
        wf_val_mask   = pd.Series(False, index=train_df.index)
        wf_train_mask.iloc[_perm[:_sp]] = True
        wf_val_mask.iloc[_perm[_sp:]]   = True

    wf_train = train_df[wf_train_mask].copy()
    wf_val   = train_df[wf_val_mask].copy()
    X_wf_train_df = wf_train[all_feature_cols]
    X_wf_val_df   = wf_val[all_feature_cols]
    y_wf_train_raw = wf_train[target_col].values.astype(float)
    y_wf_train     = np.log1p(y_wf_train_raw) if _use_log1p else y_wf_train_raw
    y_wf_val_raw   = wf_val[target_col].values.astype(float)
    y_wf_val       = y_wf_val_raw  # original space for MAE eval
    _wf_sw = _adv_weights[wf_train.index.values] if _adv_weights is not None else None

    # CatBoost probe → best_n_estimators via early stopping
    probe_params = {
        "iterations":    PROBE_ITERS,
        "loss_function": _cb_loss,
        "eval_metric":   _cb_eval_metric,
        "verbose":       False,
        "allow_writing_files": False,
        "random_seed":   42,
        "od_type":       "Iter",
        "od_wait":       100,
        "learning_rate": final_hparams.get("learning_rate", 0.05),
        "depth":         int(final_hparams.get("depth", 6)),
        "l2_leaf_reg":   final_hparams.get("l2_leaf_reg", 3.0),
    }
    _probe_bp_cat = build_pipeline(adaptive_steps, for_model="catboost")
    _probe_bp_cat.pipeline.fit(X_wf_train_df, y_wf_train)
    _X_wf_tr_t = _probe_bp_cat.pipeline.transform(X_wf_train_df)
    _X_wf_va_t = _probe_bp_cat.pipeline.transform(X_wf_val_df)
    _cf_probe_idx = [_X_wf_tr_t.columns.get_loc(c) for c in cat_feature_cols if c in _X_wf_tr_t.columns]
    probe = _cb_module.CatBoostRegressor(**{**probe_params, "cat_features": _cf_probe_idx})
    probe.fit(
        _X_wf_tr_t, y_wf_train, sample_weight=_wf_sw,
        eval_set=(_X_wf_va_t, np.log1p(y_wf_val_raw) if _use_log1p else y_wf_val_raw),
        verbose=False,
    )
    _best_iter        = probe.get_best_iteration() or 500
    best_n_estimators = int(_best_iter * 1.1)
    if DEBUG:
        best_n_estimators = min(best_n_estimators, 200)
    wf_mae            = float(mean_absolute_error(
        y_wf_val, np.clip(inv(probe.predict(_X_wf_va_t)), 0, None)))
    print(f"WF MAE (probe): {wf_mae:.4f}  best_iter: {_best_iter}  n_estimators: {best_n_estimators}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 9 — Final 5-seed CatBoost retrain on full train data + production Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    _cb_t0 = time.time()
    final_params = {
        "iterations":    best_n_estimators,
        "learning_rate": final_hparams.get("learning_rate", 0.05),
        "depth":         int(final_hparams.get("depth", 6)),
        "l2_leaf_reg":   final_hparams.get("l2_leaf_reg", 3.0),
        "loss_function": _cb_loss,
        "eval_metric":   _cb_eval_metric,
        "verbose":       False,
        "allow_writing_files": False,
    }

    production_bp_cat = build_pipeline(adaptive_steps, for_model="catboost")
    production_bp_cat.pipeline.fit(train_df[all_feature_cols], y_full)
    X_train_t = production_bp_cat.pipeline.transform(train_df[all_feature_cols])
    X_val_t   = production_bp_cat.pipeline.transform(val_df[all_feature_cols])
    _cf_prod_idx = [X_train_t.columns.get_loc(c) for c in cat_feature_cols if c in X_train_t.columns]

    _cb_trained_models = []
    _cb_seed_preds = []
    for seed in FINAL_RETRAIN_SEEDS:
        params_s = {**final_params, "cat_features": _cf_prod_idx, "random_seed": seed}
        m = _cb_module.CatBoostRegressor(**params_s)
        m.fit(X_train_t, y_full, sample_weight=_adv_weights, verbose=False)
        _cb_trained_models.append(m)
        _cb_seed_preds.append(np.clip(inv(m.predict(X_val_t)), 0, None))

    cb_ensemble_preds = _seed_agg(_cb_seed_preds, axis=0)
    cb_training_time  = int(time.time() - _cb_t0)
    last_model = _cb_trained_models[-1]
    print(f"CatBoost final: min={cb_ensemble_preds.min():.2f} max={cb_ensemble_preds.max():.2f} "
          f"mean={cb_ensemble_preds.mean():.2f}  training_time={cb_training_time}s")

    # Write OOF predictions (one row per outer-fold val row)
    _oof_ids = train_df.loc[_covered, group_cols + [time_col]].copy().reset_index(drop=True)
    _oof_ids["fold"]             = -1
    _oof_ids["predicted_target"] = oof_preds[_covered]
    _oof_path = os.path.join(REPO_ROOT, "reports", "oof_predictions.csv")
    _oof_ids.to_csv(_oof_path, index=False, encoding="utf-8")
    print(f"Written OOF predictions: {_oof_path}  shape={_oof_ids.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 10 — Ridge diagnostic (Pipeline-encoded, not in submission)
    # ─────────────────────────────────────────────────────────────────────────
    ridge_diagnostic = {
        "attempted":            False,
        "succeeded":            None,
        "oof_mae":              None,
        "best_alpha":           None,
        "top_coefficients":     [],
        "training_time_seconds": None,
        "role":                 "diagnostic_only",
        "skip_reason":          None,
    }
    ridge_top_coefficients = []

    _ridge_t0 = time.time()
    try:
        ridge_diagnostic["attempted"] = True

        # Alpha probe on the WF training portion using a Ridge-variant Pipeline
        _bp_rg_probe = build_pipeline(adaptive_steps, for_model="ridge")
        _bp_rg_probe.pipeline.fit(X_wf_train_df, y_wf_train)
        _Xwf_tr_rg = _bp_rg_probe.pipeline.transform(X_wf_train_df)
        _Xwf_va_rg = _bp_rg_probe.pipeline.transform(X_wf_val_df)

        _best_alpha     = 1.0
        _best_alpha_mae = float("inf")
        for _alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            _r = Ridge(alpha=_alpha)
            _r.fit(_Xwf_tr_rg, y_wf_train, sample_weight=_wf_sw)
            _mae = mean_absolute_error(
                y_wf_val, np.clip(inv(_r.predict(_Xwf_va_rg)), 0, None))
            if _mae < _best_alpha_mae:
                _best_alpha_mae = _mae
                _best_alpha     = _alpha
        print(f"Ridge alpha probe: best={_best_alpha}  MAE={_best_alpha_mae:.4f}")

        # Final Ridge diagnostic: refit on WF train with best alpha
        _r_final = Ridge(alpha=_best_alpha)
        _r_final.fit(_Xwf_tr_rg, y_wf_train, sample_weight=_wf_sw)
        _ridge_pred = np.clip(inv(_r_final.predict(_Xwf_va_rg)), 0, None)
        ridge_oof_mae = float(mean_absolute_error(y_wf_val, _ridge_pred))
        print(f"Ridge diagnostic WF OOF MAE: {ridge_oof_mae:.4f}")

        # Top-10 absolute coefficients — Ridge Pipeline outputs a numpy array, so
        # we reconstruct column names from the production Ridge pipeline.
        _bp_rg_named = build_pipeline(adaptive_steps, for_model="ridge")
        _bp_rg_named.pipeline.fit(X_wf_train_df, y_wf_train)
        # The last step (DropNonNumeric) stores the surviving column list
        _final_step = _bp_rg_named.pipeline.named_steps.get("to_numpy")
        _ridge_cols = list(_final_step.numeric_cols_) if _final_step is not None else []
        if len(_ridge_cols) == len(_r_final.coef_):
            ridge_top_coefficients = [
                {"feature": f, "abs_coef": float(c)}
                for f, c in sorted(zip(_ridge_cols, np.abs(_r_final.coef_)),
                                   key=lambda x: -x[1])[:10]
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
            "succeeded": False,
            "skip_reason": f"training_error: {_re}",
            "training_time_seconds": int(time.time() - _ridge_t0),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Step 11 — Final val predictions and ensemble metadata
    # ─────────────────────────────────────────────────────────────────────────
    ensemble_preds = np.clip(cb_ensemble_preds, 0, None)
    ensemble_blend = "single_catboost"
    _blend_weights_log = {"catboost": 1.0}
    _included_keys     = ["catboost"]

    _seed_stack = np.array(_cb_seed_preds)
    ensemble_disagreement = {
        "mean_disagreement":        float(np.mean(np.std(_seed_stack, axis=0))),
        "n_high_disagreement_rows": int(
            np.sum(np.std(_seed_stack, axis=0) > ensemble_preds.mean())
        ),
    }
    ensemble_oof_mae       = float(oof_mae)
    n_families_in_ensemble = 1

    # ─────────────────────────────────────────────────────────────────────────
    # Step 12 — Recursive multi-step forecasting (panel only)
    # ─────────────────────────────────────────────────────────────────────────
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
        _has_ord = (
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
            _rf_lag_cols   = [f"lag_{k}"       for k in _rf_lag_periods  if f"lag_{k}"       in all_feature_cols]
            _rf_rmean_cols = [f"roll_mean_{w}" for w in _rf_roll_windows if f"roll_mean_{w}" in all_feature_cols]
            _rf_rstd_cols  = [f"roll_std_{w}"  for w in _rf_roll_windows if f"roll_std_{w}"  in all_feature_cols]
            _rf_ceiling      = float(train_df[target_col].max()) * 10.0
            _rf_ceiling_hits = 0
            _fc_idx = {c: i for i, c in enumerate(all_feature_cols)}

            def _rf_predict(X_df):
                """Apply production Pipeline → 5-seed CatBoost mean/median."""
                X_t = production_bp_cat.pipeline.transform(X_df)
                sp = [np.clip(inv(m.predict(X_t)), 0, None) for m in _cb_trained_models]
                return np.clip(_seed_agg(sp, axis=0), 0, None)

            def _inject_target_feats(feat_arr, local_i, hist):
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

            def _build_gkeys(df):
                if len(group_cols) == 1:
                    return list(df[group_cols[0]])
                return [tuple(r) for r in df[group_cols].values.tolist()]

            _gb_cols = group_cols if len(group_cols) > 1 else group_cols[0]
            _wf_v_sorted = wf_val.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
            _wf_t_sorted = wf_train.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
            _wf_train_max_ord = int(_wf_t_sorted[_rf_ord_col].max())
            _wf_val_periods   = sorted(_wf_v_sorted[_rf_ord_col].unique().tolist())
            _n_h_steps        = len(_wf_val_periods)
            _wf_step_nums     = (_wf_v_sorted[_rf_ord_col].values - _wf_train_max_ord).astype(int)
            _y_wf_truth       = _wf_v_sorted[target_col].values

            _hist_seed = {}
            for _gk, _gdf in _wf_t_sorted.groupby(_gb_cols):
                _hist_seed[_gk] = list(_gdf[target_col].values)
            _wf_gkeys = _build_gkeys(_wf_v_sorted)

            # Build a fresh feature matrix that we can overwrite per recursive step
            _wf_base_df = _wf_v_sorted[all_feature_cols].copy().reset_index(drop=True)

            # (A) imputation holdout — use last known target as the lag seed
            print("Recursive forecasting: imputation holdout…")
            _last_known = {_gk: float(v[-1]) if v else 0.0 for _gk, v in _hist_seed.items()}
            _imp_df = _wf_base_df.copy()
            for col in _rf_lag_cols + _rf_rmean_cols:
                if col in _imp_df.columns:
                    _imp_df[col] = [_last_known.get(_gk, 0.0) for _gk in _wf_gkeys]
            for col in _rf_rstd_cols:
                if col in _imp_df.columns:
                    _imp_df[col] = 0.0
            _imp_preds_h = _rf_predict(_imp_df)
            _imp_hold_mae = float(mean_absolute_error(_y_wf_truth, _imp_preds_h))
            _per_step_imp = [
                float(mean_absolute_error(_y_wf_truth[_wf_step_nums == s],
                                           _imp_preds_h[_wf_step_nums == s]))
                for s in range(1, _n_h_steps + 1)
                if (_wf_step_nums == s).any()
            ]
            print(f"  Imputation holdout MAE: {_imp_hold_mae:.4f}")

            # (B) recursive holdout — predict step-by-step, feeding preds forward
            print("Recursive forecasting: recursive holdout…")
            _rec_preds_h = np.zeros(len(_wf_v_sorted))
            _rec_hist = {_gk: list(v) for _gk, v in _hist_seed.items()}
            _wf_base_arr = _wf_base_df.values.astype(object).copy()

            for _pord in _wf_val_periods:
                _pm = (_wf_v_sorted[_rf_ord_col] == _pord).values
                _pidxs = np.where(_pm)[0]
                _sf_arr = _wf_base_arr[_pidxs].copy()
                for _li, _ri in enumerate(_pidxs):
                    _inject_target_feats(_sf_arr, _li, _rec_hist.get(_wf_gkeys[_ri], []))
                _sf_df = pd.DataFrame(_sf_arr, columns=all_feature_cols)
                # Restore numeric dtype for non-cat columns
                for _col in all_feature_cols:
                    if _col not in cat_feature_cols:
                        _sf_df[_col] = pd.to_numeric(_sf_df[_col], errors="coerce")
                _sp = _rf_predict(_sf_df)
                _cf = _sp > _rf_ceiling
                if _cf.any():
                    _rf_ceiling_hits += int(_cf.sum())
                    _sp = np.clip(_sp, 0, _rf_ceiling)
                _rec_preds_h[_pidxs] = _sp
                for _li, _ri in enumerate(_pidxs):
                    _rec_hist.setdefault(_wf_gkeys[_ri], []).append(float(_sp[_li]))

            _rec_hold_mae = float(mean_absolute_error(_y_wf_truth, _rec_preds_h))
            _per_step_rec = [
                float(mean_absolute_error(_y_wf_truth[_wf_step_nums == s],
                                           _rec_preds_h[_wf_step_nums == s]))
                for s in range(1, _n_h_steps + 1)
                if (_wf_step_nums == s).any()
            ]
            print(f"  Recursive holdout MAE: {_rec_hold_mae:.4f}")

            _rec_wins = _rec_hold_mae <= _imp_hold_mae
            print(f"  Winner: {'RECURSIVE' if _rec_wins else 'IMPUTATION'}")

            _method_used  = "imputation"
            _n_val_steps  = int(val_df[_rf_ord_col].nunique())
            _val_ceil_hits = 0

            if _rec_wins:
                print("Recursive forecasting: generating recursive val predictions…")
                _val_s = val_df.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
                _val_orig_positions = val_df.sort_values(group_cols + [_rf_ord_col]).index.tolist()
                _val_periods = sorted(_val_s[_rf_ord_col].unique().tolist())
                _val_gkeys = _build_gkeys(_val_s)

                _train_s2 = train_df.sort_values(group_cols + [_rf_ord_col]).reset_index(drop=True)
                _full_hist = {}
                for _gk, _gdf in _train_s2.groupby(_gb_cols):
                    _full_hist[_gk] = list(_gdf[target_col].values)

                _val_base_df = _val_s[all_feature_cols].copy().reset_index(drop=True)
                _val_base_arr = _val_base_df.values.astype(object).copy()
                _rec_val_preds = np.zeros(len(_val_s))
                _val_hist = {_gk: list(v) for _gk, v in _full_hist.items()}

                for _pord in _val_periods:
                    _pm = (_val_s[_rf_ord_col] == _pord).values
                    _pidxs = np.where(_pm)[0]
                    _sf_arr = _val_base_arr[_pidxs].copy()
                    for _li, _ri in enumerate(_pidxs):
                        _inject_target_feats(_sf_arr, _li, _val_hist.get(_val_gkeys[_ri], []))
                    _sf_df = pd.DataFrame(_sf_arr, columns=all_feature_cols)
                    for _col in all_feature_cols:
                        if _col not in cat_feature_cols:
                            _sf_df[_col] = pd.to_numeric(_sf_df[_col], errors="coerce")
                    _sp = _rf_predict(_sf_df)
                    _cf = _sp > _rf_ceiling
                    if _cf.any():
                        _val_ceil_hits += int(_cf.sum())
                        _sp = np.clip(_sp, 0, _rf_ceiling)
                    _rec_val_preds[_pidxs] = _sp
                    for _li, _ri in enumerate(_pidxs):
                        _val_hist.setdefault(_val_gkeys[_ri], []).append(float(_sp[_li]))

                _aligned = np.zeros(len(val_df))
                for _new_pos, _orig_idx in enumerate(_val_orig_positions):
                    _aligned[_orig_idx] = _rec_val_preds[_new_pos]
                if np.isnan(_aligned).any() or (_aligned < 0).any():
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

    # ─────────────────────────────────────────────────────────────────────────
    # Step 13 — Write reports/predictions.csv
    # ─────────────────────────────────────────────────────────────────────────
    preds_df = val_df[group_cols + ([time_col] if time_col else [])].copy().reset_index(drop=True)
    preds_df.insert(0, "row_id", range(len(preds_df)))
    preds_df["predicted_target"] = ensemble_preds
    assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions found — abort"
    _preds_path = os.path.join(REPO_ROOT, "reports", "predictions.csv")
    preds_df.to_csv(_preds_path, index=False, encoding="utf-8")
    print(f"Written predictions.csv: {preds_df.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 14 — Write reports/model_results.json
    # ─────────────────────────────────────────────────────────────────────────
    _imp_arr = last_model.get_feature_importance()
    _imp_cols = list(X_train_t.columns) if hasattr(X_train_t, "columns") else all_feature_cols
    feat_imp = pd.Series(_imp_arr, index=_imp_cols).sort_values(ascending=False)
    top10   = [{"feature": k, "importance": float(v)} for k, v in feat_imp.head(10).items()]
    all_imp = [{"feature": k, "importance": float(v)} for k, v in feat_imp.items()]

    training_time = int(time.time() - start_time)

    results = {
        "algorithm":          "CatBoost (Ridge diagnostic)",
        "debug_mode":         DEBUG,
        "problem_subtype":    problem_subtype,
        "ensemble_path_used": "catboost_only",
        "log1p_transform":    _use_log1p,
        "target_skewness":    float(_target_skew),
        "pipeline_config_used": True,
        "nested_cv": {
            "outer_folds":          len(outer_splits),
            "outer_scheme":         outer_scheme,
            "inner_folds":          N_INNER_FOLDS,
            "outer_fold_maes":      [float(m) for m in outer_fold_maes],
            "outer_fold_train_sizes": [int(s) for s in outer_fold_sizes],
            "outer_fold_best_params": outer_fold_best_params,
            "honest_oof_mae":       float(oof_mae),
        },
        "adaptive_choice": {
            "branch":                        1,
            "families_selected":             ["catboost", "ridge_diagnostic"],
            "families_included_in_ensemble": _included_keys,
            "reasoning":                     "CatBoost sole predictor; Ridge diagnostic only",
            "ensemble_weighting":            "single_model",
            "weighting_reason":              "single CatBoost predictor",
            "ridge_excluded_reason":         "ridge_is_diagnostic_only",
            "ensemble_blend":                ensemble_blend,
            "blend_weights":                 _blend_weights_log,
            "blend_holdout_mae_equal":       None,
            "blend_holdout_mae_inv":         None,
        },
        "families": {
            "catboost": {
                "best_params":           final_hparams,
                "oof_mae":               float(oof_mae),
                "training_time_seconds": cb_training_time,
                "succeeded":             True,
                "included_in_ensemble":  True,
                "n_estimators":          best_n_estimators,
                "optuna_trials":         total_optuna_trials,
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
        "objective":                _cb_loss,
        "best_params":              final_hparams,
        "n_estimators":             best_n_estimators,
        "n_seeds":                  len(FINAL_RETRAIN_SEEDS),
        "cv_scheme":                outer_scheme,
        "oof_mae":                  float(oof_mae),
        "oof_cv_scheme":            outer_scheme,
        "per_fold_maes":            [float(m) for m in outer_fold_maes],
        "walk_forward_mae":         float(wf_mae),
        "probe_mae_80_20":          float(wf_mae),
        "training_time_seconds":    training_time,
        "optuna_trials_completed":  total_optuna_trials,
        "optuna_succeeded":         total_optuna_trials > 0,
        "val_prediction_stats": {
            "min":  float(ensemble_preds.min()),
            "max":  float(ensemble_preds.max()),
            "mean": float(ensemble_preds.mean()),
            "std":  float(ensemble_preds.std()),
        },
        "n_features":    len(all_feature_cols),
        "n_train_rows":  len(train_df),
        "n_val_rows":    len(val_df),
        "retune_applied":    _retune_applied,
        "optuna_reflection": {
            "pinned_params":   [],
            "recentered":      False,
            "best_mae_before": None,
            "best_mae_after":  None,
        },
        "postprocessing":    _postprocessing,
        "lag_forecasting":   _lag_forecasting,
    }

    _mr_path = os.path.join(REPO_ROOT, "reports", "model_results.json")
    with open(_mr_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Written model_results.json: {_mr_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 15 — Marker file
    # ─────────────────────────────────────────────────────────────────────────
    _marker_path = os.path.join(REPO_ROOT, "reports", "modeler_was_here.txt")
    with open(_marker_path, "w", encoding="utf-8") as f:
        f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    print(f"Written marker: {_marker_path}")
    print(f"Total training time: {training_time}s")
    print("=" * 60)
    print("MODELER COMPLETE")
    print("=" * 60)
