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
    parser.add_argument(
        "--transform",
        choices=["auto", "none", "log1p", "sqrt"],
        default="auto",
        help="Target transform. 'auto' (default) runs a data-driven A/B over "
             "{none, sqrt, log1p}, picking the one with the lowest held-out "
             "raw scored MAE on the WF probe split. 'none'/'log1p'/'sqrt' "
             "force the named transform and skip the A/B (manual override).",
    )
    parser.add_argument(
        "--group-relational",
        choices=["on", "off"],
        default="on",
        help="Adaptive group-relational feature family (3 features per "
             "qualifying high-cardinality + heavy-right-tail group column: "
             "group_target_rank, peer_group_mean, gap_to_top_groups). Fit per "
             "fold inside the modeler with inner-KFold OOF on train rows and "
             "full-fold lookup on val/recursion rows. 'off' (default) "
             "preserves the baseline; 'on' injects a 'group_relational' step "
             "into adaptive_steps at runtime. Detection rule is data-driven "
             "(n_unique >= 20 AND signed skew(group_means_y) >= 1.0); no-op "
             "if no column qualifies.",
    )
    parser.add_argument(
        "--tune-mode",
        choices=["per_fold", "once"],
        default="once",
        help="Hyperparameter tuning scope. 'once' (default): single Optuna study "
             "run BEFORE the outer loop on a full-train N_INNER_FOLDS-panel-purged "
             "inner CV; the picked params apply to every outer fold's final fit "
             "(per-fold weights still independent — only the hyperparameter SEARCH "
             "is shared, never weights or pipeline statistics). 'per_fold' (legacy "
             "opt-in): fresh Optuna study per outer fold on its outer-train inner "
             "CV (N_OUTER × OPTUNA_N_TRIALS × N_INNER_FOLDS inner CatBoost fits).",
    )
    _cli_args = parser.parse_args()
    DEBUG = _cli_args.debug
    TRANSFORM_REQUEST = _cli_args.transform
    TUNE_MODE = _cli_args.tune_mode
    GROUP_RELATIONAL = _cli_args.group_relational

    if DEBUG:
        print("*** DEBUG MODE — reduced compute; OOF is NOT a valid score ***")
    print(f"*** TUNE MODE: {TUNE_MODE} ***")
    print(f"*** GROUP RELATIONAL: {GROUP_RELATIONAL} ***")

    # Compute knobs gated by --debug. Default OFF = current full behavior.
    DEFAULT_ITERS          = 200 if DEBUG else 400
    PROBE_ITERS            = 200 if DEBUG else 2000
    OPTUNA_N_TRIALS        = 1   if DEBUG else 8
    FINAL_RETRAIN_SEEDS    = [42] if DEBUG else [42, 7, 123, 2024, 999]
    OUTER_FOLD_FINAL_SEEDS = (42,) if DEBUG else (42, 7, 123)
    TUNING_BUDGET_SECONDS  = 60  if DEBUG else 25 * 60

    # ── Level correction (post-hoc shift-aware bias adjustment) ──────────────
    MIN_GROUP_HOLDOUT         = 30    # min wf_val rows per group to use group-level estimate
    BIAS_CORRECTION_LAMBDA    = 0.5   # shrinkage weight: lambda*group + (1-lambda)*global
    BIAS_CORRECTION_REL_MARGIN = 0.015  # min weighted-MAE improvement as fraction of baseline
    BIAS_CORRECTION_SCALE     = 1.0   # fraction of estimated bias actually applied

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

    adaptive_steps  = list(config.get("adaptive_steps", []))
    model_settings  = config.get("model_settings", {})
    cb_objective    = model_settings.get("objective", "MAE")

    # CatBoost loss_function from config.model_settings.objective. Accept either a
    # raw CatBoost loss name ("MAE", "RMSE", "Huber:delta=1.0") or "MAE"/"RMSE".
    _cb_loss        = cb_objective
    _cb_eval_metric = "MAE" if cb_objective.startswith("MAE") else cb_objective.split(":")[0]

    # Runtime-inject the group_relational step when the flag is on. Candidates
    # are the panel group columns; the encoder's own data-driven detection rule
    # (cardinality + signed skew) decides which of them actually qualify and is
    # a no-op if none do. Keep pipeline_config.json baseline-clean.
    if GROUP_RELATIONAL == "on" and group_cols:
        adaptive_steps.append({
            "name":            "group_relational",
            "targets":         list(group_cols),
            "min_cardinality": 20,
            "skew_threshold":  1.0,
            "top_k":           3,
            "inner_folds":     5,
            "random_state":    42,
        })
        print(f"GroupRelational candidates: {list(group_cols)}")

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
    # Step 2.5 — Detect SCORED categories from the submission contract
    # ─────────────────────────────────────────────────────────────────────────
    # Generic, domain-agnostic: any column that appears in sample_submission.csv
    # AND in train_df, whose distinct submission values form a STRICT subset of
    # the distinct training values, is a scored-restricting column. If no such
    # column exists, the scored set = ALL training rows (no restriction).
    # Training still uses every row — only the EVALUATION/DECISION metric is
    # filtered to scored rows.
    def _detect_scored_filters(sample_sub_path, train_df_, target_col_,
                                time_col_, group_cols_):
        if not os.path.exists(sample_sub_path):
            return {}, None, None, (
                f"No sample_submission.csv at {sample_sub_path} — "
                "scored set = ALL training rows (no restriction)"
            )
        try:
            sample_sub = pd.read_csv(sample_sub_path)
        except Exception as e:
            return {}, None, None, (
                f"Failed to read sample_submission.csv ({e}) — "
                "scored set = ALL training rows"
            )
        exclude = {target_col_, "row_id", "id", "Id", "ID"}
        if time_col_:
            exclude.add(time_col_)
        # Probe group_cols first, then any remaining submission column.
        candidates = []
        for c in (group_cols_ or []):
            if (c in sample_sub.columns and c in train_df_.columns
                    and c not in exclude):
                candidates.append(c)
        for c in sample_sub.columns:
            if (c not in exclude and c in train_df_.columns
                    and c not in candidates):
                candidates.append(c)

        filters = {}
        for c in candidates:
            sub_vals = set(sample_sub[c].dropna().unique().tolist())
            tr_vals  = set(train_df_[c].dropna().unique().tolist())
            if sub_vals and sub_vals < tr_vals:   # proper subset
                filters[c] = sub_vals

        if not filters:
            return {}, None, None, (
                f"No submission column restricts categories vs training "
                f"(probed {candidates}) — scored set = ALL training rows"
            )
        # Choose one column for per-category breakdown (prefer the simple case).
        cat_col = next(iter(filters))
        cats    = sorted(filters[cat_col])
        if len(filters) == 1:
            msg = (f"Detected scored category column '{cat_col}' "
                   f"with {len(cats)} scored values: {cats}")
        else:
            msg = (f"Detected multiple scored-restricting columns: "
                   f"{ {k: sorted(v) for k, v in filters.items()} }; "
                   f"per-category breakdown uses '{cat_col}'")
        return filters, cats, cat_col, msg

    def _scored_mask(df, filters):
        n = len(df)
        if not filters:
            return np.ones(n, dtype=bool)
        mask = np.ones(n, dtype=bool)
        for col, vals in filters.items():
            mask &= df[col].isin(vals).values
        return mask

    def _mae_scored(y_true, y_pred, mask):
        """MAE on rows where mask is True; falls back to full MAE if mask empty."""
        if mask is None or not mask.any():
            return float(mean_absolute_error(y_true, y_pred))
        return float(mean_absolute_error(y_true[mask], y_pred[mask]))

    _sample_sub_path = os.path.join(REPO_ROOT, "data", "sample_submission.csv")
    scored_filters, scored_categories, scored_cat_col, _scored_log = \
        _detect_scored_filters(_sample_sub_path, train_df,
                               target_col, time_col, group_cols)
    print(_scored_log)
    train_scored_mask = _scored_mask(train_df, scored_filters)
    val_scored_mask   = _scored_mask(val_df,   scored_filters)
    print(f"  scored train rows: {int(train_scored_mask.sum())}/{len(train_df)}"
          f"  ({100*train_scored_mask.mean():.1f}%)")
    print(f"  scored val rows:   {int(val_scored_mask.sum())}/{len(val_df)}"
          f"  ({100*val_scored_mask.mean():.1f}%)")
    print(f"  Training still uses ALL {len(train_df)} rows — only "
          f"the evaluation/decision metric is filtered to scored rows.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Target transform selection
    #
    # 'auto' (default) runs a data-driven A/B over {none, sqrt, log1p} and
    # picks the winner by raw-space SCORED WF MAE: one CatBoost probe per
    # candidate on the existing WF probe split, predictions inverse-mapped
    # back to raw rate before scoring. Anything else is treated as a manual
    # override and the A/B is skipped. The skew is logged but no longer drives
    # the choice — optimising de-skewing is not the same as optimising the
    # scored metric.
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from scipy.stats import skew as _skew
        _target_skew = float(_skew(train_df[target_col].dropna()))
    except Exception:
        _target_skew = 0.0

    _TRANSFORM_CANDIDATES = ("none", "sqrt", "log1p")

    def _fwd_for_mode(mode, y):
        y = np.asarray(y, dtype=float)
        if mode == "log1p":
            return np.log1p(y)
        if mode == "sqrt":
            return np.sqrt(np.clip(y, 0, None))
        return y  # 'none' (identity)

    def _inv_for_mode(mode, p):
        p = np.asarray(p, dtype=float)
        if mode == "log1p":
            return np.expm1(np.clip(p, 0, None))
        if mode == "sqrt":
            return np.square(np.clip(p, 0, None))
        return np.clip(p, 0, None)

    # WF probe split — same convention as Step 8 (deterministic; both calls
    # produce identical masks).
    def _build_wf_split_for_ab():
        n_tr = len(train_df)
        if time_col and problem_type == "panel_forecasting":
            _aw = sorted(train_df[time_col].unique())
            _ci = int(len(_aw) * 0.8)
            _cw = _aw[_ci]
            tr_mask = (train_df[time_col] < _cw).values
            va_mask = (train_df[time_col] >= _cw).values
        else:
            _perm = np.random.RandomState(42).permutation(n_tr)
            _sp = int(n_tr * 0.8)
            tr_mask = np.zeros(n_tr, dtype=bool)
            va_mask = np.zeros(n_tr, dtype=bool)
            tr_mask[_perm[:_sp]] = True
            va_mask[_perm[_sp:]] = True
        return tr_mask, va_mask

    candidates_mae: dict = {}

    if TRANSFORM_REQUEST != "auto":
        _transform_mode = TRANSFORM_REQUEST
        print(f"Transform: manual override --transform={TRANSFORM_REQUEST}; "
              f"A/B skipped.")
    else:
        # One CatBoost probe per candidate on the WF probe split. Pipeline fit
        # is reused across candidates: only the target transform changes.
        import catboost as _cb_module_ab
        from sklearn.metrics import mean_absolute_error as _mae_ab

        _tr_mask_ab, _va_mask_ab = _build_wf_split_for_ab()
        _wf_tr_df_ab = train_df[_tr_mask_ab]
        _wf_va_df_ab = train_df[_va_mask_ab]
        X_tr_ab = _wf_tr_df_ab[all_feature_cols]
        X_va_ab = _wf_va_df_ab[all_feature_cols]
        y_tr_raw_ab = _wf_tr_df_ab[target_col].values.astype(float)
        y_va_raw_ab = _wf_va_df_ab[target_col].values.astype(float)
        _sw_ab = (_adv_weights[_wf_tr_df_ab.index.values]
                  if _adv_weights is not None else None)
        _va_scored_mask_ab = _scored_mask(_wf_va_df_ab, scored_filters)

        _ab_bp = build_pipeline(adaptive_steps, for_model="catboost")
        _ab_bp.pipeline.fit(X_tr_ab, y_tr_raw_ab)
        _X_tr_ab_t = _ab_bp.pipeline.transform(X_tr_ab)
        _X_va_ab_t = _ab_bp.pipeline.transform(X_va_ab)
        _cf_ab_idx = [_X_tr_ab_t.columns.get_loc(c) for c in cat_feature_cols
                      if c in _X_tr_ab_t.columns]

        _AB_PROBE_ITERS = 200 if DEBUG else min(PROBE_ITERS, 1000)
        _ab_probe_params = {
            "iterations":          _AB_PROBE_ITERS,
            "loss_function":       _cb_loss,
            "eval_metric":         _cb_eval_metric,
            "verbose":             False,
            "allow_writing_files": False,
            "random_seed":         42,
            "od_type":             "Iter",
            "od_wait":             100,
            "learning_rate":       0.05,
            "depth":               6,
            "l2_leaf_reg":         3.0,
            "cat_features":        _cf_ab_idx,
        }

        print(f"Target skewness: {_target_skew:.3f}   "
              f"--transform=auto   running A/B over {list(_TRANSFORM_CANDIDATES)}")
        for _cand in _TRANSFORM_CANDIDATES:
            _y_tr_ab = _fwd_for_mode(_cand, y_tr_raw_ab)
            _y_va_ab = _fwd_for_mode(_cand, y_va_raw_ab)
            try:
                _m = _cb_module_ab.CatBoostRegressor(**_ab_probe_params)
                _m.fit(_X_tr_ab_t, _y_tr_ab, sample_weight=_sw_ab,
                       eval_set=(_X_va_ab_t, _y_va_ab), verbose=False)
                _preds_raw = _inv_for_mode(_cand, _m.predict(_X_va_ab_t))
                if _va_scored_mask_ab.any():
                    _mae_raw = float(_mae_ab(
                        y_va_raw_ab[_va_scored_mask_ab],
                        _preds_raw[_va_scored_mask_ab],
                    ))
                else:
                    _mae_raw = float(_mae_ab(y_va_raw_ab, _preds_raw))
            except Exception as _e:
                print(f"  A/B candidate '{_cand}' failed: {_e}")
                _mae_raw = float("inf")
            candidates_mae[_cand] = _mae_raw
            print(f"  A/B '{_cand}': raw scored WF MAE = {_mae_raw:.4f}")

        _transform_mode = min(_TRANSFORM_CANDIDATES,
                              key=lambda c: candidates_mae[c])
        print(f"Transform A/B winner: '{_transform_mode}' "
              f"(candidates: {candidates_mae})")

    _use_log1p = (_transform_mode == "log1p")  # legacy alias for reporting

    def forward_transform(y):
        """Forward target transform — applied to training labels only."""
        return _fwd_for_mode(_transform_mode, y)

    def inverse_transform(p):
        """Inverse target transform — applied to model predictions to return
        them to the original target space. Always non-negative."""
        return _inv_for_mode(_transform_mode, p)

    # Legacy alias: many helpers reference `inv(...)` for prediction inversion.
    inv = inverse_transform

    print(f"Target skewness: {_target_skew:.3f}   "
          f"--transform={TRANSFORM_REQUEST}   resolved={_transform_mode}")

    # Selection log — surfaced via model_results.json so the report can
    # justify the choice ("we ran an A/B and sqrt won at 2.14 vs log1p 2.26").
    _transform_selection = {
        "candidates_mae":   {k: (None if (v is None or not np.isfinite(v))
                                 else float(v))
                             for k, v in candidates_mae.items()},
        "chosen":           _transform_mode,
        "selection_metric": "raw_scored_wf_mae",
        "skew":             float(_target_skew),
        "manual_override":  TRANSFORM_REQUEST != "auto",
        "requested":        TRANSFORM_REQUEST,
    }

    y_raw   = train_df[target_col].values.astype(float)
    y_full  = forward_transform(y_raw)
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
        in original target space, the per-seed prediction list (inverse/clipped
        target space, aligned to `seeds` slot order), and the trained models."""
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
        return preds, seed_preds, models, bp


    def _cb_objective_factory(X_outer_tr_df, y_outer_tr, sw_outer_tr,
                              inner_splits_local, outer_scored_mask):
        """Return an Optuna objective closure that runs inner-CV on the
        outer-training rows. Minimizes MAE restricted to SCORED rows
        (decision metric); falls back to all-row MAE if a fold has zero
        scored rows."""
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
                y_in_va_orig = inverse_transform(y_in_va_raw)
                sw_in = sw_outer_tr[inner_tr_idx] if sw_outer_tr is not None else None
                try:
                    preds, _, _, _ = _train_catboost_fold(
                        X_in_tr, y_in_tr, X_in_va, y_in_va_orig, params,
                        sample_weight=sw_in, seeds=(42,))
                    fold_scored = outer_scored_mask[inner_va_idx]
                    inner_maes.append(_mae_scored(y_in_va_orig, preds, fold_scored))
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
    outer_fold_maes         = []   # SCORED (decision metric)
    outer_fold_maes_all     = []   # all-category (diagnostic)
    outer_fold_per_cat_maes = []   # list[dict[cat -> mae|None]] per fold
    outer_fold_best_params  = []
    outer_fold_sizes        = []
    total_optuna_trials     = 0

    # OBSERVABILITY ONLY — per-seed OOF prediction accumulators. No model is
    # added, reordered, reseeded, or refit; we persist the per-seed arrays the
    # outer-fold scoring step ALREADY computes inside _train_catboost_fold.
    # No agent reads reports/oof/oof_per_seed.csv; aggregated OOF and every
    # downstream decision (ordinal offset, scored MAE, recursive-vs-imputation)
    # are unchanged.
    S_OUTER = len(OUTER_FOLD_FINAL_SEEDS)
    per_seed_oof        = np.full((S_OUTER, len(train_df)), np.nan)
    oof_fold_id         = np.full(len(train_df), -1, dtype=int)
    per_fold_seeds_used = []   # list[list[int]] aligned to outer_splits order

    TUNING_DEADLINE = start_time + TUNING_BUDGET_SECONDS  # cap on the nested-CV phase

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7a — TUNE-ONCE up-front Optuna (only when --tune-mode once)
    #
    # Runs ONE Optuna study on the FULL training data using N_INNER_FOLDS
    # panel-purged inner splits BEFORE the outer-fold loop. The resulting
    # tune_once_params is then applied to every outer fold's final CatBoost
    # fit; the per-fold inner-Optuna search is skipped. Folds remain
    # independent fits — only the hyperparameter SEARCH is shared, never
    # weights, pipeline statistics, or rows. The inner splits use the same
    # _panel_purged_splits machinery with EMBARGO_PERIODS=2 the outer scheme
    # uses, so each inner-val window is strictly later than its inner-train
    # window with a 2-period embargo — tuning never peeks across the boundary.
    # ─────────────────────────────────────────────────────────────────────────
    tune_once_params   = None
    tune_once_n_trials = 0
    if TUNE_MODE == "once":
        _tune_once_inner_splits = build_inner_splits(train_df)
        _tune_once_inner_splits_pos = [
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in _tune_once_inner_splits
        ]
        _tune_once_objective = _cb_objective_factory(
            train_df[all_feature_cols].reset_index(drop=True),
            y_full, _adv_weights, _tune_once_inner_splits_pos,
            train_scored_mask,
        )
        _tune_once_study   = optuna.create_study(direction="minimize")
        _tune_once_timeout = None if DEBUG else TUNING_BUDGET_SECONDS
        print(f"Tune-once: starting single Optuna study "
              f"(n_trials={OPTUNA_N_TRIALS}, timeout={_tune_once_timeout}s) on "
              f"full-train {len(_tune_once_inner_splits_pos)}-fold panel-purged "
              f"inner CV")
        _tune_once_study.optimize(
            _tune_once_objective, n_trials=OPTUNA_N_TRIALS,
            timeout=_tune_once_timeout, catch=(Exception,),
        )
        tune_once_params = (_tune_once_study.best_params
                            if _tune_once_study.best_trial is not None else {})
        tune_once_n_trials = len(_tune_once_study.trials)
        _tune_once_best_val = (_tune_once_study.best_value
                               if _tune_once_study.best_trial is not None
                               else float("inf"))
        print(f"Tune-once: {tune_once_n_trials} trials done, best inner MAE = "
              f"{_tune_once_best_val:.4f}  params={tune_once_params}")

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

        if TUNE_MODE == "once":
            # Single up-front tune produced tune_once_params; apply across folds.
            # Per-fold inner-Optuna is skipped; this fold's CatBoost still fits
            # only on its outer-train slice — only best_params (config) is shared.
            best_params = tune_once_params if tune_once_params is not None else {}
            print(f"  Outer fold {of_i}: tune-once mode — applying shared "
                  f"params {best_params} (no per-fold Optuna)")
        elif budget_exceeded:
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

            outer_tr_scored = train_scored_mask[outer_tr_idx]
            objective = _cb_objective_factory(
                X_outer_tr_df.reset_index(drop=True),
                y_outer_tr, sw_outer_tr, inner_splits_pos,
                outer_tr_scored,
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
            preds, seed_preds, _, _ = _train_catboost_fold(
                X_outer_tr_df, y_outer_tr, X_outer_va_df, y_outer_va_orig,
                best_params, sample_weight=sw_outer_tr, seeds=OUTER_FOLD_FINAL_SEEDS)
        except Exception as _e:
            print(f"  Outer fold {of_i}: final fit failed ({_e}); skipping fold")
            continue
        oof_preds[outer_va_idx] = preds
        # Persist per-seed predictions by slot. Slot k corresponds to
        # OUTER_FOLD_FINAL_SEEDS[k]; unfilled slots stay NaN (never fabricated).
        for _k in range(S_OUTER):
            if _k < len(seed_preds):
                per_seed_oof[_k, outer_va_idx] = seed_preds[_k]
        oof_fold_id[outer_va_idx] = of_i
        per_fold_seeds_used.append(
            [int(s) for s in OUTER_FOLD_FINAL_SEEDS[:len(seed_preds)]]
        )
        # All-category MAE (diagnostic) AND scored MAE (decision metric).
        fmae_all = float(mean_absolute_error(y_outer_va_orig, preds))
        va_scored_m = train_scored_mask[outer_va_idx]
        fmae_scored = _mae_scored(y_outer_va_orig, preds, va_scored_m)
        outer_fold_maes.append(fmae_scored)
        outer_fold_maes_all.append(fmae_all)
        outer_fold_best_params.append(best_params)
        # Per-scored-category fold MAE breakdown
        fold_per_cat = {}
        if scored_categories and scored_cat_col:
            cat_arr_va = train_df[scored_cat_col].values[outer_va_idx]
            for _cat in scored_categories:
                _cm = (cat_arr_va == _cat)
                fold_per_cat[_cat] = (
                    float(mean_absolute_error(y_outer_va_orig[_cm], preds[_cm]))
                    if _cm.any() else None
                )
        outer_fold_per_cat_maes.append(fold_per_cat)
        _cat_str = (
            "  per-cat={" + ", ".join(
                f"{k}={v:.4f}" if v is not None else f"{k}=NA"
                for k, v in fold_per_cat.items()) + "}"
        ) if fold_per_cat else ""
        print(f"  Outer fold {of_i}: MAE all={fmae_all:.4f}  "
              f"scored={fmae_scored:.4f}{_cat_str}")

    # Honest OOF MAE on rows covered by at least one outer fold.
    # `oof_mae` (the reported metric used by validator/critic/Optuna pinning) is
    # the SCORED-only MAE — the decision metric. `oof_mae_all_categories` is the
    # diagnostic across every training category.
    _covered = ~np.isnan(oof_preds)
    if _covered.any():
        oof_mae_all = float(mean_absolute_error(y_raw[_covered], oof_preds[_covered]))
        _scored_cov = _covered & train_scored_mask
        oof_mae_scored = _mae_scored(y_raw, oof_preds, _scored_cov)
    else:
        oof_mae_all    = float("inf")
        oof_mae_scored = float("inf")
    oof_mae = oof_mae_scored   # decision-metric alias (downstream contract)
    print(f"Nested-CV OOF MAE: all={oof_mae_all:.4f}  scored={oof_mae_scored:.4f}  "
          f"(covered {_covered.sum()}/{len(train_df)} rows)")
    print(f"Per-fold scored MAEs: {[round(m, 4) for m in outer_fold_maes]}")
    print(f"Per-fold all-cat MAEs: {[round(m, 4) for m in outer_fold_maes_all]}")

    # Per-scored-category OOF MAE breakdown (one entry per scored category)
    per_scored_cat_oof_mae = {}
    if scored_categories and scored_cat_col:
        _cat_arr = train_df[scored_cat_col].values
        for _cat in scored_categories:
            _cm = _covered & (_cat_arr == _cat)
            per_scored_cat_oof_mae[_cat] = (
                float(mean_absolute_error(y_raw[_cm], oof_preds[_cm]))
                if _cm.any() else None
            )
        _pc_str = ", ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=NA"
            for k, v in per_scored_cat_oof_mae.items()
        )
        print(f"Per-scored-category OOF MAE: {{{_pc_str}}}")

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


    # Under --tune-mode once, the per-fold loop ran zero inner Optuna studies;
    # the up-front study's trial count is the only Optuna work in this run.
    # Surface it in the same reporting field downstream consumers already read.
    if TUNE_MODE == "once":
        total_optuna_trials = tune_once_n_trials

    final_hparams = _aggregate_params(outer_fold_best_params)
    print(f"Final aggregated hyperparameters: {final_hparams}  (tune_mode={TUNE_MODE})")

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
    y_wf_train     = forward_transform(y_wf_train_raw)
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
        eval_set=(_X_wf_va_t, forward_transform(y_wf_val_raw)),
        verbose=False,
    )
    _best_iter        = probe.get_best_iteration() or 500
    best_n_estimators = int(_best_iter * 1.1)
    if DEBUG:
        best_n_estimators = min(best_n_estimators, 200)
    _wf_probe_preds = np.clip(inv(probe.predict(_X_wf_va_t)), 0, None)
    wf_mae          = float(mean_absolute_error(y_wf_val, _wf_probe_preds))
    _wf_val_scored_m = _scored_mask(wf_val, scored_filters)
    wf_mae_scored = _mae_scored(y_wf_val, _wf_probe_preds, _wf_val_scored_m)
    print(f"WF MAE (probe): all={wf_mae:.4f}  scored={wf_mae_scored:.4f}  "
          f"best_iter: {_best_iter}  n_estimators: {best_n_estimators}")

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
    # OBSERVABILITY ONLY — reports/oof/oof_per_seed.csv
    #
    # One row per OOF observation; columns expose the per-seed prediction
    # arrays already computed inside the outer-fold scoring step. The
    # row-wise aggregate of pred_seed_* (same callable as `_seed_agg`)
    # reproduces this run's aggregated OOF predictions to machine precision —
    # asserted below. No agent reads this file to make decisions.
    # ─────────────────────────────────────────────────────────────────────────
    _oof_dir = os.path.join(REPO_ROOT, "reports", "oof")
    os.makedirs(_oof_dir, exist_ok=True)
    _covered_pos = np.where(_covered)[0]
    _join_cols   = list(group_cols) + ([time_col] if time_col else [])
    _ops_df = train_df.loc[_covered, _join_cols].copy().reset_index(drop=True)
    _ops_df.insert(0, "row_index", _covered_pos)
    _ops_df["fold_id"] = oof_fold_id[_covered_pos]
    _ops_df["y_true"]  = y_raw[_covered_pos]
    for _k in range(S_OUTER):
        _ops_df[f"pred_seed_{_k}"] = per_seed_oof[_k, _covered_pos]
    _oof_seed_path = os.path.join(_oof_dir, "oof_per_seed.csv")
    _ops_df.to_csv(_oof_seed_path, index=False, encoding="utf-8")
    print(f"Written per-seed OOF: {_oof_seed_path}  shape={_ops_df.shape}  "
          f"seeds={list(OUTER_FOLD_FINAL_SEEDS)}  agg={_seed_agg.__name__}")

    # Verify: row-wise _seed_agg of per-seed arrays reproduces oof_preds.
    # Uses np.nanmean / np.nanmedian when only some seed slots are populated
    # (e.g. --debug → single seed), mirroring _seed_agg semantics on the
    # actual per-fold arrays.
    _pred_cols = [f"pred_seed_{_k}" for _k in range(S_OUTER)]
    _per_seed_arr = _ops_df[_pred_cols].values
    if _seed_agg is np.median:
        _rowwise = np.nanmedian(_per_seed_arr, axis=1)
    else:
        _rowwise = np.nanmean(_per_seed_arr, axis=1)
    _target = oof_preds[_covered_pos]
    _max_abs_diff = float(np.max(np.abs(_rowwise - _target)))
    print(f"oof_per_seed verify: n_rows={len(_ops_df)} (OOF rows={int(_covered.sum())})  "
          f"max |rowwise_{_seed_agg.__name__} - oof_preds| = {_max_abs_diff:.3e}")
    assert len(_ops_df) == int(_covered.sum()), (
        f"oof_per_seed row count {len(_ops_df)} != OOF row count {int(_covered.sum())}"
    )
    assert _max_abs_diff < 1e-9, (
        f"oof_per_seed verify failed: max |rowwise_{_seed_agg.__name__} - oof_preds| "
        f"= {_max_abs_diff:.3e} exceeds 1e-9 tolerance"
    )

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
        _rf_ord_col = f"{time_col}_ord"
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
            # Boolean mask of scored holdout rows (aligned to _wf_v_sorted)
            _wf_scored_m      = _scored_mask(_wf_v_sorted, scored_filters)

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
            _imp_hold_mae_all    = float(mean_absolute_error(_y_wf_truth, _imp_preds_h))
            _imp_hold_mae_scored = _mae_scored(_y_wf_truth, _imp_preds_h, _wf_scored_m)
            _imp_hold_mae        = _imp_hold_mae_scored   # decision metric
            _per_step_imp_all = [
                float(mean_absolute_error(_y_wf_truth[_wf_step_nums == s],
                                           _imp_preds_h[_wf_step_nums == s]))
                for s in range(1, _n_h_steps + 1)
                if (_wf_step_nums == s).any()
            ]
            _per_step_imp = [
                _mae_scored(_y_wf_truth, _imp_preds_h,
                            _wf_scored_m & (_wf_step_nums == s))
                for s in range(1, _n_h_steps + 1)
                if (_wf_step_nums == s).any()
            ]
            print(f"  Imputation holdout MAE: all={_imp_hold_mae_all:.4f}  "
                  f"scored={_imp_hold_mae_scored:.4f}")

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

            _rec_hold_mae_all    = float(mean_absolute_error(_y_wf_truth, _rec_preds_h))
            _rec_hold_mae_scored = _mae_scored(_y_wf_truth, _rec_preds_h, _wf_scored_m)
            _rec_hold_mae        = _rec_hold_mae_scored   # decision metric
            _per_step_rec_all = [
                float(mean_absolute_error(_y_wf_truth[_wf_step_nums == s],
                                           _rec_preds_h[_wf_step_nums == s]))
                for s in range(1, _n_h_steps + 1)
                if (_wf_step_nums == s).any()
            ]
            _per_step_rec = [
                _mae_scored(_y_wf_truth, _rec_preds_h,
                            _wf_scored_m & (_wf_step_nums == s))
                for s in range(1, _n_h_steps + 1)
                if (_wf_step_nums == s).any()
            ]
            print(f"  Recursive holdout MAE: all={_rec_hold_mae_all:.4f}  "
                  f"scored={_rec_hold_mae_scored:.4f}")

            # Decision uses SCORED-only MAE (the evaluation metric).
            _rec_wins = _rec_hold_mae_scored <= _imp_hold_mae_scored
            print(f"  Winner (by scored MAE): "
                  f"{'RECURSIVE' if _rec_wins else 'IMPUTATION'}")

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
                "decision_metric":         "scored_only_mae",
                "imputation_holdout_mae":              _imp_hold_mae_scored,
                "imputation_holdout_mae_all_categories": _imp_hold_mae_all,
                "recursive_holdout_mae":               _rec_hold_mae_scored,
                "recursive_holdout_mae_all_categories":  _rec_hold_mae_all,
                "per_step_mae_imputation":               _per_step_imp,
                "per_step_mae_imputation_all_categories": _per_step_imp_all,
                "per_step_mae_recursive":                _per_step_rec,
                "per_step_mae_recursive_all_categories":  _per_step_rec_all,
                "n_holdout_steps":         _n_h_steps,
                "n_val_steps":             _n_val_steps,
                "ceiling_hits_holdout":    _rf_ceiling_hits,
                "ceiling_hits_val":        _val_ceil_hits,
                "ceiling_value":           _rf_ceiling,
                "notes": (
                    f"recursive won by {_imp_hold_mae_scored - _rec_hold_mae_scored:.4f} "
                    f"(scored MAE)"
                    if _rec_wins
                    else f"imputation won by {_rec_hold_mae_scored - _imp_hold_mae_scored:.4f} "
                         f"(scored MAE)"
                ),
            }

    except Exception as _rf_exc:
        import traceback as _tb
        print(f"Recursive forecasting failed ({_rf_exc}) — using imputation path")
        _tb.print_exc()
        _lag_forecasting["notes"] = f"error: {str(_rf_exc)[:300]}"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 12.5 — Post-hoc level correction (shift-aware, gated)
    # Estimates systematic prediction bias using wf_val (true labels available)
    # weighted by adversarial_weights (P(val-like)), applies a shrunk correction
    # to ensemble_preds ONLY if weighted holdout MAE improves by > REL_MARGIN.
    # Residuals come from _wf_probe_preds (out-of-sample: probe trained on
    # wf_train only) to avoid the in-sample collapse from the production model.
    # ─────────────────────────────────────────────────────────────────────────
    _level_correction = {
        "applied":                False,
        "bias_hat_global":        None,
        "bias_hat_per_group":     {},
        "lambda":                 BIAS_CORRECTION_LAMBDA,
        "scale":                  BIAS_CORRECTION_SCALE,
        "min_group_holdout":      MIN_GROUP_HOLDOUT,
        "rel_margin":             BIAS_CORRECTION_REL_MARGIN,
        "wf_weighted_mae_before": None,
        "wf_weighted_mae_after":  None,
        "used_per_group":         False,
        "notes":                  "not_attempted",
    }
    try:
        # Out-of-sample residuals: probe trained on wf_train only.
        _wf_resid = _wf_probe_preds - y_wf_val

        # Adversarial weights: pull directly from wf_val column (safe regardless
        # of whether train_df carries a clean RangeIndex).
        _wf_adv_w = (
            wf_val["adversarial_weights"].fillna(1.0).values
            if "adversarial_weights" in wf_val.columns
            else np.ones(len(wf_val))
        )

        # Global weighted bias estimate.
        _bias_global = float(np.average(_wf_resid, weights=_wf_adv_w))

        # Per-group shrinkage: group estimate when group has >= MIN_GROUP_HOLDOUT rows.
        _wf_vr        = wf_val.reset_index(drop=True)
        _val_vr       = val_df.reset_index(drop=True)
        _bias_row_wf  = np.full(len(_wf_vr), _bias_global)
        _bias_row_val = np.full(len(_val_vr), _bias_global)
        _used_per_group = False
        _bias_per_group = {}

        if group_cols:
            for _grp, _gidx_idx in _wf_vr.groupby(group_cols).groups.items():
                _gidx    = np.array(list(_gidx_idx))
                _grp_key = str(_grp)
                if len(_gidx) >= MIN_GROUP_HOLDOUT:
                    _g_raw    = float(np.average(_wf_resid[_gidx], weights=_wf_adv_w[_gidx]))
                    _g_shrunk = (BIAS_CORRECTION_LAMBDA * _g_raw
                                 + (1 - BIAS_CORRECTION_LAMBDA) * _bias_global)
                    _bias_row_wf[_gidx] = _g_shrunk
                    _bias_per_group[_grp_key] = {
                        "raw": _g_raw, "shrunk": _g_shrunk, "n": len(_gidx),
                    }
                    _used_per_group = True
                else:
                    _bias_per_group[_grp_key] = {
                        "raw": None, "shrunk": _bias_global,
                        "n": len(_gidx), "fallback": "global",
                    }
            # Apply the same group map to val_df rows (no labels needed).
            for _grp, _gidx_idx in _val_vr.groupby(group_cols).groups.items():
                _gidx    = np.array(list(_gidx_idx))
                _grp_key = str(_grp)
                if (_grp_key in _bias_per_group
                        and _bias_per_group[_grp_key].get("raw") is not None):
                    _bias_row_val[_gidx] = _bias_per_group[_grp_key]["shrunk"]

        # Gate: weighted MAE before and after. Gate and application use identical
        # BIAS_CORRECTION_SCALE so the holdout result predicts the actual effect.
        _wf_wmae_before = float(np.average(np.abs(_wf_resid), weights=_wf_adv_w))
        _corrected_wf   = np.clip(
            _wf_probe_preds - BIAS_CORRECTION_SCALE * _bias_row_wf, 0, None
        )
        _wf_wmae_after  = float(
            np.average(np.abs(_corrected_wf - y_wf_val), weights=_wf_adv_w)
        )
        _improvement    = _wf_wmae_before - _wf_wmae_after
        _rel_margin_abs = BIAS_CORRECTION_REL_MARGIN * _wf_wmae_before
        _apply_lc       = _improvement > _rel_margin_abs

        if _apply_lc:
            ensemble_preds = np.clip(
                ensemble_preds - BIAS_CORRECTION_SCALE * _bias_row_val, 0, None
            )
            print(
                f"Level correction APPLIED: global_bias={_bias_global:.4f}  "
                f"wf_wmae {_wf_wmae_before:.4f} → {_wf_wmae_after:.4f}  "
                f"improvement={_improvement:.4f} (rel_margin={_rel_margin_abs:.4f})"
            )
        else:
            print(
                f"Level correction SKIPPED: improvement={_improvement:.4f} "
                f"<= rel_margin={_rel_margin_abs:.4f} "
                f"({BIAS_CORRECTION_REL_MARGIN:.1%} of {_wf_wmae_before:.4f})"
            )

        _level_correction.update({
            "applied":                _apply_lc,
            "bias_hat_global":        _bias_global,
            "bias_hat_per_group":     _bias_per_group,
            "used_per_group":         _used_per_group,
            "wf_weighted_mae_before": _wf_wmae_before,
            "wf_weighted_mae_after":  _wf_wmae_after,
            "notes": (
                f"improvement={_improvement:.4f}  "
                f"rel_margin={_rel_margin_abs:.4f}  applied={_apply_lc}"
            ),
        })
    except Exception as _lc_exc:
        import traceback as _lc_tb
        print(f"Level correction failed ({_lc_exc}) — skipping")
        _lc_tb.print_exc()
        _level_correction["notes"] = f"error: {str(_lc_exc)[:200]}"

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
        "target_transform":   _transform_mode,
        "target_transform_requested": TRANSFORM_REQUEST,
        "target_skewness":    float(_target_skew),
        "transform_selection": _transform_selection,
        "pipeline_config_used": True,
        "nested_cv": {
            "outer_folds":          len(outer_splits),
            "outer_scheme":         outer_scheme,
            "inner_folds":          N_INNER_FOLDS,
            "outer_fold_maes":      [float(m) for m in outer_fold_maes],
            "outer_fold_maes_all_categories": [float(m) for m in outer_fold_maes_all],
            "outer_fold_per_scored_category_maes": outer_fold_per_cat_maes,
            "outer_fold_train_sizes": [int(s) for s in outer_fold_sizes],
            "outer_fold_best_params": outer_fold_best_params,
            "honest_oof_mae":       float(oof_mae),
            "honest_oof_mae_all_categories": float(oof_mae_all),
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
        "oof_mae_all_categories":   float(oof_mae_all),
        "oof_cv_scheme":            outer_scheme,
        "per_fold_maes":            [float(m) for m in outer_fold_maes],
        "per_fold_maes_all_categories": [float(m) for m in outer_fold_maes_all],
        "per_scored_category_oof_mae":  per_scored_cat_oof_mae,
        "per_fold_per_scored_category_maes": outer_fold_per_cat_maes,
        "scored_categories":        scored_categories,
        "scored_category_column":   scored_cat_col,
        "scored_filters":           {k: sorted(v) for k, v in scored_filters.items()},
        "n_scored_train_rows":      int(train_scored_mask.sum()),
        "n_scored_val_rows":        int(val_scored_mask.sum()),
        "decision_metric":          "scored_only_mae" if scored_filters else "all_category_mae",
        "walk_forward_mae":         float(wf_mae),
        "walk_forward_mae_scored":  float(wf_mae_scored),
        "probe_mae_80_20":          float(wf_mae),
        "probe_mae_80_20_scored":   float(wf_mae_scored),
        "training_time_seconds":    training_time,
        "optuna_trials_completed":  total_optuna_trials,
        "optuna_succeeded":         total_optuna_trials > 0,
        "tune_mode":                TUNE_MODE,
        "tune_once_params":         tune_once_params,
        "tune_once_n_trials":       int(tune_once_n_trials),
        "group_relational": {
            "mode":            GROUP_RELATIONAL,
            "qualifying_cols": (
                list(production_bp_cat.pipeline.named_steps["group_relational"].qualifying_cols_)
                if (GROUP_RELATIONAL == "on"
                    and "group_relational" in production_bp_cat.pipeline.named_steps)
                else []
            ),
        },
        "oof_scoring_note":         (
            "OOF scored with shared fixed params from a single up-front "
            "Optuna study run on full-train N_INNER_FOLDS-panel-purged "
            "inner CV; each outer fold's CatBoost still fits only on its "
            "outer-train slice (per-fold weights independent — only "
            "hyperparameters shared)."
            if TUNE_MODE == "once" else
            "OOF scored with per-fold-tuned params (one Optuna study per "
            "outer fold on its outer-train inner CV); per-fold weights and "
            "hyperparameters both independent."
        ),
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
        "level_correction":  _level_correction,
        # OBSERVABILITY ONLY — pointer to per-seed OOF prediction arrays.
        # Not consumed by validator/critic/submission/report — diagnostic only.
        "oof_per_seed": {
            "seeds":           [int(s) for s in OUTER_FOLD_FINAL_SEEDS],
            "n_seeds":         int(S_OUTER),
            "n_rows":          int(_covered.sum()),
            "per_fold_seeds":  per_fold_seeds_used,
            "agg":             _seed_agg.__name__,
            "path":            "reports/oof/oof_per_seed.csv",
            "debug":           bool(DEBUG),
        },
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
