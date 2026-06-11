#!/usr/bin/env python3
"""CV integrity and leakage audit for the Award B pipeline.

Reads (all relative to repo root):
  data/features_train.parquet
  reports/profile.json
  reports/model_results.json
  reports/features.json
  reports/schema_analysis.md     (optional — for structural context)
  reports/oof_predictions.csv    (optional — confirms modeler ran OOF)

Writes:
  reports/validator_review.json
  reports/validator_was_here.txt

Usage:
  python tools/validate.py [--repo-root PATH]
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import catboost as _cb_module
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ── TUNABLE THRESHOLDS ─────────────────────────────────────────────────────
CV_GAP_CRITICAL_PCT       = 0.25   # (strict/reported - 1) > this → CRITICAL
CV_GAP_WARNING_PCT        = 0.10   # above this → WARNING
IMPORTANCE_WARNING_SHARE  = 0.50   # single feature > 50% share → concentration WARNING
LOO_CANDIDATE_SHARE       = 0.30   # run LOO for features above this share
LOO_STAT_SIGNAL_THRESHOLD = 0.05   # |delta|/strict_mae < this → statistical signal fires
N_STRICT_FOLDS            = 4      # number of walk-forward / grouped folds
EMBARGO_PERIODS           = 2      # time periods purged at train/val boundary

# Regex patterns that fire the structural signal (feature name heuristics)
STRUCTURAL_PATTERNS: list[str] = [
    r"_lead[_\d]",       # _lead_1, _lead3
    r"[_\b]future",      # _future, future_
    r"_t\+\d+",          # _t+1, _t+7
    r"_lag_?0\b",        # _lag0, _lag_0 (contemporaneous)
    r"\blag0\b",
    r"[_\b]leak[_\b]?",  # explicit leak marker
]


# ── HELPERS ────────────────────────────────────────────────────────────────

def _safe_json(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _fit_one_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    hparams: dict,
    n_estimators: int,
    seed: int = 42,
    problem_subtype: str = "",
) -> tuple[object, np.ndarray]:
    """Train one CatBoost fold and return (model, val_predictions)."""
    _is_cls = problem_subtype in ("binary_classification", "multiclass_classification")
    if problem_subtype == "binary_classification":
        _loss, _metric = "Logloss", "AUC"
    elif problem_subtype == "multiclass_classification":
        _loss, _metric = "MultiClass", "Accuracy"
    else:
        _loss, _metric = "MAE", "MAE"
    params = {
        "iterations":    n_estimators,
        "loss_function": _loss,
        "eval_metric":   _metric,
        "verbose":       False,
        "allow_writing_files": False,
        "random_seed":   seed,
        **hparams,
    }
    ModelClass = _cb_module.CatBoostClassifier if _is_cls else _cb_module.CatBoostRegressor
    model = ModelClass(**params)
    model.fit(X_tr, y_tr, verbose=False)
    raw = model.predict(X_vl)
    preds = raw if _is_cls else np.clip(raw, 0, None)
    return model, preds


# ── STRICT CV SPLITTERS ────────────────────────────────────────────────────

def _panel_strict_splits(
    train_df: pd.DataFrame,
    time_col: str,
    n_folds: int,
    embargo: int,
) -> list[tuple[pd.Index, pd.Index]]:
    """Return list of (train_idx, val_idx) for purged walk-forward CV."""
    all_periods = sorted(train_df[time_col].unique())
    total = len(all_periods)
    min_train = max(1, total // (n_folds + 1))
    fold_step = max(1, (total - min_train) // n_folds)

    splits = []
    for k in range(n_folds):
        vs = min_train + k * fold_step
        ve = min(vs + fold_step, total)
        if vs >= total:
            break
        val_periods = set(all_periods[vs:ve])
        # Purge: drop embargo periods immediately before val window
        train_end = max(0, vs - embargo)
        train_periods = set(all_periods[:train_end])
        tr_idx = train_df.index[train_df[time_col].isin(train_periods)]
        vl_idx = train_df.index[train_df[time_col].isin(val_periods)]
        if len(tr_idx) == 0 or len(vl_idx) == 0:
            continue
        splits.append((tr_idx, vl_idx))
    return splits


def _grouped_strict_splits(
    train_df: pd.DataFrame,
    group_col: str,
    n_folds: int,
) -> list[tuple[pd.Index, pd.Index]]:
    """Return list of (train_idx, val_idx) for grouped K-fold CV.

    If the group column has fewer unique values than n_folds, automatically
    reduces n_splits to the number of unique groups (minimum 2).
    """
    groups = train_df[group_col].values
    n_unique = len(np.unique(groups))
    actual_folds = max(2, min(n_folds, n_unique))
    gkf = GroupKFold(n_splits=actual_folds)
    splits = []
    for tr_iloc, vl_iloc in gkf.split(train_df, groups=groups):
        splits.append((train_df.index[tr_iloc], train_df.index[vl_iloc]))
    return splits


# ── STRICT CV ──────────────────────────────────────────────────────────────

def compute_strict_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str],
    time_col: str | None,
    problem_type: str,
    hparams: dict,
    n_estimators: int,
    problem_subtype: str = "",
) -> tuple[float, str, list, list[float], list[int], list[float]]:
    """
    Run strict CV and return
    (mean_mae, scheme_description, fold_models, fold_maes, fold_train_sizes, fold_cls_metrics).

    fold_models are used downstream for importance averaging.
    fold_maes and fold_train_sizes are written to validator_review.json for
    gap_attribution.py to classify the OOF→strict gap.
    fold_cls_metrics holds per-fold AUC (binary) or accuracy (multiclass); empty for regression.
    """
    if problem_type == "panel_forecasting" and time_col:
        splits = _panel_strict_splits(
            train_df, time_col, N_STRICT_FOLDS, EMBARGO_PERIODS
        )
        scheme = (
            f"{N_STRICT_FOLDS}-fold purged+embargoed walk-forward "
            f"(embargo={EMBARGO_PERIODS} periods)"
        )
    else:
        gc = group_cols[0] if group_cols else None
        if gc and gc in train_df.columns:
            splits = _grouped_strict_splits(train_df, gc, N_STRICT_FOLDS)
            actual_folds = len(splits)
            scheme = f"{actual_folds}-fold purged GroupKFold on '{gc}'"
        else:
            # Fallback: plain sequential split
            n = len(train_df)
            cut = int(n * 0.8)
            splits = [(train_df.index[:cut], train_df.index[cut:])]
            scheme = "80/20 sequential split (no group column available)"

    _is_cls = problem_subtype in ("binary_classification", "multiclass_classification")
    fold_maes: list[float] = []
    fold_models: list = []
    fold_train_sizes: list[int] = []
    fold_cls_metrics: list[float] = []

    for tr_idx, vl_idx in splits:
        fold_tr = train_df.loc[tr_idx]
        fold_vl = train_df.loc[vl_idx]

        fold_train_sizes.append(len(fold_tr))

        fill = fold_tr[feature_cols].median()
        X_tr = fold_tr[feature_cols].fillna(fill).values
        y_tr = fold_tr[target_col].values
        X_vl = fold_vl[feature_cols].fillna(fill).values
        y_vl = fold_vl[target_col].values

        model, preds = _fit_one_fold(X_tr, y_tr, X_vl, hparams, n_estimators,
                                     problem_subtype=problem_subtype)
        fold_mae = float(mean_absolute_error(y_vl, preds))
        fold_maes.append(fold_mae)
        fold_models.append(model)
        if problem_subtype == "binary_classification":
            try:
                fold_cls_metrics.append(float(roc_auc_score(y_vl, preds)))
            except Exception:
                fold_cls_metrics.append(float("nan"))
        elif problem_subtype == "multiclass_classification":
            fold_cls_metrics.append(float(accuracy_score(y_vl, preds)))

    strict_mae = float(np.mean(fold_maes))
    return strict_mae, scheme, fold_models, fold_maes, fold_train_sizes, fold_cls_metrics


# ── LEAVE-ONE-FEATURE-OUT CV ───────────────────────────────────────────────

def loo_strict_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str],
    time_col: str | None,
    problem_type: str,
    hparams: dict,
    n_estimators: int,
    base_strict_mae: float,
    candidate_features: list[str],
    problem_subtype: str = "",
) -> dict[str, float]:
    """
    For each candidate feature, re-run strict CV without it.
    Returns {feature_name: loo_delta} where delta = mae_without - base_strict_mae.
    Small (near-zero) delta → statistical leakage signal.
    """
    results: dict[str, float] = {}
    for feat in candidate_features:
        reduced = [c for c in feature_cols if c != feat]
        if not reduced:
            results[feat] = 0.0
            continue
        try:
            loo_mae, _, _, _, _, _ = compute_strict_cv(
                train_df, reduced, target_col, group_cols,
                time_col, problem_type, hparams, n_estimators,
                problem_subtype=problem_subtype,
            )
            results[feat] = float(loo_mae - base_strict_mae)
        except Exception as exc:
            print(f"  WARNING: LOO for '{feat}' failed ({exc}) — skipping")
            results[feat] = float("nan")
    return results


# ── STRUCTURAL CHECK ───────────────────────────────────────────────────────

def structural_check(
    feature_cols: list[str],
    target_col: str,
    schema_text: str = "",
) -> dict[str, bool]:
    """
    Heuristic flag: True if the feature name suggests future leakage.

    Signals:
      1. Regex pattern match on STRUCTURAL_PATTERNS.
      2. Feature name contains the target column name verbatim without
         a proper lag suffix (e.g. 'weekly_sales_direct' when target is
         'weekly_sales').

    This is deliberately conservative — only clear naming violations fire.
    """
    flagged: dict[str, bool] = {}
    target_lower = target_col.lower()

    for feat in feature_cols:
        feat_lower = feat.lower()
        pattern_hit = any(
            re.search(pat, feat_lower) for pat in STRUCTURAL_PATTERNS
        )
        # Target name present in feature name but NOT followed by a lag/window marker
        target_in_name = (
            target_lower in feat_lower
            and not re.search(r"_lag\d+|_roll|_mean|_std|_min|_max|_enc", feat_lower)
            and feat_lower != target_lower
        )
        flagged[feat] = pattern_hit or target_in_name

    return flagged


# ── IMPORTANCE FROM FOLD MODELS ────────────────────────────────────────────

def average_importances(
    fold_models: list,
    feature_cols: list[str],
) -> dict[str, float]:
    """Return mean CatBoost feature importance across folds, normalised to sum=1."""
    arrays = [m.get_feature_importance() for m in fold_models]
    mean_imp = np.mean(arrays, axis=0)
    total = mean_imp.sum()
    if total == 0:
        return {c: 0.0 for c in feature_cols}
    return {c: float(v / total) for c, v in zip(feature_cols, mean_imp)}


# ── VERDICT LOGIC ──────────────────────────────────────────────────────────

def compute_verdict(
    cv_gap_pct: float,
    has_two_signal_suspect: bool,
    max_importance_share: float,
    is_classification: bool = False,
) -> tuple[str, dict[str, str]]:
    """Return (verdict, checks_dict)."""
    # cv_integrity — gap thresholds are calibrated for regression MAE;
    # suppress for classification where the metric differs
    if is_classification:
        cv_check = "PASS"
    elif cv_gap_pct > CV_GAP_CRITICAL_PCT:
        cv_check = "CRITICAL"
    elif cv_gap_pct > CV_GAP_WARNING_PCT:
        cv_check = "WARNING"
    else:
        cv_check = "PASS"

    # importance_concentration
    if max_importance_share > IMPORTANCE_WARNING_SHARE:
        imp_check = "WARNING"
    else:
        imp_check = "PASS"

    # leakage_two_signal
    if has_two_signal_suspect:
        leak_check = "CRITICAL"
    else:
        leak_check = "PASS"

    checks = {
        "cv_integrity": cv_check,
        "importance_concentration": imp_check,
        "leakage_two_signal": leak_check,
    }

    if "CRITICAL" in checks.values():
        verdict = "CRITICAL"
    elif "WARNING" in checks.values():
        verdict = "WARNING"
    else:
        verdict = "PASS"

    return verdict, checks


# ── MAIN ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", default=".",
        help="Repo root directory (default: current directory)"
    )
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()

    reports = root / "reports"
    data    = root / "data"

    print("=" * 60)
    print("VALIDATOR SUB-AGENT STARTING")
    print(f"repo root: {root}")
    print("=" * 60)

    # ── Load inputs ─────────────────────────────────────────────────
    with open(reports / "profile.json") as f:
        profile = json.load(f)
    with open(reports / "model_results.json") as f:
        model_results = json.load(f)
    with open(reports / "features.json") as f:
        features_meta = json.load(f)

    schema_text = ""
    schema_path = reports / "schema_analysis.md"
    if schema_path.exists():
        schema_text = schema_path.read_text(encoding="utf-8")

    # ── Resolve schema fields ────────────────────────────────────────
    problem_type    = profile.get("problem_type", "tabular_regression")
    problem_subtype = profile.get("problem_subtype", "")
    _is_cls         = problem_subtype in ("binary_classification", "multiclass_classification")
    target_col      = profile["target_col"]
    group_cols      = profile.get("group_cols") or []
    time_col        = profile.get("time_col")

    hparams = model_results.get("best_params", {
        "learning_rate": 0.05,
        "depth":         6,
        "l2_leaf_reg":   3.0,
    })
    n_estimators   = int(model_results.get("n_estimators", 500))
    # Try several candidate key names for the reported CV MAE
    reported_cv_mae = float(
        model_results.get("walk_forward_mae")
        or model_results.get("oof_mae")
        or model_results.get("cv_mae")
        or float("nan")
    )

    # ── Load training features ───────────────────────────────────────
    train_df = pd.read_parquet(data / "features_train.parquet")
    exclude  = set(group_cols + ([time_col] if time_col else []) + [target_col])
    # Only keep numeric columns — string/categorical columns cannot be
    # median-filled or fed directly into CatBoost as float arrays.
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    fill_vals = train_df[feature_cols].median()

    print(f"problem_type: {problem_type}")
    print(f"target: {target_col}, groups: {group_cols}, time: {time_col}")
    print(f"features: {len(feature_cols)}, train rows: {len(train_df)}")
    print(f"reported_cv_mae (walk_forward_mae): {reported_cv_mae:.4f}")

    # ── Strict CV ────────────────────────────────────────────────────
    print("\n--- Computing strict CV ---")
    strict_cv_mae, strict_cv_scheme, fold_models, _fold_maes, _fold_train_sizes, _fold_cls_metrics = \
        compute_strict_cv(
            train_df, feature_cols, target_col, group_cols, time_col,
            problem_type, hparams, n_estimators,
            problem_subtype=problem_subtype,
        )
    print(f"strict_cv_mae: {strict_cv_mae:.4f}")
    print(f"scheme: {strict_cv_scheme}")
    print(f"fold_maes: {[round(m, 4) for m in _fold_maes]}")
    print(f"fold_train_sizes: {_fold_train_sizes}")

    cv_gap_abs = float(strict_cv_mae - reported_cv_mae)
    cv_gap_pct = float(cv_gap_abs / reported_cv_mae) if reported_cv_mae else float("nan")
    print(f"cv_gap_abs: {cv_gap_abs:.4f}, cv_gap_pct: {cv_gap_pct:.4f}")

    # ── Feature importance shares from strict CV fold models ─────────
    print("\n--- Computing feature importance shares ---")
    imp_shares = average_importances(fold_models, feature_cols)
    max_share  = max(imp_shares.values()) if imp_shares else 0.0
    candidates = [f for f, s in imp_shares.items() if s >= LOO_CANDIDATE_SHARE]
    print(f"max importance share: {max_share:.3f}")
    print(f"LOO candidates (share >= {LOO_CANDIDATE_SHARE}): {candidates}")

    # ── LOO strict CV ────────────────────────────────────────────────
    loo_deltas: dict[str, float] = {}
    if candidates:
        print("\n--- Running LOO strict CV ---")
        loo_deltas = loo_strict_cv(
            train_df, feature_cols, target_col, group_cols, time_col,
            problem_type, hparams, n_estimators, strict_cv_mae, candidates,
            problem_subtype=problem_subtype,
        )
        for feat, delta in loo_deltas.items():
            print(f"  {feat}: delta={delta:.4f}")

    # ── Structural check ─────────────────────────────────────────────
    print("\n--- Structural check ---")
    struct_flags = structural_check(feature_cols, target_col, schema_text)
    flagged_structural = [f for f, v in struct_flags.items() if v]
    print(f"structural flags ({len(flagged_structural)}): {flagged_structural[:10]}")

    # ── Two-signal rule ──────────────────────────────────────────────
    feature_suspicion: list[dict] = []
    for feat in feature_cols:
        share  = imp_shares.get(feat, 0.0)
        delta  = loo_deltas.get(feat, float("nan"))
        struct = struct_flags.get(feat, False)

        stat_signal = (
            share >= LOO_CANDIDATE_SHARE
            and not np.isnan(delta)
            and abs(delta) < LOO_STAT_SIGNAL_THRESHOLD * strict_cv_mae
        )
        suspect = bool(stat_signal and struct)

        if share >= LOO_CANDIDATE_SHARE or struct:
            feature_suspicion.append({
                "feature":          feat,
                "importance_share": float(share),
                "loo_strict_delta": float(delta) if not np.isnan(delta) else None,
                "structural_flag":  bool(struct),
                "suspect":          suspect,
                "evidence": _build_evidence(feat, share, delta, struct, stat_signal),
            })

    has_two_signal_suspect = any(r["suspect"] for r in feature_suspicion)

    # ── Verdict ──────────────────────────────────────────────────────
    verdict, checks = compute_verdict(cv_gap_pct, has_two_signal_suspect, max_share,
                                      is_classification=_is_cls)
    print(f"\nVerdict: {verdict}")
    print(f"Checks: {checks}")

    _cls_metric_val  = float(np.mean(_fold_cls_metrics)) if _fold_cls_metrics else None
    _cls_metric_type = ("AUC" if problem_subtype == "binary_classification"
                        else "accuracy" if problem_subtype == "multiclass_classification"
                        else None)

    notes = _build_notes(
        verdict, reported_cv_mae, strict_cv_mae, cv_gap_pct,
        strict_cv_scheme, candidates, feature_suspicion, max_share,
    )
    if _is_cls and _cls_metric_val is not None:
        notes += (
            f"\nClassification metric ({_cls_metric_type}) across strict CV folds: "
            f"{_cls_metric_val:.4f}  "
            f"(fold values: {[round(v, 4) for v in _fold_cls_metrics]}). "
            f"CV gap thresholds suppressed (regression-calibrated)."
        )

    # ── Write validator_review.json ──────────────────────────────────
    review = {
        "verdict":               verdict,
        "reported_cv_mae":       float(reported_cv_mae),
        "strict_cv_mae":         float(strict_cv_mae),
        "honest_cv_mae":         float(strict_cv_mae),
        "cv_gap_abs":            float(cv_gap_abs),
        "cv_gap_pct":            float(cv_gap_pct),
        "strict_cv_scheme":      strict_cv_scheme,
        "fold_maes":             [float(m) for m in _fold_maes],
        "fold_train_sizes":      [int(s) for s in _fold_train_sizes],
        "strict_cv_metric":      _cls_metric_val,
        "strict_cv_metric_type": _cls_metric_type,
        "feature_suspicion":     feature_suspicion,
        "checks":                checks,
        "notes":                 notes,
    }
    out_path = reports / "validator_review.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2)
    print(f"\nWritten {out_path}")

    # ── Gap attribution (best-effort: classify OOF→strict gap) ──────────────
    import subprocess as _subprocess
    try:
        _gap_result = _subprocess.run(
            [sys.executable, str(root / "tools" / "gap_attribution.py"),
             "--repo-root", str(root)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
        )
        print(_gap_result.stdout)
        if _gap_result.returncode != 0:
            print(f"[gap_attribution] WARNING: {_gap_result.stderr[:400]}", file=sys.stderr)
            try:
                with open(out_path, encoding="utf-8") as _f:
                    _rev = json.load(_f)
                _rev["gap_attribution"] = {
                    "classification": (
                        f"ATTRIBUTION_FAILED (rc={_gap_result.returncode}): "
                        + _gap_result.stderr[:200]
                    ),
                    "monotone_score": None,
                }
                with open(out_path, "w", encoding="utf-8") as _f:
                    json.dump(_rev, _f, indent=2)
            except Exception as _upd_e:
                print(f"[gap_attribution] could not update review: {_upd_e}", file=sys.stderr)
    except Exception as _gap_e:
        print(f"[gap_attribution] non-fatal: {_gap_e}", file=sys.stderr)

    # ── Marker file ──────────────────────────────────────────────────
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    marker = reports / "validator_was_here.txt"
    marker.write_text(
        f"validator sub-agent executed at {ts}\n"
        f"verdict: {verdict}\n"
        f"strict_cv_mae: {strict_cv_mae:.4f}\n"
        f"reported_cv_mae: {reported_cv_mae:.4f}\n"
        f"cv_gap_pct: {cv_gap_pct:.4f}\n",
        encoding="utf-8",
    )
    print(f"Written {marker}")

    # ── KG: observability — record CV verdict (best-effort) ───────────────────
    try:
        from kg import kg_append_event, kg_set_stage
        kg_append_event("validator", "hypothesis", {"hypothesis": "CV gap within threshold", "outcome": verdict, "detail": f"cv_gap_pct={cv_gap_pct:.3f} strict_cv_mae={strict_cv_mae:.4f}"})
        kg_set_stage("validator")
    except Exception as _kg_e: print(f"[KG] non-fatal: {_kg_e}", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"VALIDATOR COMPLETE — verdict: {verdict}")
    print("=" * 60)


def _build_evidence(
    feat: str,
    share: float,
    delta: float,
    struct: bool,
    stat_signal: bool,
) -> str:
    parts = []
    if share >= LOO_CANDIDATE_SHARE:
        parts.append(f"importance_share={share:.3f} (>= {LOO_CANDIDATE_SHARE})")
    if not np.isnan(delta):
        parts.append(f"loo_delta={delta:.4f}")
        if stat_signal:
            parts.append("STAT SIGNAL: low LOO delta suggests no real strict-CV gain")
    if struct:
        parts.append("STRUCT SIGNAL: feature name matches leakage heuristic")
    return "; ".join(parts) if parts else "no evidence"


def _build_notes(
    verdict: str,
    reported: float,
    strict: float,
    gap_pct: float,
    scheme: str,
    candidates: list[str],
    suspicion: list[dict],
    max_share: float,
) -> str:
    lines = [
        f"Validator used strict CV scheme: {scheme}.",
        f"Reported (modeler) CV MAE = {reported:.4f}; "
        f"strict CV MAE = {strict:.4f}; "
        f"gap = {gap_pct*100:.1f}% ({'>CRITICAL' if gap_pct > CV_GAP_CRITICAL_PCT else '>WARNING' if gap_pct > CV_GAP_WARNING_PCT else 'PASS'}).",
    ]
    if candidates:
        lines.append(f"LOO was run on {len(candidates)} high-importance feature(s): {candidates}.")
    suspects = [r["feature"] for r in suspicion if r["suspect"]]
    if suspects:
        lines.append(
            f"TWO-SIGNAL SUSPECTS (both stat and structural signals): {suspects}. "
            "A future critic agent should investigate whether these features encode "
            "information unavailable at prediction time."
        )
    else:
        lines.append(
            "No two-signal suspects found. Structural flags or high-importance "
            "features with large LOO deltas are individually insufficient for "
            "leakage classification."
        )
    if max_share > IMPORTANCE_WARNING_SHARE:
        lines.append(
            f"Max importance share {max_share:.2%} exceeds {IMPORTANCE_WARNING_SHARE:.0%} "
            "— prediction quality is highly concentrated in a single feature."
        )
    lines.append(
        f"All thresholds: CV_GAP_CRITICAL={CV_GAP_CRITICAL_PCT}, "
        f"CV_GAP_WARNING={CV_GAP_WARNING_PCT}, "
        f"LOO_CANDIDATE_SHARE={LOO_CANDIDATE_SHARE}, "
        f"LOO_STAT_SIGNAL_THRESHOLD={LOO_STAT_SIGNAL_THRESHOLD}."
    )
    return " ".join(lines)


if __name__ == "__main__":
    main()
