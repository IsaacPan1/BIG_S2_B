"""scheme_analysis.py — CV Architect.

The ONLY module allowed to author reports/cv_plan.json. Reads the raw data
directory and DATA_DESCRIPTION.md, classifies the problem, picks a CV scheme,
enumerates leakage risks, and writes the frozen CV_PLAN contract.

Public surface:
    write_cv_plan(data_dir, reports_dir) -> dict
"""
from __future__ import annotations

import json
import secrets
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CV-scheme selection: recent-vs-full drift diagnostic
#
# Default is TimeSeriesExpanding (uses ALL available history). Switch to
# TimeSeriesSliding ONLY when the drift between train and val is genuinely
# RECENCY-REDUCIBLE — i.e. restricting train to its most recent block brings
# the covariate distribution closer to val. This is what makes "use recent
# history only" actually buy something over expanding.
#
# Why not the old KS / adversarial-AUC gate?
#   The previous gate triggered sliding whenever train→val covariate shift was
#   "large" by KS or by an adversarial classifier. On any forecasting holdout
#   that signal saturates: val is a future window, so KS/AUC are large by
#   construction whether or not recency would help. The old gate therefore
#   picked sliding almost always, discarding history for no measurable gain.
#
# The new diagnostic:
#   - excludes window/seasonal + time-index features (those encode "val is the
#     future", not drift in the things sliding could reduce),
#   - per remaining numeric feature, computes the standardized mean shift
#     |mu_train - mu_val| / pooled_std, twice: FULL train vs val and RECENT
#     train (last RECENT_PERIODS by rank) vs val,
#   - aggregates: mean_improvement = mean(dist_full - dist_recent),
#     rel = mean_improvement / mean_dist_full,
#     frac_improved = share of features whose improvement > 0.05.
#
# Decision rule (default expanding):
#   sliding  IF rel >= DRIFT_REL_THRESHOLD
#             AND frac_improved >= DRIFT_FRAC_IMPROVED_THRESHOLD
#   else      expanding
#
# Val covariates are USED for this diagnostic. This is allowed: test
# covariates are provided by the competition; only the TARGET on val is
# forbidden, and the diagnostic never reads it.
# ─────────────────────────────────────────────────────────────────────────────
RECENT_PERIODS = 14
DRIFT_REL_THRESHOLD = 0.25
DRIFT_FRAC_IMPROVED_THRESHOLD = 0.40
DRIFT_IMPROVE_PER_FEATURE_THRESHOLD = 0.05  # min per-feature improvement to count toward frac_improved

# Engineered seasonality / time-index columns must be excluded from the drift
# scan: they're deterministic functions of period rank, so they separate train
# from val by construction. None of these exist in raw covariate CSVs (this
# script runs BEFORE feature engineering), but the exclusion is kept for
# robustness on datasets that ship raw seasonality features.
_WINDOW_FEATURES = {
    "period_id_ord", "period_id_trend", "period_id_of_cycle", "horizon",
    "period_id_sin", "period_id_cos", "period_id_sin2", "period_id_cos2",
    "period_id_quarter", "period_id_month", "month_of_year",
    "quarter_of_year", "is_quarter_start",
}
_NON_FEATURES = {"adversarial_weights"}


def _standardized_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    """|mean(a) - mean(b)| / pooled_std. Effect size; 0 = identical; does NOT
    saturate the way an adversarial AUC does. Matches recent_vs_full_dist.py."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    sp = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
    if sp == 0.0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(b)) / sp)


def _compute_drift_diagnostic(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    profile: dict,
    time_col: Optional[str],
) -> dict:
    """Recent-vs-full drift diagnostic. Returns a verdict dict containing the
    summary stats and an `ok` flag. When `ok` is False, the `reason` field
    explains why the diagnostic could not run — the caller falls back to
    expanding in that case (NEVER to sliding on missing evidence)."""
    if val_df is None or train_df is None:
        return {"ok": False, "reason": "no val frame available (profile.val_files empty or unreadable)"}
    if not time_col or time_col not in train_df.columns:
        return {"ok": False, "reason": f"time_col {time_col!r} not present in train frame"}

    common = [
        c for c in train_df.columns
        if c in val_df.columns
        and pd.api.types.is_numeric_dtype(train_df[c])
        and c not in _WINDOW_FEATURES
        and c not in _NON_FEATURES
    ]
    if not common:
        return {"ok": False, "reason": "no shared numeric non-window features between train and val"}

    info = profile.get("period_rank_info") or {}
    id_to_rank = info.get("id_to_rank") or {}
    if id_to_rank:
        tr_rank = train_df[time_col].map(id_to_rank)
    else:
        tr_rank = pd.to_numeric(train_df[time_col], errors="coerce")
    tr_rank_valid = tr_rank.dropna()
    if tr_rank_valid.empty:
        return {"ok": False, "reason": "could not derive numeric rank for train rows"}

    rank_max = int(tr_rank_valid.max())
    rank_span = rank_max + 1
    if rank_span < RECENT_PERIODS:
        return {"ok": False,
                "reason": f"train spans {rank_span} periods < RECENT_PERIODS={RECENT_PERIODS}"}
    cutoff = rank_max - RECENT_PERIODS + 1
    recent_mask = (tr_rank >= cutoff).fillna(False).to_numpy()

    rows = []
    for c in common:
        a_full = train_df[c].to_numpy(dtype=float, copy=False)
        a_recent = train_df.loc[recent_mask, c].to_numpy(dtype=float, copy=False)
        b_val = val_df[c].to_numpy(dtype=float, copy=False)
        d_full = _standardized_mean_shift(a_full, b_val)
        d_recent = _standardized_mean_shift(a_recent, b_val)
        if np.isnan(d_full) or np.isnan(d_recent):
            continue
        rows.append((c, d_full, d_recent, d_full - d_recent))

    if not rows:
        return {"ok": False, "reason": "no feature yielded a valid distance (too many NaNs?)"}

    mean_full = float(np.mean([r[1] for r in rows]))
    mean_recent = float(np.mean([r[2] for r in rows]))
    mean_impr = float(np.mean([r[3] for r in rows]))
    frac_improved = float(np.mean([r[3] > DRIFT_IMPROVE_PER_FEATURE_THRESHOLD for r in rows]))
    rel = float(mean_impr / mean_full) if mean_full > 0 else 0.0

    return {
        "ok":                  True,
        "n_features_scanned":  len(rows),
        "rank_max":            rank_max,
        "recent_periods":      RECENT_PERIODS,
        "recent_cutoff_rank":  cutoff,
        "mean_dist_full":      mean_full,
        "mean_dist_recent":    mean_recent,
        "mean_improvement":    mean_impr,
        "rel":                 rel,
        "frac_improved":       frac_improved,
        "per_feature": [
            {"feature": c, "dist_full": float(d_full),
             "dist_recent": float(d_recent), "improvement": float(d_full - d_recent)}
            for (c, d_full, d_recent, _) in rows
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data discovery — sourced entirely from reports/profile.json
# ─────────────────────────────────────────────────────────────────────────────

def _load_profile(reports_dir: Path) -> dict:
    """Read reports/profile.json or fail loudly. profile.json is the sole
    source of truth for file paths, target column, group/time columns."""
    profile_path = reports_dir / "profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"{profile_path} not found. Run `python tools/profile_data.py "
            f"--data-dir data/ --output reports/profile.json` first; "
            f"scheme_analysis derives all file/column discovery from it."
        )
    with open(profile_path, encoding="utf-8") as f:
        return json.load(f)


def _load_train_frame(data_dir: Path, profile: dict) -> tuple[pd.DataFrame, str]:
    """Assemble the training frame from profile['train_files'].

    Each entry is a path relative to data_dir (including subdirectories such as
    'train/covariates.csv'). Files containing the target column are merged onto
    covariate-only files via their shared keys.
    """
    train_files: list[str] = list(profile.get("train_files") or [])
    target_col: Optional[str] = profile.get("target_col")

    if not train_files:
        raise ValueError(
            "profile.json has no train_files; cannot build a training frame. "
            "Re-run tools/profile_data.py to regenerate profile.json."
        )

    resolved: list[tuple[str, Path]] = []
    for rel in train_files:
        p = data_dir / rel
        if not p.exists():
            raise FileNotFoundError(
                f"profile.json lists train file {rel!r} but {p} is missing."
            )
        resolved.append((rel, p))

    frames: dict[str, pd.DataFrame] = {rel: pd.read_csv(p) for rel, p in resolved}

    if len(frames) == 1:
        rel, df = next(iter(frames.items()))
        return df, rel

    if target_col:
        target_frames = {n: d for n, d in frames.items() if target_col in d.columns}
        cov_frames    = {n: d for n, d in frames.items() if target_col not in d.columns}
    else:
        target_frames, cov_frames = {}, frames

    if cov_frames and target_frames:
        cov_df = pd.concat(list(cov_frames.values()), ignore_index=True)
        tgt_df = pd.concat(list(target_frames.values()), ignore_index=True)
        join_keys = [c for c in cov_df.columns if c in tgt_df.columns]
        if not join_keys:
            raise ValueError(
                f"Cannot merge covariates {list(cov_frames)} with target file(s) "
                f"{list(target_frames)}: no shared columns to join on."
            )
        df = cov_df.merge(tgt_df, on=join_keys, how="left")
        source = "+".join(list(cov_frames) + list(target_frames))
        return df, source

    # All files either have or lack the target — concatenate them
    df = pd.concat(list(frames.values()), ignore_index=True)
    return df, "+".join(frames.keys())


def _load_val_frame(data_dir: Path, profile: dict) -> Optional[pd.DataFrame]:
    """Assemble the validation covariates frame from profile['val_files'].
    Used ONLY by the drift diagnostic — covariates are competition-provided
    test inputs; the target (which val files do not contain) is never read.
    Returns None if val files are absent or unreadable."""
    val_files: list[str] = list(profile.get("val_files") or [])
    if not val_files:
        return None
    frames: list[pd.DataFrame] = []
    for rel in val_files:
        p = data_dir / rel
        if p.exists():
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                continue
    if not frames:
        return None
    df = frames[0]
    for nxt in frames[1:]:
        on = [c for c in df.columns if c in nxt.columns]
        df = df.merge(nxt, on=on, how="left") if on else pd.concat([df, nxt], axis=1)
    return df.reset_index(drop=True)


def _load_diagnostic_frames(
    data_dir: Path,
    profile: dict,
    raw_train: pd.DataFrame,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """Pick the feature frames the drift diagnostic should scan.

    The CV-scheme choice should reflect drift in the features the MODEL will
    actually consume. If feature_engineer has already produced engineered
    parquets (data/features_train.parquet + features_val.parquet), prefer
    them — the engineered set contains the ratios / rolling stats / lags that
    can make raw-covariate drift look reducible when it is not (or vice
    versa). Otherwise fall back to raw covariates from CSVs (first-pass plan
    time on an unseen dataset). The diagnostic algorithm is identical in
    either case; only the column set changes.

    Returns (train_df, val_df_or_None, source_label)."""
    parq_tr = data_dir / "features_train.parquet"
    parq_vl = data_dir / "features_val.parquet"
    if parq_tr.exists() and parq_vl.exists():
        try:
            tr = pd.read_parquet(parq_tr)
            vl = pd.read_parquet(parq_vl)
            return tr, vl, "engineered_parquet"
        except Exception:
            pass
    val_df = _load_val_frame(data_dir, profile)
    return raw_train, val_df, "raw_csv"


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────


def _classify(df: pd.DataFrame, target_col: str,
              time_col: Optional[str], group_cols: list[str]) -> tuple[str, str, str]:
    """Return (problem_type, problem_subtype, cv_type)."""
    y = df[target_col].dropna()
    n_unique = int(y.nunique())
    is_integer = bool(pd.api.types.is_integer_dtype(y) or
                      ((y - y.round()).abs() < 1e-9).all() if len(y) else False)
    has_time = time_col is not None
    has_groups = bool(group_cols)

    if has_time and has_groups:
        problem_type = "grouped_time_series"
        cv_type = "TimeSeriesExpanding"
    elif has_time:
        problem_type = "time_series"
        cv_type = "TimeSeriesExpanding"
    elif has_groups:
        problem_type = "tabular_iid"
        cv_type = "GroupKFold"
    elif n_unique <= 2:
        problem_type = "tabular_iid"
        cv_type = "StratifiedKFold"
    elif n_unique <= 50 and is_integer:
        problem_type = "tabular_iid"
        cv_type = "StratifiedKFold"
    else:
        problem_type = "tabular_iid"
        cv_type = "KFold"

    if n_unique == 2:
        subtype = "binary_classification"
    elif n_unique <= 50 and is_integer and (y.min() >= 0 and y.max() <= 50):
        subtype = "ordinal_regression"
    elif n_unique <= 50 and not is_integer:
        subtype = "continuous_regression"
    else:
        subtype = "continuous_regression"

    return problem_type, subtype, cv_type


def _enumerate_leakage_risks(df: pd.DataFrame, target_col: str,
                             time_col: Optional[str],
                             group_cols: list[str]) -> list[dict]:
    risks: list[dict] = []
    if time_col:
        risks.append({
            "id": "L1",
            "kind": "future_timestamp",
            "column": time_col,
            "mitigation": "Validator enforces max(train_time)+gap <= min(valid_time)",
        })
    if group_cols:
        risks.append({
            "id": "L2",
            "kind": "group_overlap",
            "column": group_cols[0],
            "mitigation": ("Within-group forecast is intended for time-series; "
                           "GroupKFold enforces disjoint groups otherwise"),
        })
    derived = [c for c in df.columns
               if c != target_col and target_col.lower() in c.lower()]
    for col in derived:
        risks.append({
            "id": f"L3:{col}",
            "kind": "target_derived",
            "column": col,
            "mitigation": "Feature engineer must recompute from train_idx only",
        })
    for c in df.columns:
        if c.lower() in ("id", "row_id", "uid"):
            risks.append({
                "id": f"L4:{c}",
                "kind": "id_columns_as_features",
                "column": c,
                "mitigation": "Drop in feature_engineer (uniquely identifies the row)",
            })
    return risks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def write_cv_plan(data_dir: str | Path = "data",
                  reports_dir: str | Path = "reports") -> dict:
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_profile(reports_dir)

    target_col: Optional[str] = profile.get("target_col")
    if not target_col:
        raise ValueError(
            "profile.json has no target_col; cannot author cv_plan. "
            "Re-run tools/profile_data.py."
        )
    time_col: Optional[str] = profile.get("time_col")
    group_cols: list[str] = list(profile.get("group_cols") or [])

    df, source = _load_train_frame(data_dir, profile)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} (from profile.json) is not in the "
            f"assembled training frame columns: {list(df.columns)[:15]}…"
        )

    problem_type, subtype, cv_type = _classify(df, target_col, time_col, group_cols)

    # Default to TimeSeriesExpanding for any time-aware problem; switch to
    # TimeSeriesSliding ONLY if the recent-vs-full drift diagnostic shows the
    # drift is recency-reducible. On missing/unrunnable evidence, fall back to
    # expanding — never default to sliding on a missing diagnostic.
    default_cv_type_for_problem = cv_type
    diag_train, diag_val, diag_source = _load_diagnostic_frames(data_dir, profile, df)
    drift_diag = _compute_drift_diagnostic(diag_train, diag_val, profile, time_col)
    drift_diag["diagnostic_source"] = diag_source
    if cv_type == "TimeSeriesExpanding":
        if drift_diag.get("ok"):
            rel = drift_diag["rel"]
            frac = drift_diag["frac_improved"]
            if rel >= DRIFT_REL_THRESHOLD and frac >= DRIFT_FRAC_IMPROVED_THRESHOLD:
                cv_type = "TimeSeriesSliding"
                drift_branch = (
                    f"sliding (drift is recency-reducible: rel={rel:.3f} >= "
                    f"{DRIFT_REL_THRESHOLD} AND frac_improved={frac:.3f} >= "
                    f"{DRIFT_FRAC_IMPROVED_THRESHOLD})"
                )
            else:
                drift_branch = (
                    f"expanding (drift not recency-reducible: rel={rel:.3f}, "
                    f"frac_improved={frac:.3f}; thresholds rel>={DRIFT_REL_THRESHOLD}, "
                    f"frac_improved>={DRIFT_FRAC_IMPROVED_THRESHOLD})"
                )
        else:
            drift_branch = (
                f"expanding (fallback — drift diagnostic could not run: "
                f"{drift_diag.get('reason', 'unknown')})"
            )
    else:
        drift_branch = f"{cv_type} (not a time-series scheme; diagnostic not applied)"

    horizon = None
    n_splits = 5
    valid_size = None
    plan_gap = 0
    window_size = None
    if cv_type in ("TimeSeriesExpanding", "TimeSeriesSliding", "RollingOriginCV"):
        pri = profile.get("period_rank_info") or {}
        if pri.get("available") and int(pri.get("n_val_periods") or 0) > 0:
            # Codebook-resolved: use unique val rank count, not row counts.
            n_val_periods = int(pri["n_val_periods"])
            valid_size = n_val_periods
            horizon = n_val_periods
            # Profiler's gap_periods = val_min_rank - train_max_rank.
            # Contiguous months (offset=1) → plan gap=0 (no embargo, valid
            # starts at train_end+1 by cv_engine convention).
            profiler_offset = int(profile.get("gap_periods") or 1)
            plan_gap = max(0, profiler_offset - 1)
        else:
            # No codebook — fall back to the existing row-count estimate.
            if time_col and group_cols:
                sample_g = df.groupby(group_cols)[time_col]
                avg_len = float(sample_g.size().mean())
            else:
                avg_len = float(len(df))
            valid_size = max(1, int(avg_len * 0.1))
            horizon = valid_size

        # Sliding needs a window; default = 3 * valid_size, clamped < n_train_periods.
        if cv_type == "TimeSeriesSliding" and valid_size:
            default_win = 3 * valid_size
            n_train_periods = int(pri.get("n_train_periods") or 0)
            if n_train_periods > 0:
                window_size = max(valid_size, min(default_win, n_train_periods - 1))
            else:
                window_size = default_win

    cv_selection_reason = {
        "selected_cv_type":          cv_type,
        "default_for_problem_type":  default_cv_type_for_problem,
        "rule_branch":               drift_branch,
        "diagnostic": "recent_vs_full drift (standardized mean shift, no classifier; "
                      "val covariates used, val target NOT read)",
        "thresholds": {
            "rel":            DRIFT_REL_THRESHOLD,
            "frac_improved":  DRIFT_FRAC_IMPROVED_THRESHOLD,
            "recent_periods": RECENT_PERIODS,
        },
        "drift_metrics": {
            "ok":                 bool(drift_diag.get("ok", False)),
            "reason":             drift_diag.get("reason"),
            "diagnostic_source":  drift_diag.get("diagnostic_source"),
            "n_features_scanned": drift_diag.get("n_features_scanned"),
            "rank_max":           drift_diag.get("rank_max"),
            "recent_cutoff_rank": drift_diag.get("recent_cutoff_rank"),
            "rel":                drift_diag.get("rel"),
            "frac_improved":      drift_diag.get("frac_improved"),
            "mean_dist_full":     drift_diag.get("mean_dist_full"),
            "mean_dist_recent":   drift_diag.get("mean_dist_recent"),
            "mean_improvement":   drift_diag.get("mean_improvement"),
        },
    }

    leakage = _enumerate_leakage_risks(df, target_col, time_col, group_cols)

    plan = {
        "schema_version": "1.0",
        "plan_id": "cvplan_" + secrets.token_hex(4),
        "frozen": True,
        "immutable_note": ("CV_PLAN is created ONLY by scheme_analysis. "
                           "No other module may modify, recompute, or override this file."),
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "data_source": source,

        "problem_type": problem_type,
        "problem_subtype": subtype,
        "target_column": target_col,
        "time_column": time_col,
        "group_columns": group_cols,
        "horizon": horizon,
        "n_train_rows": int(len(df)),

        "cv": {
            "cv_type": cv_type,
            "n_splits": n_splits,
            "window_size": window_size,
            "gap": plan_gap,
            "step": None,
            "min_train_size": None,
            "valid_size": valid_size,
            "shuffle": cv_type in ("KFold", "StratifiedKFold"),
            "random_state": 42,
        },

        "cv_selection_reason": cv_selection_reason,

        "fold_index_contract": {
            "format": "list[{fold_id, train_idx, valid_idx}]",
            "storage_path": "reports/cv_folds.json",
        },

        "leakage_risks": leakage,

        "feature_engineer_contract": {
            "fit_visibility": "train_idx only",
            "transform_visibility": "train_idx + valid_idx",
            "forbidden": ["global statistics across full dataset",
                          "future timestamps in fit",
                          "cross-fold caches"],
            "required_artefacts_per_fold": [
                "data/features_train_fold_{k}.parquet",
                "data/features_valid_fold_{k}.parquet",
                "reports/feature_manifest_fold_{k}.json",
            ],
        },

        "modeler_contract": {
            "active_backends": ["catboost", "ridge"],
            "interface_only_backends": ["lightgbm", "xgboost"],
            "per_fold_outputs": [
                "reports/predictions_fold_{k}.parquet",
                "reports/importance_fold_{k}.json",
            ],
            "must_be_cv_agnostic": True,
        },

        "validator_contract": {
            "role": "CVEngine + OOF aggregator",
            "outputs": [
                "reports/cv_folds.json",
                "reports/oof_predictions.parquet",
                "reports/fold_metrics.json",
                "reports/validator_review.json",
            ],
        },

        "replan_policy": {
            "max_replans_per_run": 1,
            "triggers_allowed_from": ["critic:CV_INVALID"],
        },
    }

    out = reports_dir / "cv_plan.json"
    with open(out, "w") as f:
        json.dump(plan, f, indent=2)

    marker = reports_dir / "schema_analyst_was_here.txt"
    with open(marker, "w") as f:
        f.write(f"scheme_analysis executed at {plan['created_at_utc']} plan_id={plan['plan_id']}\n")

    return plan


def load_cv_plan(reports_dir: str | Path = "reports") -> dict:
    """Read the frozen CV_PLAN. Asserts immutability."""
    with open(Path(reports_dir) / "cv_plan.json") as f:
        plan = json.load(f)
    assert plan.get("frozen") is True, "CV_PLAN is not marked frozen"
    return plan


if __name__ == "__main__":
    plan = write_cv_plan()
    print(f"Wrote reports/cv_plan.json  plan_id={plan['plan_id']}")
    print(f"  cv_type={plan['cv']['cv_type']}  n_splits={plan['cv']['n_splits']}")
    print(f"  target={plan['target_column']}  time={plan['time_column']}  "
          f"groups={plan['group_columns']}")
