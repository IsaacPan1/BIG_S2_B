#!/usr/bin/env python3
"""
tools/family_ablation.py — Dataset-agnostic leave-one-family-out ablation.

Tests whether removing each named feature family (from features.json
"feature_families") improves strict purged CV MAE on the training data.
Uses the modeler's best_params — no hyperparameter re-tuning.

A family is NET-HARMFUL only when:
    (strict_mae_without_family + MARGIN) < strict_mae_full
where MARGIN = max(3 * seed_std, 0.005) to clear the noise floor.
Borderline improvements do NOT get flagged.

BUDGET GATE runs first: times one pilot refit; skips if estimated total
wall-clock exceeds --max-seconds (default 240 s).

All errors are non-fatal: logs and exits cleanly; never blocks submission.

Writes: reports/family_ablation.json

Usage:
    python tools/family_ablation.py [--repo-root PATH] [--max-seconds N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from validate import (
    _panel_strict_splits,
    _grouped_strict_splits,
    N_STRICT_FOLDS,
    EMBARGO_PERIODS,
)

DEFAULT_MAX_SECONDS = 240    # 4-minute hard cap for ablation phase
MIN_MARGIN          = 0.005  # absolute floor for the net-harmful margin


def _strict_cv_one_seed(
    train_df:     pd.DataFrame,
    feature_cols: list[str],
    target_col:   str,
    time_col:     str | None,
    group_cols:   list[str],
    problem_type: str,
    hparams:      dict,
    n_estimators: int,
    seed:         int,
) -> float:
    """Run one-seed strict CV; return mean fold MAE."""
    if problem_type == "panel_forecasting" and time_col and time_col in train_df.columns:
        splits = _panel_strict_splits(train_df, time_col, N_STRICT_FOLDS, EMBARGO_PERIODS)
    elif group_cols and group_cols[0] in train_df.columns:
        splits = _grouped_strict_splits(train_df, group_cols[0], N_STRICT_FOLDS)
    else:
        n   = len(train_df)
        cut = int(n * 0.8)
        splits = [(train_df.index[:cut], train_df.index[cut:])]

    params = {
        "objective":    "regression_l1",
        "metric":       "mae",
        "n_estimators": n_estimators,
        "bagging_freq": 5,
        "reg_alpha":    0.1,
        "reg_lambda":   0.1,
        "verbose":      -1,
        "n_jobs":       -1,
        "random_state": seed,
        **hparams,
    }
    fold_maes: list[float] = []
    for tr_idx, vl_idx in splits:
        fold_tr = train_df.loc[tr_idx]
        fold_vl = train_df.loc[vl_idx]
        fill    = fold_tr[feature_cols].median()
        X_tr    = fold_tr[feature_cols].fillna(fill).values.astype(np.float32)
        y_tr    = fold_tr[target_col].values.astype(np.float32)
        X_vl    = fold_vl[feature_cols].fillna(fill).values.astype(np.float32)
        y_vl    = fold_vl[target_col].values.astype(np.float32)
        model   = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
        preds   = np.clip(model.predict(X_vl), 0, None)
        fold_maes.append(float(mean_absolute_error(y_vl, preds)))

    return float(np.mean(fold_maes)) if fold_maes else float("nan")


def run_family_ablation(
    repo_root:   Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> dict:
    reports  = repo_root / "reports"
    data_dir = repo_root / "data"
    out_path = reports / "family_ablation.json"

    print("=" * 62)
    print("FAMILY ABLATION START")
    print("=" * 62)

    # ── Load inputs ────────────────────────────────────────────────────────────
    with open(reports / "features.json")      as f: feat_meta     = json.load(f)
    with open(reports / "model_results.json") as f: model_results = json.load(f)
    with open(reports / "profile.json")       as f: profile       = json.load(f)

    target_col   = feat_meta.get("target_col")   or profile.get("target_col")
    group_cols   = profile.get("group_cols")      or feat_meta.get("group_cols") or []
    time_col     = profile.get("time_col")        or feat_meta.get("time_col")
    problem_type = profile.get("problem_type", "tabular_regression")

    families: dict[str, list[str]] = feat_meta.get("feature_families", {})

    # Extract best_params and n_estimators from model_results (multi-key support)
    hparams = (
        model_results.get("best_params")
        or model_results.get("families", {}).get("lightgbm", {}).get("best_params")
        or {}
    )
    n_estimators = int(
        model_results.get("n_estimators")
        or model_results.get("families", {}).get("lightgbm", {}).get("n_estimators", 500)
    )

    train_df = pd.read_parquet(data_dir / "features_train.parquet")

    # Build the full feature set (numeric, non-identifier)
    _excl = set(group_cols) | {target_col}
    if time_col:
        _excl.add(time_col)
    all_feature_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in _excl
    ]

    # Validate that each declared family has at least one feature in the parquet
    valid_families: dict[str, list[str]] = {}
    for fam, cols in families.items():
        actual = [c for c in cols if c in all_feature_cols]
        if actual:
            valid_families[fam] = actual
        else:
            print(f"  [ablation] skip family '{fam}': no matching columns in train parquet")

    n_families = len(valid_families)
    print(f"target={target_col}  problem_type={problem_type}")
    print(f"train shape: {train_df.shape}  all_feature_cols: {len(all_feature_cols)}")
    print(f"valid families to test: {n_families}")
    for fam, cols in sorted(valid_families.items()):
        print(f"  {fam}: {len(cols)} features")

    if n_families == 0:
        result = {
            "skipped": True,
            "reason":           "no valid feature families found in features.json",
            "families":         {},
            "net_harmful_families": [],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("[ablation] no valid families — writing empty result")
        return result

    # ── BUDGET GATE: time one pilot refit ──────────────────────────────────────
    print(f"\nBudget gate: timing one pilot strict-CV refit ...")
    t_start = time.time()
    try:
        _ = _strict_cv_one_seed(
            train_df, all_feature_cols, target_col, time_col, group_cols,
            problem_type, hparams, n_estimators, seed=42,
        )
        pilot_time = time.time() - t_start
    except Exception as e:
        msg = f"pilot refit error: {e}"
        print(f"[ablation] {msg} — ablation skipped")
        result = {
            "skipped": True, "reason": msg,
            "families": {}, "net_harmful_families": [],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # Need: 2 seeds for full arm + 1 seed per family
    estimated_total = pilot_time * (2 + n_families) * 1.2   # 20% safety margin
    print(
        f"  pilot_time={pilot_time:.1f}s  n_families={n_families}  "
        f"estimated_total={estimated_total:.0f}s  max_allowed={max_seconds:.0f}s"
    )

    if estimated_total > max_seconds:
        msg = (
            f"ablation skipped: insufficient budget "
            f"(estimated {estimated_total:.0f}s > {max_seconds:.0f}s allowed)"
        )
        print(f"[ablation] {msg}")
        result = {
            "skipped":          True,
            "reason":           msg,
            "families":         {},
            "net_harmful_families": [],
            "pilot_time_s":     float(pilot_time),
            "estimated_total_s": float(estimated_total),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # ── Baseline: 2-seed full run to estimate seed_std ────────────────────────
    print(f"\nComputing full-feature baseline (2 seeds for seed_std) ...")
    seeds_full = [42, 7]
    full_seed_maes: list[float] = []
    for seed in seeds_full:
        mae = _strict_cv_one_seed(
            train_df, all_feature_cols, target_col, time_col, group_cols,
            problem_type, hparams, n_estimators, seed=seed,
        )
        full_seed_maes.append(mae)
        print(f"  seed={seed}  mae={mae:.4f}")

    full_strict_mae = float(np.mean(full_seed_maes))
    seed_std        = float(np.std(full_seed_maes))
    margin          = float(max(3.0 * seed_std, MIN_MARGIN))
    print(
        f"  full_strict_mae={full_strict_mae:.4f}  "
        f"seed_std={seed_std:.4f}  margin={margin:.4f}"
    )

    # ── Leave-one-family-out ───────────────────────────────────────────────────
    print(f"\nLeave-one-family-out ({n_families} families) ...")
    family_results: dict[str, dict] = {}
    t_deadline = t_start + max_seconds * 0.95   # stop 5% before cap

    for fam, fam_cols in sorted(valid_families.items(), key=lambda kv: -len(kv[1])):
        if time.time() > t_deadline:
            print("[ablation] time limit reached — skipping remaining families")
            break

        reduced = [c for c in all_feature_cols if c not in set(fam_cols)]
        if not reduced:
            family_results[fam] = {
                "n_features_dropped": len(fam_cols),
                "strict_mae_without": None,
                "delta_vs_full":      None,
                "seed_std":           float(seed_std),
                "margin":             float(margin),
                "flagged":            False,
                "skip_reason":        "dropping this family leaves 0 features",
            }
            print(f"  {fam:30s}  SKIP — would leave 0 features")
            continue

        t_fam = time.time()
        mae_without = _strict_cv_one_seed(
            train_df, reduced, target_col, time_col, group_cols,
            problem_type, hparams, n_estimators, seed=42,
        )
        elapsed_fam = time.time() - t_fam
        delta       = float(mae_without - full_strict_mae)

        # NET-HARMFUL: dropping the family improves strict MAE by more than the margin
        # (i.e., mae_without is notably LOWER than full_strict_mae)
        net_harmful = (mae_without + margin) < full_strict_mae

        marker = "  *** NET-HARMFUL" if net_harmful else ""
        print(
            f"  {fam:30s}  drop={len(fam_cols):3d}  "
            f"mae_without={mae_without:.4f}  delta={delta:+.4f}  "
            f"margin={margin:.4f}{marker}  ({elapsed_fam:.0f}s)"
        )

        family_results[fam] = {
            "n_features_dropped": len(fam_cols),
            "strict_mae_without": float(mae_without),
            "delta_vs_full":      float(delta),
            "seed_std":           float(seed_std),
            "margin":             float(margin),
            "flagged":            bool(net_harmful),
        }

    net_harmful_families = [
        fam for fam, r in family_results.items()
        if r.get("flagged", False)
    ]

    result = {
        "skipped":              False,
        "full_strict_mae":      float(full_strict_mae),
        "seed_std":             float(seed_std),
        "margin":               float(margin),
        "families":             family_results,
        "net_harmful_families": net_harmful_families,
        "pilot_time_s":         float(pilot_time),
        "total_time_s":         float(time.time() - t_start),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"\n[ablation] complete: {len(net_harmful_families)} net-harmful families: "
        f"{net_harmful_families}"
    )
    print(f"  Written {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-one-family-out strict-CV ablation."
    )
    parser.add_argument("--repo-root",   default=".", help="Repo root directory")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help="Max wall-clock seconds for ablation (default 240)")
    args = parser.parse_args()
    run_family_ablation(Path(args.repo_root).resolve(), args.max_seconds)
