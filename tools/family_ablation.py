#!/usr/bin/env python3
"""
tools/family_ablation.py — Statistically sound leave-one-family-out ablation.

A family is flagged as NET-HARMFUL only when ALL of the following hold:
  (a) NOT structurally protected (see rule below)
  (b) Per-fold consistency: mean_fold_delta < -(effective_k * fold_delta_std),
      where effective_k = EFFECT_K * sqrt(log1p(n_testable_families)).
      Bonferroni scaling: with n=16 testable families, effective_k ~ 3.5;
      with n=4, ~ 2.5. Slower than full Bonferroni, but avoids impossibly
      high bars; meaningfully higher than a single-test threshold.
  (c) Scale floor: |mean_fold_delta| > REL_FRAC * full_strict_mae  (~2% of MAE).
      Ensures improvement is practically meaningful, not just noise on any scale.
  (d) Direction: mean_fold_delta < 0 (removing the family must reduce MAE).

PROTECTED SET (panel_forecasting / time_series only):
  Families encoding the TARGET variable's recent history are load-bearing and
  must never be dropped even if strict CV score rises without them. Detection
  is dataset-agnostic — three rules, no hardcoded column names:

    Rule A : any feature name contains target_col as a substring
             (group means, baselines referencing the target).
    Rule A2: any feature name starts with "target_"
             (target-derived slopes, percentiles).
    Rule B : family name contains an AR keyword {lag, roll, recent, slope,
             baseline} AND none of its feature names embed a known covariate
             column name — identifies pure target lag/rolling families.
             Covariate families (cov_lags, cov_rolls*) share the keywords
             but their features are prefixed by covariate names, so they
             fail Rule B and remain testable.

  Protected families are still RUN and REPORTED in JSON with their deltas,
  but flagged=False always.

EVAL SAFETY — ENABLE_ABLATION_RETUNE = False (default):
  When False, net_harmful_families is always [] in the JSON (diagnostic mode).
  Detected families go to net_harmful_families_detected only. The critic reads
  net_harmful_families and sees [] — no ablation-triggered retune fires.
  Set True only after multi-dataset validation of the new logic.

BUDGET GATE (unchanged from prior version):
  Times the pilot refit (= seed-42 full-model run, result reused).
  Skips if estimated total > --max-seconds.

Usage:
    python tools/family_ablation.py [--repo-root PATH] [--max-seconds N]
"""
from __future__ import annotations

import argparse
import json
import math
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

# ── Decision constants ─────────────────────────────────────────────────────────
# Eval safety gate — set True only after multi-dataset validation.
ENABLE_ABLATION_RETUNE = False

# Single-test effect-size bar (before Bonferroni scaling).
# mean_fold_delta / fold_delta_std must be < -EFFECT_K to be a candidate.
EFFECT_K = 2.0

# Scale-relative floor: |mean_fold_delta| must exceed this fraction of full MAE.
REL_FRAC = 0.02

# AR keywords for temporal-core detection (Rule B in _infer_protected_families).
_AR_KEYWORDS = frozenset({"lag", "roll", "recent", "slope", "baseline"})

DEFAULT_MAX_SECONDS = 240   # 4-minute hard cap


# ── Strict CV helper ───────────────────────────────────────────────────────────

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
) -> tuple[float, list[float]]:
    """One-seed strict purged CV. Returns (mean_mae, per_fold_maes)."""
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

    mean_mae = float(np.mean(fold_maes)) if fold_maes else float("nan")
    return mean_mae, fold_maes


# ── Protection classifier ──────────────────────────────────────────────────────

def _infer_protected_families(
    valid_families: dict[str, list[str]],
    target_col:     str,
    problem_type:   str,
    covariate_cols: list[str],
) -> dict[str, tuple[bool, str]]:
    """
    Return {family: (is_protected, reason)} for every valid family.

    Only panel_forecasting / time_series problems have protected families.
    See module-level docstring for Rules A, A2, B.
    """
    cov_set    = {c.lower() for c in covariate_cols}
    target_low = target_col.lower()
    result: dict[str, tuple[bool, str]] = {}

    for fam, cols in valid_families.items():
        fam_low = fam.lower()

        if problem_type not in ("panel_forecasting", "time_series"):
            result[fam] = (False, "non-temporal problem — no structural protection")
            continue

        # Rule A: any feature embeds the target column name
        if any(target_low in c.lower() for c in cols):
            result[fam] = (True, f"features embed target name '{target_col}'")
            continue

        # Rule A2: any feature starts with "target_" (e.g. target_slope_w6)
        if any(c.lower().startswith("target_") for c in cols):
            result[fam] = (True, "features have 'target_' prefix — target-derived")
            continue

        # Rule B: AR keyword in family name AND features are NOT covariate-derived
        matched_kw = [kw for kw in _AR_KEYWORDS if kw in fam_low]
        if matched_kw:
            cov_derived = any(
                any(cov in c.lower() for cov in cov_set)
                for c in cols
            )
            if not cov_derived:
                result[fam] = (
                    True,
                    f"AR keyword(s) {matched_kw} in family name; "
                    f"no covariate-derived features — target temporal core",
                )
                continue

        result[fam] = (False, "not identified as temporal AR core")

    return result


# ── Decision logic for one family ─────────────────────────────────────────────

def _decide_family(
    fold_deltas:    list[float],
    full_strict_mae: float,
    effective_k:    float,
    scale_margin:   float,
) -> tuple[bool, str, float | None]:
    """
    Evaluate whether a family is net-harmful given paired per-fold deltas.

    Returns (flagged, reason_str, effect_size_d).
    effect_size_d = mean_delta / fold_delta_std (or None if not computable).
    """
    n = len(fold_deltas)
    if n < 2:
        return False, "insufficient folds for per-fold significance test", None

    mean_delta = float(np.mean(fold_deltas))
    if np.isnan(mean_delta):
        return False, "NaN mean_fold_delta", None

    # Direction check
    if mean_delta >= 0:
        return False, f"direction fail: mean_fold_delta={mean_delta:+.5f} >= 0", None

    # Scale floor
    if abs(mean_delta) <= scale_margin:
        return (
            False,
            f"scale floor fail: |mean_fold_delta|={abs(mean_delta):.5f} "
            f"<= {scale_margin:.5f} (REL_FRAC * full_mae)",
            None,
        )

    # Effect-size test (per-fold consistency)
    fold_delta_std = float(np.std(fold_deltas, ddof=1))
    if fold_delta_std == 0.0:
        # All folds have the same delta — scale floor already passed, so flag it.
        return (
            True,
            f"fold_delta_std=0; scale floor passed (mean_delta={mean_delta:+.5f})",
            float("inf"),
        )

    d = mean_delta / fold_delta_std   # effect size (negative = improvement)
    if d >= -effective_k:
        return (
            False,
            f"effect-size fail: d={d:.3f} >= -{effective_k:.3f} "
            f"(EFFECT_K={EFFECT_K}, Bonferroni-scaled)",
            d,
        )

    return (
        True,
        f"NET-HARMFUL: mean_fold_delta={mean_delta:+.5f}, "
        f"d={d:.3f} < -{effective_k:.3f}, "
        f"|delta|={abs(mean_delta):.5f} > scale_margin={scale_margin:.5f}",
        d,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def run_family_ablation(
    repo_root:   Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> dict:
    reports  = repo_root / "reports"
    data_dir = repo_root / "data"
    out_path = reports / "family_ablation.json"

    print("=" * 68)
    print("FAMILY ABLATION START")
    print("=" * 68)

    # ── Load inputs ──────────────────────────────────────────────────────
    with open(reports / "features.json")      as f: feat_meta     = json.load(f)
    with open(reports / "model_results.json") as f: model_results = json.load(f)
    with open(reports / "profile.json")       as f: profile       = json.load(f)

    target_col     = feat_meta.get("target_col")    or profile.get("target_col")
    group_cols     = profile.get("group_cols")       or feat_meta.get("group_cols") or []
    time_col       = profile.get("time_col")         or feat_meta.get("time_col")
    problem_type   = profile.get("problem_type", "tabular_regression")
    covariate_cols = feat_meta.get("covariate_cols") or []

    families: dict[str, list[str]] = feat_meta.get("feature_families", {})

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

    _excl = set(group_cols) | {target_col}
    if time_col:
        _excl.add(time_col)
    all_feature_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in _excl
    ]

    # Validate: keep only families whose declared columns exist in the parquet
    valid_families: dict[str, list[str]] = {}
    for fam, cols in families.items():
        actual = [c for c in cols if c in all_feature_cols]
        if actual:
            valid_families[fam] = actual
        else:
            print(f"  [ablation] skip '{fam}': no valid columns in parquet")

    n_families = len(valid_families)
    print(f"target={target_col}  problem_type={problem_type}")
    print(f"train shape: {train_df.shape}  feature_cols: {len(all_feature_cols)}")
    print(f"valid families: {n_families}")

    if n_families == 0:
        result = {
            "skipped": True,
            "reason":  "no valid feature families",
            "families": {},
            "net_harmful_families": [],
            "net_harmful_families_detected": [],
            "retune_enabled": ENABLE_ABLATION_RETUNE,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("[ablation] no valid families — writing empty result")
        return result

    # ── Structural protection ────────────────────────────────────────────
    protection   = _infer_protected_families(
        valid_families, target_col, problem_type, covariate_cols
    )
    protected_fams = sorted(f for f, (p, _) in protection.items() if p)
    testable_fams  = sorted(f for f, (p, _) in protection.items() if not p)
    n_testable     = max(1, len(testable_fams))

    print(f"\nProtected families ({len(protected_fams)}): {protected_fams}")
    print(f"Testable families  ({len(testable_fams)}): {testable_fams}")

    # Bonferroni-adjusted effect-size bar
    # effective_k = EFFECT_K * sqrt(log1p(n_testable))
    # n=1→×0.83, n=4→×1.27, n=8→×1.47, n=16→×1.74, n=22→×1.77
    effective_k  = EFFECT_K * math.sqrt(math.log1p(n_testable))
    print(
        f"\nConstants: EFFECT_K={EFFECT_K}  REL_FRAC={REL_FRAC}"
        f"  n_testable={n_testable}"
        f"  effective_k={effective_k:.3f}  (Bonferroni: EFFECT_K*sqrt(log1p(N)))"
        f"  ENABLE_ABLATION_RETUNE={ENABLE_ABLATION_RETUNE}"
    )

    # ── BUDGET GATE + seed-42 full-model run (reused as pilot) ───────────
    print(f"\nBudget gate: timing pilot / seed-42 full refit ...")
    t_start = time.time()
    try:
        full_mean_42, full_fold_maes_42 = _strict_cv_one_seed(
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
            "net_harmful_families_detected": [], "retune_enabled": ENABLE_ABLATION_RETUNE,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # 2 full-model runs (pilot reused as seed-42) + 1 per family
    estimated_total = pilot_time * (2 + n_families) * 1.2
    print(
        f"  pilot_time={pilot_time:.1f}s  n_families={n_families}"
        f"  estimated_total={estimated_total:.0f}s  max_allowed={max_seconds:.0f}s"
    )
    if estimated_total > max_seconds:
        msg = (
            f"ablation skipped: insufficient budget "
            f"(estimated {estimated_total:.0f}s > {max_seconds:.0f}s)"
        )
        print(f"[ablation] {msg}")
        result = {
            "skipped": True, "reason": msg,
            "families": {}, "net_harmful_families": [],
            "net_harmful_families_detected": [], "retune_enabled": ENABLE_ABLATION_RETUNE,
            "pilot_time_s": float(pilot_time),
            "estimated_total_s": float(estimated_total),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # ── Seed-7 run for seed_std ──────────────────────────────────────────
    print(f"\nSeed-7 run for seed_std estimation ...")
    full_mean_7, _ = _strict_cv_one_seed(
        train_df, all_feature_cols, target_col, time_col, group_cols,
        problem_type, hparams, n_estimators, seed=7,
    )
    full_strict_mae = float(np.mean([full_mean_42, full_mean_7]))
    seed_std        = float(np.std([full_mean_42, full_mean_7]))
    scale_margin    = REL_FRAC * full_strict_mae

    print(
        f"  seed42={full_mean_42:.4f}  seed7={full_mean_7:.4f}"
        f"  full_strict_mae={full_strict_mae:.4f}"
        f"  seed_std={seed_std:.4f}  scale_margin={scale_margin:.4f}"
    )
    print(f"  full fold MAEs (seed42): {[round(m, 4) for m in full_fold_maes_42]}")

    # ── Leave-one-family-out ─────────────────────────────────────────────
    print(f"\nLeave-one-family-out ({n_families} families) ...")
    family_results: dict[str, dict] = {}
    t_deadline = t_start + max_seconds * 0.95

    # Protected families first (for reporting), then testable by descending feature count
    sorted_fams = (
        [(f, valid_families[f]) for f in protected_fams]
        + sorted(
            [(f, valid_families[f]) for f in testable_fams],
            key=lambda kv: -len(kv[1]),
        )
    )

    for fam, fam_cols in sorted_fams:
        is_protected, prot_reason = protection[fam]

        if time.time() > t_deadline and not is_protected:
            print("[ablation] time limit reached — skipping remaining testable families")
            break

        reduced = [c for c in all_feature_cols if c not in set(fam_cols)]
        if not reduced:
            family_results[fam] = {
                "n_features_dropped": len(fam_cols),
                "strict_mae_without": None,
                "fold_deltas": [],
                "mean_fold_delta": None,
                "fold_delta_std": None,
                "effect_size": None,
                "effective_k": float(effective_k),
                "scale_margin": float(scale_margin),
                "protected": bool(is_protected),
                "protected_reason": prot_reason,
                "flagged": False,
                "flag_reason": "dropping family leaves 0 features",
                "seed_std": float(seed_std),
            }
            print(f"  {fam:30s}  SKIP — would leave 0 features")
            continue

        t_fam = time.time()
        try:
            abl_mean, abl_fold_maes = _strict_cv_one_seed(
                train_df, reduced, target_col, time_col, group_cols,
                problem_type, hparams, n_estimators, seed=42,
            )
        except Exception as ex:
            print(f"  {fam:30s}  ERROR: {ex}")
            family_results[fam] = {
                "n_features_dropped": len(fam_cols),
                "strict_mae_without": None,
                "fold_deltas": [], "mean_fold_delta": None,
                "fold_delta_std": None, "effect_size": None,
                "effective_k": float(effective_k), "scale_margin": float(scale_margin),
                "protected": bool(is_protected), "protected_reason": prot_reason,
                "flagged": False, "flag_reason": f"refit error: {ex}",
                "seed_std": float(seed_std),
            }
            continue

        # Paired per-fold deltas vs full seed-42 run
        n_paired    = min(len(abl_fold_maes), len(full_fold_maes_42))
        fold_deltas = [
            abl_fold_maes[i] - full_fold_maes_42[i]
            for i in range(n_paired)
        ]
        mean_delta = float(np.mean(fold_deltas)) if fold_deltas else float("nan")
        fold_delta_std_v = (
            float(np.std(fold_deltas, ddof=1)) if len(fold_deltas) > 1 else None
        )
        elapsed = time.time() - t_fam

        if is_protected:
            # Run + report, but never flag
            d_str = ""
            if fold_delta_std_v and fold_delta_std_v > 0:
                d_val = mean_delta / fold_delta_std_v
                d_str = f"  d={d_val:.2f}"
            print(
                f"  {fam:30s}  drop={len(fam_cols):3d}"
                f"  mae_without={abl_mean:.4f}  mean_delta={mean_delta:+.5f}"
                f"{d_str}  [PROTECTED: {prot_reason}]  ({elapsed:.0f}s)"
            )
            family_results[fam] = {
                "n_features_dropped": len(fam_cols),
                "strict_mae_without": float(abl_mean),
                "fold_deltas": [float(d) for d in fold_deltas],
                "mean_fold_delta": float(mean_delta),
                "fold_delta_std": fold_delta_std_v,
                "effect_size": (
                    float(mean_delta / fold_delta_std_v)
                    if fold_delta_std_v and fold_delta_std_v > 0 else None
                ),
                "effective_k": float(effective_k),
                "scale_margin": float(scale_margin),
                "protected": True,
                "protected_reason": prot_reason,
                "flagged": False,
                "flag_reason": f"protected — {prot_reason}",
                "seed_std": float(seed_std),
            }
            continue

        # ── Testable family — apply full decision rule ───────────────────
        flagged, flag_reason, effect_size = _decide_family(
            fold_deltas, full_strict_mae, effective_k, scale_margin
        )

        d_str = ""
        if effect_size is not None and not math.isinf(effect_size):
            d_str = f"  d={effect_size:.2f}"
        elif effect_size is not None and math.isinf(effect_size):
            d_str = "  d=inf(std=0)"

        marker = "  *** NET-HARMFUL" if flagged else ""
        print(
            f"  {fam:30s}  drop={len(fam_cols):3d}"
            f"  mae_without={abl_mean:.4f}  mean_delta={mean_delta:+.5f}"
            f"{d_str}{marker}  ({elapsed:.0f}s)"
        )
        print(f"    reason: {flag_reason}")

        family_results[fam] = {
            "n_features_dropped": len(fam_cols),
            "strict_mae_without": float(abl_mean),
            "fold_deltas": [float(d) for d in fold_deltas],
            "mean_fold_delta": float(mean_delta),
            "fold_delta_std": fold_delta_std_v,
            "effect_size": (
                float(effect_size)
                if effect_size is not None and not math.isinf(effect_size)
                else None
            ),
            "effective_k": float(effective_k),
            "scale_margin": float(scale_margin),
            "protected": False,
            "protected_reason": prot_reason,
            "flagged": bool(flagged),
            "flag_reason": flag_reason,
            "seed_std": float(seed_std),
        }

    # ── Print decision table ─────────────────────────────────────────────
    print(
        f"\n{'DECISION TABLE':=<68}\n"
        f"  scale_margin={scale_margin:.4f}  effective_k={effective_k:.3f}"
        f"  ENABLE_ABLATION_RETUNE={ENABLE_ABLATION_RETUNE}\n"
        f"  {'family':<28}  {'drop':>4}  {'mae_wo':>8}  "
        f"{'mean_Δ':>9}  {'d':>6}  {'prot':>4}  {'flag':>4}"
    )
    print("-" * 68)
    for fam in sorted(family_results):
        r  = family_results[fam]
        mw = f"{r['strict_mae_without']:.4f}" if r["strict_mae_without"] is not None else "  N/A  "
        md = f"{r['mean_fold_delta']:+.5f}" if r["mean_fold_delta"] is not None else "    N/A  "
        ds = r["effect_size"]
        dv = f"{ds:+.2f}" if ds is not None else "   N/A"
        pr = "YES" if r["protected"] else " no"
        fl = "YES" if r["flagged"]   else " no"
        print(
            f"  {fam:<28}  {r['n_features_dropped']:>4}  {mw:>8}  "
            f"{md:>9}  {dv:>6}  {pr:>4}  {fl:>4}"
        )
    print("=" * 68)

    # ── Summarize ────────────────────────────────────────────────────────
    detected     = [f for f, r in family_results.items() if r.get("flagged")]
    # Eval safety: only expose to critic when gate is open
    net_harmful  = detected if ENABLE_ABLATION_RETUNE else []

    result = {
        "skipped":                    False,
        "full_strict_mae":            float(full_strict_mae),
        "seed_std":                   float(seed_std),
        "scale_margin":               float(scale_margin),
        "effective_k":                float(effective_k),
        "effect_k_base":              float(EFFECT_K),
        "rel_frac":                   float(REL_FRAC),
        "n_testable_families":        n_testable,
        "n_protected_families":       len(protected_fams),
        "protected_families":         protected_fams,
        "retune_enabled":             bool(ENABLE_ABLATION_RETUNE),
        "families":                   family_results,
        "net_harmful_families":       net_harmful,
        "net_harmful_families_detected": detected,
        "pilot_time_s":               float(pilot_time),
        "total_time_s":               float(time.time() - t_start),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[ablation] complete:")
    print(f"  protected families ({len(protected_fams)}): {protected_fams}")
    print(f"  net_harmful_families_detected: {detected}")
    print(
        f"  net_harmful_families (exposed to critic, "
        f"retune={ENABLE_ABLATION_RETUNE}): {net_harmful}"
    )
    if not ENABLE_ABLATION_RETUNE and detected:
        print(
            "  NOTE: ENABLE_ABLATION_RETUNE=False — no ablation-based retune "
            "will fire even though families were detected. "
            "Set ENABLE_ABLATION_RETUNE=True after multi-dataset validation."
        )
    print(f"  Written {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistically sound leave-one-family-out strict-CV ablation."
    )
    parser.add_argument("--repo-root",   default=".", help="Repo root directory")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help="Max wall-clock seconds (default 240)")
    args = parser.parse_args()
    run_family_ablation(Path(args.repo_root).resolve(), args.max_seconds)
