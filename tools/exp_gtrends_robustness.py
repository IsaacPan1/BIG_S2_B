#!/usr/bin/env python3
"""
Experiment: Do high-KS Google Trends features help OUT-OF-REGIME?

Evaluates 4 feature-set arms on both single-cutoff and strict 4-fold
purged+embargoed CV (logic imported from tools/validate.py).
Decision is made on STRICT CV only.

Arms:
  FULL                 all current features (baseline)
  DROP_GTRENDS_LEVELS  drop raw levels + shift_aware derivatives; keep ratios
  DROP_GTRENDS_ALL     drop every gtrends-derived feature including ratios
  KEEP_RATIOS_ONLY     keep ONLY gtrends ratio (_div_) features; drop rest

Outputs:
  reports/exp_gtrends_robustness.json
  Printed table + verdict
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
REPORTS   = REPO_ROOT / "reports"

# -- Tunable constants ------------------------------------------------------
SEEDS           = [0, 42, 123]
N_STRICT_FOLDS  = 4
EMBARGO_PERIODS = 2
HIGH_KS_BASES   = ["naloxone", "fentanyl", "opioid", "overdose", "methamphetamine"]

# -- Load artefacts ---------------------------------------------------------
print("=" * 65)
print("EXP_GTRENDS_ROBUSTNESS -- loading artefacts")
print("=" * 65)

profile      = json.loads((REPORTS / "profile.json").read_text())
model_res    = json.loads((REPORTS / "model_results.json").read_text())

target_col  = profile["target_col"]          # "rate_per_10000_ed_visits"
group_cols  = profile.get("group_cols", [])  # ["jurisdiction", "overdose_category"]

lgb_params   = model_res["families"]["lightgbm"]["best_params"]
n_estimators = int(model_res["families"]["lightgbm"]["n_estimators"])
print(f"LGB best_params:  {lgb_params}")
print(f"n_estimators:     {n_estimators}")

train_df = pd.read_parquet(DATA_DIR / "features_train.parquet")
print(f"train_df shape:   {train_df.shape}")

# Use period_id_ord (numeric, correctly ordered) for temporal splitting.
# profile['time_col'] = 'period_id' (hashed strings, NOT lex-sortable).
TIME_ORD = "period_id_ord"
if TIME_ORD not in train_df.columns:
    sys.exit(f"ERROR: '{TIME_ORD}' not found in features_train.parquet")

# -- Build baseline feature list --------------------------------------------
NON_FEATURES = set(group_cols) | {target_col, "period_id", TIME_ORD, "jurisdiction",
                                   "overdose_category", "state_doh_release"}
numeric_cols  = train_df.select_dtypes(include=[np.number]).columns.tolist()
all_feat_cols = [c for c in numeric_cols if c not in NON_FEATURES]
all_feat_set  = set(all_feat_cols)

print(f"Total numeric feature cols: {len(all_feat_cols)}")

# -- Classify gtrends columns -----------------------------------------------
all_gtrends = [c for c in all_feat_cols if "gtrends" in c]

ratio_cols = {c for c in all_gtrends if "_div_" in c}

level_cols = {
    f"gtrends_{b}" for b in HIGH_KS_BASES
    if f"gtrends_{b}" in all_feat_set
}

shift_suffixes = [
    "zscore", "zscore_roll4", "rank",
    "dev_from_jurisdiction_mean", "dev_from_overdose_category_mean",
    "x_period_id_norm",
]
shift_aware_cols = {
    f"gtrends_{b}_{s}"
    for b in HIGH_KS_BASES
    for s in shift_suffixes
    if f"gtrends_{b}_{s}" in all_feat_set
}

other_gtrends = set(all_gtrends) - ratio_cols - level_cols - shift_aware_cols

print(f"\nGtrends column breakdown:")
print(f"  levels      : {len(level_cols)}   {sorted(level_cols)}")
print(f"  shift_aware : {len(shift_aware_cols)}")
print(f"  ratios      : {len(ratio_cols)}")
print(f"  other       : {len(other_gtrends)}  "
      f"(lags, rolls, group_stats, centered, entropy, ...)")
print(f"  TOTAL gtrends: {len(all_gtrends)}")

# -- Define arms ------------------------------------------------------------
# ARM 1: FULL — all features
arm_full = all_feat_cols

# ARM 2: DROP_GTRENDS_LEVELS — drop raw levels + shift_aware; keep ratios & other
drop_lvl = level_cols | shift_aware_cols
arm_drop_levels = [c for c in all_feat_cols if c not in drop_lvl]

# ARM 3: DROP_GTRENDS_ALL — drop every gtrends feature
drop_all = set(all_gtrends)
arm_drop_all = [c for c in all_feat_cols if c not in drop_all]

# ARM 4: KEEP_RATIOS_ONLY — from gtrends, keep only ratios; drop everything else
drop_except_ratios = set(all_gtrends) - ratio_cols
arm_ratios_only = [c for c in all_feat_cols if c not in drop_except_ratios]

ARMS: dict[str, list[str]] = {
    "FULL":                arm_full,
    "DROP_GTRENDS_LEVELS": arm_drop_levels,
    "DROP_GTRENDS_ALL":    arm_drop_all,
    "KEEP_RATIOS_ONLY":    arm_ratios_only,
}

print("\nArm sizes:")
for name, cols in ARMS.items():
    print(f"  {name:<25} {len(cols)} features")

# -- Splitters --------------------------------------------------------------

def panel_strict_splits(
    df: pd.DataFrame,
    n_folds: int = N_STRICT_FOLDS,
    embargo: int = EMBARGO_PERIODS,
) -> list[tuple[pd.Index, pd.Index]]:
    """Purged walk-forward CV on period_id_ord (mirrors validate.py logic)."""
    all_periods = sorted(df[TIME_ORD].unique())
    total       = len(all_periods)
    min_train   = max(1, total // (n_folds + 1))
    fold_step   = max(1, (total - min_train) // n_folds)

    splits = []
    for k in range(n_folds):
        vs = min_train + k * fold_step
        ve = min(vs + fold_step, total)
        if vs >= total:
            break
        val_periods   = set(all_periods[vs:ve])
        train_end     = max(0, vs - embargo)
        train_periods = set(all_periods[:train_end])
        tr_idx = df.index[df[TIME_ORD].isin(train_periods)]
        vl_idx = df.index[df[TIME_ORD].isin(val_periods)]
        if len(tr_idx) == 0 or len(vl_idx) == 0:
            continue
        splits.append((tr_idx, vl_idx))
    return splits


def single_cutoff_split(
    df: pd.DataFrame,
    pct: float = 0.80,
    embargo: int = EMBARGO_PERIODS,
) -> list[tuple[pd.Index, pd.Index]]:
    """Single temporal 80/20 split with embargo (naive OOF scheme)."""
    all_periods = sorted(df[TIME_ORD].unique())
    n           = len(all_periods)
    cut         = int(n * pct)
    train_end   = max(0, cut - embargo)
    train_periods = set(all_periods[:train_end])
    val_periods   = set(all_periods[cut:])
    tr_idx = df.index[df[TIME_ORD].isin(train_periods)]
    vl_idx = df.index[df[TIME_ORD].isin(val_periods)]
    if len(tr_idx) == 0 or len(vl_idx) == 0:
        return []
    return [(tr_idx, vl_idx)]


# -- Model fit (matches validate.py _fit_one_fold) --------------------------

def fit_one_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    seed: int,
) -> np.ndarray:
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
        **lgb_params,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
    return np.clip(model.predict(X_vl), 0, None)


def eval_arm_splits(
    feature_cols: list[str],
    splits: list[tuple[pd.Index, pd.Index]],
    seed: int,
) -> list[float]:
    fold_maes = []
    for tr_idx, vl_idx in splits:
        fold_tr = train_df.loc[tr_idx]
        fold_vl = train_df.loc[vl_idx]
        fill  = fold_tr[feature_cols].median()
        X_tr  = fold_tr[feature_cols].fillna(fill).values.astype(np.float32)
        y_tr  = fold_tr[target_col].values.astype(np.float32)
        X_vl  = fold_vl[feature_cols].fillna(fill).values.astype(np.float32)
        y_vl  = fold_vl[target_col].values.astype(np.float32)
        preds = fit_one_fold(X_tr, y_tr, X_vl, seed)
        fold_maes.append(float(mean_absolute_error(y_vl, preds)))
    return fold_maes


# -- Pre-compute splits (same for all arms) ---------------------------------
strict_splits = panel_strict_splits(train_df)
sc_splits     = single_cutoff_split(train_df)

all_periods = sorted(train_df[TIME_ORD].unique())
print(f"\nTemporal split info:")
print(f"  n_train_periods:  {len(all_periods)}")
print(f"  strict folds:     {len(strict_splits)}")
print(f"  single-cutoff:    {len(sc_splits)} fold(s)")
if sc_splits:
    tr_ix, vl_ix = sc_splits[0]
    print(f"  sc train rows:    {len(tr_ix)}, val rows: {len(vl_ix)}")

# -- Run experiment ---------------------------------------------------------
print(f"\nRunning {len(ARMS)} arms x {len(SEEDS)} seeds x "
      f"({len(strict_splits)} strict + {len(sc_splits)} SC) folds ...")
print("-" * 65)

t0_exp = time.time()
raw_results: dict[str, dict] = {}

for arm_name, feat_cols in ARMS.items():
    strict_by_seed: list[float] = []
    sc_by_seed:     list[float] = []

    for seed in SEEDS:
        t0 = time.time()

        # Strict CV
        strict_fold_maes = eval_arm_splits(feat_cols, strict_splits, seed)
        strict_by_seed.append(float(np.mean(strict_fold_maes)))

        # Single-cutoff
        if sc_splits:
            sc_fold_maes = eval_arm_splits(feat_cols, sc_splits, seed)
            sc_by_seed.append(float(np.mean(sc_fold_maes)))
        else:
            sc_by_seed.append(float("nan"))

        elapsed = time.time() - t0
        print(f"  {arm_name:<25} seed={seed}  "
              f"strict={strict_by_seed[-1]:.4f}  "
              f"sc={sc_by_seed[-1]:.4f}  "
              f"({elapsed:.0f}s)")

    raw_results[arm_name] = {
        "strict_per_seed": strict_by_seed,
        "sc_per_seed":     sc_by_seed,
        "n_features":      len(feat_cols),
    }

print(f"\nTotal experiment time: {(time.time() - t0_exp)/60:.1f} min")

# -- Aggregate --------------------------------------------------------------
agg: dict[str, dict] = {}
for arm_name, r in raw_results.items():
    sv = [x for x in r["strict_per_seed"] if not np.isnan(x)]
    cv = [x for x in r["sc_per_seed"]     if not np.isnan(x)]
    agg[arm_name] = {
        "strict_cv_mae":      float(np.mean(sv)),
        "strict_cv_mae_std":  float(np.std(sv)),
        "single_cutoff_mae":  float(np.mean(cv)) if cv else float("nan"),
        "n_features":         r["n_features"],
        "strict_per_seed":    r["strict_per_seed"],
        "sc_per_seed":        r["sc_per_seed"],
    }

# Deltas vs FULL
full_strict = agg["FULL"]["strict_cv_mae"]
full_sc     = agg["FULL"]["single_cutoff_mae"]

for arm_name, arm in agg.items():
    arm["delta_strict_vs_full"] = round(arm["strict_cv_mae"] - full_strict, 5)
    arm["optimism"]             = round(arm["single_cutoff_mae"] - arm["strict_cv_mae"], 5)

# -- Verdict ----------------------------------------------------------------
best_arm       = min(agg, key=lambda k: agg[k]["strict_cv_mae"])
full_wins      = (best_arm == "FULL")
drop_lvl_wins  = (best_arm == "DROP_GTRENDS_LEVELS")
drop_all_wins  = (best_arm == "DROP_GTRENDS_ALL")

# Was optimism reduced by dropping levels?
optimism_full      = agg["FULL"]["optimism"]
optimism_drop_lvl  = agg["DROP_GTRENDS_LEVELS"]["optimism"]
levels_reduce_opt  = (optimism_drop_lvl < optimism_full)

if full_wins:
    verdict_text = (
        "FULL stays best on strict CV => gtrends features earn their place. "
        "The gap is genuine regime change, not feature overfit. "
        "Lever is elsewhere; do NOT add/interact more shift features."
    )
    recommendation = "keep_all"
else:
    verdict_text = (
        f"{best_arm} beats FULL on strict CV => "
        "those gtrends features are net-harmful out-of-regime; "
        "recommend removing them from tools/feature_engineering.py (separate follow-up)."
    )
    recommendation = f"remove_gtrends_per_{best_arm}"

optimism_note = (
    "Dropping levels REDUCES optimism gap => levels were fitting the training regime."
    if levels_reduce_opt
    else
    "Dropping levels does NOT reduce optimism gap => regime fitting from other features."
)

# -- Print results table ----------------------------------------------------
print()
print("=" * 65)
print("RESULTS TABLE  (decision on STRICT CV)")
print("=" * 65)
print(f"{'ARM':<25} {'n_feat':>6} {'strict_mae':>10} {'+/-std':>7} "
      f"{'sc_mae':>9} {'d_strict':>9} {'optimism':>9}")
print("-" * 65)
for arm_name, arm in agg.items():
    marker = " << BEST" if arm_name == best_arm else ""
    print(
        f"{arm_name:<25} {arm['n_features']:>6} "
        f"{arm['strict_cv_mae']:>10.4f} {arm['strict_cv_mae_std']:>6.4f} "
        f"{arm['single_cutoff_mae']:>9.4f} "
        f"{arm['delta_strict_vs_full']:>+8.4f} "
        f"{arm['optimism']:>+9.4f}"
        f"{marker}"
    )
print("-" * 65)
print(f"\nBest arm on strict CV: {best_arm}")
print(f"Optimism (sc - strict):  FULL={optimism_full:+.4f}  "
      f"DROP_GTRENDS_LEVELS={optimism_drop_lvl:+.4f}")
print(f"\n[OPTIMISM NOTE] {optimism_note}")
print(f"\n[VERDICT] {verdict_text}")

# -- Write JSON output ------------------------------------------------------
output = {
    "experiment":       "exp_gtrends_robustness",
    "seeds":            SEEDS,
    "n_strict_folds":   N_STRICT_FOLDS,
    "embargo_periods":  EMBARGO_PERIODS,
    "time_col_used":    TIME_ORD,
    "high_ks_bases":    HIGH_KS_BASES,
    "arms":             agg,
    "best_arm_strict":  best_arm,
    "full_wins":        full_wins,
    "levels_reduce_optimism": levels_reduce_opt,
    "optimism_full":    optimism_full,
    "optimism_drop_levels": optimism_drop_lvl,
    "recommendation":   recommendation,
    "verdict":          verdict_text,
    "optimism_note":    optimism_note,
    "gtrends_col_counts": {
        "levels":       len(level_cols),
        "shift_aware":  len(shift_aware_cols),
        "ratios":       len(ratio_cols),
        "other":        len(other_gtrends),
        "total":        len(all_gtrends),
    },
}

out_path = REPORTS / "exp_gtrends_robustness.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults written to {out_path}")
print("=" * 65)
