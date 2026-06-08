# -*- coding: utf-8 -*-
"""
weight_diagnostic.py — read-only test of target-magnitude sample weighting
on the WF holdout. Generic / data-driven detector + per-alpha tail/body table.

NO modification of training. NO write to reports/. Stdout only.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from adaptive_pipeline import build_pipeline, load_config  # noqa: E402
import catboost as cb                                       # noqa: E402
from sklearn.metrics import mean_absolute_error            # noqa: E402
from scipy.stats import skew as _skew                       # noqa: E402

# ─── Artefacts ───────────────────────────────────────────────────────────────
with open(os.path.join(REPO, "reports", "profile.json"), encoding="utf-8") as f:
    profile = json.load(f)
with open(os.path.join(REPO, "reports", "features.json"), encoding="utf-8") as f:
    feat_meta = json.load(f)
with open(os.path.join(REPO, "reports", "model_results.json"), encoding="utf-8") as f:
    mr = json.load(f)
config = load_config(os.path.join(REPO, "pipeline_config.json"))

train_df = pd.read_parquet(os.path.join(REPO, "data", "features_train.parquet"))

target_col = feat_meta["target_col"]
group_cols = feat_meta["group_cols"]
time_col   = feat_meta["time_col"]
adaptive_steps = config.get("adaptive_steps", [])
_te_step = next((s for s in adaptive_steps if s.get("name") == "target_encode"), {})

# ─── Scored filter (reproduce modeler logic) ─────────────────────────────────
sample_sub = pd.read_csv(os.path.join(REPO, "data", "sample_submission.csv"))
exclude = {target_col, "row_id", "id", "Id", "ID"}
if time_col:
    exclude.add(time_col)
candidates = []
for c in (group_cols or []):
    if c in sample_sub.columns and c in train_df.columns and c not in exclude:
        candidates.append(c)
for c in sample_sub.columns:
    if c not in exclude and c in train_df.columns and c not in candidates:
        candidates.append(c)
scored_filters = {}
for c in candidates:
    sub_vals = set(sample_sub[c].dropna().unique().tolist())
    tr_vals  = set(train_df[c].dropna().unique().tolist())
    if sub_vals and sub_vals < tr_vals:
        scored_filters[c] = sub_vals


def scored_mask(df):
    m = np.ones(len(df), dtype=bool)
    for col, vals in scored_filters.items():
        m &= df[col].isin(vals).values
    return m


# ─── Feature column set (mirror modeler's logic) ─────────────────────────────
_exclude_ids = set([time_col] if time_col else []) | {target_col, "adversarial_weights"}
all_feature_cols = [c for c in train_df.columns if c not in _exclude_ids]
cat_feature_cols = [c for c in _te_step.get("targets", []) if c in all_feature_cols]

ID_RATIO_MAX = 0.99
_drop_nonnumeric = [c for c in all_feature_cols
                    if not pd.api.types.is_numeric_dtype(train_df[c])
                    and c not in cat_feature_cols]
all_feature_cols = [c for c in all_feature_cols if c not in _drop_nonnumeric]
_drop_idlike = []
for c in all_feature_cols:
    if c in cat_feature_cols:
        continue
    _nu = int(train_df[c].nunique(dropna=True))
    if _nu == len(train_df) or _nu / len(train_df) > ID_RATIO_MAX:
        _drop_idlike.append(c)
all_feature_cols = [c for c in all_feature_cols if c not in _drop_idlike]

# Adversarial weights, if present (modeler multiplies sample_weight onto these)
adv_weights = None
if "adversarial_weights" in train_df.columns:
    adv_weights = train_df["adversarial_weights"].fillna(1.0).values.astype(float)

# ─── Transform (reproduce modeler's resolved transform) ──────────────────────
transform_mode = mr.get("target_transform", "none")


def fwd(y):
    y = np.asarray(y, float)
    if transform_mode == "log1p":
        return np.log1p(y)
    if transform_mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    return y


def inv(p):
    p = np.asarray(p, float)
    if transform_mode == "log1p":
        return np.expm1(np.clip(p, 0, None))
    if transform_mode == "sqrt":
        return np.square(np.clip(p, 0, None))
    return np.clip(p, 0, None)


# ─── WF probe split — same convention as run_modeler.py:798-811 ──────────────
all_periods = sorted(train_df[time_col].unique())
cutoff_idx  = int(len(all_periods) * 0.8)
cutoff_pid  = all_periods[cutoff_idx]
wf_tr_mask  = (train_df[time_col] < cutoff_pid).values
wf_va_mask  = (train_df[time_col] >= cutoff_pid).values

wf_tr = train_df[wf_tr_mask].copy()
wf_va = train_df[wf_va_mask].copy()
X_tr_df = wf_tr[all_feature_cols]
X_va_df = wf_va[all_feature_cols]
y_tr_raw = wf_tr[target_col].values.astype(float)
y_va_raw = wf_va[target_col].values.astype(float)
sc_mask_va = scored_mask(wf_va)
adv_tr = adv_weights[wf_tr.index.values] if adv_weights is not None else None

print("=" * 78)
print("TARGET-MAGNITUDE WEIGHTING DIAGNOSTIC")
print("=" * 78)
print(f"transform_mode={transform_mode}  target={target_col}  scored={scored_filters}")
print(f"WF train periods: {cutoff_idx}/{len(all_periods)}  "
      f"holdout periods: {len(all_periods)-cutoff_idx}")
print(f"WF train rows: {len(wf_tr)}  WF holdout rows: {len(wf_va)}  "
      f"scored holdout rows: {int(sc_mask_va.sum())}")

# =============================================================================
# DETECTOR — Gate 1: SKEW
# =============================================================================
print()
print("=" * 78)
print("GATE 1 — SKEW")
print("=" * 78)
skew_full = float(_skew(train_df[target_col].dropna()))
skew_train_wf = float(_skew(y_tr_raw))
SKEW_THRESH = 1.0
print(f"  target_skewness (full train)   : {skew_full:.3f}")
print(f"  target_skewness (WF-train only): {skew_train_wf:.3f}")
print(f"  threshold                       : > {SKEW_THRESH}")
gate1_pass = skew_full > SKEW_THRESH
print(f"  GATE 1: {'PASS' if gate1_pass else 'FAIL'}  "
      f"(skew {skew_full:.2f} {'>' if gate1_pass else '<='} {SKEW_THRESH})")

# =============================================================================
# DETECTOR — Gate 2: TAIL-SHRINKAGE (must train a baseline first)
# =============================================================================
print()
print("=" * 78)
print("GATE 2 — TAIL-SHRINKAGE on current WF-probe predictions")
print("=" * 78)

# Build the WF-probe pipeline ONCE (same fit for every alpha — Pipeline only
# uses y to fit the target encoder; encoder fit doesn't change across alphas
# because the underlying train rows are identical).
bp = build_pipeline(adaptive_steps, for_model="catboost")
bp.pipeline.fit(X_tr_df, fwd(y_tr_raw))
X_tr_t = bp.pipeline.transform(X_tr_df)
X_va_t = bp.pipeline.transform(X_va_df)
cf_idx = [X_tr_t.columns.get_loc(c) for c in cat_feature_cols
          if c in X_tr_t.columns]

# CatBoost hparams (mirror Step 9 / final_params; iterations = best_n_estimators
# from the prior run so all alphas are compared at the same complexity).
best_params = mr["best_params"]
best_n_iter = int(mr["n_estimators"])
base_cb_params = {
    "iterations":          best_n_iter,
    "learning_rate":       float(best_params.get("learning_rate", 0.05)),
    "depth":               int(best_params.get("depth", 6)),
    "l2_leaf_reg":         float(best_params.get("l2_leaf_reg", 3.0)),
    "loss_function":       "MAE",
    "eval_metric":         "MAE",
    "verbose":             False,
    "allow_writing_files": False,
    "cat_features":        cf_idx,
}


def train_and_predict(alpha: float, median_y: float) -> np.ndarray:
    """Train one CatBoost with sample_weight = adv * (1 + alpha * y / median).
    Returns raw-space predictions on the WF holdout."""
    w_mag = 1.0 + alpha * (y_tr_raw / median_y)
    sw = w_mag.copy()
    if adv_tr is not None:
        sw = adv_tr * sw
    m = cb.CatBoostRegressor(**{**base_cb_params, "random_seed": 42})
    m.fit(X_tr_t, fwd(y_tr_raw), sample_weight=sw, verbose=False)
    return np.clip(inv(m.predict(X_va_t)), 0, None)


# Baseline (alpha=0) for Gate 2
median_y_train = float(np.median(y_tr_raw))
print(f"  median(y_train) = {median_y_train:.4f}  "
      f"(weight scheme: w = 1 + alpha * y / median)")
print(f"  CatBoost hparams: lr={base_cb_params['learning_rate']:.4f}  "
      f"depth={base_cb_params['depth']}  "
      f"l2={base_cb_params['l2_leaf_reg']:.3f}  "
      f"iter={base_cb_params['iterations']}  "
      f"loss=MAE  transform={transform_mode}")
print("  Training baseline (alpha=0)...")
base_preds = train_and_predict(0.0, median_y_train)

# Per-decile signed bias on SCORED rows (decile by true target)
sc_y = y_va_raw[sc_mask_va]
sc_p = base_preds[sc_mask_va]
deciles_edges = np.quantile(sc_y, np.linspace(0, 1, 11))
# Bin index for each scored row
bins = np.digitize(sc_y, deciles_edges[1:-1], right=False)
print(f"\n  Per-decile signed bias (pred - truth) on WF-holdout SCORED rows:")
print(f"  {'decile':<7}{'n':>6}{'truth_range':>22}{'truth_mean':>12}"
      f"{'pred_mean':>11}{'signed_bias':>13}{'rel_bias%':>11}{'abs_err':>10}")
top_decile_bias = None
for d in range(10):
    msk = (bins == d)
    if not msk.any():
        continue
    yt = sc_y[msk]; yp = sc_p[msk]
    bias = float(np.mean(yp - yt))
    rel  = bias / max(yt.mean(), 1e-9) * 100
    abse = float(np.mean(np.abs(yp - yt)))
    tag  = "*TOP*" if d == 9 else ""
    print(f"  {d:<7}{len(yt):>6d}{f'[{yt.min():.2f},{yt.max():.2f}]':>22}"
          f"{yt.mean():>12.3f}{yp.mean():>11.3f}{bias:>+13.3f}"
          f"{rel:>+10.2f}%{abse:>10.3f}  {tag}")
    if d == 9:
        top_decile_bias = bias

print(f"\n  Top-decile signed bias: {top_decile_bias:+.4f}  "
      f"(rule: must be NEGATIVE and material — i.e. systematic under-prediction)")
gate2_pass = (top_decile_bias is not None) and (top_decile_bias < -0.5)
print(f"  GATE 2: {'PASS' if gate2_pass else 'FAIL'}  "
      f"(top-decile bias {top_decile_bias:+.3f}; "
      f"under-prediction iff <  -0.5)")

# =============================================================================
# DETECTOR — Gate 3: METRIC
# =============================================================================
print()
print("=" * 78)
print("GATE 3 — METRIC")
print("=" * 78)
metric_decision = mr.get("decision_metric", "?")
loss = mr.get("objective", "MAE")
print(f"  decision_metric : {metric_decision}")
print(f"  CatBoost loss   : {loss}")
gate3_pass = "mae" in metric_decision.lower() or "mae" in loss.lower()
print(f"  GATE 3: {'PASS' if gate3_pass else 'FAIL'}  "
      f"(MAE is raw-scale-sensitive; weighting reshapes raw-magnitude error)")

gates_all_pass = gate1_pass and gate2_pass and gate3_pass
print()
print(f"  ALL THREE GATES: {'PASS — proceed to simulation' if gates_all_pass else 'FAIL — simulation may be uninformative'}")

# =============================================================================
# SIMULATION — per-alpha tail vs body
# =============================================================================
print()
print("=" * 78)
print("SIMULATION — sample_weight = adv * (1 + alpha * y/median(y_train))")
print("=" * 78)

ALPHAS = [0.0, 0.5, 1.0, 2.0]
print(f"  alpha grid: {ALPHAS}  (alpha=0 reproduces current production)")
print(f"  Note: weight is multiplied ONTO adversarial_weights "
      f"(not replaced).")
print()

# Define tail and body on SCORED holdout rows via TRAIN-side cutoff to be
# leak-safe (decision rule does not depend on holdout truth distribution):
# top decile cutoff = 90th percentile of WF-train target.
cutoff90 = float(np.quantile(y_tr_raw, 0.90))
sc_tail_mask = sc_mask_va & (y_va_raw >= cutoff90)
sc_body_mask = sc_mask_va & (y_va_raw < cutoff90)
print(f"  Tail/body split (LEAK-SAFE — cutoff from WF-train 90th pctile)")
print(f"    cutoff (train P90)        : {cutoff90:.4f}")
print(f"    n_scored_tail (holdout)   : {int(sc_tail_mask.sum())}")
print(f"    n_scored_body (holdout)   : {int(sc_body_mask.sum())}")
print()

rows = []
for a in ALPHAS:
    p = train_and_predict(a, median_y_train) if a != 0.0 else base_preds
    sc_p_a = p[sc_mask_va]
    mae_overall = float(np.mean(np.abs(sc_y - sc_p_a)))
    mae_tail = (float(np.mean(np.abs(y_va_raw[sc_tail_mask] - p[sc_tail_mask])))
                if sc_tail_mask.any() else float("nan"))
    mae_body = (float(np.mean(np.abs(y_va_raw[sc_body_mask] - p[sc_body_mask])))
                if sc_body_mask.any() else float("nan"))
    tail_bias = (float(np.mean(p[sc_tail_mask] - y_va_raw[sc_tail_mask]))
                 if sc_tail_mask.any() else float("nan"))
    max_pred  = float(p.max())
    rows.append({
        "alpha":    a,
        "overall":  mae_overall,
        "tail":     mae_tail,
        "body":     mae_body,
        "tail_bias":tail_bias,
        "max_pred": max_pred,
    })

base = rows[0]
print(f"  {'alpha':<7}{'overall_MAE':>13}{'tail_MAE':>11}{'body_MAE':>11}"
      f"{'tail_bias':>12}{'max_pred':>11}  {'Δoverall':>11}{'Δtail':>10}{'Δbody':>10}")
for r in rows:
    d_ov = r["overall"] - base["overall"]
    d_ta = r["tail"]    - base["tail"]
    d_bo = r["body"]    - base["body"]
    print(f"  {r['alpha']:<7.2f}{r['overall']:>13.4f}{r['tail']:>11.4f}"
          f"{r['body']:>11.4f}{r['tail_bias']:>+12.4f}{r['max_pred']:>11.3f}"
          f"  {d_ov:>+11.4f}{d_ta:>+10.4f}{d_bo:>+10.4f}")
print()
print(f"  (Δ = relative to alpha=0 baseline.  More-negative tail_bias = "
      f"under-prediction. max_pred shows whether the tail extends.)")

# =============================================================================
# VERDICT
# =============================================================================
print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
BODY_TOL = 0.05   # tolerate ≤5% body-MAE degradation
best = None
for r in rows[1:]:
    tail_helped  = r["tail"]    < base["tail"]
    body_ok      = (r["body"] - base["body"]) / max(base["body"], 1e-9) <= BODY_TOL
    overall_ok   = r["overall"] <= base["overall"] * (1 + BODY_TOL)
    if tail_helped and body_ok and overall_ok:
        if best is None or r["tail"] < best["tail"]:
            best = r

if best:
    print(f"  Best alpha that helps the tail without wrecking the body "
          f"(body-MAE tol = {BODY_TOL*100:.0f}%):")
    print(f"    alpha={best['alpha']:.2f}  "
          f"tail_MAE {base['tail']:.4f} -> {best['tail']:.4f}  "
          f"(Δ={best['tail']-base['tail']:+.4f}, "
          f"{(best['tail']-base['tail'])/base['tail']*100:+.2f}%)")
    print(f"    body_MAE {base['body']:.4f} -> {best['body']:.4f}  "
          f"(Δ={best['body']-base['body']:+.4f}, "
          f"{(best['body']-base['body'])/base['body']*100:+.2f}%)")
    print(f"    overall  {base['overall']:.4f} -> {best['overall']:.4f}  "
          f"(Δ={best['overall']-base['overall']:+.4f})")
else:
    print(f"  No alpha helps the tail without degrading the body beyond "
          f"{BODY_TOL*100:.0f}%.")
    print(f"  Weighting trades tail for body (and overall MAE typically rises).")

print()
print("=" * 78)
print("END DIAGNOSTIC — no files written.")
print("=" * 78)
