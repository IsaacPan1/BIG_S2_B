# -*- coding: utf-8 -*-
"""
tail_diagnostic.py — read-only diagnostic for two tail-accuracy hypotheses.

PART 1: Is the recursive ceiling truncating high-rate val predictions?
PART 2: Honest leave-fold-out per-group bias correction recovery on OOF.

Reads existing artefacts only — does NOT train models, does NOT modify
predictions, does NOT write to reports/. All output to stdout.
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

# ─── Load artefacts ──────────────────────────────────────────────────────────
with open(os.path.join(REPO, "reports", "profile.json"), encoding="utf-8") as f:
    profile = json.load(f)
with open(os.path.join(REPO, "reports", "cv_plan.json"), encoding="utf-8") as f:
    cv_plan = json.load(f)
with open(os.path.join(REPO, "reports", "model_results.json"), encoding="utf-8") as f:
    mr = json.load(f)
with open(os.path.join(REPO, "reports", "features.json"), encoding="utf-8") as f:
    feat_meta = json.load(f)

train_df = pd.read_parquet(os.path.join(REPO, "data", "features_train.parquet"))
val_df   = pd.read_parquet(os.path.join(REPO, "data", "features_val.parquet"))
val_preds = pd.read_csv(os.path.join(REPO, "reports", "predictions.csv"))
oof = pd.read_csv(os.path.join(REPO, "reports", "oof_predictions.csv"))

target_col = feat_meta["target_col"]      # rate_per_10000_ed_visits
group_cols = feat_meta["group_cols"]      # ['jurisdiction','overdose_category']
time_col   = feat_meta["time_col"]        # period_id

# ─── Scored filter (match modeler logic exactly) ─────────────────────────────
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
    tr_vals = set(train_df[c].dropna().unique().tolist())
    if sub_vals and sub_vals < tr_vals:
        scored_filters[c] = sub_vals


def scored_mask(df):
    m = np.ones(len(df), dtype=bool)
    for col, vals in scored_filters.items():
        m &= df[col].isin(vals).values
    return m


print("=" * 78)
print("TAIL-ACCURACY DIAGNOSTIC")
print("=" * 78)
print(f"target={target_col}  groups={group_cols}  time={time_col}")
print(f"scored_filters: {scored_filters}")

# =============================================================================
# PART 1 — Clip ceiling: is the tail being TRUNCATED or under-predicted?
# =============================================================================
print()
print("=" * 78)
print("PART 1 — clip ceiling: truncated vs genuinely low?")
print("=" * 78)

train_max = float(train_df[target_col].max())
ceiling   = train_max * 10.0   # this is _rf_ceiling at line 1026

print(f"\n[1] Ceiling spec (run_modeler.py:1026)")
print(f"    _rf_ceiling = train_df[target_col].max() * 10.0")
print(f"    multiplier  : 10.0x")
print(f"    of          : training target max (NOT per-group)")
print(f"    train_max   : {train_max:.4f}")
print(f"    ceiling     : {ceiling:.4f}")
print(f"    Applied at lines 1126-1129 (holdout) and 1189-1192 (val):")
print(f"      _cf = _sp > _rf_ceiling")
print(f"      _sp = np.clip(_sp, 0, _rf_ceiling)")

print(f"\n[2] Reported ceiling-hit counts (from model_results.json):")
lf = mr["lag_forecasting"]
print(f"    ceiling_hits_holdout : {lf['ceiling_hits_holdout']}")
print(f"    ceiling_hits_val     : {lf['ceiling_hits_val']}")
print(f"    method_used          : {lf['method_used']}")

vp = val_preds["predicted_target"].values
print(f"\n[3] Val prediction distribution (after clip)")
print(f"    n           : {len(vp)}")
print(f"    min         : {vp.min():.4f}")
print(f"    max         : {vp.max():.4f}")
print(f"    mean        : {vp.mean():.4f}")
print(f"    std         : {vp.std():.4f}")
within_1pct = (vp >= ceiling * 0.99).sum()
within_10pct = (vp >= ceiling * 0.90).sum()
within_50pct = (vp >= ceiling * 0.50).sum()
print(f"    rows within  1% of ceiling : {within_1pct}/{len(vp)}")
print(f"    rows within 10% of ceiling : {within_10pct}/{len(vp)}")
print(f"    rows within 50% of ceiling : {within_50pct}/{len(vp)}")
print(f"    max / ceiling             : {vp.max()/ceiling:.4f}  "
      f"({vp.max()/ceiling*100:.2f}% of ceiling)")

# How does the val max compare to training-history rates per group?
print(f"\n[4] 'High-rate' diagnostic")
print(f"    val truth is NaN (eval set) — cannot measure val-truth directly.")
print(f"    Proxy: identify groups whose RECENT-TRAINING rate was high; check "
      f"their val predictions.")
# recent training = last 7 periods (val_size from cv_plan)
period_rank = profile["period_rank_info"]["id_to_rank"]
train_df2 = train_df.copy()
train_df2["__rank__"] = train_df2[time_col].map(period_rank)
val_df2 = val_df.copy()
val_df2["__rank__"] = val_df2[time_col].map(period_rank)
max_rank = int(train_df2["__rank__"].max())
recent_train = train_df2[train_df2["__rank__"] > max_rank - 7].copy()
group_recent = (recent_train.groupby(group_cols)[target_col]
                .mean().reset_index()
                .rename(columns={target_col: "recent_mean"}))
group_recent_max = (recent_train.groupby(group_cols)[target_col]
                    .max().reset_index()
                    .rename(columns={target_col: "recent_max"}))
group_hist_max = (train_df.groupby(group_cols)[target_col]
                  .max().reset_index()
                  .rename(columns={target_col: "hist_max"}))
val_preds_aug = val_preds.merge(group_recent, on=group_cols, how="left")
val_preds_aug = val_preds_aug.merge(group_recent_max, on=group_cols, how="left")
val_preds_aug = val_preds_aug.merge(group_hist_max, on=group_cols, how="left")

# Top-20 highest-recent-mean groups: do their val preds approach hist_max?
print(f"\n    Top-20 highest recent-mean (last 7 training periods) groups —")
print(f"    showing per-group val predictions vs the group's historical max:")
print(f"    {'jurisdiction':12} {'category':18} {'recent_mean':>11} "
      f"{'recent_max':>10} {'hist_max':>9} {'val_pred_mean':>13} "
      f"{'val_pred_max':>12} {'pred/hist_max':>13}")
top20 = (group_recent
         .merge(group_recent_max, on=group_cols)
         .merge(group_hist_max, on=group_cols)
         .sort_values("recent_mean", ascending=False).head(20))
for _, r in top20.iterrows():
    g_pred = val_preds_aug[
        (val_preds_aug["jurisdiction"] == r["jurisdiction"])
        & (val_preds_aug["overdose_category"] == r["overdose_category"])
    ]["predicted_target"]
    ratio = (g_pred.max() / r["hist_max"]) if r["hist_max"] > 0 else float("nan")
    print(f"    {r['jurisdiction']:12} {r['overdose_category']:18} "
          f"{r['recent_mean']:11.3f} {r['recent_max']:10.3f} "
          f"{r['hist_max']:9.3f} {g_pred.mean():13.3f} "
          f"{g_pred.max():12.3f} {ratio:13.3f}")

# Predictions vs ceiling for highest predicted cells
print(f"\n[5] Top-20 HIGHEST val predictions — distance from ceiling:")
print(f"    {'jurisdiction':12} {'category':18} {'val_pred':>10} "
      f"{'hist_max':>9} {'pred/hist_max':>13} {'pred/ceil':>9}")
top20_pred = val_preds_aug.sort_values("predicted_target",
                                        ascending=False).head(20)
for _, r in top20_pred.iterrows():
    ratio_h = (r["predicted_target"] / r["hist_max"]
               if r["hist_max"] > 0 else float("nan"))
    ratio_c = r["predicted_target"] / ceiling
    print(f"    {r['jurisdiction']:12} {r['overdose_category']:18} "
          f"{r['predicted_target']:10.3f} {r['hist_max']:9.3f} "
          f"{ratio_h:13.3f} {ratio_c:9.5f}")

# Verdict for PART 1
n_at_ceil = (vp >= ceiling * 0.99).sum()
print(f"\n[6] PART 1 VERDICT")
print(f"    Rows AT clip ceiling (within 1%) : {n_at_ceil}")
print(f"    ceiling_hits_val (logged by run)  : {lf['ceiling_hits_val']}")
print(f"    val_max ({vp.max():.2f}) vs ceiling ({ceiling:.2f}) — "
      f"gap = {ceiling - vp.max():.2f}")
if n_at_ceil == 0 and lf["ceiling_hits_val"] == 0:
    print(f"    → Clip is NOT firing on val. The 10x-train-max ceiling is far above")
    print(f"      any predicted value. The tail is UNDER-PREDICTED (regression-to-")
    print(f"      mean / recursive degradation), NOT truncated. Tightening the clip")
    print(f"      would only restrict future runs; loosening it cannot recover the")
    print(f"      missing high-rate signal.")
else:
    print(f"    → Clip IS firing — investigate raw-vs-clipped further.")


# =============================================================================
# PART 2 — Honest leave-fold-out per-group bias correction recovery on OOF
# =============================================================================
print()
print("=" * 78)
print("PART 2 — honest leave-fold-out per-group bias correction (OOF)")
print("=" * 78)

# Reconstruct fold assignments from CV_PLAN (TimeSeriesExpanding, end-anchored)
T = int(train_df2["__rank__"].max()) + 1  # ranks are 0..T-1
n_splits  = int(cv_plan["cv"]["n_splits"])
valid_size = int(cv_plan["cv"]["valid_size"])
gap       = int(cv_plan["cv"].get("gap") or 0)
print(f"\n[1] CV reconstruction")
print(f"    T (train periods)   : {T}")
print(f"    n_splits            : {n_splits}")
print(f"    valid_size          : {valid_size}")
print(f"    gap                 : {gap}")

# Build per-period val-fold map (end-anchored expanding, reversed in CVEngine)
fold_per_period = {}    # period_rank -> fold_id (0..n_splits-1)
folds_built = []
for k in range(n_splits):
    valid_end   = T - k * valid_size
    valid_start = valid_end - valid_size
    train_end   = valid_start - gap
    if train_end <= 0 or valid_start <= 0:
        break
    folds_built.append((train_end, valid_start, valid_end))
folds_built.reverse()
for fold_id, (train_end, vs, ve) in enumerate(folds_built):
    for r in range(vs, ve):
        fold_per_period[r] = fold_id
    print(f"    fold {fold_id}: train ranks [0,{train_end}) "
          f"val ranks [{vs},{ve})  ({(ve-vs)} periods)")

# Attach fold to OOF
oof2 = oof.copy()
oof2["__rank__"] = oof2[time_col].map(period_rank)
oof2["fold"] = oof2["__rank__"].map(fold_per_period)
n_unfolded = oof2["fold"].isna().sum()
print(f"\n    OOF rows: {len(oof2)}  with fold assigned: {(oof2['fold'].notna()).sum()}"
      f"  unassigned: {n_unfolded}")
oof2 = oof2.dropna(subset=["fold"]).copy()
oof2["fold"] = oof2["fold"].astype(int)

# Join truth from train_df on (group_cols, time_col)
truth = train_df2[group_cols + [time_col, target_col]]
oof3 = oof2.merge(truth, on=group_cols + [time_col], how="left")
n_missing = oof3[target_col].isna().sum()
if n_missing:
    print(f"    WARNING: {n_missing} OOF rows missing truth after merge")
oof3 = oof3.dropna(subset=[target_col])
oof3["residual"] = oof3[target_col] - oof3["predicted_target"]
oof3["sc_mask"]  = scored_mask(oof3)

print(f"\n[2] Baseline (no correction) — all-category OOF MAE")
abs_err = (oof3[target_col] - oof3["predicted_target"]).abs()
print(f"    all-cat MAE         : {abs_err.mean():.4f}")
print(f"    scored MAE          : {abs_err[oof3['sc_mask']].mean():.4f}")
print(f"    (reported oof_mae   : {mr['oof_mae']:.4f} scored / "
      f"{mr['oof_mae_all_categories']:.4f} all-cat)")

# Honest LFO per-group bias: for each (group, fold) the correction is the mean
# residual over OTHER folds (a group's correction NEVER uses its own fold).
print(f"\n[3] Per-group honest leave-fold-out bias correction")
group_key = group_cols
oof3["group_key"] = list(zip(*[oof3[c] for c in group_key]))

# total residual sum & count per group, per fold
per_gf = oof3.groupby(["group_key", "fold"]).agg(
    res_sum=("residual", "sum"),
    n=("residual", "size"),
).reset_index()
per_g = per_gf.groupby("group_key").agg(
    res_sum_all=("res_sum", "sum"),
    n_all=("n", "sum"),
).reset_index()
per_gf = per_gf.merge(per_g, on="group_key")
# LFO mean = (total_sum - own_fold_sum) / (total_n - own_fold_n)
per_gf["res_sum_oth"] = per_gf["res_sum_all"] - per_gf["res_sum"]
per_gf["n_oth"]       = per_gf["n_all"] - per_gf["n"]
per_gf["lfo_mean"]    = np.where(per_gf["n_oth"] > 0,
                                  per_gf["res_sum_oth"] / per_gf["n_oth"],
                                  0.0)
# Map back to oof3
corr_map = per_gf.set_index(["group_key", "fold"])["lfo_mean"].to_dict()
oof3["lfo_corr"] = [corr_map.get((gk, f), 0.0)
                    for gk, f in zip(oof3["group_key"], oof3["fold"])]
oof3["pred_corrected"] = oof3["predicted_target"] + oof3["lfo_corr"]

abs_err_corr = (oof3[target_col] - oof3["pred_corrected"]).abs()
all_mae_corr = abs_err_corr.mean()
sc_mae_corr  = abs_err_corr[oof3["sc_mask"]].mean()
all_mae_base = abs_err.mean()
sc_mae_base  = abs_err[oof3["sc_mask"]].mean()

print(f"    OOF all-cat MAE  BEFORE : {all_mae_base:.4f}")
print(f"    OOF all-cat MAE  AFTER  : {all_mae_corr:.4f}  "
      f"(Δ={all_mae_base - all_mae_corr:+.4f}, "
      f"{(all_mae_base - all_mae_corr)/all_mae_base*100:+.2f}%)")
print(f"    OOF scored  MAE  BEFORE : {sc_mae_base:.4f}")
print(f"    OOF scored  MAE  AFTER  : {sc_mae_corr:.4f}  "
      f"(Δ={sc_mae_base - sc_mae_corr:+.4f}, "
      f"{(sc_mae_base - sc_mae_corr)/sc_mae_base*100:+.2f}%)")

# Sanity check: in-sample (biased) version to show the bug-version recovery
in_sample_corr = oof3.groupby("group_key")["residual"].transform("mean")
oof3["pred_corrected_INSAMPLE"] = oof3["predicted_target"] + in_sample_corr
abs_err_is = (oof3[target_col] - oof3["pred_corrected_INSAMPLE"]).abs()
print(f"\n    [sanity: BIASED in-sample group debias — for comparison only]")
print(f"    OOF all-cat MAE  IN-SAMPLE: {abs_err_is.mean():.4f}  "
      f"(Δ={all_mae_base - abs_err_is.mean():+.4f})")
print(f"    OOF scored  MAE  IN-SAMPLE: {abs_err_is[oof3['sc_mask']].mean():.4f}  "
      f"(Δ={sc_mae_base - abs_err_is[oof3['sc_mask']].mean():+.4f})")
print(f"    (Gap between honest-LFO and in-sample = optimism of the buggy version)")

print(f"\n[4] Recovery decomposition: top-decile high-rate groups vs the rest")
# Define group rate as mean training target per group
group_rate = train_df.groupby(group_cols)[target_col].mean()
group_rate.name = "g_rate"
oof3 = oof3.merge(group_rate.reset_index(), on=group_cols, how="left")
cut = group_rate.quantile(0.9)
print(f"    Top-decile cutoff (group mean training rate >= {cut:.3f})")

mask_hi = oof3["g_rate"] >= cut
mask_lo = ~mask_hi

def _block(mask, name):
    sub = oof3[mask]
    base = (sub[target_col] - sub["predicted_target"]).abs()
    corr = (sub[target_col] - sub["pred_corrected"]).abs()
    sc = sub["sc_mask"]
    print(f"    [{name}]  n_rows={len(sub):5d}  n_groups={sub['group_key'].nunique():3d}")
    print(f"           all-cat BEFORE={base.mean():.4f}  AFTER={corr.mean():.4f}  "
          f"Δ={base.mean() - corr.mean():+.4f}  "
          f"({(base.mean() - corr.mean())/base.mean()*100:+.2f}%)")
    if sc.any():
        b = base[sc].mean(); c = corr[sc].mean()
        print(f"           scored  BEFORE={b:.4f}  AFTER={c:.4f}  "
              f"Δ={b - c:+.4f}  ({(b-c)/b*100:+.2f}%)")
    return float(base.sum() - corr.sum())

red_hi = _block(mask_hi, "TOP DECILE")
red_lo = _block(mask_lo, "REMAINING ")
total_red = red_hi + red_lo
print(f"\n    Absolute-error reduction (sum) — high-rate share of recovery:")
print(f"      top decile     : {red_hi:8.3f}  ({(red_hi/total_red*100 if total_red else 0):+.1f}%)")
print(f"      remaining      : {red_lo:8.3f}  ({(red_lo/total_red*100 if total_red else 0):+.1f}%)")
print(f"      total          : {total_red:8.3f}")

# Apply ONLY to high-rate groups: what recovery do we get?
oof3["pred_hi_only"] = np.where(mask_hi, oof3["pred_corrected"],
                                 oof3["predicted_target"])
abs_err_hi_only = (oof3[target_col] - oof3["pred_hi_only"]).abs()
print(f"\n    If correction applied to TOP-DECILE only (rest unchanged):")
print(f"      all-cat MAE     : {abs_err_hi_only.mean():.4f}  "
      f"(Δ vs base {all_mae_base - abs_err_hi_only.mean():+.4f})")
print(f"      scored MAE      : {abs_err_hi_only[oof3['sc_mask']].mean():.4f}  "
      f"(Δ vs base {sc_mae_base - abs_err_hi_only[oof3['sc_mask']].mean():+.4f})")

print(f"\n[5] Estimation-stability check — rows-per-(group, OTHER-folds)")
# n_oth distribution per (group, fold) — how noisy is the LFO mean?
stats_n = per_gf["n_oth"].describe()
print(f"    per (group,fold), rows used to estimate the correction (n_oth):")
print(f"      min   = {stats_n['min']:.0f}")
print(f"      25%   = {stats_n['25%']:.0f}")
print(f"      50%   = {stats_n['50%']:.0f}")
print(f"      75%   = {stats_n['75%']:.0f}")
print(f"      max   = {stats_n['max']:.0f}")
print(f"      mean  = {stats_n['mean']:.1f}")
# residual std per group (proxy for noise of the estimate)
group_res_std = oof3.groupby("group_key")["residual"].std().fillna(0)
print(f"\n    Per-group residual std (proxy for noise of the offset estimate):")
print(f"      min={group_res_std.min():.3f}  25%={group_res_std.quantile(.25):.3f}  "
      f"50%={group_res_std.median():.3f}  75%={group_res_std.quantile(.75):.3f}  "
      f"max={group_res_std.max():.3f}")
# Standard error of LFO mean ≈ group_std / sqrt(n_oth) — assess if offset is
# distinguishable from noise.
per_gf2 = per_gf.merge(group_res_std.reset_index().rename(
    columns={"residual": "g_std"}), on="group_key")
per_gf2["se_lfo"] = per_gf2["g_std"] / np.sqrt(per_gf2["n_oth"].clip(lower=1))
per_gf2["t_stat"] = per_gf2["lfo_mean"] / per_gf2["se_lfo"].replace(0, np.nan)
print(f"\n    |t_stat| = |lfo_mean / SE| — how confidently distinguishable from 0:")
t_abs = per_gf2["t_stat"].abs().dropna()
print(f"      median |t|              = {t_abs.median():.2f}")
print(f"      share |t| > 2 (sig)    = {(t_abs > 2).mean()*100:.1f}%")
print(f"      share |t| < 1 (noise)  = {(t_abs < 1).mean()*100:.1f}%")

# Top high-rate groups — how big are their corrections vs their noise floor?
print(f"\n    Top-decile groups: per-group total mean residual (LFO would push by)")
hi_groups = group_rate[group_rate >= cut].index
hi_corr = (per_gf2[per_gf2["group_key"].isin([tuple(g) if isinstance(g, tuple) else (g,)
                                              for g in hi_groups])])
# safer reconstruction of group_key tuples
hi_keys = set()
for g in hi_groups:
    hi_keys.add(g if isinstance(g, tuple) else (g,))
hi_corr = per_gf2[per_gf2["group_key"].isin(hi_keys)]
print(f"      n (group,fold) cells in top decile : {len(hi_corr)}")
print(f"      |lfo_mean| stats          : mean={hi_corr['lfo_mean'].abs().mean():.3f}  "
      f"median={hi_corr['lfo_mean'].abs().median():.3f}  "
      f"max={hi_corr['lfo_mean'].abs().max():.3f}")
print(f"      se_lfo stats               : mean={hi_corr['se_lfo'].mean():.3f}  "
      f"median={hi_corr['se_lfo'].median():.3f}")
print(f"      median |t| (top decile)    : {hi_corr['t_stat'].abs().median():.2f}")

print()
print("=" * 78)
print("END DIAGNOSTIC — no files written.")
print("=" * 78)
