#!/usr/bin/env python3
"""Dataset-agnostic panel feature engineering.

Reads reports/profile.json for schema discovery (group_cols, time_col,
target_col, covariate_cols). Adapts lag and rolling depths to the available
training history. Covers all signal families needed for panel forecasting:
AR lags, rolling stats, group baselines, recent-trend stats, seasonality,
covariate derivatives, and interaction features.

Target feature count: ~43-50 for a panel problem with 90+ training periods
and 2 group cols, 2 numeric + 2 binary covariates.

Outputs
-------
data/features_train.parquet
data/features_val.parquet
reports/features.json
reports/feature_engineer_was_here.txt
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO        = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO / "data"
REPORTS_DIR = REPO / "reports"

# ── Load profile ───────────────────────────────────────────────────────────────
with open(REPORTS_DIR / "profile.json") as f:
    profile = json.load(f)

group_cols  = profile["group_cols"]        # e.g. ["store_id", "product_id"]
time_col    = profile["time_col"]          # e.g. "week"
target_col  = profile["target_col"]       # e.g. "weekly_sales"
cov_cols    = profile.get("covariate_cols", [])
schema      = profile.get("schema", {})
train_files = profile.get("train_files", [])
val_files   = profile.get("val_files", [])

# Classify covariates by cardinality and dtype
binary_cols  = [c for c in cov_cols if schema.get(c, {}).get("n_unique", 999) == 2]
# Exclude string/text covariates from numeric processing (e.g. free-text fields)
_str_dtypes  = {"str", "object", "string", "category"}
numeric_cols = [c for c in cov_cols
                if c not in binary_cols
                and schema.get(c, {}).get("dtype", "float64") not in _str_dtypes]
text_cols    = [c for c in cov_cols
                if c not in binary_cols and c not in numeric_cols]
if text_cols:
    print(f"Text/string covariates (excluded from numeric processing): {text_cols}")

print(f"Target: {target_col}   Groups: {group_cols}   Time: {time_col}")
print(f"Numeric covariates: {numeric_cols}")
print(f"Binary  covariates: {binary_cols}")

# ── Load and merge source CSV files ───────────────────────────────────────────
def load_and_merge(fnames: list[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(DATA_DIR / fn) for fn in fnames if (DATA_DIR / fn).exists()]
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        on = [c for c in result.columns if c in df.columns]
        if on:
            result = result.merge(df, on=on, how="left")
    return result

train = load_and_merge(train_files)
val   = load_and_merge(val_files)

# ── Expand val with any missing group columns via cross-join ──────────────────
# e.g. val/covariates.csv is jurisdiction×period only; overdose_category is missing.
# We cross-join with all known values from train so the val frame covers every
# (jurisdiction, overdose_category, period_id) combination needed for submission.
_missing_group_cols = [g for g in group_cols if g not in val.columns] if not val.empty else []
if _missing_group_cols:
    print(f"Val is missing group col(s) {_missing_group_cols} — expanding via cross-join with train values")
    for mg in _missing_group_cols:
        known_vals = pd.DataFrame({mg: train[mg].unique()})
        val = val.merge(known_vals, how="cross")
    print(f"Val shape after cross-join expansion: {val.shape}")

# ── Adversarial validation helper ─────────────────────────────────────────────
def _run_adversarial_validation(
    train_rows: "pd.DataFrame",
    val_rows: "pd.DataFrame",
    numeric_cov_cols: list,
) -> tuple:
    """Detect multivariate shift via a binary train-vs-val LightGBM classifier.

    Returns (meta_dict, sample_weights_for_train | None).
    Weights are parallel to train_rows; None means no weighting applied.
    Silently degrades on any error so the main pipeline is never blocked.
    """
    meta: dict = {
        "activated": False,
        "skip_reason": None,
        "auc_train_vs_val": None,
        "weights_applied": False,
        "weight_range": None,
        "weight_mean": None,
        "weight_std": None,
        "top_shift_revealing_features": None,
        "n_train_rows_weighted": None,
    }
    try:
        n_tr = len(train_rows)
        n_vl = len(val_rows)

        if n_tr + n_vl < 500:
            meta["skip_reason"] = f"insufficient_data: combined={n_tr + n_vl} < 500"
            return meta, None
        if n_tr < 100:
            meta["skip_reason"] = f"insufficient_train: n_train={n_tr} < 100"
            return meta, None
        if n_vl < 100:
            meta["skip_reason"] = f"insufficient_val: n_val={n_vl} < 100"
            return meta, None

        cand = [
            c for c in numeric_cov_cols
            if c in train_rows.columns and c in val_rows.columns
            and pd.api.types.is_numeric_dtype(train_rows[c])
        ]
        if len(cand) < 3:
            meta["skip_reason"] = f"insufficient_numeric_cols: found {len(cand)} < 3"
            return meta, None

        print(f"Adversarial validation: ACTIVATED — n_train={n_tr}, n_val={n_vl}, "
              f"n_cov_cols={len(cand)}")

        import lightgbm as _lgb_av
        from sklearn.model_selection import StratifiedKFold as _SKF_av
        from sklearn.metrics import roc_auc_score as _roc_av

        _fill  = train_rows[cand].median()
        _X_adv = pd.concat(
            [train_rows[cand].fillna(_fill), val_rows[cand].fillna(_fill)],
            ignore_index=True,
        )
        _y_adv = np.concatenate(
            [np.ones(n_tr, dtype=np.int8), np.zeros(n_vl, dtype=np.int8)]
        )
        _clf_p = {
            "objective": "binary", "metric": "auc",
            "n_estimators": 100, "learning_rate": 0.05,
            "num_leaves": 31, "min_child_samples": 20,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
        }
        _oof_p  = np.zeros(n_tr + n_vl)
        _fi_acc = np.zeros(len(cand))

        for _, (_tri, _vai) in enumerate(
            _SKF_av(n_splits=5, shuffle=True, random_state=42).split(_X_adv, _y_adv)
        ):
            _clf = _lgb_av.LGBMClassifier(**_clf_p)
            _clf.fit(_X_adv.iloc[_tri], _y_adv[_tri],
                     callbacks=[_lgb_av.log_evaluation(-1)])
            _oof_p[_vai]  = _clf.predict_proba(_X_adv.iloc[_vai])[:, 1]
            _fi_acc       += _clf.feature_importances_

        auc = float(_roc_av(_y_adv, _oof_p))
        meta["activated"]        = True
        meta["auc_train_vs_val"] = auc
        _top = (
            pd.Series(_fi_acc / 5, index=cand)
            .nlargest(min(5, len(cand))).index.tolist()
        )
        meta["top_shift_revealing_features"] = _top
        print(f"  AUC={auc:.4f}  top shift features: {_top}")

        if auc < 0.55:
            meta["weights_applied"] = False
            meta["skip_reason"]     = "no_meaningful_shift_detected"
            print("  AUC < 0.55 — no meaningful shift; skipping sample weights")
            return meta, None

        _w = np.clip(1.0 - _oof_p[:n_tr], 0.1, 10.0)
        _w = _w / _w.mean()   # normalize so mean = 1.0

        meta["weights_applied"]       = True
        meta["weight_range"]          = [float(_w.min()), float(_w.max())]
        meta["weight_mean"]           = float(_w.mean())
        meta["weight_std"]            = float(_w.std())
        meta["n_train_rows_weighted"] = n_tr
        print(f"  Weights: min={_w.min():.3f} max={_w.max():.3f} "
              f"mean={_w.mean():.3f} std={_w.std():.3f}")
        return meta, _w

    except Exception as _e:
        meta["activated"]   = False
        meta["skip_reason"] = f"classifier_error: {str(_e)[:200]}"
        print(f"Adversarial validation failed: {_e} — continuing without weights")
        return meta, None


# ── CROSS-SECTIONAL PATH (no time_col) ────────────────────────────────────────
if time_col is None:
    print("No time_col detected — running cross-sectional feature engineering path.")

    import re
    from pathlib import Path as _Path

    _families: dict[str, list[str]] = {}
    def _reg(family: str, cols: list[str]) -> None:
        _families.setdefault(family, []).extend(cols)

    id_col = profile.get("id_col")
    # Determine which columns are truly categorical covariates
    # sex is binary categorical; image_filename is an ID/linkage column
    true_numeric_covs = [c for c in cov_cols
                         if schema.get(c, {}).get("dtype", "") in ("int64", "float64")]
    true_binary_covs  = [c for c in group_cols
                         if schema.get(c, {}).get("n_unique", 999) == 2
                         and schema.get(c, {}).get("dtype", "") not in ("int64", "float64")]
    image_link_col    = profile.get("image_data", {}).get("linkage_column")

    print(f"Numeric covariates : {true_numeric_covs}")
    print(f"Binary  covariates : {true_binary_covs}")
    print(f"Image linkage col  : {image_link_col}")

    def _build_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        out = df.copy()
        # Only register features in _families on the first (train) call
        def _rg(family: str, cols: list[str]) -> None:
            if is_train:
                _reg(family, cols)

        # ── 1. Binary encoding ────────────────────────────────────────────────
        for bc in true_binary_covs:
            col = f"{bc}_enc"
            uniq = sorted(df[bc].unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            out[col] = df[bc].map(mapping).astype(np.int8)
        _rg("group_encodings", [f"{bc}_enc" for bc in true_binary_covs])

        # ── 2. Group baselines (sex-level mean/std from train stats) ──────────
        if is_train:
            _build_features._sex_stats = (
                df.groupby("sex")[target_col].agg(["mean", "std"])
                if target_col in df.columns and "sex" in df.columns
                else None
            )
        sex_stats = getattr(_build_features, "_sex_stats", None)
        if sex_stats is not None and "sex" in out.columns:
            out["sex_mean_target"] = out["sex"].map(sex_stats["mean"])
            out["sex_std_target"]  = out["sex"].map(sex_stats["std"])
            _rg("group_baselines", ["sex_mean_target", "sex_std_target"])

        # ── 3. Polynomial / interaction features ──────────────────────────────
        # Squared terms
        for nc in true_numeric_covs:
            col = f"{nc}_sq"
            out[col] = out[nc] ** 2
        _rg("interactions", [f"{nc}_sq" for nc in true_numeric_covs])

        # Pairwise products of numeric covariates
        for i, nc1 in enumerate(true_numeric_covs):
            for nc2 in true_numeric_covs[i+1:]:
                col = f"{nc1}_x_{nc2}"
                out[col] = out[nc1] * out[nc2]
                _rg("interactions", [col])

        # Numeric × binary interactions
        for nc in true_numeric_covs:
            for bc in true_binary_covs:
                bc_enc = f"{bc}_enc"
                if bc_enc in out.columns:
                    col = f"{nc}_x_{bc_enc}"
                    out[col] = out[nc] * out[bc_enc]
                    _rg("interactions", [col])

        # ── 4. Age-derived features ───────────────────────────────────────────
        if "age" in out.columns:
            out["age_decade"]    = (out["age"] // 10).astype(np.int8)
            out["age_over_60"]   = (out["age"] >= 60).astype(np.int8)
            out["age_over_75"]   = (out["age"] >= 75).astype(np.int8)
            out["age_under_40"]  = (out["age"] < 40).astype(np.int8)
            _rg("time_derived", ["age_decade", "age_over_60", "age_over_75", "age_under_40"])

        # ── 5. Comorbidity-derived features ───────────────────────────────────
        if "comorbidity_score" in out.columns:
            out["comorbidity_high"]   = (out["comorbidity_score"] >= 7.0).astype(np.int8)
            out["comorbidity_low"]    = (out["comorbidity_score"] <= 3.0).astype(np.int8)
            out["comorbidity_cubed"]  = out["comorbidity_score"] ** 3
            out["comorbidity_sqrt"]   = np.sqrt(np.maximum(out["comorbidity_score"], 0))
            _rg("interactions", ["comorbidity_high", "comorbidity_low",
                                   "comorbidity_cubed", "comorbidity_sqrt"])

        # ── 6. Prior admissions features ──────────────────────────────────────
        if "prior_admissions" in out.columns:
            out["prior_admissions_sq"]     = out["prior_admissions"] ** 2
            out["prior_admissions_zero"]   = (out["prior_admissions"] == 0).astype(np.int8)
            out["prior_admissions_high"]   = (out["prior_admissions"] >= 4).astype(np.int8)
            _rg("interactions", ["prior_admissions_sq",
                                   "prior_admissions_zero", "prior_admissions_high"])

        # ── 7. Compound risk scores ───────────────────────────────────────────
        if "age" in out.columns and "comorbidity_score" in out.columns:
            out["risk_score"]            = out["age"] / 100.0 * out["comorbidity_score"]
            out["risk_score_sq"]         = out["risk_score"] ** 2
        if "prior_admissions" in out.columns and "comorbidity_score" in out.columns:
            out["burden_score"]          = out["prior_admissions"] * out["comorbidity_score"]
        if "age" in out.columns and "prior_admissions" in out.columns:
            out["age_x_prior"]           = out["age"] * out["prior_admissions"]
        _rg("interactions", ["risk_score", "risk_score_sq", "burden_score", "age_x_prior"])

        # ── 8. Log transforms ─────────────────────────────────────────────────
        for nc in true_numeric_covs:
            col = f"{nc}_log1p"
            out[col] = np.log1p(np.maximum(out[nc], 0))
        _rg("interactions", [f"{nc}_log1p" for nc in true_numeric_covs])

        # ── 9. Image summary features ─────────────────────────────────────────
        img_dir = _Path(REPO / profile.get("image_data", {}).get("directory", "data/images"))
        if image_link_col and image_link_col in out.columns and img_dir.exists():
            print("  Extracting image summary features (mean intensity, std, region stats)…")
            try:
                from PIL import Image as _PILImage
                import numpy as _np2

                def _img_features(fname: str) -> dict:
                    fpath = img_dir / fname
                    if not fpath.exists():
                        return {"img_mean": np.nan, "img_std": np.nan,
                                "img_center_mean": np.nan, "img_corner_mean": np.nan,
                                "img_contrast": np.nan, "img_bright_pct": np.nan}
                    arr = _np2.array(_PILImage.open(fpath).convert("L"), dtype=_np2.float32) / 255.0
                    h, w = arr.shape
                    ch, cw = h // 4, w // 4
                    center = arr[ch:h-ch, cw:w-cw]
                    corner = _np2.concatenate([arr[:ch, :cw].ravel(), arr[:ch, -cw:].ravel(),
                                               arr[-ch:, :cw].ravel(), arr[-ch:, -cw:].ravel()])
                    return {
                        "img_mean":        float(_np2.mean(arr)),
                        "img_std":         float(_np2.std(arr)),
                        "img_center_mean": float(_np2.mean(center)),
                        "img_corner_mean": float(_np2.mean(corner)),
                        "img_contrast":    float(_np2.mean(center) - _np2.mean(corner)),
                        "img_bright_pct":  float((_np2.mean(arr) > 0.5).astype(float)),
                    }

                img_feats = out[image_link_col].apply(_img_features)
                img_df = pd.DataFrame(img_feats.tolist(), index=out.index)
                for col in img_df.columns:
                    out[col] = img_df[col].values
                _rg("image_features", list(img_df.columns))
                print(f"  Image features added: {list(img_df.columns)}")
            except Exception as e:
                print(f"  Image feature extraction failed: {e} — skipping image features.")
        else:
            print("  Image directory not available or no linkage col — skipping image features.")

        # ── 10. Raw covariate pass-through ─────────────────────────────────────
        _rg("covariates", list(true_numeric_covs))

        # ── 11. Horizon indicator (0 for all cross-sectional rows) ─────────────
        out["horizon"] = np.int16(0)
        _rg("horizon", ["horizon"])

        return out

    train_feat = _build_features(train, is_train=True)
    val_feat   = _build_features(val,   is_train=False)
    print(f"features_train: {train_feat.shape}   features_val: {val_feat.shape}")

    # ── Assert: val target is absent (all NaN after merge) ────────────────────
    if target_col in val_feat.columns:
        val_feat[target_col] = np.nan

    # ── Adversarial validation (multivariate shift detection) ─────────────────
    _av_meta, _av_weights = _run_adversarial_validation(
        train_feat, val_feat, true_numeric_covs,
    )
    if _av_weights is not None:
        train_feat = train_feat.copy()
        train_feat["adversarial_weights"] = _av_weights
        print("adversarial_weights column added to features_train")

    train_feat.to_parquet(DATA_DIR / "features_train.parquet", index=False)
    val_feat.to_parquet(  DATA_DIR / "features_val.parquet",   index=False)
    print(f"features_train: {train_feat.shape}")
    print(f"features_val:   {val_feat.shape}")

    # ── Enumerate feature columns ──────────────────────────────────────────────
    id_and_target_set = {id_col, target_col, image_link_col, "adversarial_weights"} | set(group_cols)
    feature_cols = [c for c in train_feat.columns
                    if c not in id_and_target_set and c != target_col]
    print(f"Total feature columns: {len(feature_cols)}")

    # ── Write features.json ───────────────────────────────────────────────────
    features_meta = {
        "problem_type":           profile.get("problem_type"),
        "target_col":             target_col,
        "group_cols":             group_cols,
        "time_col":               time_col,
        "covariate_cols":         list(cov_cols),
        "feature_families":       _families,
        "feature_columns":        feature_cols,
        "total_features_planned": len(feature_cols),
        "lag_periods":            [],
        "rolling_windows":        [],
        "train_shape":            list(train_feat.shape),
        "val_shape":              list(val_feat.shape),
        "adversarial_validation": _av_meta,
        "notes": [
            "Cross-sectional tabular regression — no time/lag features.",
            "Group baselines (sex-level) computed from train rows only — no leakage.",
            "Image summary features extracted from 64x64 grayscale PNGs.",
            "Polynomial, interaction, risk-score, and log-transform features included.",
        ],
    }
    with open(REPORTS_DIR / "features.json", "w") as fh:
        json.dump(features_meta, fh, indent=2)
    print("reports/features.json written")

    # ── Marker file ───────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(REPORTS_DIR / "feature_engineer_was_here.txt", "w") as fh:
        fh.write(f"feature_engineer sub-agent executed at {ts}\n")
        fh.write(f"train shape: {list(train_feat.shape)}\n")
        fh.write(f"val shape:   {list(val_feat.shape)}\n")
        fh.write(f"total_features_planned: {len(feature_cols)}\n")
    print("reports/feature_engineer_was_here.txt written")

    print("\n" + "=" * 60)
    print(f"FEATURE ENGINEERING COMPLETE — {len(feature_cols)} features")
    print("=" * 60)
    for fam, cols in sorted(_families.items()):
        print(f"  {fam:22s}: {len(cols):3d}  {cols[:3]}{'...' if len(cols) > 3 else ''}")

    import sys; sys.exit(0)  # Done — skip the panel path below

# ── PANEL PATH (time_col present) — original logic continues below ─────────────

# ── Convert datetime/opaque time_col to numeric ordinal BEFORE sorting ────────
_time_col_numeric = f"{time_col}_ord"
_is_datetime_time = False
_is_opaque_string_time = False
try:
    _test = int(train[time_col].max())
    # Integer time column — no conversion needed; use it directly.
except (ValueError, TypeError):
    # Could be a datetime string or an opaque hash string.
    _epoch = pd.Timestamp("2000-01-01")
    try:
        _is_datetime_time = True
        train[_time_col_numeric] = ((pd.to_datetime(train[time_col]) - _epoch)
                                    .dt.total_seconds() / 3600).astype(int)
        val[_time_col_numeric]   = ((pd.to_datetime(val[time_col]) - _epoch)
                                    .dt.total_seconds() / 3600).astype(int)
        print(f"Datetime time_col detected — created numeric ordinal '{_time_col_numeric}'")
    except Exception:
        # Opaque string IDs (e.g. base64 hashes) — assign rank-ordered integers.
        _is_datetime_time      = False
        _is_opaque_string_time = True
        # Derive ordering from train data: sort all train period IDs, then append
        # any unseen val IDs.  This preserves whatever temporal ordering is
        # implicit in the training set (sorted alphabetically as a proxy).
        _train_ids = sorted(train[time_col].unique().tolist())
        _val_ids   = [v for v in sorted(val[time_col].unique().tolist())
                      if v not in set(_train_ids)]
        _all_ids   = _train_ids + _val_ids
        _id_to_ord = {v: i for i, v in enumerate(_all_ids)}
        train[_time_col_numeric] = train[time_col].map(_id_to_ord).astype(int)
        val[_time_col_numeric]   = val[time_col].map(_id_to_ord).astype(int)
        print(f"Opaque string time_col detected — created rank ordinal '{_time_col_numeric}' "
              f"({len(_train_ids)} train IDs, {len(_val_ids)} new val IDs)")

_tc = _time_col_numeric if (_is_datetime_time or _is_opaque_string_time) else time_col

# Sort using the (now-numeric) time column so lags are computed correctly.
train = train.sort_values(group_cols + [_tc]).reset_index(drop=True)
val   = val.sort_values(group_cols + [_tc]).reset_index(drop=True)
print(f"Train: {train.shape}   Val: {val.shape}")

# ── Detect time granularity ────────────────────────────────────────────────────
def _detect_granularity() -> str:
    """Return 'hourly', 'daily', 'weekly', or 'monthly' from median inter-obs step."""
    ref = train.sort_values(group_cols + [_tc]) if group_cols else train.sort_values(_tc)
    if group_cols:
        steps = ref.groupby(group_cols)[_tc].diff().dropna().abs()
    else:
        steps = ref[_tc].diff().dropna().abs()
    med = float(steps.median()) if len(steps) > 0 else 1.0
    if _is_datetime_time:
        # _tc units are hours since epoch
        if med < 2:    return "hourly"
        if med < 48:   return "daily"
        if med < 216:  return "weekly"
        return "monthly"
    # Integer time column (e.g. retail week numbers) — assume weekly cadence
    return "weekly"

_granularity = _detect_granularity()
print(f"Detected time granularity: {_granularity}")

train_time_max = int(train[_tc].max())
min_periods    = int(train.groupby(group_cols)[_tc].count().min()) if group_cols else len(train)
print(f"min_periods per group: {min_periods}   train_time_max: {train_time_max}")

# ── Adaptive depth parameters ──────────────────────────────────────────────────
# Lag_k requires at least (k + 4) training periods per group.
LAG_CORE    = [1, 2, 3, 4]
LAG_LONG    = [k for k in [8, 12, 26] if min_periods >= k + 4]
LAG_XLONG   = [52]                        if min_periods >= 60 else []
LAG_PERIODS = LAG_CORE + LAG_LONG + LAG_XLONG

ROLL_CORE    = [4, 8]
ROLL_LONG    = [w for w in [13, 26] if min_periods >= w + 4]
ROLL_WINDOWS = ROLL_CORE + ROLL_LONG

PERIOD = 52  # approximate weekly annual cycle

print(f"Lag periods:     {LAG_PERIODS}")
print(f"Rolling windows: {ROLL_WINDOWS}")

# ── Build combined train+val frame for shift-based features ────────────────────
val_copy = val.copy()
if target_col not in val_copy.columns:
    val_copy[target_col] = np.nan   # val_features.csv has no target — add placeholder
full = pd.concat([train, val_copy], ignore_index=True)

# Ensure the numeric ordinal column exists in the combined frame
if _is_datetime_time:
    _epoch = pd.Timestamp("2000-01-01")
    full[_time_col_numeric] = ((pd.to_datetime(full[time_col]) - _epoch)
                               .dt.total_seconds() / 3600).astype(int)
elif _is_opaque_string_time:
    # Re-apply the ordinal mapping to the combined frame
    full[_time_col_numeric] = full[time_col].map(_id_to_ord).astype(int)

# Sort by the numeric ordinal column so lags are computed correctly
full = full.sort_values(group_cols + [_tc]).reset_index(drop=True)

tr_mask = full[_tc] <= train_time_max
vl_mask = full[_tc] >  train_time_max

# Registry maps family name -> list of column names for features.json
_families: dict[str, list[str]] = {}

def _reg(family: str, cols: list[str]) -> None:
    _families.setdefault(family, []).extend(cols)

# ── 1. Group label encodings ─────────────────────────────────────────────────
for g in group_cols:
    col = f"{g}_enc"
    full[col] = full[g].astype("category").cat.codes.astype(np.int16)
_reg("group_encodings", [f"{g}_enc" for g in group_cols])

# ── 2. Trig seasonality (annual cycle) ───────────────────────────────────────
# Use numeric ordinal for arithmetic; for datetime columns use 24-hour modulus
# so PERIOD_H = 52*24 = 1248 hours covers an approximate annual cycle unit.
if _is_datetime_time:
    t = full[_tc]
    PERIOD_H = PERIOD * 24  # hours per "annual unit" (52 * 24 = 1248)
    full[f"{time_col}_sin"]  = np.sin(2 * np.pi * (t % PERIOD_H) / PERIOD_H)
    full[f"{time_col}_cos"]  = np.cos(2 * np.pi * (t % PERIOD_H) / PERIOD_H)
    full[f"{time_col}_sin2"] = np.sin(4 * np.pi * (t % PERIOD_H) / PERIOD_H)
    full[f"{time_col}_cos2"] = np.cos(4 * np.pi * (t % PERIOD_H) / PERIOD_H)
else:
    # For opaque string IDs _tc is the numeric ordinal; for true integer time_col use it directly.
    t = full[_tc]
    full[f"{time_col}_sin"]  = np.sin(2 * np.pi * (t % PERIOD) / PERIOD)
    full[f"{time_col}_cos"]  = np.cos(2 * np.pi * (t % PERIOD) / PERIOD)
    full[f"{time_col}_sin2"] = np.sin(4 * np.pi * (t % PERIOD) / PERIOD)
    full[f"{time_col}_cos2"] = np.cos(4 * np.pi * (t % PERIOD) / PERIOD)
_reg("seasonality", [f"{time_col}_sin", f"{time_col}_cos",
                     f"{time_col}_sin2", f"{time_col}_cos2"])

# ── 2b. Granularity-specific seasonality (adds to existing, does not replace) ──
if _granularity == "hourly" and _is_datetime_time:
    _dt = pd.to_datetime(full[time_col])
    _hod = _dt.dt.hour.astype(float)
    _dow_g = _dt.dt.dayofweek.astype(float)
    if "hour_sin" not in full.columns:
        full["hour_sin"]  = np.sin(2 * np.pi * _hod / 24)
        full["hour_cos"]  = np.cos(2 * np.pi * _hod / 24)
        full["hour_sin2"] = np.sin(4 * np.pi * _hod / 24)
        full["hour_cos2"] = np.cos(4 * np.pi * _hod / 24)
        _reg("seasonality", ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2"])
    if "dow_sin" not in full.columns:
        full["dow_sin"] = np.sin(2 * np.pi * _dow_g / 7)
        full["dow_cos"] = np.cos(2 * np.pi * _dow_g / 7)
        _reg("seasonality", ["dow_sin", "dow_cos"])
    print("Added hourly seasonality: hour_sin/cos/sin2/cos2 + dow_sin/cos")

    # Per-(group, hour_of_day) and per-(group, day_of_week) target baselines.
    # These encode the time-of-day pattern per entity from training data, giving
    # the model strong signal even when lag features are imputed for far-out val rows.
    _dt_train_aug = pd.to_datetime(train[time_col])
    _tr_aug = train.copy()
    _tr_aug["_hod"] = _dt_train_aug.dt.hour.values
    _tr_aug["_dow"] = _dt_train_aug.dt.dayofweek.values
    full["_hod"] = _dt.dt.hour.values
    full["_dow"] = _dt.dt.dayofweek.values

    for _g in group_cols:
        _hod_mc = f"{_g}_hour_mean_{target_col}"
        _hod_sc = f"{_g}_hour_std_{target_col}"
        _dow_mc = f"{_g}_dow_mean_{target_col}"
        _dow_sc = f"{_g}_dow_std_{target_col}"
        if _hod_mc not in full.columns:
            _hod_stats = (_tr_aug.groupby([_g, "_hod"])[target_col]
                         .agg(**{_hod_mc: "mean", _hod_sc: "std"})
                         .reset_index())
            full = full.merge(_hod_stats[[_g, "_hod", _hod_mc, _hod_sc]],
                              on=[_g, "_hod"], how="left")
            _reg("group_baselines", [_hod_mc, _hod_sc])
        if _dow_mc not in full.columns:
            _dow_stats = (_tr_aug.groupby([_g, "_dow"])[target_col]
                         .agg(**{_dow_mc: "mean", _dow_sc: "std"})
                         .reset_index())
            full = full.merge(_dow_stats[[_g, "_dow", _dow_mc, _dow_sc]],
                              on=[_g, "_dow"], how="left")
            _reg("group_baselines", [_dow_mc, _dow_sc])

    full.drop(columns=["_hod", "_dow"], inplace=True)
    print(f"Added hourly group baselines: (group×hour) and (group×dow) mean/std")

elif _granularity == "daily" and _is_datetime_time:
    _dt = pd.to_datetime(full[time_col])
    _dow_g = _dt.dt.dayofweek.astype(float)
    _moy = _dt.dt.month.astype(float)
    _woy = _dt.dt.isocalendar().week.astype(float).values
    if "dow_sin" not in full.columns:
        full["dow_sin"] = np.sin(2 * np.pi * _dow_g / 7)
        full["dow_cos"] = np.cos(2 * np.pi * _dow_g / 7)
        _reg("seasonality", ["dow_sin", "dow_cos"])
    if "month_sin" not in full.columns:
        full["month_sin"] = np.sin(2 * np.pi * _moy / 12)
        full["month_cos"] = np.cos(2 * np.pi * _moy / 12)
        _reg("seasonality", ["month_sin", "month_cos"])
    if "woy_sin" not in full.columns:
        full["woy_sin"] = np.sin(2 * np.pi * _woy / 52)
        full["woy_cos"] = np.cos(2 * np.pi * _woy / 52)
        _reg("seasonality", ["woy_sin", "woy_cos"])
    print("Added daily seasonality: dow + month + woy sin/cos")

elif _granularity == "monthly" and _is_datetime_time:
    _dt = pd.to_datetime(full[time_col])
    _moy = _dt.dt.month.astype(float)
    if "month_sin" not in full.columns:
        full["month_sin"] = np.sin(2 * np.pi * _moy / 12)
        full["month_cos"] = np.cos(2 * np.pi * _moy / 12)
        _reg("seasonality", ["month_sin", "month_cos"])
    if "quarter" not in full.columns:
        full["quarter"] = _dt.dt.quarter.astype(np.int8)
        _reg("seasonality", ["quarter"])
    print("Added monthly seasonality: month_sin/cos + quarter")

# ── 3. Time-derived features (week_of_year, quarter, month, linear trend) ─────
t_num = full[_tc]   # always numeric
if _is_datetime_time:
    PERIOD_H = PERIOD * 24
    full[f"{time_col}_of_cycle"] = t_num % PERIOD_H
    full[f"{time_col}_quarter"]  = (t_num % PERIOD_H) // (13 * 24)
    full[f"{time_col}_month"]    = (t_num % PERIOD_H) // (4 * 24)
else:
    full[f"{time_col}_of_cycle"] = t_num % PERIOD
    full[f"{time_col}_quarter"]  = (t_num % PERIOD) // 13
    full[f"{time_col}_month"]    = (t_num % PERIOD) // 4
full[f"{time_col}_trend"] = t_num                    # global linear trend
_reg("time_derived", [f"{time_col}_of_cycle", f"{time_col}_quarter",
                      f"{time_col}_month",    f"{time_col}_trend"])

# ── 4. Group baselines (train-only, no leakage) ───────────────────────────────
for g in group_cols:
    col = f"{g}_mean_{target_col}"
    full[col] = full[g].map(train.groupby(g)[target_col].mean())
    _reg("group_baselines", [col])

if len(group_cols) >= 2:
    pair_key = "_".join(group_cols)
    sp_mean = train.groupby(group_cols)[target_col].mean().rename(f"{pair_key}_mean")
    sp_std  = train.groupby(group_cols)[target_col].std().rename(f"{pair_key}_std")
    full = full.merge(sp_mean, on=group_cols, how="left")
    full = full.merge(sp_std,  on=group_cols, how="left")
    _reg("group_baselines", [f"{pair_key}_mean", f"{pair_key}_std"])

# ── 5. Recent group-pair stats (last 4 / 8 training periods) ──────────────────
if len(group_cols) >= 2:
    pair_key = "_".join(group_cols)
    for w in [4, 8]:
        col = f"{pair_key}_recent{w}_mean"
        recent_vals = train[train[_tc] >= train_time_max - w + 1]
        recent_mean = recent_vals.groupby(group_cols)[target_col].mean().rename(col)
        full = full.merge(recent_mean, on=group_cols, how="left")
        _reg("recent_stats", [col])

    ratio_col = f"{pair_key}_recent4_vs_hist_ratio"
    hist_mean = train.groupby(group_cols)[target_col].mean()
    r4_mean   = train[train[_tc] >= train_time_max - 3].groupby(group_cols)[target_col].mean()
    ratio     = (r4_mean / hist_mean.replace(0, np.nan)).rename(ratio_col)
    full = full.merge(ratio, on=group_cols, how="left")
    _reg("recent_stats", [ratio_col])

# ── 6. AR lag features ────────────────────────────────────────────────────────
grp_tgt = full.groupby(group_cols)[target_col]
for k in LAG_PERIODS:
    full[f"lag_{k}"] = grp_tgt.shift(k)
_reg("lags", [f"lag_{k}" for k in LAG_PERIODS])

# ── 7. Rolling mean / std ─────────────────────────────────────────────────────
for w in ROLL_WINDOWS:
    full[f"roll_mean_{w}"] = full.groupby(group_cols)[target_col].transform(
        lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    full[f"roll_std_{w}"]  = full.groupby(group_cols)[target_col].transform(
        lambda s: s.shift(1).rolling(w, min_periods=2).std())
_reg("rolling_mean", [f"roll_mean_{w}" for w in ROLL_WINDOWS])
_reg("rolling_std",  [f"roll_std_{w}"  for w in ROLL_WINDOWS])

# ── 8. Covariate lags, deltas, rolling means ──────────────────────────────────
for cov in numeric_cols:
    lag_col  = f"{cov}_lag1"
    dlt_col  = f"{cov}_change"
    roll_col = f"{cov}_roll_mean4"
    full[lag_col]  = full.groupby(group_cols)[cov].shift(1)
    full[dlt_col]  = full[cov] - full[lag_col]
    full[roll_col] = full.groupby(group_cols)[cov].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())
_reg("cov_lags",   [f"{cov}_lag1"       for cov in numeric_cols])
_reg("cov_deltas", [f"{cov}_change"     for cov in numeric_cols])
_reg("cov_rolls",  [f"{cov}_roll_mean4" for cov in numeric_cols])

# ── 9. Primary covariate deviation / ratio vs group baseline ──────────────────
# Captures "is this week's price higher or lower than the product's usual price?"
if numeric_cols and len(group_cols) >= 2:
    primary_cov  = numeric_cols[0]   # typically the price-like covariate
    ref_group    = group_cols[-1]    # e.g. product_id
    mean_col     = f"{ref_group}_mean_{primary_cov}"
    grp_cov_mean = train.groupby(ref_group)[primary_cov].mean()
    full[mean_col]                      = full[ref_group].map(grp_cov_mean)
    full[f"{primary_cov}_deviation"]    = full[primary_cov] - full[mean_col]
    full[f"{primary_cov}_ratio"]        = full[primary_cov] / (full[mean_col] + 1e-9)
    _reg("price_derived", [mean_col,
                           f"{primary_cov}_deviation",
                           f"{primary_cov}_ratio"])

# ── 10. Interaction features ──────────────────────────────────────────────────
# Numeric × binary
for nc in numeric_cols:
    for bc in binary_cols:
        col = f"{nc}_x_{bc}"
        full[col] = full[nc] * full[bc]
        _reg("interactions", [col])

# Binary × binary
for i, b1 in enumerate(binary_cols):
    for b2 in binary_cols[i+1:]:
        col = f"{b1}_x_{b2}"
        full[col] = full[b1] * full[b2]
        _reg("interactions", [col])

# ── 11. Horizon indicator ─────────────────────────────────────────────────────
full["horizon"] = np.where(vl_mask, full[_tc] - train_time_max, 0).astype(np.int16)
_reg("horizon", ["horizon"])

# ── 12. Raw covariates (pass-through) ─────────────────────────────────────────
_reg("covariates", list(cov_cols))

# ── 12b. Hourly domain supplement: critical lags + cyclical encodings ──────────
# For hourly panel data the canonical lags are 1h, 24h, 48h, 168h.
# The generic section above covers lag_1–4 already; add the missing long ones.
_hourly_extra_lags = [k for k in [24, 48, 168] if k not in LAG_PERIODS and min_periods >= k + 4]
if _hourly_extra_lags:
    print(f"Adding hourly domain lags: {_hourly_extra_lags}")
    grp_tgt2 = full.groupby(group_cols)[target_col]
    for k in _hourly_extra_lags:
        full[f"lag_{k}"] = grp_tgt2.shift(k)
    LAG_PERIODS = LAG_PERIODS + _hourly_extra_lags
    _reg("lags", [f"lag_{k}" for k in _hourly_extra_lags])

# Cyclical encoding of hour_of_day (24-hour cycle) — skipped if already added by 2b
if "hour_of_day" in full.columns and "hour_sin" not in full.columns:
    h = full["hour_of_day"]
    full["hour_sin"] = np.sin(2 * np.pi * h / 24)
    full["hour_cos"] = np.cos(2 * np.pi * h / 24)
    full["hour_sin2"] = np.sin(4 * np.pi * h / 24)
    full["hour_cos2"] = np.cos(4 * np.pi * h / 24)
    _reg("seasonality", ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2"])

# Cyclical encoding of day_of_week (7-day cycle) — skipped if already added by 2b
if "day_of_week" in full.columns and "dow_sin" not in full.columns:
    d = full["day_of_week"]
    full["dow_sin"] = np.sin(2 * np.pi * d / 7)
    full["dow_cos"] = np.cos(2 * np.pi * d / 7)
    _reg("seasonality", ["dow_sin", "dow_cos"])

# Temperature squared to capture U-shaped energy response
if "temperature" in full.columns:
    full["temperature_sq"] = full["temperature"] ** 2
    _reg("interactions", ["temperature_sq"])

# Region fixed effect std (complement to existing mean)
if len(group_cols) == 1:
    g0 = group_cols[0]
    std_col = f"{g0}_std_{target_col}"
    full[std_col] = full[g0].map(train.groupby(g0)[target_col].std())
    _reg("group_baselines", [std_col])

# ── 12c. Distribution-shift-aware features ────────────────────────────────────
def _add_shift_aware_features(
    df: pd.DataFrame,
    prof: dict,
    tr_mask_series: pd.Series,
) -> tuple[list[str], list[str]]:
    """Add robust features for numeric covariates whose KS statistic > 0.15.

    All statistics (mean, std, rank CDF, group means) are derived from training
    rows only — validation rows receive the same transform using training params,
    so there is no leakage.

    Returns
    -------
    added_cols : list[str]
        New column names actually written to df.
    added_descs : list[str]
        Human-readable description strings for features.json.
    """
    shift_list = prof.get("distribution_shifts", [])
    if not shift_list:
        print("No distribution_shifts in profile — skipping shift-aware features.")
        return [], []

    shifted_covs = [
        entry["column"]
        for entry in shift_list
        if entry.get("ks_statistic", 0.0) > 0.15
        and entry["column"] in numeric_cols
        and entry["column"] in df.columns
    ]
    if not shifted_covs:
        print("No numeric covariate with KS > 0.15 — skipping shift-aware features.")
        return [], []

    print(f"Shift-aware features for {len(shifted_covs)} covariate(s): {shifted_covs}")
    added_cols:  list[str] = []
    added_descs: list[str] = []
    tr_rows = df[tr_mask_series]
    new_cols_dict: dict[str, pd.Series] = {}  # batch all new columns to avoid fragmentation

    # Normalized time in [0, 1] across the combined (train+val) frame
    t_min   = float(df[_tc].min())
    t_max   = float(df[_tc].max())
    t_range = max(t_max - t_min, 1.0)
    time_norm = (df[_tc] - t_min) / t_range  # Series aligned with df.index

    for cov in shifted_covs:
        tr_vals = tr_rows[cov].dropna()
        if len(tr_vals) == 0:
            print(f"  WARNING: {cov} all-NaN in training — skipping its shift features")
            continue

        tr_mean   = float(tr_vals.mean())
        tr_std    = float(tr_vals.std())
        tr_median = float(tr_vals.median())
        cov_imp   = df[cov].fillna(tr_median)  # impute NaN with training median

        # 1. Z-score normalization (training mean/std)
        if np.isnan(tr_std) or tr_std < 1e-9:
            print(f"  WARNING: {cov} std ≈ 0 — skipping z-score features")
        else:
            zs_col = f"{cov}_zscore"
            new_cols_dict[zs_col] = (cov_imp - tr_mean) / tr_std
            added_cols.append(zs_col)
            added_descs.append(
                f"{zs_col}: {cov} z-score normalized to training mean/std")

            # 2. Rolling z-score (4-period window per group)
            roll4_mean = df.groupby(group_cols)[cov].transform(
                lambda s: s.rolling(4, min_periods=1).mean())
            roll4_std  = df.groupby(group_cols)[cov].transform(
                lambda s: s.rolling(4, min_periods=2).std())
            roll4_std  = roll4_std.fillna(tr_std).replace(0.0, tr_std)
            rz_col = f"{cov}_zscore_roll4"
            new_cols_dict[rz_col] = ((cov_imp - roll4_mean) / roll4_std).fillna(0.0)
            added_cols.append(rz_col)
            added_descs.append(
                f"{rz_col}: {cov} rolling 4-period z-score (short-term anomaly)")

        # 3. Rank transform — percentile in training CDF, normalized to [0, 1]
        sorted_tr = np.sort(tr_vals.values)
        n_tr      = len(sorted_tr)
        rk_col    = f"{cov}_rank"
        new_cols_dict[rk_col] = pd.Series(
            np.searchsorted(sorted_tr, cov_imp.values, side="right") / n_tr,
            index=df.index)
        added_cols.append(rk_col)
        added_descs.append(
            f"{rk_col}: {cov} percentile rank within training distribution [0,1]")

        # 4. Group-relative deviation (one feature per group column)
        for g in group_cols:
            grp_means = tr_rows.groupby(g)[cov].mean()
            dev_col   = f"{cov}_dev_from_{g}_mean"
            new_cols_dict[dev_col] = cov_imp - df[g].map(grp_means).fillna(tr_mean)
            added_cols.append(dev_col)
            added_descs.append(
                f"{dev_col}: {cov} deviation from {g}-level training mean")

        # 5. Covariate × normalized time interaction
        tx_col = f"{cov}_x_{time_col}_norm"
        new_cols_dict[tx_col] = cov_imp * time_norm
        added_cols.append(tx_col)
        added_descs.append(
            f"{tx_col}: {cov} × normalized time — captures time-varying shift effect")

    return added_cols, added_descs, pd.DataFrame(new_cols_dict, index=df.index)

_shift_cols, _shift_descs, _shift_df = _add_shift_aware_features(full, profile, tr_mask)
if _shift_cols:
    # Concat all shift-aware columns in one shot to avoid DataFrame fragmentation
    full = pd.concat([full, _shift_df], axis=1)
    # Refresh masks after concat (index is unchanged but reassignment is safe)
    tr_mask = full[_tc] <= train_time_max
    vl_mask = full[_tc] >  train_time_max
    _reg("shift_aware", _shift_descs)
    print(f"Shift-aware features added: {len(_shift_cols)}  "
          f"{_shift_cols[:4]}{'...' if len(_shift_cols) > 4 else ''}")

# ── 12d. Codebook date features ──────────────────────────────────────────────
_codebook_info = profile.get("time_codebook", {})
if _codebook_info.get("available") and time_col is not None:
    try:
        _cb_path = DATA_DIR / _codebook_info["path"]
        _raw_cb  = json.loads(_cb_path.read_text(encoding="utf-8"))

        # Normalize to {id: date_string}
        _direction = _codebook_info.get("direction_detected", "id_to_date")
        if _direction == "date_to_id":
            _id_to_date = {str(v): str(k) for k, v in _raw_cb.items()}
        else:
            _id_to_date = {str(k): str(v) for k, v in _raw_cb.items()}

        # Map all rows; gate on training unmapped rate
        _dates_mapped    = full[time_col].astype(str).map(_id_to_date)
        _tr_unmapped_pct = float(_dates_mapped[tr_mask].isna().mean())

        if _tr_unmapped_pct > 0.1:
            print(f"WARNING: {_tr_unmapped_pct:.1%} of training rows have unmapped "
                  f"codebook entries — skipping date features")
        else:
            _dt_series  = pd.to_datetime(_dates_mapped, errors="coerce")
            _month_raw  = _dt_series.dt.month  # NaN for unmapped rows

            _tr_months     = _month_raw[tr_mask].dropna()
            _median_month  = int(_tr_months.median()) if len(_tr_months) > 0 else 6
            _month_filled  = _month_raw.fillna(_median_month).astype(np.int8)

            full["month_of_year"]    = _month_filled
            full["quarter_of_year"]  = ((_month_filled.astype(int) - 1) // 3 + 1).astype(np.int8)
            full["is_quarter_start"] = _month_filled.isin([1, 4, 7, 10]).astype(np.int8)

            _reg("date_features", ["month_of_year", "quarter_of_year", "is_quarter_start"])
            print(f"Codebook date features added: month_of_year, quarter_of_year, "
                  f"is_quarter_start  (median_month={_median_month}, "
                  f"train_unmapped={_tr_unmapped_pct:.1%})")
    except Exception as _e:
        print(f"WARNING: Codebook date feature extraction failed: {_e} — skipping")

# ── 13. Fill val NaN lags/roll features ──────────────────────────────────────
print("Filling val NaN lag/roll features…")
last_known = (full.loc[full[_tc] == train_time_max, group_cols + [target_col]]
              .rename(columns={target_col: "_lk"})
              .set_index(group_cols))
full = full.join(last_known, on=group_cols)

# Smart lag imputation: fill val NaN lag_k with cycle-position-aware group mean.
#   hourly  → mean(target | group, hour=(h-k) % 24)
#   monthly → mean(target | group, month_of_year=(m-k) % 12)
#   weekly  → mean(target | group, week_of_year=(w-k) % 52)
#   daily   → mean(target | group, day_of_week=(d-k) % 7)
# Falls back to last_known when a cycle position cannot be derived.
_lk_arr = full["_lk"].values
_smart_lag_meta: dict = {
    "activated": False,
    "granularity": _granularity,
    "method": "last_known",
    "n_val_lag_cells_filled": 0,
    "fallback_to_last_known_count": 0,
}


def _cycle_impute_lags(cycle_pos_arr: np.ndarray, cycle_len: int, method: str) -> tuple:
    """Vectorised cycle-aware lag imputation.  Modifies `full` in-place.
    Returns (n_smart_filled, n_fallback_filled).
    """
    _tr_df = full.loc[tr_mask, group_cols + [target_col]].copy()
    _tr_df["_cp"] = (cycle_pos_arr[tr_mask.values] % cycle_len).astype(int)
    _gby = group_cols + ["_cp"]
    _stats = (_tr_df.groupby(_gby)[target_col]
              .mean()
              .reset_index()
              .rename(columns={target_col: "_cm"}))

    _ns, _nf = 0, 0
    for _k in LAG_PERIODS:
        _lc = f"lag_{_k}"
        if _lc not in full.columns:
            continue
        _nan_mask = vl_mask & full[_lc].isna()
        if not _nan_mask.any():
            continue

        _aux = full[group_cols].copy()
        _aux["_cp"] = (cycle_pos_arr.astype(int) - _k) % cycle_len
        _aux = _aux.reset_index().merge(_stats, on=_gby, how="left").set_index("index")
        _fill = _aux["_cm"]

        _smart = _nan_mask & _fill.notna()
        _lkfb  = _nan_mask & _fill.isna()
        if _smart.any():
            full.loc[_smart, _lc] = _fill[_smart]
            _ns += int(_smart.sum())
        if _lkfb.any():
            full.loc[_lkfb, _lc] = full.loc[_lkfb, "_lk"]
            _nf += int(_lkfb.sum())
        print(f"  {_lc}: filled {int(_nan_mask.sum())} NaN ({method})")
    return _ns, _nf


def _fill_rolls_last_known() -> None:
    for _w in ROLL_WINDOWS:
        _col = f"roll_mean_{_w}"
        _m = vl_mask & full[_col].isna()
        if _m.sum():
            full.loc[_m, _col] = full.loc[_m, "_lk"]
            print(f"  {_col}: filled {int(_m.sum())} NaN")
    for _col in [f"roll_std_{_w}" for _w in ROLL_WINDOWS]:
        _m = vl_mask & full[_col].isna()
        if _m.sum():
            full.loc[_m, _col] = 0.0


if _granularity == "hourly" and _is_datetime_time and len(group_cols) == 1:
    _g0 = group_cols[0]
    _tr_hod_v   = pd.to_datetime(train[time_col]).dt.hour.values
    _full_hod_v = pd.to_datetime(full[time_col]).dt.hour.values
    _full_grp_v = full[_g0].values
    _hod_lists: dict = {}
    for _gv, _h, _tv in zip(train[_g0].values, _tr_hod_v, train[target_col].values):
        _hk = (_gv, int(_h))
        _hod_lists.setdefault(_hk, []).append(_tv)
    _hod_mean_d: dict = {k: float(np.mean(v)) for k, v in _hod_lists.items()}

    for k in LAG_PERIODS:
        lag_col = f"lag_{k}"
        if lag_col not in full.columns:
            continue
        nan_mask_bool = (vl_mask & full[lag_col].isna()).values
        if not nan_mask_bool.any():
            continue
        nan_idx = np.where(nan_mask_bool)[0]
        fill_vals = np.array([
            _hod_mean_d.get(
                (_full_grp_v[i], int((_full_hod_v[i] - k) % 24)),
                float(_lk_arr[i]) if not np.isnan(_lk_arr[i]) else 0.0
            )
            for i in nan_idx
        ])
        full.iloc[nan_idx, full.columns.get_loc(lag_col)] = fill_vals
        print(f"  {lag_col}: filled {len(nan_idx)} NaN (hour-aware)")

    _fill_rolls_last_known()
    _smart_lag_meta.update({"activated": True, "method": "group_x_hour_of_day"})

elif _granularity == "monthly":
    _moy_arr: np.ndarray | None = None
    _moy_method = "last_known"
    if "month_of_year" in full.columns:
        # Codebook date features already computed — reuse them
        _moy_arr    = full["month_of_year"].values.astype(int)
        _moy_method = "group_x_month_of_year_codebook"
    elif _is_datetime_time:
        _moy_arr    = pd.to_datetime(full[time_col]).dt.month.values.astype(int)
        _moy_method = "group_x_month_of_year_datetime"

    if _moy_arr is not None:
        _ns, _nf = _cycle_impute_lags(_moy_arr, 12, "monthly")
        _smart_lag_meta.update({
            "activated": True,
            "method": _moy_method,
            "n_val_lag_cells_filled": _ns,
            "fallback_to_last_known_count": _nf,
        })
        print(f"Monthly smart lag imputation: {_ns} cycle fills, {_nf} last_known fallbacks")
    else:
        for _lc in [f"lag_{k}" for k in LAG_PERIODS]:
            _m = vl_mask & full[_lc].isna()
            if _m.sum():
                full.loc[_m, _lc] = full.loc[_m, "_lk"]
                print(f"  {_lc}: filled {int(_m.sum())} NaN (last_known fallback)")
    _fill_rolls_last_known()

elif _granularity == "weekly":
    _woy_arr: np.ndarray | None = None
    _woy_method = "last_known"
    if _is_datetime_time:
        _woy_arr    = (pd.to_datetime(full[time_col])
                       .dt.isocalendar().week.astype(int).values)
        _woy_method = "group_x_week_of_year_datetime"
    elif not _is_opaque_string_time:
        # Integer time_col treated as week number; modulo 52 gives intra-year position
        _woy_arr    = full[_tc].values.astype(int) % 52
        _woy_arr    = np.where(_woy_arr == 0, 52, _woy_arr)
        _woy_method = "group_x_week_of_year_modulo"

    if _woy_arr is not None:
        _ns, _nf = _cycle_impute_lags(_woy_arr, 52, "weekly")
        _smart_lag_meta.update({
            "activated": True,
            "method": _woy_method,
            "n_val_lag_cells_filled": _ns,
            "fallback_to_last_known_count": _nf,
        })
        print(f"Weekly smart lag imputation: {_ns} cycle fills, {_nf} last_known fallbacks")
    else:
        for _lc in [f"lag_{k}" for k in LAG_PERIODS]:
            _m = vl_mask & full[_lc].isna()
            if _m.sum():
                full.loc[_m, _lc] = full.loc[_m, "_lk"]
                print(f"  {_lc}: filled {int(_m.sum())} NaN (last_known fallback)")
    _fill_rolls_last_known()

elif _granularity == "daily" and _is_datetime_time:
    _dow_arr = pd.to_datetime(full[time_col]).dt.dayofweek.values.astype(int)
    _ns, _nf = _cycle_impute_lags(_dow_arr, 7, "daily")
    _smart_lag_meta.update({
        "activated": True,
        "method": "group_x_day_of_week",
        "n_val_lag_cells_filled": _ns,
        "fallback_to_last_known_count": _nf,
    })
    print(f"Daily smart lag imputation: {_ns} cycle fills, {_nf} last_known fallbacks")
    _fill_rolls_last_known()

else:
    fill_mean_cols = ([f"lag_{k}" for k in LAG_PERIODS]
                      + [f"roll_mean_{w}" for w in ROLL_WINDOWS])
    # LAG_PERIODS may have been extended by the hourly supplement
    for col in fill_mean_cols:
        mask = vl_mask & full[col].isna()
        if mask.sum():
            full.loc[mask, col] = full.loc[mask, "_lk"]
            print(f"  {col}: filled {int(mask.sum())} NaN")
    for col in [f"roll_std_{w}" for w in ROLL_WINDOWS]:
        mask = vl_mask & full[col].isna()
        if mask.sum():
            full.loc[mask, col] = 0.0

full.drop(columns=["_lk"], inplace=True)

# ── 13.5. Adversarial validation (multivariate shift detection + sample weighting) ──
_av_tr = full[tr_mask].reset_index(drop=True)
_av_vl = full[vl_mask].reset_index(drop=True)
_av_meta, _av_weights_arr = _run_adversarial_validation(_av_tr, _av_vl, numeric_cols)
if _av_weights_arr is not None:
    _w_full = np.ones(len(full))
    _w_full[np.where(tr_mask.values)[0]] = _av_weights_arr
    full["adversarial_weights"] = _w_full
    print("adversarial_weights column added to full dataframe (training rows only)")

# ── 14. Split and save parquet ────────────────────────────────────────────────
features_train = full[full[_tc] <= train_time_max].copy()
features_val   = full[full[_tc] >  train_time_max].copy()

assert features_val[target_col].isna().all(),    "Val target should be all NaN"
assert features_train[target_col].notna().all(), "Train target has unexpected NaN"

features_train.to_parquet(DATA_DIR / "features_train.parquet", index=False)
features_val.to_parquet(  DATA_DIR / "features_val.parquet",   index=False)
print(f"features_train: {features_train.shape}")
print(f"features_val:   {features_val.shape}")

# ── 15. Enumerate feature columns ─────────────────────────────────────────────
id_and_target = set(group_cols + [time_col, target_col, "adversarial_weights"])
if _is_datetime_time:
    id_and_target.add(_time_col_numeric)  # exclude the internal ordinal column
feature_cols  = [c for c in features_train.columns if c not in id_and_target]
print(f"Total feature columns: {len(feature_cols)}")

# ── 16. Write features.json ────────────────────────────────────────────────────
features_meta = {
    "problem_type":           profile.get("problem_type"),
    "target_col":             target_col,
    "group_cols":             group_cols,
    "time_col":               time_col,
    "covariate_cols":         list(cov_cols),
    "feature_families":       _families,
    "feature_columns":        feature_cols,
    "total_features_planned": len(feature_cols),
    "lag_periods":            LAG_PERIODS,
    "rolling_windows":        ROLL_WINDOWS,
    "smart_lag_imputation":   _smart_lag_meta,
    "train_shape":            list(features_train.shape),
    "val_shape":              list(features_val.shape),
    "adversarial_validation": _av_meta,
    "notes": [
        "Group baselines computed from train rows only — no leakage.",
        "Rolling stats use shift(1) before rolling to prevent target leakage.",
        "Val NaN lag/roll features filled with cycle-aware group mean (monthly/weekly/daily/hourly) or last-known fallback.",
        "Long lags/windows skipped if min_periods_per_group < threshold.",
        *(
            [f"Shift-aware features added for {len(_shift_cols)} covariate(s) with KS > 0.15: "
             "z-score, rolling z-score, rank, group deviation, time interaction."]
            if _shift_cols else []
        ),
        *(
            ["Codebook date features added: month_of_year, quarter_of_year, is_quarter_start "
             "(from time_codebook in profile.json)."]
            if "date_features" in _families else []
        ),
    ],
}
with open(REPORTS_DIR / "features.json", "w") as fh:
    json.dump(features_meta, fh, indent=2)
print("reports/features.json written")

# ── 17. Marker file ───────────────────────────────────────────────────────────
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with open(REPORTS_DIR / "feature_engineer_was_here.txt", "w") as fh:
    fh.write(f"feature_engineer sub-agent executed at {ts}\n")
    fh.write(f"train shape: {list(features_train.shape)}\n")
    fh.write(f"val shape:   {list(features_val.shape)}\n")
    fh.write(f"total_features_planned: {len(feature_cols)}\n")
print("reports/feature_engineer_was_here.txt written")

print("\n" + "=" * 60)
print(f"FEATURE ENGINEERING COMPLETE — {len(feature_cols)} features")
print("=" * 60)
for fam, cols in sorted(_families.items()):
    print(f"  {fam:22s}: {len(cols):3d}  {cols[:3]}{'...' if len(cols) > 3 else ''}")
