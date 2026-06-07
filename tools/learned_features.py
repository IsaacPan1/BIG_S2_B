# -*- coding: utf-8 -*-
"""
learned_features.py — fold-specific "learned" feature transformer.

This module encapsulates every feature whose computation REQUIRES fitting on a
training set: target encodings (group means/stds of the target), covariate
group statistics, shift-aware scalings (z-score, rank CDF, group deviation),
slope features derived from training history, and median-based imputation.

The rule the transformer enforces:

    All parameters are fit on the rows passed to .fit() only.
    .transform() applies those stored parameters to any rows (train or val).

This is what the modeler and validator use inside every CV fold to avoid
leakage from the validation set into the feature definitions.

Usage:
    xf = LearnedFeaturesTransformer(target_col, group_cols, time_col,
                                    numeric_cov_cols, profile=profile)
    xf.fit(train_fold)
    train_fold_t = xf.transform(train_fold)
    val_fold_t   = xf.transform(val_fold)

Conventions:
  - `target_col` may be NaN in the rows passed to .transform() (val rows).
  - .fit() ignores rows where the target is NaN.
  - .transform() copies its input and adds new columns; it never mutates in place.
  - The set of columns added is deterministic given the fitted params, so
    `pd.DataFrame(columns=…)` schemas line up between train and val.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class LearnedFeaturesTransformer:
    """Encapsulates target-encoded, group-statistic, scaling, and imputation
    features that need to be fit on a training fold and applied identically
    to train and val rows.

    Features added (when applicable to the dataset):

    Target encodings (Sections 2/4/5/12 of the prior feature engineer):
      - For each g in group_cols:        `{g}_mean_{target_col}`
      - If 1 group col:                  `{g}_std_{target_col}`
      - If >= 2 group cols:              `{pair_key}_mean`, `{pair_key}_std`
      - If >= 2 group cols (recent):     `{pair_key}_recent4_mean`, `{pair_key}_recent8_mean`,
                                         `{pair_key}_recent4_vs_hist_ratio`
      - Cross-sectional ("sex" col):     `sex_mean_target`, `sex_std_target`

    Slope features (Section 8e):
      - `target_slope_w{w}` for w in [6, 12] (panel only)

    Covariate group statistics (Section 8d, 9):
      - For each numeric cov × group col: `{cov}_{g}_hist_mean`,
        `{cov}_{g}_recent4_mean`, `{cov}_{g}_drift`
      - Primary cov vs reference group:   `{ref_group}_mean_{primary_cov}`,
        `{primary_cov}_deviation`, `{primary_cov}_ratio`

    Covariate centering (Section 8f):
      - `{cov}_minus_mean`

    Shift-aware features (Section 12c) — only for KS > 0.15 covariates:
      - `{cov}_zscore`, `{cov}_zscore_roll4`, `{cov}_rank`,
        `{cov}_dev_from_{g}_mean`, `{cov}_x_{time_col}_norm`

    Median imputation:
      - `transform()` fills any NaN in feature_cols (numeric, non-id) with the
        train-fit median for the column.

    `apply_median_impute=False` disables the last step, leaving NaNs untouched
    (useful when the modeler wants to control NaN handling itself).
    """

    def __init__(
        self,
        target_col: str,
        group_cols: list[str],
        time_col: Optional[str],
        numeric_cov_cols: list[str],
        binary_cov_cols: Optional[list[str]] = None,
        profile: Optional[dict] = None,
        ks_shift_threshold: float = 0.15,
        apply_median_impute: bool = True,
    ):
        self.target_col       = target_col
        self.group_cols       = list(group_cols or [])
        self.time_col         = time_col
        self.numeric_cov_cols = list(numeric_cov_cols or [])
        self.binary_cov_cols  = list(binary_cov_cols or [])
        self.profile          = profile or {}
        self.ks_shift_threshold = ks_shift_threshold
        self.apply_median_impute = apply_median_impute

        self._fitted = False
        self._params: dict = {}
        # Column names produced by transform() — populated during fit()
        self.feature_names_: list[str] = []

    # ────────────────────────────────────────────────────────────────────────
    # FIT
    # ────────────────────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "LearnedFeaturesTransformer":
        """Compute every learned parameter from train_df only."""
        target = self.target_col
        # Use only rows where the target is observed (val rows have NaN target)
        tr = train_df[train_df[target].notna()].copy() if target in train_df.columns else train_df.copy()

        p: dict = {}
        names: list[str] = []

        # ── 1. Single-group target mean/std ─────────────────────────────────
        for g in self.group_cols:
            if g in tr.columns and target in tr.columns:
                p[f"group_target_mean_{g}"] = tr.groupby(g)[target].mean()
                names.append(f"{g}_mean_{target}")
        if len(self.group_cols) == 1:
            g = self.group_cols[0]
            if g in tr.columns and target in tr.columns:
                p[f"group_target_std_{g}"] = tr.groupby(g)[target].std()
                names.append(f"{g}_std_{target}")

        # ── 2. Pair-group target stats ──────────────────────────────────────
        if len(self.group_cols) >= 2 and target in tr.columns:
            pair_key = "_".join(self.group_cols)
            p["pair_key"]   = pair_key
            p["pair_mean"]  = tr.groupby(self.group_cols)[target].mean()
            p["pair_std"]   = tr.groupby(self.group_cols)[target].std()
            names.extend([f"{pair_key}_mean", f"{pair_key}_std"])

            # Recent windows over the last 4 and 8 time periods (panel only)
            if self.time_col and self.time_col in tr.columns:
                t_max = tr[self.time_col].max()
                for w in [4, 8]:
                    recent = tr[tr[self.time_col] >= t_max - w + 1]
                    if len(recent) > 0:
                        p[f"pair_recent_{w}_mean"] = (
                            recent.groupby(self.group_cols)[target].mean()
                        )
                        names.append(f"{pair_key}_recent{w}_mean")
                # Recent-vs-hist ratio
                r4 = tr[tr[self.time_col] >= t_max - 3].groupby(self.group_cols)[target].mean()
                hist = tr.groupby(self.group_cols)[target].mean()
                ratio = r4 / hist.replace(0, np.nan)
                p["pair_recent4_vs_hist_ratio"] = ratio
                names.append(f"{pair_key}_recent4_vs_hist_ratio")

        # ── 3. Cross-sectional "sex" target stats ────────────────────────────
        # Preserves prior cross-sectional path behaviour
        if "sex" in tr.columns and target in tr.columns:
            sex_stats = tr.groupby("sex")[target].agg(["mean", "std"])
            p["sex_target_mean"] = sex_stats["mean"]
            p["sex_target_std"]  = sex_stats["std"]
            names.extend(["sex_mean_target", "sex_std_target"])

        # ── 4. Slope features (panel only) ───────────────────────────────────
        # target_slope_w{w} for w in [6, 12]
        if (
            self.time_col and self.time_col in tr.columns
            and self.group_cols and target in tr.columns
        ):
            for w in [6, 12]:
                slope_map: dict = {}
                for key, grp in tr.groupby(self.group_cols):
                    recent = grp.nlargest(w, self.time_col)
                    if len(recent) < 2:
                        slope_map[key] = 0.0
                        continue
                    sx = recent[self.time_col].values.astype(float)
                    sy = recent[target].values.astype(float)
                    sxc = sx - sx.mean()
                    denom = float(np.dot(sxc, sxc))
                    slope_map[key] = float(np.dot(sxc, sy) / (denom + 1e-9))
                if slope_map:
                    p[f"slope_map_w{w}"] = slope_map
                    names.append(f"target_slope_w{w}")

        # ── 5. Covariate group statistics ────────────────────────────────────
        # `{cov}_{g}_hist_mean`, `{cov}_{g}_recent4_mean`, `{cov}_{g}_drift`
        recent4_tr: Optional[pd.DataFrame] = None
        if (
            self.time_col and self.time_col in tr.columns
            and self.numeric_cov_cols and self.group_cols
        ):
            t_max = tr[self.time_col].max()
            recent4_tr = tr[tr[self.time_col] >= t_max - 3]
        for cov in self.numeric_cov_cols:
            if cov not in tr.columns:
                continue
            for g in self.group_cols:
                if g not in tr.columns:
                    continue
                hist_m = tr.groupby(g)[cov].mean()
                p[f"cov_group_hist_mean_{cov}_{g}"] = hist_m
                names.append(f"{cov}_{g}_hist_mean")
                if recent4_tr is not None:
                    rec_m = recent4_tr.groupby(g)[cov].mean()
                    p[f"cov_group_recent4_mean_{cov}_{g}"] = rec_m
                    p[f"cov_group_drift_{cov}_{g}"]        = rec_m - hist_m
                    names.append(f"{cov}_{g}_recent4_mean")
                    names.append(f"{cov}_{g}_drift")

        # ── 6. Covariate minus overall mean ─────────────────────────────────
        for cov in self.numeric_cov_cols:
            if cov in tr.columns:
                p[f"cov_overall_mean_{cov}"] = float(tr[cov].mean())
                names.append(f"{cov}_minus_mean")

        # ── 7. Primary covariate deviation/ratio vs ref-group baseline ───────
        if self.numeric_cov_cols and len(self.group_cols) >= 2:
            primary_cov = self.numeric_cov_cols[0]
            ref_group   = self.group_cols[-1]
            if primary_cov in tr.columns and ref_group in tr.columns:
                p["primary_cov"] = primary_cov
                p["ref_group"]   = ref_group
                p[f"grp_cov_mean_{ref_group}_{primary_cov}"] = (
                    tr.groupby(ref_group)[primary_cov].mean()
                )
                names.extend([
                    f"{ref_group}_mean_{primary_cov}",
                    f"{primary_cov}_deviation",
                    f"{primary_cov}_ratio",
                ])

        # ── 8. Shift-aware features (KS > threshold covariates) ─────────────
        shift_list = self.profile.get("distribution_shifts", []) if self.profile else []
        shifted_covs = [
            entry.get("column")
            for entry in shift_list
            if isinstance(entry, dict)
               and entry.get("ks_statistic", 0.0) > self.ks_shift_threshold
               and entry.get("column") in self.numeric_cov_cols
               and entry.get("column") in tr.columns
        ]
        p["shifted_covs"] = shifted_covs
        for cov in shifted_covs:
            tr_vals = tr[cov].dropna()
            if len(tr_vals) == 0:
                continue
            mu  = float(tr_vals.mean())
            sd  = float(tr_vals.std())
            med = float(tr_vals.median())
            p[f"shift_mean_{cov}"]   = mu
            p[f"shift_std_{cov}"]    = sd if not np.isnan(sd) and sd > 1e-9 else None
            p[f"shift_median_{cov}"] = med
            # Rank CDF — pre-sorted training values
            p[f"shift_sorted_{cov}"] = np.sort(tr_vals.values)
            # Group means for dev_from_{g}_mean
            for g in self.group_cols:
                if g in tr.columns:
                    p[f"shift_group_mean_{cov}_{g}"] = (
                        tr[tr[cov].notna()].groupby(g)[cov].mean()
                    )

            # Column names produced
            if p[f"shift_std_{cov}"] is not None:
                names.append(f"{cov}_zscore")
                names.append(f"{cov}_zscore_roll4")
            names.append(f"{cov}_rank")
            for g in self.group_cols:
                names.append(f"{cov}_dev_from_{g}_mean")
            if self.time_col:
                names.append(f"{cov}_x_{self.time_col}_norm")

        # ── 9. Median imputation params (numeric feature cols only) ─────────
        if self.apply_median_impute:
            id_set = set(self.group_cols + ([self.time_col] if self.time_col else []) + [target, "adversarial_weights"])
            num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            impute_cols = [c for c in num_cols if c not in id_set]
            if impute_cols:
                p["impute_medians"] = train_df[impute_cols].median()

        self._params         = p
        self.feature_names_  = names
        self._fitted         = True
        return self

    # ────────────────────────────────────────────────────────────────────────
    # TRANSFORM
    # ────────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted parameters to `df` (train or val rows) and return a
        new DataFrame with the additional learned columns."""
        if not self._fitted:
            raise RuntimeError("LearnedFeaturesTransformer.transform() called before fit()")
        out = df.copy()
        p = self._params
        target = self.target_col

        new_cols: dict[str, pd.Series] = {}

        # 1. Single-group target mean/std
        for g in self.group_cols:
            mkey = f"group_target_mean_{g}"
            if mkey in p and g in out.columns:
                new_cols[f"{g}_mean_{target}"] = out[g].map(p[mkey])
        if len(self.group_cols) == 1:
            g = self.group_cols[0]
            skey = f"group_target_std_{g}"
            if skey in p and g in out.columns:
                new_cols[f"{g}_std_{target}"] = out[g].map(p[skey])

        # 2. Pair-group target stats
        if "pair_mean" in p:
            pair_key = p["pair_key"]
            grp = self.group_cols
            if all(c in out.columns for c in grp):
                key_idx = out[grp].apply(tuple, axis=1)
                pair_mean = p["pair_mean"]
                pair_std  = p["pair_std"]
                new_cols[f"{pair_key}_mean"] = key_idx.map(pair_mean.to_dict())
                new_cols[f"{pair_key}_std"]  = key_idx.map(pair_std.to_dict())
                for w in [4, 8]:
                    rkey = f"pair_recent_{w}_mean"
                    if rkey in p:
                        new_cols[f"{pair_key}_recent{w}_mean"] = key_idx.map(p[rkey].to_dict())
                if "pair_recent4_vs_hist_ratio" in p:
                    new_cols[f"{pair_key}_recent4_vs_hist_ratio"] = key_idx.map(
                        p["pair_recent4_vs_hist_ratio"].to_dict()
                    )

        # 3. Cross-sectional sex stats
        if "sex_target_mean" in p and "sex" in out.columns:
            new_cols["sex_mean_target"] = out["sex"].map(p["sex_target_mean"])
            new_cols["sex_std_target"]  = out["sex"].map(p["sex_target_std"])

        # 4. Slope features
        for w in [6, 12]:
            mkey = f"slope_map_w{w}"
            if mkey not in p:
                continue
            slope_map = p[mkey]
            if len(self.group_cols) == 1:
                g = self.group_cols[0]
                new_cols[f"target_slope_w{w}"] = out[g].map(slope_map)
            else:
                key_idx = out[self.group_cols].apply(tuple, axis=1)
                new_cols[f"target_slope_w{w}"] = key_idx.map(slope_map)

        # 5. Covariate group statistics
        for cov in self.numeric_cov_cols:
            if cov not in out.columns:
                continue
            for g in self.group_cols:
                if g not in out.columns:
                    continue
                hist_k = f"cov_group_hist_mean_{cov}_{g}"
                rec_k  = f"cov_group_recent4_mean_{cov}_{g}"
                drift_k = f"cov_group_drift_{cov}_{g}"
                if hist_k in p:
                    new_cols[f"{cov}_{g}_hist_mean"] = out[g].map(p[hist_k])
                if rec_k in p:
                    new_cols[f"{cov}_{g}_recent4_mean"] = out[g].map(p[rec_k])
                if drift_k in p:
                    new_cols[f"{cov}_{g}_drift"] = out[g].map(p[drift_k])

        # 6. Covariate minus overall mean
        for cov in self.numeric_cov_cols:
            mkey = f"cov_overall_mean_{cov}"
            if mkey in p and cov in out.columns:
                new_cols[f"{cov}_minus_mean"] = out[cov] - p[mkey]

        # 7. Primary covariate deviation/ratio vs ref-group baseline
        if "primary_cov" in p:
            primary_cov = p["primary_cov"]
            ref_group   = p["ref_group"]
            map_key     = f"grp_cov_mean_{ref_group}_{primary_cov}"
            if primary_cov in out.columns and ref_group in out.columns and map_key in p:
                ref_mean_col = out[ref_group].map(p[map_key])
                new_cols[f"{ref_group}_mean_{primary_cov}"] = ref_mean_col
                new_cols[f"{primary_cov}_deviation"]        = out[primary_cov] - ref_mean_col
                new_cols[f"{primary_cov}_ratio"]            = (
                    out[primary_cov] / (ref_mean_col + 1e-9)
                )

        # 8. Shift-aware features
        shifted_covs = p.get("shifted_covs", []) or []
        time_col = self.time_col
        time_norm = None
        if shifted_covs and time_col and time_col in out.columns:
            try:
                _tnum = pd.to_numeric(out[time_col], errors="coerce")
                t_min = float(_tnum.min())
                t_max = float(_tnum.max())
                t_range = max(t_max - t_min, 1.0)
                time_norm = (_tnum - t_min) / t_range
            except Exception:
                time_norm = None
        for cov in shifted_covs:
            if cov not in out.columns:
                continue
            mu  = p.get(f"shift_mean_{cov}")
            sd  = p.get(f"shift_std_{cov}")
            med = p.get(f"shift_median_{cov}")
            if mu is None or med is None:
                continue
            cov_imp = out[cov].fillna(med)
            if sd is not None and sd > 1e-9:
                new_cols[f"{cov}_zscore"] = (cov_imp - mu) / sd
                # Rolling 4-period z-score per group
                if self.group_cols:
                    rm = out.groupby(self.group_cols)[cov].transform(
                        lambda s: s.rolling(4, min_periods=1).mean()
                    )
                    rs = out.groupby(self.group_cols)[cov].transform(
                        lambda s: s.rolling(4, min_periods=2).std()
                    )
                    rs = rs.fillna(sd).replace(0.0, sd)
                    new_cols[f"{cov}_zscore_roll4"] = ((cov_imp - rm) / rs).fillna(0.0)
                else:
                    new_cols[f"{cov}_zscore_roll4"] = pd.Series(0.0, index=out.index)
            # Rank CDF
            sorted_tr = p.get(f"shift_sorted_{cov}")
            if sorted_tr is not None and len(sorted_tr) > 0:
                rank = np.searchsorted(sorted_tr, cov_imp.values, side="right") / len(sorted_tr)
                new_cols[f"{cov}_rank"] = pd.Series(rank, index=out.index)
            # Group deviation
            for g in self.group_cols:
                gmkey = f"shift_group_mean_{cov}_{g}"
                if gmkey in p and g in out.columns:
                    new_cols[f"{cov}_dev_from_{g}_mean"] = (
                        cov_imp - out[g].map(p[gmkey]).fillna(mu)
                    )
            # Time interaction
            if time_norm is not None:
                new_cols[f"{cov}_x_{time_col}_norm"] = cov_imp * time_norm

        # Concat in one shot to avoid DataFrame fragmentation
        if new_cols:
            out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

        # 9. Median imputation
        if self.apply_median_impute and "impute_medians" in p:
            medians = p["impute_medians"]
            cols_to_fill = [c for c in medians.index if c in out.columns]
            if cols_to_fill:
                out[cols_to_fill] = out[cols_to_fill].fillna(medians[cols_to_fill])

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def build_transformer_from_meta(
    feat_meta: dict,
    profile: Optional[dict] = None,
    apply_median_impute: bool = True,
) -> LearnedFeaturesTransformer:
    """Convenience: build a transformer from features.json + profile.json.

    feat_meta keys consumed: target_col, group_cols, time_col, covariate_cols.
    profile (optional) keys consumed: distribution_shifts, schema.
    """
    target_col = feat_meta.get("target_col")
    group_cols = feat_meta.get("group_cols", []) or []
    time_col   = feat_meta.get("time_col")
    cov_cols   = feat_meta.get("covariate_cols", []) or []
    schema     = (profile or {}).get("schema", {}) if profile else {}

    _str_dtypes = {"str", "object", "string", "category"}
    binary_cols  = [c for c in cov_cols if schema.get(c, {}).get("n_unique", 999) == 2]
    numeric_cols = [
        c for c in cov_cols
        if c not in binary_cols
           and schema.get(c, {}).get("dtype", "float64") not in _str_dtypes
    ]
    return LearnedFeaturesTransformer(
        target_col=target_col,
        group_cols=group_cols,
        time_col=time_col,
        numeric_cov_cols=numeric_cols,
        binary_cov_cols=binary_cols,
        profile=profile or {},
        apply_median_impute=apply_median_impute,
    )
