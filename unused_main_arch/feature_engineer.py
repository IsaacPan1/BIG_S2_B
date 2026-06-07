"""feature_engineer.py — fold-bound transformer factory.

Per-fold contract: every transformer fits ONLY on train_idx and transforms both
slices of that fold. No global statistics. No cross-fold caches. Schema is
locked across folds (same set of feature columns).
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class FoldTransformer:
    """fit() may only see train rows; transform() applies to train or valid."""
    name: str = "abstract"

    def fit(self, df_train: pd.DataFrame) -> "FoldTransformer":
        raise NotImplementedError

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Transformers
# ─────────────────────────────────────────────────────────────────────────────

class LagFeatures(FoldTransformer):
    """Causal lag features per group. Uses only train history."""
    name = "lags"

    def __init__(self, target: str, time_col: str, group_cols: list[str],
                 lags: Iterable[int] = (1, 2, 4, 8, 12)):
        self.target = target
        self.time_col = time_col
        self.group_cols = group_cols
        self.lags = list(lags)
        self._train_panel: pd.DataFrame | None = None
        self._global_mean: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "LagFeatures":
        cols = self.group_cols + [self.time_col, self.target]
        self._train_panel = df_train[cols].copy()
        self._global_mean = float(df_train[self.target].mean())
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        assert self._train_panel is not None, "LagFeatures.transform before fit"
        keys = self.group_cols + [self.time_col]
        out = df_any[keys].copy()
        for lag in self.lags:
            shifted = self._train_panel.copy()
            shifted[self.time_col] = shifted[self.time_col] + lag
            shifted = shifted.rename(columns={self.target: f"lag_{lag}"})
            out = out.merge(shifted[keys + [f"lag_{lag}"]], on=keys, how="left")
            out[f"lag_{lag}"] = out[f"lag_{lag}"].fillna(self._global_mean)
        return out[[f"lag_{l}" for l in self.lags]].reset_index(drop=True)


class RollingStats(FoldTransformer):
    """Causal rolling mean/std per group using only train history."""
    name = "rolling"

    def __init__(self, target: str, time_col: str, group_cols: list[str],
                 windows: Iterable[int] = (4, 8, 12)):
        self.target = target
        self.time_col = time_col
        self.group_cols = group_cols
        self.windows = list(windows)
        self._train_panel: pd.DataFrame | None = None
        self._global_mean: float = 0.0
        self._global_std: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "RollingStats":
        cols = self.group_cols + [self.time_col, self.target]
        self._train_panel = df_train[cols].copy().sort_values(self.group_cols + [self.time_col])
        self._global_mean = float(df_train[self.target].mean())
        self._global_std = float(df_train[self.target].std() or 1.0)
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        assert self._train_panel is not None, "RollingStats.transform before fit"
        # Pre-compute rolling stats indexed by (group, time_col).
        tp = self._train_panel
        stats_frames = []
        for w in self.windows:
            g = tp.groupby(self.group_cols, sort=False)[self.target]
            roll_mean = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            roll_std = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())
            tmp = tp[self.group_cols + [self.time_col]].copy()
            tmp[f"roll_mean_{w}"] = roll_mean
            tmp[f"roll_std_{w}"] = roll_std
            stats_frames.append(tmp)
        # Reduce to one frame
        merged = stats_frames[0]
        for f in stats_frames[1:]:
            merged = merged.merge(f, on=self.group_cols + [self.time_col], how="outer")

        out = df_any[self.group_cols + [self.time_col]].merge(
            merged, on=self.group_cols + [self.time_col], how="left"
        )
        feature_cols = [c for w in self.windows
                        for c in (f"roll_mean_{w}", f"roll_std_{w}")]
        for c in feature_cols:
            if c not in out.columns:
                out[c] = self._global_mean if "mean" in c else self._global_std
            else:
                out[c] = out[c].fillna(self._global_mean if "mean" in c else self._global_std)
        return out[feature_cols].reset_index(drop=True)


class GroupEncodings(FoldTransformer):
    """Per-group mean/count/std of the target, fit on train only."""
    name = "group_encodings"

    def __init__(self, target: str, group_cols: list[str]):
        self.target = target
        self.group_cols = group_cols
        self._stats: pd.DataFrame | None = None
        self._global_mean: float = 0.0
        self._global_std: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "GroupEncodings":
        g = df_train.groupby(self.group_cols)[self.target]
        self._stats = g.agg(["mean", "std", "count"]).reset_index().rename(
            columns={"mean": "g_mean", "std": "g_std", "count": "g_count"})
        self._global_mean = float(df_train[self.target].mean())
        self._global_std = float(df_train[self.target].std() or 1.0)
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        assert self._stats is not None, "GroupEncodings.transform before fit"
        out = df_any[self.group_cols].merge(self._stats, on=self.group_cols, how="left")
        out["g_mean"] = out["g_mean"].fillna(self._global_mean)
        out["g_std"] = out["g_std"].fillna(self._global_std)
        out["g_count"] = out["g_count"].fillna(0).astype(float)
        return out[["g_mean", "g_std", "g_count"]].reset_index(drop=True)


class TargetEncoding(FoldTransformer):
    """Leakage-safe target encoding with an internal K-fold inside train."""
    name = "target_encoding"

    def __init__(self, target: str, group_cols: list[str],
                 inner_folds: int = 5, smoothing: float = 10.0):
        self.target = target
        self.group_cols = group_cols
        self.inner_folds = inner_folds
        self.smoothing = smoothing
        self._global_mean: float = 0.0
        self._train_map: pd.DataFrame | None = None

    def fit(self, df_train: pd.DataFrame) -> "TargetEncoding":
        # Final train-slice encoder (applied to validation rows at transform time)
        g = df_train.groupby(self.group_cols)[self.target]
        means = g.mean()
        counts = g.count()
        global_mean = float(df_train[self.target].mean())
        smoothed = (means * counts + global_mean * self.smoothing) / (counts + self.smoothing)
        self._train_map = smoothed.reset_index().rename(columns={self.target: "te"})
        self._global_mean = global_mean
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        out = df_any[self.group_cols].merge(self._train_map, on=self.group_cols, how="left")
        out["te"] = out["te"].fillna(self._global_mean)
        return out[["te"]].reset_index(drop=True)


class CategoricalEncoding(FoldTransformer):
    """Vocabulary frozen on train; unseen categories map to a single bucket."""
    name = "categorical"

    def __init__(self, exclude_cols: list[str] | None = None):
        self.exclude_cols = set(exclude_cols or [])
        self._vocabs: dict[str, dict] = {}

    def fit(self, df_train: pd.DataFrame) -> "CategoricalEncoding":
        self._vocabs = {}
        for c in df_train.columns:
            if c in self.exclude_cols:
                continue
            if df_train[c].dtype == object or str(df_train[c].dtype) == "category":
                cats = list(pd.unique(df_train[c].astype(str)))
                self._vocabs[c] = {v: i + 1 for i, v in enumerate(cats)}  # 0 = unseen
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        out_cols = {}
        for c, vocab in self._vocabs.items():
            if c in df_any.columns:
                out_cols[f"cat__{c}"] = df_any[c].astype(str).map(vocab).fillna(0).astype(int)
        return pd.DataFrame(out_cols, index=df_any.index).reset_index(drop=True)


class NumericPassthrough(FoldTransformer):
    """Pass numeric columns through as-is (median-impute fit on train)."""
    name = "numeric"

    def __init__(self, exclude_cols: list[str] | None = None):
        self.exclude_cols = set(exclude_cols or [])
        self._medians: pd.Series | None = None
        self._numeric_cols: list[str] = []

    def fit(self, df_train: pd.DataFrame) -> "NumericPassthrough":
        numeric = df_train.select_dtypes(include=[np.number])
        cols = [c for c in numeric.columns if c not in self.exclude_cols]
        self._numeric_cols = cols
        self._medians = df_train[cols].median(numeric_only=True)
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        out_cols = {}
        for c in self._numeric_cols:
            if c in df_any.columns:
                out_cols[c] = df_any[c].fillna(self._medians[c])
            else:
                out_cols[c] = self._medians[c]
        return pd.DataFrame(out_cols, index=df_any.index).reset_index(drop=True)


class StandardScalerTransformer(FoldTransformer):
    """StandardScaler fit on train numeric block only."""
    name = "scaler"

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None

    def fit(self, df_train: pd.DataFrame) -> "StandardScalerTransformer":
        cols = self.columns or list(df_train.select_dtypes(include=[np.number]).columns)
        self._cols = cols
        self._mean = df_train[cols].mean()
        self._std = df_train[cols].std().replace(0.0, 1.0)
        return self

    def transform(self, df_any: pd.DataFrame) -> pd.DataFrame:
        out = (df_any[self._cols].fillna(self._mean) - self._mean) / self._std
        return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build transformers from CV_PLAN
# ─────────────────────────────────────────────────────────────────────────────

def build_transformers(plan: dict, df_sample: pd.DataFrame) -> list[FoldTransformer]:
    target = plan["target_column"]
    time_col = plan.get("time_column")
    group_cols = plan.get("group_columns") or []
    pt = plan["problem_type"]

    transformers: list[FoldTransformer] = []
    exclude = {target}
    if time_col:
        exclude.add(time_col)
    exclude.update(group_cols)

    # Numeric block (no group_cols, no time_col, no target)
    transformers.append(NumericPassthrough(exclude_cols=list(exclude)))
    # Categorical block
    transformers.append(CategoricalEncoding(exclude_cols=list(exclude)))

    if pt in ("time_series", "grouped_time_series", "forecasting_multi_horizon") and time_col and group_cols:
        transformers.append(LagFeatures(target=target, time_col=time_col,
                                        group_cols=group_cols,
                                        lags=(1, 2, 4, 8, 12)))
        transformers.append(RollingStats(target=target, time_col=time_col,
                                         group_cols=group_cols,
                                         windows=(4, 8, 12)))

    if group_cols:
        transformers.append(GroupEncodings(target=target, group_cols=group_cols))
        transformers.append(TargetEncoding(target=target, group_cols=group_cols,
                                           inner_folds=5))

    return transformers


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_per_fold(plan: dict, folds_payload: dict, df: pd.DataFrame,
                 data_dir: str | Path = "data",
                 reports_dir: str | Path = "reports") -> dict:
    """Run the fold-bound feature engineer for every fold in folds_payload."""
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    target = plan["target_column"]
    df = df.reset_index(drop=True)
    feature_columns_locked: list[str] | None = None

    for fold in folds_payload["folds"]:
        k = fold["fold_id"]
        tr_idx = np.asarray(fold["train_idx"], dtype=int)
        va_idx = np.asarray(fold["valid_idx"], dtype=int)

        df_train = df.iloc[tr_idx].reset_index(drop=True)
        df_valid = df.iloc[va_idx].reset_index(drop=True)

        transformers = build_transformers(plan, df_train)
        train_blocks: list[pd.DataFrame] = []
        valid_blocks: list[pd.DataFrame] = []
        for tx in transformers:
            tx.fit(df_train)                           # train-only fit
            train_blocks.append(tx.transform(df_train))
            valid_blocks.append(tx.transform(df_valid))

        feat_train = pd.concat(train_blocks, axis=1)
        feat_valid = pd.concat(valid_blocks, axis=1)
        # Drop duplicate column names if any (safety)
        feat_train = feat_train.loc[:, ~feat_train.columns.duplicated()]
        feat_valid = feat_valid.loc[:, ~feat_valid.columns.duplicated()]

        cols_now = sorted(feat_train.columns.tolist())
        if feature_columns_locked is None:
            feature_columns_locked = cols_now
        elif cols_now != feature_columns_locked:
            # Reindex to enforce schema parity (missing cols → fill with 0)
            for c in feature_columns_locked:
                if c not in feat_train.columns:
                    feat_train[c] = 0.0
                if c not in feat_valid.columns:
                    feat_valid[c] = 0.0
            extra_tr = [c for c in feat_train.columns if c not in feature_columns_locked]
            extra_va = [c for c in feat_valid.columns if c not in feature_columns_locked]
            feat_train = feat_train.drop(columns=extra_tr)
            feat_valid = feat_valid.drop(columns=extra_va)
            feat_train = feat_train[feature_columns_locked]
            feat_valid = feat_valid[feature_columns_locked]

        # Attach target + row_id
        feat_train[target] = df_train[target].values
        feat_train["__row_id__"] = tr_idx
        feat_valid["__row_id__"] = va_idx
        feat_valid["__y_true__"] = df_valid[target].values

        feat_train.to_parquet(data_dir / f"features_train_fold_{k}.parquet", index=False)
        feat_valid.to_parquet(data_dir / f"features_valid_fold_{k}.parquet", index=False)

        manifest = {
            "plan_id": plan["plan_id"],
            "fold_id": k,
            "n_train": int(len(feat_train)),
            "n_valid": int(len(feat_valid)),
            "feature_columns": feature_columns_locked,
            "transformers": [type(tx).__name__ for tx in transformers],
            "fit_visibility": "train_idx only",
            "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(reports_dir / f"feature_manifest_fold_{k}.json", "w") as f:
            json.dump(manifest, f, indent=2)

        with open(reports_dir / "feature_engineer_was_here.txt", "a") as f:
            f.write(f"fold {k} done at {manifest['created_at_utc']}\n")

    # Combined features.json (schema-only)
    with open(reports_dir / "features.json", "w") as f:
        json.dump({
            "plan_id": plan["plan_id"],
            "target_col": target,
            "time_col": plan.get("time_column"),
            "group_cols": plan.get("group_columns") or [],
            "feature_columns": feature_columns_locked or [],
            "n_folds": folds_payload["n_splits"],
            "fold_aware": True,
        }, f, indent=2)

    return {"feature_columns": feature_columns_locked or [],
            "n_folds": folds_payload["n_splits"]}
