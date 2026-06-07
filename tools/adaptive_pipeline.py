# -*- coding: utf-8 -*-
"""
adaptive_pipeline.py — fold-fit transformers + factory for the modeler.

Reads pipeline_config.json's adaptive_steps and builds a leak-proof
sklearn Pipeline that is fit on the training fold only and applied
identically to the validation fold.

Supported step types (matching the user-defined config schema):
  - impute_missing   {strategy: "mean"|"median", targets: [...]}
  - scale_features   {method: "standard_scaler", targets: [...]}
  - target_encode    {smoothing: float, targets: [...]}

CatBoost vs Ridge dispatch:
  - For CatBoost (for_model="catboost"), the target_encode step is SKIPPED —
    CatBoost's cat_features handles leak-free target encoding internally via
    ordered boosting. The target_encode targets are reported to the caller
    so they can be passed as cat_features=[...] to CatBoostRegressor.
  - For Ridge (for_model="ridge"), the target_encode step is APPLIED via
    the smoothed KFold encoder so Ridge sees numeric encodings.

All transformers preserve DataFrame structure so column-targeted steps can
be chained without losing column identity. Output of the final Pipeline is
a numpy array suitable for either model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _filter_targets(targets: list[str], columns) -> list[str]:
    """Drop target names not present in the input frame's columns."""
    col_set = set(columns)
    return [c for c in targets if c in col_set]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — FoldImputer (mean / median)
# ─────────────────────────────────────────────────────────────────────────────
class FoldImputer(BaseEstimator, TransformerMixin):
    """Median/mean impute the configured target columns from the training fold."""

    def __init__(self, strategy: str = "median", targets: Optional[list[str]] = None):
        self.strategy = strategy
        self.targets  = list(targets or [])

    def fit(self, X: pd.DataFrame, y=None):
        cols = _filter_targets(self.targets, X.columns)
        if self.strategy == "mean":
            self.fill_values_ = X[cols].mean(numeric_only=True)
        else:
            self.fill_values_ = X[cols].median(numeric_only=True)
        self.fitted_cols_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = [c for c in self.fitted_cols_ if c in out.columns]
        if cols:
            out[cols] = out[cols].fillna(self.fill_values_[cols])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — FoldScaler (StandardScaler on configured columns)
# ─────────────────────────────────────────────────────────────────────────────
class FoldScaler(BaseEstimator, TransformerMixin):
    """Standardize the configured columns; leave others untouched.

    Stores per-column mean/std from the training fold and applies them
    identically to any frame passed to transform().
    """

    def __init__(self, method: str = "standard_scaler",
                 targets: Optional[list[str]] = None):
        self.method  = method
        self.targets = list(targets or [])

    def fit(self, X: pd.DataFrame, y=None):
        cols = _filter_targets(self.targets, X.columns)
        self.fitted_cols_ = cols
        if not cols:
            self.means_ = pd.Series(dtype=float)
            self.stds_  = pd.Series(dtype=float)
            return self
        sub = X[cols]
        self.means_ = sub.mean(numeric_only=True)
        std         = sub.std(numeric_only=True).replace(0.0, 1.0)
        self.stds_  = std.fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = [c for c in self.fitted_cols_ if c in out.columns]
        if cols:
            out[cols] = (out[cols] - self.means_[cols]) / self.stds_[cols]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — TargetEncoderCV (smoothed KFold target encoder)
# ─────────────────────────────────────────────────────────────────────────────
class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """Smoothed KFold target encoder.

    During fit():
      - Computes a leak-aware encoding for the training fold via an internal
        KFold (each row encoded using the OTHER inner folds' target means).
      - Stores the FULL-fold smoothed encoding per category for use on val.

    During transform():
      - Maps each row's category to its stored full-fold encoding.
      - Unseen categories are mapped to the global training target mean.

    The smoothing formula follows the classic empirical-Bayes form:
        enc(c) = (n_c * mean_c + smoothing * mean_global) / (n_c + smoothing)
    """

    def __init__(self, smoothing: float = 10.0,
                 targets: Optional[list[str]] = None,
                 inner_folds: int = 5, random_state: int = 42):
        self.smoothing    = smoothing
        self.targets      = list(targets or [])
        self.inner_folds  = inner_folds
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        cols = _filter_targets(self.targets, X.columns)
        self.fitted_cols_ = cols
        if y is None:
            raise ValueError("TargetEncoderCV.fit requires y (training targets).")
        y_arr = np.asarray(y, dtype=float)
        global_mean = float(np.nanmean(y_arr))
        self.global_mean_ = global_mean

        # Full-fold smoothed encoding per column (for transform on val)
        self.encodings_: dict[str, pd.Series] = {}
        for c in cols:
            cats = X[c].astype("object").fillna("__NA__")
            grouped = pd.DataFrame({"cat": cats, "y": y_arr}).groupby("cat")["y"]
            counts  = grouped.count()
            means   = grouped.mean()
            smooth  = (counts * means + self.smoothing * global_mean) / (
                counts + self.smoothing
            )
            self.encodings_[c] = smooth

        # Pre-compute OOF (within-train) encodings for the training fold itself
        # so the model never sees a row encoded using its own target.
        self._train_oof_: dict[str, np.ndarray] = {}
        if self.inner_folds >= 2 and len(X) >= self.inner_folds:
            kf = KFold(n_splits=self.inner_folds, shuffle=True,
                       random_state=self.random_state)
            for c in cols:
                cats = X[c].astype("object").fillna("__NA__").values
                oof  = np.full(len(X), global_mean, dtype=float)
                for tr_idx, va_idx in kf.split(np.arange(len(X))):
                    tr_cats = cats[tr_idx]
                    tr_y    = y_arr[tr_idx]
                    grouped = pd.DataFrame({"cat": tr_cats, "y": tr_y}).groupby("cat")["y"]
                    counts  = grouped.count()
                    means   = grouped.mean()
                    inner_smooth = (counts * means + self.smoothing * global_mean) / (
                        counts + self.smoothing
                    )
                    enc_map = inner_smooth.to_dict()
                    oof[va_idx] = [enc_map.get(v, global_mean) for v in cats[va_idx]]
                self._train_oof_[c] = oof
            self._train_index_ = X.index
        else:
            self._train_oof_   = {c: np.full(len(X), global_mean) for c in cols}
            self._train_index_ = X.index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = [c for c in self.fitted_cols_ if c in out.columns]
        # Detect "is this the training frame we fit on?" via index identity.
        is_train_frame = (
            len(out) == len(self._train_index_)
            and out.index.equals(self._train_index_)
        )
        for c in cols:
            encoded_col = f"{c}__te"
            if is_train_frame:
                out[encoded_col] = self._train_oof_[c]
            else:
                cats = out[c].astype("object").fillna("__NA__")
                out[encoded_col] = cats.map(self.encodings_[c]).fillna(self.global_mean_).values
            # Drop the original categorical column so Ridge sees only numeric encodings
            out = out.drop(columns=[c])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Drop-categoricals-for-Ridge transformer
# ─────────────────────────────────────────────────────────────────────────────
class DropNonNumeric(BaseEstimator, TransformerMixin):
    """Drop any remaining non-numeric columns and return a numpy array.

    Used as the last Pipeline step for Ridge so the model never sees object
    or category dtypes that StandardScaler/Ridge cannot consume.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols_ = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.numeric_cols_ if c in X.columns]
        return X[cols].to_numpy(dtype=float, copy=False, na_value=0.0)


class PassThroughDataFrame(BaseEstimator, TransformerMixin):
    """Last step for the CatBoost pipeline: return the DataFrame unchanged.

    CatBoost accepts DataFrames directly (with cat_features list of column
    names) so we keep the structure intact rather than dumping to numpy.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BuiltPipeline:
    pipeline:               Pipeline
    target_encode_targets:  list[str]   # original column names that get target-encoded
    target_encode_outputs:  list[str]   # __te suffix output names (Ridge path); equals targets for CatBoost path
    scaled_targets:         list[str]
    imputed_targets:        list[str]


def build_pipeline(
    adaptive_steps: list[dict],
    for_model:      str = "catboost",
) -> BuiltPipeline:
    """Build an sklearn Pipeline from the config's adaptive_steps list.

    Behaviour:
      - CatBoost path: skip target_encode (leave columns raw → cat_features).
        Apply imputer to numeric cols only; scaler is omitted (boosting is
        scale-invariant).
      - Ridge path: apply all three steps. Scaler runs over imputed numeric
        cols PLUS the __te outputs from target_encode. Output is np.ndarray.

    Returns a BuiltPipeline carrying the sklearn Pipeline plus column-role
    metadata the modeler uses to wire up CatBoost cat_features.
    """
    impute_targets:        list[str] = []
    impute_strategy:       str       = "median"
    scale_targets:         list[str] = []
    target_encode_targets: list[str] = []
    target_encode_smooth:  float     = 10.0

    for step in adaptive_steps:
        nm = step.get("name")
        if nm == "impute_missing":
            impute_targets  = list(step.get("targets", []))
            impute_strategy = step.get("strategy", "median")
        elif nm == "scale_features":
            scale_targets   = list(step.get("targets", []))
        elif nm == "target_encode":
            target_encode_targets = list(step.get("targets", []))
            target_encode_smooth  = float(step.get("smoothing", 10.0))

    steps: list[tuple[str, BaseEstimator]] = []

    if impute_targets:
        steps.append((
            "impute",
            FoldImputer(strategy=impute_strategy, targets=impute_targets),
        ))

    if for_model == "ridge":
        # Ridge needs numeric encodings AND scaling
        if target_encode_targets:
            steps.append((
                "target_encode",
                TargetEncoderCV(smoothing=target_encode_smooth,
                                targets=target_encode_targets),
            ))
        # Scale: combine declared scale targets with the __te outputs
        scale_full = list(scale_targets) + [f"{c}__te" for c in target_encode_targets]
        if scale_full:
            steps.append((
                "scale",
                FoldScaler(method="standard_scaler", targets=scale_full),
            ))
        steps.append(("to_numpy", DropNonNumeric()))
    else:
        # CatBoost path: keep raw categoricals; pass them via cat_features
        steps.append(("passthrough", PassThroughDataFrame()))

    pipeline = Pipeline(steps)

    return BuiltPipeline(
        pipeline=pipeline,
        target_encode_targets=target_encode_targets,
        target_encode_outputs=(
            [f"{c}__te" for c in target_encode_targets]
            if for_model == "ridge" else list(target_encode_targets)
        ),
        scaled_targets=scale_targets,
        imputed_targets=impute_targets,
    )


def load_config(path: str) -> dict:
    """Convenience loader for pipeline_config.json."""
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)
