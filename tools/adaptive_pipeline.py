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
# Step 2.5 — GroupRelationalEncoder (leak-safe group-level relational features)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.stats import skew as _scipy_skew
except Exception:
    _scipy_skew = None


class GroupRelationalEncoder(BaseEstimator, TransformerMixin):
    """Fit-on-train, leak-safe group-relational features for high-cardinality
    heavy-right-tail categorical group columns.

    For each candidate column ``G`` that passes the data-driven detection rule
    (``n_unique >= min_cardinality`` AND ``signed_skew(group_means_y) >=
    skew_threshold``) the encoder emits three numeric features per row:

      - ``group_target_rank_<G>``: dense rank (descending; 1 = heaviest group)
        of the row's group within the fit-fold group-mean(y) distribution.
      - ``peer_group_mean_<G>``: mean of OTHER groups' mean(y), excluding the
        row's own group.
      - ``gap_to_top_groups_<G>``: row's group_mean(y) minus the mean of the
        top-K groups by mean(y). Frozen-recent v1 — the top-K reference is
        computed once at fit() from y_tr and does NOT slide on val rows or
        across recursive forecasting steps.

    Detection uses SIGNED skew (not abs) so the family targets the heavy RIGHT
    tail (whale groups); left-skewed distributions are skipped.

    Leak safety:
      - Train rows: inner-KFold over the training frame; each row's three
        encodings are computed using only OTHER inner folds' rows. The
        peer-group-mean further excludes the row's own group from the peer
        aggregate, so a train row's encoded value depends on no row that
        shares either its inner-fold OR its group_id.
      - Val / unseen rows: full fit-fold lookup table.
      - Recursive forecasting: production pipeline is fit on full train; the
        lookups are keyed on the unchanging group column and are bit-identical
        across every recursive step (the 3 features are
        constant-within-(group, fold) by construction).

    Train vs val rows are distinguished by index-identity (length + value
    equality of ``self._train_index_``) — same convention as ``TargetEncoderCV``.
    The encoder is fit AFTER ``FoldImputer`` (which preserves the index via
    ``DataFrame.copy()``) and BEFORE ``TargetEncoderCV`` (which drops the source
    group columns on the Ridge path), so the identity round-trip holds.

    The encoder is a strict no-op when no candidate column qualifies (returns
    X unchanged), making it safe to enable on arbitrary datasets.
    """

    _summary_logged = False  # class-level — print qualifying cols once per process

    def __init__(self,
                 targets: Optional[list[str]] = None,
                 min_cardinality: int = 20,
                 skew_threshold: float = 1.0,
                 top_k: int = 3,
                 inner_folds: int = 5,
                 random_state: int = 42):
        self.targets         = list(targets or [])
        self.min_cardinality = min_cardinality
        self.skew_threshold  = skew_threshold
        self.top_k           = top_k
        self.inner_folds     = inner_folds
        self.random_state    = random_state

    @staticmethod
    def _signed_skew(values: np.ndarray) -> float:
        """Fisher–Pearson moment skewness; signed (positive = right tail)."""
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size < 3:
            return 0.0
        if _scipy_skew is not None:
            try:
                return float(_scipy_skew(v))
            except Exception:
                pass
        m = v.mean()
        s = v.std(ddof=0)
        if s == 0.0:
            return 0.0
        return float(np.mean(((v - m) / s) ** 3))

    def fit(self, X: pd.DataFrame, y):
        if y is None:
            raise ValueError("GroupRelationalEncoder.fit requires y.")
        y_arr = np.asarray(y, dtype=float)
        n = len(X)
        self.global_mean_y_ = float(np.nanmean(y_arr)) if y_arr.size else 0.0

        candidate_cols = [c for c in self.targets if c in X.columns]

        self.qualifying_cols_:    list[str]              = []
        self.full_rank_:          dict[str, pd.Series]   = {}
        self.full_peer_mean_:     dict[str, pd.Series]   = {}
        self.full_gap_top_:       dict[str, pd.Series]   = {}
        self.global_mean_rank_:   dict[str, float]       = {}
        self._train_oof_:         dict[str, dict[str, np.ndarray]] = {}
        self._detection_log_:     list[dict]             = []

        for col in candidate_cols:
            cats_full = X[col].astype("object").fillna("__NA__")
            n_unique  = int(cats_full.nunique())
            log_entry = {"col": col, "n_unique": n_unique, "skew": None,
                         "qualifies": False, "reason": ""}
            if n_unique < self.min_cardinality:
                log_entry["reason"] = (
                    f"cardinality {n_unique} < {self.min_cardinality}"
                )
                self._detection_log_.append(log_entry)
                continue
            gm_full = pd.Series(y_arr).groupby(cats_full.values).mean()
            gm_clean = gm_full.dropna()
            if gm_clean.size < 3:
                log_entry["reason"] = f"only {gm_clean.size} non-NaN group means"
                self._detection_log_.append(log_entry)
                continue
            sk = self._signed_skew(gm_clean.values)
            log_entry["skew"] = round(sk, 3)
            if sk < self.skew_threshold:
                log_entry["reason"] = (
                    f"signed skew {sk:.2f} < {self.skew_threshold} "
                    f"(left-tailed or symmetric)"
                )
                self._detection_log_.append(log_entry)
                continue

            # ── full-fold lookups (val rows + recursive forecasting) ──────
            rank_full = gm_full.rank(method="dense", ascending=False)
            if len(gm_full) >= 2:
                peer_full = (gm_full.sum() - gm_full) / (len(gm_full) - 1)
            else:
                peer_full = pd.Series([self.global_mean_y_],
                                      index=gm_full.index)
            top_k_eff = min(self.top_k, len(gm_full))
            top_k_mean_full = float(gm_full.nlargest(top_k_eff).mean())
            gap_full = gm_full - top_k_mean_full

            self.qualifying_cols_.append(col)
            self.full_rank_[col]        = rank_full
            self.full_peer_mean_[col]   = peer_full
            self.full_gap_top_[col]     = gap_full
            self.global_mean_rank_[col] = float(rank_full.mean())
            log_entry["qualifies"] = True
            log_entry["reason"]    = "passed cardinality + heavy-right-tail"
            self._detection_log_.append(log_entry)

            # ── inner-KFold OOF for train rows ────────────────────────────
            # Per-row encoding uses only OTHER inner folds' rows; peer_mean
            # additionally excludes the row's own group from the aggregate.
            oof_rank = np.full(n, self.global_mean_rank_[col], dtype=float)
            oof_peer = np.full(n, self.global_mean_y_,         dtype=float)
            oof_gap  = np.full(n, 0.0,                          dtype=float)

            if self.inner_folds >= 2 and n >= self.inner_folds:
                kf = KFold(n_splits=self.inner_folds, shuffle=True,
                           random_state=self.random_state)
                cats_arr = cats_full.values
                for tr_idx, va_idx in kf.split(np.arange(n)):
                    # Group means on inner-train rows ONLY — peer-mean below
                    # then excludes each row's own group from those means, so
                    # the va row's encoding depends on neither its own inner
                    # fold NOR its own group_id.
                    inner_gm = pd.Series(y_arr[tr_idx]).groupby(
                        cats_arr[tr_idx]
                    ).mean()
                    if inner_gm.dropna().size == 0:
                        continue
                    inner_rank = inner_gm.rank(method="dense", ascending=False)
                    if len(inner_gm) >= 2:
                        # peer = (sum_all_other_groups) / (n_other_groups)
                        inner_peer = (
                            (inner_gm.sum() - inner_gm) / (len(inner_gm) - 1)
                        )
                    else:
                        inner_peer = pd.Series([self.global_mean_y_],
                                               index=inner_gm.index)
                    inner_top_k_eff  = min(self.top_k, len(inner_gm))
                    inner_top_k_mean = float(inner_gm.nlargest(inner_top_k_eff).mean())
                    inner_gap = inner_gm - inner_top_k_mean

                    va_cats = pd.Series(cats_arr[va_idx])
                    fallback_rank = float(inner_rank.mean())
                    oof_rank[va_idx] = va_cats.map(inner_rank).fillna(
                        fallback_rank
                    ).values
                    oof_peer[va_idx] = va_cats.map(inner_peer).fillna(
                        self.global_mean_y_
                    ).values
                    oof_gap[va_idx]  = va_cats.map(inner_gap).fillna(0.0).values

            self._train_oof_[col] = {
                "group_target_rank": oof_rank,
                "peer_group_mean":   oof_peer,
                "gap_to_top_groups": oof_gap,
            }

        self._train_index_ = X.index

        if not GroupRelationalEncoder._summary_logged:
            qualifying = [
                f"{e['col']}:n={e['n_unique']},skew={e['skew']:+.2f}"
                for e in self._detection_log_ if e["qualifies"]
            ]
            skipped = [
                f"{e['col']} ({e['reason']})"
                for e in self._detection_log_ if not e["qualifies"]
            ]
            print(
                f"GroupRelationalEncoder: qualifying={qualifying or 'NONE (no-op)'} "
                f"skipped={skipped or 'NONE'} "
                f"(min_card={self.min_cardinality}, skew_threshold>="
                f"{self.skew_threshold}, top_k={self.top_k}, "
                f"inner_folds={self.inner_folds})"
            )
            GroupRelationalEncoder._summary_logged = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if not getattr(self, "qualifying_cols_", []):
            return out
        is_train_frame = (
            len(out) == len(self._train_index_)
            and out.index.equals(self._train_index_)
        )
        for col in self.qualifying_cols_:
            if col not in out.columns:
                # group column was dropped upstream — emit fallback scalars
                for feat_name in ("group_target_rank", "peer_group_mean",
                                  "gap_to_top_groups"):
                    out[f"{feat_name}_{col}"] = (
                        self.global_mean_rank_[col] if feat_name == "group_target_rank"
                        else self.global_mean_y_   if feat_name == "peer_group_mean"
                        else 0.0
                    )
                continue
            cats = out[col].astype("object").fillna("__NA__")
            specs = [
                ("group_target_rank", self.full_rank_[col],
                 self.global_mean_rank_[col]),
                ("peer_group_mean",   self.full_peer_mean_[col],
                 self.global_mean_y_),
                ("gap_to_top_groups", self.full_gap_top_[col],
                 0.0),
            ]
            for feat_name, full_lookup, fallback in specs:
                out_col = f"{feat_name}_{col}"
                if is_train_frame:
                    out[out_col] = self._train_oof_[col][feat_name]
                else:
                    out[out_col] = cats.map(full_lookup).fillna(fallback).values
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
    group_relational_step: Optional[dict] = None

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
        elif nm == "group_relational":
            group_relational_step = step

    steps: list[tuple[str, BaseEstimator]] = []

    if impute_targets:
        steps.append((
            "impute",
            FoldImputer(strategy=impute_strategy, targets=impute_targets),
        ))

    # GroupRelationalEncoder runs AFTER FoldImputer (which preserves the index
    # for train-frame identity detection) and BEFORE TargetEncoderCV (which
    # drops the source group columns on the Ridge path).
    if group_relational_step is not None and group_relational_step.get("targets"):
        steps.append((
            "group_relational",
            GroupRelationalEncoder(
                targets=list(group_relational_step.get("targets", [])),
                min_cardinality=int(group_relational_step.get("min_cardinality", 20)),
                skew_threshold=float(group_relational_step.get("skew_threshold", 1.0)),
                top_k=int(group_relational_step.get("top_k", 3)),
                inner_folds=int(group_relational_step.get("inner_folds", 5)),
                random_state=int(group_relational_step.get("random_state", 42)),
            ),
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
