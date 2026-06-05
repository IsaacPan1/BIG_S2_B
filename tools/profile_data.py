#!/usr/bin/env python3
"""Profile a data directory for the Award B agentic pipeline.

Parses DATA_DESCRIPTION.md, loads CSVs, classifies the problem type,
detects panel structure and forecasting horizons, runs KS tests for
distribution shift, and detects image files.

Usage
-----
    python tools/profile_data.py --data-dir data/ --output profile.json
    python tools/profile_data.py --data-dir practice_data/retail_sales --output /tmp/retail.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings as _warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp as _scipy_ks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
)

CODEBOOK_CANDIDATES: tuple[str, ...] = (
    "period_id_codebook.json",
    "period_codebook.json",
    "period_id.json",
    "dates.json",
    "time_codebook.json",
)

# ---------------------------------------------------------------------------
# 1.  DATA_DESCRIPTION.md parsing
# ---------------------------------------------------------------------------

def parse_data_description(data_dir: Path) -> dict:
    """Read DATA_DESCRIPTION.md and extract structural hints with regex."""
    md_path = data_dir / "DATA_DESCRIPTION.md"
    if not md_path.exists():
        return {"found": False}

    text = md_path.read_text(encoding="utf-8")
    result: dict[str, Any] = {"found": True}

    # --- target column ---
    target = _extract_target_col(text)
    if target:
        result["target_col"] = target

    # --- CSV file names ---
    csvs = list(dict.fromkeys(re.findall(r"`([^`\s]+\.csv)`", text)))
    result["mentioned_files"] = csvs

    # --- column names from markdown tables ---
    result["mentioned_columns"] = _extract_table_columns(text)

    # --- train / val file hints from table rows ---
    result["train_file_hints"] = _match_file_role(text, ["train", "training"])
    result["val_file_hints"]   = _match_file_role(text, ["val", "validation", "feature"])

    # --- loose time / group keyword mentions ---
    result["time_keyword_hints"]  = _keyword_col_mentions(
        text, ["timestamp", "datetime", "date", "time", "period",
               "week", "month", "hour", "day", "year", "quarter"]
    )
    result["group_keyword_hints"] = _keyword_col_mentions(
        text, ["store", "region", "product", "patient", "entity", "group"]
    )

    return result


def _extract_target_col(text: str) -> str | None:
    # Pattern 1 — backtick-quoted column name anywhere on the same line as "TARGET"
    # Requires the column name to be backtick-quoted so we don't capture type words.
    # Allows | between cells on the same line ([^\n]* instead of [^|\n]*).
    m = re.search(r"\|\s*`(\w+)`\s*\|[^\n]*TARGET[^\n]*\|", text)
    if m:
        return m.group(1)

    # Pattern 2 — "Predict `col`" sentence
    m = re.search(r"[Pp]redict\s+`(\w+)`", text)
    if m:
        return m.group(1)

    # Pattern 3 — "target: `col`" or "**Target**: `col`"
    m = re.search(r"[Tt]arget[:\s*]+`(\w+)`", text)
    if m:
        return m.group(1)

    # Pattern 4 — "target column/variable is/= col"
    m = re.search(
        r"target\s+(?:column|variable)[:\s=]+`?(\w+)`?", text, re.IGNORECASE
    )
    if m:
        return m.group(1)

    return None


def _extract_table_columns(text: str) -> list[str]:
    return list(dict.fromkeys(
        m.group(1)
        for m in re.finditer(r"\|\s*`(\w+)`\s*\|", text)
        if not m.group(1).lower().endswith(".csv")
    ))


def _match_file_role(text: str, keywords: list[str]) -> list[str]:
    """Return CSV filenames that appear in a table row containing any keyword."""
    hits: list[str] = []
    for line in text.splitlines():
        if any(kw in line.lower() for kw in keywords):
            for m in re.finditer(r"`([^`]+\.csv)`", line):
                fname = m.group(1)
                if fname not in hits:
                    hits.append(fname)
    return hits


def _keyword_col_mentions(text: str, keywords: list[str]) -> list[str]:
    """Find backtick-quoted identifiers adjacent to any keyword."""
    hits: list[str] = []
    for kw in keywords:
        for m in re.finditer(
            rf"`(\w*{kw}\w*)`", text, re.IGNORECASE
        ):
            val = m.group(1)
            if len(val) > 1 and not val.endswith(".csv") and val not in hits:
                hits.append(val)
    return hits


# ---------------------------------------------------------------------------
# 2.  CSV loading and file-role classification
# ---------------------------------------------------------------------------

def load_top_level_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for p in sorted(data_dir.glob("*.csv")):
        try:
            result[p.name] = pd.read_csv(p)
        except Exception as exc:
            _warnings.warn(f"Could not load {p.name}: {exc}")
    return result


# ---------------------------------------------------------------------------
# 2b.  Kaggle subfolder convention helpers
# ---------------------------------------------------------------------------

def _detect_convention(data_dir: Path) -> str:
    """Return 'kaggle_subfolder' if train/ and test/ (or val/) subdirs exist."""
    if (data_dir / "train").is_dir():
        if (data_dir / "test").is_dir() or (data_dir / "val").is_dir():
            return "kaggle_subfolder"
    return "flat"


def _load_csvs_in_dir(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all *.csv files from a single directory (non-recursive)."""
    result: dict[str, pd.DataFrame] = {}
    for p in sorted(directory.glob("*.csv")):
        try:
            result[p.name] = pd.read_csv(p)
        except Exception as exc:
            _warnings.warn(f"Could not load {directory.name}/{p.name}: {exc}")
    return result


def _discover_kaggle_target_file(
    train_dir: Path,
    target_col: str | None,
    desc: dict,
) -> str | None:
    """Discover the target-bearing file inside train/.

    Priority:
    a. DATA_DESCRIPTION.md mentions a file (by base name) in train/ that has target_col.
    b. Any *.csv in train/ (other than covariates.csv) that contains target_col.
    c. Any *.csv in train/ that is not covariates.csv.
    """
    # a. Mentioned files in description — try just the basename in case the
    #    description included subdirectory prefix
    if target_col:
        for raw_name in desc.get("mentioned_files", []):
            fname = Path(raw_name).name   # strip any leading path component
            p = train_dir / fname
            if p.exists() and fname.lower() != "covariates.csv":
                try:
                    if target_col in pd.read_csv(p, nrows=5).columns:
                        return fname
                except Exception:
                    pass

    # b. Scan all CSVs in train/ for the target column
    if target_col:
        for p in sorted(train_dir.glob("*.csv")):
            if p.name.lower() == "covariates.csv":
                continue
            try:
                if target_col in pd.read_csv(p, nrows=5).columns:
                    return p.name
            except Exception:
                pass

    # c. Any non-covariates CSV
    for p in sorted(train_dir.glob("*.csv")):
        if p.name.lower() != "covariates.csv":
            return p.name

    return None


def _build_file_paths_kaggle(
    train_dir: Path,
    val_dir: Path,
    target_file: str | None,
    data_dir: Path,
    submission_files: list[str],
    target_col: str | None,
) -> dict:
    """Build file_paths dict for the Kaggle subfolder convention."""
    td = train_dir.relative_to(data_dir).as_posix()   # "train"
    vd = val_dir.relative_to(data_dir).as_posix()     # "test" or "val"
    return {
        "convention":                    "kaggle_subfolder",
        "train_data":                    f"{td}/covariates.csv",
        "train_target":                  f"{td}/{target_file}" if target_file else None,
        "val_features":                  f"{vd}/covariates.csv",
        "sample_submission":             submission_files[0] if submission_files else None,
        "train_target_in_combined_file": False,
        "target_column":                 target_col,
    }


def identify_files(
    csvs: dict[str, pd.DataFrame],
    target_col: str | None,
    desc: dict,
) -> dict:
    """Split CSV files into train / val / submission buckets.

    Priority order (highest → lowest):
      1. Filename contains "submission"/"sample"  → submission
      2. File contains the target column with non-null values  → train
      3. Filename contains "train"/"fit"  → train
      4. Filename contains "val"/"test"/"eval"/"feature"  → val
      5. Description hints (unreliable; same CSV often appears in both train
         and val sections of the markdown, so only use as a tiebreaker)
    """
    train_frames: dict[str, pd.DataFrame] = {}
    val_frames:   dict[str, pd.DataFrame] = {}
    sub_frames:   dict[str, pd.DataFrame] = {}

    desc_val_hints   = set(desc.get("val_file_hints",   []))
    desc_train_hints = set(desc.get("train_file_hints", []))

    for name, df in csvs.items():
        lname = name.lower()

        # 1. Explicit submission files
        if "submission" in lname or "sample" in lname:
            sub_frames[name] = df
            continue

        # 2. Contains target column with mostly non-null values → training
        if target_col and target_col in df.columns:
            null_ratio = df[target_col].isna().mean()
            if null_ratio < 0.5:
                train_frames[name] = df
            else:
                sub_frames[name] = df   # all-null target = held-out submission
            continue

        # 3. Filename says "train" → training (even without the target column,
        #    e.g. covariates_train.csv which will be merged with target_train.csv)
        if any(x in lname for x in ("train", "fit")):
            train_frames[name] = df
            continue

        # 4. Filename says "val" / "test" / "feature" → validation
        if any(x in lname for x in ("val", "test", "eval", "feature")):
            val_frames[name] = df
            continue

        # 5. Use description hints only as tiebreaker for ambiguous filenames
        only_in_val   = name in desc_val_hints   and name not in desc_train_hints
        only_in_train = name in desc_train_hints  and name not in desc_val_hints
        if only_in_val:
            val_frames[name] = df
        elif only_in_train:
            train_frames[name] = df
        else:
            train_frames[name] = df   # conservative default

    # If no val frames were found, grab everything not yet classified
    if not val_frames:
        remaining = {
            n: d for n, d in csvs.items()
            if n not in train_frames and n not in sub_frames
        }
        val_frames = remaining

    return {
        "train_files":      list(train_frames.keys()),
        "val_files":        list(val_frames.keys()),
        "submission_files": list(sub_frames.keys()),
        "train_frames":     train_frames,
        "val_frames":       val_frames,
    }


def build_train_df(
    frames: dict[str, pd.DataFrame],
    target_col: str | None,
) -> pd.DataFrame:
    """Merge separate covariate + target files into one training DataFrame."""
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return list(frames.values())[0].copy()

    if not target_col:
        return pd.concat(list(frames.values()), ignore_index=True)

    target_frames = {n: d for n, d in frames.items() if target_col in d.columns}
    cov_frames    = {n: d for n, d in frames.items() if target_col not in d.columns}

    if not target_frames:
        return pd.concat(list(frames.values()), ignore_index=True)
    if not cov_frames:
        return pd.concat(list(target_frames.values()), ignore_index=True)

    cov_df = pd.concat(list(cov_frames.values()), ignore_index=True)
    tgt_df = pd.concat(list(target_frames.values()), ignore_index=True)

    common = list(set(cov_df.columns) & set(tgt_df.columns))
    if common:
        return cov_df.merge(tgt_df, on=common, how="left")
    return cov_df


def build_val_df(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(list(frames.values()), ignore_index=True)


def build_file_paths(
    train_files: list[str],
    val_files: list[str],
    submission_files: list[str],
    target_col: str | None,
    csvs: dict[str, pd.DataFrame],
) -> dict:
    """Determine which file serves each role in the pipeline.

    Returns a dict understood by all downstream tools:
      train_data        — primary file with training features (may also contain target)
      train_target      — file that has the target column; equals train_data when the
                          dataset uses a combined-file convention (e.g. energy_load)
      val_features      — validation feature file (no target column)
      sample_submission — submission template
      train_target_in_combined_file — True when target lives in the same file as the
                          training covariates (combined convention)
      target_column     — copy of target_col for downstream convenience
    """
    target_files  = [
        f for f in train_files
        if target_col and target_col in csvs.get(f, pd.DataFrame()).columns
    ]
    cov_only_files = [f for f in train_files if f not in target_files]

    if cov_only_files:
        # Split convention: separate covariate and target files
        train_data   = cov_only_files[0]
        train_target = target_files[0] if target_files else train_files[0]
        combined     = False
    elif target_files:
        # Combined convention: single file holds features + target
        train_data   = target_files[0]
        train_target = target_files[0]
        combined     = True
    elif train_files:
        train_data   = train_files[0]
        train_target = train_files[0]
        combined     = True
    else:
        train_data   = None
        train_target = None
        combined     = True

    return {
        "convention":                    "split" if not combined else "combined",
        "train_data":                    train_data,
        "train_target":                  train_target,
        "val_features":                  val_files[0] if val_files else None,
        "sample_submission":             submission_files[0] if submission_files else None,
        "train_target_in_combined_file": combined,
        "target_column":                 target_col,
    }


# ---------------------------------------------------------------------------
# 3.  Target column inference (fallback when DATA_DESCRIPTION.md is silent)
# ---------------------------------------------------------------------------

def infer_target_col(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> str | None:
    """The target is usually the numeric column present in train but absent from val."""
    if val_df.empty:
        return None

    train_only = set(train_df.columns) - set(val_df.columns)

    numeric_candidates = [
        c for c in train_only
        if pd.api.types.is_numeric_dtype(train_df[c])
        and not train_df[c].isna().all()
        and "id" not in c.lower()
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    if len(numeric_candidates) > 1:
        # Prefer the column with more unique values (more likely continuous)
        return max(numeric_candidates, key=lambda c: train_df[c].nunique())

    # Fall back to any train-only column
    non_id = [c for c in train_only if "id" not in c.lower()]
    return non_id[0] if non_id else (list(train_only)[0] if train_only else None)


# ---------------------------------------------------------------------------
# 4.  Structural column detection
# ---------------------------------------------------------------------------

_TIME_KEYWORDS = (
    "timestamp", "datetime", "date", "time", "period",
    "week", "month", "hour", "day", "year", "quarter",
)

_GROUP_ENTITY_KEYWORDS = (
    "id", "region", "store", "product", "group",
    "entity", "patient", "item", "series",
)


def find_time_col(df: pd.DataFrame, target_col: str | None = None) -> str | None:
    """Return the primary time column, scoring by datetime-parseability and name."""
    n_rows = max(len(df), 1)
    exclude = {target_col} if target_col else set()
    scored: list[tuple[float, str]] = []

    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        col_lower = col.lower()
        n_unique = int(s.nunique())
        score = 0.0

        # Datetime-parseable string column  (big bonus)
        if s.dtype == object:
            sample = s.dropna().head(50)
            parsed = pd.to_datetime(sample, errors="coerce")
            if len(sample) > 0 and parsed.notna().mean() > 0.9:
                score += 100.0 + n_unique / n_rows

        # Name contains a recognised time keyword
        if any(kw == col_lower or col_lower.startswith(kw) or col_lower.endswith(kw)
               for kw in _TIME_KEYWORDS):
            if n_unique >= 5:
                score += 10.0
                # Higher unique-count ratio → more likely a real time index
                ratio = n_unique / n_rows
                if 0.0001 < ratio < 0.5:
                    score += ratio * 8

        if score > 0:
            scored.append((score, col))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def find_id_col(
    df: pd.DataFrame,
    time_col: str | None,
    target_col: str | None,
) -> str | None:
    """Detect an explicit row-ID column (near-unique, often named *_id or row_id)."""
    n_rows = len(df)
    exclude = {c for c in [time_col, target_col] if c}

    for col in df.columns:
        if col in exclude:
            continue
        col_lower = col.lower()
        n_unique = df[col].nunique()

        # Explicitly named row_id
        if col_lower in {"row_id", "id", "rowid", "record_id", "index"}:
            return col

        # Near-unique (≥ 99 % of rows) — but not if it looks like a time column
        if n_unique >= 0.99 * n_rows and not any(kw in col_lower for kw in _TIME_KEYWORDS):
            return col

    return None


def find_group_cols(
    df: pd.DataFrame,
    time_col: str | None,
    target_col: str | None,
    id_col: str | None,
) -> list[str]:
    """Detect panel group columns: low-cardinality, typically categorical."""
    n_rows = max(len(df), 1)
    exclude = {c for c in [time_col, target_col, id_col] if c}
    groups: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        col_lower = col.lower()
        n_unique = s.nunique()

        is_categorical = not pd.api.types.is_numeric_dtype(s)
        low_cardinality = n_unique <= max(500, n_rows * 0.02)

        if is_categorical and low_cardinality and n_unique >= 2:
            groups.append(col)
            continue

        # Numeric but entity-like (e.g. region stored as int with low cardinality).
        # Use word-boundary matching so "id" inside "holiday" is not a false hit.
        if pd.api.types.is_numeric_dtype(s) and n_unique <= 200:
            if any(
                re.search(rf"(?:^|_){re.escape(kw)}(?:_|$)", col_lower)
                for kw in _GROUP_ENTITY_KEYWORDS
            ):
                groups.append(col)

    return groups


# ---------------------------------------------------------------------------
# 5.  Column profiling
# ---------------------------------------------------------------------------

def profile_columns(
    df: pd.DataFrame,
    target_col: str | None,
    group_cols: list[str],
    time_col: str | None,
    id_col: str | None,
) -> dict[str, dict]:
    """Return per-column stats and inferred role."""
    schema: dict[str, dict] = {}
    group_set = set(group_cols)

    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique())
        n_null   = int(s.isna().sum())

        if col == target_col:
            role = "target"
        elif col == time_col:
            role = "time"
        elif col in group_set:
            role = "group"
        elif col == id_col:
            role = "id"
        else:
            role = "covariate"

        info: dict[str, Any] = {
            "role":     role,
            "dtype":    str(s.dtype),
            "n_unique": n_unique,
            "n_null":   n_null,
            "null_pct": round(float(n_null / max(len(s), 1) * 100), 2),
        }

        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean) > 0:
                info["min"]    = round(float(clean.min()),    6)
                info["max"]    = round(float(clean.max()),    6)
                info["mean"]   = round(float(clean.mean()),   4)
                info["std"]    = round(float(clean.std()),    4)
                info["median"] = round(float(clean.median()), 4)
        else:
            top5 = s.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top5.items()}

        schema[col] = info

    return schema


# ---------------------------------------------------------------------------
# 6.  Problem-type classification
# ---------------------------------------------------------------------------

def classify_problem(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    target_col:  str | None,
    time_col:    str | None,
    group_cols:  list[str],
) -> tuple[str, str, str]:
    """Return (problem_type, confidence, reasoning)."""
    reasons: list[str] = []

    target_is_numeric = (
        target_col is not None
        and target_col in train_df.columns
        and pd.api.types.is_numeric_dtype(train_df[target_col])
    )

    val_target_missing = (
        target_col is None
        or target_col not in val_df.columns
        or (
            target_col in val_df.columns
            and val_df[target_col].isna().mean() > 0.9
        )
    )

    has_time   = time_col is not None
    has_groups = len(group_cols) >= 1

    # ---- panel_forecasting ----
    if has_time and has_groups and target_is_numeric and val_target_missing:
        reasons += [
            f"Time column '{time_col}' identified.",
            f"{len(group_cols)} group column(s) detected: {group_cols}.",
            f"Numeric target '{target_col}'.",
            "Validation file contains covariates but no target values.",
        ]
        return "panel_forecasting", "high", " ".join(reasons)

    # ---- time_series (no groups) ----
    if has_time and not has_groups and target_is_numeric and val_target_missing:
        reasons += [
            f"Time column '{time_col}' found.",
            "No repeated group identifiers detected — treating as single series.",
            f"Numeric target '{target_col}'.",
        ]
        return "time_series", "medium", " ".join(reasons)

    # ---- tabular_regression ----
    if target_is_numeric and not has_time:
        reasons.append("Numeric target, no time structure detected.")
        conf = "medium"
        if not val_target_missing:
            reasons.append("WARNING: target appears in validation — may already be labelled.")
            conf = "low"
        return "tabular_regression", conf, " ".join(reasons)

    # ---- tabular_classification ----
    if target_col and not target_is_numeric and not has_time:
        reasons.append("Categorical or binary target, no time structure.")
        return "tabular_classification", "medium", " ".join(reasons)

    # ---- fallback ----
    return "unknown", "low", "Could not match any problem-type heuristic."


# ---------------------------------------------------------------------------
# 6b.  Problem-subtype classification (ordinal vs continuous vs classification)
# ---------------------------------------------------------------------------

def classify_problem_subtype(
    train_df: pd.DataFrame,
    target_col: str | None,
    problem_type: str,
    desc_text: str,
) -> tuple[str, str, dict]:
    """Return (problem_subtype, reasoning, target_characteristics).

    Subtypes:
      panel_forecasting       — same as problem_type
      time_series             — same as problem_type
      continuous_regression   — float or many-unique numeric target
      ordinal_regression      — integer with few consecutive unique values (count/score/days)
      binary_classification   — exactly 2 unique values
      multiclass_classification — 3-50 unordered unique values
    """
    target_chars: dict[str, Any] = {}

    # Panel/time_series: subtype equals problem_type directly.
    if problem_type in ("panel_forecasting", "time_series"):
        return problem_type, f"Inherited from problem_type={problem_type}.", target_chars

    if target_col is None or target_col not in train_df.columns:
        return "continuous_regression", "Target column not found; defaulting to continuous_regression.", target_chars

    series = train_df[target_col].dropna()
    if len(series) == 0:
        return "continuous_regression", "Target column is all-null; defaulting to continuous_regression.", target_chars

    n_unique = int(series.nunique())
    dtype_str = str(series.dtype)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    reasons: list[str] = []

    target_chars["n_unique"] = n_unique
    target_chars["dtype"] = dtype_str
    if is_numeric:
        target_chars["min"] = round(float(series.min()), 6)
        target_chars["max"] = round(float(series.max()), 6)

    # Binary classification (n_unique == 2)
    if n_unique == 2:
        target_chars["is_consecutive_integers"] = False
        target_chars["data_description_hints"] = []
        return "binary_classification", f"n_unique=2 → binary classification.", target_chars

    # Non-numeric → classification
    if not is_numeric:
        target_chars["is_consecutive_integers"] = False
        target_chars["data_description_hints"] = []
        subtype = "multiclass_classification"
        return subtype, f"Non-numeric target, {n_unique} unique values → multiclass classification.", target_chars

    # Many unique values → continuous regression
    if n_unique > 50:
        target_chars["is_consecutive_integers"] = False
        target_chars["data_description_hints"] = []
        return "continuous_regression", f"Numeric target, {n_unique} > 50 unique values → continuous regression.", target_chars

    # Numeric, 3-50 unique values — determine ordinal vs categorical
    is_integer_like = pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_float_dtype(series)
        and len(series) > 0
        and (series == series.round(0)).all()
    )

    if not is_integer_like:
        target_chars["is_consecutive_integers"] = False
        target_chars["data_description_hints"] = []
        return "continuous_regression", f"Float target, {n_unique} unique values, not integer-like → continuous regression.", target_chars

    # Integer target, 3-50 unique values — ordinal vs categorical
    min_val = int(series.min())
    max_val = int(series.max())
    actual_vals = {int(v) for v in series.unique()}
    expected_vals = set(range(min_val, max_val + 1))
    is_consecutive = actual_vals == expected_vals or (
        len(actual_vals) >= max(1, (max_val - min_val + 1) * 0.9)
        and actual_vals.issubset(expected_vals)
    )
    target_chars["is_consecutive_integers"] = is_consecutive

    ordinal_kws = ["days", "score", "count", "rating", "level", "severity",
                   "stage", "grade", "rank", "visit", "admission", "number"]
    categ_kws   = ["category", "class", "type", "label"]
    desc_lower  = desc_text.lower()

    found_ordinal = [kw for kw in ordinal_kws if kw in desc_lower]
    found_categ   = [kw for kw in categ_kws   if kw in desc_lower]
    target_chars["data_description_hints"] = found_ordinal + found_categ

    small_range = min_val in (0, 1) and max_val <= 20

    reasons.append(f"Integer target, {n_unique} unique values [{min_val}-{max_val}].")
    reasons.append("Consecutive integers." if is_consecutive else "Non-consecutive (gaps present).")

    if found_ordinal and not found_categ:
        reasons.append(f"Ordinal keywords in description: {found_ordinal}.")
        return "ordinal_regression", " ".join(reasons), target_chars

    if found_categ and not found_ordinal:
        reasons.append(f"Categorical keywords in description: {found_categ}.")
        return "multiclass_classification", " ".join(reasons), target_chars

    # Both or neither keyword — use structural heuristic
    if is_consecutive and small_range:
        reasons.append(f"Consecutive ints in small range [min={min_val}, max={max_val}] → ordinal regression.")
        return "ordinal_regression", " ".join(reasons), target_chars

    if is_consecutive:
        reasons.append("Consecutive integer values; defaulting to ordinal regression (safer for MAE).")
        return "ordinal_regression", " ".join(reasons), target_chars

    reasons.append("Non-consecutive integers → multiclass classification.")
    return "multiclass_classification", " ".join(reasons), target_chars


# ---------------------------------------------------------------------------
# 7.  Horizon detection (panel_forecasting only)
# ---------------------------------------------------------------------------

def detect_horizons(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    time_col: str,
) -> dict:
    """Compute forecasting horizon information from the validation time column."""
    if val_df.empty or time_col not in val_df.columns:
        return {"n_horizons": 0, "horizons": []}

    train_times = train_df[time_col].dropna()
    val_times   = val_df[time_col].dropna()

    if val_times.empty:
        return {"n_horizons": 0, "horizons": []}

    # --- Integer / ordinal time ---
    if pd.api.types.is_numeric_dtype(train_times):
        train_max  = float(train_times.max())
        val_unique = sorted(float(v) for v in val_times.unique())
        relative   = sorted(set(v - train_max for v in val_unique))
        horizons   = [int(h) if h == int(h) else h for h in relative]

        return {
            "n_horizons":     len(val_unique),
            "train_time_max": train_max,
            "val_time_min":   val_unique[0]  if val_unique else None,
            "val_time_max":   val_unique[-1] if val_unique else None,
            "horizons":       horizons[:20],   # cap display at 20
            "gap_periods":    horizons[0] if horizons else None,
        }

    # --- Datetime string / timestamp ---
    try:
        train_dt = pd.to_datetime(train_times, errors="coerce").dropna()
        val_dt   = pd.to_datetime(val_times,   errors="coerce").dropna()

        train_max  = train_dt.max()
        val_unique = sorted(val_dt.unique())
        n_horizons = len(val_unique)

        # Infer period from consecutive differences
        if n_horizons > 1:
            diffs   = pd.Series(val_unique).diff().dropna()
            period  = diffs.mode().iloc[0]
        else:
            period = None

        # Express horizons as relative step-numbers (1, 2, 3, …)
        horizons = list(range(1, min(n_horizons + 1, 21)))

        return {
            "n_horizons":     n_horizons,
            "train_time_max": str(train_max),
            "val_time_min":   str(min(val_unique)),
            "val_time_max":   str(max(val_unique)),
            "period_inferred": str(period) if period is not None else None,
            "horizons":        horizons,
            "gap_periods":     1,
        }
    except Exception as exc:
        return {"n_horizons": 0, "horizons": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# 8.  Distribution-shift detection (KS test)
# ---------------------------------------------------------------------------

def _numpy_ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test implemented in numpy (fallback when scipy absent)."""
    a = np.sort(a[~np.isnan(a)])
    b = np.sort(b[~np.isnan(b)])
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan")

    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, combined, side="right") / n_a
    cdf_b = np.searchsorted(b, combined, side="right") / n_b
    ks    = float(np.max(np.abs(cdf_a - cdf_b)))

    n     = n_a * n_b / (n_a + n_b)
    lam   = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks
    p_val = float(np.clip(2 * np.exp(-2 * lam ** 2), 0.0, 1.0))
    return ks, p_val


def compute_distribution_shifts(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    target_col: str | None,
    group_cols: list[str],
    time_col:   str | None,
    id_col:     str | None,
) -> list[dict]:
    """Run per-column KS tests on numeric covariates; flag KS > 0.15 & p < 0.01."""
    if val_df.empty:
        return []

    exclude = set(group_cols) | {target_col, time_col, id_col} - {None}
    results: list[dict] = []

    for col in train_df.columns:
        if col in exclude or col not in val_df.columns:
            continue
        s_tr = train_df[col].dropna()
        s_vl = val_df[col].dropna()

        if not pd.api.types.is_numeric_dtype(s_tr):
            continue
        if len(s_tr) < 20 or len(s_vl) < 20:
            continue

        if _HAS_SCIPY:
            stat, pval = _scipy_ks(s_tr.to_numpy(), s_vl.to_numpy())
        else:
            stat, pval = _numpy_ks_2samp(s_tr.to_numpy(), s_vl.to_numpy())

        results.append({
            "column":       col,
            "ks_statistic": round(float(stat), 4),
            "p_value":      round(float(pval), 6),
            "flagged":      bool(stat > 0.15 and pval < 0.01),
        })

    return sorted(results, key=lambda x: x["ks_statistic"], reverse=True)


# ---------------------------------------------------------------------------
# 9.  Image detection (preserved from prior version)
# ---------------------------------------------------------------------------

def _collect_image_files(data_dir: Path) -> list[Path]:
    found: list[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(p)
    return sorted(found)


def detect_images(data_dir: Path) -> tuple[dict, set[str]]:
    files = _collect_image_files(data_dir)
    if not files:
        return {"present": False}, set()

    parent_counts: Counter = Counter(str(f.parent) for f in files)
    primary_dir = parent_counts.most_common(1)[0][0]
    extensions  = sorted({f.suffix.lower() for f in files})
    basenames   = {f.name for f in files}

    return (
        {
            "present":      True,
            "directory":    primary_dir,
            "n_images":     len(files),
            "sample_paths": [str(f) for f in files[:3]],
            "extensions":   extensions,
        },
        basenames,
    )


def find_linkage_column(
    dataframes: list[pd.DataFrame],
    image_basenames: set[str],
) -> str | None:
    if not image_basenames:
        return None
    best_col, best_ratio = None, 0.0
    for df in dataframes:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            vals  = set(df[col].dropna().astype(str).unique())
            ratio = len(vals & image_basenames) / max(len(vals), 1)
            if ratio > best_ratio:
                best_ratio, best_col = ratio, col
    return best_col if best_ratio >= 0.50 else None


# ---------------------------------------------------------------------------
# 10.  Codebook detection (maps opaque time IDs to real dates)
# ---------------------------------------------------------------------------

def _fraction_parseable_as_dates(strings: list[str]) -> float:
    """Return fraction of strings that pandas can parse as dates."""
    if not strings:
        return 0.0
    parsed = pd.to_datetime(strings, errors="coerce")
    return float(parsed.notna().mean())


def detect_time_codebook(
    data_dir: Path,
    train_df: pd.DataFrame,
    time_col: str | None,
) -> dict:
    """Search data_dir for a JSON codebook mapping opaque time IDs to dates.

    Checks CODEBOOK_CANDIDATES in order; returns the first valid one.
    Any failure silently returns available=False rather than crashing.

    The returned dict always has these keys:
        available, path, n_entries, direction_detected, sample_mappings
    """
    _empty: dict = {
        "available": False,
        "path": None,
        "n_entries": None,
        "direction_detected": None,
        "sample_mappings": None,
    }

    if time_col is None or time_col not in train_df.columns:
        return _empty

    train_time_vals = set(train_df[time_col].dropna().astype(str).unique())
    if not train_time_vals:
        return _empty

    try:
        for fname in CODEBOOK_CANDIDATES:
            candidate = data_dir / fname
            if not candidate.exists():
                continue
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(raw, dict) or len(raw) < 10:
                continue

            sample_items = list(raw.items())[:20]
            keys_sample  = [str(k) for k, _ in sample_items]
            vals_sample  = [str(v) for _, v in sample_items]

            keys_are_dates = _fraction_parseable_as_dates(keys_sample) > 0.8
            vals_are_dates = _fraction_parseable_as_dates(vals_sample) > 0.8

            if not keys_are_dates and not vals_are_dates:
                continue

            # Which non-date side appears in the training time column?
            vals_in_train = (
                len(train_time_vals & set(vals_sample)) / max(len(vals_sample), 1)
            )
            keys_in_train = (
                len(train_time_vals & set(keys_sample)) / max(len(keys_sample), 1)
            )

            if keys_are_dates and vals_in_train > 0.3:
                # {date: id} mapping
                direction   = "date_to_id"
                id_to_date  = {str(v): str(k) for k, v in raw.items()}
            elif vals_are_dates and keys_in_train > 0.3:
                # {id: date} mapping
                direction   = "id_to_date"
                id_to_date  = {str(k): str(v) for k, v in raw.items()}
            else:
                continue

            sample_pairs = dict(list(id_to_date.items())[:5])

            return {
                "available":           True,
                "path":                fname,  # relative to data_dir
                "n_entries":           len(raw),
                "direction_detected":  direction,
                "sample_mappings":     sample_pairs,
            }
    except Exception:
        pass

    return _empty


# ---------------------------------------------------------------------------
# 11.  Human-readable summary
# ---------------------------------------------------------------------------

def print_summary(profile: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  DATA PROFILE SUMMARY")
    print(sep)

    pt   = profile.get("problem_type", "unknown")
    ptc  = profile.get("problem_type_confidence", "?")
    ptr  = profile.get("problem_type_reasoning", "")
    pst  = profile.get("problem_subtype", "")
    print(f"\nProblem type : {pt}  (confidence: {ptc})")
    if pst and pst != pt:
        print(f"Subtype      : {pst}")
    print(f"Reasoning    : {ptr}")

    tgt   = profile.get("target_col")   or "(none)"
    grp   = profile.get("group_cols",  [])
    tcol  = profile.get("time_col")    or "(none)"
    icol  = profile.get("id_col")      or "(none)"
    print(f"\nTarget col   : {tgt}")
    print(f"Group cols   : {grp if grp else '—'}")
    print(f"Time col     : {tcol}")
    print(f"ID col       : {icol}")
    cov = profile.get("covariate_cols", [])
    print(f"Covariates   : {cov}")

    n_tr = profile.get("n_train_rows", 0)
    n_vl = profile.get("n_val_rows",   0)
    tfiles = profile.get("train_files", [])
    vfiles = profile.get("val_files",   [])
    print(f"\nTrain rows   : {n_tr:,}  ({tfiles})")
    print(f"Val rows     : {n_vl:,}  ({vfiles})")

    if pt == "panel_forecasting":
        n_h = profile.get("n_horizons", 0)
        h   = profile.get("horizons",   [])
        tmax = profile.get("val_time_min", "?")
        print(f"\nForecast horizon : {n_h} unique val periods")
        display = str(h[:10]) + ("  ..." if len(h) == 20 else "")
        print(f"Horizon steps    : {display}")
        print(f"Val time start   : {profile.get('val_time_min', '?')}")
        print(f"Val time end     : {profile.get('val_time_max', '?')}")

    flagged = [s for s in profile.get("distribution_shifts", []) if s.get("flagged")]
    all_ks  = profile.get("distribution_shifts", [])
    print(f"\nKS-tested cols   : {len(all_ks)}")
    if flagged:
        print(f"Flagged shifts   : {len(flagged)}")
        for s in flagged:
            print(f"   {s['column']:25s}  KS={s['ks_statistic']:.3f}  p={s['p_value']:.4f}")
    else:
        print("Flagged shifts   : none")

    img = profile.get("image_data", {})
    if img.get("present"):
        print(f"\nImages detected  : {img['n_images']} files in {img['directory']}")
        print(f"Linkage column   : {img.get('linkage_column') or '(not identified)'}")
    else:
        print("\nImages detected  : none")

    warns = profile.get("warnings", [])
    if warns:
        print(f"\nWarnings ({len(warns)}):")
        for w in warns:
            print(f"  ! {w}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# 11.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a data directory for the Award B pipeline."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing the dataset")
    parser.add_argument("--output",   required=True, help="Path for the JSON profile output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {data_dir}")

    warns: list[str] = []

    # 1. Parse DATA_DESCRIPTION.md
    desc = parse_data_description(data_dir)
    if not desc["found"]:
        warns.append("DATA_DESCRIPTION.md not found — relying on heuristics only.")
    target_col: str | None = desc.get("target_col")

    # 2. Detect data-layout convention and load CSVs
    convention = _detect_convention(data_dir)

    if convention == "kaggle_subfolder":
        _kag_train_dir = data_dir / "train"
        _kag_val_dir   = (data_dir / "test") if (data_dir / "test").is_dir() \
                         else (data_dir / "val")
        top_csvs     = load_top_level_csvs(data_dir)
        _train_csvs  = _load_csvs_in_dir(_kag_train_dir)
        _val_csvs    = _load_csvs_in_dir(_kag_val_dir)
        if not _train_csvs:
            sys.exit(f"ERROR: no CSV files found in {_kag_train_dir}")
        _td = _kag_train_dir.name   # "train"
        _vd = _kag_val_dir.name     # "test" or "val"
        _train_frames: dict[str, pd.DataFrame] = {
            f"{_td}/{k}": v for k, v in _train_csvs.items()
        }
        _val_frames: dict[str, pd.DataFrame] = {
            f"{_vd}/{k}": v for k, v in _val_csvs.items()
        }
        _sub_files = [
            n for n in top_csvs
            if "submission" in n.lower() or "sample" in n.lower()
        ]
        csvs: dict[str, pd.DataFrame] = {
            **top_csvs,
            **_train_frames,
            **_val_frames,
        }
        file_split: dict[str, Any] = {
            "train_files":      list(_train_frames),
            "val_files":        list(_val_frames),
            "submission_files": _sub_files,
            "train_frames":     _train_frames,
            "val_frames":       _val_frames,
        }
    else:
        # Flat convention (split or combined)
        csvs = load_top_level_csvs(data_dir)
        if not csvs:
            sys.exit(f"ERROR: no CSV files found in {data_dir}")
        file_split = identify_files(csvs, target_col, desc)

    # 3. Build merged DataFrames
    train_frames = file_split["train_frames"]
    val_frames   = file_split["val_frames"]
    train_df = build_train_df(train_frames, target_col)
    val_df   = build_val_df(val_frames)

    if train_df.empty:
        sys.exit("ERROR: could not assemble a training DataFrame.")

    # 5. Infer target col from data if not found in description
    if not target_col:
        target_col = infer_target_col(train_df, val_df)
        if target_col:
            warns.append(
                f"target_col '{target_col}' inferred from data "
                "(absent in DATA_DESCRIPTION.md)."
            )
        else:
            warns.append("Could not determine target_col.")

    # 6. Structural column detection
    time_col   = find_time_col(train_df, target_col)
    id_col     = find_id_col(train_df, time_col, target_col)
    group_cols = find_group_cols(train_df, time_col, target_col, id_col)

    # 6b. Optional time codebook (maps opaque IDs → real dates)
    codebook_info = detect_time_codebook(data_dir, train_df, time_col)

    covariate_cols = [
        c for c in train_df.columns
        if c not in {target_col, time_col, id_col}
        and c not in group_cols
    ]

    # 7. Column schema
    schema = profile_columns(train_df, target_col, group_cols, time_col, id_col)

    # 8. Problem type
    problem_type, confidence, reasoning = classify_problem(
        train_df, val_df, target_col, time_col, group_cols
    )

    # 8b. Problem subtype (ordinal vs continuous vs classification)
    _desc_text = (data_dir / "DATA_DESCRIPTION.md").read_text(encoding="utf-8") \
        if (data_dir / "DATA_DESCRIPTION.md").exists() else ""
    problem_subtype, subtype_reasoning, target_chars = classify_problem_subtype(
        train_df, target_col, problem_type, _desc_text
    )

    # 9. Horizon detection
    horizon_info: dict = {}
    if problem_type in ("panel_forecasting", "time_series") and time_col:
        horizon_info = detect_horizons(train_df, val_df, time_col)

    # 10. Distribution shift
    ks_results = compute_distribution_shifts(
        train_df, val_df, target_col, group_cols, time_col, id_col
    )

    # 11. Image detection
    img_info, img_basenames = detect_images(data_dir)
    if img_info.get("present"):
        all_dfs = list(train_frames.values()) + list(val_frames.values())
        img_info["linkage_column"] = find_linkage_column(all_dfs, img_basenames)

    # 12. Build file_paths section (convention-aware)
    if convention == "kaggle_subfolder":
        _target_file = _discover_kaggle_target_file(_kag_train_dir, target_col, desc)
        file_paths = _build_file_paths_kaggle(
            _kag_train_dir,
            _kag_val_dir,
            _target_file,
            data_dir,
            file_split["submission_files"],
            target_col,
        )
    else:
        file_paths = build_file_paths(
            file_split["train_files"],
            file_split["val_files"],
            file_split.get("submission_files", []),
            target_col,
            csvs,
        )

    # Add images_dir (relative to data_dir) for all conventions
    if img_info.get("present"):
        _img_abs = Path(img_info["directory"])
        try:
            file_paths["images_dir"] = _img_abs.relative_to(data_dir).as_posix()
        except ValueError:
            file_paths["images_dir"] = img_info["directory"]
    else:
        file_paths["images_dir"] = None

    # 13. Assemble profile
    profile: dict[str, Any] = {
        "data_description_parsed": {
            k: v for k, v in desc.items() if k != "found"
        } | {"found": desc["found"]},
        "schema":                   schema,
        "problem_type":             problem_type,
        "problem_type_confidence":  confidence,
        "problem_type_reasoning":   reasoning,
        "problem_subtype":          problem_subtype,
        "problem_subtype_reasoning": subtype_reasoning,
        "target_characteristics":   target_chars,
        "target_col":               target_col,
        "group_cols":               group_cols,
        "time_col":                 time_col,
        "id_col":                   id_col,
        "covariate_cols":           covariate_cols,
        "horizons":                 horizon_info.get("horizons", []),
        "n_horizons":               horizon_info.get("n_horizons", 0),
        "val_time_min":             horizon_info.get("val_time_min"),
        "val_time_max":             horizon_info.get("val_time_max"),
        "train_time_max":           horizon_info.get("train_time_max"),
        "gap_periods":              horizon_info.get("gap_periods"),
        "period_inferred":          horizon_info.get("period_inferred"),
        "distribution_shifts":      ks_results,
        "n_train_rows":             len(train_df),
        "n_val_rows":               len(val_df),
        "train_files":              file_split["train_files"],
        "val_files":                file_split["val_files"],
        "file_paths":               file_paths,
        "image_data":               img_info,
        "time_codebook":            codebook_info,
        "warnings":                 warns,
    }

    # 14. Print summary
    print_summary(profile)

    # 15. Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, default=str)
    print(f"Profile written -> {out_path}")

    # ── KG: observability — record inferred problem (best-effort) ─────────────
    try:
        from kg import kg_init, kg_append_event, kg_set_stage
        kg_init()
        kg_set_stage("schema_analyst")
        kg_append_event("schema_analyst", "inferred_problem", {"problem_type": profile.get("problem_type"), "problem_subtype": profile.get("problem_subtype"), "target_col": profile.get("target_col"), "n_train": profile.get("n_train_rows"), "n_val": profile.get("n_val_rows"), "group_cols": profile.get("group_cols", []), "time_col": profile.get("time_col")})
    except Exception as _kg_e: print(f"[KG] non-fatal: {_kg_e}", file=sys.stderr)


if __name__ == "__main__":
    main()
