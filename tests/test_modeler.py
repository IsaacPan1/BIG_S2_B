"""Regression tests for the modeler pipeline.

Regression context
------------------
The val-lag-NaN bug let lag features in the validation set default to NaN
(zero-filled in practice) instead of being imputed from training medians.
This collapsed predictions toward the mean and produced retail_sales
prediction mean ~29 instead of ~43.

Tests that would have caught it:
  - test_prediction_mean_within_training_range  (29 is 37 % below train mean of 46)
  - test_no_nan_predictions                     (NaNs directly in predictions)

Run from the repo root:
    pytest tests/test_modeler.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO     = Path(__file__).parent.parent
PRACTICE = REPO / "practice_data"

# ---------------------------------------------------------------------------
# Per-dataset thresholds
# Derived by running the current pipeline on each practice dataset;
# MAE threshold = observed MAE * 1.30, rounded up to one decimal.
# ---------------------------------------------------------------------------
RETAIL_TRAIN_MEAN      = 46.32
RETAIL_TRAIN_MAX       = 247.25
RETAIL_MAE_THRESHOLD   = 10.0   # current: ~9.47-9.57; was 11.0 before feature-set regression

ENERGY_TRAIN_MEAN      = 846.96
ENERGY_TRAIN_MAX       = 2214.52
ENERGY_MAE_THRESHOLD   = 49.0   # current simple pipeline: 37.73

MEDICAL_TRAIN_MEAN     = 7.02
MEDICAL_TRAIN_MAX      = 14.0
MEDICAL_MAE_THRESHOLD  = 1.8    # current simple pipeline: 1.36


# ===========================================================================
# Shared assertion helper
# ===========================================================================

def assert_predictions_valid(
    preds_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    train_target: pd.Series,
    truth_target_col: str,
    id_cols: list[str],
    mae_threshold: float,
    dataset: str,
) -> None:
    """Assert all six modeler-output invariants.

    Parameters
    ----------
    preds_df         : DataFrame returned by the pipeline (must contain row_id,
                       id_cols, and 'predicted_target').
    truth_df         : Ground-truth DataFrame with id_cols + truth_target_col.
    train_target     : pd.Series of training target values (for range checks).
    truth_target_col : Column name of the target in truth_df.
    id_cols          : Columns used to merge predictions onto truth.
    mae_threshold    : Acceptable upper bound on MAE.
    dataset          : Name string used in error messages.
    """

    # ── 1. Required columns ────────────────────────────────────────────────
    required_cols = {"row_id", "predicted_target"} | set(id_cols)
    missing = required_cols - set(preds_df.columns)
    assert not missing, (
        f"[{dataset}] predictions.csv missing columns: {missing}. "
        "The file contract requires row_id, identifier columns from the "
        "validation set, and predicted_target."
    )

    # ── 2. No NaN predictions ──────────────────────────────────────────────
    nan_count = preds_df["predicted_target"].isna().sum()
    assert nan_count == 0, (
        f"[{dataset}] {nan_count} NaN values in predicted_target. "
        "NaN predictions indicate missing imputation of val-set lag/feature "
        "columns — the classic val-lag-NaN bug."
    )

    # ── 3. All non-negative ────────────────────────────────────────────────
    neg_count = (preds_df["predicted_target"] < 0).sum()
    assert neg_count == 0, (
        f"[{dataset}] {neg_count} negative predictions in predicted_target. "
        "All targets are non-negative quantities; predictions must be clipped ≥ 0."
    )

    # ── 4. Mean within 30 % of training mean ──────────────────────────────
    train_mean = float(train_target.mean())
    pred_mean  = float(preds_df["predicted_target"].mean())
    low, high  = train_mean * 0.70, train_mean * 1.30
    assert low <= pred_mean <= high, (
        f"[{dataset}] Prediction mean {pred_mean:.4f} is outside 30 % band "
        f"[{low:.4f}, {high:.4f}] around training mean {train_mean:.4f}. "
        "Possible distribution shift or systematic imputation error in val "
        "features (e.g., val-lag-NaN producing zero-filled lag features)."
    )

    # ── 5. Max at least 60 % of training max ──────────────────────────────
    train_max = float(train_target.max())
    pred_max  = float(preds_df["predicted_target"].max())
    assert pred_max >= 0.60 * train_max, (
        f"[{dataset}] Prediction max {pred_max:.4f} is below 60 % of training "
        f"max {train_max:.4f} (threshold: {0.60 * train_max:.4f}). "
        "Model may be collapsing predictions toward the mean — check for "
        "missing high-signal features or excessive regularisation."
    )

    # ── 6. MAE below threshold ─────────────────────────────────────────────
    merged = truth_df.merge(
        preds_df[id_cols + ["predicted_target"]],
        on=id_cols,
        how="inner",
    )
    assert len(merged) == len(truth_df), (
        f"[{dataset}] Only {len(merged)}/{len(truth_df)} truth rows found in "
        "predictions — row-count mismatch indicates missing groups or wrong id columns."
    )
    mae = float(np.mean(np.abs(
        merged[truth_target_col].to_numpy(float)
        - merged["predicted_target"].to_numpy(float)
    )))
    assert mae < mae_threshold, (
        f"[{dataset}] MAE {mae:.4f} exceeds threshold {mae_threshold}. "
        "Regression versus the baseline pipeline — re-run the full pipeline "
        "to confirm this is not seed variation."
    )


# ===========================================================================
# retail_sales — runs tools/run_modeler.py against existing parquet files
# ===========================================================================

@pytest.fixture(scope="class")
def retail_predictions(tmp_path_factory):
    """Run tools/run_modeler.py and return the resulting predictions DataFrame."""
    result = subprocess.run(
        [sys.executable, "tools/run_modeler.py"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, (
        f"tools/run_modeler.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-1000:]}"
    )
    return pd.read_csv(REPO / "reports" / "predictions.csv")


@pytest.fixture(scope="class")
def retail_truth():
    return pd.read_csv(PRACTICE / "retail_sales" / "_truth" / "val_truth.csv")


@pytest.fixture(scope="class")
def retail_train_target():
    return pd.read_csv(PRACTICE / "retail_sales" / "target_train.csv")["weekly_sales"]


class TestRetailSalesModeler:
    """End-to-end modeler tests for the retail_sales practice dataset."""

    def test_required_columns_present(self, retail_predictions):
        assert_predictions_valid.__wrapped__ = None  # placeholder; individual tests below
        required = {"row_id", "store_id", "product_id", "week", "predicted_target"}
        missing = required - set(retail_predictions.columns)
        assert not missing, (
            f"[retail_sales] predictions.csv missing columns: {missing}. "
            "Expected row_id, store_id, product_id, week, predicted_target."
        )

    def test_no_nan_predictions(self, retail_predictions):
        nan_count = retail_predictions["predicted_target"].isna().sum()
        assert nan_count == 0, (
            f"[retail_sales] {nan_count} NaN values in predicted_target. "
            "Classic symptom of val-lag-NaN bug: lag features left as NaN "
            "in the validation parquet propagate to NaN predictions."
        )

    def test_predictions_non_negative(self, retail_predictions):
        neg = (retail_predictions["predicted_target"] < 0).sum()
        assert neg == 0, (
            f"[retail_sales] {neg} negative predictions — weekly sales must be ≥ 0."
        )

    def test_row_count_matches_val_set(self, retail_predictions):
        assert len(retail_predictions) == 15_000, (
            f"[retail_sales] Expected 15 000 prediction rows (50 stores × 30 "
            f"products × 10 weeks), got {len(retail_predictions)}."
        )

    def test_feature_count_sufficient(self):
        """Catch regressions where the feature engineer drops feature families."""
        features_path = REPO / "reports" / "features.json"
        assert features_path.exists(), (
            "[retail_sales] reports/features.json not found — feature_engineer "
            "sub-agent did not run or failed to write its output."
        )
        with open(features_path) as f:
            meta = json.load(f)
        problem_type = meta.get("problem_type", "")
        total = meta.get("total_features_planned", 0)
        if problem_type == "panel_forecasting":
            assert total >= 35, (
                f"[retail_sales] features.json reports only {total} features for a "
                f"panel_forecasting problem. Expected ≥ 35. This likely means the "
                f"feature_engineer ran inline code instead of tools/feature_engineering.py, "
                f"or the script dropped feature families (lags, rolling stats, group "
                f"baselines, interactions). Restore the full feature set to recover MAE."
            )

    def test_prediction_mean_within_training_range(
        self, retail_predictions, retail_train_target
    ):
        train_mean = float(retail_train_target.mean())
        pred_mean  = float(retail_predictions["predicted_target"].mean())
        low, high  = train_mean * 0.70, train_mean * 1.30
        assert low <= pred_mean <= high, (
            f"[retail_sales] Prediction mean {pred_mean:.2f} outside 30 % band "
            f"[{low:.2f}, {high:.2f}] around training mean {train_mean:.2f}. "
            "If the predicted mean is near 29 (below 32.4), this is the "
            "val-lag-NaN bug: lag features imputed as 0 instead of ~40-50."
        )

    def test_prediction_max_not_collapsed(
        self, retail_predictions, retail_train_target
    ):
        train_max = float(retail_train_target.max())
        pred_max  = float(retail_predictions["predicted_target"].max())
        floor     = 0.60 * train_max
        assert pred_max >= floor, (
            f"[retail_sales] Prediction max {pred_max:.2f} < {floor:.2f} "
            f"(60 % of training max {train_max:.2f}). "
            "Predictions collapsed toward mean — check for missing lag/feature signal."
        )

    def test_mae_below_threshold(self, retail_predictions, retail_truth):
        merged = retail_truth.merge(
            retail_predictions[["store_id", "product_id", "week", "predicted_target"]],
            on=["store_id", "product_id", "week"],
        )
        mae = float(np.mean(np.abs(
            merged["weekly_sales"].to_numpy(float)
            - merged["predicted_target"].to_numpy(float)
        )))
        assert mae < RETAIL_MAE_THRESHOLD, (
            f"[retail_sales] MAE {mae:.4f} exceeds threshold {RETAIL_MAE_THRESHOLD}. "
            "Current baseline is 9.57; allowing buffer to 11.0 for seed variation."
        )


# ===========================================================================
# energy_load — inline simple LightGBM pipeline
# ===========================================================================

def _build_energy_load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a simple (no-lag) LightGBM pipeline on energy_load practice data.
    Returns (predictions_df, truth_df).
    """
    data = PRACTICE / "energy_load"
    train = pd.read_csv(data / "train.csv")
    val   = pd.read_csv(data / "val_features.csv")
    truth = pd.read_csv(data / "_truth" / "val_truth.csv")
    target = "load_mw"
    num_feats = ["hour_of_day", "day_of_week", "temperature", "is_holiday"]

    gm = train.groupby("region_id")[target].mean().rename("region_mean")
    gs = train.groupby("region_id")[target].std().rename("region_std")
    global_mean = float(train[target].mean())
    global_std  = float(train[target].std())
    for df in [train, val]:
        df["region_mean"] = df["region_id"].map(gm).fillna(global_mean)
        df["region_std"]  = df["region_id"].map(gs).fillna(global_std)

    feat_cols = ["region_mean", "region_std"] + num_feats
    fill_vals = train[feat_cols].median()
    X_tr = train[feat_cols].fillna(fill_vals)
    X_vl = val[feat_cols].fillna(fill_vals)
    y_tr = train[target].values

    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=300,
        learning_rate=0.05, num_leaves=31,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    preds = np.clip(model.predict(X_vl), 0, None)

    preds_df = val[["region_id", "timestamp"]].copy().reset_index(drop=True)
    preds_df.insert(0, "row_id", range(len(preds_df)))
    preds_df["predicted_target"] = preds
    return preds_df, truth


@pytest.fixture(scope="class")
def energy_predictions_and_truth():
    return _build_energy_load_predictions()


@pytest.fixture(scope="class")
def energy_train_target():
    return pd.read_csv(PRACTICE / "energy_load" / "train.csv")["load_mw"]


class TestEnergyLoadPipeline:
    """Modeler invariant tests for the energy_load practice dataset."""

    def test_required_columns_present(self, energy_predictions_and_truth):
        preds, _ = energy_predictions_and_truth
        required = {"row_id", "region_id", "timestamp", "predicted_target"}
        missing = required - set(preds.columns)
        assert not missing, (
            f"[energy_load] predictions missing columns: {missing}."
        )

    def test_no_nan_predictions(self, energy_predictions_and_truth):
        preds, _ = energy_predictions_and_truth
        nan_count = preds["predicted_target"].isna().sum()
        assert nan_count == 0, (
            f"[energy_load] {nan_count} NaN values in predicted_target. "
            "Missing imputation in val features."
        )

    def test_predictions_non_negative(self, energy_predictions_and_truth):
        preds, _ = energy_predictions_and_truth
        neg = (preds["predicted_target"] < 0).sum()
        assert neg == 0, (
            f"[energy_load] {neg} negative load_mw predictions — electricity "
            "demand must be ≥ 0."
        )

    def test_row_count_matches_val_set(self, energy_predictions_and_truth):
        preds, truth = energy_predictions_and_truth
        assert len(preds) == len(truth), (
            f"[energy_load] Expected {len(truth)} rows, got {len(preds)}."
        )

    def test_prediction_mean_within_training_range(
        self, energy_predictions_and_truth, energy_train_target
    ):
        preds, _ = energy_predictions_and_truth
        train_mean = float(energy_train_target.mean())
        pred_mean  = float(preds["predicted_target"].mean())
        low, high  = train_mean * 0.70, train_mean * 1.30
        assert low <= pred_mean <= high, (
            f"[energy_load] Prediction mean {pred_mean:.2f} outside 30 % band "
            f"[{low:.2f}, {high:.2f}] around training mean {train_mean:.2f}. "
            "Possible distribution shift or missing imputation in val features."
        )

    def test_prediction_max_not_collapsed(
        self, energy_predictions_and_truth, energy_train_target
    ):
        preds, _ = energy_predictions_and_truth
        train_max = float(energy_train_target.max())
        pred_max  = float(preds["predicted_target"].max())
        floor     = 0.60 * train_max
        assert pred_max >= floor, (
            f"[energy_load] Prediction max {pred_max:.2f} < {floor:.2f} "
            f"(60 % of training max {train_max:.2f}). "
            "Model collapsed to mean — check for missing high-signal features."
        )

    def test_mae_below_threshold(self, energy_predictions_and_truth):
        preds, truth = energy_predictions_and_truth
        merged = truth.merge(
            preds[["region_id", "timestamp", "predicted_target"]],
            on=["region_id", "timestamp"],
        )
        mae = float(np.mean(np.abs(
            merged["load_mw"].to_numpy(float)
            - merged["predicted_target"].to_numpy(float)
        )))
        assert mae < ENERGY_MAE_THRESHOLD, (
            f"[energy_load] MAE {mae:.4f} exceeds threshold {ENERGY_MAE_THRESHOLD}. "
            "Current simple-pipeline baseline is 37.73 MW."
        )


# ===========================================================================
# medical_imaging — inline simple LightGBM pipeline
# ===========================================================================

def _build_medical_imaging_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a simple tabular LightGBM pipeline on medical_imaging practice data.
    Returns (predictions_df, truth_df).
    """
    data = PRACTICE / "medical_imaging"
    cov_tr = pd.read_csv(data / "covariates_train.csv")
    tgt_tr = pd.read_csv(data / "target_train.csv")
    cov_vl = pd.read_csv(data / "covariates_val.csv")
    truth  = pd.read_csv(data / "_truth" / "val_truth.csv")
    target = "hospitalization_days"

    train = cov_tr.merge(tgt_tr, on="patient_id")
    num_feats = ["age", "prior_admissions", "comorbidity_score"]
    le = LabelEncoder()
    train["sex_enc"]  = le.fit_transform(train["sex"].astype(str))
    classes = set(le.classes_)
    cov_vl["sex_enc"] = [
        le.transform([str(v)])[0] if str(v) in classes else -1
        for v in cov_vl["sex"]
    ]

    feat_cols = num_feats + ["sex_enc"]
    fill_vals = train[feat_cols].median()
    X_tr = train[feat_cols].fillna(fill_vals)
    X_vl = cov_vl[feat_cols].fillna(fill_vals)
    y_tr = train[target].values

    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=300,
        learning_rate=0.05, num_leaves=31,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    preds = np.clip(model.predict(X_vl), 0, None)

    preds_df = cov_vl[["patient_id"]].copy().reset_index(drop=True)
    preds_df.insert(0, "row_id", range(len(preds_df)))
    preds_df["predicted_target"] = preds
    return preds_df, truth


@pytest.fixture(scope="class")
def medical_predictions_and_truth():
    return _build_medical_imaging_predictions()


@pytest.fixture(scope="class")
def medical_train_target():
    return pd.read_csv(PRACTICE / "medical_imaging" / "target_train.csv")[
        "hospitalization_days"
    ]


class TestMedicalImagingPipeline:
    """Modeler invariant tests for the medical_imaging practice dataset."""

    def test_required_columns_present(self, medical_predictions_and_truth):
        preds, _ = medical_predictions_and_truth
        required = {"row_id", "patient_id", "predicted_target"}
        missing = required - set(preds.columns)
        assert not missing, (
            f"[medical_imaging] predictions missing columns: {missing}."
        )

    def test_no_nan_predictions(self, medical_predictions_and_truth):
        preds, _ = medical_predictions_and_truth
        nan_count = preds["predicted_target"].isna().sum()
        assert nan_count == 0, (
            f"[medical_imaging] {nan_count} NaN values in predicted_target. "
            "Missing imputation in val features."
        )

    def test_predictions_non_negative(self, medical_predictions_and_truth):
        preds, _ = medical_predictions_and_truth
        neg = (preds["predicted_target"] < 0).sum()
        assert neg == 0, (
            f"[medical_imaging] {neg} negative hospitalization_days predictions — "
            "days must be ≥ 0."
        )

    def test_row_count_matches_val_set(self, medical_predictions_and_truth):
        preds, truth = medical_predictions_and_truth
        assert len(preds) == len(truth), (
            f"[medical_imaging] Expected {len(truth)} rows, got {len(preds)}."
        )

    def test_prediction_mean_within_training_range(
        self, medical_predictions_and_truth, medical_train_target
    ):
        preds, _ = medical_predictions_and_truth
        train_mean = float(medical_train_target.mean())
        pred_mean  = float(preds["predicted_target"].mean())
        low, high  = train_mean * 0.70, train_mean * 1.30
        assert low <= pred_mean <= high, (
            f"[medical_imaging] Prediction mean {pred_mean:.2f} outside 30 % band "
            f"[{low:.2f}, {high:.2f}] around training mean {train_mean:.2f}. "
            "Possible distribution shift or missing imputation in val features."
        )

    def test_prediction_max_not_collapsed(
        self, medical_predictions_and_truth, medical_train_target
    ):
        preds, _ = medical_predictions_and_truth
        train_max = float(medical_train_target.max())
        pred_max  = float(preds["predicted_target"].max())
        floor     = 0.60 * train_max
        assert pred_max >= floor, (
            f"[medical_imaging] Prediction max {pred_max:.2f} < {floor:.2f} "
            f"(60 % of training max {train_max:.2f}). "
            "Model collapsed to mean — check for missing features."
        )

    def test_mae_below_threshold(self, medical_predictions_and_truth):
        preds, truth = medical_predictions_and_truth
        merged = truth.merge(
            preds[["patient_id", "predicted_target"]],
            on="patient_id",
        )
        mae = float(np.mean(np.abs(
            merged["hospitalization_days"].to_numpy(float)
            - merged["predicted_target"].to_numpy(float)
        )))
        assert mae < MEDICAL_MAE_THRESHOLD, (
            f"[medical_imaging] MAE {mae:.4f} exceeds threshold {MEDICAL_MAE_THRESHOLD}. "
            "Current simple-pipeline baseline is 1.36 days."
        )
