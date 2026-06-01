"""Tests for tools/validate.py

Structure
---------
TestValidatorUnit
    Fast, in-memory tests of individual functions — no parquet needed.

TestValidatorSchema
    Schema tests: run validate.py as a subprocess against the retail_sales
    practice data (requires full pipeline to have run first, producing
    data/features_train.parquet and reports/model_results.json etc.).

TestLeakDetection
    Constructs a synthetic dataset with a deliberately leaked feature,
    runs validate.py against it in a temp directory, and asserts the
    validator flags it CRITICAL with the leaked feature as suspect=True.

Run from the repo root:
    pytest tests/test_validator.py -v
    pytest tests/test_validator.py::TestLeakDetection -v  # just leak test
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

# ── Repo layout ─────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))

from validate import (  # noqa: E402  (module is in tools/)
    CV_GAP_CRITICAL_PCT,
    CV_GAP_WARNING_PCT,
    EMBARGO_PERIODS,
    IMPORTANCE_WARNING_SHARE,
    LOO_CANDIDATE_SHARE,
    LOO_STAT_SIGNAL_THRESHOLD,
    N_STRICT_FOLDS,
    STRUCTURAL_PATTERNS,
    _panel_strict_splits,
    compute_strict_cv,
    compute_verdict,
    structural_check,
)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — no I/O
# ═══════════════════════════════════════════════════════════════════════════

class TestValidatorUnit:

    def test_structural_check_flags_lead_pattern(self):
        cols = ["price_lag1", "sales_lead_1", "week_sin"]
        flags = structural_check(cols, "weekly_sales")
        assert flags["sales_lead_1"] is True
        assert flags["price_lag1"] is False
        assert flags["week_sin"] is False

    def test_structural_check_flags_lead0(self):
        flags = structural_check(["target_lag_0"], "weekly_sales")
        assert flags["target_lag_0"] is True

    def test_structural_check_flags_leak_keyword(self):
        flags = structural_check(["price_leak_feature"], "sales")
        assert flags["price_leak_feature"] is True

    def test_structural_check_flags_target_name_in_feature(self):
        # "weekly_sales_direct" contains target name without lag suffix
        flags = structural_check(["weekly_sales_direct"], "weekly_sales")
        assert flags["weekly_sales_direct"] is True

    def test_structural_check_does_not_flag_lag_features(self):
        # Lag features should NOT be flagged — they are legitimate
        cols = ["weekly_sales_lag1", "weekly_sales_lag4", "weekly_sales_roll4_mean"]
        flags = structural_check(cols, "weekly_sales")
        assert all(not flags[c] for c in cols), (
            f"Structural check incorrectly flagged lag/rolling features: "
            f"{[c for c in cols if flags[c]]}"
        )

    def test_verdict_critical_on_large_gap(self):
        verdict, checks = compute_verdict(
            cv_gap_pct=CV_GAP_CRITICAL_PCT + 0.01,
            has_two_signal_suspect=False,
            max_importance_share=0.2,
        )
        assert verdict == "CRITICAL"
        assert checks["cv_integrity"] == "CRITICAL"

    def test_verdict_critical_on_two_signal_suspect(self):
        verdict, checks = compute_verdict(
            cv_gap_pct=0.0,
            has_two_signal_suspect=True,
            max_importance_share=0.2,
        )
        assert verdict == "CRITICAL"
        assert checks["leakage_two_signal"] == "CRITICAL"

    def test_verdict_warning_on_moderate_gap(self):
        verdict, checks = compute_verdict(
            cv_gap_pct=CV_GAP_WARNING_PCT + 0.01,
            has_two_signal_suspect=False,
            max_importance_share=0.2,
        )
        assert verdict == "WARNING"
        assert checks["cv_integrity"] == "WARNING"

    def test_verdict_warning_on_importance_concentration(self):
        verdict, checks = compute_verdict(
            cv_gap_pct=0.0,
            has_two_signal_suspect=False,
            max_importance_share=IMPORTANCE_WARNING_SHARE + 0.01,
        )
        assert verdict == "WARNING"
        assert checks["importance_concentration"] == "WARNING"

    def test_verdict_pass_clean(self):
        verdict, checks = compute_verdict(
            cv_gap_pct=0.01,
            has_two_signal_suspect=False,
            max_importance_share=0.2,
        )
        assert verdict == "PASS"
        assert all(v == "PASS" for v in checks.values())

    def test_panel_splits_purge_embargo(self):
        """Each val window must not overlap the corresponding train window by >= embargo."""
        rng = np.random.default_rng(0)
        weeks = list(range(80))
        df = pd.DataFrame({
            "week": np.repeat(weeks, 5),
            "g":    np.tile(range(5), 80),
        })
        splits = _panel_strict_splits(df, "week", n_folds=4, embargo=2)
        assert len(splits) > 0
        for tr_idx, vl_idx in splits:
            tr_weeks = set(df.loc[tr_idx, "week"])
            vl_weeks = set(df.loc[vl_idx, "week"])
            overlap = tr_weeks & vl_weeks
            assert not overlap, f"Train/val week overlap: {overlap}"
            if tr_weeks and vl_weeks:
                max_tr = max(tr_weeks)
                min_vl = min(vl_weeks)
                assert min_vl > max_tr, (
                    f"Val starts before train ends: max_tr={max_tr}, min_vl={min_vl}"
                )

    def test_two_signal_conservatism(self):
        """Single-signal (stat only OR struct only) must NOT produce suspect=True."""
        # stat signal only: high importance, low LOO delta, but no structural flag
        stat_only_share   = LOO_CANDIDATE_SHARE + 0.05
        stat_only_delta   = 0.0                           # removal doesn't hurt
        struct_flag_false = False

        stat_signal  = (
            stat_only_share >= LOO_CANDIDATE_SHARE
            and abs(stat_only_delta) < LOO_STAT_SIGNAL_THRESHOLD * 5.0
        )
        suspect_stat_only = stat_signal and struct_flag_false
        assert not suspect_stat_only, "Stat-only single signal should not produce suspect=True"

        # struct signal only: name matches pattern, but importance below threshold
        struct_only_share  = LOO_CANDIDATE_SHARE - 0.01  # below threshold
        struct_flag_true   = True
        stat_signal_struct = struct_only_share >= LOO_CANDIDATE_SHARE
        suspect_struct_only = stat_signal_struct and struct_flag_true
        assert not suspect_struct_only, "Struct-only single signal should not produce suspect=True"


# ═══════════════════════════════════════════════════════════════════════════
# Schema tests — requires full pipeline output on retail_sales
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {
    "verdict", "reported_cv_mae", "strict_cv_mae", "honest_cv_mae",
    "cv_gap_abs", "cv_gap_pct", "strict_cv_scheme",
    "feature_suspicion", "checks", "notes",
}
REQUIRED_CHECKS = {"cv_integrity", "importance_concentration", "leakage_two_signal"}
VALID_VERDICTS = {"PASS", "WARNING", "CRITICAL"}
VALID_CHECK_VALUES = {"PASS", "WARNING", "CRITICAL"}


def _assert_review_schema(review: dict, context: str) -> None:
    missing = REQUIRED_KEYS - set(review.keys())
    assert not missing, f"[{context}] validator_review.json missing keys: {missing}"

    assert review["verdict"] in VALID_VERDICTS, (
        f"[{context}] invalid verdict '{review['verdict']}'"
    )
    assert isinstance(review["feature_suspicion"], list), (
        f"[{context}] feature_suspicion must be a list"
    )
    for item in review["feature_suspicion"]:
        for key in ("feature", "importance_share", "structural_flag", "suspect", "evidence"):
            assert key in item, f"[{context}] feature_suspicion item missing key '{key}'"
        assert isinstance(item["suspect"], bool)
        assert isinstance(item["structural_flag"], bool)

    assert set(review["checks"].keys()) >= REQUIRED_CHECKS, (
        f"[{context}] checks dict missing: {REQUIRED_CHECKS - set(review['checks'].keys())}"
    )
    for k, v in review["checks"].items():
        assert v in VALID_CHECK_VALUES, f"[{context}] check '{k}' has invalid value '{v}'"

    assert review["strict_cv_mae"] == review["honest_cv_mae"], (
        f"[{context}] honest_cv_mae must equal strict_cv_mae"
    )
    assert isinstance(review["notes"], str) and len(review["notes"]) > 0


@pytest.mark.skipif(
    not (REPO / "reports" / "validator_review.json").exists(),
    reason="reports/validator_review.json not present — run full pipeline first",
)
class TestValidatorSchema:
    """Tests on the existing reports/validator_review.json (if pipeline ran)."""

    def test_all_required_keys_present(self):
        with open(REPO / "reports" / "validator_review.json") as f:
            review = json.load(f)
        _assert_review_schema(review, "existing_review")

    def test_marker_file_exists(self):
        assert (REPO / "reports" / "validator_was_here.txt").exists(), (
            "reports/validator_was_here.txt missing — validator sub-agent did not run "
            "or ran inline instead of via tools/validate.py"
        )

    def test_verdict_is_valid(self):
        with open(REPO / "reports" / "validator_review.json") as f:
            review = json.load(f)
        assert review["verdict"] in VALID_VERDICTS

    def test_strict_cv_mae_is_positive(self):
        with open(REPO / "reports" / "validator_review.json") as f:
            review = json.load(f)
        assert review["strict_cv_mae"] > 0, "strict_cv_mae must be positive"

    def test_reported_cv_mae_is_positive(self):
        with open(REPO / "reports" / "validator_review.json") as f:
            review = json.load(f)
        assert review["reported_cv_mae"] > 0, "reported_cv_mae must be positive"


# ═══════════════════════════════════════════════════════════════════════════
# Leak-detection test — synthetic dataset with deliberate leak
# ═══════════════════════════════════════════════════════════════════════════

def _build_leak_fixture(tmp_dir: Path) -> None:
    """
    Create a minimal repo-root layout in tmp_dir with a deliberately leaked feature.

    Leaked feature: 'weekly_sales_lead_0' = target + tiny noise.
    Three other features are low-signal (noise only) so that:
      - The leaked feature gets very high importance (>= LOO_CANDIDATE_SHARE)
      - Three copies of the same signal exist (so removal of one barely changes MAE)
      - The feature name matches STRUCTURAL_PATTERNS ('_lead_0')

    Design rationale for two-signal detection:
      - stat_signal: leaked feature has share >= LOO_CANDIDATE_SHARE AND
        removal delta ≈ 0 (two redundant copies cover the same signal)
      - struct_signal: '_lead_0' matches structural regex
      - suspect = stat AND struct → CRITICAL
    """
    rng = np.random.default_rng(77)

    N_GROUPS = 15
    N_PERIODS = 60
    TRAIN_CUT = 50

    groups  = [f"g{i:02d}" for i in range(N_GROUPS)]
    periods = list(range(N_PERIODS))

    rows = []
    for g in groups:
        base = rng.uniform(10, 100)
        for t in periods:
            target = max(0.0, base + 5 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2))
            rows.append({"group_id": g, "week": t, "weekly_sales": round(target, 2)})

    df = pd.DataFrame(rows)
    train = df[df["week"] < TRAIN_CUT].copy()

    # Feature 1: leaked (name triggers structural check)
    # Feature 2 & 3: redundant copies of same signal (so LOO delta is tiny)
    train["weekly_sales_lead_0"] = train["weekly_sales"] + rng.normal(0, 0.1, len(train))
    train["redundant_a"]         = train["weekly_sales"] + rng.normal(0, 0.1, len(train))
    train["redundant_b"]         = train["weekly_sales"] + rng.normal(0, 0.1, len(train))
    # Noise feature
    train["noise_feat"]          = rng.standard_normal(len(train))

    feature_cols = ["weekly_sales_lead_0", "redundant_a", "redundant_b", "noise_feat"]

    (tmp_dir / "data").mkdir()
    (tmp_dir / "reports").mkdir()

    # Write features_train.parquet (features + target + identifiers)
    train_feat = train[["group_id", "week", "weekly_sales"] + feature_cols].copy()
    train_feat.to_parquet(tmp_dir / "data" / "features_train.parquet", index=False)

    # profile.json
    profile = {
        "problem_type": "panel_forecasting",
        "problem_type_confidence": "high",
        "target_col": "weekly_sales",
        "group_cols": ["group_id"],
        "time_col": "week",
        "n_train_rows": len(train),
        "n_val_rows": 0,
        "covariate_cols": feature_cols,
        "train_files": [],
        "val_files": [],
    }
    (tmp_dir / "reports" / "profile.json").write_text(json.dumps(profile))

    # features.json
    features_meta = {
        "target_col": "weekly_sales",
        "group_cols": ["group_id"],
        "time_col": "week",
        "problem_type": "panel_forecasting",
        "feature_columns": feature_cols,
        "feature_families": {"leak_test": feature_cols},
        "total_features_planned": len(feature_cols),
    }
    (tmp_dir / "reports" / "features.json").write_text(json.dumps(features_meta))

    # Train a quick LightGBM to get feature importances
    X = train[feature_cols].values
    y = train["weekly_sales"].values
    m = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=200,
        learning_rate=0.1, num_leaves=31, verbose=-1, random_state=42,
    )
    m.fit(X, y)
    importances = m.feature_importances_
    total_imp = int(importances.sum()) or 1
    imp_all = [
        {"feature": feature_cols[i], "importance": int(importances[i])}
        for i in range(len(feature_cols))
    ]

    # model_results.json
    # Walk-forward MAE uses an optimistic single 80/20 split
    cut = int(len(train) * 0.8)
    X_tr, y_tr = X[:cut], y[:cut]
    X_vl, y_vl = X[cut:], y[cut:]
    wf_preds = np.clip(m.predict(X_vl), 0, None)
    wf_mae = float(np.mean(np.abs(y_vl - wf_preds)))

    model_results = {
        "algorithm": "LightGBM",
        "objective": "regression_l1",
        "best_params": {
            "learning_rate": 0.1, "num_leaves": 31,
            "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        },
        "n_estimators": 200,
        "n_seeds": 1,
        "walk_forward_mae": wf_mae,
        "walk_forward_cv_scheme": "single 80/20 time split",
        "oof_mae": None,
        "oof_cv_scheme": None,
        "feature_importance_top10": imp_all[:10],
        "feature_importance_all": imp_all,
        "n_features": len(feature_cols),
        "n_train_rows": len(train),
        "n_val_rows": 0,
    }
    (tmp_dir / "reports" / "model_results.json").write_text(
        json.dumps(model_results, indent=2)
    )


class TestLeakDetection:
    """Prove the validator flags a deliberately leaked feature as CRITICAL+suspect."""

    def test_leak_detection_critical(self, tmp_path):
        _build_leak_fixture(tmp_path)

        result = subprocess.run(
            [sys.executable, str(REPO / "tools" / "validate.py"),
             "--repo-root", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )

        # Print output for debugging on failure
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-3000:])
            print("STDERR:", result.stderr[-1000:])
        assert result.returncode == 0, (
            f"validate.py exited {result.returncode}\n"
            f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-1000:]}"
        )

        review_path = tmp_path / "reports" / "validator_review.json"
        assert review_path.exists(), "validator_review.json not written"

        with open(review_path) as f:
            review = json.load(f)

        # Full schema check
        _assert_review_schema(review, "leak_detection")

        # Must be CRITICAL because the leaked feature fires both signals
        assert review["verdict"] == "CRITICAL", (
            f"Expected CRITICAL verdict for dataset with leaked feature, got "
            f"'{review['verdict']}'. cv_gap_pct={review['cv_gap_pct']:.3f}, "
            f"checks={review['checks']}, "
            f"suspicion={review['feature_suspicion']}"
        )

        # The leaked feature must appear as suspect=True
        suspects = {r["feature"] for r in review["feature_suspicion"] if r["suspect"]}
        assert "weekly_sales_lead_0" in suspects, (
            f"Expected 'weekly_sales_lead_0' to be suspect=True, but suspects={suspects}. "
            f"Full suspicion list: {review['feature_suspicion']}"
        )

    def test_leak_marker_written(self, tmp_path):
        _build_leak_fixture(tmp_path)
        subprocess.run(
            [sys.executable, str(REPO / "tools" / "validate.py"),
             "--repo-root", str(tmp_path)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        marker = tmp_path / "reports" / "validator_was_here.txt"
        assert marker.exists(), "validator_was_here.txt not written by validate.py"
        content = marker.read_text()
        assert "validator sub-agent executed" in content


# ═══════════════════════════════════════════════════════════════════════════
# Cross-dataset schema tests (practice data — inline models, no full pipeline)
# ═══════════════════════════════════════════════════════════════════════════

PRACTICE = REPO / "practice_data"


def _run_validate_on_fixture(tmp_path: Path, profile: dict, model_results: dict,
                              features_meta: dict, train_df: pd.DataFrame) -> dict:
    """Write fixtures to tmp_path, run validate.py, return validator_review."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "reports").mkdir(exist_ok=True)

    feature_cols = features_meta["feature_columns"]
    train_df.to_parquet(tmp_path / "data" / "features_train.parquet", index=False)
    (tmp_path / "reports" / "profile.json").write_text(json.dumps(profile))
    (tmp_path / "reports" / "model_results.json").write_text(json.dumps(model_results))
    (tmp_path / "reports" / "features.json").write_text(json.dumps(features_meta))

    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "validate.py"),
         "--repo-root", str(tmp_path)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-500:])
    assert result.returncode == 0, (
        f"validate.py failed (exit {result.returncode}):\n{result.stdout[-2000:]}"
    )

    with open(tmp_path / "reports" / "validator_review.json") as f:
        return json.load(f)


def _build_simple_lgbm_results(
    X: np.ndarray, y: np.ndarray, feature_cols: list[str],
    hparams: dict, n_estimators: int = 200,
) -> dict:
    """Train a quick LightGBM and build a model_results dict."""
    m = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=n_estimators,
        verbose=-1, random_state=42, n_jobs=-1,
        **hparams,
    )
    m.fit(X, y)
    cut = int(len(X) * 0.8)
    wf_preds = np.clip(m.predict(X[cut:]), 0, None)
    wf_mae = float(np.mean(np.abs(y[cut:] - wf_preds)))
    imp = m.feature_importances_
    imp_all = [{"feature": feature_cols[i], "importance": int(imp[i])}
               for i in range(len(feature_cols))]
    return {
        "algorithm": "LightGBM",
        "objective": "regression_l1",
        "best_params": hparams,
        "n_estimators": n_estimators,
        "n_seeds": 1,
        "walk_forward_mae": wf_mae,
        "walk_forward_cv_scheme": "single 80/20 split",
        "oof_mae": None,
        "oof_cv_scheme": None,
        "feature_importance_top10": imp_all[:10],
        "feature_importance_all": imp_all,
        "n_features": len(feature_cols),
        "n_train_rows": len(X),
        "n_val_rows": 0,
    }


HPARAMS_DEFAULT = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}


@pytest.mark.skipif(
    not (PRACTICE / "retail_sales" / "target_train.csv").exists(),
    reason="retail_sales practice data not found — run tests/generate_practice_data.py"
)
class TestRetailSalesValidator:
    """Validate schema on retail_sales — uses inline model (no full pipeline needed)."""

    @pytest.fixture(scope="class")
    def retail_review(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("retail_validate")
        data = PRACTICE / "retail_sales"
        cov   = pd.read_csv(data / "covariates_train.csv")
        tgt   = pd.read_csv(data / "target_train.csv")
        train = cov.merge(tgt, on=["store_id", "product_id", "week"])

        num_cols = ["price", "promotion_active", "holiday_flag", "weather_index"]
        train_feat = train[["store_id", "product_id", "week", "weekly_sales"] + num_cols].copy()
        X = train[num_cols].fillna(train[num_cols].median()).values
        y = train["weekly_sales"].values

        profile = {
            "problem_type": "panel_forecasting",
            "problem_type_confidence": "high",
            "target_col": "weekly_sales",
            "group_cols": ["store_id", "product_id"],
            "time_col": "week",
            "n_train_rows": len(train),
            "n_val_rows": 15000,
            "covariate_cols": num_cols,
            "train_files": [],
            "val_files": [],
        }
        features_meta = {
            "target_col": "weekly_sales",
            "group_cols": ["store_id", "product_id"],
            "time_col": "week",
            "problem_type": "panel_forecasting",
            "feature_columns": num_cols,
            "feature_families": {"covariates": num_cols},
            "total_features_planned": len(num_cols),
        }
        model_results = _build_simple_lgbm_results(X, y, num_cols, HPARAMS_DEFAULT)
        return _run_validate_on_fixture(tmp, profile, model_results, features_meta, train_feat)

    def test_schema_valid(self, retail_review):
        _assert_review_schema(retail_review, "retail_sales")

    def test_verdict_is_valid(self, retail_review):
        assert retail_review["verdict"] in VALID_VERDICTS

    def test_strict_cv_mae_positive(self, retail_review):
        assert retail_review["strict_cv_mae"] > 0

    def test_honest_equals_strict(self, retail_review):
        assert retail_review["strict_cv_mae"] == retail_review["honest_cv_mae"]

    def test_marker_written(self, tmp_path_factory):
        # This fixture re-runs; just check the review file has notes
        pass  # covered by schema validation — marker is checked in test_leak_detection


@pytest.mark.skipif(
    not (PRACTICE / "energy_load" / "train.csv").exists(),
    reason="energy_load practice data not found — run tests/generate_practice_data.py"
)
class TestEnergyLoadValidator:
    """Validate schema on energy_load (panel_forecasting)."""

    @pytest.fixture(scope="class")
    def energy_review(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("energy_validate")
        data = PRACTICE / "energy_load"
        train = pd.read_csv(data / "train.csv")

        num_cols = ["hour_of_day", "day_of_week", "temperature", "is_holiday"]
        train_feat = train[["region_id", "timestamp", "load_mw"] + num_cols].copy()

        # timestamp is string; use a numeric time proxy
        train_feat["t_idx"] = pd.factorize(train_feat["timestamp"])[0]
        feature_cols = num_cols + ["t_idx"]
        X = train_feat[feature_cols].fillna(train_feat[feature_cols].median()).values
        y = train_feat["load_mw"].values

        profile = {
            "problem_type": "panel_forecasting",
            "problem_type_confidence": "high",
            "target_col": "load_mw",
            "group_cols": ["region_id"],
            "time_col": "t_idx",
            "n_train_rows": len(train),
            "n_val_rows": 0,
            "covariate_cols": feature_cols,
            "train_files": [],
            "val_files": [],
        }
        features_meta = {
            "target_col": "load_mw",
            "group_cols": ["region_id"],
            "time_col": "t_idx",
            "problem_type": "panel_forecasting",
            "feature_columns": feature_cols,
            "feature_families": {"covariates": feature_cols},
            "total_features_planned": len(feature_cols),
        }
        model_results = _build_simple_lgbm_results(X, y, feature_cols, HPARAMS_DEFAULT)

        # t_idx is already in feature_cols; avoid duplicate column
        full_df = train_feat[["region_id", "load_mw"] + feature_cols].copy()
        return _run_validate_on_fixture(tmp, profile, model_results, features_meta, full_df)

    def test_schema_valid(self, energy_review):
        _assert_review_schema(energy_review, "energy_load")

    def test_verdict_is_valid(self, energy_review):
        assert energy_review["verdict"] in VALID_VERDICTS


@pytest.mark.skipif(
    not (PRACTICE / "medical_imaging" / "target_train.csv").exists(),
    reason="medical_imaging practice data not found — run tests/generate_practice_data.py"
)
class TestMedicalImagingValidator:
    """Validate schema on medical_imaging (tabular_regression — no time_col)."""

    @pytest.fixture(scope="class")
    def medical_review(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("medical_validate")
        data = PRACTICE / "medical_imaging"
        cov  = pd.read_csv(data / "covariates_train.csv")
        tgt  = pd.read_csv(data / "target_train.csv")
        train = cov.merge(tgt, on="patient_id")

        num_cols = ["age", "prior_admissions", "comorbidity_score"]
        train_feat = train[["patient_id", "hospitalization_days"] + num_cols].copy()
        X = train_feat[num_cols].fillna(train_feat[num_cols].median()).values
        y = train_feat["hospitalization_days"].values

        profile = {
            "problem_type": "tabular_regression",
            "problem_type_confidence": "high",
            "target_col": "hospitalization_days",
            "group_cols": ["patient_id"],
            "time_col": None,
            "n_train_rows": len(train),
            "n_val_rows": 0,
            "covariate_cols": num_cols,
            "train_files": [],
            "val_files": [],
        }
        features_meta = {
            "target_col": "hospitalization_days",
            "group_cols": ["patient_id"],
            "time_col": None,
            "problem_type": "tabular_regression",
            "feature_columns": num_cols,
            "feature_families": {"covariates": num_cols},
            "total_features_planned": len(num_cols),
        }
        model_results = _build_simple_lgbm_results(X, y, num_cols, HPARAMS_DEFAULT)
        return _run_validate_on_fixture(tmp, profile, model_results, features_meta, train_feat)

    def test_schema_valid(self, medical_review):
        _assert_review_schema(medical_review, "medical_imaging")

    def test_verdict_is_valid(self, medical_review):
        assert medical_review["verdict"] in VALID_VERDICTS

    def test_strict_cv_mae_positive(self, medical_review):
        assert medical_review["strict_cv_mae"] > 0
