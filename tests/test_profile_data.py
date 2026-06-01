"""Tests for tools/profile_data.py.

Run from the repo root:
    pytest tests/test_profile_data.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_profiler(data_dir: str, tmp_path: Path) -> dict:
    out = tmp_path / "profile.json"
    result = subprocess.run(
        [sys.executable, "tools/profile_data.py",
         "--data-dir", data_dir,
         "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"profiler exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return json.loads(out.read_text())


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------

class TestImageDetection:
    def test_retail_sales_no_images(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["image_data"]["present"] is False

    def test_energy_load_no_images(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert profile["image_data"]["present"] is False

    def test_medical_imaging_has_images(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        assert profile["image_data"]["present"] is True

    def test_medical_imaging_count(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        assert profile["image_data"]["n_images"] == 500, (
            f"expected 500, got {profile['image_data']['n_images']}"
        )

    def test_medical_imaging_extension(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        assert ".png" in profile["image_data"]["extensions"]

    def test_medical_imaging_sample_paths_exist(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        sample = profile["image_data"]["sample_paths"]
        assert len(sample) == 3
        assert all(Path(p).exists() for p in sample), (
            f"one or more sample paths do not exist: {sample}"
        )


# ---------------------------------------------------------------------------
# Linkage-column detection
# ---------------------------------------------------------------------------

class TestLinkageColumn:
    def test_medical_imaging_linkage_column(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        assert profile["image_data"]["linkage_column"] == "image_filename", (
            f"expected 'image_filename', got {profile['image_data']['linkage_column']!r}"
        )

    def test_retail_sales_no_linkage_column(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert "linkage_column" not in profile["image_data"]


# ---------------------------------------------------------------------------
# Schema / column stats (use new top-level schema dict)
# ---------------------------------------------------------------------------

class TestColumnSchema:
    def test_retail_sales_has_schema(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert "schema" in profile
        assert len(profile["schema"]) > 0

    def test_retail_sales_week_stats(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        week = profile["schema"]["week"]
        assert week["min"] == 0.0
        assert week["max"] == 89.0

    def test_retail_sales_train_row_count(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["n_train_rows"] == 135_000

    def test_retail_sales_val_row_count(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["n_val_rows"] == 15_000

    def test_energy_load_train_row_count(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert profile["n_train_rows"] == 12_480

    def test_all_schema_cols_have_role(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        for col, info in profile["schema"].items():
            assert "role" in info, f"column '{col}' missing 'role'"
            assert info["role"] in {
                "target", "time", "group", "id", "covariate"
            }, f"column '{col}' has unexpected role '{info['role']}'"


# ---------------------------------------------------------------------------
# Panel-forecasting classification
# ---------------------------------------------------------------------------

class TestPanelForecasting:
    """Both retail_sales and energy_load should be identified as panel_forecasting
    with high confidence, correct target, and at least one group column."""

    def test_retail_sales_problem_type(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["problem_type"] == "panel_forecasting", (
            f"expected panel_forecasting, got {profile['problem_type']}"
        )

    def test_retail_sales_confidence_high(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["problem_type_confidence"] == "high"

    def test_retail_sales_target_col(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["target_col"] == "weekly_sales", (
            f"expected 'weekly_sales', got {profile['target_col']!r}"
        )

    def test_retail_sales_time_col(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["time_col"] == "week", (
            f"expected 'week', got {profile['time_col']!r}"
        )

    def test_retail_sales_group_cols(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert len(profile["group_cols"]) >= 1
        assert "store_id" in profile["group_cols"]
        assert "product_id" in profile["group_cols"]

    def test_retail_sales_horizons(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["n_horizons"] == 10        # 10 val weeks (90-99)
        assert profile["horizons"] == list(range(1, 11))

    # --- energy_load ---

    def test_energy_load_problem_type(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert profile["problem_type"] == "panel_forecasting", (
            f"expected panel_forecasting, got {profile['problem_type']}"
        )

    def test_energy_load_confidence_high(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert profile["problem_type_confidence"] == "high"

    def test_energy_load_target_col(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert profile["target_col"] == "load_mw", (
            f"expected 'load_mw', got {profile['target_col']!r}"
        )

    def test_energy_load_group_cols(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert len(profile["group_cols"]) >= 1
        assert "region_id" in profile["group_cols"]

    def test_energy_load_horizons(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        # 4 val days × 24 hours = 96 unique timestamps
        assert profile["n_horizons"] == 96

    # --- shared invariants ---

    def test_distribution_shifts_is_list_retail(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert isinstance(profile["distribution_shifts"], list)

    def test_distribution_shifts_is_list_energy(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert isinstance(profile["distribution_shifts"], list)

    def test_distribution_shift_entries_have_required_keys(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        for entry in profile["distribution_shifts"]:
            assert "column"       in entry
            assert "ks_statistic" in entry
            assert "p_value"      in entry
            assert "flagged"      in entry

    def test_covariate_cols_non_empty(self, tmp_path):
        for dataset in ("practice_data/retail_sales", "practice_data/energy_load"):
            profile = run_profiler(dataset, tmp_path)
            assert len(profile["covariate_cols"]) >= 1, (
                f"{dataset}: expected at least one covariate_col"
            )


# ---------------------------------------------------------------------------
# Kaggle subfolder convention
# ---------------------------------------------------------------------------

class TestKaggleSubfolderConvention:
    """profile_data.py on the kaggle_style fixture (train/ + test/ subdirs)."""

    def test_convention_detected(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["file_paths"]["convention"] == "kaggle_subfolder"

    def test_train_data_path(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["file_paths"]["train_data"] == "train/covariates.csv"

    def test_train_target_path(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["file_paths"]["train_target"] == "train/dose_sys_train.csv"

    def test_val_features_path(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["file_paths"]["val_features"] == "test/covariates.csv"

    def test_target_column(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["target_col"] == "dose_sys"
        assert profile["file_paths"]["target_column"] == "dose_sys"

    def test_not_combined(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["file_paths"]["train_target_in_combined_file"] is False

    def test_images_dir_null(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["file_paths"]["images_dir"] is None

    def test_problem_type(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["problem_type"] == "panel_forecasting"

    def test_problem_type_confidence(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["problem_type_confidence"] == "high"

    def test_train_rows(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["n_train_rows"] == 50

    def test_val_rows(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["n_val_rows"] == 10

    def test_group_cols(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert "patient_id" in profile["group_cols"]

    def test_time_col(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert profile["time_col"] == "time_step"

    def test_train_files_have_subdir_prefix(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert all(f.startswith("train/") for f in profile["train_files"])

    def test_val_files_have_subdir_prefix(self, tmp_path):
        profile = run_profiler("practice_data/kaggle_style", tmp_path)
        assert all(f.startswith("test/") for f in profile["val_files"])


# ---------------------------------------------------------------------------
# Backward compat: existing conventions now expose 'convention' + 'images_dir'
# ---------------------------------------------------------------------------

class TestExistingConventionBackcompat:

    def test_retail_sales_convention(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["file_paths"]["convention"] == "split"

    def test_energy_load_convention(self, tmp_path):
        profile = run_profiler("practice_data/energy_load", tmp_path)
        assert profile["file_paths"]["convention"] == "combined"

    def test_medical_imaging_convention(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        assert profile["file_paths"]["convention"] == "split"

    def test_retail_images_dir_null(self, tmp_path):
        profile = run_profiler("practice_data/retail_sales", tmp_path)
        assert profile["file_paths"]["images_dir"] is None

    def test_medical_imaging_images_dir(self, tmp_path):
        profile = run_profiler("practice_data/medical_imaging", tmp_path)
        assert profile["file_paths"]["images_dir"] == "images"
