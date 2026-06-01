#!/usr/bin/env python3
"""Run full pipeline (profile + features + modeler) on a practice dataset.

Usage:
    python tests/run_practice_pipeline.py --dataset retail_sales
    python tests/run_practice_pipeline.py --dataset energy_load
    python tests/run_practice_pipeline.py --dataset medical_imaging
    python tests/run_practice_pipeline.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
TOOLS_DIR = REPO / "tools"
PRACTICE_DIR = REPO / "practice_data"


def _patch_modeler(src: Path, dst: Path, base_dir: str) -> None:
    """Copy run_modeler.py with BASE_DIR patched to the temp directory."""
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("BASE_DIR"):
            # Use forward slashes to avoid regex/escape issues on Windows
            safe = base_dir.replace("\\", "/")
            out.append(f'BASE_DIR = r"{safe}"\n')
        else:
            out.append(line)
    dst.write_text("".join(out), encoding="utf-8")


def _score(pred_csv: Path, truth_csv: Path, target_col: str, key_cols: list[str]) -> float:
    truth = pd.read_csv(truth_csv)
    preds = pd.read_csv(pred_csv)
    # Rename predicted_target to target_col for merge
    if "predicted_target" in preds.columns and target_col not in preds.columns:
        preds = preds.rename(columns={"predicted_target": target_col})
    merged = truth.merge(preds[key_cols + [target_col]], on=key_cols, suffixes=("_truth", "_pred"))
    y_true = merged[f"{target_col}_truth"].to_numpy(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy(float)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    return float(np.mean(np.abs(y_true - y_pred)))


_DATASET_META = {
    "retail_sales": {
        "target_col": "weekly_sales",
        "key_cols": ["store_id", "product_id", "week"],
    },
    "energy_load": {
        "target_col": "load_mw",
        "key_cols": ["region_id", "timestamp"],
    },
    "medical_imaging": {
        "target_col": "hospitalization_days",
        "key_cols": ["patient_id"],
    },
    "kaggle_style": {
        "target_col": "dose_sys",
        "key_cols": ["patient_id", "time_step"],
    },
}


def run_dataset(
    dataset: str,
    verbose: bool = True,
    run_modeler: bool = True,
    run_build_submission: bool = False,
) -> dict:
    src = PRACTICE_DIR / dataset
    if not src.exists():
        print(f"ERROR: practice_data/{dataset} not found")
        return {"dataset": dataset, "error": "not found"}

    with tempfile.TemporaryDirectory(prefix=f"award_b_{dataset}_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "data").mkdir()
        (tmp_path / "reports").mkdir()
        (tmp_path / "tools").mkdir()

        # Copy tools (patch run_modeler.py with correct base_dir)
        for f in TOOLS_DIR.glob("*.py"):
            if f.name == "run_modeler.py":
                _patch_modeler(f, tmp_path / "tools" / f.name, str(tmp_path))
            else:
                shutil.copy(f, tmp_path / "tools" / f.name)

        # Copy practice data into data/ (skip _truth dir)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy(item, tmp_path / "data" / item.name)
            elif item.is_dir() and item.name != "_truth":
                shutil.copytree(item, tmp_path / "data" / item.name)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  DATASET: {dataset}")
            print(f"{'='*60}")

        # ── Step 1: profile_data.py ───────────────────────────────────────────
        r = subprocess.run(
            [sys.executable, str(tmp_path / "tools" / "profile_data.py"),
             "--data-dir", str(tmp_path / "data"),
             "--output", str(tmp_path / "reports" / "profile.json")],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        if r.returncode != 0:
            print(f"  profile_data.py FAILED:\n{r.stderr[-1500:]}")
            return {"dataset": dataset, "error": "profile failed"}

        with open(tmp_path / "reports" / "profile.json") as fh:
            profile = json.load(fh)

        if verbose:
            print(f"  problem_type : {profile.get('problem_type')}")
            print(f"  group_cols   : {profile.get('group_cols')}")
            print(f"  time_col     : {profile.get('time_col')}")
            print(f"  target_col   : {profile.get('target_col')}")

        # ── Step 2: feature_engineering.py ───────────────────────────────────
        r = subprocess.run(
            [sys.executable, str(tmp_path / "tools" / "feature_engineering.py")],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        if r.returncode != 0:
            print(f"  feature_engineering.py FAILED:\n{r.stdout[-1500:]}\n{r.stderr[-500:]}")
            return {"dataset": dataset, "error": "feature_engineering failed"}

        if verbose:
            for line in r.stdout.splitlines():
                if any(kw in line for kw in ["granularity", "Detected", "seasonality",
                                              "Total feature", "COMPLETE", "hourly", "daily"]):
                    print(f"  FE >> {line}")

        feat_json_path = tmp_path / "reports" / "features.json"
        with open(feat_json_path) as fh:
            features = json.load(fh)

        n_features = features.get("total_features_planned", 0)
        seasonality = features.get("feature_families", {}).get("seasonality", [])
        if verbose:
            print(f"  features      : {n_features}")
            print(f"  seasonality   : {seasonality}")

        result = {
            "dataset": dataset,
            "problem_type": profile.get("problem_type"),
            "total_features": n_features,
            "seasonality": seasonality,
        }

        if not run_modeler:
            return result

        # ── Step 3: run_modeler.py ────────────────────────────────────────────
        if verbose:
            print(f"  Running modeler (this may take a few minutes)…")

        r = subprocess.run(
            [sys.executable, str(tmp_path / "tools" / "run_modeler.py")],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=600,
        )
        if r.returncode != 0:
            print(f"  run_modeler.py FAILED:\n{r.stdout[-2000:]}\n{r.stderr[-500:]}")
            result["error"] = "modeler failed"
            return result

        if verbose:
            for line in r.stdout.splitlines():
                if any(kw in line for kw in ["MAE", "OOF", "CV", "val", "COMPLETE", "Trial"]):
                    print(f"  MOD >> {line}")

        # ── Step 4 (optional): build_submission.py ────────────────────────────
        if run_build_submission:
            r = subprocess.run(
                [sys.executable, str(tmp_path / "tools" / "build_submission.py"),
                 "--repo-root", str(tmp_path)],
                capture_output=True, text=True, cwd=str(tmp_path), timeout=60,
            )
            if r.returncode != 0:
                print(f"  build_submission.py FAILED:\n{r.stdout[-1000:]}\n{r.stderr[-500:]}")
                result["error"] = "build_submission failed"
                return result
            sub_csv = tmp_path / "submission.csv"
            if not sub_csv.exists():
                result["error"] = "submission.csv not produced"
                return result
            result["submission_produced"] = True
            if verbose:
                print(f"  submission.csv   : {sub_csv.stat().st_size} bytes")

        # ── Step 5: Score against truth ───────────────────────────────────────
        pred_csv = tmp_path / "reports" / "predictions.csv"
        truth_csv = src / "_truth" / "val_truth.csv"
        if not pred_csv.exists():
            result["error"] = "predictions.csv missing"
            return result

        meta = _DATASET_META[dataset]
        if not truth_csv.exists():
            if verbose:
                print(f"  (no _truth/ dir — skipping MAE scoring)")
            return result

        try:
            mae = _score(pred_csv, truth_csv, meta["target_col"], meta["key_cols"])
            result["mae"] = round(mae, 4)
            if verbose:
                print(f"  Truth MAE     : {mae:.4f}")
        except Exception as e:
            result["error"] = f"scoring failed: {e}"

        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(_DATASET_META.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--features-only", action="store_true",
                        help="Skip modeler, just check feature engineering")
    parser.add_argument("--build-submission", action="store_true",
                        help="Also run build_submission.py and verify submission.csv")
    args = parser.parse_args()

    datasets = (
        list(_DATASET_META.keys()) if args.all else [args.dataset]
    )

    results = []
    for ds in datasets:
        # kaggle_style always runs build_submission to verify the full pipeline
        do_sub = args.build_submission or ds == "kaggle_style"
        r = run_dataset(ds, verbose=True, run_modeler=not args.features_only,
                        run_build_submission=do_sub and not args.features_only)
        results.append(r)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for r in results:
        status = "ERROR" if "error" in r else "OK"
        mae_str = f"  MAE={r['mae']:.4f}" if "mae" in r else ""
        sub_str = "  submission=yes" if r.get("submission_produced") else ""
        print(f"  {r['dataset']:25s}: {status}  features={r.get('total_features','?')}{mae_str}{sub_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
