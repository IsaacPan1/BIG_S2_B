"""Diversity sweep runner — read-only test harness for scheme_analysis.

For each dataset in DATASETS, runs the pre-feature-engineering sequence from a
CLEAN state and captures everything the report needs. Does NOT touch pipeline
logic. Writes results to _sweep_results.json for the reporting turn.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
PRACTICE = ROOT / "practice_data"
BACKUP = ROOT / "_data_backup_sweep"

DATASETS = ["award_A", "retail_sales", "kaggle_style", "medical_imaging", "energy_load"]


def run(cmd: list[str], cwd: Path = ROOT, timeout: int = 300) -> dict:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": " ".join(cmd),
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except subprocess.TimeoutExpired as e:
        return {"cmd": " ".join(cmd), "returncode": -1, "stdout": e.stdout or "",
                "stderr": f"TIMEOUT after {timeout}s"}
    except Exception as e:
        return {"cmd": " ".join(cmd), "returncode": -1, "stdout": "", "stderr": str(e)}


def wipe_data() -> None:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True, exist_ok=True)


def wipe_reports_and_parquet() -> None:
    if REPORTS.exists():
        for child in REPORTS.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # also nuke any engineered parquet so the drift gate cannot read one
    for p in (DATA.glob("features_*.parquet")):
        p.unlink()


def copy_practice(name: str) -> None:
    """Copy practice_data/<name>/* into data/, excluding _truth (held-out)."""
    src = PRACTICE / name
    for child in src.iterdir():
        if child.name == "_truth":
            continue
        dest = DATA / child.name
        if child.is_dir():
            shutil.copytree(child, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_read_error": str(e)}


def detect_tail_flag(verify_stdout: str) -> dict:
    flags = []
    in_flags = False
    silent = True
    for line in verify_stdout.splitlines():
        if line.startswith("FLAGS:"):
            in_flags = True
            silent = False
            continue
        if line.startswith("OK —") or line.startswith("OK -"):
            silent = True
        if in_flags and line.strip().startswith("[!]"):
            flags.append(line.strip())
    return {"tail_flag_silent": silent, "flags": flags}


def summarize_dataset(name: str) -> dict:
    print(f"\n[ {name} ] " + "=" * (60 - len(name)))
    wipe_reports_and_parquet()
    wipe_data()
    copy_practice(name)

    # Step 1 — profile
    r_profile = run([sys.executable, "tools/profile_data.py",
                     "--data-dir", "data/", "--output", "reports/profile.json"])
    profile = read_json(REPORTS / "profile.json") or {}

    # Step 2 — scheme_analysis
    r_scheme = run([sys.executable, "tools/scheme_analysis.py"])
    cv_plan = read_json(REPORTS / "cv_plan.json") or {}

    # Step 3 — verify_cv_plan (skip if non-time-series)
    cv = cv_plan.get("cv") or {}
    cv_type = cv.get("cv_type")
    skip_verify = cv_type in ("KFold", "StratifiedKFold")
    if skip_verify:
        r_verify = {"cmd": "(skipped)", "stdout": "", "stderr": "",
                    "returncode": 0, "skipped_reason": f"cv_type={cv_type} is not time-series"}
        verify_summary = {"tail_flag_silent": None, "flags": [], "skipped": True}
        fold_lines = []
    else:
        r_verify = run([sys.executable, "verify_cv_plan.py"])
        verify_summary = detect_tail_flag(r_verify.get("stdout", ""))
        fold_lines = [ln for ln in r_verify.get("stdout", "").splitlines()
                      if "fold " in ln and ("train" in ln or "valid" in ln)]

    reason = cv_plan.get("cv_selection_reason") or {}
    metrics = reason.get("drift_metrics") or {}

    return {
        "dataset": name,
        "profile_returncode": r_profile["returncode"],
        "profile_stderr_tail": r_profile["stderr"][-400:] if r_profile["stderr"] else "",
        "scheme_returncode": r_scheme["returncode"],
        "scheme_stderr_tail": r_scheme["stderr"][-400:] if r_scheme["stderr"] else "",
        "verify_returncode": r_verify.get("returncode"),
        "verify_stdout": r_verify.get("stdout", ""),
        "verify_skipped": skip_verify,
        "profile": {
            "problem_type": profile.get("problem_type"),
            "target_col":   profile.get("target_col"),
            "time_col":     profile.get("time_col"),
            "group_cols":   profile.get("group_cols"),
            "n_train_rows": profile.get("n_train_rows"),
            "n_val_rows":   profile.get("n_val_rows"),
            "train_files":  profile.get("train_files"),
            "val_files":    profile.get("val_files"),
        },
        "cv_plan": {
            "problem_type":    cv_plan.get("problem_type"),
            "problem_subtype": cv_plan.get("problem_subtype"),
            "target_column":   cv_plan.get("target_column"),
            "time_column":     cv_plan.get("time_column"),
            "group_columns":   cv_plan.get("group_columns"),
            "horizon":         cv_plan.get("horizon"),
            "cv":              cv,
        },
        "cv_selection_reason": {
            "selected_cv_type":         reason.get("selected_cv_type"),
            "default_for_problem_type": reason.get("default_for_problem_type"),
            "rule_branch":              reason.get("rule_branch"),
            "thresholds":               reason.get("thresholds"),
            "drift_metrics":            metrics,
        },
        "fold_lines": fold_lines,
        "tail_flag_silent": verify_summary.get("tail_flag_silent"),
        "verify_flags":     verify_summary.get("flags"),
    }


def main() -> None:
    # Backup current data/
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    if DATA.exists():
        shutil.move(str(DATA), str(BACKUP))
    print(f"backed up data/ -> {BACKUP}")

    try:
        results: list[dict] = []
        for name in DATASETS:
            try:
                results.append(summarize_dataset(name))
            except Exception as e:
                results.append({"dataset": name, "harness_error": repr(e)})

        # (d) FLIP TEST on award_A: same dataset, but with the previously-engineered
        # parquet present so the drift diagnostic uses engineered_parquet instead of
        # raw_csv. Use the parquet we have in BACKUP (the prior run's engineered set).
        flip = {"dataset": "award_A__with_parquet"}
        try:
            wipe_reports_and_parquet()
            wipe_data()
            copy_practice("award_A")
            # Copy the previously-engineered parquets from BACKUP into data/
            for parq in ("features_train.parquet", "features_val.parquet"):
                src = BACKUP / parq
                if src.exists():
                    shutil.copy2(src, DATA / parq)
            flip["parquets_present"] = [p.name for p in DATA.glob("features_*.parquet")]
            run([sys.executable, "tools/profile_data.py",
                 "--data-dir", "data/", "--output", "reports/profile.json"])
            run([sys.executable, "tools/scheme_analysis.py"])
            cv_plan = read_json(REPORTS / "cv_plan.json") or {}
            reason = cv_plan.get("cv_selection_reason") or {}
            flip["cv_type"] = (cv_plan.get("cv") or {}).get("cv_type")
            flip["rule_branch"] = reason.get("rule_branch")
            flip["drift_metrics"] = reason.get("drift_metrics") or {}
        except Exception as e:
            flip["harness_error"] = repr(e)

        out = {"results": results, "flip_test": flip}
        (ROOT / "_sweep_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote _sweep_results.json with {len(results)} dataset entries + flip test.")
    finally:
        # Restore data/
        if DATA.exists():
            shutil.rmtree(DATA)
        shutil.move(str(BACKUP), str(DATA))
        print(f"restored data/ from {BACKUP}")


if __name__ == "__main__":
    main()
