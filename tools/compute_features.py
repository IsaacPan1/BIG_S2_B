"""Dataset-agnostic feature computation — delegates to feature_engineering.py.

Previously used hardcoded Windows paths and retail_sales column names.
Now reads reports/profile.json for schema and file_paths, then delegates
to tools/feature_engineering.py (the canonical script).

Usage (from repo root):
    python tools/compute_features.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR    = REPO_ROOT / "data"

# ── Validate inputs before delegating ─────────────────────────────────────────
profile_path = REPORTS_DIR / "profile.json"
if not profile_path.exists():
    sys.exit(f"ERROR: {profile_path} not found — run profile_data.py first")

with open(profile_path) as f:
    profile = json.load(f)

fp = profile.get("file_paths", {})

def _check_file(fname: str | None, fallbacks: list[str]) -> str | None:
    for name in ([fname] if fname else []) + fallbacks:
        if name and (DATA_DIR / name).exists():
            return name
    return None

train_file = _check_file(fp.get("train_data"), ["covariates_train.csv", "train.csv"])
val_file   = _check_file(fp.get("val_features"), ["covariates_val.csv", "val_features.csv"])

if not train_file:
    sys.exit("ERROR: cannot locate training data file")
if not val_file:
    sys.exit("ERROR: cannot locate validation features file")

print(f"compute_features: train={train_file}, val={val_file}")
print("Delegating to tools/feature_engineering.py ...")

result = subprocess.run(
    [sys.executable, str(REPO_ROOT / "tools" / "feature_engineering.py")],
    cwd=str(REPO_ROOT),
)
sys.exit(result.returncode)
