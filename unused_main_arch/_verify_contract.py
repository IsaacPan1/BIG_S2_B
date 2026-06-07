"""Smoke checks for the CV-as-contract pipeline.

Run from the repo root AFTER `python main.py`:
    python tools/_verify_contract.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

# Resolve all paths against the repo root, so this works regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"

from tools.cv_engine import CVEngine
from tools.modeler import LightGBMEstimator, XGBoostEstimator

# 1) Per-fold feature manifests declare train-only fit visibility
manifest_paths = sorted(REPORTS.glob("feature_manifest_fold_*.json"))
assert manifest_paths, f"No feature manifests under {REPORTS}"
for p in manifest_paths:
    m = json.load(open(p))
    assert m["fit_visibility"] == "train_idx only", f"{p.name} broke contract"
    print(f"{p.stem}: fit_visibility={m['fit_visibility']!r} "
          f"transformers={m['transformers']}")

# 2) cv_engine refuses unfrozen plans
plan = json.load(open(REPORTS / "cv_plan.json"))
plan_unfrozen = dict(plan)
plan_unfrozen["frozen"] = False
try:
    CVEngine(plan_unfrozen, pd.DataFrame({"week": list(range(10))}))
    print("BUG: unfrozen plan accepted")
except AssertionError as e:
    print(f"cv_engine refuses unfrozen plan: {e}")

# 3) Modeler refuses LightGBM/XGBoost backends
for cls in (LightGBMEstimator, XGBoostEstimator):
    try:
        cls().fit(None, None)
        print(f"BUG: {cls.__name__} ran")
    except NotImplementedError as e:
        print(f"{cls.__name__} correctly refuses: {e}")

# 4) Plan IDs match across the artefact chain
folds = json.load(open(REPORTS / "cv_folds.json"))
fm = json.load(open(REPORTS / "fold_metrics.json"))
cr = json.load(open(REPORTS / "critic_review.json"))
mr = json.load(open(REPORTS / "model_results.json"))
ids = {plan["plan_id"], folds["plan_id"], fm["plan_id"],
       cr["plan_id"], mr["plan_id"]}
assert len(ids) == 1, f"plan_id mismatch across artefacts: {ids}"
print(f"plan_id consistent across all artefacts: {plan['plan_id']}")
