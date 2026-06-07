"""critic.py — read-only CV consistency auditor.

Reads CV_PLAN, fold metrics, OOF predictions, per-fold importance, and the
validator review. Emits one structured verdict: CV_OK / CV_RISK / CV_INVALID.
Never modifies any artefact.
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _load(p: Path, parquet: bool = False):
    if not p.exists():
        return None
    return pd.read_parquet(p) if parquet else json.load(open(p))


def _escalate(a: str, b: str) -> str:
    order = {"CV_OK": 0, "CV_RISK": 1, "CV_INVALID": 2}
    return a if order[a] >= order[b] else b


def run(reports_dir: str | Path = "reports") -> dict:
    reports_dir = Path(reports_dir)
    plan = _load(reports_dir / "cv_plan.json") or {}
    folds = _load(reports_dir / "cv_folds.json") or {}
    fm = _load(reports_dir / "fold_metrics.json") or {}
    oof = _load(reports_dir / "oof_predictions.parquet", parquet=True)
    mr = _load(reports_dir / "model_results.json") or {}
    vr = _load(reports_dir / "validator_review.json") or {}

    checks: list[dict] = []
    verdict = "CV_OK"

    # Check 1 — fold stability ────────────────────────────────────────────────
    maes = list(fm.get("fold_maes") or [])
    if maes:
        mu, sd = float(np.mean(maes)), float(np.std(maes))
        cv = sd / max(mu, 1e-9)
        med = float(np.median(maes))
        if cv > 0.80 or (len(maes) >= 2 and med > 0 and max(maes) > 3 * med):
            s = "CV_INVALID"
        elif cv > 0.40:
            s = "CV_RISK"
        else:
            s = "CV_OK"
        checks.append({"name": "fold_stability", "status": s,
                       "details": f"cv={cv:.3f} n_folds={len(maes)} maes={maes}"})
        verdict = _escalate(verdict, s)

    # Check 2 — CV too good ──────────────────────────────────────────────────
    target = plan.get("target_column")
    train_std = None
    profile_p = reports_dir / "profile.json"
    if profile_p.exists() and target:
        prof = json.load(open(profile_p))
        train_std = (prof.get("schema", {}).get(target, {}) or {}).get("std")
    if train_std and fm.get("oof_mae") is not None:
        ratio = fm["oof_mae"] / max(train_std, 1e-9)
        if ratio < 0.01:
            s = "CV_INVALID"
        elif ratio < 0.05:
            s = "CV_RISK"
        else:
            s = "CV_OK"
        checks.append({"name": "cv_too_good", "status": s,
                       "details": f"oof_mae/train_std={ratio:.4f}"})
        verdict = _escalate(verdict, s)

    # Check 3 — importance stability across folds ──────────────────────────────
    try:
        from scipy.stats import spearmanr
        have_scipy = True
    except ImportError:
        have_scipy = False

    if have_scipy and folds.get("folds"):
        imps = []
        for f in folds["folds"]:
            k = f["fold_id"]
            p = reports_dir / f"importance_fold_{k}.json"
            if p.exists():
                d = json.load(open(p))
                # Prefer CatBoost importance; fall back to Ridge if CatBoost empty
                top = d.get("catboost_top") or d.get("ridge_top") or []
                imps.append({x["name"]: x["importance"] for x in top})
        if len(imps) >= 2:
            common = set.intersection(*[set(d.keys()) for d in imps])
            rhos = []
            if common and len(common) >= 2:
                for i in range(len(imps)):
                    for j in range(i + 1, len(imps)):
                        a = [imps[i][f] for f in common]
                        b = [imps[j][f] for f in common]
                        rho, _ = spearmanr(a, b)
                        if not np.isnan(rho):
                            rhos.append(float(rho))
            if rhos:
                med = float(np.median(rhos))
                s = ("CV_INVALID" if med < 0.2
                     else "CV_RISK" if med < 0.5 else "CV_OK")
                checks.append({"name": "importance_stability", "status": s,
                               "details": f"median_spearman={med:.3f} pairs={len(rhos)}"})
                verdict = _escalate(verdict, s)

    # Check 4 — prediction distribution per fold ────────────────────────────
    if isinstance(oof, pd.DataFrame) and not oof.empty:
        fold_bad = 0
        n_folds = oof["fold"].nunique()
        for _k, sub in oof.groupby("fold"):
            yt_mean = sub["y_true"].mean()
            yt_std = sub["y_true"].std() or 1.0
            if abs(sub["y_pred"].mean() - yt_mean) > 0.3 * (abs(yt_mean) or 1.0):
                fold_bad += 1
            elif sub["y_pred"].std() < 0.4 * yt_std:
                fold_bad += 1
        if n_folds:
            s = ("CV_INVALID" if fold_bad > n_folds / 2
                 else "CV_RISK" if fold_bad else "CV_OK")
            checks.append({"name": "pred_distribution", "status": s,
                           "details": f"{fold_bad}/{n_folds} folds with drift"})
            verdict = _escalate(verdict, s)

    # Check 5 — validator concordance ──────────────────────────────────────
    vv = vr.get("verdict", "PASS")
    s = ("CV_INVALID" if vv == "CRITICAL"
         else "CV_RISK" if vv == "WARNING" else "CV_OK")
    checks.append({"name": "validator_concordance", "status": s,
                   "details": f"validator verdict={vv}"})
    verdict = _escalate(verdict, s)

    review = {
        "plan_id": plan.get("plan_id"),
        "verdict": verdict,
        "checks": checks,
        "recommendation": (
            "proceed to submission" if verdict in ("CV_OK", "CV_RISK")
            else "replan requested — orchestrator should delete reports/cv_plan.json "
                 "and re-run scheme_analysis (at most one replan per run)"
        ),
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(reports_dir / "critic_review.json", "w") as f:
        json.dump(review, f, indent=2)

    if verdict == "CV_INVALID":
        with open(reports_dir / "critic_replan_requested.json", "w") as f:
            json.dump({
                "plan_id": plan.get("plan_id"),
                "reason": "CV_INVALID",
                "failing_checks": [c for c in checks if c["status"] == "CV_INVALID"],
            }, f, indent=2)

    with open(reports_dir / "critic_was_here.txt", "w") as f:
        f.write(f"critic executed at {review['created_at_utc']} verdict={verdict}\n")

    return review
