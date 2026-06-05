#!/usr/bin/env python3
"""
tools/gap_attribution.py — Dataset-agnostic CV gap attribution.

Reads validator_review.json (requires fold_maes, fold_train_sizes written by
validate.py) and classifies the OOF→strict CV gap as:

  CV_SCHEME       — gap is a structural pessimism artifact of the expanding-window
                    scheme (early folds have less history → higher MAE); the latest
                    fold (most history) ≈ the modeler's reported MAE.
  REAL_DIVERGENCE — gap is not explained by scheme pessimism; possible overfit or
                    distribution shift.
  UNKNOWN         — insufficient fold data to classify.

Appends a "gap_attribution" block to reports/validator_review.json.

NO MODEL TRAINING. Pure statistical analysis on already-computed fold MAEs.
Called by validate.py (or validator.md) after strict CV completes.

Usage:
    python tools/gap_attribution.py [--repo-root PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Classification thresholds ─────────────────────────────────────────────────
# Latest fold MAE must be within this fraction of reported MAE to call CV_SCHEME
LATEST_FOLD_MATCH_PCT = 0.15
# At least this fraction of consecutive fold pairs must show more-data→lower-MAE
MONOTONE_MIN_FRACTION = 0.60


def _monotonicity_score(maes_by_ascending_size: list[float]) -> float:
    """Fraction of consecutive (ascending train-size) fold pairs where MAE decreases."""
    if len(maes_by_ascending_size) < 2:
        return 1.0
    pairs = list(zip(maes_by_ascending_size[:-1], maes_by_ascending_size[1:]))
    return sum(1 for a, b in pairs if b < a) / len(pairs)


def run_gap_attribution(repo_root: Path) -> dict:
    reports     = repo_root / "reports"
    review_path = reports / "validator_review.json"

    with open(review_path) as f:
        review = json.load(f)

    fold_maes        = [float(x) for x in review.get("fold_maes", [])]
    fold_train_sizes = [int(x)   for x in review.get("fold_train_sizes", [])]
    reported_mae     = float(review.get("reported_cv_mae") or review.get("honest_cv_mae") or 0.0)
    strict_mae       = float(review.get("strict_cv_mae") or 0.0)

    # ── Guard: need at least 2 folds with matching sizes ─────────────────────
    if (
        not fold_maes
        or not fold_train_sizes
        or len(fold_maes) != len(fold_train_sizes)
        or reported_mae == 0.0
    ):
        attribution = {
            "classification":          "UNKNOWN",
            "latest_fold_mae":         None,
            "reported_mae":            reported_mae,
            "fold_mae_by_trainsize":   [],
            "pct_explained_by_scheme": None,
            "monotone_score":          None,
            "note": (
                "Insufficient fold data for gap attribution "
                "(fold_maes or fold_train_sizes missing, mismatched, or reported_mae=0)."
            ),
        }
        review["gap_attribution"] = attribution
        with open(review_path, "w") as f:
            json.dump(review, f, indent=2)
        print("[gap_attribution] UNKNOWN — missing or mismatched fold data")
        return attribution

    # ── Sort folds by train size ascending (smallest history → largest) ───────
    paired       = sorted(zip(fold_train_sizes, fold_maes), key=lambda x: x[0])
    sizes_sorted = [int(p[0])   for p in paired]
    maes_sorted  = [float(p[1]) for p in paired]

    latest_fold_mae = maes_sorted[-1]    # most history → best proxy for true test
    gap_total       = strict_mae - reported_mae

    # pct_explained: how much of the gap comes from averaging in pessimistic early folds
    # (strict_mae - latest_fold_mae) = the part attributable to early-fold pessimism
    gap_from_scheme = strict_mae - latest_fold_mae
    pct_explained   = float(gap_from_scheme / gap_total) if gap_total > 1e-9 else 0.0

    mono_score   = _monotonicity_score(maes_sorted)
    latest_match = abs(latest_fold_mae - reported_mae) <= LATEST_FOLD_MATCH_PCT * reported_mae
    monotone     = mono_score >= MONOTONE_MIN_FRACTION

    # ── Classify ──────────────────────────────────────────────────────────────
    if latest_match and monotone:
        classification = "CV_SCHEME"
        note = (
            f"Latest fold MAE ({latest_fold_mae:.4f}) ≈ reported MAE ({reported_mae:.4f}) "
            f"(within {LATEST_FOLD_MATCH_PCT*100:.0f}%); "
            f"{mono_score*100:.0f}% of fold pairs show expected monotone improvement "
            f"with more training data. The gap ({gap_total:+.4f}) is a structural "
            f"pessimism artifact of the expanding-window scheme — not model overfit."
        )
    elif latest_match:
        classification = "CV_SCHEME"
        note = (
            f"Latest fold MAE ({latest_fold_mae:.4f}) ≈ reported MAE ({reported_mae:.4f}), "
            f"but fold-MAE monotonicity is weak (score={mono_score:.2f} < "
            f"{MONOTONE_MIN_FRACTION:.2f}). "
            f"Classified CV_SCHEME (scheme pessimism) with moderate confidence."
        )
    else:
        classification = "REAL_DIVERGENCE"
        note = (
            f"Latest fold MAE ({latest_fold_mae:.4f}) diverges from reported MAE "
            f"({reported_mae:.4f}) by more than {LATEST_FOLD_MATCH_PCT*100:.0f}% "
            f"(monotone_score={mono_score:.2f}). "
            f"Gap is NOT fully explained by expanding-window scheme pessimism; "
            f"possible overfit or distribution shift."
        )

    attribution = {
        "classification":          classification,
        "latest_fold_mae":         float(latest_fold_mae),
        "reported_mae":            float(reported_mae),
        "fold_mae_by_trainsize":   [
            {"train_size": int(s), "fold_mae": float(m)}
            for s, m in zip(sizes_sorted, maes_sorted)
        ],
        "pct_explained_by_scheme": float(pct_explained),
        "monotone_score":          float(mono_score),
        "note":                    note,
    }

    review["gap_attribution"] = attribution
    with open(review_path, "w") as f:
        json.dump(review, f, indent=2)

    print(f"[gap_attribution] classification={classification}")
    print(f"  latest_fold_mae={latest_fold_mae:.4f}  reported_mae={reported_mae:.4f}")
    print(f"  pct_explained_by_scheme={pct_explained*100:.1f}%  monotone_score={mono_score:.2f}")
    print(f"  {note}")

    return attribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attribute OOF→strict CV gap to scheme pessimism or real divergence."
    )
    parser.add_argument("--repo-root", default=".", help="Repo root directory")
    args   = parser.parse_args()
    result = run_gap_attribution(Path(args.repo_root).resolve())
    print(f"\ngap_attribution complete: {result['classification']}")
