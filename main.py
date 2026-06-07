"""main.py — CV-as-Contract pipeline entry point.

Execution order (deterministic, file-based contracts between stages):

    tools.scheme_analysis.write_cv_plan    -> reports/cv_plan.json (frozen)
    tools.cv_engine.materialise_folds      -> reports/cv_folds.json
    tools.feature_engineer.run_per_fold    -> data/features_*_fold_{k}.parquet
                                              + reports/feature_manifest_fold_{k}.json
    tools.modeler.run_per_fold             -> reports/predictions_fold_{k}.parquet
                                              + reports/importance_fold_{k}.json
                                              + reports/model_results.json
    tools.cv_engine.aggregate_oof          -> reports/oof_predictions.parquet
                                              + reports/fold_metrics.json
                                              + reports/validator_review.json
    tools.critic.run                       -> reports/critic_review.json
    tools.report_writer.run                -> report.pdf (or report.txt)
    _write_submission_best_effort          -> submission.csv (best-effort)

Run with:
    python main.py

The entrypoint resolves all paths relative to the repo root so it works
regardless of CWD, and wipes reports/ at the start of every run so no
downstream stage can ever read a stale artefact from a previous domain.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo-root anchor ────────────────────────────────────────────────────────
# Every path in this file is derived from ROOT so `python main.py` works from
# any CWD (per the layout contract).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import scheme_analysis, cv_engine, feature_engineer, modeler, critic, report_writer


# ─────────────────────────────────────────────────────────────────────────────
# reports/ hygiene
# ─────────────────────────────────────────────────────────────────────────────

def _reset_reports_dir(reports_dir: Path) -> None:
    """Wipe reports/ contents (preserve .gitkeep) and recreate empty.

    The pipeline must NEVER read a stale artefact from a previous run/domain.
    Every reports/* file consumed downstream is produced earlier in the same
    run.
    """
    if reports_dir.exists():
        for entry in reports_dir.iterdir():
            if entry.name == ".gitkeep":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Maintain the .gitkeep marker so the empty folder is tracked.
    gitkeep = reports_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def _reset_generated_feature_parquets(data_dir: Path) -> None:
    """Remove fold-bound feature parquets from any previous run.

    These live under data/ (an input directory) and are regenerated per fold.
    They must not survive across runs/domains.
    """
    for p in data_dir.glob("features_train_fold_*.parquet"):
        p.unlink()
    for p in data_dir.glob("features_valid_fold_*.parquet"):
        p.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading (canonical training frame)
# ─────────────────────────────────────────────────────────────────────────────

def _load_canonical_train_frame(plan: dict, data_dir: Path) -> pd.DataFrame:
    """Reconstruct the training frame the CV_PLAN was built against.

    Dataset-agnostic: reads whatever `plan['data_source']` declares. Falls back
    to the first non-validation CSV found in data/.
    """
    src = plan.get("data_source", "")
    if src == "covariates_train.csv+target_train.csv":
        df_cov = pd.read_csv(data_dir / "covariates_train.csv")
        df_tgt = pd.read_csv(data_dir / "target_train.csv")
        join_keys = [c for c in df_cov.columns if c in df_tgt.columns]
        return df_cov.merge(df_tgt, on=join_keys, how="left")

    p = data_dir / src
    if p.exists():
        return pd.read_csv(p)

    for p in sorted(data_dir.glob("*.csv")):
        name = p.name.lower()
        if any(x in name for x in ("submission", "sample", "test", "val")):
            continue
        return pd.read_csv(p)

    raise FileNotFoundError(f"Cannot reload training frame for plan {plan['plan_id']}")


# ─────────────────────────────────────────────────────────────────────────────
# Submission (best-effort full-train final model)
# ─────────────────────────────────────────────────────────────────────────────

def _write_submission_best_effort(plan: dict, data_dir: Path,
                                  out_path: Path) -> str | None:
    """Train CatBoost+Ridge on the full training frame and predict the
    validation file referenced by data/sample_submission.csv.

    The CV-as-contract pipeline's primary deliverables are the OOF and the
    report; this hook produces a working submission.csv for downstream graders.
    Dataset-agnostic: column names + join keys are derived from the actual
    files, not hardcoded.
    """
    sample = data_dir / "sample_submission.csv"
    cov_val = data_dir / "covariates_val.csv"
    if not (sample.exists() and cov_val.exists()):
        return None

    df_train = _load_canonical_train_frame(plan, data_dir)
    df_val = pd.read_csv(cov_val)
    target = plan["target_column"]
    if target not in df_train.columns:
        return None

    transformers = feature_engineer.build_transformers(plan, df_train)
    train_blocks, valid_blocks = [], []
    for tx in transformers:
        tx.fit(df_train)
        train_blocks.append(tx.transform(df_train))
        valid_blocks.append(tx.transform(df_val))
    feat_train = pd.concat(train_blocks, axis=1)
    feat_valid = pd.concat(valid_blocks, axis=1)
    feat_train = feat_train.loc[:, ~feat_train.columns.duplicated()]
    feat_valid = feat_valid.loc[:, ~feat_valid.columns.duplicated()]
    feat_valid = feat_valid.reindex(columns=feat_train.columns, fill_value=0.0)

    X_train = feat_train.fillna(0.0)
    y_train = df_train[target]
    X_valid = feat_valid.fillna(0.0)

    cb_pred, ridge_pred = None, None
    try:
        cb = modeler.CatBoostEstimator(
            problem_subtype=plan.get("problem_subtype", "continuous_regression"),
            cat_indices=[],
        ).fit(X_train, y_train)
        cb_pred = cb.predict(X_valid)
    except Exception as e:
        print(f"[submission] CatBoost failed: {e}")
    try:
        r = modeler.RidgeEstimator(alpha=1.0).fit(X_train, y_train)
        ridge_pred = r.predict(X_valid)
    except Exception as e:
        print(f"[submission] Ridge failed: {e}")

    if cb_pred is not None and ridge_pred is not None:
        blend = 0.5 * cb_pred + 0.5 * ridge_pred
    elif cb_pred is not None:
        blend = cb_pred
    elif ridge_pred is not None:
        blend = ridge_pred
    else:
        blend = np.full(len(X_valid), float(y_train.mean()))
    if (y_train >= 0).all():
        blend = np.clip(blend, 0, None)

    sample_df = pd.read_csv(sample)
    pred_col = target if target in sample_df.columns else sample_df.columns[-1]
    join_keys = [c for c in sample_df.columns if c in df_val.columns and c != pred_col]
    df_val_with_pred = df_val.copy()
    df_val_with_pred["__pred__"] = blend
    if join_keys:
        merged = sample_df[join_keys].merge(
            df_val_with_pred[join_keys + ["__pred__"]],
            on=join_keys, how="left",
        )
        sample_df[pred_col] = merged["__pred__"].fillna(float(y_train.mean())).values
    else:
        if len(sample_df) == len(blend):
            sample_df[pred_col] = blend
        else:
            sample_df[pred_col] = float(y_train.mean())
    sample_df.to_csv(out_path, index=False)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: Path = None, reports_dir: Path = None,
        submission_path: Path = None, report_path: Path = None,
        write_submission: bool = True) -> dict:
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    reports_dir = Path(reports_dir) if reports_dir else ROOT / "reports"
    submission_path = Path(submission_path) if submission_path else ROOT / "submission.csv"
    report_path = Path(report_path) if report_path else ROOT / "report.pdf"

    # ── reports/ hygiene: clean slate, no stale artefacts from a prior run ──
    _reset_reports_dir(reports_dir)
    _reset_generated_feature_parquets(data_dir)
    # Also remove the prior run's submission/report at the root.
    if submission_path.exists():
        submission_path.unlink()
    if report_path.exists():
        report_path.unlink()
    txt_fallback = report_path.with_suffix(".txt")
    if txt_fallback.exists():
        txt_fallback.unlink()

    t0 = time.time()
    print("=" * 70)
    print("STAGE 1 — scheme_analysis: writing CV_PLAN (frozen contract)")
    print("=" * 70)
    plan = scheme_analysis.write_cv_plan(data_dir, reports_dir)
    print(f"  plan_id={plan['plan_id']}")
    print(f"  cv_type={plan['cv']['cv_type']}  n_splits={plan['cv']['n_splits']}")
    print(f"  target={plan['target_column']}  time={plan['time_column']}  "
          f"groups={plan['group_columns']}")
    print(f"  leakage_risks={[r['id'] for r in plan['leakage_risks']]}")

    print()
    print("=" * 70)
    print("STAGE 2 — cv_engine: materialise folds from CV_PLAN")
    print("=" * 70)
    df_train = _load_canonical_train_frame(plan, data_dir)
    print(f"  training frame loaded: shape={df_train.shape}")
    folds_payload = cv_engine.materialise_folds(plan, df_train, reports_dir)
    print(f"  produced {folds_payload['n_splits']} folds")
    for f in folds_payload["folds"]:
        print(f"    fold {f['fold_id']}: n_train={f['n_train']} n_valid={f['n_valid']}")

    print()
    print("=" * 70)
    print("STAGE 3 — feature_engineer: fold-bound transformations")
    print("=" * 70)
    fe = feature_engineer.run_per_fold(plan, folds_payload, df_train,
                                       data_dir, reports_dir)
    print(f"  feature columns ({len(fe['feature_columns'])}): "
          f"{fe['feature_columns'][:8]}{'...' if len(fe['feature_columns'])>8 else ''}")

    print()
    print("=" * 70)
    print("STAGE 4 — modeler: CatBoost + Ridge per fold")
    print("=" * 70)
    summary = modeler.run_per_fold(plan, folds_payload, data_dir, reports_dir)
    for r in summary["per_fold"]:
        print(f"  fold {r['fold_id']}: blend_mae={r['mae']:.4f} "
              f"cb={r.get('catboost_mae')} ridge={r.get('ridge_mae')}")

    print()
    print("=" * 70)
    print("STAGE 5 — cv_engine.aggregate_oof: OOF + fold metrics")
    print("=" * 70)
    agg = cv_engine.aggregate_oof(plan, folds_payload, reports_dir)
    m = agg["metrics"]
    print(f"  OOF MAE = {m['oof_mae']}")
    print(f"  fold MAE mean/std = {m['fold_mae_mean']} / {m['fold_mae_std']}")
    print(f"  validator verdict = {agg['review']['verdict']}")

    print()
    print("=" * 70)
    print("STAGE 6 — critic: CV consistency audit")
    print("=" * 70)
    review = critic.run(reports_dir)
    print(f"  critic verdict = {review['verdict']}")
    for c in review["checks"]:
        print(f"    - {c['name']}: {c['status']}")

    print()
    print("=" * 70)
    print("STAGE 7 — report_writer: report.pdf")
    print("=" * 70)
    written_report = report_writer.run(reports_dir, report_path)
    print(f"  wrote {written_report}")

    written_submission = None
    if write_submission:
        print()
        print("=" * 70)
        print("STAGE 8 — submission writer (best-effort, full-train final model)")
        print("=" * 70)
        try:
            written_submission = _write_submission_best_effort(
                plan, data_dir, submission_path,
            )
            print(f"  wrote {written_submission}")
        except Exception as e:
            print(f"  submission writer failed (non-fatal): {e}")

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  plan_id={plan['plan_id']} oof_mae={m['oof_mae']} "
          f"critic={review['verdict']}")
    print("=" * 70)

    return {
        "plan_id": plan["plan_id"],
        "oof_mae": m["oof_mae"],
        "validator_verdict": agg["review"]["verdict"],
        "critic_verdict": review["verdict"],
        "report_path": written_report,
        "submission_path": written_submission,
        "elapsed_seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="CV-as-Contract pipeline")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--submission", default=str(ROOT / "submission.csv"))
    ap.add_argument("--report", default=str(ROOT / "report.pdf"))
    ap.add_argument("--no-submission", action="store_true",
                    help="Skip the final submission.csv step")
    args = ap.parse_args()
    run(
        data_dir=Path(args.data_dir),
        reports_dir=Path(args.reports_dir),
        submission_path=Path(args.submission),
        report_path=Path(args.report),
        write_submission=not args.no_submission,
    )
    # Note: CV_INVALID does NOT block submission; it signals a replan to the
    # orchestrator. Exit 0 unconditionally.
    sys.exit(0)


if __name__ == "__main__":
    main()
