#!/usr/bin/env python3
"""verify_cv_plan.py — Tier-1 acceptance check for the CV scheme.

Run AFTER `python tools/scheme_analysis.py` on whatever dataset is in data/.
It does NOT run feature engineering or modeling — it only verifies that the
plan scheme_analysis wrote actually materializes into valid folds, and that
those folds replicate the kind of split the problem type implies.

    python verify_cv_plan.py

Reads reports/cv_plan.json + the file(s) named in plan["data_source"] under data/.
Prints: planned vs materialized fold count, per-fold time windows + gap, and a
short list of automated sanity flags. A clean run with the expected fold count
and no flags is the real pass — not just that cv_plan.json parsed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.cv_engine import CVEngine, attach_period_rank  # noqa: E402

REPORTS = ROOT / "reports"
DATA = ROOT / "data"


def load_train_frame(plan: dict) -> pd.DataFrame:
    """Rebuild the training frame from plan['data_source'] (files joined on '+')."""
    src = plan.get("data_source", "")
    files = [s.strip() for s in src.split("+") if s.strip()]
    if not files:
        raise SystemExit(f"plan['data_source'] is empty or unparseable: {src!r}")
    frames = []
    for fn in files:
        p = DATA / fn
        if not p.exists():
            raise SystemExit(f"data_source file missing: {p}")
        frames.append(pd.read_csv(p))
    df = frames[0]
    for nxt in frames[1:]:
        on = [c for c in df.columns if c in nxt.columns]
        df = df.merge(nxt, on=on, how="left") if on else pd.concat([df, nxt], axis=1)
    return df.reset_index(drop=True)


def main() -> None:
    plan_path = REPORTS / "cv_plan.json"
    if not plan_path.exists():
        raise SystemExit("reports/cv_plan.json not found — run tools/scheme_analysis.py first")
    plan = json.loads(plan_path.read_text())

    profile_path = REPORTS / "profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}

    pt = plan.get("problem_type")
    cv = plan["cv"]
    cv_type = cv["cv_type"]
    planned = int(cv["n_splits"])
    tcol = plan.get("time_column")
    gcols = plan.get("group_columns") or []
    horizon = plan.get("horizon")

    print(f"plan_id        : {plan.get('plan_id')}")
    print(f"problem_type   : {pt}")
    print(f"cv_type        : {cv_type}")
    print(f"time_column    : {tcol}")
    print(f"group_columns  : {gcols}")
    print(f"horizon        : {horizon}   valid_size: {cv.get('valid_size')}   gap: {cv.get('gap')}")
    print(f"n_splits planned: {planned}")
    print("-" * 64)

    df_raw = load_train_frame(plan)
    df, axis_col = attach_period_rank(df_raw, profile, tcol)
    using_rank = axis_col != tcol and axis_col is not None
    print(f"frame rows     : {len(df)}   cols: {list(df.columns)[:8]}{'...' if df.shape[1] > 8 else ''}")
    if tcol:
        n_uniq = df[tcol].nunique()
        if using_rank:
            print(f"unique {tcol:<8}: {n_uniq}  (rank range "
                  f"{int(df[axis_col].min())}–{int(df[axis_col].max())})")
        else:
            print(f"unique {tcol:<8}: {n_uniq}  (range {df[tcol].min()}–{df[tcol].max()})")
    print("-" * 64)

    # The real test: does the engine produce valid folds? (split() raises on
    # overlap / time-leakage / group-overlap — a clean return is meaningful.)
    folds = CVEngine(plan, df, profile=profile).split()
    print(f"n_splits materialized: {len(folds)}")

    flags: list[str] = []
    if len(folds) < planned:
        flags.append(f"only {len(folds)}/{planned} folds materialized — not enough "
                     f"unique {tcol or 'rows'} for valid_size={cv.get('valid_size')}")

    axis = axis_col or tcol
    for k, (tr, va) in enumerate(folds):
        line = f"  fold {k}: n_train={len(tr):>6}  n_valid={len(va):>5}"
        if axis:
            tr_max = df[axis].iloc[tr].max()
            va_min = df[axis].iloc[va].min()
            va_max = df[axis].iloc[va].max()
            gap = va_min - tr_max
            line += f"  train≤{tr_max}  valid {va_min}–{va_max}  gap={gap}"
            if gap <= 0:
                flags.append(f"fold {k}: valid starts at/before train end (gap={gap}) — leakage")
        if set(tr.tolist()) & set(va.tolist()):
            flags.append(f"fold {k}: train/valid index overlap")
        print(line)

    # Problem-type vs cv-type sanity (cheap misclassification catcher)
    ts_families = {"TimeSeriesExpanding", "TimeSeriesSliding", "RollingOriginCV"}
    if pt in ("time_series", "grouped_time_series", "forecasting_multi_horizon") and cv_type not in ts_families:
        flags.append(f"problem_type={pt} but cv_type={cv_type} is not time-aware — likely wrong")
    if pt == "tabular_iid" and cv_type in ts_families:
        flags.append(f"problem_type=tabular_iid but cv_type={cv_type} is time-series — likely wrong")
    if "grouped" in str(pt) and not gcols:
        flags.append("problem_type mentions groups but group_columns is empty")

    # Forecasting fidelity: do the folds reach the deployment boundary?
    # The held-out test trains on ALL history and predicts the next block, so
    # the last fold should validate near the END of the series. A large unused
    # tail means CV measures an earlier / shorter-history regime than the test —
    # and never validates the most recent (most test-like) period.
    if axis and cv_type in ts_families and folds:
        t_max = df[axis].max()
        last_valid_max = df[axis].iloc[folds[-1][1]].max()
        unused_tail = t_max - last_valid_max
        vs = cv.get("valid_size") or 1
        if unused_tail > vs:
            flags.append(
                f"folds stop at {axis}={last_valid_max} but series ends at {t_max} "
                f"({unused_tail} unused at the tail) — folds do NOT reach the deployment "
                f"boundary; the most recent period is never validated. CV MAE will measure "
                f"a different regime than the held-out test. Use more splits or an "
                f"end-anchored window.")

    print("-" * 64)
    if flags:
        print("FLAGS:")
        for f in flags:
            print(f"  [!] {f}")
    else:
        print("OK — folds materialized cleanly, no sanity flags.")


if __name__ == "__main__":
    main()