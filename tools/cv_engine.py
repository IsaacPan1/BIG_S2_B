"""cv_engine.py — CVEngine + OOF aggregator.

Owns the runtime translation of CV_PLAN into fold indices and the post-modeling
stitching of out-of-fold predictions. Refuses to operate without a frozen
CV_PLAN. No feature engineering, no model fitting.
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Period-rank chokepoint
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_RANK_COL = "__period_rank__"


def attach_period_rank(
    df: pd.DataFrame, profile: dict | None, time_col: str | None
) -> tuple[pd.DataFrame, str | None]:
    """Resolve an opaque period_id column to a chronological dense integer rank.

    This is the ONLY place id_to_rank lookups happen. Downstream code reads
    `axis_col` from the return and uses df[axis_col] for ordering, leakage
    checks, and any arithmetic on the time axis — never the raw hash.

    Returns (df, axis_col).
        - If profile["period_rank_info"]["available"] is True, returns a copy
          of df with an int column __period_rank__ added; axis_col is the new
          column name. The raw time_col stays unchanged (it is the join /
          submission key, not the ordering axis).
        - Otherwise returns (df, time_col) so codebook-less datasets keep their
          existing single-axis behaviour.
        - Idempotent: if df already has __period_rank__, returns it as-is.

    Raises ValueError when rank_info is available but any df period_id is
    missing from id_to_rank — NaN ranks must never silently flow into CV math.
    """
    if _PERIOD_RANK_COL in df.columns:
        return df, _PERIOD_RANK_COL

    pri = (profile or {}).get("period_rank_info") or {}
    if not pri.get("available") or not pri.get("id_to_rank"):
        return df, time_col

    if not time_col or time_col not in df.columns:
        return df, time_col

    id_to_rank: dict = pri["id_to_rank"]
    ids = df[time_col].astype(str)
    ranks = ids.map(id_to_rank)
    nan_mask = ranks.isna()
    if nan_mask.any():
        missing = sorted(set(ids[nan_mask].tolist()))
        raise ValueError(
            f"attach_period_rank: {len(missing)} {time_col} value(s) not in "
            f"profile['period_rank_info']['id_to_rank']: "
            f"{missing[:10]}{'…' if len(missing) > 10 else ''}"
        )

    out = df.copy()
    out[_PERIOD_RANK_COL] = ranks.astype("int64").values
    return out, _PERIOD_RANK_COL


# ─────────────────────────────────────────────────────────────────────────────
# CVEngine
# ─────────────────────────────────────────────────────────────────────────────

class CVEngine:
    """Deterministic (train_idx, valid_idx) splits driven entirely by CV_PLAN."""

    SUPPORTED = {
        "KFold", "StratifiedKFold", "GroupKFold",
        "TimeSeriesExpanding", "TimeSeriesSliding", "RollingOriginCV",
    }

    def __init__(self, plan: dict, df: pd.DataFrame, profile: dict | None = None):
        assert plan.get("frozen") is True, "CV_PLAN must be frozen"
        cv = plan["cv"]
        if cv["cv_type"] not in self.SUPPORTED:
            raise ValueError(f"Unsupported cv_type: {cv['cv_type']}")

        self.plan = plan
        raw_time_col = plan.get("time_column")
        df_aug, axis_col = attach_period_rank(df, profile, raw_time_col)
        self.df = df_aug.reset_index(drop=True)
        self.cv_type = cv["cv_type"]
        self.n_splits = int(cv["n_splits"])
        self.gap = int(cv.get("gap") or 0)
        self.window_size = cv.get("window_size")
        self.valid_size = cv.get("valid_size")
        self.random_state = int(cv.get("random_state") or 42)
        self.time_col = axis_col           # ordering axis (rank when codebook present)
        self.raw_time_col = raw_time_col   # join / submission key (always the hash)
        self.group_cols = plan.get("group_columns") or []
        self.target_col = plan["target_column"]
        self.horizon = plan.get("horizon")

    # ── public ───────────────────────────────────────────────────────────────
    def split(self) -> list[tuple[np.ndarray, np.ndarray]]:
        dispatch = {
            "KFold": self._kfold,
            "StratifiedKFold": self._stratified,
            "GroupKFold": self._group_kfold,
            "TimeSeriesExpanding": self._ts_expanding,
            "TimeSeriesSliding": self._ts_sliding,
            "RollingOriginCV": self._rolling_origin,
        }
        splits = dispatch[self.cv_type]()
        self.assert_invariants(splits)
        return splits

    def assert_invariants(self, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for k, (tr, va) in enumerate(splits):
            tr_set, va_set = set(map(int, tr)), set(map(int, va))
            if tr_set & va_set:
                raise AssertionError(f"Fold {k}: train/valid index overlap")
            if self.cv_type in ("TimeSeriesExpanding", "TimeSeriesSliding",
                                "RollingOriginCV") and self.time_col:
                t_tr = self.df[self.time_col].iloc[tr]
                t_va = self.df[self.time_col].iloc[va]
                if len(t_tr) and len(t_va):
                    if t_tr.max() + self.gap > t_va.min():
                        raise AssertionError(
                            f"Fold {k}: time leakage "
                            f"max(train_time)+gap={t_tr.max()+self.gap} > "
                            f"min(valid_time)={t_va.min()}")
            if self.cv_type == "GroupKFold" and self.group_cols:
                g_tr = set(self.df[self.group_cols[0]].iloc[tr].unique().tolist())
                g_va = set(self.df[self.group_cols[0]].iloc[va].unique().tolist())
                if g_tr & g_va:
                    raise AssertionError(f"Fold {k}: group overlap on {self.group_cols[0]}")

    # ── concrete schemes ─────────────────────────────────────────────────────
    def _kfold(self):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return [(np.asarray(tr), np.asarray(va)) for tr, va in kf.split(self.df)]

    def _stratified(self):
        from sklearn.model_selection import StratifiedKFold
        y = self.df[self.target_col].values
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        return [(np.asarray(tr), np.asarray(va)) for tr, va in skf.split(self.df, y)]

    def _group_kfold(self):
        from sklearn.model_selection import GroupKFold
        groups = self.df[self.group_cols[0]].values
        n_unique = int(pd.Series(groups).nunique())
        n = max(2, min(self.n_splits, n_unique))
        gkf = GroupKFold(n_splits=n)
        return [(np.asarray(tr), np.asarray(va))
                for tr, va in gkf.split(self.df, groups=groups)]

    # ── time-series helpers ──────────────────────────────────────────────────
    def _time_steps_to_rows(self) -> tuple[np.ndarray, dict]:
        """Map each unique time value to an array of row indices that share it."""
        t = self.df[self.time_col].values
        unique_times = np.array(sorted(pd.Series(t).unique()))
        rows_by_time: dict = {}
        idx_series = pd.Series(np.arange(len(self.df)))
        gb = idx_series.groupby(self.df[self.time_col].values, sort=False)
        for tv, grp in gb:
            rows_by_time[tv] = np.asarray(grp.values, dtype=int)
        return unique_times, rows_by_time

    @staticmethod
    def _concat_rows(time_window: np.ndarray, rows_by_time: dict) -> np.ndarray:
        parts = [rows_by_time[tv] for tv in time_window if tv in rows_by_time]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    def _ts_expanding(self):
        """End-anchored expanding window: the LAST fold's valid block ends at T
        (the final rank), so CV-MAE measures the deployment-like regime — the
        same window the held-out test will validate. Earlier folds step BACKWARD
        by valid_size; training is always [0:train_end], so train expands as
        fold index advances. Mirrors the end-anchored pattern in _ts_sliding."""
        unique_times, rows_by_time = self._time_steps_to_rows()
        T = len(unique_times)
        valid_size = int(self.valid_size or max(1, T // (self.n_splits + 1)))
        splits = []
        for k in range(self.n_splits):
            valid_end = T - k * valid_size
            valid_start = valid_end - valid_size
            train_end = valid_start - self.gap
            if train_end <= 0 or valid_start <= 0:
                break
            tr_times = unique_times[:train_end]
            va_times = unique_times[valid_start:valid_end]
            splits.append((self._concat_rows(tr_times, rows_by_time),
                           self._concat_rows(va_times, rows_by_time)))
        splits.reverse()
        return splits

    def _ts_sliding(self):
        """End-anchored sliding window: the LAST fold's training window is the most
        recent `window_size` periods, validating the final `valid_size` periods
        (mirrors deployment under drift); earlier folds slide backward."""
        unique_times, rows_by_time = self._time_steps_to_rows()
        T = len(unique_times)
        win = int(self.window_size or max(1, T // (self.n_splits + 1)))
        valid_size = int(self.valid_size or win)
        splits = []
        for k in range(self.n_splits):
            valid_end = T - k * valid_size
            valid_start = valid_end - valid_size
            train_end = valid_start - self.gap
            train_start = train_end - win
            if train_start < 0 or valid_start <= 0:
                break
            tr_times = unique_times[train_start:train_end]
            va_times = unique_times[valid_start:valid_end]
            splits.append((self._concat_rows(tr_times, rows_by_time),
                           self._concat_rows(va_times, rows_by_time)))
        splits.reverse()
        return splits

    def _rolling_origin(self):
        unique_times, rows_by_time = self._time_steps_to_rows()
        T = len(unique_times)
        H = int(self.horizon or 1)
        valid_size = int(self.valid_size or H)
        splits = []
        for k in range(self.n_splits):
            tr_end = (k + 1) * valid_size
            va_start = tr_end + self.gap
            va_end = va_start + H
            if va_end > T or tr_end == 0:
                break
            tr_times = unique_times[:tr_end]
            va_times = unique_times[va_start:va_end]
            splits.append((self._concat_rows(tr_times, rows_by_time),
                           self._concat_rows(va_times, rows_by_time)))
        return splits


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — materialise folds
# ─────────────────────────────────────────────────────────────────────────────

def materialise_folds(plan: dict, df: pd.DataFrame,
                      reports_dir: str | Path = "reports",
                      profile: dict | None = None) -> dict:
    """Build folds from CV_PLAN, enforce invariants, write reports/cv_folds.json.

    When `profile` is None, attempts to load reports/profile.json so the rank
    chokepoint runs even for legacy callers that don't pass it explicitly.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if profile is None:
        _pp = reports_dir / "profile.json"
        if _pp.exists():
            try:
                profile = json.loads(_pp.read_text(encoding="utf-8"))
            except Exception:
                profile = None

    engine = CVEngine(plan, df, profile=profile)
    folds = engine.split()

    payload = {
        "plan_id": plan["plan_id"],
        "cv_type": plan["cv"]["cv_type"],
        "n_splits": len(folds),
        "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "folds": [
            {
                "fold_id": k,
                "train_idx": [int(i) for i in tr.tolist()],
                "valid_idx": [int(i) for i in va.tolist()],
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
            }
            for k, (tr, va) in enumerate(folds)
        ],
    }

    with open(reports_dir / "cv_folds.json", "w") as f:
        json.dump(payload, f)

    with open(reports_dir / "validator_was_here.txt", "a") as f:
        f.write(f"validator PhaseA at {payload['created_at_utc']} "
                f"plan_id={plan['plan_id']} n_folds={len(folds)}\n")

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — aggregate OOF
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_oof(plan: dict, folds_payload: dict,
                  reports_dir: str | Path = "reports") -> dict:
    """Stitch per-fold predictions into OOF matrix; emit fold_metrics + review."""
    reports_dir = Path(reports_dir)
    target = plan["target_column"]
    n_planned = folds_payload["n_splits"]

    oof_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []

    for fold in folds_payload["folds"]:
        k = fold["fold_id"]
        fp = reports_dir / f"predictions_fold_{k}.parquet"
        if not fp.exists():
            continue
        preds = pd.read_parquet(fp).assign(fold=k)
        oof_rows.append(preds)
        yt = preds["y_true"].values
        yp = preds["y_pred"].values
        fold_metrics.append({
            "fold_id": k,
            "n_valid": int(len(preds)),
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "y_pred_mean": float(preds["y_pred"].mean()),
            "y_pred_std": float(preds["y_pred"].std() if len(preds) > 1 else 0.0),
        })

    if oof_rows:
        oof = pd.concat(oof_rows, ignore_index=True)
        oof.to_parquet(reports_dir / "oof_predictions.parquet", index=False)
        overall_mae = float(np.mean(np.abs(oof["y_true"].values - oof["y_pred"].values)))
    else:
        oof = pd.DataFrame()
        overall_mae = None

    maes = np.array([f["mae"] for f in fold_metrics], dtype=float)
    metrics = {
        "plan_id": plan["plan_id"],
        "n_folds": int(len(fold_metrics)),
        "n_folds_planned": n_planned,
        "oof_mae": overall_mae,
        "fold_mae_mean": float(maes.mean()) if len(maes) else None,
        "fold_mae_std": float(maes.std()) if len(maes) else None,
        "fold_mae_min": float(maes.min()) if len(maes) else None,
        "fold_mae_max": float(maes.max()) if len(maes) else None,
        "fold_maes": maes.tolist(),
        "per_fold": fold_metrics,
    }
    with open(reports_dir / "fold_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Validator review (diagnostic)
    verdict = "PASS"
    notes: list[str] = []
    if metrics["fold_mae_mean"] is not None and metrics["fold_mae_std"] is not None and metrics["fold_mae_mean"] > 0:
        coef_var = metrics["fold_mae_std"] / metrics["fold_mae_mean"]
        if coef_var > 0.5:
            verdict = "WARNING"
            notes.append(f"High fold MAE variability: std/mean={coef_var:.2f}")
    if metrics["n_folds"] < n_planned:
        verdict = "WARNING"
        notes.append(f"Only {metrics['n_folds']}/{n_planned} folds produced predictions")
    if metrics["n_folds"] == 0:
        verdict = "CRITICAL"
        notes.append("No fold predictions written")

    review = {
        "plan_id": plan["plan_id"],
        "cv_type": plan["cv"]["cv_type"],
        "verdict": verdict,
        "oof_mae": metrics["oof_mae"],
        "fold_mae_mean": metrics["fold_mae_mean"],
        "fold_mae_std": metrics["fold_mae_std"],
        "fold_maes": metrics["fold_maes"],
        "fold_train_sizes": [f["fold_id"] for f in fold_metrics],
        "notes": " | ".join(notes),
    }
    with open(reports_dir / "validator_review.json", "w") as f:
        json.dump(review, f, indent=2)

    with open(reports_dir / "validator_was_here.txt", "a") as f:
        f.write(f"validator PhaseB at {datetime.datetime.utcnow().isoformat()}Z "
                f"verdict={verdict} oof_mae={metrics['oof_mae']}\n")

    return {"metrics": metrics, "review": review, "oof": oof}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────────────────────

def load_folds(reports_dir: str | Path = "reports") -> dict:
    with open(Path(reports_dir) / "cv_folds.json") as f:
        return json.load(f)


if __name__ == "__main__":
    # Allow `python tools/cv_engine.py` from the repo root.
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from tools.scheme_analysis import load_cv_plan
    plan = load_cv_plan()
    print(f"CVEngine ready for plan_id={plan['plan_id']} cv_type={plan['cv']['cv_type']}")
