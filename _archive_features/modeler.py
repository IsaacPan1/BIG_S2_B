"""modeler.py — CV-agnostic estimator adapter.

Reads per-fold feature parquets, fits CatBoost and Ridge on (X_train, y_train),
predicts on X_valid, and writes per-fold predictions/importances. Does NOT
choose folds, does NOT aggregate across folds, does NOT touch the raw frame.

Active backends: CatBoost, Ridge. LightGBM/XGBoost are interface-only.
"""
from __future__ import annotations

import json
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Estimator interface
# ─────────────────────────────────────────────────────────────────────────────

class EstimatorInterface:
    name = "abstract"
    def fit(self, X_train, y_train): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def feature_importance(self, feature_names): return []


class CatBoostEstimator(EstimatorInterface):
    name = "catboost"
    def __init__(self, problem_subtype: str, cat_indices: list[int] | None = None):
        import catboost as cb
        regression = problem_subtype in (
            "continuous_regression", "ordinal_regression",
            "forecasting_multi_horizon",
        )
        loss = "MAE" if regression else "Logloss"
        cls = cb.CatBoostRegressor if regression else cb.CatBoostClassifier
        self._cls = cls
        self.model = cls(
            iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            loss_function=loss, verbose=False, allow_writing_files=False,
            random_seed=42,
        )
        self.cat_indices = cat_indices or []
        self._is_classifier = not regression

    def fit(self, X_train, y_train):
        kwargs = {"verbose": False}
        if self.cat_indices:
            kwargs["cat_features"] = self.cat_indices
        self.model.fit(X_train.values, y_train.values, **kwargs)
        return self

    def predict(self, X):
        if self._is_classifier and hasattr(self.model, "predict_proba"):
            return self.model.predict(X.values).ravel()
        return self.model.predict(X.values).ravel()

    def feature_importance(self, feature_names):
        imp = self.model.get_feature_importance()
        return [{"name": n, "importance": float(v)}
                for n, v in sorted(zip(feature_names, imp), key=lambda x: -x[1])[:20]]


class RidgeEstimator(EstimatorInterface):
    name = "ridge"
    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        Xs = self.scaler.fit_transform(X_train.values)
        self.model.fit(Xs, y_train.values)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X.values)).ravel()

    def feature_importance(self, feature_names):
        coef = np.abs(self.model.coef_)
        return [{"name": n, "importance": float(v)}
                for n, v in sorted(zip(feature_names, coef), key=lambda x: -x[1])[:20]]


# ─────────────────────────────────────────────────────────────────────────────
# Interface-only placeholders
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMEstimator(EstimatorInterface):
    name = "lightgbm"
    def fit(self, *_a, **_k):
        raise NotImplementedError("LightGBM is interface-only in this build")


class XGBoostEstimator(EstimatorInterface):
    name = "xgboost"
    def fit(self, *_a, **_k):
        raise NotImplementedError("XGBoost is interface-only in this build")


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_per_fold(plan: dict, folds_payload: dict,
                 data_dir: str | Path = "data",
                 reports_dir: str | Path = "reports") -> dict:
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    active = set(plan["modeler_contract"]["active_backends"])
    assert active == {"catboost", "ridge"}, (
        f"Modeler supports only CatBoost+Ridge in this build; got {active}"
    )

    target = plan["target_column"]
    subtype = plan.get("problem_subtype", "continuous_regression")
    per_fold_results: list[dict] = []

    for fold in folds_payload["folds"]:
        k = fold["fold_id"]
        tr_path = data_dir / f"features_train_fold_{k}.parquet"
        va_path = data_dir / f"features_valid_fold_{k}.parquet"
        if not (tr_path.exists() and va_path.exists()):
            print(f"[modeler] fold {k}: feature parquets missing — skipping")
            continue

        feat_train = pd.read_parquet(tr_path)
        feat_valid = pd.read_parquet(va_path)

        feature_cols = [c for c in feat_train.columns
                        if c not in (target, "__row_id__", "__y_true__")]
        X_train = feat_train[feature_cols].fillna(0.0)
        y_train = feat_train[target]
        X_valid = feat_valid[feature_cols].fillna(0.0)
        y_valid = feat_valid["__y_true__"]

        # CatBoost
        t0 = time.time()
        cat_indices = [i for i, c in enumerate(feature_cols)
                       if str(X_train[c].dtype) in ("object", "category")]
        cb_pred = None
        cb_top = []
        try:
            cb_est = CatBoostEstimator(problem_subtype=subtype,
                                       cat_indices=cat_indices).fit(X_train, y_train)
            cb_pred = cb_est.predict(X_valid)
            cb_top = cb_est.feature_importance(feature_cols)
        except Exception as e:
            print(f"[modeler] fold {k}: CatBoost failed ({e})")
        cb_time = time.time() - t0

        # Ridge
        t0 = time.time()
        ridge_pred = None
        ridge_top = []
        try:
            ridge_est = RidgeEstimator(alpha=1.0).fit(X_train, y_train)
            ridge_pred = ridge_est.predict(X_valid)
            ridge_top = ridge_est.feature_importance(feature_cols)
        except Exception as e:
            print(f"[modeler] fold {k}: Ridge failed ({e})")
        ridge_time = time.time() - t0

        # Blend (equal mean if both succeed)
        if cb_pred is not None and ridge_pred is not None:
            blend = 0.5 * cb_pred + 0.5 * ridge_pred
            backends_used = ["catboost", "ridge"]
        elif cb_pred is not None:
            blend = cb_pred
            backends_used = ["catboost"]
        elif ridge_pred is not None:
            blend = ridge_pred
            backends_used = ["ridge"]
        else:
            blend = np.full(len(X_valid), float(y_train.mean()))
            backends_used = ["fallback_mean"]

        if (y_train >= 0).all():
            blend = np.clip(blend, 0, None)

        # Persist per-fold predictions
        preds_df = pd.DataFrame({
            "__row_id__": feat_valid["__row_id__"].values,
            "y_true": y_valid.values,
            "y_pred": blend,
            "y_pred_catboost": cb_pred if cb_pred is not None else np.nan,
            "y_pred_ridge": ridge_pred if ridge_pred is not None else np.nan,
        })
        preds_df.to_parquet(reports_dir / f"predictions_fold_{k}.parquet", index=False)

        imp = {
            "fold_id": k,
            "backends_trained": backends_used,
            "catboost_top": cb_top,
            "ridge_top": ridge_top,
            "training_time_seconds": {"catboost": cb_time, "ridge": ridge_time},
        }
        with open(reports_dir / f"importance_fold_{k}.json", "w") as f:
            json.dump(imp, f, indent=2)

        fold_mae = float(np.mean(np.abs(y_valid.values - blend)))
        cb_mae = (float(np.mean(np.abs(y_valid.values - cb_pred)))
                  if cb_pred is not None else None)
        ridge_mae = (float(np.mean(np.abs(y_valid.values - ridge_pred)))
                     if ridge_pred is not None else None)

        per_fold_results.append({
            "fold_id": k, "mae": fold_mae,
            "catboost_mae": cb_mae, "ridge_mae": ridge_mae,
            "backends_used": backends_used,
        })

        with open(reports_dir / "modeler_was_here.txt", "a") as f:
            f.write(f"fold {k} mae={fold_mae:.4f} at "
                    f"{datetime.datetime.utcnow().isoformat()}Z\n")

    summary = {
        "plan_id": plan["plan_id"],
        "backends_active": ["catboost", "ridge"],
        "backends_interface_only": ["lightgbm", "xgboost"],
        "blend": "equal_mean(catboost, ridge)",
        "per_fold": per_fold_results,
        "mean_fold_mae": (float(np.mean([r["mae"] for r in per_fold_results]))
                          if per_fold_results else None),
    }
    with open(reports_dir / "model_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
