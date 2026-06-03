"""
Modeler: Adaptive LightGBM + XGBoost + CatBoost + Ridge ensemble.
Branch selected by problem_type and n_train:
  panel_forecasting  + n_train>=1000 → Branch 1: LightGBM+XGBoost+CatBoost+Ridge
  panel_forecasting  + n_train<1000  → Branch 2: LightGBM+Ridge
  tabular_regression + n_train>=1000 → Branch 3: LightGBM+XGBoost+CatBoost+Ridge
  tabular_regression + n_train<1000  → Branch 4: LightGBM+Ridge
  classification                     → classification_fallback: LightGBM only
CatBoost (Axis 3) is conditional: runs only when catboost is importable,
elapsed < 40 min, and n_train >= 500. Excluded if OOF > 1.5x best tree OOF.
Final predictions = median across included families' val predictions.
"""
import pandas as pd
import numpy as np
import json, time, warnings, datetime, os

import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

BASE_DIR  = r"C:\Users\isaac\OneDrive\Desktop\award_B"
REPORTS   = os.path.join(BASE_DIR, "reports")
DATA_DIR  = os.path.join(BASE_DIR, "data")
start_time = time.time()

print("="*60)
print("MODELER (Adaptive Ensemble) START", datetime.datetime.now().isoformat())
print("="*60)

# ── 1. Metadata ────────────────────────────────────────────────────────────────
with open(os.path.join(REPORTS, "features.json")) as f: feat_meta = json.load(f)
with open(os.path.join(REPORTS, "profile.json"))  as f: profile   = json.load(f)

target_col   = feat_meta["target_col"]
problem_type = profile.get("problem_type") or feat_meta.get("problem_type", "tabular_regression")
group_cols   = profile.get("group_cols")   or feat_meta.get("group_cols", [])
time_col     = profile.get("time_col")     or feat_meta.get("time_col")
print(f"problem_type={problem_type}  target={target_col}  groups={group_cols}  time={time_col}")

# ── 2. Critic retune ───────────────────────────────────────────────────────────
_retune_applied = None
_rpath = os.path.join(REPORTS, "critic_retune_requested.json")
if os.path.exists(_rpath):
    with open(_rpath) as f: _retune = json.load(f)
    _sug = _retune.get("suggested_change", "")
    print(f"Critic retune: {_sug}")
    if "median seed aggregation"        in _sug: _retune_applied = "median_seed_aggregation"
    if "expand Optuna"                  in _sug: _retune_applied = (_retune_applied or "") + "+expanded_optuna_bounds"
    if "val feature imputation"         in _sug: _retune_applied = (_retune_applied or "") + "+verified_imputation"
    if "np.clip applied after"          in _sug: _retune_applied = (_retune_applied or "") + "+clip_after_ensemble"
    if "multi-fold purged walk-forward" in _sug: _retune_applied = (_retune_applied or "") + "+multi_fold_wf_cv"
    if "remove suspect features"        in _sug:
        try:
            with open(os.path.join(REPORTS, "validator_review.json")) as f: _vr = json.load(f)
            _ns = len(_vr.get("feature_suspicion", []))
            if _ns: _retune_applied = (_retune_applied or "") + f"+will_remove_{_ns}_features"
        except Exception: pass

use_median_agg   = bool(_retune_applied and "median_seed_aggregation"  in (_retune_applied or ""))
_use_multi_wf    = bool(_retune_applied and "multi_fold_wf_cv"         in (_retune_applied or ""))
_expanded_bounds = bool(_retune_applied and "expanded_optuna_bounds"   in (_retune_applied or ""))

def agg(lst): return np.median(lst, axis=0) if use_median_agg else np.mean(lst, axis=0)

# ── 3. Data ────────────────────────────────────────────────────────────────────
train_df = pd.read_parquet(os.path.join(DATA_DIR, "features_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, "features_val.parquet"))
n_train  = len(train_df)
print(f"train={train_df.shape}  val={val_df.shape}")

# ── 4. Feature columns ─────────────────────────────────────────────────────────
_xc = set(group_cols) | {target_col, "horizon", "timestamp_ord"}
if time_col: _xc.add(time_col)
for _c in ["patient_id"]:
    if _c in train_df.columns: _xc.add(_c)

feature_cols = [c for c in train_df.columns
                if c not in _xc and not (
                    str(train_df[c].dtype) in ('object', 'string', 'str') or
                    hasattr(train_df[c].dtype, 'pyarrow_dtype') and str(train_df[c].dtype).startswith('string') or
                    pd.api.types.is_string_dtype(train_df[c])
                )]

if _retune_applied and "remove_" in (_retune_applied or ""):
    try:
        with open(os.path.join(REPORTS, "validator_review.json")) as f: _vr = json.load(f)
        _sus = _vr.get("feature_suspicion", [])
        if _sus: feature_cols = [c for c in feature_cols if c not in _sus]; print(f"Removed {len(_sus)} suspect feats")
    except Exception: pass

print(f"features={len(feature_cols)}: {feature_cols[:15]}...")
fill_vals = train_df[feature_cols].median()

X_full = train_df[feature_cols].fillna(fill_vals)
y_full = train_df[target_col].values
X_val  = val_df[feature_cols].fillna(fill_vals)

train_target_min  = float(y_full.min())
train_target_max  = float(y_full.max())
train_target_mean = float(y_full.mean())

# ── 5. Ensemble selection ──────────────────────────────────────────────────────
def select_ensemble(profile, n_train):
    pt = profile.get("problem_type")
    if pt == "classification":
        return {"branch": "classification_fallback", "families": ["lightgbm"],
                "reasoning": "Classification: LightGBM only (ensembling not fully tested)"}
    if n_train >= 1000:
        b = 1 if pt == "panel_forecasting" else 3
        return {"branch": b, "families": ["lightgbm", "xgboost", "catboost", "ridge"],
                "reasoning": f"{pt}, n_train={n_train}>=1000: full 4-family ensemble (LGB+XGB+CatBoost+Ridge)"}
    b = 2 if pt == "panel_forecasting" else 4
    return {"branch": b, "families": ["lightgbm", "ridge"],
            "reasoning": f"{pt}, n_train={n_train}<1000: 2-family ensemble (skip XGBoost)"}

ens_info      = select_ensemble(profile, n_train)
families_plan = ens_info["families"]
print(f"\nEnsemble: branch={ens_info['branch']} families={families_plan}")
print(f"Reason: {ens_info['reasoning']}")

# ── 5b. Shift-severity weighting decision (Axis 2) ────────────────────────────
_dist_shifts = profile.get("distribution_shifts", [])
_max_ks = 0.0
if _dist_shifts:
    _max_ks = max((d.get("ks_statistic", 0.0) for d in _dist_shifts if isinstance(d, dict)), default=0.0)
print(f"Distribution shift: max_ks={_max_ks:.4f} (threshold=0.40)")

_KS_THRESHOLD = 0.40
if _max_ks > _KS_THRESHOLD:
    _ensemble_weighting = "ridge_weighted_1.5x"
    _weighting_reason   = (f"max_ks={_max_ks:.2f} > {_KS_THRESHOLD} threshold; "
                           "weighting Ridge to hedge against severe shift")
else:
    _ensemble_weighting = "equal_median"
    _weighting_reason   = (f"max_ks={_max_ks:.2f} <= {_KS_THRESHOLD} threshold; "
                           "equal-weight median")
print(f"Ensemble weighting: {_ensemble_weighting}")
print(f"Weighting reason: {_weighting_reason}")

# ── 6. Split setup ─────────────────────────────────────────────────────────────
cv_splits   = None
all_periods = None

if problem_type == "panel_forecasting":
    all_periods   = sorted(train_df[time_col].unique())
    cutoff_idx    = int(len(all_periods) * 0.8)
    cutoff_period = all_periods[cutoff_idx]
    wf_tr_df = train_df[train_df[time_col] < cutoff_period].copy()
    wf_va_df = train_df[train_df[time_col] >= cutoff_period].copy()
    wf_fill  = wf_tr_df[feature_cols].median()
    X_wf_tr  = wf_tr_df[feature_cols].fillna(wf_fill)
    y_wf_tr  = wf_tr_df[target_col].values
    X_wf_va  = wf_va_df[feature_cols].fillna(wf_fill)
    y_wf_va  = wf_va_df[target_col].values
    X_ptr, y_ptr = X_wf_tr, y_wf_tr
    X_pva, y_pva = X_wf_va, y_wf_va
    cv_scheme = ("MultiWalkForward(4-fold,embargo=2)" if _use_multi_wf
                 else f"WalkForward(cutoff={cutoff_period})")
    print(f"WF: train={X_ptr.shape}, val={X_pva.shape}, cutoff={cutoff_period}")
else:
    np.random.seed(42)
    perm = np.random.permutation(n_train)
    sp   = int(n_train * 0.8)
    ptr_fill = X_full.iloc[perm[:sp]].median()
    X_ptr = X_full.iloc[perm[:sp]].fillna(ptr_fill); y_ptr = y_full[perm[:sp]]
    X_pva = X_full.iloc[perm[sp:]].fillna(ptr_fill);  y_pva = y_full[perm[sp:]]

    def _build_cv(pt, gc, X, y, df, ns=5):
        if pt == "tabular_regression":
            if gc:
                g0 = gc[0]
                if g0 in df.columns:
                    grps = df[g0].values
                    k    = max(2, min(ns, len(np.unique(grps))))
                    return list(GroupKFold(k).split(X, y, groups=grps)), f"GroupKFold({k},col='{g0}')"
            return list(RepeatedKFold(n_splits=ns, n_repeats=3, random_state=42).split(X, y)), "RepeatedKFold(5x3)"
        if pt == "classification":
            from sklearn.model_selection import StratifiedKFold
            return list(StratifiedKFold(ns, shuffle=True, random_state=42).split(X, y)), f"StratifiedKFold({ns})"
        return list(KFold(ns, shuffle=True, random_state=42).split(X, y)), f"KFold({ns})"

    cv_splits, cv_scheme = _build_cv(problem_type, group_cols, X_full, y_full, train_df.reset_index(drop=True))
    print(f"CV: {cv_scheme}, folds={len(cv_splits)}, probe={X_ptr.shape}")

TUNING_DEADLINE = start_time + 25 * 60

# ── OOF helpers ────────────────────────────────────────────────────────────────
def _panel_oof(pred_fn, multi):
    """Returns (mae, fold_maes, oof_df). pred_fn(Xtr,ytr,Xva)->ndarray."""
    if multi:
        N, EMB = 4, 2
        avail = all_periods[int(len(all_periods) * 0.4):]
        fsz   = max(1, len(avail) // N)
        fmaes, rows = [], []
        for fi in range(N):
            vps = avail[fi*fsz : (fi+1)*fsz if fi < N-1 else len(avail)]
            tc  = all_periods.index(vps[0]) - EMB
            if tc < 10: continue
            ftr = train_df[train_df[time_col].isin(all_periods[:tc])]
            fva = train_df[train_df[time_col].isin(vps)]
            ff  = ftr[feature_cols].median()
            Xtr = ftr[feature_cols].fillna(ff); ytr = ftr[target_col].values
            Xva = fva[feature_cols].fillna(ff); yva = fva[target_col].values
            fp  = pred_fn(Xtr, ytr, Xva)
            fm  = mean_absolute_error(yva, fp); fmaes.append(fm)
            r   = fva[group_cols + [time_col]].copy().reset_index(drop=True)
            r["fold"] = fi; r["predicted_target"] = fp; rows.append(r)
            print(f"    WF fold {fi+1}: MAE={fm:.4f}")
        mae = float(np.mean(fmaes)) if fmaes else float('nan')
        oof = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return mae, fmaes, oof
    else:
        preds = pred_fn(X_wf_tr, y_wf_tr, X_wf_va)
        mae   = float(mean_absolute_error(y_wf_va, preds))
        df_o  = wf_va_df[group_cols + [time_col]].copy().reset_index(drop=True)
        df_o["fold"] = 0; df_o["predicted_target"] = preds
        print(f"    WF OOF MAE: {mae:.4f}")
        return mae, [mae], df_o


def _tabular_oof(pred_fn):
    """Returns (mae, fold_maes, oof_arr, folds_arr)."""
    n   = len(X_full)
    acc = np.zeros(n); cnt = np.zeros(n, dtype=int); fa = np.full(n, -1, dtype=int)
    fmaes = []
    for fi, (tri, vai) in enumerate(cv_splits):
        ffl = X_full.iloc[tri].median()
        Xtr = X_full.iloc[tri].fillna(ffl); ytr = y_full[tri]
        Xva = X_full.iloc[vai].fillna(ffl); yva = y_full[vai]
        fp  = pred_fn(Xtr, ytr, Xva)
        acc[vai] += fp; cnt[vai] += 1; fa[vai] = fi
        fm = mean_absolute_error(yva, fp); fmaes.append(fm)
        print(f"    Fold {fi+1}: MAE={fm:.4f}")
    oof = np.where(cnt > 0, acc / cnt, float(y_full.mean()))
    return float(mean_absolute_error(y_full, oof)), fmaes, oof, fa


# Accumulate results
all_family_results = {}   # name -> dict
all_val_preds      = {}   # name -> np.array (only if included_in_ensemble)
master_oof_df      = None
master_oof_mae     = float('nan')
master_fold_maes   = []
probe_mae          = float('nan')
_lgbm_fi           = None
_xgb_fi            = None
_ridge_coef_top5   = []
_lgbm_nt           = 0
_lgbm_ne           = 500
_lgbm_hp           = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8}

# ══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*50 + "\nLightGBM\n" + "-"*50)
_lt0 = time.time()
_lgbm_res = {"succeeded": False, "included_in_ensemble": False,
             "oof_mae": None, "training_time_seconds": None, "exclusion_reason": "not started"}
_lgbm_vp  = np.full(len(val_df), train_target_mean)

try:
    _nl_hi  = 255 if _expanded_bounds else 127
    _mcs_lo = 3   if _expanded_bounds else 5

    def _lgbm_obj(trial):
        if time.time() > TUNING_DEADLINE: raise optuna.exceptions.TrialPruned()
        p = {"objective": "regression_l1", "metric": "mae", "n_estimators": 500,
             "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1, "n_jobs": -1,
             "learning_rate":   trial.suggest_float("learning_rate",   0.01, 0.1, log=True),
             "num_leaves":      trial.suggest_int("num_leaves",        15,   _nl_hi),
             "min_child_samples": trial.suggest_int("min_child_samples", _mcs_lo, 60),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0)}
        ms = [mean_absolute_error(y_pva, np.clip(
                  lgb.LGBMRegressor(**{**p, "random_state": s}).fit(
                      X_ptr, y_ptr,
                      eval_set=[(X_pva, y_pva)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
                  ).predict(X_pva), 0, None)) for s in [42, 7, 123]]
        return float(np.mean(ms))

    _ls = optuna.create_study(direction="minimize")
    _ls.optimize(_lgbm_obj, n_trials=15, timeout=25*60, catch=(Exception,))
    _lgbm_bp = _ls.best_params; _lgbm_nt = len(_ls.trials)
    print(f"Optuna: {_lgbm_nt} trials, best={_ls.best_value:.4f}")
except Exception as _e:
    _lgbm_bp = {}; _lgbm_nt = 0; print(f"Optuna failed: {_e}")

_lgbm_def = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
             "feature_fraction": 0.8, "bagging_fraction": 0.8}
_lgbm_hp  = {**_lgbm_def, **_lgbm_bp}
_lgbm_base = {"objective": "regression_l1", "metric": "mae", "bagging_freq": 5,
              "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1, "n_jobs": -1, **_lgbm_hp}

try:
    _pm = lgb.LGBMRegressor(**{**_lgbm_base, "n_estimators": 2000, "random_state": 42})
    _pm.fit(X_ptr, y_ptr, eval_set=[(X_pva, y_pva)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    _lgbm_ne  = max(int((_pm.best_iteration_ or 500) * 1.1), 200)
    probe_mae = float(mean_absolute_error(y_pva, np.clip(_pm.predict(X_pva), 0, None)))
    print(f"Probe MAE={probe_mae:.4f}, n_est={_lgbm_ne}")
except Exception as _e:
    _lgbm_ne = 500; print(f"Probe failed: {_e}")

_lgbm_p = {**_lgbm_base, "n_estimators": _lgbm_ne}

def _lgbm_pfn(Xtr, ytr, Xva):
    return agg([np.clip(lgb.LGBMRegressor(**{**_lgbm_p, "random_state": s}).fit(
                    Xtr, ytr, callbacks=[lgb.log_evaluation(-1)]).predict(Xva), 0, None)
                for s in [42, 7, 123]])

try:
    if problem_type == "panel_forecasting":
        _lm, _lf, _lod = _panel_oof(_lgbm_pfn, _use_multi_wf)
    else:
        _lm, _lf, _larr, _lflds = _tabular_oof(_lgbm_pfn)
        _ic = [c for c in (([profile.get("id_col")] if profile.get("id_col") else [])
                           + group_cols + ([time_col] if time_col else []))
               if c and c in train_df.columns]
        _lod = train_df[_ic].copy().reset_index(drop=True)
        _lod["fold"] = _lflds; _lod["predicted_target"] = _larr

    master_oof_mae = _lm; master_fold_maes = _lf; master_oof_df = _lod
    print(f"LightGBM OOF MAE: {_lm:.4f}")

    print("Full retrain (5 seeds)...")
    _lsps = []; _last_lgbm = None
    for _s in [42, 7, 123, 2024, 999]:
        _m = lgb.LGBMRegressor(**{**_lgbm_p, "random_state": _s})
        _m.fit(X_full, y_full, callbacks=[lgb.log_evaluation(-1)])
        _lsps.append(np.clip(_m.predict(X_val), 0, None)); _last_lgbm = _m
    _lgbm_vp = agg(_lsps)
    _lgbm_fi = pd.Series(_last_lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"LightGBM val: mean={_lgbm_vp.mean():.3f}")

    _lgbm_res = {
        "best_params": {k: (float(v) if isinstance(v, (float, np.floating))
                            else int(v) if isinstance(v, (int, np.integer)) else v)
                        for k, v in _lgbm_hp.items()},
        "oof_mae": float(_lm), "training_time_seconds": float(time.time() - _lt0),
        "succeeded": True, "included_in_ensemble": True,
        "n_estimators": _lgbm_ne, "optuna_trials": _lgbm_nt,
    }
    all_val_preds["lightgbm"] = _lgbm_vp
    print(f"LightGBM done: OOF={_lm:.4f}, time={_lgbm_res['training_time_seconds']:.1f}s")

except Exception as _lgbm_exc:
    _lgbm_res = {"succeeded": False, "included_in_ensemble": False,
                 "exclusion_reason": str(_lgbm_exc), "oof_mae": None,
                 "training_time_seconds": float(time.time() - _lt0)}
    _lgbm_fi = None
    all_val_preds["lightgbm"] = _lgbm_vp  # global mean fallback
    print(f"LightGBM FAILED: {_lgbm_exc}")

all_family_results["lightgbm"] = _lgbm_res

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ══════════════════════════════════════════════════════════════════════════════
if "xgboost" in families_plan:
    _elapsed = time.time() - start_time
    if _elapsed > 20 * 60:
        print(f"\nSkipping XGBoost: elapsed {_elapsed/60:.1f}min > 20min safeguard")
        all_family_results["xgboost"] = {
            "succeeded": False, "included_in_ensemble": False,
            "exclusion_reason": f"time safeguard: {_elapsed/60:.1f}min elapsed before XGBoost started",
            "oof_mae": None, "training_time_seconds": 0,
        }
    else:
        print("\n" + "-"*50 + "\nXGBoost\n" + "-"*50)
        _xt0 = time.time()
        _xgb_res = {"succeeded": False, "included_in_ensemble": False,
                    "oof_mae": None, "training_time_seconds": None, "exclusion_reason": "not started"}
        try:
            import xgboost as xgb

            # Optuna
            _xgb_bp = {}; _xgb_nt = 0
            try:
                def _xgb_obj(trial):
                    if time.time() > TUNING_DEADLINE: raise optuna.exceptions.TrialPruned()
                    try:
                        _obj = "reg:absoluteerror"
                        _tp = {
                            "objective": _obj, "n_estimators": 500, "verbosity": 0, "n_jobs": -1,
                            "learning_rate":    trial.suggest_float("learning_rate",    0.01, 0.3,  log=True),
                            "max_depth":        trial.suggest_int("max_depth",          3,    12),
                            "min_child_weight": trial.suggest_int("min_child_weight",   1,    10),
                            "subsample":        trial.suggest_float("subsample",        0.5,  1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,  1.0),
                            "reg_alpha":        trial.suggest_float("reg_alpha",        0.0,  1.0),
                            "reg_lambda":       trial.suggest_float("reg_lambda",       0.0,  1.0),
                        }
                        ms = []
                        for _s in [42, 7, 123]:
                            _xm = xgb.XGBRegressor(**{**_tp, "random_state": _s,
                                                      "early_stopping_rounds": 50})
                            _xm.fit(X_ptr, y_ptr, eval_set=[(X_pva, y_pva)], verbose=False)
                            ms.append(mean_absolute_error(y_pva, np.clip(_xm.predict(X_pva), 0, None)))
                        return float(np.mean(ms))
                    except Exception:
                        _tp2 = {**_tp, "objective": "reg:squarederror"}
                        _xm2b = xgb.XGBRegressor(**{**_tp2, "random_state": 42,
                                                   "early_stopping_rounds": 50})
                        _xm2b.fit(X_ptr, y_ptr, eval_set=[(X_pva, y_pva)], verbose=False)
                        return float(mean_absolute_error(y_pva, np.clip(_xm2b.predict(X_pva), 0, None)))

                _xs = optuna.create_study(direction="minimize")
                _xs.optimize(_xgb_obj, n_trials=15, timeout=max(1, TUNING_DEADLINE - time.time()), catch=(Exception,))
                _xgb_bp = _xs.best_params; _xgb_nt = len(_xs.trials)
                print(f"XGBoost Optuna: {_xgb_nt} trials, best={_xs.best_value:.4f}")
            except Exception as _e:
                print(f"XGBoost Optuna failed: {_e}")

            # Probe for n_estimators
            _xgb_def = {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1,
                        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0, "reg_lambda": 1}
            _xgb_hp  = {**_xgb_def, **_xgb_bp}
            _xgb_obj_name = "reg:squarederror"; _xgb_ne = 500
            try:
                _xgb_obj_name = "reg:absoluteerror"
                _xgb_probe = xgb.XGBRegressor(
                    **{**_xgb_hp, "objective": _xgb_obj_name, "n_estimators": 2000,
                       "verbosity": 0, "n_jobs": -1, "random_state": 42,
                       "early_stopping_rounds": 50})
                _xgb_probe.fit(X_ptr, y_ptr, eval_set=[(X_pva, y_pva)], verbose=False)
                _xgb_ne = max(int((_xgb_probe.best_iteration or 500) * 1.1), 100)
                print(f"XGBoost probe n_est={_xgb_ne}")
            except Exception as _e:
                try:
                    _xgb_obj_name = "reg:squarederror"
                    _xgb_probe = xgb.XGBRegressor(
                        **{**_xgb_hp, "objective": _xgb_obj_name, "n_estimators": 2000,
                           "verbosity": 0, "n_jobs": -1, "random_state": 42,
                           "early_stopping_rounds": 50})
                    _xgb_probe.fit(X_ptr, y_ptr, eval_set=[(X_pva, y_pva)], verbose=False)
                    _xgb_ne = max(int((_xgb_probe.best_iteration or 500) * 1.1), 100)
                    print(f"XGBoost probe (squarederror fallback) n_est={_xgb_ne}")
                except Exception as _e2:
                    print(f"XGBoost probe failed: {_e2}")

            _xgb_p = {**_xgb_hp, "objective": _xgb_obj_name, "n_estimators": _xgb_ne,
                      "verbosity": 0, "n_jobs": -1}

            def _xgb_pfn(Xtr, ytr, Xva):
                return agg([np.clip(xgb.XGBRegressor(**{**_xgb_p, "random_state": _s}).fit(
                                Xtr, ytr, verbose=False).predict(Xva), 0, None)
                            for _s in [42, 7, 123]])

            # OOF + full retrain
            if problem_type == "panel_forecasting":
                _xm2, _xf2, _xod = _panel_oof(_xgb_pfn, _use_multi_wf)
            else:
                _xm2, _xf2, _xarr, _xflds = _tabular_oof(_xgb_pfn)
            print(f"XGBoost OOF MAE: {_xm2:.4f}")

            print("XGBoost: full retrain (5 seeds)...")
            _xsps = []
            for _s in [42, 7, 123, 2024, 999]:
                _xm_ = xgb.XGBRegressor(**{**_xgb_p, "random_state": _s})
                _xm_.fit(X_full, y_full, verbose=False)
                _xsps.append(np.clip(_xm_.predict(X_val), 0, None))
            _xgb_vp = agg(_xsps)
            _xgb_fi = pd.Series(_xm_.feature_importances_, index=feature_cols).sort_values(ascending=False)
            print(f"XGBoost val: mean={_xgb_vp.mean():.3f}")

            _xgb_res = {
                "best_params": {k: (float(v) if isinstance(v, (float, np.floating))
                                    else int(v) if isinstance(v, (int, np.integer)) else v)
                                for k, v in _xgb_hp.items()},
                "oof_mae": float(_xm2), "training_time_seconds": float(time.time() - _xt0),
                "succeeded": True, "included_in_ensemble": True,
                "n_estimators": _xgb_ne, "optuna_trials": _xgb_nt, "objective": _xgb_obj_name,
            }
            all_val_preds["xgboost"] = _xgb_vp
            print(f"XGBoost done: OOF={_xm2:.4f}, time={_xgb_res['training_time_seconds']:.1f}s")

        except Exception as _xgb_exc:
            _xgb_res = {"succeeded": False, "included_in_ensemble": False,
                        "exclusion_reason": str(_xgb_exc), "oof_mae": None,
                        "training_time_seconds": float(time.time() - _xt0)}
            _xgb_fi = None
            print(f"XGBoost FAILED: {_xgb_exc}")

        all_family_results["xgboost"] = _xgb_res

# ══════════════════════════════════════════════════════════════════════════════
# CATBOOST (Axis 3 — conditional, time-gated)
# ══════════════════════════════════════════════════════════════════════════════
if "catboost" in families_plan:
    _elapsed_cb = (time.time() - start_time) / 60
    try:
        import catboost as _cb_module
        _catboost_available = True
    except ImportError:
        _catboost_available = False

    _should_run_cb = (
        _catboost_available
        and n_train >= 500
        and _elapsed_cb < 40
        and problem_type in ("panel_forecasting", "tabular_regression", "classification")
    )

    if not _should_run_cb:
        _skip_reason = (
            "skipped_import_error" if not _catboost_available else
            f"skipped_data_too_small (n_train={n_train})" if n_train < 500 else
            f"skipped_no_time (elapsed={_elapsed_cb:.1f}m >= 40m)" if _elapsed_cb >= 40 else
            "skipped_unsupported_problem_type"
        )
        print(f"\nCatBoost skipped: {_skip_reason}")
        all_family_results["catboost"] = {
            "succeeded": False, "included_in_ensemble": False,
            "skip_reason": _skip_reason, "oof_mae": None, "training_time_seconds": 0,
        }
    else:
        print("\n" + "-"*50 + "\nCatBoost (Axis 3)\n" + "-"*50)
        _ct0 = time.time()
        _cb_res = {"succeeded": False, "included_in_ensemble": False,
                   "oof_mae": None, "training_time_seconds": None,
                   "skip_reason": None, "exclusion_reason": None}
        try:
            _cb_loss = "MAE" if problem_type in ("panel_forecasting", "tabular_regression") else "Logloss"
            _cat_cols = [c for c in feature_cols if train_df[c].dtype.name in ("object", "category")]
            _cat_idx = [feature_cols.index(c) for c in _cat_cols]

            CB_DEADLINE = start_time + 50 * 60  # hard stop at 50 min total

            def _cb_obj(trial):
                if time.time() > CB_DEADLINE: raise optuna.exceptions.TrialPruned()
                _p = {
                    "iterations": 400,
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "loss_function": _cb_loss, "eval_metric": _cb_loss,
                    "verbose": False, "allow_writing_files": False,
                }
                if _cat_idx: _p["cat_features"] = _cat_idx
                _ms = []
                for _s in [42, 7]:
                    _p["random_seed"] = _s
                    _m = (_cb_module.CatBoostRegressor(**_p) if _cb_loss == "MAE"
                          else _cb_module.CatBoostClassifier(**_p))
                    _m.fit(X_ptr.values, y_ptr, verbose=False)
                    _ms.append(mean_absolute_error(y_pva, np.clip(_m.predict(X_pva.values), 0, None)))
                return float(np.mean(_ms))

            _cb_study = optuna.create_study(direction="minimize")
            _cb_study.optimize(_cb_obj, n_trials=10, timeout=15*60, catch=(Exception,))
            _cb_best = _cb_study.best_params
            print(f"CatBoost Optuna: {len(_cb_study.trials)} trials, best={_cb_study.best_value:.4f}")

            _cb_fp = {
                "iterations": 400,
                "learning_rate": _cb_best.get("learning_rate", 0.05),
                "depth": _cb_best.get("depth", 6),
                "l2_leaf_reg": _cb_best.get("l2_leaf_reg", 3.0),
                "loss_function": _cb_loss, "eval_metric": _cb_loss,
                "verbose": False, "allow_writing_files": False,
            }
            if _cat_idx: _cb_fp["cat_features"] = _cat_idx

            def _cb_pfn(Xtr, ytr, Xva):
                _preds = []
                for _s in [42, 7, 123]:
                    _cb_fp["random_seed"] = _s
                    _m = (_cb_module.CatBoostRegressor(**_cb_fp) if _cb_loss == "MAE"
                          else _cb_module.CatBoostClassifier(**_cb_fp))
                    _m.fit(Xtr.values if hasattr(Xtr, "values") else Xtr, ytr, verbose=False)
                    _preds.append(np.clip(_m.predict(Xva.values if hasattr(Xva, "values") else Xva), 0, None))
                return np.median(_preds, axis=0)

            if problem_type == "panel_forecasting":
                _cm, _cf, _cod = _panel_oof(_cb_pfn, _use_multi_wf)
            else:
                _cm, _cf, _carr, _cflds = _tabular_oof(_cb_pfn)
            print(f"CatBoost OOF MAE: {_cm:.4f}")
            _cb_res["oof_mae"] = float(_cm)

            # Competence check: cb_oof <= 1.5 * best tree OOF
            _tree_oof_maes_cb = {k: all_family_results[k]["oof_mae"]
                                 for k in ("lightgbm", "xgboost")
                                 if k in all_family_results
                                 and all_family_results[k].get("succeeded")
                                 and all_family_results[k].get("oof_mae") is not None}
            _best_tree_oof_cb = min(_tree_oof_maes_cb.values()) if _tree_oof_maes_cb else float("inf")
            _cb_passes = _cm <= 1.5 * _best_tree_oof_cb

            if not _cb_passes:
                _cb_res["exclusion_reason"] = (
                    f"excluded_too_weak: cb_oof={_cm:.4f} > 1.5x best_tree_oof={_best_tree_oof_cb:.4f}"
                )
                _cb_res["included_in_ensemble"] = False
                _cb_res["succeeded"] = True
                print(f"CatBoost EXCLUDED (competence): {_cb_res['exclusion_reason']}")
            else:
                print("CatBoost: full retrain (5 seeds)...")
                _cb_fps = []
                for _s in [42, 7, 123, 2024, 999]:
                    _cb_fp["random_seed"] = _s
                    _m = (_cb_module.CatBoostRegressor(**_cb_fp) if _cb_loss == "MAE"
                          else _cb_module.CatBoostClassifier(**_cb_fp))
                    _m.fit(X_full.values, y_full, verbose=False)
                    _cb_fps.append(_m.predict(X_val.values))
                _cb_vp = np.median(_cb_fps, axis=0)
                if train_target_min >= 0:
                    _cb_vp = np.clip(_cb_vp, 0, None)

                _cb_ok = True
                _cb_pmax, _cb_pmean = float(_cb_vp.max()), float(_cb_vp.mean())
                if _cb_pmax > 5 * train_target_max:
                    _cb_res["exclusion_reason"] = (
                        f"sanity: pred_max={_cb_pmax:.2f} > 5x train_max={train_target_max:.2f}"
                    )
                    _cb_ok = False
                    print(f"CatBoost EXCLUDED (sanity): {_cb_res['exclusion_reason']}")
                elif abs(train_target_mean) > 0 and abs(_cb_pmean - train_target_mean) / abs(train_target_mean) > 1.0:
                    _cb_res["exclusion_reason"] = (
                        f"sanity: pred_mean={_cb_pmean:.2f} deviates >100% from train_mean={train_target_mean:.2f}"
                    )
                    _cb_ok = False
                    print(f"CatBoost EXCLUDED (sanity): {_cb_res['exclusion_reason']}")

                _cb_res["succeeded"] = True
                if _cb_ok:
                    all_val_preds["catboost"] = _cb_vp
                    _cb_res["included_in_ensemble"] = True
                    print(f"CatBoost included: OOF={_cm:.4f}, val mean={_cb_vp.mean():.3f}")
                else:
                    _cb_res["included_in_ensemble"] = False

        except Exception as _cb_exc:
            _cb_res["succeeded"] = False
            _cb_res["skip_reason"] = f"training_error: {_cb_exc}"
            print(f"CatBoost FAILED: {_cb_exc}")

        _cb_res["training_time_seconds"] = float(time.time() - _ct0)
        all_family_results["catboost"] = _cb_res
        print(f"CatBoost block: time={_cb_res['training_time_seconds']:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# RIDGE
# ══════════════════════════════════════════════════════════════════════════════
if "ridge" in families_plan:
    _n_done  = sum(1 for r in all_family_results.values() if r.get("succeeded"))
    _elapsed = time.time() - start_time
    if _n_done >= 2 and _elapsed > 30 * 60:
        print(f"\nSkipping Ridge: {_n_done} families done, elapsed {_elapsed/60:.1f}min > 30min")
        all_family_results["ridge"] = {
            "succeeded": False, "included_in_ensemble": False,
            "exclusion_reason": f"time safeguard: {_n_done} families done and {_elapsed/60:.1f}min elapsed",
            "oof_mae": None, "training_time_seconds": 0,
        }
    else:
        print("\n" + "-"*50 + "\nRidge\n" + "-"*50)
        _rt0 = time.time()
        _ridge_res = {"succeeded": False, "included_in_ensemble": False,
                      "oof_mae": None, "training_time_seconds": None, "exclusion_reason": "not started"}
        try:
            # Alpha selection via probe split
            _best_alpha = 1.0; _best_alpha_mae = float('inf')
            _rsc = StandardScaler()
            _Xptr_f = X_ptr.fillna(X_ptr.median()) if hasattr(X_ptr, 'fillna') else X_ptr
            _Xpva_f = X_pva.fillna(X_ptr.median()) if hasattr(X_pva, 'fillna') else X_pva
            _Xptr_s = _rsc.fit_transform(_Xptr_f)
            _Xpva_s = _rsc.transform(_Xpva_f)
            for _al in [0.01, 0.1, 1.0, 10.0, 100.0]:
                _rm  = SkRidge(alpha=_al)
                _rm.fit(_Xptr_s, y_ptr)
                _mae = mean_absolute_error(y_pva, _rm.predict(_Xpva_s))
                print(f"  Ridge alpha={_al}: probe MAE={_mae:.4f}")
                if _mae < _best_alpha_mae:
                    _best_alpha_mae = _mae; _best_alpha = _al
            print(f"Best alpha: {_best_alpha} (probe MAE={_best_alpha_mae:.4f})")

            # OOF predictions
            def _ridge_pfn(Xtr, ytr, Xva):
                _sc = StandardScaler()
                _Xtr_f = Xtr.fillna(Xtr.median()) if hasattr(Xtr, 'fillna') else Xtr
                _Xva_f = Xva.fillna(Xtr.median()) if hasattr(Xva, 'fillna') else Xva
                _Xtr_s = _sc.fit_transform(_Xtr_f)
                _Xva_s = _sc.transform(_Xva_f)
                _m = SkRidge(alpha=_best_alpha)
                _m.fit(_Xtr_s, ytr)
                return _m.predict(_Xva_s)

            if problem_type == "panel_forecasting":
                _rm2, _rf2, _rod = _panel_oof(_ridge_pfn, _use_multi_wf)
            else:
                _rm2, _rf2, _rarr, _rflds = _tabular_oof(_ridge_pfn)
            print(f"Ridge OOF MAE: {_rm2:.4f}")

            # Full retrain on all training data
            print("Ridge: full retrain...")
            _rsc_full = StandardScaler()
            _Xfull_f  = X_full.fillna(fill_vals) if hasattr(X_full, 'fillna') else X_full
            _Xval_f   = X_val.fillna(fill_vals)  if hasattr(X_val, 'fillna') else X_val
            _Xfull_s  = _rsc_full.fit_transform(_Xfull_f)
            _Xval_s   = _rsc_full.transform(_Xval_f)
            _rfull    = SkRidge(alpha=_best_alpha)
            _rfull.fit(_Xfull_s, y_full)
            _ridge_vp  = _rfull.predict(_Xval_s)
            _ridge_coef = pd.Series(np.abs(_rfull.coef_), index=feature_cols).sort_values(ascending=False)

            # Sanity checks
            _ridge_included = True
            _ridge_excl_why = ""
            if train_target_min >= 0:
                _n_neg = (_ridge_vp < 0).sum()
                if _n_neg > 0:
                    print(f"  Ridge: clipping {_n_neg} negative predictions to 0")
                    _ridge_vp = np.clip(_ridge_vp, 0, None)
            if train_target_max != 0 and _ridge_vp.max() > 5 * train_target_max:
                _ridge_included = False
                _ridge_excl_why = (f"pred_max={_ridge_vp.max():.2f} > 5*train_max={5*train_target_max:.2f}")
                print(f"  Ridge EXCLUDED: {_ridge_excl_why}")
            elif train_target_mean != 0 and abs(_ridge_vp.mean() - train_target_mean) / abs(train_target_mean) > 1.0:
                _ridge_included = False
                _ridge_excl_why = (f"pred_mean={_ridge_vp.mean():.2f} is >100% off "
                                   f"train_mean={train_target_mean:.2f}")
                print(f"  Ridge EXCLUDED: {_ridge_excl_why}")

            # Competence check: exclude Ridge if OOF MAE > 1.5x best tree-model OOF MAE
            if _ridge_included:
                _tree_oof_maes = {k: all_family_results[k]["oof_mae"]
                                  for k in ("lightgbm", "xgboost")
                                  if k in all_family_results
                                  and all_family_results[k].get("succeeded")
                                  and all_family_results[k].get("oof_mae") is not None}
                if _tree_oof_maes:
                    _best_tree_oof = min(_tree_oof_maes.values())
                    _ridge_oof_val = float(_rm2)
                    if _ridge_oof_val > 1.5 * _best_tree_oof:
                        _ridge_included = False
                        _ridge_excl_why = (
                            f"ridge_oof={_ridge_oof_val:.3f} > 1.5x "
                            f"best_family_oof={_best_tree_oof:.3f}"
                        )
                        print(f"  Ridge EXCLUDED (competence): {_ridge_excl_why}")
                    else:
                        print(f"  Ridge competence PASS: ridge_oof={_ridge_oof_val:.3f} "
                              f"<= 1.5x best_family_oof={_best_tree_oof:.3f}")

            _ridge_coef_top5 = [{"feature": k, "abs_coef": float(v)}
                                 for k, v in _ridge_coef.head(5).items()]
            print(f"Ridge val: mean={_ridge_vp.mean():.3f}, included={_ridge_included}")

            _ridge_res = {
                "best_alpha": float(_best_alpha), "oof_mae": float(_rm2),
                "training_time_seconds": float(time.time() - _rt0),
                "succeeded": True, "included_in_ensemble": _ridge_included,
            }
            if not _ridge_included:
                _ridge_res["exclusion_reason"] = _ridge_excl_why
            if _ridge_included:
                all_val_preds["ridge"] = _ridge_vp
            print(f"Ridge done: OOF={_rm2:.4f}, time={_ridge_res['training_time_seconds']:.1f}s")

        except Exception as _ridge_exc:
            _ridge_res = {"succeeded": False, "included_in_ensemble": False,
                          "exclusion_reason": str(_ridge_exc), "oof_mae": None,
                          "training_time_seconds": float(time.time() - _rt0)}
            _ridge_coef_top5 = []
            print(f"Ridge FAILED: {_ridge_exc}")

        all_family_results["ridge"] = _ridge_res

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "-"*50 + "\nEnsemble\n" + "-"*50)
_included = {k: v for k, v in all_val_preds.items()
             if all_family_results.get(k, {}).get("included_in_ensemble", False)}

if len(_included) == 0:
    print("WARNING: no family produced valid predictions — using global mean")
    _included = {"lightgbm": np.full(len(val_df), train_target_mean)}

# ── Competence check: only apply Ridge weighting if Ridge is competitive ──────
if _ensemble_weighting == "ridge_weighted_1.5x" and "ridge" in _included:
    _oof_maes = {k: all_family_results[k]["oof_mae"]
                 for k in _included
                 if all_family_results.get(k, {}).get("oof_mae") is not None}
    if _oof_maes:
        _best_oof  = min(_oof_maes.values())
        _ridge_oof = _oof_maes.get("ridge")
        if _ridge_oof is not None and _ridge_oof <= 1.5 * _best_oof:
            _weighting_reason = (
                f"max_ks={_max_ks:.2f} > {_KS_THRESHOLD} threshold, "
                f"ridge_oof={_ridge_oof:.3f} within 1.5x best_oof={_best_oof:.3f}; "
                "ridge_weighted_1.5x applied"
            )
            print(f"Competence PASS: ridge_oof={_ridge_oof:.3f} <= 1.5*best={1.5*_best_oof:.3f}; keeping ridge_weighted_1.5x")
        else:
            _ensemble_weighting = "equal_median"
            _weighting_reason = (
                f"max_ks={_max_ks:.2f} > {_KS_THRESHOLD} threshold, "
                f"ridge_oof={_ridge_oof:.3f} > 1.5x best_oof={_best_oof:.3f}; "
                "using equal_median instead"
            )
            print(f"Competence FAIL: ridge_oof={_ridge_oof:.3f} > 1.5*best={1.5*_best_oof:.3f}; downgrading to equal_median")

if len(_included) == 1:
    ensemble_preds = next(iter(_included.values()))
elif _ensemble_weighting == "ridge_weighted_1.5x" and "ridge" in _included:
    _weights = np.array([1.5 if k == "ridge" else 1.0 for k in _included.keys()])
    _stacked = np.stack(list(_included.values()), axis=0)
    ensemble_preds = np.average(_stacked, axis=0, weights=_weights)
    print(f"Ridge-weighted avg: weights={dict(zip(_included.keys(), _weights.tolist()))}")
else:
    ensemble_preds = np.median(np.stack(list(_included.values()), axis=0), axis=0)

if _retune_applied and "clip_after_ensemble" in (_retune_applied or ""):
    ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f"Ensemble from {list(_included.keys())}: min={ensemble_preds.min():.2f}, "
      f"mean={ensemble_preds.mean():.2f}, max={ensemble_preds.max():.2f}")
print(f"NaN count: {np.isnan(ensemble_preds).sum()}")

# Ensemble disagreement
if len(_included) > 1:
    _stk = np.stack(list(_included.values()), axis=0)
    _rng = _stk.max(axis=0) - _stk.min(axis=0)
    _ens_disag = {
        "mean_disagreement": float(_rng.mean()),
        "n_high_disagreement_rows": int((_rng > 2 * _rng.std()).sum()),
    }
else:
    _ens_disag = {"mean_disagreement": 0.0, "n_high_disagreement_rows": 0}

print(f"Disagreement: mean={_ens_disag['mean_disagreement']:.4f}, "
      f"high_rows={_ens_disag['n_high_disagreement_rows']}")

# Ensemble OOF MAE
_family_oof_maes = [r["oof_mae"] for r in all_family_results.values()
                    if r.get("succeeded") and r.get("included_in_ensemble") and r.get("oof_mae") is not None]
ensemble_oof_mae = float(np.median(_family_oof_maes)) if _family_oof_maes else float('nan')

# Use LightGBM OOF as primary (validator compatibility)
oof_mae = master_oof_mae

# Algorithm name
_fam_names = [k for k, r in all_family_results.items() if r.get("included_in_ensemble")]
algorithm   = "+".join(f.capitalize() for f in _fam_names) + " ensemble"
if len(_fam_names) == 1:
    algorithm = _fam_names[0].capitalize()

# ── Write OOF predictions ──────────────────────────────────────────────────────
if master_oof_df is not None and len(master_oof_df) > 0:
    oof_path = os.path.join(REPORTS, "oof_predictions.csv")
    master_oof_df.to_csv(oof_path, index=False)
    print(f"\nWritten {oof_path}: {master_oof_df.shape}")

# ── Write predictions.csv ──────────────────────────────────────────────────────
_id_cands = ([profile.get("id_col")] if profile.get("id_col") else []) + group_cols + ([time_col] if time_col else [])
_id_out   = [c for c in _id_cands if c and c in val_df.columns]
if not _id_out:
    _id_out = [c for c in val_df.columns if c not in feature_cols][:1]

preds_df = val_df[_id_out].copy().reset_index(drop=True)
preds_df.insert(0, "row_id", range(len(preds_df)))
preds_df["predicted_target"] = ensemble_preds

_nan_ct = preds_df["predicted_target"].isna().sum()
if _nan_ct > 0:
    preds_df["predicted_target"] = preds_df["predicted_target"].fillna(train_target_mean)
    print(f"WARNING: filled {_nan_ct} NaN preds with global mean")

assert preds_df["predicted_target"].isna().sum() == 0, "NaN predictions remain!"
pred_path = os.path.join(REPORTS, "predictions.csv")
preds_df.to_csv(pred_path, index=False)
print(f"\nWritten {pred_path}: {preds_df.shape}")
print(preds_df.head(5).to_string())

# ── Feature importance ─────────────────────────────────────────────────────────
_lgbm_fi_ok = _lgbm_fi is not None
_xgb_fi_ok  = (_xgb_fi is not None and "xgboost" in all_family_results
               and all_family_results["xgboost"].get("included_in_ensemble"))

if _lgbm_fi_ok and _xgb_fi_ok:
    _lgbm_a = _lgbm_fi.reindex(feature_cols).fillna(0)
    _xgb_a  = _xgb_fi.reindex(feature_cols).fillna(0)
    _combined_fi = ((_lgbm_a / (_lgbm_a.sum() or 1)) + (_xgb_a / (_xgb_a.sum() or 1))) / 2
    _combined_fi = _combined_fi.sort_values(ascending=False)
elif _lgbm_fi_ok:
    _combined_fi = _lgbm_fi
else:
    _combined_fi = pd.Series(dtype=float)

top10    = [{"feature": str(k), "importance": float(v)} for k, v in _combined_fi.head(10).items()]
all_imp  = {str(k): float(v) for k, v in _combined_fi.items()}
_ridge_coef_top5 = _ridge_coef_top5 if "ridge" in all_family_results and all_family_results["ridge"].get("succeeded") else []

# ── Write model_results.json ───────────────────────────────────────────────────
training_time = int(time.time() - start_time)

results = {
    "algorithm": algorithm,
    "adaptive_choice": {
        "branch": ens_info["branch"],
        "families_selected": families_plan,
        "families_included_in_ensemble": _fam_names,
        "reasoning": ens_info["reasoning"],
        "ensemble_weighting": _ensemble_weighting,
        "weighting_reason": _weighting_reason,
        "ridge_excluded_reason": (
            all_family_results.get("ridge", {}).get("exclusion_reason")
            if (not all_family_results.get("ridge", {}).get("included_in_ensemble")
                and all_family_results.get("ridge", {}).get("succeeded"))
            else None
        ),
    },
    "families": {
        "lightgbm": all_family_results.get("lightgbm", {}),
        **({k: all_family_results[k] for k in ["xgboost", "catboost", "ridge"] if k in all_family_results}),
    },
    "ensemble_oof_mae": float(ensemble_oof_mae),
    "n_families_in_ensemble": len(_fam_names),
    "ensemble_disagreement": _ens_disag,
    "feature_importance_top10": top10,
    "feature_importance_all":   all_imp,
    "ridge_top_coefficients":   _ridge_coef_top5,
    # ── Backward-compat fields (validator reads these) ──
    "objective":   "regression_l1",
    "best_params": {k: (float(v) if isinstance(v, (float, np.floating))
                        else int(v) if isinstance(v, (int, np.integer)) else v)
                    for k, v in _lgbm_hp.items()},
    "n_estimators": _lgbm_ne,
    "n_seeds": 5,
    "cv_scheme": cv_scheme,
    "oof_mae": float(oof_mae),
    "oof_cv_scheme": cv_scheme,
    "per_fold_maes": [float(x) for x in master_fold_maes],
    "walk_forward_mae": float(oof_mae),
    "probe_mae_80_20": float(probe_mae),
    "training_time_seconds": training_time,
    "optuna_trials_completed": int(_lgbm_nt),
    "optuna_succeeded": _lgbm_nt > 0,
    "val_prediction_stats": {
        "min": float(ensemble_preds.min()), "max": float(ensemble_preds.max()),
        "mean": float(ensemble_preds.mean()), "std": float(ensemble_preds.std()),
    },
    "n_features": len(feature_cols),
    "n_train_rows": int(n_train),
    "n_val_rows": int(len(val_df)),
    "retune_applied": _retune_applied,
}

results_path = os.path.join(REPORTS, "model_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWritten {results_path}")

# Summary (suppress huge dicts)
_sum = {k: v for k, v in results.items() if k not in ("feature_importance_all", "feature_importance_top10")}
print(json.dumps(_sum, indent=2, default=str))

# ── Marker file ────────────────────────────────────────────────────────────────
marker_path = os.path.join(REPORTS, "modeler_was_here.txt")
with open(marker_path, "w") as f:
    f.write(f"modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n")
    f.write(f"Algorithm: {algorithm}\n")
    f.write(f"Branch: {ens_info['branch']}, families: {_fam_names}\n")
    f.write(f"OOF MAE (LightGBM, {cv_scheme}): {oof_mae:.4f}\n")
    f.write(f"Ensemble OOF MAE: {ensemble_oof_mae:.4f}\n")
    f.write(f"Training time: {training_time}s\n")
print(f"Written {marker_path}")

print("\n" + "="*60)
print("MODELER COMPLETE")
print(f"Algorithm       : {algorithm}")
print(f"Branch          : {ens_info['branch']}")
print(f"Families        : {_fam_names}")
print(f"OOF MAE (LGBM)  : {oof_mae:.4f}")
print(f"Ensemble OOF MAE: {ensemble_oof_mae:.4f}")
print(f"Val mean        : {ensemble_preds.mean():.3f}")
print(f"Time            : {training_time}s")
print("="*60)
