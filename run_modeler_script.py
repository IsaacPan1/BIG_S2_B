import pandas as pd, numpy as np, json, time, warnings, os, datetime
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

start_time = time.time()

# ── Load context ──────────────────────────────────────────────────────────────
with open('reports/features.json') as f:
    feat_meta = json.load(f)
with open('reports/profile.json') as f:
    profile = json.load(f)

train_df = pd.read_parquet('data/features_train.parquet')
val_df   = pd.read_parquet('data/features_val.parquet')

target_col   = feat_meta['target_col']
group_cols   = feat_meta['group_cols']
time_col     = feat_meta['time_col']
problem_type = profile['problem_type']

exclude      = set(group_cols + ([time_col] if time_col else []) + [target_col, 'adversarial_weights'])
feature_cols = [c for c in train_df.columns if c not in exclude and c != 'state_doh_release']
fill_vals    = train_df[feature_cols].median()
print(f'Features: {len(feature_cols)}, Target: {target_col}')

# Adversarial weights
_av_info = feat_meta.get('adversarial_validation', {})
_adv_weights = None
if _av_info.get('weights_applied', False) and 'adversarial_weights' in train_df.columns:
    _adv_weights = train_df['adversarial_weights'].fillna(1.0).values
    print(f'Adv weights: mean={_adv_weights.mean():.3f}, max={_adv_weights.max():.3f}')

# Adaptive ensemble selection
n_train = len(train_df)
_dist_shifts = profile.get('distribution_shifts', [])
_max_ks = max((d.get('ks_statistic', 0.0) for d in _dist_shifts if isinstance(d, dict)), default=0.0)
print(f'n_train={n_train}, max_ks={_max_ks:.3f}')
# Branch 1: panel_forecasting + n_train>=1000 -> LightGBM + XGBoost + Ridge
ensemble_families = ['lgbm', 'xgboost', 'ridge']
weighting_mode    = 'ridge_weighted_1.5x'  # max_ks=0.56 > 0.40
print(f'Ensemble: {ensemble_families}, weighting: {weighting_mode}')

# Log1p transform (skew=2.70 > 2.0)
use_log1p = True
def T(y):  return np.log1p(y)
def Ti(y): return np.expm1(y)
train_target_max  = float(train_df[target_col].max())
train_target_mean = float(train_df[target_col].mean())
print(f'train_target_max={train_target_max:.2f}, mean={train_target_mean:.2f}')

# ── Walk-forward split (80/20 by time) ───────────────────────────────────────
all_ords   = sorted(train_df['period_id_ord'].unique())
cutoff_idx = int(len(all_ords) * 0.8)
cutoff_ord = all_ords[cutoff_idx]
print(f'WF cutoff: ord={cutoff_ord} ({cutoff_idx}/{len(all_ords)} periods)')

wf_train_df  = train_df[train_df['period_id_ord'] < cutoff_ord].copy()
wf_val_df    = train_df[train_df['period_id_ord'] >= cutoff_ord].copy()
wf_fill      = wf_train_df[feature_cols].median()

X_wft = wf_train_df[feature_cols].fillna(wf_fill)
y_wft = T(wf_train_df[target_col].values)
X_wfv = wf_val_df[feature_cols].fillna(wf_fill)
y_wfv_raw = wf_val_df[target_col].values
y_wfv = T(y_wfv_raw)
_wf_sw = _adv_weights[wf_train_df.index.values] if _adv_weights is not None else None

X_full = train_df[feature_cols].fillna(fill_vals)
y_full = T(train_df[target_col].values)
y_full_raw = train_df[target_col].values
X_val  = val_df[feature_cols].fillna(fill_vals)
print(f'WF train={X_wft.shape}, WF val={X_wfv.shape}, Full={X_full.shape}, Val={X_val.shape}')

TUNING_DEADLINE = start_time + 20 * 60  # 20 min

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 1: LightGBM
# ══════════════════════════════════════════════════════════════════════════════
lgbm_succeeded = False
lgbm_oof_mae   = None
lgbm_val_preds = None
lgbm_exclusion = None
lgbm_trials    = 0
final_lgbm_hparams = {}
best_n_est_lgbm = 500
last_lgbm_model = None
final_lgbm_params = {}

try:
    print('\n=== LightGBM Optuna (15 trials) ===')
    def lgbm_objective(trial):
        if time.time() > TUNING_DEADLINE:
            raise optuna.exceptions.TrialPruned()
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 500,
            'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'verbose': -1, 'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves':    trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
        }
        maes = []
        for seed in [42, 7, 123]:
            m = lgb.LGBMRegressor(**{**params, 'random_state': seed})
            m.fit(X_wft, y_wft, sample_weight=_wf_sw,
                  eval_set=[(X_wfv, y_wfv)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            raw = Ti(np.clip(m.predict(X_wfv), 0, None))
            maes.append(mean_absolute_error(y_wfv_raw, raw))
        return float(np.mean(maes))

    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(lgbm_objective, n_trials=15, timeout=18 * 60, catch=(Exception,))
    best_lgbm = study_lgbm.best_params
    lgbm_trials = len([t for t in study_lgbm.trials if t.state.name == 'COMPLETE'])
    print(f'LightGBM best MAE={study_lgbm.best_value:.4f} ({lgbm_trials} trials)')
    print(f'Best params: {best_lgbm}')

    default_lgbm = {'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 20,
                    'feature_fraction': 0.8, 'bagging_fraction': 0.8}
    final_lgbm_hparams = {**default_lgbm, **best_lgbm}

    probe_params = {'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
                    'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1,
                    'n_jobs': -1, 'random_state': 42, **final_lgbm_hparams}
    probe = lgb.LGBMRegressor(**probe_params)
    probe.fit(X_wft, y_wft, sample_weight=_wf_sw,
              eval_set=[(X_wfv, y_wfv)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    best_n_est_lgbm = int((probe.best_iteration_ or 500) * 1.1)
    wf_lgbm_raw = Ti(np.clip(probe.predict(X_wfv), 0, None))
    wf_lgbm_mae = mean_absolute_error(y_wfv_raw, wf_lgbm_raw)
    print(f'WF MAE (LightGBM)={wf_lgbm_mae:.4f}, n_estimators={best_n_est_lgbm}')

    final_lgbm_params = {'objective': 'regression_l1', 'metric': 'mae',
                         'n_estimators': best_n_est_lgbm, 'bagging_freq': 5,
                         'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1,
                         **final_lgbm_hparams}

    seed_preds_lgbm = []
    for seed in [42, 7, 123, 2024, 999]:
        m = lgb.LGBMRegressor(**{**final_lgbm_params, 'random_state': seed})
        m.fit(X_full, y_full, sample_weight=_adv_weights, callbacks=[lgb.log_evaluation(-1)])
        seed_preds_lgbm.append(Ti(np.clip(m.predict(X_val), 0, None)))
    lgbm_val_preds = np.mean(seed_preds_lgbm, axis=0)
    lgbm_oof_mae   = wf_lgbm_mae
    lgbm_succeeded = True
    last_lgbm_model = m
    print(f'LightGBM val preds: min={lgbm_val_preds.min():.3f}, max={lgbm_val_preds.max():.3f}, mean={lgbm_val_preds.mean():.3f}')
    print(f'Elapsed: {(time.time()-start_time)/60:.1f} min')
except Exception as e:
    lgbm_exclusion = str(e)
    print(f'LightGBM FAILED: {e}')

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 2: XGBoost
# ══════════════════════════════════════════════════════════════════════════════
xgb_succeeded = False
xgb_oof_mae   = None
xgb_val_preds = None
xgb_exclusion = None
xgb_trials    = 0
final_xgb_hparams = {}
best_n_xgb = 500

if time.time() - start_time < 20 * 60:
    try:
        print('\n=== XGBoost Optuna (15 trials) ===')
        XGB_DEADLINE = start_time + 40 * 60

        def xgb_objective(trial):
            if time.time() > XGB_DEADLINE:
                raise optuna.exceptions.TrialPruned()
            params = {
                'objective': 'reg:absoluteerror',
                'n_estimators': 500,
                'tree_method': 'hist', 'device': 'cpu', 'verbosity': 0, 'n_jobs': -1,
                'early_stopping_rounds': 50,
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth':        trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha':        trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda':       trial.suggest_float('reg_lambda', 0, 1),
            }
            maes = []
            for seed in [42, 7, 123]:
                m2 = xgb.XGBRegressor(**{**params, 'random_state': seed})
                m2.fit(X_wft, y_wft, sample_weight=_wf_sw,
                       eval_set=[(X_wfv, y_wfv)], verbose=False)
                raw = Ti(np.clip(m2.predict(X_wfv), 0, None))
                maes.append(mean_absolute_error(y_wfv_raw, raw))
            return float(np.mean(maes))

        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(xgb_objective, n_trials=15, timeout=18 * 60, catch=(Exception,))

        completed_xgb = [t for t in study_xgb.trials if t.state.name == 'COMPLETE']
        xgb_trials = len(completed_xgb)
        if xgb_trials > 0:
            best_xgb_params = study_xgb.best_params
            print(f'XGBoost best MAE={study_xgb.best_value:.4f} ({xgb_trials} trials)')
        else:
            best_xgb_params = {}
            print('XGBoost: no complete trials, using defaults')

        default_xgb = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3,
                       'subsample': 0.8, 'colsample_bytree': 0.8,
                       'reg_alpha': 0.1, 'reg_lambda': 0.1}
        final_xgb_hparams = {**default_xgb, **best_xgb_params}

        xgb_probe = xgb.XGBRegressor(**{**final_xgb_hparams,
                                        'objective': 'reg:absoluteerror',
                                        'n_estimators': 2000, 'early_stopping_rounds': 100,
                                        'tree_method': 'hist', 'device': 'cpu',
                                        'verbosity': 0, 'n_jobs': -1, 'random_state': 42})
        xgb_probe.fit(X_wft, y_wft, sample_weight=_wf_sw,
                      eval_set=[(X_wfv, y_wfv)], verbose=False)
        best_n_xgb = int((xgb_probe.best_iteration or 500) * 1.1)
        xgb_wf_raw = Ti(np.clip(xgb_probe.predict(X_wfv), 0, None))
        xgb_wf_mae = mean_absolute_error(y_wfv_raw, xgb_wf_raw)
        print(f'WF MAE (XGBoost)={xgb_wf_mae:.4f}, n_estimators={best_n_xgb}')

        seed_preds_xgb = []
        for seed in [42, 7, 123, 2024, 999]:
            m2 = xgb.XGBRegressor(**{**final_xgb_hparams,
                                     'objective': 'reg:absoluteerror',
                                     'n_estimators': best_n_xgb,
                                     'tree_method': 'hist', 'device': 'cpu',
                                     'verbosity': 0, 'n_jobs': -1, 'random_state': seed})
            m2.fit(X_full, y_full, sample_weight=_adv_weights)
            seed_preds_xgb.append(Ti(np.clip(m2.predict(X_val), 0, None)))
        xgb_val_preds = np.mean(seed_preds_xgb, axis=0)
        xgb_oof_mae   = xgb_wf_mae
        xgb_succeeded = True
        print(f'XGBoost val preds: min={xgb_val_preds.min():.3f}, max={xgb_val_preds.max():.3f}, mean={xgb_val_preds.mean():.3f}')
        print(f'Elapsed: {(time.time()-start_time)/60:.1f} min')
    except Exception as e:
        xgb_exclusion = str(e)
        print(f'XGBoost FAILED: {e}')
else:
    xgb_exclusion = 'Skipped: elapsed > 20 min when XGBoost would start'
    print('XGBoost SKIPPED (time budget)')

# ══════════════════════════════════════════════════════════════════════════════
# FAMILY 3: Ridge
# ══════════════════════════════════════════════════════════════════════════════
ridge_succeeded = False
ridge_oof_mae   = None
ridge_val_preds = None
ridge_exclusion = None

completed_families = sum([lgbm_succeeded, xgb_succeeded])
if not (completed_families >= 2 and time.time() - start_time > 30 * 60):
    try:
        print('\n=== Ridge (alpha probe) ===')
        scaler = StandardScaler()
        X_wft_sc = scaler.fit_transform(X_wft.fillna(0))
        X_wfv_sc = scaler.transform(X_wfv.fillna(0))

        best_alpha, best_ridge_mae = 1.0, np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            r = Ridge(alpha=alpha)
            r.fit(X_wft_sc, y_wft, sample_weight=_wf_sw)
            raw = Ti(np.clip(r.predict(X_wfv_sc), 0, None))
            mae = mean_absolute_error(y_wfv_raw, raw)
            print(f'  alpha={alpha}: WF MAE={mae:.4f}')
            if mae < best_ridge_mae:
                best_ridge_mae = mae
                best_alpha = alpha

        print(f'Ridge best alpha={best_alpha}, WF MAE={best_ridge_mae:.4f}')

        scaler_full = StandardScaler()
        X_full_sc = scaler_full.fit_transform(X_full.fillna(0))
        X_val_sc  = scaler_full.transform(X_val.fillna(0))

        ridge_final = Ridge(alpha=best_alpha)
        ridge_final.fit(X_full_sc, y_full, sample_weight=_adv_weights)
        ridge_raw_preds = Ti(np.clip(ridge_final.predict(X_val_sc), 0, None))

        ridge_pred_max  = float(ridge_raw_preds.max())
        ridge_pred_mean = float(ridge_raw_preds.mean())
        print(f'Ridge val: min={ridge_raw_preds.min():.3f}, max={ridge_pred_max:.3f}, mean={ridge_pred_mean:.3f}')

        if (ridge_raw_preds < 0).any():
            ridge_raw_preds = np.clip(ridge_raw_preds, 0, None)
            print('Ridge: clipped negative predictions to 0')

        if ridge_pred_max > 5 * train_target_max:
            ridge_exclusion = f'Ridge pred_max={ridge_pred_max:.1f} > 5*train_max={5*train_target_max:.1f}'
            print(f'Ridge EXCLUDED: {ridge_exclusion}')
        elif abs(ridge_pred_mean - train_target_mean) / abs(train_target_mean) > 1.0:
            ridge_exclusion = f'Ridge pred_mean={ridge_pred_mean:.3f} deviates >100% from train_mean={train_target_mean:.3f}'
            print(f'Ridge EXCLUDED: {ridge_exclusion}')
        else:
            ridge_val_preds = ridge_raw_preds
            ridge_oof_mae   = best_ridge_mae
            ridge_succeeded = True
            print(f'Ridge INCLUDED, WF MAE={ridge_oof_mae:.4f}')
        print(f'Elapsed: {(time.time()-start_time)/60:.1f} min')
    except Exception as e:
        ridge_exclusion = str(e)
        print(f'Ridge FAILED: {e}')
else:
    ridge_exclusion = 'Skipped: 2 families done and elapsed > 30 min'
    print('Ridge SKIPPED (time budget)')

# ══════════════════════════════════════════════════════════════════════════════
# Aggregate ensemble
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Ensemble aggregation ===')
all_val_preds_list = []
all_family_names = []
all_oof_maes  = {}

if lgbm_succeeded:
    all_val_preds_list.append(lgbm_val_preds)
    all_family_names.append('lgbm')
    all_oof_maes['lgbm'] = lgbm_oof_mae

if xgb_succeeded:
    all_val_preds_list.append(xgb_val_preds)
    all_family_names.append('xgboost')
    all_oof_maes['xgboost'] = xgb_oof_mae

if ridge_succeeded:
    all_val_preds_list.append(ridge_val_preds)
    all_family_names.append('ridge')
    all_oof_maes['ridge'] = ridge_oof_mae

print(f'Families included: {all_family_names}')
print(f'OOF MAEs: {all_oof_maes}')

stack = np.vstack(all_val_preds_list)

best_oof = min(all_oof_maes.values())
weighting_reason = ''
final_weighting = weighting_mode

if weighting_mode == 'ridge_weighted_1.5x' and ridge_succeeded:
    if all_oof_maes['ridge'] <= 1.5 * best_oof:
        weighting_reason = (
            f"max_ks={_max_ks:.2f} > 0.40 threshold, "
            f"ridge_oof={all_oof_maes['ridge']:.4f} within 1.5x best_oof={best_oof:.4f}; "
            f"ridge_weighted_1.5x applied"
        )
        weights = [1.5 if n == 'ridge' else 1.0 for n in all_family_names]
        ensemble_preds = np.average(stack, axis=0, weights=weights)
    else:
        final_weighting = 'equal_median'
        weighting_reason = (
            f"max_ks={_max_ks:.2f} > 0.40 threshold, "
            f"ridge_oof={all_oof_maes['ridge']:.4f} > 1.5x best_oof={best_oof:.4f}; "
            f"using equal_median instead"
        )
        ensemble_preds = np.median(stack, axis=0)
elif weighting_mode == 'ridge_weighted_1.5x' and not ridge_succeeded:
    final_weighting = 'equal_median'
    weighting_reason = f"max_ks={_max_ks:.2f} > 0.40; Ridge not in ensemble; falling back to equal_median"
    ensemble_preds = np.median(stack, axis=0)
else:
    weighting_reason = f"max_ks={_max_ks:.2f} <= 0.40; equal_median"
    ensemble_preds = np.median(stack, axis=0)

ensemble_preds = np.clip(ensemble_preds, 0, None)
print(f'Ensemble: min={ensemble_preds.min():.3f}, max={ensemble_preds.max():.3f}, mean={ensemble_preds.mean():.3f}')
print(f'Weighting: {final_weighting}')
print(f'Reason: {weighting_reason}')
print(f'NaN count: {np.isnan(ensemble_preds).sum()}')

wf_mae  = lgbm_oof_mae if lgbm_succeeded else float('nan')
oof_mae = wf_mae

# ── Write predictions.csv ─────────────────────────────────────────────────────
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, 'row_id', range(len(preds_df)))
preds_df['predicted_target'] = ensemble_preds

assert preds_df['predicted_target'].isna().sum() == 0, 'NaN predictions!'
preds_df.to_csv('reports/predictions.csv', index=False)
print(f'\nWritten reports/predictions.csv: {preds_df.shape}')
print(preds_df.head())

# ── OOF predictions on training set (walk-forward held-out set) ───────────────
print('\n=== OOF predictions ===')
seed_preds_oof = []
for seed in [42, 7, 123, 2024, 999]:
    m_oof = lgb.LGBMRegressor(**{**final_lgbm_params, 'random_state': seed})
    m_oof.fit(X_wft, y_wft, sample_weight=_wf_sw, callbacks=[lgb.log_evaluation(-1)])
    seed_preds_oof.append(Ti(np.clip(m_oof.predict(X_wfv), 0, None)))
wf_oof_preds = np.mean(seed_preds_oof, axis=0)

oof_df = wf_val_df[group_cols + [time_col]].copy().reset_index(drop=True)
oof_df['fold'] = 0
oof_df['predicted_target'] = wf_oof_preds
oof_df.to_csv('reports/oof_predictions.csv', index=False)
oof_check = mean_absolute_error(y_wfv_raw, wf_oof_preds)
print(f'Written reports/oof_predictions.csv: {oof_df.shape}, OOF MAE={oof_check:.4f}')

# ── Write model_results.json ──────────────────────────────────────────────────
feat_imp = pd.Series(last_lgbm_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10    = [{'feature': k, 'importance': int(v)} for k, v in feat_imp.head(10).items()]
all_imp  = {k: int(v) for k, v in feat_imp.items()}

training_time = int(time.time() - start_time)

results = {
    'algorithm': 'LightGBM+XGBoost+Ridge (adaptive ensemble)',
    'objective': 'regression_l1',
    'log1p_transform': use_log1p,
    'best_params': final_lgbm_hparams,
    'n_estimators': best_n_est_lgbm,
    'n_seeds': 5,
    'cv_scheme': 'WalkForward(cutoff_ord=61, 80pct_train)',
    'oof_mae': float(oof_mae),
    'oof_cv_scheme': 'WalkForward(cutoff_ord=61, 80pct_train)',
    'per_fold_maes': [float(wf_mae)],
    'walk_forward_mae': float(wf_mae),
    'feature_importance_top10': top10,
    'feature_importance_all': all_imp,
    'training_time_seconds': training_time,
    'optuna_trials_completed': lgbm_trials,
    'val_prediction_stats': {
        'min': float(ensemble_preds.min()),
        'max': float(ensemble_preds.max()),
        'mean': float(ensemble_preds.mean()),
        'std': float(ensemble_preds.std()),
    },
    'n_features': len(feature_cols),
    'n_train_rows': len(train_df),
    'n_val_rows': len(val_df),
    'retune_applied': None,
    'adaptive_choice': {
        'problem_type': problem_type,
        'n_train': n_train,
        'ensemble_families_planned': ensemble_families,
        'ensemble_families_succeeded': all_family_names,
        'max_ks': float(_max_ks),
        'ensemble_weighting': final_weighting,
        'weighting_reason': weighting_reason,
        'family_oof_maes': {k: float(v) for k, v in all_oof_maes.items()},
        'lgbm_succeeded': lgbm_succeeded,
        'xgboost_succeeded': xgb_succeeded,
        'xgboost_exclusion_reason': xgb_exclusion,
        'ridge_succeeded': ridge_succeeded,
        'ridge_exclusion_reason': ridge_exclusion,
        'adversarial_validation': {
            'used_weights': _adv_weights is not None,
            'auc_from_feature_engineer': _av_info.get('auc_train_vs_val'),
            'weight_range_used': _av_info.get('weight_range') if _adv_weights is not None else None,
        },
    },
}

with open('reports/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Written reports/model_results.json')

# ── Marker ────────────────────────────────────────────────────────────────────
with open('reports/modeler_was_here.txt', 'w') as f:
    f.write(f'modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n')
print('Written reports/modeler_was_here.txt')
print(f'\nTotal elapsed: {(time.time()-start_time)/60:.1f} min')
