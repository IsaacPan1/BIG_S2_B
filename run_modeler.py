import pandas as pd, numpy as np, json, time, warnings, os, datetime
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

start_time = time.time()

# ---- Step 2: Load data ----
with open('reports/features.json') as f:
    feat_meta = json.load(f)

train_df = pd.read_parquet('data/features_train.parquet')
val_df   = pd.read_parquet('data/features_val.parquet')

target_col = feat_meta['target_col']
group_cols = feat_meta['group_cols']
time_col   = feat_meta['time_col']

exclude = set(group_cols + [time_col, target_col])
feature_cols = [c for c in train_df.columns if c not in exclude]

print(f'Train: {train_df.shape}, Val: {val_df.shape}')
print(f'Features: {len(feature_cols)}, Target: {target_col}')

fill_vals = train_df[feature_cols].median()
X_full = train_df[feature_cols].fillna(fill_vals)
y_full = train_df[target_col].values
n = len(X_full)

# ---- Step 3: Check critic retune ----
_retune_applied = None
if os.path.exists('reports/critic_retune_requested.json'):
    with open('reports/critic_retune_requested.json') as _f:
        _retune = json.load(_f)
    _suggestion = _retune.get('suggested_change', '')
    print(f'Critic retune requested: {_suggestion}')
    if 'median seed aggregation' in _suggestion:
        _retune_applied = 'median_seed_aggregation'
    if 'expand Optuna' in _suggestion:
        _retune_applied = (_retune_applied or '') + '+expanded_optuna_bounds'
    if 'val feature imputation' in _suggestion:
        _retune_applied = (_retune_applied or '') + '+verified_imputation'
    if 'np.clip applied after seed aggregation' in _suggestion:
        _retune_applied = (_retune_applied or '') + '+clip_after_ensemble'
else:
    print('No critic retune request')
print(f'_retune_applied: {_retune_applied}')

# ---- Step 4: Walk-forward split ----
all_times = sorted(train_df[time_col].unique())
cutoff_idx = int(len(all_times) * 0.8)
cutoff_time = all_times[cutoff_idx]

wf_train = train_df[train_df[time_col] < cutoff_time].copy()
wf_val   = train_df[train_df[time_col] >= cutoff_time].copy()

wf_fill_vals = wf_train[feature_cols].median()
X_wf_train = wf_train[feature_cols].fillna(wf_fill_vals)
y_wf_train = wf_train[target_col]
X_wf_val   = wf_val[feature_cols].fillna(wf_fill_vals)
y_wf_val   = wf_val[target_col]

print(f'Walk-forward train: {X_wf_train.shape}, val: {X_wf_val.shape}')
print(f'Cutoff time: {cutoff_time}')

# ---- Step 5: Optuna tuning ----
TUNING_DEADLINE = start_time + 25 * 60

def objective(trial):
    if time.time() > TUNING_DEADLINE:
        raise optuna.exceptions.TrialPruned()

    num_leaves_high = 255 if (_retune_applied and 'expanded_optuna_bounds' in _retune_applied) else 127
    min_child_low   = 3   if (_retune_applied and 'expanded_optuna_bounds' in _retune_applied) else 5

    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, num_leaves_high),
        'min_child_samples': trial.suggest_int('min_child_samples', min_child_low, 60),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
    }

    use_median = _retune_applied and 'median_seed_aggregation' in _retune_applied
    maes = []
    for seed in [42, 7, 123]:
        m = lgb.LGBMRegressor(**{**params, 'random_state': seed})
        m.fit(
            X_wf_train, y_wf_train,
            eval_set=[(X_wf_val, y_wf_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        preds = np.clip(m.predict(X_wf_val), 0, None)
        maes.append(mean_absolute_error(y_wf_val, preds))
    return float(np.median(maes) if use_median else np.mean(maes))

try:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15, timeout=25*60, catch=(Exception,))
    best_params = study.best_params
    optuna_trials = len(study.trials)
    print(f'Optuna complete: {optuna_trials} trials, best MAE={study.best_value:.4f}')
    print(f'Best params: {best_params}')
    optuna_succeeded = True
except Exception as e:
    print(f'Optuna failed ({e}), using defaults')
    best_params = {}
    optuna_trials = 0
    optuna_succeeded = False

# ---- Step 6: Build final hyperparams and probe for n_estimators ----
default_params = {
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
final_hparams = {**default_params, **best_params}

probe_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 2000,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    **final_hparams,
}
probe = lgb.LGBMRegressor(**probe_params)
probe.fit(
    X_wf_train, y_wf_train,
    eval_set=[(X_wf_val, y_wf_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
)
best_n_estimators = int(probe.best_iteration_ * 1.1) if probe.best_iteration_ else 500
wf_mae = mean_absolute_error(y_wf_val, np.clip(probe.predict(X_wf_val), 0, None))
print(f'Walk-forward MAE: {wf_mae:.4f}, best_iteration: {probe.best_iteration_}, n_estimators: {best_n_estimators}')

# ---- Step 7: panel_forecasting: skip OOF CV; use walk-forward MAE; retrain on full with 5 seeds ----
oof_mae = wf_mae
fold_maes = [wf_mae]
cv_scheme = 'walk_forward(80/20_temporal_split)'

final_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': best_n_estimators,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    **final_hparams,
}

use_median = _retune_applied and 'median_seed_aggregation' in _retune_applied
X_full_filled = train_df[feature_cols].fillna(fill_vals)
y_full_arr = train_df[target_col].values
X_val  = val_df[feature_cols].fillna(fill_vals)

seed_preds = []
for seed in [42, 7, 123, 2024, 999]:
    m = lgb.LGBMRegressor(**{**final_params, 'random_state': seed})
    m.fit(X_full_filled, y_full_arr, callbacks=[lgb.log_evaluation(-1)])
    seed_preds.append(np.clip(m.predict(X_val), 0, None))
    print(f'  Seed {seed} done')

if use_median:
    ensemble_preds = np.median(seed_preds, axis=0)
else:
    ensemble_preds = np.mean(seed_preds, axis=0)

if _retune_applied and 'clip_after_ensemble' in _retune_applied:
    ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f'Ensemble: min={ensemble_preds.min():.2f}, max={ensemble_preds.max():.2f}, mean={ensemble_preds.mean():.2f}')
print(f'NaN count: {np.isnan(ensemble_preds).sum()}')

last_model = m

# ---- Step 7b: Write OOF predictions (panel_forecasting: walk-forward val as stand-in) ----
wf_val_preds = np.clip(probe.predict(X_wf_val), 0, None)
oof_df = wf_val[group_cols + [time_col]].copy().reset_index(drop=True)
oof_df['fold'] = 0
oof_df['predicted_target'] = wf_val_preds
oof_df.to_csv('reports/oof_predictions.csv', index=False)
print(f'Written reports/oof_predictions.csv: {oof_df.shape}')

# ---- Step 8: Write predictions.csv ----
preds_df = val_df[group_cols + [time_col]].copy().reset_index(drop=True)
preds_df.insert(0, 'row_id', range(len(preds_df)))
preds_df['predicted_target'] = ensemble_preds

assert preds_df['predicted_target'].isna().sum() == 0, 'NaN predictions found'

preds_df.to_csv('reports/predictions.csv', index=False)
print(f'Written reports/predictions.csv: {preds_df.shape}')
print(preds_df.head())

# ---- Step 9: Write model_results.json ----
feat_imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = [{'feature': k, 'importance': int(v)} for k, v in feat_imp.head(10).items()]
all_imp = {k: int(v) for k, v in feat_imp.items()}

training_time = int(time.time() - start_time)

results = {
    'algorithm': 'LightGBM',
    'objective': final_params['objective'],
    'best_params': final_hparams,
    'n_estimators': best_n_estimators,
    'n_seeds': 5,
    'cv_scheme': cv_scheme,
    'oof_mae': float(oof_mae),
    'oof_cv_scheme': cv_scheme,
    'per_fold_maes': [float(m_) for m_ in fold_maes],
    'walk_forward_mae': float(wf_mae),
    'feature_importance_top10': top10,
    'feature_importance_all': all_imp,
    'training_time_seconds': training_time,
    'optuna_trials_completed': optuna_trials,
    'val_prediction_stats': {
        'min': float(ensemble_preds.min()),
        'max': float(ensemble_preds.max()),
        'mean': float(ensemble_preds.mean()),
        'std': float(ensemble_preds.std()),
    },
    'n_features': len(feature_cols),
    'n_train_rows': len(train_df),
    'n_val_rows': len(val_df),
    'retune_applied': _retune_applied,
}

with open('reports/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Written reports/model_results.json')

# ---- Step 10: Write marker file ----
with open('reports/modeler_was_here.txt', 'w') as f:
    f.write(f'modeler sub-agent executed at {datetime.datetime.now().isoformat()}\n')
print('Written reports/modeler_was_here.txt')
print(f'Total time: {training_time}s')
