import pandas as pd
import json
import os

sub    = pd.read_csv('submission.csv')
sample = pd.read_csv('data/sample_submission.csv')

print('=== Submission Verification ===')
print(f'submission.csv shape: {sub.shape}')
print(f'sample_submission shape: {sample.shape}')
print(f'Columns match: {sub.columns.tolist() == sample.columns.tolist()}')
print(f'Row count match: {len(sub) == len(sample)}')
print(f'NaN in weekly_sales: {sub["weekly_sales"].isna().sum()}')
print(f'Negative predictions: {(sub["weekly_sales"] < 0).sum()}')
print(f'Pred stats: min={sub["weekly_sales"].min():.2f}, max={sub["weekly_sales"].max():.2f}, mean={sub["weekly_sales"].mean():.2f}')
print()

files = [
    'reports/profile.json',
    'reports/schema_analysis.md',
    'reports/features.json',
    'reports/model_results.json',
    'reports/predictions.csv',
    'data/features_train.parquet',
    'data/features_val.parquet',
    'submission.csv',
    'report.pdf',
]
for fpath in files:
    size = os.path.getsize(fpath) if os.path.exists(fpath) else -1
    status = 'OK' if size > 0 else 'MISSING'
    print(f'{fpath}: {size} bytes [{status}]')

print()
with open('reports/profile.json') as f:
    profile = json.load(f)
pred = pd.read_csv('reports/predictions.csv')
print(f'n_val_rows in profile: {profile["n_val_rows"]}')
print(f'predictions.csv rows:  {len(pred)}')
print(f'Target col match:      {pred.columns[-1] == profile["target_col"]}')
print(f'NaN in predictions:    {pred[profile["target_col"]].isna().sum()}')
print()

all_ok = (
    len(sub) == len(sample)
    and sub.columns.tolist() == sample.columns.tolist()
    and sub['weekly_sales'].isna().sum() == 0
    and len(pred) == profile['n_val_rows']
    and os.path.getsize('report.pdf') > 0
)
print('=== ALL CHECKS PASSED ===' if all_ok else '=== SOME CHECKS FAILED ===')
