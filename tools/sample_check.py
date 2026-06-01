import pandas as pd, json

profile = json.load(open('reports/profile.json'))
train_files = profile['train_files']

df = pd.read_csv(f'data/{train_files[0]}')
print('Shape:', df.shape)
print()
print('Head:')
print(df.head())
print()
print('Dtypes:')
print(df.dtypes)
print()
print('Null rates (%):')
null_rates = df.isnull().mean() * 100
print(null_rates)
print()
print('Target range check:')
print(f'load_mw min={df.load_mw.min():.2f}, max={df.load_mw.max():.2f}, mean={df.load_mw.mean():.2f}')
print()
print('Region counts:')
print(df.groupby('region_id').size().describe())
print()
ts_min = df['timestamp'].min()
ts_max = df['timestamp'].max()
print(f'Train timestamp range: {ts_min} to {ts_max}')

val = pd.read_csv('data/val_features.csv')
val_min = val['timestamp'].min()
val_max = val['timestamp'].max()
print(f'Val timestamp range: {val_min} to {val_max}')
has_target = 'load_mw' in val.columns
print('Val has load_mw column:', has_target)
print()
print('Groups per time step:')
ts_counts = df.groupby('timestamp')['region_id'].nunique()
print(ts_counts.describe())
