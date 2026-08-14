import pandas as pd

df = pd.read_csv('processed/stream_edges.csv')
print('total_rows:', len(df))
print('max_time_step:', df['time_step'].max())
print('rows_time_step>=35:', (df['time_step'] >= 35).sum())
print('sample rows with time_step>=35:')
print(df[df['time_step'] >= 35].head().to_string(index=False))
