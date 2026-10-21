import os
import pandas as pd

nifty_path = r'C:\Users\vishv\Downloads\NIFTY 50_minute.csv'
bn_path = r'C:\Users\vishv\Downloads\bank-nifty-1m-data.csv'

print('Processing NIFTY...')
df_n = pd.read_csv(nifty_path)
df_n['date'] = pd.to_datetime(df_n['date'])
n_min_date = df_n['date'].min()
n_max_date = df_n['date'].max()
print(f'NIFTY Total Rows: {len(df_n):,}')
print(f'NIFTY Range: {n_min_date} to {n_max_date}')
df_n_2025 = df_n[df_n['date'] >= '2025-01-01']
print(f"NIFTY 2025+ Rows: {len(df_n_2025):,} ({df_n_2025['date'].dt.date.nunique()} trading days)")

print('\nProcessing BANKNIFTY...')
df_bn = pd.read_csv(bn_path)
print('BankNifty columns:', list(df_bn.columns))
df_bn['timestamp'] = pd.to_datetime(df_bn['Date'] + ' ' + df_bn['Time'], dayfirst=True, errors='coerce')
bn_min_date = df_bn['timestamp'].min()
bn_max_date = df_bn['timestamp'].max()
print(f'BANKNIFTY Total Rows: {len(df_bn):,}')
print(f'BANKNIFTY Range: {bn_min_date} to {bn_max_date}')
df_bn_1_5yr = df_bn[df_bn['timestamp'] >= '2024-11-01']
print(f"BANKNIFTY 1.5yr (from Nov 2024) Rows: {len(df_bn_1_5yr):,} ({df_bn_1_5yr['timestamp'].dt.date.nunique()} trading days)")
df_bn_2025 = df_bn[df_bn['timestamp'] >= '2025-01-01']
print(f"BANKNIFTY 2025+ Rows: {len(df_bn_2025):,} ({df_bn_2025['timestamp'].dt.date.nunique()} trading days)")
