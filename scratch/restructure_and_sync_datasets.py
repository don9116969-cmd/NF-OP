"""
Restructure and Sync Downloaded Historical Datasets for NIFTY and BankNIFTY
Source:
- C:\\Users\\vishv\\Downloads\\NIFTY 50_minute.csv
- C:\\Users\\vishv\\Downloads\\bank-nifty-1m-data.csv

Target:
- Backup existing data to data/backup_real_aug_oct/
- Save raw copies to data/raw_historical/
- Filter & restructure:
  - NIFTY: 2025-01-01 onwards merged with recent 2026 data
  - BankNIFTY: 2024-11-01 onwards (1.5 years) merged with recent 2026 data
- Save to data/nifty_1min_real.csv and data/banknifty_1min_real.csv
"""

import os
import shutil
import pandas as pd

def restructure():
    nifty_dl = r'C:\Users\vishv\Downloads\NIFTY 50_minute.csv'
    bn_dl = r'C:\Users\vishv\Downloads\bank-nifty-1m-data.csv'

    os.makedirs('data/raw_historical', exist_ok=True)
    os.makedirs('data/backup_real_aug_oct', exist_ok=True)

    # 1. Backup existing files
    if os.path.exists('data/nifty_1min_real.csv'):
        shutil.copy('data/nifty_1min_real.csv', 'data/backup_real_aug_oct/nifty_1min_real.csv')
        print("[BACKUP] Backed up data/nifty_1min_real.csv")
    if os.path.exists('data/banknifty_1min_real.csv'):
        shutil.copy('data/banknifty_1min_real.csv', 'data/backup_real_aug_oct/banknifty_1min_real.csv')
        print("[BACKUP] Backed up data/banknifty_1min_real.csv")

    # 2. Copy raw downloads to raw_historical
    raw_nifty = 'data/raw_historical/nifty_50_minute_2015_2026.csv'
    raw_bn = 'data/raw_historical/bank_nifty_1m_data_2015_2026.csv'
    
    if not os.path.exists(raw_nifty):
        print(f"[COPY] Copying {nifty_dl} -> {raw_nifty}...")
        shutil.copy(nifty_dl, raw_nifty)
    if not os.path.exists(raw_bn):
        print(f"[COPY] Copying {bn_dl} -> {raw_bn}...")
        shutil.copy(bn_dl, raw_bn)

    # 3. Process NIFTY
    print("\n--- Processing NIFTY 50 (2025-01-01 onwards) ---")
    df_n_raw = pd.read_csv(raw_nifty)
    df_n_raw.rename(columns={'date': 'timestamp'}, inplace=True)
    df_n_raw['timestamp'] = pd.to_datetime(df_n_raw['timestamp']).dt.tz_localize(None)
    
    # Filter 2025 onwards
    df_n_filtered = df_n_raw[df_n_raw['timestamp'] >= '2025-01-01'].copy()
    print(f"Downloaded NIFTY 2025+ candles: {len(df_n_filtered):,} ({df_n_filtered['timestamp'].dt.date.nunique()} trading days)")
    
    # Merge with existing real data
    if os.path.exists('data/backup_real_aug_oct/nifty_1min_real.csv'):
        df_n_exist = pd.read_csv('data/backup_real_aug_oct/nifty_1min_real.csv')
        df_n_exist['timestamp'] = pd.to_datetime(df_n_exist['timestamp']).dt.tz_localize(None)
        print(f"Existing live NIFTY candles: {len(df_n_exist):,} ({df_n_exist['timestamp'].dt.date.nunique()} trading days)")
        df_n_combined = pd.concat([df_n_filtered, df_n_exist], ignore_index=True)
    else:
        df_n_combined = df_n_filtered

    df_n_combined.drop_duplicates(subset=['timestamp'], inplace=True)
    df_n_combined.sort_values('timestamp', inplace=True)
    df_n_combined.reset_index(drop=True, inplace=True)
    
    # Standard columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for c in cols:
        if c not in df_n_combined.columns:
            df_n_combined[c] = 0
    df_n_combined = df_n_combined[cols]
    
    df_n_combined.to_csv('data/nifty_1min_real.csv', index=False)
    print(f"[SYNC SUCCESS] NIFTY Total Dataset: {len(df_n_combined):,} candles")
    print(f"  Range: {df_n_combined['timestamp'].min()} to {df_n_combined['timestamp'].max()}")
    print(f"  Unique Trading Days: {df_n_combined['timestamp'].dt.date.nunique()}")

    # 4. Process BankNIFTY
    print("\n--- Processing BankNIFTY (1.5 Years: 2024-11-01 onwards) ---")
    df_bn_raw = pd.read_csv(raw_bn)
    df_bn_raw['timestamp'] = pd.to_datetime(df_bn_raw['Date'] + ' ' + df_bn_raw['Time'], dayfirst=True, errors='coerce')
    df_bn_raw.dropna(subset=['timestamp'], inplace=True)
    df_bn_raw['timestamp'] = df_bn_raw['timestamp'].dt.tz_localize(None)

    df_bn_filtered = df_bn_raw[df_bn_raw['timestamp'] >= '2024-11-01'].copy()
    rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
    df_bn_filtered.rename(columns=rename_map, inplace=True)
    df_bn_filtered['volume'] = 0
    print(f"Downloaded BankNIFTY 1.5yr candles: {len(df_bn_filtered):,} ({df_bn_filtered['timestamp'].dt.date.nunique()} trading days)")

    # Merge with existing real data
    if os.path.exists('data/backup_real_aug_oct/banknifty_1min_real.csv'):
        df_bn_exist = pd.read_csv('data/backup_real_aug_oct/banknifty_1min_real.csv')
        df_bn_exist['timestamp'] = pd.to_datetime(df_bn_exist['timestamp']).dt.tz_localize(None)
        print(f"Existing live BankNIFTY candles: {len(df_bn_exist):,} ({df_bn_exist['timestamp'].dt.date.nunique()} trading days)")
        df_bn_combined = pd.concat([df_bn_filtered, df_bn_exist], ignore_index=True)
    else:
        df_bn_combined = df_bn_filtered

    df_bn_combined.drop_duplicates(subset=['timestamp'], inplace=True)
    df_bn_combined.sort_values('timestamp', inplace=True)
    df_bn_combined.reset_index(drop=True, inplace=True)

    df_bn_combined = df_bn_combined[cols]
    df_bn_combined.to_csv('data/banknifty_1min_real.csv', index=False)
    print(f"[SYNC SUCCESS] BankNIFTY Total Dataset: {len(df_bn_combined):,} candles")
    print(f"  Range: {df_bn_combined['timestamp'].min()} to {df_bn_combined['timestamp'].max()}")
    print(f"  Unique Trading Days: {df_bn_combined['timestamp'].dt.date.nunique()}")

if __name__ == '__main__':
    restructure()
