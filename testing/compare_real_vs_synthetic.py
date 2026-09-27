"""
Empirical Comparison: Real SmartAPI Traded Option Prices vs Synthetic Black-Scholes Prices
Period: Sep 22 to Sep 25, 2026 (1,500+ 1-minute candles).
Compares:
1. Price correlation
2. Return variance (does 10% move in real equal 9-11% in synthetic?)
3. Tracking error and IV behavior
"""
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.path.append('.')
from src.features.greeks import black_scholes_price

CACHE_DIR = "testing/data/real_options_cache"

def load_data():
    df_spot = pd.read_csv("data/nifty_1min_real.csv")
    df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'])
    df_spot.set_index('timestamp', inplace=True)
    
    # Load sample real option files: ATM strikes 23300CE, 23300PE, 23200CE, 23400PE
    real_opts = {}
    for sym in ['NIFTY29SEP2623300CE', 'NIFTY29SEP2623300PE', 'NIFTY29SEP2623200CE', 'NIFTY29SEP2623400PE']:
        p = f"{CACHE_DIR}/{sym}.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            real_opts[sym] = df
    return df_spot, real_opts

def compare_real_vs_synthetic():
    df_spot, real_opts = load_data()
    expiry_dt = pd.to_datetime("2026-09-29 15:30:00")
    
    results = []
    
    for sym, opt_df in real_opts.items():
        strike = int(sym.replace("NIFTY29SEP26", "").replace("CE", "").replace("PE", ""))
        opt_type = "CE" if "CE" in sym else "PE"
        
        # Align timestamps
        common_times = opt_df.index.intersection(df_spot.index)
        
        real_prices = []
        synth_prices = []
        
        for t in common_times:
            spot = df_spot.loc[t, 'close']
            real_p = opt_df.loc[t, 'close']
            
            # Time to expiry in calendar years
            tte_days = max(1e-4, (expiry_dt - t).total_seconds() / (24 * 3600))
            T = tte_days / 365.0
            
            # Using India VIX approx ~13.5% (0.135)
            synth_p = black_scholes_price(spot, strike, T, 0.07, 0.135, opt_type)
            
            real_prices.append(real_p)
            synth_prices.append(synth_p)
            
        df_comp = pd.DataFrame({
            'real': real_prices,
            'synth': synth_prices
        }, index=common_times)
        
        # Returns correlation (15-minute percentage return)
        df_comp['real_ret15'] = df_comp['real'].pct_change(15) * 100
        df_comp['synth_ret15'] = df_comp['synth'].pct_change(15) * 100
        df_comp.dropna(inplace=True)
        
        corr_price = df_comp['real'].corr(df_comp['synth'])
        corr_ret = df_comp['real_ret15'].corr(df_comp['synth_ret15'])
        
        # Check ratio of returns: synth_ret / real_ret for significant moves (> 5%)
        sig_moves = df_comp[abs(df_comp['real_ret15']) >= 5.0]
        if len(sig_moves) > 0:
            ratio = (sig_moves['synth_ret15'] / sig_moves['real_ret15']).median()
            mae = abs(sig_moves['synth_ret15'] - sig_moves['real_ret15']).mean()
        else:
            ratio = 1.0
            mae = 0.0
            
        results.append({
            'contract': sym,
            'price_corr': round(corr_price, 4),
            'return_corr': round(corr_ret, 4),
            'return_scale_ratio': round(ratio, 2),
            'avg_abs_return_diff_pct': round(mae, 2)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df_res = compare_real_vs_synthetic()
    print("=== EMPIRICAL COMPARISON: REAL VS SYNTHETIC BLACK-SCHOLES ===")
    print(df_res.to_string(index=False))
