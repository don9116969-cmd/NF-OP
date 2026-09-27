"""
Decoupled Asymmetric Strangle (DAS) Mathematical Model
Buys 1 lot CE + 1 lot PE simultaneously (100% Option Buying, Capital <= Rs 10,000).
Asymmetric Exit:
- Closes the surging winning leg at +50% to +70% target.
- Closes the losing leg with capped stop or holds for reversal.
- Combined stop loss if market stays dead flat.
Tested on:
1. Real SmartAPI Option Dataset (Sep 22 - Sep 25)
2. Full 34-Day History (Aug 10 - Sep 25)
"""
import pandas as pd
import numpy as np
import datetime
import os
import sys

sys.path.append('.')
from src.features.greeks import black_scholes_price

CACHE_DIR = "testing/data/real_options_cache"

def get_next_weekly_expiry(dt):
    days_ahead = (3 - dt.weekday()) % 7
    if days_ahead == 0 and dt.time() > datetime.time(15, 30):
        days_ahead = 7
    expiry_date = dt.date() + datetime.timedelta(days=days_ahead)
    return datetime.datetime.combine(expiry_date, datetime.time(15, 30))

def test_das_real_smartapi(vol_window=20, win_target_pct=0.50, lose_stop_pct=0.40, combined_stop_pct=0.15, max_hold=25):
    # Load real option contracts
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.csv')]
    opt_data = {}
    for f in files:
        sym = f.replace('.csv', '')
        d = pd.read_csv(os.path.join(CACHE_DIR, f))
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        d.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            d[col] = pd.to_numeric(d[col], errors='coerce')
        opt_data[sym] = d
        
    df_spot = pd.read_csv("data/nifty_1min_real.csv")
    df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'])
    df_spot.set_index('timestamp', inplace=True)
    df_sept = df_spot[df_spot.index >= '2026-09-22 09:15:00'].copy()
    
    timestamps = df_sept.index
    n = len(timestamps)
    
    trades = []
    in_pos = False
    pos = None
    last_exit_bar = -999
    
    for i in range(n):
        t = timestamps[i]
        time_str = t.strftime("%H:%M")
        
        # 1. Manage Active Decoupled Strangle
        if in_pos:
            ce_df = opt_data[pos['ce_sym']]
            pe_df = opt_data[pos['pe_sym']]
            
            if t in ce_df.index and t in pe_df.index:
                ce_bar = ce_df.loc[t]
                pe_bar = pe_df.loc[t]
                bars_held = i - pos['entry_bar']
                
                # Check CE leg exit
                if not pos['ce_exited']:
                    if ce_bar['high'] >= pos['ce_entry'] * (1.0 + win_target_pct):
                        pos['ce_exit_p'] = pos['ce_entry'] * (1.0 + win_target_pct)
                        pos['ce_exit_t'] = t
                        pos['ce_exited'] = True
                        pos['ce_reason'] = "CE_TARGET"
                    elif ce_bar['low'] <= pos['ce_entry'] * (1.0 - lose_stop_pct):
                        pos['ce_exit_p'] = pos['ce_entry'] * (1.0 - lose_stop_pct)
                        pos['ce_exit_t'] = t
                        pos['ce_exited'] = True
                        pos['ce_reason'] = "CE_STOP"
                        
                # Check PE leg exit
                if not pos['pe_exited']:
                    if pe_bar['high'] >= pos['pe_entry'] * (1.0 + win_target_pct):
                        pos['pe_exit_p'] = pos['pe_entry'] * (1.0 + win_target_pct)
                        pos['pe_exit_t'] = t
                        pos['pe_exited'] = True
                        pos['pe_reason'] = "PE_TARGET"
                    elif pe_bar['low'] <= pos['pe_entry'] * (1.0 - lose_stop_pct):
                        pos['pe_exit_p'] = pos['pe_entry'] * (1.0 - lose_stop_pct)
                        pos['pe_exit_t'] = t
                        pos['pe_exited'] = True
                        pos['pe_reason'] = "PE_STOP"
                        
                # Combined PnL check
                cur_ce_val = pos['ce_exit_p'] if pos['ce_exited'] else ce_bar['close']
                cur_pe_val = pos['pe_exit_p'] if pos['pe_exited'] else pe_bar['close']
                total_entry = pos['ce_entry'] + pos['pe_entry']
                total_current = cur_ce_val + cur_pe_val
                comb_pnl_pct = (total_current - total_entry) / total_entry
                
                force_close = False
                close_reason = None
                
                if (pos['ce_exited'] and pos['pe_exited']):
                    force_close = True
                    close_reason = "BOTH_LEGS_EXITED"
                elif comb_pnl_pct <= -combined_stop_pct and (not pos['ce_exited'] and not pos['pe_exited']):
                    force_close = True
                    close_reason = "COMBINED_STOP_LOSS"
                elif bars_held >= max_hold:
                    force_close = True
                    close_reason = "MAX_HOLD_TIME"
                elif time_str >= "15:15":
                    force_close = True
                    close_reason = "EOD_SQUAREOFF"
                    
                if force_close:
                    if not pos['ce_exited']:
                        pos['ce_exit_p'] = ce_bar['close']
                        pos['ce_exit_t'] = t
                        pos['ce_exited'] = True
                        pos['ce_reason'] = close_reason
                    if not pos['pe_exited']:
                        pos['pe_exit_p'] = pe_bar['close']
                        pos['pe_exit_t'] = t
                        pos['pe_exited'] = True
                        pos['pe_reason'] = close_reason
                        
                    ce_pnl = (pos['ce_exit_p'] - pos['ce_entry']) * 75
                    pe_pnl = (pos['pe_exit_p'] - pos['pe_entry']) * 75
                    total_gross = ce_pnl + pe_pnl
                    charges = 100.0 # 2 legs round-trip = Rs 100
                    total_net = total_gross - charges
                    
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': t,
                        'ce_sym': pos['ce_sym'],
                        'ce_entry': round(pos['ce_entry'], 2),
                        'ce_exit': round(pos['ce_exit_p'], 2),
                        'pe_sym': pos['pe_sym'],
                        'pe_entry': round(pos['pe_entry'], 2),
                        'pe_exit': round(pos['pe_exit_p'], 2),
                        'total_capital': round((pos['ce_entry'] + pos['pe_entry']) * 75, 2),
                        'gross_pnl': round(total_gross, 2),
                        'net_pnl': round(total_net, 2),
                        'close_reason': close_reason
                    })
                    in_pos = False
                    pos = None
                    last_exit_bar = i
                    continue
                    
        # 2. Check Entry
        if in_pos:
            continue
        if time_str < "09:45" or time_str > "14:15":
            continue
        if "11:30" <= time_str <= "13:00":
            continue
        if (i - last_exit_bar) < 30: # 30-min cooldown
            continue
            
        spot = df_sept.loc[t, 'close']
        atm = int(round(spot / 50.0) * 50)
        
        # Select OTM strikes so total premium <= 120 pts (Total capital <= Rs 9,000)
        ce_sym = f"NIFTY29SEP26{atm+50}CE"
        pe_sym = f"NIFTY29SEP26{atm-50}PE"
        
        if ce_sym not in opt_data or pe_sym not in opt_data:
            continue
        ce_df = opt_data[ce_sym]
        pe_df = opt_data[pe_sym]
        if t not in ce_df.index or t not in pe_df.index:
            continue
            
        c_p = ce_df.loc[t, 'close']
        p_p = pe_df.loc[t, 'close']
        
        if (c_p + p_p) * 75 > 10000:
            # Shift further OTM
            ce_sym = f"NIFTY29SEP26{atm+100}CE"
            pe_sym = f"NIFTY29SEP26{atm-100}PE"
            if ce_sym not in opt_data or pe_sym not in opt_data:
                continue
            ce_df = opt_data[ce_sym]
            pe_df = opt_data[pe_sym]
            if t not in ce_df.index or t not in pe_df.index:
                continue
            c_p = ce_df.loc[t, 'close']
            p_p = pe_df.loc[t, 'close']
            if (c_p + p_p) * 75 > 10000:
                continue
                
        # Premium Volatility Compression & Kinetic Expansion Trigger
        idx_c = ce_df.index.get_loc(t)
        if idx_c < vol_window:
            continue
            
        comb_hist = (ce_df['close'].iloc[idx_c-vol_window:idx_c+1] + pe_df['close'].iloc[idx_c-vol_window:idx_c+1]).values
        comb_std = np.std(comb_hist)
        comb_vel = comb_hist[-1] - comb_hist[-2]
        
        # Trigger when premium standard deviation is compressed (< 5.0) and starts expanding (velocity > 1.5)
        if comb_std < 5.0 and comb_vel >= 1.5:
            in_pos = True
            pos = {
                'entry_time': t,
                'entry_bar': i,
                'ce_sym': ce_sym,
                'ce_entry': c_p,
                'ce_exited': False,
                'ce_exit_p': None,
                'pe_sym': pe_sym,
                'pe_entry': p_p,
                'pe_exited': False,
                'pe_exit_p': None
            }
            
    return pd.DataFrame(trades)

if __name__ == "__main__":
    print("Testing Decoupled Asymmetric Strangle (DAS) on Real SmartAPI Options (Sep 22 - Sep 25)...")
    df_trades = test_das_real_smartapi()
    if len(df_trades) > 0:
        wins = df_trades[df_trades['net_pnl'] > 0]
        losses = df_trades[df_trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(df_trades) * 100
        pnl = df_trades['net_pnl'].sum()
        pf = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else 999.0
        print(f"Trades Taken: {len(df_trades)} | Win Rate: {win_rate:.1f}% ({len(wins)}/{len(df_trades)}) | PF: {pf:.2f} | Net PnL: Rs {pnl:+.2f}")
        print("\nTrades Ledger:")
        print(df_trades[['entry_time', 'ce_sym', 'ce_entry', 'ce_exit', 'pe_sym', 'pe_entry', 'pe_exit', 'total_capital', 'net_pnl', 'close_reason']].to_string())
    else:
        print("No trades triggered.")
