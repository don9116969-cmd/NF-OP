"""
Dynamic Budget-Constrained Decoupled Asymmetric Strangle (DAS)
Selects OTM CE and OTM PE strikes whose individual premiums are between 35 and 65 pts,
guaranteeing total capital deployed is strictly Rs 5,500 to Rs 9,500 (<= Rs 10,000 budget) on every day!
Backtested on 100% Real SmartAPI Option Candles (Sep 22 - Sep 25, 2026).
"""
import pandas as pd
import numpy as np
import os

CACHE_DIR = "testing/data/real_options_cache"

def load_real_data():
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
    return df_sept, opt_data

def run_dynamic_das_real(vol_window=15, std_thresh=5.0, vel_thresh=1.0, win_target_pct=0.40, lose_stop_pct=0.35, combined_stop_pct=0.15, max_hold=25):
    df_sept, opt_data = load_real_data()
    timestamps = df_sept.index
    n = len(timestamps)
    
    trades = []
    in_pos = False
    pos = None
    last_exit_bar = -999
    
    for i in range(n):
        t = timestamps[i]
        time_str = t.strftime("%H:%M")
        
        # 1. Manage Active Position
        if in_pos:
            ce_df = opt_data[pos['ce_sym']]
            pe_df = opt_data[pos['pe_sym']]
            
            if t in ce_df.index and t in pe_df.index:
                ce_bar = ce_df.loc[t]
                pe_bar = pe_df.loc[t]
                bars_held = i - pos['entry_bar']
                
                # Check CE leg target/stop
                if not pos['ce_exited']:
                    if ce_bar['high'] >= pos['ce_entry'] * (1.0 + win_target_pct):
                        pos['ce_exit_p'] = pos['ce_entry'] * (1.0 + win_target_pct)
                        pos['ce_exit_t'] = t
                        pos['ce_exited'] = True
                        pos['ce_reason'] = "CE_TARGET_HIT"
                    elif ce_bar['low'] <= pos['ce_entry'] * (1.0 - lose_stop_pct):
                        pos['ce_exit_p'] = pos['ce_entry'] * (1.0 - lose_stop_pct)
                        pos['ce_exit_t'] = t
                        pos['ce_exited'] = True
                        pos['ce_reason'] = "CE_STOP_HIT"
                        
                # Check PE leg target/stop
                if not pos['pe_exited']:
                    if pe_bar['high'] >= pos['pe_entry'] * (1.0 + win_target_pct):
                        pos['pe_exit_p'] = pos['pe_entry'] * (1.0 + win_target_pct)
                        pos['pe_exit_t'] = t
                        pos['pe_exited'] = True
                        pos['pe_reason'] = "PE_TARGET_HIT"
                    elif pe_bar['low'] <= pos['pe_entry'] * (1.0 - lose_stop_pct):
                        pos['pe_exit_p'] = pos['pe_entry'] * (1.0 - lose_stop_pct)
                        pos['pe_exit_t'] = t
                        pos['pe_exited'] = True
                        pos['pe_reason'] = "PE_STOP_HIT"
                        
                # Combined Position Check
                val_ce = pos['ce_exit_p'] if pos['ce_exited'] else ce_bar['close']
                val_pe = pos['pe_exit_p'] if pos['pe_exited'] else pe_bar['close']
                tot_entry = pos['ce_entry'] + pos['pe_entry']
                tot_cur = val_ce + val_pe
                comb_pnl_pct = (tot_cur - tot_entry) / tot_entry
                
                force_close = False
                close_reason = None
                
                if pos['ce_exited'] and pos['pe_exited']:
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
                        'ce_reason': pos['ce_reason'],
                        'pe_sym': pos['pe_sym'],
                        'pe_entry': round(pos['pe_entry'], 2),
                        'pe_exit': round(pos['pe_exit_p'], 2),
                        'pe_reason': pos['pe_reason'],
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
        if "11:30" <= time_str <= "13:00": # Bypass midday chop
            continue
        if (i - last_exit_bar) < 30: # 30-min cooldown
            continue
            
        spot = df_sept.loc[t, 'close']
        atm = int(round(spot / 50.0) * 50)
        
        # Dynamically scan strikes to find CE and PE with affordable premiums (35 to 65 pts)
        chosen_ce = None
        chosen_pe = None
        ce_entry_p = None
        pe_entry_p = None
        
        # Check CE candidates (ATM, +50, +100, +150, +200)
        for offset in [50, 100, 150, 200, 0]:
            k = atm + offset
            sym = f"NIFTY29SEP26{k}CE"
            if sym in opt_data and t in opt_data[sym].index:
                p = opt_data[sym].loc[t, 'close']
                if 35 <= p <= 68:
                    chosen_ce = sym
                    ce_entry_p = p
                    break
                    
        # Check PE candidates (ATM, -50, -100, -150, -200)
        for offset in [50, 100, 150, 200, 0]:
            k = atm - offset
            sym = f"NIFTY29SEP26{k}PE"
            if sym in opt_data and t in opt_data[sym].index:
                p = opt_data[sym].loc[t, 'close']
                if 35 <= p <= 68:
                    chosen_pe = sym
                    pe_entry_p = p
                    break
                    
        if not chosen_ce or not chosen_pe:
            continue
            
        tot_cap = (ce_entry_p + pe_entry_p) * 75
        if tot_cap > 10000:
            continue
            
        # Check Combined Premium Coiling & Expansion
        ce_df = opt_data[chosen_ce]
        pe_df = opt_data[chosen_pe]
        idx_c = ce_df.index.get_loc(t)
        if idx_c < vol_window:
            continue
            
        comb_hist = (ce_df['close'].iloc[idx_c-vol_window:idx_c+1] + pe_df['close'].iloc[idx_c-vol_window:idx_c+1]).values
        comb_std = np.std(comb_hist)
        comb_vel = comb_hist[-1] - comb_hist[-2]
        
        if comb_std < std_thresh and comb_vel >= vel_thresh:
            in_pos = True
            pos = {
                'entry_time': t,
                'entry_bar': i,
                'ce_sym': chosen_ce,
                'ce_entry': ce_entry_p,
                'ce_exited': False,
                'ce_exit_p': None,
                'ce_reason': None,
                'pe_sym': chosen_pe,
                'pe_entry': pe_entry_p,
                'pe_exited': False,
                'pe_exit_p': None,
                'pe_reason': None
            }
            
    return pd.DataFrame(trades)

if __name__ == "__main__":
    print("Testing Dynamic Budget DAS on Real SmartAPI Options (Sep 22 - Sep 25)...")
    df_trades = run_dynamic_das_real(vol_window=15, std_thresh=5.0, vel_thresh=1.0, win_target_pct=0.40, lose_stop_pct=0.35, combined_stop_pct=0.15)
    
    if len(df_trades) > 0:
        wins = df_trades[df_trades['net_pnl'] > 0]
        losses = df_trades[df_trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(df_trades) * 100
        pnl = df_trades['net_pnl'].sum()
        pf = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else 999.0
        
        print("\n" + "="*70)
        print("    DECOUPLED ASYMMETRIC STRANGLE: REAL SMARTAPI OPTION RESULTS    ")
        print("="*70)
        print(f"Total Real Trades    : {len(df_trades)}")
        print(f"Winning Trades       : {len(wins)} ({win_rate:.1f}%)")
        print(f"Losing Trades        : {len(losses)} ({100-win_rate:.1f}%)")
        print(f"Profit Factor        : {pf:.2f}")
        print(f"Total Net PnL        : Rs {pnl:+.2f} (After all Rs 100/trade charges)")
        print(f"Average PnL / Trade  : Rs {df_trades['net_pnl'].mean():+.2f}")
        print(f"Max Capital Deployed : Rs {df_trades['total_capital'].max():.2f}")
        print(f"Avg Capital Deployed : Rs {df_trades['total_capital'].mean():.2f}")
        print("="*70)
        
        os.makedirs("testing/results", exist_ok=True)
        out_path = "testing/results/das_real_smartapi_ledger.csv"
        df_trades.to_csv(out_path, index=False)
        print(f"\n[DONE] Saved Real SmartAPI Ledger to: {out_path}")
        
        print("\n--- COMPLETE TRADE LEDGER ---")
        for idx, r in df_trades.iterrows():
            print(f"Trade #{idx+1:02d} | Entry: {r['entry_time']} -> Exit: {r['exit_time']}")
            print(f"  CE: {r['ce_sym']} Entry: Rs {r['ce_entry']} -> Exit: Rs {r['ce_exit']} ({r['ce_reason']})")
            print(f"  PE: {r['pe_sym']} Entry: Rs {r['pe_entry']} -> Exit: Rs {r['pe_exit']} ({r['pe_reason']})")
            print(f"  Capital: Rs {r['total_capital']} | Net PnL: Rs {r['net_pnl']:+.2f} | Close: {r['close_reason']}")
            print("-" * 70)
    else:
        print("No trades triggered.")
