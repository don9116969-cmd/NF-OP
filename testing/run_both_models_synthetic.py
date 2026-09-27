"""
Full Historical Evaluation of Model 1 (Lee-Mykland Jump-Diffusion) & Model 2 (Decoupled Asymmetric Strangle)
Period: August 10, 2026 to September 25, 2026 (34 Trading Days, 12,321 1-minute real Nifty spot candles).
Option Pricing: Black-Scholes Synthetic Options modeled on Real 1-Minute Nifty Spot.
Capital Constraint: Strictly <= Rs 10,000 per trade, 100% Option Buying (Zero Shorting).
"""
import pandas as pd
import numpy as np
import datetime
import os
import sys

sys.path.append('.')
from src.features.greeks import black_scholes_price

def get_next_weekly_expiry(dt):
    days_ahead = (3 - dt.weekday()) % 7
    if days_ahead == 0 and dt.time() > datetime.time(15, 30):
        days_ahead = 7
    expiry_date = dt.date() + datetime.timedelta(days=days_ahead)
    return datetime.datetime.combine(expiry_date, datetime.time(15, 30))

def get_atm_strike(spot):
    return int(round(spot / 50.0) * 50)

# ==============================================================================
# MODEL 1: LEE-MYKLAND JUMP-DIFFUSION MODEL (LM-BV)
# ==============================================================================
def compute_lee_mykland_jumps(df_spot, K=20):
    log_spots = np.log(df_spot['close'].values)
    returns = np.zeros_like(log_spots)
    returns[1:] = log_spots[1:] - log_spots[:-1]
    
    n = len(returns)
    j_stats = np.zeros(n)
    sigmas = np.full(n, np.nan)
    c_factor = np.pi / 2.0
    
    for i in range(K + 1, n):
        sub_ret = returns[i - K : i]
        abs_r = np.abs(sub_ret)
        bv = c_factor * np.sum(abs_r[1:] * abs_r[:-1]) / (K - 2)
        sigma = np.sqrt(max(1e-8, bv))
        sigmas[i] = sigma
        j_stats[i] = returns[i] / sigma
        
    df_spot['ret'] = returns
    df_spot['bv_sigma'] = sigmas
    df_spot['lm_stat'] = j_stats
    return df_spot

def run_model_1_lee_mykland(df_spot, jump_thresh=3.0, target_pct=0.50, stop_pct=0.20, max_hold_bars=20, iv=0.135):
    timestamps = df_spot.index
    n = len(timestamps)
    trades = []
    in_trade = False
    current_trade = None
    last_exit_bar = -999
    
    dates = df_spot.index.date
    expiry_map = {}
    for d in np.unique(dates):
        dummy_dt = datetime.datetime.combine(d, datetime.time(12, 0))
        expiry_map[d] = get_next_weekly_expiry(dummy_dt)
        
    for i in range(n):
        t = timestamps[i]
        d = t.date()
        time_str = t.strftime("%H:%M")
        
        spot = df_spot.loc[t, 'close']
        spot_high = df_spot.loc[t, 'high']
        spot_low = df_spot.loc[t, 'low']
        
        expiry_dt = expiry_map[d]
        tte_seconds = max(60, (expiry_dt - t).total_seconds())
        T = tte_seconds / (365.0 * 24.0 * 3600.0)
        
        # 1. Manage Active Position
        if in_trade:
            strike = current_trade['strike']
            opt_type = current_trade['opt_type']
            
            cur_p = black_scholes_price(spot, strike, T, 0.07, iv, opt_type)
            if opt_type == "CE":
                high_p = black_scholes_price(spot_high, strike, T, 0.07, iv, opt_type)
                low_p = black_scholes_price(spot_low, strike, T, 0.07, iv, opt_type)
            else:
                high_p = black_scholes_price(spot_low, strike, T, 0.07, iv, opt_type)
                low_p = black_scholes_price(spot_high, strike, T, 0.07, iv, opt_type)
                
            bars_held = i - current_trade['entry_bar']
            target_p = current_trade['entry_price'] * (1.0 + target_pct)
            stop_p = current_trade['entry_price'] * (1.0 - stop_pct)
            
            exit_price = None
            exit_reason = None
            
            if high_p >= target_p:
                exit_price = target_p
                exit_reason = "TARGET_HIT"
            elif low_p <= stop_p:
                exit_price = stop_p
                exit_reason = "STOP_LOSS"
            elif bars_held >= max_hold_bars:
                exit_price = cur_p
                exit_reason = "TIME_DECAY_EXIT"
            elif time_str >= "15:15":
                exit_price = cur_p
                exit_reason = "EOD_SQUAREOFF"
                
            if exit_price is not None:
                gross_pnl = (exit_price - current_trade['entry_price']) * 75
                charges = 50.0 # Rs 50 broker & exchange costs
                net_pnl = gross_pnl - charges
                current_trade.update({
                    'exit_time': t,
                    'exit_price': round(exit_price, 2),
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                    'gross_pnl': round(gross_pnl, 2),
                    'net_pnl': round(net_pnl, 2),
                    'return_pct': round((exit_price / current_trade['entry_price'] - 1) * 100, 2)
                })
                trades.append(current_trade)
                in_trade = False
                current_trade = None
                last_exit_bar = i
                continue
                
        # 2. Check Entry
        if in_trade:
            continue
        if time_str < "09:30" or time_str > "14:45":
            continue
        if "11:30" <= time_str <= "13:00": # Bypass midday chop
            continue
        if (i - last_exit_bar) < 25: # 25-min cooldown
            continue
            
        lm = df_spot.loc[t, 'lm_stat']
        atm = int(round(spot / 50.0) * 50)
        
        signal = None
        if lm >= jump_thresh:
            signal = "BUY_CE"
        elif lm <= -jump_thresh:
            signal = "BUY_PE"
            
        if signal:
            opt_type = "CE" if signal == "BUY_CE" else "PE"
            candidates = [atm, atm + 50, atm + 100] if signal == "BUY_CE" else [atm, atm - 50, atm - 100]
            chosen_strike = None
            entry_p = None
            for s in candidates:
                p = black_scholes_price(spot, s, T, 0.07, iv, opt_type)
                if 40 <= p <= 110: # Affordable premium (Capital Rs 3,000 to Rs 8,250 <= Rs 10,000)
                    chosen_strike = s
                    entry_p = p
                    break
                    
            if chosen_strike and entry_p and (entry_p * 75 <= 10000):
                in_trade = True
                current_trade = {
                    'entry_time': t,
                    'signal': signal,
                    'strike': chosen_strike,
                    'symbol': f"NIFTY_OPT_{chosen_strike}{opt_type}",
                    'opt_type': opt_type,
                    'entry_price': round(entry_p, 2),
                    'capital_used': round(entry_p * 75, 2),
                    'entry_bar': i,
                    'lm_stat': round(lm, 2),
                    'target_pct': target_pct,
                    'stop_pct': stop_pct
                }
                
    return pd.DataFrame(trades)

# ==============================================================================
# MODEL 2: DECOUPLED ASYMMETRIC STRANGLE (DAS)
# ==============================================================================
def run_model_2_decoupled_strangle(df_spot, vol_window=20, std_thresh=4.5, vel_thresh=1.2, win_target_pct=0.50, lose_stop_pct=0.40, combined_stop_pct=0.15, max_hold=25, iv=0.135):
    timestamps = df_spot.index
    n = len(timestamps)
    trades = []
    in_pos = False
    pos = None
    last_exit_bar = -999
    
    dates = df_spot.index.date
    expiry_map = {}
    for d in np.unique(dates):
        dummy_dt = datetime.datetime.combine(d, datetime.time(12, 0))
        expiry_map[d] = get_next_weekly_expiry(dummy_dt)
        
    for i in range(n):
        t = timestamps[i]
        d = t.date()
        time_str = t.strftime("%H:%M")
        
        spot = df_spot.loc[t, 'close']
        spot_high = df_spot.loc[t, 'high']
        spot_low = df_spot.loc[t, 'low']
        
        expiry_dt = expiry_map[d]
        tte_seconds = max(60, (expiry_dt - t).total_seconds())
        T = tte_seconds / (365.0 * 24.0 * 3600.0)
        
        # 1. Manage Active Decoupled Strangle Position
        if in_pos:
            ce_s = pos['ce_strike']
            pe_s = pos['pe_strike']
            
            ce_cur = black_scholes_price(spot, ce_s, T, 0.07, iv, "CE")
            ce_high = black_scholes_price(spot_high, ce_s, T, 0.07, iv, "CE")
            ce_low = black_scholes_price(spot_low, ce_s, T, 0.07, iv, "CE")
            
            pe_cur = black_scholes_price(spot, pe_s, T, 0.07, iv, "PE")
            pe_high = black_scholes_price(spot_low, pe_s, T, 0.07, iv, "PE") # Put is highest at spot low
            pe_low = black_scholes_price(spot_high, pe_s, T, 0.07, iv, "PE")
            
            bars_held = i - pos['entry_bar']
            
            # Check CE Leg Exit
            if not pos['ce_exited']:
                if ce_high >= pos['ce_entry'] * (1.0 + win_target_pct):
                    pos['ce_exit_p'] = pos['ce_entry'] * (1.0 + win_target_pct)
                    pos['ce_exit_t'] = t
                    pos['ce_exited'] = True
                    pos['ce_reason'] = "CE_TARGET"
                elif ce_low <= pos['ce_entry'] * (1.0 - lose_stop_pct):
                    pos['ce_exit_p'] = pos['ce_entry'] * (1.0 - lose_stop_pct)
                    pos['ce_exit_t'] = t
                    pos['ce_exited'] = True
                    pos['ce_reason'] = "CE_STOP"
                    
            # Check PE Leg Exit
            if not pos['pe_exited']:
                if pe_high >= pos['pe_entry'] * (1.0 + win_target_pct):
                    pos['pe_exit_p'] = pos['pe_entry'] * (1.0 + win_target_pct)
                    pos['pe_exit_t'] = t
                    pos['pe_exited'] = True
                    pos['pe_reason'] = "PE_TARGET"
                elif pe_low <= pos['pe_entry'] * (1.0 - lose_stop_pct):
                    pos['pe_exit_p'] = pos['pe_entry'] * (1.0 - lose_stop_pct)
                    pos['pe_exit_t'] = t
                    pos['pe_exited'] = True
                    pos['pe_reason'] = "PE_STOP"
                    
            # Combined PnL evaluation
            val_ce = pos['ce_exit_p'] if pos['ce_exited'] else ce_cur
            val_pe = pos['pe_exit_p'] if pos['pe_exited'] else pe_cur
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
                    pos['ce_exit_p'] = ce_cur
                    pos['ce_exit_t'] = t
                    pos['ce_exited'] = True
                    pos['ce_reason'] = close_reason
                if not pos['pe_exited']:
                    pos['pe_exit_p'] = pe_cur
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
                    'ce_strike': ce_s,
                    'ce_entry': round(pos['ce_entry'], 2),
                    'ce_exit': round(pos['ce_exit_p'], 2),
                    'ce_reason': pos['ce_reason'],
                    'pe_strike': pe_s,
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
            
        atm = get_atm_strike(spot)
        ce_s = atm + 50
        pe_s = atm - 50
        
        c_p = black_scholes_price(spot, ce_s, T, 0.07, iv, "CE")
        p_p = black_scholes_price(spot, pe_s, T, 0.07, iv, "PE")
        
        if (c_p + p_p) * 75 > 10000:
            # Shift further OTM to fit Rs 10,000 budget
            ce_s = atm + 100
            pe_s = atm - 100
            c_p = black_scholes_price(spot, ce_s, T, 0.07, iv, "CE")
            p_p = black_scholes_price(spot, pe_s, T, 0.07, iv, "PE")
            if (c_p + p_p) * 75 > 10000:
                continue
                
        # Check combined premium coiling & expansion over rolling 20 bars
        if i < vol_window:
            continue
            
        recent_spots = df_spot['close'].iloc[i-vol_window:i+1].values
        # Theoretical combined prices over recent spots
        comb_prices = []
        for s_val in recent_spots:
            cp = black_scholes_price(s_val, ce_s, T, 0.07, iv, "CE")
            pp = black_scholes_price(s_val, pe_s, T, 0.07, iv, "PE")
            comb_prices.append(cp + pp)
            
        comb_prices = np.array(comb_prices)
        comb_std = np.std(comb_prices)
        comb_vel = comb_prices[-1] - comb_prices[-2]
        
        # Trigger when premium variance is compressed and velocity expands
        if comb_std < std_thresh and comb_vel >= vel_thresh:
            in_pos = True
            pos = {
                'entry_time': t,
                'entry_bar': i,
                'ce_strike': ce_s,
                'ce_entry': c_p,
                'ce_exited': False,
                'ce_exit_p': None,
                'ce_reason': None,
                'pe_strike': pe_s,
                'pe_entry': p_p,
                'pe_exited': False,
                'pe_exit_p': None,
                'pe_reason': None
            }
            
    return pd.DataFrame(trades)

if __name__ == "__main__":
    df_spot = pd.read_csv("data/nifty_1min_real.csv")
    df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'])
    df_spot.set_index('timestamp', inplace=True)
    df_spot = compute_lee_mykland_jumps(df_spot, K=20)
    
    print("="*75)
    print("  EVALUATING MODEL 1 & MODEL 2 ON FULL HISTORICAL DATASET (AUG 10 - SEP 25)  ")
    print("="*75)
    
    # 1. RUN MODEL 1: LEE-MYKLAND
    print("\n>>> Running Model 1: Lee-Mykland Jump-Diffusion (J_thresh=3.2, Tgt=50%, Stop=20%)...")
    df_m1 = run_model_1_lee_mykland(df_spot, jump_thresh=3.2, target_pct=0.50, stop_pct=0.20, max_hold_bars=20)
    
    if len(df_m1) > 0:
        w1 = df_m1[df_m1['net_pnl'] > 0]
        l1 = df_m1[df_m1['net_pnl'] <= 0]
        pnl1 = df_m1['net_pnl'].sum()
        pf1 = (w1['net_pnl'].sum() / abs(l1['net_pnl'].sum())) if len(l1) > 0 and l1['net_pnl'].sum() != 0 else 999.0
        
        df_m1['cum_pnl'] = df_m1['net_pnl'].cumsum()
        df_m1['peak'] = df_m1['cum_pnl'].cummax()
        df_m1['dd'] = df_m1['cum_pnl'] - df_m1['peak']
        max_dd1 = df_m1['dd'].min()
        
        print(f"MODEL 1 RESULTS (34 Days):")
        print(f"Total Trades Taken   : {len(df_m1)} ({len(df_m1)/34:.2f} trades/day)")
        print(f"Winning Trades       : {len(w1)} ({len(w1)/len(df_m1)*100:.1f}%)")
        print(f"Losing Trades        : {len(l1)} ({len(l1)/len(df_m1)*100:.1f}%)")
        print(f"Profit Factor        : {pf1:.2f}")
        print(f"Total Net PnL        : Rs {pnl1:+.2f}")
        print(f"Max Drawdown         : Rs {max_dd1:.2f}")
        print(f"Average PnL / Trade  : Rs {df_m1['net_pnl'].mean():+.2f}")
        print(f"Max Capital Deployed : Rs {df_m1['capital_used'].max():.2f}")
        
    # 2. RUN MODEL 2: DECOUPLED ASYMMETRIC STRANGLE (DAS)
    print("\n>>> Running Model 2: Decoupled Asymmetric Strangle (Std=4.5, Vel=1.2, Tgt=50%, Stop=40%)...")
    df_m2 = run_model_2_decoupled_strangle(df_spot, vol_window=20, std_thresh=4.5, vel_thresh=1.2, win_target_pct=0.50, lose_stop_pct=0.40, combined_stop_pct=0.15, max_hold=25)
    
    if len(df_m2) > 0:
        w2 = df_m2[df_m2['net_pnl'] > 0]
        l2 = df_m2[df_m2['net_pnl'] <= 0]
        pnl2 = df_m2['net_pnl'].sum()
        pf2 = (w2['net_pnl'].sum() / abs(l2['net_pnl'].sum())) if len(l2) > 0 and l2['net_pnl'].sum() != 0 else 999.0
        
        df_m2['cum_pnl'] = df_m2['net_pnl'].cumsum()
        df_m2['peak'] = df_m2['cum_pnl'].cummax()
        df_m2['dd'] = df_m2['cum_pnl'] - df_m2['peak']
        max_dd2 = df_m2['dd'].min()
        
        print(f"MODEL 2 RESULTS (34 Days):")
        print(f"Total Trades Taken   : {len(df_m2)} ({len(df_m2)/34:.2f} trades/day)")
        print(f"Winning Trades       : {len(w2)} ({len(w2)/len(df_m2)*100:.1f}%)")
        print(f"Losing Trades        : {len(l2)} ({len(l2)/len(df_m2)*100:.1f}%)")
        print(f"Profit Factor        : {pf2:.2f}")
        print(f"Total Net PnL        : Rs {pnl2:+.2f}")
        print(f"Max Drawdown         : Rs {max_dd2:.2f}")
        print(f"Average PnL / Trade  : Rs {df_m2['net_pnl'].mean():+.2f}")
        print(f"Max Capital Deployed : Rs {df_m2['total_capital'].max():.2f}")
        
    # Save ledgers
    os.makedirs("testing/results", exist_ok=True)
    df_m1.to_csv("testing/results/model1_lee_mykland_synthetic_full_ledger.csv", index=False)
    df_m2.to_csv("testing/results/model2_decoupled_strangle_synthetic_full_ledger.csv", index=False)
    print("\n[DONE] Saved ledgers to testing/results/!")
