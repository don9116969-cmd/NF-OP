"""
Dual-Session Kinetic Shock (DSKS Mode 2) Backtest across Full Historical Dataset
Period: August 10, 2026 to September 25, 2026 (34 Trading Days, 12,321 1-minute candles).
Option Pricing: Black-Scholes Synthetic Options modeled on Real 1-Minute Nifty Spot.
Constraint: 100% Option Buying, Capital <= Rs 10,000 per trade.
"""
import pandas as pd
import numpy as np
import datetime
import os
import sys

sys.path.append('.')
from src.features.greeks import black_scholes_price

def get_next_weekly_expiry(dt):
    # Nifty weekly options expire on Thursday
    # weekday(): Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4
    days_ahead = (3 - dt.weekday()) % 7
    if days_ahead == 0 and dt.time() > datetime.time(15, 30):
        days_ahead = 7
    expiry_date = dt.date() + datetime.timedelta(days=days_ahead)
    expiry_dt = datetime.datetime.combine(expiry_date, datetime.time(15, 30))
    return expiry_dt

def get_atm_strike(spot):
    return int(round(spot / 50.0) * 50)

def run_dsks_synthetic_full(window=20, z_thresh=1.8, delta_z_thresh=0.8, target_pct=0.30, stop_pct=0.15, max_hold_bars=20, iv=0.135):
    df_spot = pd.read_csv("data/nifty_1min_real.csv")
    df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'])
    df_spot.set_index('timestamp', inplace=True)
    df_spot['ema20'] = df_spot['close'].ewm(span=20).mean()
    
    timestamps = df_spot.index
    n = len(timestamps)
    
    trades = []
    in_trade = False
    current_trade = None
    r_history = []
    session_trades = {}
    
    # Pre-calculate active expiry for each day to be fast
    dates = df_spot.index.date
    expiry_map = {}
    for d in np.unique(dates):
        dummy_dt = datetime.datetime.combine(d, datetime.time(12, 0))
        expiry_map[d] = get_next_weekly_expiry(dummy_dt)
        
    for i in range(n):
        t = timestamps[i]
        d = t.date()
        date_str = t.strftime("%Y-%m-%d")
        time_str = t.strftime("%H:%M")
        
        current_session = None
        if "09:45" <= time_str <= "11:15":
            current_session = f"{date_str}_MORNING"
        elif "13:15" <= time_str <= "14:30":
            current_session = f"{date_str}_AFTERNOON"
            
        spot = df_spot.loc[t, 'close']
        spot_high = df_spot.loc[t, 'high']
        spot_low = df_spot.loc[t, 'low']
        spot_ema = df_spot.loc[t, 'ema20']
        
        expiry_dt = expiry_map[d]
        tte_seconds = max(60, (expiry_dt - t).total_seconds())
        T = tte_seconds / (365.0 * 24.0 * 3600.0)
        
        # 1. Manage Ongoing Trade
        if in_trade:
            strike = current_trade['strike']
            opt_type = current_trade['opt_type']
            
            # Theoretical option price at close, high, low of the spot bar
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
                charges = 50.0 # Rs 50 round-trip brokerage & taxes
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
                continue
                
        # 2. Check Session Eligibility
        if in_trade or current_session is None:
            continue
        if session_trades.get(current_session, 0) >= 1:
            continue
            
        # 3. Calculate ATM Log-Parity Ratio
        atm = get_atm_strike(spot)
        c_atm = black_scholes_price(spot, atm, T, 0.07, iv, "CE")
        p_atm = black_scholes_price(spot, atm, T, 0.07, iv, "PE")
        
        if c_atm <= 0 or p_atm <= 0:
            continue
            
        r_val = np.log(c_atm / p_atm)
        r_history.append((t, r_val, spot))
        
        if len(r_history) < window:
            continue
            
        recent = [x[1] for x in r_history[-window:]]
        mean_r = np.mean(recent)
        std_r = np.std(recent)
        if std_r < 1e-4:
            continue
            
        z = (r_val - mean_r) / std_r
        prev_z = (recent[-4] - np.mean(recent[:-3])) / (np.std(recent[:-3]) + 1e-4) if len(recent) >= 5 else 0
        delta_z = z - prev_z
        
        spot_trend_ce = (spot >= spot_ema)
        spot_trend_pe = (spot <= spot_ema)
        
        # Check spot acceleration as proxy for option convexity
        recent_spots = [x[2] for x in r_history[-3:]]
        spot_accel = recent_spots[2] - 2 * recent_spots[1] + recent_spots[0]
        
        signal = None
        if z >= z_thresh and delta_z >= delta_z_thresh and spot_trend_ce and spot_accel >= 0:
            signal = "BUY_CE"
        elif z <= -z_thresh and delta_z <= -delta_z_thresh and spot_trend_pe and spot_accel <= 0:
            signal = "BUY_PE"
            
        if signal:
            opt_type = "CE" if signal == "BUY_CE" else "PE"
            candidates = [atm, atm + 50, atm + 100, atm - 50] if signal == "BUY_CE" else [atm, atm - 50, atm - 100, atm + 50]
            chosen_strike = None
            entry_p = None
            for s in candidates:
                p = black_scholes_price(spot, s, T, 0.07, iv, opt_type)
                if 40 <= p <= 125: # Premium between 40 and 125 pts (Rs 3,000 to Rs 9,375 <= Rs 10,000)
                    chosen_strike = s
                    entry_p = p
                    break
                    
            if chosen_strike and entry_p and (entry_p * 75 <= 10000):
                in_trade = True
                current_trade = {
                    'entry_time': t,
                    'session': current_session,
                    'signal': signal,
                    'strike': chosen_strike,
                    'symbol': f"NIFTY_OPT_{chosen_strike}{opt_type}",
                    'opt_type': opt_type,
                    'entry_price': round(entry_p, 2),
                    'capital_used': round(entry_p * 75, 2),
                    'entry_bar': i,
                    'z_score': round(z, 2),
                    'delta_z': round(delta_z, 2),
                    'target_pct': target_pct,
                    'stop_pct': stop_pct
                }
                session_trades[current_session] = session_trades.get(current_session, 0) + 1
                
    return pd.DataFrame(trades)

if __name__ == "__main__":
    print("==========================================================================")
    print("RUNNING DSKS MODE 2 (HIGH-ALPHA) ON FULL HISTORICAL DATASET (AUG - SEP)")
    print("==========================================================================")
    df_trades = run_dsks_synthetic_full()
    
    if len(df_trades) > 0:
        wins = df_trades[df_trades['net_pnl'] > 0]
        losses = df_trades[df_trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(df_trades) * 100
        total_net = df_trades['net_pnl'].sum()
        gross_pnl = df_trades['gross_pnl'].sum()
        charges = len(df_trades) * 50.0
        pf = (wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else 999.0
        
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_time']).dt.date
        trading_days = df_trades['entry_date'].nunique()
        total_dataset_days = 34
        
        print(f"Total Dataset Days     : {total_dataset_days} Days (Aug 10, 2026 to Sep 25, 2026)")
        print(f"Total Trades Taken     : {len(df_trades)} (Avg {len(df_trades)/total_dataset_days:.2f} trades/day)")
        print(f"Winning Trades         : {len(wins)} ({win_rate:.1f}%)")
        print(f"Losing Trades          : {len(losses)} ({100-win_rate:.1f}%)")
        print(f"Profit Factor          : {pf:.2f}")
        print(f"Gross PnL              : Rs {gross_pnl:.2f}")
        print(f"Total Brokerage/Taxes  : Rs {charges:.2f} (Rs 50/trade round-trip)")
        print(f"Total Net PnL          : Rs {total_net:.2f} (Net after all charges)")
        print(f"Average PnL / Trade    : Rs {df_trades['net_pnl'].mean():.2f}")
        print(f"Max Capital Deployed   : Rs {df_trades['capital_used'].max():.2f} (Max Budget Rs 10,000)")
        print(f"Avg Capital Deployed   : Rs {df_trades['capital_used'].mean():.2f}")
        print("==========================================================================")
        
        # Save ledger
        os.makedirs("testing/results", exist_ok=True)
        out_path = "testing/results/dsks_synthetic_full_aug_sep_ledger.csv"
        df_trades.to_csv(out_path, index=False)
        print(f"\n[DONE] Full 34-day ledger saved to: {out_path}")
        
        # Weekly breakdown
        df_trades['week'] = pd.to_datetime(df_trades['entry_time']).dt.isocalendar().week
        weekly = df_trades.groupby('week').agg(
            trades=('net_pnl', 'count'),
            wins=('net_pnl', lambda x: (x > 0).sum()),
            net_pnl=('net_pnl', 'sum'),
            avg_cap=('capital_used', 'mean')
        )
        weekly['win_rate'] = (weekly['wins'] / weekly['trades'] * 100).round(1)
        print("\n--- WEEK-BY-WEEK PERFORMANCE BREAKDOWN ---")
        print(weekly.to_string())
    else:
        print("No trades triggered.")
