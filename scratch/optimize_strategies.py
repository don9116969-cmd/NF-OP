"""
Quantitative Strategy Optimization & Fine-Tuning Grid Explorer.
Explores:
1. BankNifty (Strategy 2) tuning: Box filter, breakout buffer, time cutoffs, trailing/breakeven stops.
2. Nifty (Strategy 1) tuning: Box width, buffer, trailing stop.
3. DAS Strangle (Strategy 3) tuning: Velocity threshold, std compression threshold.
4. Capital & Trade Circuit Breaker: Max 1 loss/day filter.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_full_portfolio_backtest import (
    NIFTY_CSV, BANKNIFTY_CSV, bs_vectorized, calc_charges, get_dte, compute_stats
)

def load_data():
    df_n = pd.read_csv(NIFTY_CSV)
    df_n['timestamp'] = pd.to_datetime(df_n['timestamp']).dt.tz_localize(None)
    df_n = df_n.sort_values('timestamp').reset_index(drop=True)
    df_n['date'] = df_n['timestamp'].dt.date
    df_n['time'] = df_n['timestamp'].dt.time
    df_n['ema_20'] = df_n['close'].ewm(span=20, adjust=False).mean()
    n_days = {d: group.copy().reset_index(drop=True) for d, group in df_n.groupby('date')}

    df_b = pd.read_csv(BANKNIFTY_CSV)
    df_b['timestamp'] = pd.to_datetime(df_b['timestamp']).dt.tz_localize(None)
    df_b = df_b.sort_values('timestamp').reset_index(drop=True)
    df_b['date'] = df_b['timestamp'].dt.date
    df_b['time'] = df_b['timestamp'].dt.time
    df_b['ema_20'] = df_b['close'].ewm(span=20, adjust=False).mean()
    b_days = {d: group.copy().reset_index(drop=True) for d, group in df_b.groupby('date')}

    return n_days, b_days

def test_banknifty_optimizations(b_days):
    print("=" * 100)
    print("           STRATEGY 2 (BANKNIFTY): SYSTEMATIC PARAMETER OPTIMIZATION")
    print("=" * 100)
    all_dates = sorted(list(b_days.keys()))
    results = []

    # Grid search
    for max_box in [0.50, 0.70, 0.90, 1.20]:
        for buf in [25.0, 40.0, 50.0]:
            for cutoff_hour in [11, 12, 13, 15]:
                for be_trigger in [None, 0.35, 0.50]: # Breakeven trigger: move SL to entry when +35% / +50% reached
                    trades = []
                    for d in all_dates:
                        day_df = b_days[d]
                        if len(day_df) < 20: continue
                        times = day_df['time'].values
                        t_0915 = datetime.time(9, 15)
                        t_0930 = datetime.time(9, 30)
                        t_cutoff = datetime.time(cutoff_hour, 30 if cutoff_hour == 11 else 0)
                        t_1515 = datetime.time(15, 15)

                        box_mask = (times >= t_0915) & (times <= t_0930)
                        if np.sum(box_mask) < 10: continue
                        box_high = float(day_df.loc[box_mask, 'high'].max())
                        box_low = float(day_df.loc[box_mask, 'low'].min())
                        open_p = float(day_df.loc[box_mask, 'close'].iloc[0])
                        box_range = box_high - box_low
                        box_pct = (box_range / open_p) * 100.0
                        if box_pct > max_box: continue

                        u_trig = box_high + buf
                        l_trig = box_low - buf

                        post_mask = (times > t_0930) & (times <= t_cutoff)
                        post_indices = np.where(post_mask)[0]
                        if len(post_indices) == 0: continue

                        entry_idx = None
                        opt_type = None
                        for idx in post_indices:
                            c = float(day_df.at[idx, 'close'])
                            ema = float(day_df.at[idx, 'ema_20'])
                            if c >= u_trig and c > ema:
                                entry_idx = idx
                                opt_type = 'CE'
                                break
                            elif c <= l_trig and c < ema:
                                entry_idx = idx
                                opt_type = 'PE'
                                break
                        if entry_idx is None: continue

                        entry_spot = float(day_df.at[entry_idx, 'close'])
                        atm = int(round(entry_spot / 100.0) * 100)
                        dte = get_dte(d, 'BANKNIFTY')

                        rem_df = day_df.iloc[entry_idx:].copy().reset_index(drop=True)
                        closes = rem_df['close'].values
                        highs = rem_df['high'].values
                        lows = rem_df['low'].values
                        sub_times = rem_df['time'].values

                        chosen_strike = None
                        entry_p = None
                        for off in range(100, 3600, 100):
                            k = atm + off if opt_type == 'CE' else atm - off
                            p0 = float(bs_vectorized(np.array([closes[0]]), k, dte, sigma=0.18, opt_type=opt_type)[0])
                            p0 = round(p0 * 1.005, 2)
                            c0 = round(p0 * 30, 2)
                            if c0 <= 9500.0 and p0 >= 25.0:
                                chosen_strike = k
                                entry_p = p0
                                cost = c0
                                break
                        if chosen_strike is None: continue

                        opt_p_arr = bs_vectorized(closes, chosen_strike, dte, sigma=0.18, opt_type=opt_type)
                        opt_h_arr = bs_vectorized(highs if opt_type == 'CE' else lows, chosen_strike, dte, sigma=0.18, opt_type=opt_type)
                        opt_l_arr = bs_vectorized(lows if opt_type == 'CE' else highs, chosen_strike, dte, sigma=0.18, opt_type=opt_type)

                        tgt_p = entry_p * 1.80
                        sl_p = entry_p * 0.75
                        exit_p = entry_p

                        be_active = False
                        if be_trigger is not None:
                            be_level = entry_p * (1.0 + be_trigger)

                        for j in range(1, len(rem_df)):
                            c_l = opt_l_arr[j]
                            c_h = opt_h_arr[j]
                            c_p = opt_p_arr[j]

                            # Check if breakeven reached
                            if be_trigger is not None and not be_active and c_h >= be_level:
                                be_active = True
                                sl_p = entry_p # Move SL to cost!

                            if c_l <= sl_p:
                                exit_p = round(sl_p * 0.995, 2)
                                break
                            elif c_h >= tgt_p:
                                exit_p = round(tgt_p * 0.995, 2)
                                break
                            elif sub_times[j] >= t_1515:
                                exit_p = round(float(c_p) * 0.995, 2)
                                break

                        gross, charges, net = calc_charges(entry_p, exit_p, 30)
                        trades.append({'gross': gross, 'charges': charges, 'net_pnl': net})

                    st = compute_stats(trades)
                    be_str = f"+{int(be_trigger*100)}% -> Cost" if be_trigger else "None"
                    results.append({
                        "max_box": max_box, "buf": buf, "cutoff": cutoff_hour, "be": be_str,
                        "trades": st['trades'], "wr": st['win_rate'], "pf": st['pf'],
                        "dd": st['max_dd'], "net": st['net']
                    })

    df_res = pd.DataFrame(results)
    # Sort by Profit Factor and Net PnL
    df_res = df_res.sort_values(by=["pf", "net"], ascending=False).reset_index(drop=True)
    print("\nTOP 10 BANKNIFTY PARAMETER CONFIGURATIONS (BY PROFIT FACTOR):")
    top_table = []
    for i in range(min(10, len(df_res))):
        r = df_res.iloc[i]
        top_table.append([
            f"Box <= {r['max_box']:.2f}%",
            f"Buf +{r['buf']:.0f} pt",
            f"Cutoff {int(r['cutoff'])}:00",
            f"BE: {r['be']}",
            f"{int(r['trades'])}",
            f"{r['wr']:.1f}%",
            f"{r['pf']:.2f}",
            f"INR {r['dd']:,.0f}",
            f"+INR {r['net']:,.2f}"
        ])
    print(tabulate(top_table, headers=["Max Box", "Buffer", "Entry Cutoff", "Trailing/BE", "Trades", "Win Rate", "PF", "Max DD", "Net PnL"], tablefmt="grid"))
    return df_res

def test_nifty_optimizations(n_days):
    print("\n" + "=" * 100)
    print("           STRATEGY 1 (NIFTY): SYSTEMATIC PARAMETER OPTIMIZATION")
    print("=" * 100)
    all_dates = sorted(list(n_days.keys()))
    results = []

    for max_box in [0.25, 0.30, 0.35, 0.40]:
        for buf in [6.0, 8.0, 10.0]:
            for tgt_pct in [0.50, 0.60, 0.70]:
                for sl_pct in [0.14, 0.16, 0.18]:
                    trades = []
                    for d in all_dates:
                        day_df = n_days[d]
                        if len(day_df) < 20: continue
                        times = day_df['time'].values
                        t_0915 = datetime.time(9, 15)
                        t_0930 = datetime.time(9, 30)
                        t_1500 = datetime.time(15, 0)
                        t_1515 = datetime.time(15, 15)

                        box_mask = (times >= t_0915) & (times <= t_0930)
                        if np.sum(box_mask) < 10: continue
                        box_high = float(day_df.loc[box_mask, 'high'].max())
                        box_low = float(day_df.loc[box_mask, 'low'].min())
                        open_p = float(day_df.loc[box_mask, 'close'].iloc[0])
                        box_range = box_high - box_low
                        box_pct = (box_range / open_p) * 100.0
                        if box_pct > max_box: continue

                        u_trig = box_high + buf
                        l_trig = box_low - buf

                        post_mask = (times > t_0930) & (times <= t_1500)
                        post_indices = np.where(post_mask)[0]
                        if len(post_indices) == 0: continue

                        entry_idx = None
                        opt_type = None
                        for idx in post_indices:
                            c = float(day_df.at[idx, 'close'])
                            ema = float(day_df.at[idx, 'ema_20'])
                            if c >= u_trig and c > ema:
                                entry_idx = idx
                                opt_type = 'CE'
                                break
                            elif c <= l_trig and c < ema:
                                entry_idx = idx
                                opt_type = 'PE'
                                break
                        if entry_idx is None: continue

                        entry_spot = float(day_df.at[entry_idx, 'close'])
                        strike = int(round(entry_spot / 50.0) * 50)
                        dte = get_dte(d, 'NIFTY')

                        rem_df = day_df.iloc[entry_idx:].copy().reset_index(drop=True)
                        closes = rem_df['close'].values
                        highs = rem_df['high'].values
                        lows = rem_df['low'].values
                        sub_times = rem_df['time'].values

                        opt_p_arr = bs_vectorized(closes, strike, dte, sigma=0.15, opt_type=opt_type)
                        opt_h_arr = bs_vectorized(highs if opt_type == 'CE' else lows, strike, dte, sigma=0.15, opt_type=opt_type)
                        opt_l_arr = bs_vectorized(lows if opt_type == 'CE' else highs, strike, dte, sigma=0.15, opt_type=opt_type)

                        entry_p = round(float(opt_p_arr[0]) * 1.005, 2)
                        cost = round(entry_p * 65, 2)
                        if cost > 9500.0: continue

                        tgt_p = entry_p * (1.0 + tgt_pct)
                        sl_p = entry_p * (1.0 - sl_pct)
                        exit_p = entry_p

                        for j in range(1, len(rem_df)):
                            if opt_l_arr[j] <= sl_p:
                                exit_p = round(sl_p * 0.995, 2)
                                break
                            elif opt_h_arr[j] >= tgt_p:
                                exit_p = round(tgt_p * 0.995, 2)
                                break
                            elif sub_times[j] >= t_1515:
                                exit_p = round(float(opt_p_arr[j]) * 0.995, 2)
                                break

                        gross, charges, net = calc_charges(entry_p, exit_p, 65)
                        trades.append({'gross': gross, 'charges': charges, 'net_pnl': net})

                    st = compute_stats(trades)
                    results.append({
                        "max_box": max_box, "buf": buf, "tgt": tgt_pct, "sl": sl_pct,
                        "trades": st['trades'], "wr": st['win_rate'], "pf": st['pf'],
                        "dd": st['max_dd'], "net": st['net']
                    })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["net", "pf"], ascending=False).reset_index(drop=True)
    print("\nTOP 10 NIFTY STRATEGY 1 CONFIGURATIONS:")
    top_table = []
    for i in range(min(10, len(df_res))):
        r = df_res.iloc[i]
        top_table.append([
            f"Box <= {r['max_box']:.2f}%",
            f"Buf +{r['buf']:.0f} pt",
            f"TP +{int(r['tgt']*100)}%",
            f"SL -{int(r['sl']*100)}%",
            f"{int(r['trades'])}",
            f"{r['wr']:.1f}%",
            f"{r['pf']:.2f}",
            f"INR {r['dd']:,.0f}",
            f"+INR {r['net']:,.2f}"
        ])
    print(tabulate(top_table, headers=["Max Box", "Buffer", "Target", "Stop Loss", "Trades", "Win Rate", "PF", "Max DD", "Net PnL"], tablefmt="grid"))
    return df_res

def test_das_optimizations(n_days):
    print("\n" + "=" * 100)
    print("           STRATEGY 3 (NIFTY DAS STRANGLE): PARAMETER OPTIMIZATION")
    print("=" * 100)
    all_dates = sorted(list(n_days.keys()))
    results = []

    for std_thresh in [25.0, 30.0, 35.0]:
        for vel_thresh in [30.0, 35.0, 40.0]:
            for hold_bars in [30, 45, 50, 60]:
                trades = []
                for d in all_dates:
                    day_df = n_days[d]
                    if len(day_df) < 50: continue
                    closes = day_df["close"].values
                    highs = day_df["high"].values
                    lows = day_df["low"].values
                    times = day_df["time"].values
                    timestamps = day_df["timestamp"].values
                    n = len(closes)

                    rolling_std = pd.Series(closes).rolling(20).std().values
                    rolling_vel = np.abs(closes - np.roll(closes, 5))
                    rolling_vel[:5] = 0.0

                    dte = get_dte(d, "NIFTY")
                    iv = 0.135

                    for i in range(20, n):
                        t = times[i]
                        if t < datetime.time(9, 45) or t > datetime.time(14, 0): continue
                        if datetime.time(11, 30) <= t <= datetime.time(13, 0): continue

                        std_v = rolling_std[i]
                        vel_v = rolling_vel[i]

                        if std_v < std_thresh and vel_v >= vel_thresh:
                            spot = float(closes[i])
                            atm = int(round(spot / 50.0) * 50)

                            ce_strike, ce_p = None, None
                            for off in [100, 150, 200, 50, 250]:
                                k = atm + off
                                p = float(bs_vectorized(np.array([spot]), k, dte, sigma=iv, opt_type="CE")[0])
                                if 35.0 <= p <= 68.0:
                                    ce_strike, ce_p = k, round(p, 2)
                                    break

                            pe_strike, pe_p = None, None
                            for off in [100, 150, 200, 50, 250]:
                                k = atm - off
                                p = float(bs_vectorized(np.array([spot]), k, dte, sigma=iv, opt_type="PE")[0])
                                if 35.0 <= p <= 68.0:
                                    pe_strike, pe_p = k, round(p, 2)
                                    break

                            if not (ce_strike and pe_strike): continue

                            entry_tot = round((ce_p + pe_p) * 1.005, 2)
                            cost = round(entry_tot * 65, 2)
                            if cost > 9500.0: continue

                            ce_tgt = round(ce_p * 1.60, 2)
                            ce_sl = round(ce_p * 0.88, 2)
                            pe_tgt = round(pe_p * 1.60, 2)
                            pe_sl = round(pe_p * 0.88, 2)

                            ce_exited, pe_exited = False, False
                            ce_exit_p, pe_exit_p = ce_p, pe_p

                            max_j = min(i + hold_bars + 1, n)
                            sub_h = highs[i+1:max_j]
                            sub_l = lows[i+1:max_j]
                            sub_c = closes[i+1:max_j]
                            sub_times_j = times[i+1:max_j]

                            ce_h_arr = bs_vectorized(sub_h, ce_strike, dte, sigma=iv, opt_type="CE")
                            ce_l_arr = bs_vectorized(sub_l, ce_strike, dte, sigma=iv, opt_type="CE")
                            ce_c_arr = bs_vectorized(sub_c, ce_strike, dte, sigma=iv, opt_type="CE")

                            pe_h_arr = bs_vectorized(sub_l, pe_strike, dte, sigma=iv, opt_type="PE")
                            pe_l_arr = bs_vectorized(sub_h, pe_strike, dte, sigma=iv, opt_type="PE")
                            pe_c_arr = bs_vectorized(sub_c, pe_strike, dte, sigma=iv, opt_type="PE")

                            for k_idx in range(len(sub_c)):
                                if not ce_exited:
                                    if ce_l_arr[k_idx] <= ce_sl:
                                        ce_exited = True
                                        ce_exit_p = round(ce_sl * 0.995, 2)
                                    elif ce_h_arr[k_idx] >= ce_tgt:
                                        ce_exited = True
                                        ce_exit_p = round(ce_tgt * 0.995, 2)

                                if not pe_exited:
                                    if pe_l_arr[k_idx] <= pe_sl:
                                        pe_exited = True
                                        pe_exit_p = round(pe_sl * 0.995, 2)
                                    elif pe_h_arr[k_idx] >= pe_tgt:
                                        pe_exited = True
                                        pe_exit_p = round(pe_tgt * 0.995, 2)

                                if ce_exited and pe_exited:
                                    break

                                if k_idx == len(sub_c) - 1 or sub_times_j[k_idx] >= datetime.time(15, 15):
                                    if not ce_exited:
                                        ce_exit_p = round(float(ce_c_arr[k_idx]) * 0.995, 2)
                                    if not pe_exited:
                                        pe_exit_p = round(float(pe_c_arr[k_idx]) * 0.995, 2)
                                    break

                            exit_tot = round(ce_exit_p + pe_exit_p, 2)
                            gross, charges, net = calc_charges(entry_tot, exit_tot, 65, is_strangle=True)
                            trades.append({'gross': gross, 'charges': charges, 'net_pnl': net})
                            break

                st = compute_stats(trades)
                results.append({
                    "std": std_thresh, "vel": vel_thresh, "hold": hold_bars,
                    "trades": st['trades'], "wr": st['win_rate'], "pf": st['pf'],
                    "dd": st['max_dd'], "net": st['net']
                })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["pf", "net"], ascending=False).reset_index(drop=True)
    print("\nTOP 10 NIFTY DAS STRANGLE CONFIGURATIONS:")
    top_table = []
    for i in range(min(10, len(df_res))):
        r = df_res.iloc[i]
        top_table.append([
            f"Std < {r['std']:.0f}",
            f"Vel >= {r['vel']:.0f}",
            f"{int(r['hold'])} mins",
            f"{int(r['trades'])}",
            f"{r['wr']:.1f}%",
            f"{r['pf']:.2f}",
            f"INR {r['dd']:,.0f}",
            f"+INR {r['net']:,.2f}"
        ])
    print(tabulate(top_table, headers=["Compression", "Velocity", "Max Hold", "Trades", "Win Rate", "PF", "Max DD", "Net PnL"], tablefmt="grid"))
    return df_res

if __name__ == "__main__":
    n_days, b_days = load_data()
    test_banknifty_optimizations(b_days)
    test_nifty_optimizations(n_days)
    test_das_optimizations(n_days)
