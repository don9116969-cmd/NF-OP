"""
High-Speed Vectorized Parameter Optimizer for Strategies 2, 4, 5, 7:
- Strategy 2: Nifty Expiry Gamma Squeeze (GSM) - Morning vs Afternoon Window, Target/SL sweep
- Strategy 4: Nifty Decoupled Strangle (DAS) - Compression/Velocity thresholds, Asymmetric Exits
- Strategy 5: BankNIFTY Decoupled Strangle (DAS) - Kinetic filters, Asymmetric Exits
- Strategy 7: BankNIFTY Expiry Gamma Squeeze (GSM) - Afternoon Gamma window, Target/SL sweep
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from config import config
from src.banknifty.banknifty_das import bs_price, get_banknifty_dte

# -------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------
def load_datasets():
    print("Loading datasets...")
    df_n = pd.read_csv("data/nifty_1min_real.csv")
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n["date"] = df_n["timestamp"].dt.date
    df_n.sort_values("timestamp", inplace=True)
    df_n.reset_index(drop=True, inplace=True)
    df_n["ema_20"] = df_n["close"].ewm(span=20, adjust=False).mean()

    df_bn = pd.read_csv("data/banknifty_1min_real.csv")
    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
    df_bn["date"] = df_bn["timestamp"].dt.date
    df_bn.sort_values("timestamp", inplace=True)
    df_bn.reset_index(drop=True, inplace=True)
    df_bn["ema_20"] = df_bn["close"].ewm(span=20, adjust=False).mean()

    n_days = {d: group.copy() for d, group in df_n.groupby("date") if len(group) >= 20}
    bn_days = {d: group.copy() for d, group in df_bn.groupby("date") if len(group) >= 20}

    return n_days, bn_days

# =============================================================
# 1. OPTIMIZE STRATEGY 2: NIFTY EXPIRY GAMMA SQUEEZE (GSM)
# =============================================================
def optimize_strategy_2(n_days):
    print("\n" + "=" * 90)
    print("OPTIMIZING STRATEGY 2: NIFTY EXPIRY GAMMA SQUEEZE (GSM)")
    print("=" * 90)

    # Monday (1-DTE) & Tuesday (0-DTE) Expiry Sessions
    results = []

    for window_type in ["morning", "afternoon"]:
        for compression_mins in [15, 20, 30]:
            for buf in [4.0, 6.0, 8.0]:
                trades_cache = []
                for target_date, day_bars in n_days.items():
                    weekday = target_date.weekday()
                    if weekday not in [0, 1]:  # Monday or Tuesday only
                        continue

                    # Window boundaries
                    if window_type == "morning":
                        w_start = datetime.time(9, 15)
                        tot_m = 15 + compression_mins
                        w_eval = datetime.time(9 + tot_m // 60, tot_m % 60)
                        w_end = datetime.time(11, 30)
                    else: # afternoon gamma window
                        w_start = datetime.time(12, 30)
                        tot_m = 30 + compression_mins
                        w_eval = datetime.time(12 + tot_m // 60, tot_m % 60)
                        w_end = datetime.time(14, 45)

                    c_bars = day_bars[(day_bars["timestamp"].dt.time >= w_start) & (day_bars["timestamp"].dt.time <= w_eval)]
                    if len(c_bars) < (compression_mins - 3):
                        continue

                    box_h = float(c_bars["high"].max())
                    box_l = float(c_bars["low"].min())
                    u_trig = box_h + buf
                    l_trig = box_l - buf

                    post = day_bars[(day_bars["timestamp"].dt.time > w_eval) & (day_bars["timestamp"].dt.time <= w_end)]
                    entry_idx = None
                    opt_type = None

                    for idx, row in post.iterrows():
                        c = float(row["close"])
                        if c >= u_trig:
                            entry_idx = idx
                            opt_type = "CE"
                            break
                        elif c <= l_trig:
                            entry_idx = idx
                            opt_type = "PE"
                            break

                    if entry_idx is None:
                        continue

                    entry_bar = post.loc[entry_idx]
                    entry_spot = float(entry_bar["close"])
                    entry_t = entry_bar["timestamp"].time()

                    # Select OTM/ATM strike near target premium (e.g. 35-50 pts)
                    days_ahead = 0 if weekday == 1 else 1
                    mins = max(1.0, 930 - (entry_t.hour * 60 + entry_t.minute))
                    T_years = max(0.0005, (days_ahead * 375 + mins) / (365.0 * 375.0))

                    atm = int(round(entry_spot / 50.0) * 50)
                    # Pick strike where premium is between 30 and 55 pts
                    chosen_strike = atm
                    chosen_p = 0.0
                    for off in range(0, 500, 50):
                        k = atm + off if opt_type == "CE" else atm - off
                        p = bs_price(entry_spot, k, T_years, opt_type=opt_type, sigma=0.15)
                        if p <= 50.0 or off == 450:
                            chosen_strike = k
                            chosen_p = p
                            break

                    entry_p = round(chosen_p * (1.0 + config.SLIPPAGE_PCT), 2)
                    if entry_p <= 5.0 or (entry_p * 65) > 8500.0:
                        continue

                    trade_slice = day_bars.loc[entry_idx:].copy()
                    highs = trade_slice["high"].values
                    lows = trade_slice["low"].values
                    closes = trade_slice["close"].values
                    times = trade_slice["timestamp"].dt.time.values

                    opt_highs = np.array([bs_price(h if opt_type == "CE" else l, chosen_strike, T_years, opt_type=opt_type, sigma=0.15) for h, l in zip(highs, lows)])
                    opt_lows = np.array([bs_price(l if opt_type == "CE" else h, chosen_strike, T_years, opt_type=opt_type, sigma=0.15) for h, l in zip(highs, lows)])
                    opt_closes = np.array([bs_price(c, chosen_strike, T_years, opt_type=opt_type, sigma=0.15) for c in closes])

                    trades_cache.append({
                        "entry_p": entry_p,
                        "opt_highs": opt_highs,
                        "opt_lows": opt_lows,
                        "opt_closes": opt_closes,
                        "times": times
                    })

                # Sweep Exits
                if len(trades_cache) >= 20:
                    for sl in [0.15, 0.20, 0.25]:
                        for tgt in [0.50, 0.80, 1.20, 1.50]:
                            for trail in [0.0, 0.25]:
                                pnls = []
                                for tr in trades_cache:
                                    ep = tr["entry_p"]
                                    sl_p = ep * (1.0 - sl)
                                    tgt_p = ep * (1.0 + tgt)
                                    peak = ep
                                    exit_p = ep

                                    oh = tr["opt_highs"]
                                    ol = tr["opt_lows"]
                                    oc = tr["opt_closes"]
                                    tm = tr["times"]

                                    for i in range(1, min(len(oh), 60)): # 60 min holding cutoff
                                        if oh[i] > peak: peak = oh[i]
                                        dyn_sl = max(sl_p, peak * (1.0 - trail)) if trail > 0 and peak >= ep * (1.0 + trail) else sl_p
                                        if oh[i] >= tgt_p:
                                            exit_p = tgt_p * (1.0 - config.SLIPPAGE_PCT)
                                            break
                                        elif ol[i] <= dyn_sl:
                                            exit_p = dyn_sl * (1.0 - config.SLIPPAGE_PCT)
                                            break
                                        elif tm[i] >= datetime.time(15, 15):
                                            exit_p = oc[i] * (1.0 - config.SLIPPAGE_PCT)
                                            break

                                    net = (exit_p - ep) * 65 - (45.0 + (ep + exit_p) * 65 * 0.0015)
                                    pnls.append(net)

                                wins = [p for p in pnls if p > 0]
                                losses = [p for p in pnls if p <= 0]
                                wr = (len(wins) / len(pnls)) * 100.0
                                tot_win = sum(wins)
                                tot_loss = abs(sum(losses))
                                pf = (tot_win / tot_loss) if tot_loss > 0 else 99.0
                                net_tot = sum(pnls)

                                if pf > 1.15 and net_tot > 0:
                                    results.append({
                                        "window": window_type, "comp": compression_mins, "buf": buf,
                                        "sl": sl, "tgt": tgt, "trail": trail, "trades": len(pnls),
                                        "wr": wr, "pf": pf, "net": net_tot
                                    })

    df_s2 = pd.DataFrame(results)
    if not df_s2.empty:
        df_s2.sort_values("net", ascending=False, inplace=True)
        best = df_s2.iloc[0]
        print(f"  >>> BEST FOUND (S2): Window={best['window'].upper()} ({best['comp']}m comp) | Buf={best['buf']}pt | SL:-{int(best['sl']*100)}% | TP:+{int(best['tgt']*100)}% | Trail:{int(best['trail']*100)}%")
        print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f}")
        return best.to_dict()
    else:
        print("  >>> No profitable configuration found in strict gamma sweep.")
        return None

# =============================================================
# 2. OPTIMIZE STRATEGY 4: NIFTY DECOUPLED STRANGLE (DAS)
# =============================================================
def optimize_strategy_4(n_days):
    print("\n" + "=" * 90)
    print("OPTIMIZING STRATEGY 4: NIFTY DECOUPLED STRANGLE (DAS)")
    print("=" * 90)

    results = []

    for std_thresh in [18.0, 24.0, 30.0]:
        for vel_thresh in [25.0, 35.0, 45.0]:
            trades_cache = []
            for target_date, day_bars in n_days.items():
                if len(day_bars) < 30:
                    continue

                day = day_bars.copy()
                day["std"] = day["close"].rolling(20).std()
                day["vel"] = day["close"] - day["close"].shift(3)

                entry_idx = None
                direction = None
                for idx, row in day[(day["timestamp"].dt.time >= datetime.time(9, 30)) & (day["timestamp"].dt.time <= datetime.time(11, 0))].iterrows():
                    s = row["std"]
                    v = row["vel"]
                    if pd.notna(s) and pd.notna(v) and s < std_thresh and abs(v) >= vel_thresh:
                        entry_idx = idx
                        direction = "UP" if v > 0 else "DOWN"
                        break

                if entry_idx is None:
                    continue

                entry_bar = day.loc[entry_idx]
                entry_spot = float(entry_bar["close"])
                entry_t = entry_bar["timestamp"].time()

                weekday = target_date.weekday()
                days_ahead = (1 - weekday) % 7
                T_years = max(0.001, days_ahead / 365.0)

                # Strikes for strangle
                atm = int(round(entry_spot / 50.0) * 50)
                ce_strike = atm + 50
                pe_strike = atm - 50

                ce_entry = bs_price(entry_spot, ce_strike, T_years, "CE", 0.14) * 1.005
                pe_entry = bs_price(entry_spot, pe_strike, T_years, "PE", 0.14) * 1.005
                tot_cost = (ce_entry + pe_entry) * 65
                if tot_cost > 10000.0 or ce_entry <= 5.0 or pe_entry <= 5.0:
                    continue

                post = day.loc[entry_idx:].copy()
                highs = post["high"].values
                lows = post["low"].values
                closes = post["close"].values

                ce_highs = np.array([bs_price(h, ce_strike, T_years, "CE", 0.14) for h in highs])
                ce_lows = np.array([bs_price(l, ce_strike, T_years, "CE", 0.14) for l in lows])
                pe_highs = np.array([bs_price(l, pe_strike, T_years, "PE", 0.14) for l in lows])
                pe_lows = np.array([bs_price(h, pe_strike, T_years, "PE", 0.14) for h in highs])

                trades_cache.append({
                    "ce_entry": ce_entry, "pe_entry": pe_entry,
                    "ce_highs": ce_highs, "ce_lows": ce_lows,
                    "pe_highs": pe_highs, "pe_lows": pe_lows
                })

            if len(trades_cache) >= 30:
                for win_tgt in [0.45, 0.60, 0.80]:
                    for lose_sl in [0.12, 0.18, 0.25]:
                        for max_hold in [20, 35, 50]:
                            pnls = []
                            for tr in trades_cache:
                                ce_in = tr["ce_entry"]
                                pe_in = tr["pe_entry"]
                                ce_tgt = ce_in * (1.0 + win_tgt)
                                ce_sl = ce_in * (1.0 - lose_sl)
                                pe_tgt = pe_in * (1.0 + win_tgt)
                                pe_sl = pe_in * (1.0 - lose_sl)

                                ce_out = ce_in
                                pe_out = pe_in
                                ce_done = False
                                pe_done = False

                                n_bars = min(len(tr["ce_highs"]), max_hold)
                                for i in range(1, n_bars):
                                    if not ce_done:
                                        if tr["ce_highs"][i] >= ce_tgt:
                                            ce_out = ce_tgt * 0.995; ce_done = True
                                        elif tr["ce_lows"][i] <= ce_sl:
                                            ce_out = ce_sl * 0.995; ce_done = True
                                    if not pe_done:
                                        if tr["pe_highs"][i] >= pe_tgt:
                                            pe_out = pe_tgt * 0.995; pe_done = True
                                        elif tr["pe_lows"][i] <= pe_sl:
                                            pe_out = pe_sl * 0.995; pe_done = True

                                    # Combined stop loss (-10%)
                                    comb_pnl_pct = ((ce_out - ce_in) + (pe_out - pe_in)) / (ce_in + pe_in)
                                    if comb_pnl_pct <= -0.10:
                                        break
                                    if ce_done and pe_done:
                                        break

                                gross = ((ce_out - ce_in) + (pe_out - pe_in)) * 65
                                charges = 80.0 + ((ce_in + pe_in) + (ce_out + pe_out)) * 65 * 0.0015
                                pnls.append(gross - charges)

                            wins = [p for p in pnls if p > 0]
                            losses = [p for p in pnls if p <= 0]
                            wr = (len(wins) / len(pnls)) * 100.0
                            tot_win = sum(wins)
                            tot_loss = abs(sum(losses))
                            pf = (tot_win / tot_loss) if tot_loss > 0 else 99.0
                            net_tot = sum(pnls)

                            if pf > 1.05 and net_tot > 0:
                                results.append({
                                    "std": std_thresh, "vel": vel_thresh, "win_tgt": win_tgt,
                                    "lose_sl": lose_sl, "hold": max_hold, "trades": len(pnls),
                                    "wr": wr, "pf": pf, "net": net_tot
                                })

    df_s4 = pd.DataFrame(results)
    if not df_s4.empty:
        df_s4.sort_values("net", ascending=False, inplace=True)
        best = df_s4.iloc[0]
        print(f"  >>> BEST FOUND (S4): Std<{best['std']} | Vel>={best['vel']} | WinTP:+{int(best['win_tgt']*100)}% | LoseSL:-{int(best['lose_sl']*100)}% | Hold:{int(best['hold'])}m")
        print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f}")
        return best.to_dict()
    else:
        print("  >>> No profitable configuration found for S4 strangle. Recommend pure Directional DAS.")
        return None

# =============================================================
# 3. OPTIMIZE STRATEGY 5: BANKNIFTY DECOUPLED STRANGLE (DAS)
# =============================================================
def optimize_strategy_5(bn_days):
    print("\n" + "=" * 90)
    print("OPTIMIZING STRATEGY 5: BANKNIFTY DECOUPLED STRANGLE (DAS)")
    print("=" * 90)

    results = []

    for std_thresh in [25.0, 35.0, 50.0]:
        for vel_thresh in [45.0, 70.0, 100.0]:
            trades_cache = []
            for target_date, day_bars in bn_days.items():
                if len(day_bars) < 30:
                    continue

                day = day_bars.copy()
                day["std"] = day["close"].rolling(20).std()
                day["vel"] = day["close"] - day["close"].shift(3)

                entry_idx = None
                for idx, row in day[(day["timestamp"].dt.time >= datetime.time(9, 30)) & (day["timestamp"].dt.time <= datetime.time(11, 0))].iterrows():
                    s = row["std"]
                    v = row["vel"]
                    if pd.notna(s) and pd.notna(v) and s < std_thresh and abs(v) >= vel_thresh:
                        entry_idx = idx
                        break

                if entry_idx is None:
                    continue

                entry_bar = day.loc[entry_idx]
                entry_spot = float(entry_bar["close"])
                entry_t = entry_bar["timestamp"].time()

                dte_y = max(1e-4, get_banknifty_dte(entry_bar["timestamp"], target_date))
                atm = int(round(entry_spot / 100.0) * 100)
                ce_strike = atm + 200
                pe_strike = atm - 200

                ce_entry = bs_price(entry_spot, ce_strike, dte_y, "CE", 0.18) * 1.005
                pe_entry = bs_price(entry_spot, pe_strike, dte_y, "PE", 0.18) * 1.005
                tot_cost = (ce_entry + pe_entry) * 30
                if tot_cost > 8500.0 or ce_entry <= 10.0 or pe_entry <= 10.0:
                    continue

                post = day.loc[entry_idx:].copy()
                highs = post["high"].values
                lows = post["low"].values

                ce_highs = np.array([bs_price(h, ce_strike, dte_y, "CE", 0.18) for h in highs])
                ce_lows = np.array([bs_price(l, ce_strike, dte_y, "CE", 0.18) for l in lows])
                pe_highs = np.array([bs_price(l, pe_strike, dte_y, "PE", 0.18) for l in lows])
                pe_lows = np.array([bs_price(h, pe_strike, dte_y, "PE", 0.18) for h in highs])

                trades_cache.append({
                    "ce_entry": ce_entry, "pe_entry": pe_entry,
                    "ce_highs": ce_highs, "ce_lows": ce_lows,
                    "pe_highs": pe_highs, "pe_lows": pe_lows
                })

            if len(trades_cache) >= 25:
                for win_tgt in [0.50, 0.75, 1.00]:
                    for lose_sl in [0.12, 0.18, 0.25]:
                        for max_hold in [25, 40, 60]:
                            pnls = []
                            for tr in trades_cache:
                                ce_in = tr["ce_entry"]
                                pe_in = tr["pe_entry"]
                                ce_tgt = ce_in * (1.0 + win_tgt)
                                ce_sl = ce_in * (1.0 - lose_sl)
                                pe_tgt = pe_in * (1.0 + win_tgt)
                                pe_sl = pe_in * (1.0 - lose_sl)

                                ce_out = ce_in
                                pe_out = pe_in
                                ce_done = False
                                pe_done = False

                                n_bars = min(len(tr["ce_highs"]), max_hold)
                                for i in range(1, n_bars):
                                    if not ce_done:
                                        if tr["ce_highs"][i] >= ce_tgt:
                                            ce_out = ce_tgt * 0.995; ce_done = True
                                        elif tr["ce_lows"][i] <= ce_sl:
                                            ce_out = ce_sl * 0.995; ce_done = True
                                    if not pe_done:
                                        if tr["pe_highs"][i] >= pe_tgt:
                                            pe_out = pe_tgt * 0.995; pe_done = True
                                        elif tr["pe_lows"][i] <= pe_sl:
                                            pe_out = pe_sl * 0.995; pe_done = True

                                    comb_pnl_pct = ((ce_out - ce_in) + (pe_out - pe_in)) / (ce_in + pe_in)
                                    if comb_pnl_pct <= -0.10:
                                        break
                                    if ce_done and pe_done:
                                        break

                                gross = ((ce_out - ce_in) + (pe_out - pe_in)) * 30
                                charges = 80.0 + ((ce_in + pe_in) + (ce_out + pe_out)) * 30 * 0.0015
                                pnls.append(gross - charges)

                            wins = [p for p in pnls if p > 0]
                            losses = [p for p in pnls if p <= 0]
                            wr = (len(wins) / len(pnls)) * 100.0
                            tot_win = sum(wins)
                            tot_loss = abs(sum(losses))
                            pf = (tot_win / tot_loss) if tot_loss > 0 else 99.0
                            net_tot = sum(pnls)

                            if pf > 1.10 and net_tot > 0:
                                results.append({
                                    "std": std_thresh, "vel": vel_thresh, "win_tgt": win_tgt,
                                    "lose_sl": lose_sl, "hold": max_hold, "trades": len(pnls),
                                    "wr": wr, "pf": pf, "net": net_tot
                                })

    df_s5 = pd.DataFrame(results)
    if not df_s5.empty:
        df_s5.sort_values("net", ascending=False, inplace=True)
        best = df_s5.iloc[0]
        print(f"  >>> BEST FOUND (S5): Std<{best['std']} | Vel>={best['vel']} | WinTP:+{int(best['win_tgt']*100)}% | LoseSL:-{int(best['lose_sl']*100)}% | Hold:{int(best['hold'])}m")
        print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f}")
        return best.to_dict()
    else:
        print("  >>> No profitable configuration found for BankNIFTY Strangle DAS.")
        return None

# =============================================================
# 4. OPTIMIZE STRATEGY 7: BANKNIFTY EXPIRY GAMMA SQUEEZE (GSM)
# =============================================================
def optimize_strategy_7(bn_days):
    print("\n" + "=" * 90)
    print("OPTIMIZING STRATEGY 7: BANKNIFTY EXPIRY GAMMA SQUEEZE (GSM)")
    print("=" * 90)

    results = []

    for window in ["morning", "afternoon"]:
        for compression_mins in [15, 20, 30]:
            for buf in [30.0, 50.0, 80.0]:
                trades_cache = []
                for target_date, day_bars in bn_days.items():
                    weekday = target_date.weekday()
                    if weekday not in [2, 3]:  # Wednesday or Thursday only (BN Expiry)
                        continue

                    if window == "morning":
                        w_start = datetime.time(9, 15)
                        tot_m = 15 + compression_mins
                        w_eval = datetime.time(9 + tot_m // 60, tot_m % 60)
                        w_end = datetime.time(11, 30)
                    else:
                        w_start = datetime.time(12, 30)
                        tot_m = 30 + compression_mins
                        w_eval = datetime.time(12 + tot_m // 60, tot_m % 60)
                        w_end = datetime.time(14, 45)

                    c_bars = day_bars[(day_bars["timestamp"].dt.time >= w_start) & (day_bars["timestamp"].dt.time <= w_eval)]
                    if len(c_bars) < (compression_mins - 3):
                        continue

                    box_h = float(c_bars["high"].max())
                    box_l = float(c_bars["low"].min())
                    u_trig = box_h + buf
                    l_trig = box_l - buf

                    post = day_bars[(day_bars["timestamp"].dt.time > w_eval) & (day_bars["timestamp"].dt.time <= w_end)]
                    entry_idx = None
                    opt_type = None

                    for idx, row in post.iterrows():
                        c = float(row["close"])
                        if c >= u_trig:
                            entry_idx = idx
                            opt_type = "CE"
                            break
                        elif c <= l_trig:
                            entry_idx = idx
                            opt_type = "PE"
                            break

                    if entry_idx is None:
                        continue

                    entry_bar = post.loc[entry_idx]
                    entry_spot = float(entry_bar["close"])
                    entry_t = entry_bar["timestamp"].time()

                    dte_y = max(1e-4, get_banknifty_dte(entry_bar["timestamp"], target_date))
                    atm = int(round(entry_spot / 100.0) * 100)

                    # Strike near 50-80 pts premium
                    chosen_strike = atm
                    chosen_p = 0.0
                    for off in range(0, 1000, 100):
                        k = atm + off if opt_type == "CE" else atm - off
                        p = bs_price(entry_spot, k, dte_y, opt_type=opt_type, sigma=0.18)
                        if p <= 80.0 or off == 900:
                            chosen_strike = k
                            chosen_p = p
                            break

                    entry_p = round(chosen_p * 1.005, 2)
                    if entry_p <= 10.0 or (entry_p * 30) > 8500.0:
                        continue

                    trade_slice = day_bars.loc[entry_idx:].copy()
                    highs = trade_slice["high"].values
                    lows = trade_slice["low"].values
                    closes = trade_slice["close"].values
                    times = trade_slice["timestamp"].dt.time.values

                    opt_highs = np.array([bs_price(h if opt_type == "CE" else l, chosen_strike, dte_y, opt_type=opt_type, sigma=0.18) for h, l in zip(highs, lows)])
                    opt_lows = np.array([bs_price(l if opt_type == "CE" else h, chosen_strike, dte_y, opt_type=opt_type, sigma=0.18) for h, l in zip(highs, lows)])
                    opt_closes = np.array([bs_price(c, chosen_strike, dte_y, opt_type=opt_type, sigma=0.18) for c in closes])

                    trades_cache.append({
                        "entry_p": entry_p,
                        "opt_highs": opt_highs,
                        "opt_lows": opt_lows,
                        "opt_closes": opt_closes,
                        "times": times
                    })

                if len(trades_cache) >= 20:
                    for sl in [0.15, 0.20, 0.25]:
                        for tgt in [0.50, 0.80, 1.20, 1.50]:
                            for trail in [0.0, 0.25]:
                                pnls = []
                                for tr in trades_cache:
                                    ep = tr["entry_p"]
                                    sl_p = ep * (1.0 - sl)
                                    tgt_p = ep * (1.0 + tgt)
                                    peak = ep
                                    exit_p = ep

                                    oh = tr["opt_highs"]
                                    ol = tr["opt_lows"]
                                    oc = tr["opt_closes"]
                                    tm = tr["times"]

                                    for i in range(1, min(len(oh), 60)):
                                        if oh[i] > peak: peak = oh[i]
                                        dyn_sl = max(sl_p, peak * (1.0 - trail)) if trail > 0 and peak >= ep * (1.0 + trail) else sl_p
                                        if oh[i] >= tgt_p:
                                            exit_p = tgt_p * 0.995
                                            break
                                        elif ol[i] <= dyn_sl:
                                            exit_p = dyn_sl * 0.995
                                            break
                                        elif tm[i] >= datetime.time(15, 15):
                                            exit_p = oc[i] * 0.995
                                            break

                                    net = (exit_p - ep) * 30 - (45.0 + (ep + exit_p) * 30 * 0.0015)
                                    pnls.append(net)

                                wins = [p for p in pnls if p > 0]
                                losses = [p for p in pnls if p <= 0]
                                wr = (len(wins) / len(pnls)) * 100.0
                                tot_win = sum(wins)
                                tot_loss = abs(sum(losses))
                                pf = (tot_win / tot_loss) if tot_loss > 0 else 99.0
                                net_tot = sum(pnls)

                                if pf > 1.15 and net_tot > 0:
                                    results.append({
                                        "window": window, "comp": compression_mins, "buf": buf,
                                        "sl": sl, "tgt": tgt, "trail": trail, "trades": len(pnls),
                                        "wr": wr, "pf": pf, "net": net_tot
                                    })

    df_s7 = pd.DataFrame(results)
    if not df_s7.empty:
        df_s7.sort_values("net", ascending=False, inplace=True)
        best = df_s7.iloc[0]
        print(f"  >>> BEST FOUND (S7): Window={best['window'].upper()} ({best['comp']}m comp) | Buf={best['buf']}pt | SL:-{int(best['sl']*100)}% | TP:+{int(best['tgt']*100)}% | Trail:{int(best['trail']*100)}%")
        print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f}")
        return best.to_dict()
    else:
        print("  >>> No profitable configuration found for BankNIFTY GSM.")
        return None

def run_all_optimizations():
    n_days, bn_days = load_datasets()
    best_s2 = optimize_strategy_2(n_days)
    best_s4 = optimize_strategy_4(n_days)
    best_s5 = optimize_strategy_5(bn_days)
    best_s7 = optimize_strategy_7(bn_days)

if __name__ == "__main__":
    run_all_optimizations()
