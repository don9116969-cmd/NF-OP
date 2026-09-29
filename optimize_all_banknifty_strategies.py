"""
Master BankNIFTY Optimization Engine:
Exhaustively sweeps parameters for ALL 4 strategies using precomputed vectorized option grids.
Identifies the optimal, highest-profit configurations specifically for BankNIFTY.

Strategies:
1. Decoupled Asymmetric Strangle (DAS)
2. Directional Box Breakout
3. Gamma Squeeze Momentum
4. Hedged Strangle
"""
import os
import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

LOT_SIZE = 30
FEE_STRANGLE = 80.0     # Round-trip ₹80 for 2 legs (brokerage + STT + GST + turnover)
FEE_DIRECTIONAL = 45.0  # Round-trip ₹45 for 1 leg

# Fast Vectorized Black-Scholes
def bs_vec(S, K, T, opt_type="CE", r=0.07, sigma=0.165):
    T_clip = np.maximum(1e-4, T)
    sqrt_T = np.sqrt(T_clip)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_clip) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm.cdf(d1) - K * np.exp(-r * T_clip) * norm.cdf(d2)
    else:
        p = K * np.exp(-r * T_clip) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.maximum(0.05, p)

def calc_dte_series(ts_series):
    dtes = []
    for ts in ts_series:
        weekday = ts.weekday()
        target_exp = 2  # Wednesday
        if weekday <= target_exp:
            days_ahead = target_exp - weekday
        else:
            days_ahead = 7 - (weekday - target_exp)
        mins_to_expiry = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
        frac = mins_to_expiry / 375.0
        if days_ahead == 0:
            d = max(0.01, frac * 0.5)
        else:
            d = (days_ahead - 1) + frac
        dtes.append(d / 365.0)
    return np.array(dtes)

# =============================================================================
# LOAD AND PRECOMPUTE DATA
# =============================================================================
print("Loading and precomputing BankNIFTY datasets...")

# 1-Minute Data
df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.sort_values("timestamp", inplace=True)
df_1m.reset_index(drop=True, inplace=True)

df_1m["date"] = df_1m["timestamp"].dt.date
df_1m["time"] = df_1m["timestamp"].dt.time
df_1m["dte"] = calc_dte_series(df_1m["timestamp"])
S_1m = df_1m["close"].values
K_atm_1m = np.round(S_1m / 100.0) * 100.0
T_1m = df_1m["dte"].values

# Precompute 1-min option prices
df_1m["ce_atm"] = bs_vec(S_1m, K_atm_1m, T_1m, "CE")
df_1m["pe_atm"] = bs_vec(S_1m, K_atm_1m, T_1m, "PE")
df_1m["ce_otm1"] = bs_vec(S_1m, K_atm_1m + 100, T_1m, "CE")
df_1m["pe_otm1"] = bs_vec(S_1m, K_atm_1m - 100, T_1m, "PE")
df_1m["ce_otm2"] = bs_vec(S_1m, K_atm_1m + 200, T_1m, "CE")
df_1m["pe_otm2"] = bs_vec(S_1m, K_atm_1m - 200, T_1m, "PE")

df_1m["ema_50"] = df_1m["close"].ewm(span=50).mean()

# 5-Minute Data
df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)

df_5m["date"] = df_5m["timestamp"].dt.date
df_5m["time"] = df_5m["timestamp"].dt.time
df_5m["dte"] = calc_dte_series(df_5m["timestamp"])
S_5m = df_5m["close"].values
K_atm_5m = np.round(S_5m / 100.0) * 100.0
T_5m = df_5m["dte"].values

df_5m["ce_atm"] = bs_vec(S_5m, K_atm_5m, T_5m, "CE")
df_5m["pe_atm"] = bs_vec(S_5m, K_atm_5m, T_5m, "PE")
df_5m["ema_20"] = df_5m["close"].ewm(span=20).mean()
df_5m["ema_50"] = df_5m["close"].ewm(span=50).mean()

# Daily VWAP on 5-min
df_5m["typical_p"] = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
df_5m["tp_vol"] = df_5m["typical_p"] * df_5m["volume"]
df_5m["cum_vol"] = df_5m.groupby("date")["volume"].cumsum()
df_5m["cum_tp_vol"] = df_5m.groupby("date")["tp_vol"].cumsum()
df_5m["vwap"] = df_5m["cum_tp_vol"] / df_5m["cum_vol"].clip(lower=1)
df_5m["vwap"].fillna(df_5m["close"].rolling(20).mean(), inplace=True)
if df_5m["vwap"].isna().all():
    df_5m["vwap"] = df_5m["close"].rolling(20).mean()

# ATR on 5-min
tr1 = df_5m["high"] - df_5m["low"]
tr2 = (df_5m["high"] - df_5m["close"].shift(1)).abs()
tr3 = (df_5m["low"] - df_5m["close"].shift(1)).abs()
df_5m["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().fillna(60.0)

print("Precomputation completed successfully!\n")

# =============================================================================
# 1. OPTIMIZE DIRECTIONAL BOX BREAKOUT
# =============================================================================
def optimize_directional_box():
    print("=" * 80)
    print("OPTIMIZING STRATEGY 2: DIRECTIONAL BOX BREAKOUT ON BANKNIFTY")
    print("=" * 80)

    # Pre-extract day groups
    days = []
    for d, grp in df_1m.groupby("date"):
        days.append({
            "times": grp["time"].values,
            "highs": grp["high"].values,
            "lows": grp["low"].values,
            "closes": grp["close"].values,
            "ce_atm": grp["ce_atm"].values,
            "pe_atm": grp["pe_atm"].values,
            "ce_otm1": grp["ce_otm1"].values,
            "pe_otm1": grp["pe_otm1"].values,
            "emas": grp["ema_50"].values
        })

    box_combos = []
    start_t = datetime.time(9, 15)
    eod_t = datetime.time(15, 15)

    for box_mins in [15, 30, 45]:
        end_h = 9 + ((15 + box_mins) // 60)
        end_m = (15 + box_mins) % 60
        box_end_t = datetime.time(end_h, end_m)

        for max_box_pct in [0.45, 0.60, 0.75]:
            for buffer_pts in [30.0, 50.0, 75.0]:
                for tp_pct in [0.50, 0.75, 1.00]:
                    for sl_pct in [0.20, 0.25, 0.30]:
                        for strike_type in ["atm", "otm1"]:
                            trades = []
                            for day in days:
                                times = day["times"]
                                b_mask = (times >= start_t) & (times <= box_end_t)
                                if np.sum(b_mask) < (box_mins // 2):
                                    continue
                                box_h = np.max(day["highs"][b_mask])
                                box_l = np.min(day["lows"][b_mask])
                                box_pct = (box_h - box_l) / day["closes"][b_mask][0] * 100.0
                                if box_pct > max_box_pct:
                                    continue

                                rem_mask = (times > box_end_t) & (times <= eod_t)
                                rem_idxs = np.where(rem_mask)[0]

                                in_pos = False
                                pos_dir = None
                                entry_p = 0.0

                                ce_arr = day["ce_atm"] if strike_type == "atm" else day["ce_otm1"]
                                pe_arr = day["pe_atm"] if strike_type == "atm" else day["pe_otm1"]

                                for i in rem_idxs:
                                    h = day["highs"][i]
                                    l = day["lows"][i]
                                    c = day["closes"][i]
                                    tm = times[i]

                                    if in_pos:
                                        cur_p = ce_arr[i] if pos_dir == "CE" else pe_arr[i]
                                        ret = (cur_p - entry_p) / entry_p
                                        if ret >= tp_pct or ret <= -sl_pct or tm >= eod_t:
                                            net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                                            trades.append(net)
                                            in_pos = False
                                            break
                                    else:
                                        if h >= (box_h + buffer_pts):
                                            pos_dir = "CE"
                                            entry_p = ce_arr[i]
                                            in_pos = True
                                        elif l <= (box_l - buffer_pts):
                                            pos_dir = "PE"
                                            entry_p = pe_arr[i]
                                            in_pos = True

                            if len(trades) >= 10:
                                s = pd.Series(trades)
                                w = s[s > 0].sum()
                                l = abs(s[s <= 0].sum())
                                pf = w / l if l > 0 else 99.0
                                wr = len(s[s > 0]) / len(s) * 100.0
                                tot = s.sum()
                                if pf >= 1.25 and tot > 0:
                                    box_combos.append({
                                        "box_m": box_mins, "max_pct": max_box_pct, "buffer": buffer_pts,
                                        "strike": strike_type, "tp": f"+{int(tp_pct*100)}%", "sl": f"-{int(sl_pct*100)}%",
                                        "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                                    })

    res_df = pd.DataFrame(box_combos)
    if not res_df.empty:
        res_df.sort_values(by="pf", ascending=False, inplace=True)
        print(tabulate(res_df.head(8), headers="keys", tablefmt="grid", showindex=False))
        return res_df.iloc[0].to_dict()
    else:
        print("No combinations met threshold.")
        return None

# =============================================================================
# 2. OPTIMIZE GAMMA SQUEEZE MOMENTUM
# =============================================================================
def optimize_gamma_squeeze():
    print("\n" + "=" * 80)
    print("OPTIMIZING STRATEGY 3: GAMMA SQUEEZE MOMENTUM ON BANKNIFTY (5-MIN)")
    print("=" * 80)

    closes = df_5m["close"].values
    vwaps = df_5m["vwap"].values
    atrs = df_5m["atr"].values
    emas = df_5m["ema_50"].values
    ce_arr = df_5m["ce_atm"].values
    pe_arr = df_5m["pe_atm"].values
    times = df_5m["time"].values
    n = len(df_5m)

    gamma_combos = []

    for atr_mult in [1.5, 1.8, 2.2, 2.5]:
        for tp_pct in [0.60, 0.80, 1.00]:
            for sl_pct in [0.20, 0.25, 0.30]:
                for trend_mode in ["none", "ema_50"]:
                    trades = []
                    in_pos = False
                    pos_dir = None
                    entry_p = 0.0

                    for i in range(25, n):
                        spot = closes[i]
                        vwap = vwaps[i]
                        atr = atrs[i]
                        em = emas[i]
                        tm = times[i]

                        if in_pos:
                            cur_p = ce_arr[i] if pos_dir == "CE" else pe_arr[i]
                            ret = (cur_p - entry_p) / entry_p
                            if ret >= tp_pct or ret <= -sl_pct or (pos_dir == "CE" and spot < vwap) or (pos_dir == "PE" and spot > vwap) or tm >= datetime.time(15, 15):
                                net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                                trades.append(net)
                                in_pos = False
                        else:
                            if tm < datetime.time(9, 30) or tm > datetime.time(14, 15):
                                continue

                            # Long Breakout
                            if spot > (vwap + atr_mult * atr):
                                if trend_mode == "ema_50" and spot < em:
                                    continue
                                pos_dir = "CE"
                                entry_p = ce_arr[i]
                                in_pos = True

                            # Short Breakdown
                            elif spot < (vwap - atr_mult * atr):
                                if trend_mode == "ema_50" and spot > em:
                                    continue
                                pos_dir = "PE"
                                entry_p = pe_arr[i]
                                in_pos = True

                    if len(trades) >= 15:
                        s = pd.Series(trades)
                        w = s[s > 0].sum()
                        l = abs(s[s <= 0].sum())
                        pf = w / l if l > 0 else 99.0
                        wr = len(s[s > 0]) / len(s) * 100.0
                        tot = s.sum()
                        if pf >= 1.20 and tot > 0:
                            gamma_combos.append({
                                "atr_mult": atr_mult, "trend": trend_mode,
                                "tp": f"+{int(tp_pct*100)}%", "sl": f"-{int(sl_pct*100)}%",
                                "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                            })

    res_df = pd.DataFrame(gamma_combos)
    if not res_df.empty:
        res_df.sort_values(by="pf", ascending=False, inplace=True)
        print(tabulate(res_df.head(8), headers="keys", tablefmt="grid", showindex=False))
        return res_df.iloc[0].to_dict()
    else:
        print("No combinations met threshold.")
        return None

# =============================================================================
# 3. OPTIMIZE DECOUPLED ASYMMETRIC STRANGLE (DAS)
# =============================================================================
def optimize_decoupled_strangle():
    print("\n" + "=" * 80)
    print("OPTIMIZING STRATEGY 1: DECOUPLED ASYMMETRIC STRANGLE (DAS)")
    print("=" * 80)

    closes = df_1m["close"].values
    ce_arr = df_1m["ce_otm1"].values
    pe_arr = df_1m["pe_otm1"].values
    times = df_1m["time"].values
    n = len(df_1m)

    das_combos = []

    for std_thresh in [45.0, 60.0, 75.0]:
        for vel_thresh in [25.0, 40.0, 55.0]:
            for win_target in [0.50, 0.70, 0.90]:
                for lose_stop in [0.25, 0.35]:
                    for max_hold in [45, 60, 90]:
                        trades = []
                        in_pos = False
                        ce_entry = 0.0
                        pe_entry = 0.0
                        ce_exit_p = None
                        pe_exit_p = None
                        ce_exited = False
                        pe_exited = False
                        entry_idx = 0
                        last_exit_idx = -100

                        for i in range(30, n):
                            tm = times[i]

                            if in_pos:
                                bars = i - entry_idx
                                c_ce = ce_arr[i]
                                c_pe = pe_arr[i]

                                if not ce_exited:
                                    r_ce = (c_ce - ce_entry) / ce_entry
                                    if r_ce >= win_target:
                                        ce_exited = True
                                        ce_exit_p = c_ce
                                    elif r_ce <= -lose_stop:
                                        ce_exited = True
                                        ce_exit_p = c_ce

                                if not pe_exited:
                                    r_pe = (c_pe - pe_entry) / pe_entry
                                    if r_pe >= win_target:
                                        pe_exited = True
                                        pe_exit_p = c_pe
                                    elif r_pe <= -lose_stop:
                                        pe_exited = True
                                        pe_exit_p = c_pe

                                force_close = (ce_exited and pe_exited) or (bars >= max_hold) or (tm >= datetime.time(15, 15))
                                if force_close:
                                    p_c = ce_exit_p if ce_exited else c_ce
                                    p_p = pe_exit_p if pe_exited else c_pe
                                    net = (p_c - ce_entry + p_p - pe_entry) * LOT_SIZE - FEE_STRANGLE
                                    trades.append(net)
                                    in_pos = False
                                    last_exit_idx = i

                            else:
                                if tm < datetime.time(9, 45) or tm > datetime.time(13, 30):
                                    continue
                                if datetime.time(11, 30) <= tm <= datetime.time(12, 45):
                                    continue
                                if (i - last_exit_idx) < 30:
                                    continue

                                sub_c = closes[i-20:i+1]
                                std_val = np.std(sub_c)
                                vel_val = sub_c[-1] - sub_c[-3]

                                if std_val < std_thresh and abs(vel_val) >= vel_thresh:
                                    in_pos = True
                                    entry_idx = i
                                    ce_entry = ce_arr[i]
                                    pe_entry = pe_arr[i]
                                    ce_exited = False
                                    pe_exited = False
                                    ce_exit_p = None
                                    pe_exit_p = None

                        if len(trades) >= 10:
                            s = pd.Series(trades)
                            w = s[s > 0].sum()
                            l = abs(s[s <= 0].sum())
                            pf = w / l if l > 0 else 99.0
                            wr = len(s[s > 0]) / len(s) * 100.0
                            tot = s.sum()
                            if pf >= 1.25 and tot > 0:
                                das_combos.append({
                                    "std_th": std_thresh, "vel_th": vel_thresh, "target": f"+{int(win_target*100)}%",
                                    "stop": f"-{int(lose_stop*100)}%", "hold_m": max_hold,
                                    "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                                })

    res_df = pd.DataFrame(das_combos)
    if not res_df.empty:
        res_df.sort_values(by="pf", ascending=False, inplace=True)
        print(tabulate(res_df.head(8), headers="keys", tablefmt="grid", showindex=False))
        return res_df.iloc[0].to_dict()
    else:
        print("No combinations met threshold.")
        return None

# =============================================================================
# 4. OPTIMIZE HEDGED STRANGLE (COUPLED)
# =============================================================================
def optimize_hedged_strangle():
    print("\n" + "=" * 80)
    print("OPTIMIZING STRATEGY 4: HEDGED STRANGLE (COUPLED)")
    print("=" * 80)

    ce_arr = df_1m["ce_otm2"].values
    pe_arr = df_1m["pe_otm2"].values
    times = df_1m["time"].values
    ts_list = df_1m["timestamp"].values
    n = len(df_1m)

    hedged_combos = []

    for entry_min in [20, 35, 45]:
        entry_time_target = datetime.time(9, entry_min)
        for target_pct in [0.30, 0.45, 0.60]:
            for stop_pct in [0.15, 0.20, 0.25]:
                for hold_m in [45, 60, 90]:
                    trades = []
                    in_pos = False
                    entry_tot = 0.0
                    entry_idx = 0

                    for i in range(20, n):
                        tm = times[i]
                        weekday = ts_list[i].weekday()

                        if in_pos:
                            bars = i - entry_idx
                            cur_tot = ce_arr[i] + pe_arr[i]
                            ret = (cur_tot - entry_tot) / entry_tot

                            if ret >= target_pct or ret <= -stop_pct or bars >= hold_m or tm >= datetime.time(15, 15):
                                net = (cur_tot - entry_tot) * LOT_SIZE - FEE_STRANGLE
                                trades.append(net)
                                in_pos = False
                        else:
                            if tm != entry_time_target:
                                continue
                            if weekday not in [1, 2]:  # Tuesday / Wednesday (Expiry proximity)
                                continue

                            entry_tot = ce_arr[i] + pe_arr[i]
                            in_pos = True
                            entry_idx = i

                    if len(trades) >= 5:
                        s = pd.Series(trades)
                        w = s[s > 0].sum()
                        l = abs(s[s <= 0].sum())
                        pf = w / l if l > 0 else 99.0
                        wr = len(s[s > 0]) / len(s) * 100.0
                        tot = s.sum()
                        if pf >= 1.20 and tot > 0:
                            hedged_combos.append({
                                "entry_t": str(entry_time_target)[:5], "target": f"+{int(target_pct*100)}%",
                                "stop": f"-{int(stop_pct*100)}%", "hold_m": hold_m,
                                "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                            })

    res_df = pd.DataFrame(hedged_combos)
    if not res_df.empty:
        res_df.sort_values(by="pf", ascending=False, inplace=True)
        print(tabulate(res_df.head(8), headers="keys", tablefmt="grid", showindex=False))
        return res_df.iloc[0].to_dict()
    else:
        print("No combinations met threshold.")
        return None

if __name__ == "__main__":
    t_start = datetime.datetime.now()
    best_box = optimize_directional_box()
    best_gamma = optimize_gamma_squeeze()
    best_das = optimize_decoupled_strangle()
    best_hedged = optimize_hedged_strangle()
    t_dur = (datetime.datetime.now() - t_start).total_seconds()
    print(f"\nExhaustive optimization of all 4 BankNIFTY strategies completed in {t_dur:.2f} seconds!")
