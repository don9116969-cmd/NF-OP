"""
Comprehensive BankNIFTY Parametric Search:
Captures the top performing combinations across ALL 4 strategies on both:
1. Real SmartAPI Traded Option Data (Angel One)
2. Synthetic Data Feed
"""
import os
import glob
import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_DIRECTIONAL = 45.0
CACHE_DIR = "data/banknifty_options_cache"

# Load 1-min and 5-min Spot
df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.sort_values("timestamp", inplace=True)
df_1m.reset_index(drop=True, inplace=True)
df_1m["date"] = df_1m["timestamp"].dt.date
df_1m["time"] = df_1m["timestamp"].dt.time

df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)
df_5m["date"] = df_5m["timestamp"].dt.date
df_5m["time"] = df_5m["timestamp"].dt.time

# Load Real Options
real_opts = {}
for p in glob.glob(f"{CACHE_DIR}/*.csv"):
    sym = os.path.basename(p).replace(".csv", "")
    df_opt = pd.read_csv(p)
    df_opt["timestamp"] = pd.to_datetime(df_opt["timestamp"]).dt.tz_localize(None)
    df_opt.set_index("timestamp", inplace=True)
    df_opt.sort_index(inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce")
    real_opts[sym] = df_opt

# Fast BS
def bs_price(S, K, T, opt_type="CE", r=0.07, sigma=0.165):
    T_c = max(1e-4, T)
    sqrt_T = np.sqrt(T_c)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_c) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm.cdf(d1) - K * np.exp(-r * T_c) * norm.cdf(d2)
    else:
        p = K * np.exp(-r * T_c) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0.05, float(p))

def get_dte_val(ts):
    weekday = ts.weekday()
    target_exp = 2
    if weekday <= target_exp:
        days_ahead = target_exp - weekday
    else:
        days_ahead = 7 - (weekday - target_exp)
    mins = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
    frac = mins / 375.0
    d = max(0.01, frac * 0.5) if days_ahead == 0 else (days_ahead - 1) + frac
    return d / 365.0

# =============================================================================
# 1. SWEEP STRATEGY 1: DECOUPLED ASYMMETRIC STRANGLE (DAS)
# =============================================================================
def sweep_das(use_real=True):
    feed_name = "Real Options" if use_real else "Synthetic"
    print(f"\nSweeping Strategy 1 (DAS) on {feed_name}...")
    
    closes = df_1m["close"].values
    ts_list = df_1m["timestamp"].values
    n = len(df_1m)
    
    combos = []
    
    # Test combinations
    for std_th in [45.0, 60.0, 75.0, 90.0]:
        for vel_th in [20.0, 30.0, 45.0]:
            for win_tgt in [0.40, 0.60, 0.80]:
                for lose_sp in [0.25, 0.35]:
                    for hold_m in [45, 60, 90]:
                        trades = []
                        in_pos = False
                        ce_entry = 0.0
                        pe_entry = 0.0
                        ce_exited = False
                        pe_exited = False
                        ce_exit_p = None
                        pe_exit_p = None
                        last_exit_idx = -100

                        for i in range(30, n):
                            t = pd.Timestamp(ts_list[i])
                            tm = t.time()

                            if in_pos:
                                bars = i - entry_idx
                                if use_real:
                                    c_df = real_opts.get(ce_sym)
                                    p_df = real_opts.get(pe_sym)
                                    cur_ce = c_df.loc[t, "close"] if (c_df is not None and t in c_df.index) else ce_entry
                                    cur_pe = p_df.loc[t, "close"] if (p_df is not None and t in p_df.index) else pe_entry
                                else:
                                    spot = closes[i]
                                    T = get_dte_val(t)
                                    cur_ce = bs_price(spot, ce_k, T, "CE")
                                    cur_pe = bs_price(spot, pe_k, T, "PE")

                                if not ce_exited:
                                    r_ce = (cur_ce - ce_entry) / ce_entry
                                    if r_ce >= win_tgt:
                                        ce_exited = True
                                        ce_exit_p = cur_ce
                                    elif r_ce <= -lose_sp:
                                        ce_exited = True
                                        ce_exit_p = cur_ce

                                if not pe_exited:
                                    r_pe = (cur_pe - pe_entry) / pe_entry
                                    if r_pe >= win_tgt:
                                        pe_exited = True
                                        pe_exit_p = cur_pe
                                    elif r_pe <= -lose_sp:
                                        pe_exited = True
                                        pe_exit_p = cur_pe

                                force_close = (ce_exited and pe_exited) or (bars >= hold_m) or (tm >= datetime.time(15, 15))
                                if force_close:
                                    p_c = ce_exit_p if ce_exited else cur_ce
                                    p_p = pe_exit_p if pe_exited else cur_pe
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

                                if std_val < std_th and abs(vel_val) >= vel_th:
                                    spot = closes[i]
                                    atm_k = int(round(spot / 100.0) * 100)
                                    ce_k = atm_k + 200
                                    pe_k = atm_k - 200

                                    if use_real:
                                        s_ce = f"BANKNIFTY29SEP26{ce_k}CE"
                                        s_pe = f"BANKNIFTY29SEP26{pe_k}PE"
                                        if s_ce in real_opts and s_pe in real_opts and t in real_opts[s_ce].index and t in real_opts[s_pe].index:
                                            ce_sym = s_ce
                                            pe_sym = s_pe
                                            ce_entry = real_opts[s_ce].loc[t, "close"]
                                            pe_entry = real_opts[s_pe].loc[t, "close"]
                                            in_pos = True
                                            entry_idx = i
                                            ce_exited = False
                                            pe_exited = False
                                            ce_exit_p = None
                                            pe_exit_p = None
                                    else:
                                        T = get_dte_val(t)
                                        ce_entry = bs_price(spot, ce_k, T, "CE")
                                        pe_entry = bs_price(spot, pe_k, T, "PE")
                                        in_pos = True
                                        entry_idx = i
                                        ce_exited = False
                                        pe_exited = False
                                        ce_exit_p = None
                                        pe_exit_p = None

                        if len(trades) >= (3 if use_real else 10):
                            s = pd.Series(trades)
                            w = s[s > 0].sum()
                            l = abs(s[s <= 0].sum())
                            pf = w / l if l > 0 else 99.0
                            wr = len(s[s > 0]) / len(s) * 100.0
                            tot = s.sum()
                            combos.append({
                                "std_th": std_th, "vel_th": vel_th, "win_tgt": f"+{int(win_tgt*100)}%",
                                "lose_sp": f"-{int(lose_sp*100)}%", "hold": hold_m,
                                "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                            })

    cdf = pd.DataFrame(combos)
    if not cdf.empty:
        cdf.sort_values(by="net_pnl", ascending=False, inplace=True)
        print(f"Top 5 DAS Combinations ({feed_name}):")
        print(tabulate(cdf.head(5), headers="keys", tablefmt="grid", showindex=False))
        return cdf.iloc[0].to_dict()
    else:
        print(f"No DAS combos triggered for {feed_name}.")
        return None

# =============================================================================
# 2. SWEEP STRATEGY 2: DIRECTIONAL BOX BREAKOUT
# =============================================================================
def sweep_box(use_real=True):
    feed_name = "Real Options" if use_real else "Synthetic"
    print(f"\nSweeping Strategy 2 (Directional Box) on {feed_name}...")
    
    # Pre-extract days
    days = []
    for d, grp in df_1m.groupby("date"):
        days.append({
            "date": d,
            "times": grp["time"].values,
            "highs": grp["high"].values,
            "lows": grp["low"].values,
            "closes": grp["close"].values,
            "timestamps": grp["timestamp"].values
        })

    combos = []
    start_t = datetime.time(9, 15)
    eod_t = datetime.time(15, 15)

    for box_mins in [15, 30, 45]:
        end_h = 9 + ((15 + box_mins) // 60)
        end_m = (15 + box_mins) % 60
        box_end_t = datetime.time(end_h, end_m)

        for max_box_pct in [0.50, 0.70, 0.90]:
            for buffer_pts in [30.0, 50.0, 75.0]:
                for tp_pct in [0.50, 0.75, 1.00]:
                    for sl_pct in [0.20, 0.30]:
                        trades = []
                        for day in days:
                            times = day["times"]
                            b_mask = (times >= start_t) & (times <= box_end_t)
                            if np.sum(b_mask) < (box_mins // 2):
                                continue

                            box_h = np.max(day["highs"][b_mask])
                            box_l = np.min(day["lows"][b_mask])
                            box_range = box_h - box_l
                            spot_ref = day["closes"][b_mask][0]
                            box_pct = (box_range / spot_ref) * 100.0

                            if box_pct > max_box_pct:
                                continue

                            rem_mask = (times > box_end_t) & (times <= eod_t)
                            rem_idxs = np.where(rem_mask)[0]

                            in_pos = False
                            pos_dir = None
                            entry_p = 0.0
                            opt_sym = ""

                            for i in rem_idxs:
                                h = day["highs"][i]
                                l = day["lows"][i]
                                c = day["closes"][i]
                                tm = times[i]
                                ts = pd.Timestamp(day["timestamps"][i])

                                if in_pos:
                                    if use_real:
                                        opt_df = real_opts.get(opt_sym)
                                        cur_p = opt_df.loc[ts, "close"] if (opt_df is not None and ts in opt_df.index) else entry_p
                                    else:
                                        T = get_dte_val(ts)
                                        cur_p = bs_price(c, pos_k, T, pos_dir)

                                    ret = (cur_p - entry_p) / entry_p
                                    if ret >= tp_pct or ret <= -sl_pct or tm >= eod_t:
                                        net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                                        trades.append(net)
                                        in_pos = False
                                        break
                                else:
                                    if h >= (box_h + buffer_pts):
                                        pos_dir = "CE"
                                        pos_k = int(round(c / 100.0) * 100)
                                        if use_real:
                                            s_ce = f"BANKNIFTY29SEP26{pos_k}CE"
                                            if s_ce in real_opts and ts in real_opts[s_ce].index:
                                                opt_sym = s_ce
                                                entry_p = real_opts[s_ce].loc[ts, "close"]
                                                in_pos = True
                                        else:
                                            T = get_dte_val(ts)
                                            entry_p = bs_price(c, pos_k, T, "CE")
                                            in_pos = True

                                    elif l <= (box_l - buffer_pts):
                                        pos_dir = "PE"
                                        pos_k = int(round(c / 100.0) * 100)
                                        if use_real:
                                            s_pe = f"BANKNIFTY29SEP26{pos_k}PE"
                                            if s_pe in real_opts and ts in real_opts[s_pe].index:
                                                opt_sym = s_pe
                                                entry_p = real_opts[s_pe].loc[ts, "close"]
                                                in_pos = True
                                        else:
                                            T = get_dte_val(ts)
                                            entry_p = bs_price(c, pos_k, T, "PE")
                                            in_pos = True

                        if len(trades) >= (1 if use_real else 8):
                            s = pd.Series(trades)
                            w = s[s > 0].sum()
                            l = abs(s[s <= 0].sum())
                            pf = w / l if l > 0 else 99.0
                            wr = len(s[s > 0]) / len(s) * 100.0
                            tot = s.sum()
                            combos.append({
                                "box_m": box_mins, "max_pct": max_box_pct, "buffer": buffer_pts,
                                "tp": f"+{int(tp_pct*100)}%", "sl": f"-{int(sl_pct*100)}%",
                                "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                            })

    cdf = pd.DataFrame(combos)
    if not cdf.empty:
        cdf.sort_values(by="net_pnl", ascending=False, inplace=True)
        print(f"Top 5 Directional Box Combinations ({feed_name}):")
        print(tabulate(cdf.head(5), headers="keys", tablefmt="grid", showindex=False))
        return cdf.iloc[0].to_dict()
    else:
        print(f"No Box combos triggered for {feed_name}.")
        return None

# =============================================================================
# 3. SWEEP STRATEGY 3: GAMMA SQUEEZE MOMENTUM (5-MIN)
# =============================================================================
def sweep_gamma(use_real=True):
    feed_name = "Real Options" if use_real else "Synthetic"
    print(f"\nSweeping Strategy 3 (Gamma Squeeze) on {feed_name}...")

    # Calculate VWAP & ATR on 5m
    df_5m["typical_p"] = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
    df_5m["tp_vol"] = df_5m["typical_p"] * df_5m["volume"]
    df_5m["cum_vol"] = df_5m.groupby("date")["volume"].cumsum()
    df_5m["cum_tp_vol"] = df_5m.groupby("date")["tp_vol"].cumsum()
    df_5m["vwap"] = df_5m["cum_tp_vol"] / df_5m["cum_vol"].clip(lower=1)
    df_5m["vwap"].fillna(df_5m["close"].rolling(20).mean(), inplace=True)
    if df_5m["vwap"].isna().all():
        df_5m["vwap"] = df_5m["close"].rolling(20).mean()

    tr1 = df_5m["high"] - df_5m["low"]
    tr2 = (df_5m["high"] - df_5m["close"].shift(1)).abs()
    tr3 = (df_5m["low"] - df_5m["close"].shift(1)).abs()
    df_5m["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().fillna(60.0)
    df_5m["ema_50"] = df_5m["close"].ewm(span=50).mean()

    closes = df_5m["close"].values
    vwaps = df_5m["vwap"].values
    atrs = df_5m["atr"].values
    emas = df_5m["ema_50"].values
    times = df_5m["time"].values
    ts_list = df_5m["timestamp"].values
    n = len(df_5m)

    combos = []

    for atr_mult in [1.8, 2.2, 2.6]:
        for tp_pct in [0.60, 0.80, 1.00]:
            for sl_pct in [0.20, 0.25]:
                for trend_mode in ["none", "ema_50"]:
                    trades = []
                    in_pos = False
                    pos_dir = None
                    entry_p = 0.0
                    opt_sym = ""

                    for i in range(25, n):
                        spot = closes[i]
                        vwap = vwaps[i]
                        atr = atrs[i]
                        em = emas[i]
                        tm = times[i]
                        ts = pd.Timestamp(ts_list[i])

                        if in_pos:
                            if use_real:
                                opt_df = real_opts.get(opt_sym)
                                cur_p = opt_df.loc[ts, "close"] if (opt_df is not None and ts in opt_df.index) else entry_p
                            else:
                                T = get_dte_val(ts)
                                cur_p = bs_price(spot, pos_k, T, pos_dir)

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
                                pos_k = int(round(spot / 100.0) * 100)
                                if use_real:
                                    s_ce = f"BANKNIFTY29SEP26{pos_k}CE"
                                    if s_ce in real_opts and ts in real_opts[s_ce].index:
                                        opt_sym = s_ce
                                        entry_p = real_opts[s_ce].loc[ts, "close"]
                                        in_pos = True
                                else:
                                    T = get_dte_val(ts)
                                    entry_p = bs_price(spot, pos_k, T, "CE")
                                    in_pos = True

                            # Short Breakdown
                            elif spot < (vwap - atr_mult * atr):
                                if trend_mode == "ema_50" and spot > em:
                                    continue
                                pos_dir = "PE"
                                pos_k = int(round(spot / 100.0) * 100)
                                if use_real:
                                    s_pe = f"BANKNIFTY29SEP26{pos_k}PE"
                                    if s_pe in real_opts and ts in real_opts[s_pe].index:
                                        opt_sym = s_pe
                                        entry_p = real_opts[s_pe].loc[ts, "close"]
                                        in_pos = True
                                else:
                                    T = get_dte_val(ts)
                                    entry_p = bs_price(spot, pos_k, T, "PE")
                                    in_pos = True

                    if len(trades) >= (1 if use_real else 10):
                        s = pd.Series(trades)
                        w = s[s > 0].sum()
                        l = abs(s[s <= 0].sum())
                        pf = w / l if l > 0 else 99.0
                        wr = len(s[s > 0]) / len(s) * 100.0
                        tot = s.sum()
                        combos.append({
                            "atr_mult": atr_mult, "trend": trend_mode, "tp": f"+{int(tp_pct*100)}%",
                            "sl": f"-{int(sl_pct*100)}%", "trades": len(s), "wr": round(wr, 1),
                            "pf": round(pf, 2), "net_pnl": round(tot, 1)
                        })

    cdf = pd.DataFrame(combos)
    if not cdf.empty:
        cdf.sort_values(by="net_pnl", ascending=False, inplace=True)
        print(f"Top 5 Gamma Squeeze Combinations ({feed_name}):")
        print(tabulate(cdf.head(5), headers="keys", tablefmt="grid", showindex=False))
        return cdf.iloc[0].to_dict()
    else:
        print(f"No Gamma combos triggered for {feed_name}.")
        return None

# =============================================================================
# 4. SWEEP STRATEGY 4: HEDGED STRANGLE (COUPLED)
# =============================================================================
def sweep_hedged(use_real=True):
    feed_name = "Real Options" if use_real else "Synthetic"
    print(f"\nSweeping Strategy 4 (Hedged Strangle) on {feed_name}...")

    closes = df_1m["close"].values
    times = df_1m["time"].values
    ts_list = df_1m["timestamp"].values
    n = len(df_1m)

    combos = []

    for entry_min in [20, 35, 45]:
        entry_t_target = datetime.time(9, entry_min)
        for target_pct in [0.30, 0.45, 0.60]:
            for stop_pct in [0.15, 0.20, 0.25]:
                for hold_m in [45, 60, 90]:
                    trades = []
                    in_pos = False
                    entry_tot = 0.0
                    entry_idx = 0
                    ce_sym = ""
                    pe_sym = ""

                    for i in range(20, n):
                        tm = times[i]
                        t = pd.Timestamp(ts_list[i])
                        weekday = t.weekday()

                        if in_pos:
                            bars = i - entry_idx
                            if use_real:
                                c_df = real_opts.get(ce_sym)
                                p_df = real_opts.get(pe_sym)
                                cur_ce = c_df.loc[t, "close"] if (c_df is not None and t in c_df.index) else ce_entry
                                cur_pe = p_df.loc[t, "close"] if (p_df is not None and t in p_df.index) else pe_entry
                            else:
                                spot = closes[i]
                                T = get_dte_val(t)
                                cur_ce = bs_price(spot, ce_k, T, "CE")
                                cur_pe = bs_price(spot, pe_k, T, "PE")

                            cur_tot = cur_ce + cur_pe
                            ret = (cur_tot - entry_tot) / entry_tot

                            if ret >= target_pct or ret <= -stop_pct or bars >= hold_m or tm >= datetime.time(15, 15):
                                net = (cur_tot - entry_tot) * LOT_SIZE - FEE_STRANGLE
                                trades.append(net)
                                in_pos = False
                        else:
                            if tm != entry_t_target:
                                continue
                            if weekday not in [1, 2]:  # Tuesday / Wednesday
                                continue

                            spot = closes[i]
                            atm_k = int(round(spot / 100.0) * 100)
                            ce_k = atm_k + 300
                            pe_k = atm_k - 300

                            if use_real:
                                s_ce = f"BANKNIFTY29SEP26{ce_k}CE"
                                s_pe = f"BANKNIFTY29SEP26{pe_k}PE"
                                if s_ce in real_opts and s_pe in real_opts and t in real_opts[s_ce].index and t in real_opts[s_pe].index:
                                    ce_sym = s_ce
                                    pe_sym = s_pe
                                    ce_entry = real_opts[s_ce].loc[t, "close"]
                                    pe_entry = real_opts[s_pe].loc[t, "close"]
                                    entry_tot = ce_entry + pe_entry
                                    in_pos = True
                                    entry_idx = i
                            else:
                                T = get_dte_val(t)
                                ce_entry = bs_price(spot, ce_k, T, "CE")
                                pe_entry = bs_price(spot, pe_k, T, "PE")
                                entry_tot = ce_entry + pe_entry
                                in_pos = True
                                entry_idx = i

                    if len(trades) >= (1 if use_real else 5):
                        s = pd.Series(trades)
                        w = s[s > 0].sum()
                        l = abs(s[s <= 0].sum())
                        pf = w / l if l > 0 else 99.0
                        wr = len(s[s > 0]) / len(s) * 100.0
                        tot = s.sum()
                        combos.append({
                            "entry_t": str(entry_t_target)[:5], "target": f"+{int(target_pct*100)}%",
                            "stop": f"-{int(stop_pct*100)}%", "hold": hold_m, "trades": len(s),
                            "wr": round(wr, 1), "pf": round(pf, 2), "net_pnl": round(tot, 1)
                        })

    cdf = pd.DataFrame(combos)
    if not cdf.empty:
        cdf.sort_values(by="net_pnl", ascending=False, inplace=True)
        print(f"Top 5 Hedged Strangle Combinations ({feed_name}):")
        print(tabulate(cdf.head(5), headers="keys", tablefmt="grid", showindex=False))
        return cdf.iloc[0].to_dict()
    else:
        print(f"No Hedged Strangle combos triggered for {feed_name}.")
        return None

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("        COMPREHENSIVE BANKNIFTY PARAMETRIC GRID SWEEP (REAL & SYNTHETIC)        ")
    print("=" * 90)

    # Real Options Sweep
    best_das_real = sweep_das(use_real=True)
    best_box_real = sweep_box(use_real=True)
    best_gamma_real = sweep_gamma(use_real=True)
    best_hedged_real = sweep_hedged(use_real=True)

    # Synthetic Sweep
    best_das_synth = sweep_das(use_real=False)
    best_box_synth = sweep_box(use_real=False)
    best_gamma_synth = sweep_gamma(use_real=False)
    best_hedged_synth = sweep_hedged(use_real=False)

    print("\n" + "=" * 90)
    print("                         SWEEP COMPLETED SUCCESSFULLY                          ")
    print("=" * 90)
