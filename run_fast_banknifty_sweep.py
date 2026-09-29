"""
Ultra-Fast BankNIFTY Parametric Search Engine:
Pre-aligns all Real Option Contracts and Synthetic Option Prices into contiguous 1D Numpy Arrays.
Performs 10,000+ simulation runs across all 4 strategies in seconds!
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

print("[1/4] Loading spot datasets...")
df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.sort_values("timestamp", inplace=True)
df_1m.reset_index(drop=True, inplace=True)
df_1m.set_index("timestamp", inplace=True)

df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)
df_5m.set_index("timestamp", inplace=True)

# 1-min helper arrays
times_1m = df_1m.index.time
dates_1m = df_1m.index.date
closes_1m = df_1m["close"].values
highs_1m = df_1m["high"].values
lows_1m = df_1m["low"].values
n_1m = len(df_1m)

# 5-min helper arrays
times_5m = df_5m.index.time
dates_5m = df_5m.index.date
closes_5m = df_5m["close"].values
highs_5m = df_5m["high"].values
lows_5m = df_5m["low"].values
n_5m = len(df_5m)

print("[2/4] Vectorizing and precomputing Synthetic Option Grids...")
def bs_vec(S, K, T, opt_type="CE", r=0.07, sigma=0.165):
    T_c = np.maximum(1e-4, T)
    sqrt_T = np.sqrt(T_c)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_c) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm.cdf(d1) - K * np.exp(-r * T_c) * norm.cdf(d2)
    else:
        p = K * np.exp(-r * T_c) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.maximum(0.05, p)

def calc_dtes(idx):
    dtes = []
    for ts in idx:
        weekday = ts.weekday()
        target_exp = 2
        days_ahead = target_exp - weekday if weekday <= target_exp else 7 - (weekday - target_exp)
        mins = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
        frac = mins / 375.0
        d = max(0.01, frac * 0.5) if days_ahead == 0 else (days_ahead - 1) + frac
        dtes.append(d / 365.0)
    return np.array(dtes)

dtes_1m = calc_dtes(df_1m.index)
atm_k_1m = np.round(closes_1m / 100.0) * 100.0

synth_ce_atm_1m = bs_vec(closes_1m, atm_k_1m, dtes_1m, "CE")
synth_pe_atm_1m = bs_vec(closes_1m, atm_k_1m, dtes_1m, "PE")
synth_ce_otm1_1m = bs_vec(closes_1m, atm_k_1m + 100, dtes_1m, "CE")
synth_pe_otm1_1m = bs_vec(closes_1m, atm_k_1m - 100, dtes_1m, "PE")
synth_ce_otm2_1m = bs_vec(closes_1m, atm_k_1m + 200, dtes_1m, "CE")
synth_pe_otm2_1m = bs_vec(closes_1m, atm_k_1m - 200, dtes_1m, "PE")

dtes_5m = calc_dtes(df_5m.index)
atm_k_5m = np.round(closes_5m / 100.0) * 100.0
synth_ce_atm_5m = bs_vec(closes_5m, atm_k_5m, dtes_5m, "CE")
synth_pe_atm_5m = bs_vec(closes_5m, atm_k_5m, dtes_5m, "PE")

# 5-min VWAP & ATR
df_5m["typical_p"] = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
df_5m["tp_vol"] = df_5m["typical_p"] * df_5m["volume"]
df_5m["cum_vol"] = df_5m.groupby(df_5m.index.date)["volume"].cumsum()
df_5m["cum_tp_vol"] = df_5m.groupby(df_5m.index.date)["tp_vol"].cumsum()
vwap_5m = (df_5m["cum_tp_vol"] / df_5m["cum_vol"].clip(lower=1)).fillna(df_5m["close"].rolling(20).mean()).values

tr1 = df_5m["high"] - df_5m["low"]
tr2 = (df_5m["high"] - df_5m["close"].shift(1)).abs()
tr3 = (df_5m["low"] - df_5m["close"].shift(1)).abs()
atr_5m = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().fillna(60.0).values
ema50_5m = df_5m["close"].ewm(span=50).mean().values

print("[3/4] Aligning Real Option Contracts into 1D Numpy Arrays...")
real_opt_arrays_1m = {}
real_opt_arrays_5m = {}
for p in glob.glob(f"{CACHE_DIR}/*.csv"):
    sym = os.path.basename(p).replace(".csv", "")
    dfo = pd.read_csv(p)
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"]).dt.tz_localize(None)
    dfo.sort_values("timestamp", inplace=True)
    dfo.drop_duplicates(subset=["timestamp"], inplace=True)
    dfo.set_index("timestamp", inplace=True)
    dfo["close"] = pd.to_numeric(dfo["close"], errors="coerce")
    
    # Reindex to 1-min
    aligned_1m = dfo["close"].reindex(df_1m.index).ffill().values
    real_opt_arrays_1m[sym] = aligned_1m

    # Reindex to 5-min
    aligned_5m = dfo["close"].reindex(df_5m.index).ffill().values
    real_opt_arrays_5m[sym] = aligned_5m

print(f"Aligned {len(real_opt_arrays_1m)} real option contracts. Ready for instant simulation!\n")

# =============================================================================
# STRATEGY 1: DECOUPLED ASYMMETRIC STRANGLE (DAS) FAST SIMULATOR
# =============================================================================
def run_das_fast(std_th, vel_th, win_tgt, lose_sp, hold_m, use_real=False):
    trades = []
    in_pos = False
    ce_entry = 0.0
    pe_entry = 0.0
    ce_exited = False
    pe_exited = False
    ce_exit_p = 0.0
    pe_exit_p = 0.0
    entry_idx = 0
    last_exit_idx = -100

    ce_active_arr = None
    pe_active_arr = None

    for i in range(30, n_1m):
        tm = times_1m[i]

        if in_pos:
            bars = i - entry_idx
            c_ce = ce_active_arr[i]
            c_pe = pe_active_arr[i]

            if not ce_exited:
                r_ce = (c_ce - ce_entry) / max(0.1, ce_entry)
                if r_ce >= win_tgt:
                    ce_exited = True
                    ce_exit_p = c_ce
                elif r_ce <= -lose_sp:
                    ce_exited = True
                    ce_exit_p = c_ce

            if not pe_exited:
                r_pe = (c_pe - pe_entry) / max(0.1, pe_entry)
                if r_pe >= win_tgt:
                    pe_exited = True
                    pe_exit_p = c_pe
                elif r_pe <= -lose_sp:
                    pe_exited = True
                    pe_exit_p = c_pe

            if (ce_exited and pe_exited) or (bars >= hold_m) or (tm >= datetime.time(15, 15)):
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                gross = (p_c - ce_entry + p_p - pe_entry) * LOT_SIZE
                net = gross - FEE_STRANGLE
                cap = (ce_entry + pe_entry) * LOT_SIZE
                trades.append({"net": net, "cap": cap})
                in_pos = False
                last_exit_idx = i

        else:
            if tm < datetime.time(9, 45) or tm > datetime.time(13, 30):
                continue
            if datetime.time(11, 30) <= tm <= datetime.time(12, 45):
                continue
            if (i - last_exit_idx) < 30:
                continue

            sub_c = closes_1m[i-20:i+1]
            std_val = np.std(sub_c)
            vel_val = sub_c[-1] - sub_c[-3]

            if std_val < std_th and abs(vel_val) >= vel_th:
                spot = closes_1m[i]
                atm_k = int(round(spot / 100.0) * 100)
                ce_k = atm_k + 200
                pe_k = atm_k - 200

                if use_real:
                    s_ce = f"BANKNIFTY29SEP26{ce_k}CE"
                    s_pe = f"BANKNIFTY29SEP26{pe_k}PE"
                    if s_ce in real_opt_arrays_1m and s_pe in real_opt_arrays_1m:
                        arr_c = real_opt_arrays_1m[s_ce]
                        arr_p = real_opt_arrays_1m[s_pe]
                        if not np.isnan(arr_c[i]) and not np.isnan(arr_p[i]) and arr_c[i] > 10 and arr_p[i] > 10:
                            ce_active_arr = arr_c
                            pe_active_arr = arr_p
                            ce_entry = arr_c[i]
                            pe_entry = arr_p[i]
                            in_pos = True
                            entry_idx = i
                            ce_exited = False
                            pe_exited = False
                else:
                    ce_active_arr = synth_ce_otm2_1m
                    pe_active_arr = synth_pe_otm2_1m
                    ce_entry = ce_active_arr[i]
                    pe_entry = pe_active_arr[i]
                    in_pos = True
                    entry_idx = i
                    ce_exited = False
                    pe_exited = False

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    w = tdf[tdf["net"] > 0]["net"].sum()
    l = abs(tdf[tdf["net"] <= 0]["net"].sum())
    pf = w / l if l > 0 else 99.0
    wr = len(tdf[tdf["net"] > 0]) / len(tdf) * 100.0
    return {
        "trades": len(tdf), "wr": round(wr, 1), "pf": round(pf, 2),
        "net": round(tdf["net"].sum(), 1), "avg_cap": round(tdf["cap"].mean(), 1)
    }

# =============================================================================
# STRATEGY 2: DIRECTIONAL BOX BREAKOUT FAST SIMULATOR
# =============================================================================
def run_box_fast(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, use_real=False):
    trades = []
    
    start_t = datetime.time(9, 15)
    end_h = 9 + ((15 + box_mins) // 60)
    end_m = (15 + box_mins) % 60
    box_end_t = datetime.time(end_h, end_m)
    eod_t = datetime.time(15, 15)

    # Group by date
    unique_dates = np.unique(dates_1m)
    for d in unique_dates:
        day_mask = (dates_1m == d)
        day_idxs = np.where(day_mask)[0]
        day_times = times_1m[day_idxs]

        b_mask = (day_times >= start_t) & (day_times <= box_end_t)
        if np.sum(b_mask) < (box_mins // 2):
            continue

        b_idxs = day_idxs[b_mask]
        box_h = np.max(highs_1m[b_idxs])
        box_l = np.min(lows_1m[b_idxs])
        spot_ref = closes_1m[b_idxs[0]]
        box_pct = (box_h - box_l) / spot_ref * 100.0
        if box_pct > max_box_pct:
            continue

        rem_mask = (day_times > box_end_t) & (day_times <= eod_t)
        rem_idxs = day_idxs[rem_mask]

        in_pos = False
        entry_p = 0.0
        opt_arr = None

        for idx in rem_idxs:
            h = highs_1m[idx]
            l = lows_1m[idx]
            tm = times_1m[idx]
            spot = closes_1m[idx]

            if in_pos:
                cur_p = opt_arr[idx]
                ret = (cur_p - entry_p) / entry_p
                if ret >= tp_pct or ret <= -sl_pct or tm >= eod_t:
                    net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                    trades.append({"net": net, "cap": entry_p * LOT_SIZE})
                    in_pos = False
                    break
            else:
                if h >= (box_h + buffer_pts):
                    atm_k = int(round(spot / 100.0) * 100)
                    if use_real:
                        s = f"BANKNIFTY29SEP26{atm_k}CE"
                        if s in real_opt_arrays_1m:
                            arr = real_opt_arrays_1m[s]
                            if not np.isnan(arr[idx]) and arr[idx] > 10:
                                opt_arr = arr
                                entry_p = arr[idx]
                                in_pos = True
                    else:
                        opt_arr = synth_ce_atm_1m
                        entry_p = opt_arr[idx]
                        in_pos = True

                elif l <= (box_l - buffer_pts):
                    atm_k = int(round(spot / 100.0) * 100)
                    if use_real:
                        s = f"BANKNIFTY29SEP26{atm_k}PE"
                        if s in real_opt_arrays_1m:
                            arr = real_opt_arrays_1m[s]
                            if not np.isnan(arr[idx]) and arr[idx] > 10:
                                opt_arr = arr
                                entry_p = arr[idx]
                                in_pos = True
                    else:
                        opt_arr = synth_pe_atm_1m
                        entry_p = opt_arr[idx]
                        in_pos = True

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    w = tdf[tdf["net"] > 0]["net"].sum()
    l = abs(tdf[tdf["net"] <= 0]["net"].sum())
    pf = w / l if l > 0 else 99.0
    wr = len(tdf[tdf["net"] > 0]) / len(tdf) * 100.0
    return {
        "trades": len(tdf), "wr": round(wr, 1), "pf": round(pf, 2),
        "net": round(tdf["net"].sum(), 1), "avg_cap": round(tdf["cap"].mean(), 1)
    }

# =============================================================================
# STRATEGY 3: GAMMA SQUEEZE FAST SIMULATOR (5-MIN)
# =============================================================================
def run_gamma_fast(atr_mult, tp_pct, sl_pct, use_trend=True, use_real=False):
    trades = []
    in_pos = False
    entry_p = 0.0
    opt_arr = None
    pos_dir = ""

    for i in range(25, n_5m):
        spot = closes_5m[i]
        vwap = vwap_5m[i]
        atr = atr_5m[i]
        em = ema50_5m[i]
        tm = times_5m[i]

        if in_pos:
            cur_p = opt_arr[i]
            ret = (cur_p - entry_p) / entry_p
            if ret >= tp_pct or ret <= -sl_pct or (pos_dir == "CE" and spot < vwap) or (pos_dir == "PE" and spot > vwap) or tm >= datetime.time(15, 15):
                net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                trades.append({"net": net, "cap": entry_p * LOT_SIZE})
                in_pos = False
        else:
            if tm < datetime.time(9, 30) or tm > datetime.time(14, 15):
                continue

            if spot > (vwap + atr_mult * atr):
                if use_trend and spot < em:
                    continue
                atm_k = int(round(spot / 100.0) * 100)
                if use_real:
                    s = f"BANKNIFTY29SEP26{atm_k}CE"
                    if s in real_opt_arrays_5m:
                        arr = real_opt_arrays_5m[s]
                        if not np.isnan(arr[i]) and arr[i] > 10:
                            opt_arr = arr
                            entry_p = arr[i]
                            pos_dir = "CE"
                            in_pos = True
                else:
                    opt_arr = synth_ce_atm_5m
                    entry_p = opt_arr[i]
                    pos_dir = "CE"
                    in_pos = True

            elif spot < (vwap - atr_mult * atr):
                if use_trend and spot > em:
                    continue
                atm_k = int(round(spot / 100.0) * 100)
                if use_real:
                    s = f"BANKNIFTY29SEP26{atm_k}PE"
                    if s in real_opt_arrays_5m:
                        arr = real_opt_arrays_5m[s]
                        if not np.isnan(arr[i]) and arr[i] > 10:
                            opt_arr = arr
                            entry_p = arr[i]
                            pos_dir = "PE"
                            in_pos = True
                else:
                    opt_arr = synth_pe_atm_5m
                    entry_p = opt_arr[i]
                    pos_dir = "PE"
                    in_pos = True

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    w = tdf[tdf["net"] > 0]["net"].sum()
    l = abs(tdf[tdf["net"] <= 0]["net"].sum())
    pf = w / l if l > 0 else 99.0
    wr = len(tdf[tdf["net"] > 0]) / len(tdf) * 100.0
    return {
        "trades": len(tdf), "wr": round(wr, 1), "pf": round(pf, 2),
        "net": round(tdf["net"].sum(), 1), "avg_cap": round(tdf["cap"].mean(), 1)
    }

# =============================================================================
# STRATEGY 4: HEDGED STRANGLE FAST SIMULATOR
# =============================================================================
def run_hedged_fast(target_pct, stop_pct, hold_m, use_real=False):
    trades = []
    in_pos = False
    entry_tot = 0.0
    entry_idx = 0
    ce_arr = None
    pe_arr = None

    for i in range(20, n_1m):
        tm = times_1m[i]
        t = df_1m.index[i]
        weekday = t.weekday()

        if in_pos:
            bars = i - entry_idx
            cur_tot = ce_arr[i] + pe_arr[i]
            ret = (cur_tot - entry_tot) / entry_tot

            if ret >= target_pct or ret <= -stop_pct or bars >= hold_m or tm >= datetime.time(15, 15):
                net = (cur_tot - entry_tot) * LOT_SIZE - FEE_STRANGLE
                trades.append({"net": net, "cap": entry_tot * LOT_SIZE})
                in_pos = False
        else:
            if tm != datetime.time(9, 35):
                continue
            if weekday not in [1, 2]:  # Tuesday / Wednesday only
                continue

            spot = closes_1m[i]
            atm_k = int(round(spot / 100.0) * 100)
            ce_k = atm_k + 300
            pe_k = atm_k - 300

            if use_real:
                s_ce = f"BANKNIFTY29SEP26{ce_k}CE"
                s_pe = f"BANKNIFTY29SEP26{pe_k}PE"
                if s_ce in real_opt_arrays_1m and s_pe in real_opt_arrays_1m:
                    arr_c = real_opt_arrays_1m[s_ce]
                    arr_p = real_opt_arrays_1m[s_pe]
                    if not np.isnan(arr_c[i]) and not np.isnan(arr_p[i]) and arr_c[i] > 10 and arr_p[i] > 10:
                        ce_arr = arr_c
                        pe_arr = arr_p
                        entry_tot = arr_c[i] + arr_p[i]
                        in_pos = True
                        entry_idx = i
            else:
                ce_arr = synth_ce_otm2_1m
                pe_arr = synth_pe_otm2_1m
                entry_tot = ce_arr[i] + pe_arr[i]
                in_pos = True
                entry_idx = i

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    w = tdf[tdf["net"] > 0]["net"].sum()
    l = abs(tdf[tdf["net"] <= 0]["net"].sum())
    pf = w / l if l > 0 else 99.0
    wr = len(tdf[tdf["net"] > 0]) / len(tdf) * 100.0
    return {
        "trades": len(tdf), "wr": round(wr, 1), "pf": round(pf, 2),
        "net": round(tdf["net"].sum(), 1), "avg_cap": round(tdf["cap"].mean(), 1)
    }

# =============================================================================
# RUN THE MULTI-PARAMETRIC SWEEP
# =============================================================================
if __name__ == "__main__":
    print("[4/4] Sweeping parameter combinations across ALL 4 Strategies...\n")

    # 1. Sweep DAS
    best_das = []
    for std_th in [45.0, 60.0, 75.0, 90.0]:
        for vel_th in [20.0, 35.0, 50.0]:
            for win_tgt in [0.40, 0.60, 0.80]:
                for lose_sp in [0.25, 0.35]:
                    for hold_m in [45, 60, 90]:
                        m_real = run_das_fast(std_th, vel_th, win_tgt, lose_sp, hold_m, use_real=True)
                        m_synth = run_das_fast(std_th, vel_th, win_tgt, lose_sp, hold_m, use_real=False)
                        if m_real and m_real["trades"] >= 2:
                            best_das.append({
                                "std_th": std_th, "vel_th": vel_th, "tp": f"+{int(win_tgt*100)}%", "sl": f"-{int(lose_sp*100)}%", "hold": f"{hold_m}m",
                                "r_trades": m_real["trades"], "r_wr": f"{m_real['wr']}%", "r_pf": m_real["pf"], "r_pnl": m_real["net"],
                                "s_trades": m_synth["trades"] if m_synth else 0, "s_wr": f"{m_synth['wr']}%" if m_synth else "0%", "s_pnl": m_synth["net"] if m_synth else 0
                            })

    das_df = pd.DataFrame(best_das)
    if not das_df.empty:
        das_df.sort_values(by="r_pnl", ascending=False, inplace=True)
        print("=" * 105)
        print("  TOP DECOUPLED ASYMMETRIC STRANGLE (DAS) CONFIGURATIONS ON BANKNIFTY")
        print("=" * 105)
        print(tabulate(das_df.head(6), headers="keys", tablefmt="grid", showindex=False))

    # 2. Sweep Directional Box
    best_box = []
    for box_mins in [15, 30, 45]:
        for max_box_pct in [0.45, 0.60, 0.75]:
            for buffer_pts in [30.0, 50.0, 75.0]:
                for tp_pct in [0.40, 0.60, 0.80]:
                    for sl_pct in [0.20, 0.25]:
                        m_real = run_box_fast(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, use_real=True)
                        m_synth = run_box_fast(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, use_real=False)
                        if m_real or m_synth:
                            best_box.append({
                                "box_m": f"{box_mins}m", "max_pct": f"{max_box_pct}%", "buf": f"{int(buffer_pts)}pt",
                                "tp": f"+{int(tp_pct*100)}%", "sl": f"-{int(sl_pct*100)}%",
                                "r_trades": m_real["trades"] if m_real else 0, "r_pnl": m_real["net"] if m_real else 0,
                                "s_trades": m_synth["trades"] if m_synth else 0, "s_wr": f"{m_synth['wr']}%" if m_synth else "0%",
                                "s_pf": m_synth["pf"] if m_synth else 0, "s_pnl": m_synth["net"] if m_synth else 0
                            })

    box_df = pd.DataFrame(best_box)
    if not box_df.empty:
        box_df.sort_values(by="s_pnl", ascending=False, inplace=True)
        print("\n" + "=" * 105)
        print("  TOP DIRECTIONAL BOX BREAKOUT CONFIGURATIONS ON BANKNIFTY")
        print("=" * 105)
        print(tabulate(box_df.head(6), headers="keys", tablefmt="grid", showindex=False))

    # 3. Sweep Gamma Squeeze
    best_gamma = []
    for atr_mult in [1.5, 1.8, 2.2, 2.5]:
        for tp_pct in [0.50, 0.75, 1.00]:
            for sl_pct in [0.20, 0.25]:
                for trend in [True, False]:
                    m_real = run_gamma_fast(atr_mult, tp_pct, sl_pct, use_trend=trend, use_real=True)
                    m_synth = run_gamma_fast(atr_mult, tp_pct, sl_pct, use_trend=trend, use_real=False)
                    if m_real or m_synth:
                        best_gamma.append({
                            "atr_mult": f"{atr_mult}x", "trend": "EMA50" if trend else "None",
                            "tp": f"+{int(tp_pct*100)}%", "sl": f"-{int(sl_pct*100)}%",
                            "r_trades": m_real["trades"] if m_real else 0, "r_pnl": m_real["net"] if m_real else 0,
                            "s_trades": m_synth["trades"] if m_synth else 0, "s_wr": f"{m_synth['wr']}%" if m_synth else "0%",
                            "s_pf": m_synth["pf"] if m_synth else 0, "s_pnl": m_synth["net"] if m_synth else 0
                        })

    gamma_df = pd.DataFrame(best_gamma)
    if not gamma_df.empty:
        gamma_df.sort_values(by="s_pnl", ascending=False, inplace=True)
        print("\n" + "=" * 105)
        print("  TOP GAMMA SQUEEZE MOMENTUM CONFIGURATIONS ON BANKNIFTY (5-MIN)")
        print("=" * 105)
        print(tabulate(gamma_df.head(6), headers="keys", tablefmt="grid", showindex=False))

    # 4. Sweep Hedged Strangle
    best_hedged = []
    for tp_pct in [0.30, 0.45, 0.60]:
        for sl_pct in [0.15, 0.20, 0.25]:
            for hold_m in [45, 60, 90]:
                m_real = run_hedged_fast(tp_pct, sl_pct, hold_m, use_real=True)
                m_synth = run_hedged_fast(tp_pct, sl_pct, hold_m, use_real=False)
                if m_real or m_synth:
                    best_hedged.append({
                        "tp": f"+{int(tp_pct*100)}%", "sl": f"-{int(sl_pct*100)}%", "hold": f"{hold_m}m",
                        "r_trades": m_real["trades"] if m_real else 0, "r_pnl": m_real["net"] if m_real else 0,
                        "s_trades": m_synth["trades"] if m_synth else 0, "s_wr": f"{m_synth['wr']}%" if m_synth else "0%",
                        "s_pf": m_synth["pf"] if m_synth else 0, "s_pnl": m_synth["net"] if m_synth else 0
                    })

    hedged_df = pd.DataFrame(best_hedged)
    if not hedged_df.empty:
        hedged_df.sort_values(by="s_pnl", ascending=False, inplace=True)
        print("\n" + "=" * 105)
        print("  TOP HEDGED STRANGLE CONFIGURATIONS ON BANKNIFTY")
        print("=" * 105)
        print(tabulate(hedged_df.head(6), headers="keys", tablefmt="grid", showindex=False))

    print("\n[DONE] Full parametric optimization completed!")
