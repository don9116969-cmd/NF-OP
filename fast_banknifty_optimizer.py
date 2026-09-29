"""
High-Speed Vectorized Multi-Strategy Parametric Optimizer for BankNIFTY.
Precomputes all indicators and option pricing arrays to run tens of thousands of combinations in seconds.
Produces exact comparative metrics for:
1. Real SmartAPI Traded Options (Angel One Expiry Week Sept 21-28)
2. Synthetic Black-Scholes (Same Expiry Week Sept 21-28)
3. Synthetic Black-Scholes (Full Dataset 8,664 bars)
"""

import os
import glob
import datetime
import itertools
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_DIRECTIONAL = 45.0
CACHE_DIR = "data/banknifty_options_cache"

print("=" * 85)
print("  HIGH-SPEED BANKNIFTY PARAMETRIC OPTIMIZER (REAL & SYNTHETIC)")
print("=" * 85)

# ---------------------------------------------------------
# 1. LOAD AND PREPARE SPOT DATA
# ---------------------------------------------------------
print("\n[1/5] Loading 1-min and 5-min BankNIFTY spot bars...")
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

# 1-min arrays
n_1m = len(df_1m)
times_1m = df_1m.index.time
dates_1m = df_1m.index.date
closes_1m = df_1m["close"].values
highs_1m = df_1m["high"].values
lows_1m = df_1m["low"].values
mins_of_day_1m = np.array([t.hour * 60 + t.minute for t in times_1m])

# Precompute 1-min rolling indicators
rolling_std_20_1m = pd.Series(closes_1m).rolling(20).std().fillna(50.0).values
vel_3_1m = closes_1m - pd.Series(closes_1m).shift(3).fillna(0.0).values

# 5-min arrays
n_5m = len(df_5m)
times_5m = df_5m.index.time
dates_5m = df_5m.index.date
closes_5m = df_5m["close"].values
highs_5m = df_5m["high"].values
lows_5m = df_5m["low"].values
vols_5m = df_5m["volume"].values
mins_of_day_5m = np.array([t.hour * 60 + t.minute for t in times_5m])

# 5-min indicators
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
vol_sma20_5m = pd.Series(vols_5m).rolling(20).mean().fillna(1.0).values

print(f"Loaded {n_1m} 1-min bars and {n_5m} 5-min bars.")

# ---------------------------------------------------------
# 2. LOAD REAL OPTIONS
# ---------------------------------------------------------
print("\n[2/5] Loading Real SmartAPI Option Contracts...")
opt_files = glob.glob(f"{CACHE_DIR}/*.csv")
strikes_ce = sorted(list(set([int(os.path.basename(f).replace("BANKNIFTY29SEP26", "").replace("CE.csv", "")) for f in opt_files if "CE.csv" in f])))
strikes_pe = sorted(list(set([int(os.path.basename(f).replace("BANKNIFTY29SEP26", "").replace("PE.csv", "")) for f in opt_files if "PE.csv" in f])))
strikes_ce_arr = np.array(strikes_ce)
strikes_pe_arr = np.array(strikes_pe)

real_opt_1m = {}
real_opt_5m = {}
real_opt_start_date = datetime.date(2026, 9, 21)
real_opt_end_date = datetime.date(2026, 9, 28)

for p in opt_files:
    sym = os.path.basename(p).replace(".csv", "")
    dfo = pd.read_csv(p)
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"]).dt.tz_localize(None)
    dfo.sort_values("timestamp", inplace=True)
    dfo.drop_duplicates(subset=["timestamp"], inplace=True)
    dfo.set_index("timestamp", inplace=True)
    dfo["close"] = pd.to_numeric(dfo["close"], errors="coerce")

    real_opt_1m[sym] = dfo["close"].reindex(df_1m.index).ffill().values
    real_opt_5m[sym] = dfo["close"].reindex(df_5m.index).ffill().values

def get_real_opt_array(spot, offset, opt_type, tf="1m"):
    target = spot + offset
    arr = strikes_ce_arr if opt_type == "CE" else strikes_pe_arr
    k = arr[np.argmin(np.abs(arr - target))]
    sym = f"BANKNIFTY29SEP26{k}{opt_type}"
    d = real_opt_1m if tf == "1m" else real_opt_5m
    return d.get(sym, None)

# ---------------------------------------------------------
# 3. VECTORIZED SYNTHETIC OPTIONS
# ---------------------------------------------------------
print("\n[3/5] Vectorizing Black-Scholes Synthetic Option Grids...")
def bs_vec(S, K, T, opt_type="CE", r=0.07, sigma=0.17):
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
        target_exp = 2  # Wednesday expiry
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
synth_ce_otm1_1m = bs_vec(closes_1m, atm_k_1m + 200, dtes_1m, "CE")
synth_pe_otm1_1m = bs_vec(closes_1m, atm_k_1m - 200, dtes_1m, "PE")
synth_ce_otm2_1m = bs_vec(closes_1m, atm_k_1m + 400, dtes_1m, "CE")
synth_pe_otm2_1m = bs_vec(closes_1m, atm_k_1m - 400, dtes_1m, "PE")

dtes_5m = calc_dtes(df_5m.index)
atm_k_5m = np.round(closes_5m / 100.0) * 100.0
synth_ce_atm_5m = bs_vec(closes_5m, atm_k_5m, dtes_5m, "CE")
synth_pe_atm_5m = bs_vec(closes_5m, atm_k_5m, dtes_5m, "PE")

# Metrics evaluator
def calc_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "max_dd": 0.0, "avg_trade": 0.0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net"] > 0]["net"]
    losses = tdf[tdf["net"] <= 0]["net"]
    w_sum = wins.sum()
    l_sum = abs(losses.sum())
    pf = round(w_sum / l_sum, 2) if l_sum > 0 else (99.0 if w_sum > 0 else 0.0)
    wr = round(len(wins) / len(tdf) * 100.0, 1)
    tot_pnl = round(tdf["net"].sum(), 1)
    avg_t = round(tot_pnl / len(tdf), 1)

    cum = tdf["net"].cumsum()
    peak = cum.cummax()
    dd = (peak - cum).max()
    max_dd = round(dd, 1) if not np.isnan(dd) else 0.0
    return {"trades": len(tdf), "wr": wr, "pf": pf, "pnl": tot_pnl, "max_dd": max_dd, "avg_trade": avg_t}

print("Precomputation finished in < 1 second.\n")

# ---------------------------------------------------------
# SIMULATION ENGINE 1: DAS
# ---------------------------------------------------------
def run_das(std_th, vel_th, win_tgt, lose_sp, hold_m, offset=200, mode="real"):
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

    ce_arr = None
    pe_arr = None

    for i in range(30, n_1m):
        tm_min = mins_of_day_1m[i]
        d = dates_1m[i]

        if mode in ["real", "synth_week"]:
            if d < real_opt_start_date or d > real_opt_end_date:
                continue

        if in_pos:
            bars = i - entry_idx
            c_ce = ce_arr[i]
            c_pe = pe_arr[i]

            if not ce_exited:
                r_ce = (c_ce - ce_entry) / max(0.1, ce_entry)
                if r_ce >= win_tgt or r_ce <= -lose_sp:
                    ce_exited = True
                    ce_exit_p = c_ce

            if not pe_exited:
                r_pe = (c_pe - pe_entry) / max(0.1, pe_entry)
                if r_pe >= win_tgt or r_pe <= -lose_sp:
                    pe_exited = True
                    pe_exit_p = c_pe

            if (ce_exited and pe_exited) or (bars >= hold_m) or (tm_min >= 915): # 15:15
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                gross = (p_c - ce_entry + p_p - pe_entry) * LOT_SIZE
                net = gross - FEE_STRANGLE
                trades.append({"net": net})
                in_pos = False
                last_exit_idx = i

        else:
            # 09:45 (585) to 13:30 (810), skip 11:30 (690) to 12:45 (765)
            if tm_min < 585 or tm_min > 810 or (690 <= tm_min <= 765):
                continue
            if (i - last_exit_idx) < 30:
                continue

            if rolling_std_20_1m[i] < std_th and abs(vel_3_1m[i]) >= vel_th:
                spot = closes_1m[i]
                if mode == "real":
                    arr_c = get_real_opt_array(spot, offset, "CE", "1m")
                    arr_p = get_real_opt_array(spot, -offset, "PE", "1m")
                    if arr_c is not None and arr_p is not None:
                        if not np.isnan(arr_c[i]) and not np.isnan(arr_p[i]) and arr_c[i] > 5 and arr_p[i] > 5:
                            ce_arr = arr_c
                            pe_arr = arr_p
                            ce_entry = arr_c[i]
                            pe_entry = arr_p[i]
                            in_pos = True
                            entry_idx = i
                            ce_exited = False
                            pe_exited = False
                else:
                    if offset == 0:
                        ce_arr, pe_arr = synth_ce_atm_1m, synth_pe_atm_1m
                    elif offset == 200:
                        ce_arr, pe_arr = synth_ce_otm1_1m, synth_pe_otm1_1m
                    else:
                        ce_arr, pe_arr = synth_ce_otm2_1m, synth_pe_otm2_1m
                    ce_entry = ce_arr[i]
                    pe_entry = pe_arr[i]
                    in_pos = True
                    entry_idx = i
                    ce_exited = False
                    pe_exited = False

    return calc_metrics(trades)

# ---------------------------------------------------------
# SIMULATION ENGINE 2: DIRECTIONAL BOX BREAKOUT
# ---------------------------------------------------------
def run_box(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, spot_sl=False, mode="real"):
    trades = []
    box_end_min = 555 + box_mins  # 09:15 = 555 min
    eod_min = 915                # 15:15 = 915 min

    unique_dates = np.unique(dates_1m)
    for d in unique_dates:
        if mode in ["real", "synth_week"]:
            if d < real_opt_start_date or d > real_opt_end_date:
                continue

        day_mask = (dates_1m == d)
        day_idxs = np.where(day_mask)[0]
        day_mins = mins_of_day_1m[day_idxs]

        b_mask = (day_mins >= 555) & (day_mins <= box_end_min)
        if np.sum(b_mask) < (box_mins // 2):
            continue

        b_idxs = day_idxs[b_mask]
        box_h = np.max(highs_1m[b_idxs])
        box_l = np.min(lows_1m[b_idxs])
        box_pct = (box_h - box_l) / closes_1m[b_idxs[0]] * 100.0
        if box_pct > max_box_pct:
            continue

        rem_mask = (day_mins > box_end_min) & (day_mins <= eod_min)
        rem_idxs = day_idxs[rem_mask]

        in_pos = False
        entry_p = 0.0
        opt_arr = None
        pos_side = ""

        for idx in rem_idxs:
            h = highs_1m[idx]
            l = lows_1m[idx]
            c = closes_1m[idx]
            tm = mins_of_day_1m[idx]

            if in_pos:
                cur_p = opt_arr[idx]
                ret = (cur_p - entry_p) / entry_p
                is_tp = ret >= tp_pct
                is_sl = ret <= -sl_pct
                is_spot_sl = spot_sl and ((pos_side == "CE" and c < box_h) or (pos_side == "PE" and c > box_l))
                is_eod = tm >= eod_min

                if is_tp or is_sl or is_spot_sl or is_eod:
                    net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                    trades.append({"net": net})
                    in_pos = False
                    break
            else:
                if h >= (box_h + buffer_pts):
                    pos_side = "CE"
                    if mode == "real":
                        arr = get_real_opt_array(c, 0, "CE", "1m")
                        if arr is not None and not np.isnan(arr[idx]) and arr[idx] > 5:
                            opt_arr = arr
                            entry_p = arr[idx]
                            in_pos = True
                    else:
                        opt_arr = synth_ce_atm_1m
                        entry_p = opt_arr[idx]
                        in_pos = True

                elif l <= (box_l - buffer_pts):
                    pos_side = "PE"
                    if mode == "real":
                        arr = get_real_opt_array(c, 0, "PE", "1m")
                        if arr is not None and not np.isnan(arr[idx]) and arr[idx] > 5:
                            opt_arr = arr
                            entry_p = arr[idx]
                            in_pos = True
                    else:
                        opt_arr = synth_pe_atm_1m
                        entry_p = opt_arr[idx]
                        in_pos = True

    return calc_metrics(trades)

# ---------------------------------------------------------
# SIMULATION ENGINE 3: GAMMA SQUEEZE MOMENTUM (5-MIN)
# ---------------------------------------------------------
def run_gamma(atr_mult, tp_pct, sl_pct, use_trend=True, vol_mult=1.0, mode="real"):
    trades = []
    in_pos = False
    entry_p = 0.0
    opt_arr = None
    pos_dir = ""
    last_exit_idx = -20

    for i in range(25, n_5m):
        tm_min = mins_of_day_5m[i]
        d = dates_5m[i]

        if mode in ["real", "synth_week"]:
            if d < real_opt_start_date or d > real_opt_end_date:
                continue

        spot = closes_5m[i]
        prev_spot = closes_5m[i-1]
        vwap = vwap_5m[i]
        prev_vwap = vwap_5m[i-1]
        atr = atr_5m[i]
        prev_atr = atr_5m[i-1]
        em = ema50_5m[i]
        vol = vols_5m[i]
        v_sma = vol_sma20_5m[i]

        if in_pos:
            cur_p = opt_arr[i]
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= tp_pct
            is_sl = ret <= -sl_pct
            is_rev = (pos_dir == "CE" and spot < vwap) or (pos_dir == "PE" and spot > vwap)
            is_eod = tm_min >= 915

            if is_tp or is_sl or is_rev or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                trades.append({"net": net})
                in_pos = False
                last_exit_idx = i
        else:
            if tm_min < 570 or tm_min > 870: # 09:30 to 14:30
                continue
            if (i - last_exit_idx) < 3: # 15-min cooldown
                continue

            thresh_up = vwap + atr_mult * atr
            prev_thresh_up = prev_vwap + atr_mult * prev_atr
            fresh_ce = (prev_spot <= prev_thresh_up) and (spot > thresh_up)

            thresh_dn = vwap - atr_mult * atr
            prev_thresh_dn = prev_vwap - atr_mult * prev_atr
            fresh_pe = (prev_spot >= prev_thresh_dn) and (spot < thresh_dn)

            vol_ok = (vol >= vol_mult * v_sma)

            if fresh_ce and vol_ok:
                if use_trend and spot < em:
                    continue
                if mode == "real":
                    arr = get_real_opt_array(spot, 0, "CE", "5m")
                    if arr is not None and not np.isnan(arr[i]) and arr[i] > 5:
                        opt_arr = arr
                        entry_p = arr[i]
                        pos_dir = "CE"
                        in_pos = True
                else:
                    opt_arr = synth_ce_atm_5m
                    entry_p = opt_arr[i]
                    pos_dir = "CE"
                    in_pos = True

            elif fresh_pe and vol_ok:
                if use_trend and spot > em:
                    continue
                if mode == "real":
                    arr = get_real_opt_array(spot, 0, "PE", "5m")
                    if arr is not None and not np.isnan(arr[i]) and arr[i] > 5:
                        opt_arr = arr
                        entry_p = arr[i]
                        pos_dir = "PE"
                        in_pos = True
                else:
                    opt_arr = synth_pe_atm_5m
                    entry_p = opt_arr[i]
                    pos_dir = "PE"
                    in_pos = True

    return calc_metrics(trades)

# ---------------------------------------------------------
# SIMULATION ENGINE 4: HEDGED STRANGLE (HS)
# ---------------------------------------------------------
def run_hedged(entry_time_min, target_pct, stop_pct, hold_m, offset=300, mode="real"):
    trades = []
    in_pos = False
    entry_tot = 0.0
    entry_idx = 0
    ce_arr = None
    pe_arr = None

    for i in range(20, n_1m):
        tm_min = mins_of_day_1m[i]
        d = dates_1m[i]
        weekday = df_1m.index[i].weekday()

        if mode in ["real", "synth_week"]:
            if d < real_opt_start_date or d > real_opt_end_date:
                continue

        if in_pos:
            bars = i - entry_idx
            cur_tot = ce_arr[i] + pe_arr[i]
            ret = (cur_tot - entry_tot) / entry_tot

            if ret >= target_pct or ret <= -stop_pct or bars >= hold_m or tm_min >= 915:
                net = (cur_tot - entry_tot) * LOT_SIZE - FEE_STRANGLE
                trades.append({"net": net})
                in_pos = False
        else:
            if tm_min != entry_time_min:
                continue
            if weekday not in [1, 2]:  # Tuesday, Wednesday
                continue

            spot = closes_1m[i]
            if mode == "real":
                arr_c = get_real_opt_array(spot, offset, "CE", "1m")
                arr_p = get_real_opt_array(spot, -offset, "PE", "1m")
                if arr_c is not None and arr_p is not None:
                    if not np.isnan(arr_c[i]) and not np.isnan(arr_p[i]) and arr_c[i] > 5 and arr_p[i] > 5:
                        ce_arr = arr_c
                        pe_arr = arr_p
                        entry_tot = arr_c[i] + arr_p[i]
                        in_pos = True
                        entry_idx = i
            else:
                ce_arr = synth_ce_otm1_1m if offset <= 200 else synth_ce_otm2_1m
                pe_arr = synth_pe_otm1_1m if offset <= 200 else synth_pe_otm2_1m
                entry_tot = ce_arr[i] + pe_arr[i]
                in_pos = True
                entry_idx = i

    return calc_metrics(trades)


# ---------------------------------------------------------
# SWEEPS EXECUTION
# ---------------------------------------------------------
print("[4/5] Executing Sweeps across All Parameter Spaces...")

# 1. Sweep DAS
das_results = []
das_grid = list(itertools.product(
    [35.0, 50.0, 65.0, 80.0, 100.0],     # std_th
    [30.0, 45.0, 60.0, 80.0],            # vel_th
    [0.40, 0.60, 0.80, 1.00],            # win_tgt
    [0.20, 0.25, 0.30, 0.35],            # lose_sp
    [45, 60, 90, 120],                    # hold_m
    [0, 200, 400]                         # offset
))
print(f"--> Sweeping DAS: {len(das_grid)} parameter combinations...")
for std_th, vel_th, win_tgt, lose_sp, hold_m, off in das_grid:
    mr = run_das(std_th, vel_th, win_tgt, lose_sp, hold_m, off, "real")
    if mr["trades"] >= 1:
        ms_w = run_das(std_th, vel_th, win_tgt, lose_sp, hold_m, off, "synth_week")
        ms_f = run_das(std_th, vel_th, win_tgt, lose_sp, hold_m, off, "synth_full")
        das_results.append({
            "std": std_th, "vel": vel_th, "tp": f"+{int(win_tgt*100)}%", "sl": f"-{int(lose_sp*100)}%",
            "hold": f"{hold_m}m", "otm": f"{off}pt",
            "r_trades": mr["trades"], "r_wr": f"{mr['wr']}%", "r_pf": mr["pf"], "r_pnl": mr["pnl"],
            "sw_trades": ms_w["trades"], "sw_pnl": ms_w["pnl"],
            "sf_trades": ms_f["trades"], "sf_wr": f"{ms_f['wr']}%", "sf_pf": ms_f["pf"], "sf_pnl": ms_f["pnl"]
        })

df_das = pd.DataFrame(das_results)
df_das.sort_values(by="r_pnl", ascending=False, inplace=True)

# 2. Sweep Directional Box Breakout
box_results = []
box_grid = list(itertools.product(
    [15, 30, 45, 60],                     # box_mins
    [0.50, 0.75, 1.00, 1.30],             # max_box_pct
    [30.0, 50.0, 75.0, 100.0, 150.0],     # buffer_pts
    [0.40, 0.60, 0.80, 1.00],             # tp_pct
    [0.20, 0.25, 0.30, 0.35],             # sl_pct
    [False, True]                         # spot_sl
))
print(f"--> Sweeping Box Breakout: {len(box_grid)} parameter combinations...")
for box_m, max_pct, buf, tp, sl, sp_sl in box_grid:
    mr = run_box(box_m, max_pct, buf, tp, sl, sp_sl, "real")
    if mr["trades"] >= 1:
        ms_w = run_box(box_m, max_pct, buf, tp, sl, sp_sl, "synth_week")
        ms_f = run_box(box_m, max_pct, buf, tp, sl, sp_sl, "synth_full")
        box_results.append({
            "box_m": f"{box_m}m", "max_pct": f"{max_pct}%", "buf": f"{int(buf)}pt",
            "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%", "spot_sl": sp_sl,
            "r_trades": mr["trades"], "r_wr": f"{mr['wr']}%", "r_pf": mr["pf"], "r_pnl": mr["pnl"],
            "sw_trades": ms_w["trades"], "sw_pnl": ms_w["pnl"],
            "sf_trades": ms_f["trades"], "sf_wr": f"{ms_f['wr']}%", "sf_pf": ms_f["pf"], "sf_pnl": ms_f["pnl"]
        })

df_box = pd.DataFrame(box_results)
if not df_box.empty:
    df_box.sort_values(by="r_pnl", ascending=False, inplace=True)

# 3. Sweep Gamma Squeeze
gamma_results = []
gamma_grid = list(itertools.product(
    [1.8, 2.2, 2.6, 3.0, 3.5],            # atr_mult
    [0.40, 0.60, 0.80, 1.00],             # tp_pct
    [0.20, 0.25, 0.30],                   # sl_pct
    [True, False],                        # trend
    [1.0, 1.3]                            # vol_mult
))
print(f"--> Sweeping Gamma Squeeze: {len(gamma_grid)} parameter combinations...")
for atr_m, tp, sl, trend, v_mult in gamma_grid:
    mr = run_gamma(atr_m, tp, sl, trend, v_mult, "real")
    if mr["trades"] >= 1:
        ms_w = run_gamma(atr_m, tp, sl, trend, v_mult, "synth_week")
        ms_f = run_gamma(atr_m, tp, sl, trend, v_mult, "synth_full")
        gamma_results.append({
            "atr": f"{atr_m}x", "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%",
            "trend": "EMA50" if trend else "None", "vol": f"{v_mult}x",
            "r_trades": mr["trades"], "r_wr": f"{mr['wr']}%", "r_pf": mr["pf"], "r_pnl": mr["pnl"],
            "sw_trades": ms_w["trades"], "sw_pnl": ms_w["pnl"],
            "sf_trades": ms_f["trades"], "sf_wr": f"{ms_f['wr']}%", "sf_pf": ms_f["pf"], "sf_pnl": ms_f["pnl"]
        })

df_gamma = pd.DataFrame(gamma_results)
if not df_gamma.empty:
    df_gamma.sort_values(by="r_pnl", ascending=False, inplace=True)

# 4. Sweep Hedged Strangle
hedged_results = []
hedged_grid = list(itertools.product(
    [570, 600, 660],                      # 09:30, 10:00, 11:00
    [0.25, 0.35, 0.50, 0.70],             # target_pct
    [0.15, 0.20, 0.25, 0.30],             # stop_pct
    [45, 60, 90, 150],                    # hold_m
    [200, 400]                            # offset
))
print(f"--> Sweeping Hedged Strangle: {len(hedged_grid)} parameter combinations...")
for et_min, tp, sl, hold, off in hedged_grid:
    mr = run_hedged(et_min, tp, sl, hold, off, "real")
    if mr["trades"] >= 1:
        ms_w = run_hedged(et_min, tp, sl, hold, off, "synth_week")
        ms_f = run_hedged(et_min, tp, sl, hold, off, "synth_full")
        et_str = f"{et_min//60:02d}:{et_min%60:02d}"
        hedged_results.append({
            "entry": et_str, "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%",
            "hold": f"{hold}m", "otm": f"{off}pt",
            "r_trades": mr["trades"], "r_wr": f"{mr['wr']}%", "r_pf": mr["pf"], "r_pnl": mr["pnl"],
            "sw_trades": ms_w["trades"], "sw_pnl": ms_w["pnl"],
            "sf_trades": ms_f["trades"], "sf_wr": f"{ms_f['wr']}%", "sf_pf": ms_f["pf"], "sf_pnl": ms_f["pnl"]
        })

df_hedged = pd.DataFrame(hedged_results)
if not df_hedged.empty:
    df_hedged.sort_values(by="r_pnl", ascending=False, inplace=True)

print("\n" + "=" * 85)
print("  BANKNIFTY OPTIMIZATION RESULTS SUMMARY")
print("=" * 85)

print("\n>>> 1. TOP 5 DECOUPLED ASYMMETRIC STRANGLE (DAS) CONFIGURATIONS:")
if not df_das.empty:
    print(tabulate(df_das.head(5), headers="keys", tablefmt="grid", showindex=False))

print("\n>>> 2. TOP 5 DIRECTIONAL BOX BREAKOUT CONFIGURATIONS:")
if not df_box.empty:
    print(tabulate(df_box.head(5), headers="keys", tablefmt="grid", showindex=False))
else:
    print("No qualifying box breakout setups met minimum trade threshold.")

print("\n>>> 3. TOP 5 GAMMA SQUEEZE MOMENTUM CONFIGURATIONS:")
if not df_gamma.empty:
    print(tabulate(df_gamma.head(5), headers="keys", tablefmt="grid", showindex=False))
else:
    print("No qualifying gamma squeeze setups met minimum trade threshold.")

print("\n>>> 4. TOP 5 HEDGED STRANGLE CONFIGURATIONS:")
if not df_hedged.empty:
    print(tabulate(df_hedged.head(5), headers="keys", tablefmt="grid", showindex=False))
else:
    print("No qualifying hedged strangle setups met minimum trade threshold.")

# Save CSVs
os.makedirs("data/banknifty_results", exist_ok=True)
if not df_das.empty:
    df_das.to_csv("data/banknifty_results/banknifty_das_sweep.csv", index=False)
if not df_box.empty:
    df_box.to_csv("data/banknifty_results/banknifty_box_sweep.csv", index=False)
if not df_gamma.empty:
    df_gamma.to_csv("data/banknifty_results/banknifty_gamma_sweep.csv", index=False)
if not df_hedged.empty:
    df_hedged.to_csv("data/banknifty_results/banknifty_hedged_sweep.csv", index=False)

print("\n[5/5] All results exported to data/banknifty_results/. Execution completed!")
