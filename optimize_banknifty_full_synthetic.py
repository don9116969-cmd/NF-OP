"""
Full-Dataset Synthetic Parametric Optimizer for BankNIFTY.
Searches all parameter combinations across the entire historical dataset:
- 1-min dataset: 8,664 bars (Aug 25 - Sep 28)
- 5-min dataset: 4,652 bars (July 1 - Sep 28, 3 full months)

Finds the robust champion parameters that maximize Net Realized PnL, Profit Factor, and Win Rate.
"""

import os
import datetime
import itertools
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_DIRECTIONAL = 45.0
RESULTS_DIR = "data/banknifty_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 95)
print("  EXHAUSTIVE FULL-DATASET PARAMETRIC OPTIMIZATION (SYNTHETIC BLACK-SCHOLES)")
print("=" * 95)

# -------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------
print("\n[Step 1] Loading Spot Datasets...")
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
mins_1m = np.array([t.hour * 60 + t.minute for t in times_1m])
rolling_std_20_1m = pd.Series(closes_1m).rolling(20).std().fillna(50.0).values
vel_3_1m = closes_1m - pd.Series(closes_1m).shift(3).fillna(0.0).values

# 5-min arrays
n_5m = len(df_5m)
times_5m = df_5m.index.time
dates_5m = df_5m.index.date
closes_5m = df_5m["close"].values
highs_5m = df_5m["high"].values
lows_5m = df_5m["low"].values
mins_5m = np.array([t.hour * 60 + t.minute for t in times_5m])

df_5m["typical_p"] = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
session_mean_5m = df_5m.groupby(df_5m.index.date)["typical_p"].expanding().mean().values
tr1 = df_5m["high"] - df_5m["low"]
tr2 = (df_5m["high"] - df_5m["close"].shift(1)).abs()
tr3 = (df_5m["low"] - df_5m["close"].shift(1)).abs()
atr_5m = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().fillna(60.0).values
ema50_5m = df_5m["close"].ewm(span=50).mean().values

print(f"1-Minute Data: {n_1m} bars (Aug 25 to Sep 28)")
print(f"5-Minute Data: {n_5m} bars (July 1 to Sep 28, 3 full months)")

# -------------------------------------------------------------
# 2. FAST VECTORIZED OPTION PRICER
# -------------------------------------------------------------
print("\n[Step 2] Precomputing Vectorized Synthetic Option Grids...")
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

# Metrics Helper
def evaluate_metrics(trades):
    if not trades:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "avg_trade": 0.0, "max_dd": 0.0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net"] > 0]["net"]
    losses = tdf[tdf["net"] <= 0]["net"]
    w_sum = wins.sum()
    l_sum = abs(losses.sum())
    pf = round(w_sum / l_sum, 2) if l_sum > 0 else (99.0 if w_sum > 0 else 0.0)
    wr = round(len(wins) / len(tdf) * 100.0, 1)
    tot = round(tdf["net"].sum(), 1)
    avg_t = round(tot / len(tdf), 1)

    cum = tdf["net"].cumsum()
    peak = cum.cummax()
    dd = (peak - cum).max()
    max_dd = round(dd, 1) if not np.isnan(dd) else 0.0
    return {"trades": len(tdf), "wr": wr, "pf": pf, "pnl": tot, "avg_trade": avg_t, "max_dd": max_dd}

# Pre-calculate day index slices for 1m
day_slices_1m = []
for d in np.unique(dates_1m):
    idxs = np.where(dates_1m == d)[0]
    day_slices_1m.append((d, idxs, mins_1m[idxs]))

print("Precomputation finished in < 1 second.\n")

# -------------------------------------------------------------
# 3. OPTIMIZE STRATEGY 1: DIRECTIONAL BOX BREAKOUT (DBB)
# -------------------------------------------------------------
print("[Step 3] Sweeping Parameter Space for Directional Box Breakout (DBB)...")

def sim_box_fast(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, use_spot_sl=False):
    trades = []
    box_end_min = 555 + box_mins
    eod_min = 915

    for d, day_idxs, day_mins in day_slices_1m:
        b_mask = (day_mins >= 555) & (day_mins <= box_end_min)
        if np.sum(b_mask) < (box_mins // 2):
            continue

        b_idxs = day_idxs[b_mask]
        box_h = np.max(highs_1m[b_idxs])
        box_l = np.min(lows_1m[b_idxs])
        spot_ref = closes_1m[b_idxs[0]]
        if ((box_h - box_l) / spot_ref * 100.0) > max_box_pct:
            continue

        rem_mask = (day_mins > box_end_min) & (day_mins <= eod_min)
        rem_idxs = day_idxs[rem_mask]

        in_pos = False
        entry_p = 0.0
        opt_arr = None
        pos_dir = ""

        for idx in rem_idxs:
            h = highs_1m[idx]
            l = lows_1m[idx]
            c = closes_1m[idx]
            tm = mins_1m[idx]

            if in_pos:
                cur_p = opt_arr[idx]
                ret = (cur_p - entry_p) / entry_p
                is_tp = ret >= tp_pct
                is_sl = ret <= -sl_pct
                is_spot_sl = use_spot_sl and ((pos_dir == "CE" and c < box_h) or (pos_dir == "PE" and c > box_l))
                is_eod = tm >= eod_min

                if is_tp or is_sl or is_spot_sl or is_eod:
                    net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                    trades.append({"net": net})
                    in_pos = False
                    break
            else:
                if h >= (box_h + buffer_pts):
                    opt_arr = synth_ce_atm_1m
                    entry_p = opt_arr[idx]
                    pos_dir = "CE"
                    in_pos = True
                elif l <= (box_l - buffer_pts):
                    opt_arr = synth_pe_atm_1m
                    entry_p = opt_arr[idx]
                    pos_dir = "PE"
                    in_pos = True

    return evaluate_metrics(trades)

box_results = []
box_params = list(itertools.product(
    [15, 30, 45, 60],                      # box_mins
    [0.50, 0.75, 1.00, 1.50],              # max_box_pct
    [30.0, 50.0, 75.0, 100.0, 150.0],      # buffer_pts
    [0.30, 0.40, 0.50, 0.60, 0.80],        # tp_pct
    [0.15, 0.20, 0.25, 0.30],              # sl_pct
    [False, True]                          # use_spot_sl
))

print(f"Sweeping {len(box_params)} Box Breakout configurations across full 8,664 bars...")
for bm, mb, buf, tp, sl, sp_sl in box_params:
    res = sim_box_fast(bm, mb, buf, tp, sl, sp_sl)
    if res["trades"] >= 5:
        box_results.append({
            "box_m": f"{bm}m", "max_pct": f"{mb}%", "buf": f"{int(buf)}pt",
            "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%", "spot_sl": sp_sl,
            "trades": res["trades"], "wr": f"{res['wr']}%", "pf": res["pf"],
            "pnl": res["pnl"], "avg_trade": res["avg_trade"], "max_dd": res["max_dd"]
        })

df_box_res = pd.DataFrame(box_results)
df_box_res.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 DIRECTIONAL BOX BREAKOUT CONFIGURATIONS (FULL DATASET):")
print(tabulate(df_box_res.head(5), headers="keys", tablefmt="grid", showindex=False))

# -------------------------------------------------------------
# 4. OPTIMIZE STRATEGY 2: GAMMA SQUEEZE MOMENTUM (GSM)
# -------------------------------------------------------------
print("\n[Step 4] Sweeping Parameter Space for Gamma Squeeze Momentum (GSM)...")

def sim_gamma_fast(atr_mult, tp_pct, sl_pct, use_trend=True, max_trades_day=1):
    trades = []
    in_pos = False
    entry_p = 0.0
    opt_arr = None
    pos_dir = ""
    last_exit_idx = -20
    trades_today = 0
    current_day = None

    for i in range(25, n_5m):
        tm_min = mins_5m[i]
        d = dates_5m[i]

        if d != current_day:
            current_day = d
            trades_today = 0

        spot = closes_5m[i]
        prev_spot = closes_5m[i-1]
        s_mean = session_mean_5m[i]
        prev_s_mean = session_mean_5m[i-1]
        atr = atr_5m[i]
        prev_atr = atr_5m[i-1]
        em = ema50_5m[i]

        if in_pos:
            cur_p = opt_arr[i]
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= tp_pct
            is_sl = ret <= -sl_pct
            is_rev = (pos_dir == "CE" and spot < s_mean) or (pos_dir == "PE" and spot > s_mean)
            is_eod = tm_min >= 915

            if is_tp or is_sl or is_rev or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                trades.append({"net": net})
                in_pos = False
                last_exit_idx = i
                trades_today += 1
        else:
            if tm_min < 570 or tm_min > 870:
                continue
            if (i - last_exit_idx) < 3:
                continue
            if trades_today >= max_trades_day:
                continue

            thresh_up = s_mean + atr_mult * atr
            prev_thresh_up = prev_s_mean + atr_mult * prev_atr
            fresh_ce = (prev_spot <= prev_thresh_up) and (spot > thresh_up)

            thresh_dn = s_mean - atr_mult * atr
            prev_thresh_dn = prev_s_mean - atr_mult * prev_atr
            fresh_pe = (prev_spot >= prev_thresh_dn) and (spot < thresh_dn)

            if fresh_ce:
                if use_trend and spot < em:
                    continue
                opt_arr = synth_ce_atm_5m
                entry_p = opt_arr[i]
                pos_dir = "CE"
                in_pos = True
            elif fresh_pe:
                if use_trend and spot > em:
                    continue
                opt_arr = synth_pe_atm_5m
                entry_p = opt_arr[i]
                pos_dir = "PE"
                in_pos = True

    return evaluate_metrics(trades)

gamma_results = []
gamma_params = list(itertools.product(
    [1.2, 1.5, 1.8, 2.2, 2.6, 3.0],       # atr_mult
    [0.30, 0.40, 0.50, 0.60, 0.80],       # tp_pct
    [0.15, 0.20, 0.25, 0.30],             # sl_pct
    [True, False],                        # use_trend
    [1, 2, 3]                             # max_trades_day
))

print(f"Sweeping {len(gamma_params)} Gamma Squeeze configurations across full 3 months (4,652 bars)...")
for atr_m, tp, sl, tr, mtd in gamma_params:
    res = sim_gamma_fast(atr_m, tp, sl, tr, mtd)
    if res["trades"] >= 5:
        gamma_results.append({
            "atr": f"{atr_m}x", "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%",
            "trend": "EMA50" if tr else "None", "max_t": mtd,
            "trades": res["trades"], "wr": f"{res['wr']}%", "pf": res["pf"],
            "pnl": res["pnl"], "avg_trade": res["avg_trade"], "max_dd": res["max_dd"]
        })

df_gamma_res = pd.DataFrame(gamma_results)
df_gamma_res.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 GAMMA SQUEEZE CONFIGURATIONS (FULL 3-MONTH DATASET):")
print(tabulate(df_gamma_res.head(5), headers="keys", tablefmt="grid", showindex=False))

# -------------------------------------------------------------
# 5. OPTIMIZE STRATEGY 3: DECOUPLED ASYMMETRIC STRANGLE (DAS)
# -------------------------------------------------------------
print("\n[Step 5] Sweeping Parameter Space for Decoupled Asymmetric Strangle (DAS)...")

def sim_das_fast(std_th, vel_th, win_tgt, lose_sp, hold_m, offset=200):
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

    ce_arr = synth_ce_otm1_1m if offset == 200 else (synth_ce_atm_1m if offset == 0 else synth_ce_otm2_1m)
    pe_arr = synth_pe_otm1_1m if offset == 200 else (synth_pe_atm_1m if offset == 0 else synth_pe_otm2_1m)

    for i in range(30, n_1m):
        tm_min = mins_1m[i]

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

            if (ce_exited and pe_exited) or (bars >= hold_m) or (tm_min >= 915):
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                gross = (p_c - ce_entry + p_p - pe_entry) * LOT_SIZE
                net = gross - FEE_STRANGLE
                trades.append({"net": net})
                in_pos = False
                last_exit_idx = i

        else:
            if tm_min < 585 or tm_min > 810 or (690 <= tm_min <= 765):
                continue
            if (i - last_exit_idx) < 30:
                continue

            if rolling_std_20_1m[i] < std_th and abs(vel_3_1m[i]) >= vel_th:
                ce_entry = ce_arr[i]
                pe_entry = pe_arr[i]
                in_pos = True
                entry_idx = i
                ce_exited = False
                pe_exited = False

    return evaluate_metrics(trades)

das_results = []
das_params = list(itertools.product(
    [30.0, 40.0, 50.0, 60.0, 75.0, 90.0],  # std_th
    [35.0, 50.0, 65.0, 80.0, 100.0],       # vel_th
    [0.40, 0.60, 0.80, 1.00, 1.20],        # win_tgt
    [0.15, 0.20, 0.25, 0.30],              # lose_sp
    [45, 60, 90, 120],                     # hold_m
    [0, 200, 400]                          # offset
))

print(f"Sweeping {len(das_params)} DAS configurations across full 8,664 bars...")
for st, vt, wt, ls, hm, off in das_params:
    res = sim_das_fast(st, vt, wt, ls, hm, off)
    if res["trades"] >= 5:
        das_results.append({
            "std": st, "vel": vt, "tp": f"+{int(wt*100)}%", "sl": f"-{int(ls*100)}%",
            "hold": f"{hm}m", "otm": f"{off}pt",
            "trades": res["trades"], "wr": f"{res['wr']}%", "pf": res["pf"],
            "pnl": res["pnl"], "avg_trade": res["avg_trade"], "max_dd": res["max_dd"]
        })

df_das_res = pd.DataFrame(das_results)
df_das_res.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 DECOUPLED STRANGLE CONFIGURATIONS (FULL DATASET):")
print(tabulate(df_das_res.head(5), headers="keys", tablefmt="grid", showindex=False))

# -------------------------------------------------------------
# 6. OPTIMIZE STRATEGY 4: HEDGED STRANGLE (HS)
# -------------------------------------------------------------
print("\n[Step 6] Sweeping Parameter Space for Hedged Strangle (HS)...")

def sim_hedged_fast(entry_time_min, target_pct, stop_pct, hold_m, offset=200):
    trades = []
    in_pos = False
    entry_tot = 0.0
    entry_idx = 0
    ce_arr = synth_ce_otm1_1m if offset == 200 else (synth_ce_atm_1m if offset == 0 else synth_ce_otm2_1m)
    pe_arr = synth_pe_otm1_1m if offset == 200 else (synth_pe_atm_1m if offset == 0 else synth_pe_otm2_1m)

    for i in range(20, n_1m):
        tm_min = mins_1m[i]
        weekday = df_1m.index[i].weekday()

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
            if weekday not in [1, 2]:
                continue
            entry_tot = ce_arr[i] + pe_arr[i]
            entry_idx = i
            in_pos = True

    return evaluate_metrics(trades)

hedged_results = []
hedged_params = list(itertools.product(
    [570, 600, 660, 720, 780],             # 09:30, 10:00, 11:00, 12:00, 13:00
    [0.15, 0.25, 0.35, 0.50, 0.70],        # target_pct
    [0.10, 0.15, 0.20, 0.25, 0.30],        # stop_pct
    [30, 45, 60, 90, 120],                 # hold_m
    [0, 200, 400]                          # offset
))

print(f"Sweeping {len(hedged_params)} Hedged Strangle configurations across full 8,664 bars...")
for et, tp, sl, hm, off in hedged_params:
    res = sim_hedged_fast(et, tp, sl, hm, off)
    if res["trades"] >= 4:
        et_str = f"{et//60:02d}:{et%60:02d}"
        hedged_results.append({
            "entry": et_str, "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%",
            "hold": f"{hm}m", "otm": f"{off}pt",
            "trades": res["trades"], "wr": f"{res['wr']}%", "pf": res["pf"],
            "pnl": res["pnl"], "avg_trade": res["avg_trade"], "max_dd": res["max_dd"]
        })

df_hedged_res = pd.DataFrame(hedged_results)
df_hedged_res.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 HEDGED STRANGLE CONFIGURATIONS (FULL DATASET):")
print(tabulate(df_hedged_res.head(5), headers="keys", tablefmt="grid", showindex=False))

# Export top configurations
df_box_res.to_csv(f"{RESULTS_DIR}/full_synthetic_box_optimization.csv", index=False)
df_gamma_res.to_csv(f"{RESULTS_DIR}/full_synthetic_gamma_optimization.csv", index=False)
df_das_res.to_csv(f"{RESULTS_DIR}/full_synthetic_das_optimization.csv", index=False)
df_hedged_res.to_csv(f"{RESULTS_DIR}/full_synthetic_hedged_optimization.csv", index=False)

print("\n" + "=" * 95)
print("  FULL-DATASET OPTIMIZATION COMPLETE! All results exported to data/banknifty_results/")
print("=" * 95)
