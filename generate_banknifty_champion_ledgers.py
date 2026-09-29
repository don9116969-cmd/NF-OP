"""
Final BankNIFTY Champion Verification and Ledger Generator.
Runs the optimal BankNIFTY calibrated parameter sets for:
1. Directional Box Breakout (DBB)
2. Decoupled Asymmetric Strangle (DAS)
3. Gamma Squeeze Momentum (GSM)
4. Hedged Strangle (HS)

Produces side-by-side results on:
- Real Angel One SmartAPI Options Cache (Traded Prices)
- Synthetic Black-Scholes Options (Exact Same Window & Full Period)
Saves complete trade ledgers and comparative metrics to data/banknifty_results/
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
RESULTS_DIR = "data/banknifty_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Data
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

# 1m arrays
n_1m = len(df_1m)
times_1m = df_1m.index.time
dates_1m = df_1m.index.date
closes_1m = df_1m["close"].values
highs_1m = df_1m["high"].values
lows_1m = df_1m["low"].values
mins_1m = np.array([t.hour * 60 + t.minute for t in times_1m])
rolling_std_20_1m = pd.Series(closes_1m).rolling(20).std().fillna(50.0).values
vel_3_1m = closes_1m - pd.Series(closes_1m).shift(3).fillna(0.0).values

# 5m arrays
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

# Load Real Options
opt_files = glob.glob(f"{CACHE_DIR}/*.csv")
strikes_ce = sorted(list(set([int(os.path.basename(f).replace("BANKNIFTY29SEP26", "").replace("CE.csv", "")) for f in opt_files if "CE.csv" in f])))
strikes_pe = sorted(list(set([int(os.path.basename(f).replace("BANKNIFTY29SEP26", "").replace("PE.csv", "")) for f in opt_files if "PE.csv" in f])))
strikes_ce_arr = np.array(strikes_ce)
strikes_pe_arr = np.array(strikes_pe)

real_opt_1m = {}
real_opt_5m = {}
real_start_d = datetime.date(2026, 9, 21)
real_end_d = datetime.date(2026, 9, 28)

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

def get_real_opt(spot, offset, opt_type, tf="1m"):
    target = spot + offset
    arr = strikes_ce_arr if opt_type == "CE" else strikes_pe_arr
    k = arr[np.argmin(np.abs(arr - target))]
    sym = f"BANKNIFTY29SEP26{k}{opt_type}"
    d = real_opt_1m if tf == "1m" else real_opt_5m
    return d.get(sym, None), sym

# Vectorized Synthetic Options
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
synth_ce_otm1_1m = bs_vec(closes_1m, atm_k_1m + 200, dtes_1m, "CE")
synth_pe_otm1_1m = bs_vec(closes_1m, atm_k_1m - 200, dtes_1m, "PE")

dtes_5m = calc_dtes(df_5m.index)
atm_k_5m = np.round(closes_5m / 100.0) * 100.0
synth_ce_atm_5m = bs_vec(closes_5m, atm_k_5m, dtes_5m, "CE")
synth_pe_atm_5m = bs_vec(closes_5m, atm_k_5m, dtes_5m, "PE")

# =============================================================================
# DETAILED SIMULATORS FOR CHAMPIONS
# =============================================================================

# 1. CHAMPION DIRECTIONAL BOX BREAKOUT (15m, 30pt buf, +60% TP, -20% SL)
def run_champion_box(use_real=True):
    trades = []
    box_end_min = 570  # 09:30
    eod_min = 915      # 15:15
    for d in np.unique(dates_1m):
        if d < real_start_d or d > real_end_d:
            continue
        day_idxs = np.where(dates_1m == d)[0]
        day_mins = mins_1m[day_idxs]
        b_mask = (day_mins >= 555) & (day_mins <= box_end_min)
        if np.sum(b_mask) < 8:
            continue
        b_idxs = day_idxs[b_mask]
        box_h = np.max(highs_1m[b_idxs])
        box_l = np.min(lows_1m[b_idxs])
        spot_ref = closes_1m[b_idxs[0]]
        if ((box_h - box_l) / spot_ref * 100.0) > 1.2:
            continue

        rem_idxs = day_idxs[(day_mins > box_end_min) & (day_mins <= eod_min)]
        in_pos = False
        entry_p = 0.0
        entry_ts = None
        opt_sym = ""
        opt_arr = None

        for idx in rem_idxs:
            h = highs_1m[idx]
            l = lows_1m[idx]
            c = closes_1m[idx]
            tm = mins_1m[idx]
            ts = df_1m.index[idx]

            if in_pos:
                cur_p = opt_arr[idx]
                ret = (cur_p - entry_p) / entry_p
                if ret >= 0.60 or ret <= -0.20 or tm >= eod_min:
                    gross = (cur_p - entry_p) * LOT_SIZE
                    net = gross - FEE_DIRECTIONAL
                    trades.append({
                        "date": str(d), "strategy": "Directional Box Breakout",
                        "symbol": opt_sym, "entry_time": str(entry_ts), "exit_time": str(ts),
                        "entry_price": round(entry_p, 2), "exit_price": round(cur_p, 2),
                        "return_pct": f"{round(ret * 100, 1)}%",
                        "gross_pnl": round(gross, 2), "friction": FEE_DIRECTIONAL, "net_pnl": round(net, 2)
                    })
                    in_pos = False
                    break
            else:
                if h >= (box_h + 30.0):
                    if use_real:
                        arr, sym = get_real_opt(c, 0, "CE", "1m")
                        if arr is not None and not np.isnan(arr[idx]) and arr[idx] > 5:
                            opt_arr = arr
                            opt_sym = sym
                            entry_p = arr[idx]
                            entry_ts = ts
                            in_pos = True
                    else:
                        opt_arr = synth_ce_atm_1m
                        opt_sym = f"SYNTH_ATM_CE_{int(c)}"
                        entry_p = opt_arr[idx]
                        entry_ts = ts
                        in_pos = True

                elif l <= (box_l - 30.0):
                    if use_real:
                        arr, sym = get_real_opt(c, 0, "PE", "1m")
                        if arr is not None and not np.isnan(arr[idx]) and arr[idx] > 5:
                            opt_arr = arr
                            opt_sym = sym
                            entry_p = arr[idx]
                            entry_ts = ts
                            in_pos = True
                    else:
                        opt_arr = synth_pe_atm_1m
                        opt_sym = f"SYNTH_ATM_PE_{int(c)}"
                        entry_p = opt_arr[idx]
                        entry_ts = ts
                        in_pos = True

    return pd.DataFrame(trades)

# 2. CHAMPION DAS (std 45, vel 50, +80% TP, -25% SL, 90m hold, 200pt OTM)
def run_champion_das(use_real=True):
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
    sym_c = ""
    sym_p = ""
    entry_ts = None

    for i in range(30, n_1m):
        tm_min = mins_1m[i]
        d = dates_1m[i]
        ts = df_1m.index[i]
        if d < real_start_d or d > real_end_d:
            continue

        if in_pos:
            bars = i - entry_idx
            c_ce = ce_arr[i]
            c_pe = pe_arr[i]

            if not ce_exited:
                r_ce = (c_ce - ce_entry) / max(0.1, ce_entry)
                if r_ce >= 0.80 or r_ce <= -0.25:
                    ce_exited = True
                    ce_exit_p = c_ce

            if not pe_exited:
                r_pe = (c_pe - pe_entry) / max(0.1, pe_entry)
                if r_pe >= 0.80 or r_pe <= -0.25:
                    pe_exited = True
                    pe_exit_p = c_pe

            if (ce_exited and pe_exited) or (bars >= 90) or (tm_min >= 915):
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                ce_pnl = (p_c - ce_entry) * LOT_SIZE
                pe_pnl = (p_p - pe_entry) * LOT_SIZE
                gross = ce_pnl + pe_pnl
                net = gross - FEE_STRANGLE
                trades.append({
                    "date": str(d), "strategy": "Decoupled Asymmetric Strangle",
                    "symbol": f"{sym_c} + {sym_p}", "entry_time": str(entry_ts), "exit_time": str(ts),
                    "ce_entry": round(ce_entry, 2), "ce_exit": round(p_c, 2), "ce_pnl": round(ce_pnl, 2),
                    "pe_entry": round(pe_entry, 2), "pe_exit": round(p_p, 2), "pe_pnl": round(pe_pnl, 2),
                    "gross_pnl": round(gross, 2), "friction": FEE_STRANGLE, "net_pnl": round(net, 2)
                })
                in_pos = False
                last_exit_idx = i
        else:
            if tm_min < 585 or tm_min > 810 or (690 <= tm_min <= 765):
                continue
            if (i - last_exit_idx) < 30:
                continue

            if rolling_std_20_1m[i] < 45.0 and abs(vel_3_1m[i]) >= 50.0:
                spot = closes_1m[i]
                if use_real:
                    arr_c, sc = get_real_opt(spot, 200, "CE", "1m")
                    arr_p, sp = get_real_opt(spot, -200, "PE", "1m")
                    if arr_c is not None and arr_p is not None:
                        if not np.isnan(arr_c[i]) and not np.isnan(arr_p[i]) and arr_c[i] > 5 and arr_p[i] > 5:
                            ce_arr = arr_c
                            pe_arr = arr_p
                            sym_c = sc
                            sym_p = sp
                            ce_entry = arr_c[i]
                            pe_entry = arr_p[i]
                            in_pos = True
                            entry_idx = i
                            entry_ts = ts
                            ce_exited = False
                            pe_exited = False
                else:
                    ce_arr = synth_ce_otm1_1m
                    pe_arr = synth_pe_otm1_1m
                    sym_c = f"SYNTH_OTM_CE_{int(spot+200)}"
                    sym_p = f"SYNTH_OTM_PE_{int(spot-200)}"
                    ce_entry = ce_arr[i]
                    pe_entry = pe_arr[i]
                    in_pos = True
                    entry_idx = i
                    entry_ts = ts
                    ce_exited = False
                    pe_exited = False

    return pd.DataFrame(trades)

# 3. CHAMPION GAMMA SQUEEZE (1.2x ATR, Session Mean, +40% TP, -20% SL)
def run_champion_gamma(use_real=True):
    trades = []
    in_pos = False
    entry_p = 0.0
    entry_ts = None
    opt_arr = None
    opt_sym = ""
    pos_dir = ""
    last_exit_idx = -20

    for i in range(25, n_5m):
        tm_min = mins_5m[i]
        d = dates_5m[i]
        ts = df_5m.index[i]
        if d < real_start_d or d > real_end_d:
            continue

        spot = closes_5m[i]
        s_mean = session_mean_5m[i]
        atr = atr_5m[i]

        if in_pos:
            cur_p = opt_arr[i]
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= 0.40
            is_sl = ret <= -0.20
            is_rev = (pos_dir == "CE" and spot < s_mean) or (pos_dir == "PE" and spot > s_mean)
            is_eod = tm_min >= 915

            if is_tp or is_sl or is_rev or is_eod:
                gross = (cur_p - entry_p) * LOT_SIZE
                net = gross - FEE_DIRECTIONAL
                trades.append({
                    "date": str(d), "strategy": "Gamma Squeeze Momentum",
                    "symbol": opt_sym, "entry_time": str(entry_ts), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(cur_p, 2),
                    "return_pct": f"{round(ret * 100, 1)}%",
                    "gross_pnl": round(gross, 2), "friction": FEE_DIRECTIONAL, "net_pnl": round(net, 2)
                })
                in_pos = False
                last_exit_idx = i
        else:
            if tm_min < 570 or tm_min > 870:
                continue
            if (i - last_exit_idx) < 3:
                continue

            f_ce = (closes_5m[i-1] <= session_mean_5m[i-1] + 1.2 * atr_5m[i-1]) and (spot > s_mean + 1.2 * atr)
            f_pe = (closes_5m[i-1] >= session_mean_5m[i-1] - 1.2 * atr_5m[i-1]) and (spot < s_mean - 1.2 * atr)

            if f_ce:
                if use_real:
                    arr, sym = get_real_opt(spot, 0, "CE", "5m")
                    if arr is not None and not np.isnan(arr[i]) and arr[i] > 5:
                        opt_arr = arr
                        opt_sym = sym
                        entry_p = arr[i]
                        entry_ts = ts
                        pos_dir = "CE"
                        in_pos = True
                else:
                    opt_arr = synth_ce_atm_5m
                    opt_sym = f"SYNTH_ATM_CE_{int(spot)}"
                    entry_p = opt_arr[i]
                    entry_ts = ts
                    pos_dir = "CE"
                    in_pos = True

            elif f_pe:
                if use_real:
                    arr, sym = get_real_opt(spot, 0, "PE", "5m")
                    if arr is not None and not np.isnan(arr[i]) and arr[i] > 5:
                        opt_arr = arr
                        opt_sym = sym
                        entry_p = arr[i]
                        entry_ts = ts
                        pos_dir = "PE"
                        in_pos = True
                else:
                    opt_arr = synth_pe_atm_5m
                    opt_sym = f"SYNTH_ATM_PE_{int(spot)}"
                    entry_p = opt_arr[i]
                    entry_ts = ts
                    pos_dir = "PE"
                    in_pos = True

    return pd.DataFrame(trades)

# 4. CHAMPION HEDGED STRANGLE (09:30 entry, 200pt OTM, +70% TP, -30% SL, 45m hold)
def run_champion_hedged(use_real=True):
    trades = []
    in_pos = False
    entry_tot = 0.0
    entry_idx = 0
    entry_ts = None
    ce_arr = None
    pe_arr = None
    sym_c = ""
    sym_p = ""

    for i in range(20, n_1m):
        tm_min = mins_1m[i]
        d = dates_1m[i]
        ts = df_1m.index[i]
        if d < real_start_d or d > real_end_d:
            continue

        if in_pos:
            bars = i - entry_idx
            cur_tot = ce_arr[i] + pe_arr[i]
            ret = (cur_tot - entry_tot) / entry_tot
            if ret >= 0.70 or ret <= -0.30 or bars >= 45 or tm_min >= 915:
                gross = (cur_tot - entry_tot) * LOT_SIZE
                net = gross - FEE_STRANGLE
                trades.append({
                    "date": str(d), "strategy": "Hedged Strangle",
                    "symbol": f"{sym_c} + {sym_p}", "entry_time": str(entry_ts), "exit_time": str(ts),
                    "entry_premium": round(entry_tot, 2), "exit_premium": round(cur_tot, 2),
                    "return_pct": f"{round(ret * 100, 1)}%",
                    "gross_pnl": round(gross, 2), "friction": FEE_STRANGLE, "net_pnl": round(net, 2)
                })
                in_pos = False
        else:
            if tm_min != 570:  # 09:30
                continue
            if ts.weekday() not in [1, 2]:
                continue
            spot = closes_1m[i]
            if use_real:
                arr_c, sc = get_real_opt(spot, 200, "CE", "1m")
                arr_p, sp = get_real_opt(spot, -200, "PE", "1m")
                if arr_c is not None and arr_p is not None:
                    if not np.isnan(arr_c[i]) and not np.isnan(arr_p[i]) and arr_c[i] > 5 and arr_p[i] > 5:
                        ce_arr = arr_c
                        pe_arr = arr_p
                        sym_c = sc
                        sym_p = sp
                        entry_tot = arr_c[i] + arr_p[i]
                        entry_ts = ts
                        entry_idx = i
                        in_pos = True
            else:
                ce_arr = synth_ce_otm1_1m
                pe_arr = synth_pe_otm1_1m
                sym_c = f"SYNTH_OTM_CE_{int(spot+200)}"
                sym_p = f"SYNTH_OTM_PE_{int(spot-200)}"
                entry_tot = ce_arr[i] + pe_arr[i]
                entry_ts = ts
                entry_idx = i
                in_pos = True

    return pd.DataFrame(trades)

# Generate Ledgers
print("\n>>> Simulating all 4 calibrated strategies on Real SmartAPI Options and Synthetic Data...")

df_box_real = run_champion_box(use_real=True)
df_box_synth = run_champion_box(use_real=False)

df_das_real = run_champion_das(use_real=True)
df_das_synth = run_champion_das(use_real=False)

df_gamma_real = run_champion_gamma(use_real=True)
df_gamma_synth = run_champion_gamma(use_real=False)

df_hedged_real = run_champion_hedged(use_real=True)
df_hedged_synth = run_champion_hedged(use_real=False)

# Export Ledgers
df_box_real.to_csv(f"{RESULTS_DIR}/box_breakout_banknifty_real_ledger.csv", index=False)
df_box_synth.to_csv(f"{RESULTS_DIR}/box_breakout_banknifty_synth_ledger.csv", index=False)

df_das_real.to_csv(f"{RESULTS_DIR}/das_banknifty_real_ledger.csv", index=False)
df_das_synth.to_csv(f"{RESULTS_DIR}/das_banknifty_synth_ledger.csv", index=False)

df_gamma_real.to_csv(f"{RESULTS_DIR}/gamma_squeeze_banknifty_real_ledger.csv", index=False)
df_gamma_synth.to_csv(f"{RESULTS_DIR}/gamma_squeeze_banknifty_synth_ledger.csv", index=False)

df_hedged_real.to_csv(f"{RESULTS_DIR}/hedged_strangle_banknifty_real_ledger.csv", index=False)
df_hedged_synth.to_csv(f"{RESULTS_DIR}/hedged_strangle_banknifty_synth_ledger.csv", index=False)

# Summary table generator
def summarize_df(df, name, source):
    if df.empty:
        return {"Strategy": name, "Data Source": source, "Trades": 0, "Win Rate": "0.0%", "Profit Factor": 0.0, "Net PnL (INR)": 0.0, "Avg Trade (INR)": 0.0}
    w = df[df["net_pnl"] > 0]["net_pnl"]
    l = abs(df[df["net_pnl"] <= 0]["net_pnl"].sum())
    w_sum = w.sum()
    pf = round(w_sum / l, 2) if l > 0 else (99.0 if w_sum > 0 else 0.0)
    wr = round(len(w) / len(df) * 100.0, 1)
    tot = round(df["net_pnl"].sum(), 2)
    avg_t = round(tot / len(df), 2)
    return {
        "Strategy": name, "Data Source": source,
        "Trades": len(df), "Win Rate": f"{wr}%", "Profit Factor": pf,
        "Net PnL (INR)": f"Rs. {tot:,.2f}", "Avg Trade (INR)": f"Rs. {avg_t:,.2f}"
    }

summary = [
    summarize_df(df_box_real, "Directional Box Breakout (DBB)", "Real SmartAPI Options"),
    summarize_df(df_box_synth, "Directional Box Breakout (DBB)", "Synthetic Black-Scholes"),
    summarize_df(df_das_real, "Decoupled Asymmetric Strangle (DAS)", "Real SmartAPI Options"),
    summarize_df(df_das_synth, "Decoupled Asymmetric Strangle (DAS)", "Synthetic Black-Scholes"),
    summarize_df(df_gamma_real, "Gamma Squeeze Momentum (GSM)", "Real SmartAPI Options"),
    summarize_df(df_gamma_synth, "Gamma Squeeze Momentum (GSM)", "Synthetic Black-Scholes"),
    summarize_df(df_hedged_real, "Hedged Strangle (HS)", "Real SmartAPI Options"),
    summarize_df(df_hedged_synth, "Hedged Strangle (HS)", "Synthetic Black-Scholes"),
]

df_summary = pd.DataFrame(summary)
print("\n" + "=" * 95)
print("  FINAL BANKNIFTY BENCHMARK COMPARISON TABLE (CALIBRATED PARAMETERS)")
print("=" * 95)
print(tabulate(df_summary, headers="keys", tablefmt="grid", showindex=False))

print("\nLedgers saved successfully in data/banknifty_results/.")
