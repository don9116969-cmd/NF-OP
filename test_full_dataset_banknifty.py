"""
Full Dataset BankNIFTY Strategy Simulator using Synthetic Black-Scholes Options.
Evaluates all 4 calibrated BankNIFTY strategies over the complete historical dataset:
1. Directional Box Breakout (DBB) - 1-min data (Aug 25 to Sep 28, 8,664 bars)
2. Gamma Squeeze Momentum (GSM) - 5-min data (July 1 to Sep 28, 4,652 bars)
3. Decoupled Asymmetric Strangle (DAS) - 1-min data (Aug 25 to Sep 28, 8,664 bars)
4. Hedged Strangle (HS) - 1-min data (Aug 25 to Sep 28, 8,664 bars)
"""

import os
import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_DIRECTIONAL = 45.0
RESULTS_DIR = "data/banknifty_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 90)
print("  BANKNIFTY FULL-DATASET STRATEGY SIMULATION (SYNTHETIC OPTIONS)")
print("=" * 90)

# -------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------
print("\n[1/4] Loading full historical spot datasets...")
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

print(f"1-Minute Data: {n_1m} bars ({df_1m.index[0].date()} to {df_1m.index[-1].date()})")
print(f"5-Minute Data: {n_5m} bars ({df_5m.index[0].date()} to {df_5m.index[-1].date()})")

# -------------------------------------------------------------
# 2. BLACK-SCHOLES SYNTHETIC OPTION PRICER
# -------------------------------------------------------------
print("\n[2/4] Initializing Black-Scholes synthetic option pricer...")
def bs_price(S, K, T, opt_type="CE", r=0.07, sigma=0.17):
    T_c = max(1e-4, T)
    sqrt_T = np.sqrt(T_c)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_c) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm.cdf(d1) - K * np.exp(-r * T_c) * norm.cdf(d2)
    else:
        p = K * np.exp(-r * T_c) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0.05, p)

def get_dte(ts):
    weekday = ts.weekday()
    target_exp = 2  # Wednesday expiry
    days_ahead = target_exp - weekday if weekday <= target_exp else 7 - (weekday - target_exp)
    mins = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
    frac = mins / 375.0
    d = max(0.01, frac * 0.5) if days_ahead == 0 else (days_ahead - 1) + frac
    return d / 365.0

# -------------------------------------------------------------
# 3. STRATEGY 1: DIRECTIONAL BOX BREAKOUT (DBB, 1-MIN)
# -------------------------------------------------------------
def run_full_box(box_mins=15, max_box_pct=1.2, buffer_pts=30.0, tp_pct=0.60, sl_pct=0.20):
    trades = []
    box_end_min = 555 + box_mins
    eod_min = 915

    for d in np.unique(dates_1m):
        day_idxs = np.where(dates_1m == d)[0]
        day_mins = mins_1m[day_idxs]

        b_mask = (day_mins >= 555) & (day_mins <= box_end_min)
        if np.sum(b_mask) < (box_mins // 2):
            continue

        b_idxs = day_idxs[b_mask]
        box_h = np.max(highs_1m[b_idxs])
        box_l = np.min(lows_1m[b_idxs])
        box_range = box_h - box_l
        spot_ref = closes_1m[b_idxs[0]]

        if (box_range / spot_ref * 100.0) > max_box_pct:
            continue

        rem_idxs = day_idxs[(day_mins > box_end_min) & (day_mins <= eod_min)]
        in_pos = False
        entry_p = 0.0
        entry_idx = 0
        strike = 0
        opt_type = ""

        for idx in rem_idxs:
            h = highs_1m[idx]
            l = lows_1m[idx]
            c = closes_1m[idx]
            tm = mins_1m[idx]
            ts = df_1m.index[idx]

            if in_pos:
                dte = get_dte(ts)
                cur_p = bs_price(c, strike, dte, opt_type)
                ret = (cur_p - entry_p) / entry_p

                if ret >= tp_pct or ret <= -sl_pct or tm >= eod_min:
                    gross = (cur_p - entry_p) * LOT_SIZE
                    net = gross - FEE_DIRECTIONAL
                    exit_reason = "Target Hit (+60%)" if ret >= tp_pct else ("Stop Loss (-20%)" if ret <= -sl_pct else "EOD Squareoff")
                    trades.append({
                        "date": str(d), "strategy": "Directional Box Breakout", "type": opt_type,
                        "strike": strike, "entry_time": str(df_1m.index[entry_idx].time()),
                        "exit_time": str(ts.time()), "entry_price": round(entry_p, 2),
                        "exit_price": round(cur_p, 2), "return_pct": f"{round(ret * 100, 1)}%",
                        "gross_pnl": round(gross, 2), "friction": FEE_DIRECTIONAL, "net_pnl": round(net, 2),
                        "exit_reason": exit_reason
                    })
                    in_pos = False
                    break
            else:
                if h >= (box_h + buffer_pts):
                    opt_type = "CE"
                    strike = int(round(c / 100.0) * 100)
                    dte = get_dte(ts)
                    entry_p = bs_price(c, strike, dte, "CE")
                    entry_idx = idx
                    in_pos = True

                elif l <= (box_l - buffer_pts):
                    opt_type = "PE"
                    strike = int(round(c / 100.0) * 100)
                    dte = get_dte(ts)
                    entry_p = bs_price(c, strike, dte, "PE")
                    entry_idx = idx
                    in_pos = True

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# 4. STRATEGY 2: GAMMA SQUEEZE MOMENTUM (GSM, 5-MIN)
# -------------------------------------------------------------
def run_full_gamma(atr_mult=1.2, tp_pct=0.40, sl_pct=0.20):
    trades = []
    in_pos = False
    entry_p = 0.0
    entry_idx = 0
    strike = 0
    opt_type = ""
    last_exit_idx = -20

    for i in range(25, n_5m):
        tm = mins_5m[i]
        d = dates_5m[i]
        ts = df_5m.index[i]
        spot = closes_5m[i]
        s_mean = session_mean_5m[i]
        atr = atr_5m[i]

        if in_pos:
            dte = get_dte(ts)
            cur_p = bs_price(spot, strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= tp_pct
            is_sl = ret <= -sl_pct
            is_rev = (opt_type == "CE" and spot < s_mean) or (opt_type == "PE" and spot > s_mean)
            is_eod = tm >= 915

            if is_tp or is_sl or is_rev or is_eod:
                gross = (cur_p - entry_p) * LOT_SIZE
                net = gross - FEE_DIRECTIONAL
                exit_reason = "Target Hit (+40%)" if is_tp else ("Stop Loss (-20%)" if is_sl else ("Reversal Cross" if is_rev else "EOD Exit"))
                trades.append({
                    "date": str(d), "strategy": "Gamma Squeeze Momentum", "type": opt_type,
                    "strike": strike, "entry_time": str(df_5m.index[entry_idx].time()),
                    "exit_time": str(ts.time()), "entry_price": round(entry_p, 2),
                    "exit_price": round(cur_p, 2), "return_pct": f"{round(ret * 100, 1)}%",
                    "gross_pnl": round(gross, 2), "friction": FEE_DIRECTIONAL, "net_pnl": round(net, 2),
                    "exit_reason": exit_reason
                })
                in_pos = False
                last_exit_idx = i
        else:
            if tm < 570 or tm > 870:
                continue
            if (i - last_exit_idx) < 3:
                continue

            f_ce = (closes_5m[i-1] <= session_mean_5m[i-1] + atr_mult * atr_5m[i-1]) and (spot > s_mean + atr_mult * atr)
            f_pe = (closes_5m[i-1] >= session_mean_5m[i-1] - atr_mult * atr_5m[i-1]) and (spot < s_mean - atr_mult * atr)

            if f_ce:
                opt_type = "CE"
                strike = int(round(spot / 100.0) * 100)
                dte = get_dte(ts)
                entry_p = bs_price(spot, strike, dte, "CE")
                entry_idx = i
                in_pos = True
            elif f_pe:
                opt_type = "PE"
                strike = int(round(spot / 100.0) * 100)
                dte = get_dte(ts)
                entry_p = bs_price(spot, strike, dte, "PE")
                entry_idx = i
                in_pos = True

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# 5. STRATEGY 3: DECOUPLED ASYMMETRIC STRANGLE (DAS, 1-MIN)
# -------------------------------------------------------------
def run_full_das(std_th=45.0, vel_th=50.0, win_tgt=0.80, lose_sp=0.25, hold_m=90, offset=200):
    trades = []
    in_pos = False
    ce_entry = 0.0
    pe_entry = 0.0
    ce_strike = 0
    pe_strike = 0
    ce_exited = False
    pe_exited = False
    ce_exit_p = 0.0
    pe_exit_p = 0.0
    entry_idx = 0
    last_exit_idx = -100

    for i in range(30, n_1m):
        tm_min = mins_1m[i]
        d = dates_1m[i]
        ts = df_1m.index[i]
        c = closes_1m[i]

        if in_pos:
            bars = i - entry_idx
            dte = get_dte(ts)
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")

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
                ce_pnl = (p_c - ce_entry) * LOT_SIZE
                pe_pnl = (p_p - pe_entry) * LOT_SIZE
                gross = ce_pnl + pe_pnl
                net = gross - FEE_STRANGLE
                trades.append({
                    "date": str(d), "strategy": "Decoupled Asymmetric Strangle",
                    "strikes": f"{ce_strike}CE / {pe_strike}PE",
                    "entry_time": str(df_1m.index[entry_idx].time()), "exit_time": str(ts.time()),
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

            if rolling_std_20_1m[i] < std_th and abs(vel_3_1m[i]) >= vel_th:
                atm = int(round(c / 100.0) * 100)
                ce_strike = atm + offset
                pe_strike = atm - offset
                dte = get_dte(ts)
                ce_entry = bs_price(c, ce_strike, dte, "CE")
                pe_entry = bs_price(c, pe_strike, dte, "PE")
                in_pos = True
                entry_idx = i
                ce_exited = False
                pe_exited = False

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# 6. STRATEGY 4: HEDGED STRANGLE (HS, 1-MIN)
# -------------------------------------------------------------
def run_full_hedged(target_pct=0.70, stop_pct=0.30, hold_m=45, offset=200):
    trades = []
    in_pos = False
    entry_tot = 0.0
    entry_idx = 0
    ce_strike = 0
    pe_strike = 0

    for i in range(20, n_1m):
        tm_min = mins_1m[i]
        d = dates_1m[i]
        ts = df_1m.index[i]
        c = closes_1m[i]
        weekday = ts.weekday()

        if in_pos:
            bars = i - entry_idx
            dte = get_dte(ts)
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")
            cur_tot = c_ce + c_pe
            ret = (cur_tot - entry_tot) / entry_tot

            if ret >= target_pct or ret <= -stop_pct or bars >= hold_m or tm_min >= 915:
                gross = (cur_tot - entry_tot) * LOT_SIZE
                net = gross - FEE_STRANGLE
                trades.append({
                    "date": str(d), "strategy": "Hedged Strangle",
                    "strikes": f"{ce_strike}CE / {pe_strike}PE",
                    "entry_time": str(df_1m.index[entry_idx].time()), "exit_time": str(ts.time()),
                    "entry_premium": round(entry_tot, 2), "exit_premium": round(cur_tot, 2),
                    "return_pct": f"{round(ret * 100, 1)}%",
                    "gross_pnl": round(gross, 2), "friction": FEE_STRANGLE, "net_pnl": round(net, 2)
                })
                in_pos = False
        else:
            if tm_min != 570:  # 09:30 AM
                continue
            if weekday not in [1, 2]:  # Tuesday / Wednesday
                continue

            atm = int(round(c / 100.0) * 100)
            ce_strike = atm + offset
            pe_strike = atm - offset
            dte = get_dte(ts)
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")
            entry_tot = c_ce + c_pe
            entry_idx = i
            in_pos = True

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# RUN ALL AND GENERATE AUDIT
# -------------------------------------------------------------
print("\n[3/4] Running simulations across full historical periods...")

df_box = run_full_box()
df_gamma = run_full_gamma()
df_das = run_full_das()
df_hedged = run_full_hedged()

# Export full ledgers
df_box.to_csv(f"{RESULTS_DIR}/full_history_box_breakout_ledger.csv", index=False)
df_gamma.to_csv(f"{RESULTS_DIR}/full_history_gamma_squeeze_ledger.csv", index=False)
df_das.to_csv(f"{RESULTS_DIR}/full_history_das_ledger.csv", index=False)
df_hedged.to_csv(f"{RESULTS_DIR}/full_history_hedged_strangle_ledger.csv", index=False)

def summarize_strat(df, name, dataset_period):
    if df.empty:
        return {"Strategy": name, "Coverage": dataset_period, "Trades": 0, "Win Rate": "0.0%", "Profit Factor": 0.0, "Net PnL (INR)": "Rs. 0.00", "Avg Trade (INR)": "Rs. 0.00", "Max Drawdown": "Rs. 0.00"}
    w = df[df["net_pnl"] > 0]["net_pnl"]
    l = abs(df[df["net_pnl"] <= 0]["net_pnl"].sum())
    w_sum = w.sum()
    pf = round(w_sum / l, 2) if l > 0 else (99.0 if w_sum > 0 else 0.0)
    wr = round(len(w) / len(df) * 100.0, 1)
    tot = round(df["net_pnl"].sum(), 2)
    avg_t = round(tot / len(df), 2)
    
    cum = df["net_pnl"].cumsum()
    peak = cum.cummax()
    dd = (peak - cum).max()
    max_dd = round(dd, 2) if not np.isnan(dd) else 0.0

    return {
        "Strategy": name, "Coverage": dataset_period,
        "Trades": len(df), "Win Rate": f"{wr}%", "Profit Factor": pf,
        "Net PnL (INR)": f"Rs. {tot:,.2f}", "Avg Trade (INR)": f"Rs. {avg_t:,.2f}",
        "Max Drawdown": f"Rs. {max_dd:,.2f}"
    }

summary = [
    summarize_strat(df_box, "1. Directional Box Breakout (DBB)", "Aug 25 - Sep 28 (1-Min, 8,664 bars)"),
    summarize_strat(df_gamma, "2. Gamma Squeeze Momentum (GSM)", "July 1 - Sep 28 (5-Min, 4,652 bars)"),
    summarize_strat(df_das, "3. Decoupled Asymmetric Strangle (DAS)", "Aug 25 - Sep 28 (1-Min, 8,664 bars)"),
    summarize_strat(df_hedged, "4. Hedged Strangle (HS)", "Aug 25 - Sep 28 (1-Min, 8,664 bars)")
]

df_summary = pd.DataFrame(summary)
print("\n" + "=" * 105)
print("  BANKNIFTY FULL-DATASET PERFORMANCE SUMMARY (SYNTHETIC BLACK-SCHOLES)")
print("=" * 105)
print(tabulate(df_summary, headers="keys", tablefmt="grid", showindex=False))

print(f"\n[4/4] All 4 full-history ledgers saved in {RESULTS_DIR}/.")
