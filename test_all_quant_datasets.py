"""
Comprehensive Quantitative Engine Benchmark on ALL BankNIFTY Datasets:
1. Real Angel One SmartAPI Traded Contracts (Sept 21 - Sept 28, 2026)
2. Full Historical Synthetic Dataset (1-minute, 8,664 bars, Aug 25 - Sept 28)
3. Full Historical Synthetic Dataset (5-minute, 4,652 bars, July 1 - Sept 28)

Mathematical Models Tested:
- Model 1: Variance Risk Premium (VRP) & 9:20 AM Delta-Neutral Theta Harvest
- Model 2: Ornstein-Uhlenbeck (O-U) Stochastic Mean-Reversion (exploiting H < 0.40 anti-persistence)
- Model 3: Parkinson Volatility Diffusion Shock (two-legged asymmetric strangle during extreme sigma expansion)
"""

import os
import glob
import math
import numpy as np
import pandas as pd
from tabulate import tabulate

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_SINGLE = 45.0
RESULTS_DIR = "data/banknifty_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. MATHEMATICAL FORMULAS
# -------------------------------------------------------------
INV_SQRT_2 = 1.0 / math.sqrt(2.0)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))

def bs_price(S, K, T, opt_type="CE", r=0.07, sigma=0.18):
    T_c = max(1e-4, T)
    sqrt_T = math.sqrt(T_c)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_c) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm_cdf(d1) - K * math.exp(-r * T_c) * norm_cdf(d2)
    else:
        p = K * math.exp(-r * T_c) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(0.05, p)

def get_dte(ts):
    weekday = ts.weekday()
    target_exp = 2  # Wednesday expiry
    days_ahead = target_exp - weekday if weekday <= target_exp else 7 - (weekday - target_exp)
    mins = max(1.0, 930 - (ts.hour * 60 + ts.minute))
    frac = mins / 375.0
    d = max(0.01, frac * 0.5) if days_ahead == 0 else (days_ahead - 1) + frac
    return d / 365.0

def calc_parkinson(highs, lows, window=20):
    log_hl_sq = (np.log(np.maximum(1e-4, highs / lows))) ** 2
    factor = 1.0 / (4.0 * math.log(2.0))
    return np.sqrt(factor * pd.Series(log_hl_sq).rolling(window).mean().values) * 100.0

def eval_metrics(trades_df, title, dataset_name):
    if len(trades_df) == 0:
        return {"Model": title, "Dataset": dataset_name, "Trades": 0, "Win Rate": "0.0%", "PF": 0.0, "Net PnL (INR)": 0.0, "Avg Trade": 0.0, "Max DD": 0.0}
    wins = trades_df[trades_df["net_pnl"] > 0]["net_pnl"]
    losses = trades_df[trades_df["net_pnl"] <= 0]["net_pnl"]
    w_sum = float(wins.sum()) if len(wins) > 0 else 0.0
    l_sum = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    pf = round(w_sum / l_sum, 2) if l_sum > 0 else 99.0
    wr = round(len(wins) / len(trades_df) * 100.0, 1)
    tot = round(float(trades_df["net_pnl"].sum()), 2)
    avg_t = round(tot / len(trades_df), 2)
    cum = trades_df["net_pnl"].cumsum()
    peak = cum.cummax()
    dd = float((peak - cum).max()) if len(cum) > 0 else 0.0
    return {"Model": title, "Dataset": dataset_name, "Trades": len(trades_df), "Win Rate": f"{wr}%", "PF": pf, "Net PnL (INR)": tot, "Avg Trade": avg_t, "Max DD": round(dd, 2)}

# -------------------------------------------------------------
# 2. LOAD ALL DATASETS
# -------------------------------------------------------------
print("Loading real cached options and spot datasets...")
real_opt_files = glob.glob("data/banknifty_options_cache/*.csv")
opt_data = {}
for f in real_opt_files:
    sym = os.path.basename(f).replace(".csv", "")
    df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index("timestamp", inplace=True)
    opt_data[sym] = df

df_spot_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_spot_1m["timestamp"] = pd.to_datetime(df_spot_1m["timestamp"]).dt.tz_localize(None)
df_spot_1m.sort_values("timestamp", inplace=True)
df_spot_1m.reset_index(drop=True, inplace=True)

df_spot_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_spot_5m["timestamp"] = pd.to_datetime(df_spot_5m["timestamp"]).dt.tz_localize(None)
df_spot_5m.sort_values("timestamp", inplace=True)
df_spot_5m.reset_index(drop=True, inplace=True)

# =============================================================
# MODEL 1: VRP 9:20 AM DELTA-NEUTRAL HARVEST (OPTION SELLING)
# =============================================================
def run_model_1_vrp(dataset_type="REAL"):
    trades = []
    df = df_spot_1m.copy()
    if dataset_type == "REAL":
        df = df[(df["timestamp"] >= "2026-09-21") & (df["timestamp"] <= "2026-09-28 23:59:59")].reset_index(drop=True)

    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    for d in df["date"].unique():
        day_df = df[df["date"] == d].reset_index(drop=True)
        m920 = day_df[day_df["mins"] == 560]
        if len(m920) == 0:
            continue

        spot_entry = m920.iloc[0]["close"]
        atm_k = int(round(spot_entry / 100.0) * 100)
        ts_entry = m920.iloc[0]["timestamp"]
        dte_entry = get_dte(ts_entry)

        ce_name = f"BANKNIFTY29SEP26{atm_k}CE"
        pe_name = f"BANKNIFTY29SEP26{atm_k}PE"

        if dataset_type == "REAL":
            if ce_name not in opt_data or pe_name not in opt_data:
                continue
            if ts_entry not in opt_data[ce_name].index or ts_entry not in opt_data[pe_name].index:
                continue
            ce_entry = opt_data[ce_name].loc[ts_entry, "close"]
            pe_entry = opt_data[pe_name].loc[ts_entry, "close"]
        else:
            ce_entry = bs_price(spot_entry, atm_k, dte_entry, "CE")
            pe_entry = bs_price(spot_entry, atm_k, dte_entry, "PE")

        ce_sl = ce_entry * 1.25  # 25% SL on short
        pe_sl = pe_entry * 1.25

        ce_exited = False
        pe_exited = False
        ce_exit_p = 0.0
        pe_exit_p = 0.0

        rem = day_df[day_df["mins"] > 560].reset_index(drop=True)
        for j in range(len(rem)):
            s = rem.loc[j, "close"]
            ts = rem.loc[j, "timestamp"]
            tm = rem.loc[j, "mins"]
            dte = get_dte(ts)

            if dataset_type == "REAL":
                cur_ce = opt_data[ce_name].loc[ts, "close"] if ts in opt_data[ce_name].index else ce_entry
                cur_pe = opt_data[pe_name].loc[ts, "close"] if ts in opt_data[pe_name].index else pe_entry
            else:
                cur_ce = bs_price(s, atm_k, dte, "CE")
                cur_pe = bs_price(s, atm_k, dte, "PE")

            if not ce_exited and cur_ce >= ce_sl:
                ce_exited = True
                ce_exit_p = cur_ce
            if not pe_exited and cur_pe >= pe_sl:
                pe_exited = True
                pe_exit_p = cur_pe

            if tm >= 910 or (ce_exited and pe_exited):
                if not ce_exited:
                    ce_exit_p = cur_ce
                if not pe_exited:
                    pe_exit_p = cur_pe
                break

        ce_pnl = (ce_entry - ce_exit_p) * LOT_SIZE
        pe_pnl = (pe_entry - pe_exit_p) * LOT_SIZE
        net = ce_pnl + pe_pnl - FEE_STRANGLE
        trades.append({"date": str(d), "net_pnl": round(net, 2)})

    return pd.DataFrame(trades)

# =============================================================
# MODEL 2: O-U STOCHASTIC MEAN REVERSION (OPTION BUYING)
# =============================================================
def run_model_2_ou(dataset_type="REAL"):
    trades = []
    df = df_spot_1m.copy() if dataset_type == "REAL" else df_spot_5m.copy()
    if dataset_type == "REAL":
        df = df[(df["timestamp"] >= "2026-09-21") & (df["timestamp"] <= "2026-09-28 23:59:59")].reset_index(drop=True)

    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    closes = df["close"].values
    n = len(df)

    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    entry_idx = 0
    trades_today = 0
    cur_date = None

    for i in range(35, n):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        s = closes[i]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            bars = i - entry_idx
            if dataset_type == "REAL":
                opt_sym = f"BANKNIFTY29SEP26{opt_strike}{opt_type}"
                cur_p = opt_data[opt_sym].loc[ts, "close"] if (opt_sym in opt_data and ts in opt_data[opt_sym].index) else entry_p
            else:
                cur_p = bs_price(s, opt_strike, dte, opt_type)

            ret = (cur_p - entry_p) / max(0.1, entry_p)
            is_tp = ret >= 0.40
            is_sl = ret <= -0.15
            is_time = bars >= (45 if dataset_type == "REAL" else 9) or tm >= 915

            if is_tp or is_sl or is_time:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"date": str(d), "net_pnl": round(net, 2)})
                in_pos = False
                trades_today += 1
        else:
            if tm < 570 or tm > 870 or trades_today >= 2:
                continue

            x = closes[i-25:i]
            x_prev = x[:-1]
            dx = np.diff(x)
            slope, intercept = np.polyfit(x_prev, dx, 1)
            theta = -slope
            if theta <= 0:
                continue

            bar_mins = 1.0 if dataset_type == "REAL" else 5.0
            half_life = (math.log(2.0) / theta) * bar_mins

            if 10.0 <= half_life <= 45.0:
                mu = -intercept / slope
                sigma_eq = np.std(x)
                z = (s - mu) / max(1.0, sigma_eq)
                atm_k = int(round(s / 100.0) * 100)

                if z <= -1.8:  # Deeply oversold -> Buy CE
                    opt_strike = atm_k
                    opt_type = "CE"
                    if dataset_type == "REAL":
                        opt_sym = f"BANKNIFTY29SEP26{opt_strike}CE"
                        if opt_sym not in opt_data or ts not in opt_data[opt_sym].index:
                            continue
                        entry_p = opt_data[opt_sym].loc[ts, "close"]
                    else:
                        entry_p = bs_price(s, opt_strike, dte, opt_type)

                    if entry_p > 0:
                        in_pos = True
                        entry_idx = i
                elif z >= 1.8:  # Deeply overbought -> Buy PE
                    opt_strike = atm_k
                    opt_type = "PE"
                    if dataset_type == "REAL":
                        opt_sym = f"BANKNIFTY29SEP26{opt_strike}PE"
                        if opt_sym not in opt_data or ts not in opt_data[opt_sym].index:
                            continue
                        entry_p = opt_data[opt_sym].loc[ts, "close"]
                    else:
                        entry_p = bs_price(s, opt_strike, dte, opt_type)

                    if entry_p > 0:
                        in_pos = True
                        entry_idx = i

    return pd.DataFrame(trades)

# =============================================================
# MODEL 3: PARKINSON DIFFUSION EXPANSION (ASYMMETRIC STRANGLE)
# =============================================================
def run_model_3_parkinson(dataset_type="REAL"):
    trades = []
    df = df_spot_1m.copy() if dataset_type == "REAL" else df_spot_1m.copy()
    if dataset_type == "REAL":
        df = df[(df["timestamp"] >= "2026-09-21") & (df["timestamp"] <= "2026-09-28 23:59:59")].reset_index(drop=True)

    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    pv = calc_parkinson(df["high"].values, df["low"].values, window=20)
    closes = df["close"].values
    n = len(df)

    in_pos = False
    ce_strike = 0
    pe_strike = 0
    ce_entry = 0.0
    pe_entry = 0.0
    ce_exited = False
    pe_exited = False
    ce_exit_p = 0.0
    pe_exit_p = 0.0
    entry_idx = 0
    trades_today = 0
    cur_date = None

    for i in range(35, n):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        s = closes[i]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            bars = i - entry_idx
            if dataset_type == "REAL":
                ce_sym = f"BANKNIFTY29SEP26{ce_strike}CE"
                pe_sym = f"BANKNIFTY29SEP26{pe_strike}PE"
                c_ce = opt_data[ce_sym].loc[ts, "close"] if (ce_sym in opt_data and ts in opt_data[ce_sym].index) else ce_entry
                c_pe = opt_data[pe_sym].loc[ts, "close"] if (pe_sym in opt_data and ts in opt_data[pe_sym].index) else pe_entry
            else:
                c_ce = bs_price(s, ce_strike, dte, "CE")
                c_pe = bs_price(s, pe_strike, dte, "PE")

            if not ce_exited:
                r_ce = (c_ce - ce_entry) / max(0.1, ce_entry)
                if r_ce >= 0.80 or r_ce <= -0.15:
                    ce_exited = True
                    ce_exit_p = c_ce

            if not pe_exited:
                r_pe = (c_pe - pe_entry) / max(0.1, pe_entry)
                if r_pe >= 0.80 or r_pe <= -0.15:
                    pe_exited = True
                    pe_exit_p = c_pe

            if (ce_exited and pe_exited) or bars >= 45 or tm >= 915:
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                gross = (p_c - ce_entry + p_p - pe_entry) * LOT_SIZE
                net = gross - FEE_STRANGLE
                trades.append({"date": str(d), "net_pnl": round(net, 2)})
                in_pos = False
                trades_today += 1
        else:
            if tm < 585 or tm > 840 or trades_today >= 1:
                continue

            pv_cur = pv[i]
            pv_prev = pv[i-1]
            pv_hist = pv[max(0, i-60):i]
            pctl = np.sum(pv_hist < pv_cur) / len(pv_hist) if len(pv_hist) > 0 else 0.5

            # Extreme Volatility Squeeze (<25th percentile) followed by diffusion burst (>15% jump)
            if pctl < 0.30 and (pv_cur > pv_prev * 1.15):
                atm = int(round(s / 100.0) * 100)
                ce_strike = atm
                pe_strike = atm

                if dataset_type == "REAL":
                    ce_sym = f"BANKNIFTY29SEP26{ce_strike}CE"
                    pe_sym = f"BANKNIFTY29SEP26{pe_strike}PE"
                    if ce_sym not in opt_data or pe_sym not in opt_data:
                        continue
                    if ts not in opt_data[ce_sym].index or ts not in opt_data[pe_sym].index:
                        continue
                    ce_entry = opt_data[ce_sym].loc[ts, "close"]
                    pe_entry = opt_data[pe_sym].loc[ts, "close"]
                else:
                    ce_entry = bs_price(s, ce_strike, dte, "CE")
                    pe_entry = bs_price(s, pe_strike, dte, "PE")

                if ce_entry > 0 and pe_entry > 0:
                    in_pos = True
                    entry_idx = i
                    ce_exited = False
                    pe_exited = False

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# RUN AND COMPILE BENCHMARK ACROSS ALL DATASETS
# -------------------------------------------------------------
print("\n[Step 3] Running Mathematical Models across all datasets...")

# 1. Real Options Dataset
m1_real = run_model_1_vrp("REAL")
m2_real = run_model_2_ou("REAL")
m3_real = run_model_3_parkinson("REAL")

# 2. Full Synthetic Dataset
m1_synth = run_model_1_vrp("SYNTHETIC")
m2_synth = run_model_2_ou("SYNTHETIC")
m3_synth = run_model_3_parkinson("SYNTHETIC")

# Export ledgers
m1_real.to_csv(f"{RESULTS_DIR}/math_vrp_real_ledger.csv", index=False)
m2_real.to_csv(f"{RESULTS_DIR}/math_ou_real_ledger.csv", index=False)
m3_real.to_csv(f"{RESULTS_DIR}/math_parkinson_real_ledger.csv", index=False)

m1_synth.to_csv(f"{RESULTS_DIR}/math_vrp_synth_ledger.csv", index=False)
m2_synth.to_csv(f"{RESULTS_DIR}/math_ou_synth_ledger.csv", index=False)
m3_synth.to_csv(f"{RESULTS_DIR}/math_parkinson_synth_ledger.csv", index=False)

summary = [
    # REAL OPTIONS DATASET (SEPT 21 - 28)
    eval_metrics(m1_real, "Model 1: VRP 9:20 Delta-Neutral Harvest (Short)", "Real SmartAPI Options"),
    eval_metrics(m2_real, "Model 2: O-U Stochastic Mean-Reversion (Long)", "Real SmartAPI Options"),
    eval_metrics(m3_real, "Model 3: Parkinson Volatility Squeeze (Long)", "Real SmartAPI Options"),
    
    # FULL SYNTHETIC MULTI-MONTH DATASET
    eval_metrics(m1_synth, "Model 1: VRP 9:20 Delta-Neutral Harvest (Short)", "Full Synthetic (Aug-Sep)"),
    eval_metrics(m2_synth, "Model 2: O-U Stochastic Mean-Reversion (Long)", "Full Synthetic (July-Sep)"),
    eval_metrics(m3_synth, "Model 3: Parkinson Volatility Squeeze (Long)", "Full Synthetic (Aug-Sep)")
]

df_res = pd.DataFrame(summary)
print("\n" + "=" * 115)
print("       MASTER BENCHMARK: MATHEMATICAL QUANTITATIVE MODELS ON ALL BANKNIFTY DATASETS")
print("=" * 115)
print(tabulate(df_res, headers="keys", tablefmt="grid", showindex=False))
