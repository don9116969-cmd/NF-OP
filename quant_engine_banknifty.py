"""
BankNIFTY Pure Mathematical & Quantitative Trading Engine:
1. Continuous Parkinson High-Low Volatility Squeeze & Diffusion Acceleration
2. Rolling Hurst Exponent (H) Fractal Regime Classifier (Persistent vs Anti-Persistent vs Noise)
3. Ornstein-Uhlenbeck (O-U) Stochastic Process with Calibrated Half-Life (tau_1/2) Filter
4. Dynamic Convex Strangle & Statistical Fade Execution

Evaluated on:
- Real Angel One SmartAPI Traded Options Dataset (Sept 21 - Sept 28)
- Full Historical Multi-Month Synthetic Dataset (Aug 25 - Sept 28, 8,664 1-min bars & July 1 - Sept 28, 4,652 5-min bars)
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

print("=" * 95)
print("  BANKNIFTY QUANTITATIVE MATHEMATICAL ENGINE BENCHMARK (REAL & SYNTHETIC)")
print("=" * 95)

# -------------------------------------------------------------
# 1. MATHEMATICAL CORE FUNCTIONS
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

# Parkinson Volatility Estimator
def calc_parkinson(highs, lows, window=20):
    log_hl_sq = (np.log(np.maximum(1e-4, highs / lows))) ** 2
    factor = 1.0 / (4.0 * math.log(2.0))
    return np.sqrt(factor * pd.Series(log_hl_sq).rolling(window).mean().values) * 100.0

# Hurst Exponent (R/S method)
def calc_hurst(series, max_lag=16):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 3:
        return 0.5
    reg = np.polyfit(np.log([v[0] for v in valid]), np.log([v[1] for v in valid]), 1)
    return reg[0]

# Metrics evaluator
def eval_metrics(trades_df, title):
    if len(trades_df) == 0:
        return {"Setup": title, "Trades": 0, "Win Rate": "0.0%", "PF": 0.0, "Net PnL (INR)": 0.0, "Avg Trade": 0.0, "Max DD": 0.0}
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
    return {"Setup": title, "Trades": len(trades_df), "Win Rate": f"{wr}%", "PF": pf, "Net PnL (INR)": tot, "Avg Trade": avg_t, "Max DD": round(dd, 2)}

# -------------------------------------------------------------
# 2. RUN ON REAL ANGEL ONE SMARTAPI OPTIONS DATASET (SEPT 21-28)
# -------------------------------------------------------------
print("\n[Step 1] Loading Real SmartAPI Traded Contracts (Sept 21-28)...")
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

# Spot 1-minute
df_spot_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_spot_1m["timestamp"] = pd.to_datetime(df_spot_1m["timestamp"]).dt.tz_localize(None)
df_spot_1m.sort_values("timestamp", inplace=True)
df_spot_1m.reset_index(drop=True, inplace=True)
df_spot_1m.set_index("timestamp", inplace=True)

# Filter spot for the real test week (2026-09-21 to 2026-09-28)
real_spot = df_spot_1m.loc["2026-09-21":"2026-09-28"].copy()
real_n = len(real_spot)
real_h = real_spot["high"].values
real_l = real_spot["low"].values
real_c = real_spot["close"].values
real_pv = calc_parkinson(real_h, real_l, window=20)
real_mins = np.array([t.hour * 60 + t.minute for t in real_spot.index.time])
real_dates = real_spot.index.date

def run_quant_real_options():
    trades = []
    in_pos = False
    mode = ""
    ce_sym = ""
    pe_sym = ""
    ce_entry = 0.0
    pe_entry = 0.0
    entry_p = 0.0
    entry_idx = 0
    trades_today = 0
    cur_date = None

    for i in range(40, real_n):
        ts = real_spot.index[i]
        d = real_dates[i]
        tm = real_mins[i]
        spot = real_c[i]

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            bars = i - entry_idx
            if mode == "CONVEX_STRANGLE":
                cur_ce = opt_data[ce_sym].loc[ts, "close"] if ts in opt_data[ce_sym].index else ce_entry
                cur_pe = opt_data[pe_sym].loc[ts, "close"] if ts in opt_data[pe_sym].index else pe_entry

                r_ce = (cur_ce - ce_entry) / max(0.1, ce_entry)
                r_pe = (cur_pe - pe_entry) / max(0.1, pe_entry)

                is_ce_tp = r_ce >= 0.80
                is_pe_tp = r_pe >= 0.80
                is_sl = (cur_ce + cur_pe - (ce_entry + pe_entry)) / (ce_entry + pe_entry) <= -0.15
                is_time = bars >= 45 or tm >= 915

                if is_ce_tp or is_pe_tp or is_sl or is_time:
                    gross = (cur_ce - ce_entry + cur_pe - pe_entry) * LOT_SIZE
                    net = gross - FEE_STRANGLE
                    reason = "Target Hit" if (is_ce_tp or is_pe_tp) else ("Stop Loss" if is_sl else "Time Exit")
                    trades.append({
                        "date": str(d), "model": "Convex Volatility Expansion (Real Options)",
                        "contract": f"{ce_sym} + {pe_sym}", "entry_time": str(real_spot.index[entry_idx].time()),
                        "exit_time": str(ts.time()), "entry_price": round(ce_entry + pe_entry, 2),
                        "exit_price": round(cur_ce + cur_pe, 2), "net_pnl": round(net, 2), "reason": reason
                    })
                    in_pos = False
                    trades_today += 1
            elif mode == "OU_REVERSION":
                cur_opt = opt_data[ce_sym].loc[ts, "close"] if ts in opt_data[ce_sym].index else entry_p
                ret = (cur_opt - entry_p) / max(0.1, entry_p)
                is_tp = ret >= 0.40
                is_sl = ret <= -0.18
                is_time = bars >= 45 or tm >= 915

                if is_tp or is_sl or is_time:
                    gross = (cur_opt - entry_p) * LOT_SIZE
                    net = gross - FEE_SINGLE
                    reason = "Target Hit" if is_tp else ("Stop Loss" if is_sl else "Time Exit")
                    trades.append({
                        "date": str(d), "model": "O-U Stochastic Reversion (Real Options)",
                        "contract": ce_sym, "entry_time": str(real_spot.index[entry_idx].time()),
                        "exit_time": str(ts.time()), "entry_price": round(entry_p, 2),
                        "exit_price": round(cur_opt, 2), "net_pnl": round(net, 2), "reason": reason
                    })
                    in_pos = False
                    trades_today += 1
        else:
            if tm < 585 or tm > 840 or trades_today >= 2:
                continue

            # 1. Compute Hurst on 30-bar window
            w_closes = real_c[i-30:i]
            H = calc_hurst(w_closes, max_lag=12)

            # 2. Compute Parkinson Volatility Percentile
            pv_cur = real_pv[i]
            pv_hist = real_pv[max(0, i-60):i]
            pv_pctl = np.sum(pv_hist < pv_cur) / len(pv_hist) if len(pv_hist) > 0 else 0.5

            # 3. Fit O-U SDE
            x = w_closes
            x_prev = x[:-1]
            dx = np.diff(x)
            slope, intercept = np.polyfit(x_prev, dx, 1)
            theta = -slope
            half_life = (math.log(2.0) / theta) if theta > 0 else 999.0

            atm_k = int(round(spot / 100.0) * 100)
            ce_name = f"BANKNIFTY29SEP26{atm_k}CE"
            pe_name = f"BANKNIFTY29SEP26{atm_k}PE"

            # Check if option contracts exist in cache
            if ce_name not in opt_data or pe_name not in opt_data:
                continue
            if ts not in opt_data[ce_name].index or ts not in opt_data[pe_name].index:
                continue

            # SETUP 1: CONVEX VOLATILITY EXPANSION (H > 0.55 & Parkinson Vol Squeeze Burst)
            if H > 0.55 and pv_pctl < 0.25 and (real_pv[i] > real_pv[i-1] * 1.15):
                ce_sym = ce_name
                pe_sym = pe_name
                ce_entry = opt_data[ce_sym].loc[ts, "close"]
                pe_entry = opt_data[pe_sym].loc[ts, "close"]
                if ce_entry > 0 and pe_entry > 0:
                    mode = "CONVEX_STRANGLE"
                    in_pos = True
                    entry_idx = i

            # SETUP 2: O-U STOCHASTIC MEAN REVERSION (H < 0.48 & tau_1/2 <= 40m & |Z| >= 1.8)
            elif H < 0.48 and (10.0 <= half_life <= 40.0):
                mu = -intercept / slope
                sigma_eq = np.std(x)
                z = (spot - mu) / max(1.0, sigma_eq)

                if z <= -1.8:  # Deeply oversold -> Buy CE for fast reversion
                    ce_sym = ce_name
                    entry_p = opt_data[ce_sym].loc[ts, "close"]
                    if entry_p > 0:
                        mode = "OU_REVERSION"
                        in_pos = True
                        entry_idx = i
                elif z >= 1.8:  # Deeply overbought -> Buy PE for fast reversion
                    ce_sym = pe_name
                    entry_p = opt_data[ce_sym].loc[ts, "close"]
                    if entry_p > 0:
                        mode = "OU_REVERSION"
                        in_pos = True
                        entry_idx = i

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# 3. RUN ON FULL MULTI-MONTH SYNTHETIC DATASET (8,664 & 4,652 BARS)
# -------------------------------------------------------------
print("\n[Step 2] Running Quant Engine on Full Multi-Month Synthetic Dataset...")
df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)

synth_n = len(df_5m)
synth_h = df_5m["high"].values
synth_l = df_5m["low"].values
synth_c = df_5m["close"].values
synth_pv = calc_parkinson(synth_h, synth_l, window=20)
synth_mins = np.array([t.hour * 60 + t.minute for t in df_5m["timestamp"].dt.time])
synth_dates = df_5m["timestamp"].dt.date

def run_quant_synthetic():
    trades = []
    in_pos = False
    mode = ""
    opt_strike = 0
    opt_type = ""
    ce_entry = 0.0
    pe_entry = 0.0
    entry_p = 0.0
    entry_idx = 0
    trades_today = 0
    cur_date = None

    for i in range(40, synth_n):
        ts = df_5m.loc[i, "timestamp"]
        d = synth_dates[i]
        tm = synth_mins[i]
        spot = synth_c[i]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            bars = i - entry_idx
            if mode == "CONVEX_STRANGLE":
                cur_ce = bs_price(spot, opt_strike, dte, "CE")
                cur_pe = bs_price(spot, opt_strike, dte, "PE")

                r_ce = (cur_ce - ce_entry) / max(0.1, ce_entry)
                r_pe = (cur_pe - pe_entry) / max(0.1, pe_entry)

                is_ce_tp = r_ce >= 0.80
                is_pe_tp = r_pe >= 0.80
                is_sl = (cur_ce + cur_pe - (ce_entry + pe_entry)) / (ce_entry + pe_entry) <= -0.15
                is_time = bars >= 9 or tm >= 915  # 9 bars = 45 mins

                if is_ce_tp or is_pe_tp or is_sl or is_time:
                    gross = (cur_ce - ce_entry + cur_pe - pe_entry) * LOT_SIZE
                    net = gross - FEE_STRANGLE
                    reason = "Target Hit" if (is_ce_tp or is_pe_tp) else ("Stop Loss" if is_sl else "Time Exit")
                    trades.append({
                        "date": str(d), "model": "Convex Volatility Expansion (Synthetic)",
                        "contract": f"{opt_strike}CE + {opt_strike}PE", "entry_time": str(df_5m.loc[entry_idx, "timestamp"].time()),
                        "exit_time": str(ts.time()), "entry_price": round(ce_entry + pe_entry, 2),
                        "exit_price": round(cur_ce + cur_pe, 2), "net_pnl": round(net, 2), "reason": reason
                    })
                    in_pos = False
                    trades_today += 1
            elif mode == "OU_REVERSION":
                cur_opt = bs_price(spot, opt_strike, dte, opt_type)
                ret = (cur_opt - entry_p) / max(0.1, entry_p)
                is_tp = ret >= 0.40
                is_sl = ret <= -0.18
                is_time = bars >= 9 or tm >= 915

                if is_tp or is_sl or is_time:
                    gross = (cur_opt - entry_p) * LOT_SIZE
                    net = gross - FEE_SINGLE
                    reason = "Target Hit" if is_tp else ("Stop Loss" if is_sl else "Time Exit")
                    trades.append({
                        "date": str(d), "model": "O-U Stochastic Reversion (Synthetic)",
                        "contract": f"{opt_strike}{opt_type}", "entry_time": str(df_5m.loc[entry_idx, "timestamp"].time()),
                        "exit_time": str(ts.time()), "entry_price": round(entry_p, 2),
                        "exit_price": round(cur_opt, 2), "net_pnl": round(net, 2), "reason": reason
                    })
                    in_pos = False
                    trades_today += 1
        else:
            if tm < 585 or tm > 840 or trades_today >= 2:
                continue

            w_closes = synth_c[i-30:i]
            H = calc_hurst(w_closes, max_lag=12)

            pv_cur = synth_pv[i]
            pv_hist = synth_pv[max(0, i-60):i]
            pv_pctl = np.sum(pv_hist < pv_cur) / len(pv_hist) if len(pv_hist) > 0 else 0.5

            x = w_closes
            x_prev = x[:-1]
            dx = np.diff(x)
            slope, intercept = np.polyfit(x_prev, dx, 1)
            theta = -slope
            half_life_bars = (math.log(2.0) / theta) if theta > 0 else 999.0
            half_life_mins = half_life_bars * 5.0

            atm_k = int(round(spot / 100.0) * 100)

            # SETUP 1: CONVEX VOLATILITY EXPANSION (H > 0.55 & Parkinson Vol Squeeze Burst)
            if H > 0.55 and pv_pctl < 0.25 and (synth_pv[i] > synth_pv[i-1] * 1.15):
                opt_strike = atm_k
                ce_entry = bs_price(spot, opt_strike, dte, "CE")
                pe_entry = bs_price(spot, opt_strike, dte, "PE")
                mode = "CONVEX_STRANGLE"
                in_pos = True
                entry_idx = i

            # SETUP 2: O-U STOCHASTIC MEAN REVERSION (H < 0.48 & tau_1/2 <= 40m & |Z| >= 1.8)
            elif H < 0.48 and (10.0 <= half_life_mins <= 40.0):
                mu = -intercept / slope
                sigma_eq = np.std(x)
                z = (spot - mu) / max(1.0, sigma_eq)

                if z <= -1.8:
                    opt_strike = atm_k
                    opt_type = "CE"
                    entry_p = bs_price(spot, opt_strike, dte, opt_type)
                    mode = "OU_REVERSION"
                    in_pos = True
                    entry_idx = i
                elif z >= 1.8:
                    opt_strike = atm_k
                    opt_type = "PE"
                    entry_p = bs_price(spot, opt_strike, dte, opt_type)
                    mode = "OU_REVERSION"
                    in_pos = True
                    entry_idx = i

    return pd.DataFrame(trades)

# Run evaluations
df_real_res = run_quant_real_options()
df_synth_res = run_quant_synthetic()

# Export ledgers
df_real_res.to_csv(f"{RESULTS_DIR}/quant_model_real_options_ledger.csv", index=False)
df_synth_res.to_csv(f"{RESULTS_DIR}/quant_model_synthetic_ledger.csv", index=False)

summary_rows = [
    eval_metrics(df_real_res, "1. Quant Mathematical Engine (Real SmartAPI Options, Sept 21-28)"),
    eval_metrics(df_synth_res, "2. Quant Mathematical Engine (Full 3-Month Synthetic Dataset)")
]

df_summary = pd.DataFrame(summary_rows)
print("\n" + "=" * 95)
print("  BANKNIFTY QUANTITATIVE MATHEMATICAL ENGINE PERFORMANCE SUMMARY")
print("=" * 95)
print(tabulate(df_summary, headers="keys", tablefmt="grid", showindex=False))

print("\nLedgers saved to:")
print(" - data/banknifty_results/quant_model_real_options_ledger.csv")
print(" - data/banknifty_results/quant_model_synthetic_ledger.csv")
