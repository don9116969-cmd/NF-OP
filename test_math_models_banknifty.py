"""
Mathematical & Quantitative Models for BankNIFTY Options Trading:
1. Hurst Exponent Regime-Switching Engine (Fractal Dimension & Persistence)
2. Ornstein-Uhlenbeck (O-U) Stochastic Mean-Reversion with Half-Life Filter
3. Parkinson High-Low Volatility Squeeze & Volatility Expansion Model
4. 1D Kalman Filter Dynamic State-Space Innovation Engine

Evaluated on BankNIFTY 1-min (8,664 bars) and 5-min (4,652 bars) with 30 Qty lot size.
"""

import math
import numpy as np
import pandas as pd
from tabulate import tabulate

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_SINGLE = 45.0

# -------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------
print("Loading BankNIFTY datasets...")
df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.sort_values("timestamp", inplace=True)
df_1m.reset_index(drop=True, inplace=True)

df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)

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

def eval_trades(trades, name):
    if len(trades) == 0:
        return {"Model": name, "Trades": 0, "Win Rate": "0.0%", "PF": 0.0, "Net PnL": 0.0, "Avg Trade": 0.0, "Max DD": 0.0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net"] > 0]["net"]
    losses = tdf[tdf["net"] <= 0]["net"]
    w_sum = wins.sum() if len(wins) > 0 else 0.0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0.0
    pf = round(w_sum / l_sum, 2) if l_sum > 0 else 99.0
    wr = round(len(wins) / len(tdf) * 100.0, 1)
    tot = round(tdf["net"].sum(), 2)
    avg_t = round(tot / len(tdf), 2)
    cum = tdf["net"].cumsum()
    peak = cum.cummax()
    dd = round(float((peak - cum).max()), 2)
    return {"Model": name, "Trades": len(tdf), "Win Rate": f"{wr}%", "PF": pf, "Net PnL": tot, "Avg Trade": avg_t, "Max DD": dd}

# =============================================================
# MODEL 1: HURST EXPONENT (H) REGIME-SWITCHING STRANGLE ENGINE
# =============================================================
# Rescaled range (R/S) algorithm to compute Hurst exponent over rolling window
def compute_hurst(ts_series, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts_series[lag:], ts_series[:-lag])) for lag in lags]
    # Filter non-zero
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 3:
        return 0.5
    reg = np.polyfit(np.log([v[0] for v in valid]), np.log([v[1] for v in valid]), 1)
    return reg[0]

def test_hurst_model():
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    closes = df["close"].values

    trades = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    trades_today = 0
    cur_date = None

    # Rolling window of 60 bars (5 hours) to compute Hurst
    for i in range(60, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        c = closes[i]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            cur_p = bs_price(c, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= 0.50
            is_sl = ret <= -0.15
            is_eod = tm >= 915

            if is_tp or is_sl or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"net": net})
                in_pos = False
                trades_today += 1
        else:
            if tm < 585 or tm > 870 or trades_today >= 1:
                continue

            window = closes[i-40:i]
            H = compute_hurst(window, max_lag=16)

            # Mathematical persistence: H > 0.60 indicates strong trend continuation
            # Combine with price velocity relative to 20-bar mean
            mu = np.mean(window[-20:])
            std = np.std(window[-20:])
            z_score = (c - mu) / max(1.0, std)

            if H > 0.62 and z_score > 1.8:
                opt_type = "CE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                in_pos = True
            elif H > 0.62 and z_score < -1.8:
                opt_type = "PE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                in_pos = True

    return eval_trades(trades, "1. Hurst Exponent (H > 0.62) Persistent Trend Model")

# =============================================================
# MODEL 2: ORNSTEIN-UHLENBECK (O-U) MEAN REVERSION WITH HALF-LIFE
# =============================================================
# Fits dX_t = theta * (mu - X_t) dt + sigma * dW_t via AR(1) regression
def test_ornstein_uhlenbeck():
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    closes = df["close"].values

    trades = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    trades_today = 0
    cur_date = None

    for i in range(50, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        c = closes[i]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            cur_p = bs_price(c, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= 0.35
            is_sl = ret <= -0.15
            is_eod = tm >= 915

            if is_tp or is_sl or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"net": net})
                in_pos = False
                trades_today += 1
        else:
            if tm < 570 or tm > 870 or trades_today >= 1:
                continue

            # Linear regression of X_t - X_{t-1} on X_{t-1} over 30 bars
            x = closes[i-30:i]
            x_prev = x[:-1]
            dx = np.diff(x)
            slope, intercept = np.polyfit(x_prev, dx, 1)

            # theta = -slope
            theta = -slope
            if theta <= 0:
                continue  # Not mean reverting

            # Half life = ln(2) / theta in 5-min bars
            half_life_bars = math.log(2.0) / theta
            half_life_mins = half_life_bars * 5.0

            # Only trade if half-life is fast: between 15 mins and 45 mins!
            if 15.0 <= half_life_mins <= 45.0:
                mu = -intercept / slope
                sigma_eq = np.std(x)
                z = (c - mu) / max(1.0, sigma_eq)

                # If price is 2 standard deviations below mean, buy Call (betting on reversion)
                if z <= -2.0:
                    opt_type = "CE"
                    opt_strike = int(round(c / 100.0) * 100)
                    entry_p = bs_price(c, opt_strike, dte, opt_type)
                    in_pos = True
                # If price is 2 standard deviations above mean, buy Put
                elif z >= 2.0:
                    opt_type = "PE"
                    opt_strike = int(round(c / 100.0) * 100)
                    entry_p = bs_price(c, opt_strike, dte, opt_type)
                    in_pos = True

    return eval_trades(trades, "2. Ornstein-Uhlenbeck (O-U) Half-Life Reversion")

# =============================================================
# MODEL 3: PARKINSON HIGH-LOW VOLATILITY COMPRESSION & EXPANSION
# =============================================================
def test_parkinson_volatility():
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    # Parkinson Volatility = sqrt( 1/(4*ln(2)) * sum( (ln(H/L))^2 ) )
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    log_hl_sq = (np.log(np.maximum(1e-4, h / l))) ** 2
    factor = 1.0 / (4.0 * math.log(2.0))
    parkinson_20 = np.sqrt(factor * pd.Series(log_hl_sq).rolling(20).mean().values) * 100.0

    trades = []
    in_pos = False
    ce_strike = 0
    pe_strike = 0
    ce_entry = 0.0
    pe_entry = 0.0
    entry_idx = 0
    trades_today = 0
    cur_date = None

    for i in range(30, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        spot = c[i]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            c_ce = bs_price(spot, ce_strike, dte, "CE")
            c_pe = bs_price(spot, pe_strike, dte, "PE")
            cur_tot = c_ce + c_pe
            ret = (cur_tot - (ce_entry + pe_entry)) / (ce_entry + pe_entry)

            is_tp = ret >= 0.40
            is_sl = ret <= -0.15
            is_eod = tm >= 915 or (i - entry_idx) >= 12  # max 60 mins

            if is_tp or is_sl or is_eod:
                net = (cur_tot - (ce_entry + pe_entry)) * LOT_SIZE - FEE_STRANGLE
                trades.append({"net": net})
                in_pos = False
                trades_today += 1
        else:
            if tm < 585 or tm > 840 or trades_today >= 1:
                continue

            pv = parkinson_20[i]
            prev_pv = parkinson_20[i-1]
            pv_roll = parkinson_20[max(0, i-60):i]
            pv_10th = np.percentile(pv_roll, 15)  # 15th percentile extreme compression

            # Volatility Squeeze breakout: PV was compressed in bottom 15%, now expanding
            if prev_pv <= pv_10th and pv > (pv_10th * 1.3):
                atm = int(round(spot / 100.0) * 100)
                ce_strike = atm
                pe_strike = atm
                ce_entry = bs_price(spot, ce_strike, dte, "CE")
                pe_entry = bs_price(spot, pe_strike, dte, "PE")
                in_pos = True
                entry_idx = i

    return eval_trades(trades, "3. Parkinson High-Low Volatility Squeeze Strangle")

# =============================================================
# MODEL 4: 1D KALMAN FILTER DYNAMIC STATE-SPACE INNOVATION
# =============================================================
def test_kalman_filter():
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    closes = df["close"].values

    # Kalman Filter State: x = price, P = estimation error covariance
    # Q = process noise variance, R = measurement noise variance
    Q = 1e-4
    R = 0.01

    trades = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    trades_today = 0
    cur_date = None

    # Track Kalman per session
    x_hat = closes[0]
    P = 1.0

    for i in range(20, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        z = closes[i]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0
            x_hat = z
            P = 1.0

        # Time update (Predict)
        x_hat_minus = x_hat
        P_minus = P + Q

        # Measurement update (Correct)
        K = P_minus / (P_minus + R)
        innovation = z - x_hat_minus
        x_hat = x_hat_minus + K * innovation
        P = (1.0 - K) * P_minus

        # Rolling std of innovations over last 20 bars
        if in_pos:
            cur_p = bs_price(z, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= 0.50
            is_sl = ret <= -0.15
            is_eod = tm >= 915

            if is_tp or is_sl or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"net": net})
                in_pos = False
                trades_today += 1
        else:
            if tm < 570 or tm > 870 or trades_today >= 1:
                continue

            # If innovation is positive & large (> 120 pts beyond predicted state)
            if innovation > 120.0:
                opt_type = "CE"
                opt_strike = int(round(z / 100.0) * 100)
                entry_p = bs_price(z, opt_strike, dte, opt_type)
                in_pos = True
            elif innovation < -120.0:
                opt_type = "PE"
                opt_strike = int(round(z / 100.0) * 100)
                entry_p = bs_price(z, opt_strike, dte, opt_type)
                in_pos = True

    return eval_trades(trades, "4. 1D Kalman Filter State-Space Innovation Engine")

# -------------------------------------------------------------
# RUN ALL 4 MATHEMATICAL MODELS
# -------------------------------------------------------------
print("\n" + "=" * 95)
print("  EVALUATING QUANTITATIVE & MATHEMATICAL MODELS ON HISTORICAL DATA")
print("=" * 95)

results = [
    test_hurst_model(),
    test_ornstein_uhlenbeck(),
    test_parkinson_volatility(),
    test_kalman_filter()
]

df_res = pd.DataFrame(results)
print("\n" + tabulate(df_res, headers="keys", tablefmt="grid", showindex=False))
