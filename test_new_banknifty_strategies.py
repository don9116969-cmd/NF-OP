"""
Backtesting New BankNIFTY-Specific Intraday Strategies:
1. 5 EMA Mean-Reversion Scalper (Subhashish Pani - 5-min)
2. Central Pivot Range (CPR) Width Breakout & Trend (5-min / Daily Pivot)
3. Supertrend (10, 2) + Session VWAP Pullback Trend Following (5-min)
4. Wednesday 1:30 PM 0-DTE Expiry Gamma Blast (1-min)

All tested with 30 Qty BankNIFTY lot size and statutory transaction costs.
"""

import math
import numpy as np
import pandas as pd
from tabulate import tabulate

LOT_SIZE = 30
FEE_SINGLE = 45.0
FEE_STRANGLE = 80.0

# -------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------
print("Loading BankNIFTY datasets...")
df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)

df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.sort_values("timestamp", inplace=True)
df_1m.reset_index(drop=True, inplace=True)

# -------------------------------------------------------------
# C-Level Black-Scholes Helper
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

def eval_trades(trades, name):
    if len(trades) == 0:
        return {"Strategy": name, "Trades": 0, "Win Rate": "0.0%", "PF": 0.0, "Net PnL": 0.0, "Avg Trade": 0.0, "Max DD": 0.0}
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
    return {"Strategy": name, "Trades": len(tdf), "Win Rate": f"{wr}%", "PF": pf, "Net PnL": tot, "Avg Trade": avg_t, "Max DD": dd}

# =============================================================
# STRATEGY 1: 5 EMA REVERSAL (POWER OF STOCKS - SUBHASHISH PANI)
# =============================================================
def test_5ema_strategy():
    df = df_5m.copy()
    df["ema5"] = df["close"].ewm(span=5).mean()
    df["time"] = df["timestamp"].dt.time
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    trades = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    entry_idx = 0
    sl_spot = 0.0
    tp_pct = 0.50
    sl_pct = 0.20
    trades_today = 0
    cur_date = None

    for i in range(10, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        c = df.loc[i, "close"]
        h = df.loc[i, "high"]
        l = df.loc[i, "low"]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            cur_p = bs_price(c, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= tp_pct
            is_sl = ret <= -sl_pct
            # Spot level SL
            is_spot_sl = (opt_type == "PE" and c > sl_spot) or (opt_type == "CE" and c < sl_spot)
            is_eod = tm >= 915

            if is_tp or is_sl or is_spot_sl or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"net": net, "date": str(d)})
                in_pos = False
                trades_today += 1
        else:
            if tm < 570 or tm > 870 or trades_today >= 2:
                continue

            prev_l = df.loc[i-1, "low"]
            prev_h = df.loc[i-1, "high"]
            prev_ema = df.loc[i-1, "ema5"]

            # Alert candle for PE: previous candle low is completely above 5 EMA
            if prev_l > prev_ema and l < prev_l:
                opt_type = "PE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                sl_spot = prev_h
                in_pos = True
                entry_idx = i

            # Alert candle for CE: previous candle high is completely below 5 EMA
            elif prev_h < prev_ema and h > prev_h:
                opt_type = "CE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                sl_spot = prev_l
                in_pos = True
                entry_idx = i

    return eval_trades(trades, "1. BankNIFTY 5-EMA Mean-Reversion Scalp")

# =============================================================
# STRATEGY 2: CENTRAL PIVOT RANGE (CPR) BREAKOUT (GOMATHI SHANKAR)
# =============================================================
def test_cpr_strategy():
    # Calculate daily high, low, close to get daily CPR
    df_daily = df_5m.groupby(df_5m["timestamp"].dt.date).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).shift(1)  # previous day's metrics

    df_daily["pivot"] = (df_daily["high"] + df_daily["low"] + df_daily["close"]) / 3.0
    df_daily["bc"] = (df_daily["high"] + df_daily["low"]) / 2.0
    df_daily["tc"] = (df_daily["pivot"] - df_daily["bc"]) + df_daily["pivot"]
    df_daily["cpr_top"] = np.maximum(df_daily["tc"], df_daily["bc"])
    df_daily["cpr_bot"] = np.minimum(df_daily["tc"], df_daily["bc"])
    df_daily["cpr_width_pct"] = (df_daily["cpr_top"] - df_daily["cpr_bot"]) / df_daily["pivot"] * 100.0

    cpr_dict = df_daily.to_dict(orient="index")

    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    trades = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    entry_idx = 0
    trades_today = 0
    cur_date = None

    for i in range(10, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        c = df.loc[i, "close"]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if d not in cpr_dict or np.isnan(cpr_dict[d]["cpr_top"]):
            continue

        cpr_top = cpr_dict[d]["cpr_top"]
        cpr_bot = cpr_dict[d]["cpr_bot"]
        cpr_w = cpr_dict[d]["cpr_width_pct"]

        if in_pos:
            cur_p = bs_price(c, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= 0.40
            is_sl = ret <= -0.20
            # If price falls back inside CPR, exit
            is_cpr_sl = (opt_type == "CE" and c < cpr_top) or (opt_type == "PE" and c > cpr_bot)
            is_eod = tm >= 915

            if is_tp or is_sl or is_cpr_sl or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"net": net, "date": str(d)})
                in_pos = False
                trades_today += 1
        else:
            # We trade Narrow CPR breakout (cpr_width < 0.35%) between 09:30 and 11:30
            if tm < 570 or tm > 690 or trades_today >= 1:
                continue
            if cpr_w > 0.40:
                continue  # Skip wide CPR (choppy day)

            prev_c = df.loc[i-1, "close"]
            # Bullish Breakout above Top CPR
            if prev_c <= cpr_top and c > (cpr_top + 30):
                opt_type = "CE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                in_pos = True
                entry_idx = i

            # Bearish Breakdown below Bottom CPR
            elif prev_c >= cpr_bot and c < (cpr_bot - 30):
                opt_type = "PE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                in_pos = True
                entry_idx = i

    return eval_trades(trades, "2. BankNIFTY Narrow CPR Breakout")

# =============================================================
# STRATEGY 3: SUPERTREND (10, 2) + VWAP PULLBACK SCALPER
# =============================================================
def test_supertrend_vwap():
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.date
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    # Calculate Session VWAP (typical price mean)
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = df.groupby("date")["typical"].expanding().mean().values

    # Calculate Supertrend (10, 2)
    n = 10
    mult = 2.0
    tr = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    atr = tr.rolling(n).mean().fillna(50.0).values

    upper_band = df["typical"].values + mult * atr
    lower_band = df["typical"].values - mult * atr
    st = np.zeros(len(df))
    direction = np.ones(len(df))  # 1 for Bullish, -1 for Bearish

    for i in range(1, len(df)):
        if df.loc[i, "close"] > upper_band[i-1]:
            direction[i] = 1
        elif df.loc[i, "close"] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]

    df["st_dir"] = direction

    trades = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    trades_today = 0
    cur_date = None

    for i in range(20, len(df)):
        d = df.loc[i, "date"]
        tm = df.loc[i, "mins"]
        c = df.loc[i, "close"]
        l = df.loc[i, "low"]
        h = df.loc[i, "high"]
        vwap = df.loc[i, "vwap"]
        st_dir = df.loc[i, "st_dir"]
        ts = df.loc[i, "timestamp"]
        dte = get_dte(ts)

        if d != cur_date:
            cur_date = d
            trades_today = 0

        if in_pos:
            cur_p = bs_price(c, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= 0.40
            is_sl = ret <= -0.15
            is_rev = (opt_type == "CE" and st_dir == -1) or (opt_type == "PE" and st_dir == 1)
            is_eod = tm >= 915

            if is_tp or is_sl or is_rev or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                trades.append({"net": net, "date": str(d)})
                in_pos = False
                trades_today += 1
        else:
            if tm < 570 or tm > 870 or trades_today >= 2:
                continue

            # Bullish pullback: Supertrend Bullish, price dipped to VWAP and bounced
            if st_dir == 1 and c > vwap and l <= (vwap + 25):
                opt_type = "CE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                in_pos = True

            # Bearish pullback: Supertrend Bearish, price rallied to VWAP and rejected
            elif st_dir == -1 and c < vwap and h >= (vwap - 25):
                opt_type = "PE"
                opt_strike = int(round(c / 100.0) * 100)
                entry_p = bs_price(c, opt_strike, dte, opt_type)
                in_pos = True

    return eval_trades(trades, "3. BankNIFTY Supertrend + VWAP Pullback")

# =============================================================
# STRATEGY 4: WEDNESDAY 1:30 PM 0-DTE EXPIRY GAMMA BLAST (1-MIN)
# =============================================================
def test_expiry_gamma_blast():
    df = df_1m.copy()
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    df["weekday"] = df["timestamp"].dt.weekday
    df["mins"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    trades = []
    # Loop over all Wednesdays in the 1-minute dataset
    wednesdays = df[df["weekday"] == 2]["date"].unique()

    for w_date in wednesdays:
        day_df = df[df["date"] == w_date].copy().reset_index(drop=True)
        # Look at the consolidation between 13:00 and 13:30 (780 to 810 mins)
        cons = day_df[(day_df["mins"] >= 780) & (day_df["mins"] <= 810)]
        if len(cons) < 25:
            continue

        h_30 = cons["high"].max()
        l_30 = cons["low"].min()
        post_130 = day_df[day_df["mins"] > 810].copy().reset_index(drop=True)

        in_pos = False
        opt_type = ""
        opt_strike = 0
        entry_p = 0.0

        for j in range(len(post_130)):
            c = post_130.loc[j, "close"]
            h = post_130.loc[j, "high"]
            l = post_130.loc[j, "low"]
            tm = post_130.loc[j, "mins"]
            ts = post_130.loc[j, "timestamp"]
            dte = get_dte(ts)

            if in_pos:
                cur_p = bs_price(c, opt_strike, dte, opt_type)
                ret = (cur_p - entry_p) / entry_p
                is_tp = ret >= 1.0  # +100% target (double)
                is_sl = ret <= -0.30  # -30% stop loss
                is_eod = tm >= 915

                if is_tp or is_sl or is_eod:
                    net = (cur_p - entry_p) * LOT_SIZE - FEE_SINGLE
                    trades.append({"net": net, "date": str(w_date)})
                    in_pos = False
                    break
            else:
                if h >= (h_30 + 35):
                    opt_type = "CE"
                    opt_strike = int(round(c / 100.0) * 100)
                    entry_p = bs_price(c, opt_strike, dte, opt_type)
                    in_pos = True
                elif l <= (l_30 - 35):
                    opt_type = "PE"
                    opt_strike = int(round(c / 100.0) * 100)
                    entry_p = bs_price(c, opt_strike, dte, opt_type)
                    in_pos = True

    return eval_trades(trades, "4. BankNIFTY 1:30 PM 0-DTE Expiry Gamma Blast")

# -------------------------------------------------------------
# RUN ALL 4 STRATEGIES AND COMPARE
# -------------------------------------------------------------
print("\n" + "=" * 95)
print("  EVALUATING NEW BANKNIFTY-SPECIFIC STRATEGIES ON HISTORICAL DATA")
print("=" * 95)

results = [
    test_5ema_strategy(),
    test_cpr_strategy(),
    test_supertrend_vwap(),
    test_expiry_gamma_blast()
]

df_res = pd.DataFrame(results)
print("\n" + tabulate(df_res, headers="keys", tablefmt="grid", showindex=False))
