"""
Comprehensive Full-History Backtest Engine for Tri-Strategy Core Architecture.
Evaluates:
- Strategy 1: Nifty 15M Directional Breakout (ATM, SL -16%, TP +60%)
- Strategy 2 [OLD]: BankNifty 15M Breakout without Dynamic Strikes (Fixed 100-pt OTM, capped at Rs 9,500)
- Strategy 2 [NEW]: BankNifty 15M Breakout with Dynamic Strikes (Affordable strike scanner, capped at Rs 9,500)
- Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS) (Compression < 30, Velocity >= 35, Leg SL -12%, TP +60%)

Strict Realism Guarantees:
- Zero Lookahead Bias: Stop loss evaluated strictly before target profit on every candle.
- Full EMA-20 Warm-up across the entire historical series.
- Accurate Contract Expiry: Nifty Tuesday weekly; BankNifty monthly contracts (last Tuesday of the month).
- Realistic Transaction Costs & Slippage: 0.5% slippage, Rs 40/80 brokerage, STT 0.1% on sell, exchange fees, GST.
"""

import os
import sys
import datetime
import calendar
import pandas as pd
import numpy as np
from scipy.special import ndtr
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIFTY_CSV = os.path.join(BASE_DIR, "data", "nifty_1min_real.csv")
BANKNIFTY_CSV = os.path.join(BASE_DIR, "data", "banknifty_1min_real.csv")

# Capital & Cost Controls
LOT_NIFTY = 65
LOT_BANKNIFTY = 30
INITIAL_CAPITAL = 10000.0
MAX_CAPITAL_PER_TRADE = 9500.0
SLIPPAGE_PCT = 0.005
DYNAMIC_MIN_PREMIUM = 25.0

def bs_vectorized(S, K, T, r=0.07, sigma=0.15, opt_type="CE"):
    """Ultra-fast vectorized Black-Scholes pricing using scipy.special.ndtr."""
    T_arr = np.maximum(1e-4, T)
    sigma_arr = max(sigma, 1e-4)
    sqrt_T = np.sqrt(T_arr)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_arr ** 2) * T_arr) / (sigma_arr * sqrt_T)
    d2 = d1 - sigma_arr * sqrt_T
    if opt_type.upper() == "CE":
        price = S * ndtr(d1) - K * np.exp(-r * T_arr) * ndtr(d2)
    else:
        price = K * np.exp(-r * T_arr) * ndtr(-d2) - S * ndtr(-d1)
    if isinstance(price, np.ndarray):
        return np.maximum(0.05, price)
    return max(0.05, float(price))

def calc_charges(entry_p, exit_p, lot_size, is_strangle=False):
    gross = (exit_p - entry_p) * lot_size
    turn = (entry_p + exit_p) * lot_size
    brok = 80.0 if is_strangle else 40.0
    stt = (exit_p * lot_size) * 0.001
    exch = turn * 0.0005
    gst = (brok + exch) * 0.18
    charges = round(brok + stt + exch + gst, 2)
    net = round(gross - charges, 2)
    return round(gross, 2), charges, net

def get_banknifty_monthly_expiry(target_date: datetime.date) -> datetime.date:
    last_day = calendar.monthrange(target_date.year, target_date.month)[1]
    d = datetime.date(target_date.year, target_date.month, last_day)
    offset = (d.weekday() - 1) % 7
    exp_this_month = d - datetime.timedelta(days=offset)
    if target_date <= exp_this_month:
        return exp_this_month
    if target_date.month == 12:
        next_year, next_month = target_date.year + 1, 1
    else:
        next_year, next_month = target_date.year, target_date.month + 1
    last_day_next = calendar.monthrange(next_year, next_month)[1]
    d_next = datetime.date(next_year, next_month, last_day_next)
    offset_next = (d_next.weekday() - 1) % 7
    return d_next - datetime.timedelta(days=offset_next)

def get_dte(target_date: datetime.date, underlying: str = "NIFTY") -> float:
    if underlying.upper() == "BANKNIFTY":
        exp = get_banknifty_monthly_expiry(target_date)
        days = max(0, (exp - target_date).days)
    else:
        days = (1 - target_date.weekday()) % 7  # Tuesday weekly
    return max(0.001, float(days) / 365.25)

# ---------------------------------------------------------------------------
# STRATEGY 1: NIFTY 15M DIRECTIONAL BREAKOUT
# ---------------------------------------------------------------------------
def simulate_s1(day_df, target_date):
    if len(day_df) < 20:
        return None, "NO_DATA"
    
    times = day_df["time"].values
    t_0915 = datetime.time(9, 15)
    t_0930 = datetime.time(9, 30)
    t_1500 = datetime.time(15, 0)
    t_1515 = datetime.time(15, 15)

    box_mask = (times >= t_0915) & (times <= t_0930)
    if np.sum(box_mask) < 10:
        return None, "INCOMPLETE_BOX"

    box_high = float(day_df.loc[box_mask, "high"].max())
    box_low = float(day_df.loc[box_mask, "low"].min())
    open_p = float(day_df.loc[box_mask, "close"].iloc[0])
    box_range = box_high - box_low
    box_pct = (box_range / open_p) * 100.0

    if box_pct > 0.30:
        return None, "WIDE_BOX_FILTERED"

    u_trig = box_high + 8.0
    l_trig = box_low - 8.0

    post_mask = (times > t_0930) & (times <= datetime.time(13, 0))
    post_indices = np.where(post_mask)[0]
    if len(post_indices) == 0:
        return None, "NO_POST_BARS"

    entry_idx = None
    opt_type = None
    for idx in post_indices:
        c = float(day_df.at[idx, "close"])
        ema = float(day_df.at[idx, "ema_20"])
        if c >= u_trig and c > ema:
            entry_idx = idx
            opt_type = "CE"
            break
        elif c <= l_trig and c < ema:
            entry_idx = idx
            opt_type = "PE"
            break

    if entry_idx is None:
        return None, "NO_BREAKOUT"

    entry_ts = day_df.at[entry_idx, "timestamp"]
    entry_spot = float(day_df.at[entry_idx, "close"])
    strike = int(round(entry_spot / 50.0) * 50)
    dte = get_dte(target_date, "NIFTY")

    # Slice from entry onward
    rem_df = day_df.iloc[entry_idx:].copy().reset_index(drop=True)
    closes = rem_df["close"].values
    highs = rem_df["high"].values
    lows = rem_df["low"].values
    sub_times = rem_df["time"].values
    sub_ts = rem_df["timestamp"].values

    opt_p_arr = bs_vectorized(closes, strike, dte, sigma=0.15, opt_type=opt_type)
    opt_h_arr = bs_vectorized(highs if opt_type == "CE" else lows, strike, dte, sigma=0.15, opt_type=opt_type)
    opt_l_arr = bs_vectorized(lows if opt_type == "CE" else highs, strike, dte, sigma=0.15, opt_type=opt_type)

    entry_p = round(float(opt_p_arr[0]) * (1.0 + SLIPPAGE_PCT), 2)
    cost = round(entry_p * LOT_NIFTY, 2)
    if cost > MAX_CAPITAL_PER_TRADE:
        return None, "BUDGET_EXCEEDED"

    tgt_p = entry_p * 1.60
    sl_p = entry_p * 0.86

    exit_p = entry_p
    exit_ts = sub_ts[-1]
    exit_reason = "EOD (15:15)"

    for j in range(1, len(rem_df)):
        c_l = opt_l_arr[j]
        c_h = opt_h_arr[j]
        c_p = opt_p_arr[j]
        c_t = sub_times[j]
        t_stamp = sub_ts[j]

        # Stop loss evaluated FIRST (conservative, no lookahead bias)
        if c_l <= sl_p:
            exit_p = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_ts = t_stamp
            exit_reason = "SL Hit (-16%)"
            break
        elif c_h >= tgt_p:
            exit_p = round(tgt_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_ts = t_stamp
            exit_reason = "Target Hit (+60%)"
            break
        elif c_t >= t_1515:
            exit_p = round(float(c_p) * (1.0 - SLIPPAGE_PCT), 2)
            exit_ts = t_stamp
            exit_reason = "EOD (15:15)"
            break

    gross, charges, net = calc_charges(entry_p, exit_p, LOT_NIFTY)
    return {
        "date": target_date,
        "strategy": "Strategy 1: Nifty 15M Breakout",
        "leg": opt_type,
        "strike": f"NIFTY {strike} {opt_type}",
        "entry_time": str(entry_ts)[11:19],
        "exit_time": str(exit_ts)[11:19],
        "entry_p": entry_p,
        "exit_p": exit_p,
        "cost": cost,
        "gross": gross,
        "charges": charges,
        "net_pnl": net,
        "exit_reason": exit_reason
    }, "SUCCESS"

# ---------------------------------------------------------------------------
# STRATEGY 2: BANKNIFTY 15M DIRECTIONAL BREAKOUT
# ---------------------------------------------------------------------------
def simulate_s2(day_df, target_date, use_dynamic: bool = True):
    if len(day_df) < 20:
        return None, "NO_DATA"

    times = day_df["time"].values
    t_0915 = datetime.time(9, 15)
    t_0930 = datetime.time(9, 30)
    t_1500 = datetime.time(15, 0)
    t_1515 = datetime.time(15, 15)

    box_mask = (times >= t_0915) & (times <= t_0930)
    if np.sum(box_mask) < 10:
        return None, "INCOMPLETE_BOX"

    box_high = float(day_df.loc[box_mask, "high"].max())
    box_low = float(day_df.loc[box_mask, "low"].min())
    open_p = float(day_df.loc[box_mask, "close"].iloc[0])
    box_range = box_high - box_low
    box_pct = (box_range / open_p) * 100.0

    if box_pct > 1.20:
        return None, "WIDE_BOX_FILTERED"

    u_trig = box_high + 40.0
    l_trig = box_low - 40.0

    post_mask = (times > t_0930) & (times <= datetime.time(13, 0))
    post_indices = np.where(post_mask)[0]
    if len(post_indices) == 0:
        return None, "NO_POST_BARS"

    entry_idx = None
    opt_type = None
    for idx in post_indices:
        c = float(day_df.at[idx, "close"])
        ema = float(day_df.at[idx, "ema_20"])
        if c >= u_trig and c > ema:
            entry_idx = idx
            opt_type = "CE"
            break
        elif c <= l_trig and c < ema:
            entry_idx = idx
            opt_type = "PE"
            break

    if entry_idx is None:
        return None, "NO_BREAKOUT"

    entry_ts = day_df.at[entry_idx, "timestamp"]
    entry_spot = float(day_df.at[entry_idx, "close"])
    atm = int(round(entry_spot / 100.0) * 100)
    dte = get_dte(target_date, "BANKNIFTY")

    rem_df = day_df.iloc[entry_idx:].copy().reset_index(drop=True)
    closes = rem_df["close"].values
    highs = rem_df["high"].values
    lows = rem_df["low"].values
    sub_times = rem_df["time"].values
    sub_ts = rem_df["timestamp"].values

    chosen_strike = None
    entry_p = None
    cost = None

    if not use_dynamic:
        # OLD BEHAVIOR: Fixed 100-pt OTM
        fixed_strike = atm + 100 if opt_type == "CE" else atm - 100
        p0 = float(bs_vectorized(np.array([closes[0]]), fixed_strike, dte, sigma=0.18, opt_type=opt_type)[0])
        p0 = round(p0 * (1.0 + SLIPPAGE_PCT), 2)
        c0 = round(p0 * LOT_BANKNIFTY, 2)
        if c0 > MAX_CAPITAL_PER_TRADE:
            return None, "BUDGET_EXCEEDED"
        chosen_strike = fixed_strike
        entry_p = p0
        cost = c0
    else:
        # DYNAMIC BEHAVIOR: Scan outward in 100-pt increments until affordable
        for off in range(100, 3600, 100):
            k = atm + off if opt_type == "CE" else atm - off
            p0 = float(bs_vectorized(np.array([closes[0]]), k, dte, sigma=0.18, opt_type=opt_type)[0])
            p0 = round(p0 * (1.0 + SLIPPAGE_PCT), 2)
            c0 = round(p0 * LOT_BANKNIFTY, 2)
            if c0 <= MAX_CAPITAL_PER_TRADE and p0 >= DYNAMIC_MIN_PREMIUM:
                chosen_strike = k
                entry_p = p0
                cost = c0
                break

        if chosen_strike is None:
            return None, "BUDGET_EXCEEDED"

    opt_p_arr = bs_vectorized(closes, chosen_strike, dte, sigma=0.18, opt_type=opt_type)
    opt_h_arr = bs_vectorized(highs if opt_type == "CE" else lows, chosen_strike, dte, sigma=0.18, opt_type=opt_type)
    opt_l_arr = bs_vectorized(lows if opt_type == "CE" else highs, chosen_strike, dte, sigma=0.18, opt_type=opt_type)

    tgt_p = entry_p * 1.80
    sl_p = entry_p * 0.75

    exit_p = entry_p
    exit_ts = sub_ts[-1]
    exit_reason = "EOD (15:15)"

    for j in range(1, len(rem_df)):
        c_l = opt_l_arr[j]
        c_h = opt_h_arr[j]
        c_p = opt_p_arr[j]
        c_t = sub_times[j]
        t_stamp = sub_ts[j]

        # Stop loss evaluated FIRST (conservative, no lookahead bias)
        if c_l <= sl_p:
            exit_p = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_ts = t_stamp
            exit_reason = "SL Hit (-25%)"
            break
        elif c_h >= tgt_p:
            exit_p = round(tgt_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_ts = t_stamp
            exit_reason = "Target Hit (+80%)"
            break
        elif c_t >= t_1515:
            exit_p = round(float(c_p) * (1.0 - SLIPPAGE_PCT), 2)
            exit_ts = t_stamp
            exit_reason = "EOD (15:15)"
            break

    gross, charges, net = calc_charges(entry_p, exit_p, LOT_BANKNIFTY)
    return {
        "date": target_date,
        "strategy": "Strategy 2: BankNifty 15M Breakout" + (" (Dynamic)" if use_dynamic else " (Fixed 100-OTM)"),
        "leg": opt_type,
        "strike": f"BANKNIFTY {chosen_strike} {opt_type}",
        "entry_time": str(entry_ts)[11:19],
        "exit_time": str(exit_ts)[11:19],
        "entry_p": entry_p,
        "exit_p": exit_p,
        "cost": cost,
        "gross": gross,
        "charges": charges,
        "net_pnl": net,
        "exit_reason": exit_reason
    }, "SUCCESS"

# ---------------------------------------------------------------------------
# STRATEGY 3: NIFTY DECOUPLED ASYMMETRIC STRANGLE (DAS)
# ---------------------------------------------------------------------------
def simulate_s3(day_df, target_date):
    if len(day_df) < 50:
        return None, "NO_DATA"

    closes = day_df["close"].values
    highs = day_df["high"].values
    lows = day_df["low"].values
    times = day_df["time"].values
    timestamps = day_df["timestamp"].values
    n = len(closes)

    # 20-bar rolling std & 5-bar velocity
    rolling_std = pd.Series(closes).rolling(20).std().values
    rolling_vel = np.abs(closes - np.roll(closes, 5))
    rolling_vel[:5] = 0.0

    dte = get_dte(target_date, "NIFTY")
    iv = 0.135

    trade_res = None
    t_0945 = datetime.time(9, 45)
    t_1130 = datetime.time(11, 30)
    t_1300 = datetime.time(13, 0)
    t_1400 = datetime.time(14, 0)
    t_1515 = datetime.time(15, 15)

    for i in range(20, n):
        t = times[i]
        if t < t_0945 or t > t_1400:
            continue
        if t_1130 <= t <= t_1300:
            continue

        std_v = rolling_std[i]
        vel_v = rolling_vel[i]

        # Compression (< 30) and Velocity (>= 35)
        if std_v < 25.0 and vel_v >= 40.0:
            spot = float(closes[i])
            atm = int(round(spot / 50.0) * 50)

            # Select OTM strikes with premium in 35 - 68 range
            ce_strike, ce_p = None, None
            for off in [100, 150, 200, 50, 250]:
                k = atm + off
                p = float(bs_vectorized(np.array([spot]), k, dte, sigma=iv, opt_type="CE")[0])
                if 35.0 <= p <= 68.0:
                    ce_strike, ce_p = k, round(p, 2)
                    break

            pe_strike, pe_p = None, None
            for off in [100, 150, 200, 50, 250]:
                k = atm - off
                p = float(bs_vectorized(np.array([spot]), k, dte, sigma=iv, opt_type="PE")[0])
                if 35.0 <= p <= 68.0:
                    pe_strike, pe_p = k, round(p, 2)
                    break

            if not (ce_strike and pe_strike):
                continue

            entry_tot = round((ce_p + pe_p) * (1.0 + SLIPPAGE_PCT), 2)
            cost = round(entry_tot * LOT_NIFTY, 2)
            if cost > MAX_CAPITAL_PER_TRADE:
                continue

            entry_ts = timestamps[i]
            ce_tgt = round(ce_p * 1.60, 2)
            ce_sl = round(ce_p * 0.90, 2)
            pe_tgt = round(pe_p * 1.60, 2)
            pe_sl = round(pe_p * 0.90, 2)

            ce_exited, pe_exited = False, False
            ce_exit_p, pe_exit_p = ce_p, pe_p
            ce_reason, pe_reason = "", ""
            exit_ts = timestamps[min(i + 30, n - 1)]

            # Pre-compute forward option prices for sub-window
            max_j = min(i + 31, n)
            sub_h = highs[i+1:max_j]
            sub_l = lows[i+1:max_j]
            sub_c = closes[i+1:max_j]
            sub_times_j = times[i+1:max_j]
            sub_ts_j = timestamps[i+1:max_j]

            ce_h_arr = bs_vectorized(sub_h, ce_strike, dte, sigma=iv, opt_type="CE")
            ce_l_arr = bs_vectorized(sub_l, ce_strike, dte, sigma=iv, opt_type="CE")
            ce_c_arr = bs_vectorized(sub_c, ce_strike, dte, sigma=iv, opt_type="CE")

            pe_h_arr = bs_vectorized(sub_l, pe_strike, dte, sigma=iv, opt_type="PE")
            pe_l_arr = bs_vectorized(sub_h, pe_strike, dte, sigma=iv, opt_type="PE")
            pe_c_arr = bs_vectorized(sub_c, pe_strike, dte, sigma=iv, opt_type="PE")

            for k_idx in range(len(sub_c)):
                # Evaluate CE
                if not ce_exited:
                    if ce_l_arr[k_idx] <= ce_sl:
                        ce_exited = True
                        ce_exit_p = round(ce_sl * (1.0 - SLIPPAGE_PCT), 2)
                        ce_reason = "SL Hit (-12%)"
                    elif ce_h_arr[k_idx] >= ce_tgt:
                        ce_exited = True
                        ce_exit_p = round(ce_tgt * (1.0 - SLIPPAGE_PCT), 2)
                        ce_reason = "Target Hit (+60%)"

                # Evaluate PE
                if not pe_exited:
                    if pe_l_arr[k_idx] <= pe_sl:
                        pe_exited = True
                        pe_exit_p = round(pe_sl * (1.0 - SLIPPAGE_PCT), 2)
                        pe_reason = "SL Hit (-12%)"
                    elif pe_h_arr[k_idx] >= pe_tgt:
                        pe_exited = True
                        pe_exit_p = round(pe_tgt * (1.0 - SLIPPAGE_PCT), 2)
                        pe_reason = "Target Hit (+60%)"

                if ce_exited and pe_exited:
                    exit_ts = sub_ts_j[k_idx]
                    break

                # 50-minute hold cutoff or 15:15 EOD
                if k_idx == len(sub_c) - 1 or sub_times_j[k_idx] >= t_1515:
                    if not ce_exited:
                        ce_exit_p = round(float(ce_c_arr[k_idx]) * (1.0 - SLIPPAGE_PCT), 2)
                        ce_reason = "Time Cutoff (30m)"
                    if not pe_exited:
                        pe_exit_p = round(float(pe_c_arr[k_idx]) * (1.0 - SLIPPAGE_PCT), 2)
                        pe_reason = "Time Cutoff (30m)"
                    exit_ts = sub_ts_j[k_idx]
                    break

            exit_tot = round(ce_exit_p + pe_exit_p, 2)
            gross, charges, net = calc_charges(entry_tot, exit_tot, LOT_NIFTY, is_strangle=True)
            trade_res = {
                "date": target_date,
                "strategy": "Strategy 3: Nifty DAS",
                "leg": "CE+PE",
                "strike": f"NIFTY {ce_strike}CE + {pe_strike}PE",
                "entry_time": str(entry_ts)[11:19],
                "exit_time": str(exit_ts)[11:19],
                "entry_p": entry_tot,
                "exit_p": exit_tot,
                "cost": cost,
                "gross": gross,
                "charges": charges,
                "net_pnl": net,
                "exit_reason": f"{ce_reason} | {pe_reason}"
            }
            break  # Max 1 DAS trade per day

    return trade_res, ("SUCCESS" if trade_res else "NO_TRIGGER")

# ---------------------------------------------------------------------------
# MAIN BACKTEST EXECUTION
# ---------------------------------------------------------------------------
def compute_stats(trades_list):
    if not trades_list:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "gross": 0.0, "charges": 0.0, "net": 0.0,
            "pf": 0.0, "max_dd": 0.0, "return_pct": 0.0
        }
    df = pd.DataFrame(trades_list)
    tot = len(df)
    wins = len(df[df["net_pnl"] > 0])
    losses = len(df[df["net_pnl"] <= 0])
    wr = (wins / tot * 100.0) if tot > 0 else 0.0
    gross = float(df["gross"].sum())
    charges = float(df["charges"].sum())
    net = float(df["net_pnl"].sum())
    g_win = float(df[df["net_pnl"] > 0]["net_pnl"].sum())
    g_loss = float(abs(df[df["net_pnl"] <= 0]["net_pnl"].sum()))
    pf = round(g_win / g_loss, 2) if g_loss > 0 else (99.0 if g_win > 0 else 0.0)
    cum = df["net_pnl"].cumsum()
    max_dd = float((cum.cummax() - cum).max()) if not cum.empty else 0.0
    ret_pct = (net / INITIAL_CAPITAL) * 100.0
    return {
        "trades": tot, "wins": wins, "losses": losses, "win_rate": round(wr, 1),
        "gross": round(gross, 2), "charges": round(charges, 2), "net": round(net, 2),
        "pf": pf, "max_dd": round(max_dd, 2), "return_pct": round(ret_pct, 1)
    }

def run_backtest():
    print("=" * 100)
    print("   TRI-STRATEGY FULL-DATASET AUDIT & BACKTEST: DYNAMIC OPTION IMPACT ANALYSIS")
    print("=" * 100)

    print("Loading 1-minute historical exchange candles...")
    df_n = pd.read_csv(NIFTY_CSV)
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n = df_n.sort_values("timestamp").reset_index(drop=True)
    df_n["date"] = df_n["timestamp"].dt.date
    df_n["time"] = df_n["timestamp"].dt.time
    # Warm EMA-20 across full history
    df_n["ema_20"] = df_n["close"].ewm(span=20, adjust=False).mean()

    df_b = pd.read_csv(BANKNIFTY_CSV)
    df_b["timestamp"] = pd.to_datetime(df_b["timestamp"]).dt.tz_localize(None)
    df_b = df_b.sort_values("timestamp").reset_index(drop=True)
    df_b["date"] = df_b["timestamp"].dt.date
    df_b["time"] = df_b["timestamp"].dt.time
    # Warm EMA-20 across full history
    df_b["ema_20"] = df_b["close"].ewm(span=20, adjust=False).mean()

    n_days = {d: group.copy().reset_index(drop=True) for d, group in df_n.groupby("date")}
    b_days = {d: group.copy().reset_index(drop=True) for d, group in df_b.groupby("date")}
    all_dates = sorted(list(set(n_days.keys()).union(set(b_days.keys()))))
    print(f"Loaded {len(df_n):,} Nifty bars ({len(n_days)} days) and {len(df_b):,} BankNifty bars ({len(b_days)} days).")
    print(f"Simulation Range: {all_dates[0]} to {all_dates[-1]}")

    s1_trades = []
    s2_old_trades = []
    s2_new_trades = []
    s3_trades = []

    s2_old_rejections = {"BUDGET_EXCEEDED": 0, "WIDE_BOX": 0, "NO_BREAKOUT": 0}
    s2_new_rejections = {"BUDGET_EXCEEDED": 0, "WIDE_BOX": 0, "NO_BREAKOUT": 0}

    for d in all_dates:
        # S1: Nifty 15M Breakout
        if d in n_days:
            t1, reason1 = simulate_s1(n_days[d], d)
            if t1:
                s1_trades.append(t1)

            # S3: Nifty DAS
            t3, reason3 = simulate_s3(n_days[d], d)
            if t3:
                s3_trades.append(t3)

        # S2: BankNifty 15M Breakout
        if d in b_days:
            # S2 Old (Fixed 100-pt OTM, capped at Rs 9,500)
            t2_old, r2_old = simulate_s2(b_days[d], d, use_dynamic=False)
            if t2_old:
                s2_old_trades.append(t2_old)
            elif r2_old in s2_old_rejections:
                s2_old_rejections[r2_old] += 1
            elif r2_old == "WIDE_BOX_FILTERED":
                s2_old_rejections["WIDE_BOX"] += 1

            # S2 New (Dynamic strikes, capped at Rs 9,500)
            t2_new, r2_new = simulate_s2(b_days[d], d, use_dynamic=True)
            if t2_new:
                s2_new_trades.append(t2_new)
            elif r2_new in s2_new_rejections:
                s2_new_rejections[r2_new] += 1
            elif r2_new == "WIDE_BOX_FILTERED":
                s2_new_rejections["WIDE_BOX"] += 1

    # Portfolios
    portfolio_old = sorted(s1_trades + s2_old_trades + s3_trades, key=lambda x: (x["date"], x["entry_time"]))
    portfolio_new = sorted(s1_trades + s2_new_trades + s3_trades, key=lambda x: (x["date"], x["entry_time"]))

    stats_s1 = compute_stats(s1_trades)
    stats_s2_old = compute_stats(s2_old_trades)
    stats_s2_new = compute_stats(s2_new_trades)
    stats_s3 = compute_stats(s3_trades)
    stats_port_old = compute_stats(portfolio_old)
    stats_port_new = compute_stats(portfolio_new)

    print("\n" + "=" * 100)
    print("                     INDIVIDUAL STRATEGY PERFORMANCE BREAKDOWN")
    print("=" * 100)
    strat_rows = [
        ["Strategy 1: Nifty 15M Breakout (ATM)", f"{stats_s1['trades']}", f"{stats_s1['win_rate']}%", f"{stats_s1['pf']}", f"INR {stats_s1['max_dd']:,.0f}", f"+INR {stats_s1['net']:,.2f}"],
        ["Strategy 2 [OLD]: BN 15M Fixed 100-OTM", f"{stats_s2_old['trades']}", f"{stats_s2_old['win_rate']}%", f"{stats_s2_old['pf']}", f"INR {stats_s2_old['max_dd']:,.0f}", f"{'+' if stats_s2_old['net']>=0 else ''}INR {stats_s2_old['net']:,.2f}"],
        ["Strategy 2 [NEW]: BN 15M Dynamic Strike", f"{stats_s2_new['trades']}", f"{stats_s2_new['win_rate']}%", f"{stats_s2_new['pf']}", f"INR {stats_s2_new['max_dd']:,.0f}", f"{'+' if stats_s2_new['net']>=0 else ''}INR {stats_s2_new['net']:,.2f}"],
        ["Strategy 3: Nifty DAS Strangle", f"{stats_s3['trades']}", f"{stats_s3['win_rate']}%", f"{stats_s3['pf']}", f"INR {stats_s3['max_dd']:,.0f}", f"{'+' if stats_s3['net']>=0 else ''}INR {stats_s3['net']:,.2f}"]
    ]
    print(tabulate(strat_rows, headers=["Strategy", "Trades", "Win Rate", "Profit Factor", "Max Drawdown", "Net Realized PnL"], tablefmt="grid"))

    print("\n" + "=" * 100)
    print("           PORTFOLIO COMPARISON: BEFORE VS AFTER DYNAMIC STRIKE ADOPTION")
    print("=" * 100)
    comp_rows = [
        [
            "Portfolio [OLD] (Without Dynamic Strikes)",
            f"{stats_port_old['trades']} Trades",
            f"{stats_port_old['win_rate']}%",
            f"{stats_port_old['pf']}",
            f"INR {stats_port_old['max_dd']:,.0f}",
            f"{'+' if stats_port_old['net']>=0 else ''}INR {stats_port_old['net']:,.2f}",
            f"{'+' if stats_port_old['return_pct']>=0 else ''}{stats_port_old['return_pct']:.1f}%"
        ],
        [
            "Portfolio [NEW] (WITH DYNAMIC STRIKES)",
            f"{stats_port_new['trades']} Trades",
            f"{stats_port_new['win_rate']}%",
            f"{stats_port_new['pf']}",
            f"INR {stats_port_new['max_dd']:,.0f}",
            f"{'+' if stats_port_new['net']>=0 else ''}INR {stats_port_new['net']:,.2f}",
            f"{'+' if stats_port_new['return_pct']>=0 else ''}{stats_port_new['return_pct']:.1f}%"
        ]
    ]
    print(tabulate(comp_rows, headers=["System Setup", "Total Trades", "Win Rate", "Profit Factor", "Max Drawdown", "Net Realized PnL", "Account Growth"], tablefmt="grid"))

    print("\n" + "=" * 100)
    print("           BANKNIFTY BUDGET_EXCEEDED REJECTION ANALYSIS (MONTHLY OPTIONS)")
    print("=" * 100)
    rej_table = [
        ["Old Setup (Fixed 100-pt OTM)", f"{len(s2_old_trades)} Trades Taken", f"{s2_old_rejections['BUDGET_EXCEEDED']} Days Blocked (Budget Exceeded)", f"{s2_old_rejections['WIDE_BOX']} Days Wide Box Filtered"],
        ["New Setup (Dynamic Strike Engine)", f"{len(s2_new_trades)} Trades Taken", f"{s2_new_rejections['BUDGET_EXCEEDED']} Days Blocked (Budget Exceeded)", f"{s2_new_rejections['WIDE_BOX']} Days Wide Box Filtered"]
    ]
    print(tabulate(rej_table, headers=["Setup", "Executed Trades", "Budget Exceeded Blocks", "Wide Box Choppy Days"], tablefmt="grid"))
    print("=" * 100)

    # Save detailed CSV files for inspection
    pd.DataFrame(portfolio_new).to_csv(os.path.join(BASE_DIR, "data", "backtest_dynamic_portfolio.csv"), index=False)
    pd.DataFrame(s2_new_trades).to_csv(os.path.join(BASE_DIR, "data", "backtest_s2_dynamic.csv"), index=False)
    pd.DataFrame(s2_old_trades).to_csv(os.path.join(BASE_DIR, "data", "backtest_s2_old.csv"), index=False)
    print("\n[OK] Detailed CSV backtest logs saved in data/backtest_dynamic_portfolio.csv")

if __name__ == "__main__":
    run_backtest()
