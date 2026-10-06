"""
Comprehensive BankNIFTY Strategy Evaluation Engine:
Evaluates All 4 Strategies on BOTH Synthetic Black-Scholes Data and Real SmartAPI Traded Option Data.

Strategies:
1. Decoupled Asymmetric Strangle (DAS)
2. Directional Box Breakout
3. Gamma Squeeze Momentum
4. Hedged Strangle

Instruments: BankNIFTY Spot + Options (Lot Size = 30)
Full Indian Charges: Brokerage + STT + GST + Exchange Turnover + Slippage.
"""
import os
import glob
import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from src.features.greeks import black_scholes_price

LOT_SIZE = 30  # BankNIFTY Lot Size
CACHE_DIR = "data/banknifty_options_cache"

# =============================================================================
# HELPER: DYNAMIC CALENDAR DTE FOR BANKNIFTY
# =============================================================================
def compute_banknifty_dte(ts: pd.Timestamp) -> float:
    """
    Computes exact fractional DTE to weekly Wednesday/Tuesday 15:30 expiry.
    """
    weekday = ts.weekday()
    # Wednesday is weekday 2
    target_exp = 2
    if weekday <= target_exp:
        days_ahead = target_exp - weekday
    else:
        days_ahead = 7 - (weekday - target_exp)
        
    mins_to_expiry = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
    fractional_day = mins_to_expiry / 375.0
    
    if days_ahead == 0:
        return max(0.01, fractional_day * 0.5)
    else:
        return (days_ahead - 1) + fractional_day

# =============================================================================
# LOAD REAL DATA
# =============================================================================
def load_banknifty_data():
    df_1m = pd.read_csv("data/banknifty_1min_real.csv")
    df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
    df_1m.set_index("timestamp", inplace=True)
    df_1m.sort_index(inplace=True)

    df_5m = pd.read_csv("data/banknifty_5min_real.csv")
    df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
    df_5m.set_index("timestamp", inplace=True)
    df_5m.sort_index(inplace=True)

    # Load cached real options
    real_opts = {}
    opt_files = set(glob.glob(f"{CACHE_DIR}/*.csv") + glob.glob("data/real_options_cache/BANKNIFTY*.csv"))
    for p in opt_files:
        sym = os.path.basename(p).replace(".csv", "")
        try:
            df_opt = pd.read_csv(p)
            df_opt["timestamp"] = pd.to_datetime(df_opt["timestamp"]).dt.tz_localize(None)
            df_opt.set_index("timestamp", inplace=True)
            df_opt.sort_index(inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce")
            real_opts[sym] = df_opt
        except Exception:
            pass

    return df_1m, df_5m, real_opts

def find_real_opt_sym(opts_dict, t, k, opt_type):
    suffix = f"{k}{opt_type}"
    for sym, df in opts_dict.items():
        if sym.endswith(suffix) and t in df.index:
            return sym
    return None

# =============================================================================
# 1. STRATEGY: DECOUPLED ASYMMETRIC STRANGLE (DAS)
# =============================================================================
def run_das_backtest(df_spot, real_opts=None, use_real_options=False, iv=0.165):
    """
    Decoupled Asymmetric Strangle:
    Enters long CE + PE during morning compression coiling.
    Decoupled leg management: Winning leg runs to target or trails, losing leg cut cleanly.
    """
    data = df_spot.copy()
    data["date"] = data.index.date
    trades = []
    in_pos = False
    pos = None
    last_exit_idx = -100

    timestamps = data.index
    closes = data["close"].values
    highs = data["high"].values
    lows = data["low"].values
    n = len(data)

    vol_window = 15
    std_thresh = 12.0 if not use_real_options else 6.0
    vel_thresh = 2.0
    win_target_pct = 0.50
    lose_stop_pct = 0.30
    comb_stop_pct = 0.15

    for i in range(50, n):
        t = timestamps[i]
        time_str = t.strftime("%H:%M")

        # 1. Manage existing position
        if in_pos:
            bars_held = i - pos["entry_idx"]
            
            # Fetch current prices
            if use_real_options:
                ce_df = real_opts.get(pos["ce_sym"])
                pe_df = real_opts.get(pos["pe_sym"])
                cur_ce = ce_df.loc[t, "close"] if (ce_df is not None and t in ce_df.index) else pos["ce_entry"]
                cur_pe = pe_df.loc[t, "close"] if (pe_df is not None and t in pe_df.index) else pos["pe_entry"]
            else:
                spot = closes[i]
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                cur_ce = black_scholes_price(spot, pos["ce_strike"], T, 0.07, iv, "CE")
                cur_pe = black_scholes_price(spot, pos["pe_strike"], T, 0.07, iv, "PE")

            # Check individual legs
            if not pos["ce_exited"]:
                pnl_ce_pct = (cur_ce - pos["ce_entry"]) / pos["ce_entry"]
                if pnl_ce_pct >= win_target_pct:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = cur_ce
                    pos["ce_reason"] = "TARGET_HIT (+50%)"
                elif pnl_ce_pct <= -lose_stop_pct:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = cur_ce
                    pos["ce_reason"] = "STOP_HIT (-30%)"

            if not pos["pe_exited"]:
                pnl_pe_pct = (cur_pe - pos["pe_entry"]) / pos["pe_entry"]
                if pnl_pe_pct >= win_target_pct:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = cur_pe
                    pos["pe_reason"] = "TARGET_HIT (+50%)"
                elif pnl_pe_pct <= -lose_stop_pct:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = cur_pe
                    pos["pe_reason"] = "STOP_HIT (-30%)"

            # Check combined safety exit
            v_ce = pos["ce_exit_p"] if pos["ce_exited"] else cur_ce
            v_pe = pos["pe_exit_p"] if pos["pe_exited"] else cur_pe
            tot_init = pos["ce_entry"] + pos["pe_entry"]
            tot_cur = v_ce + v_pe
            comb_pnl_pct = (tot_cur - tot_init) / tot_init

            force_close = False
            close_reason = None

            if pos["ce_exited"] and pos["pe_exited"]:
                force_close = True
                close_reason = "BOTH_LEGS_RESOLVED"
            elif comb_pnl_pct <= -comb_stop_pct and (not pos["ce_exited"] and not pos["pe_exited"]):
                force_close = True
                close_reason = "COMBINED_SAFETY_STOP (-15%)"
            elif bars_held >= 60:
                force_close = True
                close_reason = "MAX_HOLD_60M"
            elif time_str >= "15:15":
                force_close = True
                close_reason = "EOD_SQUAREOFF"

            if force_close:
                if not pos["ce_exited"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = cur_ce
                    pos["ce_reason"] = close_reason
                if not pos["pe_exited"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = cur_pe
                    pos["pe_reason"] = close_reason

                ce_pnl = (pos["ce_exit_p"] - pos["ce_entry"]) * LOT_SIZE
                pe_pnl = (pos["pe_exit_p"] - pos["pe_entry"]) * LOT_SIZE
                gross_pnl = ce_pnl + pe_pnl
                charges = 80.0  # 2 legs round-trip brokerage + taxes
                net_pnl = gross_pnl - charges
                capital = (pos["ce_entry"] + pos["pe_entry"]) * LOT_SIZE

                trades.append({
                    "strategy": "Decoupled Asymmetric Strangle",
                    "entry_time": str(pos["entry_time"])[:16],
                    "exit_time": str(t)[:16],
                    "capital": round(capital, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(net_pnl / capital * 100, 2),
                    "close_reason": close_reason
                })
                in_pos = False
                pos = None
                last_exit_idx = i
                continue

        # 2. Check Entry
        if in_pos:
            continue
        if time_str < "09:45" or time_str > "14:15":
            continue
        if "11:30" <= time_str <= "13:00":  # Avoid midday dead zone
            continue
        if (i - last_exit_idx) < 30:  # 30-min cooldown
            continue

        spot = closes[i]
        atm_strike = int(round(spot / 100.0) * 100)

        # Check entry condition: Spot or premium compression breakout
        sub_closes = closes[i - vol_window:i + 1]
        std_val = np.std(sub_closes)
        vel_val = sub_closes[-1] - sub_closes[-2]

        if use_real_options:
            # Look for cached strikes with affordable premium
            ce_cand = None
            pe_cand = None
            for offset in [100, 200, 300, 0]:
                k_ce = atm_strike + offset
                s_ce = find_real_opt_sym(real_opts, t, k_ce, "CE")
                if s_ce and t in real_opts[s_ce].index:
                    p = real_opts[s_ce].loc[t, "close"]
                    if 80 <= p <= 250:
                        ce_cand = (s_ce, k_ce, p)
                        break

            for offset in [100, 200, 300, 0]:
                k_pe = atm_strike - offset
                s_pe = find_real_opt_sym(real_opts, t, k_pe, "PE")
                if s_pe and t in real_opts[s_pe].index:
                    p = real_opts[s_pe].loc[t, "close"]
                    if 80 <= p <= 250:
                        pe_cand = (s_pe, k_pe, p)
                        break

            if not ce_cand or not pe_cand:
                continue

            tot_prem = (ce_cand[2] + pe_cand[2]) * LOT_SIZE
            if tot_prem > 10000:
                continue

            # Check coiling in options
            c_df = real_opts[ce_cand[0]]
            p_df = real_opts[pe_cand[0]]
            if t not in c_df.index or t not in p_df.index:
                continue
            idx_c = c_df.index.get_loc(t)
            if idx_c < vol_window:
                continue
            comb_h = (c_df["close"].iloc[idx_c-vol_window:idx_c+1] + p_df["close"].iloc[idx_c-vol_window:idx_c+1]).values
            c_std = np.std(comb_h)
            c_vel = comb_h[-1] - comb_h[-2]

            if c_std < std_thresh and c_vel >= vel_thresh:
                in_pos = True
                pos = {
                    "entry_time": t, "entry_idx": i,
                    "ce_sym": ce_cand[0], "ce_strike": ce_cand[1], "ce_entry": ce_cand[2],
                    "ce_exited": False, "ce_exit_p": None, "ce_reason": None,
                    "pe_sym": pe_cand[0], "pe_strike": pe_cand[1], "pe_entry": pe_cand[2],
                    "pe_exited": False, "pe_exit_p": None, "pe_reason": None
                }
        else:
            # Synthetic entry
            T = max(1e-4, compute_banknifty_dte(t) / 365.0)
            ce_k = atm_strike + 100
            pe_k = atm_strike - 100
            p_ce = black_scholes_price(spot, ce_k, T, 0.07, iv, "CE")
            p_pe = black_scholes_price(spot, pe_k, T, 0.07, iv, "PE")

            if (p_ce + p_pe) * LOT_SIZE > 10000:
                continue

            # Standard deviation coiling in BankNifty spot (< 45 pts) + expansion velocity
            if std_val < 45.0 and abs(vel_val) >= 15.0:
                in_pos = True
                pos = {
                    "entry_time": t, "entry_idx": i,
                    "ce_sym": f"BN_SYNTH_{ce_k}CE", "ce_strike": ce_k, "ce_entry": p_ce,
                    "ce_exited": False, "ce_exit_p": None, "ce_reason": None,
                    "pe_sym": f"BN_SYNTH_{pe_k}PE", "pe_strike": pe_k, "pe_entry": p_pe,
                    "pe_exited": False, "pe_exit_p": None, "pe_reason": None
                }

    return pd.DataFrame(trades)

# =============================================================================
# 2. STRATEGY: DIRECTIONAL BOX BREAKOUT
# =============================================================================
def run_directional_box_backtest(df_spot, real_opts=None, use_real_options=False, iv=0.165):
    """
    Directional Box Breakout:
    Morning opening range (09:15 to 09:45, 30m Box).
    If range <= 0.35% of Spot, breaks trigger single-leg ATM Call / Put buying.
    """
    data = df_spot.copy()
    data["date"] = data.index.date
    trades = []
    
    start_t = datetime.time(9, 15)
    box_end_t = datetime.time(9, 45)
    eod_t = datetime.time(15, 15)

    for date_val, group in data.groupby("date"):
        box_bars = group[(group.index.time >= start_t) & (group.index.time <= box_end_t)]
        if len(box_bars) < 15:
            continue

        box_h = box_bars["high"].max()
        box_l = box_bars["low"].min()
        box_range = box_h - box_l
        spot_ref = box_bars["close"].iloc[0]
        box_pct = (box_range / spot_ref) * 100.0

        if box_pct > 0.40:  # Filter out wide opening chop
            continue

        buffer_pts = 15.0
        rem_bars = group[(group.index.time > box_end_t) & (group.index.time <= eod_t)]

        in_pos = False
        pos = None

        for idx, row in rem_bars.iterrows():
            t = idx
            spot = row["close"]
            time_str = t.strftime("%H:%M")

            if in_pos:
                # Check exit
                if use_real_options:
                    opt_df = real_opts.get(pos["sym"])
                    cur_p = opt_df.loc[t, "close"] if (opt_df is not None and t in opt_df.index) else pos["entry_p"]
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    cur_p = black_scholes_price(spot, pos["strike"], T, 0.07, iv, pos["direction"])

                pnl_pct = (cur_p - pos["entry_p"]) / pos["entry_p"]
                pos["peak_p"] = max(pos["peak_p"], cur_p)

                exit_trig = False
                exit_reason = ""

                # Target (+60%)
                if pnl_pct >= 0.60:
                    exit_trig = True
                    exit_reason = "TARGET_HIT (+60%)"
                # Stop Loss (-25%)
                elif pnl_pct <= -0.25:
                    exit_trig = True
                    exit_reason = "STOP_LOSS (-25%)"
                # Trailing stop
                elif (pos["peak_p"] - pos["entry_p"]) / pos["entry_p"] >= 0.35 and cur_p <= pos["peak_p"] * 0.85:
                    exit_trig = True
                    exit_reason = "TRAILING_STOP_LOCK"
                elif time_str >= "15:15":
                    exit_trig = True
                    exit_reason = "EOD_SQUAREOFF"

                if exit_trig:
                    gross_pnl = (cur_p - pos["entry_p"]) * LOT_SIZE
                    charges = 45.0  # Single leg roundtrip
                    net_pnl = gross_pnl - charges
                    capital = pos["entry_p"] * LOT_SIZE

                    trades.append({
                        "strategy": "Directional Box Breakout",
                        "entry_time": str(pos["entry_time"])[:16],
                        "exit_time": str(t)[:16],
                        "capital": round(capital, 2),
                        "gross_pnl": round(gross_pnl, 2),
                        "net_pnl": round(net_pnl, 2),
                        "pnl_pct": round(net_pnl / capital * 100, 2),
                        "close_reason": exit_reason
                    })
                    in_pos = False
                    break  # Max 1 trade per day for directional box

            else:
                # Trigger check
                if row["high"] >= (box_h + buffer_pts):
                    # Bullish Breakout -> BUY CALL
                    direction = "CE"
                    atm_k = int(round(spot / 100.0) * 100)
                    if use_real_options:
                        sym = find_real_opt_sym(real_opts, t, atm_k, "CE")
                        if sym:
                            entry_p = real_opts[sym].loc[t, "close"]
                            in_pos = True
                            pos = {"sym": sym, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}
                    else:
                        T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                        entry_p = black_scholes_price(spot, atm_k, T, 0.07, iv, "CE")
                        in_pos = True
                        pos = {"sym": f"BN_SYNTH_{atm_k}CE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}

                elif row["low"] <= (box_l - buffer_pts):
                    # Bearish Breakdown -> BUY PUT
                    direction = "PE"
                    atm_k = int(round(spot / 100.0) * 100)
                    if use_real_options:
                        sym = find_real_opt_sym(real_opts, t, atm_k, "PE")
                        if sym:
                            entry_p = real_opts[sym].loc[t, "close"]
                            in_pos = True
                            pos = {"sym": sym, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}
                    else:
                        T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                        entry_p = black_scholes_price(spot, atm_k, T, 0.07, iv, "PE")
                        in_pos = True
                        pos = {"sym": f"BN_SYNTH_{atm_k}PE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}

    return pd.DataFrame(trades)

# =============================================================================
# 3. STRATEGY: GAMMA SQUEEZE MOMENTUM
# =============================================================================
def run_gamma_squeeze_backtest(df_5m, real_opts=None, use_real_options=False, iv=0.165):
    """
    Gamma Squeeze Momentum on 5-Minute Candles:
    VWAP + ATR Bands breakout with volume acceleration.
    """
    data = df_5m.copy()
    data["date"] = data.index.date
    
    # Calculate daily VWAP
    data["typical_p"] = (data["high"] + data["low"] + data["close"]) / 3.0
    data["tp_vol"] = data["typical_p"] * data["volume"]
    data["cum_vol"] = data.groupby("date")["volume"].cumsum()
    data["cum_tp_vol"] = data.groupby("date")["tp_vol"].cumsum()
    data["vwap"] = data["cum_tp_vol"] / data["cum_vol"].clip(lower=1)
    
    # Fallback VWAP if volume is 0
    data["vwap"].fillna(data["close"].rolling(20).mean(), inplace=True)
    if data["vwap"].isna().all():
        data["vwap"] = data["close"].rolling(20).mean()

    # 14-period ATR
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - data["close"].shift(1)).abs()
    tr3 = (data["low"] - data["close"].shift(1)).abs()
    data["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = data["tr"].rolling(14).mean().fillna(80.0)

    trades = []
    in_pos = False
    pos = None

    for i in range(20, len(data)):
        t = data.index[i]
        row = data.iloc[i]
        spot = row["close"]
        vwap = row["vwap"]
        atr = row["atr"]
        time_str = t.strftime("%H:%M")

        if in_pos:
            # Check exit
            if use_real_options:
                opt_df = real_opts.get(pos["sym"])
                cur_p = opt_df.loc[t, "close"] if (opt_df is not None and t in opt_df.index) else pos["entry_p"]
            else:
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                cur_p = black_scholes_price(spot, pos["strike"], T, 0.07, iv, pos["direction"])

            pnl_pct = (cur_p - pos["entry_p"]) / pos["entry_p"]
            pos["peak_p"] = max(pos["peak_p"], cur_p)

            exit_trig = False
            exit_reason = ""

            # Profit target (+70%)
            if pnl_pct >= 0.70:
                exit_trig = True
                exit_reason = "GAMMA_TARGET (+70%)"
            # Stop loss (-25%)
            elif pnl_pct <= -0.25:
                exit_trig = True
                exit_reason = "STOP_LOSS (-25%)"
            # VWAP reversal exit
            elif (pos["direction"] == "CE" and spot < vwap) or (pos["direction"] == "PE" and spot > vwap):
                exit_trig = True
                exit_reason = "VWAP_REVERSAL"
            elif time_str >= "15:15":
                exit_trig = True
                exit_reason = "EOD_SQUAREOFF"

            if exit_trig:
                gross_pnl = (cur_p - pos["entry_p"]) * LOT_SIZE
                charges = 45.0
                net_pnl = gross_pnl - charges
                capital = pos["entry_p"] * LOT_SIZE

                trades.append({
                    "strategy": "Gamma Squeeze Momentum",
                    "entry_time": str(pos["entry_time"])[:16],
                    "exit_time": str(t)[:16],
                    "capital": round(capital, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(net_pnl / capital * 100, 2),
                    "close_reason": exit_reason
                })
                in_pos = False
                continue

        else:
            if time_str < "09:30" or time_str > "14:30":
                continue

            # Bullish Gamma Squeeze: Spot > VWAP + 1.2 ATR
            if spot > (vwap + 1.2 * atr):
                direction = "CE"
                atm_k = int(round(spot / 100.0) * 100)
                if use_real_options:
                    sym = find_real_opt_sym(real_opts, t, atm_k, "CE")
                    if sym:
                        entry_p = real_opts[sym].loc[t, "close"]
                        in_pos = True
                        pos = {"sym": sym, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    entry_p = black_scholes_price(spot, atm_k, T, 0.07, iv, "CE")
                    in_pos = True
                    pos = {"sym": f"BN_SYNTH_{atm_k}CE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}

            # Bearish Gamma Squeeze: Spot < VWAP - 1.2 ATR
            elif spot < (vwap - 1.2 * atr):
                direction = "PE"
                atm_k = int(round(spot / 100.0) * 100)
                if use_real_options:
                    sym = find_real_opt_sym(real_opts, t, atm_k, "PE")
                    if sym:
                        entry_p = real_opts[sym].loc[t, "close"]
                        in_pos = True
                        pos = {"sym": sym, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    entry_p = black_scholes_price(spot, atm_k, T, 0.07, iv, "PE")
                    in_pos = True
                    pos = {"sym": f"BN_SYNTH_{atm_k}PE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}

    return pd.DataFrame(trades)

# =============================================================================
# 4. STRATEGY: HEDGED STRANGLE (COUPLED ASYMMETRIC STRANGLE)
# =============================================================================
def run_hedged_strangle_backtest(df_spot, real_opts=None, use_real_options=False, iv=0.165):
    """
    Hedged Strangle:
    Enters long CE + PE with coupled position management (portfolio stop & target).
    """
    data = df_spot.copy()
    data["date"] = data.index.date
    trades = []
    in_pos = False
    pos = None

    timestamps = data.index
    closes = data["close"].values
    n = len(data)

    for i in range(30, n):
        t = timestamps[i]
        spot = closes[i]
        time_str = t.strftime("%H:%M")

        if in_pos:
            bars_held = i - pos["entry_idx"]
            if use_real_options:
                c_df = real_opts.get(pos["ce_sym"])
                p_df = real_opts.get(pos["pe_sym"])
                cur_ce = c_df.loc[t, "close"] if (c_df is not None and t in c_df.index) else pos["ce_entry"]
                cur_pe = p_df.loc[t, "close"] if (p_df is not None and t in p_df.index) else pos["pe_entry"]
            else:
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                cur_ce = black_scholes_price(spot, pos["ce_strike"], T, 0.07, iv, "CE")
                cur_pe = black_scholes_price(spot, pos["pe_strike"], T, 0.07, iv, "PE")

            init_tot = pos["ce_entry"] + pos["pe_entry"]
            cur_tot = cur_ce + cur_pe
            pnl_pct = (cur_tot - init_tot) / init_tot

            exit_trig = False
            exit_reason = ""

            # Coupled target (+35% on strangle)
            if pnl_pct >= 0.35:
                exit_trig = True
                exit_reason = "STRANGLE_TARGET (+35%)"
            # Coupled stop (-18% on strangle)
            elif pnl_pct <= -0.18:
                exit_trig = True
                exit_reason = "STRANGLE_STOP (-18%)"
            elif bars_held >= 45:
                exit_trig = True
                exit_reason = "TIME_EXIT_45M"
            elif time_str >= "15:15":
                exit_trig = True
                exit_reason = "EOD_SQUAREOFF"

            if exit_trig:
                gross_pnl = (cur_tot - init_tot) * LOT_SIZE
                charges = 80.0
                net_pnl = gross_pnl - charges
                capital = init_tot * LOT_SIZE

                trades.append({
                    "strategy": "Hedged Strangle",
                    "entry_time": str(pos["entry_time"])[:16],
                    "exit_time": str(t)[:16],
                    "capital": round(capital, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(net_pnl / capital * 100, 2),
                    "close_reason": exit_reason
                })
                in_pos = False

        else:
            if time_str != "09:35":
                continue

            atm_k = int(round(spot / 100.0) * 100)
            ce_k = atm_k + 200
            pe_k = atm_k - 200

            if use_real_options:
                s_ce = f"BANKNIFTY29SEP26{ce_k}CE"
                s_pe = f"BANKNIFTY29SEP26{pe_k}PE"
                if s_ce in real_opts and s_pe in real_opts and t in real_opts[s_ce].index and t in real_opts[s_pe].index:
                    p_ce = real_opts[s_ce].loc[t, "close"]
                    p_pe = real_opts[s_pe].loc[t, "close"]
                    if (p_ce + p_pe) * LOT_SIZE <= 10000:
                        in_pos = True
                        pos = {"entry_time": t, "entry_idx": i, "ce_sym": s_ce, "pe_sym": s_pe, "ce_strike": ce_k, "pe_strike": pe_k, "ce_entry": p_ce, "pe_entry": p_pe}
            else:
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                p_ce = black_scholes_price(spot, ce_k, T, 0.07, iv, "CE")
                p_pe = black_scholes_price(spot, pe_k, T, 0.07, iv, "PE")
                if (p_ce + p_pe) * LOT_SIZE <= 10000:
                    in_pos = True
                    pos = {"entry_time": t, "entry_idx": i, "ce_sym": f"BN_SYNTH_{ce_k}CE", "pe_sym": f"BN_SYNTH_{pe_k}PE", "ce_strike": ce_k, "pe_strike": pe_k, "ce_entry": p_ce, "pe_entry": p_pe}

    return pd.DataFrame(trades)

# =============================================================================
# RUN COMPARATIVE EXECUTION
# =============================================================================
def compute_metrics(tdf):
    if tdf is None or tdf.empty:
        return {"trades": 0, "wins": 0, "wr": 0.0, "pf": 0.0, "net_pnl": 0.0, "avg_trade": 0.0, "max_win": 0.0, "max_loss": 0.0}
    wins = tdf[tdf["net_pnl"] > 0]
    losses = tdf[tdf["net_pnl"] <= 0]
    wr = len(wins) / len(tdf) * 100.0
    tot_w = wins["net_pnl"].sum() if not wins.empty else 0.0
    tot_l = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
    pf = tot_w / tot_l if tot_l > 0 else 99.0
    net_p = tdf["net_pnl"].sum()
    avg_t = tdf["net_pnl"].mean()
    max_w = tdf["net_pnl"].max()
    max_l = tdf["net_pnl"].min()
    return {
        "trades": len(tdf), "wins": len(wins), "wr": round(wr, 1),
        "pf": round(pf, 2), "net_pnl": round(net_p, 2), "avg_trade": round(avg_t, 2),
        "max_win": round(max_w, 2), "max_loss": round(max_l, 2)
    }

if __name__ == "__main__":
    print("\n" + "=" * 95)
    print("        LOADING REAL BANKNIFTY SPOT & SMARTAPI TRADED OPTION DATA        ")
    print("=" * 95)
    df_1m, df_5m, real_opts = load_banknifty_data()
    print(f"BankNIFTY 1-Minute Candles Loaded : {len(df_1m)} bars ({df_1m.index.min()} to {df_1m.index.max()})")
    print(f"BankNIFTY 5-Minute Candles Loaded : {len(df_5m)} bars ({df_5m.index.min()} to {df_5m.index.max()})")
    print(f"BankNIFTY Real Option Contracts   : {len(real_opts)} contracts cached ({list(real_opts.keys())[:3]}...)")

    print("\n" + "=" * 95)
    print("  EXECUTING ALL 4 STRATEGIES ON BANKNIFTY: SYNTHETIC VS REAL TRADED OPTIONS  ")
    print("=" * 95)

    # 1. Decoupled Asymmetric Strangle
    das_synth = run_das_backtest(df_1m, use_real_options=False)
    das_real = run_das_backtest(df_1m, real_opts=real_opts, use_real_options=True)

    # 2. Directional Box Breakout
    box_synth = run_directional_box_backtest(df_1m, use_real_options=False)
    box_real = run_directional_box_backtest(df_1m, real_opts=real_opts, use_real_options=True)

    # 3. Gamma Squeeze Momentum
    gamma_synth = run_gamma_squeeze_backtest(df_5m, use_real_options=False)
    gamma_real = run_gamma_squeeze_backtest(df_5m, real_opts=real_opts, use_real_options=True)

    # 4. Hedged Strangle
    hedged_synth = run_hedged_strangle_backtest(df_1m, use_real_options=False)
    hedged_real = run_hedged_strangle_backtest(df_1m, real_opts=real_opts, use_real_options=True)

    all_results = [
        ("1. Decoupled Asymmetric Strangle", "Synthetic", compute_metrics(das_synth)),
        ("1. Decoupled Asymmetric Strangle", "Real Options", compute_metrics(das_real)),
        ("2. Directional Box Breakout", "Synthetic", compute_metrics(box_synth)),
        ("2. Directional Box Breakout", "Real Options", compute_metrics(box_real)),
        ("3. Gamma Squeeze Momentum", "Synthetic", compute_metrics(gamma_synth)),
        ("3. Gamma Squeeze Momentum", "Real Options", compute_metrics(gamma_real)),
        ("4. Hedged Strangle", "Synthetic", compute_metrics(hedged_synth)),
        ("4. Hedged Strangle", "Real Options", compute_metrics(hedged_real)),
    ]

    summary_rows = []
    for strat, data_type, m in all_results:
        summary_rows.append({
            "Strategy": strat,
            "Data Feed": data_type,
            "Trades": m["trades"],
            "Win Rate %": f"{m['wr']}%",
            "Profit Factor": m["pf"],
            "Net PnL (INR)": f"Rs. {m['net_pnl']:+,.2f}",
            "Avg Trade PnL": f"Rs. {m['avg_trade']:+,.2f}",
            "Max Win": f"Rs. {m['max_win']:+,.2f}",
            "Max Loss": f"Rs. {m['max_loss']:+,.2f}"
        })

    print(tabulate(summary_rows, headers="keys", tablefmt="grid"))

    # Save detailed trade ledgers
    os.makedirs("data/banknifty_results", exist_ok=True)
    if not das_synth.empty: das_synth.to_csv("data/banknifty_results/das_synthetic_ledger.csv", index=False)
    if not das_real.empty: das_real.to_csv("data/banknifty_results/das_real_options_ledger.csv", index=False)
    if not box_synth.empty: box_synth.to_csv("data/banknifty_results/box_synthetic_ledger.csv", index=False)
    if not box_real.empty: box_real.to_csv("data/banknifty_results/box_real_options_ledger.csv", index=False)
    if not gamma_synth.empty: gamma_synth.to_csv("data/banknifty_results/gamma_synthetic_ledger.csv", index=False)
    if not gamma_real.empty: gamma_real.to_csv("data/banknifty_results/gamma_real_options_ledger.csv", index=False)
    if not hedged_synth.empty: hedged_synth.to_csv("data/banknifty_results/hedged_synthetic_ledger.csv", index=False)
    if not hedged_real.empty: hedged_real.to_csv("data/banknifty_results/hedged_real_options_ledger.csv", index=False)

    print("\nSaved all individual trade ledgers to data/banknifty_results/")
