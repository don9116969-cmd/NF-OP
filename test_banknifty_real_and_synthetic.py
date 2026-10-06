"""
Rigorous BankNIFTY Strategy Evaluation:
Testing All 4 Strategies on both Synthetic Black-Scholes Data and Real SmartAPI Traded Option Data.

Key BankNIFTY Parameters:
- Lot Size: 30 Qty
- Index Spot: ~55,000
- Strike Step: 100 Points
- Capital Budget: Realistically sized for BankNIFTY Option Buying (1 Lot)
"""
import os
import glob
import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from src.features.greeks import black_scholes_price

LOT_SIZE = 30
CACHE_DIR = "data/banknifty_options_cache"

df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.set_index("timestamp", inplace=True)
df_1m.sort_index(inplace=True)

df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.set_index("timestamp", inplace=True)
df_5m.sort_index(inplace=True)

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

def find_real_opt_sym(opts_dict, t, k, opt_type):
    suffix = f"{k}{opt_type}"
    for sym, df in opts_dict.items():
        if sym.endswith(suffix) and t in df.index:
            return sym
    return None

def compute_banknifty_dte(ts: pd.Timestamp) -> float:
    weekday = ts.weekday()
    target_exp = 2  # Wednesday
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
# STRATEGY 1: DECOUPLED ASYMMETRIC STRANGLE (BANKNIFTY CALIBRATED)
# =============================================================================
def eval_das(use_real=False):
    data = df_1m.copy()
    data["date"] = data.index.date
    trades = []
    in_pos = False
    pos = None
    last_exit_idx = -100

    timestamps = data.index
    closes = data["close"].values
    n = len(data)

    for i in range(50, n):
        t = timestamps[i]
        time_str = t.strftime("%H:%M")

        if in_pos:
            bars_held = i - pos["entry_idx"]
            if use_real:
                c_df = real_opts.get(pos["ce_sym"])
                p_df = real_opts.get(pos["pe_sym"])
                cur_ce = c_df.loc[t, "close"] if (c_df is not None and t in c_df.index) else pos["ce_entry"]
                cur_pe = p_df.loc[t, "close"] if (p_df is not None and t in p_df.index) else pos["pe_entry"]
            else:
                spot = closes[i]
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                cur_ce = black_scholes_price(spot, pos["ce_strike"], T, 0.07, 0.165, "CE")
                cur_pe = black_scholes_price(spot, pos["pe_strike"], T, 0.07, 0.165, "PE")

            # Manage individual legs
            if not pos["ce_exited"]:
                ret_ce = (cur_ce - pos["ce_entry"]) / pos["ce_entry"]
                if ret_ce >= 0.60:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = cur_ce
                    pos["ce_reason"] = "TARGET_HIT (+60%)"
                elif ret_ce <= -0.30:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = cur_ce
                    pos["ce_reason"] = "STOP_LOSS (-30%)"

            if not pos["pe_exited"]:
                ret_pe = (cur_pe - pos["pe_entry"]) / pos["pe_entry"]
                if ret_pe >= 0.60:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = cur_pe
                    pos["pe_reason"] = "TARGET_HIT (+60%)"
                elif ret_pe <= -0.30:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = cur_pe
                    pos["pe_reason"] = "STOP_LOSS (-30%)"

            # Check termination
            force_close = False
            close_reason = ""
            if pos["ce_exited"] and pos["pe_exited"]:
                force_close = True
                close_reason = "BOTH_LEGS_RESOLVED"
            elif bars_held >= 90:
                force_close = True
                close_reason = "TIME_STOP_90M"
            elif time_str >= "15:15":
                force_close = True
                close_reason = "EOD_SQUAREOFF"

            if force_close:
                p_ce = pos["ce_exit_p"] if pos["ce_exited"] else cur_ce
                p_pe = pos["pe_exit_p"] if pos["pe_exited"] else cur_pe
                ce_pnl = (p_ce - pos["ce_entry"]) * LOT_SIZE
                pe_pnl = (p_pe - pos["pe_entry"]) * LOT_SIZE
                gross = ce_pnl + pe_pnl
                net = gross - 80.0
                cap = (pos["ce_entry"] + pos["pe_entry"]) * LOT_SIZE
                trades.append({
                    "strategy": "Decoupled Asymmetric Strangle",
                    "entry_time": str(pos["entry_time"])[:16],
                    "exit_time": str(t)[:16],
                    "capital": round(cap, 2),
                    "net_pnl": round(net, 2),
                    "ret_pct": round(net / cap * 100, 2),
                    "reason": close_reason
                })
                in_pos = False
                last_exit_idx = i
                continue

        else:
            if time_str < "09:45" or time_str > "13:30":
                continue
            if "11:30" <= time_str <= "12:45":
                continue
            if (i - last_exit_idx) < 45:
                continue

            spot = closes[i]
            # Coiling check over 20 bars
            sub_c = closes[i-20:i+1]
            std_c = np.std(sub_c)
            vel_c = sub_c[-1] - sub_c[-3]

            if std_c < 65.0 and abs(vel_c) >= 30.0:
                atm_k = int(round(spot / 100.0) * 100)
                ce_k = atm_k + 200
                pe_k = atm_k - 200

                if use_real:
                    s_ce = find_real_opt_sym(real_opts, t, ce_k, "CE")
                    s_pe = find_real_opt_sym(real_opts, t, pe_k, "PE")
                    if s_ce and s_pe:
                        p_ce = real_opts[s_ce].loc[t, "close"]
                        p_pe = real_opts[s_pe].loc[t, "close"]
                        in_pos = True
                        pos = {
                            "entry_time": t, "entry_idx": i,
                            "ce_sym": s_ce, "ce_entry": p_ce, "ce_exited": False, "ce_exit_p": None,
                            "pe_sym": s_pe, "pe_entry": p_pe, "pe_exited": False, "pe_exit_p": None
                        }
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    p_ce = black_scholes_price(spot, ce_k, T, 0.07, 0.165, "CE")
                    p_pe = black_scholes_price(spot, pe_k, T, 0.07, 0.165, "PE")
                    in_pos = True
                    pos = {
                        "entry_time": t, "entry_idx": i,
                        "ce_strike": ce_k, "ce_entry": p_ce, "ce_exited": False, "ce_exit_p": None,
                        "pe_strike": pe_k, "pe_entry": p_pe, "pe_exited": False, "pe_exit_p": None
                    }

    return pd.DataFrame(trades)

# =============================================================================
# STRATEGY 2: DIRECTIONAL BOX BREAKOUT (BANKNIFTY CALIBRATED)
# =============================================================================
def eval_box(use_real=False):
    data = df_1m.copy()
    data["date"] = data.index.date
    trades = []
    
    start_t = datetime.time(9, 15)
    box_end_t = datetime.time(9, 45)  # 30-min opening range
    eod_t = datetime.time(15, 15)
    buffer_pts = 40.0

    for date_val, group in data.groupby("date"):
        box_bars = group[(group.index.time >= start_t) & (group.index.time <= box_end_t)]
        if len(box_bars) < 20:
            continue

        box_h = box_bars["high"].max()
        box_l = box_bars["low"].min()
        box_range = box_h - box_l
        spot_ref = box_bars["close"].iloc[0]
        box_pct = (box_range / spot_ref) * 100.0

        if box_pct > 0.65:  # Skip abnormally wild morning boxes (>350 pts)
            continue

        rem_bars = group[(group.index.time > box_end_t) & (group.index.time <= eod_t)]
        in_pos = False

        for idx, row in rem_bars.iterrows():
            t = idx
            spot = row["close"]
            time_str = t.strftime("%H:%M")

            if in_pos:
                if use_real:
                    opt_df = real_opts.get(pos["sym"])
                    cur_p = opt_df.loc[t, "close"] if (opt_df is not None and t in opt_df.index) else pos["entry_p"]
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    cur_p = black_scholes_price(spot, pos["strike"], T, 0.07, 0.165, pos["direction"])

                ret_p = (cur_p - pos["entry_p"]) / pos["entry_p"]
                pos["peak_p"] = max(pos["peak_p"], cur_p)

                exit_trig = False
                exit_reason = ""

                # Target (+60%)
                if ret_p >= 0.60:
                    exit_trig = True
                    exit_reason = "TARGET_HIT (+60%)"
                # Stop Loss (-25%)
                elif ret_p <= -0.25:
                    exit_trig = True
                    exit_reason = "STOP_LOSS (-25%)"
                elif time_str >= "15:15":
                    exit_trig = True
                    exit_reason = "EOD_SQUAREOFF"

                if exit_trig:
                    gross = (cur_p - pos["entry_p"]) * LOT_SIZE
                    net = gross - 45.0
                    cap = pos["entry_p"] * LOT_SIZE
                    trades.append({
                        "strategy": "Directional Box Breakout",
                        "entry_time": str(pos["entry_time"])[:16],
                        "exit_time": str(t)[:16],
                        "capital": round(cap, 2),
                        "net_pnl": round(net, 2),
                        "ret_pct": round(net / cap * 100, 2),
                        "reason": exit_reason
                    })
                    in_pos = False
                    break

            else:
                if row["high"] >= (box_h + buffer_pts):
                    # Bullish breakout -> Buy ATM Call
                    direction = "CE"
                    atm_k = int(round(spot / 100.0) * 100)
                    if use_real:
                        s_call = find_real_opt_sym(real_opts, t, atm_k, "CE")
                        if s_call:
                            entry_p = real_opts[s_call].loc[t, "close"]
                            in_pos = True
                            pos = {"sym": s_call, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}
                    else:
                        T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                        entry_p = black_scholes_price(spot, atm_k, T, 0.07, 0.165, "CE")
                        in_pos = True
                        pos = {"sym": f"BN_SYNTH_{atm_k}CE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}

                elif row["low"] <= (box_l - buffer_pts):
                    # Bearish breakdown -> Buy ATM Put
                    direction = "PE"
                    atm_k = int(round(spot / 100.0) * 100)
                    if use_real:
                        s_put = find_real_opt_sym(real_opts, t, atm_k, "PE")
                        if s_put:
                            entry_p = real_opts[s_put].loc[t, "close"]
                            in_pos = True
                            pos = {"sym": s_put, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}
                    else:
                        T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                        entry_p = black_scholes_price(spot, atm_k, T, 0.07, 0.165, "PE")
                        in_pos = True
                        pos = {"sym": f"BN_SYNTH_{atm_k}PE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t, "peak_p": entry_p}

    return pd.DataFrame(trades)

# =============================================================================
# STRATEGY 3: GAMMA SQUEEZE MOMENTUM (BANKNIFTY CALIBRATED)
# =============================================================================
def eval_gamma(use_real=False):
    data = df_5m.copy()
    data["date"] = data.index.date
    
    # VWAP
    data["typical_p"] = (data["high"] + data["low"] + data["close"]) / 3.0
    data["tp_vol"] = data["typical_p"] * data["volume"]
    data["cum_vol"] = data.groupby("date")["volume"].cumsum()
    data["cum_tp_vol"] = data.groupby("date")["tp_vol"].cumsum()
    data["vwap"] = data["cum_tp_vol"] / data["cum_vol"].clip(lower=1)
    data["vwap"].fillna(data["close"].rolling(20).mean(), inplace=True)
    if data["vwap"].isna().all():
        data["vwap"] = data["close"].rolling(20).mean()

    # 14 ATR
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - data["close"].shift(1)).abs()
    tr3 = (data["low"] - data["close"].shift(1)).abs()
    data["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = data["tr"].rolling(14).mean().fillna(80.0)

    # 50 EMA Macro Trend
    data["ema_50"] = data["close"].ewm(span=50).mean()

    trades = []
    in_pos = False
    pos = None

    for i in range(25, len(data)):
        t = data.index[i]
        row = data.iloc[i]
        spot = row["close"]
        vwap = row["vwap"]
        atr = row["atr"]
        ema50 = row["ema_50"]
        time_str = t.strftime("%H:%M")

        if in_pos:
            if use_real:
                opt_df = real_opts.get(pos["sym"])
                cur_p = opt_df.loc[t, "close"] if (opt_df is not None and t in opt_df.index) else pos["entry_p"]
            else:
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                cur_p = black_scholes_price(spot, pos["strike"], T, 0.07, 0.165, pos["direction"])

            ret_p = (cur_p - pos["entry_p"]) / pos["entry_p"]
            exit_trig = False
            exit_reason = ""

            if ret_p >= 0.70:
                exit_trig = True
                exit_reason = "GAMMA_TARGET (+70%)"
            elif ret_p <= -0.25:
                exit_trig = True
                exit_reason = "STOP_LOSS (-25%)"
            elif (pos["direction"] == "CE" and spot < vwap) or (pos["direction"] == "PE" and spot > vwap):
                exit_trig = True
                exit_reason = "VWAP_REVERSAL"
            elif time_str >= "15:15":
                exit_trig = True
                exit_reason = "EOD_SQUAREOFF"

            if exit_trig:
                gross = (cur_p - pos["entry_p"]) * LOT_SIZE
                net = gross - 45.0
                cap = pos["entry_p"] * LOT_SIZE
                trades.append({
                    "strategy": "Gamma Squeeze Momentum",
                    "entry_time": str(pos["entry_time"])[:16],
                    "exit_time": str(t)[:16],
                    "capital": round(cap, 2),
                    "net_pnl": round(net, 2),
                    "ret_pct": round(net / cap * 100, 2),
                    "reason": exit_reason
                })
                in_pos = False

        else:
            if time_str < "09:30" or time_str > "14:15":
                continue

            # Strict 2.0x ATR expansion beyond VWAP aligned with 50 EMA
            if spot > (vwap + 2.0 * atr) and spot > ema50:
                direction = "CE"
                atm_k = int(round(spot / 100.0) * 100)
                if use_real:
                    s_call = find_real_opt_sym(real_opts, t, atm_k, "CE")
                    if s_call:
                        entry_p = real_opts[s_call].loc[t, "close"]
                        in_pos = True
                        pos = {"sym": s_call, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t}
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    entry_p = black_scholes_price(spot, atm_k, T, 0.07, 0.165, "CE")
                    in_pos = True
                    pos = {"sym": f"BN_SYNTH_{atm_k}CE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t}

            elif spot < (vwap - 2.0 * atr) and spot < ema50:
                direction = "PE"
                atm_k = int(round(spot / 100.0) * 100)
                if use_real:
                    s_put = find_real_opt_sym(real_opts, t, atm_k, "PE")
                    if s_put:
                        entry_p = real_opts[s_put].loc[t, "close"]
                        in_pos = True
                        pos = {"sym": s_put, "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t}
                else:
                    T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                    entry_p = black_scholes_price(spot, atm_k, T, 0.07, 0.165, "PE")
                    in_pos = True
                    pos = {"sym": f"BN_SYNTH_{atm_k}PE", "strike": atm_k, "direction": direction, "entry_p": entry_p, "entry_time": t}

    return pd.DataFrame(trades)

# =============================================================================
# STRATEGY 4: HEDGED STRANGLE (BANKNIFTY CALIBRATED)
# =============================================================================
def eval_hedged_strangle(use_real=False):
    data = df_1m.copy()
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
            if use_real:
                c_df = real_opts.get(pos["ce_sym"])
                p_df = real_opts.get(pos["pe_sym"])
                cur_ce = c_df.loc[t, "close"] if (c_df is not None and t in c_df.index) else pos["ce_entry"]
                cur_pe = p_df.loc[t, "close"] if (p_df is not None and t in p_df.index) else pos["pe_entry"]
            else:
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                cur_ce = black_scholes_price(spot, pos["ce_strike"], T, 0.07, 0.165, "CE")
                cur_pe = black_scholes_price(spot, pos["pe_strike"], T, 0.07, 0.165, "PE")

            init_tot = pos["ce_entry"] + pos["pe_entry"]
            cur_tot = cur_ce + cur_pe
            pnl_pct = (cur_tot - init_tot) / init_tot

            exit_trig = False
            exit_reason = ""

            if pnl_pct >= 0.40:
                exit_trig = True
                exit_reason = "STRANGLE_TARGET (+40%)"
            elif pnl_pct <= -0.20:
                exit_trig = True
                exit_reason = "STRANGLE_STOP (-20%)"
            elif bars_held >= 60:
                exit_trig = True
                exit_reason = "TIME_EXIT_60M"
            elif time_str >= "15:15":
                exit_trig = True
                exit_reason = "EOD_SQUAREOFF"

            if exit_trig:
                gross = (cur_tot - init_tot) * LOT_SIZE
                net = gross - 80.0
                cap = init_tot * LOT_SIZE
                trades.append({
                    "strategy": "Hedged Strangle",
                    "entry_time": str(pos["entry_time"])[:16],
                    "exit_time": str(t)[:16],
                    "capital": round(cap, 2),
                    "net_pnl": round(net, 2),
                    "ret_pct": round(net / cap * 100, 2),
                    "reason": exit_reason
                })
                in_pos = False

        else:
            # Enter strictly on Tuesday/Wednesday morning at 09:35 when theta/vega setup is favorable
            if time_str != "09:35":
                continue
            if t.weekday() not in [1, 2]:  # Tuesday or Wednesday only
                continue

            atm_k = int(round(spot / 100.0) * 100)
            ce_k = atm_k + 300
            pe_k = atm_k - 300

            if use_real:
                s_ce = find_real_opt_sym(real_opts, t, ce_k, "CE")
                s_pe = find_real_opt_sym(real_opts, t, pe_k, "PE")
                if s_ce and s_pe:
                    p_ce = real_opts[s_ce].loc[t, "close"]
                    p_pe = real_opts[s_pe].loc[t, "close"]
                    in_pos = True
                    pos = {"entry_time": t, "entry_idx": i, "ce_sym": s_ce, "pe_sym": s_pe, "ce_entry": p_ce, "pe_entry": p_pe}
            else:
                T = max(1e-4, compute_banknifty_dte(t) / 365.0)
                p_ce = black_scholes_price(spot, ce_k, T, 0.07, 0.165, "CE")
                p_pe = black_scholes_price(spot, pe_k, T, 0.07, 0.165, "PE")
                in_pos = True
                pos = {"entry_time": t, "entry_idx": i, "ce_strike": ce_k, "pe_strike": pe_k, "ce_entry": p_ce, "pe_entry": p_pe}

    return pd.DataFrame(trades)

def get_stats(df):
    if df is None or df.empty:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "avg": 0.0, "max_w": 0.0, "max_l": 0.0}
    w = df[df["net_pnl"] > 0]
    l = df[df["net_pnl"] <= 0]
    wr = len(w) / len(df) * 100
    tot_w = w["net_pnl"].sum() if not w.empty else 0.0
    tot_l = abs(l["net_pnl"].sum()) if not l.empty else 0.0
    pf = tot_w / tot_l if tot_l > 0 else 99.0
    return {
        "trades": len(df), "wr": round(wr, 1), "pf": round(pf, 2),
        "pnl": round(df["net_pnl"].sum(), 2), "avg": round(df["net_pnl"].mean(), 2),
        "max_w": round(df["net_pnl"].max(), 2), "max_l": round(df["net_pnl"].min(), 2)
    }

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("      BANKNIFTY CALIBRATED STRATEGY EVALUATION: SYNTHETIC VS REAL TRADED OPTIONS      ")
    print("=" * 100)

    das_s = eval_das(use_real=False)
    das_r = eval_das(use_real=True)

    box_s = eval_box(use_real=False)
    box_r = eval_box(use_real=True)

    gamma_s = eval_gamma(use_real=False)
    gamma_r = eval_gamma(use_real=True)

    hedged_s = eval_hedged_strangle(use_real=False)
    hedged_r = eval_hedged_strangle(use_real=True)

    results = [
        ("1. Decoupled Asymmetric Strangle", "Synthetic (Black-Scholes)", get_stats(das_s)),
        ("1. Decoupled Asymmetric Strangle", "Real SmartAPI Options", get_stats(das_r)),
        ("2. Directional Box Breakout", "Synthetic (Black-Scholes)", get_stats(box_s)),
        ("2. Directional Box Breakout", "Real SmartAPI Options", get_stats(box_r)),
        ("3. Gamma Squeeze Momentum", "Synthetic (Black-Scholes)", get_stats(gamma_s)),
        ("3. Gamma Squeeze Momentum", "Real SmartAPI Options", get_stats(gamma_r)),
        ("4. Hedged Strangle", "Synthetic (Black-Scholes)", get_stats(hedged_s)),
        ("4. Hedged Strangle", "Real SmartAPI Options", get_stats(hedged_r)),
    ]

    table_data = []
    for strat, feed, m in results:
        table_data.append({
            "Strategy": strat,
            "Feed": feed,
            "Trades": m["trades"],
            "Win Rate %": f"{m['wr']}%",
            "Profit Factor": m["pf"],
            "Net PnL (INR)": f"Rs. {m['pnl']:+,.2f}",
            "Avg Trade PnL": f"Rs. {m['avg']:+,.2f}",
            "Max Win": f"Rs. {m['max_w']:+,.2f}",
            "Max Loss": f"Rs. {m['max_l']:+,.2f}"
        })

    print(tabulate(table_data, headers="keys", tablefmt="grid"))

    os.makedirs("data/banknifty_results", exist_ok=True)
    if not das_s.empty: das_s.to_csv("data/banknifty_results/das_synthetic_ledger.csv", index=False)
    if not das_r.empty: das_r.to_csv("data/banknifty_results/das_real_options_ledger.csv", index=False)
    if not box_s.empty: box_s.to_csv("data/banknifty_results/box_synthetic_ledger.csv", index=False)
    if not box_r.empty: box_r.to_csv("data/banknifty_results/box_real_options_ledger.csv", index=False)
    if not gamma_s.empty: gamma_s.to_csv("data/banknifty_results/gamma_synthetic_ledger.csv", index=False)
    if not gamma_r.empty: gamma_r.to_csv("data/banknifty_results/gamma_real_options_ledger.csv", index=False)
    if not hedged_s.empty: hedged_s.to_csv("data/banknifty_results/hedged_synthetic_ledger.csv", index=False)
    if not hedged_r.empty: hedged_r.to_csv("data/banknifty_results/hedged_real_options_ledger.csv", index=False)
    print("\nSaved all individual trade ledgers to data/banknifty_results/")

