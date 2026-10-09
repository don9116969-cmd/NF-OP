"""
BankNIFTY Multi-Strategy Engine:
1. Strategy 6: BankNIFTY Directional Box Breakout (DBB)
2. Strategy 7: BankNIFTY Expiry Gamma Squeeze Momentum (GSM)
3. Strategy 8: BankNIFTY Hedged Long Strangle (15M Box)

Connects to Angel One SmartAPI for real option contract selection and pricing,
with Black-Scholes mathematical fallback for offline/historical backtesting.
Enforces INR 10,000 maximum budget per trade.
"""

import os
import math
import datetime
import pandas as pd
import numpy as np

from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles
from src.banknifty.banknifty_das import bs_price, get_banknifty_dte, LOT_SIZE, SLIPPAGE_PCT

FEE_DIRECTIONAL = 45.0
FEE_STRANGLE = 80.0
MAX_TRADE_BUDGET = 8500.0

def select_banknifty_directional_strike(spot: float, opt_type: str, dte: float, target_date: datetime.date, api=None):
    """Selects an affordable BankNIFTY strike so capital <= Rs 8,500."""
    atm = int(round(spot / 100.0) * 100)
    chosen_strike = None
    chosen_p = 0.0

    for offset in range(0, 8000, 100):
        k = atm + offset if opt_type == "CE" else atm - offset
        p = bs_price(spot, k, dte, opt_type=opt_type, sigma=0.18)
        if p * LOT_SIZE <= MAX_TRADE_BUDGET:
            chosen_strike = k
            chosen_p = p
            break

    if chosen_strike is None:
        return None, 0.0, None

    c_opt = get_active_option_contract(target_date, chosen_strike, opt_type, underlying="BANKNIFTY") if api else None
    return chosen_strike, chosen_p, c_opt

def select_banknifty_strangle_strikes(spot: float, dte: float, target_date: datetime.date, api=None):
    """Selects affordable CE and PE strikes for BankNIFTY Strangle so total capital <= Rs 8,500."""
    atm = int(round(spot / 100.0) * 100)
    chosen_offset = None
    chosen_ce_p = 0.0
    chosen_pe_p = 0.0

    for offset in range(200, 8000, 100):
        k_ce = atm + offset
        k_pe = atm - offset
        p_ce = bs_price(spot, k_ce, dte, "CE", sigma=0.18)
        p_pe = bs_price(spot, k_pe, dte, "PE", sigma=0.18)
        if (p_ce + p_pe) * LOT_SIZE <= MAX_TRADE_BUDGET:
            chosen_offset = offset
            chosen_ce_p = p_ce
            chosen_pe_p = p_pe
            break

    if chosen_offset is None:
        return None, None, 0.0, 0.0, None, None

    k_ce = atm + chosen_offset
    k_pe = atm - chosen_offset
    c_ce = get_active_option_contract(target_date, k_ce, "CE", underlying="BANKNIFTY") if api else None
    c_pe = get_active_option_contract(target_date, k_pe, "PE", underlying="BANKNIFTY") if api else None
    return k_ce, k_pe, chosen_ce_p, chosen_pe_p, c_ce, c_pe


def evaluate_banknifty_box_for_day(df_bn: pd.DataFrame, target_date: datetime.date, api=None) -> dict:
    """
    Strategy 6: BankNIFTY Directional Box Breakout (DBB)
    - Box Window: 09:15 to 09:30 IST (15 minutes).
    - Filter: (box_high - box_low) / spot <= 1.2% (filters wide chop).
    - Breakout Buffer: 30.0 points above box_high (CE) or below box_low (PE).
    - Strike: Affordable ATM/OTM Strike (budget <= Rs 8,500).
    - Exit: Target +60%, Stop Loss -20%, EOD 15:15 IST.
    """
    info = {"trade_occurred": False, "status": "NO_DATA"}
    if df_bn.empty:
        info["reason"] = "No market data available"
        return info

    df = df_bn.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    day_df = df[df["timestamp"].dt.date == target_date].sort_values("timestamp")

    if len(day_df) < 15:
        info["reason"] = "Insufficient candles (<15 bars) for BankNIFTY box formation"
        return info

    box_bars = day_df[(day_df["timestamp"].dt.time >= datetime.time(9, 15)) & (day_df["timestamp"].dt.time <= datetime.time(9, 30))]
    if len(box_bars) < 8:
        info["status"] = "INCOMPLETE"
        info["reason"] = "15-minute box is still forming"
        return info

    box_h = float(box_bars["high"].max())
    box_l = float(box_bars["low"].min())
    spot_open = float(box_bars["close"].iloc[0])
    box_pct = ((box_h - box_l) / spot_open) * 100.0

    info["box_high"] = round(box_h, 2)
    info["box_low"] = round(box_l, 2)
    info["box_range"] = round(box_h - box_l, 2)
    info["box_pct"] = round(box_pct, 2)
    info["upper_trigger"] = round(box_h + 30.0, 2)
    info["lower_trigger"] = round(box_l - 30.0, 2)

    if box_pct > 1.2:
        info["status"] = "FILTERED_WIDE_BOX"
        info["reason"] = f"Box width {box_h - box_l:.1f} pts ({box_pct:.2f}% > 1.2%). Filtered out to avoid chop."
        return info

    post_box = day_df[(day_df["timestamp"].dt.time > datetime.time(9, 30)) & (day_df["timestamp"].dt.time <= datetime.time(15, 15))]
    if post_box.empty:
        info["status"] = "NO_BREAKOUT"
        info["reason"] = "No candles post-box window"
        return info

    for _, bar in post_box.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        ts = bar["timestamp"]

        is_call_break = h >= (box_h + 30.0)
        is_put_break = l <= (box_l - 30.0)

        if not (is_call_break or is_put_break):
            continue

        opt_type = "CE" if is_call_break else "PE"
        dte_y = max(1e-4, get_banknifty_dte(ts, target_date))
        chosen_strike, chosen_p, c_opt = select_banknifty_directional_strike(c, opt_type, dte_y, target_date, api=api)
        if chosen_strike is None:
            info["status"] = "BUDGET_EXCEEDED"
            info["trade_occurred"] = False
            info["reason"] = "Breakout triggered, but option premiums exceeded the INR 8,500 capital limit."
            return info

        # Real option resolution
        if api and c_opt:
            df_opt = fetch_real_option_candles(api, c_opt, ts, day_df["timestamp"].max())
            if not df_opt.empty:
                sub_opt = df_opt[df_opt.index >= ts]
                if not sub_opt.empty:
                    entry_p = round(float(sub_opt.iloc[0]["close"]) * (1.0 + SLIPPAGE_PCT), 2)
                    cost = round(entry_p * LOT_SIZE, 2)
                    if cost > MAX_TRADE_BUDGET:
                        info["status"] = "BUDGET_EXCEEDED"
                        info["trade_occurred"] = False
                        info["reason"] = f"Breakout triggered, but real option contract {c_opt['symbol']} cost (INR {cost:.2f}) exceeded INR 8,500 budget limit."
                        return info

                    target_p = entry_p * 1.60
                    sl_p = entry_p * 0.80

                    exit_p = entry_p
                    exit_ts = sub_opt.index[-1]
                    exit_reason = "EOD (15:15 PM)"

                    for o_ts, o_bar in sub_opt.iloc[1:].iterrows():
                        o_h = float(o_bar["high"])
                        o_l = float(o_bar["low"])
                        o_c = float(o_bar["close"])

                        if o_h >= target_p:
                            exit_p = round(target_p * (1.0 - SLIPPAGE_PCT), 2)
                            exit_ts = o_ts
                            exit_reason = "Target Hit (+60%)"
                            break
                        elif o_l <= sl_p:
                            exit_p = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
                            exit_ts = o_ts
                            exit_reason = "SL Hit (-20%)"
                            break
                        elif o_ts.time() >= datetime.time(15, 15):
                            exit_p = round(o_c * (1.0 - SLIPPAGE_PCT), 2)
                            exit_ts = o_ts
                            exit_reason = "EOD (15:15 PM)"
                            break

                    gross = (exit_p - entry_p) * LOT_SIZE
                    turnover = (entry_p + exit_p) * LOT_SIZE
                    stt = (exit_p * LOT_SIZE) * 0.001
                    exch = turnover * 0.0005
                    charges = round(FEE_DIRECTIONAL + stt + exch + (FEE_DIRECTIONAL + exch) * 0.18, 2)
                    net = round(gross - charges, 2)

                    info.update({
                        "trade_occurred": True,
                        "status": "TRADE_EXECUTED",
                        "strategy": "BankNIFTY Directional Box Breakout (DBB)",
                        "strike": f"{c_opt['symbol']}",
                        "opt_type": opt_type,
                        "entry_time": ts.time().strftime("%H:%M:%S"),
                        "exit_time": exit_ts.time().strftime("%H:%M:%S") if hasattr(exit_ts, 'time') else str(exit_ts),
                        "entry_p": entry_p,
                        "exit_p": exit_p,
                        "cost": cost,
                        "gross_pnl": round(gross, 2),
                        "charges": charges,
                        "net_pnl": net,
                        "exit_reason": exit_reason,
                        "data_feed": f"Angel One Real Traded ({c_opt['symbol']})"
                    })
                    return info

        # Mathematical BSM Mode
        entry_p = chosen_p
        cost = round(entry_p * LOT_SIZE, 2)
        target_p = entry_p * 1.60
        sl_p = entry_p * 0.80

        exit_p = entry_p
        exit_ts = day_df["timestamp"].max()
        exit_reason = "EOD (15:15 PM)"

        for _, sub_b in post_box[post_box["timestamp"] > ts].iterrows():
            sub_ts = sub_b["timestamp"]
            sub_c = float(sub_b["close"])
            sub_dte = max(1e-4, get_banknifty_dte(sub_ts, target_date))
            cur_p = bs_price(sub_c, chosen_strike, sub_dte, opt_type=opt_type, sigma=0.18)

            if cur_p >= target_p:
                exit_p = target_p
                exit_ts = sub_ts
                exit_reason = "Target Hit (+60%)"
                break
            elif cur_p <= sl_p:
                exit_p = sl_p
                exit_ts = sub_ts
                exit_reason = "SL Hit (-20%)"
                break
            elif sub_ts.time() >= datetime.time(15, 15):
                exit_p = cur_p
                exit_ts = sub_ts
                exit_reason = "EOD (15:15 PM)"
                break

        gross = (exit_p - entry_p) * LOT_SIZE
        charges = round(FEE_DIRECTIONAL + (entry_p + exit_p) * LOT_SIZE * 0.0015, 2)
        net = round(gross - charges, 2)

        info.update({
            "trade_occurred": True,
            "status": "TRADE_EXECUTED",
            "strategy": "BankNIFTY Directional Box Breakout (DBB)",
            "strike": f"{chosen_strike} {opt_type}",
            "opt_type": opt_type,
            "entry_time": ts.time().strftime("%H:%M:%S"),
            "exit_time": exit_ts.time().strftime("%H:%M:%S") if hasattr(exit_ts, 'time') else str(exit_ts),
            "entry_p": round(entry_p, 2),
            "exit_p": round(exit_p, 2),
            "cost": cost,
            "gross_pnl": round(gross, 2),
            "charges": charges,
            "net_pnl": net,
            "exit_reason": exit_reason,
            "data_feed": "Mathematical (BSM)"
        })
        return info

    info["status"] = "NO_BREAKOUT"
    info["reason"] = f"Price stayed within triggers ({info['lower_trigger']} - {info['upper_trigger']}). No breakout."
    return info


def evaluate_banknifty_gamma_for_day(df_bn: pd.DataFrame, target_date: datetime.date, api=None) -> dict:
    """
    Strategy 7: BankNIFTY Expiry Gamma Squeeze Momentum (GSM)
    - Window: 09:30 to 14:30 IST.
    - Session Mean (expanding typical price) + 1.2x ATR(14) breakout.
    - Strike: Affordable ATM/OTM Strike (budget <= Rs 8,500).
    - Exit: Target +40%, Stop Loss -20%, Mean Reversal, or 15:15 EOD.
    """
    info = {"trade_occurred": False, "status": "NO_DATA"}
    if df_bn.empty:
        info["reason"] = "No market data available"
        return info

    df = df_bn.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    day_df = df[df["timestamp"].dt.date == target_date].sort_values("timestamp")

    if len(day_df) < 25:
        info["reason"] = "Insufficient bars for 5M aggregation & ATR calculation"
        return info

    # Resample 5-minute bars
    day_5m = day_df.set_index("timestamp").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()

    if len(day_5m) < 10:
        info["status"] = "INCOMPLETE"
        info["reason"] = "Insufficient 5-minute bars"
        return info

    day_5m["typical_p"] = (day_5m["high"] + day_5m["low"] + day_5m["close"]) / 3.0
    day_5m["session_mean"] = day_5m["typical_p"].expanding().mean()
    tr1 = day_5m["high"] - day_5m["low"]
    tr2 = (day_5m["high"] - day_5m["close"].shift(1)).abs()
    tr3 = (day_5m["low"] - day_5m["close"].shift(1)).abs()
    day_5m["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=5).mean().bfill()

    valid_window = day_5m[(day_5m["timestamp"].dt.time >= datetime.time(9, 30)) & (day_5m["timestamp"].dt.time <= datetime.time(14, 30))]

    for i in range(1, len(valid_window)):
        prev = valid_window.iloc[i-1]
        cur = valid_window.iloc[i]
        ts = cur["timestamp"]

        spot = float(cur["close"])
        s_mean = float(cur["session_mean"])
        atr = float(cur["atr"])
        prev_s_mean = float(prev["session_mean"])
        prev_atr = float(prev["atr"])
        prev_spot = float(prev["close"])

        f_ce = (prev_spot <= prev_s_mean + 1.2 * prev_atr) and (spot > s_mean + 1.2 * atr)
        f_pe = (prev_spot >= prev_s_mean - 1.2 * prev_atr) and (spot < s_mean - 1.2 * atr)

        if not (f_ce or f_pe):
            continue

        opt_type = "CE" if f_ce else "PE"
        dte_y = max(1e-4, get_banknifty_dte(ts, target_date))
        chosen_strike, chosen_p, c_opt = select_banknifty_directional_strike(spot, opt_type, dte_y, target_date, api=api)
        if chosen_strike is None:
            info["status"] = "BUDGET_EXCEEDED"
            info["trade_occurred"] = False
            info["reason"] = "Breakout triggered, but option premiums exceeded the INR 8,500 capital limit."
            return info

        # Real option execution
        if api and c_opt:
            df_opt = fetch_real_option_candles(api, c_opt, ts, day_df["timestamp"].max())
            if not df_opt.empty:
                sub_opt = df_opt[df_opt.index >= ts]
                if not sub_opt.empty:
                    entry_p = round(float(sub_opt.iloc[0]["close"]) * (1.0 + SLIPPAGE_PCT), 2)
                    cost = round(entry_p * LOT_SIZE, 2)
                    if cost > MAX_TRADE_BUDGET:
                        info["status"] = "BUDGET_EXCEEDED"
                        info["trade_occurred"] = False
                        info["reason"] = f"Breakout triggered, but real option contract {c_opt['symbol']} cost (INR {cost:.2f}) exceeded INR 8,500 budget limit."
                        return info

                    target_p = entry_p * 1.40
                    sl_p = entry_p * 0.80

                    exit_p = entry_p
                    exit_ts = sub_opt.index[-1]
                    exit_reason = "EOD (15:15 PM)"

                    for o_ts, o_bar in sub_opt.iloc[1:].iterrows():
                        o_h = float(o_bar["high"])
                        o_l = float(o_bar["low"])
                        o_c = float(o_bar["close"])

                        if o_h >= target_p:
                            exit_p = round(target_p * (1.0 - SLIPPAGE_PCT), 2)
                            exit_ts = o_ts
                            exit_reason = "Target Hit (+40%)"
                            break
                        elif o_l <= sl_p:
                            exit_p = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
                            exit_ts = o_ts
                            exit_reason = "SL Hit (-20%)"
                            break
                        elif o_ts.time() >= datetime.time(15, 15):
                            exit_p = round(o_c * (1.0 - SLIPPAGE_PCT), 2)
                            exit_ts = o_ts
                            exit_reason = "EOD (15:15 PM)"
                            break

                    gross = (exit_p - entry_p) * LOT_SIZE
                    turnover = (entry_p + exit_p) * LOT_SIZE
                    stt = (exit_p * LOT_SIZE) * 0.001
                    exch = turnover * 0.0005
                    charges = round(FEE_DIRECTIONAL + stt + exch + (FEE_DIRECTIONAL + exch) * 0.18, 2)
                    net = round(gross - charges, 2)

                    info.update({
                        "trade_occurred": True,
                        "status": "TRADE_EXECUTED",
                        "strategy": "BankNIFTY Expiry Gamma Squeeze (GSM)",
                        "strike": f"{c_opt['symbol']}",
                        "opt_type": opt_type,
                        "entry_time": ts.time().strftime("%H:%M:%S"),
                        "exit_time": exit_ts.time().strftime("%H:%M:%S") if hasattr(exit_ts, 'time') else str(exit_ts),
                        "entry_p": entry_p,
                        "exit_p": exit_p,
                        "cost": cost,
                        "gross_pnl": round(gross, 2),
                        "charges": charges,
                        "net_pnl": net,
                        "exit_reason": exit_reason,
                        "data_feed": f"Angel One Real Traded ({c_opt['symbol']})"
                    })
                    return info

        # Mathematical BSM Mode
        entry_p = chosen_p
        cost = round(entry_p * LOT_SIZE, 2)
        target_p = entry_p * 1.40
        sl_p = entry_p * 0.80

        exit_p = entry_p
        exit_ts = day_5m["timestamp"].max()
        exit_reason = "EOD (15:15 PM)"

        for _, sub_b in day_5m[day_5m["timestamp"] > ts].iterrows():
            sub_ts = sub_b["timestamp"]
            sub_c = float(sub_b["close"])
            sub_mean = float(sub_b["session_mean"])
            sub_dte = max(1e-4, get_banknifty_dte(sub_ts, target_date))
            cur_p = bs_price(sub_c, chosen_strike, sub_dte, opt_type=opt_type, sigma=0.18)

            is_rev = (opt_type == "CE" and sub_c < sub_mean) or (opt_type == "PE" and sub_c > sub_mean)

            if cur_p >= target_p:
                exit_p = target_p
                exit_ts = sub_ts
                exit_reason = "Target Hit (+40%)"
                break
            elif cur_p <= sl_p:
                exit_p = sl_p
                exit_ts = sub_ts
                exit_reason = "SL Hit (-20%)"
                break
            elif is_rev:
                exit_p = cur_p
                exit_ts = sub_ts
                exit_reason = "Session Mean Reversal"
                break
            elif sub_ts.time() >= datetime.time(15, 15):
                exit_p = cur_p
                exit_ts = sub_ts
                exit_reason = "EOD (15:15 PM)"
                break

        gross = (exit_p - entry_p) * LOT_SIZE
        charges = round(FEE_DIRECTIONAL + (entry_p + exit_p) * LOT_SIZE * 0.0015, 2)
        net = round(gross - charges, 2)

        info.update({
            "trade_occurred": True,
            "status": "TRADE_EXECUTED",
            "strategy": "BankNIFTY Expiry Gamma Squeeze (GSM)",
            "strike": f"{chosen_strike} {opt_type}",
            "opt_type": opt_type,
            "entry_time": ts.time().strftime("%H:%M:%S"),
            "exit_time": exit_ts.time().strftime("%H:%M:%S") if hasattr(exit_ts, 'time') else str(exit_ts),
            "entry_p": round(entry_p, 2),
            "exit_p": round(exit_p, 2),
            "cost": cost,
            "gross_pnl": round(gross, 2),
            "charges": charges,
            "net_pnl": net,
            "exit_reason": exit_reason,
            "data_feed": "Mathematical (BSM)"
        })
        return info

    info["status"] = "NO_TRIGGER"
    info["reason"] = "No 1.2x ATR breakout from expanding session mean."
    return info


def evaluate_banknifty_strangle_for_day(df_bn: pd.DataFrame, target_date: datetime.date, api=None) -> dict:
    """
    Strategy 8: BankNIFTY Hedged Long Strangle (15M Box)
    - Entry: 09:30 IST upon completion of opening 15-minute box.
    - Strikes: Affordable Spot +/- OTM strikes (budget <= Rs 8,500).
    - Exit: Target +70%, Stop Loss -30%, Max Hold 45 mins, EOD 15:15 IST.
    """
    info = {"trade_occurred": False, "status": "NO_DATA"}
    if df_bn.empty:
        info["reason"] = "No market data available"
        return info

    df = df_bn.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    day_df = df[df["timestamp"].dt.date == target_date].sort_values("timestamp")

    if len(day_df) < 15:
        info["reason"] = "Insufficient bars (<15 bars) for 15M box"
        return info

    entry_bars = day_df[day_df["timestamp"].dt.time >= datetime.time(9, 30)]
    if entry_bars.empty:
        info["status"] = "INCOMPLETE"
        info["reason"] = "Market has not reached 09:30 AM IST"
        return info

    entry_row = entry_bars.iloc[0]
    entry_ts = entry_row["timestamp"]
    spot = float(entry_row["close"])

    dte_y = max(1e-4, get_banknifty_dte(entry_ts, target_date))
    k_ce, k_pe, p_ce, p_pe, c_ce, c_pe = select_banknifty_strangle_strikes(spot, dte_y, target_date, api=api)
    if k_ce is None or k_pe is None:
        info["status"] = "BUDGET_EXCEEDED"
        info["trade_occurred"] = False
        info["reason"] = "Strangle option premiums exceeded the INR 8,500 capital limit."
        return info

    if api and c_ce and c_pe:
        df_ce = fetch_real_option_candles(api, c_ce, entry_ts, day_df["timestamp"].max())
        df_pe = fetch_real_option_candles(api, c_pe, entry_ts, day_df["timestamp"].max())
        if not df_ce.empty and not df_pe.empty:
            sub_ce = df_ce[df_ce.index >= entry_ts]
            sub_pe = df_pe[df_pe.index >= entry_ts]
            if not sub_ce.empty and not sub_pe.empty:
                entry_ce = round(float(sub_ce.iloc[0]["close"]) * (1.0 + SLIPPAGE_PCT), 2)
                entry_pe = round(float(sub_pe.iloc[0]["close"]) * (1.0 + SLIPPAGE_PCT), 2)
                entry_tot = round(entry_ce + entry_pe, 2)
                cost = round(entry_tot * LOT_SIZE, 2)
                if cost > MAX_TRADE_BUDGET:
                    info["status"] = "BUDGET_EXCEEDED"
                    info["trade_occurred"] = False
                    info["reason"] = f"Real strangle cost (INR {cost:.2f}) exceeded INR 8,500 budget limit."
                    return info

                target_p = entry_tot * 1.70
                sl_p = entry_tot * 0.70

                exit_tot = entry_tot
                exit_ts = entry_ts
                exit_reason = "Time Exit (45m)"

                common_idx = sub_ce.index.intersection(sub_pe.index)
                for ts in common_idx[1:]:
                    elapsed_m = (ts - entry_ts).total_seconds() / 60.0
                    c_tot = float(sub_ce.loc[ts]["close"]) + float(sub_pe.loc[ts]["close"])

                    if c_tot >= target_p:
                        exit_tot = round(target_p * (1.0 - SLIPPAGE_PCT), 2)
                        exit_ts = ts
                        exit_reason = "Target Hit (+70%)"
                        break
                    elif c_tot <= sl_p:
                        exit_tot = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
                        exit_ts = ts
                        exit_reason = "Basket SL Hit (-30%)"
                        break
                    elif elapsed_m >= 45.0 or ts.time() >= datetime.time(15, 15):
                        exit_tot = round(c_tot * (1.0 - SLIPPAGE_PCT), 2)
                        exit_ts = ts
                        exit_reason = "Time Exit (45m)"
                        break

                gross = (exit_tot - entry_tot) * LOT_SIZE
                turnover = (entry_tot + exit_tot) * LOT_SIZE
                stt = (exit_tot * LOT_SIZE) * 0.001
                exch = turnover * 0.0005
                charges = round(FEE_STRANGLE + stt + exch + (FEE_STRANGLE + exch) * 0.18, 2)
                net = round(gross - charges, 2)

                info.update({
                    "trade_occurred": True,
                    "status": "TRADE_EXECUTED",
                    "strategy": "BankNIFTY Hedged Long Strangle (15M Box)",
                    "strike": f"{c_ce['symbol']} + {c_pe['symbol']}",
                    "opt_type": "CE+PE",
                    "entry_time": entry_ts.time().strftime("%H:%M:%S"),
                    "exit_time": exit_ts.time().strftime("%H:%M:%S") if hasattr(exit_ts, 'time') else str(exit_ts),
                    "entry_p": entry_tot,
                    "exit_p": exit_tot,
                    "cost": cost,
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": f"Angel One Real Traded ({c_ce['symbol']} + {c_pe['symbol']})"
                })
                return info

    # Mathematical BSM Mode
    entry_tot = round(p_ce + p_pe, 2)
    cost = round(entry_tot * LOT_SIZE, 2)

    target_p = entry_tot * 1.70
    sl_p = entry_tot * 0.70

    exit_tot = entry_tot
    exit_ts = day_df["timestamp"].max()
    exit_reason = "Time Exit (45m)"

    post_entry = day_df[day_df["timestamp"] > entry_ts]
    for _, b in post_entry.iterrows():
        b_ts = b["timestamp"]
        b_spot = float(b["close"])
        elapsed_m = (b_ts - entry_ts).total_seconds() / 60.0
        b_dte = max(1e-4, get_banknifty_dte(b_ts, target_date))

        cur_tot = bs_price(b_spot, k_ce, b_dte, "CE", sigma=0.18) + bs_price(b_spot, k_pe, b_dte, "PE", sigma=0.18)

        if cur_tot >= target_p:
            exit_tot = target_p
            exit_ts = b_ts
            exit_reason = "Target Hit (+70%)"
            break
        elif cur_tot <= sl_p:
            exit_tot = sl_p
            exit_ts = b_ts
            exit_reason = "Basket SL Hit (-30%)"
            break
        elif elapsed_m >= 45.0 or b_ts.time() >= datetime.time(15, 15):
            exit_tot = cur_tot
            exit_ts = b_ts
            exit_reason = "Time Exit (45m)"
            break

    gross = (exit_tot - entry_tot) * LOT_SIZE
    charges = round(FEE_STRANGLE + (entry_tot + exit_tot) * LOT_SIZE * 0.0015, 2)
    net = round(gross - charges, 2)

    info.update({
        "trade_occurred": True,
        "status": "TRADE_EXECUTED",
        "strategy": "BankNIFTY Hedged Long Strangle (15M Box)",
        "strike": f"{k_ce}CE + {k_pe}PE",
        "opt_type": "CE+PE",
        "entry_time": entry_ts.time().strftime("%H:%M:%S"),
        "exit_time": exit_ts.time().strftime("%H:%M:%S") if hasattr(exit_ts, 'time') else str(exit_ts),
        "entry_p": entry_tot,
        "exit_p": round(exit_tot, 2),
        "cost": cost,
        "gross_pnl": round(gross, 2),
        "charges": charges,
        "net_pnl": net,
        "exit_reason": exit_reason,
        "data_feed": "Mathematical (BSM)"
    })
    return info
