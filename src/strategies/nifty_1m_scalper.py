"""
Strategy 4: Nifty 1M Pullback Option Scalper (Behavioral Mirror of FirstOne.xlsx).
Execution Architecture:
- Timeframe: 1-Minute Nifty 50 Spot Candles
- Indicators: VWAP, EMA 9, EMA 21, EMA 50
- Morning Bias (09:20–09:30 AM):
    * Spot > VWAP and EMA 9 > EMA 21 -> CALL Only (+1)
    * Spot < VWAP and EMA 9 < EMA 21 -> PUT Only (-1)
- Entry Trigger (Pullback Continuation Bounce):
    * Bullish: Low touches/nears EMA 21 (<= EMA 21 * 1.0015) and closes green above Open & EMA 9.
    * Bearish: High touches/nears EMA 21 (>= EMA 21 * 0.9985) and closes red below Open & EMA 9.
- Strike Selection: Nearest weekly ATM strike (budget <= Rs 9,500).
- Exit Rules:
    * Profit Target: +10.0% on Option Premium
    * Stop Loss: -11.5% on Option Premium
    * Single-Loss Kill Switch: Exactly 1 loss terminates trading for the day.
    * Win Cap: Max 4 wins per day.
    * Cooldown: 10 minutes between trades.
    * EOD Squareoff: 15:15 IST.
"""

import os
import math
import datetime
import pandas as pd
import numpy as np

from config import config
from src.features.greeks import black_scholes_price, compute_calendar_dte
from src.data.real_option_feed import (
    get_active_option_contract,
    fetch_real_option_candles,
    get_candle_feed,
    select_dynamic_affordable_strike
)

def compute_indicators(df_candles: pd.DataFrame) -> pd.DataFrame:
    """Computes VWAP, EMA 9, EMA 21, and EMA 50 on 1-min candles."""
    df = df_candles.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["date"] = df["timestamp"].dt.date
        df["time_min"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    df["hlc3"] = hlc3
    df["vwap"] = df.groupby("date")["hlc3"].transform(lambda s: s.cumsum() / np.arange(1, len(s) + 1))
    df["ema_9"] = df["close"].ewm(span=config.S4_FAST_EMA, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=config.S4_SLOW_EMA, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=config.S4_TREND_EMA, adjust=False).mean()
    return df

def determine_morning_bias(day_bars: pd.DataFrame) -> int:
    """
    Determines single directional bias (+1 for CALL, -1 for PUT) based on 09:20-09:30 AM price action.
    Zero lookahead: Only reads bars up to 09:30 AM.
    """
    bars_30 = day_bars[day_bars["timestamp"].dt.time <= datetime.time(9, 30)]
    if bars_30.empty:
        return 0

    last_bar = bars_30.iloc[-1]
    c = float(last_bar["close"])
    v = float(last_bar["vwap"])
    e9 = float(last_bar["ema_9"])
    e21 = float(last_bar["ema_21"])

    if c > v and e9 > e21:
        return 1  # Bullish -> CALL only
    elif c < v and e9 < e21:
        return -1 # Bearish -> PUT only
    else:
        return 1 if c > v else -1

def check_pullback_trigger(row: pd.Series, prev_row: pd.Series, bias: int) -> bool:
    """Checks whether a 1-minute candle triggers a pullback bounce in the direction of bias."""
    c = float(row["close"])
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    v = float(row["vwap"])
    e9 = float(row["ema_9"])
    e21 = float(row["ema_21"])

    if bias == 1:
        # Bullish: Above VWAP, EMA9 > EMA21, Low dipped near EMA21, and Close bounced green
        if c > v and e9 > e21 and l <= e21 * 1.0015 and c > o and c > e9:
            return True
    elif bias == -1:
        # Bearish: Below VWAP, EMA9 < EMA21, High pulled back to EMA21, and Close bounced red
        if c < v and e9 < e21 and h >= e21 * 0.9985 and c < o and c < e9:
            return True

    return False

def evaluate_strategy_4(df_full: pd.DataFrame, target_date: datetime.date, api = None) -> dict:
    """
    Evaluates Strategy 4 for a single day.
    Returns full day trade statistics matching the daily result checker format.
    """
    if config.S4_SKIP_TUESDAY and target_date.weekday() == 1:
        return {
            "status": "SKIPPED_EXPIRY",
            "trade_occurred": False,
            "reason": "Tuesday Expiry Day skipped as per S4_SKIP_TUESDAY rule."
        }

    # Ensure indicators are computed
    if "ema_9" not in df_full.columns or "vwap" not in df_full.columns:
        data = compute_indicators(df_full)
    else:
        data = df_full

    day_bars = data[data["timestamp"].dt.date == target_date].copy().reset_index(drop=True)
    if day_bars.empty or len(day_bars) < 20:
        return {"status": "NO_DATA", "trade_occurred": False, "reason": "No market data (holiday/weekend)"}

    # Determine morning bias
    bias = determine_morning_bias(day_bars)
    bias_str = "CALL (BUY CE)" if bias == 1 else ("PUT (BUY PE)" if bias == -1 else "NEUTRAL")

    start_h, start_m = map(int, config.S4_START_TIME.split(":"))
    cutoff_h, cutoff_m = map(int, config.S4_ENTRY_CUTOFF.split(":"))

    eligible_bars = day_bars[
        (day_bars["timestamp"].dt.time >= datetime.time(start_h, start_m)) &
        (day_bars["timestamp"].dt.time <= datetime.time(cutoff_h, cutoff_m))
    ].reset_index(drop=True)

    if eligible_bars.empty:
        return {"status": "NO_DATA", "trade_occurred": False, "reason": "No bars within trading window."}

    daily_trades = []
    wins_count = 0
    losses_count = 0
    next_eligible_idx = 0

    lot_size = config.LOT_SIZE

    for idx in range(1, len(eligible_bars)):
        if idx < next_eligible_idx:
            continue
        if wins_count >= config.S4_MAX_WINS:
            break
        if losses_count >= config.S4_MAX_LOSSES:
            # Single-loss kill switch activated!
            break

        curr_row = eligible_bars.iloc[idx]
        prev_row = eligible_bars.iloc[idx - 1]

        if not check_pullback_trigger(curr_row, prev_row, bias):
            continue

        # Trigger fired!
        entry_ts = curr_row["timestamp"]
        entry_spot = float(curr_row["close"])
        opt_type = "CE" if bias == 1 else "PE"
        initial_strike = int(round(entry_spot / 50.0) * 50)

        # Dynamic strike selection for real ATM/OTM option contract
        if config.DYNAMIC_STRIKE_ENABLED:
            strike, df_opt, sym, feed_label, _, _ = select_dynamic_affordable_strike(
                api, target_date, entry_spot, opt_type, "NIFTY", day_bars, entry_ts=entry_ts, max_budget=config.S4_MAX_BUDGET
            )
        else:
            strike = initial_strike
            df_opt, sym, feed_label = get_candle_feed(api, target_date, strike, opt_type, "NIFTY", day_bars)

        sub_opt = df_opt[df_opt.index >= entry_ts]
        if sub_opt.empty:
            continue

        entry_p = round(float(sub_opt.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
        cost = round(entry_p * lot_size, 2)
        if cost > config.S4_MAX_BUDGET:
            continue

        tgt_p = round(entry_p * (1.0 + config.S4_TARGET_PCT), 2)
        sl_p = round(entry_p * (1.0 - config.S4_SL_PCT), 2)

        exit_p = entry_p
        exit_ts = sub_opt.index[-1]
        exit_reason = "EOD (15:15)"
        bars_held = 0

        # Evaluate sub-candles
        for o_ts, o_row in sub_opt.iloc[1:].iterrows():
            bars_held += 1
            # Strict Risk-First evaluation (SL checked before Target to prevent lookahead)
            if float(o_row["low"]) <= sl_p:
                exit_p = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_ts = o_ts
                exit_reason = f"SL Hit (-{int(config.S4_SL_PCT * 100)}%)"
                break
            elif float(o_row["high"]) >= tgt_p:
                exit_p = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_ts = o_ts
                exit_reason = f"Target Hit (+{int(config.S4_TARGET_PCT * 100)}%)"
                break
            elif o_ts.time() >= datetime.time(15, 15):
                exit_p = round(float(o_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_ts = o_ts
                exit_reason = "EOD (15:15)"
                break

        gross = (exit_p - entry_p) * lot_size
        turn = (entry_p + exit_p) * lot_size
        brok = 40.0
        stt = (exit_p * lot_size) * config.STT_PCT_ON_SELL
        exch = turn * config.EXCHANGE_TURNOVER_PCT
        gst = (brok + exch) * config.GST_PCT
        charges = round(brok + stt + exch + gst, 2)
        net = round(gross - charges, 2)

        if net > 0:
            wins_count += 1
        else:
            losses_count += 1

        trade_info = {
            "trade_id": len(daily_trades) + 1,
            "strategy": "Strategy 4: Nifty 1M Pullback Scalper",
            "symbol": sym,
            "entry_time": entry_ts.strftime("%H:%M:%S") if hasattr(entry_ts, "strftime") else str(entry_ts),
            "exit_time": exit_ts.strftime("%H:%M:%S") if hasattr(exit_ts, "strftime") else str(exit_ts),
            "strike": strike,
            "opt_type": opt_type,
            "entry_p": entry_p,
            "exit_p": exit_p,
            "cost": cost,
            "capital_used": cost,
            "gross_pnl": round(gross, 2),
            "charges": charges,
            "net_pnl": net,
            "exit_reason": exit_reason,
            "data_feed": feed_label
        }
        daily_trades.append(trade_info)

        # Apply 10-minute cooldown
        next_eligible_idx = idx + bars_held + config.S4_COOLDOWN_MINS

    if not daily_trades:
        return {
            "status": "NO_SETUP",
            "trade_occurred": False,
            "strategy": "Strategy 4: Nifty 1M Pullback Scalper",
            "reason": f"No pullback bounce triggered today under {bias_str} morning bias."
        }

    tot_net = round(sum(t["net_pnl"] for t in daily_trades), 2)
    tot_gross = round(sum(t["gross_pnl"] for t in daily_trades), 2)
    tot_charges = round(sum(t["charges"] for t in daily_trades), 2)

    return {
        "status": "TRADE_EXECUTED",
        "trade_occurred": True,
        "strategy": "Strategy 4: Nifty 1M Pullback Scalper",
        "num_trades_day": len(daily_trades),
        "entry_time": daily_trades[0]["entry_time"],
        "exit_time": daily_trades[-1]["exit_time"],
        "strike": daily_trades[0]["strike"],
        "opt_type": daily_trades[0]["opt_type"],
        "entry_p": daily_trades[0]["entry_p"],
        "exit_p": daily_trades[-1]["exit_p"],
        "cost": daily_trades[0]["cost"],
        "gross_pnl": tot_gross,
        "charges": tot_charges,
        "net_pnl": tot_net,
        "exit_reason": " | ".join(t["exit_reason"] for t in daily_trades),
        "data_feed": daily_trades[0]["data_feed"],
        "all_trades": daily_trades
    }

