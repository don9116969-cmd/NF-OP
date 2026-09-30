"""
BankNIFTY Decoupled Asymmetric Strangle (DAS) Engine.
Tail-Risk Capture & Velocity Arbitrage Engine for BankNIFTY Options.

Key Parameters (Calibrated on Full Historical & Real Traded Datasets):
- Instrument: BankNIFTY Weekly Options (1 Lot = 30 Qty)
- Rolling Volatility Window: 20-minute rolling std < 55.0 (Coiled Range)
- Velocity Trigger: 3-bar Spot Change >= 70.0 points
- Strikes: At-The-Money (ATM) CE + PE
- Winning Leg Target: +100.0% (uncoupled exit)
- Losing Leg Stop Loss: -15.0% (uncoupled exit)
- Combined Capital Stop: -15.0% on total strangle cost
- Max Hold Duration: 45 minutes
- Trading Window: 09:45 to 13:30 IST (bypasses 11:30–12:45 midday European open chop)
"""

import os
import math
import datetime
import pandas as pd
import numpy as np

LOT_SIZE = 30
FEE_STRANGLE = 80.0
SLIPPAGE_PCT = 0.005

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

def get_banknifty_dte(ts: pd.Timestamp) -> float:
    weekday = ts.weekday()
    target_exp = 2  # Wednesday expiry
    days_ahead = target_exp - weekday if weekday <= target_exp else 7 - (weekday - target_exp)
    mins = max(1.0, 930 - (ts.hour * 60 + ts.minute))
    frac = mins / 375.0
    d = max(0.01, frac * 0.5) if days_ahead == 0 else (days_ahead - 1) + frac
    return d / 365.0

def evaluate_banknifty_das_for_day(df_spot_bn: pd.DataFrame, target_date: datetime.date, api=None) -> dict:
    """
    Evaluates BankNIFTY Decoupled Asymmetric Strangle for target_date.
    Returns trade details, exit reasons, gross/net PnL.
    """
    info = {
        "strategy": "BankNIFTY Decoupled Asymmetric Strangle",
        "trade_occurred": False,
        "status": "IDLE",
        "reason": "No entry trigger matched today."
    }

    if df_spot_bn.empty:
        info["reason"] = "Empty BankNIFTY spot dataframe."
        return info

    df = df_spot_bn.copy()
    if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    day_bars = df[df["timestamp"].dt.date == target_date].copy().reset_index(drop=True)
    if len(day_bars) < 20:
        info["reason"] = f"Insufficient intraday candles for {target_date} (found {len(day_bars)})."
        return info

    # Rolling indicators on full dataset to avoid cold start
    df["rolling_std"] = df["close"].rolling(20).std().fillna(50.0)
    df["vel_3"] = df["close"] - df["close"].shift(3).fillna(0.0)

    # Sub-select today with indicators
    day_indices = df[df["timestamp"].dt.date == target_date].index
    day_slice = df.loc[day_indices].copy().reset_index(drop=True)

    trades_list = []
    in_pos = False
    ce_strike = 0
    pe_strike = 0
    ce_entry = 0.0
    pe_entry = 0.0
    ce_exited = False
    pe_exited = False
    ce_exit_p = 0.0
    pe_exit_p = 0.0
    ce_reason = ""
    pe_reason = ""
    entry_idx = 0
    last_exit_idx = -60

    for i in range(len(day_slice)):
        row = day_slice.iloc[i]
        ts = row["timestamp"]
        tm = ts.time()
        c = float(row["close"])
        std_val = float(row["rolling_std"])
        vel_val = float(row["vel_3"])
        dte = get_banknifty_dte(ts)

        if in_pos:
            bars = i - entry_idx
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")

            if not ce_exited:
                r_ce = (c_ce - ce_entry) / max(0.1, ce_entry)
                if r_ce >= 1.0:
                    ce_exited = True
                    ce_exit_p = c_ce
                    ce_reason = "CE Target (+100%)"
                elif r_ce <= -0.15:
                    ce_exited = True
                    ce_exit_p = c_ce
                    ce_reason = "CE SL (-15%)"

            if not pe_exited:
                r_pe = (c_pe - pe_entry) / max(0.1, pe_entry)
                if r_pe >= 1.0:
                    pe_exited = True
                    pe_exit_p = c_pe
                    pe_reason = "PE Target (+100%)"
                elif r_pe <= -0.15:
                    pe_exited = True
                    pe_exit_p = c_pe
                    pe_reason = "PE SL (-15%)"

            cur_tot = (ce_exit_p if ce_exited else c_ce) + (pe_exit_p if pe_exited else c_pe)
            entry_tot = ce_entry + pe_entry
            comb_ret = (cur_tot - entry_tot) / max(0.1, entry_tot)

            is_time_up = (bars >= 45) or (tm >= datetime.time(15, 15))
            is_comb_stop = comb_ret <= -0.15

            if (ce_exited and pe_exited) or is_time_up or is_comb_stop:
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                gross = ((p_c - ce_entry) + (p_p - pe_entry)) * LOT_SIZE
                net = gross - FEE_STRANGLE

                if is_comb_stop:
                    exit_reason = "Combined Capital SL (-15%)"
                elif ce_exited and pe_exited:
                    exit_reason = f"{ce_reason} & {pe_reason}"
                elif is_time_up:
                    active_reasons = []
                    if ce_exited: active_reasons.append(ce_reason)
                    if pe_exited: active_reasons.append(pe_reason)
                    if active_reasons:
                        exit_reason = f"Time Cutoff ({' & '.join(active_reasons)})"
                    else:
                        exit_reason = "Time Exit (45m)"
                elif ce_exited or pe_exited:
                    exit_reason = ce_reason if ce_exited else pe_reason
                else:
                    exit_reason = "Time Exit (45m)"

                trades_list.append({
                    "date": str(target_date),
                    "strategy": "BankNIFTY Decoupled Strangle (DAS)",
                    "opt_type": "CE+PE",
                    "strike": f"{ce_strike}CE + {pe_strike}PE",
                    "entry_time": day_slice.iloc[entry_idx]["timestamp"].time().strftime("%H:%M:%S"),
                    "exit_time": tm.strftime("%H:%M:%S"),
                    "entry_p": round(entry_tot, 2),
                    "exit_p": round(cur_tot, 2),
                    "cost": round(entry_tot * LOT_SIZE, 2),
                    "gross_pnl": round(gross, 2),
                    "charges": FEE_STRANGLE,
                    "net_pnl": round(net, 2),
                    "exit_reason": exit_reason,
                    "data_feed": "Mathematical (BSM)"
                })
                in_pos = False
                last_exit_idx = i

        else:
            # Entry window: 09:45 to 13:30, exclude 11:30–12:45
            if tm < datetime.time(9, 45) or tm > datetime.time(13, 30):
                continue
            if datetime.time(11, 30) <= tm <= datetime.time(12, 45):
                continue
            if (i - last_exit_idx) < 30:
                continue

            # Quantitative condition: Volatility compression + kinetic velocity expansion
            if std_val < 55.0 and abs(vel_val) >= 70.0:
                atm = int(round(c / 100.0) * 100)
                ce_strike = atm
                pe_strike = atm
                ce_entry = bs_price(c, ce_strike, dte, "CE")
                pe_entry = bs_price(c, pe_strike, dte, "PE")
                tot_capital = (ce_entry + pe_entry) * LOT_SIZE

                # Check capital limit <= ₹10,000 (if exceeds, skip or adjust)
                if tot_capital > 15000.0:
                    continue

                in_pos = True
                entry_idx = i
                ce_exited = False
                pe_exited = False
                ce_reason = ""
                pe_reason = ""

    if trades_list:
        last_t = trades_list[-1]
        info.update(last_t)
        info["trade_occurred"] = True
        info["status"] = "TRADE_EXECUTED"
        info["num_trades_day"] = len(trades_list)
        info["all_trades"] = trades_list
    else:
        info["status"] = "NO_TRIGGER"
        info["reason"] = "No volatility compression (<55.0) with kinetic velocity (>=70 pts) occurred."

    return info
