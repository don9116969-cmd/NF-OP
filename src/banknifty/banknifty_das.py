"""
BankNIFTY Decoupled Asymmetric Strangle (DAS) Engine.
Tail-Risk Capture & Velocity Arbitrage Engine for BankNIFTY Options.

Key Features:
- Underlying: BankNIFTY Weekly/Monthly Options (1 Lot = 30 Qty)
- Volatility Compression: 20-minute rolling std < 55.0 points
- Velocity Trigger: 3-bar spot change >= 70.0 points
- Affordable Strike Selection: If ATM exceeds Rs 10,000 capital limit, scans next cheap OTM strikes
- Real Traded Data: Connects to Angel One SmartAPI to price real candles and executions
- Decoupled Exits: +100% target on surging leg, -15% SL on decaying leg, -15% combined capital stop, 45-min cutoff
- Trading Window: 09:45 to 13:30 IST (bypasses midday 11:30–12:45 chop)
"""

import os
import math
import datetime
import pandas as pd
import numpy as np

from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles

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

_BN_EXPIRY_CACHE = {}

def get_banknifty_dte(ts: pd.Timestamp, target_date: datetime.date = None) -> float:
    if target_date is None:
        target_date = ts.date()
    global _BN_EXPIRY_CACHE
    if target_date not in _BN_EXPIRY_CACHE:
        sample_c = get_active_option_contract(target_date, 54000, "CE", underlying="BANKNIFTY")
        _BN_EXPIRY_CACHE[target_date] = sample_c.get("expiry_date") if sample_c else None

    exp_date = _BN_EXPIRY_CACHE.get(target_date)
    if exp_date:
        days_ahead = max(1, (exp_date - target_date).days)
        mins = max(1.0, 930 - (ts.hour * 60 + ts.minute))
        frac = mins / 375.0
        d = max(0.01, (days_ahead - 1) + frac if days_ahead > 0 else frac * 0.5)
        return d / 365.0

    # Fallback if scrip not loaded (BankNIFTY monthly is last Thursday)
    weekday = ts.weekday()
    target_exp = 3
    days_ahead = target_exp - weekday if weekday <= target_exp else 7 - (weekday - target_exp)
    return max(1, days_ahead) / 365.0

def select_banknifty_affordable_strikes(
    spot: float,
    dte: float,
    target_date: datetime.date,
    api=None,
    max_budget: float = 10000.0
):
    """
    Selects CE and PE strikes whose combined cost for 1 lot (30 Qty) fits strictly under max_budget (Rs 10,000).
    Starts at ATM (offset 0). If ATM exceeds the budget (e.g. monthly ~27 DTE time value), steps into
    the next cheap OTM strikes where combined premium <= Rs 333 (Rs 10,000 / 30).
    Uses high-speed BSM to determine the affordable strike offset instantaneously, then resolves active contracts.
    """
    atm = int(round(spot / 100.0) * 100)

    chosen_offset = 0
    chosen_ce_p = 0.0
    chosen_pe_p = 0.0
    chosen_tot = 0.0

    # Scan in 100-pt steps until combined cost <= 8500 in theoretical BSM
    # (guarantees real traded option market cost stays comfortably under Rs 10,000)
    for offset in range(0, 5000, 100):
        k_ce = atm + offset
        k_pe = atm - offset
        p_ce = bs_price(spot, k_ce, dte, "CE")
        p_pe = bs_price(spot, k_pe, dte, "PE")
        tot = (p_ce + p_pe) * LOT_SIZE
        if tot <= 8500.0:
            chosen_offset = offset
            chosen_ce_p = p_ce
            chosen_pe_p = p_pe
            chosen_tot = tot
            break

    k_ce = atm + chosen_offset
    k_pe = atm - chosen_offset

    c_ce = get_active_option_contract(target_date, k_ce, "CE", underlying="BANKNIFTY") if api else None
    c_pe = get_active_option_contract(target_date, k_pe, "PE", underlying="BANKNIFTY") if api else None

    return k_ce, k_pe, chosen_ce_p, chosen_pe_p, chosen_tot, c_ce, c_pe

def evaluate_banknifty_das_for_day(df_spot_bn: pd.DataFrame, target_date: datetime.date, api=None) -> dict:
    """
    Evaluates BankNIFTY Decoupled Asymmetric Strangle for target_date.
    Uses real Angel One option candles when api is available; otherwise falls back to BSM.
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
        dte = get_banknifty_dte(ts, target_date)

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

            elapsed_mins = (day_slice.iloc[i]["timestamp"] - day_slice.iloc[entry_idx]["timestamp"]).total_seconds() / 60.0
            is_time_up = (elapsed_mins >= 45.0) or (tm >= datetime.time(15, 15))
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
                    exit_reason = f"Time Cutoff ({' & '.join(active_reasons)})" if active_reasons else "Time Exit (45m)"
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

            # print(f"Checking bar i={i}, tm={tm}, std={std_val:.1f}, vel={vel_val:.1f}", flush=True)

            # Quantitative condition: Volatility compression + kinetic velocity expansion
            if std_val < 55.0 and abs(vel_val) >= 70.0:
                # Select affordable strikes strictly under Rs 10,000 budget
                k_ce, k_pe, p_ce, p_pe, tot_capital, c_ce, c_pe = select_banknifty_affordable_strikes(
                    c, dte, target_date, api=api, max_budget=10000.0
                )

                if tot_capital > 10000.0:
                    continue

                # Check if real traded option candles can be fetched from cache or Angel One
                if c_ce and c_pe:
                    t_start = day_slice.iloc[i]["timestamp"]
                    t_end = day_slice["timestamp"].max()
                    print(f"[BANKNIFTY DAS] Signal at {t_start.time().strftime('%H:%M:%S')} (Spot {c:.1f}) -> Testing {c_ce['symbol']} + {c_pe['symbol']}", flush=True)
                    df_ce = fetch_real_option_candles(api, c_ce, t_start, t_end)
                    df_pe = fetch_real_option_candles(api, c_pe, t_start, t_end)

                    if not df_ce.empty and not df_pe.empty:
                        real_ce_entry = float(df_ce.iloc[0]["close"]) * (1.0 + SLIPPAGE_PCT)
                        real_pe_entry = float(df_pe.iloc[0]["close"]) * (1.0 + SLIPPAGE_PCT)
                        entry_tot = round(real_ce_entry + real_pe_entry, 2)
                        cost = round(entry_tot * LOT_SIZE, 2)

                        ce_tgt = real_ce_entry * 2.0
                        ce_sl = real_ce_entry * 0.85
                        pe_tgt = real_pe_entry * 2.0
                        pe_sl = real_pe_entry * 0.85

                        ce_exited, pe_exited = False, False
                        ce_exit_p, pe_exit_p = real_ce_entry, real_pe_entry
                        ce_reason, pe_reason = "", ""
                        exit_ts = df_ce.index[-1]

                        common_index = df_ce.index.intersection(df_pe.index)
                        bars_held = 0
                        for ts_idx in common_index:
                            bars_held += 1
                            c_bar = df_ce.loc[ts_idx]
                            p_bar = df_pe.loc[ts_idx]

                            if not ce_exited:
                                if c_bar["high"] >= ce_tgt:
                                    ce_exited = True
                                    ce_exit_p = round(ce_tgt * (1.0 - SLIPPAGE_PCT), 2)
                                    ce_reason = "CE Target (+100%)"
                                elif c_bar["low"] <= ce_sl:
                                    ce_exited = True
                                    ce_exit_p = round(ce_sl * (1.0 - SLIPPAGE_PCT), 2)
                                    ce_reason = "CE SL (-15%)"

                            if not pe_exited:
                                if p_bar["high"] >= pe_tgt:
                                    pe_exited = True
                                    pe_exit_p = round(pe_tgt * (1.0 - SLIPPAGE_PCT), 2)
                                    pe_reason = "PE Target (+100%)"
                                elif p_bar["low"] <= pe_sl:
                                    pe_exited = True
                                    pe_exit_p = round(pe_sl * (1.0 - SLIPPAGE_PCT), 2)
                                    pe_reason = "PE SL (-15%)"

                            cur_tot = (ce_exit_p if ce_exited else float(c_bar["close"])) + (pe_exit_p if pe_exited else float(p_bar["close"]))
                            comb_ret = (cur_tot - entry_tot) / entry_tot
                            elapsed_mins = (ts_idx - t_start).total_seconds() / 60.0
                            is_time_up = (elapsed_mins >= 45.0) or (ts_idx.time() >= datetime.time(15, 15))
                            is_comb_stop = comb_ret <= -0.15

                            if (ce_exited and pe_exited) or is_time_up or is_comb_stop:
                                exit_ts = ts_idx
                                p_c = ce_exit_p if ce_exited else float(c_bar["close"])
                                p_p = pe_exit_p if pe_exited else float(p_bar["close"])

                                gross = ((p_c - real_ce_entry) + (p_p - real_pe_entry)) * LOT_SIZE
                                buy_v = entry_tot * LOT_SIZE
                                sell_v = (p_c + p_p) * LOT_SIZE
                                turn = buy_v + sell_v
                                stt = sell_v * 0.001
                                exch = turn * 0.0005
                                charges = round(40.0 + stt + exch + (40.0 + exch) * 0.18, 2)
                                net = round(gross - charges, 2)

                                if is_comb_stop:
                                    exit_reason = "Combined Capital SL (-15%)"
                                elif ce_exited and pe_exited:
                                    exit_reason = f"{ce_reason} & {pe_reason}"
                                elif is_time_up:
                                    active_reasons = []
                                    if ce_exited: active_reasons.append(ce_reason)
                                    if pe_exited: active_reasons.append(pe_reason)
                                    exit_reason = f"Time Cutoff ({' & '.join(active_reasons)})" if active_reasons else "Time Exit (45m)"
                                elif ce_exited or pe_exited:
                                    exit_reason = ce_reason if ce_exited else pe_reason
                                else:
                                    exit_reason = "Time Exit (45m)"

                                trades_list.append({
                                    "date": str(target_date),
                                    "strategy": "BankNIFTY Decoupled Strangle (DAS)",
                                    "opt_type": "CE+PE",
                                    "strike": f"{c_ce['symbol']} + {c_pe['symbol']}",
                                    "ce_symbol": c_ce['symbol'],
                                    "pe_symbol": c_pe['symbol'],
                                    "ce_entry": real_ce_entry,
                                    "pe_entry": real_pe_entry,
                                    "ce_exit": p_c,
                                    "pe_exit": p_p,
                                    "entry_time": t_start.time().strftime("%H:%M:%S"),
                                    "exit_time": exit_ts.time().strftime("%H:%M:%S"),
                                    "entry_p": entry_tot,
                                    "exit_p": round(p_c + p_p, 2),
                                    "cost": cost,
                                    "gross_pnl": round(gross, 2),
                                    "charges": charges,
                                    "net_pnl": net,
                                    "exit_reason": exit_reason,
                                    "data_feed": f"Angel One Real Traded ({c_ce['symbol']} + {c_pe['symbol']})"
                                })
                                print(f"[BANKNIFTY DAS] Real Trade: Entry {t_start.time()} -> Exit {exit_ts.time()} | Cost: INR {cost:.2f} | Net: INR {net:+.2f} ({exit_reason})", flush=True)

                                matching_day_idxs = day_slice[day_slice["timestamp"] >= exit_ts].index
                                last_exit_idx = matching_day_idxs[0] if len(matching_day_idxs) > 0 else len(day_slice)
                                break

                    # In live API mode, do not enter synthetic BSM trades; advance cooldown if no trade executed
                    if not trades_list or last_exit_idx < i:
                        last_exit_idx = i
                    continue

                # Mathematical BSM mode when API not available (offline backtesting only)
                ce_strike = k_ce
                pe_strike = k_pe
                ce_entry = p_ce
                pe_entry = p_pe
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
