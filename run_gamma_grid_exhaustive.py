"""
Exhaustive Parametric Sweep & Backtest Engine for Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze
Tested on 10,483 Real 1-Minute Nifty Candles from Angel One SmartAPI.
Constraints:
- Capital: INR 10,000 (1 Lot = 75 Qty)
- Frequency: 1 to 2 Quality Trades per Week (Wednesday / Thursday Only)
- Max 1 Trade per Day
- Realistic Indian Taxes: Brokerage ₹40 (2 orders) + STT (0.1% on sell) + Exch + GST + 0.8% Slippage
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import time
from tabulate import tabulate
from src.features.greeks import black_scholes_price
from run_strangle_grid_exhaustive import compute_calendar_dte
from src.gamma_squeeze.gamma_signals import GammaSignalGenerators
from config import config

def select_gamma_strike(spot: float, opt_type: str, dte: float, target_premium: float = 30.0) -> tuple:
    """
    Finds the strike closest to the target premium (e.g. ₹25 - ₹40) for 0-DTE / 1-DTE.
    Returns: (strike, estimated_price)
    """
    atm = int(round(spot / 50.0) * 50)
    T = max(dte, 0.005) / 365.0
    best_strike = atm
    best_diff = 999.0
    best_price = 30.0

    # Search strikes within 300 points
    step = 50
    candidates = range(atm, atm + 350, step) if opt_type == "CE" else range(atm, atm - 350, -step)

    for k in candidates:
        p = black_scholes_price(spot, k, T, 0.07, 0.14, opt_type)
        diff = abs(p - target_premium)
        if diff < best_diff:
            best_diff = diff
            best_strike = k
            best_price = p

    return best_strike, best_price

def run_single_gamma_simulation(
    data: pd.DataFrame,
    target_premium: float = 30.0,
    sl_pct: float = 0.35,
    target_pct: float = 1.50, # +150% (2.5x)
    trail_pct: float = 0.50,   # Trail to breakeven once up +50%
    time_stop_mins: int = 45   # Exit if trade stagnates for 45 mins
) -> dict:
    n = len(data)
    lot_size = config.LOT_SIZE
    capital = 10000.0
    equity = capital
    trades = []

    current_day = None
    daily_trades = 0
    in_pos = False
    opt_type = ""
    strike = 0
    entry_price = 0.0
    peak_price = 0.0
    entry_time = None
    pos_bars = 0
    eod_time = time(15, 15)

    for i in range(n):
        bar = data.iloc[i]
        bar_time = data.index[i]
        bar_date = bar['date']

        if bar_date != current_day:
            current_day = bar_date
            daily_trades = 0

        curr_dte = compute_calendar_dte(bar_time)
        T = max(curr_dte, 0.005) / 365.0

        if in_pos:
            pos_bars += 1
            spot = bar['close']
            curr_p = black_scholes_price(spot, strike, T, 0.07, 0.14, opt_type)
            peak_price = max(peak_price, curr_p)

            exit_triggered = False
            reason = ""

            # 1. Stop Loss Hit
            if curr_p <= entry_price * (1.0 - sl_pct):
                exit_triggered = True
                reason = f"SL Hit (-{int(sl_pct*100)}%)"

            # 2. Profit Target Hit (Gamma Explosion)
            elif curr_p >= entry_price * (1.0 + target_pct):
                exit_triggered = True
                reason = f"Target Hit (+{int(target_pct*100)}%)"

            # 3. Trailing Stop (Once up +50%, exit if drops 15% from peak)
            elif trail_pct > 0 and peak_price >= entry_price * (1.0 + trail_pct) and curr_p <= peak_price * 0.85:
                exit_triggered = True
                reason = "Trailing Exit"

            # 4. Time-based Decay Stop
            elif time_stop_mins > 0 and pos_bars >= time_stop_mins and curr_p <= entry_price:
                exit_triggered = True
                reason = f"Time Stop ({time_stop_mins}m)"

            # 5. EOD Exit
            elif bar_time.time() >= eod_time:
                exit_triggered = True
                reason = "EOD (15:15 PM)"

            if exit_triggered:
                exit_price = max(0.5, curr_p * (1.0 - config.SLIPPAGE_PCT))
                gross = (exit_price - entry_price) * lot_size

                # Brokerage (2 orders) + taxes
                buy_val = entry_price * lot_size
                sell_val = exit_price * lot_size
                turn = buy_val + sell_val
                brok = 40.0
                stt = sell_val * 0.001
                exch = turn * 0.0005
                gst = (brok + exch) * 0.18
                charges = round(brok + stt + exch + gst, 2)
                net = round(gross - charges, 2)

                trades.append({
                    "entry_time": entry_time, "exit_time": bar_time,
                    "opt_type": opt_type, "strike": strike,
                    "entry_p": entry_price, "exit_p": exit_price,
                    "cost": round(buy_val, 2), "gross": gross, "charges": charges, "net": net,
                    "exit_reason": reason, "bars": pos_bars
                })
                equity += net
                in_pos = False

        # Entry Check (Max 1 trade/day on Wed/Thu)
        if not in_pos and daily_trades < 1:
            sig = bar.get('signal', 0)
            if sig in [1, -1] and bar_time.time() < eod_time:
                opt_type = "CE" if sig == 1 else "PE"
                spot = bar['close']
                best_k, raw_p = select_gamma_strike(spot, opt_type, curr_dte, target_premium)
                entry_price = round(raw_p * (1.0 + config.SLIPPAGE_PCT), 2)
                cost = entry_price * lot_size

                if cost <= capital: # Within 10,000 budget
                    strike = best_k
                    peak_price = entry_price
                    entry_time = bar_time
                    pos_bars = 0
                    in_pos = True
                    daily_trades += 1

    if not trades:
        return {"trades": 0, "net": 0.0, "wr": 0.0, "pf": 0.0, "max_dd": 0.0}

    df_t = pd.DataFrame(trades)
    df_t['cum_net'] = df_t['net'].cumsum()
    peak_eq = df_t['cum_net'].cummax()
    dd = peak_eq - df_t['cum_net']
    max_dd = round(float(dd.max()), 2)

    wins = df_t[df_t['net'] > 0]
    losses = df_t[df_t['net'] <= 0]
    gain = wins['net'].sum() if not wins.empty else 0.0
    loss = abs(losses['net'].sum()) if not losses.empty else 0.0
    pf = round(gain / loss, 2) if loss > 0 else (99.0 if gain > 0 else 0.0)
    wr = round(len(wins) / len(df_t) * 100.0, 2)
    net = round(df_t['net'].sum(), 2)

    return {
        "trades": len(df_t),
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "pf": pf,
        "net": net,
        "max_dd": max_dd,
        "avg_win": round(wins['net'].mean(), 2) if not wins.empty else 0.0,
        "avg_loss": round(losses['net'].mean(), 2) if not losses.empty else 0.0,
        "trades_df": df_t
    }

def run_gamma_sweep():
    df = pd.read_csv("data/nifty_1min_real.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    print("=" * 105)
    print("      EXHAUSTIVE STRATEGY 2 (EXPIRY GAMMA SQUEEZE) SWEEP ON 10,483 REAL CANDLES")
    print("      Days: Monday (1-DTE) & Tuesday (0-DTE Expiry) | Capital: INR 10,000 | 1 Lot (75 Qty)")
    print("=" * 105)

    # Pre-generate signals across windows and compressions
    windows = ["afternoon", "morning", "both"]
    compressions = [20, 30, 45]
    precomputed_signals = {}

    for w in windows:
        for c in compressions:
            key = f"{w}_{c}m"
            precomputed_signals[key] = GammaSignalGenerators.expiry_consolidation_breakout(
                df, compression_mins=c, buffer_pts=5.0, window=w, include_pre_expiry_monday=True
            )

    all_results = []
    premiums = [22.0, 32.0, 45.0]
    sl_options = [0.25, 0.35, 0.45]
    tgt_options = [0.80, 1.20, 1.80, 2.50]
    time_stops = [30, 45]

    total_runs = len(precomputed_signals) * len(premiums) * len(sl_options) * len(tgt_options) * len(time_stops)
    print(f"Sweeping through {total_runs} distinct parameter combinations across windows, strikes, and SL/TP...\n")

    for sig_key, sig_data in precomputed_signals.items():
        w_name, c_name = sig_key.split("_")
        for p in premiums:
            for sl in sl_options:
                for tgt in tgt_options:
                    for ts in time_stops:
                        res = run_single_gamma_simulation(
                            data=sig_data,
                            target_premium=p,
                            sl_pct=sl,
                            target_pct=tgt,
                            trail_pct=0.40,
                            time_stop_mins=ts
                        )

                        if res["trades"] > 0:
                            all_results.append({
                                "Window": w_name,
                                "Compression": c_name,
                                "Target Premium": f"~INR {int(p)}",
                                "SL": f"-{int(sl*100)}%",
                                "Target": f"+{int(tgt*100)}%",
                                "Time Stop": f"{ts}m",
                                "Trades": res["trades"],
                                "Wins": res["wins"],
                                "Losses": res["losses"],
                                "Win Rate (%)": f"{res['wr']}%",
                                "Profit Factor": res["pf"],
                                "Net PnL (INR)": res["net"],
                                "Max DD (INR)": res["max_dd"],
                                "Avg Win": res["avg_win"],
                                "Avg Loss": res["avg_loss"]
                            })

    df_res = pd.DataFrame(all_results)
    df_res.sort_values(by=["Net PnL (INR)", "Profit Factor"], ascending=[False, False], inplace=True)

    print("\n--- TOP 15 BEST PERFORMING GAMMA SQUEEZE COMBINATIONS ---")
    print(tabulate(df_res.head(15), headers="keys", tablefmt="grid", showindex=False))

    profitable = len(df_res[df_res["Net PnL (INR)"] > 0])
    print(f"\nTotal Configurations Tested: {len(df_res)}")
    print(f"Profitable Configurations: {profitable} ({profitable/len(df_res)*100:.1f}%)")
    print(f"Max Net PnL Achieved: INR {df_res['Net PnL (INR)'].max()}")

    df_res.to_csv("data/gamma_sweep_results.csv", index=False)
    print("\nFull parameter sweep saved to: data/gamma_sweep_results.csv")

if __name__ == "__main__":
    run_gamma_sweep()
