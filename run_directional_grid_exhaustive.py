"""
Exhaustive Parametric Sweep & Backtest Engine for Strategy 3: 30-Minute Statistical Box Breakout.
Directional ATM Option Buying on 10,483 Real 1-Minute Nifty Candles from Angel One SmartAPI.
Constraints:
- Capital: Strictly INR 10,000 (1 Lot = 75 Qty)
- Max 1 Trade per Day
- Tuesday Weekly Expiry DTE Calculation (NSE Rule)
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
from src.directional_box.directional_signals import DirectionalBoxSignals
from config import config

def run_single_directional_simulation(
    data: pd.DataFrame,
    strike_type: str = "ATM", # "ATM", "ITM_50", "OTM_50"
    sl_pct: float = 0.20,
    target_pct: float = 0.60,
    trail_trigger_pct: float = 0.25, # Move SL to breakeven once up 25%
    trailing_peak_pct: float = 0.15  # Trail 15% from peak once in profit
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

            # 2. Profit Target Hit
            elif curr_p >= entry_price * (1.0 + target_pct):
                exit_triggered = True
                reason = f"Target Hit (+{int(target_pct*100)}%)"

            # 3. Trailing Stop / Breakeven
            elif trail_trigger_pct > 0 and peak_price >= entry_price * (1.0 + trail_trigger_pct) and curr_p <= peak_price * (1.0 - trailing_peak_pct):
                exit_triggered = True
                reason = "Trailing Peak Exit"

            # 4. EOD Exit (15:15 PM)
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

        # Entry Check (Max 1 trade/day)
        if not in_pos and daily_trades < 1:
            sig = bar.get('signal', 0)
            if sig in [1, -1] and bar_time.time() < eod_time:
                opt_type = "CE" if sig == 1 else "PE"
                spot = bar['close']
                atm = int(round(spot / 50.0) * 50)

                if strike_type == "ITM_50":
                    preferred_offset = -50 if opt_type == "CE" else 50
                elif strike_type == "ATM":
                    preferred_offset = 0
                else: # "OTM_50"
                    preferred_offset = 50 if opt_type == "CE" else -50

                # Candidate strikes starting from preferred towards cheaper OTM strikes
                if opt_type == "CE":
                    offsets = [preferred_offset] + [off for off in [0, 50, 100, 150, 200, 250, 300, 350, 400] if off > preferred_offset]
                else:
                    offsets = [preferred_offset] + [off for off in [0, -50, -100, -150, -200, -250, -300, -350, -400] if off < preferred_offset]

                found = False
                for off in offsets:
                    cand_strike = atm + off
                    raw_p = black_scholes_price(spot, cand_strike, T, 0.07, 0.14, opt_type)
                    cand_entry = round(raw_p * (1.0 + config.SLIPPAGE_PCT), 2)
                    cand_cost = cand_entry * lot_size
                    if cand_cost <= capital:
                        strike = cand_strike
                        entry_price = cand_entry
                        cost = cand_cost
                        found = True
                        break

                if not found:
                    continue

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

def run_directional_sweep():
    df = pd.read_csv("data/nifty_1min_real.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    print("=" * 105)
    print("      EXHAUSTIVE STRATEGY 3 (30M STATISTICAL BOX BREAKOUT) SWEEP ON 10,483 REAL CANDLES")
    print("      Capital: INR 10,000 | 1 Lot (75 Qty) | Max 1 Trade/Day | Tuesday Expiry DTE")
    print("=" * 105)

    # Pre-generate signals across box compression and buffers
    box_pcts = [0.28, 0.32, 0.36]
    buffers = [6.0, 8.0, 10.0]
    precomputed_signals = {}

    for bp in box_pcts:
        for buf in buffers:
            key = f"bp{int(bp*100)}_buf{int(buf)}"
            precomputed_signals[key] = DirectionalBoxSignals.generate_30m_box_signals(
                df, max_box_pct=bp, buffer_pts=buf, trade_cutoff_str="13:30"
            )

    all_results = []
    strike_types = ["ATM", "ITM_50", "OTM_50"]
    sl_options = [0.15, 0.20, 0.25, 0.30]
    tgt_options = [0.40, 0.60, 0.80, 1.00]
    trail_triggers = [0.25, 0.30]

    total_runs = len(precomputed_signals) * len(strike_types) * len(sl_options) * len(tgt_options) * len(trail_triggers)
    print(f"Sweeping through {total_runs} distinct parameter combinations across strikes, SL, TP, and trailing...\n")

    for sig_key, sig_data in precomputed_signals.items():
        parts = sig_key.split("_")
        bp_val = parts[0].replace("bp", "") + "%"
        buf_val = parts[1].replace("buf", "") + " pts"

        for st in strike_types:
            for sl in sl_options:
                for tgt in tgt_options:
                    for tr in trail_triggers:
                        res = run_single_directional_simulation(
                            data=sig_data,
                            strike_type=st,
                            sl_pct=sl,
                            target_pct=tgt,
                            trail_trigger_pct=tr,
                            trailing_peak_pct=0.15
                        )

                        if res["trades"] > 0:
                            all_results.append({
                                "Box Pct": bp_val,
                                "Buffer": buf_val,
                                "Strike": st,
                                "SL": f"-{int(sl*100)}%",
                                "Target": f"+{int(tgt*100)}%",
                                "Trail Trigger": f"+{int(tr*100)}%",
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

    print("\n--- TOP 15 BEST PERFORMING DIRECTIONAL BOX COMBINATIONS ---")
    print(tabulate(df_res.head(15), headers="keys", tablefmt="grid", showindex=False))

    profitable = len(df_res[df_res["Net PnL (INR)"] > 0])
    print(f"\nTotal Configurations Tested: {len(df_res)}")
    print(f"Profitable Configurations: {profitable} ({profitable/len(df_res)*100:.1f}%)")
    print(f"Max Net PnL Achieved: INR {df_res['Net PnL (INR)'].max()}")

    df_res.to_csv("data/directional_sweep_results.csv", index=False)
    print("\nFull parameter sweep saved to: data/directional_sweep_results.csv")

if __name__ == "__main__":
    run_directional_sweep()
