"""
Exhaustive Parametric Sweep & Backtest Engine for Strategy 1: Hedged Long Strangle
Designed for:
- Nifty 50 Options (1 Lot = 75 Qty)
- Strict ₹10,000 Capital Limit
- 2 to 4 Quality Trades per Week (Max 1 trade/day)
- 9,031 Real 1-Minute Candles from Angel One SmartAPI (Aug 10 - Sep 11, 2026)
- Dynamic Calendar DTE calculation (Mon=3, Tue=2, Wed=1, Thu=0, Fri=6)
- Realistic Indian Charges (Brokerage ₹80/roundtrip + STT + Exch + GST + Slippage)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import time
from tabulate import tabulate
from src.features.greeks import black_scholes_price
from config import config

def compute_calendar_dte(ts: pd.Timestamp) -> float:
    """
    Computes exact fractional DTE to weekly Tuesday 15:30 expiry (Official NSE Rule).
    Tuesday (1): 0-DTE (Expiry Day, fractional 0.5 to 0.01)
    Monday (0): 1-DTE (~0.88 to 1.0 day)
    Friday (4): 3-DTE (~3.88 days)
    Thursday (3): 4-DTE (~4.88 days)
    Wednesday (2): 5-DTE (~5.88 days)
    """
    weekday = ts.weekday()
    target_tuesday = 1
    if weekday <= target_tuesday:
        days_ahead = target_tuesday - weekday
    else:
        days_ahead = 7 - (weekday - target_tuesday)
        
    mins_to_expiry = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
    fractional_day = mins_to_expiry / 375.0
    
    if days_ahead == 0:
        return max(0.01, fractional_day * 0.5)
    else:
        return (days_ahead - 1) + fractional_day

def generate_signals(df: pd.DataFrame, setup_type: str = "30m_box", max_box_pct: float = 0.30, buffer_pts: float = 6.0) -> pd.DataFrame:
    data = df.copy()
    if 'date' not in data.columns:
        data['date'] = data.index.date

    signals = [0] * len(data)
    eod_cutoff = time(13, 30)

    if setup_type in ["15m_box", "30m_box", "45m_box"]:
        box_mins = 15 if setup_type == "15m_box" else (30 if setup_type == "30m_box" else 45)
        start_t = time(9, 15)
        end_minute = 15 + box_mins
        end_h = 9 + (end_minute // 60)
        end_m = end_minute % 60
        end_t = time(end_h, end_m)

        for date_val, group in data.groupby('date'):
            box_bars = group[(group.index.time >= start_t) & (group.index.time <= end_t)]
            if len(box_bars) < (box_mins // 2):
                continue
            box_h = box_bars['high'].max()
            box_l = box_bars['low'].min()
            box_range = box_h - box_l
            spot_ref = box_bars['close'].iloc[0]
            box_pct = (box_range / spot_ref) * 100.0

            if box_pct <= max_box_pct:
                rem = group[(group.index.time > end_t) & (group.index.time <= eod_cutoff)]
                for idx in rem.index:
                    row = data.loc[idx]
                    pos = data.index.get_loc(idx)
                    if row['high'] >= (box_h + buffer_pts) or row['low'] <= (box_l - buffer_pts):
                        signals[pos] = 1
                        break

    elif setup_type == "nr7":
        daily = data.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
        daily['range'] = daily['high'] - daily['low']
        daily['is_nr7'] = daily['range'] == daily['range'].rolling(7).min()
        daily['trigger_today'] = daily['is_nr7'].shift(1).fillna(False)
        nr7_dates = set(daily[daily['trigger_today']]['date'].values)
        trigger_t = time(9, 30)
        for i in range(len(data)):
            if data['date'].iloc[i] in nr7_dates and data.index[i].time() == trigger_t:
                signals[i] = 1

    data['signal'] = signals
    return data

def run_single_simulation_with_data(
    data: pd.DataFrame,
    strike_mode: str = "dynamic_affordable",
    fixed_otm: int = 100,
    manage_mode: str = "uncoupled", # "coupled" or "uncoupled"
    leg_sl_pct: float = 0.20,
    target_pct: float = 0.50,
    trailing_pct: float = 0.25
) -> dict:
    n = len(data)
    lot_size = config.LOT_SIZE
    initial_cap = 10000.0
    equity = initial_cap
    trades = []
    
    current_day = None
    daily_trades = 0
    in_pos = False
    ce_active = False
    pe_active = False
    ce_strike = 0
    pe_strike = 0
    ce_entry = 0.0
    pe_entry = 0.0
    ce_peak = 0.0
    pe_peak = 0.0
    comb_entry = 0.0
    comb_peak = 0.0
    ce_exit_val = None
    pe_exit_val = None
    ce_reason = ""
    pe_reason = ""
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
            curr_ce = black_scholes_price(spot, ce_strike, T, config.RISK_FREE_RATE, 0.14, "CE")
            curr_pe = black_scholes_price(spot, pe_strike, T, config.RISK_FREE_RATE, 0.14, "PE")

            if manage_mode == "uncoupled":
                # Uncoupled leg management
                if ce_active:
                    ce_peak = max(ce_peak, curr_ce)
                    if curr_ce <= ce_entry * (1 - leg_sl_pct):
                        ce_active = False
                        ce_exit_val = max(0.5, curr_ce * (1 - config.SLIPPAGE_PCT))
                        ce_reason = "CE SL"
                    elif curr_ce >= ce_entry * (1 + target_pct):
                        ce_active = False
                        ce_exit_val = max(0.5, curr_ce * (1 - config.SLIPPAGE_PCT))
                        ce_reason = "CE Target"
                    elif trailing_pct > 0 and ce_peak >= ce_entry * (1 + trailing_pct) and curr_ce <= ce_peak * 0.90:
                        ce_active = False
                        ce_exit_val = max(0.5, curr_ce * (1 - config.SLIPPAGE_PCT))
                        ce_reason = "CE Trail"

                if pe_active:
                    pe_peak = max(pe_peak, curr_pe)
                    if curr_pe <= pe_entry * (1 - leg_sl_pct):
                        pe_active = False
                        pe_exit_val = max(0.5, curr_pe * (1 - config.SLIPPAGE_PCT))
                        pe_reason = "PE SL"
                    elif curr_pe >= pe_entry * (1 + target_pct):
                        pe_active = False
                        pe_exit_val = max(0.5, curr_pe * (1 - config.SLIPPAGE_PCT))
                        pe_reason = "PE Target"
                    elif trailing_pct > 0 and pe_peak >= pe_entry * (1 + trailing_pct) and curr_pe <= pe_peak * 0.90:
                        pe_active = False
                        pe_exit_val = max(0.5, curr_pe * (1 - config.SLIPPAGE_PCT))
                        pe_reason = "PE Trail"

                if bar_time.time() >= eod_time:
                    if ce_active:
                        ce_active = False
                        ce_exit_val = max(0.5, curr_ce * (1 - config.SLIPPAGE_PCT))
                        ce_reason = "EOD"
                    if pe_active:
                        pe_active = False
                        pe_exit_val = max(0.5, curr_pe * (1 - config.SLIPPAGE_PCT))
                        pe_reason = "EOD"

                if (not ce_active) and (not pe_active):
                    ce_real_exit = ce_exit_val if ce_exit_val is not None else ce_entry
                    pe_real_exit = pe_exit_val if pe_exit_val is not None else pe_entry
                    gross_ce = (ce_real_exit - ce_entry) * lot_size
                    gross_pe = (pe_real_exit - pe_entry) * lot_size
                    gross = gross_ce + gross_pe

                    sell_v = (ce_real_exit + pe_real_exit) * lot_size
                    buy_v = (ce_entry + pe_entry) * lot_size
                    turn = buy_v + sell_v
                    brok = 80.0
                    stt = sell_v * 0.001
                    exch = turn * 0.0005
                    gst = (brok + exch) * 0.18
                    charges = round(brok + stt + exch + gst, 2)
                    net = round(gross - charges, 2)

                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "ce_strike": ce_strike, "pe_strike": pe_strike,
                        "strike": f"{ce_strike}CE / {pe_strike}PE",
                        "opt_type": "CE+PE",
                        "entry_p": round(comb_entry, 2),
                        "exit_p": round(ce_real_exit + pe_real_exit, 2),
                        "cost": round(buy_v, 2),
                        "gross": gross, "charges": charges, "net": net,
                        "exit_reason": f"{ce_reason} | {pe_reason}", "bars": pos_bars
                    })
                    equity += net
                    in_pos = False

            else: # "coupled" mode
                curr_comb = curr_ce + curr_pe
                comb_peak = max(comb_peak, curr_comb)
                comb_pnl_pct = (curr_comb - comb_entry) / comb_entry

                exit_triggered = False
                reason = ""
                if comb_pnl_pct <= -leg_sl_pct:
                    exit_triggered = True
                    reason = "Comb SL"
                elif comb_pnl_pct >= target_pct:
                    exit_triggered = True
                    reason = "Comb Target"
                elif trailing_pct > 0 and comb_peak >= comb_entry * (1 + trailing_pct) and curr_comb <= comb_peak * 0.90:
                    exit_triggered = True
                    reason = "Comb Trail"
                elif bar_time.time() >= eod_time:
                    exit_triggered = True
                    reason = "EOD"

                if exit_triggered:
                    ce_real_exit = max(0.5, curr_ce * (1 - config.SLIPPAGE_PCT))
                    pe_real_exit = max(0.5, curr_pe * (1 - config.SLIPPAGE_PCT))
                    gross = (ce_real_exit + pe_real_exit - comb_entry) * lot_size

                    sell_v = (ce_real_exit + pe_real_exit) * lot_size
                    turn = (comb_entry * lot_size) + sell_v
                    brok = 80.0
                    stt = sell_v * 0.001
                    exch = turn * 0.0005
                    gst = (brok + exch) * 0.18
                    charges = round(brok + stt + exch + gst, 2)
                    net = round(gross - charges, 2)

                    trades.append({
                        "entry_time": entry_time, "exit_time": bar_time,
                        "gross": gross, "charges": charges, "net": net,
                        "exit_reason": reason, "bars": pos_bars
                    })
                    equity += net
                    in_pos = False

        # Entry Check (Max 1 trade/day)
        if not in_pos and daily_trades < 1:
            sig = bar.get('signal', 0)
            if sig == 1 and bar_time.time() < eod_time:
                spot = bar['close']
                atm = int(round(spot / config.STRIKE_STEP) * config.STRIKE_STEP)

                if strike_mode == "fixed":
                    # Try preferred fixed_otm first (e.g. 150). If exceeds budget, fallback to next cheaper OTM (200, 250, 300, etc.)
                    found = False
                    otm_candidates = [fixed_otm] + [fixed_otm + step for step in [50, 100, 150, 200, 250, 300, 350]]
                    for otm_cand in otm_candidates:
                        c_strike = atm + otm_cand
                        p_strike = atm - otm_cand
                        raw_c = black_scholes_price(spot, c_strike, T, config.RISK_FREE_RATE, 0.14, "CE")
                        raw_p = black_scholes_price(spot, p_strike, T, config.RISK_FREE_RATE, 0.14, "PE")
                        cost = (raw_c + raw_p) * (1 + config.SLIPPAGE_PCT) * lot_size
                        if cost <= initial_cap:
                            found = True
                            break
                    if not found:
                        continue
                else: # "dynamic_affordable"
                    # Find closest OTM strikes where (CE + PE) * 75 <= 9,800
                    found = False
                    for otm_candidate in [50, 100, 150, 200, 250, 300, 350, 400]:
                        c_strike = atm + otm_candidate
                        p_strike = atm - otm_candidate
                        raw_c = black_scholes_price(spot, c_strike, T, config.RISK_FREE_RATE, 0.14, "CE")
                        raw_p = black_scholes_price(spot, p_strike, T, config.RISK_FREE_RATE, 0.14, "PE")
                        cost = (raw_c + raw_p) * (1 + config.SLIPPAGE_PCT) * lot_size
                        if cost <= initial_cap:
                            found = True
                            break
                    if not found:
                        continue

                ce_strike = c_strike
                pe_strike = p_strike
                ce_entry = round(raw_c * (1 + config.SLIPPAGE_PCT), 2)
                pe_entry = round(raw_p * (1 + config.SLIPPAGE_PCT), 2)
                comb_entry = ce_entry + pe_entry
                ce_peak = ce_entry
                pe_peak = pe_entry
                comb_peak = comb_entry
                ce_active = True
                pe_active = True
                ce_exit_val = None
                pe_exit_val = None
                ce_reason = ""
                pe_reason = ""
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
    charges = round(df_t['charges'].sum(), 2)

    return {
        "trades": len(df_t),
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "pf": pf,
        "net": net,
        "charges": charges,
        "max_dd": max_dd,
        "avg_win": round(wins['net'].mean(), 2) if not wins.empty else 0.0,
        "avg_loss": round(losses['net'].mean(), 2) if not losses.empty else 0.0,
        "trades_df": df_t
    }

def run_exhaustive_sweep():
    df = pd.read_csv("data/nifty_1min_real.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["date"] = df.index.date

    print("=" * 105)
    print("      EXHAUSTIVE STRATEGY 1 (HEDGED STRANGLE) PARAMETER SWEEP ON 9,031 REAL CANDLES")
    print("      Constraints: Capital = INR 10,000 | 1 Lot (75 Qty) | Max 1 Trade/Day")
    print("=" * 105)

    # Pre-generate signals for each setup to avoid redundant work
    precomputed_signals = {
        "15m_box": generate_signals(df, "15m_box", max_box_pct=0.30, buffer_pts=6.0),
        "30m_box": generate_signals(df, "30m_box", max_box_pct=0.30, buffer_pts=6.0),
        "45m_box": generate_signals(df, "45m_box", max_box_pct=0.30, buffer_pts=6.0),
        "nr7": generate_signals(df, "nr7")
    }

    all_results = []
    setups = ["15m_box", "30m_box", "45m_box", "nr7"]
    strike_modes = ["dynamic_affordable", "fixed_100", "fixed_150", "fixed_200"]
    manage_modes = ["uncoupled", "coupled"]
    sl_options = [0.15, 0.20, 0.25, 0.30]
    target_options = [0.30, 0.50, 0.75, 1.00]

    total_runs = len(setups) * len(strike_modes) * len(manage_modes) * len(sl_options) * len(target_options)
    print(f"Sweeping through {total_runs} distinct parameter combinations across timeframes and leg architectures...\n")

    for setup in setups:
        sig_data = precomputed_signals[setup]
        for sm in strike_modes:
            s_mode = "dynamic_affordable" if sm == "dynamic_affordable" else "fixed"
            f_otm = int(sm.split("_")[1]) if s_mode == "fixed" else 100

            for mm in manage_modes:
                for sl in sl_options:
                    for tgt in target_options:
                        res = run_single_simulation_with_data(
                            data=sig_data,
                            strike_mode=s_mode,
                            fixed_otm=f_otm,
                            manage_mode=mm,
                            leg_sl_pct=sl,
                            target_pct=tgt,
                            trailing_pct=0.20
                        )

                        if res["trades"] > 0:
                            all_results.append({
                                "Setup": setup,
                                "Strike": sm,
                                "Mode": mm,
                                "SL": f"-{int(sl*100)}%",
                                "Target": f"+{int(tgt*100)}%",
                                "Trades": res["trades"],
                                "Wins": res["wins"],
                                "Losses": res["losses"],
                                "Win Rate (%)": f"{res['wr']}%",
                                "Profit Factor": res["pf"],
                                "Net PnL (INR)": res["net"],
                                "Max DD (INR)": res["max_dd"],
                                "Total Charges": res["charges"],
                                "Avg Win": res["avg_win"],
                                "Avg Loss": res["avg_loss"]
                            })

    df_results = pd.DataFrame(all_results)
    df_results.sort_values(by=["Net PnL (INR)", "Profit Factor"], ascending=[False, False], inplace=True)

    print("\n--- TOP 15 BEST PERFORMING PARAMETER COMBINATIONS ---")
    print(tabulate(df_results.head(15), headers="keys", tablefmt="grid", showindex=False))

    print("\n--- BOTTOM 5 WORST COMBINATIONS (For Asymmetry Comparison) ---")
    print(tabulate(df_results.tail(5), headers="keys", tablefmt="grid", showindex=False))

    profitable = len(df_results[df_results["Net PnL (INR)"] > 0])
    unprofitable = len(df_results[df_results["Net PnL (INR)"] <= 0])
    print(f"\nTotal Configurations Tested: {len(df_results)}")
    print(f"Profitable Configurations: {profitable} ({profitable/len(df_results)*100:.1f}%)" if len(df_results) > 0 else "")
    print(f"Loss-Making Configurations: {unprofitable} ({unprofitable/len(df_results)*100:.1f}%)" if len(df_results) > 0 else "")
    if len(df_results) > 0:
        print(f"Max Net PnL Achieved: INR {df_results['Net PnL (INR)'].max()}")
        print(f"Min Net PnL: INR {df_results['Net PnL (INR)'].min()}")

    df_results.to_csv("data/strangle_sweep_results.csv", index=False)
    print("\nFull parameter sweep saved to: data/strangle_sweep_results.csv")

if __name__ == "__main__":
    run_exhaustive_sweep()
