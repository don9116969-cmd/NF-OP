"""
Comprehensive Whole-Year Black-Scholes (BSM) Trade Journal Generator.
Evaluates the Tri-Strategy Core Architecture across the entire 2025 - 2026 historical candle dataset (389 trading days).
Strategies Evaluated:
1. Strategy 1: Nifty 15M Directional Breakout (Trend-Aligned ATM, SL -16%, TP +60%)
2. Strategy 2: BankNIFTY 15M Directional Breakout (Trend-Aligned 100-pt OTM, SL -25%, TP +80%)
3. Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS) (Volatility Compression & Kinetic Expansion)
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIFTY_FILE = os.path.join(BASE_DIR, "data", "nifty_1min_real.csv")
BANKNIFTY_FILE = os.path.join(BASE_DIR, "data", "banknifty_1min_real.csv")
JOURNAL_CSV = os.path.join(BASE_DIR, "data", "trade_journal_whole_year_bsm.csv")
JOURNAL_MD = os.path.join(BASE_DIR, "data", "trade_journal_whole_year_bsm.md")

# Capital & Cost Controls
LOT_NIFTY = 65
LOT_BN = 30
INITIAL_CAPITAL = 10000.0
MAX_CAPITAL_PER_TRADE = 8500.0
SLIPPAGE_PCT = 0.005

def bs_calc(S, K, T, r=0.07, sigma=0.15, opt_type="CE"):
    """Analytical Black-Scholes option pricing."""
    if T <= 0:
        return max(0.05, S - K if opt_type == "CE" else K - S)
    sigma = max(sigma, 1e-4)
    T = max(T, 1e-4)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
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
    return gross, charges, net

def eval_s1(day_df, target_date):
    """Strategy 1: Nifty 15M Directional Breakout."""
    if len(day_df) < 20:
        return None
    box_bars = day_df[(day_df["time"] >= datetime.time(9, 15)) & (day_df["time"] <= datetime.time(9, 30))]
    if len(box_bars) < 10:
        return None

    box_h = float(box_bars["high"].max())
    box_l = float(box_bars["low"].min())
    open_p = float(box_bars["close"].iloc[0])
    b_rng = box_h - box_l
    if (b_rng / open_p) * 100.0 > 0.30:  # Box max pct 0.30%
        return None

    u_trig = box_h + 8.0
    l_trig = box_l - 8.0

    post_bars = day_df[(day_df["time"] > datetime.time(9, 30)) & (day_df["time"] <= datetime.time(15, 0))].copy()
    post_bars["ema_20"] = day_df["close"].ewm(span=20, adjust=False).mean().loc[post_bars.index]

    entry_row = None
    opt_type = None
    for _, row in post_bars.iterrows():
        c = float(row["close"])
        ema = float(row["ema_20"])
        if c >= u_trig and c > ema:
            entry_row = row
            opt_type = "CE"
            break
        elif c <= l_trig and c < ema:
            entry_row = row
            opt_type = "PE"
            break

    if entry_row is None:
        return None

    entry_t = entry_row["timestamp"]
    c_spot = float(entry_row["close"])
    strike = int(round(c_spot / 50.0) * 50)

    # Tuesday expiry DTE
    days_ahead = (1 - target_date.weekday()) % 7
    dte_base = max(0.001, float(days_ahead) / 365.25)

    sub = day_df[day_df["timestamp"] >= entry_t].copy()
    if sub.empty:
        return None

    # Calculate option prices across sub
    sub["opt_p"] = [bs_calc(float(s), strike, dte_base, sigma=0.15, opt_type=opt_type) for s in sub["close"]]
    sub["opt_h"] = [bs_calc(float(s), strike, dte_base, sigma=0.15, opt_type=opt_type) for s in sub["high"]]
    sub["opt_l"] = [bs_calc(float(s), strike, dte_base, sigma=0.15, opt_type=opt_type) for s in sub["low"]]

    entry_p = round(float(sub.iloc[0]["opt_p"]) * (1.0 + SLIPPAGE_PCT), 2)
    cost = round(entry_p * LOT_NIFTY, 2)
    if cost > MAX_CAPITAL_PER_TRADE:
        return None

    tgt_p = round(entry_p * 1.60, 2)
    sl_p = round(entry_p * 0.84, 2)

    exit_p = entry_p
    exit_t = sub.iloc[-1]["timestamp"]
    exit_reason = "EOD (15:15)"

    for _, r in sub.iloc[1:].iterrows():
        h = float(r["opt_h"])
        l = float(r["opt_l"])
        t = r["time"]

        if h >= tgt_p:
            exit_p = round(tgt_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_t = r["timestamp"]
            exit_reason = "Target Hit (+60%)"
            break
        elif l <= sl_p:
            exit_p = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_t = r["timestamp"]
            exit_reason = "SL Hit (-16%)"
            break
        elif t >= datetime.time(15, 15):
            exit_p = round(float(r["opt_p"]) * (1.0 - SLIPPAGE_PCT), 2)
            exit_t = r["timestamp"]
            exit_reason = "EOD (15:15)"
            break

    gross, charges, net = calc_charges(entry_p, exit_p, LOT_NIFTY)
    return {
        "date": str(target_date),
        "strategy": "Strategy 1: Nifty 15M Directional Breakout",
        "leg": opt_type,
        "strike": f"NIFTY_{strike}_{opt_type}",
        "entry_time": entry_t.strftime("%H:%M:%S"),
        "exit_time": exit_t.strftime("%H:%M:%S"),
        "entry_p": entry_p,
        "exit_p": exit_p,
        "capital_used": cost,
        "gross_pnl": round(gross, 2),
        "charges": charges,
        "net_pnl": net,
        "exit_reason": exit_reason,
        "data_feed": "Black-Scholes (BSM)"
    }

def eval_s2(day_df, target_date):
    """Strategy 2: BankNIFTY 15M Directional Breakout."""
    if len(day_df) < 20:
        return None
    box_bars = day_df[(day_df["time"] >= datetime.time(9, 15)) & (day_df["time"] <= datetime.time(9, 30))]
    if len(box_bars) < 10:
        return None

    box_h = float(box_bars["high"].max())
    box_l = float(box_bars["low"].min())
    open_p = float(box_bars["close"].iloc[0])
    b_rng = box_h - box_l
    if (b_rng / open_p) * 100.0 > 1.20:  # Box max pct 1.20%
        return None

    u_trig = box_h + 25.0
    l_trig = box_l - 25.0

    post_bars = day_df[(day_df["time"] > datetime.time(9, 30)) & (day_df["time"] <= datetime.time(15, 0))].copy()
    post_bars["ema_20"] = day_df["close"].ewm(span=20, adjust=False).mean().loc[post_bars.index]

    entry_row = None
    opt_type = None
    for _, row in post_bars.iterrows():
        c = float(row["close"])
        ema = float(row["ema_20"])
        if c >= u_trig and c > ema:
            entry_row = row
            opt_type = "CE"
            break
        elif c <= l_trig and c < ema:
            entry_row = row
            opt_type = "PE"
            break

    if entry_row is None:
        return None

    entry_t = entry_row["timestamp"]
    c_spot = float(entry_row["close"])
    atm = int(round(c_spot / 100.0) * 100)
    strike = atm + 100 if opt_type == "CE" else atm - 100

    # Wednesday expiry DTE
    days_ahead = (2 - target_date.weekday()) % 7
    dte_base = max(0.001, float(days_ahead) / 365.25)

    sub = day_df[day_df["timestamp"] >= entry_t].copy()
    if sub.empty:
        return None

    sub["opt_p"] = [bs_calc(float(s), strike, dte_base, sigma=0.18, opt_type=opt_type) for s in sub["close"]]
    sub["opt_h"] = [bs_calc(float(s), strike, dte_base, sigma=0.18, opt_type=opt_type) for s in sub["high"]]
    sub["opt_l"] = [bs_calc(float(s), strike, dte_base, sigma=0.18, opt_type=opt_type) for s in sub["low"]]

    entry_p = round(float(sub.iloc[0]["opt_p"]) * (1.0 + SLIPPAGE_PCT), 2)
    cost = round(entry_p * LOT_BN, 2)
    if cost > MAX_CAPITAL_PER_TRADE:
        return None

    tgt_p = round(entry_p * 1.80, 2)
    sl_p = round(entry_p * 0.75, 2)

    exit_p = entry_p
    exit_t = sub.iloc[-1]["timestamp"]
    exit_reason = "EOD (15:15)"

    for _, r in sub.iloc[1:].iterrows():
        h = float(r["opt_h"])
        l = float(r["opt_l"])
        t = r["time"]

        if h >= tgt_p:
            exit_p = round(tgt_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_t = r["timestamp"]
            exit_reason = "Target Hit (+80%)"
            break
        elif l <= sl_p:
            exit_p = round(sl_p * (1.0 - SLIPPAGE_PCT), 2)
            exit_t = r["timestamp"]
            exit_reason = "SL Hit (-25%)"
            break
        elif t >= datetime.time(15, 15):
            exit_p = round(float(r["opt_p"]) * (1.0 - SLIPPAGE_PCT), 2)
            exit_t = r["timestamp"]
            exit_reason = "EOD (15:15)"
            break

    gross, charges, net = calc_charges(entry_p, exit_p, LOT_BN)
    return {
        "date": str(target_date),
        "strategy": "Strategy 2: BankNIFTY 15M Directional Breakout",
        "leg": opt_type,
        "strike": f"BANKNIFTY_{strike}_{opt_type}",
        "entry_time": entry_t.strftime("%H:%M:%S"),
        "exit_time": exit_t.strftime("%H:%M:%S"),
        "entry_p": entry_p,
        "exit_p": exit_p,
        "capital_used": cost,
        "gross_pnl": round(gross, 2),
        "charges": charges,
        "net_pnl": net,
        "exit_reason": exit_reason,
        "data_feed": "Black-Scholes (BSM)"
    }

def eval_s3(day_df, target_date):
    """Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)."""
    if len(day_df) < 50:
        return None

    days_ahead = (1 - target_date.weekday()) % 7
    dte_base = max(0.001, float(days_ahead) / 365.25)
    iv = 0.135

    # Compute rolling 20-min std and 5-min velocity of spot
    closes = day_df["close"].values
    highs = day_df["high"].values
    lows = day_df["low"].values
    times = day_df["time"].tolist()
    timestamps = day_df["timestamp"].tolist()
    n = len(closes)

    # Rolling std of 20 bars
    rolling_std = pd.Series(closes).rolling(20).std().values
    rolling_vel = np.abs(closes - np.roll(closes, 5))

    in_pos = False
    cooldown_until_idx = 0
    trade_res = None

    for i in range(20, n):
        t = times[i]
        # Trade window 09:45 to 14:00 (excludes chop 11:30 - 13:00)
        if t < datetime.time(9, 45) or t > datetime.time(14, 0):
            continue
        if datetime.time(11, 30) <= t <= datetime.time(13, 0):
            continue
        if i < cooldown_until_idx:
            continue

        std_v = rolling_std[i]
        vel_v = rolling_vel[i]

        # Compression (< 30 pts) and Kinetic Expansion (>= 35 pts)
        if std_v < 30.0 and vel_v >= 35.0:
            spot = float(closes[i])
            atm = int(round(spot / 50.0) * 50)

            # Select OTM strikes with premiums in 35-68 range
            ce_strike, pe_strike = None, None
            ce_p, pe_p = None, None

            for off in [100, 150, 200, 50]:
                k_ce = atm + off
                p = bs_calc(spot, k_ce, dte_base, sigma=iv, opt_type="CE")
                if 35.0 <= p <= 68.0:
                    ce_strike, ce_p = k_ce, p
                    break

            for off in [100, 150, 200, 50]:
                k_pe = atm - off
                p = bs_calc(spot, k_pe, dte_base, sigma=iv, opt_type="PE")
                if 35.0 <= p <= 68.0:
                    pe_strike, pe_p = k_pe, p
                    break

            if not (ce_strike and pe_strike):
                continue

            entry_tot = round(ce_p + pe_p, 2)
            cost = round(entry_tot * LOT_NIFTY, 2)
            if cost > MAX_CAPITAL_PER_TRADE:
                continue

            # Decoupled tracking forward
            entry_t = timestamps[i]
            ce_tgt = round(ce_p * 1.60, 2)
            ce_sl = round(ce_p * 0.88, 2)
            pe_tgt = round(pe_p * 1.60, 2)
            pe_sl = round(pe_p * 0.88, 2)

            ce_exited, pe_exited = False, False
            ce_exit_p, pe_exit_p = ce_p, pe_p
            ce_reason, pe_reason = "", ""
            exit_t = timestamps[min(i + 50, n - 1)]

            for j in range(i + 1, min(i + 51, n)):
                s_h = float(highs[j])
                s_l = float(lows[j])
                s_c = float(closes[j])
                t_j = times[j]

                # CE Leg
                if not ce_exited:
                    c_h = bs_calc(s_h, ce_strike, dte_base, sigma=iv, opt_type="CE")
                    c_l = bs_calc(s_l, ce_strike, dte_base, sigma=iv, opt_type="CE")
                    if c_h >= ce_tgt:
                        ce_exited = True
                        ce_exit_p = round(ce_tgt * (1.0 - SLIPPAGE_PCT), 2)
                        ce_reason = "TP Hit (+60%)"
                    elif c_l <= ce_sl:
                        ce_exited = True
                        ce_exit_p = round(ce_sl * (1.0 - SLIPPAGE_PCT), 2)
                        ce_reason = "SL Hit (-12%)"

                # PE Leg
                if not pe_exited:
                    p_h = bs_calc(s_l, pe_strike, dte_base, sigma=iv, opt_type="PE")
                    p_l = bs_calc(s_h, pe_strike, dte_base, sigma=iv, opt_type="PE")
                    if p_h >= pe_tgt:
                        pe_exited = True
                        pe_exit_p = round(pe_tgt * (1.0 - SLIPPAGE_PCT), 2)
                        pe_reason = "TP Hit (+60%)"
                    elif p_l <= pe_sl:
                        pe_exited = True
                        pe_exit_p = round(pe_sl * (1.0 - SLIPPAGE_PCT), 2)
                        pe_reason = "SL Hit (-12%)"

                if ce_exited and pe_exited:
                    exit_t = timestamps[j]
                    break

                if j == min(i + 50, n - 1) or t_j >= datetime.time(15, 15):
                    if not ce_exited:
                        ce_exit_p = round(bs_calc(s_c, ce_strike, dte_base, sigma=iv, opt_type="CE") * (1.0 - SLIPPAGE_PCT), 2)
                        ce_reason = "Time Cutoff"
                    if not pe_exited:
                        pe_exit_p = round(bs_calc(s_c, pe_strike, dte_base, sigma=iv, opt_type="PE") * (1.0 - SLIPPAGE_PCT), 2)
                        pe_reason = "Time Cutoff"
                    exit_t = timestamps[j]
                    break

            exit_tot = round(ce_exit_p + pe_exit_p, 2)
            gross, charges, net = calc_charges(entry_tot, exit_tot, LOT_NIFTY, is_strangle=True)
            trade_res = {
                "date": str(target_date),
                "strategy": "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)",
                "leg": "CE+PE",
                "strike": f"NIFTY_{ce_strike}CE+NIFTY_{pe_strike}PE",
                "entry_time": entry_t.strftime("%H:%M:%S"),
                "exit_time": exit_t.strftime("%H:%M:%S"),
                "entry_p": entry_tot,
                "exit_p": exit_tot,
                "capital_used": cost,
                "gross_pnl": round(gross, 2),
                "charges": charges,
                "net_pnl": net,
                "exit_reason": f"{ce_reason} | {pe_reason}",
                "data_feed": "Black-Scholes (BSM)"
            }
            break  # Max 1 trade per day for DAS

    return trade_res

def run_whole_year_simulation():
    print("=" * 95)
    print("   RUNNING WHOLE-YEAR (2025 - 2026) BLACK-SCHOLES (BSM) BACKTEST FOR STRATEGIES 1, 2, 3")
    print("=" * 95)

    df_n = pd.read_csv(NIFTY_FILE)
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n["date"] = df_n["timestamp"].dt.date
    df_n["time"] = df_n["timestamp"].dt.time

    df_b = pd.read_csv(BANKNIFTY_FILE)
    df_b["timestamp"] = pd.to_datetime(df_b["timestamp"]).dt.tz_localize(None)
    df_b["date"] = df_b["timestamp"].dt.date
    df_b["time"] = df_b["timestamp"].dt.time

    dates = sorted(df_n["date"].unique())
    print(f"Loaded {len(df_n)} Nifty candles & {len(df_b)} BankNifty candles across {len(dates)} trading days.")
    print(f"Evaluating each session from {dates[0]} to {dates[-1]}...")

    trades = []
    # Max concurrent positions = 1 across portfolio
    for idx, d in enumerate(dates):
        day_n = df_n[df_n["date"] == d].copy()
        day_b = df_b[df_b["date"] == d].copy()

        # Check Strategy 1 (Nifty Directional)
        t1 = eval_s1(day_n, d)
        # Check Strategy 2 (BankNifty Directional)
        t2 = eval_s2(day_b, d) if not day_b.empty else None
        # Check Strategy 3 (Nifty DAS)
        t3 = eval_s3(day_n, d)

        day_candidates = [t for t in [t1, t2, t3] if t is not None]
        if not day_candidates:
            continue

        # Sort by entry time
        day_candidates.sort(key=lambda x: x["entry_time"])

        # Select non-overlapping trades (Max concurrent = 1)
        cur_exit = "00:00:00"
        for tr in day_candidates:
            if tr["entry_time"] >= cur_exit:
                trades.append(tr)
                cur_exit = tr["exit_time"]

    df_trades = pd.DataFrame(trades)
    df_trades.sort_values(by=["date", "entry_time"], inplace=True)
    df_trades.to_csv(JOURNAL_CSV, index=False)
    print(f"\n[DONE] Successfully simulated {len(df_trades)} trades across the full year dataset!")
    print(f"Saved CSV journal to: {JOURNAL_CSV}")

    # Generate Markdown Journal
    tot_trades = len(df_trades)
    wins = len(df_trades[df_trades["net_pnl"] > 0])
    losses = len(df_trades[df_trades["net_pnl"] <= 0])
    wr = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    tot_net = round(float(df_trades["net_pnl"].sum()), 2)
    tot_gross = round(float(df_trades["gross_pnl"].sum()), 2)
    tot_charges = round(float(df_trades["charges"].sum()), 2)
    g_win = float(df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum())
    g_loss = float(abs(df_trades[df_trades["net_pnl"] <= 0]["net_pnl"].sum()))
    pf = round(g_win / g_loss, 2) if g_loss > 0 else 99.0
    cum = df_trades["net_pnl"].cumsum()
    max_dd = float((cum.cummax() - cum).max())
    final_balance = round(INITIAL_CAPITAL + tot_net, 2)
    ret_pct = round((tot_net / INITIAL_CAPITAL) * 100.0, 1)

    # Strategy breakdown
    summary_rows = []
    for s_name in [
        "Strategy 1: Nifty 15M Directional Breakout",
        "Strategy 2: BankNIFTY 15M Directional Breakout",
        "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)"
    ]:
        sub_s = df_trades[df_trades["strategy"] == s_name]
        c_count = len(sub_s)
        if c_count > 0:
            c_w = len(sub_s[sub_s["net_pnl"] > 0])
            c_wr = (c_w / c_count * 100.0)
            c_net = float(sub_s["net_pnl"].sum())
            c_gwin = float(sub_s[sub_s["net_pnl"] > 0]["net_pnl"].sum())
            c_gloss = float(abs(sub_s[sub_s["net_pnl"] <= 0]["net_pnl"].sum()))
            c_pf = round(c_gwin / c_gloss, 2) if c_gloss > 0 else 99.0
            c_cum = sub_s["net_pnl"].cumsum()
            c_dd = float((c_cum.cummax() - c_cum).max())
            summary_rows.append([s_name, f"{c_count} Trades", f"{c_wr:.1f}% Win Rate", f"{c_pf:.2f} PF", f"Max DD: INR {c_dd:,.0f}", f"{'+' if c_net>=0 else ''}INR {c_net:,.2f}"])
        else:
            summary_rows.append([s_name, "0 Trades", "0.0% Win Rate", "0.00 PF", "Max DD: INR 0", "INR 0.00"])

    summary_rows.append([
        "COMBINED TRI-STRATEGY WHOLE-YEAR PORTFOLIO",
        f"{tot_trades} Trades",
        f"{wr:.1f}% Win Rate",
        f"{pf:.2f} PF",
        f"Max DD: INR {max_dd:,.0f}",
        f"{'+' if tot_net>=0 else ''}INR {tot_net:,.2f}"
    ])

    print("\n" + " " * 20 + "--- WHOLE-YEAR (2025-2026) BLACK-SCHOLES PORTFOLIO SUMMARY ---")
    print(tabulate(summary_rows, headers=["Strategy", "Trades", "Win Rate", "Profit Factor", "Risk (Max DD)", "Net Return"], tablefmt="grid"))
    print(f"\nAccount Capital: INR 10,000.00  -->  Final Balance: INR {final_balance:,.2f} ({'+' if ret_pct>=0 else ''}{ret_pct:.1f}% Return)")
    print("=" * 95)

    # Write Markdown file
    with open(JOURNAL_MD, "w", encoding="utf-8") as f:
        f.write("# 📈 Whole-Year Black-Scholes (BSM) Trade Journal (2025 – 2026)\n\n")
        f.write(f"*Evaluated: Full 389 Trading Days ({dates[0]} to {dates[-1]}) on 1-Minute Exchange Candles*\n\n")
        f.write("---\n\n")
        f.write("## 📊 Whole-Year Performance Summary\n\n")
        f.write(f"- **Starting Portfolio Capital**: INR 10,000.00\n")
        f.write(f"- **Total Executed Trades**: **{tot_trades} Trades** (Max 1 concurrent position)\n")
        f.write(f"- **Win Rate**: **{wr:.1f}%** ({wins} Wins / {losses} Losses)\n")
        f.write(f"- **Profit Factor**: **{pf:.2f}**\n")
        f.write(f"- **Max Drawdown**: **INR {max_dd:,.2f}**\n")
        f.write(f"- **Total Gross Profit**: INR {tot_gross:+,.2f}\n")
        f.write(f"- **Total Statutory & Brokerage Deductions**: INR {tot_charges:,.2f}\n")
        f.write(f"- **Net Realized PnL**: **INR {tot_net:+,.2f}**\n")
        f.write(f"- **Final Account Balance**: **INR {final_balance:+,.2f}** (**{'+' if ret_pct>=0 else ''}{ret_pct:.1f}% Return**)\n\n")

        f.write("### Strategy Performance Breakdown\n\n")
        f.write("| Strategy | Trades | Win Rate | Profit Factor | Max DD | Net Return |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for row in summary_rows:
            f.write(f"| **{row[0]}** | {row[1]} | {row[2]} | {row[3]} | {row[4]} | **{row[5]}** |\n")
        f.write("\n---\n\n")

        f.write("## 📜 Complete Whole-Year Historical Trade Ledger\n\n")
        f.write("| # | Date | Strategy | Leg | Strike | Entry Time | Exit Time | Entry P | Exit P | Capital Used | Net PnL | Exit Reason |\n")
        f.write("|:---:|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|\n")
        for idx, r in df_trades.iterrows():
            net_val = float(r["net_pnl"])
            net_str = f"**+₹{net_val:.2f}**" if net_val > 0 else f"**-₹{abs(net_val):.2f}**"
            f.write(f"| {idx+1} | {r['date']} | {r['strategy']} | `{r['leg']}` | {r['strike']} | {r['entry_time']} | {r['exit_time']} | ₹{float(r['entry_p']):.2f} | ₹{float(r['exit_p']):.2f} | ₹{float(r['capital_used']):.2f} | {net_str} | {r['exit_reason']} |\n")
        f.write("\n")

    print(f"Saved Markdown journal to: {JOURNAL_MD}")

if __name__ == "__main__":
    run_whole_year_simulation()
