"""
High-Speed Vectorized Parameter Optimizer for Directional Nifty Strategies
Sweeps:
- Strategy 3: 30M Directional ITM Breakout
- Strategy 1: 15M Directional Breakout (converted from losing strangle)
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from config import config
from src.banknifty.banknifty_das import bs_price

def load_data():
    nifty_csv = "data/nifty_1min_real.csv"
    df_n = pd.read_csv(nifty_csv)
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n["date"] = df_n["timestamp"].dt.date
    df_n.sort_values("timestamp", inplace=True)
    df_n.reset_index(drop=True, inplace=True)

    df_n["ema_20"] = df_n["close"].ewm(span=20, adjust=False).mean()
    df_n["ema_50"] = df_n["close"].ewm(span=50, adjust=False).mean()

    n_days = {d: group.copy() for d, group in df_n.groupby("date") if len(group) >= 20}
    return n_days

def precompute_option_paths(n_days, box_mins=30, buf_pts=6.0, max_box_pct=0.35, itm_offset=-100):
    """
    Extracts all triggered breakout trades and precomputes their option price trajectory
    so thousands of (SL, TP, Trail) parameter combinations can be evaluated instantaneously.
    """
    trades_data = []

    for target_date, day_bars in n_days.items():
        box_end_time = datetime.time(9, 15 + box_mins)
        box_bars = day_bars[(day_bars["timestamp"].dt.time >= datetime.time(9, 15)) & (day_bars["timestamp"].dt.time <= box_end_time)]
        if len(box_bars) < (box_mins - 5):
            continue

        box_h = float(box_bars["high"].max())
        box_l = float(box_bars["low"].min())
        box_range = box_h - box_l
        spot_open = float(box_bars["close"].iloc[0])
        box_pct = (box_range / spot_open) * 100.0

        if box_pct > max_box_pct:
            continue

        upper_trig = box_h + buf_pts
        lower_trig = box_l - buf_pts

        post = day_bars[(day_bars["timestamp"].dt.time > box_end_time) & (day_bars["timestamp"].dt.time <= datetime.time(15, 0))]
        if post.empty:
            continue

        entry_idx = None
        opt_type = None

        for idx, row in post.iterrows():
            c = float(row["close"])
            if c >= upper_trig:
                entry_idx = idx
                opt_type = "CE"
                break
            elif c <= lower_trig:
                entry_idx = idx
                opt_type = "PE"
                break

        if entry_idx is None:
            continue

        entry_bar = post.loc[entry_idx]
        entry_spot = float(entry_bar["close"])
        entry_t = entry_bar["timestamp"].time()
        c_ema20 = float(entry_bar["ema_20"])
        c_ema50 = float(entry_bar["ema_50"])

        # Determine option strike
        atm = int(round(entry_spot / 50.0) * 50)
        strike = atm + itm_offset if opt_type == "CE" else atm - itm_offset

        # DTE
        weekday = target_date.weekday()
        days_ahead = (1 - weekday) % 7
        T_years = max(0.001, days_ahead / 365.0)

        entry_p = bs_price(entry_spot, strike, T_years, opt_type=opt_type, sigma=0.14)
        entry_p = round(entry_p * (1.0 + config.SLIPPAGE_PCT), 2)
        if entry_p <= 2.0 or (entry_p * 65) > 10000.0:
            continue

        # Extract remaining path
        trade_slice = day_bars.loc[entry_idx:].copy()
        highs = trade_slice["high"].values
        lows = trade_slice["low"].values
        closes = trade_slice["close"].values
        times = trade_slice["timestamp"].dt.time.values

        # Compute option path
        opt_highs = np.array([bs_price(h if opt_type == "CE" else l, strike, T_years, opt_type=opt_type, sigma=0.14) for h, l in zip(highs, lows)])
        opt_lows = np.array([bs_price(l if opt_type == "CE" else h, strike, T_years, opt_type=opt_type, sigma=0.14) for h, l in zip(highs, lows)])
        opt_closes = np.array([bs_price(c, strike, T_years, opt_type=opt_type, sigma=0.14) for c in closes])

        trades_data.append({
            "date": target_date,
            "opt_type": opt_type,
            "entry_p": entry_p,
            "entry_spot": entry_spot,
            "ema_20": c_ema20,
            "ema_50": c_ema50,
            "is_trend_aligned": (opt_type == "CE" and entry_spot > c_ema20) or (opt_type == "PE" and entry_spot < c_ema20),
            "opt_highs": opt_highs,
            "opt_lows": opt_lows,
            "opt_closes": opt_closes,
            "times": times
        })

    return trades_data

def evaluate_fast(trades_data, sl_pct, tgt_pct, trail_pct, use_trend=False, lot_size=65):
    pnls = []

    for t in trades_data:
        if use_trend and not t["is_trend_aligned"]:
            continue

        entry_p = t["entry_p"]
        sl_p = entry_p * (1.0 - sl_pct)
        tgt_p = entry_p * (1.0 + tgt_pct)
        peak_p = entry_p
        exit_p = entry_p

        opt_highs = t["opt_highs"]
        opt_lows = t["opt_lows"]
        opt_closes = t["opt_closes"]
        times = t["times"]

        for i in range(1, len(opt_highs)):
            h_opt = opt_highs[i]
            l_opt = opt_lows[i]
            c_opt = opt_closes[i]
            tm = times[i]

            if h_opt > peak_p:
                peak_p = h_opt

            dyn_sl = max(sl_p, peak_p * (1.0 - trail_pct)) if trail_pct > 0 and peak_p >= entry_p * (1.0 + trail_pct) else sl_p

            if h_opt >= tgt_p:
                exit_p = tgt_p * (1.0 - config.SLIPPAGE_PCT)
                break
            elif l_opt <= dyn_sl:
                exit_p = dyn_sl * (1.0 - config.SLIPPAGE_PCT)
                break
            elif tm >= datetime.time(15, 15):
                exit_p = c_opt * (1.0 - config.SLIPPAGE_PCT)
                break

        gross = (exit_p - entry_p) * lot_size
        charges = 45.0 + (entry_p + exit_p) * lot_size * 0.0015
        net = gross - charges
        pnls.append(net)

    if not pnls or len(pnls) < 20:
        return None

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = (len(wins) / len(pnls)) * 100.0
    tot_win = sum(wins)
    tot_loss = abs(sum(losses))
    pf = (tot_win / tot_loss) if tot_loss > 0 else 99.0
    net_tot = sum(pnls)

    cum = np.cumsum(pnls)
    dd = np.max(np.maximum.accumulate(cum) - cum)

    return {
        "trades": len(pnls),
        "wr": wr,
        "pf": pf,
        "net": net_tot,
        "dd": dd,
        "avg_trade": net_tot / len(pnls)
    }

def run_optimization():
    print("Loading NIFTY dataset...")
    n_days = load_data()
    print(f"Loaded {len(n_days)} trading sessions.")

    # 1. OPTIMIZE STRATEGY 3 (30M Directional ITM)
    print("\n" + "=" * 90)
    print("1. OPTIMIZING STRATEGY 3: 30M DIRECTIONAL ITM BREAKOUT")
    print("=" * 90)

    for buf in [4.0, 6.0, 10.0]:
        for max_box in [0.25, 0.32, 0.40]:
            trades_cache = precompute_option_paths(n_days, box_mins=30, buf_pts=buf, max_box_pct=max_box, itm_offset=-100)
            print(f"Precomputed {len(trades_cache)} breakouts for Buffer={buf}pt, MaxBox={max_box}%...")

            grid_results = []
            for sl in [0.10, 0.14, 0.18, 0.22]:
                for tgt in [0.40, 0.60, 0.80, 1.10]:
                    for trail in [0.0, 0.15, 0.25]:
                        for use_trend in [False, True]:
                            m = evaluate_fast(trades_cache, sl, tgt, trail, use_trend=use_trend)
                            if m and m["pf"] > 1.1:
                                grid_results.append({
                                    "buf": buf, "max_box": max_box, "sl": sl, "tgt": tgt,
                                    "trail": trail, "trend": use_trend, **m
                                })

            if grid_results:
                df_g = pd.DataFrame(grid_results).sort_values("net", ascending=False)
                best = df_g.iloc[0]
                print(f"  >>> BEST FOUND: Buf={best['buf']}pt | Box<{best['max_box']}% | Trend={best['trend']} | SL:-{int(best['sl']*100)}% | TP:+{int(best['tgt']*100)}% | Trail:{int(best['trail']*100)}%")
                print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f} | MaxDD: INR {best['dd']:,.2f}")

    # 2. OPTIMIZE STRATEGY 1 (CONVERT 15M BOX FROM STRANGLE TO DIRECTIONAL)
    print("\n" + "=" * 90)
    print("2. OPTIMIZING STRATEGY 1: 15M DIRECTIONAL BREAKOUT (CONVERTED FROM STRANGLE)")
    print("=" * 90)

    for buf in [4.0, 6.0, 8.0]:
        for max_box in [0.20, 0.30, 0.40]:
            trades_cache = precompute_option_paths(n_days, box_mins=15, buf_pts=buf, max_box_pct=max_box, itm_offset=0) # ATM
            grid_results = []
            for sl in [0.12, 0.16, 0.20]:
                for tgt in [0.40, 0.60, 0.80]:
                    for trail in [0.0, 0.15, 0.20]:
                        for use_trend in [False, True]:
                            m = evaluate_fast(trades_cache, sl, tgt, trail, use_trend=use_trend)
                            if m and m["pf"] > 1.05:
                                grid_results.append({
                                    "buf": buf, "max_box": max_box, "sl": sl, "tgt": tgt,
                                    "trail": trail, "trend": use_trend, **m
                                })

            if grid_results:
                df_g = pd.DataFrame(grid_results).sort_values("net", ascending=False)
                best = df_g.iloc[0]
                print(f"  >>> BEST FOUND (15M): Buf={best['buf']}pt | Box<{best['max_box']}% | Trend={best['trend']} | SL:-{int(best['sl']*100)}% | TP:+{int(best['tgt']*100)}% | Trail:{int(best['trail']*100)}%")
                print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f} | MaxDD: INR {best['dd']:,.2f}")

if __name__ == "__main__":
    run_optimization()
