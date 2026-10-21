"""
High-Speed Vectorized Parameter Optimizer for BankNIFTY Directional Strategies:
- Strategy 6: BankNIFTY Directional Box Breakout (DBB)
- Strategy 8: BankNIFTY 15M Directional Breakout (converted from losing strangle)
- Strategy 7: BankNIFTY Expiry Gamma Squeeze (GSM - Window & Exit sweep)
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
from src.banknifty.banknifty_das import bs_price, get_banknifty_dte

LOT_SIZE = 30
FEE_DIRECTIONAL = 45.0

def load_bn_data():
    bn_csv = "data/banknifty_1min_real.csv"
    df_bn = pd.read_csv(bn_csv)
    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
    df_bn["date"] = df_bn["timestamp"].dt.date
    df_bn.sort_values("timestamp", inplace=True)
    df_bn.reset_index(drop=True, inplace=True)

    df_bn["ema_20"] = df_bn["close"].ewm(span=20, adjust=False).mean()
    df_bn["ema_50"] = df_bn["close"].ewm(span=50, adjust=False).mean()

    bn_days = {d: group.copy() for d, group in df_bn.groupby("date") if len(group) >= 20}
    return bn_days

def precompute_bn_box_trades(bn_days, box_mins=15, buf_pts=30.0, max_box_pct=0.80):
    trades = []
    for target_date, day_bars in bn_days.items():
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
        c_ema20 = float(entry_bar["ema_20"])

        # Strike selection: 100-pt OTM or ATM
        atm = int(round(entry_spot / 100.0) * 100)
        strike = atm + 100 if opt_type == "CE" else atm - 100

        dte_y = max(1e-4, get_banknifty_dte(entry_bar["timestamp"], target_date))
        entry_p = bs_price(entry_spot, strike, dte_y, opt_type=opt_type, sigma=0.18)
        entry_p = round(entry_p * (1.0 + 0.005), 2)
        if entry_p <= 5.0 or (entry_p * LOT_SIZE) > 8500.0:
            continue

        trade_slice = day_bars.loc[entry_idx:].copy()
        highs = trade_slice["high"].values
        lows = trade_slice["low"].values
        closes = trade_slice["close"].values
        times = trade_slice["timestamp"].dt.time.values

        opt_highs = np.array([bs_price(h if opt_type == "CE" else l, strike, dte_y, opt_type=opt_type, sigma=0.18) for h, l in zip(highs, lows)])
        opt_lows = np.array([bs_price(l if opt_type == "CE" else h, strike, dte_y, opt_type=opt_type, sigma=0.18) for h, l in zip(highs, lows)])
        opt_closes = np.array([bs_price(c, strike, dte_y, opt_type=opt_type, sigma=0.18) for c in closes])

        trades.append({
            "date": target_date,
            "opt_type": opt_type,
            "entry_p": entry_p,
            "entry_spot": entry_spot,
            "is_trend_aligned": (opt_type == "CE" and entry_spot > c_ema20) or (opt_type == "PE" and entry_spot < c_ema20),
            "opt_highs": opt_highs,
            "opt_lows": opt_lows,
            "opt_closes": opt_closes,
            "times": times
        })

    return trades

def evaluate_bn_fast(trades_data, sl_pct, tgt_pct, trail_pct, use_trend=False):
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
                exit_p = tgt_p * (1.0 - 0.005)
                break
            elif l_opt <= dyn_sl:
                exit_p = dyn_sl * (1.0 - 0.005)
                break
            elif tm >= datetime.time(15, 15):
                exit_p = c_opt * (1.0 - 0.005)
                break

        gross = (exit_p - entry_p) * LOT_SIZE
        charges = FEE_DIRECTIONAL + (entry_p + exit_p) * LOT_SIZE * 0.0015
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

def run_bn_optimization():
    print("Loading BankNIFTY dataset...")
    bn_days = load_bn_data()
    print(f"Loaded {len(bn_days)} trading sessions.")

    # 1. OPTIMIZE STRATEGY 8: 15M DIRECTIONAL BREAKOUT (CONVERTED FROM STRANGLE)
    print("\n" + "=" * 90)
    print("1. OPTIMIZING STRATEGY 8: BANKNIFTY 15M DIRECTIONAL BREAKOUT (CONVERTED FROM STRANGLE)")
    print("=" * 90)

    for buf in [25.0, 40.0, 60.0]:
        for max_box in [0.70, 0.90, 1.20]:
            trades = precompute_bn_box_trades(bn_days, box_mins=15, buf_pts=buf, max_box_pct=max_box)
            grid = []
            for sl in [0.15, 0.20, 0.25]:
                for tgt in [0.40, 0.60, 0.80]:
                    for trail in [0.0, 0.15, 0.25]:
                        for use_trend in [False, True]:
                            m = evaluate_bn_fast(trades, sl, tgt, trail, use_trend=use_trend)
                            if m and m["pf"] > 1.1:
                                grid.append({"buf": buf, "max_box": max_box, "sl": sl, "tgt": tgt, "trail": trail, "trend": use_trend, **m})
            if grid:
                df_g = pd.DataFrame(grid).sort_values("net", ascending=False)
                best = df_g.iloc[0]
                print(f"  >>> BEST (15M BN): Buf={best['buf']}pt | Box<{best['max_box']}% | Trend={best['trend']} | SL:-{int(best['sl']*100)}% | TP:+{int(best['tgt']*100)}% | Trail:{int(best['trail']*100)}%")
                print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f} | MaxDD: INR {best['dd']:,.2f}")

    # 2. OPTIMIZE STRATEGY 6: 30M DIRECTIONAL BREAKOUT (DBB)
    print("\n" + "=" * 90)
    print("2. OPTIMIZING STRATEGY 6: BANKNIFTY 30M DIRECTIONAL BREAKOUT (DBB)")
    print("=" * 90)

    for buf in [30.0, 50.0, 70.0]:
        for max_box in [0.80, 1.00, 1.30]:
            trades = precompute_bn_box_trades(bn_days, box_mins=30, buf_pts=buf, max_box_pct=max_box)
            grid = []
            for sl in [0.15, 0.20, 0.25]:
                for tgt in [0.40, 0.60, 0.80]:
                    for trail in [0.0, 0.15, 0.25]:
                        for use_trend in [False, True]:
                            m = evaluate_bn_fast(trades, sl, tgt, trail, use_trend=use_trend)
                            if m and m["pf"] > 1.1:
                                grid.append({"buf": buf, "max_box": max_box, "sl": sl, "tgt": tgt, "trail": trail, "trend": use_trend, **m})
            if grid:
                df_g = pd.DataFrame(grid).sort_values("net", ascending=False)
                best = df_g.iloc[0]
                print(f"  >>> BEST (30M BN): Buf={best['buf']}pt | Box<{best['max_box']}% | Trend={best['trend']} | SL:-{int(best['sl']*100)}% | TP:+{int(best['tgt']*100)}% | Trail:{int(best['trail']*100)}%")
                print(f"      Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f} | Net: INR {best['net']:+,.2f} | MaxDD: INR {best['dd']:,.2f}")

if __name__ == "__main__":
    run_bn_optimization()
