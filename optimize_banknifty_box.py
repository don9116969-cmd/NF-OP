"""
Exhaustive Parametric Optimization for Strategy 2: Directional Box Breakout on BankNIFTY (1-Minute Candles).
Sweeps:
- Box Periods: 15m, 30m, 45m, 60m
- Max Box Range %: 0.30% to 0.90% (step 0.10%)
- Breakout Buffers: 20, 30, 40, 50, 65, 80, 100 points
- Option Take Profit: +40%, +50%, +60%, +80%, +100%, +120%
- Option Stop Loss: -15%, -20%, -25%, -30%, -35%
- Trailing Stop Lock: None, +30%, +40%, +50%
- Trend Filter: None, EMA 50, VWAP
"""
import os
import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

LOT_SIZE = 30
CHARGES_PER_TRADE = 45.0  # ₹45 round-trip statutory charges + brokerage for 1 lot

# Load BankNIFTY 1-Minute data
df = pd.read_csv("data/banknifty_1min_real.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time
df["ema_50"] = df["close"].ewm(span=50).mean()

# Calculate daily VWAP
df["typical_p"] = (df["high"] + df["low"] + df["close"]) / 3.0
df["tp_vol"] = df["typical_p"] * df["volume"]
df["cum_vol"] = df.groupby("date")["volume"].cumsum()
df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].clip(lower=1)
df["vwap"].fillna(df["close"].rolling(20).mean(), inplace=True)
if df["vwap"].isna().all():
    df["vwap"] = df["close"].rolling(20).mean()

# Fast Black-Scholes pricing function
def bs_price(S, K, T, opt_type, r=0.07, sigma=0.165):
    if T <= 0:
        return np.maximum(0.05, S - K if opt_type == "CE" else K - S)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        p = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.maximum(0.05, p)

def get_dte(ts):
    weekday = ts.weekday()
    target_exp = 2  # Wednesday
    if weekday <= target_exp:
        days_ahead = target_exp - weekday
    else:
        days_ahead = 7 - (weekday - target_exp)
    mins_to_expiry = max(1.0, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
    fractional_day = mins_to_expiry / 375.0
    if days_ahead == 0:
        return max(0.01, fractional_day * 0.5) / 365.0
    else:
        return ((days_ahead - 1) + fractional_day) / 365.0

print(f"Loaded {len(df)} 1-minute BankNIFTY candles across {df['date'].nunique()} trading days.")

# Pre-group by date for lightning fast box calculations
day_groups = []
for d, grp in df.groupby("date"):
    times = grp["time"].values
    highs = grp["high"].values
    lows = grp["low"].values
    closes = grp["close"].values
    emas = grp["ema_50"].values
    vwaps = grp["vwap"].values
    ts_list = grp["timestamp"].values
    day_groups.append({
        "date": d, "times": times, "highs": highs, "lows": lows,
        "closes": closes, "emas": emas, "vwaps": vwaps, "timestamps": ts_list
    })

def simulate_box(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, trail_pct, trend_filter):
    trades = []
    
    start_t = datetime.time(9, 15)
    end_h = 9 + ((15 + box_mins) // 60)
    end_m = (15 + box_mins) % 60
    box_end_t = datetime.time(end_h, end_m)
    eod_t = datetime.time(15, 15)

    for day in day_groups:
        times = day["times"]
        highs = day["highs"]
        lows = day["lows"]
        closes = day["closes"]
        emas = day["emas"]
        vwaps = day["vwaps"]
        ts_list = day["timestamps"]

        # 1. Identify box
        box_mask = (times >= start_t) & (times <= box_end_t)
        if np.sum(box_mask) < (box_mins // 2):
            continue

        b_highs = highs[box_mask]
        b_lows = lows[box_mask]
        box_h = np.max(b_highs)
        box_l = np.min(b_lows)
        box_range = box_h - box_l
        spot_ref = closes[box_mask][0]
        box_pct = (box_range / spot_ref) * 100.0

        if box_pct > max_box_pct:
            continue

        # 2. Iterate remaining bars of the day
        rem_mask = (times > box_end_t) & (times <= eod_t)
        rem_indices = np.where(rem_mask)[0]

        in_pos = False
        pos_dir = None
        pos_k = 0
        entry_p = 0.0
        peak_p = 0.0

        for idx in rem_indices:
            spot = closes[idx]
            h = highs[idx]
            l = lows[idx]
            t = ts_list[idx]
            tm = times[idx]
            em = emas[idx]
            vw = vwaps[idx]

            if in_pos:
                T = get_dte(pd.Timestamp(t))
                cur_p = bs_price(spot, pos_k, T, pos_dir)
                peak_p = max(peak_p, cur_p)
                ret_p = (cur_p - entry_p) / entry_p

                exit_trig = False
                # Take profit
                if ret_p >= tp_pct:
                    exit_trig = True
                # Stop loss
                elif ret_p <= -sl_pct:
                    exit_trig = True
                # Trailing stop
                elif trail_pct > 0 and (peak_p - entry_p) / entry_p >= trail_pct and cur_p <= peak_p * 0.85:
                    exit_trig = True
                elif tm >= eod_t:
                    exit_trig = True

                if exit_trig:
                    gross = (cur_p - entry_p) * LOT_SIZE
                    net = gross - CHARGES_PER_TRADE
                    trades.append(net)
                    in_pos = False
                    break  # Max 1 trade per day

            else:
                # Long entry
                if h >= (box_h + buffer_pts):
                    if trend_filter == "ema" and spot < em:
                        continue
                    if trend_filter == "vwap" and spot < vw:
                        continue
                    pos_dir = "CE"
                    pos_k = int(round(spot / 100.0) * 100)
                    T = get_dte(pd.Timestamp(t))
                    entry_p = bs_price(spot, pos_k, T, pos_dir)
                    peak_p = entry_p
                    in_pos = True

                # Short entry
                elif l <= (box_l - buffer_pts):
                    if trend_filter == "ema" and spot > em:
                        continue
                    if trend_filter == "vwap" and spot > vw:
                        continue
                    pos_dir = "PE"
                    pos_k = int(round(spot / 100.0) * 100)
                    T = get_dte(pd.Timestamp(t))
                    entry_p = bs_price(spot, pos_k, T, pos_dir)
                    peak_p = entry_p
                    in_pos = True

    if not trades:
        return None
    s = pd.Series(trades)
    w = s[s > 0].sum()
    l = abs(s[s <= 0].sum())
    pf = w / l if l > 0 else 99.0
    wr = len(s[s > 0]) / len(s) * 100.0
    tot = s.sum()
    return {
        "trades": len(s), "wr": round(wr, 1), "pf": round(pf, 2),
        "net_pnl": round(tot, 1), "avg": round(s.mean(), 1),
        "max_w": round(s.max(), 1), "max_l": round(s.min(), 1)
    }

# Run exhaustive parameter sweep
print("Running Exhaustive Directional Box Grid Search on BankNIFTY...")
results = []
count = 0

for box_mins in [15, 30, 45, 60]:
    for max_box_pct in [0.45, 0.60, 0.75, 0.90]:
        for buffer_pts in [25.0, 40.0, 60.0, 80.0]:
            for tp_pct in [0.50, 0.70, 0.90, 1.20]:
                for sl_pct in [0.20, 0.25, 0.30]:
                    for trail in [0.0, 0.40]:
                        for trend in ["none", "ema", "vwap"]:
                            count += 1
                            m = simulate_box(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, trail, trend)
                            if m and m["trades"] >= 10 and m["pf"] >= 1.50 and m["net_pnl"] > 0:
                                results.append({
                                    "box_m": box_mins, "max_pct": max_box_pct, "buffer": buffer_pts,
                                    "tp": tp_pct, "sl": sl_pct, "trail": trail, "trend": trend,
                                    "trades": m["trades"], "wr": m["wr"], "pf": m["pf"],
                                    "net_pnl": m["net_pnl"], "avg": m["avg"]
                                })

print(f"Evaluated {count} parameter combinations!")
res_df = pd.DataFrame(results)
if not res_df.empty:
    res_df.sort_values(by="pf", ascending=False, inplace=True)
    out_csv = "data/banknifty_results/box_optimization_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nTOP 15 COMBINATIONS FOR DIRECTIONAL BOX ON BANKNIFTY (Saved to {out_csv}):")
    print(tabulate(res_df.head(15), headers="keys", tablefmt="grid", showindex=False))
else:
    print("No parameter combinations met the criteria.")
