"""
Robust Fixed-Strike Parametric Optimizer for BankNIFTY over the entire Synthetic Dataset.
Ensures option strikes are FIXED at trade entry and priced realistically via ultra-fast C-level Black-Scholes.

Covers:
- 1-Minute Dataset: 8,664 bars (Aug 25 to Sep 28)
- 5-Minute Dataset: 4,652 bars (July 1 to Sep 28)
"""

import os
import math
import itertools
import numpy as np
import pandas as pd
from tabulate import tabulate

LOT_SIZE = 30
FEE_STRANGLE = 80.0
FEE_DIRECTIONAL = 45.0
RESULTS_DIR = "data/banknifty_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 95, flush=True)
print("  EXHAUSTIVE FIXED-STRIKE PARAMETRIC OPTIMIZATION (FULL SYNTHETIC DATASET)", flush=True)
print("=" * 95, flush=True)

# -------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------
print("\n[Step 1] Loading Spot Datasets...", flush=True)
df_1m = pd.read_csv("data/banknifty_1min_real.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"]).dt.tz_localize(None)
df_1m.sort_values("timestamp", inplace=True)
df_1m.reset_index(drop=True, inplace=True)
df_1m.set_index("timestamp", inplace=True)

df_5m = pd.read_csv("data/banknifty_5min_real.csv")
df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"]).dt.tz_localize(None)
df_5m.sort_values("timestamp", inplace=True)
df_5m.reset_index(drop=True, inplace=True)
df_5m.set_index("timestamp", inplace=True)

# 1-min arrays
n_1m = len(df_1m)
times_1m = df_1m.index.time
dates_1m = df_1m.index.date
closes_1m = df_1m["close"].values
highs_1m = df_1m["high"].values
lows_1m = df_1m["low"].values
mins_1m = np.array([t.hour * 60 + t.minute for t in times_1m])
weekdays_1m = np.array([ts.weekday() for ts in df_1m.index])

rolling_std_20_1m = pd.Series(closes_1m).rolling(20).std().fillna(50.0).values
vel_3_1m = closes_1m - pd.Series(closes_1m).shift(3).fillna(0.0).values

# 5-min arrays
n_5m = len(df_5m)
times_5m = df_5m.index.time
dates_5m = df_5m.index.date
closes_5m = df_5m["close"].values
highs_5m = df_5m["high"].values
lows_5m = df_5m["low"].values
mins_5m = np.array([t.hour * 60 + t.minute for t in times_5m])

df_5m["typical_p"] = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
session_mean_5m = df_5m.groupby(df_5m.index.date)["typical_p"].expanding().mean().values
tr1 = df_5m["high"] - df_5m["low"]
tr2 = (df_5m["high"] - df_5m["close"].shift(1)).abs()
tr3 = (df_5m["low"] - df_5m["close"].shift(1)).abs()
atr_5m = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().fillna(60.0).values
ema50_5m = df_5m["close"].ewm(span=50).mean().values
ema20_5m = df_5m["close"].ewm(span=20).mean().values

print(f"1-Minute Data: {n_1m} bars (Aug 25 to Sep 28)", flush=True)
print(f"5-Minute Data: {n_5m} bars (July 1 to Sep 28, 3 full months)", flush=True)

# -------------------------------------------------------------
# 2. ULTRA-FAST C-LEVEL BLACK-SCHOLES PRICER
# -------------------------------------------------------------
INV_SQRT_2 = 1.0 / math.sqrt(2.0)

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))

def bs_price(S, K, T, opt_type="CE", r=0.07, sigma=0.17):
    T_c = max(1e-4, T)
    sqrt_T = math.sqrt(T_c)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_c) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "CE":
        p = S * norm_cdf(d1) - K * math.exp(-r * T_c) * norm_cdf(d2)
    else:
        p = K * math.exp(-r * T_c) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(0.05, p)

def calc_dte(weekday, tm_min):
    target_exp = 2  # Wednesday expiry
    days_ahead = target_exp - weekday if weekday <= target_exp else 7 - (weekday - target_exp)
    mins = max(1.0, 930 - tm_min)
    frac = mins / 375.0
    d = max(0.01, frac * 0.5) if days_ahead == 0 else (days_ahead - 1) + frac
    return d / 365.0

dte_1m = np.array([calc_dte(w, m) for w, m in zip(weekdays_1m, mins_1m)])
weekdays_5m = np.array([ts.weekday() for ts in df_5m.index])
dte_5m = np.array([calc_dte(w, m) for w, m in zip(weekdays_5m, mins_5m)])

def eval_pnl(nets):
    if len(nets) == 0:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "avg": 0.0, "max_dd": 0.0}
    nets = np.array(nets)
    wins = nets[nets > 0]
    losses = nets[nets <= 0]
    w_sum = float(wins.sum()) if len(wins) > 0 else 0.0
    l_sum = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    pf = round(w_sum / l_sum, 2) if l_sum > 0 else (99.0 if w_sum > 0 else 0.0)
    wr = round(len(wins) / len(nets) * 100.0, 1)
    tot = round(float(nets.sum()), 2)
    avg_t = round(tot / len(nets), 2)
    cum = np.cumsum(nets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.max(peak - cum)) if len(cum) > 0 else 0.0
    return {"trades": len(nets), "wr": wr, "pf": pf, "pnl": tot, "avg": avg_t, "max_dd": round(dd, 2)}

# -------------------------------------------------------------
# 3. OPTIMIZE STRATEGY 1: DIRECTIONAL BOX BREAKOUT (DBB)
# -------------------------------------------------------------
print("\n" + "=" * 90, flush=True)
print("[Step 3] Sweeping Parameter Space for Directional Box Breakout (DBB)...", flush=True)
print("=" * 90, flush=True)

day_slices_1m = []
for d in np.unique(dates_1m):
    idxs = np.where(dates_1m == d)[0]
    day_slices_1m.append((d, idxs, mins_1m[idxs]))

def run_box_sim(box_mins, max_box_pct, buffer_pts, tp_pct, sl_pct, spot_sl=True, return_trades=False):
    trades = []
    nets = []
    box_end_min = 555 + box_mins

    for d, day_idxs, day_mins in day_slices_1m:
        b_mask = (day_mins >= 555) & (day_mins <= box_end_min)
        if np.sum(b_mask) < (box_mins // 2):
            continue
        b_idxs = day_idxs[b_mask]
        box_h = np.max(highs_1m[b_idxs])
        box_l = np.min(lows_1m[b_idxs])
        spot_ref = closes_1m[b_idxs[0]]
        if ((box_h - box_l) / spot_ref * 100.0) > max_box_pct:
            continue

        rem_mask = (day_mins > box_end_min) & (day_mins <= 915)
        rem_idxs = day_idxs[rem_mask]

        in_pos = False
        opt_type = ""
        opt_strike = 0
        entry_p = 0.0
        entry_idx = 0

        for idx in rem_idxs:
            h = highs_1m[idx]
            l = lows_1m[idx]
            c = closes_1m[idx]
            tm = mins_1m[idx]
            dte = dte_1m[idx]

            if in_pos:
                cur_p = bs_price(c, opt_strike, dte, opt_type)
                ret = (cur_p - entry_p) / entry_p
                is_tp = ret >= tp_pct
                is_sl = ret <= -sl_pct
                is_spot_sl = spot_sl and ((opt_type == "CE" and c < box_h) or (opt_type == "PE" and c > box_l))
                is_eod = tm >= 915

                if is_tp or is_sl or is_spot_sl or is_eod:
                    net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                    nets.append(net)
                    if return_trades:
                        reason = "TP Target" if is_tp else ("Stop Loss" if is_sl else ("Spot Return SL" if is_spot_sl else "EOD Exit"))
                        trades.append({
                            "date": str(d), "strategy": "Directional Box Breakout", "type": opt_type,
                            "strike": opt_strike, "entry_time": str(df_1m.index[entry_idx].time()),
                            "exit_time": str(df_1m.index[idx].time()), "entry_price": round(entry_p, 2),
                            "exit_price": round(cur_p, 2), "net_pnl": round(net, 2), "exit_reason": reason
                        })
                    in_pos = False
                    break
            else:
                if h >= (box_h + buffer_pts):
                    opt_type = "CE"
                    opt_strike = int(round(c / 100.0) * 100)
                    entry_p = bs_price(c, opt_strike, dte, opt_type)
                    entry_idx = idx
                    in_pos = True
                elif l <= (box_l - buffer_pts):
                    opt_type = "PE"
                    opt_strike = int(round(c / 100.0) * 100)
                    entry_p = bs_price(c, opt_strike, dte, opt_type)
                    entry_idx = idx
                    in_pos = True

    if return_trades:
        return pd.DataFrame(trades)
    return eval_pnl(nets)

box_results = []
box_grid = list(itertools.product(
    [30, 45, 60],                      # box_mins
    [0.75, 1.0, 1.25, 1.50],           # max_box_pct
    [30.0, 50.0, 75.0, 100.0],         # buffer_pts
    [0.30, 0.40, 0.50, 0.70],          # tp_pct
    [0.15, 0.20, 0.25],                # sl_pct
    [True, False]                      # spot_sl
))

print(f"Sweeping {len(box_grid)} DBB configurations...", flush=True)
for bm, mb, buf, tp, sl, sp_sl in box_grid:
    m = run_box_sim(bm, mb, buf, tp, sl, sp_sl)
    if m["trades"] >= 5:
        box_results.append({
            "box_m": f"{bm}m", "max_pct": f"{mb}%", "buf": f"{int(buf)}pt",
            "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%", "spot_sl": sp_sl,
            "trades": m["trades"], "wr": f"{m['wr']}%", "pf": m["pf"],
            "pnl": m["pnl"], "avg": m["avg"], "max_dd": m["max_dd"],
            "_params": (bm, mb, buf, tp, sl, sp_sl)
        })

df_box_opt = pd.DataFrame(box_results)
df_box_opt.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 BOX BREAKOUT PARAMETERS (FULL DATASET):", flush=True)
print(tabulate(df_box_opt.drop(columns=["_params"]).head(5), headers="keys", tablefmt="grid", showindex=False), flush=True)

# Export champion ledger
best_box_params = df_box_opt.iloc[0]["_params"]
df_box_champ = run_box_sim(*best_box_params, return_trades=True)
df_box_champ.to_csv(f"{RESULTS_DIR}/full_synthetic_box_champion_ledger.csv", index=False)

# -------------------------------------------------------------
# 4. OPTIMIZE STRATEGY 2: GAMMA SQUEEZE MOMENTUM (GSM)
# -------------------------------------------------------------
print("\n" + "=" * 90, flush=True)
print("[Step 4] Sweeping Parameter Space for Gamma Squeeze Momentum (GSM)...", flush=True)
print("=" * 90, flush=True)

def run_gsm_sim(atr_mult, tp_pct, sl_pct, max_trades_day=1, trend_filter="None", return_trades=False):
    trades = []
    nets = []
    in_pos = False
    opt_type = ""
    opt_strike = 0
    entry_p = 0.0
    entry_idx = 0
    last_exit_idx = -20
    trades_today = 0
    current_day = None

    for i in range(25, n_5m):
        tm_min = mins_5m[i]
        d = dates_5m[i]
        dte = dte_5m[i]

        if d != current_day:
            current_day = d
            trades_today = 0

        spot = closes_5m[i]
        prev_spot = closes_5m[i-1]
        s_mean = session_mean_5m[i]
        prev_s_mean = session_mean_5m[i-1]
        atr = atr_5m[i]
        prev_atr = atr_5m[i-1]

        if in_pos:
            cur_p = bs_price(spot, opt_strike, dte, opt_type)
            ret = (cur_p - entry_p) / entry_p
            is_tp = ret >= tp_pct
            is_sl = ret <= -sl_pct
            is_rev = (opt_type == "CE" and spot < s_mean) or (opt_type == "PE" and spot > s_mean)
            is_eod = tm_min >= 915

            if is_tp or is_sl or is_rev or is_eod:
                net = (cur_p - entry_p) * LOT_SIZE - FEE_DIRECTIONAL
                nets.append(net)
                if return_trades:
                    reason = "TP Target" if is_tp else ("Stop Loss" if is_sl else ("Mean Reversion" if is_rev else "EOD Exit"))
                    trades.append({
                        "date": str(d), "strategy": "Gamma Squeeze Momentum", "type": opt_type,
                        "strike": opt_strike, "entry_time": str(df_5m.index[entry_idx].time()),
                        "exit_time": str(df_5m.index[i].time()), "entry_price": round(entry_p, 2),
                        "exit_price": round(cur_p, 2), "net_pnl": round(net, 2), "exit_reason": reason
                    })
                in_pos = False
                last_exit_idx = i
                trades_today += 1
        else:
            if tm_min < 570 or tm_min > 870:
                continue
            if (i - last_exit_idx) < 3:
                continue
            if trades_today >= max_trades_day:
                continue

            thresh_up = s_mean + atr_mult * atr
            prev_thresh_up = prev_s_mean + atr_mult * prev_atr
            fresh_ce = (prev_spot <= prev_thresh_up) and (spot > thresh_up)

            thresh_dn = s_mean - atr_mult * atr
            prev_thresh_dn = prev_s_mean - atr_mult * prev_atr
            fresh_pe = (prev_spot >= prev_thresh_dn) and (spot < thresh_dn)

            if fresh_ce:
                if trend_filter == "EMA50" and spot < ema50_5m[i]:
                    continue
                if trend_filter == "EMA20" and spot < ema20_5m[i]:
                    continue
                opt_type = "CE"
                opt_strike = int(round(spot / 100.0) * 100)
                entry_p = bs_price(spot, opt_strike, dte, opt_type)
                entry_idx = i
                in_pos = True
            elif fresh_pe:
                if trend_filter == "EMA50" and spot > ema50_5m[i]:
                    continue
                if trend_filter == "EMA20" and spot > ema20_5m[i]:
                    continue
                opt_type = "PE"
                opt_strike = int(round(spot / 100.0) * 100)
                entry_p = bs_price(spot, opt_strike, dte, opt_type)
                entry_idx = i
                in_pos = True

    if return_trades:
        return pd.DataFrame(trades)
    return eval_pnl(nets)

gsm_results = []
gsm_grid = list(itertools.product(
    [1.8, 2.2, 2.5, 3.0],                  # atr_mult
    [0.30, 0.40, 0.50, 0.70],              # tp_pct
    [0.15, 0.20, 0.25],                    # sl_pct
    [1, 2],                                # max_trades_day
    ["None", "EMA50", "EMA20"]             # trend_filter
))

print(f"Sweeping {len(gsm_grid)} GSM configurations across full 3 months...", flush=True)
for atr_m, tp, sl, mtd, tf in gsm_grid:
    m = run_gsm_sim(atr_m, tp, sl, mtd, tf)
    if m["trades"] >= 5:
        gsm_results.append({
            "atr": f"{atr_m}x", "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%",
            "max_t": mtd, "filter": tf,
            "trades": m["trades"], "wr": f"{m['wr']}%", "pf": m["pf"],
            "pnl": m["pnl"], "avg": m["avg"], "max_dd": m["max_dd"],
            "_params": (atr_m, tp, sl, mtd, tf)
        })

df_gsm_opt = pd.DataFrame(gsm_results)
df_gsm_opt.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 GAMMA SQUEEZE PARAMETERS (FULL 3-MONTH DATASET):", flush=True)
print(tabulate(df_gsm_opt.drop(columns=["_params"]).head(5), headers="keys", tablefmt="grid", showindex=False), flush=True)

best_gsm_params = df_gsm_opt.iloc[0]["_params"]
df_gsm_champ = run_gsm_sim(*best_gsm_params, return_trades=True)
df_gsm_champ.to_csv(f"{RESULTS_DIR}/full_synthetic_gsm_champion_ledger.csv", index=False)

# -------------------------------------------------------------
# 5. OPTIMIZE STRATEGY 3: DECOUPLED ASYMMETRIC STRANGLE (DAS)
# -------------------------------------------------------------
print("\n" + "=" * 90, flush=True)
print("[Step 5] Sweeping Parameter Space for Decoupled Asymmetric Strangle (DAS)...", flush=True)
print("=" * 90, flush=True)

def run_das_sim(std_th, vel_th, win_tgt, lose_sp, hold_m, offset=200, return_trades=False):
    trades = []
    nets = []
    in_pos = False
    ce_strike = 0
    pe_strike = 0
    ce_entry = 0.0
    pe_entry = 0.0
    ce_exited = False
    pe_exited = False
    ce_exit_p = 0.0
    pe_exit_p = 0.0
    entry_idx = 0
    last_exit_idx = -100

    for i in range(30, n_1m):
        tm_min = mins_1m[i]
        c = closes_1m[i]
        dte = dte_1m[i]

        if in_pos:
            bars = i - entry_idx
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")

            if not ce_exited:
                r_ce = (c_ce - ce_entry) / max(0.1, ce_entry)
                if r_ce >= win_tgt or r_ce <= -lose_sp:
                    ce_exited = True
                    ce_exit_p = c_ce

            if not pe_exited:
                r_pe = (c_pe - pe_entry) / max(0.1, pe_entry)
                if r_pe >= win_tgt or r_pe <= -lose_sp:
                    pe_exited = True
                    pe_exit_p = c_pe

            if (ce_exited and pe_exited) or (bars >= hold_m) or (tm_min >= 915):
                p_c = ce_exit_p if ce_exited else c_ce
                p_p = pe_exit_p if pe_exited else c_pe
                ce_pnl = (p_c - ce_entry) * LOT_SIZE
                pe_pnl = (p_p - pe_entry) * LOT_SIZE
                gross = ce_pnl + pe_pnl
                net = gross - FEE_STRANGLE
                nets.append(net)
                if return_trades:
                    trades.append({
                        "date": str(dates_1m[i]), "strategy": "Decoupled Asymmetric Strangle",
                        "strikes": f"{ce_strike}CE / {pe_strike}PE",
                        "entry_time": str(df_1m.index[entry_idx].time()),
                        "exit_time": str(df_1m.index[i].time()),
                        "ce_entry": round(ce_entry, 2), "ce_exit": round(p_c, 2), "ce_pnl": round(ce_pnl, 2),
                        "pe_entry": round(pe_entry, 2), "pe_exit": round(p_p, 2), "pe_pnl": round(pe_pnl, 2),
                        "gross_pnl": round(gross, 2), "friction": FEE_STRANGLE, "net_pnl": round(net, 2)
                    })
                in_pos = False
                last_exit_idx = i
        else:
            if tm_min < 585 or tm_min > 810 or (690 <= tm_min <= 765):
                continue
            if (i - last_exit_idx) < 30:
                continue

            if rolling_std_20_1m[i] < std_th and abs(vel_3_1m[i]) >= vel_th:
                atm = int(round(c / 100.0) * 100)
                ce_strike = atm + offset
                pe_strike = atm - offset
                ce_entry = bs_price(c, ce_strike, dte, "CE")
                pe_entry = bs_price(c, pe_strike, dte, "PE")
                in_pos = True
                entry_idx = i
                ce_exited = False
                pe_exited = False

    if return_trades:
        return pd.DataFrame(trades)
    return eval_pnl(nets)

das_results = []
das_grid = list(itertools.product(
    [35.0, 45.0, 55.0, 65.0],              # std_th
    [40.0, 55.0, 70.0, 85.0],              # vel_th
    [0.60, 0.80, 1.00, 1.20],              # win_tgt
    [0.15, 0.20, 0.25],                    # lose_sp
    [45, 60, 90],                          # hold_m
    [0, 100, 200]                          # offset
))

print(f"Sweeping {len(das_grid)} DAS configurations across full 8,664 bars...", flush=True)
for st, vt, wt, ls, hm, off in das_grid:
    m = run_das_sim(st, vt, wt, ls, hm, off)
    if m["trades"] >= 5:
        das_results.append({
            "std": st, "vel": vt, "tp": f"+{int(wt*100)}%", "sl": f"-{int(ls*100)}%",
            "hold": f"{hm}m", "otm": f"{off}pt",
            "trades": m["trades"], "wr": f"{m['wr']}%", "pf": m["pf"],
            "pnl": m["pnl"], "avg": m["avg"], "max_dd": m["max_dd"],
            "_params": (st, vt, wt, ls, hm, off)
        })

df_das_opt = pd.DataFrame(das_results)
df_das_opt.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 DECOUPLED STRANGLE PARAMETERS (FULL DATASET):", flush=True)
print(tabulate(df_das_opt.drop(columns=["_params"]).head(5), headers="keys", tablefmt="grid", showindex=False), flush=True)

best_das_params = df_das_opt.iloc[0]["_params"]
df_das_champ = run_das_sim(*best_das_params, return_trades=True)
df_das_champ.to_csv(f"{RESULTS_DIR}/full_synthetic_das_champion_ledger.csv", index=False)

# -------------------------------------------------------------
# 6. OPTIMIZE STRATEGY 4: HEDGED STRANGLE (HS)
# -------------------------------------------------------------
print("\n" + "=" * 90, flush=True)
print("[Step 6] Sweeping Parameter Space for Hedged Strangle (HS)...", flush=True)
print("=" * 90, flush=True)

def run_hs_sim(entry_time_min, target_pct, stop_pct, hold_m, offset=200, return_trades=False):
    trades = []
    nets = []
    in_pos = False
    entry_tot = 0.0
    entry_idx = 0
    ce_strike = 0
    pe_strike = 0

    for i in range(20, n_1m):
        tm_min = mins_1m[i]
        c = closes_1m[i]
        dte = dte_1m[i]
        w = weekdays_1m[i]

        if in_pos:
            bars = i - entry_idx
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")
            cur_tot = c_ce + c_pe
            ret = (cur_tot - entry_tot) / entry_tot

            if ret >= target_pct or ret <= -stop_pct or bars >= hold_m or tm_min >= 915:
                net = (cur_tot - entry_tot) * LOT_SIZE - FEE_STRANGLE
                nets.append(net)
                if return_trades:
                    reason = "Target Hit" if ret >= target_pct else ("Stop Loss" if ret <= -stop_pct else ("Time Expiry" if bars >= hold_m else "EOD Exit"))
                    trades.append({
                        "date": str(dates_1m[i]), "strategy": "Hedged Strangle",
                        "strikes": f"{ce_strike}CE / {pe_strike}PE",
                        "entry_time": str(df_1m.index[entry_idx].time()),
                        "exit_time": str(df_1m.index[i].time()),
                        "entry_cost": round(entry_tot, 2), "exit_cost": round(cur_tot, 2),
                        "gross_pnl": round((cur_tot - entry_tot) * LOT_SIZE, 2),
                        "friction": FEE_STRANGLE, "net_pnl": round(net, 2), "exit_reason": reason
                    })
                in_pos = False
        else:
            if tm_min != entry_time_min:
                continue
            if w not in [1, 2]:
                continue
            atm = int(round(c / 100.0) * 100)
            ce_strike = atm + offset
            pe_strike = atm - offset
            c_ce = bs_price(c, ce_strike, dte, "CE")
            c_pe = bs_price(c, pe_strike, dte, "PE")
            entry_tot = c_ce + c_pe
            entry_idx = i
            in_pos = True

    if return_trades:
        return pd.DataFrame(trades)
    return eval_pnl(nets)

hs_results = []
hs_grid = list(itertools.product(
    [570, 600, 660, 720, 780, 810],        # 09:30, 10:00, 11:00, 12:00, 13:00, 13:30
    [0.20, 0.30, 0.40, 0.50, 0.70],        # target_pct
    [0.15, 0.20, 0.25, 0.30],              # stop_pct
    [30, 45, 60, 90, 120],                 # hold_m
    [0, 100, 200]                          # offset
))

print(f"Sweeping {len(hs_grid)} Hedged Strangle configurations...", flush=True)
for et, tp, sl, hm, off in hs_grid:
    m = run_hs_sim(et, tp, sl, hm, off)
    if m["trades"] >= 4:
        et_str = f"{et//60:02d}:{et%60:02d}"
        hs_results.append({
            "entry": et_str, "tp": f"+{int(tp*100)}%", "sl": f"-{int(sl*100)}%",
            "hold": f"{hm}m", "otm": f"{off}pt",
            "trades": m["trades"], "wr": f"{m['wr']}%", "pf": m["pf"],
            "pnl": m["pnl"], "avg": m["avg"], "max_dd": m["max_dd"],
            "_params": (et, tp, sl, hm, off)
        })

df_hs_opt = pd.DataFrame(hs_results)
df_hs_opt.sort_values(by="pnl", ascending=False, inplace=True)
print("\n>>> TOP 5 HEDGED STRANGLE PARAMETERS (FULL DATASET):", flush=True)
print(tabulate(df_hs_opt.drop(columns=["_params"]).head(5), headers="keys", tablefmt="grid", showindex=False), flush=True)

best_hs_params = df_hs_opt.iloc[0]["_params"]
df_hs_champ = run_hs_sim(*best_hs_params, return_trades=True)
df_hs_champ.to_csv(f"{RESULTS_DIR}/full_synthetic_hs_champion_ledger.csv", index=False)

# Export all sweep results
df_box_opt.drop(columns=["_params"]).to_csv(f"{RESULTS_DIR}/fixed_strike_box_optimization.csv", index=False)
df_gsm_opt.drop(columns=["_params"]).to_csv(f"{RESULTS_DIR}/fixed_strike_gsm_optimization.csv", index=False)
df_das_opt.drop(columns=["_params"]).to_csv(f"{RESULTS_DIR}/fixed_strike_das_optimization.csv", index=False)
df_hs_opt.drop(columns=["_params"]).to_csv(f"{RESULTS_DIR}/fixed_strike_hs_optimization.csv", index=False)

print("\n" + "=" * 95, flush=True)
print("  EXHAUSTIVE OPTIMIZATION COMPLETE! All ledgers and parameter tables exported.", flush=True)
print("=" * 95, flush=True)
