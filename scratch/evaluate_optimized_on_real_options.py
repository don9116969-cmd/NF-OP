"""
Evaluate ALL 8 NEW Fine-Tuned Strategy Parameters on REAL OPTION CONTRACTS (SmartAPI / Cache):
Audits actual traded 1-minute OHLC tick candles from Angel One for the September - October 2026 dataset.
Directly compares the new optimized parameters against the old baseline (-INR 5,869.27 across 82 trades).
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
from daily_result_checker import get_smart_api
from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles
from src.banknifty.banknifty_das import bs_price, get_banknifty_dte

LOT_NIFTY = 65
LOT_BN = 30
FEE_DIR = 45.0
FEE_STR = 80.0

def get_nifty_dte(target_date: datetime.date) -> float:
    # Nifty weekly expiry on Tuesday (weekday 1)
    days_ahead = (1 - target_date.weekday()) % 7
    return max(0.001, float(days_ahead) / 365.25)

def get_candle_feed(api, target_date, strike, opt_type, underlying, day_df):
    contract = get_active_option_contract(target_date, strike, opt_type, underlying=underlying)
    if contract:
        t_min = day_df["timestamp"].min()
        t_max = day_df["timestamp"].max()
        # Pass None so it loads instantly from our pre-cached SmartAPI dataset without network lag
        df_opt = fetch_real_option_candles(None, contract, t_min, t_max)
        if not df_opt.empty and len(df_opt) >= 5:
            return df_opt, contract["symbol"], "Real Option Tick"
    
    # Mathematical fallback if outside cached strike range
    dte = get_banknifty_dte(pd.Timestamp(target_date)) if underlying == "BANKNIFTY" else get_nifty_dte(target_date)
    df_synth = pd.DataFrame(index=day_df["timestamp"])
    prices = [bs_price(float(s), float(strike), dte, opt_type=opt_type, sigma=0.15) for s in day_df["close"]]
    df_synth["open"] = prices
    df_synth["high"] = prices
    df_synth["low"] = prices
    df_synth["close"] = prices
    sym = f"{underlying}_{target_date.strftime('%d%b%y').upper()}_{strike}_{opt_type}"
    return df_synth, sym, "BSM Fallback"

def eval_real_options():
    print("=" * 105)
    print("   EVALUATING ALL 8 NEW OPTIMIZED STRATEGIES ON REAL TRADED OPTION DATASET (SMARTAPI / CACHE)")
    print("=" * 105)

    api = get_smart_api()
    if api:
        print("[CONNECTED] Angel One SmartAPI connected successfully.")
    else:
        print("[OFFLINE] Using local pre-cached SmartAPI option candle data.")

    df_n = pd.read_csv("data/nifty_1min_real.csv")
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n["date"] = df_n["timestamp"].dt.date
    df_n.sort_values("timestamp", inplace=True)
    df_n.reset_index(drop=True, inplace=True)
    df_n["ema_20"] = df_n["close"].ewm(span=20, adjust=False).mean()

    df_bn = pd.read_csv("data/banknifty_1min_real.csv")
    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
    df_bn["date"] = df_bn["timestamp"].dt.date
    df_bn.sort_values("timestamp", inplace=True)
    df_bn.reset_index(drop=True, inplace=True)
    df_bn["ema_20"] = df_bn["close"].ewm(span=20, adjust=False).mean()

    n_days = {d: group.copy() for d, group in df_n.groupby("date") if d >= datetime.date(2026, 9, 21) and len(group) >= 20}
    bn_days = {d: group.copy() for d, group in df_bn.groupby("date") if d >= datetime.date(2026, 9, 21) and len(group) >= 20}

    all_dates = sorted(list(set(n_days.keys()).union(set(bn_days.keys()))))
    print(f"Auditing {len(all_dates)} live market sessions ({all_dates[0]} to {all_dates[-1]})...\n")

    executed_trades = []

    for target_date in all_dates:
        # =========================================================================
        # STRATEGY 1: NIFTY 15M DIRECTIONAL BREAKOUT (NEW OPTIMIZED)
        # Buffer 8.0pt | Box <= 0.30% | 20-EMA Trend | SL -16% | TP +60%
        # =========================================================================
        if target_date in n_days:
            day = n_days[target_date]
            b_bars = day[(day["timestamp"].dt.time >= datetime.time(9, 15)) & (day["timestamp"].dt.time <= datetime.time(9, 30))]
            if len(b_bars) >= 10:
                bh = float(b_bars["high"].max())
                bl = float(b_bars["low"].min())
                bpct = ((bh - bl) / b_bars["close"].iloc[0]) * 100.0

                if bpct <= 0.30:
                    u_trig = bh + 8.0
                    l_trig = bl - 8.0
                    post = day[(day["timestamp"].dt.time > datetime.time(9, 30)) & (day["timestamp"].dt.time <= datetime.time(15, 0))]

                    for idx, row in post.iterrows():
                        c = float(row["close"])
                        ema = float(row["ema_20"])
                        ts = row["timestamp"]

                        opt_type = None
                        if c >= u_trig and c > ema: opt_type = "CE"
                        elif c <= l_trig and c < ema: opt_type = "PE"

                        if opt_type:
                            atm = int(round(c / 50.0) * 50)
                            df_opt, sym, feed = get_candle_feed(api, target_date, atm, opt_type, "NIFTY", day)
                            sub = df_opt[df_opt.index >= ts]
                            if not sub.empty:
                                p_in = round(float(sub.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                                if (p_in * LOT_NIFTY) <= 10000.0:
                                    tgt_p = p_in * 1.60
                                    sl_p = p_in * 0.84
                                    p_out = p_in
                                    exit_t = sub.index[-1]
                                    reason = "EOD (15:15)"

                                    for o_ts, o_row in sub.iloc[1:].iterrows():
                                        if float(o_row["high"]) >= tgt_p:
                                            p_out = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "Target Hit (+60%)"
                                            break
                                        elif float(o_row["low"]) <= sl_p:
                                            p_out = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "SL Hit (-16%)"
                                            break
                                        elif o_ts.time() >= datetime.time(15, 15):
                                            p_out = round(float(o_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "EOD (15:15)"
                                            break

                                    gross = (p_out - p_in) * LOT_NIFTY
                                    turn = (p_in + p_out) * LOT_NIFTY
                                    charges = round(FEE_DIR + (p_out * LOT_NIFTY * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                    net = round(gross - charges, 2)

                                    executed_trades.append({
                                        "date": str(target_date),
                                        "strategy": "S1: Nifty 15M Directional",
                                        "symbol": sym,
                                        "side": opt_type,
                                        "entry_t": ts.strftime("%H:%M"),
                                        "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                        "entry_p": p_in, "exit_p": p_out,
                                        "net_pnl": net, "reason": reason, "feed": feed
                                    })
                            break

        # =========================================================================
        # STRATEGY 2: NIFTY AFTERNOON EXPIRY GAMMA (NEW OPTIMIZED)
        # Mon/Tue Expiry (12:30-14:45) | Buf 6.0pt | OTM_50 | SL -20% | TP +100%
        # =========================================================================
        if target_date in n_days and target_date.weekday() in [0, 1]:
            day = n_days[target_date]
            c_bars = day[(day["timestamp"].dt.time >= datetime.time(12, 30)) & (day["timestamp"].dt.time <= datetime.time(13, 0))]
            if len(c_bars) >= 20:
                bh = float(c_bars["high"].max())
                bl = float(c_bars["low"].min())
                post = day[(day["timestamp"].dt.time > datetime.time(13, 0)) & (day["timestamp"].dt.time <= datetime.time(14, 45))]

                for idx, row in post.iterrows():
                    c = float(row["close"])
                    ts = row["timestamp"]
                    opt_type = None
                    if c >= (bh + 6.0): opt_type = "CE"
                    elif c <= (bl - 6.0): opt_type = "PE"

                    if opt_type:
                        atm = int(round(c / 50.0) * 50)
                        strike = atm + 50 if opt_type == "CE" else atm - 50
                        df_opt, sym, feed = get_candle_feed(api, target_date, strike, opt_type, "NIFTY", day)
                        sub = df_opt[df_opt.index >= ts]
                        if not sub.empty:
                            p_in = round(float(sub.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            if (p_in * LOT_NIFTY) <= 8500.0:
                                tgt_p = p_in * 2.00
                                sl_p = p_in * 0.80
                                p_out = p_in
                                exit_t = sub.index[-1]
                                reason = "EOD (15:15)"

                                for o_ts, o_row in sub.iloc[1:].iterrows():
                                    if float(o_row["high"]) >= tgt_p:
                                        p_out = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                        exit_t = o_ts
                                        reason = "Target Hit (+100%)"
                                        break
                                    elif float(o_row["low"]) <= sl_p:
                                        p_out = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                        exit_t = o_ts
                                        reason = "SL Hit (-20%)"
                                        break
                                    elif o_ts.time() >= datetime.time(15, 15):
                                        p_out = round(float(o_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                                        exit_t = o_ts
                                        reason = "EOD (15:15)"
                                        break

                                gross = (p_out - p_in) * LOT_NIFTY
                                turn = (p_in + p_out) * LOT_NIFTY
                                charges = round(FEE_DIR + (p_out * LOT_NIFTY * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                net = round(gross - charges, 2)

                                executed_trades.append({
                                    "date": str(target_date),
                                    "strategy": "S2: Nifty Afternoon Gamma",
                                    "symbol": sym,
                                    "side": opt_type,
                                    "entry_t": ts.strftime("%H:%M"),
                                    "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                    "entry_p": p_in, "exit_p": p_out,
                                    "net_pnl": net, "reason": reason, "feed": feed
                                })
                        break

        # =========================================================================
        # STRATEGY 3: NIFTY 30M DIRECTIONAL ITM (NEW OPTIMIZED)
        # Buffer 4.0pt | Box <= 0.32% | 20-EMA Trend | ITM_100 | SL -22% | TP +110%
        # =========================================================================
        if target_date in n_days:
            day = n_days[target_date]
            b_bars = day[(day["timestamp"].dt.time >= datetime.time(9, 15)) & (day["timestamp"].dt.time <= datetime.time(9, 45))]
            if len(b_bars) >= 20:
                bh = float(b_bars["high"].max())
                bl = float(b_bars["low"].min())
                bpct = ((bh - bl) / b_bars["close"].iloc[0]) * 100.0

                if bpct <= 0.32:
                    u_trig = bh + 4.0
                    l_trig = bl - 4.0
                    post = day[(day["timestamp"].dt.time > datetime.time(9, 45)) & (day["timestamp"].dt.time <= datetime.time(15, 0))]

                    for idx, row in post.iterrows():
                        c = float(row["close"])
                        ema = float(row["ema_20"])
                        ts = row["timestamp"]

                        opt_type = None
                        if c >= u_trig and c > ema: opt_type = "CE"
                        elif c <= l_trig and c < ema: opt_type = "PE"

                        if opt_type:
                            atm = int(round(c / 50.0) * 50)
                            strike = atm - 100 if opt_type == "CE" else atm + 100
                            df_opt, sym, feed = get_candle_feed(api, target_date, strike, opt_type, "NIFTY", day)
                            sub = df_opt[df_opt.index >= ts]
                            if not sub.empty:
                                p_in = round(float(sub.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                                if (p_in * LOT_NIFTY) <= 10000.0:
                                    tgt_p = p_in * 2.10
                                    sl_p = p_in * 0.78
                                    p_out = p_in
                                    exit_t = sub.index[-1]
                                    reason = "EOD (15:15)"

                                    for o_ts, o_row in sub.iloc[1:].iterrows():
                                        if float(o_row["high"]) >= tgt_p:
                                            p_out = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "Target Hit (+110%)"
                                            break
                                        elif float(o_row["low"]) <= sl_p:
                                            p_out = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "SL Hit (-22%)"
                                            break
                                        elif o_ts.time() >= datetime.time(15, 15):
                                            p_out = round(float(o_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "EOD (15:15)"
                                            break

                                    gross = (p_out - p_in) * LOT_NIFTY
                                    turn = (p_in + p_out) * LOT_NIFTY
                                    charges = round(FEE_DIR + (p_out * LOT_NIFTY * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                    net = round(gross - charges, 2)

                                    executed_trades.append({
                                        "date": str(target_date),
                                        "strategy": "S3: Nifty 30M Directional ITM",
                                        "symbol": sym,
                                        "side": opt_type,
                                        "entry_t": ts.strftime("%H:%M"),
                                        "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                        "entry_p": p_in, "exit_p": p_out,
                                        "net_pnl": net, "reason": reason, "feed": feed
                                    })
                            break

        # =========================================================================
        # STRATEGY 4: NIFTY DECOUPLED ASYMMETRIC STRANGLE (DAS - NEW OPTIMIZED)
        # Std < 30.0 | Vel >= 35.0 | Win TP +60% | Lose SL -12% | Hold 50m
        # =========================================================================
        if target_date in n_days:
            day = n_days[target_date].copy().reset_index(drop=True)
            if len(day) >= 50:
                day["std20"] = day["close"].rolling(20).std()
                day["vel5"] = (day["close"] - day["close"].shift(5)).abs()

                trade_taken = False
                skip_until = 0
                for i in range(25, len(day) - 15):
                    if trade_taken: break
                    if i < skip_until: continue
                    row = day.iloc[i]
                    ts = row["timestamp"]
                    if ts.time() < datetime.time(9, 30) or ts.time() > datetime.time(14, 0): continue

                    if float(row["std20"]) < 30.0 and float(row["vel5"]) >= 35.0:
                        skip_until = i + 15
                        spot = float(row["close"])
                        atm = int(round(spot / 50.0) * 50)
                        ce_strk = atm + 100
                        pe_strk = atm - 100

                        df_ce, sym_ce, f_ce = get_candle_feed(api, target_date, ce_strk, "CE", "NIFTY", day)
                        df_pe, sym_pe, f_pe = get_candle_feed(api, target_date, pe_strk, "PE", "NIFTY", day)

                        sub_ce = df_ce[df_ce.index >= ts]
                        sub_pe = df_pe[df_pe.index >= ts]
                        if not sub_ce.empty and not sub_pe.empty:
                            p_ce_in = round(float(sub_ce.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            p_pe_in = round(float(sub_pe.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            tot_cost = (p_ce_in + p_pe_in) * LOT_NIFTY

                            if tot_cost <= 10000.0:
                                # Decoupled exits: Win TP +60%, Lose SL -12%
                                ce_tgt = p_ce_in * 1.60
                                ce_sl = p_ce_in * 0.88
                                pe_tgt = p_pe_in * 1.60
                                pe_sl = p_pe_in * 0.88

                                p_ce_out, p_pe_out = p_ce_in, p_pe_in
                                ce_done, pe_done = False, False
                                exit_t = ts
                                r_ce, r_pe = "Hold Limit", "Hold Limit"

                                c_idx = sub_ce.index.intersection(sub_pe.index)
                                bars_count = 0
                                for bar_t in c_idx[1:]:
                                    bars_count += 1
                                    c_row = sub_ce.loc[bar_t]
                                    p_row = sub_pe.loc[bar_t]

                                    if not ce_done:
                                        if float(c_row["high"]) >= ce_tgt:
                                            p_ce_out = round(ce_tgt * 0.995, 2)
                                            ce_done = True
                                            r_ce = "TP Hit (+60%)"
                                        elif float(c_row["low"]) <= ce_sl:
                                            p_ce_out = round(ce_sl * 0.995, 2)
                                            ce_done = True
                                            r_ce = "SL Hit (-12%)"

                                    if not pe_done:
                                        if float(p_row["high"]) >= pe_tgt:
                                            p_pe_out = round(pe_tgt * 0.995, 2)
                                            pe_done = True
                                            r_pe = "TP Hit (+60%)"
                                        elif float(p_row["low"]) <= pe_sl:
                                            p_pe_out = round(pe_sl * 0.995, 2)
                                            pe_done = True
                                            r_pe = "SL Hit (-12%)"

                                    if (ce_done and pe_done) or bars_count >= 50 or bar_t.time() >= datetime.time(15, 15):
                                        if not ce_done: p_ce_out = round(float(c_row["close"]) * 0.995, 2)
                                        if not pe_done: p_pe_out = round(float(p_row["close"]) * 0.995, 2)
                                        exit_t = bar_t
                                        break

                                gross = ((p_ce_out - p_ce_in) + (p_pe_out - p_pe_in)) * LOT_NIFTY
                                turn = (p_ce_in + p_ce_out + p_pe_in + p_pe_out) * LOT_NIFTY
                                charges = round(FEE_STR + (p_ce_out + p_pe_out) * LOT_NIFTY * 0.001 + turn * 0.0005 + FEE_STR * 0.18, 2)
                                net = round(gross - charges, 2)

                                executed_trades.append({
                                    "date": str(target_date),
                                    "strategy": "S4: Nifty Decoupled DAS",
                                    "symbol": f"{sym_ce}+{sym_pe}",
                                    "side": "CE+PE",
                                    "entry_t": ts.strftime("%H:%M"),
                                    "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                    "entry_p": round(p_ce_in + p_pe_in, 2),
                                    "exit_p": round(p_ce_out + p_pe_out, 2),
                                    "net_pnl": net,
                                    "reason": f"{r_ce} | {r_pe}",
                                    "feed": f_ce if f_ce == f_pe else "Mixed Feed"
                                })
                                trade_taken = True

        # =========================================================================
        # STRATEGY 5: BANKNIFTY KINETIC TREND DAS (NEW OPTIMIZED)
        # Std < 60.0 | Vel >= 120.0 | 20-EMA Trend | SL -25% | TP +80% | Hold 45m
        # =========================================================================
        if target_date in bn_days:
            day_b = bn_days[target_date].copy().reset_index(drop=True)
            if len(day_b) >= 50:
                day_b["std20"] = day_b["close"].rolling(20).std()
                day_b["vel5"] = (day_b["close"] - day_b["close"].shift(5)).abs()

                trade_taken = False
                skip_until = 0
                for i in range(25, len(day_b) - 15):
                    if trade_taken: break
                    if i < skip_until: continue
                    row = day_b.iloc[i]
                    ts = row["timestamp"]
                    if ts.time() < datetime.time(9, 45) or ts.time() > datetime.time(11, 30): continue

                    c = float(row["close"])
                    ema = float(row["ema_20"])
                    if float(row["std20"]) < 60.0 and float(row["vel5"]) >= 120.0:
                        skip_until = i + 15
                        opt_type = "CE" if c > ema else "PE"
                        atm = int(round(c / 100.0) * 100)
                        strike = atm + 100 if opt_type == "CE" else atm - 100

                        df_opt, sym, feed = get_candle_feed(api, target_date, strike, opt_type, "BANKNIFTY", day_b)
                        sub = df_opt[df_opt.index >= ts]
                        if not sub.empty:
                            p_in = round(float(sub.iloc[0]["close"]) * (1.0 + 0.005), 2)
                            if (p_in * LOT_BN) <= 8500.0:
                                tgt_p = p_in * 1.80
                                sl_p = p_in * 0.75
                                p_out = p_in
                                exit_t = sub.index[-1]
                                reason = "Hold Limit (45m)"

                                bars = 0
                                for o_ts, o_row in sub.iloc[1:].iterrows():
                                    bars += 1
                                    if float(o_row["high"]) >= tgt_p:
                                        p_out = round(tgt_p * 0.995, 2)
                                        exit_t = o_ts
                                        reason = "Target Hit (+80%)"
                                        break
                                    elif float(o_row["low"]) <= sl_p:
                                        p_out = round(sl_p * 0.995, 2)
                                        exit_t = o_ts
                                        reason = "SL Hit (-25%)"
                                        break
                                    elif bars >= 45 or o_ts.time() >= datetime.time(15, 15):
                                        p_out = round(float(o_row["close"]) * 0.995, 2)
                                        exit_t = o_ts
                                        reason = "Hold Limit (45m)"
                                        break

                                gross = (p_out - p_in) * LOT_BN
                                turn = (p_in + p_out) * LOT_BN
                                charges = round(FEE_DIR + (p_out * LOT_BN * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                net = round(gross - charges, 2)

                                executed_trades.append({
                                    "date": str(target_date),
                                    "strategy": "S5: BankNIFTY Kinetic DAS",
                                    "symbol": sym,
                                    "side": opt_type,
                                    "entry_t": ts.strftime("%H:%M"),
                                    "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                    "entry_p": p_in, "exit_p": p_out,
                                    "net_pnl": net, "reason": reason, "feed": feed
                                })
                                trade_taken = True

        # =========================================================================
        # STRATEGY 6: BANKNIFTY 30M DIRECTIONAL BOX BREAKOUT (DBB - NEW OPTIMIZED)
        # Buffer 30.0pt | Box <= 1.30% | 100-pt OTM | SL -25% | TP +80%
        # =========================================================================
        if target_date in bn_days:
            day_b = bn_days[target_date]
            b_bars = day_b[(day_b["timestamp"].dt.time >= datetime.time(9, 15)) & (day_b["timestamp"].dt.time <= datetime.time(9, 45))]
            if len(b_bars) >= 20:
                bh = float(b_bars["high"].max())
                bl = float(b_bars["low"].min())
                bpct = ((bh - bl) / b_bars["close"].iloc[0]) * 100.0

                if bpct <= 1.30:
                    u_trig = bh + 30.0
                    l_trig = bl - 30.0
                    post = day_b[(day_b["timestamp"].dt.time > datetime.time(9, 45)) & (day_b["timestamp"].dt.time <= datetime.time(15, 0))]

                    for idx, row in post.iterrows():
                        c = float(row["close"])
                        ts = row["timestamp"]
                        opt_type = None
                        if c >= u_trig: opt_type = "CE"
                        elif c <= l_trig: opt_type = "PE"

                        if opt_type:
                            atm = int(round(c / 100.0) * 100)
                            strike = atm + 100 if opt_type == "CE" else atm - 100
                            df_opt, sym, feed = get_candle_feed(api, target_date, strike, opt_type, "BANKNIFTY", day_b)
                            sub = df_opt[df_opt.index >= ts]
                            if not sub.empty:
                                p_in = round(float(sub.iloc[0]["close"]) * (1.0 + 0.005), 2)
                                if (p_in * LOT_BN) <= 8500.0:
                                    tgt_p = p_in * 1.80
                                    sl_p = p_in * 0.75
                                    p_out = p_in
                                    exit_t = sub.index[-1]
                                    reason = "EOD (15:15)"

                                    for o_ts, o_row in sub.iloc[1:].iterrows():
                                        if float(o_row["high"]) >= tgt_p:
                                            p_out = round(tgt_p * 0.995, 2)
                                            exit_t = o_ts
                                            reason = "Target Hit (+80%)"
                                            break
                                        elif float(o_row["low"]) <= sl_p:
                                            p_out = round(sl_p * 0.995, 2)
                                            exit_t = o_ts
                                            reason = "SL Hit (-25%)"
                                            break
                                        elif o_ts.time() >= datetime.time(15, 15):
                                            p_out = round(float(o_row["close"]) * 0.995, 2)
                                            exit_t = o_ts
                                            reason = "EOD (15:15)"
                                            break

                                    gross = (p_out - p_in) * LOT_BN
                                    turn = (p_in + p_out) * LOT_BN
                                    charges = round(FEE_DIR + (p_out * LOT_BN * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                    net = round(gross - charges, 2)

                                    executed_trades.append({
                                        "date": str(target_date),
                                        "strategy": "S6: BankNIFTY 30M DBB",
                                        "symbol": sym,
                                        "side": opt_type,
                                        "entry_t": ts.strftime("%H:%M"),
                                        "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                        "entry_p": p_in, "exit_p": p_out,
                                        "net_pnl": net, "reason": reason, "feed": feed
                                    })
                            break

        # =========================================================================
        # STRATEGY 7: BANKNIFTY AFTERNOON EXPIRY GAMMA (GSM - NEW OPTIMIZED)
        # Wed/Thu Expiry (12:30-14:45) | Buf 80.0pt | SL -20% | TP +150%
        # =========================================================================
        if target_date in bn_days and target_date.weekday() in [2, 3]:
            day_b = bn_days[target_date]
            c_bars = day_b[(day_b["timestamp"].dt.time >= datetime.time(12, 30)) & (day_b["timestamp"].dt.time <= datetime.time(13, 0))]
            if len(c_bars) >= 20:
                bh = float(c_bars["high"].max())
                bl = float(c_bars["low"].min())
                post = day_b[(day_b["timestamp"].dt.time > datetime.time(13, 0)) & (day_b["timestamp"].dt.time <= datetime.time(14, 45))]

                for idx, row in post.iterrows():
                    c = float(row["close"])
                    ts = row["timestamp"]
                    opt_type = None
                    if c >= (bh + 80.0): opt_type = "CE"
                    elif c <= (bl - 80.0): opt_type = "PE"

                    if opt_type:
                        atm = int(round(c / 100.0) * 100)
                        df_opt, sym, feed = get_candle_feed(api, target_date, atm, opt_type, "BANKNIFTY", day_b)
                        sub = df_opt[df_opt.index >= ts]
                        if not sub.empty:
                            p_in = round(float(sub.iloc[0]["close"]) * (1.0 + 0.005), 2)
                            if (p_in * LOT_BN) <= 8500.0:
                                tgt_p = p_in * 2.50
                                sl_p = p_in * 0.80
                                p_out = p_in
                                exit_t = sub.index[-1]
                                reason = "EOD (15:15)"

                                for o_ts, o_row in sub.iloc[1:].iterrows():
                                    if float(o_row["high"]) >= tgt_p:
                                        p_out = round(tgt_p * 0.995, 2)
                                        exit_t = o_ts
                                        reason = "Target Hit (+150%)"
                                        break
                                    elif float(o_row["low"]) <= sl_p:
                                        p_out = round(sl_p * 0.995, 2)
                                        exit_t = o_ts
                                        reason = "SL Hit (-20%)"
                                        break
                                    elif o_ts.time() >= datetime.time(15, 15):
                                        p_out = round(float(o_row["close"]) * 0.995, 2)
                                        exit_t = o_ts
                                        reason = "EOD (15:15)"
                                        break

                                gross = (p_out - p_in) * LOT_BN
                                turn = (p_in + p_out) * LOT_BN
                                charges = round(FEE_DIR + (p_out * LOT_BN * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                net = round(gross - charges, 2)

                                executed_trades.append({
                                    "date": str(target_date),
                                    "strategy": "S7: BankNIFTY Expiry GSM",
                                    "symbol": sym,
                                    "side": opt_type,
                                    "entry_t": ts.strftime("%H:%M"),
                                    "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                    "entry_p": p_in, "exit_p": p_out,
                                    "net_pnl": net, "reason": reason, "feed": feed
                                })
                        break

        # =========================================================================
        # STRATEGY 8: BANKNIFTY 15M DIRECTIONAL BREAKOUT (NEW OPTIMIZED)
        # Buffer 25.0pt | Box <= 1.20% | 20-EMA Trend | 100-pt OTM | SL -25% | TP +80%
        # =========================================================================
        if target_date in bn_days:
            day_b = bn_days[target_date]
            b_bars = day_b[(day_b["timestamp"].dt.time >= datetime.time(9, 15)) & (day_b["timestamp"].dt.time <= datetime.time(9, 30))]
            if len(b_bars) >= 10:
                bh = float(b_bars["high"].max())
                bl = float(b_bars["low"].min())
                bpct = ((bh - bl) / b_bars["close"].iloc[0]) * 100.0

                if bpct <= 1.20:
                    u_trig = bh + 25.0
                    l_trig = bl - 25.0
                    post = day_b[(day_b["timestamp"].dt.time > datetime.time(9, 30)) & (day_b["timestamp"].dt.time <= datetime.time(15, 0))]

                    for idx, row in post.iterrows():
                        c = float(row["close"])
                        ema = float(row["ema_20"])
                        ts = row["timestamp"]

                        opt_type = None
                        if c >= u_trig and c > ema: opt_type = "CE"
                        elif c <= l_trig and c < ema: opt_type = "PE"

                        if opt_type:
                            atm = int(round(c / 100.0) * 100)
                            strike = atm + 100 if opt_type == "CE" else atm - 100
                            df_opt, sym, feed = get_candle_feed(api, target_date, strike, opt_type, "BANKNIFTY", day_b)
                            sub = df_opt[df_opt.index >= ts]
                            if not sub.empty:
                                p_in = round(float(sub.iloc[0]["close"]) * (1.0 + 0.005), 2)
                                if (p_in * LOT_BN) <= 8500.0:
                                    tgt_p = p_in * 1.80
                                    sl_p = p_in * 0.75
                                    p_out = p_in
                                    exit_t = sub.index[-1]
                                    reason = "EOD (15:15)"

                                    for o_ts, o_row in sub.iloc[1:].iterrows():
                                        if float(o_row["high"]) >= tgt_p:
                                            p_out = round(tgt_p * 0.995, 2)
                                            exit_t = o_ts
                                            reason = "Target Hit (+80%)"
                                            break
                                        elif float(o_row["low"]) <= sl_p:
                                            p_out = round(sl_p * 0.995, 2)
                                            exit_t = o_ts
                                            reason = "SL Hit (-25%)"
                                            break
                                        elif o_ts.time() >= datetime.time(15, 15):
                                            p_out = round(float(o_row["close"]) * 0.995, 2)
                                            exit_t = o_ts
                                            reason = "EOD (15:15)"
                                            break

                                    gross = (p_out - p_in) * LOT_BN
                                    turn = (p_in + p_out) * LOT_BN
                                    charges = round(FEE_DIR + (p_out * LOT_BN * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                    net = round(gross - charges, 2)

                                    executed_trades.append({
                                        "date": str(target_date),
                                        "strategy": "S8: BankNIFTY 15M Directional",
                                        "symbol": sym,
                                        "side": opt_type,
                                        "entry_t": ts.strftime("%H:%M"),
                                        "exit_t": exit_t.strftime("%H:%M") if hasattr(exit_t, 'strftime') else str(exit_t),
                                        "net_pnl": net, "reason": reason, "feed": feed
                                    })
                            break
        print(f"[{target_date}] Done. Cumulative trades: {len(executed_trades)}")

    # =========================================================================
    # DISPLAY COMPLETE AUDIT REPORT
    # =========================================================================
    df_res = pd.DataFrame(executed_trades)
    if df_res.empty:
        print("[WARN] No trades executed on real options dataset.")
        return

    print("\n" + "=" * 105)
    print("ALL EXECUTED TRADES ON REAL TRADED OPTION DATASET (SEP 21 - OCT 21, 2026):")
    table_trades = []
    for idx, r in df_res.iterrows():
        table_trades.append([
            r["date"], r["strategy"], r["symbol"], f"{r['entry_t']} -> {r['exit_t']}",
            f"INR {r['entry_p']:.1f}", f"INR {r['exit_p']:.1f}", f"INR {r['net_pnl']:+,.2f}",
            r["reason"], r["feed"]
        ])
    headers = ["Date", "Strategy", "Traded Symbol", "Times", "Entry", "Exit", "Net PnL", "Exit Reason", "Data Feed"]
    print(tabulate(table_trades, headers=headers, tablefmt="grid"))

    # Summary by Strategy
    print("\n" + "=" * 105)
    print("STRATEGY PERFORMANCE ON REAL SMARTAPI OPTION DATASET (SEP 21 - OCT 21, 2026):")
    strat_summary = []
    for strat, group in df_res.groupby("strategy"):
        w = group[group["net_pnl"] > 0]
        l = group[group["net_pnl"] <= 0]
        wr = (len(w) / len(group)) * 100.0
        w_sum = w["net_pnl"].sum()
        l_sum = abs(l["net_pnl"].sum())
        pf = (w_sum / l_sum) if l_sum > 0 else 99.0
        net = group["net_pnl"].sum()
        strat_summary.append([strat, f"{len(group)} Trades", f"{wr:.1f}%", f"{pf:.2f} PF", f"INR {net:+,.2f}"])

    print(tabulate(strat_summary, headers=["Strategy", "Trades", "Win Rate", "Profit Factor", "Net Realized PnL"], tablefmt="grid"))

    tot_trades = len(df_res)
    tot_w = len(df_res[df_res["net_pnl"] > 0])
    tot_wr = (tot_w / tot_trades) * 100.0
    tot_w_sum = df_res[df_res["net_pnl"] > 0]["net_pnl"].sum()
    tot_l_sum = abs(df_res[df_res["net_pnl"] <= 0]["net_pnl"].sum())
    tot_pf = (tot_w_sum / tot_l_sum) if tot_l_sum > 0 else 99.0
    tot_net = df_res["net_pnl"].sum()

    print("\n" + "=" * 105)
    print("THE REAL OPTION DATASET COMPARISON (SEP 21 - OCT 21, 2026):")
    comp_table = [
        ["OLD BASELINE (With 4 Losing Strangles)", "82 Trades", "36.6%", "0.78 PF", "-INR 5,869.27 (-58.7%)"],
        ["NEW OPTIMIZED STRATEGIES (Directional Alpha)", f"{tot_trades} Trades", f"{tot_wr:.1f}%", f"{tot_pf:.2f} PF", f"INR {tot_net:+,.2f} ({(tot_net/10000)*100:+.1f}%)"]
    ]
    print(tabulate(comp_table, headers=["Configuration", "Trades", "Win Rate", "Profit Factor", "Net Realized PnL"], tablefmt="grid"))
    print("=" * 105)

if __name__ == "__main__":
    eval_real_options()
