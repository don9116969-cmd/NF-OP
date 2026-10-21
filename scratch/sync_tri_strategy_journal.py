"""
Sync and populate data/trade_journal.csv & data/trade_journal.md exclusively
for the Tri-Strategy Core Architecture (Strategies 1, 2, 3):
1. Strategy 1: Nifty 15M Directional Breakout
2. Strategy 2: BankNIFTY 15M Directional Breakout
3. Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)

Replaces all old 8-strategy baseline clutter with the verified real-options trade log.
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from config import config
from daily_result_checker import CANDLE_FILE, BANKNIFTY_CANDLE_FILE
from scratch.evaluate_optimized_on_real_options import get_candle_feed
from src.decoupled_strangle.das_signals import check_compression_expansion, is_trade_window_valid
from src.features.greeks import black_scholes_price
from src.utils.journal_manager import refresh_markdown_journal, save_audited_days

LOT_NIFTY = config.LOT_SIZE
LOT_BN = config.BANKNIFTY_LOT_SIZE
FEE_DIR = 45.0
FEE_STR = 80.0

def generate_tri_strategy_journal():
    df_nifty = pd.read_csv(CANDLE_FILE)
    df_nifty["timestamp"] = pd.to_datetime(df_nifty["timestamp"]).dt.tz_localize(None)
    df_nifty["date"] = df_nifty["timestamp"].dt.date
    df_nifty["ema_20"] = df_nifty["close"].ewm(span=20, adjust=False).mean()

    df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE)
    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
    df_bn["date"] = df_bn["timestamp"].dt.date
    df_bn["ema_20"] = df_bn["close"].ewm(span=20, adjust=False).mean()

    min_date = datetime.date(2026, 9, 21)
    n_days = {d: g.sort_values("timestamp") for d, g in df_nifty.groupby("date") if d >= min_date}
    bn_days = {d: g.sort_values("timestamp") for d, g in df_bn.groupby("date") if d >= min_date}
    all_dates = sorted(list(set(n_days.keys()).union(set(bn_days.keys()))))

    executed_trades = []

    for target_date in all_dates:
        # ---------------------------------------------------------------------
        # STRATEGY 1: NIFTY 15M DIRECTIONAL BREAKOUT
        # ---------------------------------------------------------------------
        if target_date in n_days:
            day = n_days[target_date]
            b_bars = day[(day["timestamp"].dt.time >= datetime.time(9, 15)) & (day["timestamp"].dt.time <= datetime.time(9, 30))]
            if len(b_bars) >= 10:
                bh = float(b_bars["high"].max())
                bl = float(b_bars["low"].min())
                bpct = ((bh - bl) / b_bars["close"].iloc[0]) * 100.0

                if bpct <= config.S1_MAX_BOX_PCT:
                    u_trig = bh + config.S1_BUFFER_PTS
                    l_trig = bl - config.S1_BUFFER_PTS
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
                            df_opt, sym, feed = get_candle_feed(None, target_date, atm, opt_type, "NIFTY", day)
                            sub = df_opt[df_opt.index >= ts]
                            if not sub.empty:
                                p_in = round(float(sub.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                                cost = round(p_in * LOT_NIFTY, 2)
                                if cost <= config.MAX_CAPITAL_PER_TRADE:
                                    tgt_p = p_in * (1.0 + config.S1_TARGET_PCT)
                                    sl_p = p_in * (1.0 - config.S1_SL_PCT)
                                    p_out = p_in
                                    exit_t = sub.index[-1]
                                    reason = "EOD (15:15)"

                                    for o_ts, o_row in sub.iloc[1:].iterrows():
                                        if float(o_row["high"]) >= tgt_p:
                                            p_out = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = f"Target Hit (+{int(config.S1_TARGET_PCT*100)}%)"
                                            break
                                        elif float(o_row["low"]) <= sl_p:
                                            p_out = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = f"SL Hit (-{int(config.S1_SL_PCT*100)}%)"
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
                                        "strategy": "Strategy 1: Nifty 15M Directional Breakout",
                                        "leg": opt_type,
                                        "strike": sym,
                                        "entry_time": ts.strftime("%H:%M:%S"),
                                        "exit_time": exit_t.strftime("%H:%M:%S") if hasattr(exit_t, 'strftime') else str(exit_t),
                                        "entry_p": p_in,
                                        "exit_p": p_out,
                                        "capital_used": cost,
                                        "gross_pnl": round(gross, 2),
                                        "charges": charges,
                                        "net_pnl": net,
                                        "exit_reason": reason,
                                        "data_feed": feed
                                    })
                            break

        # ---------------------------------------------------------------------
        # STRATEGY 2: BANKNIFTY 15M DIRECTIONAL BREAKOUT
        # ---------------------------------------------------------------------
        if target_date in bn_days:
            day_b = bn_days[target_date]
            b_bars = day_b[(day_b["timestamp"].dt.time >= datetime.time(9, 15)) & (day_b["timestamp"].dt.time <= datetime.time(9, 30))]
            if len(b_bars) >= 10:
                bh = float(b_bars["high"].max())
                bl = float(b_bars["low"].min())
                bpct = ((bh - bl) / b_bars["close"].iloc[0]) * 100.0

                if bpct <= config.S2_MAX_BOX_PCT:
                    u_trig = bh + config.S2_BUFFER_PTS
                    l_trig = bl - config.S2_BUFFER_PTS
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
                            strike = atm + config.S2_OTM_OFFSET if opt_type == "CE" else atm - config.S2_OTM_OFFSET
                            df_opt, sym, feed = get_candle_feed(None, target_date, strike, opt_type, "BANKNIFTY", day_b)
                            sub = df_opt[df_opt.index >= ts]
                            if not sub.empty:
                                p_in = round(float(sub.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                                cost = round(p_in * LOT_BN, 2)
                                if cost <= config.MAX_CAPITAL_PER_TRADE:
                                    tgt_p = p_in * (1.0 + config.S2_TARGET_PCT)
                                    sl_p = p_in * (1.0 - config.S2_SL_PCT)
                                    p_out = p_in
                                    exit_t = sub.index[-1]
                                    reason = "EOD (15:15)"

                                    for o_ts, o_row in sub.iloc[1:].iterrows():
                                        if float(o_row["high"]) >= tgt_p:
                                            p_out = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = f"Target Hit (+{int(config.S2_TARGET_PCT*100)}%)"
                                            break
                                        elif float(o_row["low"]) <= sl_p:
                                            p_out = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = f"SL Hit (-{int(config.S2_SL_PCT*100)}%)"
                                            break
                                        elif o_ts.time() >= datetime.time(15, 15):
                                            p_out = round(float(o_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                                            exit_t = o_ts
                                            reason = "EOD (15:15)"
                                            break

                                    gross = (p_out - p_in) * LOT_BN
                                    turn = (p_in + p_out) * LOT_BN
                                    charges = round(FEE_DIR + (p_out * LOT_BN * 0.001) + turn * 0.0005 + FEE_DIR * 0.18, 2)
                                    net = round(gross - charges, 2)

                                    executed_trades.append({
                                        "date": str(target_date),
                                        "strategy": "Strategy 2: BankNIFTY 15M Directional Breakout",
                                        "leg": opt_type,
                                        "strike": sym,
                                        "entry_time": ts.strftime("%H:%M:%S"),
                                        "exit_time": exit_t.strftime("%H:%M:%S") if hasattr(exit_t, 'strftime') else str(exit_t),
                                        "entry_p": p_in,
                                        "exit_p": p_out,
                                        "capital_used": cost,
                                        "gross_pnl": round(gross, 2),
                                        "charges": charges,
                                        "net_pnl": net,
                                        "exit_reason": reason,
                                        "data_feed": feed
                                    })
                            break

        # ---------------------------------------------------------------------
        # STRATEGY 3: NIFTY DECOUPLED ASYMMETRIC STRANGLE (DAS)
        # ---------------------------------------------------------------------
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
                    if not is_trade_window_valid(ts.time()): continue

                    if float(row["std20"]) < config.DAS_STD_THRESH and float(row["vel5"]) >= config.DAS_VEL_THRESH:
                        skip_until = i + config.DAS_COOLDOWN_BARS
                        c = float(row["close"])
                        atm = int(round(c / 50.0) * 50)
                        ce_k = atm + 100
                        pe_k = atm - 100

                        df_ce, sym_ce, f_ce = get_candle_feed(None, target_date, ce_k, "CE", "NIFTY", day)
                        df_pe, sym_pe, f_pe = get_candle_feed(None, target_date, pe_k, "PE", "NIFTY", day)

                        sub_ce = df_ce[df_ce.index >= ts]
                        sub_pe = df_pe[df_pe.index >= ts]

                        if not sub_ce.empty and not sub_pe.empty:
                            p_ce_in = round(float(sub_ce.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            p_pe_in = round(float(sub_pe.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            cost = round((p_ce_in + p_pe_in) * LOT_NIFTY, 2)

                            if cost <= config.MAX_CAPITAL_PER_TRADE:
                                ce_tgt = p_ce_in * (1.0 + config.DAS_WIN_TARGET_PCT)
                                ce_sl = p_ce_in * (1.0 - config.DAS_LOSE_STOP_PCT)
                                pe_tgt = p_pe_in * (1.0 + config.DAS_WIN_TARGET_PCT)
                                pe_sl = p_pe_in * (1.0 - config.DAS_LOSE_STOP_PCT)

                                ce_done, pe_done = False, False
                                p_ce_out, p_pe_out = p_ce_in, p_pe_in
                                exit_t = sub_ce.index[-1]
                                r_ce, r_pe = "EOD", "EOD"

                                c_idx = sub_ce.index.intersection(sub_pe.index)
                                bars_count = 0
                                for bar_t in c_idx[1:]:
                                    bars_count += 1
                                    c_row = sub_ce.loc[bar_t]
                                    p_row = sub_pe.loc[bar_t]

                                    if not ce_done:
                                        if float(c_row["high"]) >= ce_tgt:
                                            p_ce_out = round(ce_tgt * (1.0 - config.SLIPPAGE_PCT), 2)
                                            ce_done = True
                                            r_ce = f"TP Hit (+{int(config.DAS_WIN_TARGET_PCT*100)}%)"
                                        elif float(c_row["low"]) <= ce_sl:
                                            p_ce_out = round(ce_sl * (1.0 - config.SLIPPAGE_PCT), 2)
                                            ce_done = True
                                            r_ce = f"SL Hit (-{int(config.DAS_LOSE_STOP_PCT*100)}%)"

                                    if not pe_done:
                                        if float(p_row["high"]) >= pe_tgt:
                                            p_pe_out = round(pe_tgt * (1.0 - config.SLIPPAGE_PCT), 2)
                                            pe_done = True
                                            r_pe = f"TP Hit (+{int(config.DAS_WIN_TARGET_PCT*100)}%)"
                                        elif float(p_row["low"]) <= pe_sl:
                                            p_pe_out = round(pe_sl * (1.0 - config.SLIPPAGE_PCT), 2)
                                            pe_done = True
                                            r_pe = f"SL Hit (-{int(config.DAS_LOSE_STOP_PCT*100)}%)"

                                    if (ce_done and pe_done) or bars_count >= config.DAS_MAX_HOLD_MINS or bar_t.time() >= datetime.time(15, 15):
                                        if not ce_done: p_ce_out = round(float(c_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                                        if not pe_done: p_pe_out = round(float(p_row["close"]) * (1.0 - config.SLIPPAGE_PCT), 2)
                                        exit_t = bar_t
                                        break

                                gross = ((p_ce_out - p_ce_in) + (p_pe_out - p_pe_in)) * LOT_NIFTY
                                turn = (p_ce_in + p_ce_out + p_pe_in + p_pe_out) * LOT_NIFTY
                                charges = round(FEE_STR + (p_ce_out + p_pe_out) * LOT_NIFTY * 0.001 + turn * 0.0005 + FEE_STR * 0.18, 2)
                                net = round(gross - charges, 2)

                                executed_trades.append({
                                    "date": str(target_date),
                                    "strategy": "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)",
                                    "leg": "CE+PE",
                                    "strike": f"{sym_ce}+{sym_pe}",
                                    "entry_time": ts.strftime("%H:%M:%S"),
                                    "exit_time": exit_t.strftime("%H:%M:%S") if hasattr(exit_t, 'strftime') else str(exit_t),
                                    "entry_p": round(p_ce_in + p_pe_in, 2),
                                    "exit_p": round(p_ce_out + p_pe_out, 2),
                                    "capital_used": cost,
                                    "gross_pnl": round(gross, 2),
                                    "charges": charges,
                                    "net_pnl": net,
                                    "exit_reason": f"{r_ce} | {r_pe}",
                                    "data_feed": f_ce if f_ce == f_pe else "Mixed Feed"
                                })
                                trade_taken = True

    df_out = pd.DataFrame(executed_trades)
    df_out.sort_values(by=["date", "entry_time"], inplace=True)
    df_out.to_csv("data/trade_journal.csv", index=False)
    print(f"[SUCCESS] Written {len(df_out)} clean trades to data/trade_journal.csv!")
    print(f"Total Net PnL: INR {df_out['net_pnl'].sum():+,.2f}")

    # Regenerate markdown journal
    refresh_markdown_journal()
    print("[SUCCESS] Refreshed data/trade_journal.md!")

    # Mark all audited days
    audited = [str(d) for d in all_dates]
    save_audited_days(audited)
    print(f"[SUCCESS] Saved {len(audited)} dates to data/audited_days.json!")

if __name__ == "__main__":
    generate_tri_strategy_journal()
