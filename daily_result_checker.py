"""
Triple-Strategy Daily Result Checker & Live Portfolio Engine.
Portfolio Components:
1. Strategy 1: Hedged Long Strangle (15M Box Breakout, Uncoupled Leg SL -20%, Target +75%)
2. Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze (Tuesday Expiry & Monday Pre-Expiry, SL -25%, Target +80%)
3. Strategy 3: 30-Minute Statistical Box Breakout (Directional ITM_50 Buying, SL -15%, Target +80%, Trail +25%)

Features:
- Auto-syncs latest 1-min candles from Angel One SmartAPI to data/nifty_1min_real.csv
- Checks all 3 strategies for today or any past date
- Displays complete trade breakdown, exit triggers, and net PnL
- Tracks cumulative INR 10,000 portfolio ledger
"""

import os
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate
from dotenv import dotenv_values
import pyotp
from SmartApi import SmartConnect

from src.features.greeks import black_scholes_price
from run_strangle_grid_exhaustive import compute_calendar_dte, generate_signals, run_single_simulation_with_data
from src.gamma_squeeze.gamma_signals import GammaSignalGenerators
from run_gamma_grid_exhaustive import run_single_gamma_simulation
from src.directional_box.directional_signals import DirectionalBoxSignals
from run_directional_grid_exhaustive import run_single_directional_simulation
from src.decoupled_strangle.das_engine import evaluate_das_for_day
from src.banknifty.banknifty_das import evaluate_banknifty_das_for_day
from src.banknifty.banknifty_strategies import (
    evaluate_banknifty_box_for_day,
    evaluate_banknifty_gamma_for_day,
    evaluate_banknifty_strangle_for_day
)
from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles
from config import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
LEDGER_PATH = os.path.join(BASE_DIR, "data", "paper_trading_ledger.csv")
CANDLE_FILE = os.path.join(BASE_DIR, "data", "nifty_1min_real.csv")
BANKNIFTY_CANDLE_FILE = os.path.join(BASE_DIR, "data", "banknifty_1min_real.csv")

def get_smart_api() -> SmartConnect:
    env = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    api_key = os.getenv("ANGEL_API_KEY") or env.get("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_CODE") or env.get("ANGEL_CLIENT_CODE")
    password = os.getenv("ANGEL_PASSWORD") or os.getenv("ANGEL_PIN") or os.getenv("ANGEI_PASSWORD") or env.get("ANGEL_PASSWORD")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET") or env.get("ANGEL_TOTP_SECRET")

    if not all([api_key, client_code, password, totp_secret]):
        return None

    import time as time_lib
    for attempt in range(3):
        try:
            api = SmartConnect(api_key=api_key, timeout=30)
            totp = pyotp.TOTP(totp_secret).now()
            session = api.generateSession(client_code, password, totp)
            if session and session.get("status"):
                return api
            elif session and "exceeding access rate" in str(session):
                time_lib.sleep(1.5)
        except Exception as e:
            if attempt < 2:
                time_lib.sleep(1.5)
            else:
                print(f"[WARN] Angel One login connection error: {e}")
    return None

def sync_and_update_candles(api: SmartConnect, local_file: str = CANDLE_FILE, symbol_token: str = "99926000") -> pd.DataFrame:
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    df_existing = pd.DataFrame()
    last_dt = None

    if os.path.exists(local_file):
        try:
            df_existing = pd.read_csv(local_file)
            if not df_existing.empty and "timestamp" in df_existing.columns:
                df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], errors="coerce").dt.tz_localize(None)
                df_existing.dropna(subset=["timestamp"], inplace=True)
                if not df_existing.empty:
                    last_dt = df_existing["timestamp"].max()
        except Exception as e:
            print(f"[WARN] Error reading {local_file}: {e}")

    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
    now = now.replace(tzinfo=None)
    if last_dt is None:
        from_dt = now - datetime.timedelta(days=5)
    else:
        # Start directly from last recorded candle to minimize data size
        from_dt = last_dt

    if api and (now - last_dt if last_dt is not None else datetime.timedelta(days=1)) >= datetime.timedelta(minutes=1):
        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str = now.strftime("%Y-%m-%d %H:%M")
        sym_label = "BankNIFTY" if symbol_token == "99926009" else "NIFTY 50"
        print(f"[AUTO-SYNC] Checking Angel One for new {sym_label} candles from {from_str} to {to_str}...")
        
        import time as time_lib
        res = None
        for attempt in range(5):
            try:
                time_lib.sleep(1.5 + (attempt * 2.0))
                res = api.getCandleData({
                    "exchange": "NSE",
                    "symboltoken": symbol_token,
                    "interval": "ONE_MINUTE",
                    "fromdate": from_str,
                    "todate": to_str
                })
                if res and res.get("status"):
                    break
                elif res and "exceeding access rate" in str(res):
                    print(f"[RATE-LIMIT] Angel One rate limit response received ({attempt+1}/5). Backing off...")
                    time_lib.sleep(3.0 * (attempt + 1))
            except Exception as e:
                err_str = str(e)
                if "exceeding access rate" in err_str or "DataException" in str(type(e)):
                    print(f"[RATE-LIMIT] Angel One rate limit error ({attempt+1}/5). Backing off...")
                    time_lib.sleep(3.0 * (attempt + 1))
                elif attempt < 4:
                    time_lib.sleep(2.5)
                else:
                    print(f"[WARN] Candle sync error on attempt {attempt+1}: {e}")
                    break

        try:
            if res and res.get("status") and res.get("data"):
                new_bars = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                new_bars["timestamp"] = pd.to_datetime(new_bars["timestamp"]).dt.tz_localize(None)
                print(f"[AUTO-SYNC] Downloaded {len(new_bars)} live/recent candles from Angel One.")

                if not df_existing.empty:
                    df_combined = pd.concat([df_existing, new_bars], ignore_index=True)
                else:
                    df_combined = new_bars

                df_combined.drop_duplicates(subset=["timestamp"], inplace=True)
                df_combined.sort_values("timestamp", inplace=True)
                df_combined.reset_index(drop=True, inplace=True)
                df_combined.to_csv(local_file, index=False)
                print(f"[AUTO-SYNC] Successfully updated {local_file}! Total dataset: {len(df_combined)} candles (Latest: {df_combined['timestamp'].max()}).")
                return df_combined
            else:
                print(f"[AUTO-SYNC] Candles are up to date.")
        except Exception as e:
            print(f"[WARN] Auto-sync failed: {e}")

    return df_existing

def evaluate_strategy_1(df_full: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    data = df_full.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
    data['date'] = data['timestamp'].dt.date

    day_bars = data[data['date'] == target_date].copy()
    if day_bars.empty or len(day_bars) < 15:
        return {"status": "NO_DATA", "reason": "No market data (holiday/weekend)"}

    box_bars = day_bars[(day_bars["timestamp"].dt.time >= datetime.time(9, 15)) & (day_bars["timestamp"].dt.time <= datetime.time(9, 30))]
    if len(box_bars) < 10:
        return {"status": "INCOMPLETE", "reason": "15-minute box is forming"}

    box_h = box_bars["high"].max()
    box_l = box_bars["low"].min()
    box_range = box_h - box_l
    spot_open = box_bars["close"].iloc[0]
    box_pct = (box_range / spot_open) * 100.0

    info = {
        "box_high": round(box_h, 2), "box_low": round(box_l, 2),
        "box_range_pts": round(box_range, 2), "box_pct": round(box_pct, 2),
        "upper_trigger": round(box_h + 6.0, 2), "lower_trigger": round(box_l - 6.0, 2)
    }

    if box_pct > 0.30:
        info["status"] = "FILTERED_WIDE_BOX"
        info["trade_occurred"] = False
        info["reason"] = f"Box width {box_range:.1f} pts ({box_pct:.2f}% > 0.30%). Filtered out to avoid chop."
        return info

    sig_day = generate_signals(day_bars.set_index('timestamp'), '15m_box', max_box_pct=0.30, buffer_pts=6.0)
    res = run_single_simulation_with_data(sig_day, strike_mode="fixed", fixed_otm=150, manage_mode="uncoupled", leg_sl_pct=0.20, target_pct=0.75, trailing_pct=0.20)

    if res.get("trades", 0) == 0:
        has_signal = (sig_day['signal'] != 0).any()
        if has_signal:
            info["status"] = "BUDGET_EXCEEDED"
            info["trade_occurred"] = False
            info["reason"] = f"Breakout triggered, but option premiums exceeded the INR 10,000 capital limit (Wednesday 6-DTE premiums)."
        else:
            info["status"] = "NO_BREAKOUT"
            info["trade_occurred"] = False
            info["reason"] = f"Price stayed within triggers ({info['lower_trigger']} - {info['upper_trigger']}). No trade."
        return info

    trade = res["trades_df"].iloc[0].to_dict()
    feed_label = "Mathematical (BSM)"
    if "ce_strike" in trade and "pe_strike" in trade:
        ce_contract = get_active_option_contract(target_date, trade["ce_strike"], "CE")
        pe_contract = get_active_option_contract(target_date, trade["pe_strike"], "PE")
        if ce_contract and pe_contract:
            t_min = day_bars["timestamp"].min()
            t_max = day_bars["timestamp"].max()
            df_ce = fetch_real_option_candles(api, ce_contract, t_min, t_max)
            df_pe = fetch_real_option_candles(api, pe_contract, t_min, t_max)
            if not df_ce.empty and not df_pe.empty:
                entry_ts = pd.to_datetime(trade["entry_time"])
                sub_ce = df_ce[df_ce.index >= entry_ts]
                sub_pe = df_pe[df_pe.index >= entry_ts]
                if not sub_ce.empty and not sub_pe.empty:
                    real_ce_entry = round(float(sub_ce.iloc[0]['close']) * (1.0 + config.SLIPPAGE_PCT), 2)
                    real_pe_entry = round(float(sub_pe.iloc[0]['close']) * (1.0 + config.SLIPPAGE_PCT), 2)
                    trade["entry_p"] = round(real_ce_entry + real_pe_entry, 2)
                    trade["cost"] = round(trade["entry_p"] * config.LOT_SIZE, 2)

                    ce_sl = real_ce_entry * (1 - 0.20)
                    ce_tgt = real_ce_entry * (1 + 0.75)
                    pe_sl = real_pe_entry * (1 - 0.20)
                    pe_tgt = real_pe_entry * (1 + 0.75)

                    ce_active, pe_active = True, True
                    ce_exit_p, pe_exit_p = real_ce_entry, real_pe_entry
                    ce_reason, pe_reason = 'EOD', 'EOD'
                    ce_exit_time, pe_exit_time = entry_ts, entry_ts

                    common_index = sub_pe.index.intersection(sub_ce.index)
                    for ts in common_index:
                        c_bar = sub_ce.loc[ts]
                        p_bar = sub_pe.loc[ts]

                        if pe_active:
                            if p_bar['high'] >= pe_tgt:
                                pe_active = False
                                pe_exit_p = round(pe_tgt * (1.0 - config.SLIPPAGE_PCT), 2)
                                pe_reason = 'PE Target (+75%)'
                                pe_exit_time = ts
                            elif p_bar['low'] <= pe_sl:
                                pe_active = False
                                pe_exit_p = round(pe_sl * (1.0 - config.SLIPPAGE_PCT), 2)
                                pe_reason = 'PE SL (-20%)'
                                pe_exit_time = ts

                        if ce_active:
                            if c_bar['low'] <= ce_sl:
                                ce_active = False
                                ce_exit_p = round(ce_sl * (1.0 - config.SLIPPAGE_PCT), 2)
                                ce_reason = 'CE SL (-20%)'
                                ce_exit_time = ts
                            elif c_bar['high'] >= ce_tgt:
                                ce_active = False
                                ce_exit_p = round(ce_tgt * (1.0 - config.SLIPPAGE_PCT), 2)
                                ce_reason = 'CE Target (+75%)'
                                ce_exit_time = ts

                        # Check 120-minute time cutoff or market close (15:15 IST)
                        elapsed_m = (ts - entry_ts).total_seconds() / 60.0
                        if elapsed_m >= 120.0 or ts.time() >= datetime.time(15, 15):
                            if ce_active:
                                ce_active = False
                                ce_exit_p = round(float(c_bar['close']) * (1.0 - config.SLIPPAGE_PCT), 2)
                                ce_reason = 'Time Stop (120m)'
                                ce_exit_time = ts
                            if pe_active:
                                pe_active = False
                                pe_exit_p = round(float(p_bar['close']) * (1.0 - config.SLIPPAGE_PCT), 2)
                                pe_reason = 'Time Stop (120m)'
                                pe_exit_time = ts
                            break

                        if not ce_active and not pe_active:
                            break

                    final_exit_time = max(ce_exit_time, pe_exit_time) if (ce_exit_time and pe_exit_time) else trade["exit_time"]
                    trade["exit_time"] = final_exit_time
                    trade["exit_p"] = round(ce_exit_p + pe_exit_p, 2)
                    gross_ce = (ce_exit_p - real_ce_entry) * config.LOT_SIZE
                    gross_pe = (pe_exit_p - real_pe_entry) * config.LOT_SIZE
                    trade["gross"] = round(gross_ce + gross_pe, 2)

                    buy_v = trade["entry_p"] * config.LOT_SIZE
                    sell_v = trade["exit_p"] * config.LOT_SIZE
                    turn = buy_v + sell_v
                    brok = 80.0
                    stt = sell_v * 0.001
                    exch = turn * 0.0005
                    gst = (brok + exch) * 0.18
                    trade["charges"] = round(brok + stt + exch + gst, 2)
                    trade["net"] = round(trade["gross"] - trade["charges"], 2)
                    trade["exit_reason"] = f"{pe_reason} | {ce_reason}"
                    feed_label = f"Angel One Real Traded ({ce_contract['symbol']} + {pe_contract['symbol']})"

    info["status"] = "TRADE_EXECUTED"
    info["trade_occurred"] = True
    info["entry_time"] = trade["entry_time"].strftime("%H:%M:%S") if hasattr(trade["entry_time"], 'strftime') else str(trade["entry_time"])
    info["exit_time"] = trade["exit_time"].strftime("%H:%M:%S") if hasattr(trade["exit_time"], 'strftime') else str(trade["exit_time"])
    info["strike"] = trade.get("strike", "Dual OTM")
    info["opt_type"] = trade.get("opt_type", "CE+PE")
    info["entry_p"] = trade.get("entry_p", 0.0)
    info["exit_p"] = trade.get("exit_p", 0.0)
    info["cost"] = trade.get("cost", 0.0)
    info["gross_pnl"] = round(trade["gross"], 2)
    info["charges"] = round(trade["charges"], 2)
    info["net_pnl"] = round(trade["net"], 2)
    info["exit_reason"] = trade["exit_reason"]
    info["data_feed"] = feed_label
    return info

def evaluate_strategy_2(df_full: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    data = df_full.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
    data['date'] = data['timestamp'].dt.date

    day_bars = data[data['date'] == target_date].copy()
    if day_bars.empty or len(day_bars) < 15:
        return {"status": "NO_DATA", "reason": "No market data"}

    weekday = target_date.weekday()
    if weekday not in [0, 1]:
        return {
            "status": "NON_EXPIRY_DAY",
            "trade_occurred": False,
            "reason": f"{target_date.strftime('%A')} is a Non-Expiry day (Strategy 2 trades Monday 1-DTE & Tuesday 0-DTE only)."
        }

    sig_day = GammaSignalGenerators.expiry_consolidation_breakout(
        day_bars.set_index('timestamp'), compression_mins=20, buffer_pts=5.0, window="morning", include_pre_expiry_monday=True
    )
    res = run_single_gamma_simulation(sig_day, target_premium=45.0, sl_pct=0.25, target_pct=0.80, trail_pct=0.40, time_stop_mins=45)

    if res.get("trades", 0) == 0:
        return {
            "status": "NO_GAMMA_TRIGGER",
            "trade_occurred": False,
            "reason": "No 20-minute consolidation breakout triggered during the morning window."
        }

    trade = res["trades_df"].iloc[0].to_dict()
    feed_label = "Mathematical (BSM)"
    contract = get_active_option_contract(target_date, trade["strike"], trade["opt_type"])
    if contract:
            t_min = day_bars["timestamp"].min()
            t_max = day_bars["timestamp"].max()
            df_opt = fetch_real_option_candles(api, contract, t_min, t_max)
            if not df_opt.empty:
                entry_ts = pd.to_datetime(trade["entry_time"])
                sub_entry = df_opt[df_opt.index >= entry_ts]
                if not sub_entry.empty:
                    real_entry_p = round(float(sub_entry.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                    trade["entry_p"] = real_entry_p
                    trade["cost"] = round(real_entry_p * config.LOT_SIZE, 2)

                    tgt_p = round(real_entry_p * 1.80, 2)
                    sl_p = round(real_entry_p * 0.75, 2)

                    exit_p = real_entry_p
                    exit_reason = "Time Stop (45m)"
                    exit_ts = sub_entry.index[-1]

                    for ts, bar in sub_entry.iterrows():
                        elapsed_m = (ts - entry_ts).total_seconds() / 60.0
                        b_high = float(bar['high'])
                        b_low = float(bar['low'])
                        b_close = float(bar['close'])

                        if b_high >= tgt_p:
                            exit_p = round(tgt_p * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "Target Hit (+80%)"
                            exit_ts = ts
                            break
                        elif b_low <= sl_p:
                            exit_p = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "SL Hit (-25%)"
                            exit_ts = ts
                            break
                        elif elapsed_m >= 45.0 or ts.time() >= datetime.time(15, 15):
                            exit_p = round(b_close * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "Time Stop (45m)"
                            exit_ts = ts
                            break

                    trade["exit_p"] = exit_p
                    trade["exit_time"] = exit_ts
                    trade["exit_reason"] = exit_reason
                    gross = (exit_p - real_entry_p) * config.LOT_SIZE
                    trade["gross"] = round(gross, 2)
                    buy_val = real_entry_p * config.LOT_SIZE
                    sell_val = exit_p * config.LOT_SIZE
                    turn = buy_val + sell_val
                    brok = 40.0
                    stt = sell_val * 0.001
                    exch = turn * 0.0005
                    gst = (brok + exch) * 0.18
                    trade["charges"] = round(brok + stt + exch + gst, 2)
                    trade["net"] = round(trade["gross"] - trade["charges"], 2)
                feed_label = f"Angel One Real Traded ({contract['symbol']})"

    return {
        "status": "TRADE_EXECUTED",
        "trade_occurred": True,
        "entry_time": trade["entry_time"].strftime("%H:%M:%S") if hasattr(trade["entry_time"], 'strftime') else str(trade["entry_time"]),
        "exit_time": trade["exit_time"].strftime("%H:%M:%S") if hasattr(trade["exit_time"], 'strftime') else str(trade["exit_time"]),
        "opt_type": trade["opt_type"],
        "strike": trade["strike"],
        "entry_p": trade["entry_p"],
        "exit_p": trade["exit_p"],
        "cost": trade["cost"],
        "gross_pnl": round(trade["gross"], 2),
        "charges": round(trade["charges"], 2),
        "net_pnl": round(trade["net"], 2),
        "exit_reason": trade["exit_reason"],
        "data_feed": feed_label
    }

def evaluate_strategy_3(df_full: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    data = df_full.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
    data['date'] = data['timestamp'].dt.date

    day_bars = data[data['date'] == target_date].copy()
    if day_bars.empty or len(day_bars) < 25:
        return {"status": "NO_DATA", "reason": "No market data"}

    box_bars = day_bars[(day_bars["timestamp"].dt.time >= datetime.time(9, 15)) & (day_bars["timestamp"].dt.time <= datetime.time(9, 45))]
    if len(box_bars) < 15:
        return {"status": "INCOMPLETE", "reason": "30-minute box is forming"}

    box_h = box_bars["high"].max()
    box_l = box_bars["low"].min()
    box_range = box_h - box_l
    spot_open = box_bars["close"].iloc[0]
    box_pct = (box_range / spot_open) * 100.0

    info = {
        "box_high": round(box_h, 2), "box_low": round(box_l, 2),
        "box_range_pts": round(box_range, 2), "box_pct": round(box_pct, 2),
        "upper_trigger": round(box_h + 6.0, 2), "lower_trigger": round(box_l - 6.0, 2)
    }

    if box_pct > 0.32:
        info["status"] = "FILTERED_WIDE_BOX"
        info["trade_occurred"] = False
        info["reason"] = f"30M Range {box_range:.1f} pts ({box_pct:.2f}% > 0.32%). Filtered out to avoid chop."
        return info

    sig_day = DirectionalBoxSignals.generate_30m_box_signals(day_bars.set_index('timestamp'), max_box_pct=0.32, buffer_pts=6.0, trade_cutoff_str="13:30")
    res = run_single_directional_simulation(sig_day, strike_type="ITM_50", sl_pct=0.15, target_pct=0.80, trail_trigger_pct=0.25, trailing_peak_pct=0.15)

    if res.get("trades", 0) == 0:
        has_signal = (sig_day['signal'] != 0).any()
        if has_signal:
            info["status"] = "BUDGET_EXCEEDED"
            info["trade_occurred"] = False
            info["reason"] = f"Breakout triggered, but ITM option cost exceeded the INR 10,000 budget (Wednesday far-DTE premiums)."
        else:
            info["status"] = "NO_BREAKOUT"
            info["trade_occurred"] = False
            info["reason"] = f"Price stayed within triggers ({info['lower_trigger']} - {info['upper_trigger']}). No breakout."
        return info

    trade = res["trades_df"].iloc[0].to_dict()
    feed_label = "Mathematical (BSM)"
    contract = get_active_option_contract(target_date, trade["strike"], trade["opt_type"])
    if contract:
            t_min = day_bars["timestamp"].min()
            t_max = day_bars["timestamp"].max()
            df_opt = fetch_real_option_candles(api, contract, t_min, t_max)
            if not df_opt.empty:
                entry_ts = pd.to_datetime(trade["entry_time"])
                sub_opt = df_opt[df_opt.index >= entry_ts]
                if not sub_opt.empty:
                    entry_p = round(float(sub_opt.iloc[0]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                    trade["entry_p"] = entry_p
                    trade["cost"] = round(entry_p * config.LOT_SIZE, 2)

                    sl_price = round(entry_p * (1.0 - 0.15), 2)
                    tgt_price = round(entry_p * (1.0 + 0.80), 2)
                    trail_act_price = round(entry_p * (1.0 + 0.25), 2)

                    peak_p = entry_p
                    exit_p = entry_p
                    exit_reason = "EOD (15:15 PM)"
                    exit_ts = sub_opt.index[-1]

                    for ts, bar in sub_opt.iterrows():
                        p_high = float(bar['high'])
                        p_low = float(bar['low'])
                        p_close = float(bar['close'])
                        peak_p = max(peak_p, p_high)

                        if p_high >= tgt_price:
                            exit_p = round(tgt_price * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "Target Hit (+80%)"
                            exit_ts = ts
                            break
                        elif p_low <= sl_price:
                            exit_p = round(sl_price * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "SL Hit (-15%)"
                            exit_ts = ts
                            break
                        elif peak_p >= trail_act_price and p_close <= peak_p * 0.85:
                            exit_p = round(p_close * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "Trailing Peak Exit"
                            exit_ts = ts
                            break
                        elif ts.time() >= datetime.time(15, 15):
                            exit_p = round(p_close * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = "EOD (15:15 PM)"
                            exit_ts = ts
                            break

                    trade["exit_p"] = exit_p
                    trade["exit_time"] = exit_ts
                    trade["exit_reason"] = exit_reason
                    gross = (trade["exit_p"] - trade["entry_p"]) * config.LOT_SIZE
                    trade["gross"] = round(gross, 2)

                    buy_val = trade["entry_p"] * config.LOT_SIZE
                    sell_val = trade["exit_p"] * config.LOT_SIZE
                    turn = buy_val + sell_val
                    brok = 40.0
                    stt = sell_val * 0.001
                    exch = turn * 0.0005
                    gst = (brok + exch) * 0.18
                    trade["charges"] = round(brok + stt + exch + gst, 2)
                    trade["net"] = round(trade["gross"] - trade["charges"], 2)
                feed_label = f"Angel One Real Traded ({contract['symbol']})"

    info["status"] = "TRADE_EXECUTED"
    info["trade_occurred"] = True
    info["entry_time"] = trade["entry_time"].strftime("%H:%M:%S") if hasattr(trade["entry_time"], 'strftime') else str(trade["entry_time"])
    info["exit_time"] = trade["exit_time"].strftime("%H:%M:%S") if hasattr(trade["exit_time"], 'strftime') else str(trade["exit_time"])
    info["opt_type"] = trade["opt_type"]
    info["strike"] = trade["strike"]
    info["entry_p"] = trade["entry_p"]
    info["exit_p"] = trade["exit_p"]
    info["cost"] = trade["cost"]
    info["gross_pnl"] = round(trade["gross"], 2)
    info["charges"] = round(trade["charges"], 2)
    info["net_pnl"] = round(trade["net"], 2)
    info["exit_reason"] = trade["exit_reason"]
    info["data_feed"] = feed_label
    return info

def evaluate_strategy_4(df_full: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    """
    Strategy 4: Decoupled Asymmetric Strangle (DAS - Nifty)
    Delta-Neutral Volatility Compression & Kinetic Velocity Expansion.
    Dynamically scans OTM CE and PE strikes (35-68 pts premium, budget <= Rs 10,000).
    Decoupled Asymmetric Exit:
    - Surging winning leg captured at +50% target
    - Decaying losing leg stopped at -35% or held for reversal
    - Combined stop loss at -15% or 25-minute holding cutoff
    """
    return evaluate_das_for_day(df_full, target_date, api=api)

def evaluate_strategy_5(df_banknifty: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    """
    Strategy 5: BankNIFTY Decoupled Asymmetric Strangle (DAS)
    Delta-Neutral Volatility Compression & Kinetic Expansion on BankNIFTY.
    Instrument: 1 Lot = 30 Qty.
    Fine-tuned Exit: +55% Target on surging leg, -12% SL on decaying leg, -12% Combined SL, 35-min hold.
    Entry Window: 09:45 - 11:00 AM IST (Max 1 trade/day).
    """
    return evaluate_banknifty_das_for_day(df_banknifty, target_date, api=api)

def evaluate_strategy_6(df_banknifty: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    """
    Strategy 6: BankNIFTY Directional Box Breakout (DBB)
    15M Box Breakout (09:15-09:30 IST) with Institutional VWAP confirmation (20-pt buffer).
    Fine-tuned Exit: +40% Target, -20% SL, EOD 15:15 IST.
    Budget <= INR 8,500 (fits INR 10,000 account limit).
    """
    return evaluate_banknifty_box_for_day(df_banknifty, target_date, api=api)

def evaluate_strategy_7(df_banknifty: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    """
    Strategy 7: BankNIFTY Expiry Gamma Squeeze Momentum (GSM)
    Tuesday (1-DTE) & Wednesday (0-DTE) Expiry Sessions Only.
    5M Session Mean + 2.0x ATR(14) Breakout (09:30-14:30 IST), +40% Target, -20% SL, Mean Reversal.
    Budget <= INR 8,500 (fits INR 10,000 account limit).
    """
    return evaluate_banknifty_gamma_for_day(df_banknifty, target_date, api=api)

def evaluate_strategy_8(df_banknifty: pd.DataFrame, target_date: datetime.date, api: SmartConnect = None) -> dict:
    """
    Strategy 8: BankNIFTY Hedged Long Strangle (15M Box)
    Tuesday (1-DTE) & Wednesday (0-DTE) Expiry Sessions with Opening Compression (<0.35%).
    15M Box OTM Strangle at 09:30 IST, +70% Target, -30% Basket SL, 45-min hold.
    Budget <= INR 8,500 (fits INR 10,000 account limit).
    """
    return evaluate_banknifty_strangle_for_day(df_banknifty, target_date, api=api)

def print_portfolio_dashboard(target_date: datetime.date, s1: dict, s2: dict, s3: dict, s4: dict, s5: dict, s6: dict = None, s7: dict = None, s8: dict = None):
    print("\n" + "=" * 95)
    print(f"      MULTI-INDEX QUANT PORTFOLIO DAILY RESULT CHECKER: {target_date.strftime('%A, %d-%b-%Y')}")
    print("=" * 95)

    def print_strategy_box(title, s):
        print(f"[{title}]")
        if s.get("trade_occurred", False):
            p_col = "+" if s.get("net_pnl", 0) > 0 else ""
            n_trades = s.get("num_trades_day", 1)
            t_str = f" ({n_trades} Trades Executed Today)" if n_trades > 1 else ""
            print(f"-> STATUS: TRADE EXECUTED | Net PnL: INR {p_col}{s.get('net_pnl', 0.0):.2f}{t_str}")
            
            all_t = s.get("all_trades") if (s.get("all_trades") and len(s.get("all_trades")) > 1) else [s]
            for idx, tr in enumerate(all_t):
                prefix = f"   Trade {idx+1}: " if len(all_t) > 1 else "   "
                tr_pnl = float(tr.get("net_pnl", tr.get("net", 0.0)))
                p_c = "+" if tr_pnl > 0 else ""
                leg_str = f"BUY {tr.get('strike', '')} ({tr.get('opt_type', tr.get('leg', ''))})"
                print(f"{prefix}{leg_str} @ INR {float(tr.get('entry_p', 0)):.2f} (Capital: INR {float(tr.get('cost', tr.get('capital_used', 0))):.2f})")
                print(f"   Entry: {tr.get('entry_time')} | Exit: {tr.get('exit_time')} @ INR {float(tr.get('exit_p', 0)):.2f} | Reason: {tr.get('exit_reason')}")
                print(f"   Gross PnL: INR {float(tr.get('gross_pnl', tr.get('gross', 0))):.2f} | Brokerage+Govt Charges: INR {float(tr.get('charges', 0)):.2f} | Net: INR {p_c}{tr_pnl:.2f}")
                if "data_feed" in tr:
                    print(f"   Option Data Feed: {tr.get('data_feed')}")
        else:
            print(f"-> STATUS: NO TRADE TODAY ({s.get('status', 'IDLE')})")
            print(f"   Reason: {s.get('reason', '')}")
        print("-" * 95)

    print_strategy_box("STRATEGY 1: HEDGED LONG STRANGLE (15M BOX BREAKOUT - NIFTY)", s1)
    print_strategy_box("STRATEGY 2: 0-DTE / 1-DTE EXPIRY GAMMA SQUEEZE (NIFTY TUESDAY EXPIRY)", s2)
    print_strategy_box("STRATEGY 3: 30-MINUTE STATISTICAL BOX BREAKOUT (DIRECTIONAL ITM_50 - NIFTY)", s3)
    print_strategy_box("STRATEGY 4: DECOUPLED ASYMMETRIC STRANGLE (DAS - NIFTY)", s4)
    print_strategy_box("STRATEGY 5: DECOUPLED ASYMMETRIC STRANGLE (DAS - BANKNIFTY)", s5)
    if s6 is not None: print_strategy_box("STRATEGY 6: DIRECTIONAL BOX BREAKOUT (DBB - BANKNIFTY)", s6)
    if s7 is not None: print_strategy_box("STRATEGY 7: EXPIRY GAMMA SQUEEZE MOMENTUM (GSM - BANKNIFTY)", s7)
    if s8 is not None: print_strategy_box("STRATEGY 8: HEDGED LONG STRANGLE (15M BOX - BANKNIFTY)", s8)

    # Dynamic computation of portfolio table directly from trade_journal.csv
    portfolio_table = []
    tot_trades_all = 0
    tot_wins_all = 0
    tot_net_all = 0.0
    tot_gross_win_all = 0.0
    tot_gross_loss_all = 0.0

    if os.path.exists(TRADE_JOURNAL_CSV):
        try:
            df_j = pd.read_csv(TRADE_JOURNAL_CSV)
        except Exception:
            df_j = pd.DataFrame()
    else:
        df_j = pd.DataFrame()

    strat_list = [
        ("Strategy 1", "Strategy 1: Hedged Strangle (15M Box - NIFTY)"),
        ("Strategy 2", "Strategy 2: Expiry Gamma Squeeze (NIFTY)"),
        ("Strategy 3", "Strategy 3: 30M Directional ITM (NIFTY)"),
        ("Strategy 4", "Strategy 4: Decoupled Strangle (DAS - NIFTY)"),
        ("Strategy 5", "Strategy 5: Decoupled Strangle (DAS - BANKNIFTY)"),
        ("Strategy 6", "Strategy 6: Directional Box Breakout (DBB - BANKNIFTY)"),
        ("Strategy 7", "Strategy 7: Expiry Gamma Squeeze (GSM - BANKNIFTY)"),
        ("Strategy 8", "Strategy 8: Hedged Long Strangle (15M Box - BANKNIFTY)"),
    ]

    for key_pattern, label in strat_list:
        if not df_j.empty and "strategy" in df_j.columns:
            g = df_j[df_j["strategy"].str.contains(key_pattern, na=False, case=False)]
        else:
            g = pd.DataFrame()

        t_count = len(g)
        if t_count > 0:
            wins = len(g[g["net_pnl"] > 0])
            wr = (wins / t_count) * 100.0
            g_win = float(g[g["net_pnl"] > 0]["net_pnl"].sum())
            g_loss = float(abs(g[g["net_pnl"] <= 0]["net_pnl"].sum()))
            pf = round(g_win / g_loss, 2) if g_loss > 0 else (99.0 if g_win > 0 else 0.0)
            net_val = float(g["net_pnl"].sum())
            cum = g["net_pnl"].cumsum()
            peak = cum.cummax()
            dd = float((peak - cum).max()) if not cum.empty else 0.0
            
            tot_trades_all += t_count
            tot_wins_all += wins
            tot_net_all += net_val
            tot_gross_win_all += g_win
            tot_gross_loss_all += g_loss
            
            p_sign = "+" if net_val >= 0 else ""
            portfolio_table.append([
                label,
                f"{t_count} Trades",
                f"{wr:.1f}% Win Rate",
                f"{pf:.2f} PF",
                f"Max DD: INR {dd:,.0f}",
                f"{p_sign}INR {net_val:,.2f}"
            ])
        else:
            portfolio_table.append([label, "0 Trades", "0.0% Win Rate", "0.00 PF", "Max DD: INR 0", "INR 0.00"])

    comb_wr = (tot_wins_all / tot_trades_all * 100.0) if tot_trades_all > 0 else 0.0
    comb_pf = round(tot_gross_win_all / tot_gross_loss_all, 2) if tot_gross_loss_all > 0 else 0.0
    comb_cum = df_j["net_pnl"].cumsum() if not df_j.empty and "net_pnl" in df_j.columns else pd.Series([0.0])
    comb_dd = float((comb_cum.cummax() - comb_cum).max()) if not comb_cum.empty else 0.0
    comb_sign = "+" if tot_net_all >= 0 else ""

    portfolio_table.append([
        "COMBINED 8-STRATEGY MULTI-INDEX PORTFOLIO",
        f"{tot_trades_all} Trades",
        f"{comb_wr:.1f}% Win Rate",
        f"{comb_pf:.2f} PF",
        f"Max DD: INR {comb_dd:,.0f}",
        f"{comb_sign}INR {tot_net_all:,.2f}"
    ])

    print("\n" + " " * 20 + "--- CUMULATIVE INR 10,000 MULTI-INDEX PORTFOLIO SUMMARY ---")
    print(tabulate(portfolio_table, headers=["Strategy", "Trades", "Win Rate", "Profit Factor", "Risk (Max DD)", "Net Return"], tablefmt="grid"))
    curr_bal = 10000.0 + tot_net_all
    ret_pct = (tot_net_all / 10000.0) * 100.0
    r_sign = "+" if ret_pct >= 0 else ""
    print(f"\nAccount Capital: INR 10,000.00  -->  Current Multi-Index Balance: INR {curr_bal:,.2f} ({r_sign}{ret_pct:.1f}% Growth)")
    print("=" * 95 + "\n")

TRADE_JOURNAL_CSV = os.path.join(BASE_DIR, "data", "trade_journal.csv")
TRADE_JOURNAL_MD = os.path.join(BASE_DIR, "data", "trade_journal.md")

def log_trade_to_journal(target_date, strategy_name: str, trade_res: dict):
    from src.utils.journal_manager import log_trade_to_journal as _log_trade
    _log_trade(target_date, strategy_name, trade_res)

def run():
    parser = argparse.ArgumentParser(description="Multi-Strategy Daily Result Checker")
    parser.add_argument("--date", type=str, default=None, help="Date to check (YYYY-MM-DD).")
    parser.add_argument("--telegram", action="store_true", default=True, help="Sync and push complete daily result summary to Telegram (default: True).")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram push.")
    args = parser.parse_args()

    from src.utils.journal_manager import (
        auto_catchup_missing_trading_days,
        mark_day_as_audited,
        refresh_markdown_journal
    )
    from src.utils.git_sync import git_sync_push, git_sync_pull

    # 1. First ensure local state is synchronized with remote live trader commits
    git_sync_pull()

    api = get_smart_api()
    if api:
        print("[CONNECTED] Angel One SmartAPI connected successfully.")
        import time as time_lib
        df_all = sync_and_update_candles(api, CANDLE_FILE, "99926000")
        time_lib.sleep(1.5)  # Enforce 1.5s pause to prevent Angel One rate limit between instruments
        df_bn = sync_and_update_candles(api, BANKNIFTY_CANDLE_FILE, "99926009")
    else:
        print("[OFFLINE] Running with local candle dataset.")
        df_all = pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame()
        df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame()

    if df_all.empty:
        print("[ERROR] No Nifty data available.")
        return

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.tz_localize(None)
    if not df_bn.empty:
        df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)

    if args.date:
        target_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = df_all["timestamp"].dt.date.max()
        prior_date = target_date - datetime.timedelta(days=1)
        auto_catchup_missing_trading_days(api, df_all, df_bn, up_to_date=prior_date)

    # Check if live trades were already recorded today in trade_journal.csv
    today_live_trades = []
    if os.path.exists(TRADE_JOURNAL_CSV):
        try:
            df_j = pd.read_csv(TRADE_JOURNAL_CSV)
            if not df_j.empty and "date" in df_j.columns:
                today_live_trades = df_j[df_j["date"].astype(str) == str(target_date)].to_dict("records")
        except Exception:
            today_live_trades = []

    def get_strategy_evaluation(strat_key, strat_full_name, eval_func, df_candles, is_bn=False):
        # Check if actual live trades occurred today for this strategy
        strat_trades = [t for t in today_live_trades if strat_key.lower() in str(t.get("strategy", "")).lower()]
        if strat_trades:
            net_tot = round(sum(float(t.get("net_pnl", 0.0)) for t in strat_trades), 2)
            gross_tot = round(sum(float(t.get("gross_pnl", 0.0)) for t in strat_trades), 2)
            charges_tot = round(sum(float(t.get("charges", 0.0)) for t in strat_trades), 2)
            cost_tot = round(sum(float(t.get("capital_used", 0.0)) for t in strat_trades), 2)
            reasons = " | ".join(str(t.get("exit_reason", "")) for t in strat_trades)
            return {
                "trade_occurred": True,
                "status": "TRADE_EXECUTED",
                "strategy": strat_full_name,
                "num_trades_day": len(strat_trades),
                "entry_time": strat_trades[0].get("entry_time"),
                "exit_time": strat_trades[-1].get("exit_time"),
                "strike": strat_trades[0].get("strike"),
                "opt_type": strat_trades[0].get("leg", "CE+PE"),
                "entry_p": strat_trades[0].get("entry_p"),
                "exit_p": strat_trades[-1].get("exit_p"),
                "cost": cost_tot,
                "gross_pnl": gross_tot,
                "charges": charges_tot,
                "net_pnl": net_tot,
                "exit_reason": reasons,
                "data_feed": strat_trades[0].get("data_feed", "Angel One Real Traded"),
                "all_trades": strat_trades
            }
        # Fallback to simulation to provide explanation why no trade occurred
        if df_candles.empty:
            return {"status": "NO_DATA", "trade_occurred": False, "reason": "No market data available"}
        return eval_func(df_candles, target_date, api=api)

    s1_res = get_strategy_evaluation("Strategy 1", "Strategy 1: Hedged Strangle", evaluate_strategy_1, df_all)
    s2_res = get_strategy_evaluation("Strategy 2", "Strategy 2: Expiry Gamma Squeeze", evaluate_strategy_2, df_all)
    s3_res = get_strategy_evaluation("Strategy 3", "Strategy 3: 30M Directional ITM", evaluate_strategy_3, df_all)
    s4_res = get_strategy_evaluation("Strategy 4", "Strategy 4: Decoupled Asymmetric Strangle", evaluate_strategy_4, df_all)
    s5_res = get_strategy_evaluation("Strategy 5", "Strategy 5: BankNIFTY Decoupled Strangle (DAS)", evaluate_strategy_5, df_bn, is_bn=True)
    s6_res = get_strategy_evaluation("Strategy 6", "Strategy 6: BankNIFTY Directional Box Breakout (DBB)", evaluate_strategy_6, df_bn, is_bn=True)
    s7_res = get_strategy_evaluation("Strategy 7", "Strategy 7: BankNIFTY Expiry Gamma Squeeze (GSM)", evaluate_strategy_7, df_bn, is_bn=True)
    s8_res = get_strategy_evaluation("Strategy 8", "Strategy 8: BankNIFTY Hedged Long Strangle", evaluate_strategy_8, df_bn, is_bn=True)

    print_portfolio_dashboard(target_date, s1_res, s2_res, s3_res, s4_res, s5_res, s6_res, s7_res, s8_res)

    # Record any unrecorded simulation trades only if no live trades existed today
    if not today_live_trades:
        for s_name, s_res in [
            ("Strategy 1: Hedged Strangle", s1_res),
            ("Strategy 2: Expiry Gamma Squeeze", s2_res),
            ("Strategy 3: 30M Directional ITM", s3_res),
            ("Strategy 4: Decoupled Asymmetric Strangle", s4_res),
            ("Strategy 5: BankNIFTY Decoupled Strangle (DAS)", s5_res),
            ("Strategy 6: BankNIFTY Directional Box Breakout (DBB)", s6_res),
            ("Strategy 7: BankNIFTY Expiry Gamma Squeeze (GSM)", s7_res),
            ("Strategy 8: BankNIFTY Hedged Long Strangle", s8_res)
        ]:
            if s_res.get("trade_occurred", False):
                if "all_trades" in s_res and len(s_res["all_trades"]) > 1:
                    for sub_tr in s_res["all_trades"]:
                        log_trade_to_journal(target_date, s_name, sub_tr)
                else:
                    log_trade_to_journal(target_date, s_name, s_res)

    # Mark day as audited
    mark_day_as_audited(str(target_date))
    refresh_markdown_journal()

    # Automatically push journal update to GitHub
    git_sync_push(f"Daily Result Audit: {target_date.strftime('%d-%b-%Y')}")

    should_push_telegram = args.telegram and not args.no_telegram
    if should_push_telegram:
        tot_pnl = 0.0
        daily_trades_count = 0
        daily_net_pnl = 0.0

        if os.path.exists(TRADE_JOURNAL_CSV):
            try:
                df_j = pd.read_csv(TRADE_JOURNAL_CSV)
                if not df_j.empty and "net_pnl" in df_j.columns:
                    tot_pnl = round(float(df_j["net_pnl"].sum()), 2)
                    today_rows = df_j[df_j["date"].astype(str) == str(target_date)]
                    daily_trades_count = len(today_rows)
                    daily_net_pnl = round(float(today_rows["net_pnl"].sum()), 2)
            except Exception:
                pass

        try:
            from src.notifications.notifier import (
                send_daily_summary_alert,
                is_alert_already_sent,
                mark_alert_as_sent
            )
            summary_key = f"SUMMARY:{target_date}"
            if not is_alert_already_sent(summary_key):
                print(f"[TELEGRAM] Pushing daily portfolio summary to Telegram for {target_date.strftime('%d-%b-%Y')}...")
                send_daily_summary_alert(
                    date_str=target_date.strftime("%d-%b-%Y"),
                    trades_count=daily_trades_count,
                    daily_pnl=round(daily_net_pnl, 2),
                    total_pnl=tot_pnl,
                    current_balance=round(10000.0 + tot_pnl, 2)
                )
                mark_alert_as_sent(summary_key)
                print("[TELEGRAM] Daily summary delivered successfully!")
            else:
                print(f"[TELEGRAM] Daily summary for {target_date} was already sent today. Skipping duplicate.")
        except Exception as e:
            print(f"[WARN] Failed to push Telegram sync: {e}")

if __name__ == "__main__":
    run()
