"""
Live Market Paper Trading Bot for Nifty Options (Quad-Strategy Unified Engine).
Executes and monitors all 4 quantitative strategies concurrently:
1. Strategy 1: Hedged Long Strangle (15M Box Breakout, Uncoupled Leg SL -20%, Target +75%)
2. Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze (Monday & Tuesday Expiry Breakout, Target +80%)
3. Strategy 3: 30M Statistical Directional ITM Breakout (Trailing Peak Exit, Target +80%)
4. Strategy 4: Decoupled Asymmetric Strangle (DAS - Delta-Neutral Volatility Compression & Kinetic Expansion)

Connects to Angel One SmartAPI to track Nifty Spot and option candles in real-time.
Pushes instant alerts to Telegram for every entry, target, stop, and EOD daily report.
Logs all paper trades to data/trade_journal.csv and data/trade_journal.md.
"""

import os
import sys
import time
import math
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta, time as dtime
from dotenv import load_dotenv

# Try importing SmartApi
try:
    from SmartApi import SmartConnect
    import pyotp
except ImportError:
    SmartConnect = None

from src.features.greeks import black_scholes_price
from config import config
from src.decoupled_strangle.das_signals import (
    select_affordable_strikes,
    check_compression_expansion,
    is_trade_window_valid
)
from src.decoupled_strangle.das_engine import load_cached_real_options
from src.notifications.notifier import (
    send_telegram_alert,
    send_trade_entry_alert,
    send_trade_exit_alert,
    send_daily_summary_alert
)
from daily_result_checker import (
    evaluate_strategy_1,
    evaluate_strategy_2,
    evaluate_strategy_3,
    evaluate_strategy_4,
    evaluate_strategy_5,
    sync_and_update_candles,
    CANDLE_FILE,
    BANKNIFTY_CANDLE_FILE
)
from src.utils.journal_manager import (
    load_active_positions,
    save_active_positions,
    log_trade_to_journal,
    refresh_markdown_journal,
    auto_catchup_missing_trading_days,
    mark_day_as_audited
)
from src.utils.git_sync import git_sync_push
from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles
from src.banknifty.banknifty_das import select_banknifty_affordable_strikes, get_banknifty_dte
from run_strangle_grid_exhaustive import compute_calendar_dte

load_dotenv()

def get_ist_now() -> datetime:
    """Returns the current datetime in Indian Standard Time (IST, UTC+5:30)."""
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

class LiveQuadPaperTrader:
    """
    Unified Quad-Strategy Paper Trading Engine.
    Executes and monitors all 4 strategies during live market hours.
    """
    def __init__(self, target_strategy: str = "all"):
        self.target_strategy = target_strategy.lower() # 'all', 'das', 'bndas', 'strangle', 'gamma', 'directional'
        self.lot_size = config.LOT_SIZE
        self.bn_lot_size = 30
        self.capital = config.INITIAL_CAPITAL

        self.smart_api = None
        self.auth_token = None
        self.feed_token = None
        self.opt_data = load_cached_real_options()

        # Real-time state machine tracking for live positions
        self.active_positions = {
            "s1": None,
            "s2": None,
            "s3": None,
            "s4": None,
            "s5": None,
        }
        self.cooldown_until = {
            "s1": None,
            "s2": None,
            "s3": None,
            "s4": None,
            "s5": None,
        }
        self.daily_trade_count = {
            "s1": 0,
            "s2": 0,
            "s3": 0,
            "s4": 0,
            "s5": 0,
        }

        # Output log
        self.log_file = "data/live_paper_trades.csv"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=[
                "date", "strategy", "entry_time", "exit_time", "spot_entry", "ce_strike", "pe_strike",
                "ce_entry", "ce_exit", "pe_entry", "pe_exit", "gross_pnl", "charges", "net_pnl", "exit_reason"
            ]).to_csv(self.log_file, index=False)

    def connect_angel_one(self) -> bool:
        api_key = os.getenv("ANGEL_API_KEY")
        client_code = os.getenv("ANGEL_CLIENT_CODE")
        pin = os.getenv("ANGEL_PASSWORD") or os.getenv("ANGEL_PIN") or os.getenv("ANGEI_PASSWORD")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET")

        if not all([api_key, client_code, pin, totp_secret]):
            print("[INFO] Angel One credentials not fully configured in .env. Running in offline/simulation mode.")
            return False

        import time as time_lib
        for attempt in range(3):
            try:
                self.smart_api = SmartConnect(api_key=api_key, timeout=30)
                totp = pyotp.TOTP(totp_secret).now()
                data = self.smart_api.generateSession(client_code, pin, totp)
                if data and data.get("status"):
                    print(f"[SUCCESS] Connected to Angel One SmartAPI for Multi-Index Live Trading. (Client: {client_code})")
                    return True
                elif data and "exceeding access rate" in str(data):
                    time_lib.sleep(1.5)
                else:
                    if attempt == 2:
                        print(f"[WARN] Angel One login failed: {data.get('message', 'Unknown error')}")
            except Exception as e:
                if attempt < 2:
                    time_lib.sleep(1.5)
                else:
                    print(f"[ERROR] Angel One connection error: {e}")
        return False

    def get_nifty_spot_price(self) -> float:
        """Fetches live Nifty 50 spot price from Angel One LTP."""
        if self.smart_api:
            try:
                ltp_data = self.smart_api.ltpData("NSE", "Nifty 50", "99926000")
                if ltp_data and ltp_data.get("status"):
                    return float(ltp_data["data"]["ltp"])
            except Exception:
                pass
        return 24800.0

    def get_banknifty_spot_price(self) -> float:
        """Fetches live BankNIFTY spot price from Angel One LTP."""
        if self.smart_api:
            try:
                ltp_data = self.smart_api.ltpData("NSE", "Nifty Bank", "99926009")
                if ltp_data and ltp_data.get("status"):
                    return float(ltp_data["data"]["ltp"])
            except Exception:
                pass
        return 55000.0

    def is_market_open_now(self) -> bool:
        """Checks if current time is within live NSE market hours (Mon-Fri 09:15 - 15:25 IST)."""
        now = get_ist_now()
        if now.weekday() in [5, 6]:  # Saturday or Sunday
            return False
        return dtime(9, 15) <= now.time() <= dtime(15, 25)

    def wait_for_market_open(self):
        """If started before 09:15 AM IST on a weekday, waits until 09:15:00 IST."""
        now = get_ist_now()
        if now.weekday() in [5, 6]:
            return False
        if now.time() < dtime(9, 15):
            open_time = datetime(now.year, now.month, now.day, 9, 15, 0)
            now_naive = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
            wait_secs = (open_time - now_naive).total_seconds()
            if 0 < wait_secs <= 3600:
                print(f"\n[PRE-MARKET STANDBY] Bot initialized early at {now.strftime('%I:%M:%S %p')} IST.")
                print(f"Waiting {int(wait_secs)} seconds until NSE market opens at 09:15:00 AM IST...")
                send_telegram_alert(
                    f"⏳ *Option Trading Bot Initialized Early (Pre-Market)*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⏰ *Current Time:* {now.strftime('%I:%M:%S %p')} IST\n"
                    f"📅 *Date:* {now.strftime('%A, %d-%b-%Y')}\n\n"
                    f"Market opens at 09:15:00 AM IST. Bot is standing by and will begin active tracking across Nifty & BankNifty at the opening bell! 🔔"
                )
                time.sleep(max(1.0, wait_secs))
                return True
        return False

    def save_active_positions(self):
        """Persists in-flight open positions to data/active_positions.json for crash resilience."""
        data_to_save = {}
        for s_key, pos in self.active_positions.items():
            if pos is not None:
                p_copy = dict(pos)
                if "entry_time" in p_copy and hasattr(p_copy["entry_time"], "isoformat"):
                    p_copy["entry_time"] = p_copy["entry_time"].isoformat()
                # remove non-serializable objects
                if "c_ce" in p_copy and isinstance(p_copy["c_ce"], dict):
                    p_copy["c_ce"] = {k: str(v) for k, v in p_copy["c_ce"].items() if k in ["token", "symbol"]}
                if "c_pe" in p_copy and isinstance(p_copy["c_pe"], dict):
                    p_copy["c_pe"] = {k: str(v) for k, v in p_copy["c_pe"].items() if k in ["token", "symbol"]}
                if "contract" in p_copy and isinstance(p_copy["contract"], dict):
                    p_copy["contract"] = {k: str(v) for k, v in p_copy["contract"].items() if k in ["token", "symbol"]}
                data_to_save[s_key] = p_copy
        os.makedirs("data", exist_ok=True)
        try:
            with open("data/active_positions.json", "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save active positions: {e}")

    def on_trade_entry(self, s_key: str, s_name: str, pos: dict):
        """Called immediately upon trade entry: updates active positions, journal, and pushes to git."""
        pos["strategy_name"] = s_name
        self.active_positions[s_key] = pos
        self.daily_trade_count[s_key] += 1
        self.save_active_positions()
        refresh_markdown_journal()
        sym_str = pos.get("strike") or pos.get("symbol") or f"{pos.get('ce_symbol', '')} + {pos.get('pe_symbol', '')}"
        print(f"[LIVE JOURNAL] [ENTRY] Active trade logged in trade_journal.md for {s_name} ({sym_str})")
        git_sync_push(f"Auto-Journal: 🟢 Entry {s_name} ({sym_str})")

    def on_trade_exit(self, s_key: str, s_name: str, tr_log: dict):
        """Called immediately upon trade exit: clears active position, appends to journal, and pushes to git."""
        self.active_positions[s_key] = None
        self.save_active_positions()
        log_trade_to_journal(tr_log["date"], s_name, tr_log)
        refresh_markdown_journal()
        net_val = float(tr_log.get("net_pnl", 0.0))
        print(f"[LIVE JOURNAL] [EXIT] Trade exit logged in trade_journal.md for {s_name} [PnL: INR {net_val:+.2f}]")
        git_sync_push(f"Auto-Journal: 🔴 Exit {s_name} [PnL: INR {net_val:+.2f}]")

    def load_active_positions(self, today_date):
        """Loads and restores any in-flight open positions for today from data/active_positions.json."""
        pos_file = "data/active_positions.json"
        if os.path.exists(pos_file):
            try:
                with open(pos_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for s_key, p in data.items():
                    if s_key in self.active_positions and p:
                        if "entry_time" in p and isinstance(p["entry_time"], str):
                            p["entry_time"] = datetime.fromisoformat(p["entry_time"])
                        if p["entry_time"].date() == today_date:
                            self.active_positions[s_key] = p
                            print(f"[RECOVERY] Restored in-flight position for {s_key} entered at {p['entry_time']}")
            except Exception as e:
                print(f"[WARN] Failed to load active positions: {e}")

    def init_today_state(self, today_date):
        """Initializes position tracking and synchronizes with existing journal entries today."""
        self.active_positions = {
            "s1": None,
            "s2": None,
            "s3": None,
            "s4": None,
            "s5": None,
        }
        self.cooldown_until = {
            "s1": None,
            "s2": None,
            "s3": None,
            "s4": None,
            "s5": None,
        }
        self.daily_trade_count = {
            "s1": 0,
            "s2": 0,
            "s3": 0,
            "s4": 0,
            "s5": 0,
        }
        if os.path.exists("data/trade_journal.csv"):
            try:
                df_j = pd.read_csv("data/trade_journal.csv")
                if not df_j.empty and "date" in df_j.columns and "strategy" in df_j.columns:
                    today_j = df_j[df_j["date"].astype(str) == str(today_date)]
                    for _, row in today_j.iterrows():
                        s_name = str(row["strategy"])
                        if "Strategy 1" in s_name: self.daily_trade_count["s1"] += 1
                        if "Strategy 2" in s_name: self.daily_trade_count["s2"] += 1
                        if "Strategy 3" in s_name: self.daily_trade_count["s3"] += 1
                        if "Strategy 4" in s_name: self.daily_trade_count["s4"] += 1
                        if "Strategy 5" in s_name: self.daily_trade_count["s5"] += 1
            except Exception:
                pass
        self.load_active_positions(today_date)

    def process_live_strategy_5(self, df_bn: pd.DataFrame, today_date, ist_now, bn_spot: float):
        """Real-time live position tracking and entry engine for Strategy 5 (BankNIFTY DAS)."""
        pos = self.active_positions["s5"]

        # 1. POSITION IS OPEN -> MONITOR AND EXIT IN REAL TIME
        if pos is not None:
            elapsed_mins = (ist_now - pos["entry_time"]).total_seconds() / 60.0
            p_c = pos["ce_entry"]
            p_p = pos["pe_entry"]
            high_c, low_c = p_c, p_c
            high_p, low_p = p_p, p_p

            if self.smart_api and pos.get("c_ce") and pos.get("c_pe"):
                df_c = fetch_real_option_candles(self.smart_api, pos["c_ce"], pos["entry_time"], ist_now)
                df_p = fetch_real_option_candles(self.smart_api, pos["c_pe"], pos["entry_time"], ist_now)
                if not df_c.empty:
                    p_c = float(df_c.iloc[-1]["close"])
                    high_c = float(df_c["high"].max())
                    low_c = float(df_c["low"].min())
                if not df_p.empty:
                    p_p = float(df_p.iloc[-1]["close"])
                    high_p = float(df_p["high"].max())
                    low_p = float(df_p["low"].min())
            else:
                dte = get_banknifty_dte(ist_now, today_date)
                p_c = black_scholes_price(bn_spot, pos["ce_strike"], dte, config.RISK_FREE_RATE, 0.16, "CE")
                p_p = black_scholes_price(bn_spot, pos["pe_strike"], dte, config.RISK_FREE_RATE, 0.16, "PE")
                high_c, low_c = p_c, p_c
                high_p, low_p = p_p, p_p

            if not pos["ce_exited"]:
                if high_c >= pos["ce_tgt"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = "CE Target (+100%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 5: BankNIFTY Decoupled Strangle (DAS)`\n"
                            f"🟢 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (+100% Target)\n"
                            f"🛡️ *PUT Leg Status:* Still active with capital protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )
                elif low_c <= pos["ce_sl"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = "CE SL (-15%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 5: BankNIFTY Decoupled Strangle (DAS)`\n"
                            f"🔴 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (-15% SL)\n"
                            f"🛡️ *PUT Leg Status:* Still active under decoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )

            if not pos["pe_exited"]:
                if high_p >= pos["pe_tgt"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = "PE Target (+100%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 5: BankNIFTY Decoupled Strangle (DAS)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (+100% Target)\n"
                            f"🛡️ *CALL Leg Status:* Still active with capital protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )
                elif low_p <= pos["pe_sl"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = "PE SL (-15%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 5: BankNIFTY Decoupled Strangle (DAS)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (-15% SL)\n"
                            f"🛡️ *CALL Leg Status:* Still active under decoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )

            exit_c = pos["ce_exit_p"] if pos["ce_exited"] else round(p_c * (1.0 - config.SLIPPAGE_PCT), 2)
            exit_p = pos["pe_exit_p"] if pos["pe_exited"] else round(p_p * (1.0 - config.SLIPPAGE_PCT), 2)
            cur_tot = exit_c + exit_p
            comb_ret = (cur_tot - pos["entry_tot"]) / max(0.1, pos["entry_tot"])
            is_time_up = (elapsed_mins >= 45.0) or (ist_now.time() >= dtime(15, 15))
            is_comb_stop = comb_ret <= -0.15

            if (pos["ce_exited"] and pos["pe_exited"]) or is_time_up or is_comb_stop:
                if is_comb_stop:
                    exit_reason = "Combined Capital SL (-15%)"
                elif pos["ce_exited"] and pos["pe_exited"]:
                    exit_reason = f"{pos['ce_reason']} & {pos['pe_reason']}"
                elif is_time_up:
                    active_reasons = []
                    if pos["ce_exited"]: active_reasons.append(pos["ce_reason"])
                    if pos["pe_exited"]: active_reasons.append(pos["pe_reason"])
                    exit_reason = f"Time Cutoff ({' & '.join(active_reasons)})" if active_reasons else "Time Exit (45m)"
                elif pos["ce_exited"] or pos["pe_exited"]:
                    exit_reason = pos["ce_reason"] if pos["ce_exited"] else pos["pe_reason"]
                else:
                    exit_reason = "Time Exit (45m)"

                gross = ((exit_c - pos["ce_entry"]) + (exit_p - pos["pe_entry"])) * self.bn_lot_size
                buy_v = pos["entry_tot"] * self.bn_lot_size
                sell_v = cur_tot * self.bn_lot_size
                turn = buy_v + sell_v
                stt = sell_v * 0.001
                exch = turn * 0.0005
                charges = round(80.0 + stt + exch + (80.0 + exch) * 0.18, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 5: BankNIFTY Decoupled Strangle (DAS)",
                    "opt_type": "CE+PE",
                    "strike": f"{pos['ce_symbol']} + {pos['pe_symbol']}",
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_tot"],
                    "exit_p": round(cur_tot, 2),
                    "cost": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                self.cooldown_until["s5"] = ist_now + timedelta(minutes=30)
                self.on_trade_exit("s5", "Strategy 5: BankNIFTY Decoupled Strangle (DAS)", tr_log)
                send_trade_exit_alert(
                    strategy="Strategy 5: BankNIFTY Decoupled Strangle (DAS)",
                    exit_reason=exit_reason,
                    gross_pnl=round(gross, 2),
                    charges=charges,
                    net_pnl=net,
                    details=f"Entry: ₹{pos['entry_tot']:.2f} -> Exit: ₹{cur_tot:.2f} ({pos['ce_symbol']} + {pos['pe_symbol']})",
                    time_str=ist_now.strftime("%H:%M:%S")
                )

        # 2. POSITION IS NOT OPEN -> CHECK FOR LIVE BREAKOUT ENTRY
        else:
            if self.cooldown_until["s5"] and ist_now < self.cooldown_until["s5"]:
                return
            if ist_now.time() < dtime(9, 45) or ist_now.time() > dtime(13, 30):
                return
            if dtime(11, 30) <= ist_now.time() <= dtime(12, 45):
                return
            if df_bn.empty or len(df_bn) < 15:
                return

            recent_bars = df_bn.iloc[-15:]
            std_val = float(recent_bars["close"].std())
            vel_val = float(df_bn["close"].iloc[-1] - df_bn["close"].iloc[-4]) if len(df_bn) >= 4 else 0.0

            if std_val < 55.0 and abs(vel_val) >= 70.0:
                dte = get_banknifty_dte(ist_now, today_date)
                k_ce, k_pe, p_ce, p_pe, tot_capital, c_ce, c_pe = select_banknifty_affordable_strikes(
                    bn_spot, dte, today_date, api=self.smart_api, max_budget=10000.0
                )
                real_ce = p_ce
                real_pe = p_pe
                feed_label = "Mathematical (BSM)"

                if self.smart_api and c_ce and c_pe:
                    df_ce = fetch_real_option_candles(self.smart_api, c_ce, ist_now - timedelta(minutes=10), ist_now)
                    df_pe = fetch_real_option_candles(self.smart_api, c_pe, ist_now - timedelta(minutes=10), ist_now)
                    if not df_ce.empty and not df_pe.empty:
                        real_ce = round(float(df_ce.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        real_pe = round(float(df_pe.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        feed_label = f"Angel One Real Traded ({c_ce['symbol']} + {c_pe['symbol']})"

                entry_tot = round(real_ce + real_pe, 2)
                cost = round(entry_tot * self.bn_lot_size, 2)
                if cost > 10000.0:
                    return

                ce_sym = c_ce["symbol"] if c_ce else f"{k_ce}CE"
                pe_sym = c_pe["symbol"] if c_pe else f"{k_pe}PE"
                curr_bar = df_bn.iloc[-1]
                candle_ts = pd.to_datetime(curr_bar["timestamp"]).to_pydatetime() if "timestamp" in curr_bar else ist_now

                pos_data = {
                    "entry_time": candle_ts,
                    "spot_entry": bn_spot,
                    "c_ce": c_ce,
                    "c_pe": c_pe,
                    "ce_strike": k_ce,
                    "pe_strike": k_pe,
                    "ce_symbol": ce_sym,
                    "pe_symbol": pe_sym,
                    "ce_entry": real_ce,
                    "pe_entry": real_pe,
                    "entry_tot": entry_tot,
                    "cost": cost,
                    "ce_tgt": round(real_ce * 2.0, 2),
                    "ce_sl": round(real_ce * 0.85, 2),
                    "pe_tgt": round(real_pe * 2.0, 2),
                    "pe_sl": round(real_pe * 0.85, 2),
                    "ce_exited": False,
                    "pe_exited": False,
                    "ce_exit_p": None,
                    "pe_exit_p": None,
                    "ce_reason": "",
                    "pe_reason": "",
                    "data_feed": feed_label
                }
                self.on_trade_entry("s5", "Strategy 5: BankNIFTY Decoupled Strangle (DAS)", pos_data)
                send_trade_entry_alert(
                    strategy="Strategy 5: BankNIFTY Decoupled Strangle (DAS)",
                    spot=bn_spot,
                    ce_str=ce_sym,
                    pe_str=pe_sym,
                    ce_p=real_ce,
                    pe_p=real_pe,
                    tot_cost=cost,
                    win_target_pct=1.00,
                    lose_stop_pct=0.15,
                    max_hold_mins=45,
                    time_str=candle_ts.strftime("%H:%M:%S"),
                    qty=self.bn_lot_size
                )

    def process_live_strategy_4(self, df_all: pd.DataFrame, today_date, ist_now, spot: float):
        """Real-time live position tracking and entry engine for Strategy 4 (Nifty DAS)."""
        pos = self.active_positions["s4"]

        if pos is not None:
            elapsed_mins = (ist_now - pos["entry_time"]).total_seconds() / 60.0
            p_c = pos["ce_entry"]
            p_p = pos["pe_entry"]
            high_c, low_c = p_c, p_c
            high_p, low_p = p_p, p_p

            if self.smart_api and pos.get("c_ce") and pos.get("c_pe"):
                df_c = fetch_real_option_candles(self.smart_api, pos["c_ce"], pos["entry_time"], ist_now)
                df_p = fetch_real_option_candles(self.smart_api, pos["c_pe"], pos["entry_time"], ist_now)
                if not df_c.empty:
                    p_c = float(df_c.iloc[-1]["close"])
                    high_c = float(df_c["high"].max())
                    low_c = float(df_c["low"].min())
                if not df_p.empty:
                    p_p = float(df_p.iloc[-1]["close"])
                    high_p = float(df_p["high"].max())
                    low_p = float(df_p["low"].min())
            else:
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                p_c = black_scholes_price(spot, pos["ce_strike"], t_exp_years, config.RISK_FREE_RATE, 0.135, "CE")
                p_p = black_scholes_price(spot, pos["pe_strike"], t_exp_years, config.RISK_FREE_RATE, 0.135, "PE")
                high_c, low_c = p_c, p_c
                high_p, low_p = p_p, p_p

            if not pos["ce_exited"]:
                if high_c >= pos["ce_tgt"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = "CE Target (+50%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 4: Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🟢 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (+50% Target)\n"
                            f"🛡️ *PUT Leg Status:* Still active with capital protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )
                elif low_c <= pos["ce_sl"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = "CE SL (-35%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 4: Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🔴 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (-35% SL)\n"
                            f"🛡️ *PUT Leg Status:* Still active under decoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )

            if not pos["pe_exited"]:
                if high_p >= pos["pe_tgt"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = "PE Target (+50%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 4: Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (+50% Target)\n"
                            f"🛡️ *CALL Leg Status:* Still active with capital protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )
                elif low_p <= pos["pe_sl"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = "PE SL (-35%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 4: Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (-35% SL)\n"
                            f"🛡️ *CALL Leg Status:* Still active under decoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )

            exit_c = pos["ce_exit_p"] if pos["ce_exited"] else round(p_c * (1.0 - config.SLIPPAGE_PCT), 2)
            exit_p = pos["pe_exit_p"] if pos["pe_exited"] else round(p_p * (1.0 - config.SLIPPAGE_PCT), 2)
            cur_tot = exit_c + exit_p
            comb_ret = (cur_tot - pos["entry_tot"]) / max(0.1, pos["entry_tot"])
            is_time_up = (elapsed_mins >= config.DAS_MAX_HOLD_MINS) or (ist_now.time() >= dtime(15, 15))
            is_comb_stop = comb_ret <= -config.DAS_COMBINED_STOP_PCT

            if (pos["ce_exited"] and pos["pe_exited"]) or is_time_up or is_comb_stop:
                if is_comb_stop:
                    exit_reason = "Combined Capital SL (-15%)"
                elif pos["ce_exited"] and pos["pe_exited"]:
                    exit_reason = f"{pos['ce_reason']} & {pos['pe_reason']}"
                elif is_time_up:
                    active_reasons = []
                    if pos["ce_exited"]: active_reasons.append(pos["ce_reason"])
                    if pos["pe_exited"]: active_reasons.append(pos["pe_reason"])
                    exit_reason = f"Time Cutoff ({' & '.join(active_reasons)})" if active_reasons else f"Max Hold ({config.DAS_MAX_HOLD_MINS}m)"
                elif pos["ce_exited"] or pos["pe_exited"]:
                    exit_reason = pos["ce_reason"] if pos["ce_exited"] else pos["pe_reason"]
                else:
                    exit_reason = f"Max Hold ({config.DAS_MAX_HOLD_MINS}m)"

                gross = ((exit_c - pos["ce_entry"]) + (exit_p - pos["pe_entry"])) * self.lot_size
                buy_v = pos["entry_tot"] * self.lot_size
                sell_v = cur_tot * self.lot_size
                turn = buy_v + sell_v
                stt = sell_v * 0.001
                exch = turn * 0.0005
                charges = round(80.0 + stt + exch + (80.0 + exch) * 0.18, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 4: Decoupled Asymmetric Strangle",
                    "opt_type": "CE+PE",
                    "strike": f"{pos['ce_symbol']} + {pos['pe_symbol']}",
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_tot"],
                    "exit_p": round(cur_tot, 2),
                    "cost": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                self.cooldown_until["s4"] = ist_now + timedelta(minutes=config.DAS_COOLDOWN_BARS)
                self.on_trade_exit("s4", "Strategy 4: Decoupled Asymmetric Strangle", tr_log)
                send_trade_exit_alert(
                    strategy="Strategy 4: Decoupled Asymmetric Strangle (DAS)",
                    exit_reason=exit_reason,
                    gross_pnl=round(gross, 2),
                    charges=charges,
                    net_pnl=net,
                    details=f"Entry: ₹{pos['entry_tot']:.2f} -> Exit: ₹{cur_tot:.2f} ({pos['ce_symbol']} + {pos['pe_symbol']})",
                    time_str=ist_now.strftime("%H:%M:%S")
                )

        else:
            if self.cooldown_until["s4"] and ist_now < self.cooldown_until["s4"]:
                return
            if not is_trade_window_valid(ist_now.time()):
                return
            if df_all.empty:
                return

            day_bars = df_all[df_all["timestamp"].dt.date == today_date]
            if day_bars.empty or len(day_bars) < 15:
                return

            recent_bars = day_bars.iloc[-15:]
            std_val = float(recent_bars["close"].std())
            vel_val = float(day_bars["close"].iloc[-1] - day_bars["close"].iloc[-4]) if len(day_bars) >= 4 else 0.0

            if std_val < 5.0 and abs(vel_val) >= 3.0:
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                strike_res = select_affordable_strikes(spot, ist_now, T_years=t_exp_years, min_prem=35.0, max_prem=68.0, max_budget=10000.0)
                if not strike_res:
                    return

                ce_k, pe_k, p_ce, p_pe, _ = strike_res
                c_ce = get_active_option_contract(today_date, ce_k, "CE", underlying="NIFTY") if self.smart_api else None
                c_pe = get_active_option_contract(today_date, pe_k, "PE", underlying="NIFTY") if self.smart_api else None

                real_ce = p_ce
                real_pe = p_pe
                feed_label = "Mathematical (BSM)"

                if self.smart_api and c_ce and c_pe:
                    df_ce = fetch_real_option_candles(self.smart_api, c_ce, ist_now - timedelta(minutes=10), ist_now)
                    df_pe = fetch_real_option_candles(self.smart_api, c_pe, ist_now - timedelta(minutes=10), ist_now)
                    if not df_ce.empty and not df_pe.empty:
                        real_ce = round(float(df_ce.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        real_pe = round(float(df_pe.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        feed_label = f"Angel One Real Traded ({c_ce['symbol']} + {c_pe['symbol']})"

                entry_tot = round(real_ce + real_pe, 2)
                cost = round(entry_tot * self.lot_size, 2)
                if cost > 10000.0:
                    return

                ce_sym = c_ce["symbol"] if c_ce else f"{ce_k}CE"
                pe_sym = c_pe["symbol"] if c_pe else f"{pe_k}PE"
                curr_bar = day_bars.iloc[-1]
                candle_ts = pd.to_datetime(curr_bar["timestamp"]).to_pydatetime() if "timestamp" in curr_bar else ist_now

                pos_data = {
                    "entry_time": candle_ts,
                    "spot_entry": spot,
                    "c_ce": c_ce,
                    "c_pe": c_pe,
                    "ce_strike": ce_k,
                    "pe_strike": pe_k,
                    "ce_symbol": ce_sym,
                    "pe_symbol": pe_sym,
                    "ce_entry": real_ce,
                    "pe_entry": real_pe,
                    "entry_tot": entry_tot,
                    "cost": cost,
                    "ce_tgt": round(real_ce * (1.0 + config.DAS_WIN_TARGET_PCT), 2),
                    "ce_sl": round(real_ce * (1.0 - config.DAS_LOSE_STOP_PCT), 2),
                    "pe_tgt": round(real_pe * (1.0 + config.DAS_WIN_TARGET_PCT), 2),
                    "pe_sl": round(real_pe * (1.0 - config.DAS_LOSE_STOP_PCT), 2),
                    "ce_exited": False,
                    "pe_exited": False,
                    "ce_exit_p": None,
                    "pe_exit_p": None,
                    "ce_reason": "",
                    "pe_reason": "",
                    "data_feed": feed_label
                }
                self.on_trade_entry("s4", "Strategy 4: Decoupled Asymmetric Strangle", pos_data)
                send_trade_entry_alert(
                    strategy="Strategy 4: Decoupled Asymmetric Strangle (DAS)",
                    spot=spot,
                    ce_str=ce_sym,
                    pe_str=pe_sym,
                    ce_p=real_ce,
                    pe_p=real_pe,
                    tot_cost=cost,
                    win_target_pct=config.DAS_WIN_TARGET_PCT,
                    lose_stop_pct=config.DAS_LOSE_STOP_PCT,
                    max_hold_mins=config.DAS_MAX_HOLD_MINS,
                    time_str=candle_ts.strftime("%H:%M:%S"),
                    qty=self.lot_size
                )

    def process_live_strategy_1(self, df_all: pd.DataFrame, today_date, ist_now, spot: float):
        """Real-time live position tracking and entry engine for Strategy 1 (15M Box Strangle)."""
        pos = self.active_positions["s1"]

        if pos is not None:
            elapsed_mins = (ist_now - pos["entry_time"]).total_seconds() / 60.0
            p_c = pos["ce_entry"]
            p_p = pos["pe_entry"]
            high_c, low_c = p_c, p_c
            high_p, low_p = p_p, p_p

            if self.smart_api and pos.get("c_ce") and pos.get("c_pe"):
                df_c = fetch_real_option_candles(self.smart_api, pos["c_ce"], pos["entry_time"], ist_now)
                df_p = fetch_real_option_candles(self.smart_api, pos["c_pe"], pos["entry_time"], ist_now)
                if not df_c.empty:
                    p_c = float(df_c.iloc[-1]["close"])
                    high_c = float(df_c["high"].max())
                    low_c = float(df_c["low"].min())
                if not df_p.empty:
                    p_p = float(df_p.iloc[-1]["close"])
                    high_p = float(df_p["high"].max())
                    low_p = float(df_p["low"].min())
            else:
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                p_c = black_scholes_price(spot, pos["ce_strike"], t_exp_years, config.RISK_FREE_RATE, 0.14, "CE")
                p_p = black_scholes_price(spot, pos["pe_strike"], t_exp_years, config.RISK_FREE_RATE, 0.14, "PE")
                high_c, low_c = p_c, p_c
                high_p, low_p = p_p, p_p

            if not pos["ce_exited"]:
                if high_c >= pos["ce_tgt"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = "CE Target (+75%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 1: Hedged Long Strangle (15M Box)`\n"
                            f"🟢 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (+75% Target)\n"
                            f"🛡️ *PUT Leg Status:* Still active with uncoupled protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )
                elif low_c <= pos["ce_sl"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = "CE SL (-20%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 1: Hedged Long Strangle (15M Box)`\n"
                            f"🔴 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (-20% SL)\n"
                            f"🛡️ *PUT Leg Status:* Still active under uncoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )

            if not pos["pe_exited"]:
                if high_p >= pos["pe_tgt"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = "PE Target (+75%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 1: Hedged Long Strangle (15M Box)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (+75% Target)\n"
                            f"🛡️ *CALL Leg Status:* Still active with uncoupled protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )
                elif low_p <= pos["pe_sl"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = "PE SL (-20%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 1: Hedged Long Strangle (15M Box)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (-20% SL)\n"
                            f"🛡️ *CALL Leg Status:* Still active under uncoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━"
                        )

            exit_c = pos["ce_exit_p"] if pos["ce_exited"] else round(p_c * (1.0 - config.SLIPPAGE_PCT), 2)
            exit_p = pos["pe_exit_p"] if pos["pe_exited"] else round(p_p * (1.0 - config.SLIPPAGE_PCT), 2)
            cur_tot = exit_c + exit_p
            is_time_up = (elapsed_mins >= 120.0) or (ist_now.time() >= dtime(15, 15))

            if is_time_up:
                if not pos["ce_exited"]:
                    pos["ce_reason"] = "Time Stop (120m)"
                if not pos["pe_exited"]:
                    pos["pe_reason"] = "Time Stop (120m)"

            if (pos["ce_exited"] and pos["pe_exited"]) or is_time_up:
                exit_reason = f"{pos['ce_reason']} | {pos['pe_reason']}"

                gross = ((exit_c - pos["ce_entry"]) + (exit_p - pos["pe_entry"])) * self.lot_size
                buy_v = pos["entry_tot"] * self.lot_size
                sell_v = cur_tot * self.lot_size
                turn = buy_v + sell_v
                stt = sell_v * 0.001
                exch = turn * 0.0005
                charges = round(80.0 + stt + exch + (80.0 + exch) * 0.18, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 1: Hedged Strangle",
                    "opt_type": "CE+PE",
                    "strike": f"{pos['ce_symbol']} / {pos['pe_symbol']}",
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_tot"],
                    "exit_p": round(cur_tot, 2),
                    "cost": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                self.on_trade_exit("s1", "Strategy 1: Hedged Strangle", tr_log)
                send_trade_exit_alert(
                    strategy="Strategy 1: Hedged Long Strangle (15M Box)",
                    exit_reason=exit_reason,
                    gross_pnl=round(gross, 2),
                    charges=charges,
                    net_pnl=net,
                    details=f"Entry: ₹{pos['entry_tot']:.2f} -> Exit: ₹{cur_tot:.2f} ({pos['ce_symbol']} + {pos['pe_symbol']})",
                    time_str=ist_now.strftime("%H:%M:%S")
                )

        else:
            if self.daily_trade_count["s1"] >= 1:
                return
            if ist_now.time() < dtime(9, 30) or ist_now.time() > dtime(13, 30):
                return
            if df_all.empty:
                return

            day_bars = df_all[df_all["timestamp"].dt.date == today_date]
            if day_bars.empty:
                return

            box_bars = day_bars[(day_bars["timestamp"].dt.time >= dtime(9, 15)) & (day_bars["timestamp"].dt.time <= dtime(9, 30))]
            if len(box_bars) < 10:
                return

            box_h = float(box_bars["high"].max())
            box_l = float(box_bars["low"].min())
            open_p = float(box_bars["close"].iloc[0])
            box_pct = (box_h - box_l) / open_p * 100.0
            if box_pct > 0.30:
                return

            upper_trig = box_h + 6.0
            lower_trig = box_l - 6.0
            curr_bar = day_bars.iloc[-1]
            candle_ts = pd.to_datetime(curr_bar["timestamp"]).to_pydatetime() if "timestamp" in curr_bar else ist_now

            if float(curr_bar["high"]) >= upper_trig or float(curr_bar["low"]) <= lower_trig or spot > upper_trig or spot < lower_trig:
                atm = int(round(spot / 50.0) * 50)
                ce_k = atm + 150
                pe_k = atm - 150
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)

                c_ce = get_active_option_contract(today_date, ce_k, "CE", underlying="NIFTY") if self.smart_api else None
                c_pe = get_active_option_contract(today_date, pe_k, "PE", underlying="NIFTY") if self.smart_api else None

                p_ce = black_scholes_price(spot, ce_k, t_exp_years, config.RISK_FREE_RATE, 0.14, "CE")
                p_pe = black_scholes_price(spot, pe_k, t_exp_years, config.RISK_FREE_RATE, 0.14, "PE")
                real_ce = p_ce
                real_pe = p_pe
                feed_label = "Mathematical (BSM)"

                if self.smart_api and c_ce and c_pe:
                    df_ce = fetch_real_option_candles(self.smart_api, c_ce, ist_now - timedelta(minutes=10), ist_now)
                    df_pe = fetch_real_option_candles(self.smart_api, c_pe, ist_now - timedelta(minutes=10), ist_now)
                    if not df_ce.empty and not df_pe.empty:
                        real_ce = round(float(df_ce.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        real_pe = round(float(df_pe.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        feed_label = f"Angel One Real Traded ({c_ce['symbol']} + {c_pe['symbol']})"

                entry_tot = round(real_ce + real_pe, 2)
                cost = round(entry_tot * self.lot_size, 2)
                if cost > 10000.0:
                    return

                ce_sym = c_ce["symbol"] if c_ce else f"{ce_k}CE"
                pe_sym = c_pe["symbol"] if c_pe else f"{pe_k}PE"

                pos_data = {
                    "entry_time": candle_ts,
                    "spot_entry": spot,
                    "c_ce": c_ce,
                    "c_pe": c_pe,
                    "ce_strike": ce_k,
                    "pe_strike": pe_k,
                    "ce_symbol": ce_sym,
                    "pe_symbol": pe_sym,
                    "ce_entry": real_ce,
                    "pe_entry": real_pe,
                    "entry_tot": entry_tot,
                    "cost": cost,
                    "ce_tgt": round(real_ce * 1.75, 2),
                    "ce_sl": round(real_ce * 0.80, 2),
                    "pe_tgt": round(real_pe * 1.75, 2),
                    "pe_sl": round(real_pe * 0.80, 2),
                    "ce_exited": False,
                    "pe_exited": False,
                    "ce_exit_p": None,
                    "pe_exit_p": None,
                    "ce_reason": "",
                    "pe_reason": "",
                    "data_feed": feed_label
                }
                self.on_trade_entry("s1", "Strategy 1: Hedged Strangle", pos_data)
                send_trade_entry_alert(
                    strategy="Strategy 1: Hedged Long Strangle (15M Box)",
                    spot=spot,
                    ce_str=ce_sym,
                    pe_str=pe_sym,
                    ce_p=real_ce,
                    pe_p=real_pe,
                    tot_cost=cost,
                    win_target_pct=0.75,
                    lose_stop_pct=0.20,
                    max_hold_mins=120,
                    time_str=candle_ts.strftime("%H:%M:%S"),
                    qty=self.lot_size
                )

    def process_live_strategy_2(self, df_all: pd.DataFrame, today_date, ist_now, spot: float):
        """Real-time live position tracking and entry engine for Strategy 2 (Gamma Squeeze)."""
        pos = self.active_positions["s2"]

        if pos is not None:
            elapsed_mins = (ist_now - pos["entry_time"]).total_seconds() / 60.0
            curr_p = pos["entry_p"]
            high_p = curr_p
            low_p = curr_p

            if self.smart_api and pos.get("contract"):
                df_opt = fetch_real_option_candles(self.smart_api, pos["contract"], pos["entry_time"], ist_now)
                if not df_opt.empty:
                    curr_p = float(df_opt.iloc[-1]["close"])
                    high_p = float(df_opt["high"].max())
                    low_p = float(df_opt["low"].min())
            else:
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                curr_p = black_scholes_price(spot, pos["strike"], t_exp_years, config.RISK_FREE_RATE, 0.14, pos["opt_type"])
                high_p, low_p = curr_p, curr_p

            exit_triggered = False
            exit_reason = ""
            exit_p = curr_p

            if high_p >= pos["target_p"]:
                exit_triggered = True
                exit_p = round(pos["target_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "Target Hit (+80%)"
            elif low_p <= pos["stop_p"]:
                exit_triggered = True
                exit_p = round(pos["stop_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "SL Hit (-25%)"
            elif elapsed_mins >= 45.0 or ist_now.time() >= dtime(15, 15):
                exit_triggered = True
                exit_p = round(curr_p * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "Time Stop (45m)"

            if exit_triggered:
                gross = (exit_p - pos["entry_p"]) * self.lot_size
                buy_v = pos["entry_p"] * self.lot_size
                sell_v = exit_p * self.lot_size
                turn = buy_v + sell_v
                stt = sell_v * 0.001
                exch = turn * 0.0005
                charges = round(40.0 + stt + exch + (40.0 + exch) * 0.18, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 2: Expiry Gamma Squeeze",
                    "opt_type": pos["opt_type"],
                    "strike": str(pos["strike"]),
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_p"],
                    "exit_p": exit_p,
                    "cost": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                self.on_trade_exit("s2", "Strategy 2: Expiry Gamma Squeeze", tr_log)
                send_trade_exit_alert(
                    strategy="Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze",
                    exit_reason=exit_reason,
                    gross_pnl=round(gross, 2),
                    charges=charges,
                    net_pnl=net,
                    details=f"BUY {pos['symbol']} @ ₹{pos['entry_p']:.2f} exited @ ₹{exit_p:.2f}",
                    time_str=ist_now.strftime("%H:%M:%S")
                )

        else:
            if today_date.weekday() not in [0, 1]:
                return
            if self.daily_trade_count["s2"] >= 1:
                return
            if ist_now.time() < dtime(9, 45) or ist_now.time() > dtime(11, 30):
                return
            if df_all.empty:
                return

            day_bars = df_all[df_all["timestamp"].dt.date == today_date]
            if day_bars.empty:
                return

            win_bars = day_bars[(day_bars["timestamp"].dt.time >= dtime(9, 45)) & (day_bars["timestamp"].dt.time <= dtime(11, 30))]
            if len(win_bars) < 21:
                return

            lookback = win_bars.iloc[-21:-1]
            box_h = float(lookback["high"].max())
            box_l = float(lookback["low"].min())
            box_rng = box_h - box_l
            spot_ref = float(lookback["close"].iloc[-1])

            if (box_rng / spot_ref) * 100.0 > 0.25:
                return

            upper_trig = box_h + 5.0
            lower_trig = box_l - 5.0
            curr_bar = win_bars.iloc[-1]

            opt_type = ""
            if float(curr_bar["high"]) >= upper_trig or spot >= upper_trig:
                opt_type = "CE"
            elif float(curr_bar["low"]) <= lower_trig or spot <= lower_trig:
                opt_type = "PE"
            else:
                return

            t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
            atm = int(round(spot / 50.0) * 50)
            chosen_k = atm
            best_diff = 999.0
            for offset in range(-500, 550, 50):
                k = atm + offset
                p_est = black_scholes_price(spot, k, t_exp_years, config.RISK_FREE_RATE, 0.14, opt_type)
                if abs(p_est - 50.0) < best_diff and p_est * self.lot_size <= 10000.0:
                    best_diff = abs(p_est - 50.0)
                    chosen_k = k

            contract = get_active_option_contract(today_date, chosen_k, opt_type, underlying="NIFTY") if self.smart_api else None
            real_p = black_scholes_price(spot, chosen_k, t_exp_years, config.RISK_FREE_RATE, 0.14, opt_type)
            feed_label = "Mathematical (BSM)"

            if self.smart_api and contract:
                df_opt = fetch_real_option_candles(self.smart_api, contract, ist_now - timedelta(minutes=10), ist_now)
                if not df_opt.empty:
                    real_p = round(float(df_opt.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                    feed_label = f"Angel One Real Traded ({contract['symbol']})"

            cost = round(real_p * self.lot_size, 2)
            if cost > 10000.0:
                return

            sym = contract["symbol"] if contract else f"{chosen_k} {opt_type}"
            candle_ts = pd.to_datetime(curr_bar["timestamp"]).to_pydatetime() if "timestamp" in curr_bar else ist_now

            pos_data = {
                "entry_time": candle_ts,
                "spot_entry": spot,
                "contract": contract,
                "symbol": sym,
                "opt_type": opt_type,
                "strike": chosen_k,
                "entry_p": real_p,
                "cost": cost,
                "target_p": round(real_p * 1.80, 2),
                "stop_p": round(real_p * 0.75, 2),
                "data_feed": feed_label
            }
            self.on_trade_entry("s2", "Strategy 2: Expiry Gamma Squeeze", pos_data)
            send_trade_entry_alert(
                strategy="Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze",
                spot=spot,
                opt_type=opt_type,
                strike=sym,
                entry_p=real_p,
                tot_cost=cost,
                win_target_pct=0.80,
                lose_stop_pct=0.25,
                max_hold_mins=45,
                time_str=candle_ts.strftime("%H:%M:%S"),
                qty=self.lot_size
            )

    def process_live_strategy_3(self, df_all: pd.DataFrame, today_date, ist_now, spot: float):
        """Real-time live position tracking and entry engine for Strategy 3 (Directional ITM)."""
        pos = self.active_positions["s3"]

        if pos is not None:
            elapsed_mins = (ist_now - pos["entry_time"]).total_seconds() / 60.0
            curr_p = pos["entry_p"]
            high_p = curr_p
            low_p = curr_p

            if self.smart_api and pos.get("contract"):
                df_opt = fetch_real_option_candles(self.smart_api, pos["contract"], pos["entry_time"], ist_now)
                if not df_opt.empty:
                    curr_p = float(df_opt.iloc[-1]["close"])
                    high_p = float(df_opt["high"].max())
                    low_p = float(df_opt["low"].min())
            else:
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                curr_p = black_scholes_price(spot, pos["strike"], t_exp_years, config.RISK_FREE_RATE, 0.14, pos["opt_type"])
                high_p, low_p = curr_p, curr_p

            pos["peak_p"] = max(pos.get("peak_p", pos["entry_p"]), high_p)

            exit_triggered = False
            exit_reason = ""
            exit_p = curr_p

            if high_p >= pos["target_p"]:
                exit_triggered = True
                exit_p = round(pos["target_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "Target Hit (+80%)"
            elif low_p <= pos["stop_p"]:
                exit_triggered = True
                exit_p = round(pos["stop_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "SL Hit (-15%)"
            elif pos["peak_p"] >= pos["trail_act_p"] and curr_p <= pos["peak_p"] * 0.85:
                exit_triggered = True
                exit_p = round(curr_p * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "Trailing Peak Exit"
            elif elapsed_mins >= 60.0 or ist_now.time() >= dtime(15, 15):
                exit_triggered = True
                exit_p = round(curr_p * (1.0 - config.SLIPPAGE_PCT), 2)
                exit_reason = "Time Exit (60m)"

            if exit_triggered:
                gross = (exit_p - pos["entry_p"]) * self.lot_size
                buy_v = pos["entry_p"] * self.lot_size
                sell_v = exit_p * self.lot_size
                turn = buy_v + sell_v
                stt = sell_v * 0.001
                exch = turn * 0.0005
                charges = round(40.0 + stt + exch + (40.0 + exch) * 0.18, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 3: 30M Directional ITM",
                    "opt_type": pos["opt_type"],
                    "strike": str(pos["strike"]),
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_p"],
                    "exit_p": exit_p,
                    "cost": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                self.on_trade_exit("s3", "Strategy 3: 30M Directional ITM", tr_log)
                send_trade_exit_alert(
                    strategy="Strategy 3: 30M Statistical Directional ITM",
                    exit_reason=exit_reason,
                    gross_pnl=round(gross, 2),
                    charges=charges,
                    net_pnl=net,
                    details=f"BUY {pos['symbol']} @ ₹{pos['entry_p']:.2f} exited @ ₹{exit_p:.2f}",
                    time_str=ist_now.strftime("%H:%M:%S")
                )

        else:
            if self.daily_trade_count["s3"] >= 1:
                return
            if ist_now.time() < dtime(9, 45) or ist_now.time() > dtime(13, 30):
                return
            if df_all.empty:
                return

            day_bars = df_all[df_all["timestamp"].dt.date == today_date]
            if day_bars.empty:
                return

            box_bars = day_bars[(day_bars["timestamp"].dt.time >= dtime(9, 15)) & (day_bars["timestamp"].dt.time <= dtime(9, 45))]
            if len(box_bars) < 15:
                return

            box_h = float(box_bars["high"].max())
            box_l = float(box_bars["low"].min())
            open_p = float(box_bars["close"].iloc[0])
            box_pct = (box_h - box_l) / open_p * 100.0
            if box_pct > 0.32:
                return

            upper_trig = box_h + 6.0
            lower_trig = box_l - 6.0
            curr_bar = day_bars.iloc[-1]

            opt_type = ""
            atm = int(round(spot / 50.0) * 50)
            if float(curr_bar["high"]) >= upper_trig or spot >= upper_trig:
                opt_type = "CE"
                chosen_k = atm - 50
            elif float(curr_bar["low"]) <= lower_trig or spot <= lower_trig:
                opt_type = "PE"
                chosen_k = atm + 50
            else:
                return

            t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
            contract = get_active_option_contract(today_date, chosen_k, opt_type, underlying="NIFTY") if self.smart_api else None
            real_p = black_scholes_price(spot, chosen_k, t_exp_years, config.RISK_FREE_RATE, 0.14, opt_type)
            feed_label = "Mathematical (BSM)"

            if self.smart_api and contract:
                df_opt = fetch_real_option_candles(self.smart_api, contract, ist_now - timedelta(minutes=10), ist_now)
                if not df_opt.empty:
                    real_p = round(float(df_opt.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                    feed_label = f"Angel One Real Traded ({contract['symbol']})"

            cost = round(real_p * self.lot_size, 2)
            if cost > 10000.0:
                return

            sym = contract["symbol"] if contract else f"{chosen_k} {opt_type}"
            candle_ts = pd.to_datetime(curr_bar["timestamp"]).to_pydatetime() if "timestamp" in curr_bar else ist_now

            pos_data = {
                "entry_time": candle_ts,
                "spot_entry": spot,
                "contract": contract,
                "symbol": sym,
                "opt_type": opt_type,
                "strike": chosen_k,
                "entry_p": real_p,
                "peak_p": real_p,
                "cost": cost,
                "target_p": round(real_p * 1.80, 2),
                "stop_p": round(real_p * 0.85, 2),
                "trail_act_p": round(real_p * 1.25, 2),
                "data_feed": feed_label
            }
            self.on_trade_entry("s3", "Strategy 3: 30M Directional ITM", pos_data)
            send_trade_entry_alert(
                strategy="Strategy 3: 30M Statistical Directional ITM",
                spot=spot,
                opt_type=opt_type,
                strike=sym,
                entry_p=real_p,
                tot_cost=cost,
                win_target_pct=0.80,
                lose_stop_pct=0.15,
                max_hold_mins=60,
                time_str=candle_ts.strftime("%H:%M:%S"),
                qty=self.lot_size
            )

    def run_live_loop(self):
        """Continuously monitors live market during trading hours across all active strategies in real-time."""
        print("\n" + "=" * 85)
        print("   MULTI-INDEX LIVE MARKET TRADING ENGINE ACTIVE (REAL-TIME STATE MACHINE)")
        print("   Tracking Strategies: Strategy 1, 2, 3, 4 (Nifty) & Strategy 5 (BankNifty DAS)")
        print("=" * 85)

        ist_now = get_ist_now()
        today_date = ist_now.date()
        self.init_today_state(today_date)

        send_telegram_alert(
            f"🚀 *Option Trading Bot Woke Up (Live Engine Active)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Session Time:* {ist_now.strftime('%I:%M:%S %p')} IST\n"
            f"📅 *Date:* {ist_now.strftime('%A, %d-%b-%Y')}\n\n"
            f"Bot is actively monitoring all 5 quantitative strategies in real-time:\n"
            f"• Strategy 1: Hedged Long Strangle (15M Box - Nifty)\n"
            f"• Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze (Nifty)\n"
            f"• Strategy 3: 30M Statistical Directional ITM Breakout (Nifty)\n"
            f"• Strategy 4: Decoupled Asymmetric Strangle (DAS - Nifty)\n"
            f"• Strategy 5: Decoupled Asymmetric Strangle (DAS - BankNifty)\n\n"
            f"💰 *Capital Limit:* ₹10,000 per trade\n"
            f"🔔 Instant alerts will be sent here the exact second a trade enters and exits!"
        )

        while get_ist_now().time() <= dtime(15, 25):
            ist_now = get_ist_now()
            today_date = ist_now.date()

            # 1. Sync live candles from Angel One SmartAPI
            try:
                if self.smart_api:
                    df_all = sync_and_update_candles(self.smart_api, CANDLE_FILE, "99926000")
                else:
                    df_all = pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame()
            except Exception as e:
                print(f"[WARN] Nifty candle sync error: {e}")
                df_all = pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame()

            try:
                if self.smart_api:
                    df_bn = sync_and_update_candles(self.smart_api, BANKNIFTY_CANDLE_FILE, "99926009")
                else:
                    df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame()
            except Exception as e:
                print(f"[WARN] BankNifty candle sync error: {e}")
                df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame()

            if df_all.empty and df_bn.empty:
                time.sleep(30)
                continue

            if not df_all.empty:
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.tz_localize(None)
                day_candles = df_all[df_all["timestamp"].dt.date == today_date]
                spot = float(day_candles["close"].iloc[-1]) if not day_candles.empty else self.get_nifty_spot_price()
            else:
                spot = self.get_nifty_spot_price()

            if not df_bn.empty:
                df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
                day_bn_candles = df_bn[df_bn["timestamp"].dt.date == today_date]
                bn_spot = float(day_bn_candles["close"].iloc[-1]) if not day_bn_candles.empty else self.get_banknifty_spot_price()
            else:
                bn_spot = self.get_banknifty_spot_price()

            # Execute real-time tracking across all 5 strategies
            if self.target_strategy in ["all", "strangle"] and not df_all.empty:
                try:
                    self.process_live_strategy_1(df_all, today_date, ist_now, spot)
                except Exception as e:
                    print(f"[WARN] Strategy 1 live error: {e}")

            if self.target_strategy in ["all", "gamma"] and not df_all.empty:
                try:
                    self.process_live_strategy_2(df_all, today_date, ist_now, spot)
                except Exception as e:
                    print(f"[WARN] Strategy 2 live error: {e}")

            if self.target_strategy in ["all", "directional"] and not df_all.empty:
                try:
                    self.process_live_strategy_3(df_all, today_date, ist_now, spot)
                except Exception as e:
                    print(f"[WARN] Strategy 3 live error: {e}")

            if self.target_strategy in ["all", "das"] and not df_all.empty:
                try:
                    self.process_live_strategy_4(df_all, today_date, ist_now, spot)
                except Exception as e:
                    print(f"[WARN] Strategy 4 live error: {e}")

            if self.target_strategy in ["all", "bndas", "das"] and not df_bn.empty:
                try:
                    self.process_live_strategy_5(df_bn, today_date, ist_now, bn_spot)
                except Exception as e:
                    print(f"[WARN] Strategy 5 live error: {e}")

            time.sleep(60)

        # Market Close Procedure at 15:25 IST
        print("[MARKET CLOSE] Squareoff time reached (15:25 IST). Performing EOD audit...")
        self.run_eod_accounting()

    def run_eod_accounting(self):
        """Runs end-of-day result check and logs to journal and Telegram."""
        df_all = pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame()
        df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame()
        today = get_ist_now().date()

        # Check journal for trades already recorded today from live state machine
        daily_net = 0.0
        trades_today = 0
        has_live_trades_today = False
        if os.path.exists("data/trade_journal.csv"):
            try:
                df_j = pd.read_csv("data/trade_journal.csv")
                if not df_j.empty and "date" in df_j.columns and "net_pnl" in df_j.columns:
                    today_j = df_j[df_j["date"].astype(str) == str(today)]
                    if not today_j.empty:
                        has_live_trades_today = True
                        trades_today = len(today_j)
                        daily_net = round(float(today_j["net_pnl"].sum()), 2)
            except Exception:
                pass

        if not has_live_trades_today:
            if not df_all.empty:
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
                s1_res = evaluate_strategy_1(df_all, today, api=self.smart_api)
                s2_res = evaluate_strategy_2(df_all, today, api=self.smart_api)
                s3_res = evaluate_strategy_3(df_all, today, api=self.smart_api)
                s4_res = evaluate_strategy_4(df_all, today, api=self.smart_api)
            else:
                s1_res, s2_res, s3_res, s4_res = {}, {}, {}, {}

            if not df_bn.empty:
                df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
                s5_res = evaluate_strategy_5(df_bn, today, api=self.smart_api)
            else:
                s5_res = {}

            # Log any executed trades to journal
            for s_name, s_res in [
                ("Strategy 1: Hedged Strangle", s1_res),
                ("Strategy 2: Expiry Gamma Squeeze", s2_res),
                ("Strategy 3: 30M Directional ITM", s3_res),
                ("Strategy 4: Decoupled Asymmetric Strangle", s4_res),
                ("Strategy 5: BankNIFTY Decoupled Strangle (DAS)", s5_res)
            ]:
                if s_res.get("trade_occurred", False):
                    trades_today += s_res.get("num_trades_day", 1)
                    daily_net += s_res.get("net_pnl", 0.0)
                    trades_to_notify = s_res["all_trades"] if ("all_trades" in s_res and len(s_res["all_trades"]) > 1) else [s_res]
                    for sub_tr in trades_to_notify:
                        log_trade_to_journal(today, s_name, sub_tr)
                        send_trade_exit_alert(
                            strategy=s_name,
                            exit_reason=sub_tr.get("exit_reason", "EOD Reconciliation"),
                            gross_pnl=sub_tr.get("gross_pnl", sub_tr.get("gross", 0.0)),
                            charges=sub_tr.get("charges", 0.0),
                            net_pnl=sub_tr.get("net_pnl", sub_tr.get("net", 0.0)),
                            details=f"{sub_tr.get('strike', '')} | Entry: ₹{sub_tr.get('entry_p', 0):.2f} ({sub_tr.get('entry_time', '')}) -> Exit: ₹{sub_tr.get('exit_p', 0):.2f} ({sub_tr.get('exit_time', '')})",
                            time_str=str(sub_tr.get("exit_time", ""))
                        )

        # Read cumulative PnL directly from journal
        total_pnl = 0.0
        if os.path.exists("data/trade_journal.csv"):
            try:
                df_j = pd.read_csv("data/trade_journal.csv")
                if not df_j.empty and "net_pnl" in df_j.columns:
                    total_pnl = round(float(df_j["net_pnl"].sum()), 2)
            except Exception:
                total_pnl = daily_net
        else:
            total_pnl = daily_net

        current_balance = round(10000.0 + total_pnl, 2)

        send_daily_summary_alert(
            date_str=today.strftime("%d-%b-%Y"),
            trades_count=trades_today,
            daily_pnl=daily_net,
            total_pnl=total_pnl,
            current_balance=current_balance
        )

    def run_replay_demonstration(self):
        """Simulates all 5 strategies on the latest market session for offline testing."""
        print("=" * 85)
        print("   MULTI-INDEX STRATEGY REPLAY DEMONSTRATION & TELEGRAM TEST")
        print("=" * 85)
        df_n = pd.read_csv("data/nifty_1min_real.csv") if os.path.exists("data/nifty_1min_real.csv") else pd.DataFrame()
        df_b = pd.read_csv("data/banknifty_1min_real.csv") if os.path.exists("data/banknifty_1min_real.csv") else pd.DataFrame()

        if not df_n.empty:
            df_n["timestamp"] = pd.to_datetime(df_n["timestamp"])
            last_date = df_n["timestamp"].dt.date.max()
        elif not df_b.empty:
            df_b["timestamp"] = pd.to_datetime(df_b["timestamp"]).dt.tz_localize(None)
            last_date = df_b["timestamp"].dt.date.max()
        else:
            print("[ERROR] No candle data for demonstration.")
            return

        if not df_b.empty:
            df_b["timestamp"] = pd.to_datetime(df_b["timestamp"]).dt.tz_localize(None)

        print(f"Testing latest recorded market session: {last_date}")
        s1 = evaluate_strategy_1(df_n, last_date, api=self.smart_api) if not df_n.empty else {}
        s2 = evaluate_strategy_2(df_n, last_date, api=self.smart_api) if not df_n.empty else {}
        s3 = evaluate_strategy_3(df_n, last_date, api=self.smart_api) if not df_n.empty else {}
        s4 = evaluate_strategy_4(df_n, last_date, api=self.smart_api) if not df_n.empty else {}
        s5 = evaluate_strategy_5(df_b, last_date, api=self.smart_api) if not df_b.empty else {}

        print(f"\n[EVALUATION RESULTS FOR {last_date}]")
        print(f"Strategy 1 (15M Strangle - Nifty)    : {s1.get('status')} | Trade: {s1.get('trade_occurred')}")
        print(f"Strategy 2 (Gamma Squeeze - Nifty)   : {s2.get('status')} | Trade: {s2.get('trade_occurred')}")
        print(f"Strategy 3 (Directional ITM - Nifty) : {s3.get('status')} | Trade: {s3.get('trade_occurred')}")
        print(f"Strategy 4 (DAS Strangle - Nifty)    : {s4.get('status')} | Trade: {s4.get('trade_occurred')}")
        print(f"Strategy 5 (DAS Strangle - BankNifty): {s5.get('status')} | Trade: {s5.get('trade_occurred')}")

    def run(self):
        """Master execution entry point."""
        now = get_ist_now()

        # Weekend check
        if now.weekday() in [5, 6]:
            print(f"[WEEKEND] Today is {now.strftime('%A')}. NSE Market is closed on weekends.")
            self.run_replay_demonstration()
            return

        # On weekdays, automatically catch up any un-audited historical trading days before market open!
        try:
            print("\n[STARTUP] Checking for any un-audited historical trading days...")
            df_all = sync_and_update_candles(self.smart_api, CANDLE_FILE, "99926000") if self.smart_api else (pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame())
            df_bn = sync_and_update_candles(self.smart_api, BANKNIFTY_CANDLE_FILE, "99926009") if self.smart_api else (pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame())
            if not df_all.empty:
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.tz_localize(None)
                if not df_bn.empty:
                    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
                prior_dt = now.date() - timedelta(days=1)
                caught_up = auto_catchup_missing_trading_days(self.smart_api, df_all, df_bn, up_to_date=prior_dt)
                if caught_up:
                    print(f"[STARTUP] Successfully caught up {len(caught_up)} missed day(s): {caught_up}")
        except Exception as e:
            print(f"[WARN] Startup catchup error: {e}")

        # Weekday Pre-Market (e.g. 08:30 - 09:14 AM IST)
        if now.time() < dtime(9, 15):
            self.wait_for_market_open()
            self.run_live_loop()

        # Weekday Live Trading Session (09:15 - 15:25 IST)
        elif dtime(9, 15) <= now.time() <= dtime(15, 25):
            self.run_live_loop()

        # Weekday Post-Market (after 15:25 PM IST - e.g. if runner was delayed or backup trigger)
        else:
            print(f"[MARKET CLOSED] Runner started at {now.strftime('%I:%M:%S %p')} IST (Market closed at 15:25 IST).")
            today_str = str(now.date())
            already_audited = False
            if os.path.exists("data/trade_journal.md"):
                try:
                    with open("data/trade_journal.md", "r", encoding="utf-8") as f:
                        if f"Last Updated: {today_str}" in f.read():
                            already_audited = True
                except Exception:
                    pass

            if already_audited:
                print(f"[IDLE] Today's session ({today_str}) has already completed and was audited. Exiting cleanly.")
                return

            print("[EOD RECONCILIATION] Performing End-of-Day audit and journal updates...")
            send_telegram_alert(
                f"⚠️ *Trading Bot Started After Market Close*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ *Started At:* {now.strftime('%I:%M:%S %p')} IST\n"
                f"📅 *Date:* {now.strftime('%A, %d-%b-%Y')}\n\n"
                f"The GitHub Actions runner was delayed past market close (15:25 IST).\n"
                f"Performing final EOD reconciliation report."
            )
            self.run_eod_accounting()

def main():
    parser = argparse.ArgumentParser(description="Multi-Index Live Market Paper Trading Bot")
    parser.add_argument("--strategy", type=str, default="all", choices=["strangle", "gamma", "directional", "das", "bndas", "all"], help="Strategy to trade (strangle, gamma, directional, das, bndas, or all).")
    args = parser.parse_args()

    bot = LiveQuadPaperTrader(target_strategy=args.strategy)
    bot.connect_angel_one()
    bot.run()

if __name__ == "__main__":
    main()
