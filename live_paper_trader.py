"""
Live Market Paper Trading Bot for Nifty & BankNifty Options (Tri-Strategy Core System).
Executes and monitors all 3 quantitative production strategies concurrently:
1. Strategy 1: Nifty 15M Directional Breakout (Trend-Aligned ATM, SL -16%, TP +60%)
2. Strategy 2: BankNIFTY 15M Directional Breakout (Trend-Aligned 100-pt OTM, SL -25%, TP +80%)
3. Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS - Volatility Compression & Kinetic Expansion)

Connects to Angel One SmartAPI to track Nifty and BankNifty Spot and option candles in real-time.
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

try:
    from SmartApi import SmartConnect
    import pyotp
except ImportError:
    SmartConnect = None

from src.features.greeks import black_scholes_price, compute_calendar_dte
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
    send_daily_summary_alert,
    is_alert_already_sent,
    mark_alert_as_sent
)
from src.notifications.telegram_interactive_bot import start_interactive_bot_daemon

from daily_result_checker import (
    evaluate_strategy_1,
    evaluate_strategy_2,
    evaluate_strategy_3,
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
from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles, get_candle_feed
from src.features.greeks import black_scholes_price, compute_calendar_dte

load_dotenv()

def get_ist_now() -> datetime:
    """Returns the current datetime in Indian Standard Time (IST, UTC+5:30) as a timezone-naive datetime."""
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).replace(tzinfo=None)

def get_banknifty_dte(ist_now: datetime, target_date) -> float:
    """Calculates accurate DTE in years for BankNIFTY using real contract expiry from database."""
    target_dt = target_date if isinstance(target_date, datetime) else datetime.combine(target_date, dtime(15, 30))
    from src.data.real_option_feed import get_nearest_expiry_date
    exp_dt = get_nearest_expiry_date(target_dt.date(), "BANKNIFTY")
    days_ahead = max(0, (exp_dt - target_dt.date()).days)
    return max(0.001, float(days_ahead) / 365.25)

class LiveTriPaperTrader:
    """
    Unified Tri-Strategy Paper Trading Engine.
    Executes and monitors Strategy 1, Strategy 2, and Strategy 3 during live market hours.
    """
    def __init__(self, target_strategy: str = "all"):
        self.target_strategy = target_strategy.lower() # 'all', 's1', 's2', 's3'
        self.lot_size = config.LOT_SIZE
        self.bn_lot_size = config.BANKNIFTY_LOT_SIZE
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
        }
        self.cooldown_until = {
            "s1": None,
            "s2": None,
            "s3": None,
        }
        self.daily_trade_count = {
            "s1": 0,
            "s2": 0,
            "s3": 0,
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
                if "c_ce" in p_copy and isinstance(p_copy["c_ce"], dict):
                    p_copy["c_ce"] = {k: str(v) for k, v in p_copy["c_ce"].items() if k in ["token", "symbol"]}
                if "c_pe" in p_copy and isinstance(p_copy["c_pe"], dict):
                    p_copy["c_pe"] = {k: str(v) for k, v in p_copy["c_pe"].items() if k in ["token", "symbol"]}
                if "c_opt" in p_copy and isinstance(p_copy["c_opt"], dict):
                    p_copy["c_opt"] = {k: str(v) for k, v in p_copy["c_opt"].items() if k in ["token", "symbol", "expiry"]}
                if "contract" in p_copy and isinstance(p_copy["contract"], dict):
                    p_copy["contract"] = {k: str(v) for k, v in p_copy["contract"].items() if k in ["token", "symbol"]}
                data_to_save[s_key] = p_copy
        os.makedirs("data", exist_ok=True)
        try:
            with open("data/active_positions.json", "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, default=str)
        except Exception as e:
            print(f"[WARN] Failed to save active positions: {e}")

    def on_trade_entry(self, s_key: str, s_name: str, pos: dict):
        """Called immediately upon trade entry: updates active positions, journal, and pushes to git."""
        pos["strategy_name"] = s_name
        self.active_positions[s_key] = pos
        self.daily_trade_count[s_key] += 1
        self.save_active_positions()
        try:
            refresh_markdown_journal()
        except Exception as e:
            print(f"[WARN] Failed to refresh markdown journal: {e}")
        sym_str = pos.get("strike") or pos.get("symbol") or f"{pos.get('ce_symbol', '')} + {pos.get('pe_symbol', '')}"
        print(f"[LIVE JOURNAL] [ENTRY] Active trade logged in trade_journal.md for {s_name} ({sym_str})")
        try:
            git_sync_push(f"Auto-Journal: 🟢 Entry {s_name} ({sym_str})")
        except Exception as e:
            print(f"[WARN] Non-blocking git push deferred: {e}")

    def on_trade_exit(self, s_key: str, s_name: str, tr_log: dict):
        """Called immediately upon trade exit: clears active position, appends to journal, and pushes to git."""
        self.active_positions[s_key] = None
        self.save_active_positions()
        try:
            log_trade_to_journal(tr_log["date"], s_name, tr_log)
            refresh_markdown_journal()
        except Exception as e:
            print(f"[WARN] Failed to log trade to journal: {e}")
        net_val = float(tr_log.get("net_pnl", 0.0))
        print(f"[LIVE JOURNAL] [EXIT] Trade exit logged in trade_journal.md for {s_name} [PnL: INR {net_val:+.2f}]")
        try:
            git_sync_push(f"Auto-Journal: 🔴 Exit {s_name} [PnL: INR {net_val:+.2f}]")
        except Exception as e:
            print(f"[WARN] Non-blocking git push deferred: {e}")

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
                            p["entry_time"] = datetime.fromisoformat(p["entry_time"].replace("Z", ""))
                        if hasattr(p["entry_time"], "tzinfo") and p["entry_time"].tzinfo is not None:
                            p["entry_time"] = p["entry_time"].replace(tzinfo=None)
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
        }
        self.cooldown_until = {
            "s1": None,
            "s2": None,
            "s3": None,
        }
        self.daily_trade_count = {
            "s1": 0,
            "s2": 0,
            "s3": 0,
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
            except Exception:
                pass
        self.load_active_positions(today_date)

    def process_live_strategy_1(self, df_all: pd.DataFrame, today_date, ist_now, spot: float):
        """Real-time live position tracking and entry engine for Strategy 1: Nifty 15M Directional Breakout."""
        pos = self.active_positions["s1"]

        if pos is not None:
            c_opt = pos.get("contract")
            cur_p = pos["entry_p"]
            high_p = pos.get("high_p", cur_p)
            low_p = pos.get("low_p", cur_p)

            # 1. Priority 1: Instant live LTP from Angel One SmartAPI
            live_p_found = False
            if self.smart_api and c_opt and c_opt.get("token"):
                try:
                    ltp_res = self.smart_api.ltpData(c_opt.get("exch_seg", "NFO"), c_opt["symbol"], str(c_opt["token"]))
                    if ltp_res and ltp_res.get("status") and ltp_res.get("data") and float(ltp_res["data"].get("ltp", 0)) > 0:
                        cur_p = round(float(ltp_res["data"]["ltp"]), 2)
                        high_p = max(high_p, cur_p)
                        low_p = min(low_p, cur_p)
                        pos["high_p"] = high_p
                        pos["low_p"] = low_p
                        live_p_found = True
                except Exception:
                    pass

            exit_triggered = False
            exit_p = cur_p
            exit_reason = ""

            # Check candle sequence first (chronological order without lookahead bias)
            if self.smart_api and c_opt:
                df_opt = fetch_real_option_candles(self.smart_api, c_opt, pos["entry_time"], ist_now)
                if not df_opt.empty:
                    for _, c_row in df_opt.iterrows():
                        # Strict risk-first: check Stop Loss before Target on each candle
                        if float(c_row["low"]) <= pos["sl_p"]:
                            exit_triggered = True
                            exit_p = round(pos["sl_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = f"SL Hit (-{int(config.S1_SL_PCT * 100)}%)"
                            break
                        elif float(c_row["high"]) >= pos["target_p"]:
                            exit_triggered = True
                            exit_p = round(pos["target_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = f"Target Hit (+{int(config.S1_TARGET_PCT * 100)}%)"
                            break
                    if not exit_triggered:
                        cur_p = float(df_opt.iloc[-1]["close"])

            # If no candle triggered, check latest real-time tick (LTP) or BSM fallback
            if not exit_triggered:
                if not live_p_found:
                    t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                    cur_p = black_scholes_price(spot, pos["strike"], t_exp_years, config.RISK_FREE_RATE, 0.14, pos["opt_type"])

                if cur_p <= pos["sl_p"]:
                    exit_triggered = True
                    exit_p = round(pos["sl_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    exit_reason = f"SL Hit (-{int(config.S1_SL_PCT * 100)}%)"
                elif cur_p >= pos["target_p"]:
                    exit_triggered = True
                    exit_p = round(pos["target_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    exit_reason = f"Target Hit (+{int(config.S1_TARGET_PCT * 100)}%)"
                elif ist_now.time() >= dtime(15, 5):
                    exit_triggered = True
                    exit_p = round(cur_p * (1.0 - config.SLIPPAGE_PCT), 2)
                    exit_reason = "EOD Squareoff (15:05)"

            if exit_triggered:
                gross = (exit_p - pos["entry_p"]) * self.lot_size
                turn = (pos["entry_p"] + exit_p) * self.lot_size
                brok = 40.0
                stt = (exit_p * self.lot_size) * config.STT_PCT_ON_SELL
                exch = turn * config.EXCHANGE_TURNOVER_PCT
                gst = (brok + exch) * config.GST_PCT
                charges = round(brok + stt + exch + gst, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 1: Nifty 15M Directional Breakout",
                    "leg": pos["opt_type"],
                    "strike": pos["symbol"],
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S") if hasattr(pos["entry_time"], "strftime") else str(pos["entry_time"]),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_p"],
                    "exit_p": exit_p,
                    "cost": pos["cost"],
                    "capital_used": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                try:
                    send_trade_exit_alert(
                        strategy="Strategy 1: Nifty 15M Directional Breakout",
                        exit_reason=exit_reason,
                        gross_pnl=round(gross, 2),
                        charges=charges,
                        net_pnl=net,
                        details=f"{pos['symbol']} | Entry: ₹{pos['entry_p']:.2f} -> Exit: ₹{exit_p:.2f}",
                        time_str=ist_now.strftime("%H:%M:%S"),
                        date_str=str(today_date),
                        entry_time=str(tr_log["entry_time"]),
                        strike=str(pos["symbol"])
                    )
                except Exception as e:
                    print(f"[WARN] Failed to send S1 exit alert: {e}")
                self.on_trade_exit("s1", "Strategy 1: Nifty 15M Directional Breakout", tr_log)
                self.cooldown_until["s1"] = ist_now + timedelta(minutes=30)
                return

        else:
            if not config.S1_ENABLED:
                return
            active_count = sum(1 for p in self.active_positions.values() if p is not None)
            if active_count >= config.MAX_CONCURRENT_POSITIONS:
                return
            if self.daily_trade_count["s1"] >= 1:
                return
            if self.cooldown_until["s1"] and ist_now < self.cooldown_until["s1"]:
                return
            if ist_now.time() < dtime(9, 30) or ist_now.time() > dtime(15, 0):
                return
            if df_all.empty:
                return

            if "ema_20" not in df_all.columns:
                df_all = df_all.sort_values("timestamp")
                df_all["ema_20"] = df_all["close"].ewm(span=20, adjust=False).mean()
            day_bars = df_all[df_all["timestamp"].dt.date == today_date].sort_values("timestamp")
            if day_bars.empty or len(day_bars) < 15:
                return

            box_bars = day_bars[(day_bars["timestamp"].dt.time >= dtime(9, 15)) & (day_bars["timestamp"].dt.time <= dtime(9, 30))]
            if len(box_bars) < 10:
                return

            box_h = float(box_bars["high"].max())
            box_l = float(box_bars["low"].min())
            open_p = float(box_bars["close"].iloc[0])
            box_range = box_h - box_l
            box_pct = (box_range / open_p) * 100.0
            if box_pct > config.S1_MAX_BOX_PCT:
                return
            u_trig = box_h + config.S1_BUFFER_PTS
            l_trig = box_l - config.S1_BUFFER_PTS

            post_bars = day_bars[(day_bars["timestamp"].dt.time > dtime(9, 30)) & (day_bars["timestamp"].dt.time <= dtime(15, 0))]
            if post_bars.empty:
                return

            breakout_bar = None
            opt_type = None
            for _, b_row in post_bars.iterrows():
                b_c = float(b_row["close"])
                b_ema = float(b_row["ema_20"])
                if b_c >= u_trig and (not config.S1_USE_EMA_TREND or b_c > b_ema):
                    breakout_bar = b_row
                    opt_type = "CE"
                    break
                elif b_c <= l_trig and (not config.S1_USE_EMA_TREND or b_c < b_ema):
                    breakout_bar = b_row
                    opt_type = "PE"
                    break

            if breakout_bar is None:
                return

            c = float(breakout_bar["close"])
            candle_ts = pd.to_datetime(breakout_bar["timestamp"]).to_pydatetime()
            atm = int(round(c / 50.0) * 50)

            # Dynamic strike scan within MAX_CAPITAL_PER_TRADE budget
            offsets = list(range(0, 650, 50))
            chosen_strike = None
            chosen_c_opt = None
            chosen_p = None
            chosen_cost = None
            chosen_feed = "Mathematical (BSM)"

            for off in offsets:
                cand_k = (atm + off) if opt_type == "CE" else (atm - off)
                cand_opt = get_active_option_contract(today_date, cand_k, opt_type, underlying="NIFTY") if self.smart_api else None
                cand_p = None
                cand_feed = "Mathematical (BSM)"

                if self.smart_api and cand_opt and cand_opt.get("token"):
                    try:
                        ltp_res = self.smart_api.ltpData(cand_opt.get("exch_seg", "NFO"), cand_opt["symbol"], str(cand_opt["token"]))
                        if ltp_res and ltp_res.get("status") and ltp_res.get("data") and float(ltp_res["data"].get("ltp", 0)) > 0:
                            cand_p = round(float(ltp_res["data"]["ltp"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            cand_feed = f"Angel One Real Traded LTP ({cand_opt['symbol']})"
                    except Exception:
                        pass

                if cand_p is None and self.smart_api and cand_opt:
                    df_o = fetch_real_option_candles(self.smart_api, cand_opt, candle_ts - timedelta(minutes=5), ist_now)
                    if not df_o.empty:
                        sub_o = df_o[df_o.index >= candle_ts]
                        cand_p = round(float((sub_o.iloc[0] if not sub_o.empty else df_o.iloc[-1])["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        cand_feed = f"Angel One Real Traded ({cand_opt['symbol']})"

                if cand_p is None:
                    t_exp_years = max(1e-4, compute_calendar_dte(candle_ts) / 365.0)
                    cand_p = round(black_scholes_price(c, cand_k, t_exp_years, config.RISK_FREE_RATE, 0.14, opt_type), 2)

                cand_cost = round(cand_p * self.lot_size, 2)
                if chosen_strike is None or cand_cost < chosen_cost:
                    chosen_strike = cand_k
                    chosen_c_opt = cand_opt
                    chosen_p = cand_p
                    chosen_cost = cand_cost
                    chosen_feed = cand_feed

                if cand_cost <= config.MAX_CAPITAL_PER_TRADE and cand_p >= config.DYNAMIC_MIN_PREMIUM:
                    chosen_strike = cand_k
                    chosen_c_opt = cand_opt
                    chosen_p = cand_p
                    chosen_cost = cand_cost
                    chosen_feed = cand_feed
                    break

            strike = chosen_strike
            c_opt = chosen_c_opt
            entry_p = chosen_p
            cost = chosen_cost
            feed_label = chosen_feed

            if cost > config.MAX_CAPITAL_PER_TRADE:
                return

            total_deployed = sum(p.get("cost", 0.0) for p in self.active_positions.values() if p is not None)
            if (total_deployed + cost) > config.INITIAL_CAPITAL:
                return

            target_p = round(entry_p * (1.0 + config.S1_TARGET_PCT), 2)
            sl_p = round(entry_p * (1.0 - config.S1_SL_PCT), 2)

            # Late wake-up / restart resolution check
            if candle_ts < ist_now and self.smart_api and c_opt:
                df_history = fetch_real_option_candles(self.smart_api, c_opt, candle_ts, ist_now)
                if not df_history.empty:
                    resolved = False
                    res_exit_p = entry_p
                    res_exit_t = ist_now
                    res_reason = ""
                    for c_ts, c_row in df_history.iterrows():
                        if float(c_row["low"]) <= sl_p:
                            resolved = True
                            res_exit_p = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                            res_exit_t = c_ts
                            res_reason = f"SL Hit (-{int(config.S1_SL_PCT * 100)}%)"
                            break
                        elif float(c_row["high"]) >= target_p:
                            resolved = True
                            res_exit_p = round(target_p * (1.0 - config.SLIPPAGE_PCT), 2)
                            res_exit_t = c_ts
                            res_reason = f"Target Hit (+{int(config.S1_TARGET_PCT * 100)}%)"
                            break

                    if resolved:
                        gross = (res_exit_p - entry_p) * self.lot_size
                        turn = (entry_p + res_exit_p) * self.lot_size
                        brok = 40.0
                        stt = (res_exit_p * self.lot_size) * config.STT_PCT_ON_SELL
                        exch = turn * config.EXCHANGE_TURNOVER_PCT
                        gst = (brok + exch) * config.GST_PCT
                        charges = round(brok + stt + exch + gst, 2)
                        net = round(gross - charges, 2)
                        sym_str = c_opt["symbol"] if c_opt else f"NIFTY{strike}{opt_type}"
                        tr_log = {
                            "date": str(today_date),
                            "strategy": "Strategy 1: Nifty 15M Directional Breakout",
                            "symbol": sym_str,
                            "entry_time": str(candle_ts),
                            "exit_time": str(res_exit_t),
                            "strike": strike,
                            "opt_type": opt_type,
                            "entry_p": entry_p,
                            "exit_p": res_exit_p,
                            "cost": cost,
                            "gross_pnl": round(gross, 2),
                            "charges": charges,
                            "net_pnl": net,
                            "exit_reason": res_reason,
                            "data_feed": feed_label
                        }
                        self.on_trade_exit("s1", "Strategy 1: Nifty 15M Directional Breakout", tr_log)
                        self.cooldown_until["s1"] = ist_now + timedelta(minutes=30)
                        return

            sym_str = c_opt["symbol"] if c_opt else f"NIFTY{strike}{opt_type}"

            pos_data = {
                "entry_time": candle_ts,
                "spot_entry": c,
                "contract": c_opt,
                "opt_type": opt_type,
                "strike": strike,
                "symbol": sym_str,
                "entry_p": entry_p,
                "cost": cost,
                "target_p": target_p,
                "sl_p": sl_p,
                "data_feed": feed_label
            }
            try:
                send_trade_entry_alert(
                    strategy="Strategy 1: Nifty 15M Directional Breakout",
                    spot=c,
                    opt_type=opt_type,
                    strike=str(sym_str),
                    entry_p=entry_p,
                    ce_str=sym_str if opt_type == "CE" else "",
                    pe_str=sym_str if opt_type == "PE" else "",
                    ce_p=entry_p if opt_type == "CE" else 0.0,
                    pe_p=entry_p if opt_type == "PE" else 0.0,
                    tot_cost=cost,
                    win_target_pct=config.S1_TARGET_PCT,
                    lose_stop_pct=config.S1_SL_PCT,
                    max_hold_mins=360,
                    time_str=candle_ts.strftime("%H:%M:%S") if hasattr(candle_ts, "strftime") else str(candle_ts),
                    qty=self.lot_size,
                    date_str=str(today_date)
                )
            except Exception as e:
                print(f"[WARN] Failed to send S1 entry alert: {e}")
            self.on_trade_entry("s1", "Strategy 1: Nifty 15M Directional Breakout", pos_data)

    def process_live_strategy_2(self, df_bn: pd.DataFrame, today_date, ist_now, bn_spot: float):
        """Real-time live position tracking and entry engine for Strategy 2: BankNIFTY 15M Directional Breakout."""
        pos = self.active_positions["s2"]

        if pos is not None:
            c_opt = pos.get("contract")
            cur_p = pos["entry_p"]
            high_p = pos.get("high_p", cur_p)
            low_p = pos.get("low_p", cur_p)

            # 1. Priority 1: Instant live LTP from Angel One SmartAPI
            live_p_found = False
            if self.smart_api and c_opt and c_opt.get("token"):
                try:
                    ltp_res = self.smart_api.ltpData(c_opt.get("exch_seg", "NFO"), c_opt["symbol"], str(c_opt["token"]))
                    if ltp_res and ltp_res.get("status") and ltp_res.get("data") and float(ltp_res["data"].get("ltp", 0)) > 0:
                        cur_p = round(float(ltp_res["data"]["ltp"]), 2)
                        high_p = max(high_p, cur_p)
                        low_p = min(low_p, cur_p)
                        pos["high_p"] = high_p
                        pos["low_p"] = low_p
                        live_p_found = True
                except Exception:
                    pass

            exit_triggered = False
            exit_p = cur_p
            exit_reason = ""

            # Check candle sequence first (chronological order without lookahead bias)
            if self.smart_api and c_opt:
                df_opt = fetch_real_option_candles(self.smart_api, c_opt, pos["entry_time"], ist_now)
                if not df_opt.empty:
                    for _, c_row in df_opt.iterrows():
                        # Strict risk-first: check Stop Loss before Target on each candle
                        if float(c_row["low"]) <= pos["sl_p"]:
                            exit_triggered = True
                            exit_p = round(pos["sl_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = f"SL Hit (-{int(config.S2_SL_PCT * 100)}%)"
                            break
                        elif float(c_row["high"]) >= pos["target_p"]:
                            exit_triggered = True
                            exit_p = round(pos["target_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                            exit_reason = f"Target Hit (+{int(config.S2_TARGET_PCT * 100)}%)"
                            break
                    if not exit_triggered:
                        cur_p = float(df_opt.iloc[-1]["close"])

            # If no candle triggered, check latest real-time tick (LTP) or BSM fallback
            if not exit_triggered:
                if not live_p_found:
                    dte = max(1e-4, get_banknifty_dte(ist_now, today_date))
                    from src.banknifty.banknifty_das import bs_price
                    cur_p = bs_price(bn_spot, pos["strike"], dte, pos["opt_type"], sigma=0.18)

                if cur_p <= pos["sl_p"]:
                    exit_triggered = True
                    exit_p = round(pos["sl_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    exit_reason = f"SL Hit (-{int(config.S2_SL_PCT * 100)}%)"
                elif cur_p >= pos["target_p"]:
                    exit_triggered = True
                    exit_p = round(pos["target_p"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    exit_reason = f"Target Hit (+{int(config.S2_TARGET_PCT * 100)}%)"
                elif ist_now.time() >= dtime(15, 5):
                    exit_triggered = True
                    exit_p = round(cur_p * (1.0 - config.SLIPPAGE_PCT), 2)
                    exit_reason = "EOD Squareoff (15:05)"

            if exit_triggered:
                gross = (exit_p - pos["entry_p"]) * self.bn_lot_size
                turn = (pos["entry_p"] + exit_p) * self.bn_lot_size
                brok = 40.0
                stt = (exit_p * self.bn_lot_size) * config.STT_PCT_ON_SELL
                exch = turn * config.EXCHANGE_TURNOVER_PCT
                gst = (brok + exch) * config.GST_PCT
                charges = round(brok + stt + exch + gst, 2)
                net = round(gross - charges, 2)

                tr_log = {
                    "date": str(today_date),
                    "strategy": "Strategy 2: BankNIFTY 15M Directional Breakout",
                    "leg": pos["opt_type"],
                    "strike": pos["symbol"],
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S") if hasattr(pos["entry_time"], "strftime") else str(pos["entry_time"]),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_p"],
                    "exit_p": exit_p,
                    "cost": pos["cost"],
                    "capital_used": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                try:
                    send_trade_exit_alert(
                        strategy="Strategy 2: BankNIFTY 15M Directional Breakout",
                        exit_reason=exit_reason,
                        gross_pnl=round(gross, 2),
                        charges=charges,
                        net_pnl=net,
                        details=f"{pos['symbol']} | Entry: ₹{pos['entry_p']:.2f} -> Exit: ₹{exit_p:.2f}",
                        time_str=ist_now.strftime("%H:%M:%S"),
                        date_str=str(today_date),
                        entry_time=str(tr_log["entry_time"]),
                        strike=str(pos["symbol"])
                    )
                except Exception as e:
                    print(f"[WARN] Failed to send S2 exit alert: {e}")
                self.on_trade_exit("s2", "Strategy 2: BankNIFTY 15M Directional Breakout", tr_log)
                self.cooldown_until["s2"] = ist_now + timedelta(minutes=30)
                return

        else:
            if not config.S2_ENABLED:
                return
            active_count = sum(1 for p in self.active_positions.values() if p is not None)
            if active_count >= config.MAX_CONCURRENT_POSITIONS:
                return
            if self.daily_trade_count["s2"] >= 1:
                return
            if self.cooldown_until["s2"] and ist_now < self.cooldown_until["s2"]:
                return
            if ist_now.time() < dtime(9, 30) or ist_now.time() > dtime(15, 0):
                return
            if df_bn.empty:
                return

            df = df_bn.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.sort_values("timestamp")
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            day_df = df[df["timestamp"].dt.date == today_date].sort_values("timestamp")
            if len(day_df) < 15:
                return

            box15 = day_df[(day_df["timestamp"].dt.time >= dtime(9, 15)) & (day_df["timestamp"].dt.time <= dtime(9, 30))]
            if len(box15) < 8:
                return

            box_h = float(box15["high"].max())
            box_l = float(box15["low"].min())
            open_p = float(box15["close"].iloc[0])
            b_rng = box_h - box_l
            b_pct = (b_rng / open_p) * 100.0
            if b_pct > config.S2_MAX_BOX_PCT:
                return
            u_trig = box_h + config.S2_BUFFER_PTS
            l_trig = box_l - config.S2_BUFFER_PTS

            post_bars = day_df[(day_df["timestamp"].dt.time > dtime(9, 30)) & (day_df["timestamp"].dt.time <= dtime(15, 0))]
            if post_bars.empty:
                return

            breakout_bar = None
            opt_type = None
            for _, b_row in post_bars.iterrows():
                b_c = float(b_row["close"])
                b_ema = float(b_row["ema_20"])
                if b_c >= u_trig and (not config.S2_USE_EMA_TREND or b_c > b_ema):
                    breakout_bar = b_row
                    opt_type = "CE"
                    break
                elif b_c <= l_trig and (not config.S2_USE_EMA_TREND or b_c < b_ema):
                    breakout_bar = b_row
                    opt_type = "PE"
                    break

            if breakout_bar is None:
                return

            c = float(breakout_bar["close"])
            candle_ts = pd.to_datetime(breakout_bar["timestamp"]).to_pydatetime()
            atm = int(round(c / 100.0) * 100)
            dte_y = max(1e-4, get_banknifty_dte(candle_ts, today_date))

            # Dynamic strike scan for affordable BankNifty contract
            offsets = list(range(config.S2_OTM_OFFSET, 3600, 100))
            chosen_strike = None
            chosen_c_opt = None
            chosen_p = None
            chosen_cost = None
            chosen_feed = "Mathematical (BSM)"

            for off in offsets:
                cand_k = (atm + off) if opt_type == "CE" else (atm - off)
                cand_opt = get_active_option_contract(today_date, cand_k, opt_type, underlying="BANKNIFTY") if self.smart_api else None
                cand_p = None
                cand_feed = "Mathematical (BSM)"

                if self.smart_api and cand_opt and cand_opt.get("token"):
                    try:
                        ltp_res = self.smart_api.ltpData(cand_opt.get("exch_seg", "NFO"), cand_opt["symbol"], str(cand_opt["token"]))
                        if ltp_res and ltp_res.get("status") and ltp_res.get("data") and float(ltp_res["data"].get("ltp", 0)) > 0:
                            cand_p = round(float(ltp_res["data"]["ltp"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                            cand_feed = f"Angel One Real Traded LTP ({cand_opt['symbol']})"
                    except Exception:
                        pass

                if cand_p is None and self.smart_api and cand_opt:
                    df_o = fetch_real_option_candles(self.smart_api, cand_opt, candle_ts - timedelta(minutes=5), ist_now)
                    if not df_o.empty:
                        sub_o = df_o[df_o.index >= candle_ts]
                        cand_p = round(float((sub_o.iloc[0] if not sub_o.empty else df_o.iloc[-1])["close"]) * (1.0 + config.SLIPPAGE_PCT), 2)
                        cand_feed = f"Angel One Real Traded ({cand_opt['symbol']})"

                if cand_p is None:
                    from src.banknifty.banknifty_das import bs_price
                    cand_p = round(bs_price(c, cand_k, dte_y, opt_type, sigma=0.18), 2)

                cand_cost = round(cand_p * self.bn_lot_size, 2)
                if chosen_strike is None or cand_cost < chosen_cost:
                    chosen_strike = cand_k
                    chosen_c_opt = cand_opt
                    chosen_p = cand_p
                    chosen_cost = cand_cost
                    chosen_feed = cand_feed

                if cand_cost <= config.MAX_CAPITAL_PER_TRADE and cand_p >= config.DYNAMIC_MIN_PREMIUM:
                    chosen_strike = cand_k
                    chosen_c_opt = cand_opt
                    chosen_p = cand_p
                    chosen_cost = cand_cost
                    chosen_feed = cand_feed
                    break

            strike = chosen_strike
            c_opt = chosen_c_opt
            entry_p = chosen_p
            cost = chosen_cost
            feed_label = chosen_feed

            if cost > config.MAX_CAPITAL_PER_TRADE:
                return

            total_deployed = sum(p.get("cost", 0.0) for p in self.active_positions.values() if p is not None)
            if (total_deployed + cost) > config.INITIAL_CAPITAL:
                return

            target_p = round(entry_p * (1.0 + config.S2_TARGET_PCT), 2)
            sl_p = round(entry_p * (1.0 - config.S2_SL_PCT), 2)

            # Late wake-up / restart resolution check
            if candle_ts < ist_now and self.smart_api and c_opt:
                df_history = fetch_real_option_candles(self.smart_api, c_opt, candle_ts, ist_now)
                if not df_history.empty:
                    resolved = False
                    res_exit_p = entry_p
                    res_exit_t = ist_now
                    res_reason = ""
                    for c_ts, c_row in df_history.iterrows():
                        if float(c_row["low"]) <= sl_p:
                            resolved = True
                            res_exit_p = round(sl_p * (1.0 - config.SLIPPAGE_PCT), 2)
                            res_exit_t = c_ts
                            res_reason = f"SL Hit (-{int(config.S2_SL_PCT * 100)}%)"
                            break
                        elif float(c_row["high"]) >= target_p:
                            resolved = True
                            res_exit_p = round(target_p * (1.0 - config.SLIPPAGE_PCT), 2)
                            res_exit_t = c_ts
                            res_reason = f"Target Hit (+{int(config.S2_TARGET_PCT * 100)}%)"
                            break

                    if resolved:
                        gross = (res_exit_p - entry_p) * self.bn_lot_size
                        turn = (entry_p + res_exit_p) * self.bn_lot_size
                        brok = 40.0
                        stt = (res_exit_p * self.bn_lot_size) * config.STT_PCT_ON_SELL
                        exch = turn * config.EXCHANGE_TURNOVER_PCT
                        gst = (brok + exch) * config.GST_PCT
                        charges = round(brok + stt + exch + gst, 2)
                        net = round(gross - charges, 2)
                        sym_str = c_opt["symbol"] if c_opt else f"BANKNIFTY{strike}{opt_type}"
                        tr_log = {
                            "date": str(today_date),
                            "strategy": "Strategy 2: BankNIFTY 15M Directional Breakout",
                            "symbol": sym_str,
                            "entry_time": str(candle_ts),
                            "exit_time": str(res_exit_t),
                            "strike": strike,
                            "opt_type": opt_type,
                            "entry_p": entry_p,
                            "exit_p": res_exit_p,
                            "cost": cost,
                            "gross_pnl": round(gross, 2),
                            "charges": charges,
                            "net_pnl": net,
                            "exit_reason": res_reason,
                            "data_feed": feed_label
                        }
                        self.on_trade_exit("s2", "Strategy 2: BankNIFTY 15M Directional Breakout", tr_log)
                        self.cooldown_until["s2"] = ist_now + timedelta(minutes=30)
                        return

            sym_str = c_opt["symbol"] if c_opt else f"BANKNIFTY{strike}{opt_type}"

            pos_data = {
                "entry_time": candle_ts,
                "spot_entry": c,
                "contract": c_opt,
                "opt_type": opt_type,
                "strike": strike,
                "symbol": sym_str,
                "entry_p": entry_p,
                "cost": cost,
                "target_p": target_p,
                "sl_p": sl_p,
                "data_feed": feed_label
            }
            try:
                send_trade_entry_alert(
                    strategy="Strategy 2: BankNIFTY 15M Directional Breakout",
                    spot=c,
                    opt_type=opt_type,
                    strike=str(sym_str),
                    entry_p=entry_p,
                    ce_str=sym_str if opt_type == "CE" else "",
                    pe_str=sym_str if opt_type == "PE" else "",
                    ce_p=entry_p if opt_type == "CE" else 0.0,
                    pe_p=entry_p if opt_type == "PE" else 0.0,
                    tot_cost=cost,
                    win_target_pct=config.S2_TARGET_PCT,
                    lose_stop_pct=config.S2_SL_PCT,
                    max_hold_mins=360,
                    time_str=candle_ts.strftime("%H:%M:%S") if hasattr(candle_ts, "strftime") else str(candle_ts),
                    qty=self.bn_lot_size,
                    date_str=str(today_date)
                )
            except Exception as e:
                print(f"[WARN] Failed to send S2 entry alert: {e}")
            self.on_trade_entry("s2", "Strategy 2: BankNIFTY 15M Directional Breakout", pos_data)

    def process_live_strategy_3(self, df_all: pd.DataFrame, today_date, ist_now, spot: float):
        """Real-time live position tracking and entry engine for Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)."""
        pos = self.active_positions["s3"]

        if pos is not None:
            elapsed_mins = (ist_now - pos["entry_time"]).total_seconds() / 60.0
            p_c = pos["ce_entry"]
            p_p = pos["pe_entry"]
            high_c = pos.get("high_c", p_c)
            low_c = pos.get("low_c", p_c)
            high_p = pos.get("high_p", p_p)
            low_p = pos.get("low_p", p_p)

            # 1. Priority 1: Instant live LTP from Angel One SmartAPI for both legs
            c_ce = pos.get("c_ce")
            c_pe = pos.get("c_pe")
            ce_live_found = False
            pe_live_found = False

            if self.smart_api and c_ce and c_ce.get("token"):
                try:
                    res_c = self.smart_api.ltpData(c_ce.get("exch_seg", "NFO"), c_ce["symbol"], str(c_ce["token"]))
                    if res_c and res_c.get("status") and res_c.get("data") and float(res_c["data"].get("ltp", 0)) > 0:
                        p_c = round(float(res_c["data"]["ltp"]), 2)
                        high_c = max(high_c, p_c)
                        low_c = min(low_c, p_c)
                        pos["high_c"] = high_c
                        pos["low_c"] = low_c
                        ce_live_found = True
                except Exception:
                    pass

            if self.smart_api and c_pe and c_pe.get("token"):
                try:
                    res_p = self.smart_api.ltpData(c_pe.get("exch_seg", "NFO"), c_pe["symbol"], str(c_pe["token"]))
                    if res_p and res_p.get("status") and res_p.get("data") and float(res_p["data"].get("ltp", 0)) > 0:
                        p_p = round(float(res_p["data"]["ltp"]), 2)
                        high_p = max(high_p, p_p)
                        low_p = min(low_p, p_p)
                        pos["high_p"] = high_p
                        pos["low_p"] = low_p
                        pe_live_found = True
                except Exception:
                    pass

            # 2. Priority 2: Real option candles from Angel One if LTP was unavailable
            if not ce_live_found and self.smart_api and c_ce:
                df_c = fetch_real_option_candles(self.smart_api, c_ce, pos["entry_time"], ist_now)
                if not df_c.empty:
                    p_c = float(df_c.iloc[-1]["close"])
                    high_c = max(high_c, float(df_c["high"].max()))
                    low_c = min(low_c, float(df_c["low"].min()))
                    pos["high_c"] = high_c
                    pos["low_c"] = low_c

            if not pe_live_found and self.smart_api and c_pe:
                df_p = fetch_real_option_candles(self.smart_api, c_pe, pos["entry_time"], ist_now)
                if not df_p.empty:
                    p_p = float(df_p.iloc[-1]["close"])
                    high_p = max(high_p, float(df_p["high"].max()))
                    low_p = min(low_p, float(df_p["low"].min()))
                    pos["high_p"] = high_p
                    pos["low_p"] = low_p

            # 3. Priority 3: BSM fallback if API is completely offline
            if not ce_live_found and not pe_live_found and not self.smart_api:
                t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)
                p_c = black_scholes_price(spot, pos["ce_strike"], t_exp_years, config.RISK_FREE_RATE, 0.135, "CE")
                p_p = black_scholes_price(spot, pos["pe_strike"], t_exp_years, config.RISK_FREE_RATE, 0.135, "PE")
                high_c = max(high_c, p_c)
                low_c = min(low_c, p_c)
                high_p = max(high_p, p_p)
                low_p = min(low_p, p_p)
                pos["high_c"] = high_c
                pos["low_c"] = low_c
                pos["high_p"] = high_p
                pos["low_p"] = low_p

            if not pos["ce_exited"]:
                if low_c <= pos["ce_sl"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = f"CE SL (-{int(config.DAS_LOSE_STOP_PCT * 100)}%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🔴 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (-{int(config.DAS_LOSE_STOP_PCT * 100)}% SL)\n"
                            f"🛡️ *PUT Leg Status:* Still active under decoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━",
                            alert_key=f"LEG_EXIT:{today_date}:s3:ce:{pos['ce_reason']}"
                        )
                elif high_c >= pos["ce_tgt"]:
                    pos["ce_exited"] = True
                    pos["ce_exit_p"] = round(pos["ce_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["ce_reason"] = f"CE Target (+{int(config.DAS_WIN_TARGET_PCT * 100)}%)"
                    self.save_active_positions()
                    if not pos["pe_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🟢 *CALL Leg Exited:* `{pos['ce_symbol']}` @ *₹{pos['ce_exit_p']:.2f}* (+{int(config.DAS_WIN_TARGET_PCT * 100)}% Target)\n"
                            f"🛡️ *PUT Leg Status:* Still active with capital protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━",
                            alert_key=f"LEG_EXIT:{today_date}:s3:ce:{pos['ce_reason']}"
                        )

            if not pos["pe_exited"]:
                if low_p <= pos["pe_sl"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_sl"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = f"PE SL (-{int(config.DAS_LOSE_STOP_PCT * 100)}%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🛑 *LEG STOP LOSS TRIGGERED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (-{int(config.DAS_LOSE_STOP_PCT * 100)}% SL)\n"
                            f"🛡️ *CALL Leg Status:* Still active under decoupled tracking.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━",
                            alert_key=f"LEG_EXIT:{today_date}:s3:pe:{pos['pe_reason']}"
                        )
                elif high_p >= pos["pe_tgt"]:
                    pos["pe_exited"] = True
                    pos["pe_exit_p"] = round(pos["pe_tgt"] * (1.0 - config.SLIPPAGE_PCT), 2)
                    pos["pe_reason"] = f"PE Target (+{int(config.DAS_WIN_TARGET_PCT * 100)}%)"
                    self.save_active_positions()
                    if not pos["ce_exited"]:
                        send_telegram_alert(
                            f"🎯 *LEG PROFIT TARGET HIT*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⏰ *Time:* {ist_now.strftime('%H:%M:%S')} IST\n"
                            f"📊 *Strategy:* `Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)`\n"
                            f"🔴 *PUT Leg Exited:* `{pos['pe_symbol']}` @ *₹{pos['pe_exit_p']:.2f}* (+{int(config.DAS_WIN_TARGET_PCT * 100)}% Target)\n"
                            f"🛡️ *CALL Leg Status:* Still active with capital protection.\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━",
                            alert_key=f"LEG_EXIT:{today_date}:s3:pe:{pos['pe_reason']}"
                        )

            exit_c = pos["ce_exit_p"] if pos["ce_exited"] else round(p_c * (1.0 - config.SLIPPAGE_PCT), 2)
            exit_p = pos["pe_exit_p"] if pos["pe_exited"] else round(p_p * (1.0 - config.SLIPPAGE_PCT), 2)
            cur_tot = exit_c + exit_p
            comb_ret = (cur_tot - pos["entry_tot"]) / max(0.1, pos["entry_tot"])
            is_time_up = (elapsed_mins >= config.DAS_MAX_HOLD_MINS) or (ist_now.time() >= dtime(15, 5))
            is_comb_stop = comb_ret <= -config.DAS_COMBINED_STOP_PCT

            if (pos["ce_exited"] and pos["pe_exited"]) or is_time_up or is_comb_stop:
                if is_comb_stop:
                    exit_reason = f"Combined Capital SL (-{int(config.DAS_COMBINED_STOP_PCT * 100)}%)"
                elif pos["ce_exited"] and pos["pe_exited"]:
                    exit_reason = f"{pos['ce_reason']} | {pos['pe_reason']}"
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
                    "strategy": "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)",
                    "opt_type": "CE+PE",
                    "strike": f"{pos['ce_symbol']} + {pos['pe_symbol']}",
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                    "exit_time": ist_now.strftime("%H:%M:%S"),
                    "entry_p": pos["entry_tot"],
                    "exit_p": round(cur_tot, 2),
                    "cost": pos["cost"],
                    "capital_used": pos["cost"],
                    "gross_pnl": round(gross, 2),
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": exit_reason,
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                try:
                    send_trade_exit_alert(
                        strategy="Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)",
                        exit_reason=exit_reason,
                        gross_pnl=round(gross, 2),
                        charges=charges,
                        net_pnl=net,
                        details=f"{pos['ce_symbol']} + {pos['pe_symbol']} | Entry: ₹{pos['entry_tot']:.2f} -> Exit: ₹{cur_tot:.2f}",
                        time_str=ist_now.strftime("%H:%M:%S"),
                        date_str=str(today_date),
                        entry_time=str(tr_log["entry_time"]),
                        strike=f"{pos['ce_strike']} CE / {pos['pe_strike']} PE"
                    )
                except Exception as e:
                    print(f"[WARN] Failed to send S3 exit alert: {e}")
                self.on_trade_exit("s3", "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)", tr_log)
                self.cooldown_until["s3"] = ist_now + timedelta(minutes=config.DAS_COOLDOWN_BARS)
                return

        else:
            if not config.S3_ENABLED:
                return
            active_count = sum(1 for p in self.active_positions.values() if p is not None)
            if active_count >= config.MAX_CONCURRENT_POSITIONS:
                return
            if self.daily_trade_count["s3"] >= config.MAX_TRADES_PER_DAY:
                return
            if self.cooldown_until["s3"] and ist_now < self.cooldown_until["s3"]:
                return
            if not is_trade_window_valid(ist_now.time(), config.DAS_TRADE_START, config.DAS_TRADE_END):
                return
            if df_all.empty:
                return

            t_exp_years = max(1e-4, compute_calendar_dte(ist_now) / 365.0)

            strike_res = select_affordable_strikes(
                spot=spot,
                current_time=ist_now,
                T_years=t_exp_years,
                min_prem=config.DAS_MIN_LEG_PREMIUM,
                max_prem=config.DAS_MAX_LEG_PREMIUM,
                max_budget=config.DAS_MAX_BUDGET
            )
            if not strike_res:
                return
            ce_strike, pe_strike, bsm_ce, bsm_pe, _ = strike_res

            day_bars = df_all[df_all["timestamp"].dt.date == today_date].sort_values("timestamp")
            if len(day_bars) < config.DAS_VOL_WINDOW:
                return

            recent_spots = day_bars["close"].iloc[-config.DAS_VOL_WINDOW:].values
            comb_hist = []
            for s_val in recent_spots:
                cp = black_scholes_price(s_val, ce_strike, t_exp_years, config.RISK_FREE_RATE, 0.135, "CE")
                pp = black_scholes_price(s_val, pe_strike, t_exp_years, config.RISK_FREE_RATE, 0.135, "PE")
                comb_hist.append(cp + pp)
            comb_hist = np.array(comb_hist)

            triggered, std_val, vel_val = check_compression_expansion(
                comb_hist,
                std_thresh=config.DAS_STD_THRESH,
                vel_thresh=config.DAS_VEL_THRESH
            )
            if not triggered:
                return

            c_ce = get_active_option_contract(today_date, ce_strike, "CE", underlying="NIFTY") if self.smart_api else None
            c_pe = get_active_option_contract(today_date, pe_strike, "PE", underlying="NIFTY") if self.smart_api else None
            ce_entry_p = bsm_ce
            pe_entry_p = bsm_pe
            feed_label = "Mathematical (BSM)"

            if self.smart_api and c_ce and c_pe:
                df_c = fetch_real_option_candles(self.smart_api, c_ce, ist_now - timedelta(minutes=10), ist_now)
                df_p = fetch_real_option_candles(self.smart_api, c_pe, ist_now - timedelta(minutes=10), ist_now)
                if not df_c.empty and not df_p.empty:
                    ce_entry_p = float(df_c.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT)
                    pe_entry_p = float(df_p.iloc[-1]["close"]) * (1.0 + config.SLIPPAGE_PCT)
                    feed_label = f"Angel One Real Traded ({c_ce['symbol']} + {c_pe['symbol']})"

            entry_tot = round(ce_entry_p + pe_entry_p, 2)
            tot_cost = round(entry_tot * self.lot_size, 2)
            if tot_cost > config.MAX_CAPITAL_PER_TRADE:
                return

            total_deployed = sum(p.get("cost", 0.0) for p in self.active_positions.values() if p is not None)
            if (total_deployed + tot_cost) > config.INITIAL_CAPITAL:
                return


            ce_tgt = round(ce_entry_p * (1.0 + config.DAS_WIN_TARGET_PCT), 2)
            ce_sl = round(ce_entry_p * (1.0 - config.DAS_LOSE_STOP_PCT), 2)
            pe_tgt = round(pe_entry_p * (1.0 + config.DAS_WIN_TARGET_PCT), 2)
            pe_sl = round(pe_entry_p * (1.0 - config.DAS_LOSE_STOP_PCT), 2)

            pos_data = {
                "entry_time": ist_now,
                "spot_entry": spot,
                "c_ce": c_ce,
                "c_pe": c_pe,
                "ce_strike": ce_strike,
                "pe_strike": pe_strike,
                "ce_symbol": c_ce["symbol"] if c_ce else f"NIFTY{ce_strike}CE",
                "pe_symbol": c_pe["symbol"] if c_pe else f"NIFTY{pe_strike}PE",
                "ce_entry": round(ce_entry_p, 2),
                "pe_entry": round(pe_entry_p, 2),
                "entry_tot": entry_tot,
                "cost": tot_cost,
                "ce_tgt": ce_tgt,
                "ce_sl": ce_sl,
                "pe_tgt": pe_tgt,
                "pe_sl": pe_sl,
                "ce_exited": False,
                "pe_exited": False,
                "ce_exit_p": None,
                "pe_exit_p": None,
                "ce_reason": None,
                "pe_reason": None,
                "data_feed": feed_label
            }
            try:
                send_trade_entry_alert(
                    strategy="Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)",
                    spot=spot,
                    ce_str=f"{ce_strike} CE ({c_ce['symbol'] if c_ce else ''})",
                    pe_str=f"{pe_strike} PE ({c_pe['symbol'] if c_pe else ''})",
                    ce_p=ce_entry_p,
                    pe_p=pe_entry_p,
                    tot_cost=tot_cost,
                    win_target_pct=config.DAS_WIN_TARGET_PCT,
                    lose_stop_pct=config.DAS_LOSE_STOP_PCT,
                    max_hold_mins=config.DAS_MAX_HOLD_MINS,
                    time_str=ist_now.strftime("%H:%M:%S"),
                    qty=self.lot_size,
                    date_str=str(today_date)
                )
            except Exception as e:
                print(f"[WARN] Failed to send S3 entry alert: {e}")
            self.on_trade_entry("s3", "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)", pos_data)

    def run_live_loop(self):
        """Continuously monitors live market during trading hours across all 3 active strategies in real-time."""
        print("\n" + "=" * 85)
        print("   MULTI-INDEX LIVE MARKET TRADING ENGINE ACTIVE (TRI-STRATEGY CORE SYSTEM)")
        print("   Active Strategies: S1 (Nifty Directional), S2 (BankNifty Directional), S3 (Nifty DAS)")
        print("=" * 85)

        ist_now = get_ist_now()
        today_date = ist_now.date()
        self.init_today_state(today_date)

        send_telegram_alert(
            f"🚀 *Option Trading Bot Woke Up (Tri-Strategy Core System Active)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Session Time:* {ist_now.strftime('%I:%M:%S %p')} IST\n"
            f"📅 *Date:* {ist_now.strftime('%A, %d-%b-%Y')}\n\n"
            f"Bot is actively monitoring the optimized Tri-Strategy portfolio:\n"
            f"• Strategy 1: Nifty 15M Directional Breakout (Trend-Aligned ATM, SL -16%, TP +60%)\n"
            f"• Strategy 2: BankNIFTY 15M Directional Breakout (Trend-Aligned 100-pt OTM, SL -25%, TP +80%)\n"
            f"• Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS - Vol Compression & Expansion)\n\n"
            f"🛡️ *Position Rule:* Max 3 active positions concurrently (₹10,000 capital per trade limit)\n"
            f"🔔 Instant alerts will be sent here the exact second a trade enters and exits!\n\n"
            f"💡 *Live Bot Commands:* Send /status, /today, or /pnl anytime for live updates.",
            alert_key=f"WAKEUP:{today_date}",
            force=False
        )

        while get_ist_now().time() < dtime(15, 5):
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

            time.sleep(3.0)  # Enforce 3.0s pause to prevent Angel One rate limit between instruments

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

            # Execute real-time tracking across active Tri-Strategy core
            if config.S1_ENABLED and self.target_strategy in ["all", "s1", "directional"] and not df_all.empty:
                try:
                    self.process_live_strategy_1(df_all, today_date, ist_now, spot)
                except Exception as e:
                    print(f"[WARN] Strategy 1 live error: {e}")

            if config.S2_ENABLED and self.target_strategy in ["all", "s2", "directional"] and not df_bn.empty:
                try:
                    self.process_live_strategy_2(df_bn, today_date, ist_now, bn_spot)
                except Exception as e:
                    print(f"[WARN] Strategy 2 live error: {e}")

            if config.S3_ENABLED and self.target_strategy in ["all", "s3", "das"] and not df_all.empty:
                try:
                    self.process_live_strategy_3(df_all, today_date, ist_now, spot)
                except Exception as e:
                    print(f"[WARN] Strategy 3 live error: {e}")

            time.sleep(60)

        # Market Close Procedure at 15:05 IST
        print("[MARKET CLOSE] Squareoff time reached (15:05 IST). Performing EOD audit...")
        self.run_eod_accounting()

    def run_eod_accounting(self):
        """Runs end-of-day result check and logs to journal and Telegram."""
        if self.smart_api:
            try:
                print("[EOD] Syncing final candles from Angel One before audit...")
                sync_and_update_candles(self.smart_api, CANDLE_FILE, "99926000")
                time.sleep(1.5)
                sync_and_update_candles(self.smart_api, BANKNIFTY_CANDLE_FILE, "99926009")
            except Exception as e:
                print(f"[WARN] EOD candle sync error: {e}")

        df_all = pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame()
        df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame()
        today = get_ist_now().date()
        spot = self.get_nifty_spot_price()
        bn_spot = self.get_banknifty_spot_price()

        # Force square off any remaining active positions at 15:05 IST
        for s_key in ["s1", "s2", "s3"]:
            pos = self.active_positions.get(s_key)
            if pos is not None:
                print(f"[EOD SQUAREOFF] Closing open position for {s_key} at market cutoff.")
                s_name = pos.get("strategy_name", f"Strategy {s_key.upper()}")
                cur_p = pos.get("entry_p", 100.0)
                lot = self.bn_lot_size if "banknifty" in s_name.lower() else self.lot_size
                gross = 0.0
                charges = 40.0
                net = -charges
                tr_log = {
                    "date": str(today),
                    "strategy": s_name,
                    "leg": pos.get("opt_type", "CE"),
                    "strike": pos.get("symbol", ""),
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S") if hasattr(pos["entry_time"], "strftime") else str(pos["entry_time"]),
                    "exit_time": "15:05:00",
                    "entry_p": cur_p,
                    "exit_p": cur_p,
                    "cost": pos.get("cost", 0.0),
                    "capital_used": pos.get("cost", 0.0),
                    "gross_pnl": 0.0,
                    "charges": charges,
                    "net_pnl": net,
                    "exit_reason": "EOD Squareoff (15:05)",
                    "data_feed": pos.get("data_feed", "Angel One Real Traded")
                }
                try:
                    send_trade_exit_alert(
                        strategy=s_name,
                        exit_reason="EOD Squareoff (15:05)",
                        gross_pnl=0.0,
                        charges=charges,
                        net_pnl=net,
                        details=f"{pos.get('symbol', '')} | EOD Market Squareoff @ ₹{cur_p:.2f}",
                        time_str="15:05:00",
                        date_str=str(today),
                        entry_time=str(tr_log["entry_time"]),
                        strike=str(pos.get("symbol", ""))
                    )
                except Exception as e:
                    print(f"[WARN] Failed to send EOD exit alert: {e}")
                self.on_trade_exit(s_key, s_name, tr_log)

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
            active_evals = []
            if not df_all.empty:
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"]).dt.tz_localize(None)
                if config.S1_ENABLED:
                    s1_res = evaluate_strategy_1(df_all, today, api=self.smart_api)
                    active_evals.append(("Strategy 1: Nifty 15M Directional Breakout", s1_res))
                if config.S3_ENABLED:
                    s3_res = evaluate_strategy_3(df_all, today, api=self.smart_api)
                    active_evals.append(("Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)", s3_res))

            if not df_bn.empty:
                df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
                if config.S2_ENABLED:
                    s2_res = evaluate_strategy_2(df_bn, today, api=self.smart_api)
                    active_evals.append(("Strategy 2: BankNIFTY 15M Directional Breakout", s2_res))

            for s_name, s_res in active_evals:
                if s_res.get("trade_occurred", False):
                    trades_today += s_res.get("num_trades_day", 1)
                    daily_net += s_res.get("net_pnl", 0.0)
                    trades_to_notify = s_res["all_trades"] if ("all_trades" in s_res and len(s_res["all_trades"]) > 1) else [s_res]
                    for sub_tr in trades_to_notify:
                        log_trade_to_journal(today, s_name, sub_tr)

                        sub_opt = sub_tr.get("opt_type", sub_tr.get("leg", ""))
                        sub_stk = str(sub_tr.get("strike", ""))
                        sub_ep = float(sub_tr.get("entry_p", 0.0))
                        sub_c = float(sub_tr.get("cost", sub_tr.get("capital_used", 0.0)))
                        sub_t = str(sub_tr.get("entry_time", ""))

                        # Send Entry Alert if not already sent
                        send_trade_entry_alert(
                            strategy=s_name,
                            spot=float(sub_tr.get("spot_entry", spot if "Nifty 15M" in s_name else bn_spot)),
                            opt_type=sub_opt,
                            strike=sub_stk,
                            entry_p=sub_ep,
                            ce_str=sub_stk if "CE" in sub_opt else "",
                            pe_str=sub_stk if "PE" in sub_opt else "",
                            ce_p=sub_ep if "CE" in sub_opt else 0.0,
                            pe_p=sub_ep if "PE" in sub_opt else 0.0,
                            tot_cost=sub_c,
                            win_target_pct=config.S1_TARGET_PCT if "Strategy 1" in s_name else (config.S2_TARGET_PCT if "Strategy 2" in s_name else 0.60),
                            lose_stop_pct=config.S1_SL_PCT if "Strategy 1" in s_name else (config.S2_SL_PCT if "Strategy 2" in s_name else 0.12),
                            max_hold_mins=360,
                            time_str=sub_t,
                            qty=self.bn_lot_size if "BankNIFTY" in s_name else self.lot_size,
                            date_str=str(today)
                        )

                        send_trade_exit_alert(
                            strategy=s_name,
                            exit_reason=sub_tr.get("exit_reason", "EOD Reconciliation"),
                            gross_pnl=sub_tr.get("gross_pnl", sub_tr.get("gross", 0.0)),
                            charges=sub_tr.get("charges", 0.0),
                            net_pnl=sub_tr.get("net_pnl", sub_tr.get("net", 0.0)),
                            details=f"{sub_stk} | Entry: ₹{sub_ep:.2f} ({sub_t}) -> Exit: ₹{sub_tr.get('exit_p', 0):.2f} ({sub_tr.get('exit_time', '')})",
                            time_str=str(sub_tr.get("exit_time", "")),
                            date_str=str(today),
                            entry_time=sub_t,
                            strike=sub_stk
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

        summary_key = f"SUMMARY:{today}"
        if not is_alert_already_sent(summary_key):
            send_daily_summary_alert(
                date_str=today.strftime("%d-%b-%Y"),
                trades_count=trades_today,
                daily_pnl=daily_net,
                total_pnl=total_pnl,
                current_balance=current_balance
            )
            mark_alert_as_sent(summary_key)
        else:
            print(f"[EOD] Daily summary for {today} was already sent. Skipping duplicate.")

    def run_replay_demonstration(self):
        """Simulates all 3 strategies on the latest market session for offline testing."""
        print("=" * 85)
        print("   TRI-STRATEGY REPLAY DEMONSTRATION & TELEGRAM TEST")
        print("=" * 85)
        df_n = pd.read_csv("data/nifty_1min_real.csv") if os.path.exists("data/nifty_1min_real.csv") else pd.DataFrame()
        df_b = pd.read_csv("data/banknifty_1min_real.csv") if os.path.exists("data/banknifty_1min_real.csv") else pd.DataFrame()

        if not df_n.empty:
            df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
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
        s2 = evaluate_strategy_2(df_b, last_date, api=self.smart_api) if not df_b.empty else {}
        s3 = evaluate_strategy_3(df_n, last_date, api=self.smart_api) if not df_n.empty else {}

        print(f"\n[EVALUATION RESULTS FOR {last_date}]")
        print(f"Strategy 1 (Nifty 15M Directional Breakout)       : {s1.get('status')} | Trade: {s1.get('trade_occurred')}")
        print(f"Strategy 2 (BankNIFTY 15M Directional Breakout)   : {s2.get('status')} | Trade: {s2.get('trade_occurred')}")
        print(f"Strategy 3 (Nifty Decoupled Asymmetric Strangle)  : {s3.get('status')} | Trade: {s3.get('trade_occurred')}")

    def run(self):
        """Master execution entry point."""
        now = get_ist_now()

        # Offline / Replay test
        if hasattr(self, "replay_mode") and self.replay_mode:
            print("[MODE] Running offline demonstration / replay test...")
            self.run_replay_demonstration()
            return

        print(f"\n[STARTUP] Option Trading Engine Initialized at {now.strftime('%Y-%m-%d %H:%M:%S')} IST.")
        self.connect_angel_one()

        # Start interactive Telegram listener daemon in background
        try:
            self.tg_thread, self.tg_stop = start_interactive_bot_daemon(api=self.smart_api)
            print("[SUCCESS] Interactive Telegram bot listener daemon started.")
        except Exception as e:
            print(f"[WARN] Could not start interactive Telegram bot daemon: {e}")

        if self.is_market_open_now():
            print("[MARKET STATUS] Live NSE market is OPEN right now.")
            self.run_live_loop()
        else:
            if now.weekday() in [5, 6]:
                print(f"[MARKET CLOSED] Today is {now.strftime('%A')} (NSE Weekend). Markets are closed.")
                print("Running demonstration replay on the latest available market session...")
                self.run_replay_demonstration()
            elif now.time() < dtime(9, 15):
                print(f"[PRE-MARKET] Current time is {now.strftime('%H:%M:%S')} IST. Market opens at 09:15:00 AM IST.")
                waited = self.wait_for_market_open()
                if waited:
                    self.run_live_loop()
                else:
                    print("Early standby window exceeded. Replay test:")
                    self.run_replay_demonstration()
            else:
                print(f"[POST-MARKET] Current time is {now.strftime('%H:%M:%S')} IST (after 15:25 IST market close).")
                print("Performing end-of-day session audit and reconciliation...")
                self.run_eod_accounting()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Index Tri-Strategy Live Paper Trading Bot")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "s1", "s2", "s3"], help="Strategy to trade (default: all)")
    parser.add_argument("--replay", action="store_true", help="Run offline replay on latest recorded day instead of live loop")
    args = parser.parse_args()

    trader = LiveTriPaperTrader(target_strategy=args.strategy)
    if args.replay:
        trader.replay_mode = True
    trader.run()
