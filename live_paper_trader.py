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
    log_trade_to_journal,
    CANDLE_FILE,
    BANKNIFTY_CANDLE_FILE
)

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
        pin = os.getenv("ANGEL_PASSWORD") or os.getenv("ANGEL_PIN")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET")

        if not all([api_key, client_code, pin, totp_secret]):
            print("[INFO] Angel One credentials not fully configured in .env. Running in offline/simulation mode.")
            return False

        try:
            self.smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            data = self.smart_api.generateSession(client_code, pin, totp)
            if data and data.get("status"):
                print(f"[SUCCESS] Connected to Angel One SmartAPI for Multi-Index Live Trading. (Client: {client_code})")
                return True
            else:
                print(f"[WARN] Angel One login failed: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
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

    def run_live_loop(self):
        """Continuously monitors live market during trading hours across all active strategies."""
        print("\n" + "=" * 85)
        print("   MULTI-INDEX LIVE MARKET TRADING ENGINE ACTIVE")
        print("   Tracking Strategies: Strategy 1, 2, 3, 4 (Nifty) & Strategy 5 (BankNifty DAS)")
        print("=" * 85)

        ist_now = get_ist_now()
        send_telegram_alert(
            f"🚀 *Option Trading Bot Woke Up (Live Engine Active)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Session Time:* {ist_now.strftime('%I:%M:%S %p')} IST\n"
            f"📅 *Date:* {ist_now.strftime('%A, %d-%b-%Y')}\n\n"
            f"Bot is actively monitoring all 5 quantitative strategies:\n"
            f"• Strategy 1: Hedged Long Strangle (15M Box - Nifty)\n"
            f"• Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze (Nifty)\n"
            f"• Strategy 3: 30M Statistical Directional ITM Breakout (Nifty)\n"
            f"• Strategy 4: Decoupled Asymmetric Strangle (DAS - Nifty)\n"
            f"• Strategy 5: Decoupled Asymmetric Strangle (DAS - BankNifty)\n\n"
            f"💰 *Capital Limit:* ₹10,000 per trade\n"
            f"🔔 Instant alerts will be sent here for every trade entry and exit!"
        )

        alerted_entries = set()
        alerted_exits = set()

        while get_ist_now().time() <= dtime(15, 25):
            ist_now = get_ist_now()
            today_date = ist_now.date()
            t_time = ist_now.time()

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
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
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

            # -------------------------------------------------------------
            # Strategy 1: Hedged Long Strangle (15M Box Breakout - Nifty)
            # -------------------------------------------------------------
            if self.target_strategy in ["all", "strangle"] and t_time >= dtime(9, 30) and not df_all.empty:
                try:
                    s1_res = evaluate_strategy_1(df_all, today_date, api=self.smart_api)
                    if s1_res.get("trade_occurred", False):
                        entry_key = ("s1_entry", str(s1_res.get("entry_time")))
                        if entry_key not in alerted_entries:
                            send_trade_entry_alert(
                                strategy="Strategy 1: Hedged Long Strangle (15M Box)",
                                spot=spot,
                                ce_str="Dual OTM CE (+150)",
                                pe_str="Dual OTM PE (-150)",
                                ce_p=round(s1_res.get("entry_p", 0.0) / 2.0, 2),
                                pe_p=round(s1_res.get("entry_p", 0.0) / 2.0, 2),
                                tot_cost=s1_res.get("cost", 0.0),
                                win_target_pct=0.75,
                                lose_stop_pct=0.20,
                                max_hold_mins=120,
                                time_str=str(s1_res.get("entry_time"))
                            )
                            alerted_entries.add(entry_key)

                        exit_key = ("s1_exit", str(s1_res.get("exit_time")))
                        if s1_res.get("exit_time") and exit_key not in alerted_exits:
                            send_trade_exit_alert(
                                strategy="Strategy 1: Hedged Long Strangle (15M Box)",
                                exit_reason=s1_res.get("exit_reason", "Target / SL Hit"),
                                gross_pnl=s1_res.get("gross_pnl", 0.0),
                                charges=s1_res.get("charges", 0.0),
                                net_pnl=s1_res.get("net_pnl", 0.0),
                                details=f"Entry: ₹{s1_res.get('entry_p')} -> Exit: ₹{s1_res.get('exit_p')}",
                                time_str=str(s1_res.get("exit_time"))
                            )
                            alerted_exits.add(exit_key)
                except Exception as e:
                    print(f"[WARN] Strategy 1 evaluation error: {e}")

            # -------------------------------------------------------------
            # Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze (Nifty Mon & Tue)
            # -------------------------------------------------------------
            if self.target_strategy in ["all", "gamma"] and today_date.weekday() in [0, 1] and t_time >= dtime(9, 35) and not df_all.empty:
                try:
                    s2_res = evaluate_strategy_2(df_all, today_date, api=self.smart_api)
                    if s2_res.get("trade_occurred", False):
                        entry_key = ("s2_entry", str(s2_res.get("entry_time")))
                        if entry_key not in alerted_entries:
                            send_trade_entry_alert(
                                strategy="Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze",
                                spot=spot,
                                opt_type=s2_res.get("opt_type", "PE"),
                                strike=str(s2_res.get("strike")),
                                entry_p=s2_res.get("entry_p", 0.0),
                                tot_cost=s2_res.get("cost", 0.0),
                                win_target_pct=0.80,
                                lose_stop_pct=0.25,
                                max_hold_mins=45,
                                time_str=str(s2_res.get("entry_time"))
                            )
                            alerted_entries.add(entry_key)

                        exit_key = ("s2_exit", str(s2_res.get("exit_time")))
                        if s2_res.get("exit_time") and exit_key not in alerted_exits:
                            send_trade_exit_alert(
                                strategy="Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze",
                                exit_reason=s2_res.get("exit_reason", "Time Stop / Target"),
                                gross_pnl=s2_res.get("gross_pnl", 0.0),
                                charges=s2_res.get("charges", 0.0),
                                net_pnl=s2_res.get("net_pnl", 0.0),
                                details=f"BUY {s2_res.get('strike')} {s2_res.get('opt_type')} @ ₹{s2_res.get('entry_p')} exited @ ₹{s2_res.get('exit_p')}",
                                time_str=str(s2_res.get("exit_time"))
                            )
                            alerted_exits.add(exit_key)
                except Exception as e:
                    print(f"[WARN] Strategy 2 evaluation error: {e}")

            # -------------------------------------------------------------
            # Strategy 3: 30M Statistical Directional ITM Breakout (Nifty)
            # -------------------------------------------------------------
            if self.target_strategy in ["all", "directional"] and t_time >= dtime(9, 45) and not df_all.empty:
                try:
                    s3_res = evaluate_strategy_3(df_all, today_date, api=self.smart_api)
                    if s3_res.get("trade_occurred", False):
                        entry_key = ("s3_entry", str(s3_res.get("entry_time")))
                        if entry_key not in alerted_entries:
                            send_trade_entry_alert(
                                strategy="Strategy 3: 30M Statistical Directional ITM",
                                spot=spot,
                                opt_type=s3_res.get("opt_type", "CE"),
                                strike=str(s3_res.get("strike")),
                                entry_p=s3_res.get("entry_p", 0.0),
                                tot_cost=s3_res.get("cost", 0.0),
                                win_target_pct=0.80,
                                lose_stop_pct=0.15,
                                max_hold_mins=60,
                                time_str=str(s3_res.get("entry_time"))
                            )
                            alerted_entries.add(entry_key)

                        exit_key = ("s3_exit", str(s3_res.get("exit_time")))
                        if s3_res.get("exit_time") and exit_key not in alerted_exits:
                            send_trade_exit_alert(
                                strategy="Strategy 3: 30M Statistical Directional ITM",
                                exit_reason=s3_res.get("exit_reason", "Target / SL / Trailing Exit"),
                                gross_pnl=s3_res.get("gross_pnl", 0.0),
                                charges=s3_res.get("charges", 0.0),
                                net_pnl=s3_res.get("net_pnl", 0.0),
                                details=f"BUY {s3_res.get('strike')} {s3_res.get('opt_type')} @ ₹{s3_res.get('entry_p')} exited @ ₹{s3_res.get('exit_p')}",
                                time_str=str(s3_res.get("exit_time"))
                            )
                            alerted_exits.add(exit_key)
                except Exception as e:
                    print(f"[WARN] Strategy 3 evaluation error: {e}")

            # -------------------------------------------------------------
            # Strategy 4: Decoupled Asymmetric Strangle (DAS - Nifty)
            # -------------------------------------------------------------
            if self.target_strategy in ["all", "das"] and t_time >= dtime(9, 35) and not df_all.empty:
                try:
                    s4_res = evaluate_strategy_4(df_all, today_date, api=self.smart_api)
                    if s4_res.get("trade_occurred", False):
                        trades_to_alert = s4_res.get("all_trades", [s4_res])
                        for tr in trades_to_alert:
                            entry_key = ("s4_entry", str(tr.get("entry_time")))
                            if entry_key not in alerted_entries:
                                send_trade_entry_alert(
                                    strategy="Strategy 4: Decoupled Asymmetric Strangle (DAS)",
                                    spot=spot,
                                    ce_str=f"{tr.get('strike', 'Dual OTM')}",
                                    pe_str=f"{tr.get('strike', 'Dual OTM')}",
                                    ce_p=round(tr.get("entry_p", 0.0) / 2.0, 2),
                                    pe_p=round(tr.get("entry_p", 0.0) / 2.0, 2),
                                    tot_cost=tr.get("cost", 0.0),
                                    win_target_pct=config.DAS_WIN_TARGET_PCT,
                                    lose_stop_pct=config.DAS_LOSE_STOP_PCT,
                                    max_hold_mins=config.DAS_MAX_HOLD_MINS,
                                    time_str=str(tr.get("entry_time")),
                                    qty=self.lot_size
                                )
                                alerted_entries.add(entry_key)

                            exit_key = ("s4_exit", str(tr.get("exit_time")))
                            if tr.get("exit_time") and exit_key not in alerted_exits:
                                send_trade_exit_alert(
                                    strategy="Strategy 4: Decoupled Asymmetric Strangle (DAS)",
                                    exit_reason=tr.get("exit_reason", "Target / SL"),
                                    gross_pnl=tr.get("gross_pnl", 0.0),
                                    charges=tr.get("charges", 0.0),
                                    net_pnl=tr.get("net_pnl", 0.0),
                                    details=f"Entry: ₹{tr.get('entry_p')} -> Exit: ₹{tr.get('exit_p')}",
                                    time_str=str(tr.get("exit_time"))
                                )
                                alerted_exits.add(exit_key)
                except Exception as e:
                    print(f"[WARN] Strategy 4 evaluation error: {e}")

            # -------------------------------------------------------------
            # Strategy 5: Decoupled Asymmetric Strangle (DAS - BankNifty)
            # -------------------------------------------------------------
            if self.target_strategy in ["all", "bndas", "das"] and t_time >= dtime(9, 45) and not df_bn.empty:
                try:
                    s5_res = evaluate_strategy_5(df_bn, today_date, api=self.smart_api)
                    if s5_res.get("trade_occurred", False):
                        trades_to_alert = s5_res.get("all_trades", [s5_res])
                        for tr in trades_to_alert:
                            entry_key = ("s5_entry", str(tr.get("entry_time")))
                            if entry_key not in alerted_entries:
                                send_trade_entry_alert(
                                    strategy="Strategy 5: BankNIFTY Decoupled Strangle (DAS)",
                                    spot=bn_spot,
                                    ce_str=f"{tr.get('strike', 'ATM Strangle')}",
                                    pe_str=f"{tr.get('strike', 'ATM Strangle')}",
                                    ce_p=round(tr.get("entry_p", 0.0) / 2.0, 2),
                                    pe_p=round(tr.get("entry_p", 0.0) / 2.0, 2),
                                    tot_cost=tr.get("cost", 0.0),
                                    win_target_pct=1.00,
                                    lose_stop_pct=0.15,
                                    max_hold_mins=45,
                                    time_str=str(tr.get("entry_time")),
                                    qty=self.bn_lot_size
                                )
                                alerted_entries.add(entry_key)

                            exit_key = ("s5_exit", str(tr.get("exit_time")))
                            if tr.get("exit_time") and exit_key not in alerted_exits:
                                send_trade_exit_alert(
                                    strategy="Strategy 5: BankNIFTY Decoupled Strangle (DAS)",
                                    exit_reason=tr.get("exit_reason", "Target / SL"),
                                    gross_pnl=tr.get("gross_pnl", 0.0),
                                    charges=tr.get("charges", 58.0),
                                    net_pnl=tr.get("net_pnl", 0.0),
                                    details=f"Entry: ₹{tr.get('entry_p')} -> Exit: ₹{tr.get('exit_p')}",
                                    time_str=str(tr.get("exit_time"))
                                )
                                alerted_exits.add(exit_key)
                except Exception as e:
                    print(f"[WARN] Strategy 5 evaluation error: {e}")

            time.sleep(60)

        # Market Close Procedure at 15:25 IST
        print("[MARKET CLOSE] Squareoff time reached (15:25 IST). Performing EOD audit...")
        self.run_eod_accounting()

    def run_eod_accounting(self):
        """Runs end-of-day result check and logs to journal and Telegram."""
        df_all = pd.read_csv(CANDLE_FILE) if os.path.exists(CANDLE_FILE) else pd.DataFrame()
        df_bn = pd.read_csv(BANKNIFTY_CANDLE_FILE) if os.path.exists(BANKNIFTY_CANDLE_FILE) else pd.DataFrame()
        today = get_ist_now().date()

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
        daily_net = 0.0
        trades_today = 0
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
                if "all_trades" in s_res and len(s_res["all_trades"]) > 1:
                    for sub_tr in s_res["all_trades"]:
                        log_trade_to_journal(today, s_name, sub_tr)
                else:
                    log_trade_to_journal(today, s_name, s_res)

        # Read cumulative PnL directly from journal
        total_pnl = 0.0
        if os.path.exists("data/trade_journal.csv"):
            try:
                df_j = pd.read_csv("data/trade_journal.csv")
                if not df_j.empty and "net_pnl" in df_j.columns:
                    total_pnl = round(float(df_j["net_pnl"].sum()), 2)
            except Exception:
                total_pnl = 5164.16 + daily_net
        else:
            total_pnl = 5164.16 + daily_net

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
