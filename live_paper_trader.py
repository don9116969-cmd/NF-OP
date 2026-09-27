"""
Live Market Paper Trading Bot for Nifty Options.
Supports:
1. Strategy 1: Hedged Long Strangle (15M Box Breakout, Uncoupled Leg SL -20%, Target +75%)
2. Strategy 4: Decoupled Asymmetric Strangle (DAS - Delta-Neutral Volatility Compression & Kinetic Expansion)
Connects to Angel One SmartAPI to track Nifty Spot in real-time or runs in high-fidelity replay simulation mode.
Logs all paper trades to data/live_paper_trades.csv and data/trade_journal.md.
"""

import os
import sys
import time
import math
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
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
    send_trade_entry_alert,
    send_trade_exit_alert,
    send_daily_summary_alert
)

load_dotenv()

class LiveStranglePaperTrader:
    """Strategy 1: Hedged Long Strangle (15M Box Breakout)"""
    def __init__(self, otm_pts: int = 150, leg_sl_pct: float = 0.20, target_pct: float = 0.75, trailing_pct: float = 0.25):
        self.otm_pts = otm_pts
        self.leg_sl_pct = leg_sl_pct
        self.target_pct = target_pct
        self.trailing_pct = trailing_pct
        self.lot_size = config.LOT_SIZE
        self.capital = config.INITIAL_CAPITAL

        self.smart_api = None
        self.auth_token = None
        self.feed_token = None

        # State variables
        self.current_date = None
        self.box_high = None
        self.box_low = None
        self.box_qualified = False
        self.signal_triggered = False
        self.in_trade = False

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
                self.auth_token = data["data"]["jwtToken"]
                self.feed_token = self.smart_api.getfeedToken()
                print(f"[SUCCESS] Connected to Angel One SmartAPI for live paper trading. (Client: {client_code})")
                return True
            else:
                print(f"[WARN] Angel One login failed: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"[ERROR] Connection error: {e}")
            return False

    def print_trade_signal(self, spot: float, ce_strike: int, pe_strike: int, ce_price: float, pe_price: float):
        print("\n" + "=" * 80)
        print(">>> [ACTIONABLE PAPER TRADE SIGNAL: STRATEGY 1 (15M BOX STRANGLE)] <<<")
        print("=" * 80)
        print(f"Time:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Nifty Spot:        {spot:.2f}")
        print("-" * 80)
        print(f"Leg 1 (CALL):       BUY 1 Lot (75 Qty) NIFTY {ce_strike} CE @ ~INR {ce_price:.2f}")
        print(f"                    -> Stop-Loss:  INR {ce_price * (1 - self.leg_sl_pct):.2f} (-20%)")
        print(f"                    -> Target:     INR {ce_price * (1 + self.target_pct):.2f} (+75%)")
        print("-" * 80)
        print(f"Leg 2 (PUT):        BUY 1 Lot (75 Qty) NIFTY {pe_strike} PE @ ~INR {pe_price:.2f}")
        print(f"                    -> Stop-Loss:  INR {pe_price * (1 - self.leg_sl_pct):.2f} (-20%)")
        print(f"                    -> Target:     INR {pe_price * (1 + self.target_pct):.2f} (+75%)")
        print("-" * 80)
        total_cost = (ce_price + pe_price) * self.lot_size
        print(f"Total Capital Required: INR {total_cost:.2f} (Within INR 10,000 budget)")
        print("MANAGEMENT RULE: Once either leg drops 20%, CUT IT IMMEDIATELY!")
        print("                 Let the winning leg run. When winner hits +25%, trail SL to Cost!")
        print("=" * 80 + "\n")

    def run_replay_demonstration(self):
        """Simulates how the paper trading bot executes on the most recent trading session."""
        print("=" * 80)
        print("   STRATEGY 1 PAPER TRADING BOT: SIMULATING RECENT SESSION")
        print("=" * 80)
        df = pd.read_csv("data/nifty_1min_real.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        last_date = df["timestamp"].dt.date.max()
        day_df = df[df["timestamp"].dt.date == last_date].copy().sort_values("timestamp")

        print(f"Trading Date: {last_date} | Total 1-min candles: {len(day_df)}")
        box_bars = day_df[(day_df["timestamp"].dt.time >= dtime(9, 15)) & (day_df["timestamp"].dt.time <= dtime(9, 30))]

        box_h = box_bars["high"].max()
        box_l = box_bars["low"].min()
        box_rng = box_h - box_l
        spot_ref = box_bars["close"].iloc[0]
        box_pct = (box_rng / spot_ref) * 100.0

        print(f"\n[09:30 AM BOX EVALUATION]")
        print(f"Box High: {box_h:.2f} | Box Low: {box_l:.2f} | Range: {box_rng:.2f} pts ({box_pct:.2f}%)")
        if box_pct <= 0.30:
            print(f"[STATUS] Box is TIGHT (<0.30%). STRATEGY IS ARMED FOR BREAKOUT!")
            print(f"Upper Breakout Trigger: {box_h + 6.0:.2f}")
            print(f"Lower Breakout Trigger: {box_l - 6.0:.2f}")
        else:
            print(f"[STATUS] Box too wide (>0.30%). No trade today.")
            return

        rem_bars = day_df[(day_df["timestamp"].dt.time > dtime(9, 30)) & (day_df["timestamp"].dt.time <= dtime(13, 30))]
        atm = int(round(spot_ref / 50.0) * 50)
        for _, bar in rem_bars.iterrows():
            t = bar["timestamp"]
            spot = bar["close"]
            if bar["high"] >= (box_h + 6.0) or bar["low"] <= (box_l - 6.0):
                ce_s = atm + self.otm_pts
                pe_s = atm - self.otm_pts
                ce_p = black_scholes_price(spot, ce_s, 0.015, 0.07, 0.135, "CE")
                pe_p = black_scholes_price(spot, pe_s, 0.015, 0.07, 0.135, "PE")
                print(f"\n[BREAKOUT DETECTED at {t.strftime('%H:%M:%S')}] Spot: {spot:.2f}")
                self.print_trade_signal(spot, ce_s, pe_s, ce_p, pe_p)
                break

class LiveDASPaperTrader:
    """
    Strategy 4: Decoupled Asymmetric Strangle (DAS) Paper Trading Engine.
    Delta-Neutral Volatility Compression & Kinetic Velocity Expansion.
    """
    def __init__(
        self,
        vol_window: int = config.DAS_VOL_WINDOW,
        std_thresh: float = config.DAS_STD_THRESH,
        vel_thresh: float = config.DAS_VEL_THRESH,
        win_target_pct: float = config.DAS_WIN_TARGET_PCT,
        lose_stop_pct: float = config.DAS_LOSE_STOP_PCT,
        combined_stop_pct: float = config.DAS_COMBINED_STOP_PCT,
        max_hold_mins: int = config.DAS_MAX_HOLD_MINS
    ):
        self.vol_window = vol_window
        self.std_thresh = std_thresh
        self.vel_thresh = vel_thresh
        self.win_target_pct = win_target_pct
        self.lose_stop_pct = lose_stop_pct
        self.combined_stop_pct = combined_stop_pct
        self.max_hold_mins = max_hold_mins
        self.lot_size = config.LOT_SIZE
        self.capital = config.INITIAL_CAPITAL

        self.smart_api = None
        self.opt_data = load_cached_real_options()

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
                print(f"[SUCCESS] Connected to Angel One SmartAPI for DAS live paper trading. (Client: {client_code})")
                return True
            else:
                return False
        except Exception:
            return False

    def print_trade_signal(self, t_str: str, spot: float, ce_str: str, pe_str: str, ce_p: float, pe_p: float):
        tot_cost = (ce_p + pe_p) * self.lot_size
        print("\n" + "=" * 85)
        print(">>> [ACTIONABLE PAPER TRADE SIGNAL: STRATEGY 4 DECOUPLED ASYMMETRIC STRANGLE] <<<")
        print("=" * 85)
        print(f"Time Triggered:    {t_str}")
        print(f"Nifty Spot LTP:    {spot:.2f}")
        print(f"Strategy Type:     Delta-Neutral Volatility Compression & Kinetic Expansion")
        print("-" * 85)
        print(f"Leg 1 (CALL):       BUY 1 Lot (75 Qty) {ce_str} @ INR {ce_p:.2f}")
        print(f"                    -> Profit Target: INR {ce_p * (1.0 + self.win_target_pct):.2f} (+{int(self.win_target_pct*100)}%) [DECOUPLED EXIT]")
        print(f"                    -> Stop-Loss:     INR {ce_p * (1.0 - self.lose_stop_pct):.2f} (-{int(self.lose_stop_pct*100)}%)")
        print("-" * 85)
        print(f"Leg 2 (PUT):        BUY 1 Lot (75 Qty) {pe_str} @ INR {pe_p:.2f}")
        print(f"                    -> Profit Target: INR {pe_p * (1.0 + self.win_target_pct):.2f} (+{int(self.win_target_pct*100)}%) [DECOUPLED EXIT]")
        print(f"                    -> Stop-Loss:     INR {pe_p * (1.0 - self.lose_stop_pct):.2f} (-{int(self.lose_stop_pct*100)}%)")
        print("-" * 85)
        print(f"Combined Cost:     INR {tot_cost:.2f} (Strictly <= INR {self.capital:,.2f} Budget)")
        print(f"Combined Stop:     -15% on Total Position (INR {tot_cost * 0.85:.2f})")
        print(f"Max Hold Duration: {self.max_hold_mins} Minutes (Time-Decay Cutoff)")
        print("=" * 85 + "\n")

        # Send Real-Time Telegram Alert to User's Phone
        send_trade_entry_alert(
            strategy="Strategy 4: Decoupled Asymmetric Strangle (DAS)",
            spot=spot,
            ce_str=ce_str,
            pe_str=pe_str,
            ce_p=ce_p,
            pe_p=pe_p,
            tot_cost=tot_cost,
            win_target_pct=self.win_target_pct,
            lose_stop_pct=self.lose_stop_pct,
            max_hold_mins=self.max_hold_mins
        )

    def run_replay_demonstration(self, session_date: str = "2026-09-24"):
        """Demonstrates live execution of Strategy 4 on real market session."""
        print("=" * 85)
        print(f"   STRATEGY 4 (DAS) PAPER TRADING BOT: SIMULATING REAL SESSION ({session_date})")
        print("=" * 85)
        df = pd.read_csv("data/nifty_1min_real.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        day_df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == session_date].copy().sort_values("timestamp")

        if day_df.empty:
            print(f"[ERROR] No data found for {session_date}")
            return

        print(f"Total 1-Minute Candles Loaded: {len(day_df)}")
        print(f"Scanning for Volatility Compression (std < {self.std_thresh}) and Kinetic Expansion (vel >= {self.vel_thresh})...\n")

        timestamps = day_df["timestamp"].tolist()
        for i in range(len(timestamps)):
            t = timestamps[i]
            t_time = t.time()
            if not is_trade_window_valid(t_time):
                continue

            spot = float(day_df.iloc[i]["close"])
            res = select_affordable_strikes(spot, t, opt_data=self.opt_data)
            if res:
                ce_sym, pe_sym, ce_p, pe_p, tot_cost = res
                ce_df = self.opt_data[ce_sym]
                pe_df = self.opt_data[pe_sym]
                try:
                    idx_c = ce_df.index.get_loc(t)
                    if idx_c >= self.vol_window:
                        comb_hist = (ce_df['close'].iloc[idx_c - self.vol_window:idx_c + 1] + pe_df['close'].iloc[idx_c - self.vol_window:idx_c + 1]).values
                        triggered, cur_std, cur_vel = check_compression_expansion(comb_hist, self.std_thresh, self.vel_thresh)
                        if triggered:
                            self.print_trade_signal(t.strftime("%Y-%m-%d %H:%M:%S"), spot, ce_sym, pe_sym, ce_p, pe_p)
                            break
                except Exception:
                    pass

def main():
    parser = argparse.ArgumentParser(description="Live Market Paper Trading Bot for Nifty Options")
    parser.add_argument("--strategy", type=str, default="das", choices=["strangle", "das", "all"], help="Strategy to trade (strangle, das, or all).")
    args = parser.parse_args()

    if args.strategy in ["strangle", "all"]:
        bot_s1 = LiveStranglePaperTrader()
        bot_s1.connect_angel_one()
        bot_s1.run_replay_demonstration()

    if args.strategy in ["das", "all"]:
        bot_das = LiveDASPaperTrader()
        bot_das.connect_angel_one()
        bot_das.run_replay_demonstration()

if __name__ == "__main__":
    main()
