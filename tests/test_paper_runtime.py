import datetime as dt
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import live_paper_trader as live
from config import config, option_charges
from src.data.smartapi_client import _SmartApiLogFilter
from src.notifications import notifier
from src.notifications.telegram_interactive_bot import format_portfolio_summary_card
from src.utils.market_calendar import is_trading_day, weekly_expiry


class PaperRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        original_directory = os.getcwd()
        os.chdir(self.work.name)
        self.addCleanup(os.chdir, original_directory)
        self.trader = live.LiveTriPaperTrader()
        self.trader.smart_api = Mock()
        self.trader.smart_api.ltpData.return_value = {"status": True, "data": {"ltp": 120}}
        self.now = dt.datetime(2026, 11, 23, 10, 0)

    def test_budget_calendar_and_charges(self):
        self.assertEqual(config.MAX_CONCURRENT_POSITIONS, 4)
        self.assertEqual(4 * config.MAX_CAPITAL_PER_TRADE, config.INITIAL_CAPITAL)
        self.assertIn("38,000", format_portfolio_summary_card()[0])
        self.assertEqual(weekly_expiry(dt.date(2026, 11, 23)), dt.date(2026, 11, 23))
        self.assertTrue(is_trading_day(dt.date(2026, 11, 9)))
        self.assertFalse(is_trading_day(dt.date(2026, 11, 24)))
        self.assertEqual(round(option_charges(8000, 10000, "2026-04-01")
                               - option_charges(8000, 10000, "2026-03-31"), 2), 5)

    def test_sdk_filter_excludes_request_logs(self):
        record = logging.LogRecord("logzero", logging.ERROR,
                                   os.path.join("site-packages", "SmartApi", "smartConnect.py"),
                                   1, "fixture", (), None)
        self.assertFalse(_SmartApiLogFilter().filter(record))

    def test_fourth_slot_entry_alert_exit_and_journal(self):
        from src.utils import journal_manager as journal
        from src.strategies import nifty_1m_scalper as scalper

        for key in ("s1", "s2", "s3"):
            self.trader.active_positions[key] = {"cost": 9500, "entry_time": self.now}
        stamps = pd.date_range(end=self.now - dt.timedelta(minutes=1), periods=10, freq="min")
        spot = pd.DataFrame({"timestamp": stamps, "open": 25000, "close": 25001,
                             "high": 25002, "low": 24998})
        contract = {"token": "1", "symbol": "TESTCE"}
        fills = pd.DataFrame({"close": [100]}, index=stamps[-1:])
        self.trader.smart_api.getMarketData.return_value = {"data": {"fetched": [{
            "symbolToken": "1", "depth": {"sell": [{"price": 101}], "buy": [{"price": 110}]}
        }]}}
        for name, filename in (("DATA_DIR", "data"), ("TRADE_JOURNAL_CSV", "data/journal.csv"),
                               ("TRADE_JOURNAL_MD", "data/journal.md"),
                               ("ACTIVE_POSITIONS_JSON", "data/active_positions.json"),
                               ("LEDGER_CSV", "data/ledger.csv")):
            self.enterContext(patch.object(journal, name, filename))
        self.enterContext(patch.object(live, "git_sync_push"))
        self.enterContext(patch.object(live, "get_ist_now", return_value=self.now))
        self.enterContext(patch.object(scalper, "determine_morning_bias", return_value=1))
        self.enterContext(patch.object(scalper, "check_pullback_trigger", return_value=True))
        self.enterContext(patch.object(live, "get_active_option_contract", return_value=contract))
        self.enterContext(patch.object(live, "select_dynamic_affordable_strike",
                                      return_value=(25000, fills, "TESTCE", "fixture", 100, 6500)))
        entry = self.enterContext(patch.object(live, "send_trade_entry_alert", return_value=True))
        self.trader.process_live_strategy_4(spot, self.now.date(), self.now, 25000)
        entry.assert_called_once()
        self.assertEqual(sum(p["cost"] for p in self.trader.active_positions.values()), 35000)
        self.enterContext(patch.object(live, "fetch_real_option_candles", return_value=pd.DataFrame()))
        self.enterContext(patch.object(live, "send_trade_exit_alert", return_value=True))
        self.trader.process_live_strategy_4(spot, self.now.date(), self.now + dt.timedelta(minutes=1), 25000)
        row = pd.read_csv("data/journal.csv").iloc[0]
        self.assertEqual(row["quoted_net_pnl"], round(9 * 65 - option_charges(101*65, 110*65, self.now.date()), 2))
        self.assertEqual(row["record_type"], "forward_paper")
        self.assertTrue(row["entry_alert_accepted"] and row["exit_alert_accepted"])
        self.assertEqual(json.loads(row["entry_depth"])["quotes"]["1"], 101)
        self.assertEqual(json.loads(row["exit_depth"])["quotes"]["1"], 110)
        self.assertNotIn("s4", json.loads(Path("data/active_positions.json").read_text()))

    def test_telegram_restart_retry_and_legacy_dedup(self):
        path = Path(self.work.name) / "alerts.json"
        path.write_text(json.dumps({"legacy": "timestamp"}))
        with patch.object(notifier, "SENT_ALERTS_FILE", str(path)), patch.dict(os.environ, {
            "TELEGRAM_CHAT_ID": "123", "TELEGRAM_BOT_TOKEN": "",
        }):
            self.assertTrue(notifier.is_alert_already_sent("legacy"))
            self.assertFalse(notifier.send_telegram_alert("summary", alert_key="SUMMARY:test"))
            self.assertFalse(notifier.is_alert_already_sent("SUMMARY:test"))
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "fixture"}), patch("requests.post") as post:
                post.return_value.status_code = 200
                post.return_value.json.return_value = {"ok": True, "result": {"message_id": 42}}
                notifier.retry_pending_alerts()
                notifier.retry_pending_alerts()
                post.assert_called_once()
                self.assertEqual(json.loads(path.read_text())["SUMMARY:test"]["message_id"], 42)

    def test_trailing_stop_survives_restart(self):
        position = {"entry_time": self.now, "entry_p": 100, "cost": 3000,
                    "sl_p": 75, "target_p": 180, "strike": 63000, "opt_type": "CE",
                    "contract": {"symbol": "TEST", "token": "1"}, "symbol": "TEST"}
        self.trader.active_positions["s2"] = position
        bars = pd.DataFrame([{"high": 104, "low": 99, "close": 101},
                             {"high": 151, "low": 105, "close": 120}],
                            index=[self.now + dt.timedelta(minutes=i) for i in (1, 2)])
        poll = self.now + dt.timedelta(minutes=3)
        with patch.object(live, "fetch_real_option_candles", return_value=bars), patch.object(live, "send_trade_exit_alert"):
            self.trader.process_live_strategy_2(pd.DataFrame(), poll.date(), poll, 63000)
            self.assertEqual(position["sl_p"], 110)
            self.trader.init_today_state(poll.date())
            with patch.object(self.trader, "on_trade_exit") as close:
                self.trader.process_live_strategy_2(pd.DataFrame(), poll.date(), poll, 63000)
                close.assert_not_called()
                bars.loc[poll] = [115, 108, 111]
                self.trader.process_live_strategy_2(pd.DataFrame(), poll.date(), poll + dt.timedelta(minutes=1), 63000)
                self.assertEqual(close.call_args.args[2]["exit_reason"], "Profit Floor Hit (+10%)")

    def test_eod_strangle_keeps_realized_leg(self):
        end = self.now.replace(hour=15, minute=5)
        self.trader.active_positions["s3"] = {
            "entry_time": self.now, "entry_tot": 115, "ce_entry": 55, "pe_entry": 60,
            "ce_exited": True, "ce_exit_p": 49.5, "ce_reason": "CE stop",
            "pe_exited": False, "pe_sl": 54, "pe_tgt": 96, "pe_exit_p": None,
            "ce_symbol": "TESTCE", "pe_symbol": "TESTPE", "ce_strike": 25000,
            "pe_strike": 24900, "cost": 7475, "c_pe": {"symbol": "TESTPE", "token": "2"},
        }
        self.trader.smart_api.ltpData.return_value = {"status": True, "data": {"ltp": 65}}
        self.addCleanup(patch.stopall)
        patch("live_paper_trader.time.sleep").start()
        with patch.object(live, "get_ist_now", return_value=end), patch.object(live, "sync_and_update_candles"), patch.object(live, "send_trade_exit_alert"), patch.object(live, "is_alert_already_sent", return_value=True), patch("src.utils.git_sync.git_sync_pull"), patch.object(self.trader, "on_trade_exit") as closed:
            self.trader.run_eod_accounting()
            trade = closed.call_args.args[2]
            self.assertEqual(trade["entry_p"], 115)
            self.assertEqual(trade["exit_p"], round(49.5 + round(65 * .995, 2), 2))
            self.assertEqual(trade["opt_type"], "CE+PE")


if __name__ == "__main__":
    unittest.main()
