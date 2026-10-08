import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from src.notifications.notifier import (
    normalize_date_str,
    normalize_strategy_code,
    normalize_contract_key,
    is_alert_already_sent,
    mark_alert_as_sent
)
from src.notifications.telegram_interactive_bot import (
    format_live_status_card,
    format_today_trades_card,
    format_portfolio_summary_card,
    format_journal_recent_card,
    load_live_active_positions
)

print("=== 1. Test Normalization Functions ===")
print("Date '08-Oct-2026' ->", normalize_date_str("08-Oct-2026"))
print("Date '2026-10-08' ->", normalize_date_str("2026-10-08"))
print("Strategy 'Strategy 5: BankNIFTY Decoupled Strangle (DAS)' ->", normalize_strategy_code("Strategy 5: BankNIFTY Decoupled Strangle (DAS)"))
print("Strategy 'Strategy 4: Decoupled Asymmetric Strangle' ->", normalize_strategy_code("Strategy 4: Decoupled Asymmetric Strangle"))
print("Strategy 'Strategy 1: Hedged Strangle (15M Box)' ->", normalize_strategy_code("Strategy 1: Hedged Strangle (15M Box)"))
print("Strategy 'Strategy 2: 0-DTE / 1-DTE Gamma Squeeze' ->", normalize_strategy_code("Strategy 2: 0-DTE / 1-DTE Gamma Squeeze"))
print("Strategy 'Strategy 3: 30M Directional ITM' ->", normalize_strategy_code("Strategy 3: 30M Directional ITM"))

print("\n=== 2. Test Deduplication Mechanism ===")
test_key = "TEST_ALERT_KEY_123"
print("Before marking:", is_alert_already_sent(test_key))
mark_alert_as_sent(test_key)
print("After marking:", is_alert_already_sent(test_key))

print("\n=== 3. Test Live Status Card (Standby / Idle) ===")
msg_idle, kb_idle = format_live_status_card()
print(msg_idle)
print("Inline Keyboard:", json.dumps(kb_idle, indent=2))

print("\n=== 4. Test Live Status Card with In-Flight Active Position ===")
active_pos_file = "data/active_positions.json"
os.makedirs("data", exist_ok=True)
mock_pos = {
    "s5": {
        "strategy_name": "Strategy 5: BankNIFTY Decoupled Strangle (DAS)",
        "entry_time": "2026-10-08T10:12:00",
        "spot_entry": 52150.0,
        "ce_strike": 57600,
        "pe_strike": 52400,
        "ce_symbol": "BANKNIFTY27OCT2657600CE",
        "pe_symbol": "BANKNIFTY27OCT2652400PE",
        "ce_entry": 95.10,
        "pe_entry": 95.10,
        "entry_tot": 190.20,
        "cost": 5706.00,
        "ce_tgt": 190.20,
        "ce_sl": 80.84,
        "pe_tgt": 190.20,
        "pe_sl": 80.84,
        "ce_exited": False,
        "pe_exited": False,
        "data_feed": "Angel One Real Traded"
    }
}
with open(active_pos_file, "w", encoding="utf-8") as f:
    json.dump(mock_pos, f, indent=2)

msg_active, kb_active = format_live_status_card()
print(msg_active)

# Clean up mock active position file so it returns to empty state
with open(active_pos_file, "w", encoding="utf-8") as f:
    json.dump({}, f)

print("\n=== 5. Test Today's Trades Card ===")
msg_today, kb_today = format_today_trades_card()
print(msg_today)

print("\n=== 6. Test Portfolio Summary Card ===")
msg_summary, kb_summary = format_portfolio_summary_card()
print(msg_summary)

print("\n=== 7. Test Recent Journal Card ===")
msg_journal, kb_journal = format_journal_recent_card()
print(msg_journal)
