"""
Notifications Package for Real-time Trading Alerts.
"""
from .notifier import (
    send_telegram_alert,
    send_trade_entry_alert,
    send_trade_exit_alert,
    send_daily_summary_alert,
    get_or_detect_chat_id
)

__all__ = [
    "send_telegram_alert",
    "send_trade_entry_alert",
    "send_trade_exit_alert",
    "send_daily_summary_alert",
    "get_or_detect_chat_id"
]
