"""
Standalone Interactive Telegram Bot Runner.
Listens for user commands (/status, /pnl, /today, /summary) and button clicks.
Allows checking live positions, current PnL, and trade journal directly from Telegram.
"""
from src.notifications.telegram_interactive_bot import run_telegram_polling_loop

if __name__ == "__main__":
    print("=" * 75)
    print("   NIFTY MULTI-INDEX OPTION BOT - INTERACTIVE TELEGRAM CONTROLLER")
    print("   Available Commands & Buttons: /status, /pnl, /today, /summary, /journal")
    print("=" * 75)
    run_telegram_polling_loop()
