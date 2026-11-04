# Market Data Feeds & Production Records

This directory contains the live-synced market candles, trade journals, and execution persistence ledgers for the **Quad-Strategy Intraday Option Buying Portfolio**.

---

## File Manifest

| File | Purpose | Updated By | Git Status |
| :--- | :--- | :--- | :--- |
| `nifty_1min_real.csv` | Official NIFTY 50 1-minute OHLCV candles (~148k+ bars). | Auto-synced daily via Angel One SmartAPI | Tracked |
| `banknifty_1min_real.csv` | Official BANKNIFTY 1-minute OHLCV candles (~153k+ bars). | Auto-synced daily via Angel One SmartAPI | Tracked |
| `trade_journal.csv` | Audited real trade execution log across all 4 strategies (CSV). | `daily_result_checker.py` & `live_paper_trader.py` | Tracked |
| `trade_journal.md` | Human-readable Markdown trade journal with formatted PnL tables. | `src/utils/journal_manager.py` | Tracked |
| `active_positions.json` | JSON store tracking open positions across bot restarts. | `live_paper_trader.py` | Tracked |
| `sent_alerts.json` | Deduplication ledger preventing duplicate Telegram notifications. | `src/notifications/notifier.py` | Tracked |
| `audited_days.json` | Ledger of all trading dates that have undergone reconciliation. | `daily_result_checker.py` | Tracked |
| `live_paper_trades.csv` | Paper trade execution transaction history log. | `live_paper_trader.py` | Tracked |
| `angel_scrip_master.json` | Angel One instrument scrip master for token-symbol resolution. | `src/data/angel_connector.py` | Ignored (`.gitignore`) |

---

## Important Rules

1. **Do not manually edit `trade_journal.csv` or `trade_journal.md`**: These files are automatically updated and reconciled by the daily auditor with exact Angel One option tick executions.
2. **Options Tick Caches**: Real historical option tick data downloaded from SmartAPI is stored in `data/real_options_cache/` and `data/banknifty_options_cache/` (both are gitignored to keep the repository lightweight).
