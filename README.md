# 🚀 Multi-Index Quad-Strategy Option Trading Bot (Nifty 50 & BankNIFTY)

Production-grade algorithmic options paper-trading and live execution system for Indian Stock Exchanges (NSE). Built for high-Sharpe option buying with strict institutional risk controls, real-time Angel One SmartAPI integration, automated GitHub Actions scheduling, and instant Telegram execution alerts.

---

## 📊 Quad-Strategy Production Architecture

The portfolio operates across **four uncorrelated, mathematically vetted quantitative strategies**, designed to harvest alpha in trending, range-bound, and high-velocity market regimes while enforcing strict capital preservation.

```
                          ┌──────────────────────────────────────────────┐
                          │    QUAD-STRATEGY QUANTITATIVE PORTFOLIO      │
                          │        (INR 10,000 Capital Cap/Trade)        │
                          └──────────────────────┬───────────────────────┘
                                                 │
            ┌───────────────────┬────────────────┴──────────────────┬───────────────────┐
            │                   │                                   │                   │
            ▼                   ▼                                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐                   ┌───────────────┐   ┌───────────────┐
    │  STRATEGY 1   │   │  STRATEGY 2   │                   │  STRATEGY 3   │   │  STRATEGY 4   │
    │  Nifty 15M    │   │ BankNifty 15M │                   │  Nifty DAS    │   │   Nifty 1M    │
    │  Directional  │   │  Directional  │                   │ Decoupled     │   │   Pullback    │
    │   Breakout    │   │   Breakout    │                   │   Strangle    │   │    Scalper    │
    └───────┬───────┘   └───────┬───────┘                   └───────┬───────┘   └───────┬───────┘
            │                   │                                   │                   │
     ATM Call / Put      100-pt OTM Call/Put                 CE + PE Strangle    ATM Call / Put
     SL: -16.0%          SL: -25.0%                          SL: -10.0% / leg    SL: -11.5%
     TP: +60.0%          TP: +80.0%                          TP: +60.0% / leg    TP: +10.0%
     Cutoff: 13:00 IST   Cutoff: 13:00 IST                   Vol Compression     Single-Loss Stop
```

### 1. Strategy 1: Nifty 15M Directional Breakout
* **Underlying**: Nifty 50 Index (`99926000`)
* **Timeframe**: 15-Minute Opening Range (09:15–09:30 AM IST) with 1-minute execution resolution.
* **Filter Rules**: Opening box range must be $\le 0.30\%$ of spot to filter out volatile chop.
* **Trigger**: Directional breakout past opening high/low with 20-period EMA trend confirmation.
* **Strike Selection**: Nearest weekly ATM strike (budget $\le$ ₹9,500).
* **Risk Management**: Stop Loss **-16.0%**, Profit Target **+60.0%**, Cutoff 13:00 IST.

### 2. Strategy 2: BankNIFTY 15M Directional Breakout
* **Underlying**: BankNIFTY Index (`99926009`)
* **Timeframe**: 15-Minute Opening Range with 1-minute execution resolution.
* **Filter Rules**: Buffer threshold +40 points beyond high/low with 20-period EMA confirmation.
* **Strike Selection**: Dynamic 100–300 pt OTM strike selected strictly within ₹9,500 budget.
* **Risk Management**: Stop Loss **-25.0%**, Profit Target **+80.0%**, Cutoff 13:00 IST.

### 3. Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)
* **Underlying**: Nifty 50 Index Options (Simultaneous CE + PE Buying)
* **Regime**: Exploits volatility compression bursts (standard deviation $< 25$) followed by kinetic expansion (velocity $\ge 40$).
* **Strike Selection**: Delta-neutral 200-point wide OTM wings (Budget $\le$ ₹9,500 combined).
* **Decoupled Exit Rules**: Each option leg is tracked and managed independently:
  * Winning Leg: Profit Target **+60.0%**
  * Losing Leg: Hard Stop Loss **-10.0%**
  * Maximum Hold Time: 25 minutes.

### 4. Strategy 4: Nifty 1M Pullback Option Scalper
* **Underlying**: Nifty 50 Index Options (1-Minute Candles)
* **Morning Directional Bias**: Established at 09:20–09:30 AM IST via VWAP and EMA 9/21 alignment.
* **Entry Trigger**: Pullback bounce into the 21-period EMA with confirmation close above EMA 9.
* **Strike Selection**: Dynamic nearest weekly ATM strike (budget $\le$ ₹9,500).
* **Execution & Kill Switch**:
  * Profit Target: **+10.0%** on option premium
  * Stop Loss: **-11.5%** on option premium
  * **Single-Loss Kill Switch**: Exactly 1 loss terminates S4 trading for the remainder of the session.
  * Maximum 4 wins per day with a 10-minute cooldown between trades.

---

## 🛡️ Institutional Risk & Execution Standards

* **Conservative Capital Cap**: Strict ₹10,000 maximum capital allocation per trade.
* **NSE Lot Sizes**: Official standard exchange lot sizes:
  * **Nifty 50**: 65 Qty
  * **BankNIFTY**: 30 Qty
* **Slippage Modeling**: 0.5% conservative slippage applied to both entry and exit prices.
* **Indian Statutory Charges & Taxes Included**:
  * Brokerage: Flat ₹40 per round-trip trade (₹20 entry + ₹20 exit)
  * STT (Securities Transaction Tax): 0.1% on option sell turnover
  * Exchange Turnover Fees: 0.05%
  * GST: 18% on (Brokerage + Exchange Fees)
  * Stamp Duty: 0.003% on buy turnover
* **Zero Lookahead Bias**: Live and backtest engines evaluate 1-minute candles strictly chronologically without peeking into future bars. Stop losses are always evaluated prior to profit targets within every candle.

---

## ⏰ Automated Cloud Operation (GitHub Actions)

The repository runs completely autonomously on GitHub Actions:
* **Workflow**: `.github/workflows/trading_bot.yml`
* **Cron Schedule**: `'44 3 * * 1-5'` (03:44 UTC = **09:14 AM IST**). Runner boots 1 minute prior to the 09:15 AM opening bell.
* **Continuous Session**: Full trading day monitoring (09:15 AM to 15:05 IST).
* **Concurrency Protection**: `cancel-in-progress: false` ensures that running sessions are never aborted prematurely.
* **Holiday Protection**: Built-in NSE market calendar (`src/utils/market_calendar.py`) detects all trading holidays (including Diwali sessions), posts an official notification to Telegram, and sleeps safely.
* **Daily Auto-Commit**: At 15:05 IST market close, the runner audits the session, squareoffs any active trades, updates `data/trade_journal.csv` and `data/trade_journal.md`, and commits the results to GitHub automatically.

---

## 📱 Telegram Alert & Interactive Command Center

The bot is directly linked to Telegram for instant real-time oversight:

| Alert / Feature | Timing | Description |
|:---|:---|:---|
| **🚨 Trade Entry Siren** | Instant (Live Tick) | Contract symbol, entry price, profit target, stop loss, capital used. |
| **🎯 / 🛑 Trade Exit Alert** | Instant (Live Tick) | Exit reason, gross PnL, statutory charges, net PnL, duration. |
| **📜 Daily Performance Summary** | 15:05 IST (Market Close) | Daily win/loss record, day net PnL, total portfolio cumulative balance. |
| **🏛️ NSE Holiday Notice** | 09:15 AM IST (Holidays) | Confirms market holiday status and lists next trading session date. |

### Interactive Bot Commands
You can message your Telegram bot at any time:
* `/status` — Displays live in-flight open positions, entry prices, live unrealized PnL, and current spot prices.
* `/today` — Summarizes today's closed trades, win rate, and realized net PnL.
* `/pnl` — Full cumulative portfolio balance, total percentage return, and historical win rate.

---

## 💻 Local CLI Usage

### 1. Run Live Market Paper Trading
```powershell
python live_paper_trader.py --strategy all
```
*Options for `--strategy`: `all`, `s1`, `s2`, `s3`, `s4`.*

### 2. Run Quad-Strategy Offline Replay
```powershell
python live_paper_trader.py --replay
```
*Replays the latest recorded market day from local exchange candles to verify strategy execution.*

### 3. Run Daily Result Checker & Auditor
```powershell
python daily_result_checker.py --date 2026-11-04
```
*Evaluates all 4 strategies, reconciles any unrecorded trades, updates `trade_journal.md`, and prints the portfolio ledger table.*

### 4. Run Interactive Telegram Controller Bot
```powershell
python telegram_interactive_bot.py
```
*Listens for user commands (`/status`, `/today`, `/pnl`, `/summary`, `/journal`) and sends instant responses.*

### 5. Run Historical Portfolio Backtest
```powershell
python tools/backtest_portfolio.py --start 2026-08-01 --end 2026-11-04
```
*Backtests all 4 strategies across the continuous datasets and displays institutional performance metrics.*

---

## 📂 Project Directory Structure

```
NF-OP/
├── .github/workflows/
│   └── trading_bot.yml           # Automated GitHub Actions cloud execution workflow
├── data/
│   ├── nifty_1min_real.csv       # Continuous 1-minute real Nifty spot candles (auto-synced)
│   ├── banknifty_1min_real.csv   # Continuous 1-minute real BankNifty spot candles (auto-synced)
│   ├── trade_journal.csv         # Canonical chronological trade ledger (all 4 strategies)
│   ├── trade_journal.md          # Formatted live Markdown performance journal
│   ├── active_positions.json     # Real-time state persistence for crash resilience
│   ├── sent_alerts.json          # Deterministic deduplication record for Telegram
│   ├── audited_days.json         # Reconciled daily audit records
│   ├── live_paper_trades.csv     # Paper execution transaction history log
│   └── README.md                 # Complete documentation of data feeds & records
├── src/
│   ├── banknifty/
│   │   └── banknifty_das.py      # Strategy 2: BankNIFTY 15M Directional Breakout
│   ├── data/
│   │   ├── angel_connector.py    # Angel One SmartAPI connector & session manager
│   │   └── real_option_feed.py   # Angel One SmartAPI live candle and LTP fetcher
│   ├── decoupled_strangle/
│   │   ├── das_engine.py         # Strategy 3: Decoupled Asymmetric Strangle engine
│   │   └── das_signals.py        # Volatility compression & expansion signal detector
│   ├── features/
│   │   ├── greeks.py             # Black-Scholes pricing & calendar DTE calculator
│   │   └── indicators.py         # ATR, EMA, Supertrend, and technical indicators
│   ├── notifications/
│   │   ├── notifier.py           # Telegram notification formatters and deduplication
│   │   └── telegram_interactive_bot.py # Interactive /status, /today, /pnl listener
│   ├── strategies/
│   │   └── nifty_1m_scalper.py   # Strategy 4: Nifty 1M Pullback Scalper implementation
│   └── utils/
│       ├── git_sync.py           # Git synchronization and commit utility
│       ├── journal_manager.py    # Automated trade ledger and Markdown journal writer
│       └── market_calendar.py    # NSE official holiday schedule & verification
├── tools/
│   ├── backtest_portfolio.py     # Unified Quad-Strategy Portfolio Backtester
│   ├── plot_equity.py            # Equity Curves & Drawdown Visualization Generator
│   └── README.md                 # Guide to running backtests and generating charts
├── config.py                     # Centralized configuration dataclass & risk rules
├── daily_result_checker.py       # EOD reconciliation, audit, and dashboard utility
├── live_paper_trader.py          # Master live market execution engine
├── telegram_interactive_bot.py   # Standalone interactive Telegram bot launcher
├── requirements.txt              # Production Python package dependencies
├── .gitignore                    # Production git exclusion rules
└── README.md                     # Complete institutional system documentation
```

---

## 🔒 Environment Variables (`.env` & GitHub Secrets)

```ini
ANGEL_API_KEY=your_smartapi_key
ANGEL_CLIENT_CODE=your_client_code
ANGEL_PASSWORD=your_trading_pin
ANGEL_TOTP_SECRET=your_authenticator_qr_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

---

*Institutional Rigor. Zero Lookahead Bias. Systematic Execution.*

