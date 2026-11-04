# Quantitative Research & Backtesting Tools

This directory contains standalone tools for conducting empirical research, historical backtesting, and performance visualization across the **Quad-Strategy Intraday Option Buying Portfolio**.

---

## 1. Unified Portfolio Backtester (`backtest_portfolio.py`)

Simulates all 4 quantitative strategies across the historical 1-minute datasets (`data/nifty_1min_real.csv` and `data/banknifty_1min_real.csv`).

### Key Features:
- **Zero Lookahead Bias**: Operates in causal, bar-by-bar sequence.
- **Institutional Friction**: Applies exact Angel One brokerage (₹20/order), STT, Exchange Turnover, GST, SEBI, Stamp duty, and a realistic **0.5% slippage**.
- **Official Lot Sizes**: 65 (NIFTY) and 30 (BANKNIFTY).
- **Single-Loss Kill Switch**: Strategy 4 enforces strict max 1-loss termination per day.

### Usage Examples:

```bash
# Backtest the entire historical dataset
python tools/backtest_portfolio.py

# Backtest a specific date window
python tools/backtest_portfolio.py --start 2026-08-01 --end 2026-11-04

# Export trade-by-trade CSV log
python tools/backtest_portfolio.py --output tools/backtest_trades.csv
```

---

## 2. Equity Curve & Drawdown Visualizer (`plot_equity.py`)

Generates high-resolution performance charts showing portfolio equity growth, individual strategy contributions, and underwater drawdown.

### Usage Example:

```bash
# Generate backtest trades first
python tools/backtest_portfolio.py --output tools/backtest_trades.csv

# Render performance chart
python tools/plot_equity.py --trades tools/backtest_trades.csv --output tools/portfolio_performance.png
```
