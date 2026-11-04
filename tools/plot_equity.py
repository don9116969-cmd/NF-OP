"""
Equity Curve and Drawdown Visualization Generator.
Plots institutional-grade multi-strategy equity curves and underwater drawdown charts.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def plot_equity_charts(trades_csv: str, output_image: str = "equity_chart.png", initial_capital: float = 10000.0):
    if not os.path.exists(trades_csv):
        print(f"[ERROR] Trade log file '{trades_csv}' does not exist.")
        return

    df = pd.read_csv(trades_csv)
    if df.empty:
        print("[ERROR] Trade log is empty.")
        return

    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["entry_time"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    strat_meta = {
        "Strategy 1: Nifty 15M Breakout": {"label": "S1: Nifty 15M Breakout", "color": "#2563eb"},
        "Strategy 2: BankNIFTY 15M Breakout": {"label": "S2: BankNifty 15M Breakout", "color": "#7c3aed"},
        "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)": {"label": "S3: Nifty DAS Strangle", "color": "#059669"},
        "Strategy 4: Nifty 1M Pullback Scalper": {"label": "S4: Nifty 1M Scalper", "color": "#d97706"}
    }

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    # Equity curves
    df["cum_pnl"] = df["net_pnl"].cumsum()
    df["portfolio_equity"] = initial_capital + df["cum_pnl"]
    df["peak_equity"] = df["portfolio_equity"].cummax()
    df["drawdown"] = df["portfolio_equity"] - df["peak_equity"]

    total_net = df["net_pnl"].sum()
    pct_return = (total_net / initial_capital) * 100.0
    max_dd = df["drawdown"].min()

    ax1.plot(df["timestamp"], df["portfolio_equity"], label=f"Quad Portfolio (₹{df['portfolio_equity'].iloc[-1]:,.0f} | {pct_return:+.1f}%)", color="#0f172a", lw=2.2)

    for strat, meta in strat_meta.items():
        sub = df[df["strategy"] == strat].copy()
        if not sub.empty:
            sub["strat_equity"] = initial_capital + sub["net_pnl"].cumsum()
            ax1.plot(sub["timestamp"], sub["strat_equity"], label=f"{meta['label']} (₹{sub['net_pnl'].sum():+,.0f})", color=meta["color"], lw=1.4, alpha=0.85)

    ax1.set_title(f"QUAD-STRATEGY QUANTITATIVE PORTFOLIO PERFORMANCE\nInitial Capital: ₹{initial_capital:,.0f} | Net Realized PnL: ₹{total_net:+,.2f} | Max Drawdown: ₹{abs(max_dd):,.0f}", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Capital (₹)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", framealpha=0.9)

    # Underwater Drawdown chart
    ax2.fill_between(df["timestamp"], df["drawdown"], 0, color="#ef4444", alpha=0.35, label=f"Max Drawdown: ₹{abs(max_dd):,.0f}")
    ax2.plot(df["timestamp"], df["drawdown"], color="#dc2626", lw=1.2)
    ax2.set_ylabel("Drawdown (₹)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left", frameon=True, facecolor="#ffffff", framealpha=0.9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()
    print(f"[SAVED] Performance chart generated: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Portfolio Equity Charts")
    parser.add_argument("--trades", type=str, required=True, help="Path to trades CSV")
    parser.add_argument("--output", type=str, default="equity_chart.png", help="Output PNG file path")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial Capital (₹)")
    args = parser.parse_args()

    plot_equity_charts(args.trades, args.output, args.capital)
