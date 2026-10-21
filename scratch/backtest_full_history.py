"""
Full Historical Multi-Year Quant Backtest (2024/2025 to 2026)
Simulates all 8 strategies on real 1-minute market data:
- NIFTY 50: 2025-01-01 to 2026-10-21 (389 trading sessions)
- BankNIFTY: 2024-11-01 to 2026-10-21 (402 trading sessions)
Utilizes Black-Scholes pricing with full transaction charges and slippage.
"""

import os
import sys
import math
import datetime
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from config import config
from daily_result_checker import (
    evaluate_strategy_1,
    evaluate_strategy_2,
    evaluate_strategy_3,
    evaluate_strategy_4,
    evaluate_strategy_5,
    evaluate_strategy_6,
    evaluate_strategy_7,
    evaluate_strategy_8
)

def run_historical_backtest():
    nifty_csv = "data/nifty_1min_real.csv"
    bn_csv = "data/banknifty_1min_real.csv"

    if not os.path.exists(nifty_csv) or not os.path.exists(bn_csv):
        print("[ERROR] Datasets missing.")
        return

    print("Loading datasets...")
    df_n = pd.read_csv(nifty_csv)
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n["date"] = df_n["timestamp"].dt.date
    df_n.sort_values("timestamp", inplace=True)
    df_n.reset_index(drop=True, inplace=True)

    df_bn = pd.read_csv(bn_csv)
    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
    df_bn["date"] = df_bn["timestamp"].dt.date
    df_bn.sort_values("timestamp", inplace=True)
    df_bn.reset_index(drop=True, inplace=True)

    # Pre-slice by date to maximize backtesting speed
    print("Pre-indexing trading sessions...")
    n_days = {d: group.copy() for d, group in df_n.groupby("date") if len(group) >= 15}
    bn_days = {d: group.copy() for d, group in df_bn.groupby("date") if len(group) >= 15}

    all_dates = sorted(list(set(n_days.keys()).union(set(bn_days.keys()))))

    print(f"\n=========================================================================================")
    print(f"   FULL MULTI-YEAR QUANT BACKTEST (2024/2025 - 2026)")
    print(f"   Date Range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} Total Trading Sessions)")
    print(f"   NIFTY Sessions: {len(n_days):,} ({len(df_n):,} candles)")
    print(f"   BankNIFTY Sessions: {len(bn_days):,} ({len(df_bn):,} candles)")
    print(f"=========================================================================================\n")

    strat_trades = {f"Strategy {i}": [] for i in range(1, 9)}
    processed = 0
    total_sessions = len(all_dates)

    start_time = datetime.datetime.now()

    for target_date in all_dates:
        processed += 1
        if processed % 50 == 0 or processed == total_sessions:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            print(f"[{processed}/{total_sessions}] Backtesting session {target_date} ({elapsed:.1f}s elapsed)...", flush=True)

        # NIFTY Strategies
        if target_date in n_days:
            day_n = n_days[target_date]

            # Strategy 1 (Nifty 15M Box Strangle)
            s1 = evaluate_strategy_1(day_n, target_date, api=None)
            if s1.get("trade_occurred"):
                strat_trades["Strategy 1"].append({"date": target_date, **s1})

            # Strategy 2 (Nifty Expiry Gamma Squeeze)
            s2 = evaluate_strategy_2(day_n, target_date, api=None)
            if s2.get("trade_occurred"):
                strat_trades["Strategy 2"].append({"date": target_date, **s2})

            # Strategy 3 (Nifty 30M Directional ITM)
            s3 = evaluate_strategy_3(day_n, target_date, api=None)
            if s3.get("trade_occurred"):
                strat_trades["Strategy 3"].append({"date": target_date, **s3})

            # Strategy 4 (Nifty Decoupled Strangle DAS)
            s4 = evaluate_strategy_4(day_n, target_date, api=None)
            if s4.get("trade_occurred"):
                if "all_trades" in s4 and len(s4["all_trades"]) > 1:
                    for sub in s4["all_trades"]:
                        strat_trades["Strategy 4"].append({"date": target_date, **sub})
                else:
                    strat_trades["Strategy 4"].append({"date": target_date, **s4})

        # BankNIFTY Strategies
        if target_date in bn_days:
            day_bn = bn_days[target_date]

            # Strategy 5 (BankNIFTY Decoupled Strangle)
            s5 = evaluate_strategy_5(day_bn, target_date, api=None)
            if s5.get("trade_occurred"):
                strat_trades["Strategy 5"].append({"date": target_date, **s5})

            # Strategy 6 (BankNIFTY Directional Box Breakout)
            s6 = evaluate_strategy_6(day_bn, target_date, api=None)
            if s6.get("trade_occurred"):
                strat_trades["Strategy 6"].append({"date": target_date, **s6})

            # Strategy 7 (BankNIFTY Expiry Gamma Squeeze GSM)
            s7 = evaluate_strategy_7(day_bn, target_date, api=None)
            if s7.get("trade_occurred"):
                strat_trades["Strategy 7"].append({"date": target_date, **s7})

            # Strategy 8 (BankNIFTY Hedged Strangle 15M Box)
            s8 = evaluate_strategy_8(day_bn, target_date, api=None)
            if s8.get("trade_occurred"):
                strat_trades["Strategy 8"].append({"date": target_date, **s8})

    # Strategy Metrics
    strat_names = {
        "Strategy 1": "Strategy 1: Hedged Strangle (Nifty 15M Box)",
        "Strategy 2": "Strategy 2: Expiry Gamma Squeeze (Nifty)",
        "Strategy 3": "Strategy 3: 30M Directional ITM (Nifty)",
        "Strategy 4": "Strategy 4: Decoupled Strangle (Nifty DAS)",
        "Strategy 5": "Strategy 5: Decoupled Strangle (BankNIFTY DAS)",
        "Strategy 6": "Strategy 6: Directional Box Breakout (BankNIFTY DBB)",
        "Strategy 7": "Strategy 7: Expiry Gamma Squeeze (BankNIFTY GSM)",
        "Strategy 8": "Strategy 8: Hedged Strangle (BankNIFTY 15M Box)",
    }

    all_trades_flat = []
    summary_data = []

    for s_key, trs in strat_trades.items():
        name = strat_names[s_key]
        if not trs:
            summary_data.append([name, 0, "0.0%", "0.00", "INR 0.00", "INR 0.00"])
            continue

        pnls = [float(t.get("net_pnl", 0.0)) for t in trs]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = (len(wins) / len(pnls)) * 100.0 if pnls else 0.0
        tot_win = sum(wins)
        tot_loss = abs(sum(losses))
        pf = (tot_win / tot_loss) if tot_loss > 0 else (99.0 if tot_win > 0 else 0.0)
        tot_pnl = sum(pnls)

        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd) if len(dd) > 0 else 0.0

        for t in trs:
            all_trades_flat.append({
                "date": str(t["date"]),
                "strategy": name,
                "strategy_id": s_key,
                "pnl": float(t.get("net_pnl", 0.0)),
                "exit_reason": t.get("exit_reason", "")
            })

        summary_data.append([
            name,
            f"{len(trs)} Trades",
            f"{wr:.1f}%",
            f"{pf:.2f} PF",
            f"INR {max_dd:,.2f}",
            f"INR {tot_pnl:+,.2f}"
        ])

    print("\n" + "=" * 90)
    print("STRATEGY PERFORMANCE METRICS (FULL 1.5 - 2 YEAR DATASET):")
    headers = ["Strategy", "Trades", "Win Rate", "Profit Factor", "Max Drawdown", "Net Realized Return"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Convert all flat trades to DataFrame
    df_all_trades = pd.DataFrame(all_trades_flat)
    if not df_all_trades.empty:
        df_all_trades["year"] = pd.to_datetime(df_all_trades["date"]).dt.year
        df_all_trades.to_csv("data/historical_backtest_results_2025_2026.csv", index=False)
        print(f"\n[SAVED] Saved {len(df_all_trades)} historical trades to data/historical_backtest_results_2025_2026.csv")

        # Yearly breakdown
        years = sorted(df_all_trades["year"].unique())
        print("\n" + "=" * 90)
        print("YEARLY BREAKDOWN:")
        yearly_rows = []
        for yr in years:
            sub = df_all_trades[df_all_trades["year"] == yr]
            w_sub = sub[sub["pnl"] > 0]
            l_sub = sub[sub["pnl"] <= 0]
            wr = (len(w_sub) / len(sub)) * 100.0 if len(sub) > 0 else 0.0
            w_sum = w_sub["pnl"].sum()
            l_sum = abs(l_sub["pnl"].sum())
            pf = (w_sum / l_sum) if l_sum > 0 else 99.0
            net = sub["pnl"].sum()
            yearly_rows.append([f"Year {yr}", len(sub), f"{wr:.1f}%", f"{pf:.2f} PF", f"INR {net:+,.2f}"])
        print(tabulate(yearly_rows, headers=["Period", "Trades", "Win Rate", "Profit Factor", "Net Return"], tablefmt="grid"))

        # Core 4 Directional vs Morning Strangles Comparison
        core4_keys = ["Strategy 2", "Strategy 3", "Strategy 4", "Strategy 7"]
        strangle_keys = ["Strategy 1", "Strategy 5", "Strategy 8"]

        df_c4 = df_all_trades[df_all_trades["strategy_id"].isin(core4_keys)]
        df_str = df_all_trades[df_all_trades["strategy_id"].isin(strangle_keys)]

        def calc_metrics(sub_df):
            if sub_df.empty:
                return 0, 0.0, 0.0, 0.0
            w = sub_df[sub_df["pnl"] > 0]
            l = sub_df[sub_df["pnl"] <= 0]
            wr = (len(w) / len(sub_df)) * 100.0
            w_sum = w["pnl"].sum()
            l_sum = abs(l["pnl"].sum())
            pf = (w_sum / l_sum) if l_sum > 0 else 99.0
            net = sub_df["pnl"].sum()
            return len(sub_df), wr, pf, net

        c4_n, c4_wr, c4_pf, c4_net = calc_metrics(df_c4)
        str_n, str_wr, str_pf, str_net = calc_metrics(df_str)
        all_n, all_wr, all_pf, all_net = calc_metrics(df_all_trades)

        print("\n" + "=" * 90)
        print("THE DEFINITIVE COMPARISON: CORE 4 DIRECTIONAL WINNERS vs 3 MORNING STRANGLES:")
        portfolio_comp = [
            ["Core 4 Directional Winners (S2, S3, S4, S7)", f"{c4_n} Trades", f"{c4_wr:.1f}%", f"{c4_pf:.2f} PF", f"INR {c4_net:+,.2f} ({(c4_net/10000)*100:+.1f}%)"],
            ["3 Losing Morning Strangles (S1, S5, S8)", f"{str_n} Trades", f"{str_wr:.1f}%", f"{str_pf:.2f} PF", f"INR {str_net:+,.2f} ({(str_net/10000)*100:+.1f}%)"],
            ["All Strategies Combined", f"{all_n} Trades", f"{all_wr:.1f}%", f"{all_pf:.2f} PF", f"INR {all_net:+,.2f} ({(all_net/10000)*100:+.1f}%)"]
        ]
        print(tabulate(portfolio_comp, headers=["Portfolio", "Total Trades", "Win Rate", "Profit Factor", "Total Net PnL (% on Rs 10k)"], tablefmt="grid"))
        print("=" * 90)

if __name__ == "__main__":
    run_historical_backtest()
