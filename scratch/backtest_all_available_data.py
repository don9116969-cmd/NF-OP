"""
Comprehensive Multi-Month Black-Scholes Backtest across all 8 Strategies:
Runs on all available real 1-minute market data (August 2026 to October 2026, 18,048+ candles).
Evaluates whether losing strategies (S1, S5, S6, S8) fail consistently or only in October.
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

def run_multi_month_backtest():
    nifty_csv = "data/nifty_1min_real.csv"
    bn_csv = "data/banknifty_1min_real.csv"
    
    if not os.path.exists(nifty_csv) or not os.path.exists(bn_csv):
        print("[ERROR] Candle files missing.")
        return

    df_n = pd.read_csv(nifty_csv)
    df_n["timestamp"] = pd.to_datetime(df_n["timestamp"]).dt.tz_localize(None)
    df_n.sort_values("timestamp", inplace=True)
    df_n.reset_index(drop=True, inplace=True)

    df_bn = pd.read_csv(bn_csv)
    df_bn["timestamp"] = pd.to_datetime(df_bn["timestamp"]).dt.tz_localize(None)
    df_bn.sort_values("timestamp", inplace=True)
    df_bn.reset_index(drop=True, inplace=True)

    # Unique dates
    n_dates = sorted(df_n["timestamp"].dt.date.unique())
    bn_dates = sorted(df_bn["timestamp"].dt.date.unique())
    all_dates = sorted(list(set(n_dates).union(set(bn_dates))))

    print(f"=========================================================================================")
    print(f"   MULTI-MONTH COMPREHENSIVE QUANT BACKTEST (BLACK-SCHOLES SIMULATION MODE)")
    print(f"   Date Range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} Trading Sessions)")
    print(f"   Nifty Candles: {len(df_n):,} | BankNifty Candles: {len(df_bn):,}")
    print(f"=========================================================================================\n")

    strat_trades = {f"Strategy {i}": [] for i in range(1, 9)}

    for target_date in all_dates:
        # Strategy 1 (Nifty 15M Box)
        s1 = evaluate_strategy_1(df_n, target_date, api=None)
        if s1.get("trade_occurred"):
            strat_trades["Strategy 1"].append({"date": target_date, **s1})

        # Strategy 2 (Nifty Expiry Gamma)
        s2 = evaluate_strategy_2(df_n, target_date, api=None)
        if s2.get("trade_occurred"):
            strat_trades["Strategy 2"].append({"date": target_date, **s2})

        # Strategy 3 (Nifty 30M Directional ITM)
        s3 = evaluate_strategy_3(df_n, target_date, api=None)
        if s3.get("trade_occurred"):
            strat_trades["Strategy 3"].append({"date": target_date, **s3})

        # Strategy 4 (Nifty DAS)
        s4 = evaluate_strategy_4(df_n, target_date, api=None)
        if s4.get("trade_occurred"):
            if "all_trades" in s4 and len(s4["all_trades"]) > 1:
                for sub in s4["all_trades"]:
                    strat_trades["Strategy 4"].append({"date": target_date, **sub})
            else:
                strat_trades["Strategy 4"].append({"date": target_date, **s4})

        # BankNifty strategies
        if target_date in bn_dates:
            # Strategy 5 (BankNIFTY DAS)
            s5 = evaluate_strategy_5(df_bn, target_date, api=None)
            if s5.get("trade_occurred"):
                strat_trades["Strategy 5"].append({"date": target_date, **s5})

            # Strategy 6 (BankNIFTY DBB)
            s6 = evaluate_strategy_6(df_bn, target_date, api=None)
            if s6.get("trade_occurred"):
                strat_trades["Strategy 6"].append({"date": target_date, **s6})

            # Strategy 7 (BankNIFTY GSM)
            s7 = evaluate_strategy_7(df_bn, target_date, api=None)
            if s7.get("trade_occurred"):
                strat_trades["Strategy 7"].append({"date": target_date, **s7})

            # Strategy 8 (BankNIFTY Hedged Strangle)
            s8 = evaluate_strategy_8(df_bn, target_date, api=None)
            if s8.get("trade_occurred"):
                strat_trades["Strategy 8"].append({"date": target_date, **s8})

    # Summary Metrics Table
    summary_data = []
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

    all_pnls = []

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

        # Drawdown calculation
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = np.max(dd) if len(dd) > 0 else 0.0

        for t in trs:
            all_pnls.append({"date": str(t["date"]), "strategy": name, "pnl": float(t.get("net_pnl", 0.0))})

        summary_data.append([
            name,
            f"{len(trs)} Trades",
            f"{wr:.1f}%",
            f"{pf:.2f} PF",
            f"INR {max_dd:,.2f}",
            f"INR {tot_pnl:+,.2f}"
        ])

    headers = ["Strategy", "Trades", "Win Rate", "Profit Factor", "Max Drawdown", "Net Realized Return"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Combined Portfolio Metrics
    if all_pnls:
        df_all_tr = pd.DataFrame(all_pnls).sort_values("date")
        tot_c = len(df_all_tr)
        w_c = len(df_all_tr[df_all_tr["pnl"] > 0])
        wr_tot = (w_c / tot_c) * 100.0 if tot_c > 0 else 0.0
        w_sum = df_all_tr[df_all_tr["pnl"] > 0]["pnl"].sum()
        l_sum = abs(df_all_tr[df_all_tr["pnl"] <= 0]["pnl"].sum())
        pf_tot = (w_sum / l_sum) if l_sum > 0 else 99.0
        net_tot = df_all_tr["pnl"].sum()
        cum_tot = df_all_tr["pnl"].cumsum()
        peak_tot = np.maximum.accumulate(cum_tot)
        dd_tot = np.max(peak_tot - cum_tot) if len(cum_tot) > 0 else 0.0

        print("\n" + "=" * 90)
        print(f"ALL 8 STRATEGIES COMBINED ACROSS 50 TRADING SESSIONS (AUG - OCT 2026):")
        print(f"Total Trades: {tot_c} | Win Rate: {wr_tot:.1f}% | Profit Factor: {pf_tot:.2f}")
        print(f"Max Portfolio Drawdown: INR {dd_tot:,.2f}")
        print(f"Total Net PnL: INR {net_tot:+,.2f} on INR 10,000 Base Account")
        print(f"Ending Balance: INR {10000 + net_tot:,.2f} ({(net_tot/10000)*100:+.1f}% Overall Return)")
        print("=" * 90)

        # Core 4 Directional Momentum (S2, S3, S4, S7) Breakdown
        core4_names = [
            "Strategy 2: Expiry Gamma Squeeze (Nifty)",
            "Strategy 3: 30M Directional ITM (Nifty)",
            "Strategy 4: Decoupled Strangle (Nifty DAS)",
            "Strategy 7: Expiry Gamma Squeeze (BankNIFTY GSM)"
        ]
        df_core4 = df_all_tr[df_all_tr["strategy"].isin(core4_names)]
        if not df_core4.empty:
            c4_trades = len(df_core4)
            c4_wins = len(df_core4[df_core4["pnl"] > 0])
            c4_wr = (c4_wins / c4_trades) * 100.0
            c4_wsum = df_core4[df_core4["pnl"] > 0]["pnl"].sum()
            c4_lsum = abs(df_core4[df_core4["pnl"] <= 0]["pnl"].sum())
            c4_pf = (c4_wsum / c4_lsum) if c4_lsum > 0 else 99.0
            c4_net = df_core4["pnl"].sum()

            print(f"\n=========================================================================================")
            print(f"BENCHMARK: CORE 4 DIRECTIONAL WINNERS ONLY (S2, S3, S4, S7):")
            print(f"Trades: {c4_trades} | Win Rate: {c4_wr:.1f}% | Profit Factor: {c4_pf:.2f} PF")
            print(f"Net Realized PnL: INR {c4_net:+,.2f} ({(c4_net/10000)*100:+.1f}% Return on Capital!)")
            print(f"=========================================================================================")

if __name__ == "__main__":
    run_multi_month_backtest()

