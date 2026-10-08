"""
Automated Trade Journal Manager & Historical Backfill Engine.
Manages:
1. data/audited_days.json - Tracks all audited historical trading dates to prevent duplicate evaluations.
2. data/active_positions.json - Tracks live in-flight positions across all 5 strategies.
3. data/trade_journal.csv - Single source of truth for all completed trades.
4. data/trade_journal.md - Live formatted markdown journal showing active open positions, performance stats, and trade ledger.
5. auto_catchup_missing_trading_days - Backfills and audits any missed trading days in chronological order.
"""

import os
import sys
import json
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from src.utils.git_sync import git_sync_push

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRADE_JOURNAL_CSV = os.path.join(DATA_DIR, "trade_journal.csv")
TRADE_JOURNAL_MD = os.path.join(DATA_DIR, "trade_journal.md")
ACTIVE_POSITIONS_JSON = os.path.join(DATA_DIR, "active_positions.json")
AUDITED_DAYS_JSON = os.path.join(DATA_DIR, "audited_days.json")
LEDGER_CSV = os.path.join(DATA_DIR, "paper_trading_ledger.csv")

def load_audited_days() -> List[str]:
    """Loads list of audited trading dates (YYYY-MM-DD)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(AUDITED_DAYS_JSON):
        try:
            with open(AUDITED_DAYS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return sorted(list(set(str(d) for d in data)))
        except Exception:
            pass
    # Fallback initialize from trade_journal.csv if audited_days.json doesn't exist
    dates = set()
    if os.path.exists(TRADE_JOURNAL_CSV):
        try:
            df = pd.read_csv(TRADE_JOURNAL_CSV)
            if not df.empty and "date" in df.columns:
                dates = set(df["date"].dropna().astype(str).unique().tolist())
        except Exception:
            pass
    return sorted(list(dates))

def save_audited_days(audited_days: List[str]):
    """Persists audited trading dates list to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    clean_days = sorted(list(set(str(d) for d in audited_days)))
    with open(AUDITED_DAYS_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_days, f, indent=2)

def mark_day_as_audited(date_str: str):
    """Marks a specific trading day as audited."""
    days = load_audited_days()
    d_clean = str(date_str)
    if d_clean not in days:
        days.append(d_clean)
        save_audited_days(days)

def load_active_positions() -> Dict[str, Any]:
    """Loads current live active in-flight positions."""
    if os.path.exists(ACTIVE_POSITIONS_JSON):
        try:
            with open(ACTIVE_POSITIONS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def save_active_positions(positions: Dict[str, Any]):
    """Persists active positions to active_positions.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    clean_pos = {}
    for k, v in positions.items():
        if v is not None:
            c = dict(v)
            if "entry_time" in c and hasattr(c["entry_time"], "isoformat"):
                c["entry_time"] = c["entry_time"].isoformat()
            # remove unpicklable objects like SmartConnect contract objects if present
            if "c_ce" in c and isinstance(c["c_ce"], dict):
                c["c_ce"] = {k1: str(v1) for k1, v1 in c["c_ce"].items() if k1 in ["token", "symbol"]}
            if "c_pe" in c and isinstance(c["c_pe"], dict):
                c["c_pe"] = {k1: str(v1) for k1, v1 in c["c_pe"].items() if k1 in ["token", "symbol"]}
            clean_pos[k] = c
    with open(ACTIVE_POSITIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_pos, f, indent=2)

def log_trade_to_journal(target_date, strategy_name: str, trade_res: Dict[str, Any]):
    """
    Appends a closed trade to trade_journal.csv without duplicates,
    and updates paper_trading_ledger.csv and trade_journal.md.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    df_j = pd.DataFrame()
    if os.path.exists(TRADE_JOURNAL_CSV):
        try:
            df_j = pd.read_csv(TRADE_JOURNAL_CSV)
        except Exception:
            df_j = pd.DataFrame()

    entry_t = str(trade_res.get("entry_time", ""))
    leg_str = str(trade_res.get("leg", trade_res.get("opt_type", "CE+PE")))
    strike_str = str(trade_res.get("strike", ""))

    # Deduplication check
    if not df_j.empty and "date" in df_j.columns and "entry_time" in df_j.columns:
        dup = df_j[
            (df_j["date"].astype(str) == str(target_date)) &
            (df_j["strategy"] == strategy_name) &
            (df_j["entry_time"].astype(str) == entry_t) &
            (df_j["strike"].astype(str) == strike_str)
        ]
        if not dup.empty:
            return

    row = {
        "date": str(target_date),
        "strategy": strategy_name,
        "leg": leg_str,
        "strike": strike_str,
        "entry_time": entry_t,
        "exit_time": str(trade_res.get("exit_time", "")),
        "entry_p": round(float(trade_res.get("entry_p", 0.0)), 2),
        "exit_p": round(float(trade_res.get("exit_p", 0.0)), 2),
        "capital_used": round(float(trade_res.get("cost", trade_res.get("capital_used", 0.0))), 2),
        "gross_pnl": round(float(trade_res.get("gross_pnl", trade_res.get("gross", 0.0))), 2),
        "charges": round(float(trade_res.get("charges", 0.0)), 2),
        "net_pnl": round(float(trade_res.get("net_pnl", trade_res.get("net", 0.0))), 2),
        "exit_reason": str(trade_res.get("exit_reason", "")),
        "data_feed": str(trade_res.get("data_feed", "Angel One Real Traded"))
    }

    df_j = pd.concat([df_j, pd.DataFrame([row])], ignore_index=True)
    df_j.sort_values(by=["date", "entry_time"], inplace=True)
    df_j.to_csv(TRADE_JOURNAL_CSV, index=False)

    # Sync ledger
    if os.path.exists(LEDGER_CSV):
        try:
            df_ledger = pd.read_csv(LEDGER_CSV)
            if not ((df_ledger["date"].astype(str) == str(target_date)) & (df_ledger["entry_time"].astype(str) == entry_t)).any():
                last_cum = df_ledger["cum_net"].iloc[-1] if not df_ledger.empty and "cum_net" in df_ledger.columns else 0.0
                new_cum = round(last_cum + row["net_pnl"], 2)
                l_row = {
                    "entry_time": row["entry_time"], "exit_time": row["exit_time"],
                    "gross": row["gross_pnl"], "charges": row["charges"], "net": row["net_pnl"],
                    "exit_reason": row["exit_reason"], "bars": 26, "cum_net": new_cum,
                    "date": row["date"], "net_pnl": row["net_pnl"], "gross_pnl": row["gross_pnl"]
                }
                df_ledger = pd.concat([df_ledger, pd.DataFrame([l_row])], ignore_index=True)
                df_ledger.to_csv(LEDGER_CSV, index=False)
        except Exception:
            pass

    refresh_markdown_journal()

def refresh_markdown_journal():
    """
    Regenerates data/trade_journal.md with:
    1. 🟢 Live Active Open Positions (In-Flight)
    2. 📊 Performance Summary
    3. 📜 Complete Chronological Trade Ledger
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load active positions
    active_pos = load_active_positions()
    active_rows = []
    for s_key, pos in active_pos.items():
        if pos:
            e_time = pos.get("entry_time", "")
            if isinstance(e_time, str) and "T" in e_time:
                e_time = e_time.split("T")[-1].split(".")[0]
            elif hasattr(e_time, "strftime"):
                e_time = e_time.strftime("%H:%M:%S")

            strat_label = pos.get("strategy_name", f"Strategy {s_key.upper()}")
            strike = pos.get("strike", f"{pos.get('ce_symbol', '')} + {pos.get('pe_symbol', '')}")
            side = pos.get("opt_type", "CE+PE")
            entry_p = pos.get("entry_tot", pos.get("entry_p", 0.0))
            cost = pos.get("cost", 0.0)

            # Target & Stop Loss
            tgt = pos.get("target_p", "")
            if not tgt:
                ce_tgt = pos.get("ce_tgt", 0)
                pe_tgt = pos.get("pe_tgt", 0)
                tgt = f"CE: ₹{ce_tgt:.1f} | PE: ₹{pe_tgt:.1f}" if ce_tgt else "N/A"
            else:
                tgt = f"₹{float(tgt):.2f}"

            sl = pos.get("stop_p", "")
            if not sl:
                ce_sl = pos.get("ce_sl", 0)
                pe_sl = pos.get("pe_sl", 0)
                sl = f"CE: ₹{ce_sl:.1f} | PE: ₹{pe_sl:.1f}" if ce_sl else "N/A"
            else:
                sl = f"₹{float(sl):.2f}"

            active_rows.append({
                "strategy": strat_label,
                "side": side,
                "strike": strike,
                "entry_time": str(e_time),
                "entry_p": f"₹{float(entry_p):.2f}",
                "tgt": tgt,
                "sl": sl,
                "cost": f"₹{float(cost):.2f}",
                "status": "🟢 ACTIVE / IN-PROGRESS"
            })

    # Load trade ledger
    df_j = pd.DataFrame()
    if os.path.exists(TRADE_JOURNAL_CSV):
        try:
            df_j = pd.read_csv(TRADE_JOURNAL_CSV)
            if not df_j.empty and "date" in df_j.columns and "entry_time" in df_j.columns:
                df_j.sort_values(by=["date", "entry_time"], inplace=True)
        except Exception:
            df_j = pd.DataFrame()

    with open(TRADE_JOURNAL_MD, "w", encoding="utf-8") as f:
        f.write("# 📈 Live Option Trading Journal & Paper Execution Log\n\n")
        f.write(f"*Last Updated: {now_str} IST*\n\n")
        f.write("---\n\n")

        # 1. Active Positions Section
        f.write("## 🟢 Live Active Open Positions (In-Flight)\n\n")
        if active_rows:
            f.write("| Strategy | Side | Strike / Contract | Entry Time | Entry Price | Target Price | Stop Loss | Capital Deployed | Status |\n")
            f.write("|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for ar in active_rows:
                f.write(f"| **{ar['strategy']}** | `{ar['side']}` | {ar['strike']} | {ar['entry_time']} | {ar['entry_p']} | {ar['tgt']} | {ar['sl']} | {ar['cost']} | `{ar['status']}` |\n")
            f.write("\n")
        else:
            f.write("> ℹ️ **No active open positions currently in-flight.**  \n")
            f.write("> *Capital deployed: ₹0.00 | Bot is actively scanning 1-minute market candles for breakout triggers.*  \n\n")

        f.write("---\n\n")

        # 2. Performance Summary
        f.write("## 📊 Live Paper Trading Performance Summary\n\n")
        tot = len(df_j) if not df_j.empty else 0
        wins = len(df_j[df_j["net_pnl"] > 0]) if not df_j.empty and "net_pnl" in df_j.columns else 0
        losses = len(df_j[df_j["net_pnl"] <= 0]) if not df_j.empty and "net_pnl" in df_j.columns else 0
        wr = (wins / tot * 100) if tot > 0 else 0.0
        tot_pnl = float(df_j["net_pnl"].sum()) if not df_j.empty and "net_pnl" in df_j.columns else 0.0
        current_bal = 10000.0 + tot_pnl

        f.write(f"- **Starting Portfolio Capital**: INR 10,000.00\n")
        f.write(f"- **Total Executed Trades**: {tot}\n")
        f.write(f"- **Win Rate**: {wr:.1f}% ({wins} Wins / {losses} Losses)\n")
        f.write(f"- **Net Realized PnL**: **INR {tot_pnl:+,.2f}**\n")
        f.write(f"- **Current Account Balance**: **INR {current_bal:+,.2f}** ({((current_bal-10000)/10000*100):+.1f}% Return)\n\n")

        f.write("---\n\n")

        # 3. Complete Trade Ledger
        f.write("## 📜 Complete Historical Trade Ledger\n\n")
        if not df_j.empty:
            f.write("| Date | Strategy | Leg | Strike | Entry | Exit | Entry P | Exit P | Capital | Net PnL | Exit Reason | Data Feed |\n")
            f.write("|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|\n")
            for _, r in df_j.iterrows():
                net_val = float(r["net_pnl"])
                net_str = f"**+₹{net_val:.2f}**" if net_val > 0 else f"**-₹{abs(net_val):.2f}**"
                f.write(f"| {r['date']} | {r['strategy']} | {r['leg']} | {r['strike']} | {r['entry_time']} | {r['exit_time']} | ₹{float(r['entry_p']):.2f} | ₹{float(r['exit_p']):.2f} | ₹{float(r['capital_used']):.2f} | {net_str} | {r['exit_reason']} | {r['data_feed']} |\n")
            f.write("\n")
        else:
            f.write("*No closed trades logged yet.*\n\n")

def auto_catchup_missing_trading_days(api, df_all: pd.DataFrame, df_bn: pd.DataFrame, up_to_date=None) -> List[str]:
    """
    Detects any historical trading days present in candles that have not yet been audited.
    Runs evaluation chronologically for each missing day and records trades to journal.
    Returns list of dates that were caught up.
    """
    if df_all.empty:
        return []

    # Import strategy evaluation functions lazily to avoid circular imports
    from daily_result_checker import (
        evaluate_strategy_1,
        evaluate_strategy_2,
        evaluate_strategy_3,
        evaluate_strategy_4,
        evaluate_strategy_5
    )

    df_all_dt = df_all.copy()
    df_all_dt["date_only"] = pd.to_datetime(df_all_dt["timestamp"]).dt.date
    min_date = datetime.date(2026, 9, 21) # Start date of active paper trading journal
    available_dates = sorted([d for d in df_all_dt["date_only"].unique().tolist() if d >= min_date])

    if up_to_date is not None:
        available_dates = [d for d in available_dates if d <= up_to_date]

    audited_days = load_audited_days()
    missed_dates = [d for d in available_dates if str(d) not in audited_days]

    if not missed_dates:
        print(f"[AUTO-CATCHUP] All {len(available_dates)} historical trading days are already audited. Up to date!")
        return []

    print(f"\n[AUTO-CATCHUP] [NOTICE] Found {len(missed_dates)} un-audited historical trading day(s): {[str(d) for d in missed_dates]}")
    print("[AUTO-CATCHUP] Automatically evaluating and backfilling missed days in chronological order...\n")

    caught_up = []
    for d in missed_dates:
        print(f"=== [AUTO-CATCHUP] Auditing Missed Day: {d.strftime('%A, %d-%b-%Y')} ===")
        s1_res = evaluate_strategy_1(df_all, d, api=api)
        s2_res = evaluate_strategy_2(df_all, d, api=api)
        s3_res = evaluate_strategy_3(df_all, d, api=api)
        s4_res = evaluate_strategy_4(df_all, d, api=api)
        s5_res = evaluate_strategy_5(df_bn, d, api=api) if not df_bn.empty else {"status": "NO_DATA"}

        # Log any trades that occurred
        for s_name, s_res in [
            ("Strategy 1: Hedged Strangle", s1_res),
            ("Strategy 2: Expiry Gamma Squeeze", s2_res),
            ("Strategy 3: 30M Directional ITM", s3_res),
            ("Strategy 4: Decoupled Asymmetric Strangle", s4_res),
            ("Strategy 5: BankNIFTY Decoupled Strangle (DAS)", s5_res)
        ]:
            if s_res.get("trade_occurred", False):
                trades_to_log = s_res["all_trades"] if ("all_trades" in s_res and len(s_res["all_trades"]) > 1) else [s_res]
                for sub_tr in trades_to_log:
                    log_trade_to_journal(d, s_name, sub_tr)
                    print(f"  -> Recorded Trade: {s_name} | Net: INR {sub_tr.get('net_pnl', 0.0):+.2f} ({sub_tr.get('exit_reason', '')})")

        mark_day_as_audited(str(d))
        caught_up.append(str(d))
        print(f"=== [AUTO-CATCHUP] Completed Audit for {d} ===\n")

    # Refresh journal & push changes
    refresh_markdown_journal()
    git_sync_push(f"Auto-Catchup: Backfilled missed trading days: {', '.join(caught_up)}")
    return caught_up
