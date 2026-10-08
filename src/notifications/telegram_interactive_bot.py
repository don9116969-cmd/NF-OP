"""
Interactive Telegram Bot Engine for Nifty Option Paper Trading.
Provides 1-tap interactive menu buttons and live commands:
- 📊 Live Status & PnL (/status, /pnl): Displays in-flight open positions, entry price, targets, SL, and live unrealized PnL.
- 📜 Today's Trades (/today): Displays today's executed trades, win/loss breakdown, and net daily PnL.
- 💼 Portfolio Balance (/summary, /balance): Displays cumulative PnL, current capital, and overall performance.
- 📖 Trade Journal (/journal): Shows the latest closed trades from the permanent trade ledger.
"""

import os
import time
import json
import threading
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

from .notifier import (
    get_telegram_token,
    get_or_detect_chat_id,
    get_main_keyboard_markup,
    get_inline_status_markup,
    send_telegram_alert
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
ACTIVE_POS_FILE = os.path.join(DATA_DIR, "active_positions.json")
JOURNAL_CSV_FILE = os.path.join(DATA_DIR, "trade_journal.csv")
JOURNAL_MD_FILE = os.path.join(DATA_DIR, "trade_journal.md")

NIFTY_CANDLE_FILE = os.path.join(DATA_DIR, "nifty_1min_real.csv")
BANKNIFTY_CANDLE_FILE = os.path.join(DATA_DIR, "banknifty_1min_real.csv")

def get_ist_now() -> datetime:
    """Returns current datetime in Indian Standard Time (IST)."""
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

def get_latest_spot_price(underlying: str = "NIFTY") -> float:
    """Gets the latest recorded spot price from local candle datasets."""
    try:
        fpath = BANKNIFTY_CANDLE_FILE if "BANK" in underlying.upper() else NIFTY_CANDLE_FILE
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])
    except Exception:
        pass
    return 52200.0 if "BANK" in underlying.upper() else 25100.0

def load_live_active_positions() -> dict:
    """Loads active positions currently in-flight from active_positions.json."""
    if os.path.exists(ACTIVE_POS_FILE):
        try:
            with open(ACTIVE_POS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if v is not None}
        except Exception:
            pass
    return {}

def format_live_status_card(api=None) -> tuple[str, dict]:
    """Generates the live status & PnL message card and inline keyboard."""
    ist_now = get_ist_now()
    active_pos = load_live_active_positions()
    inline_kb = get_inline_status_markup()

    if not active_pos:
        today_str = str(ist_now.date())
        today_trades_count = 0
        today_net_pnl = 0.0
        if os.path.exists(JOURNAL_CSV_FILE):
            try:
                df = pd.read_csv(JOURNAL_CSV_FILE)
                if not df.empty and "date" in df.columns and "net_pnl" in df.columns:
                    t_df = df[df["date"].astype(str) == today_str]
                    today_trades_count = len(t_df)
                    today_net_pnl = float(t_df["net_pnl"].sum())
            except Exception:
                pass

        p_sign = "+" if today_net_pnl > 0 else ""
        today_perf_str = (
            f"• Executed Trades: {today_trades_count}\n"
            f"• Realized Net PnL: *{p_sign}₹{today_net_pnl:+,.2f}*\n"
        ) if today_trades_count > 0 else "• Closed Trades Today: 0 (No completed exits yet)\n"

        msg = (
            f"⚪ *NO OPEN POSITIONS IN-FLIGHT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Status Time:* {ist_now.strftime('%I:%M:%S %p')} IST\n"
            f"📅 *Date:* {ist_now.strftime('%A, %d-%b-%Y')}\n"
            f"💼 *Capital Deployed:* ₹0.00 / ₹10,000.00 Max\n"
            f"🛡️ *State:* `Standby & Monitoring Live Candles`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 *Today's Session So Far:*\n"
            f"{today_perf_str}"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Bot is actively scanning 1-minute market candles across all 5 quantitative strategies:\n"
            f"• Strategy 1: Hedged Strangle (Nifty 15M Box)\n"
            f"• Strategy 2: Expiry Gamma Squeeze (Nifty Mon/Tue)\n"
            f"• Strategy 3: 30M Statistical Directional ITM (Nifty)\n"
            f"• Strategy 4: Decoupled Strangle (Nifty DAS)\n"
            f"• Strategy 5: Decoupled Strangle (BankNifty DAS)\n\n"
            f"🔔 *Instant alert will fire the exact second a trade enters!*"
        )
        return msg, inline_kb

    # Format active position
    msg = (
        f"🟢 *ACTIVE IN-FLIGHT POSITION DETECTED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ *Check Time:* {ist_now.strftime('%I:%M:%S %p')} IST\n"
    )

    for s_key, pos in active_pos.items():
        s_name = pos.get("strategy_name") or f"Strategy {s_key.upper()}"
        is_banknifty = "banknifty" in s_name.lower() or "s5" in str(s_key).lower()
        underlying = "BANKNIFTY" if is_banknifty else "NIFTY"
        lot_size = 30 if is_banknifty else 65

        entry_t_str = str(pos.get("entry_time", ""))
        cost = float(pos.get("cost", 0.0))
        entry_tot = float(pos.get("entry_tot", pos.get("entry_p", 0.0)))
        spot_entry = float(pos.get("spot_entry", 0.0))
        cur_spot = get_latest_spot_price(underlying)
        if spot_entry == 0.0:
            spot_entry = cur_spot
        spot_diff = cur_spot - spot_entry

        # Elapsed minutes
        elapsed_str = "Recent"
        try:
            if "entry_time" in pos and pos["entry_time"]:
                e_dt = datetime.fromisoformat(str(pos["entry_time"]).replace("Z", ""))
                mins = int((ist_now.replace(tzinfo=None) - e_dt.replace(tzinfo=None)).total_seconds() / 60.0)
                elapsed_str = f"{mins} mins"
        except Exception:
            pass

        ce_sym = pos.get("ce_symbol") or (f"{pos.get('ce_strike', '')}CE" if pos.get("ce_strike") else "")
        pe_sym = pos.get("pe_symbol") or (f"{pos.get('pe_strike', '')}PE" if pos.get("pe_strike") else "")
        sym_str = pos.get("strike") or pos.get("symbol") or f"{ce_sym} + {pe_sym}"

        # Calculate estimated live unrealized PnL
        if "ce_entry" in pos and "pe_entry" in pos:
            ce_p = float(pos.get("ce_entry", 0.0))
            pe_p = float(pos.get("pe_entry", 0.0))
            ce_tgt = float(pos.get("ce_tgt", ce_p * 2.0))
            ce_sl = float(pos.get("ce_sl", ce_p * 0.85))
            pe_tgt = float(pos.get("pe_tgt", pe_p * 2.0))
            pe_sl = float(pos.get("pe_sl", pe_p * 0.85))
            ce_status = "Exited" if pos.get("ce_exited") else "Active"
            pe_status = "Exited" if pos.get("pe_exited") else "Active"

            ce_est = float(pos["ce_exit_p"]) if pos.get("ce_exited") and pos.get("ce_exit_p") else max(0.5, round(ce_p + (spot_diff * 0.45), 2))
            pe_est = float(pos["pe_exit_p"]) if pos.get("pe_exited") and pos.get("pe_exit_p") else max(0.5, round(pe_p - (spot_diff * 0.45), 2))
            cur_tot = ce_est + pe_est
            unrealized_pnl = round(((ce_est - ce_p) + (pe_est - pe_p)) * lot_size, 2)
            unrealized_pct = round(((cur_tot - entry_tot) / max(0.1, entry_tot)) * 100.0, 1)

            p_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            pnl_str = f"+₹{unrealized_pnl:,.2f}" if unrealized_pnl >= 0 else f"-₹{abs(unrealized_pnl):,.2f}"
            pct_str = f"+{unrealized_pct:.1f}%" if unrealized_pct >= 0 else f"{unrealized_pct:.1f}%"

            msg += (
                f"📊 *Strategy:* `{s_name}`\n"
                f"🎯 *Contract(s):* `{sym_str}`\n"
                f"⏰ *Entry Time:* `{entry_t_str}` ({elapsed_str} in-flight)\n"
                f"📍 *Spot:* Entry `{spot_entry:.2f}` → Live `{cur_spot:.2f}` ({spot_diff:+.2f} pts)\n"
                f"💰 *Capital Used:* *₹{cost:,.2f}* (1 Lot | Limit: ₹10,000)\n"
                f"💵 *Combined Premium:* Entry ₹{entry_tot:.2f} → Cur ₹{cur_tot:.2f}\n"
                f"{p_emoji} *LIVE UNREALIZED PnL:* *{pnl_str}* ({pct_str})\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🟢 *CALL Leg ({ce_status}):* `{ce_sym}`\n"
                f"   • Entry: ₹{ce_p:.2f} | Cur: ₹{ce_est:.2f} | Tgt (+100%): ₹{ce_tgt:.2f} | SL (-15%): ₹{ce_sl:.2f}\n"
                f"🔴 *PUT Leg ({pe_status}):* `{pe_sym}`\n"
                f"   • Entry: ₹{pe_p:.2f} | Cur: ₹{pe_est:.2f} | Tgt (+100%): ₹{pe_tgt:.2f} | SL (-15%): ₹{pe_sl:.2f}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
            )
        else:
            p_val = float(pos.get("entry_p", 0.0))
            leg_type = str(pos.get("opt_type", "CE")).upper()
            tgt_p = float(pos.get("target_p", p_val * 1.80))
            sl_p = float(pos.get("stop_p", p_val * 0.75))
            mult = 0.55 if leg_type == "CE" else -0.55
            cur_p = max(0.5, round(p_val + (spot_diff * mult), 2))
            unrealized_pnl = round((cur_p - p_val) * lot_size, 2)
            unrealized_pct = round(((cur_p - p_val) / max(0.1, p_val)) * 100.0, 1)

            p_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            pnl_str = f"+₹{unrealized_pnl:,.2f}" if unrealized_pnl >= 0 else f"-₹{abs(unrealized_pnl):,.2f}"
            pct_str = f"+{unrealized_pct:.1f}%" if unrealized_pct >= 0 else f"{unrealized_pct:.1f}%"

            msg += (
                f"📊 *Strategy:* `{s_name}`\n"
                f"🎯 *Contract:* `{sym_str}`\n"
                f"⏰ *Entry Time:* `{entry_t_str}` ({elapsed_str} in-flight)\n"
                f"📍 *Spot:* Entry `{spot_entry:.2f}` → Live `{cur_spot:.2f}` ({spot_diff:+.2f} pts)\n"
                f"💰 *Capital Used:* *₹{cost:,.2f}* (1 Lot | Limit: ₹10,000)\n"
                f"💵 *Premium:* Entry ₹{p_val:.2f} → Cur ₹{cur_p:.2f}\n"
                f"{p_emoji} *LIVE UNREALIZED PnL:* *{pnl_str}* ({pct_str})\n"
                f"🎯 *Target Price:* ₹{tgt_p:.2f} | 🛑 *Stop Loss:* ₹{sl_p:.2f}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
            )

        feed = pos.get("data_feed", "Angel One Real Traded")
        msg += (
            f"🛡️ *Risk Management:* Decoupled SL & Profit Protection Active\n"
            f"📡 *Feed:* `{feed}`\n"
        )

    return msg, inline_kb

def format_today_trades_card() -> tuple[str, dict]:
    """Generates today's executed trades report."""
    ist_now = get_ist_now()
    today_str = str(ist_now.date())
    inline_kb = get_inline_status_markup()

    if not os.path.exists(JOURNAL_CSV_FILE):
        return f"ℹ️ *No trade journal found yet.*", inline_kb

    try:
        df = pd.read_csv(JOURNAL_CSV_FILE)
    except Exception as e:
        return f"[WARN] Failed to read trade journal: {e}", inline_kb

    if df.empty or "date" not in df.columns:
        return f"ℹ️ *No trades recorded in journal yet.*", inline_kb

    today_trades = df[df["date"].astype(str) == today_str]

    if today_trades.empty:
        msg = (
            f"📜 *TODAY'S TRADE SUMMARY: {ist_now.strftime('%A, %d-%b-%Y')}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"ℹ️ *No closed trades executed yet today.*\n\n"
            f"Bot is monitoring live candles during market hours (09:15 to 15:25 IST).\n"
            f"Tap *'📊 Live Status & PnL'* to check current active open positions."
        )
        return msg, inline_kb

    count = len(today_trades)
    net_pnl = float(today_trades["net_pnl"].sum())
    gross_pnl = float(today_trades["gross_pnl"].sum()) if "gross_pnl" in today_trades.columns else net_pnl
    charges = float(today_trades["charges"].sum()) if "charges" in today_trades.columns else 0.0
    wins = len(today_trades[today_trades["net_pnl"] > 0])
    losses = len(today_trades[today_trades["net_pnl"] <= 0])
    p_sign = "+" if net_pnl > 0 else ""

    msg = (
        f"📜 *TODAY'S TRADE SUMMARY: {ist_now.strftime('%A, %d-%b-%Y')}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔢 *Total Executed Trades:* {count} ({wins} Wins / {losses} Losses)\n"
        f"💵 *Gross PnL:* ₹{gross_pnl:+,.2f}\n"
        f"📉 *Charges & Brokerage:* ₹{charges:,.2f}\n"
        f"💎 *TODAY'S NET PnL:* *{p_sign}₹{net_pnl:+,.2f}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"*Executed Trades Breakdown:*\n"
    )

    for i, (_, r) in enumerate(today_trades.iterrows(), 1):
        tr_net = float(r.get("net_pnl", 0.0))
        tr_sign = "+" if tr_net > 0 else ""
        tr_emoji = "🟢" if tr_net > 0 else "🔴"
        s_name = str(r.get("strategy", ""))
        strike_str = str(r.get("strike", ""))
        entry_t = str(r.get("entry_time", ""))
        exit_t = str(r.get("exit_time", ""))
        reason = str(r.get("exit_reason", ""))

        msg += (
            f"{tr_emoji} *Trade {i}:* `{s_name}`\n"
            f"   • Strikes: `{strike_str}`\n"
            f"   • Time: {entry_t} → {exit_t}\n"
            f"   • Net PnL: *{tr_sign}₹{tr_net:,.2f}* ({reason})\n"
        )

    return msg, inline_kb

def format_portfolio_summary_card() -> tuple[str, dict]:
    """Generates complete cumulative portfolio balance report."""
    inline_kb = get_inline_status_markup()
    if not os.path.exists(JOURNAL_CSV_FILE):
        return "ℹ️ *No trade journal found yet.*", inline_kb

    try:
        df = pd.read_csv(JOURNAL_CSV_FILE)
    except Exception as e:
        return f"[WARN] Failed to read trade journal: {e}", inline_kb

    tot_trades = len(df)
    tot_pnl = float(df["net_pnl"].sum()) if not df.empty and "net_pnl" in df.columns else 0.0
    wins = len(df[df["net_pnl"] > 0]) if not df.empty and "net_pnl" in df.columns else 0
    losses = len(df[df["net_pnl"] <= 0]) if not df.empty and "net_pnl" in df.columns else 0
    wr = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    current_bal = round(10000.0 + tot_pnl, 2)
    ret_pct = ((current_bal - 10000.0) / 10000.0) * 100.0
    pnl_str = f"+₹{tot_pnl:,.2f}" if tot_pnl >= 0 else f"-₹{abs(tot_pnl):,.2f}"
    ret_str = f"+{ret_pct:.1f}%" if ret_pct >= 0 else f"{ret_pct:.1f}%"

    msg = (
        f"💼 *PORTFOLIO ACCOUNT BALANCE & PERFORMANCE*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🏦 *Starting Capital:* INR 10,000.00\n"
        f"💎 *Cumulative Net PnL:* *{pnl_str}*\n"
        f"💰 *CURRENT ACCOUNT BALANCE:* *₹{current_bal:,.2f}* ({ret_str} Return)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Historical Metrics:*\n"
        f"   • Total Trades Taken: {tot_trades}\n"
        f"   • Win Rate: *{wr:.1f}%* ({wins} Wins / {losses} Losses)\n"
        f"   • Max Trade Capital Limit: *₹10,000 strictly enforced*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *Live Bot Engine:* Active & Synchronized with GitHub Journal"
    )
    return msg, inline_kb

def format_journal_recent_card() -> tuple[str, dict]:
    """Generates the latest 5 closed trades from the journal."""
    inline_kb = get_inline_status_markup()
    if not os.path.exists(JOURNAL_CSV_FILE):
        return "ℹ️ *No trade journal found yet.*", inline_kb

    try:
        df = pd.read_csv(JOURNAL_CSV_FILE)
    except Exception as e:
        return f"[WARN] Failed to read trade journal: {e}", inline_kb

    if df.empty:
        return "ℹ️ *No closed trades logged yet.*", inline_kb

    recent = df.tail(5).iloc[::-1]
    msg = (
        f"📖 *LATEST 5 CLOSED TRADES (TRADE JOURNAL)*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    for _, r in recent.iterrows():
        net = float(r.get("net_pnl", 0.0))
        sign = "+" if net > 0 else ""
        emoji = "🟢" if net > 0 else "🔴"
        msg += (
            f"{emoji} *{r.get('date')}* | `{r.get('strategy')}`\n"
            f"   • Strikes: `{r.get('strike')}`\n"
            f"   • Times: {r.get('entry_time')} → {r.get('exit_time')}\n"
            f"   • Net PnL: *{sign}₹{net:,.2f}* | Reason: `{r.get('exit_reason')}`\n\n"
        )
    return msg, inline_kb

def reply_telegram_message(chat_id: str, text: str, reply_markup: dict = None, reply_to_message_id: int = None) -> bool:
    """Sends a direct response to the user's command or button tap."""
    token = get_telegram_token()
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup if reply_markup is not None else get_main_keyboard_markup()
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=12)
            if r.status_code == 200:
                return True
            elif "can't parse entities" in r.text:
                payload["parse_mode"] = ""
                r2 = requests.post(url, json=payload, headers=headers, timeout=12)
                return r2.status_code == 200
        except Exception:
            time.sleep(1.0)
    return False

def edit_telegram_message(chat_id: str, message_id: int, text: str, reply_markup: dict = None) -> bool:
    """Edits an existing Telegram message in place for smooth UI updates."""
    token = get_telegram_token()
    if not token or not chat_id or not message_id:
        return False

    url = f"https://api.telegram.org/bot{token}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup
    }
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code == 200:
            return True
        elif "message is not modified" in r.text:
            return True
        elif "can't parse entities" in r.text:
            payload["parse_mode"] = ""
            r2 = requests.post(url, json=payload, headers=headers, timeout=10)
            return r2.status_code == 200
    except Exception:
        pass
    return False

def answer_callback_query(callback_query_id: str, text: str = "") -> bool:
    """Dismisses the Telegram loading state for an inline button tap."""
    token = get_telegram_token()
    if not token or not callback_query_id:
        return False
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        requests.post(url, json={"callback_query_id": callback_query_id, "text": text}, timeout=8)
        return True
    except Exception:
        return False

def handle_incoming_update(update: dict, api=None):
    """Processes a single incoming Telegram update (Message or Inline Button Callback)."""
    token = get_telegram_token()
    user_chat_id = get_or_detect_chat_id()

    # 1. Handle Inline Button Callbacks
    if "callback_query" in update:
        cb = update["callback_query"]
        cb_id = cb.get("id")
        cb_data = cb.get("data", "")
        msg = cb.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id") or user_chat_id)
        msg_id = msg.get("message_id")

        answer_callback_query(cb_id, text="Updating status...")

        text, markup = "", None
        if cb_data in ["status", "refresh"]:
            text, markup = format_live_status_card(api=api)
        elif cb_data == "today":
            text, markup = format_today_trades_card()
        elif cb_data in ["summary", "balance"]:
            text, markup = format_portfolio_summary_card()
        elif cb_data == "journal":
            text, markup = format_journal_recent_card()

        if text:
            edited = False
            if msg_id:
                edited = edit_telegram_message(chat_id, msg_id, text, reply_markup=markup)
            if not edited:
                reply_telegram_message(chat_id, text, reply_markup=markup)
        return

    # 2. Handle Text Messages and Reply Keyboard button presses
    if "message" in update:
        m = update["message"]
        text_raw = str(m.get("text", "")).strip()
        chat_id = str(m.get("chat", {}).get("id") or user_chat_id)
        msg_id = m.get("message_id")

        if not text_raw:
            return

        cmd = text_raw.lower()

        if any(cmd.startswith(x) for x in ["/status", "/pnl", "status", "pnl", "📊 live status & pnl", "📊"]):
            resp, markup = format_live_status_card(api=api)
            reply_telegram_message(chat_id, resp, reply_markup=markup, reply_to_message_id=msg_id)

        elif any(cmd.startswith(x) for x in ["/today", "today", "📜 today's trades", "📜"]):
            resp, markup = format_today_trades_card()
            reply_telegram_message(chat_id, resp, reply_markup=markup, reply_to_message_id=msg_id)

        elif any(cmd.startswith(x) for x in ["/summary", "/balance", "summary", "balance", "💼 portfolio balance", "💼"]):
            resp, markup = format_portfolio_summary_card()
            reply_telegram_message(chat_id, resp, reply_markup=markup, reply_to_message_id=msg_id)

        elif any(cmd.startswith(x) for x in ["/journal", "journal", "📖 trade journal", "📖"]):
            resp, markup = format_journal_recent_card()
            reply_telegram_message(chat_id, resp, reply_markup=markup, reply_to_message_id=msg_id)

        elif any(cmd.startswith(x) for x in ["/start", "/help", "/menu", "help", "menu"]):
            welcome_msg = (
                f"🤖 *Nifty Multi-Index Trading Bot Control Panel*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Welcome! Use the permanent buttons below or type commands anytime:\n\n"
                f"• *📊 Live Status & PnL* (`/status`, `/pnl`)\n"
                f"  Check if a trade is currently running, entry prices, targets, and live PnL.\n\n"
                f"• *📜 Today's Trades* (`/today`)\n"
                f"  View all trades executed today and daily net return.\n\n"
                f"• *💼 Portfolio Balance* (`/summary`, `/balance`)\n"
                f"  View starting capital, cumulative PnL, and current balance.\n\n"
                f"• *📖 Trade Journal* (`/journal`)\n"
                f"  View recent closed trades from the permanent trade ledger.\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔔 *All entry and exit alerts are pushed instantly with 0 duplicates!*"
            )
            reply_telegram_message(chat_id, welcome_msg, reply_markup=get_main_keyboard_markup(), reply_to_message_id=msg_id)

def run_telegram_polling_loop(api=None, stop_event: threading.Event = None):
    """Background polling loop that continuously listens for incoming user button taps and commands."""
    token = get_telegram_token()
    if not token:
        print("[TELEGRAM-BOT] Token not found. Interactive listener disabled.")
        return

    print("[TELEGRAM-BOT] Interactive Telegram Listener started (Listening for buttons & commands)...")
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    offset = None

    while stop_event is None or not stop_event.is_set():
        params = {"timeout": 10}
        if offset is not None:
            params["offset"] = offset

        try:
            r = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.json()
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        offset = update["update_id"] + 1
                        try:
                            handle_incoming_update(update, api=api)
                        except Exception as u_err:
                            print(f"[WARN] Error handling Telegram update: {u_err}")
            elif r.status_code == 409:
                # Conflict (another instance polling) - back off politely
                time.sleep(5)
        except Exception:
            time.sleep(2)

def start_interactive_bot_daemon(api=None) -> tuple[threading.Thread, threading.Event]:
    """Starts the interactive bot listener in a background daemon thread."""
    stop_event = threading.Event()
    thread = threading.Thread(target=run_telegram_polling_loop, kwargs={"api": api, "stop_event": stop_event}, daemon=True)
    thread.start()
    return thread, stop_event

if __name__ == "__main__":
    print("Starting standalone interactive Telegram bot service...")
    run_telegram_polling_loop()
