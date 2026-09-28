"""
Telegram Notification Engine for Nifty Option Trading Bot.
Sends formatted real-time trade entry, exit, and daily portfolio reports to your phone.
"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_telegram_token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

def get_or_detect_chat_id() -> str:
    """
    Returns TELEGRAM_CHAT_ID from environment.
    If empty, queries Telegram getUpdates to automatically detect the user's Chat ID.
    """
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if chat_id:
        return chat_id

    token = get_telegram_token()
    if not token:
        return ""

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data and data.get("ok") and data.get("result"):
                # Extract chat id from latest update
                for update in reversed(data["result"]):
                    msg = update.get("message") or update.get("channel_post")
                    if msg and "chat" in msg:
                        detected_id = str(msg["chat"]["id"])
                        return detected_id
    except Exception as e:
        print(f"[WARN] Failed to auto-detect Telegram Chat ID: {e}")

    return ""

def send_telegram_alert(message: str) -> bool:
    """Sends a markdown/plain text alert to your Telegram."""
    import requests
    token = get_telegram_token()
    chat_id = get_or_detect_chat_id()

    if not token:
        print("[NOTIFIER] Telegram Token not configured.")
        return False

    if not chat_id:
        print("[NOTIFIER] Telegram Chat ID not found.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    headers = {
        "Connection": "close",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    import time as time_lib
    data_bytes = json.dumps(payload).encode('utf-8')
    headers_req = {
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for attempt in range(4):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                return True
            elif "can't parse entities" in r.text or "Bad Request" in r.text:
                payload["parse_mode"] = ""
                r2 = requests.post(url, json=payload, headers=headers, timeout=15)
                return r2.status_code == 200
        except Exception:
            pass

        # Urllib fallback attempt
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers_req)
            with urllib.request.urlopen(req, timeout=12) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass

        time_lib.sleep(1.5 * (attempt + 1))

    return False

def send_trade_entry_alert(
    strategy: str,
    spot: float = 0.0,
    ce_str: str = "",
    pe_str: str = "",
    ce_p: float = 0.0,
    pe_p: float = 0.0,
    tot_cost: float = 0.0,
    win_target_pct: float = 0.50,
    lose_stop_pct: float = 0.35,
    max_hold_mins: int = 25,
    opt_type: str = "",
    strike: str = "",
    entry_p: float = 0.0,
    time_str: str = ""
) -> bool:
    """Formats and sends an instant Trade Entry alert for both single-leg and dual-leg trades."""
    if not time_str:
        time_str = datetime.now().strftime("%I:%M:%S %p")

    # Detect if single-leg trade (e.g. Strategy 2 Gamma Squeeze or Strategy 3 Directional ITM)
    if opt_type in ["CE", "PE"] or (entry_p > 0 and not (ce_str and pe_str)):
        leg_type = opt_type if opt_type in ["CE", "PE"] else ("CE" if "CE" in str(strike) else "PE")
        p_val = entry_p if entry_p > 0 else (ce_p if leg_type == "CE" else pe_p)
        sym_val = f"{strike} {leg_type}" if strike and leg_type not in str(strike) else (strike or f"NIFTY {leg_type}")
        cost_val = tot_cost if tot_cost > 0 else (p_val * 75.0)
        tgt_val = p_val * (1.0 + win_target_pct)
        sl_val = p_val * (1.0 - lose_stop_pct)
        emoji = "🟢" if leg_type == "CE" else "🔴"

        msg = (
            f"🚨 *TRADE ENTRY TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Time:* {time_str} | Spot: `{spot:.2f}`\n"
            f"📊 *Strategy:* `{strategy}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} *{leg_type} Option Leg:* BUY 1 Lot (75 Qty)\n"
            f"   • `{sym_val}` @ *₹{p_val:.2f}*\n"
            f"   • Profit Target (+{int(win_target_pct*100)}%): *₹{tgt_val:.2f}*\n"
            f"   • Stop Loss (-{int(lose_stop_pct*100)}%): *₹{sl_val:.2f}*\n"
            f"   • Time Stop: {max_hold_mins} Mins\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Capital Used:* *₹{cost_val:,.2f}* (<= ₹10,000 budget)\n"
        )
    else:
        # Dual-leg Strangle (Strategy 1 or Strategy 4)
        ce_tgt = ce_p * (1.0 + win_target_pct)
        ce_sl = ce_p * (1.0 - lose_stop_pct)
        pe_tgt = pe_p * (1.0 + win_target_pct)
        pe_sl = pe_p * (1.0 - lose_stop_pct)
        cost_val = tot_cost if tot_cost > 0 else ((ce_p + pe_p) * 75.0)

        msg = (
            f"🚨 *TRADE ENTRY TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Time:* {time_str} | Spot: `{spot:.2f}`\n"
            f"📊 *Strategy:* `{strategy}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 *CALL Leg:* BUY 1 Lot (75 Qty)\n"
            f"   • `{ce_str}` @ *₹{ce_p:.2f}*\n"
            f"   • Target (+{int(win_target_pct*100)}%): *₹{ce_tgt:.2f}*\n"
            f"   • Stop (-{int(lose_stop_pct*100)}%): *₹{ce_sl:.2f}*\n\n"
            f"🔴 *PUT Leg:* BUY 1 Lot (75 Qty)\n"
            f"   • `{pe_str}` @ *₹{pe_p:.2f}*\n"
            f"   • Target (+{int(win_target_pct*100)}%): *₹{pe_tgt:.2f}*\n"
            f"   • Stop (-{int(lose_stop_pct*100)}%): *₹{pe_sl:.2f}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Capital Used:* *₹{cost_val:,.2f}* (<= ₹10,000 budget)\n"
            f"⏱️ *Max Hold:* {max_hold_mins} Minutes\n"
            f"🛡️ *Exit Mode:* Decoupled Asymmetric Profit & SL\n"
        )
    return send_telegram_alert(msg)

def send_trade_exit_alert(
    strategy: str,
    exit_reason: str,
    gross_pnl: float,
    charges: float,
    net_pnl: float,
    details: str = "",
    time_str: str = ""
) -> bool:
    """Formats and sends an instant Trade Exit alert."""
    if not time_str:
        time_str = datetime.now().strftime("%I:%M:%S %p")
    p_emoji = "🎯" if net_pnl > 0 else "🛑"
    p_sign = "+" if net_pnl > 0 else ""

    msg = (
        f"{p_emoji} *TRADE EXIT NOTIFICATION*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ *Time:* {time_str}\n"
        f"📊 *Strategy:* `{strategy}`\n"
        f"📌 *Exit Reason:* `{exit_reason}`\n"
    )
    if details:
        msg += f"ℹ️ *Trade Summary:* {details}\n"

    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 *Gross PnL:* ₹{gross_pnl:+,.2f}\n"
        f"📉 *Charges & STT:* ₹{charges:.2f}\n"
        f"💎 *NET REALIZED PnL:* *{p_sign}₹{net_pnl:+,.2f}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    return send_telegram_alert(msg)

def send_daily_summary_alert(
    date_str: str,
    trades_count: int,
    daily_pnl: float,
    total_pnl: float,
    current_balance: float
) -> bool:
    """Formats and sends an EOD portfolio report."""
    p_sign = "+" if daily_pnl > 0 else ""
    c_sign = "+" if total_pnl > 0 else ""

    msg = (
        f"📈 *DAILY PORTFOLIO SUMMARY*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 *Date:* {date_str}\n"
        f"🔢 *Trades Taken Today:* {trades_count}\n"
        f"💵 *Today's Net PnL:* *{p_sign}₹{daily_pnl:,.2f}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💎 *Total Cumulative PnL:* *{c_sign}₹{total_pnl:,.2f}*\n"
        f"💼 *Current Account Balance:* *₹{current_balance:,.2f}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *Nifty Option Bot Status:* Sleeping until next market session.\n"
    )
    return send_telegram_alert(msg)
