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

import re

SENT_ALERTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "sent_alerts.json")

def normalize_date_str(d) -> str:
    """Normalizes any date object or date string to standard YYYY-MM-DD format."""
    if not d:
        return datetime.now().strftime("%Y-%m-%d")
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    d_str = str(d).strip()
    for fmt in ["%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(d_str, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return d_str[:10]

def normalize_strategy_code(strategy: str) -> str:
    """Normalizes any strategy name variant to a short canonical code (S1..S5)."""
    s = str(strategy).lower()
    if "strategy 1" in s or "s1" in s or "15m" in s:
        return "S1"
    if "strategy 2" in s or "s2" in s or "gamma" in s:
        return "S2"
    if "strategy 3" in s or "s3" in s or "directional" in s:
        return "S3"
    if "strategy 5" in s or "s5" in s or "banknifty" in s:
        return "S5"
    if "strategy 4" in s or "s4" in s or "das" in s:
        return "S4"
    return re.sub(r'[^A-Z0-9]', '', str(strategy).upper())

def normalize_contract_key(ce_str: str = "", pe_str: str = "", strike: str = "", opt_type: str = "") -> str:
    """Creates a clean alphanumeric identifier for strikes/contracts."""
    tokens = []
    for val in [ce_str, pe_str, strike, opt_type]:
        if val:
            cleaned = re.sub(r'[^A-Za-z0-9]', '', str(val)).upper()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
    return "+".join(tokens) if tokens else "CONTRACT"

def get_main_keyboard_markup() -> dict:
    """Returns the persistent interactive Reply Keyboard with 1-tap buttons."""
    return {
        "keyboard": [
            [{"text": "📊 Live Status & PnL"}, {"text": "📜 Today's Trades"}],
            [{"text": "💼 Portfolio Balance"}, {"text": "📖 Trade Journal"}]
        ],
        "resize_keyboard": True,
        "is_persistent": True
    }

def get_inline_status_markup() -> dict:
    """Returns inline keyboard markup with status and refresh buttons."""
    return {
        "inline_keyboard": [
            [
                {"text": "📊 Live Status & PnL", "callback_data": "status"},
                {"text": "📜 Today's Trades", "callback_data": "today"}
            ],
            [
                {"text": "💼 Portfolio Balance", "callback_data": "summary"},
                {"text": "🔄 Refresh", "callback_data": "refresh"}
            ]
        ]
    }

def is_alert_already_sent(alert_key: str) -> bool:
    """Checks if an alert key has already been sent to prevent duplicate notifications."""
    if not alert_key or not os.path.exists(SENT_ALERTS_FILE):
        return False
    try:
        with open(SENT_ALERTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return alert_key in data
    except Exception:
        return False

def mark_alert_as_sent(alert_key: str):
    """Records an alert key in sent_alerts.json."""
    if not alert_key:
        return
    os.makedirs(os.path.dirname(SENT_ALERTS_FILE) or "data", exist_ok=True)
    data = {}
    if os.path.exists(SENT_ALERTS_FILE):
        try:
            with open(SENT_ALERTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[alert_key] = datetime.now().isoformat()
    # Prune old keys to keep maximum 2000 records
    if len(data) > 2000:
        keys_to_remove = list(data.keys())[:-2000]
        for k in keys_to_remove:
            del data[k]
    try:
        with open(SENT_ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write sent_alerts.json: {e}")

def send_telegram_alert(
    message: str,
    reply_markup: dict = None,
    alert_key: str = None,
    force: bool = False
) -> bool:
    """Sends a markdown/plain text alert to your Telegram with duplicate protection and interactive buttons."""
    if alert_key and not force:
        if is_alert_already_sent(alert_key):
            print(f"[NOTIFIER] Duplicate alert blocked ({alert_key}). Not sending twice.")
            return True

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
        "parse_mode": "Markdown",
        "reply_markup": reply_markup if reply_markup is not None else get_main_keyboard_markup()
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

    delivered = False
    for attempt in range(4):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                delivered = True
                break
            elif "can't parse entities" in r.text or "Bad Request" in r.text:
                payload["parse_mode"] = ""
                r2 = requests.post(url, json=payload, headers=headers, timeout=15)
                if r2.status_code == 200:
                    delivered = True
                    break
        except Exception:
            pass

        # Urllib fallback attempt
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers_req)
            with urllib.request.urlopen(req, timeout=12) as response:
                if response.status == 200:
                    delivered = True
                    break
        except Exception:
            pass

        time_lib.sleep(1.5 * (attempt + 1))

    if delivered:
        if alert_key:
            mark_alert_as_sent(alert_key)
        return True

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
    time_str: str = "",
    qty: int = 65,
    date_str: str = "",
    alert_key: str = None,
    force: bool = False
) -> bool:
    """Formats and sends an instant Trade Entry alert with deduplication protection."""
    if not time_str:
        time_str = datetime.now().strftime("%I:%M:%S %p")
    norm_date = normalize_date_str(date_str)
    s_code = normalize_strategy_code(strategy)

    # Detect if single-leg trade (e.g. Strategy 2 Gamma Squeeze or Strategy 3 Directional ITM)
    if opt_type in ["CE", "PE"] or (entry_p > 0 and not (ce_str and pe_str)):
        leg_type = opt_type if opt_type in ["CE", "PE"] else ("CE" if "CE" in str(strike) else "PE")
        p_val = entry_p if entry_p > 0 else (ce_p if leg_type == "CE" else pe_p)
        sym_val = f"{strike} {leg_type}" if strike and leg_type not in str(strike) else (strike or f"OPTION {leg_type}")
        cost_val = tot_cost if tot_cost > 0 else (p_val * qty)
        tgt_val = p_val * (1.0 + win_target_pct)
        sl_val = p_val * (1.0 - lose_stop_pct)
        emoji = "🟢" if leg_type == "CE" else "🔴"

        msg = (
            f"🚨 *TRADE ENTRY TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Time:* {time_str} | Spot: `{spot:.2f}`\n"
            f"📊 *Strategy:* `{strategy}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} *{leg_type} Option Leg:* BUY 1 Lot ({qty} Qty)\n"
            f"   • `{sym_val}` @ *₹{p_val:.2f}*\n"
            f"   • Profit Target (+{int(win_target_pct*100)}%): *₹{tgt_val:.2f}*\n"
            f"   • Stop Loss (-{int(lose_stop_pct*100)}%): *₹{sl_val:.2f}*\n"
            f"   • Time Stop: {max_hold_mins} Mins\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Capital Used:* *₹{cost_val:,.2f}* (<= ₹10,000 budget)\n"
        )
        contract_k = normalize_contract_key(strike=sym_val, opt_type=leg_type)
        if not alert_key:
            alert_key = f"ENTRY:{norm_date}:{s_code}:{contract_k}:{time_str[:5]}"
    else:
        # Dual-leg Strangle (Strategy 1, Strategy 4, or Strategy 5 BankNifty)
        ce_tgt = ce_p * (1.0 + win_target_pct)
        ce_sl = ce_p * (1.0 - lose_stop_pct)
        pe_tgt = pe_p * (1.0 + win_target_pct)
        pe_sl = pe_p * (1.0 - lose_stop_pct)
        cost_val = tot_cost if tot_cost > 0 else ((ce_p + pe_p) * qty)

        msg = (
            f"🚨 *TRADE ENTRY TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Time:* {time_str} | Spot: `{spot:.2f}`\n"
            f"📊 *Strategy:* `{strategy}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 *CALL Leg:* BUY 1 Lot ({qty} Qty)\n"
            f"   • `{ce_str}` @ *₹{ce_p:.2f}*\n"
            f"   • Target (+{int(win_target_pct*100)}%): *₹{ce_tgt:.2f}*\n"
            f"   • Stop (-{int(lose_stop_pct*100)}%): *₹{ce_sl:.2f}*\n\n"
            f"🔴 *PUT Leg:* BUY 1 Lot ({qty} Qty)\n"
            f"   • `{pe_str}` @ *₹{pe_p:.2f}*\n"
            f"   • Target (+{int(win_target_pct*100)}%): *₹{pe_tgt:.2f}*\n"
            f"   • Stop (-{int(lose_stop_pct*100)}%): *₹{pe_sl:.2f}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Capital Used:* *₹{cost_val:,.2f}* (<= ₹10,000 budget)\n"
            f"⏱️ *Max Hold:* {max_hold_mins} Minutes\n"
            f"🛡️ *Exit Mode:* Decoupled Asymmetric Profit & SL\n"
        )
        contract_k = normalize_contract_key(ce_str=ce_str, pe_str=pe_str)
        if not alert_key:
            alert_key = f"ENTRY:{norm_date}:{s_code}:{contract_k}:{time_str[:5]}"

    return send_telegram_alert(msg, alert_key=alert_key, force=force)

def send_trade_exit_alert(
    strategy: str,
    exit_reason: str,
    gross_pnl: float,
    charges: float,
    net_pnl: float,
    details: str = "",
    time_str: str = "",
    date_str: str = "",
    entry_time: str = "",
    strike: str = "",
    alert_key: str = None,
    force: bool = False
) -> bool:
    """Formats and sends an instant Trade Exit alert with canonical duplicate prevention."""
    if not time_str:
        time_str = datetime.now().strftime("%I:%M:%S %p")
    norm_date = normalize_date_str(date_str)
    s_code = normalize_strategy_code(strategy)

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

    if not alert_key:
        contract_k = normalize_contract_key(strike=strike) if strike else ""
        if not contract_k and details:
            # Extract strike tokens from details
            found_contracts = re.findall(r'[A-Za-z0-9]+[CP]E', details)
            if found_contracts:
                contract_k = "+".join(found_contracts)
            else:
                contract_k = re.sub(r'[^A-Za-z0-9]', '', details[:30]).upper()

        e_ref = str(entry_time)[:5] if entry_time else ""
        if not e_ref and details:
            match_e = re.search(r'Entry:.*?(\d{2}:\d{2})', details)
            if match_e:
                e_ref = match_e.group(1)

        t_part = e_ref if e_ref else str(time_str)[:5]
        alert_key = f"EXIT:{norm_date}:{s_code}:{contract_k}:{t_part}"

    return send_telegram_alert(msg, alert_key=alert_key, force=force)

def send_daily_summary_alert(
    date_str: str,
    trades_count: int,
    daily_pnl: float,
    total_pnl: float,
    current_balance: float,
    alert_key: str = None,
    force: bool = False
) -> bool:
    """Formats and sends an EOD portfolio report with deduplication protection."""
    norm_date = normalize_date_str(date_str)
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
    if not alert_key:
        alert_key = f"SUMMARY:{norm_date}"

    return send_telegram_alert(msg, alert_key=alert_key, force=force)

