"""
Real Option Feed Engine for Angel One SmartAPI.
Fetches actual traded 1-minute OHLCV candles for active weekly Nifty option contracts.
"""

import os
import json
import datetime
import urllib.request
import pandas as pd
from SmartApi import SmartConnect

SCRIP_MASTER_PATH = r"c:\Desktop\NF-OP\data\angel_scrip_master.json"

def ensure_scrip_master(max_age_hours: int = 168) -> list:
    """Ensures local cache of Angel One Scrip Master is fresh."""
    os.makedirs(os.path.dirname(SCRIP_MASTER_PATH), exist_ok=True)
    needs_download = True

    if os.path.exists(SCRIP_MASTER_PATH):
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(SCRIP_MASTER_PATH))
        if (datetime.datetime.now() - file_time).total_seconds() < (max_age_hours * 3600):
            needs_download = False

    if needs_download:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            with open(SCRIP_MASTER_PATH, "wb") as f:
                f.write(content)
        except Exception as e:
            if not os.path.exists(SCRIP_MASTER_PATH):
                raise RuntimeError(f"Failed to download Angel One scrip master: {e}")

    with open(SCRIP_MASTER_PATH, "r", encoding="utf-8") as f:
        master = json.load(f)

    return [d for d in master if d.get("name") == "NIFTY" and d.get("instrumenttype") == "OPTIDX"]

def get_active_option_contract(target_date: datetime.date, strike: int, opt_type: str) -> dict:
    """
    Finds the active weekly Nifty option contract matching the strike, opt_type, and nearest Tuesday expiry.
    """
    nifty_opts = ensure_scrip_master()
    opt_type_upper = opt_type.upper()

    # Collect all available expiries
    exp_map = {}
    for d in nifty_opts:
        exp_str = d.get("expiry")
        if exp_str and exp_str not in exp_map:
            try:
                exp_dt = datetime.datetime.strptime(exp_str, "%d%b%Y").date()
                exp_map[exp_str] = exp_dt
            except Exception:
                pass

    # Find nearest expiry on or after target_date
    valid_expiries = [(exp_str, exp_dt) for exp_str, exp_dt in exp_map.items() if exp_dt >= target_date]
    if not valid_expiries:
        return None

    valid_expiries.sort(key=lambda x: x[1])
    nearest_exp_str, nearest_exp_dt = valid_expiries[0]

    # Target strike in paise
    target_paise = float(strike * 100)

    for d in nifty_opts:
        if d.get("expiry") == nearest_exp_str:
            sym = d.get("symbol", "")
            if sym.endswith(opt_type_upper):
                d_strike = float(d.get("strike", 0))
                if abs(d_strike - target_paise) < 1.0 or int(round(d_strike / 100.0)) == strike:
                    return {
                        "token": d["token"],
                        "symbol": d["symbol"],
                        "expiry": nearest_exp_str,
                        "expiry_date": nearest_exp_dt,
                        "strike": strike,
                        "opt_type": opt_type_upper,
                        "exch_seg": d.get("exch_seg", "NFO")
                    }

    return None

def fetch_real_option_candles(api: SmartConnect, contract: dict, from_date: datetime.datetime, to_date: datetime.datetime) -> pd.DataFrame:
    """
    Fetches real 1-minute OHLCV candles for the option contract from Angel One SmartAPI.
    """
    if not api or not contract:
        return pd.DataFrame()

    from_str = from_date.strftime("%Y-%m-%d %H:%M")
    to_str = to_date.strftime("%Y-%m-%d %H:%M")

    try:
        res = api.getCandleData({
            "exchange": contract.get("exch_seg", "NFO"),
            "symboltoken": contract["token"],
            "interval": "ONE_MINUTE",
            "fromdate": from_str,
            "todate": to_str
        })

        if res and res.get("status") and res.get("data"):
            df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df
    except Exception as e:
        print(f"[WARN] Failed to fetch real candles for {contract.get('symbol')}: {e}")

    return pd.DataFrame()
