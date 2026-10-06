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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIP_MASTER_PATH = os.path.join(BASE_DIR, "data", "angel_scrip_master.json")

_SCRIP_CACHE = {}

def ensure_scrip_master(max_age_hours: int = 168, name: str = "NIFTY") -> list:
    """Ensures local cache of Angel One Scrip Master is fresh and cached in memory."""
    global _SCRIP_CACHE
    target_name = name.upper()
    if target_name in _SCRIP_CACHE:
        return _SCRIP_CACHE[target_name]

    parent_dir = os.path.dirname(SCRIP_MASTER_PATH)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
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

    for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        _SCRIP_CACHE[idx] = [d for d in master if d.get("name") == idx and d.get("instrumenttype") == "OPTIDX"]

    return _SCRIP_CACHE.get(target_name, [])

def get_active_option_contract(target_date: datetime.date, strike: int, opt_type: str, underlying: str = "NIFTY") -> dict:
    """
    Finds the active option contract matching the strike, opt_type, and nearest expiry for NIFTY or BANKNIFTY.
    """
    opts = ensure_scrip_master(name=underlying)
    opt_type_upper = opt_type.upper()

    # Collect all available expiries
    exp_map = {}
    for d in opts:
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
        # Fallback to the latest available expiry before target_date for historical playback
        past_expiries = [(exp_str, exp_dt) for exp_str, exp_dt in exp_map.items() if exp_dt <= target_date]
        if past_expiries:
            past_expiries.sort(key=lambda x: x[1], reverse=True)
            valid_expiries = [past_expiries[0]]
        else:
            return None

    valid_expiries.sort(key=lambda x: x[1])
    nearest_exp_str, nearest_exp_dt = valid_expiries[0]

    # Target strike in paise
    target_paise = float(strike * 100)

    for d in opts:
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
                        "exch_seg": d.get("exch_seg", "NFO"),
                        "lotsize": int(d.get("lotsize", 65 if underlying == "NIFTY" else 30))
                    }

    return None

OPTIONS_CACHE_DIR = os.path.join(BASE_DIR, "data", "real_options_cache")
BN_LEGACY_CACHE_DIR = os.path.join(BASE_DIR, "data", "banknifty_options_cache")
os.makedirs(OPTIONS_CACHE_DIR, exist_ok=True)

def load_cached_contract_candles(symbol: str) -> pd.DataFrame:
    """Loads cached 1-minute OHLCV candles for a contract symbol from disk."""
    if not symbol:
        return pd.DataFrame()

    p_file = os.path.join(OPTIONS_CACHE_DIR, f"{symbol}.parquet")
    if os.path.exists(p_file):
        try:
            df = pd.read_parquet(p_file)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                    df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            pass

    c_file = os.path.join(OPTIONS_CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(c_file):
        try:
            df = pd.read_csv(c_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            pass

    # Fallback to legacy banknifty cache if present
    bn_file = os.path.join(BN_LEGACY_CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(bn_file):
        try:
            df = pd.read_csv(bn_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            pass

    return pd.DataFrame()

def save_contract_candles_to_cache(symbol: str, df_new: pd.DataFrame):
    """Merges new OHLCV candles with any existing cache and writes to disk."""
    if df_new.empty or not symbol:
        return
    try:
        df_existing = load_cached_contract_candles(symbol)
        if not df_existing.empty:
            df_combined = pd.concat([df_existing, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
            df_combined.sort_index(inplace=True)
        else:
            df_combined = df_new[~df_new.index.duplicated(keep="last")].sort_index()

        p_file = os.path.join(OPTIONS_CACHE_DIR, f"{symbol}.parquet")
        df_combined.to_parquet(p_file)
        c_file = os.path.join(OPTIONS_CACHE_DIR, f"{symbol}.csv")
        df_combined.to_csv(c_file)
        if symbol.startswith("BANKNIFTY"):
            os.makedirs(BN_LEGACY_CACHE_DIR, exist_ok=True)
            df_combined.to_csv(os.path.join(BN_LEGACY_CACHE_DIR, f"{symbol}.csv"))
    except Exception as e:
        print(f"[WARN] Failed to write option cache for {symbol}: {e}")

def fetch_real_option_candles(api: SmartConnect, contract: dict, from_date: datetime.datetime, to_date: datetime.datetime, force_api: bool = False) -> pd.DataFrame:
    """
    Fetches real 1-minute OHLCV candles for the option contract.
    First checks disk cache (data/real_options_cache).
    If cache misses or API is required, fetches from Angel One SmartAPI and updates cache.
    """
    if not contract:
        return pd.DataFrame()

    sym = contract.get("symbol", "")
    if isinstance(from_date, str):
        from_date = pd.to_datetime(from_date)
    if isinstance(to_date, str):
        to_date = pd.to_datetime(to_date)

    # 1. Check local disk cache first
    if not force_api and sym:
        df_cached = load_cached_contract_candles(sym)
        if not df_cached.empty:
            min_ts = df_cached.index.min()
            max_ts = df_cached.index.max()
            if min_ts <= from_date and max_ts >= to_date:
                sub = df_cached[(df_cached.index >= from_date) & (df_cached.index <= to_date)]
                if not sub.empty:
                    return sub

    if not api:
        # Return whatever is in cache if API is offline
        df_cached = load_cached_contract_candles(sym) if sym else pd.DataFrame()
        if not df_cached.empty:
            return df_cached[(df_cached.index >= from_date) & (df_cached.index <= to_date)]
        return pd.DataFrame()

    # 2. Fetch from Angel One SmartAPI
    import time as time_lib
    from_str = from_date.strftime("%Y-%m-%d %H:%M")
    now_ist = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
    now_ist = now_ist.replace(tzinfo=None)
    effective_to = min(to_date, now_ist)
    to_str = effective_to.strftime("%Y-%m-%d %H:%M")

    for attempt in range(4):
        try:
            time_lib.sleep(0.4)
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

                # Persist to disk cache
                save_contract_candles_to_cache(sym, df)
                return df[(df.index >= from_date) & (df.index <= to_date)]
            elif res and "exceeding access rate" in str(res):
                time_lib.sleep(1.5 * (attempt + 1))
            else:
                break
        except Exception as e:
            if "exceeding access rate" in str(e):
                time_lib.sleep(1.5 * (attempt + 1))
            else:
                print(f"[WARN] Failed to fetch real candles for {contract.get('symbol')}: {e}")
                break

    # If API returned empty but we have cached data, return cached
    df_cached = load_cached_contract_candles(sym) if sym else pd.DataFrame()
    if not df_cached.empty:
        return df_cached[(df_cached.index >= from_date) & (df_cached.index <= to_date)]

    return pd.DataFrame()
