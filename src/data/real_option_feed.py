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

_EXP_MAP_CACHE = {}
_LOOKUP_CACHE = {}

def _init_lookup_cache(underlying: str):
    global _EXP_MAP_CACHE, _LOOKUP_CACHE
    if underlying in _LOOKUP_CACHE:
        return
    opts = ensure_scrip_master(name=underlying)
    exp_map = {}
    lookup = {}
    for d in opts:
        exp_str = d.get("expiry")
        if exp_str and exp_str not in exp_map:
            try:
                exp_dt = datetime.datetime.strptime(exp_str, "%d%b%Y").date()
                exp_map[exp_str] = exp_dt
            except Exception:
                pass
        
        sym = d.get("symbol", "")
        opt_type = "CE" if sym.endswith("CE") else ("PE" if sym.endswith("PE") else "")
        if opt_type and exp_str:
            d_strike = float(d.get("strike", 0))
            strike_pt = int(round(d_strike / 100.0))
            exp_dt = exp_map.get(exp_str)
            if exp_dt:
                contract_info = {
                    "token": d["token"],
                    "symbol": d["symbol"],
                    "expiry": exp_str,
                    "expiry_date": exp_dt,
                    "strike": strike_pt,
                    "opt_type": opt_type,
                    "exch_seg": d.get("exch_seg", "NFO"),
                    "lotsize": int(d.get("lotsize", 65 if underlying == "NIFTY" else 30))
                }
                lookup[(exp_str, opt_type, strike_pt)] = contract_info

    # Also index all pre-downloaded contracts from cache directories
    import re
    cache_dirs = [
        os.path.join(BASE_DIR, "data", "real_options_cache"),
        os.path.join(BASE_DIR, "data", "banknifty_options_cache"),
        os.path.join(BASE_DIR, "testing", "data", "real_options_cache")
    ]
    for c_dir in cache_dirs:
        if not os.path.exists(c_dir):
            continue
        for f in os.listdir(c_dir):
            if not f.endswith(".csv"):
                continue
            base = f[:-4]
            m = re.match(r"^([A-Z]+)([0-9]{2}[A-Z]{3}[0-9]{2})([0-9]+)(CE|PE)$", base)
            if not m:
                continue
            u, exp_str, strike_str, o_type = m.groups()
            if u != underlying:
                continue
            try:
                exp_dt = datetime.datetime.strptime(exp_str, "%d%b%y").date()
                norm_exp_str = exp_dt.strftime("%d%b%Y").upper()
            except Exception:
                continue
            if norm_exp_str not in exp_map:
                exp_map[norm_exp_str] = exp_dt
            s_pt = int(strike_str)
            key = (norm_exp_str, o_type, s_pt)
            if key not in lookup:
                lookup[key] = {
                    "token": base,
                    "symbol": base,
                    "expiry": norm_exp_str,
                    "expiry_date": exp_dt,
                    "strike": s_pt,
                    "opt_type": o_type,
                    "exch_seg": "NFO",
                    "lotsize": 65 if underlying == "NIFTY" else 30
                }

    _EXP_MAP_CACHE[underlying] = sorted(list(exp_map.items()), key=lambda x: x[1])
    _LOOKUP_CACHE[underlying] = lookup

def get_active_option_contract(target_date: datetime.date, strike: int, opt_type: str, underlying: str = "NIFTY") -> dict:
    """
    Finds the active option contract matching the strike, opt_type, and nearest expiry for NIFTY or BANKNIFTY in O(1) time.
    """
    underlying = underlying.upper()
    _init_lookup_cache(underlying)

    exp_list = _EXP_MAP_CACHE.get(underlying, [])
    if not exp_list:
        return None

    opt_type_upper = opt_type.upper()

    # Find nearest expiry on or after target_date
    valid = [e for e in exp_list if e[1] >= target_date]
    if valid:
        nearest_exp_str, nearest_exp_dt = valid[0]
    else:
        # Fallback to latest available expiry before target_date for historical playback
        past = [e for e in exp_list if e[1] <= target_date]
        if past:
            nearest_exp_str, nearest_exp_dt = past[-1]
        else:
            return None

    lookup = _LOOKUP_CACHE.get(underlying, {})
    return lookup.get((nearest_exp_str, opt_type_upper, strike))

OPTIONS_CACHE_DIR = os.path.join(BASE_DIR, "data", "real_options_cache")
BN_LEGACY_CACHE_DIR = os.path.join(BASE_DIR, "data", "banknifty_options_cache")
TEST_CACHE_DIR = os.path.join(BASE_DIR, "testing", "data", "real_options_cache")
os.makedirs(OPTIONS_CACHE_DIR, exist_ok=True)

def load_cached_contract_candles(symbol: str) -> pd.DataFrame:
    """Loads cached 1-minute OHLCV candles for a contract symbol from disk."""
    if not symbol:
        return pd.DataFrame()

    for c_dir in [OPTIONS_CACHE_DIR, BN_LEGACY_CACHE_DIR, TEST_CACHE_DIR]:
        if not os.path.exists(c_dir):
            continue
        p_file = os.path.join(c_dir, f"{symbol}.parquet")
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

        c_file = os.path.join(c_dir, f"{symbol}.csv")
        if os.path.exists(c_file):
            try:
                df = pd.read_csv(c_file)
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
        try:
            df_combined.to_parquet(p_file)
        except Exception:
            pass  # Fall back gracefully if pyarrow/fastparquet engine is unavailable
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

    if hasattr(from_date, "tzinfo") and from_date.tzinfo is not None:
        from_date = from_date.replace(tzinfo=None)
    if hasattr(to_date, "tzinfo") and to_date.tzinfo is not None:
        to_date = to_date.replace(tzinfo=None)

    # 1. Check local disk cache first
    if not force_api and sym:
        df_cached = load_cached_contract_candles(sym)
        if not df_cached.empty:
            sub = df_cached[(df_cached.index >= from_date) & (df_cached.index <= to_date)]
            if len(sub) >= 5:
                return sub
            # If the session date matches any data in df_cached, return what we have
            day_sub = df_cached[df_cached.index.date == from_date.date()]
            if not day_sub.empty:
                return day_sub

    # If token is non-numeric (e.g. symbol name from offline indexing), cannot query SmartAPI
    if not str(contract.get("token", "")).isdigit():
        df_cached = load_cached_contract_candles(sym) if sym else pd.DataFrame()
        if not df_cached.empty:
            sub = df_cached[(df_cached.index >= from_date) & (df_cached.index <= to_date)]
            if not sub.empty:
                return sub
        return pd.DataFrame()

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
