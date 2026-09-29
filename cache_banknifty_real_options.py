"""
Caches real 1-minute OHLCV candles for BankNIFTY options (Sep 2026 expiry) from Angel One SmartAPI.
"""
import os
import json
import time
import datetime
import logging
import pandas as pd
from download_real_data import get_authenticated_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = "data/banknifty_options_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

with open("data/angel_scrip_master.json", "r") as f:
    master = json.load(f)

bn_opts = [d for d in master if d.get("name") == "BANKNIFTY" and d.get("expiry") == "29SEP2026"]
token_map = {d["symbol"]: d["token"] for d in bn_opts}

target_strikes = [54000, 54200, 54500, 54700, 54800, 55000, 55200, 55300, 55500, 55700, 56000]
target_symbols = []
for k in target_strikes:
    target_symbols.append(f"BANKNIFTY29SEP26{k}CE")
    target_symbols.append(f"BANKNIFTY29SEP26{k}PE")

print(f"Total target option symbols to cache: {len(target_symbols)}")

api = get_authenticated_api()

# We fetch September 21 to September 28 (the active week for 29SEP2026 expiry)
from_str = "2026-09-21 09:15"
to_str = "2026-09-28 15:30"

cached_count = 0
for sym in target_symbols:
    out_file = os.path.join(CACHE_DIR, f"{sym}.csv")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        logger.info(f"Already cached: {sym}")
        cached_count += 1
        continue

    tok = token_map.get(sym)
    if not tok:
        logger.warning(f"Token not found for {sym}")
        continue

    logger.info(f"Fetching real option candles for {sym} (Token {tok})...")
    try:
        res = api.getCandleData({
            "exchange": "NFO",
            "symboltoken": tok,
            "interval": "ONE_MINUTE",
            "fromdate": from_str,
            "todate": to_str
        })
        if res and res.get("status") and res.get("data"):
            df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.to_csv(out_file, index=False)
            logger.info(f"  Saved {len(df)} candles for {sym}")
            cached_count += 1
        else:
            logger.warning(f"  No data for {sym}: {res.get('message') if res else 'empty'}")
        time.sleep(0.35)  # Rate limiting
    except Exception as e:
        logger.error(f"  Error fetching {sym}: {e}")
        time.sleep(1.0)

print(f"\nCompleted! Cached {cached_count} / {len(target_symbols)} BankNIFTY option contracts.")
