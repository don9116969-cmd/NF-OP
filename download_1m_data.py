"""
Downloads 30-45 days of 1-minute real Nifty 50 historical candles from Angel One SmartAPI.
"""
import os
import datetime
import logging
import pandas as pd
from download_real_data import get_authenticated_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_1min_candles(days_back: int = 35, output_file: str = "data/nifty_1min_real.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    api = get_authenticated_api()
    now = datetime.datetime.now()
    
    chunks = []
    chunk_days = 7  # 7 days of 1-min data is ~2600 bars, well within Angel API limit of 5000 bars per call
    
    for i in range(0, days_back, chunk_days):
        chunk_to = now - datetime.timedelta(days=i)
        chunk_from = now - datetime.timedelta(days=min(i + chunk_days, days_back))
        
        from_str = chunk_from.strftime("%Y-%m-%d 09:15")
        to_str = chunk_to.strftime("%Y-%m-%d 15:30")
        
        logger.info(f"Fetching 1-minute candles: {from_str} -> {to_str}...")
        try:
            res = api.getCandleData({
                "exchange": "NSE",
                "symboltoken": "99926000",
                "interval": "ONE_MINUTE",
                "fromdate": from_str,
                "todate": to_str
            })
            if res and res.get("status") and res.get("data"):
                df_chunk = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                chunks.append(df_chunk)
                logger.info(f"  Got {len(df_chunk)} 1-minute bars.")
            else:
                logger.warning(f"  No data: {res.get('message', 'empty')}")
        except Exception as e:
            logger.error(f"  Error: {e}")

    if not chunks:
        raise RuntimeError("Failed to fetch 1-min candles.")

    df_full = pd.concat(chunks, ignore_index=True)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_full.drop_duplicates(subset=["timestamp"], inplace=True)
    df_full.sort_values("timestamp", inplace=True)
    df_full.reset_index(drop=True, inplace=True)

    df_full.to_csv(output_file, index=False)
    df_full.to_parquet(output_file.replace(".csv", ".parquet"), index=False)
    logger.info(f"SUCCESS: Saved {len(df_full)} 1-minute real bars to {output_file}")
    return df_full

if __name__ == "__main__":
    download_1min_candles(days_back=35)
