"""
Downloads real BankNIFTY historical candles (1-minute and 5-minute) from Angel One SmartAPI.
"""
import os
import datetime
import logging
import pandas as pd
from download_real_data import get_authenticated_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def download_banknifty_spot(
    days_back: int = 90,
    interval: str = "FIVE_MINUTE",
    output_csv: str = "data/banknifty_5min_real.csv"
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    api = get_authenticated_api()
    now = datetime.datetime.now()
    
    chunks = []
    chunk_days = 20 if interval == "FIVE_MINUTE" else 5
    
    for i in range(0, days_back, chunk_days):
        chunk_to = now - datetime.timedelta(days=i)
        chunk_from = now - datetime.timedelta(days=min(i + chunk_days, days_back))
        
        from_str = chunk_from.strftime("%Y-%m-%d 09:15")
        to_str = chunk_to.strftime("%Y-%m-%d 15:30")
        
        logger.info(f"Fetching real {interval} BankNIFTY candles: {from_str} -> {to_str}...")
        try:
            res = api.getCandleData({
                "exchange": "NSE",
                "symboltoken": "99926009",  # BankNifty Index Spot
                "interval": interval,
                "fromdate": from_str,
                "todate": to_str
            })
            if res and res.get("status") and res.get("data"):
                df_chunk = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                chunks.append(df_chunk)
                logger.info(f"  Retrieved {len(df_chunk)} candles.")
            else:
                logger.warning(f"  No data or message: {res.get('message', 'empty')}")
        except Exception as e:
            logger.error(f"  Error: {e}")

    if not chunks:
        raise RuntimeError("No historical candle data retrieved for BankNIFTY.")

    full_df = pd.concat(chunks, ignore_index=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df.drop_duplicates(subset=["timestamp"], inplace=True)
    full_df.sort_values(by="timestamp", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    full_df.to_csv(output_csv, index=False)
    parquet_path = output_csv.replace(".csv", ".parquet")
    full_df.to_parquet(parquet_path, index=False)
    
    logger.info(f"SUCCESS: Saved {len(full_df)} BankNIFTY candles to {output_csv} and {parquet_path}")
    return full_df

if __name__ == "__main__":
    print("Downloading 5-Minute BankNIFTY Candles (90 days)...")
    download_banknifty_spot(days_back=90, interval="FIVE_MINUTE", output_csv="data/banknifty_5min_real.csv")
    print("\nDownloading 1-Minute BankNIFTY Candles (35 days)...")
    download_banknifty_spot(days_back=35, interval="ONE_MINUTE", output_csv="data/banknifty_1min_real.csv")
