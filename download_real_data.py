"""
Official Real Market Data Downloader using Angel One SmartAPI.
Fetches 100% real historical candles for Nifty 50 Spot and options.
"""
import os
import json
import logging
import datetime
import pandas as pd
from dotenv import dotenv_values
import pyotp
from SmartApi import SmartConnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

def get_authenticated_api() -> SmartConnect:
    """Authenticates with Angel One SmartAPI using credentials in .env."""
    env = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    api_key = os.getenv("ANGEL_API_KEY") or env.get("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_CODE") or env.get("ANGEL_CLIENT_CODE")
    password = os.getenv("ANGEL_PASSWORD") or os.getenv("ANGEL_PIN") or os.getenv("ANGEI_PASSWORD") or env.get("ANGEL_PASSWORD")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET") or env.get("ANGEL_TOTP_SECRET")

    if not all([api_key, client_code, password, totp_secret]):
        raise ValueError("Missing Angel One credentials in .env file.")

    api = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    session = api.generateSession(client_code, password, totp)
    
    if not session.get("status"):
        raise ConnectionError(f"SmartAPI Login Failed: {session.get('message')}")
        
    logger.info(f"Successfully authenticated as: {session['data'].get('name', client_code)}")
    return api

def download_nifty_spot_historical(
    api: SmartConnect,
    days_back: int = 90,
    interval: str = "FIVE_MINUTE",
    output_csv: str = "data/nifty_5min_real.csv"
) -> pd.DataFrame:
    """
    Downloads real historical candles in 30-day batches to avoid API limit caps.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    now = datetime.datetime.now()
    
    all_chunks = []
    chunk_days = 25  # Safe chunk size for Angel API candle limits
    
    for i in range(0, days_back, chunk_days):
        chunk_to = now - datetime.timedelta(days=i)
        chunk_from = now - datetime.timedelta(days=min(i + chunk_days, days_back))
        
        from_str = chunk_from.strftime("%Y-%m-%d 09:15")
        to_str = chunk_to.strftime("%Y-%m-%d 15:30")
        
        logger.info(f"Fetching real {interval} Nifty candles: {from_str} -> {to_str}...")
        try:
            res = api.getCandleData({
                "exchange": "NSE",
                "symboltoken": "99926000",  # Nifty 50 Index Spot
                "interval": interval,
                "fromdate": from_str,
                "todate": to_str
            })
            if res and res.get("status") and res.get("data"):
                chunk_df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                all_chunks.append(chunk_df)
                logger.info(f"  Retrieved {len(chunk_df)} candles.")
            else:
                logger.warning(f"  No data or error: {res.get('message', 'empty response')}")
        except Exception as e:
            logger.error(f"  Error fetching chunk: {e}")

    if not all_chunks:
        raise RuntimeError("No historical candle data retrieved from Angel One.")

    full_df = pd.concat(all_chunks, ignore_index=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df.drop_duplicates(subset=["timestamp"], inplace=True)
    full_df.sort_values(by="timestamp", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    # Save to CSV and Parquet
    full_df.to_csv(output_csv, index=False)
    parquet_path = output_csv.replace(".csv", ".parquet")
    full_df.to_parquet(parquet_path, index=False)
    
    logger.info(f"SUCCESS: Saved {len(full_df)} real candles to {output_csv} and {parquet_path}")
    return full_df

if __name__ == "__main__":
    api = get_authenticated_api()
    df = download_nifty_spot_historical(api, days_back=90, interval="FIVE_MINUTE")
    print(df.head())
    print(df.tail())
