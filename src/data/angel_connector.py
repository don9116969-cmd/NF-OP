import os
import json
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

try:
    from SmartApi import SmartConnect
    import pyotp
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

class AngelOneConnector:
    """
    Connects to Angel One SmartAPI to download instrument master tokens
    and fetch historical candle data.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("ANGEL_API_KEY", "")
        self.client_code = client_code or os.getenv("ANGEL_CLIENT_CODE", "")
        self.password = password or os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET", "")
        
        self.smart_api = None
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        self.scrip_master: Optional[pd.DataFrame] = None

    def login(self) -> bool:
        """Logs into Angel One SmartAPI using TOTP."""
        if not SMARTAPI_AVAILABLE:
            logger.error("SmartApi package is not available.")
            return False
            
        if not all([self.api_key, self.client_code, self.password, self.totp_secret]):
            logger.warning("Angel One credentials missing. Operating in offline/mock mode.")
            return False

        try:
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            data = self.smart_api.generateSession(self.client_code, self.password, totp)
            
            if data and data.get("status"):
                self.auth_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = data['data']['feedToken']
                logger.info("Successfully authenticated with Angel One SmartAPI.")
                return True
            else:
                logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Exception during Angel One login: {e}")
            return False

    def load_scrip_master(self, cache_file: str = "data/scrip_master.parquet") -> pd.DataFrame:
        """Downloads or loads cached Angel One Scrip Master to lookup tokens."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.exists(cache_file):
            logger.info(f"Loading cached scrip master from {cache_file}...")
            self.scrip_master = pd.read_parquet(cache_file)
            return self.scrip_master

        logger.info(f"Downloading Angel One Scrip Master from {SCRIP_MASTER_URL}...")
        resp = requests.get(SCRIP_MASTER_URL, timeout=30)
        data = resp.json()
        df = pd.DataFrame(data)
        df.to_parquet(cache_file, index=False)
        self.scrip_master = df
        logger.info(f"Downloaded and cached {len(df)} instrument tokens.")
        return self.scrip_master

    def fetch_historical_candles(
        self,
        token: str,
        exchange: str = "NSE",
        interval: str = "FIVE_MINUTE",
        from_date: str = "2026-03-01 09:15",
        to_date: str = "2026-03-10 15:30"
    ) -> pd.DataFrame:
        """
        Fetches historical candles from SmartAPI.
        Intervals: ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_DAY
        """
        if not self.smart_api:
            logger.warning("SmartAPI not connected. Generating realistic synthetic data instead (60 days).")
            return self.generate_synthetic_nifty_data(days=60, interval_mins=5)

        try:
            param = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            res = self.smart_api.getCandleData(param)
            if res and res.get("status"):
                data = res["data"]
                # Angel format: [timestamp, open, high, low, close, volume]
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                return df
            else:
                logger.error(f"Failed to fetch candles: {res.get('message', '')}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical candles: {e}")
            return pd.DataFrame()

    @staticmethod
    def generate_synthetic_nifty_data(
        start_date: str = "2026-01-01",
        days: int = 30,
        interval_mins: int = 5,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generates realistic 5-minute Nifty 50 intraday data modeled after actual Indian market dynamics:
        - 9:15 AM to 3:30 PM IST (75 five-minute bars per trading day)
        - Realistic intraday volatility, trends, pullbacks, and volume curves (high at open and close).
        """
        np.random.seed(seed)
        all_bars = []
        current_price = 24500.0  # Realistic base Nifty spot level
        start_dt = pd.to_datetime(start_date)

        trading_day = 0
        date_cursor = start_dt

        while trading_day < days:
            # Skip weekends (Saturday=5, Sunday=6)
            if date_cursor.weekday() >= 5:
                date_cursor += datetime.timedelta(days=1)
                continue

            day_str = date_cursor.strftime("%Y-%m-%d")
            # Session times: 09:15 to 15:25 (last 5-min bar closes at 15:30)
            times = pd.date_range(f"{day_str} 09:15", f"{day_str} 15:25", freq=f"{interval_mins}min")
            
            # Intraday regime for this day (Trending Bullish, Trending Bearish, or Choppy/Mean Reverting)
            day_type = np.random.choice(["bullish", "bearish", "choppy"], p=[0.35, 0.30, 0.35])
            
            # Morning gap (opening overnight jump)
            gap_pct = np.random.normal(0.001, 0.003)
            current_price *= (1 + gap_pct)
            
            day_drift = 0.00015 if day_type == "bullish" else (-0.00015 if day_type == "bearish" else 0.0)
            volatility = 0.0012

            for i, bar_time in enumerate(times):
                ret = np.random.normal(day_drift, volatility)
                open_p = current_price
                close_p = open_p * (1 + ret)
                
                # High and Low wicks
                high_p = max(open_p, close_p) + abs(np.random.normal(0, 0.0008)) * open_p
                low_p = min(open_p, close_p) - abs(np.random.normal(0, 0.0008)) * open_p
                
                # U-shaped intraday volume profile (higher at 9:15 and 15:00)
                u_factor = 1.0 + 3.0 * ((i - 37.5) / 37.5) ** 2
                volume = int(np.random.normal(150000, 30000) * u_factor)
                volume = max(10000, volume)

                current_price = close_p
                all_bars.append({
                    "timestamp": bar_time,
                    "open": round(open_p, 2),
                    "high": round(high_p, 2),
                    "low": round(low_p, 2),
                    "close": round(close_p, 2),
                    "volume": volume
                })

            trading_day += 1
            date_cursor += datetime.timedelta(days=1)

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df
