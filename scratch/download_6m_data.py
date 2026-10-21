"""
Download 6 months of 1-minute historical candles for NIFTY 50 and BankNIFTY from Angel One SmartAPI.
Properly handles timezone normalization (Asia/Kolkata -> naive), SmartConnect timeouts, and rate limits.
"""
import os
import sys
import time
import datetime
import pandas as pd
from dotenv import dotenv_values, load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

try:
    from SmartApi import SmartConnect
    import pyotp
except ImportError:
    print("[ERROR] SmartApi or pyotp not installed.")
    sys.exit(1)

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
env = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}

def get_api():
    api_key = os.getenv("ANGEL_API_KEY") or env.get("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_CODE") or env.get("ANGEL_CLIENT_CODE")
    password = os.getenv("ANGEL_PASSWORD") or os.getenv("ANGEL_PIN") or os.getenv("ANGEI_PASSWORD") or env.get("ANGEL_PASSWORD")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET") or env.get("ANGEL_TOTP_SECRET")

    for attempt in range(5):
        try:
            api = SmartConnect(api_key=api_key, timeout=25)
            totp = pyotp.TOTP(totp_secret).now()
            sess = api.generateSession(client_code, password, totp)
            if sess and sess.get("status"):
                print(f"[AUTH] Successfully connected to Angel One as {client_code}")
                return api
            else:
                print(f"[AUTH] Attempt {attempt+1} failed: {sess.get('message')}. Retrying in 2.5s...")
                time.sleep(2.5)
        except Exception as e:
            print(f"[AUTH] Connection error: {e}. Retrying in 3s...")
            time.sleep(3.0)
    raise ConnectionError("Failed to authenticate with Angel One SmartAPI.")

def normalize_timestamps(series):
    """Converts a series of timestamp strings (with or without tz) to timezone-naive datetime64."""
    ts = pd.to_datetime(series, errors='coerce', utc=True)
    return ts.dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)

def fetch_symbol_6m(api, symbol_token: str, symbol_label: str, output_csv: str, days_back: int = 180):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df_existing = pd.DataFrame()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            if not df_existing.empty and "timestamp" in df_existing.columns:
                df_existing["timestamp"] = normalize_timestamps(df_existing["timestamp"])
                df_existing.dropna(subset=["timestamp"], inplace=True)
                print(f"[LOCAL] Loaded {len(df_existing)} existing candles for {symbol_label} ({df_existing['timestamp'].min()} to {df_existing['timestamp'].max()})")
        except Exception as e:
            print(f"[WARN] Error reading existing {output_csv}: {e}")
            df_existing = pd.DataFrame()

    now = datetime.datetime.now()
    chunk_days = 12 # 12 days per chunk (~4500 bars)
    all_chunks = []
    if not df_existing.empty:
        all_chunks.append(df_existing)

    print(f"\n[DOWNLOAD] Fetching up to {days_back} days of 1-minute data for {symbol_label} ({symbol_token})...")
    
    for i in range(0, days_back, chunk_days):
        chunk_to = now - datetime.timedelta(days=i)
        chunk_from = now - datetime.timedelta(days=min(i + chunk_days, days_back))
        
        from_str = chunk_from.strftime("%Y-%m-%d 09:15")
        to_str = chunk_to.strftime("%Y-%m-%d 15:30")
        
        # Check if already covered in existing
        if not df_existing.empty:
            earliest_have = df_existing["timestamp"].min()
            latest_have = df_existing["timestamp"].max()
            if pd.to_datetime(from_str) >= earliest_have and pd.to_datetime(to_str) <= latest_have:
                continue

        print(f"  -> Fetching: {from_str} to {to_str}...")
        
        chunk_data = None
        for req_attempt in range(5):
            try:
                time.sleep(1.8) # Respect Angel One rate limit
                res = api.getCandleData({
                    "exchange": "NSE",
                    "symboltoken": symbol_token,
                    "interval": "ONE_MINUTE",
                    "fromdate": from_str,
                    "todate": to_str
                })
                if res and res.get("status") and res.get("data"):
                    chunk_data = res["data"]
                    break
                elif res and "exceeding access rate" in str(res):
                    wait_s = 4.0 + req_attempt * 2.0
                    print(f"     [RATE-LIMIT] Waiting {wait_s:.0f}s...")
                    time.sleep(wait_s)
                else:
                    msg = str(res.get("message", "")) if res else ""
                    if "No data" in msg or "AB1004" in msg:
                        print(f"     [INFO] No further historical data: {msg}")
                        break
                    time.sleep(2.0)
            except Exception as e:
                err_str = str(e)
                if "exceeding access rate" in err_str:
                    time.sleep(4.0)
                else:
                    print(f"     [WARN] Request attempt {req_attempt+1} failed: {e}")
                    time.sleep(2.5)

        if chunk_data:
            df_c = pd.DataFrame(chunk_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_c["timestamp"] = normalize_timestamps(df_c["timestamp"])
            df_c.dropna(subset=["timestamp"], inplace=True)
            all_chunks.append(df_c)
            print(f"     [OK] Retrieved {len(df_c)} candles ({df_c['timestamp'].min()} to {df_c['timestamp'].max()})")

    if all_chunks:
        df_merged = pd.concat(all_chunks, ignore_index=True)
        df_merged.drop_duplicates(subset=["timestamp"], inplace=True)
        df_merged.sort_values("timestamp", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)
        
        df_merged.to_csv(output_csv, index=False)
        print(f"[SUCCESS] {symbol_label}: {len(df_merged)} total 1-minute bars from {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()} saved to {output_csv}\n")
        return df_merged
    return df_existing

if __name__ == "__main__":
    api = get_api()
    fetch_symbol_6m(api, "99926000", "NIFTY 50", "data/nifty_1min_real.csv", days_back=180)
    fetch_symbol_6m(api, "99926009", "BankNIFTY", "data/banknifty_1min_real.csv", days_back=180)
