"""
Fast, robust 6-month historical 1-minute data downloader for NIFTY 50 & BankNIFTY.
Uses format='mixed' to cleanly combine existing CSV records with new Angel One API chunks.
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

    api = SmartConnect(api_key=api_key, timeout=30)
    totp = pyotp.TOTP(totp_secret).now()
    sess = api.generateSession(client_code, password, totp)
    if sess and sess.get("status"):
        print(f"[AUTH] Successfully connected to Angel One SmartAPI ({client_code})")
        return api
    raise ConnectionError(f"SmartAPI login failed: {sess.get('message')}")

def clean_and_save(file_path, new_df=None):
    dfs = []
    if os.path.exists(file_path):
        try:
            d_old = pd.read_csv(file_path)
            if not d_old.empty:
                dfs.append(d_old)
        except Exception:
            pass
            
    if new_df is not None and not new_df.empty:
        dfs.append(new_df)
        
    if not dfs:
        return pd.DataFrame()
        
    comb = pd.concat(dfs, ignore_index=True)
    # Use format='mixed' to seamlessly parse both naive and ISO timestamps
    comb["timestamp"] = pd.to_datetime(comb["timestamp"], format="mixed").dt.tz_localize(None)
    comb.drop_duplicates(subset=["timestamp"], inplace=True)
    comb.sort_values("timestamp", inplace=True)
    comb.reset_index(drop=True, inplace=True)
    comb.to_csv(file_path, index=False)
    return comb

def download_symbol(api, token, label, output_file, target_start="2026-04-21"):
    print(f"\n=======================================================")
    print(f"Syncing 6-Month 1-Min Data for {label} (Token: {token})")
    print(f"Target Start Date: {target_start}")
    print(f"=======================================================")
    
    df = clean_and_save(output_file)
    target_dt = pd.to_datetime(target_start)
    
    if not df.empty:
        earliest_dt = df["timestamp"].min()
        print(f"[STATUS] Currently have {len(df)} candles from {earliest_dt} to {df['timestamp'].max()}")
    else:
        earliest_dt = datetime.datetime.now()
        print(f"[STATUS] No prior data. Starting from current time.")

    cur_end = earliest_dt
    while cur_end > target_dt:
        cur_start = max(target_dt, cur_end - datetime.timedelta(days=12))
        from_str = cur_start.strftime("%Y-%m-%d 09:15")
        to_str = cur_end.strftime("%Y-%m-%d 15:30")
        
        print(f"  -> Fetching: {from_str} to {to_str}...")
        
        chunk_data = None
        for attempt in range(4):
            time.sleep(2.2) # Guarantees clean rate limit headroom
            try:
                res = api.getCandleData({
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": "ONE_MINUTE",
                    "fromdate": from_str,
                    "todate": to_str
                })
                if res and res.get("status") and res.get("data"):
                    chunk_data = res["data"]
                    break
                elif res and "exceeding access rate" in str(res):
                    print(f"     [RATE-LIMIT] Waiting 5s...")
                    time.sleep(5.0)
                else:
                    msg = str(res.get("message", "")) if res else "empty"
                    if "No data" in msg or "AB1004" in msg:
                        print(f"     [INFO] Oldest available data reached: {msg}")
                        break
                    time.sleep(2.5)
            except Exception as e:
                print(f"     [WARN] Attempt {attempt+1} error: {e}")
                time.sleep(3.0)
                
        if chunk_data and len(chunk_data) > 0:
            df_c = pd.DataFrame(chunk_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = clean_and_save(output_file, df_c)
            earliest_dt = df["timestamp"].min()
            print(f"     [OK] Added {len(df_c)} candles. New Earliest: {earliest_dt} (Total: {len(df)} candles)")
            cur_end = earliest_dt
        else:
            print(f"     [SKIP] No candles in window, moving back...")
            cur_end = cur_start
            
        if cur_end <= target_dt:
            break

    print(f"\n[DONE] {label}: {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()}) saved to {output_file}\n")
    return df

if __name__ == "__main__":
    api = get_api()
    download_symbol(api, "99926000", "NIFTY 50", "data/nifty_1min_real.csv", target_start="2026-04-21")
    download_symbol(api, "99926009", "BankNIFTY", "data/banknifty_1min_real.csv", target_start="2026-04-21")
