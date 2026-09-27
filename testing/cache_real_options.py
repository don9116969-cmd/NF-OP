"""
Cache 100% Real 1-Minute Option Candles directly from Angel One SmartAPI
Covers the entire active weekly expiry (29SEP2026) across all 4 days:
- 2026-09-22
- 2026-09-23
- 2026-09-24
- 2026-09-25
"""

import os
import sys
import datetime
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from daily_result_checker import get_smart_api
from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles

def main():
    api = get_smart_api()
    if not api:
        print("[ERROR] SmartAPI connection failed.")
        return

    out_dir = "testing/data/real_options_cache"
    os.makedirs(out_dir, exist_ok=True)

    dt_start = datetime.datetime(2026, 9, 22, 9, 15)
    dt_end = datetime.datetime(2026, 9, 25, 15, 30)

    # All strikes traded around current Nifty level (23,000 to 23,600)
    strikes = [23000, 23050, 23100, 23150, 23200, 23250, 23300, 23350, 23400, 23450, 23500, 23550, 23600]
    total_cached = 0

    print("[INFO] Downloading real 1-minute option contracts from Angel One SmartAPI...")
    for s in strikes:
        for opt in ['CE', 'PE']:
            c = get_active_option_contract(datetime.date(2026, 9, 22), s, opt)
            if c:
                df = fetch_real_option_candles(api, c, dt_start, dt_end)
                if len(df) > 0:
                    sym = c['symbol']
                    out_path = os.path.join(out_dir, f"{sym}.csv")
                    df.to_csv(out_path)
                    total_cached += 1
                    print(f"[{total_cached:02d}] Cached {sym}: {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    print(f"\n[DONE] Successfully downloaded and cached {total_cached} real option contracts to {out_dir}!")

if __name__ == "__main__":
    main()
