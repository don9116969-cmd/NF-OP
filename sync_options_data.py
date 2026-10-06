"""
Automated Real Options Data Downloader & Synchronizer for Angel One SmartAPI.
Fetches, caches, and syncs 1-minute OHLCV candles for NIFTY and BankNIFTY options
into data/real_options_cache/ (.parquet & .csv) for 100% real-data backtesting.
"""

import os
import sys
import time
import json
import glob
import logging
import argparse
import datetime
import pandas as pd
from typing import List, Dict, Optional
from SmartApi import SmartConnect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from download_real_data import get_authenticated_api
from src.data.real_option_feed import (
    load_cached_contract_candles,
    save_contract_candles_to_cache,
    OPTIONS_CACHE_DIR,
    BN_LEGACY_CACHE_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OptionSyncer")

SCRIP_MASTER_PATH = os.path.join(BASE_DIR, "data", "angel_scrip_master.json")
NIFTY_SPOT_PATH = os.path.join(BASE_DIR, "data", "nifty_1min_real.csv")
BANKNIFTY_SPOT_PATH = os.path.join(BASE_DIR, "data", "banknifty_1min_real.csv")

def load_scrip_master() -> List[dict]:
    """Loads Angel One scrip master JSON."""
    if not os.path.exists(SCRIP_MASTER_PATH):
        raise FileNotFoundError(f"Scrip master file not found at: {SCRIP_MASTER_PATH}")
    with open(SCRIP_MASTER_PATH, "r") as f:
        return json.load(f)

def get_latest_spot(file_path: str, default: float) -> float:
    """Reads the latest spot price from CSV."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if not df.empty and "close" in df.columns:
                return float(df.iloc[-1]["close"])
        except Exception:
            pass
    return default

def get_target_contracts(
    master: List[dict],
    index: str,
    atm_strike: int,
    step: int,
    strikes_count: int = 15,
    expiries: Optional[List[str]] = None
) -> List[dict]:
    """
    Finds active option contracts within [atm_strike - strikes_count * step, atm_strike + strikes_count * step].
    """
    min_strike = atm_strike - strikes_count * step
    max_strike = atm_strike + strikes_count * step

    contracts = []
    for d in master:
        if d.get("name") != index:
            continue
        if d.get("instrumenttype") != "OPTIDX":
            continue

        exp = d.get("expiry")
        if expiries and exp not in expiries:
            continue

        raw_strike = d.get("strike", 0)
        try:
            strike_val = int(float(raw_strike)) // 100
        except (ValueError, TypeError):
            continue

        if min_strike <= strike_val <= max_strike:
            contracts.append({
                "token": d["token"],
                "symbol": d["symbol"],
                "expiry": exp,
                "strike": strike_val,
                "opt_type": d.get("symbol", "")[-2:],  # CE or PE
                "exch_seg": d.get("exch_seg", "NFO")
            })

    # Sort deterministically
    contracts.sort(key=lambda x: (x["expiry"], x["strike"], x["opt_type"]))
    return contracts

def fetch_and_save_contract_data(
    api: SmartConnect,
    contract: dict,
    from_str: str,
    to_str: str,
    force: bool = False
) -> int:
    """
    Downloads historical candles for a contract from Angel One SmartAPI and saves to cache.
    Returns number of candles retrieved.
    """
    sym = contract["symbol"]
    tok = contract["token"]

    if not force:
        cached = load_cached_contract_candles(sym)
        if not cached.empty:
            req_from = pd.to_datetime(from_str)
            req_to = pd.to_datetime(to_str)
            if cached.index.min() <= req_from and cached.index.max() >= req_to:
                logger.info(f"[CACHE HIT] {sym}: {len(cached)} candles already cached.")
                return len(cached)

    logger.info(f"[DOWNLOADING] {sym} (Token {tok}) from {from_str} to {to_str}...")
    candles_retrieved = 0
    for attempt in range(4):
        try:
            res = api.getCandleData({
                "exchange": contract.get("exch_seg", "NFO"),
                "symboltoken": tok,
                "interval": "ONE_MINUTE",
                "fromdate": from_str,
                "todate": to_str
            })
            if res and res.get("status") and res.get("data"):
                df = pd.DataFrame(res["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                save_contract_candles_to_cache(sym, df)
                candles_retrieved = len(df)
                logger.info(f"  [SAVED] {sym}: {candles_retrieved} candles stored to cache.")
                break
            elif res and "exceeding access rate" in str(res):
                sleep_s = 1.5 * (attempt + 1)
                logger.warning(f"  Rate limit hit for {sym}. Waiting {sleep_s}s...")
                time.sleep(sleep_s)
            else:
                msg = res.get("message") if res else "empty response"
                logger.warning(f"  No candle data for {sym}: {msg}")
                break
        except Exception as e:
            if "exceeding access rate" in str(e):
                time.sleep(2.0)
            else:
                logger.error(f"  Error fetching {sym}: {e}")
                break

    time.sleep(0.35)  # Enforce SmartAPI rate limit (< 3 req/sec)
    return candles_retrieved

def print_cache_status_report():
    """Generates a detailed summary of all option contracts currently cached on disk."""
    p_files = glob.glob(os.path.join(OPTIONS_CACHE_DIR, "*.parquet"))
    c_files = glob.glob(os.path.join(OPTIONS_CACHE_DIR, "*.csv"))

    print("\n" + "=" * 90)
    print("                    REAL OPTIONS DISK CACHE STATUS REPORT")
    print("=" * 90)
    print(f"Cache Directory: {OPTIONS_CACHE_DIR}")
    print(f"Total Parquet Files: {len(p_files)} | Total CSV Files: {len(c_files)}")

    nifty_contracts = []
    banknifty_contracts = []

    for f in p_files:
        basename = os.path.basename(f).replace(".parquet", "")
        size_kb = os.path.getsize(f) / 1024.0
        try:
            df = pd.read_parquet(f)
            count = len(df)
            min_t = str(df.index.min())[:16] if not df.empty else "N/A"
            max_t = str(df.index.max())[:16] if not df.empty else "N/A"
        except Exception:
            count = 0
            min_t, max_t = "ERR", "ERR"

        item = {
            "symbol": basename,
            "candles": count,
            "from": min_t,
            "to": max_t,
            "size_kb": round(size_kb, 1)
        }
        if basename.startswith("NIFTY"):
            nifty_contracts.append(item)
        elif basename.startswith("BANKNIFTY"):
            banknifty_contracts.append(item)

    print(f"\n--- NIFTY Contracts ({len(nifty_contracts)} cached) ---")
    if nifty_contracts:
        df_n = pd.DataFrame(nifty_contracts)
        print(df_n.head(10).to_string(index=False))
        if len(df_n) > 10:
            print(f"... and {len(df_n) - 10} more contracts")
        total_n_candles = sum(x["candles"] for x in nifty_contracts)
        print(f"Total NIFTY 1-min candles: {total_n_candles:,}")

    print(f"\n--- BANKNIFTY Contracts ({len(banknifty_contracts)} cached) ---")
    if banknifty_contracts:
        df_b = pd.DataFrame(banknifty_contracts)
        print(df_b.head(10).to_string(index=False))
        if len(df_b) > 10:
            print(f"... and {len(df_b) - 10} more contracts")
        total_b_candles = sum(x["candles"] for x in banknifty_contracts)
        print(f"Total BANKNIFTY 1-min candles: {total_b_candles:,}")

    print("=" * 90 + "\n")

def sync_options(
    sync_nifty: bool = True,
    sync_banknifty: bool = True,
    strikes_count: int = 15,
    force: bool = False
):
    """Main synchronizer loop to fetch all option data from Angel One SmartAPI."""
    logger.info("Connecting to Angel One SmartAPI...")
    api = get_authenticated_api()
    master = load_scrip_master()

    now_ist = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)
    now_ist = now_ist.replace(tzinfo=None)
    # Market closes at 15:30 IST
    effective_to = now_ist.replace(hour=15, minute=30) if now_ist.hour >= 16 or (now_ist.hour == 15 and now_ist.minute >= 30) else now_ist
    to_str = effective_to.strftime("%Y-%m-%d %H:%M")

    total_synced = 0

    if sync_nifty:
        spot_n = get_latest_spot(NIFTY_SPOT_PATH, 22700.0)
        atm_n = int(round(spot_n / 50.0) * 50)
        logger.info(f"\n[SYNC NIFTY] Current Spot: {spot_n:.2f} | ATM: {atm_n}")
        # Active expiries for NIFTY
        target_exps = ["06OCT2026", "13OCT2026"]
        n_contracts = get_target_contracts(master, "NIFTY", atm_n, step=50, strikes_count=strikes_count, expiries=target_exps)
        logger.info(f"Found {len(n_contracts)} NIFTY contracts across {target_exps}")

        from_str_n = "2026-10-01 09:15"
        for i, c in enumerate(n_contracts, 1):
            logger.info(f"[{i}/{len(n_contracts)}] NIFTY: {c['symbol']}")
            candles = fetch_and_save_contract_data(api, c, from_str_n, to_str, force=force)
            if candles > 0:
                total_synced += 1

    if sync_banknifty:
        spot_b = get_latest_spot(BANKNIFTY_SPOT_PATH, 55000.0)
        atm_b = int(round(spot_b / 100.0) * 100)
        logger.info(f"\n[SYNC BANKNIFTY] Current Spot: {spot_b:.2f} | ATM: {atm_b}")
        # Active monthly expiry for BANKNIFTY
        target_exps_b = ["27OCT2026"]
        b_contracts = get_target_contracts(master, "BANKNIFTY", atm_b, step=100, strikes_count=strikes_count, expiries=target_exps_b)
        logger.info(f"Found {len(b_contracts)} BANKNIFTY contracts across {target_exps_b}")

        from_str_b = "2026-09-28 09:15"
        for i, c in enumerate(b_contracts, 1):
            logger.info(f"[{i}/{len(b_contracts)}] BANKNIFTY: {c['symbol']}")
            candles = fetch_and_save_contract_data(api, c, from_str_b, to_str, force=force)
            if candles > 0:
                total_synced += 1

    logger.info(f"\nSUCCESS! Synchronized {total_synced} option contracts to disk cache.")
    print_cache_status_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Angel One Real Options Data Downloader & Synchronizer")
    parser.add_argument("--report", action="store_true", help="Print cache status summary without downloading")
    parser.add_argument("--nifty-only", action="store_true", help="Download only NIFTY options")
    parser.add_argument("--banknifty-only", action="store_true", help="Download only BankNIFTY options")
    parser.add_argument("--strikes", type=int, default=15, help="Number of strikes above & below ATM to download (default: 15)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if already present in disk cache")

    args = parser.parse_args()

    if args.report:
        print_cache_status_report()
    else:
        sync_nifty = not args.banknifty_only
        sync_bn = not args.nifty_only
        sync_options(
            sync_nifty=sync_nifty,
            sync_banknifty=sync_bn,
            strikes_count=args.strikes,
            force=args.force
        )
