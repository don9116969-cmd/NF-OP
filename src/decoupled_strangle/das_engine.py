"""
Decoupled Asymmetric Strangle (DAS) Execution Engine.
Simulates and evaluates DAS trades with decoupled leg management:
- Surging winning leg captured at +50% target
- Decaying losing leg stopped at -35% or held for reversal
- Combined stop loss at -15%
- Time cutoff at 25 minutes
"""

import os
import datetime
import numpy as np
import pandas as pd
from config import config
from src.features.greeks import black_scholes_price
from .das_signals import (
    select_affordable_strikes,
    check_compression_expansion,
    is_trade_window_valid
)
from src.data.real_option_feed import get_active_option_contract, fetch_real_option_candles

def load_cached_real_options(cache_dirs: list = None) -> dict:
    """Loads all pre-downloaded 1-minute option candle CSVs from cache directories."""
    if cache_dirs is None:
        cache_dirs = ["data/real_options_cache", "testing/data/real_options_cache"]

    opt_data = {}
    for c_dir in cache_dirs:
        if os.path.exists(c_dir):
            for f in os.listdir(c_dir):
                if f.endswith('.csv'):
                    sym = f.replace('.csv', '')
                    if sym not in opt_data:
                        try:
                            d = pd.read_csv(os.path.join(c_dir, f))
                            d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
                            d.set_index('timestamp', inplace=True)
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                if col in d.columns:
                                    d[col] = pd.to_numeric(d[col], errors='coerce')
                            opt_data[sym] = d
                        except Exception:
                            pass
    return opt_data

def evaluate_das_for_day(
    df_full: pd.DataFrame,
    target_date: datetime.date,
    api = None,
    opt_cache_dir: str = None
) -> dict:
    """
    Evaluates Strategy 4 (Decoupled Asymmetric Strangle) for a specific trading session.
    Utilizes real option tick candles if available from Angel One / local cache,
    or mathematical Black-Scholes model if outside historical option cache.
    """
    data = df_full.copy()
    if 'date' not in data.columns:
        data['date'] = pd.to_datetime(data['timestamp']).dt.date

    day_bars = data[data['date'] == target_date].copy().sort_values("timestamp")
    if day_bars.empty or len(day_bars) < 20:
        return {"status": "NO_DATA", "trade_occurred": False, "reason": "No market data (holiday/weekend)"}

    # Load real option data from cache
    cache_dirs = [opt_cache_dir] if opt_cache_dir else ["data/real_options_cache", "testing/data/real_options_cache"]
    opt_data = load_cached_real_options(cache_dirs)

    # Strategy Parameters from config
    vol_window = config.DAS_VOL_WINDOW
    std_thresh = config.DAS_STD_THRESH
    vel_thresh = config.DAS_VEL_THRESH
    win_target_pct = config.DAS_WIN_TARGET_PCT
    lose_stop_pct = config.DAS_LOSE_STOP_PCT
    comb_stop_pct = config.DAS_COMBINED_STOP_PCT
    max_hold_bars = config.DAS_MAX_HOLD_MINS
    cooldown_bars = config.DAS_COOLDOWN_BARS
    lot_size = config.LOT_SIZE

    timestamps = day_bars['timestamp'].tolist()
    n = len(timestamps)

    trades = []
    in_pos = False
    pos = None
    last_exit_bar = -999

    # Determine next weekly expiry for mathematical fallback
    days_ahead = (1 - target_date.weekday()) % 7  # Tuesday expiry (weekday 1)
    if days_ahead == 0:
        expiry_date = target_date
    else:
        expiry_date = target_date + datetime.timedelta(days=days_ahead)
    expiry_dt = datetime.datetime.combine(expiry_date, datetime.time(15, 30))

    # Detect if we have real option data for this day
    has_real_options = False
    if opt_data:
        # Check if any strike has data on target_date
        for sym, odf in opt_data.items():
            if not odf.empty and (odf.index.date == target_date).any():
                has_real_options = True
                break

    feed_label = "Angel One Real Traded" if has_real_options else "Mathematical (BSM)"

    for i in range(n):
        t = timestamps[i]
        t_time = t.time()
        time_str = t.strftime("%H:%M")
        bar_spot = day_bars.iloc[i]
        spot = float(bar_spot['close'])
        spot_h = float(bar_spot['high'])
        spot_l = float(bar_spot['low'])

        # Time-to-expiry in years
        tte_sec = max(60, (expiry_dt - t).total_seconds())
        T_years = tte_sec / (365.0 * 24.0 * 3600.0)
        iv = 0.135

        # -------------------------------------------------------------
        # 1. MANAGE ACTIVE DECOUPLED POSITION
        # -------------------------------------------------------------
        if in_pos:
            bars_held = i - pos['entry_bar']

            # Option leg valuations
            if has_real_options and pos['ce_sym'] in opt_data and pos['pe_sym'] in opt_data:
                ce_df = opt_data[pos['ce_sym']]
                pe_df = opt_data[pos['pe_sym']]
                ce_bar = ce_df.loc[t] if t in ce_df.index else None
                pe_bar = pe_df.loc[t] if t in pe_df.index else None

                ce_cur = float(ce_bar['close']) if ce_bar is not None else pos['ce_entry']
                ce_high = float(ce_bar['high']) if ce_bar is not None else ce_cur
                ce_low = float(ce_bar['low']) if ce_bar is not None else ce_cur

                pe_cur = float(pe_bar['close']) if pe_bar is not None else pos['pe_entry']
                pe_high = float(pe_bar['high']) if pe_bar is not None else pe_cur
                pe_low = float(pe_bar['low']) if pe_bar is not None else pe_cur
            else:
                ce_cur = black_scholes_price(spot, pos['ce_strike'], T_years, config.RISK_FREE_RATE, iv, "CE")
                ce_high = black_scholes_price(spot_h, pos['ce_strike'], T_years, config.RISK_FREE_RATE, iv, "CE")
                ce_low = black_scholes_price(spot_l, pos['ce_strike'], T_years, config.RISK_FREE_RATE, iv, "CE")

                pe_cur = black_scholes_price(spot, pos['pe_strike'], T_years, config.RISK_FREE_RATE, iv, "PE")
                pe_high = black_scholes_price(spot_l, pos['pe_strike'], T_years, config.RISK_FREE_RATE, iv, "PE")
                pe_low = black_scholes_price(spot_h, pos['pe_strike'], T_years, config.RISK_FREE_RATE, iv, "PE")

            # Check CE Leg Exit
            if not pos['ce_exited']:
                if ce_high >= pos['ce_entry'] * (1.0 + win_target_pct):
                    pos['ce_exit_p'] = round(pos['ce_entry'] * (1.0 + win_target_pct), 2)
                    pos['ce_exit_t'] = t
                    pos['ce_exited'] = True
                    pos['ce_reason'] = f"CE Target (+{int(win_target_pct*100)}%)"
                elif ce_low <= pos['ce_entry'] * (1.0 - lose_stop_pct):
                    pos['ce_exit_p'] = round(pos['ce_entry'] * (1.0 - lose_stop_pct), 2)
                    pos['ce_exit_t'] = t
                    pos['ce_exited'] = True
                    pos['ce_reason'] = f"CE SL (-{int(lose_stop_pct*100)}%)"

            # Check PE Leg Exit
            if not pos['pe_exited']:
                if pe_high >= pos['pe_entry'] * (1.0 + win_target_pct):
                    pos['pe_exit_p'] = round(pos['pe_entry'] * (1.0 + win_target_pct), 2)
                    pos['pe_exit_t'] = t
                    pos['pe_exited'] = True
                    pos['pe_reason'] = f"PE Target (+{int(win_target_pct*100)}%)"
                elif pe_low <= pos['pe_entry'] * (1.0 - lose_stop_pct):
                    pos['pe_exit_p'] = round(pos['pe_entry'] * (1.0 - lose_stop_pct), 2)
                    pos['pe_exit_t'] = t
                    pos['pe_exited'] = True
                    pos['pe_reason'] = f"PE SL (-{int(lose_stop_pct*100)}%)"

            # Combined Position Check
            val_ce = pos['ce_exit_p'] if pos['ce_exited'] else ce_cur
            val_pe = pos['pe_exit_p'] if pos['pe_exited'] else pe_cur
            tot_entry = pos['ce_entry'] + pos['pe_entry']
            tot_cur = val_ce + val_pe
            comb_pnl_pct = (tot_cur - tot_entry) / tot_entry

            force_close = False
            close_reason = None

            if pos['ce_exited'] and pos['pe_exited']:
                force_close = True
                close_reason = "Both Legs Exited"
            elif comb_pnl_pct <= -comb_stop_pct and (not pos['ce_exited'] and not pos['pe_exited']):
                force_close = True
                close_reason = f"Combined SL (-{int(comb_stop_pct*100)}%)"
            elif (t - pos['entry_time']).total_seconds() / 60.0 >= max_hold_bars:
                force_close = True
                close_reason = f"Max Hold ({max_hold_bars}m)"
            elif time_str >= "15:15":
                force_close = True
                close_reason = "EOD Squareoff"

            if force_close:
                if not pos['ce_exited']:
                    pos['ce_exit_p'] = round(ce_cur, 2)
                    pos['ce_exit_t'] = t
                    pos['ce_exited'] = True
                    pos['ce_reason'] = close_reason
                if not pos['pe_exited']:
                    pos['pe_exit_p'] = round(pe_cur, 2)
                    pos['pe_exit_t'] = t
                    pos['pe_exited'] = True
                    pos['pe_reason'] = close_reason

                ce_gross = (pos['ce_exit_p'] - pos['ce_entry']) * lot_size
                pe_gross = (pos['pe_exit_p'] - pos['pe_entry']) * lot_size
                total_gross = ce_gross + pe_gross

                # Statutory charges calculation
                buy_val = (pos['ce_entry'] + pos['pe_entry']) * lot_size
                sell_val = (pos['ce_exit_p'] + pos['pe_exit_p']) * lot_size
                turn = buy_val + sell_val
                brok = 80.0  # 4 legs execution @ Rs 20
                stt = sell_val * 0.001
                exch = turn * 0.0005
                gst = (brok + exch) * 0.18
                charges = round(brok + stt + exch + gst, 2)
                total_net = round(total_gross - charges, 2)

                ce_str = pos['ce_sym'] if 'ce_sym' in pos and pos['ce_sym'] else f"{pos['ce_strike']}CE"
                pe_str = pos['pe_sym'] if 'pe_sym' in pos and pos['pe_sym'] else f"{pos['pe_strike']}PE"

                entry_t_str = pos['entry_time'].strftime("%H:%M:%S") if hasattr(pos['entry_time'], 'strftime') else str(pos['entry_time'])
                exit_t_str = t.strftime("%H:%M:%S") if hasattr(t, 'strftime') else str(t)

                trades.append({
                    'entry_time': entry_t_str,
                    'exit_time': exit_t_str,
                    'ce_strike': ce_str,
                    'pe_strike': pe_str,
                    'strike': f"{pos.get('ce_strike', '')}CE / {pos.get('pe_strike', '')}PE",
                    'ce_entry': pos['ce_entry'],
                    'ce_exit': pos['ce_exit_p'],
                    'ce_reason': pos['ce_reason'],
                    'pe_entry': pos['pe_entry'],
                    'pe_exit': pos['pe_exit_p'],
                    'pe_reason': pos['pe_reason'],
                    'entry_p': round(pos['ce_entry'] + pos['pe_entry'], 2),
                    'exit_p': round(pos['ce_exit_p'] + pos['pe_exit_p'], 2),
                    'cost': round(buy_val, 2),
                    'gross_pnl': round(total_gross, 2),
                    'charges': charges,
                    'net_pnl': total_net,
                    'exit_reason': f"{pos['ce_reason']} | {pos['pe_reason']}",
                    'close_reason': close_reason,
                    'data_feed': feed_label
                })
                in_pos = False
                pos = None
                last_exit_bar = i
                continue

        # -------------------------------------------------------------
        # 2. CHECK ENTRY TRIGGERS
        # -------------------------------------------------------------
        if in_pos:
            continue
        if not is_trade_window_valid(t_time):
            continue
        if (i - last_exit_bar) < cooldown_bars:
            continue

        atm = int(round(spot / 50.0) * 50)

        # Dynamic strike scanner
        chosen_ce = None
        chosen_pe = None
        ce_entry_p = None
        pe_entry_p = None
        ce_strike = None
        pe_strike = None

        if has_real_options:
            # Check candidate strikes dynamically across all cached options
            for offset in [50, 100, 150, 200, 0]:
                k = atm + offset
                for sym, df_o in opt_data.items():
                    if sym.endswith(f"{k}CE") and t in df_o.index:
                        p = float(df_o.loc[t, 'close'])
                        if config.DAS_MIN_LEG_PREMIUM <= p <= config.DAS_MAX_LEG_PREMIUM:
                            chosen_ce = sym
                            ce_entry_p = p
                            ce_strike = k
                            break
                if chosen_ce:
                    break

            for offset in [50, 100, 150, 200, 0]:
                k = atm - offset
                for sym, df_o in opt_data.items():
                    if sym.endswith(f"{k}PE") and t in df_o.index:
                        p = float(df_o.loc[t, 'close'])
                        if config.DAS_MIN_LEG_PREMIUM <= p <= config.DAS_MAX_LEG_PREMIUM:
                            chosen_pe = sym
                            pe_entry_p = p
                            pe_strike = k
                            break
                if chosen_pe:
                    break
        else:
            # Mathematical BSM Mode
            for offset in [50, 100, 150, 200, 0]:
                k_ce = atm + offset
                k_pe = atm - offset
                c_p = black_scholes_price(spot, k_ce, T_years, config.RISK_FREE_RATE, iv, "CE")
                p_p = black_scholes_price(spot, k_pe, T_years, config.RISK_FREE_RATE, iv, "PE")
                if (c_p + p_p) * lot_size <= config.INITIAL_CAPITAL and (config.DAS_MIN_LEG_PREMIUM <= c_p <= config.DAS_MAX_LEG_PREMIUM) and (config.DAS_MIN_LEG_PREMIUM <= p_p <= config.DAS_MAX_LEG_PREMIUM):
                    ce_strike = k_ce
                    pe_strike = k_pe
                    ce_entry_p = c_p
                    pe_entry_p = p_p
                    break

        if not ce_entry_p or not pe_entry_p:
            continue

        tot_cap = (ce_entry_p + pe_entry_p) * lot_size
        if tot_cap > config.INITIAL_CAPITAL:
            continue

        # Check Rolling Premium Volatility Compression & Kinetic Expansion
        if has_real_options and chosen_ce and chosen_pe:
            ce_df = opt_data[chosen_ce]
            pe_df = opt_data[chosen_pe]
            try:
                idx_c = ce_df.index.get_loc(t)
                if idx_c < vol_window:
                    continue
                comb_hist = (ce_df['close'].iloc[idx_c - vol_window:idx_c + 1] + pe_df['close'].iloc[idx_c - vol_window:idx_c + 1]).values
            except Exception:
                continue
        else:
            if i < vol_window:
                continue
            recent_spots = day_bars['close'].iloc[i - vol_window:i + 1].values
            comb_hist = []
            for s_val in recent_spots:
                cp = black_scholes_price(s_val, ce_strike, T_years, config.RISK_FREE_RATE, iv, "CE")
                pp = black_scholes_price(s_val, pe_strike, T_years, config.RISK_FREE_RATE, iv, "PE")
                comb_hist.append(cp + pp)
            comb_hist = np.array(comb_hist)

        triggered, cur_std, cur_vel = check_compression_expansion(comb_hist, std_thresh, vel_thresh)

        if triggered:
            in_pos = True
            pos = {
                'entry_time': t,
                'entry_bar': i,
                'ce_sym': chosen_ce,
                'pe_sym': chosen_pe,
                'ce_strike': ce_strike,
                'pe_strike': pe_strike,
                'ce_entry': round(ce_entry_p, 2),
                'pe_entry': round(pe_entry_p, 2),
                'ce_exited': False,
                'ce_exit_p': None,
                'ce_reason': None,
                'pe_exited': False,
                'pe_exit_p': None,
                'pe_reason': None
            }

    # Format return result
    if len(trades) > 0:
        primary_trade = trades[0]
        # Aggregate if multiple trades occurred on that day
        total_gross = sum(tr['gross_pnl'] for tr in trades)
        total_charges = sum(tr['charges'] for tr in trades)
        total_net = sum(tr['net_pnl'] for tr in trades)

        return {
            "status": "TRADE_EXECUTED",
            "trade_occurred": True,
            "entry_time": primary_trade["entry_time"].strftime("%H:%M:%S") if hasattr(primary_trade["entry_time"], 'strftime') else str(primary_trade["entry_time"]),
            "exit_time": primary_trade["exit_time"].strftime("%H:%M:%S") if hasattr(primary_trade["exit_time"], 'strftime') else str(primary_trade["exit_time"]),
            "opt_type": "CE+PE",
            "strike": f"{primary_trade['ce_strike']} / {primary_trade['pe_strike']}",
            "entry_p": primary_trade["entry_p"],
            "exit_p": primary_trade["exit_p"],
            "cost": primary_trade["cost"],
            "gross_pnl": round(total_gross, 2),
            "charges": round(total_charges, 2),
            "net_pnl": round(total_net, 2),
            "exit_reason": primary_trade["exit_reason"],
            "data_feed": primary_trade["data_feed"],
            "num_trades_day": len(trades),
            "all_trades": trades
        }

    return {
        "status": "NO_TRIGGER",
        "trade_occurred": False,
        "reason": f"No premium volatility compression (<{std_thresh}) with kinetic expansion (>={vel_thresh}) detected during market hours."
    }

def run_das_simulation(df_spot: pd.DataFrame, opt_data: dict = None) -> pd.DataFrame:
    """Runs DAS backtest across multiple days."""
    dates = df_spot['timestamp'].dt.date.unique() if 'timestamp' in df_spot.columns else df_spot.index.date.unique()
    all_trades = []
    for d in sorted(dates):
        res = evaluate_das_for_day(df_spot, d, opt_cache_dir=None)
        if res.get("trade_occurred", False) and "all_trades" in res:
            all_trades.extend(res["all_trades"])
    return pd.DataFrame(all_trades)
