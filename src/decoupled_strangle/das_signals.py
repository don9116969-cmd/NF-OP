"""
Decoupled Asymmetric Strangle (DAS) Signal Generators & Strike Selection.
Selects delta-neutral OTM strikes matching user's INR 10,000 budget and identifies
volatility compression coiling with kinetic velocity expansion.
"""

import numpy as np
import pandas as pd
from datetime import time as dtime
from config import config
from src.features.greeks import black_scholes_price

def is_trade_window_valid(t_time: dtime) -> bool:
    """
    Validates if current time falls within active trading window (09:45 - 14:15)
    and excludes midday low-liquidity chop (11:30 - 13:00).
    """
    t_start = dtime(9, 45)
    t_end = dtime(14, 15)
    chop_start = dtime(11, 30)
    chop_end = dtime(13, 0)

    if t_time < t_start or t_time > t_end:
        return False
    if chop_start <= t_time <= chop_end:
        return False
    return True

def select_affordable_strikes(
    spot: float,
    current_time,
    opt_data: dict = None,
    expiry_prefix: str = "NIFTY29SEP26",
    T_years: float = 0.015,
    iv: float = 0.135,
    min_prem: float = config.DAS_MIN_LEG_PREMIUM,
    max_prem: float = config.DAS_MAX_LEG_PREMIUM,
    max_budget: float = config.INITIAL_CAPITAL
):
    """
    Dynamically scans strikes around spot to find CE and PE strikes whose
    individual premiums fit the target range (35 to 68 pts) and combined capital <= Rs 10,000.
    
    Returns:
        (chosen_ce_sym_or_strike, chosen_pe_sym_or_strike, ce_price, pe_price, total_cost)
        or None if no valid strikes found.
    """
    atm = int(round(spot / 50.0) * 50)
    
    # 1. Real Option Data Mode
    if opt_data is not None:
        chosen_ce = None
        chosen_pe = None
        ce_p = None
        pe_p = None

        # Check CE candidates (OTM +50, +100, +150, +200, or ATM 0)
        for offset in [50, 100, 150, 200, 0]:
            k = atm + offset
            sym = f"{expiry_prefix}{k}CE"
            if sym in opt_data and current_time in opt_data[sym].index:
                p = float(opt_data[sym].loc[current_time, 'close'])
                if min_prem <= p <= max_prem:
                    chosen_ce = sym
                    ce_p = p
                    break

        # Check PE candidates (OTM -50, -100, -150, -200, or ATM 0)
        for offset in [50, 100, 150, 200, 0]:
            k = atm - offset
            sym = f"{expiry_prefix}{k}PE"
            if sym in opt_data and current_time in opt_data[sym].index:
                p = float(opt_data[sym].loc[current_time, 'close'])
                if min_prem <= p <= max_prem:
                    chosen_pe = sym
                    pe_p = p
                    break

        if chosen_ce and chosen_pe:
            tot_cost = (ce_p + pe_p) * config.LOT_SIZE
            if tot_cost <= max_budget:
                return chosen_ce, chosen_pe, ce_p, pe_p, tot_cost
        return None

    # 2. Mathematical / BSM Simulation Mode
    for offset in [50, 100, 150, 200, 0]:
        ce_strike = atm + offset
        pe_strike = atm - offset
        
        ce_val = black_scholes_price(spot, ce_strike, T_years, config.RISK_FREE_RATE, iv, "CE")
        pe_val = black_scholes_price(spot, pe_strike, T_years, config.RISK_FREE_RATE, iv, "PE")
        
        tot_cost = (ce_val + pe_val) * config.LOT_SIZE
        if tot_cost <= max_budget and (min_prem <= ce_val <= max_prem) and (min_prem <= pe_val <= max_prem):
            return ce_strike, pe_strike, ce_val, pe_val, tot_cost

    # Fallback to standard OTM +50/-50 if within budget
    ce_strike = atm + 50
    pe_strike = atm - 50
    ce_val = black_scholes_price(spot, ce_strike, T_years, config.RISK_FREE_RATE, iv, "CE")
    pe_val = black_scholes_price(spot, pe_strike, T_years, config.RISK_FREE_RATE, iv, "PE")
    tot_cost = (ce_val + pe_val) * config.LOT_SIZE
    if tot_cost <= max_budget:
        return ce_strike, pe_strike, ce_val, pe_val, tot_cost

    return None

def check_compression_expansion(
    combined_history: np.ndarray,
    std_thresh: float = config.DAS_STD_THRESH,
    vel_thresh: float = config.DAS_VEL_THRESH
) -> tuple:
    """
    Checks if combined premium series exhibits standard deviation coiling
    below std_thresh and instantaneous kinetic velocity expansion >= vel_thresh.
    
    Returns:
        (triggered: bool, current_std: float, current_vel: float)
    """
    if len(combined_history) < 2:
        return False, 0.0, 0.0

    comb_std = float(np.std(combined_history))
    comb_vel = float(combined_history[-1] - combined_history[-2])

    triggered = (comb_std < std_thresh) and (comb_vel >= vel_thresh)
    return triggered, round(comb_std, 2), round(comb_vel, 2)
