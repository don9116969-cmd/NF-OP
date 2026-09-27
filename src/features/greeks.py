"""
Black-Scholes Options Pricing & Greeks Calculator for Indian Index Options.
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple

def black_scholes_price(
    S: float,       # Underlying Spot Price
    K: float,       # Strike Price
    T: float,       # Time to Expiry in Years (e.g. 1 day = 1/365 or 1/252)
    r: float,       # Risk-free interest rate (e.g. 0.07 for 7%)
    sigma: float,   # Implied Volatility (annualized, e.g. 0.15 for 15%)
    option_type: str = "CE" # "CE" for Call, "PE" for Put
) -> float:
    """Calculates theoretical Black-Scholes option price."""
    if T <= 0:
        if option_type == "CE":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
            
    if sigma <= 0:
        sigma = 1e-4

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "PE":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'CE' or 'PE'")

    return max(0.05, float(price))


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "CE"
) -> Dict[str, float]:
    """
    Computes Delta, Gamma, Theta (per calendar day), Vega (per 1% IV change), and Rho.
    """
    if T <= 0:
        intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
        delta = 1.0 if (option_type == "CE" and S > K) else (-1.0 if (option_type == "PE" and S < K) else 0.0)
        return {"price": intrinsic, "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sigma = max(1e-4, sigma)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Price
    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        # Theta per day (divide by 365)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

    # Gamma and Vega are identical for Call and Put
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = (S * sqrt_T * norm.pdf(d1)) / 100.0  # Vega per 1% change in IV

    return {
        "price": max(0.05, float(price)),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega)
    }


def find_implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    tol: float = 1e-4,
    max_iter: int = 100
) -> float:
    """Calculates Implied Volatility (IV) using Newton-Raphson method with bisection fallback."""
    intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    if market_price <= intrinsic or T <= 0:
        return 0.15  # Fallback default IV

    # Initial guess
    sigma = 0.20
    for _ in range(max_iter):
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        price_diff = greeks["price"] - market_price
        if abs(price_diff) < tol:
            return float(sigma)
        vega = greeks["vega"] * 100.0  # Unscale vega
        if abs(vega) < 1e-6:
            break
        sigma -= price_diff / vega
        if sigma <= 0.001 or sigma > 5.0:
            break

    # Bisection method fallback
    low, high = 0.001, 3.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        p = black_scholes_price(S, K, T, r, mid, option_type)
        if abs(p - market_price) < tol:
            return float(mid)
        if p > market_price:
            high = mid
        else:
            low = mid

    return float((low + high) / 2.0)
