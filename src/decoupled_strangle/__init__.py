"""
Decoupled Asymmetric Strangle (DAS) Module.
Delta-Neutral Volatility Compression and Kinetic Expansion Strategy for Nifty Options.
"""
from .das_signals import (
    select_affordable_strikes,
    check_compression_expansion,
    is_trade_window_valid
)
from .das_engine import (
    evaluate_das_for_day,
    run_das_simulation
)

__all__ = [
    "select_affordable_strikes",
    "check_compression_expansion",
    "is_trade_window_valid",
    "evaluate_das_for_day",
    "run_das_simulation"
]
