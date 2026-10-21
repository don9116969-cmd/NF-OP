"""
Configuration module for Nifty & BankNifty Option Buying Portfolio Engine.
Tri-Strategy Core Production Architecture:
1. Strategy 1: Nifty 15M Directional Breakout (Trend Aligned ATM, SL -16%, TP +60%)
2. Strategy 2: BankNIFTY 15M Directional Breakout (Trend Aligned 100-pt OTM, SL -25%, TP +80%)
3. Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS) (Std < 30, Vel >= 35, Win +60%, Lose -12%)
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class TradingConfig:
    # Asset settings
    INDEX_NAME: str = "NIFTY"
    LOT_SIZE: int = 65  # Official NSE Nifty standard lot size
    BANKNIFTY_LOT_SIZE: int = 30  # Official NSE BankNifty standard lot size
    STRIKE_STEP: int = 50  # Nifty strike intervals
    BANKNIFTY_STRIKE_STEP: int = 100  # BankNifty strike intervals
    
    # Capital & Risk Controls (₹10,000 Budget)
    INITIAL_CAPITAL: float = 10000.0  # User's exact budget in INR
    MAX_CAPITAL_PER_TRADE: float = 8500.0  # Safe cap to always preserve ₹1,500 buffer
    MAX_CONCURRENT_POSITIONS: int = 1  # Strictly 1 active trade at any time across account
    
    # Execution & Charges (Indian Stock Market)
    BROKERAGE_PER_ORDER: float = 20.0  # Angel One flat brokerage (INR 20 per order)
    STT_PCT_ON_SELL: float = 0.001  # STT 0.1% on option sell turnover
    EXCHANGE_TURNOVER_PCT: float = 0.0005  # ~0.05% NSE exchange charge
    GST_PCT: float = 0.18  # 18% GST on (Brokerage + Exchange turnover)
    SEBI_CHARGES_PCT: float = 0.000001  # INR 10 per crore
    STAMP_DUTY_PCT_BUY: float = 0.00003  # 0.003% on buy value
    SLIPPAGE_PCT: float = 0.005  # 0.5% realistic execution slippage
    
    # Greeks & Market Parameters
    RISK_FREE_RATE: float = 0.07  # RBI repo / risk-free rate ~7%
    DEFAULT_IV: float = 0.15  # Benchmark IV 15%
    
    # Timing (IST)
    MARKET_START_TIME: str = "09:15"
    TRADE_START_TIME: str = "09:30"
    SQUARE_OFF_TIME: str = "15:15"

    # =========================================================================
    # CORE STRATEGY PARAMETERS
    # =========================================================================

    # STRATEGY 1: NIFTY 15M DIRECTIONAL BREAKOUT
    S1_ENABLED: bool = True
    S1_BOX_START: str = "09:15"
    S1_BOX_END: str = "09:30"
    S1_BUFFER_PTS: float = 8.0  # 8.0 pt breakout buffer
    S1_MAX_BOX_PCT: float = 0.30  # Filter out boxes wider than 0.30% to avoid chop
    S1_USE_EMA_TREND: bool = True  # Confirm Spot > 20-EMA for CE, Spot < 20-EMA for PE
    S1_SL_PCT: float = 0.16  # -16% tight stop loss
    S1_TARGET_PCT: float = 0.60  # +60% profit target
    S1_MAX_BUDGET: float = 8500.0

    # STRATEGY 2: BANKNIFTY 15M DIRECTIONAL BREAKOUT
    S2_ENABLED: bool = True
    S2_BOX_START: str = "09:15"
    S2_BOX_END: str = "09:30"
    S2_BUFFER_PTS: float = 25.0  # 25.0 pt breakout buffer
    S2_MAX_BOX_PCT: float = 1.20  # Filter out boxes wider than 1.20%
    S2_USE_EMA_TREND: bool = True  # Confirm Spot > 20-EMA for CE, Spot < 20-EMA for PE
    S2_OTM_OFFSET: int = 100  # 100-pt OTM option
    S2_SL_PCT: float = 0.25  # -25% stop loss
    S2_TARGET_PCT: float = 0.80  # +80% profit target
    S2_MAX_BUDGET: float = 8500.0

    # STRATEGY 3: NIFTY DECOUPLED ASYMMETRIC STRANGLE (DAS)
    S3_ENABLED: bool = True
    DAS_VOL_WINDOW: int = 20  # 20-min rolling window for std
    DAS_STD_THRESH: float = 30.0  # Rolling standard deviation compression (< 30.0)
    DAS_VEL_THRESH: float = 35.0  # Kinetic expansion velocity (>= 35.0 pts in 5 mins)
    DAS_MIN_LEG_PREMIUM: float = 35.0  # Min premium per leg for DAS strike selection
    DAS_MAX_LEG_PREMIUM: float = 68.0  # Max premium per leg for DAS strike selection
    DAS_WIN_TARGET_PCT: float = 0.60  # +60% target on surging winning leg
    DAS_LOSE_STOP_PCT: float = 0.12  # -12% tight stop loss on decaying losing leg
    DAS_COMBINED_STOP_PCT: float = 0.25  # Combined basket stop loss
    DAS_MAX_HOLD_MINS: int = 50  # 50-minute maximum holding cutoff
    DAS_COOLDOWN_BARS: int = 15  # 15-minute cooldown between entries
    DAS_TRADE_START: str = "09:30"
    DAS_TRADE_END: str = "14:00"
    DAS_MAX_BUDGET: float = 8500.0

    # Risk Controls
    MAX_DAILY_LOSS: float = 2500.0  # Max account daily loss limit
    MAX_TRADES_PER_DAY: int = 4  # Max trades across portfolio per day

    ACTIVE_STRATEGIES: List[str] = field(default_factory=lambda: [
        "Strategy 1: Nifty 15M Directional Breakout",
        "Strategy 2: BankNIFTY 15M Directional Breakout",
        "Strategy 3: Nifty Decoupled Asymmetric Strangle (DAS)"
    ])

config = TradingConfig()
