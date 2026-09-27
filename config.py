"""
Configuration module for Nifty Option Buying Model and Backtesting Engine.
"""
from dataclasses import dataclass

@dataclass
class TradingConfig:
    # Asset settings
    INDEX_NAME: str = "NIFTY"
    LOT_SIZE: int = 75  # Current standard Nifty lot size
    STRIKE_STEP: int = 50  # Nifty strike intervals (e.g. 24000, 24050, 24100)
    
    # Capital & Risk Controls (₹10,000 Budget)
    INITIAL_CAPITAL: float = 10000.0  # User's exact budget in INR
    MAX_RISK_PER_TRADE: float = 1500.0  # Max risk on combined strangle per trade
    MAX_TRADES_PER_WEEK: int = 4  # Strictly 3 to 4 quality trades per week
    MAX_TRADES_PER_DAY: int = 1  # At most 1 trade per day (some days 0)
    
    # Execution & Charges (Indian Stock Market)
    BROKERAGE_PER_ORDER: float = 20.0  # Angel One / Flat brokerage (INR 20 per order)
    STT_PCT_ON_SELL: float = 0.001  # STT 0.1% on option sell turnover
    EXCHANGE_TURNOVER_PCT: float = 0.0005  # ~0.05% NSE exchange charge
    GST_PCT: float = 0.18  # 18% GST on (Brokerage + Exchange turnover)
    SEBI_CHARGES_PCT: float = 0.000001  # INR 10 per crore
    STAMP_DUTY_PCT_BUY: float = 0.00003  # 0.003% on buy value
    SLIPPAGE_PCT: float = 0.008  # 0.8% realistic execution slippage on option premium
    
    # Option Greeks & Strike Selection
    PREFERRED_STRIKE: str = "ATM"  # ATM or ITM_1 (Avoid OTM for low risk)
    RISK_FREE_RATE: float = 0.07  # RBI repo / risk-free rate ~7%
    
    # Timing (IST)
    MARKET_START_TIME: str = "09:15"
    TRADE_START_TIME: str = "09:25"  # Let initial 10 mins noise settle
    SQUARE_OFF_TIME: str = "15:15"  # Force exit all intraday trades before close

    # Strategy 4: Decoupled Asymmetric Strangle (DAS) Settings
    DAS_VOL_WINDOW: int = 15  # 15-minute rolling window for premium volatility
    DAS_STD_THRESH: float = 5.0  # Premium standard deviation coiling threshold (<= 5.0)
    DAS_VEL_THRESH: float = 1.0  # Premium velocity expansion threshold (>= 1.0)
    DAS_WIN_TARGET_PCT: float = 0.50  # +50% profit target on winning leg
    DAS_LOSE_STOP_PCT: float = 0.35  # -35% stop loss on decaying leg
    DAS_COMBINED_STOP_PCT: float = 0.15  # -15% stop loss on total position
    DAS_MAX_HOLD_MINS: int = 25  # 25-minute maximum holding time
    DAS_MIN_LEG_PREMIUM: float = 35.0  # Minimum leg premium in points
    DAS_MAX_LEG_PREMIUM: float = 68.0  # Maximum leg premium in points (ensures <= Rs 10k budget)
    DAS_COOLDOWN_BARS: int = 30  # 30-minute cooldown between trades
    DAS_TRADE_START: str = "09:45"  # DAS active trading window start
    DAS_TRADE_END: str = "14:15"  # DAS active trading window end
    DAS_CHOP_START: str = "11:30"  # Midday chop filter start
    DAS_CHOP_END: str = "13:00"  # Midday chop filter end

config = TradingConfig()

