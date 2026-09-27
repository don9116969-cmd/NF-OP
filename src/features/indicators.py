"""
Technical and Microstructure Indicators for Nifty Intraday Trading.
"""
import pandas as pd
import numpy as np

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculates intraday Volume Weighted Average Price (VWAP).
    Resets at the start of each new trading day.
    """
    df = df.copy()
    if 'date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['date'] = df.index.date
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume']
    
    # If all volume is 0 (common for index spot candles), use expanding session mean
    if vol.sum() == 0:
        return df.groupby('date')['close'].transform(lambda s: s.expanding().mean())

    vol_clean = vol.replace(0, 1)
    df['tp_vol'] = typical_price * vol_clean
    df['vol_clean'] = vol_clean

    cum_vol = df.groupby('date')['vol_clean'].cumsum()
    cum_tp_vol = df.groupby('date')['tp_vol'].cumsum()
    
    vwap = cum_tp_vol / cum_vol
    return vwap


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculates SuperTrend indicator.
    Returns DataFrame with columns ['supertrend', 'direction'].
    direction: 1 for Bullish (Green), -1 for Bearish (Red).
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    atr = calculate_atr(df, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = [0.0] * len(df)
    direction = [1] * len(df)
    
    for i in range(1, len(df)):
        curr_close = close.iloc[i]
        prev_close = close.iloc[i - 1]
        
        # Lower band adjustment
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or prev_close < lower_band.iloc[i - 1]:
            curr_lower = lower_band.iloc[i]
        else:
            curr_lower = lower_band.iloc[i - 1]
            
        # Upper band adjustment
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or prev_close > upper_band.iloc[i - 1]:
            curr_upper = upper_band.iloc[i]
        else:
            curr_upper = upper_band.iloc[i - 1]
            
        # Trend direction
        if curr_close > curr_upper:
            direction[i] = 1
        elif curr_close < curr_lower:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            
        supertrend[i] = curr_lower if direction[i] == 1 else curr_upper
        
    res = pd.DataFrame(index=df.index)
    res['supertrend'] = supertrend
    res['direction'] = direction
    return res


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Calculates Bollinger Bands (Middle, Upper, Lower, Bandwidth)."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    
    return middle, upper, lower, bandwidth
