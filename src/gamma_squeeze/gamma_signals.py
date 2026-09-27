"""
Mathematical Signal Generators for Strategy 2: 0-DTE / 1-DTE Expiry Gamma Squeeze.
Targets:
1. Wednesday Afternoon (1-DTE Pre-Expiry Positioning: 13:00 - 15:00)
2. Thursday Expiry Morning (0-DTE Initial Balance Breakout: 09:45 - 11:30)
3. Thursday Expiry Afternoon (0-DTE European Open / Short-Covering Squeeze: 13:15 - 14:45)

Filters:
- Consolidation Squeeze: Requires price to compress in a tight range for N minutes.
- Breakout: Sharp expansion above range high (Call trigger) or below range low (Put trigger).
- Frequency: Max 1 trade per day, strictly 1-2 quality trades per week.
"""

import pandas as pd
import numpy as np
from datetime import time

class GammaSignalGenerators:
    @staticmethod
    def expiry_consolidation_breakout(
        df: pd.DataFrame,
        compression_mins: int = 30,
        buffer_pts: float = 6.0,
        window: str = "afternoon",   # "afternoon" (13:00-14:45), "morning" (09:45-11:30), or "both"
        include_pre_expiry_monday: bool = True
    ) -> pd.DataFrame:
        """
        Detects consolidation squeeze on 0-DTE Expiry (Tuesday) and 1-DTE Pre-Expiry (Monday),
        followed by high-momentum breakout to capture Gamma explosions under NSE Tuesday Expiry rules.
        """
        data = df.copy()
        if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)

        data['date'] = data.index.date
        data['weekday'] = data.index.weekday # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4

        # Allowed trading days under NSE Rules: Tuesday (1) 0-DTE Expiry; Monday (0) 1-DTE
        allowed_weekdays = {1}
        if include_pre_expiry_monday:
            allowed_weekdays.add(0)

        signals = [0] * len(data) # +1 for CE Buy, -1 for PE Buy
        n = len(data)

        for date_val, group in data.groupby('date'):
            day_weekday = group['weekday'].iloc[0]
            if day_weekday not in allowed_weekdays:
                continue

            # Define active windows
            windows = []
            if window in ["morning", "both"]:
                windows.append((time(9, 45), time(11, 30)))
            if window in ["afternoon", "both"]:
                # 13:15 to 14:45 is the prime gamma squeeze window
                windows.append((time(13, 0), time(14, 45)))

            trade_taken = False

            for win_start, win_end in windows:
                if trade_taken:
                    break

                win_bars = group[(group.index.time >= win_start) & (group.index.time <= win_end)]
                if len(win_bars) < (compression_mins + 5):
                    continue

                # Sliding window of compression_mins
                for i in range(compression_mins, len(win_bars)):
                    lookback_slice = win_bars.iloc[i - compression_mins : i]
                    box_h = lookback_slice['high'].max()
                    box_l = lookback_slice['low'].min()
                    box_rng = box_h - box_l
                    spot_ref = lookback_slice['close'].iloc[-1]

                    # Compression check: range must be tight (< 0.25% of spot, ~60 pts)
                    if (box_rng / spot_ref) * 100.0 <= 0.25:
                        curr_bar = win_bars.iloc[i]
                        pos_idx = data.index.get_loc(curr_bar.name)

                        # Upside breakout -> CALL BUY
                        if curr_bar['high'] >= (box_h + buffer_pts):
                            signals[pos_idx] = 1
                            trade_taken = True
                            break
                        # Downside breakout -> PUT BUY
                        elif curr_bar['low'] <= (box_l - buffer_pts):
                            signals[pos_idx] = -1
                            trade_taken = True
                            break

        data['signal'] = signals
        return data
