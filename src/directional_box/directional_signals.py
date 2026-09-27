"""
Signal Generators for Strategy 3: 30-Minute Statistical Box Breakout.
Methodology:
1. Calculates the 30-minute Initial Balance (IB) range from 09:15 to 09:45 AM IST.
2. Filters out choppy/wide days where IB > max_box_pct (default 0.32% of spot).
3. Detects directional breakout above IB High + buffer (CALL Buy) or below IB Low - buffer (PUT Buy).
4. Limits to max 1 trade per day, window 09:45 to 13:30 PM.
"""

import pandas as pd
import numpy as np
from datetime import time

class DirectionalBoxSignals:
    @staticmethod
    def generate_30m_box_signals(
        df: pd.DataFrame,
        max_box_pct: float = 0.32,
        buffer_pts: float = 8.0,
        trade_cutoff_str: str = "13:30"
    ) -> pd.DataFrame:
        data = df.copy()
        if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)

        data['date'] = data.index.date
        signals = [0] * len(data) # +1 for CE, -1 for PE
        
        start_t = time(9, 15)
        box_end_t = time(9, 45)
        cutoff_t = time(int(trade_cutoff_str.split(":")[0]), int(trade_cutoff_str.split(":")[1]))

        for date_val, group in data.groupby('date'):
            box_bars = group[(group.index.time >= start_t) & (group.index.time <= box_end_t)]
            if len(box_bars) < 20:
                continue

            box_h = box_bars['high'].max()
            box_l = box_bars['low'].min()
            box_range = box_h - box_l
            spot_ref = box_bars['close'].iloc[0]
            box_pct = (box_range / spot_ref) * 100.0

            # Only trade if morning box was tightly compressed
            if box_pct <= max_box_pct:
                rem_bars = group[(group.index.time > box_end_t) & (group.index.time <= cutoff_t)]
                for idx in rem_bars.index:
                    row = data.loc[idx]
                    pos_idx = data.index.get_loc(idx)

                    # Upside Breakout -> CALL BUY
                    if row['high'] >= (box_h + buffer_pts):
                        signals[pos_idx] = 1
                        break
                    # Downside Breakout -> PUT BUY
                    elif row['low'] <= (box_l - buffer_pts):
                        signals[pos_idx] = -1
                        break

        data['signal'] = signals
        return data
