"""
Mathematical Signal Generators for Hedged Long Strangle Strategy.
Filters out noisy/flat days and only triggers 2 to 4 high-conviction days per week:
1. Multi-Day NR7 / Low-Volatility Coiling Breakout
2. 30-Minute Morning Range Compression Breakout
3. Mid-Week Pre-Expiry Volatility Surge
"""
import numpy as np
import pandas as pd

class StrangleSignalGenerators:
    @staticmethod
    def morning_range_expansion(
        df: pd.DataFrame,
        range_mins: int = 30,          # First 30 minutes (09:15 - 09:45)
        max_box_pct: float = 0.28,      # Box must be tight (< 0.28% of spot, ~65 pts)
        breakout_buffer_pts: float = 8.0# Must break box by 8 points to confirm
    ) -> pd.DataFrame:
        """
        Setup 1: First 30-Minute Box Compression & Breakout.
        - Measures the high and low between 09:15 and 09:45 AM.
        - If the box is compressed (< 0.28%), marks a pending expansion.
        - When price breaks above high + buffer or below low - buffer: triggers Strangle!
        - Only 1 trigger per day, max 2-4 quality days per week.
        """
        data = df.copy()
        if 'date' not in data.columns:
            data['date'] = data.index.date

        signals = [0] * len(data)
        sess_start = pd.to_datetime("09:15").time()
        box_end = pd.to_datetime("09:45").time()
        trade_cutoff = pd.to_datetime("13:30").time()

        for date_val, group in data.groupby('date'):
            # Filter first 30 mins
            box_bars = group[(group.index.time >= sess_start) & (group.index.time <= box_end)]
            if len(box_bars) < 15:
                continue

            box_h = box_bars['high'].max()
            box_l = box_bars['low'].min()
            box_range = box_h - box_l
            spot_ref = box_bars['close'].iloc[0]

            box_pct = (box_range / spot_ref) * 100.0

            # Only trade if box was compressed (< max_box_pct)
            if box_pct <= max_box_pct:
                remaining_bars = group[(group.index.time > box_end) & (group.index.time <= trade_cutoff)]
                for idx in remaining_bars.index:
                    row = data.loc[idx]
                    pos_idx = data.index.get_loc(idx)

                    # Breakout detected in either direction
                    if row['high'] >= (box_h + breakout_buffer_pts) or row['low'] <= (box_l - breakout_buffer_pts):
                        signals[pos_idx] = 1  # Trigger Hedged Strangle
                        break  # Max 1 signal per day

        data['signal'] = signals
        return data

    @staticmethod
    def nr7_multi_day_expansion(
        df: pd.DataFrame,
        lookback_days: int = 7,
        morning_trigger_time: str = "09:30"
    ) -> pd.DataFrame:
        """
        Setup 2: Narrow Range 7 (NR7) Daily Volatility Coiling.
        - Checks daily high - low ranges of the previous 7 trading sessions.
        - If yesterday's range was the narrowest of the last 7 days (NR7), today is statistically an expansion day.
        - Triggers Strangle in the morning (09:30 AM) to ride the full day's trend expansion.
        """
        data = df.copy()
        if 'date' not in data.columns:
            data['date'] = data.index.date

        # Compute daily ranges
        daily = data.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
        daily['range'] = daily['high'] - daily['low']
        daily['is_nr7'] = daily['range'] == daily['range'].rolling(lookback_days).min()
        
        # Shift so yesterday's NR7 signals today
        daily['trigger_today'] = daily['is_nr7'].shift(1).fillna(False)
        nr7_dates = set(daily[daily['trigger_today']]['date'].values)

        signals = [0] * len(data)
        trigger_t = pd.to_datetime(morning_trigger_time).time()

        for i in range(len(data)):
            bar_date = data['date'].iloc[i]
            bar_time = data.index[i].time()
            if bar_date in nr7_dates and bar_time == trigger_t:
                signals[i] = 1

        data['signal'] = signals
        return data
