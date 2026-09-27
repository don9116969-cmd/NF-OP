"""
High-Precision 1-Minute Option Backtest Engine with Exact PineScript Exits:
- 0.3% Target & 0.2% Stop Loss on Spot (and Option Delta equivalent)
- 3 Time-Based Exits (10m @ +0.1%, 15m @ +0.15%, 30m @ +0.2%)
- Daily Limit: Max 2 wins OR 1 loss per day
- EOD Exit at 15:15 IST
- Full Black-Scholes Greeks, 1-minute Theta decay, 0.8% slippage, and Indian taxes
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.features.greeks import black_scholes_price
from config import config

@dataclass
class Trade1mRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    trade_type: str        # 'CALL' or 'PUT'
    strike: int
    entry_spot: float
    exit_spot: float
    spot_pnl_pct: float
    entry_premium: float
    exit_premium: float
    quantity: int
    gross_pnl: float
    charges: float
    net_pnl: float
    exit_reason: str       # 'Target', 'StopLoss', 'TBE_10m', 'TBE_15m', 'TBE_30m', 'EOD'
    bars_held: int

class Engine1Min:
    def __init__(
        self,
        initial_capital: float = 50000.0,
        lot_size: int = 75,
        num_lots: int = 1,
        target_pct: float = 0.3,         # 0.3% spot target
        sl_pct: float = 0.2,             # 0.2% spot SL
        tbe_enabled: bool = True,
        tbe1_min: int = 10, tbe1_pct: float = 0.10,
        tbe2_min: int = 15, tbe2_pct: float = 0.15,
        tbe3_min: int = 30, tbe3_pct: float = 0.20,
        max_win_trades: int = 2,
        max_loss_trades: int = 1,
        iv_annual: float = 0.14,
        days_to_expiry: float = 3.0
    ):
        self.initial_capital = initial_capital
        self.quantity = lot_size * num_lots
        self.target_pct = target_pct
        self.sl_pct = sl_pct
        self.tbe_enabled = tbe_enabled
        self.tbe1_min = tbe1_min
        self.tbe1_pct = tbe1_pct
        self.tbe2_min = tbe2_min
        self.tbe2_pct = tbe2_pct
        self.tbe3_min = tbe3_min
        self.tbe3_pct = tbe3_pct
        self.max_win_trades = max_win_trades
        self.max_loss_trades = max_loss_trades
        self.iv_annual = iv_annual
        self.days_to_expiry = days_to_expiry

    def _calc_option_prem(self, spot: float, strike: int, opt_type: str, dte_days: float) -> float:
        T = max(dte_days, 0.005) / 365.0
        return black_scholes_price(
            S=spot, K=strike, T=T, r=config.RISK_FREE_RATE, sigma=self.iv_annual, option_type=opt_type
        )

    def _calc_charges(self, buy_p: float, sell_p: float) -> float:
        buy_val = buy_p * self.quantity
        sell_val = sell_p * self.quantity
        turnover = buy_val + sell_val
        brokerage = config.BROKERAGE_PER_ORDER * 2.0
        stt = sell_val * config.STT_PCT_ON_SELL
        exch = turnover * config.EXCHANGE_TURNOVER_PCT
        gst = (brokerage + exch) * config.GST_PCT
        sebi = turnover * config.SEBI_CHARGES_PCT
        stamp = buy_val * config.STAMP_DUTY_PCT_BUY
        return round(brokerage + stt + exch + gst + sebi + stamp, 2)

    def run(self, df_with_signals: pd.DataFrame) -> Dict[str, Any]:
        data = df_with_signals.copy()
        if 'date' not in data.columns:
            data['date'] = data.index.date

        trades: List[Trade1mRecord] = []
        equity = self.initial_capital
        equity_curve = [equity]

        current_day = None
        win_count = 0
        loss_count = 0

        in_pos = False
        pos_type = ""
        pos_strike = 0
        pos_entry_spot = 0.0
        pos_entry_prem = 0.0
        pos_entry_time = None
        pos_bars_held = 0
        current_dte = self.days_to_expiry

        n = len(data)
        times = data.index.time
        eod_time = pd.to_datetime("15:15").time()

        for i in range(n):
            bar = data.iloc[i]
            bar_time = data.index[i]
            bar_date = bar['date']

            # New day reset
            if bar_date != current_day:
                current_day = bar_date
                win_count = 0
                loss_count = 0
                current_dte = self.days_to_expiry

            # 1-minute theta decay: ~1/375 of a day per 1-minute bar
            current_dte -= (1.0 / 375.0)

            # 1. Manage Active Position
            if in_pos:
                pos_bars_held += 1
                curr_spot = bar['close']
                h_spot = bar['high']
                l_spot = bar['low']

                exit_now = False
                exit_reason = ""
                exit_spot_p = curr_spot

                # Check Stop Loss & Target on spot
                if pos_type == "CALL":
                    target_level = pos_entry_spot * (1 + self.target_pct / 100.0)
                    sl_level = pos_entry_spot * (1 - self.sl_pct / 100.0)

                    # Check Target
                    if h_spot >= target_level:
                        exit_now = True
                        exit_reason = "Target"
                        exit_spot_p = target_level
                    # Check SL
                    elif l_spot <= sl_level:
                        exit_now = True
                        exit_reason = "StopLoss"
                        exit_spot_p = sl_level
                    # Time-Based Exits
                    elif self.tbe_enabled:
                        curr_gain_pct = (h_spot - pos_entry_spot) / pos_entry_spot * 100.0
                        if pos_bars_held >= self.tbe1_min and curr_gain_pct >= self.tbe1_pct:
                            exit_now = True
                            exit_reason = f"TBE_{self.tbe1_min}m"
                            exit_spot_p = curr_spot
                        elif pos_bars_held >= self.tbe2_min and curr_gain_pct >= self.tbe2_pct:
                            exit_now = True
                            exit_reason = f"TBE_{self.tbe2_min}m"
                            exit_spot_p = curr_spot
                        elif pos_bars_held >= self.tbe3_min and curr_gain_pct >= self.tbe3_pct:
                            exit_now = True
                            exit_reason = f"TBE_{self.tbe3_min}m"
                            exit_spot_p = curr_spot

                else:  # PUT
                    target_level = pos_entry_spot * (1 - self.target_pct / 100.0)
                    sl_level = pos_entry_spot * (1 + self.sl_pct / 100.0)

                    # Check Target
                    if l_spot <= target_level:
                        exit_now = True
                        exit_reason = "Target"
                        exit_spot_p = target_level
                    # Check SL
                    elif h_spot >= sl_level:
                        exit_now = True
                        exit_reason = "StopLoss"
                        exit_spot_p = sl_level
                    # Time-Based Exits
                    elif self.tbe_enabled:
                        curr_gain_pct = (pos_entry_spot - l_spot) / pos_entry_spot * 100.0
                        if pos_bars_held >= self.tbe1_min and curr_gain_pct >= self.tbe1_pct:
                            exit_now = True
                            exit_reason = f"TBE_{self.tbe1_min}m"
                            exit_spot_p = curr_spot
                        elif pos_bars_held >= self.tbe2_min and curr_gain_pct >= self.tbe2_pct:
                            exit_now = True
                            exit_reason = f"TBE_{self.tbe2_min}m"
                            exit_spot_p = curr_spot
                        elif pos_bars_held >= self.tbe3_min and curr_gain_pct >= self.tbe3_pct:
                            exit_now = True
                            exit_reason = f"TBE_{self.tbe3_min}m"
                            exit_spot_p = curr_spot

                # EOD Exit at 15:15 IST
                if not exit_now and bar_time.time() >= eod_time:
                    exit_now = True
                    exit_reason = "EOD"
                    exit_spot_p = curr_spot

                if exit_now:
                    opt_type_code = "CE" if pos_type == "CALL" else "PE"
                    raw_exit_prem = self._calc_option_prem(exit_spot_p, pos_strike, opt_type_code, current_dte)
                    # Apply slippage on exit
                    realized_exit_prem = max(0.5, raw_exit_prem * (1 - config.SLIPPAGE_PCT))

                    gross = (realized_exit_prem - pos_entry_prem) * self.quantity
                    charges = self._calc_charges(pos_entry_prem, realized_exit_prem)
                    net = gross - charges

                    spot_pct = round(((exit_spot_p - pos_entry_spot) / pos_entry_spot) * 100.0 * (1 if pos_type == "CALL" else -1), 3)

                    trades.append(Trade1mRecord(
                        entry_time=pos_entry_time,
                        exit_time=bar_time,
                        trade_type=pos_type,
                        strike=pos_strike,
                        entry_spot=round(pos_entry_spot, 2),
                        exit_spot=round(exit_spot_p, 2),
                        spot_pnl_pct=spot_pct,
                        entry_premium=round(pos_entry_prem, 2),
                        exit_premium=round(realized_exit_prem, 2),
                        quantity=self.quantity,
                        gross_pnl=round(gross, 2),
                        charges=round(charges, 2),
                        net_pnl=round(net, 2),
                        exit_reason=exit_reason,
                        bars_held=pos_bars_held
                    ))

                    equity += net
                    if net > 0:
                        win_count += 1
                    else:
                        loss_count += 1

                    in_pos = False

            # 2. Check for New Entry Signal
            if not in_pos:
                daily_limit_hit = (win_count >= self.max_win_trades) or (loss_count >= self.max_loss_trades)
                sig = bar.get('signal', 0)

                if (not daily_limit_hit) and (sig != 0) and (bar_time.time() < eod_time):
                    pos_type = "CALL" if sig == 1 else "PUT"
                    pos_entry_spot = bar['close']
                    pos_strike = int(round(pos_entry_spot / config.STRIKE_STEP) * config.STRIKE_STEP)

                    opt_type_code = "CE" if pos_type == "CALL" else "PE"
                    raw_entry_prem = self._calc_option_prem(pos_entry_spot, pos_strike, opt_type_code, current_dte)
                    # Apply slippage on entry
                    pos_entry_prem = round(raw_entry_prem * (1 + config.SLIPPAGE_PCT), 2)

                    pos_entry_time = bar_time
                    pos_bars_held = 0
                    in_pos = True

            equity_curve.append(equity)

        # 3. Summarize
        return self._summarize(trades, equity_curve)

    def _summarize(self, trades: List[Trade1mRecord], equity_curve: List[float]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
                "net_pnl_inr": 0.0, "return_on_capital_pct": 0.0, "max_drawdown_inr": 0.0,
                "max_drawdown_pct": 0.0, "total_charges_inr": 0.0, "trades_df": pd.DataFrame(),
                "exit_reasons": {}
            }

        df_t = pd.DataFrame([t.__dict__ for t in trades])
        wins = df_t[df_t['net_pnl'] > 0]
        losses = df_t[df_t['net_pnl'] <= 0]

        total_gain = wins['net_pnl'].sum() if not wins.empty else 0.0
        total_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0

        pf = round(total_gain / total_loss, 2) if total_loss > 0 else (99.0 if total_gain > 0 else 0.0)
        win_rate = round((len(wins) / len(df_t)) * 100.0, 2)
        net_pnl = round(df_t['net_pnl'].sum(), 2)

        eq = pd.Series(equity_curve)
        peak = eq.cummax()
        dd = (peak - eq) / peak
        max_dd_pct = round(dd.max() * 100.0, 2)
        max_dd_inr = round((peak - eq).max(), 2)

        return {
            "total_trades": len(df_t),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": win_rate,
            "profit_factor": pf,
            "net_pnl_inr": net_pnl,
            "return_on_capital_pct": round((net_pnl / self.initial_capital) * 100.0, 2),
            "total_charges_inr": round(df_t['charges'].sum(), 2),
            "max_drawdown_inr": max_dd_inr,
            "max_drawdown_pct": max_dd_pct,
            "exit_reasons": df_t['exit_reason'].value_counts().to_dict(),
            "trades_df": df_t
        }
