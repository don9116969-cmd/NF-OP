"""
Realistic Event-Driven Option Buying Backtest Engine with Indian Taxes and Slippage.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from config import config
from src.features.greeks import black_scholes_price, calculate_greeks

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    trade_type: str        # 'CE' or 'PE'
    strike: int
    entry_spot: float
    exit_spot: float
    entry_premium: float
    exit_premium: float
    quantity: int
    gross_pnl: float
    charges: float
    net_pnl: float
    exit_reason: str       # 'Target', 'StopLoss', 'TimeStop', 'SquareOff'
    bars_held: int

class OptionBacktestEngine:
    def __init__(
        self,
        initial_capital: float = config.INITIAL_CAPITAL,
        lot_size: int = config.LOT_SIZE,
        num_lots: int = 1,
        stop_loss_pct: float = 0.15,     # 15% SL on option premium
        target_pct: float = 0.25,        # 25% Target on option premium
        trailing_sl: bool = True,        # Move SL to breakeven at +15% profit
        time_stop_bars: int = 4,         # Exit if no momentum after 20 mins (4 x 5-min bars)
        iv_annual: float = 0.15,         # 15% Implied Volatility
        days_to_expiry: float = 3.0      # Average days to expiry for weekly options
    ):
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.num_lots = num_lots
        self.quantity = lot_size * num_lots
        
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.trailing_sl = trailing_sl
        self.time_stop_bars = time_stop_bars
        self.iv_annual = iv_annual
        self.days_to_expiry = days_to_expiry

    def _calculate_option_price(self, spot: float, strike: int, option_type: str, dte_days: float) -> float:
        """Computes realistic option premium accounting for theta decay."""
        T = max(dte_days, 0.01) / 365.0
        return black_scholes_price(
            S=spot,
            K=strike,
            T=T,
            r=config.RISK_FREE_RATE,
            sigma=self.iv_annual,
            option_type=option_type
        )

    def _calculate_taxes_and_charges(self, buy_prem: float, sell_prem: float, qty: int) -> float:
        """Calculates exact Indian STT, exchange fees, GST, SEBI charges, stamp duty, and brokerage."""
        buy_val = buy_prem * qty
        sell_val = sell_prem * qty
        turnover = buy_val + sell_val

        # Brokerage (INR 20 per order x 2)
        brokerage = config.BROKERAGE_PER_ORDER * 2.0
        # STT is only on sell value for options: 0.1% (or 0.0625%)
        stt = sell_val * config.STT_PCT_ON_SELL
        # Exchange turnover charge (NSE: 0.05%)
        exchange_charges = turnover * config.EXCHANGE_TURNOVER_PCT
        # GST (18% on Brokerage + Exchange charges)
        gst = (brokerage + exchange_charges) * config.GST_PCT
        # SEBI Turnover fee
        sebi = turnover * config.SEBI_CHARGES_PCT
        # Stamp duty (0.003% on buy value)
        stamp_duty = buy_val * config.STAMP_DUTY_PCT_BUY

        total_charges = brokerage + stt + exchange_charges + gst + sebi + stamp_duty
        return round(total_charges, 2)

    def run(self, df_with_signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes the backtest bar by bar.
        Enforces daily trade limits, strict stops, and realistic execution.
        """
        data = df_with_signals.copy()
        if 'date' not in data.columns:
            data['date'] = data.index.date

        trades: List[TradeRecord] = []
        equity = self.initial_capital
        equity_curve = [equity]

        # Daily tracking
        current_day = None
        daily_trades_count = 0
        daily_pnl = 0.0

        in_position = False
        pos_type = None
        pos_strike = 0
        pos_entry_spot = 0.0
        pos_entry_prem = 0.0
        pos_sl_prem = 0.0
        pos_target_prem = 0.0
        pos_entry_time = None
        bars_in_trade = 0
        current_dte = self.days_to_expiry

        for i in range(len(data)):
            bar = data.iloc[i]
            bar_time = data.index[i]
            bar_date = bar['date']

            # New day reset
            if bar_date != current_day:
                current_day = bar_date
                daily_trades_count = 0
                daily_pnl = 0.0
                current_dte = self.days_to_expiry

            # Intraday theta decay: fraction of a day per 5-minute bar (75 bars in a day)
            current_dte -= (1.0 / 75.0)

            # 1. Manage Active Position
            if in_position:
                bars_in_trade += 1
                curr_spot = bar['close']
                # Current option price on this bar
                curr_prem = self._calculate_option_price(curr_spot, pos_strike, pos_type, current_dte)

                exit_trade = False
                exit_reason = ""
                exit_price = curr_prem

                # Target Hit
                if curr_prem >= pos_target_prem:
                    exit_trade = True
                    exit_reason = "Target"
                    exit_price = pos_target_prem
                # Stop Loss Hit
                elif curr_prem <= pos_sl_prem:
                    exit_trade = True
                    exit_reason = "StopLoss"
                    exit_price = pos_sl_prem
                # Trailing SL: if profit >= +15%, move SL to break-even (entry price)
                elif self.trailing_sl and curr_prem >= pos_entry_prem * (1 + self.stop_loss_pct):
                    pos_sl_prem = max(pos_sl_prem, pos_entry_prem)

                # Time Stop: If position is stagnant after `time_stop_bars`, exit
                if not exit_trade and bars_in_trade >= self.time_stop_bars:
                    gain_pct = (curr_prem - pos_entry_prem) / pos_entry_prem
                    if gain_pct < 0.05:  # Stagnant or decaying
                        exit_trade = True
                        exit_reason = "TimeStop"
                        exit_price = curr_prem

                # Intraday 15:15 Auto Square-off
                if not exit_trade and bar_time.time() >= pd.to_datetime(config.SQUARE_OFF_TIME).time():
                    exit_trade = True
                    exit_reason = "SquareOff"
                    exit_price = curr_prem

                if exit_trade:
                    # Apply slippage on exit (getting 0.8% lower than theoretical bid)
                    realized_exit_prem = max(0.5, exit_price * (1 - config.SLIPPAGE_PCT))
                    gross = (realized_exit_prem - pos_entry_prem) * self.quantity
                    charges = self._calculate_taxes_and_charges(pos_entry_prem, realized_exit_prem, self.quantity)
                    net = gross - charges

                    trades.append(TradeRecord(
                        entry_time=pos_entry_time,
                        exit_time=bar_time,
                        trade_type=pos_type,
                        strike=pos_strike,
                        entry_spot=pos_entry_spot,
                        exit_spot=curr_spot,
                        entry_premium=round(pos_entry_prem, 2),
                        exit_premium=round(realized_exit_prem, 2),
                        quantity=self.quantity,
                        gross_pnl=round(gross, 2),
                        charges=round(charges, 2),
                        net_pnl=round(net, 2),
                        exit_reason=exit_reason,
                        bars_held=bars_in_trade
                    ))

                    equity += net
                    daily_pnl += net
                    in_position = False

            # 2. Check for New Entry Signal
            if not in_position:
                # Enforce Low-Risk Guardrails: Max trades per day and Daily loss circuit breaker
                can_trade = (daily_trades_count < config.MAX_TRADES_PER_DAY) and (daily_pnl > -config.MAX_DAILY_LOSS)
                is_trade_time = (bar_time.time() >= pd.to_datetime(config.TRADE_START_TIME).time()) and \
                                (bar_time.time() < pd.to_datetime("14:30").time())

                sig = bar.get('signal', 0)
                if can_trade and is_trade_time and sig != 0:
                    pos_type = "CE" if sig == 1 else "PE"
                    entry_spot = bar['close']
                    # Strike selection: round to nearest 50 for Nifty ATM
                    pos_strike = int(round(entry_spot / config.STRIKE_STEP) * config.STRIKE_STEP)
                    
                    raw_entry_prem = self._calculate_option_price(entry_spot, pos_strike, pos_type, current_dte)
                    # Apply slippage on entry (paying 0.8% higher than theoretical ask)
                    pos_entry_prem = round(raw_entry_prem * (1 + config.SLIPPAGE_PCT), 2)
                    
                    pos_sl_prem = round(pos_entry_prem * (1 - self.stop_loss_pct), 2)
                    pos_target_prem = round(pos_entry_prem * (1 + self.target_pct), 2)
                    
                    pos_entry_spot = entry_spot
                    pos_entry_time = bar_time
                    bars_in_trade = 0
                    in_position = True
                    daily_trades_count += 1

            equity_curve.append(equity)

        # 3. Calculate Performance Metrics
        return self._summarize_results(trades, equity_curve)

    def _summarize_results(self, trades: List[TradeRecord], equity_curve: List[float]) -> Dict[str, Any]:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "max_drawdown_pct": 0.0,
                "trades": []
            }

        df_trades = pd.DataFrame([t.__dict__ for t in trades])
        wins = df_trades[df_trades['net_pnl'] > 0]
        losses = df_trades[df_trades['net_pnl'] <= 0]

        total_gain = wins['net_pnl'].sum() if not wins.empty else 0.0
        total_loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
        
        profit_factor = round(total_gain / total_loss, 2) if total_loss > 0 else (99.0 if total_gain > 0 else 0.0)
        win_rate = round((len(wins) / len(df_trades)) * 100.0, 2)
        net_pnl = round(df_trades['net_pnl'].sum(), 2)
        total_charges = round(df_trades['charges'].sum(), 2)

        # Drawdown calculation
        eq = pd.Series(equity_curve)
        peak = eq.cummax()
        dd = (peak - eq) / peak
        max_dd_pct = round(dd.max() * 100.0, 2)
        max_dd_inr = round((peak - eq).max(), 2)

        return {
            "total_trades": len(df_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "net_pnl_inr": net_pnl,
            "return_on_capital_pct": round((net_pnl / self.initial_capital) * 100.0, 2),
            "total_charges_inr": total_charges,
            "max_drawdown_inr": max_dd_inr,
            "max_drawdown_pct": max_dd_pct,
            "avg_trade_pnl_inr": round(df_trades['net_pnl'].mean(), 2),
            "exit_reasons": df_trades['exit_reason'].value_counts().to_dict(),
            "trades_df": df_trades
        }
