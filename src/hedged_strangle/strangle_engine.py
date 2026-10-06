"""
Hedged Long Strangle Engine for Nifty Options Trading (Budget: INR 10,000).
Executes a simultaneous OTM Call + OTM Put position when mathematical volatility expansion conditions are met.
Features:
- Accurate Black-Scholes pricing for both Call and Put legs simultaneously.
- Combined premium PnL tracking (accounting for non-linear Gamma expansion on the winning leg and Theta decay on the losing leg).
- Indian regulatory charges for multi-leg orders (2 buy orders + 2 sell orders = 4 orders brokerage, STT on both sell legs, GST, exchange turnover).
- Trailing Stop-Loss on combined premium (Breakeven lock).
- Strict 1 trade per day limit, targeting 2 to 4 high-conviction trades per week.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from src.features.greeks import black_scholes_price
from config import config

@dataclass
class StrangleTradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_spot: float
    exit_spot: float
    spot_move_pts: float
    call_strike: int
    put_strike: int
    ce_entry_prem: float
    pe_entry_prem: float
    combined_entry_prem: float
    ce_exit_prem: float
    pe_exit_prem: float
    combined_exit_prem: float
    quantity: int
    gross_pnl: float
    charges: float
    net_pnl: float
    exit_reason: str
    bars_held: int

class HedgedStrangleEngine:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        lot_size: int = config.LOT_SIZE,
        otm_distance_pts: int = 100,      # E.g. 100 points OTM (Call +100, Put -100)
        combined_target_pct: float = 0.35, # +35% target on combined entry cost
        combined_sl_pct: float = 0.20,     # -20% stop loss on combined entry cost
        breakeven_profit_pct: float = 0.18,# Lock SL to breakeven when profit reaches +18%
        iv_annual: float = 0.14,
        days_to_expiry: float = 3.0
    ):
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.otm_distance_pts = otm_distance_pts
        self.combined_target_pct = combined_target_pct
        self.combined_sl_pct = combined_sl_pct
        self.breakeven_profit_pct = breakeven_profit_pct
        self.iv_annual = iv_annual
        self.days_to_expiry = days_to_expiry

    def _calculate_strangle_charges(self, ce_buy: float, pe_buy: float, ce_sell: float, pe_sell: float) -> float:
        """Calculates exact Indian STT, exchange fees, GST, SEBI charges, stamp duty, and brokerage for 4 orders."""
        qty = self.lot_size
        buy_val = (ce_buy + pe_buy) * qty
        sell_val = (ce_sell + pe_sell) * qty
        turnover = buy_val + sell_val

        # 4 orders (Buy CE, Buy PE, Sell CE, Sell PE)
        brokerage = config.BROKERAGE_PER_ORDER * 4.0
        # STT on sell turnover (0.1%)
        stt = sell_val * config.STT_PCT_ON_SELL
        # Exchange turnover (0.05%)
        exchange_charges = turnover * config.EXCHANGE_TURNOVER_PCT
        # GST (18% on brokerage + exchange)
        gst = (brokerage + exchange_charges) * config.GST_PCT
        # SEBI fee
        sebi = turnover * config.SEBI_CHARGES_PCT
        # Stamp duty on buy
        stamp_duty = buy_val * config.STAMP_DUTY_PCT_BUY

        total = brokerage + stt + exchange_charges + gst + sebi + stamp_duty
        return round(total, 2)

    def run_backtest(self, df_signals: pd.DataFrame) -> Dict[str, Any]:
        data = df_signals.copy()
        if 'date' not in data.columns:
            data['date'] = data.index.date

        trades: List[StrangleTradeRecord] = []
        equity = self.initial_capital
        equity_curve = [equity]

        current_day = None
        daily_trades = 0

        in_pos = False
        ce_strike = 0
        pe_strike = 0
        pos_entry_spot = 0.0
        pos_ce_entry = 0.0
        pos_pe_entry = 0.0
        pos_comb_entry = 0.0
        pos_comb_sl = 0.0
        pos_comb_target = 0.0
        pos_entry_time = None
        pos_bars = 0
        current_dte = self.days_to_expiry
        moved_to_be = False

        n = len(data)
        times = data.index.time
        eod_time = pd.to_datetime("15:15").time()

        for i in range(n):
            bar = data.iloc[i]
            bar_time = data.index[i]
            bar_date = bar['date']

            if bar_date != current_day:
                current_day = bar_date
                daily_trades = 0
                current_dte = self.days_to_expiry

            current_dte -= (1.0 / 375.0)

            # 1. Manage active Strangle position
            if in_pos:
                pos_bars += 1
                curr_spot = bar['close']
                T = max(current_dte, 0.01) / 365.0

                # Current prices of both legs
                curr_ce = black_scholes_price(curr_spot, ce_strike, T, config.RISK_FREE_RATE, self.iv_annual, "CE")
                curr_pe = black_scholes_price(curr_spot, pe_strike, T, config.RISK_FREE_RATE, self.iv_annual, "PE")
                curr_comb = curr_ce + curr_pe

                exit_now = False
                exit_reason = ""
                exit_ce = curr_ce
                exit_pe = curr_pe

                # Breakeven Ratchet: If combined gain >= breakeven_profit_pct (+18%), move SL to entry cost!
                if not moved_to_be and curr_comb >= (pos_comb_entry * (1 + self.breakeven_profit_pct)):
                    pos_comb_sl = pos_comb_entry * 1.01  # Lock cost + small buffer
                    moved_to_be = True

                # Target Hit
                if curr_comb >= pos_comb_target:
                    exit_now = True
                    exit_reason = "Target Hit"
                    exit_ce = curr_ce
                    exit_pe = curr_pe
                # Stop Loss Hit
                elif curr_comb <= pos_comb_sl:
                    exit_now = True
                    exit_reason = "Breakeven Guard" if moved_to_be else "StopLoss Hit"
                    exit_ce = curr_ce
                    exit_pe = curr_pe
                # EOD Square-Off at 15:15
                elif bar_time.time() >= eod_time:
                    exit_now = True
                    exit_reason = "EOD Square-Off"
                    exit_ce = curr_ce
                    exit_pe = curr_pe

                if exit_now:
                    # Apply 0.8% slippage on exit for both legs
                    realized_ce_exit = max(0.5, exit_ce * (1 - config.SLIPPAGE_PCT))
                    realized_pe_exit = max(0.5, exit_pe * (1 - config.SLIPPAGE_PCT))
                    realized_comb_exit = realized_ce_exit + realized_pe_exit

                    gross = (realized_comb_exit - pos_comb_entry) * self.lot_size
                    charges = self._calculate_strangle_charges(pos_ce_entry, pos_pe_entry, realized_ce_exit, realized_pe_exit)
                    net = round(gross - charges, 2)

                    spot_move = round(curr_spot - pos_entry_spot, 2)

                    trades.append(StrangleTradeRecord(
                        entry_time=pos_entry_time, exit_time=bar_time, entry_spot=round(pos_entry_spot, 2),
                        exit_spot=round(curr_spot, 2), spot_move_pts=spot_move, call_strike=ce_strike,
                        put_strike=pe_strike, ce_entry_prem=round(pos_ce_entry, 2), pe_entry_prem=round(pos_pe_entry, 2),
                        combined_entry_prem=round(pos_comb_entry, 2), ce_exit_prem=round(realized_ce_exit, 2),
                        pe_exit_prem=round(realized_pe_exit, 2), combined_exit_prem=round(realized_comb_exit, 2),
                        quantity=self.lot_size, gross_pnl=round(gross, 2), charges=charges, net_pnl=net,
                        exit_reason=exit_reason, bars_held=pos_bars
                    ))

                    equity += net
                    in_pos = False

            # 2. Check for New Strangle Entry Signal
            if not in_pos and daily_trades < 1:
                sig = bar.get('signal', 0)
                # Entry signal == 1 indicates mathematical volatility expansion conditions are met
                if sig == 1 and bar_time.time() < eod_time:
                    entry_spot = bar['close']
                    atm_strike = int(round(entry_spot / config.STRIKE_STEP) * config.STRIKE_STEP)

                    ce_strike = atm_strike + self.otm_distance_pts
                    pe_strike = atm_strike - self.otm_distance_pts

                    T = max(current_dte, 0.01) / 365.0
                    raw_ce = black_scholes_price(entry_spot, ce_strike, T, config.RISK_FREE_RATE, self.iv_annual, "CE")
                    raw_pe = black_scholes_price(entry_spot, pe_strike, T, config.RISK_FREE_RATE, self.iv_annual, "PE")

                    # Apply slippage on entry
                    pos_ce_entry = round(raw_ce * (1 + config.SLIPPAGE_PCT), 2)
                    pos_pe_entry = round(raw_pe * (1 + config.SLIPPAGE_PCT), 2)
                    pos_comb_entry = pos_ce_entry + pos_pe_entry

                    capital_needed = pos_comb_entry * self.lot_size
                    # Verify it fits in our INR 10,000 budget
                    if capital_needed <= self.initial_capital:
                        pos_comb_target = pos_comb_entry * (1 + self.combined_target_pct)
                        pos_comb_sl = pos_comb_entry * (1 - self.combined_sl_pct)

                        pos_entry_spot = entry_spot
                        pos_entry_time = bar_time
                        pos_bars = 0
                        moved_to_be = False
                        in_pos = True
                        daily_trades += 1

            equity_curve.append(equity)

        # 3. Summarize Performance Metrics
        if not trades:
            return {"total_trades": 0, "net_pnl_inr": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}

        df_t = pd.DataFrame([t.__dict__ for t in trades])
        wins = df_t[df_t['net_pnl'] > 0]
        losses = df_t[df_t['net_pnl'] <= 0]
        gain = wins['net_pnl'].sum() if not wins.empty else 0.0
        loss = abs(losses['net_pnl'].sum()) if not losses.empty else 0.0
        pf = round(gain / loss, 2) if loss > 0 else 99.0
        wr = round(len(wins) / len(df_t) * 100.0, 2)
        net = round(df_t['net_pnl'].sum(), 2)

        eq = pd.Series(equity_curve)
        peak = eq.cummax()
        dd = (peak - eq) / peak

        return {
            "total_trades": len(df_t),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": wr,
            "profit_factor": pf,
            "net_pnl_inr": net,
            "return_on_capital_pct": round((net / self.initial_capital) * 100.0, 2),
            "avg_win_inr": round(wins['net_pnl'].mean(), 2) if not wins.empty else 0.0,
            "avg_loss_inr": round(losses['net_pnl'].mean(), 2) if not losses.empty else 0.0,
            "reward_risk_ratio": round(abs(wins['net_pnl'].mean() / losses['net_pnl'].mean()), 2) if not losses.empty and not wins.empty else 0.0,
            "total_charges_inr": round(df_t['charges'].sum(), 2),
            "max_drawdown_inr": round((peak - eq).max(), 2),
            "max_drawdown_pct": round(dd.max() * 100.0, 2),
            "exit_reasons": df_t['exit_reason'].value_counts().to_dict(),
            "trades_df": df_t
        }
