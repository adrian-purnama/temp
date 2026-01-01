# -*- coding: utf-8 -*-
"""
Backtesting configuration schema.

Defines all configurable parameters for the backtesting engine.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    
    # Execution Parameters
    execution_mode: str = "next_open"  # "next_open" or "next_close"
    fee_pct: float = 0.0  # Transaction fee percentage (e.g., 0.1 for 0.1%)
    slippage_atr_mult: float = 0.0  # Slippage as ATR multiplier (e.g., 0.1 for 0.1x ATR)
    
    # Position Sizing
    risk_per_trade: float = 0.01  # Risk per trade as fraction (0.01 = 1%)
    max_position_value_pct: Optional[float] = None  # Max position value as % of equity (None = no cap)
    
    # Volatility Capping
    atr_percentile_window: int = 100  # Window for rolling ATR percentile
    atr_percentile_threshold: float = 95.0  # Percentile threshold (e.g., 95.0 for 95th percentile)
    cap_stop_loss: bool = True  # Cap stop loss distances using ATR percentile
    cap_take_profit: bool = True  # Cap take profit distances using ATR percentile
    
    # Risk Controls
    per_zone_cooldown_candles: int = 0  # Cooldown candles after exiting trade from zone
    max_daily_risk_pct: Optional[float] = None  # Max daily risk as % of equity (None = no limit)
    max_daily_trades: Optional[int] = None  # Max trades per day (None = no limit)
    max_daily_loss_pct: Optional[float] = None  # Max daily loss as % of equity (None = no limit)
    
    # Capital Accounting
    enable_unrealized_pnl: bool = False  # Track unrealized PnL for equity calculation
    
    # Signal Filters
    enable_rsi_filter: bool = False  # Enable RSI-based entry/exit filtering
    rsi_period: int = 14  # RSI calculation period
    rsi_oversold_threshold: float = 30.0  # RSI oversold threshold
    rsi_overbought_threshold: float = 70.0  # RSI overbought threshold
    rsi_lookback_candles: int = 5  # Lookback candles for RSI recent touch check
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'execution_mode': self.execution_mode,
            'fee_pct': self.fee_pct,
            'slippage_atr_mult': self.slippage_atr_mult,
            'risk_per_trade': self.risk_per_trade,
            'max_position_value_pct': self.max_position_value_pct,
            'atr_percentile_window': self.atr_percentile_window,
            'atr_percentile_threshold': self.atr_percentile_threshold,
            'cap_stop_loss': self.cap_stop_loss,
            'cap_take_profit': self.cap_take_profit,
            'per_zone_cooldown_candles': self.per_zone_cooldown_candles,
            'max_daily_risk_pct': self.max_daily_risk_pct,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'enable_unrealized_pnl': self.enable_unrealized_pnl,
            'enable_rsi_filter': self.enable_rsi_filter,
            'rsi_period': self.rsi_period,
            'rsi_oversold_threshold': self.rsi_oversold_threshold,
            'rsi_overbought_threshold': self.rsi_overbought_threshold,
            'rsi_lookback_candles': self.rsi_lookback_candles
        }






