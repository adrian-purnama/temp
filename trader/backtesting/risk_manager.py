# -*- coding: utf-8 -*-
"""
Risk management module.

Handles position sizing, position caps, and volatility capping.
"""

import numpy as np
import pandas as pd
from typing import Optional


class RiskManager:
    """Manages position sizing, caps, and volatility adjustments."""
    
    def __init__(self, max_position_value_pct: Optional[float] = None,
                 atr_percentile_window: int = 100,
                 atr_percentile_threshold: float = 95.0,
                 cap_stop_loss: bool = True,
                 cap_take_profit: bool = True):
        """
        Initialize risk manager.
        
        Parameters
        ----------
        max_position_value_pct : float or None, default None
            Maximum position value as % of equity (None = no cap)
        atr_percentile_window : int, default 100
            Window for rolling ATR percentile calculation
        atr_percentile_threshold : float, default 95.0
            Percentile threshold for ATR capping (e.g., 95.0 for 95th percentile)
        cap_stop_loss : bool, default True
            Whether to cap stop loss distances using ATR percentile
        cap_take_profit : bool, default True
            Whether to cap take profit distances using ATR percentile
        """
        self.max_position_value_pct = max_position_value_pct
        self.atr_percentile_window = atr_percentile_window
        self.atr_percentile_threshold = atr_percentile_threshold
        self.cap_stop_loss = cap_stop_loss
        self.cap_take_profit = cap_take_profit
        self._atr_history = []  # Track ATR history for percentile calculation
    
    def cap_atr_value(self, atr_value: float) -> float:
        """
        Cap ATR value using rolling percentile to prevent extreme volatility.
        
        Parameters
        ----------
        atr_value : float
            Current ATR value
        
        Returns
        -------
        float
            Capped ATR value (original if below threshold)
        """
        if len(self._atr_history) < self.atr_percentile_window:
            # Not enough history yet, just track and return original
            self._atr_history.append(atr_value)
            return atr_value
        
        # Calculate percentile threshold
        percentile_value = np.percentile(self._atr_history, self.atr_percentile_threshold)
        
        # Cap ATR if it exceeds percentile threshold
        capped_atr = min(atr_value, percentile_value)
        
        # Update history (rolling window)
        self._atr_history.append(atr_value)
        if len(self._atr_history) > self.atr_percentile_window:
            self._atr_history.pop(0)
        
        return capped_atr
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                               risk_amount: float, account_equity: float) -> float:
        """
        Calculate position size based on risk amount and position cap.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        stop_loss_price : float
            Stop loss price
        risk_amount : float
            Risk amount in currency
        account_equity : float
            Current account equity
        
        Returns
        -------
        float
            Position size (capped by max position value if configured)
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0.0
        
        # Calculate position size based on risk
        risk_based_size = risk_amount / risk_per_unit
        
        # Apply position cap if configured
        if self.max_position_value_pct is not None:
            max_position_value = account_equity * (self.max_position_value_pct / 100.0)
            max_size = max_position_value / entry_price
            return min(risk_based_size, max_size)
        
        return risk_based_size
    
    def apply_position_cap(self, position_size: float, entry_price: float,
                          account_equity: float) -> float:
        """
        Apply position value cap to position size.
        
        Parameters
        ----------
        position_size : float
            Unconstrained position size
        entry_price : float
            Entry price
        account_equity : float
            Current account equity
        
        Returns
        -------
        float
            Capped position size
        """
        if self.max_position_value_pct is None:
            return position_size
        
        max_position_value = account_equity * (self.max_position_value_pct / 100.0)
        max_size = max_position_value / entry_price
        
        return min(position_size, max_size)
    
    def reset_atr_history(self):
        """Reset ATR history (useful for new simulations)."""
        self._atr_history = []






