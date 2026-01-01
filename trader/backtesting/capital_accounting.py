# -*- coding: utf-8 -*-
"""
Capital accounting module.

Tracks equity, unrealized PnL, and equity history for accurate drawdown calculation.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class CapitalAccountant:
    """Manages capital accounting and equity tracking."""
    
    def __init__(self, enable_unrealized_pnl: bool = False):
        """
        Initialize capital accountant.
        
        Parameters
        ----------
        enable_unrealized_pnl : bool, default False
            Whether to track unrealized PnL for equity calculation
        """
        self.enable_unrealized_pnl = enable_unrealized_pnl
        self.equity_history: List[Tuple[datetime, float, float, float]] = []  # (timestamp, balance, equity, unrealized_pnl)
    
    def calculate_unrealized_pnl(self, active_trade: Optional[Dict[str, Any]],
                                 current_price: float) -> float:
        """
        Calculate unrealized PnL for active position.
        
        Parameters
        ----------
        active_trade : dict or None
            Active trade record with entry_price, position_size, direction
        current_price : float
            Current market price
        
        Returns
        -------
        float
            Unrealized PnL (0.0 if no active trade)
        """
        if active_trade is None:
            return 0.0
        
        entry_price = active_trade['entry_price']
        position_size = active_trade['position_size']
        direction = active_trade['direction']
        
        if direction == 'long':
            unrealized_pnl = (current_price - entry_price) * position_size
        else:  # short
            unrealized_pnl = (entry_price - current_price) * position_size
        
        return unrealized_pnl
    
    def calculate_equity(self, account_balance: float,
                        active_trade: Optional[Dict[str, Any]],
                        current_price: float) -> float:
        """
        Calculate total equity including unrealized PnL.
        
        Parameters
        ----------
        account_balance : float
            Current account balance (realized PnL only)
        active_trade : dict or None
            Active trade record
        current_price : float
            Current market price
        
        Returns
        -------
        float
            Total equity (balance + unrealized PnL if enabled)
        """
        if not self.enable_unrealized_pnl or active_trade is None:
            return account_balance
        
        unrealized_pnl = self.calculate_unrealized_pnl(active_trade, current_price)
        return account_balance + unrealized_pnl
    
    def update_equity_history(self, timestamp: datetime, balance: float,
                             equity: float, unrealized_pnl: float):
        """
        Update equity history for drawdown calculation.
        
        Parameters
        ----------
        timestamp : datetime
            Timestamp of the update
        balance : float
            Account balance
        equity : float
            Total equity
        unrealized_pnl : float
            Unrealized PnL
        """
        self.equity_history.append((timestamp, balance, equity, unrealized_pnl))
    
    def get_equity_series(self) -> List[float]:
        """
        Get equity values as a list.
        
        Returns
        -------
        list of float
            Equity values over time
        """
        return [eq[2] for eq in self.equity_history]
    
    def reset(self):
        """Reset equity history (useful for new simulations)."""
        self.equity_history = []






