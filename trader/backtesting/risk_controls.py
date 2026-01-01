# -*- coding: utf-8 -*-
"""
Risk controls module.

Handles cooldowns, daily risk limits, trade limits, and loss limits.
"""

from typing import Dict, Optional
from datetime import datetime, date


class RiskController:
    """Manages risk controls including cooldowns and daily limits."""
    
    def __init__(self, per_zone_cooldown_candles: int = 0,
                 max_daily_risk_pct: Optional[float] = None,
                 max_daily_trades: Optional[int] = None,
                 max_daily_loss_pct: Optional[float] = None):
        """
        Initialize risk controller.
        
        Parameters
        ----------
        per_zone_cooldown_candles : int, default 0
            Cooldown candles after exiting trade from zone
        max_daily_risk_pct : float or None, default None
            Max daily risk as % of equity (None = no limit)
        max_daily_trades : int or None, default None
            Max trades per day (None = no limit)
        max_daily_loss_pct : float or None, default None
            Max daily loss as % of equity (None = no limit)
        """
        self.per_zone_cooldown_candles = per_zone_cooldown_candles
        self.max_daily_risk_pct = max_daily_risk_pct
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        
        # Trackers
        self.last_trade_exit_by_zone: Dict[int, int] = {}  # zone_index -> exit_index
        self.daily_risk_tracker: Dict[date, float] = {}  # date -> total risk
        self.daily_trade_tracker: Dict[date, int] = {}  # date -> trade count
        self.daily_pnl_tracker: Dict[date, float] = {}  # date -> total PnL
    
    def check_zone_cooldown(self, zone_index: int, current_index: int) -> bool:
        """
        Check if zone is in cooldown period.
        
        Parameters
        ----------
        zone_index : int
            Zone index to check
        current_index : int
            Current candle index
        
        Returns
        -------
        bool
            True if zone is available (not in cooldown), False if in cooldown
        """
        if self.per_zone_cooldown_candles == 0:
            return True
        
        if zone_index not in self.last_trade_exit_by_zone:
            return True
        
        last_exit_index = self.last_trade_exit_by_zone[zone_index]
        candles_since_exit = current_index - last_exit_index
        
        return candles_since_exit >= self.per_zone_cooldown_candles
    
    def check_daily_risk_limit(self, risk_amount: float, account_equity: float,
                               current_date: date) -> bool:
        """
        Check if daily risk limit would be exceeded.
        
        Parameters
        ----------
        risk_amount : float
            Risk amount for this trade
        account_equity : float
            Current account equity
        current_date : date
            Current date
        
        Returns
        -------
        bool
            True if trade allowed, False if limit exceeded
        """
        if self.max_daily_risk_pct is None:
            return True
        
        # Reset tracker if new day
        if current_date not in self.daily_risk_tracker:
            self.daily_risk_tracker[current_date] = 0.0
        
        current_daily_risk = self.daily_risk_tracker[current_date]
        max_daily_risk = account_equity * (self.max_daily_risk_pct / 100.0)
        
        return (current_daily_risk + risk_amount) <= max_daily_risk
    
    def check_daily_trade_limit(self, current_date: date) -> bool:
        """
        Check if daily trade limit would be exceeded.
        
        Parameters
        ----------
        current_date : date
            Current date
        
        Returns
        -------
        bool
            True if trade allowed, False if limit exceeded
        """
        if self.max_daily_trades is None:
            return True
        
        # Reset tracker if new day
        if current_date not in self.daily_trade_tracker:
            self.daily_trade_tracker[current_date] = 0
        
        current_trades = self.daily_trade_tracker[current_date]
        
        return current_trades < self.max_daily_trades
    
    def check_daily_loss_limit(self, account_equity: float, initial_equity: float,
                               current_date: date) -> bool:
        """
        Check if daily loss limit would be exceeded.
        
        Parameters
        ----------
        account_equity : float
            Current account equity
        initial_equity : float
            Equity at start of day
        current_date : date
            Current date
        
        Returns
        -------
        bool
            True if trading allowed, False if loss limit exceeded
        """
        if self.max_daily_loss_pct is None:
            return True
        
        # Get initial equity for this day (or use current if first trade)
        if current_date not in self.daily_pnl_tracker:
            # First trade of the day, use current equity as baseline
            return True
        
        daily_pnl = self.daily_pnl_tracker[current_date]
        max_daily_loss = initial_equity * (self.max_daily_loss_pct / 100.0)
        
        # Check if current loss exceeds limit
        return daily_pnl >= -max_daily_loss
    
    def update_daily_trackers(self, zone_index: int, risk_amount: float,
                             pnl: float, current_date: date, current_index: int):
        """
        Update all daily trackers after trade execution.
        
        Parameters
        ----------
        zone_index : int
            Zone index of the trade
        risk_amount : float
            Risk amount for the trade
        pnl : float
            PnL from the trade
        current_date : date
            Current date
        current_index : int
            Current candle index
        """
        # Update zone cooldown tracker
        self.last_trade_exit_by_zone[zone_index] = current_index
        
        # Update daily risk tracker
        if current_date not in self.daily_risk_tracker:
            self.daily_risk_tracker[current_date] = 0.0
        self.daily_risk_tracker[current_date] += risk_amount
        
        # Update daily trade tracker
        if current_date not in self.daily_trade_tracker:
            self.daily_trade_tracker[current_date] = 0
        self.daily_trade_tracker[current_date] += 1
        
        # Update daily PnL tracker
        if current_date not in self.daily_pnl_tracker:
            self.daily_pnl_tracker[current_date] = 0.0
        self.daily_pnl_tracker[current_date] += pnl
    
    def reset(self):
        """Reset all trackers (useful for new simulations)."""
        self.last_trade_exit_by_zone = {}
        self.daily_risk_tracker = {}
        self.daily_trade_tracker = {}
        self.daily_pnl_tracker = {}






