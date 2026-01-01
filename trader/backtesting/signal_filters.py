# -*- coding: utf-8 -*-
"""
Signal filters module.

Provides RSI-based filtering for entry and exit signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Literal


class SignalFilter:
    """Filters signals based on RSI conditions."""
    
    def __init__(self, rsi_period: int = 14,
                 rsi_oversold_threshold: float = 30.0,
                 rsi_overbought_threshold: float = 70.0,
                 rsi_lookback_candles: int = 5):
        """
        Initialize signal filter.
        
        Parameters
        ----------
        rsi_period : int, default 14
            RSI calculation period
        rsi_oversold_threshold : float, default 30.0
            RSI oversold threshold
        rsi_overbought_threshold : float, default 70.0
            RSI overbought threshold
        rsi_lookback_candles : int, default 5
            Lookback candles for RSI recent touch check
        """
        self.rsi_period = rsi_period
        self.rsi_oversold_threshold = rsi_oversold_threshold
        self.rsi_overbought_threshold = rsi_overbought_threshold
        self.rsi_lookback_candles = rsi_lookback_candles
    
    def calculate_rsi(self, context_df: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI for context window.
        
        Parameters
        ----------
        context_df : pd.DataFrame
            Context window with 'close' column
        
        Returns
        -------
        pd.Series
            RSI values indexed by timestamp
        """
        if len(context_df) < self.rsi_period + 1:
            # Not enough data, return NaN series
            return pd.Series(index=context_df.index, dtype=float)
        
        close = context_df['close']
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calculate average gain and loss using Wilder's smoothing
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_rsi_entry_filter(self, confirmed_setup: Dict[str, Any],
                               context_df: pd.DataFrame) -> bool:
        """
        Check if entry is allowed based on RSI conditions.
        
        For rebound setups, entry is allowed if:
        - Current RSI < oversold threshold, OR
        - Any of past lookback candles had RSI <= oversold threshold
        
        Parameters
        ----------
        confirmed_setup : dict
            Confirmed setup event with 'event_type' and 'direction'
        context_df : pd.DataFrame
            Context window with 'close' column
        
        Returns
        -------
        bool
            True if entry allowed, False otherwise
        """
        # Only apply filter to rebound setups
        event_type = confirmed_setup.get('event_type', '')
        if 'rebound' not in event_type:
            # Not a rebound setup, allow entry
            return True
        
        # Calculate RSI
        rsi_series = self.calculate_rsi(context_df)
        
        if len(rsi_series) == 0 or rsi_series.isna().all():
            # Can't calculate RSI, allow entry (fail open)
            return True
        
        # Get current RSI (last value)
        current_rsi = rsi_series.iloc[-1]
        
        # Check current RSI condition
        if pd.notna(current_rsi) and current_rsi < self.rsi_oversold_threshold:
            return True
        
        # Check recent touch condition (past lookback candles)
        if len(rsi_series) >= self.rsi_lookback_candles:
            recent_rsi = rsi_series.iloc[-self.rsi_lookback_candles:]
            if (recent_rsi <= self.rsi_oversold_threshold).any():
                return True
        
        # Conditions not met
        return False
    
    def check_rsi_exit_filter(self, active_trade: Dict[str, Any],
                             current_candle: pd.Series,
                             context_df: pd.DataFrame) -> bool:
        """
        Check if early exit should be triggered based on RSI.
        
        For long positions: exit if RSI > overbought threshold
        For short positions: exit if RSI < oversold threshold
        
        Parameters
        ----------
        active_trade : dict
            Active trade record with 'direction'
        current_candle : pd.Series
            Current candle
        context_df : pd.DataFrame
            Context window with 'close' column
        
        Returns
        -------
        bool
            True if RSI exit should be triggered, False otherwise
        """
        # Calculate RSI
        rsi_series = self.calculate_rsi(context_df)
        
        if len(rsi_series) == 0 or rsi_series.isna().all():
            # Can't calculate RSI, don't exit (fail closed)
            return False
        
        # Get current RSI (last value)
        current_rsi = rsi_series.iloc[-1]
        
        if pd.isna(current_rsi):
            return False
        
        direction = active_trade['direction']
        
        if direction == 'long':
            # Long: exit if RSI > overbought threshold
            return current_rsi > self.rsi_overbought_threshold
        else:  # short
            # Short: exit if RSI < oversold threshold
            return current_rsi < self.rsi_oversold_threshold
    
    def filter_entry_signal(self, confirmed_setup: Optional[Dict[str, Any]],
                           context_df: pd.DataFrame) -> bool:
        """
        Apply RSI entry filter to confirmed setup.
        
        Parameters
        ----------
        confirmed_setup : dict or None
            Confirmed setup event
        context_df : pd.DataFrame
            Context window
        
        Returns
        -------
        bool
            True if entry allowed, False if filtered out
        """
        if confirmed_setup is None:
            return False
        
        return self.check_rsi_entry_filter(confirmed_setup, context_df)






