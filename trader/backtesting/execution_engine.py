# -*- coding: utf-8 -*-
"""
Execution engine for trade entry and exit.

Handles realistic execution modeling including fees and slippage.
"""

from typing import Literal


class ExecutionEngine:
    """Handles trade execution with fees and slippage."""
    
    def __init__(self, fee_pct: float = 0.0, slippage_atr_mult: float = 0.0):
        """
        Initialize execution engine.
        
        Parameters
        ----------
        fee_pct : float, default 0.0
            Transaction fee percentage (e.g., 0.1 for 0.1%)
        slippage_atr_mult : float, default 0.0
            Slippage as ATR multiplier (e.g., 0.1 for 0.1x ATR)
        """
        self.fee_pct = fee_pct
        self.slippage_atr_mult = slippage_atr_mult
    
    def apply_slippage(self, price: float, direction: Literal['long', 'short'], 
                      atr_value: float) -> float:
        """
        Apply ATR-based slippage to price.
        
        Parameters
        ----------
        price : float
            Base price
        direction : 'long' or 'short'
            Trade direction
        atr_value : float
            Current ATR value
        
        Returns
        -------
        float
            Price with slippage applied
        """
        slippage = self.slippage_atr_mult * atr_value
        
        if direction == 'long':
            # Long: slippage increases entry price, decreases exit price
            return price + slippage
        else:  # short
            # Short: slippage decreases entry price, increases exit price
            return price - slippage
    
    def apply_fees(self, price: float, position_size: float) -> float:
        """
        Calculate fee impact on effective price.
        
        Parameters
        ----------
        price : float
            Base price
        position_size : float
            Position size
        
        Returns
        -------
        float
            Fee amount per unit (for entry or exit)
        """
        if self.fee_pct == 0.0:
            return 0.0
        
        # Fee is percentage of notional value
        fee_per_unit = price * (self.fee_pct / 100.0)
        return fee_per_unit
    
    def execute_entry(self, price: float, direction: Literal['long', 'short'],
                     atr_value: float) -> float:
        """
        Calculate effective entry price with slippage and fees.
        
        Parameters
        ----------
        price : float
            Base entry price
        direction : 'long' or 'short'
            Trade direction
        atr_value : float
            Current ATR value
        
        Returns
        -------
        float
            Effective entry price (including slippage and fees)
        """
        # Apply slippage
        price_with_slippage = self.apply_slippage(price, direction, atr_value)
        
        # Apply fees (fees increase effective entry price)
        fee_per_unit = self.apply_fees(price_with_slippage, 1.0)
        
        if direction == 'long':
            # Long: slippage and fees both increase entry price
            return price_with_slippage + fee_per_unit
        else:  # short
            # Short: slippage decreases entry, fees increase it (net effect depends)
            return price_with_slippage + fee_per_unit
    
    def execute_exit(self, price: float, direction: Literal['long', 'short'],
                    atr_value: float) -> float:
        """
        Calculate effective exit price with slippage and fees.
        
        Parameters
        ----------
        price : float
            Base exit price
        direction : 'long' or 'short'
            Trade direction
        atr_value : float
            Current ATR value
        
        Returns
        -------
        float
            Effective exit price (including slippage and fees)
        """
        # Apply slippage (opposite direction from entry)
        if direction == 'long':
            # Long exit: slippage decreases exit price
            slippage = self.slippage_atr_mult * atr_value
            price_with_slippage = price - slippage
        else:  # short
            # Short exit: slippage increases exit price
            slippage = self.slippage_atr_mult * atr_value
            price_with_slippage = price + slippage
        
        # Apply fees (fees decrease effective exit price)
        fee_per_unit = self.apply_fees(price_with_slippage, 1.0)
        
        return price_with_slippage - fee_per_unit
    
    def calculate_total_fees(self, entry_price: float, exit_price: float,
                            position_size: float) -> float:
        """
        Calculate total fees for a round-trip trade.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        position_size : float
            Position size
        
        Returns
        -------
        float
            Total fees paid (entry + exit)
        """
        entry_fee = entry_price * position_size * (self.fee_pct / 100.0)
        exit_fee = exit_price * position_size * (self.fee_pct / 100.0)
        return entry_fee + exit_fee






