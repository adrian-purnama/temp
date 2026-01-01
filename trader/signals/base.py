# -*- coding: utf-8 -*-
"""
Base signal interface and standardized signal format.

Defines the contract that all signal modules must follow, ensuring consistency
across different signal types (SR, RSI, MA, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


# Standardized signal format structure
STANDARD_SIGNAL_FORMAT = {
    'direction': int,      # -1 (short), 0 (flat), 1 (long)
    'strength': float,    # 0.0 to 1.0 (confidence/strength)
    'timestamp': Any,     # Timestamp of signal (datetime, int, etc.)
    'index': int,         # Index position in data
    'metadata': dict,     # Strategy-specific metadata
    'source': str         # Signal module name (e.g., 'sr', 'rsi')
}


class BaseSignal(ABC):
    """
    Abstract base class for all signal modules.
    
    Signal modules are pure functions that consume market data and output
    standardized trading signals. They have no knowledge of positions, money,
    stop-loss, take-profit, or account balance.
    
    Each signal module may maintain internal state (e.g., waiting states,
    indicators) but this state is managed internally and passed between calls.
    """
    
    def __init__(self, name: str):
        """
        Initialize signal module.
        
        Parameters
        ----------
        name : str
            Name of the signal module (e.g., 'sr', 'rsi', 'ma').
        """
        self.name = name
    
    @abstractmethod
    def generate_signal(self, candle: pd.Series, context: pd.DataFrame, 
                       state: Optional[Dict[str, Any]] = None, 
                       **kwargs) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Generate standardized signal from market data.
        
        This is the main interface method that all signal modules must implement.
        It consumes market data and optional state, and returns a standardized
        signal along with updated state.
        
        Parameters
        ----------
        candle : pd.Series
            Current candle being analyzed. Must have columns: open, high, low, close.
        context : pd.DataFrame
            Historical context window (ending at t-1). Used for indicators/analysis.
        state : dict or None, optional
            Internal state from previous call. Signal modules may maintain state
            (e.g., waiting states, indicator buffers). If None, initializes new state.
        **kwargs
            Additional parameters specific to the signal module.
        
        Returns
        -------
        tuple of (dict, dict or None)
            - signal: Standardized signal dictionary with keys:
                * direction: int (-1 for short, 0 for flat, 1 for long)
                * strength: float (0.0 to 1.0, confidence/strength)
                * timestamp: timestamp of signal
                * index: int (index position in data)
                * metadata: dict (strategy-specific metadata)
                * source: str (signal module name)
            - updated_state: Updated internal state (or None if stateless)
        
        Notes
        -----
        Signal Format:
        - direction: -1 (short), 0 (flat), 1 (long)
        - strength: 0.0 (weak/no signal) to 1.0 (strong signal)
        - metadata: Can contain any strategy-specific information (e.g., entry_price,
          zone_info, indicator values) but must not contain trading logic
        
        State Management:
        - State is passed in and returned, ensuring pure function behavior
        - State should be serializable (dict-based) for persistence/debugging
        - If signal module is stateless, return None for updated_state
        """
        pass
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate that signal conforms to standard format.
        
        Parameters
        ----------
        signal : dict
            Signal dictionary to validate.
        
        Returns
        -------
        bool
            True if signal is valid, False otherwise.
        
        Raises
        ------
        ValueError
            If signal is invalid (missing required keys, wrong types, etc.)
        """
        # Check required keys
        required_keys = ['direction', 'strength', 'timestamp', 'index', 'metadata', 'source']
        missing_keys = [key for key in required_keys if key not in signal]
        if missing_keys:
            raise ValueError(f"Signal missing required keys: {missing_keys}")
        
        # Validate direction
        if signal['direction'] not in [-1, 0, 1]:
            raise ValueError(f"Invalid direction: {signal['direction']}. Must be -1, 0, or 1.")
        
        # Validate strength
        if not isinstance(signal['strength'], (int, float)):
            raise ValueError(f"Invalid strength type: {type(signal['strength'])}. Must be numeric.")
        if signal['strength'] < 0.0 or signal['strength'] > 1.0:
            raise ValueError(f"Invalid strength value: {signal['strength']}. Must be between 0.0 and 1.0.")
        
        # Validate index
        if not isinstance(signal['index'], (int, pd.api.types.is_integer)):
            raise ValueError(f"Invalid index type: {type(signal['index'])}. Must be integer.")
        
        # Validate metadata
        if not isinstance(signal['metadata'], dict):
            raise ValueError(f"Invalid metadata type: {type(signal['metadata'])}. Must be dict.")
        
        # Validate source
        if not isinstance(signal['source'], str):
            raise ValueError(f"Invalid source type: {type(signal['source'])}. Must be string.")
        
        return True
    
    def create_flat_signal(self, timestamp: Any, index: int, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized flat (no signal) signal.
        
        Convenience method for creating a flat signal when no trading opportunity exists.
        
        Parameters
        ----------
        timestamp : Any
            Timestamp of the signal.
        index : int
            Index position in data.
        metadata : dict or None, optional
            Optional metadata to include in signal.
        
        Returns
        -------
        dict
            Standardized flat signal dictionary.
        """
        return {
            'direction': 0,
            'strength': 0.0,
            'timestamp': timestamp,
            'index': index,
            'metadata': metadata or {},
            'source': self.name
        }

