# -*- coding: utf-8 -*-
"""
Position tracker for live trading.

Tracks active positions from Binance API and maps them to internal trade records.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from binance.client import Client
from binance.exceptions import BinanceAPIException


logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks active positions and updates PnL in real-time."""
    
    def __init__(self, client: Client, symbol: str):
        """
        Initialize position tracker.
        
        Parameters
        ----------
        client : Client
            Binance client instance
        symbol : str
            Trading pair symbol
        """
        self.client = client
        self.symbol = symbol
        
        # Track positions: trade_id -> position_info
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized PositionTracker for {symbol}")
    
    def _get_base_asset_balance(self) -> float:
        """
        Get base asset balance (e.g., BTC for BTCUSDT).
        
        Returns
        -------
        float
            Base asset balance
        """
        try:
            base_asset = self.symbol.replace("USDT", "").replace("USD", "")
            account = self.client.get_account()
            
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    return float(balance['free']) + float(balance['locked'])
            
            return 0.0
            
        except BinanceAPIException as e:
            logger.error(f"Error getting base asset balance: {e}")
            return 0.0
    
    def _get_current_price(self) -> float:
        """
        Get current price for symbol.
        
        Returns
        -------
        float
            Current price
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def register_position(self, trade_id: str, entry_price: float,
                        position_size: float, direction: str,
                        stop_loss_price: Optional[float] = None,
                        take_profit_price: Optional[float] = None,
                        entry_order_id: Optional[int] = None,
                        stop_loss_order_id: Optional[int] = None,
                        take_profit_order_id: Optional[int] = None):
        """
        Register a new position.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        entry_price : float
            Entry price
        position_size : float
            Position size (base asset quantity)
        direction : str
            Trade direction: "long" or "short"
        stop_loss_price : float or None, optional
            Stop loss price
        take_profit_price : float or None, optional
            Take profit price
        entry_order_id : int or None, optional
            Entry order ID
        stop_loss_order_id : int or None, optional
            Stop loss order ID
        take_profit_order_id : int or None, optional
            Take profit order ID
        """
        position_info = {
            'trade_id': trade_id,
            'symbol': self.symbol,
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'entry_order_id': entry_order_id,
            'stop_loss_order_id': stop_loss_order_id,
            'take_profit_order_id': take_profit_order_id,
            'current_price': entry_price,  # Will be updated
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0
        }
        
        self.positions[trade_id] = position_info
        
        logger.info(f"Registered position for trade {trade_id}: {direction} {position_size} @ {entry_price}")
        
        # Auto-save positions after registration (non-blocking)
        # This will be called periodically by LiveTrader, but we can also save here
    
    def update_position_price(self, trade_id: str, current_price: Optional[float] = None):
        """
        Update position with current market price and calculate unrealized PnL.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        current_price : float or None, optional
            Current price. If None, fetches from API.
        """
        if trade_id not in self.positions:
            logger.warning(f"Position for trade {trade_id} not found")
            return
        
        if current_price is None:
            current_price = self._get_current_price()
        
        position = self.positions[trade_id]
        position['current_price'] = current_price
        
        # Calculate unrealized PnL
        entry_price = position['entry_price']
        position_size = position['position_size']
        direction = position['direction']
        
        if direction == 'long':
            unrealized_pnl = (current_price - entry_price) * position_size
        else:  # short
            unrealized_pnl = (entry_price - current_price) * position_size
        
        position['unrealized_pnl'] = unrealized_pnl
        position['unrealized_pnl_pct'] = (unrealized_pnl / (entry_price * position_size)) * 100 if entry_price > 0 else 0.0
    
    def update_all_positions(self):
        """Update all positions with current market price."""
        current_price = self._get_current_price()
        
        for trade_id in self.positions:
            self.update_position_price(trade_id, current_price)
    
    def get_position(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get position information.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        
        Returns
        -------
        dict or None
            Position info, or None if not found
        """
        return self.positions.get(trade_id)
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all tracked positions.
        
        Returns
        -------
        list of dict
            List of position info dicts
        """
        return list(self.positions.values())
    
    def close_position(self, trade_id: str, exit_price: float, exit_reason: str):
        """
        Close a position and calculate realized PnL.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        exit_price : float
            Exit price
        exit_reason : str
            Exit reason (e.g., "stop_loss", "take_profit", "rsi_exit")
        
        Returns
        -------
        dict
            Trade record with PnL
        """
        if trade_id not in self.positions:
            logger.warning(f"Position for trade {trade_id} not found")
            return None
        
        position = self.positions[trade_id]
        
        entry_price = position['entry_price']
        position_size = position['position_size']
        direction = position['direction']
        
        # Calculate realized PnL
        if direction == 'long':
            realized_pnl = (exit_price - entry_price) * position_size
        else:  # short
            realized_pnl = (entry_price - exit_price) * position_size
        
        trade_record = {
            'trade_id': trade_id,
            'symbol': self.symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'direction': direction,
            'exit_reason': exit_reason,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': (realized_pnl / (entry_price * position_size)) * 100 if entry_price > 0 else 0.0
        }
        
        # Remove from tracking
        del self.positions[trade_id]
        
        logger.info(f"Closed position for trade {trade_id}: PnL = {realized_pnl:.2f}")
        
        return trade_record
    
    def sync_with_binance(self) -> List[Dict[str, Any]]:
        """
        Sync positions with Binance API.
        
        Reconstructs positions from account balance and open orders.
        
        Returns
        -------
        list of dict
            List of synced positions
        """
        try:
            base_asset_balance = self._get_base_asset_balance()
            current_price = self._get_current_price()
            
            # Get open orders
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            
            # Group orders by type
            stop_loss_orders = [o for o in open_orders if o['type'] == 'STOP_LOSS_LIMIT']
            take_profit_orders = [o for o in open_orders if o['type'] == 'LIMIT']
            
            # If we have a balance and orders, we likely have a position
            if base_asset_balance > 0:
                # Try to match with existing tracked positions
                # If no match, create a new position entry
                # Note: This is a simplified approach. In practice, you'd want to
                # match orders with positions more carefully.
                
                for trade_id, position in self.positions.items():
                    # Update existing position
                    self.update_position_price(trade_id, current_price)
            
            return list(self.positions.values())
            
        except BinanceAPIException as e:
            logger.error(f"Error syncing with Binance: {e}")
            return []
    
    def get_total_unrealized_pnl(self) -> float:
        """
        Get total unrealized PnL across all positions.
        
        Returns
        -------
        float
            Total unrealized PnL
        """
        self.update_all_positions()
        return sum(pos['unrealized_pnl'] for pos in self.positions.values())
    
    def get_total_exposure(self) -> float:
        """
        Get total exposure (position value) across all positions.
        
        Returns
        -------
        float
            Total exposure
        """
        self.update_all_positions()
        return sum(
            pos['current_price'] * pos['position_size']
            for pos in self.positions.values()
        )
    
    def save_positions(self, filepath: str):
        """
        Save positions to JSON file for persistence.
        
        Parameters
        ----------
        filepath : str
            Path to save positions file
        """
        try:
            # Update all positions before saving
            self.update_all_positions()
            
            # Convert positions to serializable format
            positions_data = {
                'symbol': self.symbol,
                'positions': self.positions
            }
            
            with open(filepath, 'w') as f:
                json.dump(positions_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.positions)} position(s) to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving positions to {filepath}: {e}")
    
    def load_positions(self, filepath: str):
        """
        Load positions from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to positions file
        """
        try:
            if not os.path.exists(filepath):
                logger.debug(f"Positions backup file {filepath} does not exist")
                return
            
            with open(filepath, 'r') as f:
                positions_data = json.load(f)
            
            # Verify symbol matches
            if positions_data.get('symbol') != self.symbol:
                logger.warning(f"Symbol mismatch in backup file: {positions_data.get('symbol')} != {self.symbol}")
                return
            
            # Restore positions
            self.positions = positions_data.get('positions', {})
            
            # Convert string keys back to proper types if needed
            for trade_id, position in self.positions.items():
                # Ensure numeric fields are floats
                for key in ['entry_price', 'position_size', 'current_price', 
                           'stop_loss_price', 'take_profit_price', 
                           'unrealized_pnl', 'unrealized_pnl_pct']:
                    if key in position and position[key] is not None:
                        position[key] = float(position[key])
            
            logger.info(f"Loaded {len(self.positions)} position(s) from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading positions from {filepath}: {e}")




