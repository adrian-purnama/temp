# -*- coding: utf-8 -*-
"""
State recovery for live trading.

Queries Binance API on startup to reconstruct internal trade state from
open positions and orders.
"""

import logging
from typing import Dict, Any, Optional, List
from binance.client import Client
from binance.exceptions import BinanceAPIException


logger = logging.getLogger(__name__)


class StateRecovery:
    """Recovers internal trade state from Binance API."""
    
    def __init__(self, client: Client, symbol: str):
        """
        Initialize state recovery.
        
        Parameters
        ----------
        client : Client
            Binance client instance
        symbol : str
            Trading pair symbol
        """
        self.client = client
        self.symbol = symbol
        
        logger.info(f"Initialized StateRecovery for {symbol}")
    
    def recover_positions(self) -> List[Dict[str, Any]]:
        """
        Recover open positions from Binance API.
        
        Returns
        -------
        list of dict
            List of recovered position info dicts
        """
        try:
            # Get account balance for base asset
            base_asset = self.symbol.replace("USDT", "").replace("USD", "")
            try:
                account = self.client.get_account()
            except BinanceAPIException as e:
                if e.code == -1022:  # Invalid signature
                    logger.warning("Signature error when recovering positions - API keys may be invalid or permissions insufficient")
                raise
            
            base_balance = 0.0
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    base_balance = float(balance['free']) + float(balance['locked'])
                    break
            
            if base_balance == 0:
                logger.info("No open positions found")
                return []
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Get open orders to find stop-loss and take-profit levels
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            
            stop_loss_orders = [
                o for o in open_orders
                if o['type'] == 'STOP_LOSS_LIMIT' and o['side'] == 'SELL'
            ]
            
            take_profit_orders = [
                o for o in open_orders
                if o['type'] == 'LIMIT' and o['side'] == 'SELL'
            ]
            
            positions = []
            
            # Create position entry
            # Note: We don't have the original entry price, so we'll use current price as estimate
            # In practice, you'd want to store this in a database or log file
            position_info = {
                'symbol': self.symbol,
                'base_asset': base_asset,
                'quantity': base_balance,
                'current_price': current_price,
                'estimated_entry_price': current_price,  # Best guess
                'direction': 'long',  # Assuming long for spot
                'stop_loss_price': float(stop_loss_orders[0]['stopPrice']) if stop_loss_orders else None,
                'take_profit_price': float(take_profit_orders[0]['price']) if take_profit_orders else None,
                'stop_loss_order_id': stop_loss_orders[0]['orderId'] if stop_loss_orders else None,
                'take_profit_order_id': take_profit_orders[0]['orderId'] if take_profit_orders else None
            }
            
            positions.append(position_info)
            
            logger.info(f"Recovered {len(positions)} position(s)")
            return positions
            
        except BinanceAPIException as e:
            logger.error(f"Error recovering positions: {e}")
            return []
    
    def recover_orders(self) -> List[Dict[str, Any]]:
        """
        Recover open orders from Binance API.
        
        Returns
        -------
        list of dict
            List of recovered order info dicts
        """
        try:
            try:
                open_orders = self.client.get_open_orders(symbol=self.symbol)
            except BinanceAPIException as e:
                if e.code == -1022:  # Invalid signature
                    logger.warning("Signature error when recovering orders - API keys may be invalid or permissions insufficient")
                raise
            
            recovered_orders = []
            for order in open_orders:
                order_info = {
                    'order_id': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'status': order['status'],
                    'quantity': float(order['origQty']),
                    'price': float(order['price']) if order.get('price') else None,
                    'stop_price': float(order['stopPrice']) if order.get('stopPrice') else None,
                    'executed_qty': float(order['executedQty']),
                    'time': order['time']
                }
                recovered_orders.append(order_info)
            
            logger.info(f"Recovered {len(recovered_orders)} open order(s)")
            return recovered_orders
            
        except BinanceAPIException as e:
            logger.error(f"Error recovering orders: {e}")
            return []
    
    def recover_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Recover recent trade history from Binance API.
        
        Parameters
        ----------
        limit : int, default 50
            Number of recent trades to retrieve
        
        Returns
        -------
        list of dict
            List of recent trade info dicts
        """
        try:
            try:
                trades = self.client.get_my_trades(symbol=self.symbol, limit=limit)
            except BinanceAPIException as e:
                if e.code == -1022:  # Invalid signature
                    logger.warning("Signature error when recovering trade history - API keys may be invalid or permissions insufficient")
                raise
            
            trade_history = []
            for trade in trades:
                trade_info = {
                    'trade_id': trade['id'],
                    'symbol': trade['symbol'],
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'quote_qty': float(trade['quoteQty']),
                    'commission': float(trade['commission']),
                    'commission_asset': trade['commissionAsset'],
                    'time': trade['time'],
                    'is_buyer': trade['isBuyer'],
                    'is_maker': trade['isMaker']
                }
                trade_history.append(trade_info)
            
            logger.info(f"Recovered {len(trade_history)} recent trade(s)")
            return trade_history
            
        except BinanceAPIException as e:
            logger.error(f"Error recovering trade history: {e}")
            return []
    
    def recover_full_state(self) -> Dict[str, Any]:
        """
        Recover full trading state from Binance API.
        
        Returns
        -------
        dict
            Recovered state containing positions, orders, and trade history
        """
        logger.info("Starting state recovery...")
        
        state = {
            'positions': self.recover_positions(),
            'orders': self.recover_orders(),
            'trade_history': self.recover_trade_history()
        }
        
        logger.info("State recovery complete")
        return state
    
    def validate_state_consistency(self, internal_positions: List[Dict[str, Any]],
                                   internal_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate consistency between internal state and Binance state.
        
        Parameters
        ----------
        internal_positions : list of dict
            Internal position tracking
        internal_orders : list of dict
            Internal order tracking
        
        Returns
        -------
        dict
            Validation results with any mismatches
        """
        binance_state = self.recover_full_state()
        
        validation = {
            'consistent': True,
            'mismatches': []
        }
        
        # Check positions
        binance_position_count = len(binance_state['positions'])
        internal_position_count = len(internal_positions)
        
        if binance_position_count != internal_position_count:
            validation['consistent'] = False
            validation['mismatches'].append({
                'type': 'position_count',
                'binance': binance_position_count,
                'internal': internal_position_count
            })
        
        # Check orders
        binance_order_ids = {o['order_id'] for o in binance_state['orders']}
        internal_order_ids = {o['order_id'] for o in internal_orders}
        
        missing_in_internal = binance_order_ids - internal_order_ids
        extra_in_internal = internal_order_ids - binance_order_ids
        
        if missing_in_internal or extra_in_internal:
            validation['consistent'] = False
            validation['mismatches'].append({
                'type': 'order_ids',
                'missing_in_internal': list(missing_in_internal),
                'extra_in_internal': list(extra_in_internal)
            })
        
        if validation['consistent']:
            logger.info("State validation: consistent")
        else:
            logger.warning(f"State validation: inconsistencies found: {validation['mismatches']}")
        
        return validation



