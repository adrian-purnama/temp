# -*- coding: utf-8 -*-
"""
Order manager for tracking order lifecycle.

Manages order status updates, error handling, and order state tracking.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from binance.client import Client
from binance.exceptions import BinanceAPIException


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderManager:
    """Manages order lifecycle and status tracking."""
    
    def __init__(self, client: Client, symbol: str,
                 on_order_update: Optional[Callable] = None):
        """
        Initialize order manager.
        
        Parameters
        ----------
        client : Client
            Binance client instance
        symbol : str
            Trading pair symbol
        on_order_update : callable or None, optional
            Callback function called when order status updates
        """
        self.client = client
        self.symbol = symbol
        self.on_order_update = on_order_update
        
        # Track orders: order_id -> order_info
        self.orders: Dict[int, Dict[str, Any]] = {}
        
        # Track order groups (e.g., entry + stop-loss + take-profit)
        self.order_groups: Dict[str, List[int]] = {}  # trade_id -> [order_ids]
        
        logger.info(f"Initialized OrderManager for {symbol}")
    
    def _map_binance_status(self, status: str) -> OrderStatus:
        """
        Map Binance order status to internal OrderStatus.
        
        Parameters
        ----------
        status : str
            Binance order status
        
        Returns
        -------
        OrderStatus
            Internal order status
        """
        status_map = {
            'NEW': OrderStatus.NEW,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELED,
            'PENDING_CANCEL': OrderStatus.PENDING_CANCEL,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return status_map.get(status, OrderStatus.NEW)
    
    def register_order(self, order_response: Dict[str, Any],
                      trade_id: Optional[str] = None,
                      order_type: Optional[str] = None) -> int:
        """
        Register a new order for tracking.
        
        Parameters
        ----------
        order_response : dict
            Order response from Binance API
        trade_id : str or None, optional
            Associated trade ID
        order_type : str or None, optional
            Order type (e.g., "entry", "stop_loss", "take_profit")
        
        Returns
        -------
        int
            Order ID
        """
        order_id = order_response['orderId']
        
        order_info = {
            'order_id': order_id,
            'symbol': order_response.get('symbol', self.symbol),
            'side': order_response.get('side'),
            'type': order_response.get('type'),
            'status': self._map_binance_status(order_response.get('status', 'NEW')),
            'quantity': float(order_response.get('origQty', 0)),
            'price': float(order_response.get('price', 0)) if order_response.get('price') else None,
            'executed_qty': float(order_response.get('executedQty', 0)),
            'cumulative_quote_qty': float(order_response.get('cumulativeQuoteQty', 0)),
            'time': order_response.get('transactTime', int(time.time() * 1000)),
            'trade_id': trade_id,
            'order_type': order_type,
            'raw_response': order_response
        }
        
        self.orders[order_id] = order_info
        
        # Add to order group if trade_id provided
        if trade_id:
            if trade_id not in self.order_groups:
                self.order_groups[trade_id] = []
            self.order_groups[trade_id].append(order_id)
        
        logger.info(f"Registered order {order_id} ({order_type}) for trade {trade_id}")
        
        return order_id
    
    def update_order_status(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Update order status by querying Binance API.
        
        Parameters
        ----------
        order_id : int
            Order ID
        
        Returns
        -------
        dict or None
            Updated order info, or None if order not found
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not registered")
            return None
        
        try:
            order_response = self.client.get_order(
                symbol=self.symbol,
                orderId=order_id
            )
            
            old_status = self.orders[order_id]['status']
            new_status = self._map_binance_status(order_response.get('status', 'NEW'))
            
            # Update order info
            self.orders[order_id].update({
                'status': new_status,
                'executed_qty': float(order_response.get('executedQty', 0)),
                'cumulative_quote_qty': float(order_response.get('cumulativeQuoteQty', 0)),
                'raw_response': order_response
            })
            
            # Call callback if status changed
            if old_status != new_status and self.on_order_update:
                self.on_order_update(self.orders[order_id])
            
            logger.debug(f"Updated order {order_id}: {old_status.value} -> {new_status.value}")
            
            return self.orders[order_id]
            
        except BinanceAPIException as e:
            logger.error(f"Error updating order status: {e}")
            return None
    
    def get_order(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Get order information.
        
        Parameters
        ----------
        order_id : int
            Order ID
        
        Returns
        -------
        dict or None
            Order info, or None if not found
        """
        return self.orders.get(order_id)
    
    def get_orders_for_trade(self, trade_id: str) -> List[Dict[str, Any]]:
        """
        Get all orders associated with a trade.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        
        Returns
        -------
        list of dict
            List of order info dicts
        """
        if trade_id not in self.order_groups:
            return []
        
        order_ids = self.order_groups[trade_id]
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Parameters
        ----------
        order_id : int
            Order ID to cancel
        
        Returns
        -------
        bool
            True if cancellation successful
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not registered")
            return False
        
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
            
            # Update status
            self.orders[order_id]['status'] = OrderStatus.CANCELED
            
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def cancel_all_orders_for_trade(self, trade_id: str) -> int:
        """
        Cancel all orders associated with a trade.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        
        Returns
        -------
        int
            Number of orders cancelled
        """
        orders = self.get_orders_for_trade(trade_id)
        cancelled = 0
        
        for order in orders:
            if order['status'] not in (OrderStatus.FILLED, OrderStatus.CANCELED):
                if self.cancel_order(order['order_id']):
                    cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders for trade {trade_id}")
        return cancelled
    
    def sync_open_orders(self) -> List[Dict[str, Any]]:
        """
        Sync with Binance API to get all open orders and update status.
        
        Returns
        -------
        list of dict
            List of updated order info dicts
        """
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            updated = []
            
            for order_response in open_orders:
                order_id = order_response['orderId']
                
                if order_id in self.orders:
                    # Update existing order
                    self.update_order_status(order_id)
                    updated.append(self.orders[order_id])
                else:
                    # Register new order (might be from external source)
                    self.register_order(order_response)
                    updated.append(self.orders[order_id])
            
            return updated
            
        except BinanceAPIException as e:
            logger.error(f"Error syncing open orders: {e}")
            return []
    
    def get_filled_orders(self) -> List[Dict[str, Any]]:
        """
        Get all filled orders.
        
        Returns
        -------
        list of dict
            List of filled order info dicts
        """
        return [
            order for order in self.orders.values()
            if order['status'] == OrderStatus.FILLED
        ]
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open (non-filled, non-cancelled) orders.
        
        Returns
        -------
        list of dict
            List of open order info dicts
        """
        open_statuses = {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL}
        return [
            order for order in self.orders.values()
            if order['status'] in open_statuses
        ]







