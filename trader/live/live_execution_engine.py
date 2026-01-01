# -*- coding: utf-8 -*-
"""
Live execution engine for Binance.

Extends ExecutionEngine to actually execute orders on Binance while maintaining
the same interface for fee/slippage calculation.
"""

import logging
import time
from typing import Literal, Optional, Dict, Any
from trader.backtesting.execution_engine import ExecutionEngine
from trader.live.binance_client import BinanceClient
from trader.live.order_manager import OrderManager


logger = logging.getLogger(__name__)


class LiveExecutionEngine(ExecutionEngine):
    """Live execution engine that executes orders on Binance."""
    
    def __init__(self, binance_client: BinanceClient, order_manager: OrderManager,
                 fee_pct: float = 0.1, slippage_atr_mult: float = 0.0):
        """
        Initialize live execution engine.
        
        Parameters
        ----------
        binance_client : BinanceClient
            Binance client instance
        order_manager : OrderManager
            Order manager instance
        fee_pct : float, default 0.1
            Transaction fee percentage (Binance default: 0.1%)
        slippage_atr_mult : float, default 0.0
            Slippage as ATR multiplier (0.0 = no slippage modeling)
        """
        super().__init__(fee_pct=fee_pct, slippage_atr_mult=slippage_atr_mult)
        self.binance_client = binance_client
        self.order_manager = order_manager
        
        logger.info("Initialized LiveExecutionEngine")
    
    def execute_entry_order(self, symbol: str, direction: Literal['long', 'short'],
                           quantity: float, trade_id: str,
                           quote_order_qty: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute entry order on Binance.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        direction : 'long' or 'short'
            Trade direction
        quantity : float
            Base asset quantity (for SELL) or None (for BUY with quote_order_qty)
        trade_id : str
            Trade ID for tracking
        quote_order_qty : float or None, optional
            Quote asset quantity (for market BUY orders)
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            side = "BUY" if direction == "long" else "SELL"
            
            # Place market order
            order_response = self.binance_client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                quote_order_qty=quote_order_qty
            )
            
            # Register order with order manager
            self.order_manager.register_order(
                order_response,
                trade_id=trade_id,
                order_type="entry"
            )
            
            logger.info(f"Executed entry order for trade {trade_id}: {side} {quantity}")
            
            return order_response
            
        except Exception as e:
            logger.error(f"Error executing entry order: {e}")
            raise
    
    def execute_exit_order(self, symbol: str, direction: Literal['long', 'short'],
                          quantity: float, trade_id: str) -> Dict[str, Any]:
        """
        Execute exit order on Binance.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        direction : 'long' or 'short'
            Trade direction
        quantity : float
            Base asset quantity to sell
        trade_id : str
            Trade ID for tracking
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            # For exit, we always sell (close long) or buy (close short)
            side = "SELL" if direction == "long" else "BUY"
            
            # Place market order
            order_response = self.binance_client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            # Register order with order manager
            self.order_manager.register_order(
                order_response,
                trade_id=trade_id,
                order_type="exit"
            )
            
            logger.info(f"Executed exit order for trade {trade_id}: {side} {quantity}")
            
            return order_response
            
        except Exception as e:
            logger.error(f"Error executing exit order: {e}")
            raise
    
    def place_stop_loss_take_profit(self, symbol: str, direction: Literal['long', 'short'],
                                    quantity: float, stop_loss_price: float,
                                    take_profit_price: float, trade_id: str) -> Dict[str, Any]:
        """
        Place stop-loss and take-profit orders.
        
        For spot trading, places separate STOP_LOSS_LIMIT and LIMIT orders.
        Note: Binance Spot doesn't support true OCO orders, so we place both
        and manage cancellation manually.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        direction : 'long' or 'short'
            Trade direction
        quantity : float
            Base asset quantity
        stop_loss_price : float
            Stop loss price
        take_profit_price : float
            Take profit price
        trade_id : str
            Trade ID for tracking
        
        Returns
        -------
        dict
            Dictionary containing stop_loss_order and take_profit_order
        """
        try:
            # For spot trading, we place separate orders
            # Stop loss: STOP_LOSS_LIMIT order (SELL for long position)
            # Take profit: LIMIT order (SELL for long position)
            
            if direction == "long":
                # Place stop-loss order
                stop_loss_order = self.binance_client.place_stop_loss_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    stop_price=stop_loss_price,
                    limit_price=stop_loss_price * 0.999  # Slightly below stop price
                )
                
                # Register stop-loss order
                stop_loss_order_id = self.order_manager.register_order(
                    stop_loss_order,
                    trade_id=trade_id,
                    order_type="stop_loss"
                )
                
                # Place take-profit order
                take_profit_order = self.binance_client.place_take_profit_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    price=take_profit_price
                )
                
                # Register take-profit order
                take_profit_order_id = self.order_manager.register_order(
                    take_profit_order,
                    trade_id=trade_id,
                    order_type="take_profit"
                )
                
                logger.info(f"Placed SL/TP orders for trade {trade_id}: SL@{stop_loss_price}, TP@{take_profit_price}")
                
                return {
                    'stop_loss_order': stop_loss_order,
                    'take_profit_order': take_profit_order,
                    'stop_loss_order_id': stop_loss_order_id,
                    'take_profit_order_id': take_profit_order_id
                }
            else:
                # Short positions (less common in spot trading)
                # Would need BUY orders instead
                raise ValueError("Short positions not fully supported for spot trading")
                
        except Exception as e:
            logger.error(f"Error placing stop-loss/take-profit orders: {e}")
            raise
    
    def cancel_stop_loss_take_profit(self, trade_id: str) -> int:
        """
        Cancel stop-loss and take-profit orders for a trade.
        
        Parameters
        ----------
        trade_id : str
            Trade ID
        
        Returns
        -------
        int
            Number of orders cancelled
        """
        return self.order_manager.cancel_all_orders_for_trade(trade_id)
    
    def wait_for_order_fill(self, order_id: int, timeout_seconds: int = 30,
                           poll_interval: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Wait for an order to be filled.
        
        Parameters
        ----------
        order_id : int
            Order ID
        timeout_seconds : int, default 30
            Maximum time to wait
        poll_interval : float, default 1.0
            Polling interval in seconds
        
        Returns
        -------
        dict or None
            Updated order info if filled, None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            order_info = self.order_manager.update_order_status(order_id)
            
            if order_info is None:
                break
            
            status = order_info['status']
            if status.value in ('FILLED', 'PARTIALLY_FILLED'):
                return order_info
            
            time.sleep(poll_interval)
        
        logger.warning(f"Order {order_id} did not fill within {timeout_seconds} seconds")
        return None
    
    def get_fill_price(self, order_response: Dict[str, Any]) -> float:
        """
        Get actual fill price from order response.
        
        Parameters
        ----------
        order_response : dict
            Order response from Binance
        
        Returns
        -------
        float
            Average fill price
        """
        executed_qty = float(order_response.get('executedQty', 0))
        cumulative_quote_qty = float(order_response.get('cumulativeQuoteQty', 0))
        
        if executed_qty > 0:
            return cumulative_quote_qty / executed_qty
        
        # Fallback to order price if available
        if order_response.get('price'):
            return float(order_response['price'])
        
        return 0.0





