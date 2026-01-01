# -*- coding: utf-8 -*-
"""
Binance API client wrapper.

Handles authentication, rate limiting, and provides a clean interface for Binance API operations.
Supports both Testnet and Realnet environments.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException


logger = logging.getLogger(__name__)


class BinanceClient:
    """Wrapper around python-binance client for Testnet and Realnet support."""
    
    # Binance API endpoints
    TESTNET_BASE_URL = "https://testnet.binance.vision"
    REALNET_BASE_URL = "https://api.binance.com"
    
    def __init__(self, api_key: str, api_secret: str, use_testnet: bool = True):
        """
        Initialize Binance client.
        
        Parameters
        ----------
        api_key : str
            Binance API key
        api_secret : str
            Binance API secret
        use_testnet : bool, default True
            Whether to use Binance Testnet
        """
        # Clean API keys (remove any whitespace)
        self.api_key = api_key.strip() if api_key else ""
        self.api_secret = api_secret.strip() if api_secret else ""
        self.use_testnet = use_testnet
        
        # Validate API keys
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be provided")
        
        # Initialize python-binance Client
        # The library handles testnet automatically - when testnet=True, it uses testnet.binance.vision
        # We use the simplest initialization to avoid interfering with signature generation
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=use_testnet
            )
            
            # Test connection with a simple public endpoint first
            server_time = self.client.get_server_time()
            endpoint = "TESTNET (testnet.binance.vision)" if use_testnet else "REALNET (api.binance.com)"
            logger.info(f"✓ Connected to Binance {endpoint}")
            
            # Test private endpoint to verify signature works
            try:
                # Simple test - get account info (this requires valid signature)
                account_info = self.client.get_account()
                logger.info("✓ API key authentication successful")
            except BinanceAPIException as e:
                if e.code == -1022:
                    logger.error("✗ Signature error - API key authentication failed")
                    logger.error("  Possible causes:")
                    logger.error("  1. API key/secret mismatch")
                    logger.error("  2. API key doesn't have required permissions")
                    logger.error("  3. IP whitelist is enabled and your IP is not whitelisted")
                    logger.error("  4. Using mainnet keys with testnet (or vice versa)")
                    # Don't raise - let the bot continue, but log the issue
                else:
                    raise
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def get_account_balance(self, asset: str = "USDT") -> float:
        """
        Get account balance for a specific asset.
        
        Parameters
        ----------
        asset : str, default "USDT"
            Asset symbol (e.g., "USDT", "BTC")
        
        Returns
        -------
        float
            Available balance
        """
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except BinanceAPIException as e:
            if e.code == -1022:  # Invalid signature
                logger.warning(f"Signature error getting account balance: {e}")
                logger.warning("This usually means API key permissions or IP whitelist issue")
                # Return 0 instead of raising - allows bot to continue with override balance
                return 0.0
            else:
                logger.error(f"Error getting account balance: {e}")
                raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get full account information.
        
        Returns
        -------
        dict
            Account information including balances
        """
        try:
            return self.client.get_account()
        except BinanceAPIException as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get open positions for a symbol.
        
        Note: Binance Spot doesn't have "positions" like futures.
        This returns open orders and account balances for the symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., "BTCUSDT")
        
        Returns
        -------
        list of dict
            List of open positions/orders
        """
        try:
            # Get open orders for the symbol
            open_orders = self.client.get_open_orders(symbol=symbol)
            
            # Get account balance for base asset
            base_asset = symbol.replace("USDT", "").replace("USD", "")
            balance = self.get_account_balance(asset=base_asset)
            
            positions = []
            if balance > 0:
                # We have a position if we hold the base asset
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                positions.append({
                    'symbol': symbol,
                    'base_asset': base_asset,
                    'quantity': balance,
                    'current_price': current_price,
                    'value': balance * current_price
                })
            
            return positions
        except BinanceAPIException as e:
            logger.error(f"Error getting open positions: {e}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Parameters
        ----------
        symbol : str or None, optional
            Trading pair symbol. If None, returns all open orders.
        
        Returns
        -------
        list of dict
            List of open orders
        """
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            else:
                return self.client.get_open_orders()
        except BinanceAPIException as e:
            logger.error(f"Error getting open orders: {e}")
            raise
    
    def place_market_order(self, symbol: str, side: str, quantity: float,
                          quote_order_qty: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a market order.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., "BTCUSDT")
        side : str
            Order side: "BUY" or "SELL"
        quantity : float
            Base asset quantity (for BUY, use quote_order_qty instead)
        quote_order_qty : float or None, optional
            Quote asset quantity (for market BUY orders)
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            if side == "BUY" and quote_order_qty is not None:
                # Market buy using quote quantity
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quoteOrderQty=quote_order_qty
                )
            else:
                # Market sell using base quantity
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
            
            logger.info(f"Placed market {side} order: {order['orderId']} for {symbol}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Error placing market order: {e}")
            raise
    
    def place_limit_order(self, symbol: str, side: str, quantity: float,
                         price: float, time_in_force: str = "GTC") -> Dict[str, Any]:
        """
        Place a limit order.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        side : str
            Order side: "BUY" or "SELL"
        quantity : float
            Base asset quantity
        price : float
            Limit price
        time_in_force : str, default "GTC"
            Time in force: "GTC", "IOC", "FOK"
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            if side == "BUY":
                order = self.client.order_limit_buy(
                    symbol=symbol,
                    quantity=quantity,
                    price=str(price),
                    timeInForce=time_in_force
                )
            else:
                order = self.client.order_limit_sell(
                    symbol=symbol,
                    quantity=quantity,
                    price=str(price),
                    timeInForce=time_in_force
                )
            
            logger.info(f"Placed limit {side} order: {order['orderId']} for {symbol} @ {price}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Error placing limit order: {e}")
            raise
    
    def place_oco_order(self, symbol: str, side: str, quantity: float,
                       price: float, stop_price: float, stop_limit_price: float,
                       stop_limit_time_in_force: str = "GTC") -> Dict[str, Any]:
        """
        Place an OCO (One-Cancels-Other) order.
        
        OCO orders combine a limit order and a stop-limit order.
        When one is executed, the other is automatically cancelled.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        side : str
            Order side: "BUY" or "SELL"
        quantity : float
            Base asset quantity
        price : float
            Limit price (take profit)
        stop_price : float
            Stop price (stop loss trigger)
        stop_limit_price : float
            Stop limit price (stop loss execution)
        stop_limit_time_in_force : str, default "GTC"
            Time in force for stop limit order
        
        Returns
        -------
        dict
            OCO order response from Binance
        """
        try:
            # For spot trading, we use STOP_LOSS_LIMIT order type
            # OCO is primarily for futures, but we can simulate with separate orders
            # For now, we'll place a stop-limit order for stop loss
            # and a limit order for take profit
            
            # Note: Binance Spot doesn't support true OCO orders
            # We'll need to manage this manually by placing both orders
            # and cancelling one when the other fills
            
            # Place stop-limit order for stop loss
            if side == "SELL":  # Closing a long position
                stop_order = self.client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type="STOP_LOSS_LIMIT",
                    quantity=quantity,
                    stopPrice=str(stop_price),
                    price=str(stop_limit_price),
                    timeInForce=stop_limit_time_in_force
                )
                
                # Place limit order for take profit
                limit_order = self.client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type="LIMIT",
                    quantity=quantity,
                    price=str(price),
                    timeInForce="GTC"
                )
                
                return {
                    'orderListId': -1,  # Not a true OCO, so no list ID
                    'stop_order': stop_order,
                    'limit_order': limit_order
                }
            else:  # BUY (closing a short position - not applicable for spot)
                raise ValueError("BUY OCO orders not supported for spot trading")
                
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Error placing OCO order: {e}")
            raise
    
    def place_stop_loss_order(self, symbol: str, side: str, quantity: float,
                              stop_price: float, limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a stop-loss order.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        side : str
            Order side: "SELL" (for long) or "BUY" (for short)
        quantity : float
            Base asset quantity
        stop_price : float
            Stop price (trigger price)
        limit_price : float or None, optional
            Limit price. If None, uses stop_price (STOP_MARKET equivalent)
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            if limit_price is None:
                # Use STOP_MARKET (market order when stop price hit)
                # Note: Binance Spot doesn't have STOP_MARKET, so we use STOP_LOSS_LIMIT with tight limit
                limit_price = stop_price * 0.999 if side == "SELL" else stop_price * 1.001
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type="STOP_LOSS_LIMIT",
                quantity=quantity,
                stopPrice=str(stop_price),
                price=str(limit_price),
                timeInForce="GTC"
            )
            
            logger.info(f"Placed stop-loss order: {order['orderId']} for {symbol} @ {stop_price}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Error placing stop-loss order: {e}")
            raise
    
    def place_take_profit_order(self, symbol: str, side: str, quantity: float,
                                price: float) -> Dict[str, Any]:
        """
        Place a take-profit limit order.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        side : str
            Order side: "SELL" (for long) or "BUY" (for short)
        quantity : float
            Base asset quantity
        price : float
            Limit price (take profit price)
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            order = self.client.order_limit_sell(
                symbol=symbol,
                quantity=quantity,
                price=str(price)
            ) if side == "SELL" else self.client.order_limit_buy(
                symbol=symbol,
                quantity=quantity,
                price=str(price)
            )
            
            logger.info(f"Placed take-profit order: {order['orderId']} for {symbol} @ {price}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Error placing take-profit order: {e}")
            raise
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        order_id : int
            Order ID to cancel
        
        Returns
        -------
        dict
            Cancellation response from Binance
        """
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Error cancelling order: {e}")
            raise
    
    def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Cancel all open orders for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        
        Returns
        -------
        list of dict
            List of cancelled orders
        """
        try:
            result = self.client.cancel_open_orders(symbol=symbol)
            logger.info(f"Cancelled all orders for {symbol}")
            return result
        except BinanceAPIException as e:
            logger.error(f"Error cancelling all orders: {e}")
            raise
    
    def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Get order status.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        order_id : int
            Order ID
        
        Returns
        -------
        dict
            Order status information
        """
        try:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            logger.error(f"Error getting order status: {e}")
            raise
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500,
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[List]:
        """
        Get kline/candlestick data.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        interval : str
            Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        limit : int, default 500
            Number of klines to retrieve (max 1000)
        start_time : int or None, optional
            Start timestamp in milliseconds
        end_time : int or None, optional
            End timestamp in milliseconds
        
        Returns
        -------
        list of list
            List of kline data: [open_time, open, high, low, close, volume, ...]
        """
        try:
            return self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
        except BinanceAPIException as e:
            logger.error(f"Error getting klines: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information (filters, precision, etc.).
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        
        Returns
        -------
        dict
            Symbol information
        """
        try:
            exchange_info = self.client.get_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    return s
            raise ValueError(f"Symbol {symbol} not found")
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        
        Returns
        -------
        float
            Current price
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting current price: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns
        -------
        bool
            True if connection successful
        """
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


