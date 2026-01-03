# -*- coding: utf-8 -*-
"""
Binance API client wrapper.

Handles authentication, rate limiting, and provides a clean interface for Binance API operations.
Supports both Testnet and Realnet environments.
"""

import time
import logging
import requests
import json
from typing import Dict, Any, Optional, List, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException


logger = logging.getLogger(__name__)


def create_insufficient_balance_exception(message: str) -> BinanceAPIException:
    """
    Create a BinanceAPIException with code -2010 (insufficient balance).
    
    Parameters
    ----------
    message : str
        Error message
    
    Returns
    -------
    BinanceAPIException
        Exception with code -2010
    """
    # Create a mock response-like object
    class MockResponse:
        def __init__(self):
            self.text = json.dumps({"code": -2010, "msg": message})
    
    mock_response = MockResponse()
    exc = BinanceAPIException(
        response=mock_response,
        status_code=400,
        text=mock_response.text
    )
    return exc


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
    
    def check_sufficient_balance(self, asset: str, required_amount: float, 
                                 buffer_pct: float = 0.01) -> Tuple[bool, float]:
        """
        Check if account has sufficient balance for a trade.
        
        Parameters
        ----------
        asset : str
            Asset symbol (e.g., "USDT", "BTC")
        required_amount : float
            Required amount for the trade
        buffer_pct : float, default 0.01
            Buffer percentage to account for fees (1% default)
        
        Returns
        -------
        tuple
            (has_sufficient_balance: bool, current_balance: float)
        """
        try:
            current_balance = self.get_account_balance(asset)
            required_with_buffer = required_amount * (1 + buffer_pct)
            has_sufficient = current_balance >= required_with_buffer
            
            logger.debug(f"Balance check for {asset}: "
                        f"Current={current_balance:.8f}, "
                        f"Required={required_amount:.8f}, "
                        f"Required+Buffer={required_with_buffer:.8f}, "
                        f"Sufficient={has_sufficient}")
            
            return has_sufficient, current_balance
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            # Return False on error to be safe
            return False, 0.0
    
    def request_faucet_funds(self, asset: str = "USDT", 
                            amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Request testnet funds from Binance faucet.
        
        Note: Faucet typically allows 1 request per 24 hours per asset.
        This only works on testnet.
        
        Parameters
        ----------
        asset : str, default "USDT"
            Asset to request (e.g., "USDT", "BTC")
        amount : float or None, optional
            Amount to request. If None, requests default amount.
            Note: Faucet may ignore this and give default amount.
        
        Returns
        -------
        dict
            Response containing success status and new balance
        """
        if not self.use_testnet:
            logger.warning("Faucet requests only available on testnet")
            return {
                'success': False,
                'error': 'Faucet only available on testnet',
                'balance': self.get_account_balance(asset)
            }
        
        faucet_url = f"{self.TESTNET_BASE_URL}/api/v3/faucet/claim"
        
        try:
            # Get balance before request
            balance_before = self.get_account_balance(asset)
            
            # Prepare request
            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Binance testnet faucet typically uses POST with asset in body
            # Some versions may require amount, but most ignore it
            payload = {'asset': asset}
            if amount is not None:
                payload['amount'] = amount
            
            logger.info(f"Requesting {amount if amount else 'default'} {asset} from testnet faucet...")
            
            response = requests.post(
                faucet_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                # Wait a moment for balance to update
                time.sleep(2)
                balance_after = self.get_account_balance(asset)
                received = balance_after - balance_before
                
                logger.info(f"✓ Faucet request successful! "
                          f"Received {received:.8f} {asset}. "
                          f"New balance: {balance_after:.8f} {asset}")
                
                return {
                    'success': True,
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'received': received,
                    'response': result
                }
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('msg', error_msg)
                except:
                    error_msg = response.text or error_msg
                
                logger.warning(f"Faucet request failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'balance': self.get_account_balance(asset)
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting faucet funds: {e}")
            return {
                'success': False,
                'error': str(e),
                'balance': self.get_account_balance(asset)
            }
        except Exception as e:
            logger.error(f"Unexpected error requesting faucet funds: {e}")
            return {
                'success': False,
                'error': str(e),
                'balance': self.get_account_balance(asset)
            }
    
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
                          quote_order_qty: Optional[float] = None,
                          auto_request_faucet: bool = True) -> Dict[str, Any]:
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
        auto_request_faucet : bool, default True
            Automatically request faucet funds if balance insufficient (testnet only)
        
        Returns
        -------
        dict
            Order response from Binance
        """
        try:
            # Get minimum requirements
            min_notional = self.get_min_notional(symbol)
            min_quantity = self.get_min_quantity(symbol)
            current_price = self.get_current_price(symbol)
            
            if side == "BUY" and quote_order_qty is not None:
                # Validate minimum notional for BUY orders
                if quote_order_qty < min_notional:
                    logger.warning(f"Order value {quote_order_qty} below minimum notional {min_notional}. "
                                 f"Increasing to minimum.")
                    quote_order_qty = min_notional
                
                # Check balance for BUY orders (need quote asset - USDT)
                quote_asset = "USDT"  # Assuming USDT pairs
                has_sufficient, current_balance = self.check_sufficient_balance(
                    quote_asset, quote_order_qty
                )
                
                if not has_sufficient:
                    required_with_buffer = quote_order_qty * 1.01
                    shortfall = max(0, required_with_buffer - current_balance)
                    logger.warning(f"Insufficient {quote_asset} balance for BUY order. "
                                 f"Current: {current_balance:.8f}, Required: {quote_order_qty:.8f}, "
                                 f"Required+Buffer: {required_with_buffer:.8f}, Shortfall: {shortfall:.8f}")
                    
                    # Try faucet if on testnet and enabled
                    if self.use_testnet and auto_request_faucet:
                        logger.info(f"Attempting to request {shortfall:.8f} {quote_asset} from faucet...")
                        faucet_result = self.request_faucet_funds(quote_asset, amount=shortfall)
                        if faucet_result['success']:
                            # Recheck balance after faucet
                            has_sufficient, current_balance = self.check_sufficient_balance(
                                quote_asset, quote_order_qty
                            )
                            if not has_sufficient:
                                raise create_insufficient_balance_exception(
                                    f"Insufficient balance after faucet request. "
                                    f"Current: {current_balance:.8f}, Required: {quote_order_qty:.8f}"
                                )
                        else:
                            raise create_insufficient_balance_exception(
                                f"Insufficient balance and faucet request failed: {faucet_result.get('error', 'Unknown error')}"
                            )
                    else:
                        raise create_insufficient_balance_exception(
                            f"Insufficient {quote_asset} balance. Current: {current_balance:.8f}, Required: {quote_order_qty:.8f}"
                        )
                
                # Format quote order quantity (remove decimals for quote)
                quote_order_qty_str = f"{quote_order_qty:.2f}".rstrip('0').rstrip('.')
                
                # Market buy using quote quantity
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quoteOrderQty=quote_order_qty_str
                )
            else:
                # Validate minimum quantity and notional for SELL orders
                if quantity < min_quantity:
                    logger.warning(f"Quantity {quantity} below minimum {min_quantity}. "
                                 f"Increasing to minimum.")
                    quantity = min_quantity
                
                order_value = quantity * current_price
                if order_value < min_notional:
                    # Increase quantity to meet minimum notional
                    required_quantity = min_notional / current_price
                    # Round up to meet minimum quantity
                    quantity = max(required_quantity, min_quantity)
                    logger.warning(f"Order value {order_value} below minimum notional {min_notional}. "
                                 f"Increasing quantity to {quantity}.")
                
                # Check balance for SELL orders (need base asset)
                base_asset = symbol.replace("USDT", "").replace("USD", "")
                has_sufficient, current_balance = self.check_sufficient_balance(
                    base_asset, quantity
                )
                
                if not has_sufficient:
                    required_with_buffer = quantity * 1.01
                    shortfall = max(0, required_with_buffer - current_balance)
                    logger.warning(f"Insufficient {base_asset} balance for SELL order. "
                                 f"Current: {current_balance:.8f}, Required: {quantity:.8f}, "
                                 f"Required+Buffer: {required_with_buffer:.8f}, Shortfall: {shortfall:.8f}")
                    
                    # Try faucet if on testnet and enabled
                    if self.use_testnet and auto_request_faucet:
                        logger.info(f"Attempting to request {shortfall:.8f} {base_asset} from faucet...")
                        faucet_result = self.request_faucet_funds(base_asset, amount=shortfall)
                        if faucet_result['success']:
                            # Recheck balance after faucet
                            has_sufficient, current_balance = self.check_sufficient_balance(
                                base_asset, quantity
                            )
                            if not has_sufficient:
                                raise create_insufficient_balance_exception(
                                    f"Insufficient balance after faucet request. "
                                    f"Current: {current_balance:.8f}, Required: {quantity:.8f}"
                                )
                        else:
                            raise create_insufficient_balance_exception(
                                f"Insufficient balance and faucet request failed: {faucet_result.get('error', 'Unknown error')}"
                            )
                    else:
                        raise create_insufficient_balance_exception(
                            f"Insufficient {base_asset} balance. Current: {current_balance:.8f}, Required: {quantity:.8f}"
                        )
                
                # Format quantity according to Binance precision requirements
                quantity_str = self.format_quantity(symbol, quantity)
                
                # Market sell using base quantity
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity_str
                )
            
            logger.info(f"Placed market {side} order: {order['orderId']} for {symbol}")
            return order
        except BinanceAPIException as e:
            # Handle -2010 insufficient balance error specifically
            if e.code == -2010:
                logger.error(f"Insufficient balance error (-2010): {e}")
                # Log balance details
                if side == "BUY" and quote_order_qty is not None:
                    quote_asset = "USDT"
                    current_balance = self.get_account_balance(quote_asset)
                    logger.error(f"Current {quote_asset} balance: {current_balance:.8f}, "
                               f"Required: {quote_order_qty:.8f}")
                else:
                    base_asset = symbol.replace("USDT", "").replace("USD", "")
                    current_balance = self.get_account_balance(base_asset)
                    logger.error(f"Current {base_asset} balance: {current_balance:.8f}, "
                               f"Required: {quantity:.8f}")
            raise
        except BinanceOrderException as e:
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
    
    def get_min_notional(self, symbol: str) -> float:
        """
        Get minimum notional (order value) requirement for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        
        Returns
        -------
        float
            Minimum notional value in quote asset (e.g., USDT)
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            filters = symbol_info.get('filters', [])
            
            for filter_item in filters:
                if filter_item.get('filterType') == 'MIN_NOTIONAL':
                    min_notional = float(filter_item.get('minNotional', 10.0))
                    return min_notional
                elif filter_item.get('filterType') == 'NOTIONAL':
                    min_notional = float(filter_item.get('minNotional', 10.0))
                    return min_notional
            
            # Default minimum notional for most pairs is 10 USDT
            return 10.0
        except Exception as e:
            logger.warning(f"Error getting min notional for {symbol}: {e}. Using default 10.0")
            return 10.0
    
    def get_min_quantity(self, symbol: str) -> float:
        """
        Get minimum quantity requirement for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        
        Returns
        -------
        float
            Minimum quantity in base asset
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            filters = symbol_info.get('filters', [])
            
            for filter_item in filters:
                if filter_item.get('filterType') == 'LOT_SIZE':
                    min_qty = float(filter_item.get('minQty', 0.001))
                    return min_qty
            
            return 0.001  # Default
        except Exception as e:
            logger.warning(f"Error getting min quantity for {symbol}: {e}. Using default 0.001")
            return 0.001
    
    def get_step_size(self, symbol: str) -> float:
        """
        Get step size (quantity precision) for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        
        Returns
        -------
        float
            Step size (e.g., 0.001 means quantities must be multiples of 0.001)
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            filters = symbol_info.get('filters', [])
            
            for filter_item in filters:
                if filter_item.get('filterType') == 'LOT_SIZE':
                    step_size = float(filter_item.get('stepSize', 0.001))
                    return step_size
            
            return 0.001  # Default
        except Exception as e:
            logger.warning(f"Error getting step size for {symbol}: {e}. Using default 0.001")
            return 0.001
    
    def format_quantity(self, symbol: str, quantity: float) -> str:
        """
        Format quantity according to Binance precision requirements.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol
        quantity : float
            Raw quantity value
        
        Returns
        -------
        str
            Formatted quantity string matching Binance requirements
        """
        try:
            step_size = self.get_step_size(symbol)
            
            # Round to step size
            if step_size >= 1.0:
                # Integer step size
                quantity = int(quantity / step_size) * step_size
                return str(int(quantity))
            else:
                # Decimal step size - calculate decimal places
                step_str = f"{step_size:.20f}".rstrip('0')
                if '.' in step_str:
                    decimal_places = len(step_str.split('.')[1])
                else:
                    decimal_places = 0
                
                # Round to step size
                quantity = round(quantity / step_size) * step_size
                
                # Format with appropriate decimal places (but not scientific notation)
                return f"{quantity:.{decimal_places}f}".rstrip('0').rstrip('.')
        except Exception as e:
            logger.warning(f"Error formatting quantity: {e}. Using simple format.")
            # Fallback: format to 8 decimal places max, remove trailing zeros
            return f"{quantity:.8f}".rstrip('0').rstrip('.')
    
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


