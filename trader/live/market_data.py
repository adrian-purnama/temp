# -*- coding: utf-8 -*-
"""
Market data consumer for live trading.

Handles WebSocket connection to Binance kline stream and maintains rolling window
of candles for signal generation.
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from binance.websocket import BinanceSocketManager
from binance.client import Client
try:
    # Try newer API first (python-binance >= 1.0.19)
    from binance import ThreadedWebsocketManager
    USE_THREADED_WS = True
except ImportError:
    try:
        # Fallback to older API
        from binance.websocket import BinanceSocketManager
        USE_THREADED_WS = False
    except ImportError:
        raise ImportError("Could not import WebSocket manager. Please ensure python-binance is installed.")
from binance.exceptions import BinanceAPIException
from trader.signals.sr_signal import add_atr


logger = logging.getLogger(__name__)


class MarketData:
    """Consumes real-time market data via WebSocket and maintains context window."""
    
    def __init__(self, client: Client, symbol: str, timeframe: str,
                 window_size: int = 192, atr_period: int = 14,
                 atr_method: str = "wilder", on_new_candle: Optional[Callable] = None,
                 ping_interval: float = 20.0, ping_timeout: float = 10.0,
                 connection_timeout: float = 30.0, close_timeout: float = 10.0,
                 max_queue_size: int = 500):
        """
        Initialize market data consumer.
        
        Parameters
        ----------
        client : Client
            Binance client instance
        symbol : str
            Trading pair symbol (e.g., "BTCUSDT")
        timeframe : str
            Kline interval (1m, 3m, 5m, 15m, 30m, 1h, etc.)
        window_size : int, default 192
            Size of rolling context window
        atr_period : int, default 14
            ATR calculation period
        atr_method : str, default "wilder"
            ATR calculation method
        on_new_candle : callable or None, optional
            Callback function called when new candle is received
        """
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.atr_period = atr_period
        self.atr_method = atr_method
        self.on_new_candle = on_new_candle
        
        # WebSocket configuration
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.connection_timeout = connection_timeout
        self.close_timeout = close_timeout
        self.max_queue_size = max_queue_size
        
        # Initialize data storage
        self.candles_df: Optional[pd.DataFrame] = None
        self.current_candle: Optional[pd.Series] = None
        self.is_running = False
        self.socket_manager: Optional[Any] = None
        self.socket_key: Optional[str] = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.last_message_time = None
        self._monitor_thread = None
        self._monitoring = False
        self._fallback_polling = False
        self._fallback_thread = None
        
        # Load initial historical data
        self._load_initial_data()
        
        logger.info(f"Initialized MarketData for {symbol} {timeframe}")
    
    def _load_initial_data(self):
        """Load initial historical data to build context window."""
        try:
            # Get historical klines
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=self.window_size + self.atr_period + 10  # Extra for ATR calculation
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Set index
            df = df.set_index('open_time')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Calculate ATR
            df = add_atr(df, period=self.atr_period, method=self.atr_method)
            
            # Keep only window_size candles
            self.candles_df = df.tail(self.window_size).copy()
            
            logger.info(f"Loaded {len(self.candles_df)} historical candles")
            
        except BinanceAPIException as e:
            logger.error(f"Error loading initial data: {e}")
            raise
    
    def _process_kline_update(self, msg: Dict[str, Any]):
        """
        Process kline update from WebSocket.
        
        Parameters
        ----------
        msg : dict
            WebSocket message containing kline data
        """
        try:
            # Reset reconnect attempts on successful message
            self.reconnect_attempts = 0
            self.last_message_time = time.time()
            
            # Stop fallback polling if websocket is working
            if self._fallback_polling:
                logger.info("WebSocket reconnected, stopping fallback polling")
                self._fallback_polling = False
            
            # Handle error messages from websocket
            if isinstance(msg, dict) and msg.get('e') == 'error':
                error_type = msg.get('type', 'Unknown')
                error_msg = msg.get('m', 'No message')
                logger.warning(f"WebSocket error message: {error_type} - {error_msg}")
                
                # If it's a connection error, try to reconnect
                if 'ConnectionClosed' in error_type or 'timeout' in error_msg.lower():
                    if self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
                        logger.info("Attempting to reconnect websocket...")
                        self._reconnect_websocket()
                    return
                return
            
            if 'k' not in msg:
                return
            
            kline = msg['k']
            
            # Check if this is a closed candle
            if not kline.get('x', False):  # 'x' indicates if candle is closed
                return
            
            # Extract candle data
            open_time = pd.to_datetime(kline['t'], unit='ms', utc=True)
            open_price = float(kline['o'])
            high_price = float(kline['h'])
            low_price = float(kline['l'])
            close_price = float(kline['c'])
            volume = float(kline['v'])
            
            # Create new candle Series
            new_candle = pd.Series({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }, name=open_time)
            
            # Add to DataFrame
            if self.candles_df is None:
                self.candles_df = pd.DataFrame([new_candle])
            else:
                # Check if candle already exists (update) or is new (append)
                if open_time in self.candles_df.index:
                    # Update existing candle
                    self.candles_df.loc[open_time] = new_candle
                else:
                    # Append new candle
                    self.candles_df = pd.concat([self.candles_df, pd.DataFrame([new_candle])])
                    # Keep only window_size candles
                    if len(self.candles_df) > self.window_size:
                        self.candles_df = self.candles_df.tail(self.window_size)
            
            # Recalculate ATR
            self.candles_df = add_atr(self.candles_df, period=self.atr_period, method=self.atr_method)
            
            # Update current candle
            self.current_candle = self.candles_df.iloc[-1]
            
            # Call callback if provided
            if self.on_new_candle is not None:
                self.on_new_candle(self.current_candle, self.candles_df)
            
            logger.debug(f"Processed new candle: {open_time} @ {close_price}")
            
        except Exception as e:
            logger.error(f"Error processing kline update: {e}", exc_info=True)
    
    def _configure_websocket_manager(self):
        """Configure websocket manager with keepalive settings."""
        if USE_THREADED_WS:
            # ThreadedWebsocketManager configuration
            manager = ThreadedWebsocketManager(
                testnet=getattr(self.client, 'testnet', False),
                max_queue_size=self.max_queue_size
            )
            # Configure underlying BinanceSocketManager's ws_kwargs
            # This will be done after manager starts, we need to patch the underlying manager
            return manager
        else:
            # BinanceSocketManager configuration
            manager = BinanceSocketManager(
                self.client,
                max_queue_size=self.max_queue_size,
                verbose=False
            )
            # Configure websocket kwargs for ping/pong keepalive
            # Note: close_timeout is already set by the library (0.1), so we don't include it
            manager.ws_kwargs = {
                'ping_interval': self.ping_interval,
                'ping_timeout': self.ping_timeout,
            }
            return manager
    
    def _reconnect_websocket(self):
        """Reconnect websocket with exponential backoff."""
        if not self.is_running or self._monitoring is False:
            return
        
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping websocket.")
            self.is_running = False
            self._monitoring = False
            return
        
        # Exponential backoff: 2^attempts seconds, max 60 seconds
        wait_time = min(2 ** self.reconnect_attempts, 60)
        logger.info(f"Reconnecting websocket (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) after {wait_time}s...")
        
        try:
            # Temporarily stop monitoring during reconnection
            monitoring_state = self._monitoring
            self._monitoring = False
            
            # Stop current connection (but don't stop the manager if it's still good)
            if self.socket_key and self.socket_manager:
                try:
                    self.socket_manager.stop_socket(self.socket_key)
                except Exception as e:
                    logger.debug(f"Error stopping socket during reconnect: {e}")
            
            self.is_running = False
            self.socket_key = None
            
            # Wait before reconnecting
            time.sleep(wait_time)
            
            # Restart connection
            if monitoring_state:  # Only reconnect if we were supposed to be monitoring
                self.start()
        except Exception as e:
            logger.error(f"Reconnection attempt {self.reconnect_attempts} failed: {e}")
            if self.reconnect_attempts < self.max_reconnect_attempts and self._monitoring:
                # Schedule next reconnection attempt in a separate thread
                import threading
                threading.Timer(wait_time, self._reconnect_websocket).start()
    
    def start(self):
        """Start WebSocket connection."""
        if self.is_running:
            logger.warning("MarketData already running")
            return
        
        try:
            # Initialize WebSocket manager with configuration
            self.socket_manager = self._configure_websocket_manager()
                self.socket_manager.start()
                
            # Wait for initialization (ThreadedWebsocketManager needs time to start)
                time.sleep(2)
                
            # For ThreadedWebsocketManager, we need to configure the underlying BinanceSocketManager
            if USE_THREADED_WS and hasattr(self.socket_manager, '_bsm'):
                # Wait for _bsm to be initialized
                max_wait = 10
                waited = 0
                while not hasattr(self.socket_manager, '_bsm') or self.socket_manager._bsm is None:
                    time.sleep(0.5)
                    waited += 0.5
                    if waited >= max_wait:
                        logger.warning("BinanceSocketManager not initialized, proceeding anyway...")
                        break
                
                # Configure websocket kwargs if _bsm is available
                # Note: close_timeout is already set by the library (0.1), so we don't include it
                if hasattr(self.socket_manager, '_bsm') and self.socket_manager._bsm:
                    self.socket_manager._bsm.ws_kwargs = {
                        'ping_interval': self.ping_interval,
                        'ping_timeout': self.ping_timeout,
                    }
            
            # Create kline stream
            if USE_THREADED_WS:
                self.socket_key = self.socket_manager.start_kline_socket(
                    callback=self._process_kline_update,
                    symbol=self.symbol,
                    interval=self.timeframe
                )
            else:
                self.socket_key = self.socket_manager.start_kline_socket(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    callback=self._process_kline_update
                )
            
            self.is_running = True
            self.reconnect_attempts = 0
            self.last_message_time = time.time()
            
            logger.info(f"Started WebSocket stream for {self.symbol} {self.timeframe} "
                       f"(ping_interval={self.ping_interval}s, ping_timeout={self.ping_timeout}s)")
            
            # Start connection monitoring
            self._start_connection_monitor()
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}", exc_info=True)
            # Try alternative approach if first fails
            if USE_THREADED_WS:
                logger.info("Trying alternative WebSocket initialization...")
                try:
                    self.socket_manager = ThreadedWebsocketManager(
                        api_key=getattr(self.client, 'api_key', None),
                        api_secret=getattr(self.client, 'api_secret', None),
                        testnet=getattr(self.client, 'testnet', False),
                        max_queue_size=self.max_queue_size
                    )
                    self.socket_manager.start()
                    time.sleep(2)
                    
                    # Configure underlying manager
                    # Note: close_timeout is already set by the library (0.1), so we don't include it
                    if hasattr(self.socket_manager, '_bsm') and self.socket_manager._bsm:
                        self.socket_manager._bsm.ws_kwargs = {
                            'ping_interval': self.ping_interval,
                            'ping_timeout': self.ping_timeout,
                        }
                    
                    self.socket_key = self.socket_manager.start_kline_socket(
                        callback=self._process_kline_update,
                        symbol=self.symbol,
                        interval=self.timeframe
                    )
                    self.is_running = True
                    self.reconnect_attempts = 0
                    self.last_message_time = time.time()
                    logger.info(f"Started WebSocket stream (alternative method) for {self.symbol} {self.timeframe}")
                    
                    # Start connection monitoring
                    self._start_connection_monitor()
                except Exception as e2:
                    logger.error(f"Alternative WebSocket initialization also failed: {e2}")
                    raise
            else:
                raise
    
    def _start_fallback_polling(self):
        """Start REST API polling as fallback when websocket fails."""
        if self._fallback_polling:
            return
        
        self._fallback_polling = True
        import threading
        
        def poll():
            """Poll REST API for latest candle when websocket is down."""
            last_candle_time = None
            
            while self._fallback_polling and self.is_running:
                try:
                    # Convert timeframe to milliseconds for polling interval
                    timeframe_map = {
                        '1m': 60, '3m': 180, '5m': 300, '15m': 900,
                        '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
                        '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400
                    }
                    poll_interval = timeframe_map.get(self.timeframe, 900)  # Default 15m
                    
                    time.sleep(poll_interval)
                    
                    if not self.is_running:
                        break
                    
                    # Only poll if websocket is not receiving messages
                    if self.last_message_time:
                        time_since_last = time.time() - self.last_message_time
                        if time_since_last < poll_interval * 2:
                            # Websocket is working, skip polling
                            continue
                    
                    # Fetch latest candle from REST API
                    klines = self.client.get_klines(
                        symbol=self.symbol,
                        interval=self.timeframe,
                        limit=2  # Get last 2 candles to check if new
                    )
                    
                    if len(klines) >= 2:
                        latest_kline = klines[-1]
                        candle_time = pd.to_datetime(latest_kline[0], unit='ms', utc=True)
                        
                        # Process if this is a new candle
                        if last_candle_time is None or candle_time > last_candle_time:
                            # Format as websocket message
                            msg = {
                                'k': {
                                    't': latest_kline[0],  # open time
                                    'T': latest_kline[6],  # close time
                                    'o': latest_kline[1],  # open
                                    'h': latest_kline[2],  # high
                                    'l': latest_kline[3],  # low
                                    'c': latest_kline[4],  # close
                                    'v': latest_kline[5],  # volume
                                    'x': True  # closed candle
                                }
                            }
                            
                            logger.info(f"Fallback polling: Processing candle from REST API @ {candle_time}")
                            self._process_kline_update(msg)
                            last_candle_time = candle_time
                    
                except Exception as e:
                    logger.error(f"Error in fallback polling: {e}")
        
        self._fallback_thread = threading.Thread(target=poll, daemon=True)
        self._fallback_thread.start()
        logger.info("Started fallback REST API polling")
    
    def _start_connection_monitor(self):
        """Start background thread to monitor connection health."""
        if self._monitoring:
            return
        
        self._monitoring = True
        import threading
        
        def monitor():
            """Monitor connection health and attempt reconnection if needed."""
            while self._monitoring and self.is_running:
                try:
                    time.sleep(60)  # Check every 60 seconds
                    
                    if not self.is_running:
                        break
                    
                    # Check if we've received messages recently
                    # If no messages for 2 * ping_interval, connection might be stale
                    if self.last_message_time:
                        time_since_last = time.time() - self.last_message_time
                        max_silence = max(self.ping_interval * 2, 120)  # At least 2 minutes
                        
                        if time_since_last > max_silence:
                            logger.warning(f"No messages received for {time_since_last:.0f}s. "
                                         f"Connection may be stale. Starting fallback polling...")
                            
                            # Start fallback polling if not already running
                            if not self._fallback_polling:
                                self._start_fallback_polling()
                            
                            if self.reconnect_attempts < self.max_reconnect_attempts:
                                self._reconnect_websocket()
                            else:
                                logger.error("Max reconnection attempts reached. Using fallback polling only.")
                    
                except Exception as e:
                    logger.error(f"Error in connection monitor: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        logger.debug("Started connection health monitor")
    
    def stop(self):
        """Stop WebSocket connection."""
        if not self.is_running:
            return
        
        # Stop monitoring
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        # Stop fallback polling
        self._fallback_polling = False
        if self._fallback_thread:
            self._fallback_thread.join(timeout=5)
        
        try:
            if self.socket_manager:
                if self.socket_key:
                    try:
                        self.socket_manager.stop_socket(self.socket_key)
                    except Exception as e:
                        logger.warning(f"Error stopping socket: {e}")
                
                # Close the manager
                try:
                    if hasattr(self.socket_manager, 'close'):
                        self.socket_manager.close()
                    elif hasattr(self.socket_manager, 'stop'):
                        self.socket_manager.stop()
                except Exception as e:
                    logger.warning(f"Error closing socket manager: {e}")
            
            self.is_running = False
            logger.info("Stopped WebSocket stream")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket: {e}")
    
    def get_context_window(self) -> pd.DataFrame:
        """
        Get current context window.
        
        Returns
        -------
        pd.DataFrame
            Context window DataFrame with OHLCV and ATR
        """
        if self.candles_df is None:
            raise ValueError("No data available. Call start() first.")
        
        return self.candles_df.copy()
    
    def get_current_candle(self) -> Optional[pd.Series]:
        """
        Get current candle.
        
        Returns
        -------
        pd.Series or None
            Current candle data
        """
        return self.current_candle
    
    def get_atr_value(self) -> float:
        """
        Get current ATR value.
        
        Returns
        -------
        float
            Current ATR value
        """
        if self.candles_df is None or 'atr' not in self.candles_df.columns:
            # Fallback: use high-low range as proxy
            if self.current_candle is not None:
                return (self.current_candle['high'] - self.current_candle['low']) * 0.5
            return 0.0
        
        atr_value = self.candles_df['atr'].iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            # Fallback
            if self.current_candle is not None:
                return (self.current_candle['high'] - self.current_candle['low']) * 0.5
            return 0.0
        
        return float(atr_value)
    
    def update_historical_data(self):
        """Refresh historical data from API (useful for recovery)."""
        try:
            self._load_initial_data()
            logger.info("Updated historical data")
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")

