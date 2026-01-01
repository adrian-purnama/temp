# -*- coding: utf-8 -*-
"""
Live trading configuration schema.

Defines all configurable parameters for live trading on Binance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LiveTradingConfig:
    """Configuration for live trading engine."""
    
    # Binance Connection
    use_testnet: bool = True
    api_key: str = "DSAmKuRluJwGG76oleQnXl45Wa8eWLVINBUNQNDgSfXDnDNM0Rr40LynS297RGii"
    api_secret: str = "KfrQLOj3ji9E7bOdG0f4qEkPGZqpVfL3Wu9Fux5B0mDZpu1BqtewZNvJ9n26tg4J"
    
    # api_key: str = "hHsWE5juhEC9YeVeI5oT5azi3z4gv0PcE09OcClZ47jKfYGgJNJFWN1S4fMb4dU2"
    # api_secret: str = "CdNKNTiEsc3wuNoAJGiF4qECyaTznkonKReRGpxXafJdxrIDgJie7xuBFRLHYPlm"
    
    # Trading Pair
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"  # For klines (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    
    # Realnet Safeguards (only active when use_testnet=False)
    max_position_size_pct: float = 0.5  # Reduce position size by 50% in realnet
    max_exposure_pct: float = 10.0  # Max total exposure as % of equity
    enable_kill_switch: bool = True  # Enable kill switch
    kill_switch_file: str = "kill_switch.txt"  # File to check for kill signal
    
    # Execution Parameters (from BacktestConfig)
    execution_mode: str = "next_open"  # "next_open" or "next_close"
    fee_pct: float = 0.1  # Binance spot trading fee (0.1% default)
    slippage_atr_mult: float = 0.0  # Slippage as ATR multiplier (0.0 = no slippage modeling)
    
    # Position Sizing
    risk_per_trade: float = 0.01  # Risk per trade as fraction (0.01 = 1%)
    max_position_value_pct: Optional[float] = 2.0  # Max position value as % of equity (None = no cap)
    
    # Volatility Capping
    atr_percentile_window: int = 100  # Window for rolling ATR percentile
    atr_percentile_threshold: float = 95.0  # Percentile threshold (e.g., 95.0 for 95th percentile)
    cap_stop_loss: bool = True  # Cap stop loss distances using ATR percentile
    cap_take_profit: bool = True  # Cap take profit distances using ATR percentile
    
    # Risk Controls
    per_zone_cooldown_candles: int = 0  # Cooldown candles after exiting trade from zone
    max_daily_risk_pct: Optional[float] = None  # Max daily risk as % of equity (None = no limit)
    max_daily_trades: Optional[int] = None  # Max trades per day (None = no limit)
    max_daily_loss_pct: Optional[float] = None  # Max daily loss as % of equity (None = no limit)
    
    # Capital Accounting
    enable_unrealized_pnl: bool = True  # Track unrealized PnL for equity calculation (enabled for live)
    
    # Signal Filters
    enable_rsi_filter: bool = False  # Enable RSI-based entry/exit filtering
    rsi_period: int = 14  # RSI calculation period
    rsi_oversold_threshold: float = 30.0  # RSI oversold threshold
    rsi_overbought_threshold: float = 70.0  # RSI overbought threshold
    rsi_lookback_candles: int = 5  # Lookback candles for RSI recent touch check
    
    # Strategy Parameters (from forward_test.py)
    rebound_stop_atr_mult: float = 0.5  # Stop loss buffer for rebound trades
    breakout_stop_atr_mult: float = 0.5  # Stop loss buffer for breakout trades
    take_profit_r_multiple: float = 2.0  # Take profit R-multiple (2R = 2x risk)
    cooldown_candles: int = 0  # No cooldown period
    
    # SR Signal Parameters
    window_size: int = 192  # Context window size
    atr_period: int = 14  # ATR calculation period
    atr_method: str = "wilder"  # ATR calculation method
    left_bars: int = 3  # Left bars for pivot detection
    right_bars: int = 3  # Right bars for pivot detection
    cluster_atr_mult: float = 0.85  # Cluster ATR multiplier
    zone_width_atr_mult: float = 0.25  # Zone width ATR multiplier
    min_pivots_per_zone: int = 3  # Minimum pivots per zone
    rejection_body_ratio: float = 0.3  # Rejection body ratio
    breakout_buffer_atr_mult: float = 0.2  # Breakout buffer ATR multiplier
    rebound_confirmation_candles: int = 2  # Rebound confirmation candles
    breakout_confirmation_candles: int = 2  # Breakout confirmation candles
    
    # State Recovery
    state_recovery_enabled: bool = True
    
    # Balance Override (for testing/consistent sizing)
    override_initial_balance: Optional[float] = 100.0  # If set, use this instead of actual balance (default: $100)
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "live_trading.log"
    trades_log_file: str = "live_trades.csv"
    
    # WebSocket Settings
    websocket_reconnect_delay: int = 5  # Seconds to wait before reconnecting
    websocket_max_reconnect_attempts: int = 10  # Max reconnection attempts
    websocket_ping_interval: float = 20.0  # Ping interval in seconds (websocket keepalive)
    websocket_ping_timeout: float = 10.0  # Ping timeout in seconds (must be < ping_interval)
    websocket_connection_timeout: float = 30.0  # Connection timeout in seconds
    websocket_close_timeout: float = 10.0  # Close timeout in seconds
    websocket_max_queue_size: int = 500  # Max queue size for incoming messages
    
    # Position Management
    close_positions_on_start: bool = True  # Close all positions on startup
    position_backup_file: str = "positions_backup.json"  # File to backup positions
    
    # Dashboard Settings
    enable_dashboard: bool = False  # Enable terminal dashboard
    dashboard_refresh_rate: float = 0.1  # Dashboard refresh rate in seconds (real-time)
    
    # Order Settings
    order_timeout_seconds: int = 30  # Timeout for order execution
    order_retry_attempts: int = 3  # Number of retry attempts for failed orders
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'use_testnet': self.use_testnet,
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'max_position_size_pct': self.max_position_size_pct,
            'max_exposure_pct': self.max_exposure_pct,
            'enable_kill_switch': self.enable_kill_switch,
            'kill_switch_file': self.kill_switch_file,
            'execution_mode': self.execution_mode,
            'fee_pct': self.fee_pct,
            'slippage_atr_mult': self.slippage_atr_mult,
            'risk_per_trade': self.risk_per_trade,
            'max_position_value_pct': self.max_position_value_pct,
            'atr_percentile_window': self.atr_percentile_window,
            'atr_percentile_threshold': self.atr_percentile_threshold,
            'cap_stop_loss': self.cap_stop_loss,
            'cap_take_profit': self.cap_take_profit,
            'per_zone_cooldown_candles': self.per_zone_cooldown_candles,
            'max_daily_risk_pct': self.max_daily_risk_pct,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'enable_unrealized_pnl': self.enable_unrealized_pnl,
            'enable_rsi_filter': self.enable_rsi_filter,
            'rsi_period': self.rsi_period,
            'rsi_oversold_threshold': self.rsi_oversold_threshold,
            'rsi_overbought_threshold': self.rsi_overbought_threshold,
            'rsi_lookback_candles': self.rsi_lookback_candles,
            'rebound_stop_atr_mult': self.rebound_stop_atr_mult,
            'breakout_stop_atr_mult': self.breakout_stop_atr_mult,
            'take_profit_r_multiple': self.take_profit_r_multiple,
            'cooldown_candles': self.cooldown_candles,
            'window_size': self.window_size,
            'atr_period': self.atr_period,
            'atr_method': self.atr_method,
            'left_bars': self.left_bars,
            'right_bars': self.right_bars,
            'cluster_atr_mult': self.cluster_atr_mult,
            'zone_width_atr_mult': self.zone_width_atr_mult,
            'min_pivots_per_zone': self.min_pivots_per_zone,
            'rejection_body_ratio': self.rejection_body_ratio,
            'breakout_buffer_atr_mult': self.breakout_buffer_atr_mult,
            'rebound_confirmation_candles': self.rebound_confirmation_candles,
            'breakout_confirmation_candles': self.breakout_confirmation_candles,
            'state_recovery_enabled': self.state_recovery_enabled,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'trades_log_file': self.trades_log_file,
            'websocket_reconnect_delay': self.websocket_reconnect_delay,
            'websocket_max_reconnect_attempts': self.websocket_max_reconnect_attempts,
            'websocket_ping_interval': self.websocket_ping_interval,
            'websocket_ping_timeout': self.websocket_ping_timeout,
            'websocket_connection_timeout': self.websocket_connection_timeout,
            'websocket_close_timeout': self.websocket_close_timeout,
            'websocket_max_queue_size': self.websocket_max_queue_size,
            'close_positions_on_start': self.close_positions_on_start,
            'position_backup_file': self.position_backup_file,
            'enable_dashboard': self.enable_dashboard,
            'dashboard_refresh_rate': self.dashboard_refresh_rate,
            'order_timeout_seconds': self.order_timeout_seconds,
            'order_retry_attempts': self.order_retry_attempts,
            'override_initial_balance': self.override_initial_balance
        }
    
    @classmethod
    def from_env(cls) -> 'LiveTradingConfig':
        """Create config from environment variables."""
        import os
        
        config = cls()
        
        # Binance connection
        testnet_env = os.getenv('BINANCE_TESTNET', '').lower()
        config.use_testnet = testnet_env in ('true', '1', 'yes')
        config.api_key = os.getenv('BINANCE_API_KEY', '')
        config.api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        # Trading pair
        config.symbol = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
        config.timeframe = os.getenv('TRADING_TIMEFRAME', '1h')
        
        return config

