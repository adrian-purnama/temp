# -*- coding: utf-8 -*-
"""
Forward test script for SR Signal module.

Implements walk-forward simulation matching the notebook's approach:
- Processes candles sequentially
- Uses SR signal module for signal generation
- Executes trades using trade state machine
- Reports comprehensive performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import date
from trader.signals.sr_signal import (
    add_atr, get_structure_context, detect_pivots, cluster_pivots_to_zones,
    classify_zone_interactions, process_interaction_with_confirmation,
    update_zone_metadata
)
from trader.backtesting import (
    ExecutionEngine, RiskManager, CapitalAccountant, RiskController,
    SignalFilter, BacktestConfig
)


# ============================================================================
# Configuration Constants (matching notebook)
# ============================================================================

# Trade Execution Parameters
EXECUTION_MODE = "next_open"  # Execute entry at current candle's open
REBOUND_STOP_ATR_MULT = 0.5  # Stop loss buffer for rebound trades
BREAKOUT_STOP_ATR_MULT = 0.5  # Stop loss buffer for breakout trades
TAKE_PROFIT_R_MULTIPLE = 2.0  # Take profit R-multiple (2R = 2x risk)
RISK_PER_TRADE = 0.01  # Risk 1% of capital per trade
COOLDOWN_CANDLES = 0  # No cooldown period

# Simulation Configuration
SIMULATION_CONFIG = {
    'WINDOW_SIZE': 192,
    'ATR_PERIOD': 14,
    'ATR_METHOD': 'wilder',
    'LEFT_BARS': 3,
    'RIGHT_BARS': 3,
    'CLUSTER_ATR_MULT': 0.85,
    'ZONE_WIDTH_ATR_MULT': 0.25,
    'MIN_PIVOTS_PER_ZONE': 3,
    'REJECTION_BODY_RATIO': 0.3,
    'BREAKOUT_BUFFER_ATR_MULT': 0.2,
    'REBOUND_CONFIRMATION_CANDLES': 2,
    'BREAKOUT_CONFIRMATION_CANDLES': 2,
    'EXECUTION_MODE': EXECUTION_MODE,
    'REBOUND_STOP_ATR_MULT': REBOUND_STOP_ATR_MULT,
    'BREAKOUT_STOP_ATR_MULT': BREAKOUT_STOP_ATR_MULT,
    'TAKE_PROFIT_R_MULTIPLE': TAKE_PROFIT_R_MULTIPLE,
    'RISK_PER_TRADE': RISK_PER_TRADE,
    'INITIAL_BALANCE': 10000,
    'COOLDOWN_CANDLES': COOLDOWN_CANDLES,
    # Enhanced backtesting parameters
    'FEE_PCT': 0.0,  # Transaction fee percentage (0.0 = no fees)
    'SLIPPAGE_ATR_MULT': 0.0,  # Slippage as ATR multiplier (0.0 = no slippage)
    'MAX_POSITION_VALUE_PCT': 5.0,  # Max position value as % of equity (50% = realistic cap to prevent exponential growth)
    'ATR_PERCENTILE_WINDOW': 100,  # Window for rolling ATR percentile
    'ATR_PERCENTILE_THRESHOLD': 95.0,  # Percentile threshold for ATR capping
    'CAP_STOP_LOSS': True,  # Cap stop loss distances using ATR percentile
    'CAP_TAKE_PROFIT': True,  # Cap take profit distances using ATR percentile
    'PER_ZONE_COOLDOWN_CANDLES': 0,  # Cooldown candles after exiting trade from zone
    'MAX_DAILY_RISK_PCT': None,  # Max daily risk as % of equity (None = no limit)
    'MAX_DAILY_TRADES': None,  # Max trades per day (None = no limit)
    'MAX_DAILY_LOSS_PCT': None,  # Max daily loss as % of equity (None = no limit)
    'ENABLE_UNREALIZED_PNL': False,  # Track unrealized PnL for equity calculation
    'ENABLE_RSI_FILTER': True,  # Enable RSI-based entry/exit filtering
    'RSI_PERIOD': 14,  # RSI calculation period
    'RSI_OVERSOLD_THRESHOLD': 30.0,  # RSI oversold threshold
    'RSI_OVERBOUGHT_THRESHOLD': 70.0,  # RSI overbought threshold
    'RSI_LOOKBACK_CANDLES': 5  # Lookback candles for RSI recent touch check
}

# Trade State Constants
STATE_IDLE = "IDLE"
STATE_SETUP_CONFIRMED = "SETUP_CONFIRMED"
STATE_IN_TRADE = "IN_TRADE"
STATE_COOLDOWN = "COOLDOWN"


# ============================================================================
# Data Loading
# ============================================================================

BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def parse_epoch(x):
    """Parse epoch timestamp to datetime."""
    x = int(x)
    if x > 1e17:
        return pd.to_datetime(x, unit="ns", utc=True)
    elif x > 1e14:
        return pd.to_datetime(x, unit="us", utc=True)
    elif x > 1e11:
        return pd.to_datetime(x, unit="ms", utc=True)
    else:
        return pd.to_datetime(x, unit="s", utc=True)


def parse_timestamp(value):
    """Parse timestamp that could be epoch (int), datetime string, or already a datetime."""
    # If already a datetime/Timestamp, return as-is
    if isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
        return value if value.tz is not None else value.tz_localize('UTC')
    if pd.api.types.is_datetime64_any_dtype(type(value)):
        return pd.to_datetime(value, utc=True)
    
    # Try parsing as datetime string first
    try:
        result = pd.to_datetime(value, utc=True)
        if pd.notna(result):
            return result
    except (ValueError, TypeError):
        pass
    
    # Try parsing as epoch
    try:
        return parse_epoch(value)
    except (ValueError, TypeError):
        pass
    
    # If both fail, return as-is and let pandas handle it
    return pd.to_datetime(value, errors='coerce', utc=True)


def load_binance_csv(path: str) -> pd.DataFrame:
    """
    Load Binance klines CSV file.
    
    Supports two formats:
    1. No header with epoch timestamps (combine.csv)
    2. Header row with datetime strings (combine2.csv)
    """
    # Try to detect if file has header by reading first line
    with open(path, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if first line looks like a header (contains text, not just numbers)
    has_header = False
    if first_line:
        # Check if first column of first line is numeric (epoch) or text (header)
        first_col = first_line.split(',')[0].strip()
        try:
            float(first_col)
            # If it's numeric, likely no header
            has_header = False
        except ValueError:
            # If it's not numeric, likely a header
            has_header = True
    
    # Read CSV with or without header
    if has_header:
        # Read with header and map column names
        df = pd.read_csv(path)
        
        # Create mapping from various column name formats to standard names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'open time' in col_lower or 'open_time' in col_lower:
                col_mapping[col] = 'open_time'
            elif col_lower == 'open':
                col_mapping[col] = 'open'
            elif col_lower == 'high':
                col_mapping[col] = 'high'
            elif col_lower == 'low':
                col_mapping[col] = 'low'
            elif col_lower == 'close':
                col_mapping[col] = 'close'
            elif col_lower == 'volume':
                col_mapping[col] = 'volume'
        
        df = df.rename(columns=col_mapping)
    else:
        # Read without header
        df = pd.read_csv(path, header=None, names=BINANCE_COLS)
    
    # Ensure required columns exist
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}. Found columns: {list(df.columns)}")
    
    # Convert numeric columns
    num_cols = ["open", "high", "low", "close", "volume"]
    df[num_cols] = df[num_cols].astype(float)
    
    # Parse datetime column (handles both epoch and datetime strings)
    df["datetime"] = df["open_time"].apply(parse_timestamp)
    
    df = (
        df
        .dropna(subset=["datetime"])
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .set_index("datetime")
    )
    
    df = df[["open", "high", "low", "close", "volume"]]
    return df


# ============================================================================
# Trade Execution Functions (matching notebook)
# ============================================================================

def calculate_risk_parameters(confirmed_setup, zone, entry_price, atr_value,
                              r_multiple=TAKE_PROFIT_R_MULTIPLE,
                              rebound_stop_atr_mult=REBOUND_STOP_ATR_MULT,
                              breakout_stop_atr_mult=BREAKOUT_STOP_ATR_MULT,
                              risk_manager=None):
    """Calculate stop loss and take profit prices for a confirmed setup."""
    # Cap ATR if risk manager provided
    if risk_manager is not None:
        atr_value = risk_manager.cap_atr_value(atr_value)
    
    setup_type = confirmed_setup['event_type']
    direction = confirmed_setup['direction']
    zone_type = zone['zone_type']
    lower_boundary = zone['lower_boundary']
    upper_boundary = zone['upper_boundary']
    
    # Calculate stop loss
    if 'rebound' in setup_type:
        if direction == 'long':
            stop_loss_price = lower_boundary - (rebound_stop_atr_mult * atr_value)
        else:  # short
            stop_loss_price = upper_boundary + (rebound_stop_atr_mult * atr_value)
    elif 'breakout' in setup_type:
        if direction == 'long':
            stop_loss_price = upper_boundary - (breakout_stop_atr_mult * atr_value)
        else:  # short
            stop_loss_price = lower_boundary + (breakout_stop_atr_mult * atr_value)
    else:
        # Default fallback
        if direction == 'long':
            stop_loss_price = entry_price - (2.0 * atr_value)
        else:
            stop_loss_price = entry_price + (2.0 * atr_value)
    
    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    # Calculate take profit using R-multiple
    if direction == 'long':
        take_profit_price = entry_price + (risk_per_unit * r_multiple)
    else:  # short
        take_profit_price = entry_price - (risk_per_unit * r_multiple)
    
    return {
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'risk_per_unit': risk_per_unit
    }


def calculate_position_size(entry_price, stop_loss_price, risk_amount):
    """Calculate position size based on risk amount and stop loss distance."""
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        return 0.0
    
    position_size = risk_amount / risk_per_unit
    return position_size


def execute_trade_entry(confirmed_setup, zone, current_candle, atr_value,
                       execution_mode=EXECUTION_MODE,
                       risk_per_trade=RISK_PER_TRADE,
                       account_balance=10000,
                       account_equity=None,
                       trade_id=None,
                       execution_engine=None,
                       risk_manager=None):
    """Execute trade entry for a confirmed setup."""
    # Use equity for position sizing if provided, otherwise use balance
    sizing_capital = account_equity if account_equity is not None else account_balance
    
    # Determine entry price based on execution mode
    if execution_mode == "next_open":
        base_entry_price = current_candle['open']
    elif execution_mode == "next_close":
        base_entry_price = current_candle['close']
    else:
        base_entry_price = current_candle['open']
    
    # Calculate risk parameters (with ATR capping if risk_manager provided)
    risk_params = calculate_risk_parameters(
        confirmed_setup, zone, base_entry_price, atr_value,
        risk_manager=risk_manager
    )
    
    # Calculate risk amount in currency
    risk_amount = sizing_capital * risk_per_trade
    
    # Calculate position size (with position cap if risk_manager provided)
    if risk_manager is not None:
        position_size = risk_manager.calculate_position_size(
            base_entry_price, risk_params['stop_loss_price'],
            risk_amount, sizing_capital
        )
    else:
        position_size = calculate_position_size(
            base_entry_price, risk_params['stop_loss_price'], risk_amount
        )
    
    # Apply execution engine for effective entry price
    if execution_engine is not None:
        direction = confirmed_setup['direction']
        effective_entry_price = execution_engine.execute_entry(
            base_entry_price, direction, atr_value
        )
    else:
        effective_entry_price = base_entry_price
    
    # Determine setup type
    setup_type = 'rebound' if 'rebound' in confirmed_setup['event_type'] else 'breakout'
    
    # Generate trade ID if not provided
    if trade_id is None:
        timestamp = current_candle.name if hasattr(current_candle, 'name') else None
        if timestamp is not None:
            trade_id = f"trade_{int(timestamp.timestamp() * 1000)}"
        else:
            trade_id = f"trade_{len(str(hash(str(current_candle))))}"
    
    # Create active trade record
    active_trade = {
        'trade_id': trade_id,
        'entry_timestamp': current_candle.name if hasattr(current_candle, 'name') else None,
        'entry_price': effective_entry_price,  # Use effective price with fees/slippage
        'base_entry_price': base_entry_price,  # Store base price for reference
        'direction': confirmed_setup['direction'],
        'zone_index': confirmed_setup['zone_index'],
        'zone_type': zone['zone_type'],
        'setup_type': setup_type,
        'stop_loss_price': risk_params['stop_loss_price'],
        'take_profit_price': risk_params['take_profit_price'],
        'position_size': position_size,
        'risk_amount': risk_amount,
        'risk_per_unit': risk_params['risk_per_unit']
    }
    
    return active_trade


def evaluate_trade_exit(active_trade, current_candle, atr_value=None,
                       execution_engine=None, signal_filter=None, context_df=None):
    """Evaluate if stop loss, RSI exit, or take profit was hit on current candle."""
    stop_loss_price = active_trade['stop_loss_price']
    take_profit_price = active_trade['take_profit_price']
    direction = active_trade['direction']
    
    candle_high = current_candle['high']
    candle_low = current_candle['low']
    
    exit_triggered = False
    exit_price = None
    exit_reason = None
    
    # Priority 1: Check stop loss (always highest priority)
    if direction == 'long':
        stop_hit = candle_low <= stop_loss_price
    else:  # short
        stop_hit = candle_high >= stop_loss_price
    
    if stop_hit:
        exit_triggered = True
        exit_price = stop_loss_price
        exit_reason = 'stop_loss'
    
    # Priority 2: Check RSI exit (only if stop loss not hit and filter enabled)
    elif signal_filter is not None and context_df is not None:
        rsi_exit = signal_filter.check_rsi_exit_filter(
            active_trade, current_candle, context_df
        )
        if rsi_exit:
            exit_triggered = True
            # Use current close for RSI exit
            exit_price = current_candle['close']
            exit_reason = 'rsi_exit'
    
    # Priority 3: Check take profit (only if stop loss and RSI exit not hit)
    if not exit_triggered:
        if direction == 'long':
            target_hit = candle_high >= take_profit_price
        else:  # short
            target_hit = candle_low <= take_profit_price
        
        if target_hit:
            exit_triggered = True
            exit_price = take_profit_price
            exit_reason = 'take_profit'
    
    # Apply execution engine for effective exit price
    if exit_triggered and execution_engine is not None and atr_value is not None:
        effective_exit_price = execution_engine.execute_exit(
            exit_price, direction, atr_value
        )
        return exit_triggered, effective_exit_price, exit_reason
    
    return exit_triggered, exit_price, exit_reason


def process_trade_state(current_candle, trade_state_manager, confirmed_setup_event,
                        zones_df, atr_value, current_index, context_df=None,
                        execution_mode=EXECUTION_MODE,
                        risk_per_trade=RISK_PER_TRADE,
                        account_balance=10000,
                        account_equity=None,
                        cooldown_candles=COOLDOWN_CANDLES,
                        execution_engine=None,
                        risk_manager=None,
                        risk_controller=None,
                        signal_filter=None,
                        current_date=None,
                        initial_equity=None):
    """Process trade state machine for current candle."""
    updated_state = trade_state_manager.copy()
    current_state = updated_state.get('current_state', STATE_IDLE)
    active_trade = updated_state.get('active_trade', None)
    cooldown_remaining = updated_state.get('cooldown_remaining', 0)
    
    trade_record = None
    sizing_capital = account_equity if account_equity is not None else account_balance
    
    # Process based on current state
    if current_state == STATE_IDLE:
        # IDLE: Check for confirmed setup event
        if confirmed_setup_event is not None:
            # Apply RSI entry filter if enabled
            if signal_filter is not None and context_df is not None:
                if not signal_filter.filter_entry_signal(confirmed_setup_event, context_df):
                    # RSI filter blocked entry
                    return updated_state, None, current_index
            
            # Check risk controls before entry
            if risk_controller is not None:
                zone_idx = confirmed_setup_event['zone_index']
                
                # Check zone cooldown
                if not risk_controller.check_zone_cooldown(zone_idx, current_index):
                    return updated_state, None, current_index
                
                # Check daily limits (need to calculate risk amount first)
                # We'll do a preliminary check, full check happens in SETUP_CONFIRMED
                if current_date is not None:
                    if not risk_controller.check_daily_trade_limit(current_date):
                        return updated_state, None, current_index
                    
                    # Daily loss limit check uses current equity as baseline if first trade of day
                    if not risk_controller.check_daily_loss_limit(
                        sizing_capital, sizing_capital, current_date
                    ):
                        return updated_state, None, current_index
            
            # Transition to SETUP_CONFIRMED
            updated_state['current_state'] = STATE_SETUP_CONFIRMED
            updated_state['pending_setup'] = confirmed_setup_event
            updated_state['active_trade'] = None
            updated_state['cooldown_remaining'] = 0
    
    elif current_state == STATE_SETUP_CONFIRMED:
        # SETUP_CONFIRMED: Execute trade entry
        pending_setup = updated_state.get('pending_setup')
        if pending_setup is not None:
            zone_idx = pending_setup['zone_index']
            
            try:
                if zone_idx in zones_df.index:
                    zone = zones_df.loc[zone_idx]
                elif isinstance(zone_idx, (int, np.integer)) and 0 <= zone_idx < len(zones_df):
                    zone = zones_df.iloc[zone_idx]
                else:
                    zone = None
                
                if zone is not None:
                    # Calculate preliminary risk amount for daily risk check
                    if risk_controller is not None and current_date is not None:
                        base_price = current_candle['open'] if execution_mode == "next_open" else current_candle['close']
                        temp_risk_params = calculate_risk_parameters(
                            pending_setup, zone, base_price, atr_value, risk_manager=risk_manager
                        )
                        temp_risk_amount = sizing_capital * risk_per_trade
                        
                        # Check daily risk limit
                        if not risk_controller.check_daily_risk_limit(
                            temp_risk_amount, sizing_capital, current_date
                        ):
                            # Daily risk limit exceeded, cancel entry
                            updated_state['current_state'] = STATE_IDLE
                            updated_state['pending_setup'] = None
                            return updated_state, None, current_index
                    
                    # Execute trade entry
                    active_trade = execute_trade_entry(
                        pending_setup, zone, current_candle, atr_value,
                        execution_mode=execution_mode,
                        risk_per_trade=risk_per_trade,
                        account_balance=account_balance,
                        account_equity=sizing_capital,
                        execution_engine=execution_engine,
                        risk_manager=risk_manager
                    )
                    
                    # Transition to IN_TRADE
                    updated_state['current_state'] = STATE_IN_TRADE
                    updated_state['active_trade'] = active_trade
                    updated_state['pending_setup'] = None
            except (KeyError, IndexError, ValueError):
                # Zone not found, invalidate setup
                updated_state['current_state'] = STATE_IDLE
                updated_state['pending_setup'] = None
    
    elif current_state == STATE_IN_TRADE:
        # IN_TRADE: Check for exit
        if active_trade is not None:
            exit_triggered, exit_price, exit_reason = evaluate_trade_exit(
                active_trade, current_candle, atr_value,
                execution_engine=execution_engine,
                signal_filter=signal_filter,
                context_df=context_df
            )
            
            if exit_triggered:
                # Calculate trade metrics
                entry_price = active_trade['entry_price']
                position_size = active_trade['position_size']
                direction = active_trade['direction']
                risk_amount = active_trade['risk_amount']
                risk_per_unit = active_trade['risk_per_unit']
                zone_idx = active_trade['zone_index']
                
                # Calculate PnL
                if direction == 'long':
                    pnl = (exit_price - entry_price) * position_size
                else:  # short
                    pnl = (entry_price - exit_price) * position_size
                
                # Subtract fees if execution engine provided
                if execution_engine is not None:
                    total_fees = execution_engine.calculate_total_fees(
                        entry_price, exit_price, position_size
                    )
                    pnl -= total_fees
                
                # Calculate R-multiple
                r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
                
                # Calculate PnL percentage
                pnl_pct = (pnl / account_balance) * 100 if account_balance > 0 else 0.0
                
                # Create trade record
                trade_record = {
                    'trade_id': active_trade['trade_id'],
                    'entry_timestamp': active_trade['entry_timestamp'],
                    'exit_timestamp': current_candle.name if hasattr(current_candle, 'name') else None,
                    'entry_index': active_trade.get('entry_index', current_index),
                    'exit_index': current_index,
                    'side': direction,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'stop_loss_price': active_trade['stop_loss_price'],
                    'take_profit_price': active_trade['take_profit_price'],
                    'position_size': position_size,
                    'risk_amount': risk_amount,
                    'risk_per_unit': risk_per_unit,
                    'r_multiple': r_multiple,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'setup_type': active_trade['setup_type']
                }
                
                # Update risk controller trackers
                if risk_controller is not None and current_date is not None:
                    risk_controller.update_daily_trackers(
                        zone_idx, risk_amount, pnl, current_date, current_index
                    )
                
                # Transition to COOLDOWN or IDLE
                if cooldown_candles > 0:
                    updated_state['current_state'] = STATE_COOLDOWN
                    updated_state['cooldown_remaining'] = cooldown_candles
                else:
                    updated_state['current_state'] = STATE_IDLE
                    updated_state['cooldown_remaining'] = 0
                
                updated_state['active_trade'] = None
    
    elif current_state == STATE_COOLDOWN:
        # COOLDOWN: Decrement cooldown counter
        if cooldown_remaining > 0:
            updated_state['cooldown_remaining'] = cooldown_remaining - 1
        else:
            updated_state['current_state'] = STATE_IDLE
            updated_state['cooldown_remaining'] = 0
    
    return updated_state, trade_record, current_index


# ============================================================================
# Simulation Functions
# ============================================================================

def initialize_simulation(df, config):
    """Initialize simulation state and validate data."""
    required_cols = ['open', 'high', 'low', 'close', 'atr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    window_size = config['WINDOW_SIZE']
    atr_period = config['ATR_PERIOD']
    right_bars = config['RIGHT_BARS']
    
    start_index = window_size + max(atr_period, right_bars)
    end_index = len(df)
    
    if start_index >= end_index:
        raise ValueError(f"Insufficient data: start_index ({start_index}) >= end_index ({end_index})")
    
    trade_state = {
        'current_state': STATE_IDLE,
        'active_trade': None,
        'pending_setup': None,
        'cooldown_remaining': 0
    }
    
    # Initialize backtesting components
    backtest_config = BacktestConfig(
        execution_mode=config.get('EXECUTION_MODE', 'next_open'),
        fee_pct=config.get('FEE_PCT', 0.0),
        slippage_atr_mult=config.get('SLIPPAGE_ATR_MULT', 0.0),
        risk_per_trade=config.get('RISK_PER_TRADE', 0.01),
        max_position_value_pct=config.get('MAX_POSITION_VALUE_PCT'),
        atr_percentile_window=config.get('ATR_PERCENTILE_WINDOW', 100),
        atr_percentile_threshold=config.get('ATR_PERCENTILE_THRESHOLD', 95.0),
        cap_stop_loss=config.get('CAP_STOP_LOSS', True),
        cap_take_profit=config.get('CAP_TAKE_PROFIT', True),
        per_zone_cooldown_candles=config.get('PER_ZONE_COOLDOWN_CANDLES', 0),
        max_daily_risk_pct=config.get('MAX_DAILY_RISK_PCT'),
        max_daily_trades=config.get('MAX_DAILY_TRADES'),
        max_daily_loss_pct=config.get('MAX_DAILY_LOSS_PCT'),
        enable_unrealized_pnl=config.get('ENABLE_UNREALIZED_PNL', False),
        enable_rsi_filter=config.get('ENABLE_RSI_FILTER', False),
        rsi_period=config.get('RSI_PERIOD', 14),
        rsi_oversold_threshold=config.get('RSI_OVERSOLD_THRESHOLD', 30.0),
        rsi_overbought_threshold=config.get('RSI_OVERBOUGHT_THRESHOLD', 70.0),
        rsi_lookback_candles=config.get('RSI_LOOKBACK_CANDLES', 5)
    )
    
    execution_engine = ExecutionEngine(
        fee_pct=backtest_config.fee_pct,
        slippage_atr_mult=backtest_config.slippage_atr_mult
    )
    
    risk_manager = RiskManager(
        max_position_value_pct=backtest_config.max_position_value_pct,
        atr_percentile_window=backtest_config.atr_percentile_window,
        atr_percentile_threshold=backtest_config.atr_percentile_threshold,
        cap_stop_loss=backtest_config.cap_stop_loss,
        cap_take_profit=backtest_config.cap_take_profit
    )
    
    capital_accountant = CapitalAccountant(
        enable_unrealized_pnl=backtest_config.enable_unrealized_pnl
    )
    
    risk_controller = RiskController(
        per_zone_cooldown_candles=backtest_config.per_zone_cooldown_candles,
        max_daily_risk_pct=backtest_config.max_daily_risk_pct,
        max_daily_trades=backtest_config.max_daily_trades,
        max_daily_loss_pct=backtest_config.max_daily_loss_pct
    )
    
    signal_filter = None
    if backtest_config.enable_rsi_filter:
        signal_filter = SignalFilter(
            rsi_period=backtest_config.rsi_period,
            rsi_oversold_threshold=backtest_config.rsi_oversold_threshold,
            rsi_overbought_threshold=backtest_config.rsi_overbought_threshold,
            rsi_lookback_candles=backtest_config.rsi_lookback_candles
        )
    
    simulation_state = {
        'start_index': start_index,
        'end_index': end_index,
        'account_balance': config['INITIAL_BALANCE'],
        'initial_balance': config['INITIAL_BALANCE'],
        'trade_state': trade_state,
        'active_waiting_state': None,
        'trades_log': [],
        'confirmed_setups_count': 0,
        'invalidated_setups_count': 0,
        'ignored_setups_count': 0,
        'equity_history': [],
        'backtest_config': backtest_config,
        'execution_engine': execution_engine,
        'risk_manager': risk_manager,
        'capital_accountant': capital_accountant,
        'risk_controller': risk_controller,
        'signal_filter': signal_filter
    }
    
    return simulation_state


def run_walk_forward_simulation(df, config):
    """Execute full walk-forward simulation through entire dataset."""
    # Initialize simulation state
    sim_state = initialize_simulation(df, config)
    
    start_index = sim_state['start_index']
    end_index = sim_state['end_index']
    
    print(f"Starting walk-forward simulation...")
    print(f"  Start index: {start_index} ({df.index[start_index]})")
    print(f"  End index: {end_index} ({df.index[end_index-1]})")
    print(f"  Total candles to process: {end_index - start_index}")
    print(f"  Initial balance: ${config['INITIAL_BALANCE']:.2f}")
    
    # Process each candle sequentially
    for t in range(start_index, end_index):
        # Get context window (strictly no future data)
        context_df = df.iloc[t - config['WINDOW_SIZE']:t].copy()
        current_candle = df.iloc[t]
        
        # Skip if context window is too small
        if len(context_df) < config['WINDOW_SIZE']:
            continue
        
        # Get current ATR value
        atr_value = context_df['atr'].iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            atr_value = (current_candle['high'] - current_candle['low']) * 0.5
        
        # Get previous candle's close for direction checks
        previous_candle = df.iloc[t-1] if t > 0 else None
        previous_close = previous_candle['close'] if previous_candle is not None else None
        
        # Step 1: Detect pivots
        pivots = detect_pivots(context_df, 
                              left_bars=config['LEFT_BARS'],
                              right_bars=config['RIGHT_BARS'])
        
        # Step 2: Cluster pivots into zones
        zones = cluster_pivots_to_zones(pivots, context_df,
                                        cluster_atr_mult=config['CLUSTER_ATR_MULT'],
                                        zone_width_atr_mult=config['ZONE_WIDTH_ATR_MULT'],
                                        min_pivots=config['MIN_PIVOTS_PER_ZONE'])
        
        # Step 3: Classify interactions for current candle
        interactions = classify_zone_interactions(current_candle, zones, atr_value,
                                                rejection_body_ratio=config['REJECTION_BODY_RATIO'],
                                                breakout_buffer_atr_mult=config['BREAKOUT_BUFFER_ATR_MULT'])
        
        # Step 4: Process interactions with confirmation logic
        updated_waiting_state, confirmation_events = process_interaction_with_confirmation(
            current_candle, interactions, sim_state['active_waiting_state'],
            zones, atr_value, t,
            rebound_confirmation_candles=config['REBOUND_CONFIRMATION_CANDLES'],
            breakout_confirmation_candles=config['BREAKOUT_CONFIRMATION_CANDLES'],
            previous_close=previous_close
        )
        
        sim_state['active_waiting_state'] = updated_waiting_state
        
        # Track confirmed/invalidated setups
        if confirmation_events:
            for event in confirmation_events:
                if 'confirmed' in event['event_type']:
                    sim_state['confirmed_setups_count'] += 1
                else:
                    sim_state['invalidated_setups_count'] += 1
        
        # Step 5: Calculate equity with unrealized PnL (if enabled)
        current_price = current_candle['close']
        account_equity = sim_state['capital_accountant'].calculate_equity(
            sim_state['account_balance'],
            sim_state['trade_state'].get('active_trade'),
            current_price
        )
        
        # Get current date for daily limit tracking
        current_date = None
        if hasattr(current_candle, 'name') and current_candle.name is not None:
            if isinstance(current_candle.name, pd.Timestamp):
                current_date = current_candle.name.date()
            elif isinstance(current_candle.name, date):
                current_date = current_candle.name
        
        # Step 6: Process trade state machine
        confirmed_setup = confirmation_events[0] if confirmation_events else None
        trade_state, trade_record, _ = process_trade_state(
            current_candle, sim_state['trade_state'], confirmed_setup, zones, atr_value, t,
            context_df=context_df,
            execution_mode=config['EXECUTION_MODE'],
            risk_per_trade=config['RISK_PER_TRADE'],
            account_balance=sim_state['account_balance'],
            account_equity=account_equity,
            cooldown_candles=config['COOLDOWN_CANDLES'],
            execution_engine=sim_state['execution_engine'],
            risk_manager=sim_state['risk_manager'],
            risk_controller=sim_state['risk_controller'],
            signal_filter=sim_state['signal_filter'],
            current_date=current_date,
            initial_equity=sim_state['initial_balance']
        )
        
        sim_state['trade_state'] = trade_state
        
        # Log completed trades
        if trade_record is not None:
            sim_state['trades_log'].append(trade_record)
            sim_state['account_balance'] += trade_record['pnl']
            
            # Recalculate equity after trade
            account_equity = sim_state['capital_accountant'].calculate_equity(
                sim_state['account_balance'],
                None,  # No active trade after exit
                current_price
            )
        
        # Update equity history (with unrealized PnL if enabled)
        unrealized_pnl = sim_state['capital_accountant'].calculate_unrealized_pnl(
            sim_state['trade_state'].get('active_trade'),
            current_price
        )
        timestamp = current_candle.name if hasattr(current_candle, 'name') else None
        if timestamp is not None:
            sim_state['capital_accountant'].update_equity_history(
                timestamp, sim_state['account_balance'], account_equity, unrealized_pnl
            )
            sim_state['equity_history'].append((timestamp, account_equity))
        
        # Progress indicator (every 1000 candles)
        if (t - start_index) % 1000 == 0 and t > start_index:
            print(f"  Processed {t - start_index} candles... ({len(sim_state['trades_log'])} trades)")
    
    print(f"\nSimulation complete!")
    print(f"  Total trades executed: {len(sim_state['trades_log'])}")
    print(f"  Final balance: ${sim_state['account_balance']:.2f}")
    
    # Convert trades log to DataFrame
    if sim_state['trades_log']:
        trades_df = pd.DataFrame(sim_state['trades_log'])
    else:
        trades_df = pd.DataFrame()
    
    return {
        'trades_df': trades_df,
        'simulation_state': sim_state
    }


# ============================================================================
# Performance Metrics
# ============================================================================

def calculate_sharpe_ratio(equity_values, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio from equity curve.
    
    Parameters
    ----------
    equity_values : list or array
        List of equity values over time
    risk_free_rate : float
        Risk-free rate (default 0.0)
    periods_per_year : int
        Number of periods per year for annualization (default 252 for daily)
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if len(equity_values) < 2:
        return 0.0
    
    # Convert to numpy array for easier calculation
    equity_array = np.array(equity_values)
    
    # Calculate returns
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Filter out NaN and infinite values
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate mean and std of returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize: multiply by sqrt(periods_per_year)
    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_performance_metrics(trades_df, initial_balance, simulation_state):
    """Calculate comprehensive performance metrics."""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'average_r': 0.0,
            'median_r': 0.0,
            'expectancy': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'best_trade_r': 0.0,
            'worst_trade_r': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0
        }
    
    winning_trades = trades_df[trades_df['r_multiple'] > 0]
    losing_trades = trades_df[trades_df['r_multiple'] <= 0]
    
    total_trades = len(trades_df)
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0.0
    loss_rate = (len(losing_trades) / total_trades) * 100 if total_trades > 0 else 0.0
    
    average_r = trades_df['r_multiple'].mean()
    median_r = trades_df['r_multiple'].median()
    expectancy = trades_df['r_multiple'].mean()  # Same as average_r
    
    # Profit factor
    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    # Drawdown calculation (use equity history from capital accountant if available)
    capital_accountant = simulation_state.get('capital_accountant')
    if capital_accountant is not None and len(capital_accountant.equity_history) > 0:
        equity_values = capital_accountant.get_equity_series()
    else:
        equity_history = simulation_state.get('equity_history', [])
        equity_values = [eq[1] for eq in equity_history] if equity_history else []
    
    if equity_values:
        peak = equity_values[0]
        max_drawdown = 0.0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            drawdown = ((peak - eq) / peak) * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    else:
        max_drawdown = 0.0
    
    best_trade_r = trades_df['r_multiple'].max()
    worst_trade_r = trades_df['r_multiple'].min()
    
    final_balance = simulation_state['account_balance']
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    # Calculate Sharpe ratio from equity curve
    capital_accountant = simulation_state.get('capital_accountant')
    if capital_accountant is not None and len(capital_accountant.equity_history) > 0:
        equity_values = capital_accountant.get_equity_series()
    else:
        equity_history = simulation_state.get('equity_history', [])
        equity_values = [eq[1] for eq in equity_history] if equity_history else []
    
    sharpe_ratio = calculate_sharpe_ratio(equity_values) if equity_values else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'average_r': average_r,
        'median_r': median_r,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'best_trade_r': best_trade_r,
        'worst_trade_r': worst_trade_r,
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe_ratio
    }


def print_performance_report(performance_metrics, simulation_state):
    """Print comprehensive performance report."""
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)
    
    print("\n1. SIMULATION SUMMARY")
    print("-" * 80)
    print(f"  Total Trades Executed: {performance_metrics['total_trades']}")
    print(f"  Confirmed Setups: {simulation_state.get('confirmed_setups_count', 0)}")
    print(f"  Invalidated Setups: {simulation_state.get('invalidated_setups_count', 0)}")
    print(f"  Final Account Balance: ${simulation_state['account_balance']:.2f}")
    print(f"  Total Return: {performance_metrics['total_return_pct']:.2f}%")
    
    if performance_metrics['total_trades'] == 0:
        print("\n  No trades executed. Cannot compute performance metrics.")
        return
    
    print("\n2. OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"  Win Rate: {performance_metrics['win_rate']:.2f}%")
    print(f"  Loss Rate: {performance_metrics['loss_rate']:.2f}%")
    print(f"  Average R-Multiple: {performance_metrics['average_r']:.2f}R")
    print(f"  Median R-Multiple: {performance_metrics['median_r']:.2f}R")
    print(f"  Expectancy (Mean R): {performance_metrics['expectancy']:.2f}R")
    print(f"  Profit Factor: {performance_metrics['profit_factor']:.2f}")
    
    print("\n3. RISK METRICS")
    print("-" * 80)
    print(f"  Maximum Drawdown: {performance_metrics['max_drawdown']:.2f}%")
    print(f"  Best Trade: {performance_metrics['best_trade_r']:.2f}R")
    print(f"  Worst Trade: {performance_metrics['worst_trade_r']:.2f}R")
    
    print("\n" + "=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Path to combine.csv
    csv_path = "combine2.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find combine.csv at {csv_path}")
        print("Please ensure combine.csv is in the current directory.")
        sys.exit(1)
    
    print("=" * 80)
    print("SR SIGNAL FORWARD TEST")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = load_binance_csv(csv_path)
    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Calculate ATR
    print("Calculating ATR...")
    df_with_atr = add_atr(df, period=SIMULATION_CONFIG['ATR_PERIOD'], 
                          method=SIMULATION_CONFIG['ATR_METHOD'])
    print("ATR calculation complete")
    
    # Run walk-forward simulation
    print("\n" + "=" * 80)
    print("RUNNING WALK-FORWARD SIMULATION")
    print("=" * 80)
    
    results = run_walk_forward_simulation(df_with_atr, SIMULATION_CONFIG)
    
    trades_df = results['trades_df']
    simulation_state = results['simulation_state']
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(
        trades_df, SIMULATION_CONFIG['INITIAL_BALANCE'], simulation_state
    )
    
    # Print performance report
    print_performance_report(performance_metrics, simulation_state)
    
    # Print first 10 trades
    if len(trades_df) > 0:
        print("\nFirst 10 Trades:")
        print("-" * 80)
        for i, trade in trades_df.head(10).iterrows():
            pnl_sign = "+" if trade['pnl'] > 0 else ""
            print(f"\n  Trade {i+1} ({trade['trade_id']}):")
            print(f"    Side: {trade['side']} | Entry: ${trade['entry_price']:.2f} @ {trade['entry_timestamp']}")
            print(f"    Exit: ${trade['exit_price']:.2f} @ {trade['exit_timestamp']} ({trade['exit_reason']})")
            print(f"    R-Multiple: {trade['r_multiple']:.2f}R | Net PnL: {pnl_sign}${trade['pnl']:.2f} ({pnl_sign}{trade['pnl_pct']:.2f}%)")
            print(f"    SL: ${trade['stop_loss_price']:.2f} | TP: ${trade['take_profit_price']:.2f}")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

