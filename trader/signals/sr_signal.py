# -*- coding: utf-8 -*-
"""
Support/Resistance signal module.

Pure signal generation module that detects support/resistance zones and
generates standardized trading signals based on zone interactions and confirmations.

This module contains ALL SR-specific logic (ATR, pivots, zones, confirmations)
and exports the SRSignal class for use in the modular trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from trader.signals.base import BaseSignal


# ============================================================================
# Configuration Constants
# ============================================================================

# Structure Context Window Configuration
WINDOW_SIZE = 192  # 192 candles * 15 minutes = 48 hours of historical data

# Pivot Detection Configuration
LEFT_BARS = 3  # Number of bars before pivot to check
RIGHT_BARS = 3  # Number of bars after pivot to check (prevents repainting)

# Zone Clustering Configuration
CLUSTER_ATR_MULT = 0.85  # Multiplier for clustering distance
ZONE_WIDTH_ATR_MULT = 0.25  # Multiplier for zone boundary width
MIN_PIVOTS_PER_ZONE = 3  # Minimum pivots required to form a zone

# Price-Zone Interaction Classification Configuration
REJECTION_BODY_RATIO = 0.3  # Minimum body/range ratio for rejection
BREAKOUT_BUFFER_ATR_MULT = 0.2  # ATR multiplier for breakout buffer

# Confirmation and Waiting Logic Configuration
REBOUND_CONFIRMATION_CANDLES = 2  # Candles to wait for rebound confirmation
BREAKOUT_CONFIRMATION_CANDLES = 2  # Candles to wait for breakout confirmation


# ============================================================================
# ATR Calculation
# ============================================================================

def _validate_and_prepare_df(df):
    """Validate and standardize input DataFrame for ATR calculation."""
    df_out = df.copy()
    
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df_out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle timestamp
    if df_out.index.name == 'datetime' or isinstance(df_out.index, pd.DatetimeIndex):
        if not isinstance(df_out.index, pd.DatetimeIndex):
            df_out.index = pd.to_datetime(df_out.index)
    elif 'timestamp' in df_out.columns:
        df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])
        df_out = df_out.set_index('timestamp')
    elif 'datetime' in df_out.columns:
        df_out['datetime'] = pd.to_datetime(df_out['datetime'])
        df_out = df_out.set_index('datetime')
    else:
        try:
            df_out.index = pd.to_datetime(df_out.index)
        except (ValueError, TypeError):
            raise ValueError("DataFrame must be indexed by datetime or have a 'timestamp' or 'datetime' column")
    
    # Ensure ascending time order
    if not df_out.index.is_monotonic_increasing:
        df_out = df_out.sort_index()
    
    # Convert OHLC columns to numeric
    for col in required_cols:
        if df_out[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
    
    # Remove duplicate timestamps
    if df_out.index.duplicated().any():
        df_out = df_out[~df_out.index.duplicated(keep='first')]
    
    # Forward-fill missing values only (no backward-fill to avoid lookahead)
    df_out[required_cols] = df_out[required_cols].ffill()
    
    return df_out


def _calculate_true_range(df):
    """Calculate True Range (TR) for each candle."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    prev_close = close.shift(1)
    
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr.iloc[0] = hl.iloc[0]
    
    return tr


def _calculate_atr(tr_series, period, method="wilder"):
    """Calculate Average True Range (ATR) from True Range series."""
    if method == "wilder":
        atr = pd.Series(index=tr_series.index, dtype=float)
        initial_seed = tr_series.iloc[:period].mean()
        atr.iloc[period - 1] = initial_seed
        
        for i in range(period, len(tr_series)):
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr_series.iloc[i]) / period
        
        return atr
    elif method == "sma":
        atr = tr_series.rolling(window=period).mean()
        return atr
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'wilder' or 'sma'")


def add_atr(df, period=14, method="wilder"):
    """Add True Range (TR) and Average True Range (ATR) columns to DataFrame."""
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    if method not in ["wilder", "sma"]:
        raise ValueError(f"method must be 'wilder' or 'sma', got {method}")
    
    df_prepared = _validate_and_prepare_df(df)
    tr = _calculate_true_range(df_prepared)
    atr = _calculate_atr(tr, period=period, method=method)
    
    df_out = df_prepared.copy()
    df_out['tr'] = tr
    df_out['atr'] = atr
    
    return df_out


# ============================================================================
# Structure Context Window
# ============================================================================

def get_structure_context(df, t, window_size=WINDOW_SIZE):
    """Get structure context window for a specific index position."""
    if len(df) < window_size:
        raise ValueError(f"Insufficient data: DataFrame has {len(df)} rows, but window_size requires at least {window_size} rows")
    if t < window_size:
        raise ValueError(f"Index t={t} is too small: must be >= window_size={window_size}")
    if t >= len(df):
        raise ValueError(f"Index t={t} is out of bounds: DataFrame has {len(df)} rows")
    
    context_df = df.iloc[t - window_size : t].copy()
    current_candle = df.iloc[t].copy()
    
    return context_df, current_candle


def iterate_structure_context(df, window_size=WINDOW_SIZE, validate=False):
    """
    Iterate through DataFrame with strict rolling window boundaries.
    
    Generator function that yields context windows and current candles,
    ensuring no forward-looking leakage. Each iteration provides exactly
    window_size candles of history ending at t-1, and the current candle at t.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by timestamp (ascending, already validated).
        Must have at least window_size rows.
    window_size : int, default WINDOW_SIZE
        Size of rolling window (number of historical candles).
    validate : bool, default False
        Whether to perform validation checks. Set to True for debugging.
    
    Yields
    ------
    tuple of (pd.DataFrame, pd.Series, int)
        - context_df: DataFrame slice df.iloc[t-window_size : t] (ends at t-1)
        - current_candle: Series df.iloc[t] (current candle being analyzed)
        - t: Integer index position
    
    Raises
    ------
    ValueError
        If len(df) < window_size (insufficient data)
    
    Notes
    -----
    Iteration starts at t = window_size and continues to t = len(df) - 1.
    This ensures:
    - First iteration has exactly window_size candles of history
    - Last iteration processes the last candle in the DataFrame
    - context_df always ends at t-1 (never includes current_candle)
    - No logic can access candles beyond index t
    """
    # Validate sufficient data
    if len(df) < window_size:
        raise ValueError(
            f"Insufficient data: DataFrame has {len(df)} rows, "
            f"but window_size requires at least {window_size} rows"
        )
    
    # Iterate from t = window_size to len(df) - 1
    # This ensures we always have window_size candles of history
    for t in range(window_size, len(df)):
        # Get context window and current candle
        context_df, current_candle = get_structure_context(df, t, window_size)
        
        # Validation checks (only if validate=True)
        if validate:
            assert len(context_df) == window_size, (
                f"Window size mismatch: expected {window_size}, "
                f"got {len(context_df)} at t={t}"
            )
            assert context_df.index[-1] == df.index[t - 1], (
                f"Context window end mismatch: expected index {t-1} "
                f"({df.index[t-1]}), got {context_df.index[-1]}"
            )
            assert current_candle.name == df.index[t], (
                f"Current candle index mismatch: expected {df.index[t]}, "
                f"got {current_candle.name}"
            )
            assert context_df.index[-1] < current_candle.name, (
                f"Forward-looking leakage detected: context ends at "
                f"{context_df.index[-1]}, current candle is at {current_candle.name}"
            )
        
        yield context_df, current_candle, t


# ============================================================================
# Pivot Detection
# ============================================================================

def detect_pivots(context_df, left_bars=LEFT_BARS, right_bars=RIGHT_BARS):
    """Detect pivot highs and pivot lows within the structure context window."""
    if len(context_df) < left_bars + right_bars + 1:
        return pd.DataFrame(columns=['timestamp', 'price', 'type', 'index_in_context'])
    
    if 'high' not in context_df.columns or 'low' not in context_df.columns:
        raise ValueError("context_df must have 'high' and 'low' columns")
    
    pivots = []
    highs = context_df['high'].values
    lows = context_df['low'].values
    
    for i in range(left_bars, len(context_df) - right_bars):
        window_start = i - left_bars
        window_end = i + right_bars + 1
        
        window_highs = highs[window_start:window_end]
        window_lows = lows[window_start:window_end]
        
        # Check for pivot high
        if highs[i] == np.max(window_highs):
            max_indices = np.where(window_highs == highs[i])[0]
            max_absolute_indices = [window_start + idx for idx in max_indices]
            if i in max_absolute_indices:
                pivots.append({
                    'timestamp': context_df.index[i],
                    'price': highs[i],
                    'type': 'high',
                    'index_in_context': i
                })
        
        # Check for pivot low
        if lows[i] == np.min(window_lows):
            min_indices = np.where(window_lows == lows[i])[0]
            min_absolute_indices = [window_start + idx for idx in min_indices]
            if i in min_absolute_indices:
                pivots.append({
                    'timestamp': context_df.index[i],
                    'price': lows[i],
                    'type': 'low',
                    'index_in_context': i
                })
    
    if pivots:
        pivots_df = pd.DataFrame(pivots)
        pivots_df = pivots_df.sort_values('timestamp').reset_index(drop=True)
    else:
        pivots_df = pd.DataFrame(columns=['timestamp', 'price', 'type', 'index_in_context'])
    
    return pivots_df


# ============================================================================
# Zone Clustering
# ============================================================================

def cluster_pivots_to_zones(pivots_df, context_df, cluster_atr_mult=CLUSTER_ATR_MULT, 
                            zone_width_atr_mult=ZONE_WIDTH_ATR_MULT, min_pivots=MIN_PIVOTS_PER_ZONE):
    """Cluster pivot points into support and resistance zones using ATR-normalized distances."""
    if len(pivots_df) == 0:
        return pd.DataFrame(columns=['center_price', 'lower_boundary', 'upper_boundary',
                                    'pivot_count', 'pivot_indices', 'zone_type', 'atr_value'])
    
    if 'atr' not in context_df.columns:
        raise ValueError("context_df must have 'atr' column")
    
    valid_atr = context_df['atr'].dropna()
    if len(valid_atr) == 0:
        atr_value = context_df['high'].iloc[-1] * 0.01
    else:
        atr_value = valid_atr.iloc[-1]
    
    pivot_highs = pivots_df[pivots_df['type'] == 'high'].copy()
    pivot_lows = pivots_df[pivots_df['type'] == 'low'].copy()
    
    zones = []
    
    # Cluster pivot highs into resistance zones
    if len(pivot_highs) > 0:
        pivot_highs = pivot_highs.sort_values('price', ascending=False).reset_index(drop=True)
        resistance_zones = []
        
        for _, pivot in pivot_highs.iterrows():
            pivot_price = pivot['price']
            pivot_idx = pivot['index_in_context']
            
            assigned = False
            for zone in resistance_zones:
                zone_mean_price = np.mean(zone['pivot_prices'])
                distance_atr = abs(pivot_price - zone_mean_price) / atr_value
                
                if distance_atr <= cluster_atr_mult:
                    zone['pivot_prices'].append(pivot_price)
                    zone['pivot_indices'].append(pivot_idx)
                    assigned = True
                    break
            
            if not assigned:
                resistance_zones.append({
                    'pivot_prices': [pivot_price],
                    'pivot_indices': [pivot_idx],
                    'center_price': None
                })
        
        for zone in resistance_zones:
            zone['center_price'] = np.mean(zone['pivot_prices'])
            zone['lower_boundary'] = zone['center_price'] - zone_width_atr_mult * atr_value
            zone['upper_boundary'] = zone['center_price'] + zone_width_atr_mult * atr_value
            zone['pivot_count'] = len(zone['pivot_prices'])
            zone['zone_type'] = 'resistance'
            zone['atr_value'] = atr_value
            del zone['pivot_prices']
            zones.append(zone)
    
    # Cluster pivot lows into support zones
    if len(pivot_lows) > 0:
        pivot_lows = pivot_lows.sort_values('price', ascending=True).reset_index(drop=True)
        support_zones = []
        
        for _, pivot in pivot_lows.iterrows():
            pivot_price = pivot['price']
            pivot_idx = pivot['index_in_context']
            
            assigned = False
            for zone in support_zones:
                zone_mean_price = np.mean(zone['pivot_prices'])
                distance_atr = abs(pivot_price - zone_mean_price) / atr_value
                
                if distance_atr <= cluster_atr_mult:
                    zone['pivot_prices'].append(pivot_price)
                    zone['pivot_indices'].append(pivot_idx)
                    assigned = True
                    break
            
            if not assigned:
                support_zones.append({
                    'pivot_prices': [pivot_price],
                    'pivot_indices': [pivot_idx],
                    'center_price': None
                })
        
        for zone in support_zones:
            zone['center_price'] = np.mean(zone['pivot_prices'])
            zone['lower_boundary'] = zone['center_price'] - zone_width_atr_mult * atr_value
            zone['upper_boundary'] = zone['center_price'] + zone_width_atr_mult * atr_value
            zone['pivot_count'] = len(zone['pivot_prices'])
            zone['zone_type'] = 'support'
            zone['atr_value'] = atr_value
            del zone['pivot_prices']
            zones.append(zone)
    
    if zones:
        zones_df = pd.DataFrame(zones)
        zones_df = zones_df.sort_values(['zone_type', 'center_price'],
                                       ascending=[True, False]).reset_index(drop=True)
        zones_df = zones_df[zones_df['pivot_count'] >= min_pivots].reset_index(drop=True)
    else:
        zones_df = pd.DataFrame(columns=['center_price', 'lower_boundary', 'upper_boundary',
                                        'pivot_count', 'pivot_indices', 'zone_type', 'atr_value'])
    
    return zones_df


# ============================================================================
# Zone Interaction Classification
# ============================================================================

def classify_zone_interactions(current_candle, zones_df, atr_value, 
                               rejection_body_ratio=REJECTION_BODY_RATIO,
                               breakout_buffer_atr_mult=BREAKOUT_BUFFER_ATR_MULT):
    """Classify how current_candle interacts with each zone in zones_df."""
    if len(zones_df) == 0:
        return pd.DataFrame(columns=['zone_index', 'zone_type', 'zone_center',
                                    'interaction_type', 'candle_timestamp',
                                    'candle_close', 'candle_body_ratio'])
    
    open_price = current_candle['open']
    high = current_candle['high']
    low = current_candle['low']
    close = current_candle['close']
    
    candle_range = high - low
    candle_body = abs(close - open_price)
    body_ratio = candle_body / candle_range if candle_range > 0 else 0.0
    
    candle_timestamp = current_candle.name if hasattr(current_candle, 'name') else None
    breakout_buffer = breakout_buffer_atr_mult * atr_value
    
    interactions = []
    
    for zone_idx, zone in zones_df.iterrows():
        lower_boundary = zone['lower_boundary']
        upper_boundary = zone['upper_boundary']
        zone_type = zone['zone_type']
        zone_center = zone['center_price']
        
        touches_zone = (low <= upper_boundary) and (high >= lower_boundary)
        
        if not touches_zone:
            continue
        
        interaction_type = None
        
        # Check for breakout
        if zone_type == 'support':
            if close < (lower_boundary - breakout_buffer):
                interaction_type = 'breakout'
        else:  # resistance
            if close > (upper_boundary + breakout_buffer):
                interaction_type = 'breakout'
        
        # Check for acceptance
        if interaction_type is None:
            if lower_boundary <= close <= upper_boundary:
                interaction_type = 'acceptance'
        
        # Check for rejection
        if interaction_type is None:
            if body_ratio >= rejection_body_ratio:
                if zone_type == 'support':
                    if low < upper_boundary and close > upper_boundary:
                        interaction_type = 'rejection'
                else:  # resistance
                    if high > lower_boundary and close < lower_boundary:
                        interaction_type = 'rejection'
        
        if interaction_type is None:
            interaction_type = 'touch'
        
        interactions.append({
            'zone_index': zone_idx,
            'zone_type': zone_type,
            'zone_center': zone_center,
            'interaction_type': interaction_type,
            'candle_timestamp': candle_timestamp,
            'candle_close': close,
            'candle_body_ratio': body_ratio
        })
    
    if interactions:
        interactions_df = pd.DataFrame(interactions)
    else:
        interactions_df = pd.DataFrame(columns=['zone_index', 'zone_type', 'zone_center',
                                                'interaction_type', 'candle_timestamp',
                                                'candle_close', 'candle_body_ratio'])
    
    return interactions_df


def update_zone_metadata(zones_df, interactions_df, current_index):
    """Update zone metadata based on interaction classifications."""
    zones_out = zones_df.copy()
    
    if 'last_touched_index' not in zones_out.columns:
        zones_out['last_touched_index'] = None
    if 'touch_count' not in zones_out.columns:
        zones_out['touch_count'] = 0
    if 'rejection_count' not in zones_out.columns:
        zones_out['rejection_count'] = 0
    if 'acceptance_count' not in zones_out.columns:
        zones_out['acceptance_count'] = 0
    if 'breakout_count' not in zones_out.columns:
        zones_out['breakout_count'] = 0
    
    if len(interactions_df) > 0:
        for _, interaction in interactions_df.iterrows():
            zone_idx = interaction['zone_index']
            interaction_type = interaction['interaction_type']
            
            zones_out.loc[zone_idx, 'last_touched_index'] = current_index
            
            if interaction_type == 'touch':
                zones_out.loc[zone_idx, 'touch_count'] += 1
            elif interaction_type == 'rejection':
                zones_out.loc[zone_idx, 'rejection_count'] += 1
            elif interaction_type == 'acceptance':
                zones_out.loc[zone_idx, 'acceptance_count'] += 1
            elif interaction_type == 'breakout':
                zones_out.loc[zone_idx, 'breakout_count'] += 1
    
    return zones_out


# ============================================================================
# Confirmation Logic
# ============================================================================

def process_confirmation(current_candle, active_waiting_state, zones_df, atr_value,
                        rebound_confirmation_candles=REBOUND_CONFIRMATION_CANDLES,
                        breakout_confirmation_candles=BREAKOUT_CONFIRMATION_CANDLES,
                        previous_close=None):
    """Process confirmation logic for current candle given an active waiting state."""
    if active_waiting_state is None:
        return None, None, None
    
    current_close = current_candle['close']
    current_timestamp = current_candle.name if hasattr(current_candle, 'name') else None
    
    state_type = active_waiting_state['state_type']
    zone_idx = active_waiting_state['zone_index']
    
    # Try to get zone from zones_df, but use stored boundaries as fallback
    try:
        if zone_idx in zones_df.index:
            zone = zones_df.loc[zone_idx]
            lower_boundary = zone['lower_boundary']
            upper_boundary = zone['upper_boundary']
            zone_type = zone['zone_type']
        elif isinstance(zone_idx, (int, np.integer)) and 0 <= zone_idx < len(zones_df):
            zone = zones_df.iloc[zone_idx]
            lower_boundary = zone['lower_boundary']
            upper_boundary = zone['upper_boundary']
            zone_type = zone['zone_type']
        else:
            lower_boundary = active_waiting_state.get('zone_lower_boundary')
            upper_boundary = active_waiting_state.get('zone_upper_boundary')
            zone_type = active_waiting_state.get('zone_type')
    except (KeyError, IndexError, ValueError):
        lower_boundary = active_waiting_state.get('zone_lower_boundary')
        upper_boundary = active_waiting_state.get('zone_upper_boundary')
        zone_type = active_waiting_state.get('zone_type')
    
    if lower_boundary is None or upper_boundary is None:
        confirmation_event = {
            'event_type': 'invalidated_rebound_setup' if 'rebound' in state_type else 'false_breakout',
            'zone_index': zone_idx,
            'zone_type': zone_type,
            'direction': active_waiting_state.get('direction'),
            'interaction_timestamp': active_waiting_state.get('start_timestamp'),
            'confirmation_timestamp': current_candle.name if hasattr(current_candle, 'name') else None,
            'start_index': active_waiting_state.get('start_index'),
            'confirmation_index': None,
            'candles_waited': active_waiting_state.get('candles_waited', 0),
            'reason': 'zone_no_longer_exists'
        }
        return None, confirmation_event, None
    
    direction = active_waiting_state['direction']
    candles_waited = active_waiting_state.get('candles_waited', 0)
    confirmation_candles_required = active_waiting_state['confirmation_candles_required']
    
    candles_waited += 1
    active_waiting_state['candles_waited'] = candles_waited
    
    if 'last_close' in active_waiting_state:
        previous_close = active_waiting_state['last_close']
    elif previous_close is None:
        if direction == 'long':
            previous_close = upper_boundary
        else:
            previous_close = lower_boundary
    
    active_waiting_state['last_close'] = current_close
    
    confirmation_event = None
    
    if state_type == 'WAIT_REBOUND_CONFIRMATION':
        # Check invalidation: price closes back inside zone
        if lower_boundary <= current_close <= upper_boundary:
            confirmation_event = {
                'event_type': 'invalidated_rebound_setup',
                'zone_index': zone_idx,
                'zone_type': zone_type,
                'direction': direction,
                'interaction_timestamp': active_waiting_state['start_timestamp'],
                'confirmation_timestamp': current_timestamp,
                'start_index': active_waiting_state['start_index'],
                'confirmation_index': None,
                'candles_waited': candles_waited,
                'reason': 'price_reentered_zone'
            }
            return None, confirmation_event, None
        
        # Check invalidation: price violates direction
        if direction == 'long':
            if current_close <= previous_close:
                confirmation_event = {
                    'event_type': 'invalidated_rebound_setup',
                    'zone_index': zone_idx,
                    'zone_type': zone_type,
                    'direction': direction,
                    'interaction_timestamp': active_waiting_state['start_timestamp'],
                    'confirmation_timestamp': current_timestamp,
                    'start_index': active_waiting_state['start_index'],
                    'confirmation_index': None,
                    'candles_waited': candles_waited,
                    'reason': 'direction_violated'
                }
                return None, confirmation_event, None
        else:  # short
            if current_close >= previous_close:
                confirmation_event = {
                    'event_type': 'invalidated_rebound_setup',
                    'zone_index': zone_idx,
                    'zone_type': zone_type,
                    'direction': direction,
                    'interaction_timestamp': active_waiting_state['start_timestamp'],
                    'confirmation_timestamp': current_timestamp,
                    'start_index': active_waiting_state['start_index'],
                    'confirmation_index': None,
                    'candles_waited': candles_waited,
                    'reason': 'direction_violated'
                }
                return None, confirmation_event, None
        
        # Check confirmation
        if candles_waited >= confirmation_candles_required:
            confirmation_event = {
                'event_type': 'confirmed_rebound_setup',
                'zone_index': zone_idx,
                'zone_type': zone_type,
                'direction': direction,
                'interaction_timestamp': active_waiting_state['start_timestamp'],
                'confirmation_timestamp': current_timestamp,
                'start_index': active_waiting_state['start_index'],
                'confirmation_index': None,
                'candles_waited': candles_waited,
                'reason': None
            }
            return None, confirmation_event, None
        
        return active_waiting_state, None, None
    
    elif state_type == 'WAIT_BREAKOUT_CONFIRMATION':
        breakout_buffer = active_waiting_state.get('breakout_buffer', BREAKOUT_BUFFER_ATR_MULT * atr_value)
        
        # Check invalidation: price re-enters zone
        if lower_boundary <= current_close <= upper_boundary:
            confirmation_event = {
                'event_type': 'false_breakout',
                'zone_index': zone_idx,
                'zone_type': zone_type,
                'direction': direction,
                'interaction_timestamp': active_waiting_state['start_timestamp'],
                'confirmation_timestamp': current_timestamp,
                'start_index': active_waiting_state['start_index'],
                'confirmation_index': None,
                'candles_waited': candles_waited,
                'reason': 'price_reentered_zone'
            }
            return None, confirmation_event, None
        
        # Check invalidation: price fails to hold beyond boundary + buffer
        if direction == 'long':
            if current_close <= (upper_boundary + breakout_buffer):
                confirmation_event = {
                    'event_type': 'invalidated_breakout_setup',
                    'zone_index': zone_idx,
                    'zone_type': zone_type,
                    'direction': direction,
                    'interaction_timestamp': active_waiting_state['start_timestamp'],
                    'confirmation_timestamp': current_timestamp,
                    'start_index': active_waiting_state['start_index'],
                    'confirmation_index': None,
                    'candles_waited': candles_waited,
                    'reason': 'failed_to_hold_beyond_buffer'
                }
                return None, confirmation_event, None
        else:  # short
            if current_close >= (lower_boundary - breakout_buffer):
                confirmation_event = {
                    'event_type': 'invalidated_breakout_setup',
                    'zone_index': zone_idx,
                    'zone_type': zone_type,
                    'direction': direction,
                    'interaction_timestamp': active_waiting_state['start_timestamp'],
                    'confirmation_timestamp': current_timestamp,
                    'start_index': active_waiting_state['start_index'],
                    'confirmation_index': None,
                    'candles_waited': candles_waited,
                    'reason': 'failed_to_hold_beyond_buffer'
                }
                return None, confirmation_event, None
        
        # Check confirmation
        if candles_waited >= confirmation_candles_required:
            confirmation_event = {
                'event_type': 'confirmed_breakout_setup',
                'zone_index': zone_idx,
                'zone_type': zone_type,
                'direction': direction,
                'interaction_timestamp': active_waiting_state['start_timestamp'],
                'confirmation_timestamp': current_timestamp,
                'start_index': active_waiting_state['start_index'],
                'confirmation_index': None,
                'candles_waited': candles_waited,
                'reason': None
            }
            return None, confirmation_event, None
        
        return active_waiting_state, None, None
    
    return None, None, None


def process_interaction_with_confirmation(current_candle, interactions_df, active_waiting_state,
                                         zones_df, atr_value, current_index,
                                         rebound_confirmation_candles=REBOUND_CONFIRMATION_CANDLES,
                                         breakout_confirmation_candles=BREAKOUT_CONFIRMATION_CANDLES,
                                         previous_close=None):
    """Process interactions and handle confirmation logic in a single function."""
    confirmation_events = []
    updated_waiting_state = active_waiting_state
    
    # Step 1: Process existing waiting state (if any)
    if active_waiting_state is not None:
        updated_state, confirmation_event, _ = process_confirmation(
            current_candle, active_waiting_state, zones_df, atr_value,
            rebound_confirmation_candles, breakout_confirmation_candles,
            previous_close
        )
        
        if confirmation_event is not None:
            confirmation_event['confirmation_index'] = current_index
            confirmation_events.append(confirmation_event)
            updated_waiting_state = None
    
    # Step 2: Process new interactions (only if no active waiting state)
    if updated_waiting_state is None and len(interactions_df) > 0:
        for _, interaction in interactions_df.iterrows():
            interaction_type = interaction['interaction_type']
            zone_idx = interaction['zone_index']
            zone = zones_df.loc[zone_idx]
            
            if interaction_type == 'rejection':
                if zone['zone_type'] == 'support':
                    direction = 'long'
                else:
                    direction = 'short'
                
                updated_waiting_state = {
                    'state_type': 'WAIT_REBOUND_CONFIRMATION',
                    'zone_index': zone_idx,
                    'zone_type': zone['zone_type'],
                    'zone_center': zone['center_price'],
                    'zone_lower_boundary': zone['lower_boundary'],
                    'zone_upper_boundary': zone['upper_boundary'],
                    'interaction_type': 'rejection',
                    'direction': direction,
                    'start_index': current_index,
                    'start_timestamp': current_candle.name if hasattr(current_candle, 'name') else None,
                    'confirmation_candles_required': rebound_confirmation_candles,
                    'candles_waited': 0,
                    'last_close': current_candle['close']
                }
                break
            
            elif interaction_type == 'breakout':
                if zone['zone_type'] == 'support':
                    direction = 'short'
                else:
                    direction = 'long'
                
                updated_waiting_state = {
                    'state_type': 'WAIT_BREAKOUT_CONFIRMATION',
                    'zone_index': zone_idx,
                    'zone_type': zone['zone_type'],
                    'zone_center': zone['center_price'],
                    'zone_lower_boundary': zone['lower_boundary'],
                    'zone_upper_boundary': zone['upper_boundary'],
                    'interaction_type': 'breakout',
                    'direction': direction,
                    'start_index': current_index,
                    'start_timestamp': current_candle.name if hasattr(current_candle, 'name') else None,
                    'confirmation_candles_required': breakout_confirmation_candles,
                    'candles_waited': 0,
                    'breakout_buffer': BREAKOUT_BUFFER_ATR_MULT * atr_value,
                    'last_close': current_candle['close']
                }
                break
    
    return updated_waiting_state, confirmation_events


# ============================================================================
# Main Signal Processing Function (Exported for External Use)
# ============================================================================

def process_candle_for_signals(current_candle, context_df, zones_df, atr_value, current_index,
                               active_waiting_state=None,
                               rebound_confirmation_candles=REBOUND_CONFIRMATION_CANDLES,
                               breakout_confirmation_candles=BREAKOUT_CONFIRMATION_CANDLES,
                               previous_close=None):
    """
    Process a single candle and generate SR signals.
    
    This is the main exported function that can be used elsewhere. It processes
    a single candle through the complete SR signal pipeline:
    1. Detect pivots
    2. Cluster pivots into zones
    3. Classify zone interactions
    4. Update zone metadata
    5. Process interactions with confirmation logic
    6. Return confirmed events and updated state
    
    Parameters
    ----------
    current_candle : pd.Series
        Current candle being analyzed. Must have columns: open, high, low, close.
    context_df : pd.DataFrame
        Historical context window (ending at t-1). Must have 'atr' column.
    zones_df : pd.DataFrame or None
        Existing zones DataFrame (if None, will be created from pivots).
    atr_value : float
        Current ATR value for volatility normalization.
    current_index : int
        Index position of current candle.
    active_waiting_state : dict or None, optional
        Current waiting state from previous call, or None.
    rebound_confirmation_candles : int, default REBOUND_CONFIRMATION_CANDLES
        Number of candles to wait for rebound confirmation.
    breakout_confirmation_candles : int, default BREAKOUT_CONFIRMATION_CANDLES
        Number of candles to wait for breakout confirmation.
    previous_close : float, optional
        Previous candle's close price (for direction checks).
    
    Returns
    -------
    tuple of (dict, pd.DataFrame, dict or None, list)
        - signal: Signal dictionary with confirmed events and metadata
        - updated_zones_df: Updated zones DataFrame
        - updated_waiting_state: Updated waiting state (or None)
        - confirmed_events: List of confirmation event dicts
    """
    # Step 1: Detect pivots
    pivots_df = detect_pivots(context_df, left_bars=LEFT_BARS, right_bars=RIGHT_BARS)
    
    # Step 2: Cluster pivots into zones (if zones_df is None, create from pivots)
    if zones_df is None:
        zones_df = cluster_pivots_to_zones(
            pivots_df, context_df,
            cluster_atr_mult=CLUSTER_ATR_MULT,
            zone_width_atr_mult=ZONE_WIDTH_ATR_MULT,
            min_pivots=MIN_PIVOTS_PER_ZONE
        )
    
    # Step 3: Classify zone interactions
    interactions_df = classify_zone_interactions(
        current_candle, zones_df, atr_value,
        rejection_body_ratio=REJECTION_BODY_RATIO,
        breakout_buffer_atr_mult=BREAKOUT_BUFFER_ATR_MULT
    )
    
    # Step 4: Update zone metadata
    updated_zones_df = update_zone_metadata(zones_df, interactions_df, current_index)
    
    # Step 5: Process interactions with confirmation logic
    updated_waiting_state, confirmed_events = process_interaction_with_confirmation(
        current_candle, interactions_df, active_waiting_state,
        updated_zones_df, atr_value, current_index,
        rebound_confirmation_candles=rebound_confirmation_candles,
        breakout_confirmation_candles=breakout_confirmation_candles,
        previous_close=previous_close
    )
    
    # Step 6: Build signal dictionary
    timestamp = current_candle.name if hasattr(current_candle, 'name') else None
    
    signal = {
        'index': current_index,
        'timestamp': timestamp,
        'price': {
            'open': current_candle['open'],
            'high': current_candle['high'],
            'low': current_candle['low'],
            'close': current_candle['close']
        },
        'atr_value': atr_value,
        'zones_df': updated_zones_df,
        'interactions_df': interactions_df,
        'confirmed_events': confirmed_events,
        'waiting_state': updated_waiting_state,
        'summary': {
            'has_confirmed_rebound': any(e['event_type'] == 'confirmed_rebound_setup' for e in confirmed_events),
            'has_confirmed_breakout': any(e['event_type'] == 'confirmed_breakout_setup' for e in confirmed_events),
            'has_invalidated_setup': any('invalidated' in e['event_type'] or 'false_breakout' in e['event_type'] for e in confirmed_events),
            'num_zones': len(updated_zones_df),
            'num_interactions': len(interactions_df),
            'num_confirmed_events': len(confirmed_events)
        }
    }
    
    return signal, updated_zones_df, updated_waiting_state, confirmed_events


# ============================================================================
# SRSignal Class
# ============================================================================

class SRSignal(BaseSignal):
    """
    Support/Resistance signal module.
    
    Generates standardized trading signals based on support/resistance zone
    analysis. Detects pivots, clusters them into zones, classifies interactions,
    and waits for confirmation before emitting signals.
    
    This module maintains internal state (waiting_state) for confirmation logic
    but outputs pure signals with no knowledge of positions or money.
    """
    
    def __init__(self, 
                 window_size=WINDOW_SIZE,
                 rebound_confirmation_candles=REBOUND_CONFIRMATION_CANDLES,
                 breakout_confirmation_candles=BREAKOUT_CONFIRMATION_CANDLES):
        """
        Initialize SR signal module.
        
        Parameters
        ----------
        window_size : int, default WINDOW_SIZE
            Size of context window for structure analysis.
        rebound_confirmation_candles : int, default REBOUND_CONFIRMATION_CANDLES
            Number of candles to wait for rebound confirmation.
        breakout_confirmation_candles : int, default BREAKOUT_CONFIRMATION_CANDLES
            Number of candles to wait for breakout confirmation.
        """
        super().__init__(name='sr')
        self.window_size = window_size
        self.rebound_confirmation_candles = rebound_confirmation_candles
        self.breakout_confirmation_candles = breakout_confirmation_candles
    
    def generate_signal(self, candle: pd.Series, context: pd.DataFrame,
                       state: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Generate standardized SR signal from market data.
        
        Parameters
        ----------
        candle : pd.Series
            Current candle being analyzed. Must have columns: open, high, low, close.
        context : pd.DataFrame
            Historical context window (ending at t-1). Must have 'atr' column.
        state : dict or None, optional
            Internal state from previous call. Contains:
            - waiting_state: Waiting state dict (or None)
            - zones_df: Zones DataFrame (or None)
            - previous_close: Previous candle close (or None)
        **kwargs
            Additional parameters (e.g., current_index).
        
        Returns
        -------
        tuple of (dict, dict or None)
            - signal: Standardized signal dictionary
            - updated_state: Updated internal state
        """
        # Extract current_index from kwargs
        current_index = kwargs.get('current_index')
        if current_index is None:
            current_index = len(context) if context is not None else 0
        
        # Extract state components
        if state is None:
            state = {}
        
        waiting_state = state.get('waiting_state')
        zones_df = state.get('zones_df')
        previous_close = state.get('previous_close')
        
        # Get ATR value from context
        if context is not None and len(context) > 0 and 'atr' in context.columns:
            atr_value = context['atr'].iloc[-1]
            if pd.isna(atr_value) or atr_value <= 0:
                atr_value = (candle['high'] - candle['low']) * 0.5
        else:
            atr_value = (candle['high'] - candle['low']) * 0.5
        
        # Step 1: Detect pivots
        pivots_df = detect_pivots(context, left_bars=LEFT_BARS, right_bars=RIGHT_BARS)
        
        # Step 2: Cluster pivots into zones
        # IMPORTANT: Recalculate zones every candle (matching notebook behavior)
        # Zones must be recalculated because:
        # - Context window shifts forward each candle
        # - New pivots become available as more data arrives
        # - Old pivots might be invalidated (broken zones)
        # - ATR value changes, affecting zone boundaries
        zones_df = cluster_pivots_to_zones(
            pivots_df, context,
            cluster_atr_mult=CLUSTER_ATR_MULT,
            zone_width_atr_mult=ZONE_WIDTH_ATR_MULT,
            min_pivots=MIN_PIVOTS_PER_ZONE
        )
        
        # Step 3: Classify zone interactions
        interactions_df = classify_zone_interactions(
            candle, zones_df, atr_value,
            rejection_body_ratio=REJECTION_BODY_RATIO,
            breakout_buffer_atr_mult=BREAKOUT_BUFFER_ATR_MULT
        )
        
        # Step 4: Update zone metadata
        zones_df = update_zone_metadata(zones_df, interactions_df, current_index)
        
        # Step 5: Process interactions with confirmation logic
        updated_waiting_state, confirmed_events = process_interaction_with_confirmation(
            candle, interactions_df, waiting_state,
            zones_df, atr_value, current_index,
            rebound_confirmation_candles=self.rebound_confirmation_candles,
            breakout_confirmation_candles=self.breakout_confirmation_candles,
            previous_close=previous_close
        )
        
        # Step 6: Generate standardized signal from confirmed events
        signal = self._generate_standardized_signal(
            candle, zones_df, confirmed_events, updated_waiting_state,
            atr_value, current_index
        )
        
        # Update state
        updated_state = {
            'waiting_state': updated_waiting_state,
            'zones_df': zones_df,
            'previous_close': candle['close']
        }
        
        return signal, updated_state
    
    def _generate_standardized_signal(self, candle, zones_df, confirmed_events,
                                     waiting_state, atr_value, current_index):
        """Convert confirmed events to standardized signal format."""
        timestamp = candle.name if hasattr(candle, 'name') else None
        
        # Default: flat signal
        direction = 0
        strength = 0.0
        metadata = {
            'entry_price': None,
            'zone_index': None,
            'zone_type': None,
            'zone_center': None,
            'zone_lower': None,
            'zone_upper': None,
            'setup_type': None,
            'interaction_type': None,
            'atr_value': atr_value,
            'confirmed_event': None
        }
        
        # Check for confirmed events (highest priority)
        if confirmed_events:
            event = confirmed_events[0]
            event_type = event['event_type']
            
            if event_type == 'confirmed_rebound_setup':
                # Rebound setup confirmed
                direction = 1 if event['direction'] == 'long' else -1
                metadata['setup_type'] = 'rebound'
                metadata['interaction_type'] = 'rejection'
                metadata['zone_index'] = event['zone_index']
                metadata['zone_type'] = event['zone_type']
                metadata['confirmed_event'] = event
                
                # Get zone information
                zone_idx = event['zone_index']
                try:
                    if zone_idx in zones_df.index:
                        zone = zones_df.loc[zone_idx]
                    elif isinstance(zone_idx, (int, np.integer)) and 0 <= zone_idx < len(zones_df):
                        zone = zones_df.iloc[zone_idx]
                    else:
                        zone = None
                    
                    if zone is not None:
                        metadata['zone_center'] = zone['center_price']
                        metadata['zone_lower'] = zone['lower_boundary']
                        metadata['zone_upper'] = zone['upper_boundary']
                        
                        # Entry price: zone boundary
                        if event['direction'] == 'long':
                            metadata['entry_price'] = zone['upper_boundary']
                        else:
                            metadata['entry_price'] = zone['lower_boundary']
                        
                        # Determine strength based on zone characteristics
                        pivot_count = zone.get('pivot_count', 0)
                        touch_count = zone.get('touch_count', 0)
                        rejection_count = zone.get('rejection_count', 0)
                        
                        if pivot_count >= 5 or touch_count >= 5 or rejection_count >= 2:
                            strength = 1.0
                        elif pivot_count >= 3 or touch_count >= 3:
                            strength = 0.7
                        else:
                            strength = 0.5
                except (KeyError, IndexError, ValueError):
                    pass
            
            elif event_type == 'confirmed_breakout_setup':
                # Breakout setup confirmed
                direction = 1 if event['direction'] == 'long' else -1
                metadata['setup_type'] = 'breakout'
                metadata['interaction_type'] = 'breakout'
                metadata['zone_index'] = event['zone_index']
                metadata['zone_type'] = event['zone_type']
                metadata['entry_price'] = candle['close']  # Momentum entry
                metadata['confirmed_event'] = event
                
                # Get zone information
                zone_idx = event['zone_index']
                try:
                    if zone_idx in zones_df.index:
                        zone = zones_df.loc[zone_idx]
                    elif isinstance(zone_idx, (int, np.integer)) and 0 <= zone_idx < len(zones_df):
                        zone = zones_df.iloc[zone_idx]
                    else:
                        zone = None
                    
                    if zone is not None:
                        metadata['zone_center'] = zone['center_price']
                        metadata['zone_lower'] = zone['lower_boundary']
                        metadata['zone_upper'] = zone['upper_boundary']
                        
                        # Determine strength
                        pivot_count = zone.get('pivot_count', 0)
                        breakout_count = zone.get('breakout_count', 0)
                        
                        if pivot_count >= 5 or breakout_count >= 1:
                            strength = 1.0
                        elif pivot_count >= 3:
                            strength = 0.7
                        else:
                            strength = 0.5
                except (KeyError, IndexError, ValueError):
                    pass
        
        # If waiting state exists but no confirmed event, add metadata
        elif waiting_state is not None:
            metadata['waiting_state'] = waiting_state['state_type']
            metadata['waiting_direction'] = waiting_state.get('direction')
            metadata['waiting_zone_index'] = waiting_state.get('zone_index')
            metadata['candles_waited'] = waiting_state.get('candles_waited', 0)
            metadata['candles_required'] = waiting_state.get('confirmation_candles_required', 0)
        
        # Create standardized signal
        signal = {
            'direction': direction,
            'strength': strength,
            'timestamp': timestamp,
            'index': current_index,
            'metadata': metadata,
            'source': self.name
        }
        
        # Validate signal
        self.validate_signal(signal)
        
        return signal

