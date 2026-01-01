# -*- coding: utf-8 -*-
"""
Signal modules for trading system.

This package contains pure signal generation modules that consume market data
and output standardized trading signals. Each signal module is strategy-agnostic
and knows nothing about positions, money, or execution.
"""

from trader.signals.base import BaseSignal, STANDARD_SIGNAL_FORMAT
from trader.signals.sr_signal import (
    SRSignal,
    add_atr,
    iterate_structure_context,
    get_structure_context,
    detect_pivots,
    cluster_pivots_to_zones,
    classify_zone_interactions,
    update_zone_metadata,
    process_confirmation,
    process_interaction_with_confirmation,
    process_candle_for_signals,
    # Constants
    WINDOW_SIZE,
    REBOUND_CONFIRMATION_CANDLES,
    BREAKOUT_CONFIRMATION_CANDLES
)

__all__ = [
    'BaseSignal',
    'STANDARD_SIGNAL_FORMAT',
    'SRSignal',
    'add_atr',
    'iterate_structure_context',
    'get_structure_context',
    'detect_pivots',
    'cluster_pivots_to_zones',
    'classify_zone_interactions',
    'update_zone_metadata',
    'process_confirmation',
    'process_interaction_with_confirmation',
    'process_candle_for_signals',
    'WINDOW_SIZE',
    'REBOUND_CONFIRMATION_CANDLES',
    'BREAKOUT_CONFIRMATION_CANDLES'
]

