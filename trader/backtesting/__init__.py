# -*- coding: utf-8 -*-
"""
Backtesting engine module.

Provides modular components for realistic trade execution, risk management,
capital accounting, and signal filtering in walk-forward simulations.
"""

from trader.backtesting.execution_engine import ExecutionEngine
from trader.backtesting.risk_manager import RiskManager
from trader.backtesting.capital_accounting import CapitalAccountant
from trader.backtesting.risk_controls import RiskController
from trader.backtesting.signal_filters import SignalFilter
from trader.backtesting.config import BacktestConfig

__all__ = [
    'ExecutionEngine',
    'RiskManager',
    'CapitalAccountant',
    'RiskController',
    'SignalFilter',
    'BacktestConfig'
]






