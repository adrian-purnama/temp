# -*- coding: utf-8 -*-
"""
Live trading orchestrator.

Main entry point for live trading, integrating all components and managing
the trading loop.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, date
from binance.client import Client

from trader.live.config import LiveTradingConfig
from trader.live.binance_client import BinanceClient
from trader.live.market_data import MarketData
from trader.live.order_manager import OrderManager, OrderStatus
from trader.live.position_tracker import PositionTracker
from trader.live.state_recovery import StateRecovery
from trader.live.live_execution_engine import LiveExecutionEngine

from trader.backtesting import (
    ExecutionEngine, RiskManager, CapitalAccountant, RiskController,
    SignalFilter, BacktestConfig
)
from trader.signals.sr_signal import (
    add_atr, detect_pivots, cluster_pivots_to_zones,
    classify_zone_interactions, process_interaction_with_confirmation,
    update_zone_metadata
)


# Trade State Constants (from forward_test.py)
STATE_IDLE = "IDLE"
STATE_SETUP_CONFIRMED = "SETUP_CONFIRMED"
STATE_IN_TRADE = "IN_TRADE"
STATE_COOLDOWN = "COOLDOWN"


logger = logging.getLogger(__name__)


class LiveTrader:
    """Main orchestrator for live trading."""
    
    def __init__(self, config: LiveTradingConfig):
        """
        Initialize live trader.
        
        Parameters
        ----------
        config : LiveTradingConfig
            Live trading configuration
        """
        self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Initialize Binance client
        self.binance_client = BinanceClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            use_testnet=config.use_testnet
        )
        
        # Test connection
        if not self.binance_client.test_connection():
            raise ConnectionError("Failed to connect to Binance API")
        
        logger.info(f"Connected to Binance {'Testnet' if config.use_testnet else 'Realnet'}")
        
        # Initialize components
        self.order_manager = OrderManager(
            client=self.binance_client.client,
            symbol=config.symbol
        )
        
        self.position_tracker = PositionTracker(
            client=self.binance_client.client,
            symbol=config.symbol
        )
        
        self.state_recovery = StateRecovery(
            client=self.binance_client.client,
            symbol=config.symbol
        )
        
        # Initialize market data consumer
        self.market_data = MarketData(
            client=self.binance_client.client,
            symbol=config.symbol,
            timeframe=config.timeframe,
            window_size=config.window_size,
            atr_period=config.atr_period,
            atr_method=config.atr_method,
            on_new_candle=self._on_new_candle,
            ping_interval=config.websocket_ping_interval,
            ping_timeout=config.websocket_ping_timeout,
            connection_timeout=config.websocket_connection_timeout,
            close_timeout=config.websocket_close_timeout,
            max_queue_size=config.websocket_max_queue_size
        )
        
        # Initialize live execution engine
        self.live_execution_engine = LiveExecutionEngine(
            binance_client=self.binance_client,
            order_manager=self.order_manager,
            fee_pct=config.fee_pct,
            slippage_atr_mult=config.slippage_atr_mult
        )
        
        # Initialize backtesting components (reuse)
        backtest_config = BacktestConfig(
            execution_mode=config.execution_mode,
            fee_pct=config.fee_pct,
            slippage_atr_mult=config.slippage_atr_mult,
            risk_per_trade=config.risk_per_trade,
            max_position_value_pct=config.max_position_value_pct,
            atr_percentile_window=config.atr_percentile_window,
            atr_percentile_threshold=config.atr_percentile_threshold,
            cap_stop_loss=config.cap_stop_loss,
            cap_take_profit=config.cap_take_profit,
            per_zone_cooldown_candles=config.per_zone_cooldown_candles,
            max_daily_risk_pct=config.max_daily_risk_pct,
            max_daily_trades=config.max_daily_trades,
            max_daily_loss_pct=config.max_daily_loss_pct,
            enable_unrealized_pnl=config.enable_unrealized_pnl,
            enable_rsi_filter=config.enable_rsi_filter,
            rsi_period=config.rsi_period,
            rsi_oversold_threshold=config.rsi_oversold_threshold,
            rsi_overbought_threshold=config.rsi_overbought_threshold,
            rsi_lookback_candles=config.rsi_lookback_candles
        )
        
        self.risk_manager = RiskManager(
            max_position_value_pct=backtest_config.max_position_value_pct,
            atr_percentile_window=backtest_config.atr_percentile_window,
            atr_percentile_threshold=backtest_config.atr_percentile_threshold,
            cap_stop_loss=backtest_config.cap_stop_loss,
            cap_take_profit=backtest_config.cap_take_profit
        )
        
        self.capital_accountant = CapitalAccountant(
            enable_unrealized_pnl=backtest_config.enable_unrealized_pnl
        )
        
        self.risk_controller = RiskController(
            per_zone_cooldown_candles=backtest_config.per_zone_cooldown_candles,
            max_daily_risk_pct=backtest_config.max_daily_risk_pct,
            max_daily_trades=backtest_config.max_daily_trades,
            max_daily_loss_pct=backtest_config.max_daily_loss_pct
        )
        
        self.signal_filter = None
        if backtest_config.enable_rsi_filter:
            self.signal_filter = SignalFilter(
                rsi_period=backtest_config.rsi_period,
                rsi_oversold_threshold=backtest_config.rsi_oversold_threshold,
                rsi_overbought_threshold=backtest_config.rsi_overbought_threshold,
                rsi_lookback_candles=backtest_config.rsi_lookback_candles
            )
        
        # Trading state
        self.trade_state = {
            'current_state': STATE_IDLE,
            'active_trade': None,
            'pending_setup': None,
            'cooldown_remaining': 0
        }
        
        self.active_waiting_state = None
        self.trades_log = []
        self.account_balance = 0.0
        self.initial_balance = 0.0
        
        # Load initial balance (or use override)
        if config.override_initial_balance is not None:
            # Use override balance
            self.account_balance = config.override_initial_balance
            self.initial_balance = config.override_initial_balance
            logger.info(f"Using override initial balance: ${self.account_balance:.2f}")
        else:
            # Fetch actual balance from Binance
            self._update_account_balance()
            self.initial_balance = self.account_balance
        
        logger.info(f"Initialized LiveTrader with balance: ${self.account_balance:.2f}")
        
        # Dashboard (will be initialized in start() if enabled)
        self.dashboard = None
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create logger
        logger.setLevel(log_level)
        
        # File handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _update_account_balance(self):
        """Update account balance from Binance."""
        # Skip API call if using override balance
        if self.config.override_initial_balance is not None:
            # Keep balance at override value (don't update from API)
            return
        
        try:
            self.account_balance = self.binance_client.get_account_balance("USDT")
        except Exception as e:
            logger.warning(f"Error updating account balance: {e} (using last known balance: ${self.account_balance:.2f})")
    
    def _check_kill_switch(self) -> bool:
        """
        Check kill switch file.
        
        Returns
        -------
        bool
            True if kill switch is active
        """
        if not self.config.enable_kill_switch:
            return False
        
        kill_switch_path = self.config.kill_switch_file
        if os.path.exists(kill_switch_path):
            logger.warning("Kill switch activated! Stopping trading.")
            return True
        
        return False
    
    def _apply_realnet_safeguards(self, position_size: float, entry_price: float) -> float:
        """
        Apply realnet safeguards to position size.
        
        Parameters
        ----------
        position_size : float
            Original position size
        entry_price : float
            Entry price
        
        Returns
        -------
        float
            Adjusted position size
        """
        if self.config.use_testnet:
            return position_size
        
        # Reduce position size
        adjusted_size = position_size * (self.config.max_position_size_pct / 100.0)
        
        # Check max exposure limit
        position_value = adjusted_size * entry_price
        max_exposure = self.account_balance * (self.config.max_exposure_pct / 100.0)
        
        if position_value > max_exposure:
            adjusted_size = max_exposure / entry_price
            logger.warning(f"Position size capped by max exposure limit: {adjusted_size}")
        
        return adjusted_size
    
    def _on_new_candle(self, candle: pd.Series, context_df: pd.DataFrame):
        """
        Callback when new candle is received.
        
        Parameters
        ----------
        candle : pd.Series
            New candle data
        context_df : pd.DataFrame
            Updated context window
        """
        try:
            self._process_candle(candle, context_df)
            
            # Update dashboard if enabled (non-blocking)
            if self.dashboard:
                try:
                    self.dashboard.update()
                except Exception as e:
                    logger.debug(f"Dashboard update error: {e}")
        except Exception as e:
            logger.error(f"Error processing candle: {e}", exc_info=True)
    
    def _process_candle(self, current_candle: pd.Series, context_df: pd.DataFrame):
        """Process a new candle through the trading system."""
        # Check kill switch
        if self._check_kill_switch():
            self.stop()
            return
        
        # Update account balance
        self._update_account_balance()
        
        # Get ATR value
        atr_value = self.market_data.get_atr_value()
        
        # Get previous close
        previous_close = context_df['close'].iloc[-2] if len(context_df) > 1 else None
        
        # Step 1: Detect pivots
        pivots = detect_pivots(
            context_df,
            left_bars=self.config.left_bars,
            right_bars=self.config.right_bars
        )
        
        # Step 2: Cluster pivots into zones
        zones = cluster_pivots_to_zones(
            pivots, context_df,
            cluster_atr_mult=self.config.cluster_atr_mult,
            zone_width_atr_mult=self.config.zone_width_atr_mult,
            min_pivots=self.config.min_pivots_per_zone
        )
        
        # Step 3: Classify interactions
        interactions = classify_zone_interactions(
            current_candle, zones, atr_value,
            rejection_body_ratio=self.config.rejection_body_ratio,
            breakout_buffer_atr_mult=self.config.breakout_buffer_atr_mult
        )
        
        # Step 4: Process interactions with confirmation
        updated_waiting_state, confirmation_events = process_interaction_with_confirmation(
            current_candle, interactions, self.active_waiting_state,
            zones, atr_value, len(context_df) - 1,
            rebound_confirmation_candles=self.config.rebound_confirmation_candles,
            breakout_confirmation_candles=self.config.breakout_confirmation_candles,
            previous_close=previous_close
        )
        
        self.active_waiting_state = updated_waiting_state
        
        # Step 5: Process trade state machine
        confirmed_setup = confirmation_events[0] if confirmation_events else None
        self._process_trade_state(current_candle, zones, atr_value, context_df, confirmed_setup)
        
        # Step 6: Monitor open positions
        self._monitor_positions(current_candle, context_df)
        
        # Step 7: Save positions periodically (every 10 candles)
        if hasattr(self, '_candle_count'):
            self._candle_count += 1
        else:
            self._candle_count = 1
        
        if self._candle_count % 10 == 0 and self.config.state_recovery_enabled:
            try:
                self.position_tracker.save_positions(self.config.position_backup_file)
            except Exception as e:
                logger.debug(f"Error saving positions backup: {e}")
    
    def _process_trade_state(self, current_candle: pd.Series, zones: pd.DataFrame,
                            atr_value: float, context_df: pd.DataFrame,
                            confirmed_setup: Optional[Dict[str, Any]]):
        """Process trade state machine (similar to forward_test.py)."""
        current_state = self.trade_state['current_state']
        active_trade = self.trade_state.get('active_trade')
        
        # Calculate equity
        current_price = current_candle['close']
        account_equity = self.capital_accountant.calculate_equity(
            self.account_balance,
            active_trade,
            current_price
        )
        
        # Get current date
        current_date = current_candle.name.date() if hasattr(current_candle.name, 'date') else date.today()
        
        if current_state == STATE_IDLE:
            if confirmed_setup is not None:
                # Apply RSI entry filter
                if self.signal_filter is not None:
                    if not self.signal_filter.filter_entry_signal(confirmed_setup, context_df):
                        return
                
                # Check risk controls
                zone_idx = confirmed_setup['zone_index']
                if not self.risk_controller.check_zone_cooldown(zone_idx, len(context_df) - 1):
                    return
                
                if not self.risk_controller.check_daily_trade_limit(current_date):
                    return
                
                # Transition to SETUP_CONFIRMED
                self.trade_state['current_state'] = STATE_SETUP_CONFIRMED
                self.trade_state['pending_setup'] = confirmed_setup
        
        elif current_state == STATE_SETUP_CONFIRMED:
            pending_setup = self.trade_state.get('pending_setup')
            if pending_setup is not None:
                try:
                    zone_idx = pending_setup['zone_index']
                    if zone_idx in zones.index:
                        zone = zones.loc[zone_idx]
                    elif isinstance(zone_idx, (int, np.integer)) and 0 <= zone_idx < len(zones):
                        zone = zones.iloc[zone_idx]
                    else:
                        zone = None
                    
                    if zone is not None:
                        # Execute trade entry
                        self._execute_trade_entry(pending_setup, zone, current_candle, atr_value, account_equity, current_date)
                        
                except Exception as e:
                    logger.error(f"Error executing trade entry: {e}", exc_info=True)
                    self.trade_state['current_state'] = STATE_IDLE
                    self.trade_state['pending_setup'] = None
        
        elif current_state == STATE_IN_TRADE:
            # Monitor for exit (handled by _monitor_positions)
            pass
        
        elif current_state == STATE_COOLDOWN:
            cooldown_remaining = self.trade_state.get('cooldown_remaining', 0)
            if cooldown_remaining > 0:
                self.trade_state['cooldown_remaining'] = cooldown_remaining - 1
            else:
                self.trade_state['current_state'] = STATE_IDLE
    
    def _execute_trade_entry(self, confirmed_setup: Dict[str, Any], zone: pd.Series,
                            current_candle: pd.Series, atr_value: float,
                            account_equity: float, current_date: date):
        """Execute trade entry on Binance."""
        try:
            # Determine entry price
            if self.config.execution_mode == "next_open":
                base_entry_price = current_candle['open']
            else:
                base_entry_price = current_candle['close']
            
            # Calculate risk parameters
            risk_params = self._calculate_risk_parameters(
                confirmed_setup, zone, base_entry_price, atr_value
            )
            
            # Calculate position size
            risk_amount = account_equity * self.config.risk_per_trade
            position_size = self.risk_manager.calculate_position_size(
                base_entry_price,
                risk_params['stop_loss_price'],
                risk_amount,
                account_equity
            )
            
            # Apply realnet safeguards
            position_size = self._apply_realnet_safeguards(position_size, base_entry_price)
            
            # Check daily risk limit
            if not self.risk_controller.check_daily_risk_limit(risk_amount, account_equity, current_date):
                logger.warning("Daily risk limit exceeded, skipping trade")
                self.trade_state['current_state'] = STATE_IDLE
                self.trade_state['pending_setup'] = None
                return
            
            # Generate trade ID
            trade_id = f"trade_{int(time.time() * 1000)}"
            
            # Get minimum notional requirement
            min_notional = self.binance_client.get_min_notional(self.config.symbol)
            current_price = base_entry_price
            
            # Calculate quote order quantity for market buy
            quote_order_qty = position_size * base_entry_price
            
            # Ensure minimum notional is met
            if quote_order_qty < min_notional:
                logger.warning(f"Calculated order value ${quote_order_qty:.2f} below minimum notional ${min_notional:.2f}. "
                             f"Increasing to minimum.")
                quote_order_qty = min_notional
                # Recalculate position size based on minimum notional
                position_size = quote_order_qty / current_price
            
            # Execute entry order
            direction = confirmed_setup['direction']
            entry_order = self.live_execution_engine.execute_entry_order(
                symbol=self.config.symbol,
                direction=direction,
                quantity=position_size,
                trade_id=trade_id,
                quote_order_qty=quote_order_qty if direction == "long" else None
            )
            
            # Wait for fill
            entry_order_id = entry_order['orderId']
            filled_order = self.live_execution_engine.wait_for_order_fill(
                entry_order_id,
                timeout_seconds=self.config.order_timeout_seconds
            )
            
            if filled_order is None:
                logger.error(f"Entry order {entry_order_id} did not fill")
                self.trade_state['current_state'] = STATE_IDLE
                self.trade_state['pending_setup'] = None
                return
            
            # Get actual fill price
            actual_entry_price = self.live_execution_engine.get_fill_price(filled_order['raw_response'])
            actual_position_size = filled_order['executed_qty']
            
            # Place stop-loss and take-profit orders
            sl_tp_orders = self.live_execution_engine.place_stop_loss_take_profit(
                symbol=self.config.symbol,
                direction=direction,
                quantity=actual_position_size,
                stop_loss_price=risk_params['stop_loss_price'],
                take_profit_price=risk_params['take_profit_price'],
                trade_id=trade_id
            )
            
            # Create active trade record
            active_trade = {
                'trade_id': trade_id,
                'entry_timestamp': current_candle.name,
                'entry_price': actual_entry_price,
                'direction': direction,
                'zone_index': confirmed_setup['zone_index'],
                'zone_type': zone['zone_type'],
                'setup_type': 'rebound' if 'rebound' in confirmed_setup['event_type'] else 'breakout',
                'stop_loss_price': risk_params['stop_loss_price'],
                'take_profit_price': risk_params['take_profit_price'],
                'position_size': actual_position_size,
                'risk_amount': risk_amount,
                'risk_per_unit': risk_params['risk_per_unit'],
                'entry_order_id': entry_order_id,
                'stop_loss_order_id': sl_tp_orders.get('stop_loss_order_id'),
                'take_profit_order_id': sl_tp_orders.get('take_profit_order_id')
            }
            
            # Register position
            self.position_tracker.register_position(
                trade_id=trade_id,
                entry_price=actual_entry_price,
                position_size=actual_position_size,
                direction=direction,
                stop_loss_price=risk_params['stop_loss_price'],
                take_profit_price=risk_params['take_profit_price'],
                entry_order_id=entry_order_id,
                stop_loss_order_id=sl_tp_orders.get('stop_loss_order_id'),
                take_profit_order_id=sl_tp_orders.get('take_profit_order_id')
            )
            
            # Update trade state
            self.trade_state['current_state'] = STATE_IN_TRADE
            self.trade_state['active_trade'] = active_trade
            self.trade_state['pending_setup'] = None
            
            logger.info(f"Entered trade {trade_id}: {direction} {actual_position_size} @ {actual_entry_price}")
            
        except Exception as e:
            logger.error(f"Error executing trade entry: {e}", exc_info=True)
            self.trade_state['current_state'] = STATE_IDLE
            self.trade_state['pending_setup'] = None
    
    def _monitor_positions(self, current_candle: pd.Series, context_df: pd.DataFrame):
        """Monitor open positions for exit conditions."""
        active_trade = self.trade_state.get('active_trade')
        
        if active_trade is None:
            return
        
        trade_id = active_trade['trade_id']
        
        # Update position price
        current_price = current_candle['close']
        self.position_tracker.update_position_price(trade_id, current_price)
        
        # Check for RSI exit
        if self.signal_filter is not None:
            if self.signal_filter.check_rsi_exit_filter(active_trade, current_candle, context_df):
                logger.info(f"RSI exit triggered for trade {trade_id}")
                self._execute_trade_exit(trade_id, current_price, "rsi_exit")
                return
        
        # Check order status (stop-loss/take-profit handled by Binance)
        orders = self.order_manager.get_orders_for_trade(trade_id)
        for order in orders:
            if order['order_type'] in ('stop_loss', 'take_profit'):
                self.order_manager.update_order_status(order['order_id'])
                order_status = order['status']
                
                if order_status == OrderStatus.FILLED:
                    exit_reason = 'stop_loss' if order['order_type'] == 'stop_loss' else 'take_profit'
                    logger.info(f"{exit_reason} triggered for trade {trade_id}")
                    self._execute_trade_exit(trade_id, current_price, exit_reason)
                    return
    
    def _execute_trade_exit(self, trade_id: str, exit_price: float, exit_reason: str):
        """Execute trade exit."""
        try:
            active_trade = self.trade_state.get('active_trade')
            if active_trade is None or active_trade['trade_id'] != trade_id:
                return
            
            # Cancel remaining orders
            self.live_execution_engine.cancel_stop_loss_take_profit(trade_id)
            
            # If manual exit (not stop-loss/take-profit), place market exit order
            if exit_reason == "rsi_exit":
                direction = active_trade['direction']
                position_size = active_trade['position_size']
                
                exit_order = self.live_execution_engine.execute_exit_order(
                    symbol=self.config.symbol,
                    direction=direction,
                    quantity=position_size,
                    trade_id=trade_id
                )
                
                # Wait for fill
                exit_order_id = exit_order['orderId']
                filled_order = self.live_execution_engine.wait_for_order_fill(
                    exit_order_id,
                    timeout_seconds=self.config.order_timeout_seconds
                )
                
                if filled_order:
                    exit_price = self.live_execution_engine.get_fill_price(filled_order['raw_response'])
            
            # Close position
            trade_record = self.position_tracker.close_position(trade_id, exit_price, exit_reason)
            
            if trade_record:
                # Update account balance
                self.account_balance += trade_record['realized_pnl']
                
                # Log trade
                self.trades_log.append(trade_record)
                self._log_trade(trade_record)
                
                logger.info(f"Exited trade {trade_id}: PnL = ${trade_record['realized_pnl']:.2f}")
            
            # Update trade state
            if self.config.cooldown_candles > 0:
                self.trade_state['current_state'] = STATE_COOLDOWN
                self.trade_state['cooldown_remaining'] = self.config.cooldown_candles
            else:
                self.trade_state['current_state'] = STATE_IDLE
            
            self.trade_state['active_trade'] = None
            
        except Exception as e:
            logger.error(f"Error executing trade exit: {e}", exc_info=True)
    
    def _calculate_risk_parameters(self, confirmed_setup: Dict[str, Any], zone: pd.Series,
                                   entry_price: float, atr_value: float) -> Dict[str, Any]:
        """Calculate stop loss and take profit prices for a confirmed setup."""
        # Cap ATR if risk manager provided
        if self.risk_manager is not None:
            atr_value = self.risk_manager.cap_atr_value(atr_value)
        
        setup_type = confirmed_setup['event_type']
        direction = confirmed_setup['direction']
        zone_type = zone['zone_type']
        lower_boundary = zone['lower_boundary']
        upper_boundary = zone['upper_boundary']
        
        # Calculate stop loss
        if 'rebound' in setup_type:
            if direction == 'long':
                stop_loss_price = lower_boundary - (self.config.rebound_stop_atr_mult * atr_value)
            else:  # short
                stop_loss_price = upper_boundary + (self.config.rebound_stop_atr_mult * atr_value)
        elif 'breakout' in setup_type:
            if direction == 'long':
                stop_loss_price = upper_boundary - (self.config.breakout_stop_atr_mult * atr_value)
            else:  # short
                stop_loss_price = lower_boundary + (self.config.breakout_stop_atr_mult * atr_value)
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
            take_profit_price = entry_price + (risk_per_unit * self.config.take_profit_r_multiple)
        else:  # short
            take_profit_price = entry_price - (risk_per_unit * self.config.take_profit_r_multiple)
        
        return {
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'risk_per_unit': risk_per_unit
        }
    
    def _log_trade(self, trade_record: Dict[str, Any]):
        """Log trade to CSV file."""
        try:
            import csv
            
            file_exists = os.path.exists(self.config.trades_log_file)
            
            with open(self.config.trades_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade_record.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade_record)
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def recover_state(self):
        """Recover trading state from Binance API."""
        if not self.config.state_recovery_enabled:
            return
        
        logger.info("Recovering state from Binance API...")
        
        recovered_items = 0
        
        # Try to recover positions (non-fatal)
        try:
            positions = self.state_recovery.recover_positions()
            for position in positions:
                logger.info(f"Recovered position: {position['quantity']} @ {position['current_price']}")
                recovered_items += 1
        except Exception as e:
            logger.warning(f"Could not recover positions: {e} (this is normal if no positions exist)")
        
        # Try to recover orders (non-fatal)
        try:
            orders = self.state_recovery.recover_orders()
            if orders:
                logger.info(f"Recovered {len(orders)} open order(s)")
                recovered_items += len(orders)
        except Exception as e:
            logger.warning(f"Could not recover orders: {e} (this is normal if no orders exist)")
        
        # Try to sync order manager (non-fatal)
        try:
            synced_orders = self.order_manager.sync_open_orders()
            if synced_orders:
                logger.info(f"Synced {len(synced_orders)} order(s) with order manager")
        except Exception as e:
            logger.warning(f"Could not sync orders: {e}")
        
        if recovered_items > 0:
            logger.info(f"State recovery complete: recovered {recovered_items} item(s)")
        else:
            logger.info("State recovery complete: no existing positions or orders found (starting fresh)")
    
    def _close_all_positions_on_start(self):
        """Close all open positions and cancel all orders on startup."""
        if not self.config.close_positions_on_start:
            logger.info("Position cleanup on start is disabled")
            return
        
        logger.info("Closing all positions and cancelling orders on startup...")
        
        try:
            # Step 1: Cancel all open orders
            open_orders = self.binance_client.get_open_orders(symbol=self.config.symbol)
            if open_orders:
                logger.info(f"Found {len(open_orders)} open order(s) to cancel")
                cancelled = self.binance_client.cancel_all_orders(symbol=self.config.symbol)
                logger.info(f"Cancelled {len(cancelled)} order(s)")
            else:
                logger.info("No open orders to cancel")
            
            # Step 2: Check for open positions (base asset balance > 0)
            base_asset = self.config.symbol.replace("USDT", "").replace("USD", "")
            base_balance = self.binance_client.get_account_balance(asset=base_asset)
            
            if base_balance > 0:
                logger.info(f"Found open position: {base_balance} {base_asset}")
                
                # Get current price
                current_price = self.binance_client.get_current_price(self.config.symbol)
                
                # Place market SELL order to close position
                logger.info(f"Closing position with market SELL order @ {current_price}")
                sell_order = self.binance_client.place_market_order(
                    symbol=self.config.symbol,
                    side="SELL",
                    quantity=base_balance
                )
                
                # Wait for order to fill
                order_id = sell_order['orderId']
                logger.info(f"Placed market sell order {order_id}, waiting for fill...")
                
                # Poll for order status
                import time
                max_wait = 30
                waited = 0
                while waited < max_wait:
                    order_status = self.binance_client.get_order_status(
                        symbol=self.config.symbol,
                        order_id=order_id
                    )
                    
                    if order_status['status'] == 'FILLED':
                        logger.info(f"Position closed successfully. Order {order_id} filled.")
                        break
                    
                    time.sleep(1)
                    waited += 1
                
                if waited >= max_wait:
                    logger.warning(f"Order {order_id} did not fill within {max_wait} seconds")
            else:
                logger.info("No open positions to close")
            
            # Step 3: Clear position tracker
            self.position_tracker.positions.clear()
            logger.info("Cleared position tracker")
            
            # Step 4: Reset trade state
            self.trade_state = {
                'current_state': STATE_IDLE,
                'active_trade': None,
                'pending_setup': None,
                'cooldown_remaining': 0
            }
            logger.info("Reset trade state to IDLE")
            
            logger.info("Position cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during position cleanup: {e}", exc_info=True)
            # Don't raise - allow bot to continue even if cleanup fails
    
    def start(self):
        """Start live trading."""
        logger.info("Starting live trading...")
        
        # Close all positions on start if enabled
        if self.config.close_positions_on_start:
            self._close_all_positions_on_start()
        
        # Load positions from backup if exists
        if self.config.state_recovery_enabled:
            self.position_tracker.load_positions(self.config.position_backup_file)
        
        # Recover state if enabled
        if self.config.state_recovery_enabled:
            self.recover_state()
        
        # Sync positions with Binance after recovery
        if self.config.state_recovery_enabled:
            synced_positions = self.position_tracker.sync_with_binance()
            if synced_positions:
                logger.info(f"Synced {len(synced_positions)} position(s) with Binance")
                # Restore active trade state if we have a position
                if synced_positions:
                    # Use the first synced position to restore trade state
                    position = synced_positions[0]
                    # Try to match with existing trade_id or create synthetic one
                    trade_id = position.get('trade_id', f"recovered_{int(time.time() * 1000)}")
                    self.trade_state['active_trade'] = {
                        'trade_id': trade_id,
                        'entry_price': position.get('entry_price', position.get('current_price', 0)),
                        'direction': position.get('direction', 'long'),
                        'position_size': position.get('position_size', 0),
                        'stop_loss_price': position.get('stop_loss_price'),
                        'take_profit_price': position.get('take_profit_price')
                    }
                    self.trade_state['current_state'] = STATE_IN_TRADE
                    logger.info(f"Restored trade state from synced position: {trade_id}")
        
        # Initialize dashboard if enabled
        if self.config.enable_dashboard:
            try:
                from trader.live.dashboard import LiveTradingDashboard
                self.dashboard = LiveTradingDashboard(self)
                self.dashboard.start()
                logger.info("Dashboard started")
            except Exception as e:
                logger.warning(f"Failed to start dashboard: {e}. Continuing without dashboard.")
                self.dashboard = None
        
        # Start market data stream
        self.market_data.start()
        
        logger.info("Live trading started. Waiting for signals...")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
                
                # Check kill switch periodically
                if self._check_kill_switch():
                    break
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping...")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop live trading."""
        logger.info("Stopping live trading...")
        
        # Stop dashboard if running
        if self.dashboard:
            try:
                self.dashboard.stop()
            except Exception as e:
                logger.warning(f"Error stopping dashboard: {e}")
        
        # Save positions to backup
        if self.config.state_recovery_enabled:
            try:
                self.position_tracker.save_positions(self.config.position_backup_file)
            except Exception as e:
                logger.warning(f"Error saving positions backup: {e}")
        
        # Stop market data stream
        self.market_data.stop()
        
        logger.info("Live trading stopped")

