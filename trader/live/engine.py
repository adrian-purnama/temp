import sys
from pathlib import Path
import pandas as pd
import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import Config
from trader.signals.sr_signal import SRSignal, add_atr

# Global state
candles: List[Dict[str, Any]] = []
signal_state: Optional[Dict[str, Any]] = None
signal_generator = SRSignal()
min_candles = 200  # Need enough for ATR + window_size
candle_count = 0  # Counter for periodic status updates

# Global positions dictionary (stored in RAM for fast access)
open_positions: Dict[str, Dict[str, Any]] = {}

# Global paper balance (stored in RAM for fast access)
paper_balance: float = 0.0

# Telegram notification scheduler
telegram_scheduler_thread: Optional[threading.Timer] = None
last_telegram_notification: Optional[datetime] = None
initial_balance: float = 0.0  # Track initial balance for PnL calculation


def load_positions() -> Dict[str, Dict[str, Any]]:
    """
    Load open positions from JSON file.
    
    Returns
    -------
    dict
        Dictionary of positions loaded from file
    """
    positions_file = project_root / "open_positions.json"
    
    if not positions_file.exists():
        return {}
    
    try:
        with open(positions_file, 'r') as f:
            positions = json.load(f)
            print(f"Loaded {len(positions)} position(s) from file")
            return positions
    except Exception as e:
        print(f"Error loading positions: {e}")
        return {}


def save_positions():
    positions_file = project_root / "open_positions.json"
    try:
        with open(positions_file, 'w') as f:
            json.dump(open_positions, f, indent=2, default=str)
        print(f"Saved {len(open_positions)} position(s) to file")
    except Exception as e:
        print(f"Error saving positions: {e}")


def load_paper_balance() -> float:

    balance_file = project_root / "paper_balance.txt"
    
    if not balance_file.exists():
        config = Config()
        initial_balance = config.paper_initial_balance
        print(f"No balance file found. Using initial balance: {initial_balance:.2f} USDT")
        return initial_balance
    
    try:
        with open(balance_file, 'r') as f:
            balance = float(f.read().strip())
        print(f"Loaded paper balance: {balance:.2f} USDT")
        return balance
    except Exception as e:
        print(f"Error loading paper balance: {e}")
        config = Config()
        return config.paper_initial_balance


def save_paper_balance():
    """
    Save paper balance to text file.
    """
    global paper_balance
    balance_file = project_root / "paper_balance.txt"
    
    try:
        with open(balance_file, 'w') as f:
            f.write(str(paper_balance))
        print(f"Saved paper balance: {paper_balance:.2f} USDT")
    except Exception as e:
        print(f"Error saving paper balance: {e}")


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0  # Default neutral RSI
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0


def paper_buy(signal: Dict[str, Any], client=None):
    global open_positions, candles, paper_balance
    
    # Generate unique position ID
    position_id = str(uuid.uuid4())[:8]
    
    # Get entry price from signal metadata or current candle close
    metadata = signal.get('metadata', {})
    buy_price = metadata.get('entry_price')
    
    # If no entry_price in metadata, use current price from last candle
    if buy_price is None:
        if candles:
            buy_price = candles[-1]['close']
        else:
            print("[PAPER BUY] Error: No price available")
            return
    
    # Only support long positions (buy low, sell high)
    # Only execute if direction is long (1)
    direction = signal['direction']
    if direction != 1:
        print(f"[PAPER BUY] Skipping non-long signal (direction: {direction})")
        return
    
    # Calculate sell price (take profit) and stop loss
    # These should come from signal metadata or be calculated
    atr_value = metadata.get('atr_value', buy_price * 0.02)  # Default 2% if no ATR
    
    # Use config values for ATR multipliers
    config = Config()
    sell_price = buy_price + (config.take_profit_atr_multiplier * atr_value)
    stop_loss = buy_price - (config.stop_loss_atr_multiplier * atr_value)
    
    # Calculate position quantity based on percentage of paper balance
    config = Config()
    position_size_usdt = paper_balance * (config.paper_position_size_percent / 100.0)
    quantity = position_size_usdt / buy_price
    position_cost = buy_price * quantity
    
    # Check if we have enough balance
    if paper_balance < position_cost:
        print(f"[PAPER BUY] Insufficient balance! Required: {position_cost:.2f} USDT, Available: {paper_balance:.2f} USDT")
        return
    
    # Deduct balance for the purchase
    paper_balance -= position_cost
    
    # Create position entry (long only)
    position = {
        'buy_price': float(buy_price),
        'sell_price': float(sell_price),  # Take profit price (higher than buy)
        'stop_loss': float(stop_loss),   # Cut loss price (lower than buy)
        'entry_time': datetime.now().isoformat(),
        'direction': 1,  # Always long (buy low, sell high)
        'quantity': float(quantity),
        'cost': float(position_cost),  # Store the cost for balance tracking
        'signal': signal  # Store original signal data
    }
    
    # Add to RAM dictionary
    open_positions[position_id] = position
    
    print(f"[PAPER BUY] Position {position_id} | "
          f"Buy: {buy_price:.2f} | Sell: {sell_price:.2f} | Stop Loss: {stop_loss:.2f}")
    print(f"[PAPER BUY] Quantity: {quantity} | Cost: {position_cost:.2f} USDT | Balance: {paper_balance:.2f} USDT")
    print(f"[PAPER BUY] Direction: LONG (buy low, sell high) | "
          f"Strength: {signal['strength']}")
    
    # Save balance after purchase
    save_paper_balance()


def paper_sell(position_id: str, reason: str, candle_data: Dict[str, Any], client=None):
    global open_positions, paper_balance
    
    if position_id not in open_positions:
        return
    
    position = open_positions[position_id]
    current_price = candle_data['close']
    buy_price = position['buy_price']
    quantity = position.get('quantity', 1.0)
    
    # Calculate proceeds from sale
    sale_proceeds = current_price * quantity
    
    # Calculate PnL
    direction = position['direction']
    cost = position.get('cost', buy_price * quantity)
    pnl = sale_proceeds - cost
    pnl_pct = (pnl / cost) * 100
    
    # Add proceeds back to balance
    paper_balance += sale_proceeds
    
    print(f"[PAPER SELL] Position {position_id} | "
          f"Buy: {buy_price:.2f} | Sell: {current_price:.2f} | "
          f"Quantity: {quantity} | PnL: {pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
    print(f"[PAPER SELL] Sale Proceeds: {sale_proceeds:.2f} USDT | Balance: {paper_balance:.2f} USDT")
    
    # Remove from RAM
    del open_positions[position_id]
    
    # Save to file immediately after selling
    save_positions()
    save_paper_balance()


def check_exit_strategy(candle_data: Dict[str, Any], client=None):
    global open_positions, candles
    
    config = Config()
    if config.trading_mode.lower() != "paper":
        return  # Only handle paper trading for now
    
    if not open_positions:
        return
    
    current_price = candle_data['close']
    positions_to_remove = []
    

    rsi = 50.0  # Default
    if len(candles) >= 15:
        recent_closes = pd.Series([c['close'] for c in candles[-15:]])
        rsi = calculate_rsi(recent_closes)
    
    # Check RSI > 70 condition first
    if rsi > 70:
        # If RSI > 70 AND current_price > buy_price → sell position
        for position_id, position in list(open_positions.items()):
            buy_price = position['buy_price']
            if current_price > buy_price:
                paper_sell(position_id, "rsi_exit", candle_data, client)
                positions_to_remove.append(position_id)
    
    # Check stop loss and take profit for remaining positions (long only - buy low, sell high)
    for position_id, position in list(open_positions.items()):
        if position_id in positions_to_remove:
            continue
        
        buy_price = float(position['buy_price'])
        sell_price = float(position['sell_price'])
        stop_loss = float(position['stop_loss'])
        
        # Only support long positions (buy low, sell high)
        # Check stop loss (cut loss) - price goes down
        if current_price <= stop_loss:
            paper_sell(position_id, "stop_loss", candle_data, client)
            positions_to_remove.append(position_id)
            continue
        
        # Check take profit - price goes up
        if current_price >= sell_price:
            paper_sell(position_id, "take_profit", candle_data, client)
            positions_to_remove.append(position_id)
            continue


def process_candle_update(candle_data: Dict[str, Any], client=None):
    global candles, signal_state, candle_count
    
    is_closed = candle_data.get('is_closed', False)
    
    # Always check exit strategy (for all candles - closed and non-closed)
    check_exit_strategy(candle_data, client)
    
    # Only generate signals on closed candles
    if not is_closed:
        return None
    
    # For closed candles, add to history and generate signal
    candles.append(candle_data)
    candle_count += 1
    
    if len(candles) > min_candles + 50:
        candles = candles[-min_candles:]
    
    if len(candles) < min_candles:
        print(f"[SR DEBUG] Collecting candles: {len(candles)}/{min_candles} (need {min_candles - len(candles)} more)")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Add ATR
    df = add_atr(df, period=14)
    
    # Get context window (ending at t-1) and current candle
    current_idx = len(df) - 1
    context_df = df.iloc[:current_idx].tail(192)  # Last 192 candles
    current_candle = df.iloc[current_idx]
    
    # Generate signal (only on closed candles)
    signal, updated_state = signal_generator.generate_signal(
        candle=current_candle,
        context=context_df,
        state=signal_state,
        current_index=current_idx
    )
    
    # Update state
    signal_state = updated_state
    
    # Extract signal metadata for logging
    metadata = signal.get('metadata', {})
    direction = signal.get('direction', 0)
    strength = signal.get('strength', 0.0)
    
    # Log signal generation result (including flat signals)
    if direction != 0:
        print(f"[SR SIGNAL] ✓ Actionable signal generated: direction={direction}, strength={strength:.2f}")
        print(f"           Setup: {metadata.get('setup_type', 'N/A')}, Zone: {metadata.get('zone_type', 'N/A')}")
        if metadata.get('entry_price'):
            print(f"           Entry price: {metadata.get('entry_price'):.2f}")
    else:
        # Log why signal is flat
        waiting_state = metadata.get('waiting_state')
        if waiting_state:
            candles_waited = metadata.get('candles_waited', 0)
            candles_required = metadata.get('candles_required', 0)
            waiting_direction = metadata.get('waiting_direction', 'N/A')
            print(f"[SR SIGNAL] → Flat signal (waiting for confirmation): {waiting_state}")
            print(f"           Direction: {waiting_direction}, Progress: {candles_waited}/{candles_required} candles")
        else:
            print(f"[SR SIGNAL] → Flat signal (no confirmed events, no waiting state)")
    
    # Periodic status summary (every 20 candles)
    if candle_count % 20 == 0:
        zones_df = updated_state.get('zones_df')
        waiting_state = updated_state.get('waiting_state')
        num_zones = len(zones_df) if zones_df is not None and hasattr(zones_df, '__len__') else 0
        
        print(f"[SR STATUS] Candle #{candle_count} | Zones: {num_zones} | ", end="")
        if waiting_state:
            print(f"Waiting: {waiting_state.get('state_type', 'N/A')} ({waiting_state.get('candles_waited', 0)}/{waiting_state.get('confirmation_candles_required', 0)})")
        else:
            print("Waiting: None")
    
    # Return signal if direction is not flat
    if signal['direction'] != 0:
        return signal
    
    return None


def run_trading_loop(get_next_candle, client=None):
    config = Config()
    trading_mode = config.trading_mode.lower()
    
    print(f"Starting trading loop... Mode: {trading_mode.upper()}")
    
    if trading_mode == "paper":
        # Paper trading loop
        while True:
            try:
                candle_data = get_next_candle()
                is_closed = candle_data.get('is_closed', False)
                
                if is_closed:
                    # Closed candle - process and generate signal
                    signal = process_candle_update(candle_data, client)
                    
                    if signal:
                        print(f"Signal generated: {signal['direction']} (strength: {signal['strength']})")
                        paper_buy(signal, client)
                else:
                    # Non-closed candle (streaming) - check exit strategy only
                    check_exit_strategy(candle_data, client)
                    
            except KeyboardInterrupt:
                print("\nStopping trading loop...")
                # Save positions and balance before shutdown
                save_positions()
                save_paper_balance()
                break
            except Exception as e:
                print(f"Error processing candle: {e}")
                continue
                
    elif trading_mode == "live":
        # Live trading loop
        while True:
            try:
                candle_data = get_next_candle()
                is_closed = candle_data.get('is_closed', False)
                
                if is_closed:
                    # Closed candle - process and generate signal
                    signal = process_candle_update(candle_data, client)
                    
                    if signal:
                        print(f"Signal generated: {signal['direction']} (strength: {signal['strength']})")
                        print(f"[LIVE] Executing trade: {signal['direction']}")
                        # TODO: Execute real trade
                else:
                    # Non-closed candle (streaming) - check exit strategy only
                    check_exit_strategy(candle_data, client)
                    
            except KeyboardInterrupt:
                print("\nStopping trading loop...")
                # Save positions and balance before shutdown
                save_positions()
                save_paper_balance()
                break
            except Exception as e:
                print(f"Error processing candle: {e}")
                continue


def send_daily_balance_notification():
    """Send daily balance notification to Telegram and schedule the next one."""
    global paper_balance, open_positions, initial_balance, telegram_scheduler_thread, last_telegram_notification
    
    from trader.live.telegram_notifier import send_balance_update
    config = Config()
    
    # Only send if Telegram is configured
    if config.telegram_bot_token and config.telegram_chat_id:
        success = send_balance_update(
            balance=paper_balance,
            initial_balance=initial_balance,
            open_positions=open_positions,
            symbol=config.symbol,
            interval=config.interval
        )
        
        if success:
            last_telegram_notification = datetime.now()
            print(f"[TELEGRAM] Daily balance notification sent. Next notification in 24 hours.")
        else:
            print(f"[TELEGRAM] Failed to send notification. Will retry in 24 hours.")
            print(f"[TELEGRAM] Please check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
    else:
        print(f"[TELEGRAM] Telegram not configured. Skipping notification.")
    
    # Schedule next notification in 24 hours (even if this one failed)
    schedule_next_daily_notification()


def schedule_next_daily_notification():
    """Schedule the next daily balance notification."""
    global telegram_scheduler_thread
    
    # Cancel existing timer if any
    if telegram_scheduler_thread and telegram_scheduler_thread.is_alive():
        telegram_scheduler_thread.cancel()
    
    # Schedule next notification in 24 hours (86400 seconds)
    telegram_scheduler_thread = threading.Timer(86400.0, send_daily_balance_notification)
    telegram_scheduler_thread.daemon = True  # Allow program to exit even if timer is running
    telegram_scheduler_thread.start()
    next_time = datetime.now() + timedelta(hours=24)
    print(f"[TELEGRAM] Next daily notification scheduled for: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")


def start_telegram_notifications():
    """Start the daily Telegram notification scheduler."""
    global initial_balance, paper_balance
    
    config = Config()
    
    # Only start if Telegram is configured
    if not config.telegram_bot_token or not config.telegram_chat_id:
        print("[TELEGRAM] Telegram not configured. Daily notifications disabled.")
        print("[TELEGRAM] Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.")
        return
    
    # Test connection first
    from trader.live.telegram_notifier import test_telegram_connection
    print("[TELEGRAM] Testing bot connection...")
    if not test_telegram_connection():
        print("[TELEGRAM] Connection test failed. Please check your configuration.")
        print("[TELEGRAM] Daily notifications will still be scheduled, but may fail.")
    
    # Store initial balance for PnL calculation (use config value, not current balance)
    # This ensures PnL is calculated from the true starting point
    initial_balance = config.paper_initial_balance
    
    # Send immediate notification on startup
    print("[TELEGRAM] Sending startup balance notification...")
    send_daily_balance_notification()
    
    # Schedule daily notifications
    schedule_next_daily_notification()


def run_live_trading(client=None):
    global open_positions, paper_balance, candles, initial_balance
    
    from trader.live.connect import start_market_data_stream, get_next_candle, bootstrap_historical_candles
    
    if client is None:
        from trader.live.connect import connect_binance
        client = connect_binance()
    
    # Load positions and balance from file on startup
    open_positions = load_positions()
    paper_balance = load_paper_balance()
    
    # Store initial balance for PnL tracking
    config = Config()
    # Always use the configured initial balance for PnL calculation
    initial_balance = config.paper_initial_balance
    
    # Bootstrap historical candles before starting websocket
    interval = config.interval  # Get interval from config (can be set via env var)
    
    historical_candles = bootstrap_historical_candles(
        client, 
        symbol=config.symbol, 
        interval=interval, 
        limit=min_candles
    )
    
    if historical_candles:
        candles = historical_candles
        print(f"[BOOTSTRAP] Loaded {len(candles)} candles into memory. Ready to start trading.")
    else:
        print(f"[BOOTSTRAP] Warning: No historical candles loaded. Will collect from websocket.")
        candles = []
    
    try:
        twm = start_market_data_stream(client, symbol=config.symbol, interval=interval)
    except ConnectionError as e:
        print(f"\n[FATAL ERROR] Cannot start trading bot: {e}")
        print(f"[FATAL ERROR] Please fix the connection issue and try again.")
        return
    
    # Start Telegram daily notifications
    start_telegram_notifications()
    
    try:
        # Run trading loop (this blocks and processes candles)
        run_trading_loop(get_next_candle, client)
    except KeyboardInterrupt:
        # Save positions and balance before shutdown
        print("\nSaving positions and balance before shutdown...")
        save_positions()
        save_paper_balance()
        raise
    except ConnectionError as e:
        print(f"\n[FATAL ERROR] Websocket connection lost: {e}")
        print(f"[FATAL ERROR] Saving state and shutting down...")
        save_positions()
        save_paper_balance()
        raise
    finally:
        # Cancel Telegram scheduler
        global telegram_scheduler_thread
        if telegram_scheduler_thread and telegram_scheduler_thread.is_alive():
            telegram_scheduler_thread.cancel()
            print("Stopped Telegram notification scheduler")
        
        # Stop websocket when done
        try:
            print("Stopping websocket...")
            twm.stop()
            print("Stopped")
        except Exception as e:
            print(f"[WARNING] Error stopping websocket: {e}")


if __name__ == "__main__":
    # Run live trading - this starts websocket and processes candles
    run_live_trading()
