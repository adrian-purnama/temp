import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import Config
import requests
from binance.client import Client
from binance import ThreadedWebsocketManager
import queue
import pandas as pd


def test_connection():
    """Test connection to Binance API. Returns True if successful, False otherwise."""
    try:
        config = Config()
        api_endpoint = f"{config.get_endPoint()}/api/v3/ping"
        result = requests.get(api_endpoint, timeout=10)
        result.raise_for_status()  
        return True
    except Exception:
        return False


def connect_binance():

    client = Client(
        api_key=Config.api_key,
        api_secret=Config.api_secret,
        testnet=Config.testnet
    )
    
    return client


def get_account_info(api_key=None, api_secret=None):
    if api_key and api_secret:
        Config.api_key = api_key
        Config.api_secret = api_secret

    print("Getting account info using these values:")    
    print(f"API Key: {Config.api_key[0:10]}...")
    print(f"API Secret: {Config.api_secret[0:10]}...")
    print(f"Testnet: {Config.testnet}")
    
    try:
        client = connect_binance()
        account_info = client.get_account()
        
        # Extract only important fields
        important_info = {
            'account_type': 'TESTNET' if Config.testnet else 'MAINNET',
            'can_trade': account_info.get('canTrade', False),
            'can_withdraw': account_info.get('canWithdraw', False),
            'can_deposit': account_info.get('canDeposit', False),
            'maker_commission': account_info.get('makerCommission', 0),
            'taker_commission': account_info.get('takerCommission', 0),
            'balances': []
        }
        
        # Only include non-zero balances
        for balance in account_info.get('balances', []):
            free = float(balance.get('free', 0))
            locked = float(balance.get('locked', 0))
            if free > 0 or locked > 0:
                important_info['balances'].append({
                    'asset': balance.get('asset'),
                    'free': free,
                    'locked': locked,
                    'total': free + locked
                })
        
        return important_info
    except Exception as e:
        print(f"Error getting account info: {e}")
        return None


# Global queue for candles
candle_queue = queue.Queue()

def handle_socket_error(error):
    """Handle websocket connection errors."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    print(f"[WEBSOCKET ERROR] {error_type}: {error_msg}")
    
    # Provide helpful context based on error type
    if isinstance(error, ConnectionRefusedError):
        print(f"[WEBSOCKET ERROR] Connection was refused by Binance server")
        print(f"[WEBSOCKET ERROR] This may be temporary - the connection will attempt to reconnect automatically")
    elif isinstance(error, OSError):
        errno = getattr(error, 'errno', None)
        if errno == 111:
            print(f"[WEBSOCKET ERROR] Connection refused (errno 111)")
        elif errno == 110:
            print(f"[WEBSOCKET ERROR] Connection timeout (errno 110)")
        else:
            print(f"[WEBSOCKET ERROR] Network error (errno {errno})")
        print(f"[WEBSOCKET ERROR] The websocket manager will attempt to reconnect automatically")
    else:
        print(f"[WEBSOCKET ERROR] Unexpected error type: {error_type}")
        import traceback
        traceback.print_exc()


def handle_socket_message(msg):
    """Handle incoming websocket message and convert to candle format."""
    try:
        # Check if this is an error message (multiple formats possible)
        if isinstance(msg, dict):
            # Check for error indicator
            if 'e' in msg:
                if msg.get('e') == 'error' or msg.get('e') == 'ERROR':
                    error_msg = msg.get('m', msg.get('msg', 'Unknown error'))
                    print(f"[WEBSOCKET ERROR] {error_msg}")
                    return
            
            # Check if message has error-like structure ['e', 'type', 'm']
            if 'e' in msg and 'm' in msg and 'type' in msg:
                # This appears to be an error/system message format
                error_type = msg.get('type', 'unknown')
                error_msg = msg.get('m', 'No message')
                if error_type != 'ping' and error_type != 'pong':
                    print(f"[WEBSOCKET] System message - type: {error_type}, message: {error_msg}")
                return
        
        # Check if this is a kline message
        if 'k' in msg:
            kline = msg['k']
            
            # Process ALL candles (closed and non-closed) for exit strategy
            # But mark if candle is closed for signal generation
            candle_data = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': kline['x']  # x = is_closed
            }
            
            # Print market data - show streaming updates
            # if candle_data['is_closed']:
            #     # Closed candle - full details
            #     print(f"[{candle_data['timestamp']}] {Config.symbol} | "
            #           f"O:{candle_data['open']:.2f} H:{candle_data['high']:.2f} "
            #           f"L:{candle_data['low']:.2f} C:{candle_data['close']:.2f} "
            #           f"V:{candle_data['volume']:.4f} [CLOSED]")
            # else:
            #     # Non-closed candle - show price update (streaming data for exit strategy)
            #     print(f"[{candle_data['timestamp']}] {Config.symbol} | "
            #           f"Price: {candle_data['close']:.2f} (streaming...)")
            
            # Put ALL candles in queue (for exit strategy)
            candle_queue.put(candle_data)
        # Other message types are handled above (errors, system messages)
        # If we reach here, it's an unhandled message type - silently ignore
                
    except Exception as e:
        print(f"Error handling socket message: {e}")
        import traceback
        traceback.print_exc()


def start_market_data_stream(client, symbol=None, interval=None, max_retries=3):
    """
    Start market data websocket stream with improved error handling.
    
    Parameters
    ----------
    client : binance.client.Client
        Binance client instance
    symbol : str, optional
        Trading symbol. If None, uses Config.symbol
    interval : str, optional
        Kline interval. If None, uses Config.interval
    max_retries : int, default 3
        Maximum number of connection retry attempts
    
    Returns
    -------
    ThreadedWebsocketManager
        Websocket manager instance
    
    Raises
    ------
    ConnectionError
        If websocket connection fails after all retries
    """
    if symbol is None:
        symbol = Config.symbol
    if interval is None:
        config = Config()
        interval = config.interval
    
    print(f"[WEBSOCKET] Starting market data stream for {symbol} @ {interval}")
    
    import time
    
    for attempt in range(1, max_retries + 1):
        try:
            # Create websocket manager
            twm = ThreadedWebsocketManager()
            twm.start()
            
            # Give it a moment to initialize
            time.sleep(1)
            
            # Start kline stream
            print(f"[WEBSOCKET] Attempting to connect (attempt {attempt}/{max_retries})...")
            twm.start_kline_socket(
                callback=handle_socket_message,
                symbol=symbol.lower(),
                interval=interval
            )
            
            # Give it a moment to establish connection
            time.sleep(2)
            
            print(f"[WEBSOCKET] ✓ Successfully connected to Binance websocket")
            return twm
            
        except ConnectionRefusedError as e:
            error_msg = str(e)
            print(f"[WEBSOCKET ERROR] Connection refused (attempt {attempt}/{max_retries})")
            print(f"[WEBSOCKET ERROR] Details: {error_msg}")
            
            if attempt < max_retries:
                wait_time = attempt * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"[WEBSOCKET] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[WEBSOCKET ERROR] Failed to connect after {max_retries} attempts")
                print(f"[WEBSOCKET ERROR] Possible causes:")
                print(f"  - Network connectivity issues")
                print(f"  - Firewall blocking port 9443 (Binance websocket)")
                print(f"  - Binance API server temporarily unavailable")
                print(f"  - IP restrictions on your Binance API key")
                print(f"  - VPN/proxy configuration issues")
                print(f"[WEBSOCKET ERROR] Troubleshooting:")
                print(f"  1. Check your internet connection")
                print(f"  2. Verify firewall allows outbound connections on port 9443")
                print(f"  3. Check Binance API key IP whitelist settings")
                print(f"  4. Try again in a few minutes (server may be temporarily down)")
                raise ConnectionError(f"Failed to connect to Binance websocket after {max_retries} attempts: {error_msg}")
        
        except OSError as e:
            error_msg = str(e)
            errno = e.errno if hasattr(e, 'errno') else None
            print(f"[WEBSOCKET ERROR] Network error (attempt {attempt}/{max_retries})")
            print(f"[WEBSOCKET ERROR] Error code: {errno}, Details: {error_msg}")
            
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"[WEBSOCKET] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[WEBSOCKET ERROR] Failed to connect after {max_retries} attempts")
                print(f"[WEBSOCKET ERROR] This is a network connectivity issue.")
                print(f"[WEBSOCKET ERROR] Please check your network connection and try again.")
                raise ConnectionError(f"Network error connecting to Binance websocket: {error_msg}")
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"[WEBSOCKET ERROR] Unexpected error: {error_type} (attempt {attempt}/{max_retries})")
            print(f"[WEBSOCKET ERROR] Details: {error_msg}")
            
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"[WEBSOCKET] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[WEBSOCKET ERROR] Failed to connect after {max_retries} attempts")
                print(f"[WEBSOCKET ERROR] Error type: {error_type}")
                import traceback
                print(f"[WEBSOCKET ERROR] Full traceback:")
                traceback.print_exc()
                raise ConnectionError(f"Unexpected error connecting to Binance websocket: {error_msg}")
    
    # Should never reach here, but just in case
    raise ConnectionError("Failed to establish websocket connection")


def get_next_candle():
    return candle_queue.get()


def bootstrap_historical_candles(client, symbol=None, interval=None, limit=200):
    if symbol is None:
        symbol = Config.symbol
    if interval is None:
        config = Config()
        interval = config.interval
    
    print(f"[BOOTSTRAP] Fetching {limit} historical candles for {symbol} @ {interval}...")
    
    try:
        # Fetch historical klines
        klines = client.get_klines(
            symbol=symbol.upper(),
            interval=interval,
            limit=limit
        )
        
        # Convert to same format as websocket messages
        candles = []
        for kline in klines:
            candle_data = {
                'timestamp': pd.to_datetime(kline[0], unit='ms'),  # Open time
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'is_closed': True  # Historical candles are always closed
            }
            candles.append(candle_data)
        
        print(f"[BOOTSTRAP] Successfully loaded {len(candles)} historical candles")
        if len(candles) > 0:
            print(f"[BOOTSTRAP] Date range: {candles[0]['timestamp']} to {candles[-1]['timestamp']}")
        
        return candles
        
    except Exception as e:
        print(f"[BOOTSTRAP ERROR] Failed to fetch historical candles: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    import signal
    import time
    
    print("Testing connection...")
    print(test_connection())
    print("Getting account info...")
    print(get_account_info())  # Uses Config defaults
    
    # Start websocket stream
    twm = start_market_data_stream(connect_binance())
    
    # Keep script running to receive websocket messages
    print("\nWebsocket stream running... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)  # Keep alive
    except KeyboardInterrupt:
        print("\nStopping websocket...")
        twm.stop()
        print("Stopped")