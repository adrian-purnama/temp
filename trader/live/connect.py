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
def handle_socket_message(msg):
    """Handle incoming websocket message and convert to candle format."""
    try:
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
        else:
            # Debug: show if we're getting other message types
            print(f"Received non-kline message: {list(msg.keys())}")
                
    except Exception as e:
        print(f"Error handling socket message: {e}")
        import traceback
        traceback.print_exc()


def start_market_data_stream(client, symbol=None, interval='15m'):
    if symbol is None:
        symbol = Config.symbol
    
    print(f"Starting market data stream for {symbol} @ {interval}")
    
    # Create websocket manager
    twm = ThreadedWebsocketManager()
    twm.start()
    
    # Give it a moment to initialize
    import time
    time.sleep(1)
    
    # Start kline stream
    twm.start_kline_socket(
        callback=handle_socket_message,
        symbol=symbol.lower(),
        interval=interval
    )
    
    return twm


def get_next_candle():
    return candle_queue.get()


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