import sys
from pathlib import Path
import requests
from datetime import datetime
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import Config


def test_telegram_connection(bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """
    Test Telegram bot connection by calling getMe API.
    
    Parameters
    ----------
    bot_token : str, optional
        Telegram bot token. If not provided, will use Config.
    chat_id : str, optional
        Telegram chat ID. If not provided, will use Config.
    
    Returns
    -------
    bool
        True if connection is valid, False otherwise
    """
    config = Config()
    
    bot_token = (bot_token or config.telegram_bot_token).strip()
    chat_id = (chat_id or config.telegram_chat_id).strip()
    
    if not bot_token:
        print("[TELEGRAM TEST] Bot token not configured")
        return False
    
    # Validate token format
    if ':' not in bot_token:
        print(f"[TELEGRAM TEST] Invalid bot token format. Token should be in format 'number:alphanumeric'")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                bot_info = data.get('result', {})
                print(f"[TELEGRAM TEST] ✓ Bot connection successful!")
                print(f"[TELEGRAM TEST] Bot name: {bot_info.get('first_name', 'N/A')}")
                print(f"[TELEGRAM TEST] Bot username: @{bot_info.get('username', 'N/A')}")
                print(f"[TELEGRAM TEST] Chat ID: {chat_id}")
                print(f"[TELEGRAM TEST] Note: Make sure you've sent /start to your bot first!")
                return True
            else:
                print(f"[TELEGRAM TEST] ✗ Bot API returned error: {data.get('description', 'Unknown')}")
                return False
        elif response.status_code == 401:
            print(f"[TELEGRAM TEST] ✗ 401 Unauthorized: Invalid bot token")
            return False
        else:
            print(f"[TELEGRAM TEST] ✗ HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[TELEGRAM TEST] ✗ Connection test failed: {e}")
        return False


def send_telegram_message(message: str, bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """
    Send a message to Telegram.
    
    Parameters
    ----------
    message : str
        Message to send
    bot_token : str, optional
        Telegram bot token. If not provided, will use Config.
    chat_id : str, optional
        Telegram chat ID. If not provided, will use Config.
    
    Returns
    -------
    bool
        True if message was sent successfully, False otherwise
    """
    config = Config()
    
    bot_token = bot_token or config.telegram_bot_token
    chat_id = chat_id or config.telegram_chat_id
    
    if not bot_token or not chat_id:
        print("[TELEGRAM] Bot token or chat ID not configured. Skipping notification.")
        return False
    
    # Clean and validate token (remove whitespace)
    bot_token = bot_token.strip()
    chat_id = chat_id.strip()
    
    # Validate token format (should be like "123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
    if ':' not in bot_token:
        print(f"[TELEGRAM ERROR] Invalid bot token format. Token should be in format 'number:alphanumeric'")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        
        # Check response status
        if response.status_code == 404:
            error_data = response.json() if response.content else {}
            error_desc = error_data.get('description', 'Unknown error')
            print(f"[TELEGRAM ERROR] 404 Not Found: {error_desc}")
            print(f"[TELEGRAM ERROR] Possible causes:")
            print(f"  - Invalid bot token (check TELEGRAM_BOT_TOKEN in .env)")
            print(f"  - Invalid chat ID (check TELEGRAM_CHAT_ID in .env)")
            print(f"  - Bot not started (send /start to your bot first)")
            print(f"  - Bot token format: {bot_token[:10]}...{bot_token[-5:]}")
            return False
        elif response.status_code == 400:
            error_data = response.json() if response.content else {}
            error_desc = error_data.get('description', 'Unknown error')
            print(f"[TELEGRAM ERROR] 400 Bad Request: {error_desc}")
            return False
        elif response.status_code == 401:
            print(f"[TELEGRAM ERROR] 401 Unauthorized: Invalid bot token")
            return False
        
        response.raise_for_status()
        print(f"[TELEGRAM] Message sent successfully")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[TELEGRAM ERROR] Network error: {e}")
        return False
    except Exception as e:
        print(f"[TELEGRAM ERROR] Failed to send message: {e}")
        return False


def format_balance_message(balance: float, initial_balance: float, open_positions: dict, symbol: str, interval: str = "15m") -> str:
    """
    Format a balance update message for Telegram.
    
    Parameters
    ----------
    balance : float
        Current paper balance
    initial_balance : float
        Initial starting balance
    open_positions : dict
        Dictionary of open positions
    symbol : str
        Trading symbol
    interval : str, default "15m"
        Trading time frame interval
    
    Returns
    -------
    str
        Formatted message
    """
    # Calculate PnL
    pnl = balance - initial_balance
    pnl_pct = (pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    # Format PnL with emoji
    pnl_emoji = "📈" if pnl >= 0 else "📉"
    
    message = f"<b>💰 Daily Balance Update</b>\n\n"
    message += f"<b>Symbol:</b> {symbol}\n"
    message += f"<b>Time Frame:</b> {interval}\n"
    message += f"<b>Current Balance:</b> {balance:.2f} USDT\n"
    message += f"<b>Initial Balance:</b> {initial_balance:.2f} USDT\n"
    message += f"<b>PnL:</b> {pnl_emoji} {pnl:+.2f} USDT ({pnl_pct:+.2f}%)\n"
    message += f"<b>Open Positions:</b> {len(open_positions)}\n"
    message += f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return message


def send_balance_update(balance: float, initial_balance: float, open_positions: dict, symbol: str, interval: str = "15m") -> bool:
    """
    Send a balance update message to Telegram.
    
    Parameters
    ----------
    balance : float
        Current paper balance
    initial_balance : float
        Initial starting balance
    open_positions : dict
        Dictionary of open positions
    symbol : str
        Trading symbol
    interval : str, default "15m"
        Trading time frame interval
    
    Returns
    -------
    bool
        True if message was sent successfully, False otherwise
    """
    message = format_balance_message(balance, initial_balance, open_positions, symbol, interval)
    return send_telegram_message(message)


if __name__ == "__main__":
    # Test function
    config = Config()
    
    print("=" * 50)
    print("Telegram Connection Test")
    print("=" * 50)
    
    # Test connection
    test_telegram_connection()
    
    print("\n" + "=" * 50)
    print("Test Message")
    print("=" * 50)
    
    test_message = format_balance_message(
        balance=105.50,
        initial_balance=100.0,
        open_positions={},
        symbol=config.symbol,
        interval=config.interval
    )
    print("Test message:")
    print(test_message)
    print("\nSending test message...")
    send_telegram_message(test_message)

