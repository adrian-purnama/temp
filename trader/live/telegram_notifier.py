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
        response.raise_for_status()
        print(f"[TELEGRAM] Message sent successfully")
        return True
    except Exception as e:
        print(f"[TELEGRAM ERROR] Failed to send message: {e}")
        return False


def format_balance_message(balance: float, initial_balance: float, open_positions: dict, symbol: str) -> str:
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
    message += f"<b>Current Balance:</b> {balance:.2f} USDT\n"
    message += f"<b>Initial Balance:</b> {initial_balance:.2f} USDT\n"
    message += f"<b>PnL:</b> {pnl_emoji} {pnl:+.2f} USDT ({pnl_pct:+.2f}%)\n"
    message += f"<b>Open Positions:</b> {len(open_positions)}\n"
    message += f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return message


def send_balance_update(balance: float, initial_balance: float, open_positions: dict, symbol: str) -> bool:
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
    
    Returns
    -------
    bool
        True if message was sent successfully, False otherwise
    """
    message = format_balance_message(balance, initial_balance, open_positions, symbol)
    return send_telegram_message(message)


if __name__ == "__main__":
    # Test function
    config = Config()
    test_message = format_balance_message(
        balance=105.50,
        initial_balance=100.0,
        open_positions={},
        symbol=config.symbol
    )
    print("Test message:")
    print(test_message)
    print("\nSending test message...")
    send_telegram_message(test_message)

