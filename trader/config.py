import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from project root if it exists
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded environment variables from {env_file}")

@dataclass
class Config:
    # Required fields (no defaults) must come first
    api_key: str = "LZavWGGWvQVDrgxR9iVLzhhrpMGjynvliBYfQfXggyvR3qE7ccoiaAnbWnh3cWh6"
    api_secret: str = "5P3KuQ9AOD1EUSYYW7Nwm9xCZez3f3azyauHZdc55oY2rzHmHm9KV5GiiQIvSu2J"
    # Optional fields with defaults come after
    api_endpoint: str = "https://api.binance.com"
    symbol: str = "BTCUSDT"
    testnet: bool = False
    trading_mode: str = "paper"
    interval: str = "15m"  # Kline interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
    
    # Risk management multipliers (ATR-based)
    take_profit_atr_multiplier: float = 1.5  # Take profit at 2x ATR above entry
    stop_loss_atr_multiplier: float = 1.0    # Stop loss at 1x ATR below entry
    
    # Paper trading balance
    paper_initial_balance: float = 100.0  # Starting balance for paper trading (USDT)
    paper_position_size_percent: float = 10.0  # Percentage of balance to use per position (e.g., 10.0 = 10%)
    
    # Telegram notifications
    telegram_bot_token: str = ""  # Telegram bot token (get from @BotFather)
    telegram_chat_id: str = ""  # Telegram chat ID to send messages to
    
    def __post_init__(self):
        """Override defaults with environment variables if present."""
        # API credentials
        self.api_key = os.getenv("BINANCE_API_KEY", self.api_key)
        self.api_secret = os.getenv("BINANCE_API_SECRET", self.api_secret)
        
        # API endpoint
        self.api_endpoint = os.getenv("BINANCE_API_ENDPOINT", self.api_endpoint)
        
        # Trading settings
        self.symbol = os.getenv("TRADING_SYMBOL", self.symbol)
        self.testnet = os.getenv("BINANCE_TESTNET", str(self.testnet)).lower() in ("true", "1", "yes")
        self.trading_mode = os.getenv("TRADING_MODE", self.trading_mode)
        self.interval = os.getenv("TRADING_INTERVAL", self.interval)
        
        # Risk management
        if os.getenv("TAKE_PROFIT_ATR_MULTIPLIER"):
            self.take_profit_atr_multiplier = float(os.getenv("TAKE_PROFIT_ATR_MULTIPLIER"))
        if os.getenv("STOP_LOSS_ATR_MULTIPLIER"):
            self.stop_loss_atr_multiplier = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER"))
        
        # Paper trading
        if os.getenv("PAPER_INITIAL_BALANCE"):
            self.paper_initial_balance = float(os.getenv("PAPER_INITIAL_BALANCE"))
        if os.getenv("PAPER_POSITION_SIZE_PERCENT"):
            self.paper_position_size_percent = float(os.getenv("PAPER_POSITION_SIZE_PERCENT"))
        
        # Telegram notifications
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.telegram_bot_token)
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", self.telegram_chat_id)

    def get_endPoint(self):
        if self.testnet:
            return "https://testnet.binance.vision"
        else:
            return "https://api.binance.com"