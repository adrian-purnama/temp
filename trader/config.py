from dataclasses import dataclass

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
    
    # Risk management multipliers (ATR-based)
    take_profit_atr_multiplier: float = 1.5  # Take profit at 2x ATR above entry
    stop_loss_atr_multiplier: float = 1.0    # Stop loss at 1x ATR below entry
    
    # Paper trading balance
    paper_initial_balance: float = 100.0  # Starting balance for paper trading (USDT)
    paper_position_size_percent: float = 10.0  # Percentage of balance to use per position (e.g., 10.0 = 10%)

    def get_endPoint(self):
        if self.testnet:
            return "https://testnet.binance.vision"
        else:
            return "https://api.binance.com"