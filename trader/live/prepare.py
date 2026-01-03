import sys
from pathlib import Path
import os
import json
from typing import Dict, Any

# Add project root to Python path FIRST (before any trader imports)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import trader modules
from trader.config import Config
from trader.live.connect import connect_binance


def prepare_live_trading(client):
    print("Preparing live trading...")
    
    # Get project root directory
    root_dir = Path(__file__).parent.parent.parent
    
    trading_mode = Config.trading_mode.lower()
    
    if trading_mode not in ["paper", "live"]:
        print(f"Warning: Invalid trading_mode '{trading_mode}'. Defaulting to 'paper'.")
        trading_mode = "paper"
    
    mode_display = "PAPER TRADING" if trading_mode == "paper" else "LIVE TRADING"
    print(f"Trading Mode: {mode_display}")
    
    # 2. Check/Create log file
    log_file = root_dir / "live_trading.log"
    log_exists = log_file.exists()
    
    if not log_exists:
        log_file.touch()
        print(f"✓ Created log file: {log_file}")
    else:
        print(f"✓ Log file exists: {log_file}")
    
    # 3. Check/Create positions JSON file
    positions_file = root_dir / "open_positions.json"
    positions_exists = positions_file.exists()
    
    if not positions_exists:
        # Create file with empty dict
        with open(positions_file, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"✓ Created positions file: {positions_file}")
    else:
        print(f"✓ Positions file exists: {positions_file}")
    
    # 4. Check/Create paper balance file
    balance_file = root_dir / "paper_balance.txt"
    balance_exists = balance_file.exists()
    config = Config()
    
    if not balance_exists:
        # Create file with initial balance from config
        initial_balance = config.paper_initial_balance
        with open(balance_file, 'w') as f:
            f.write(str(initial_balance))
        print(f"✓ Created balance file: {balance_file} with initial balance: {initial_balance:.2f} USDT")
    else:
        # Read existing balance
        try:
            with open(balance_file, 'r') as f:
                balance = float(f.read().strip())
            print(f"✓ Balance file exists: {balance_file} with balance: {balance:.2f} USDT")
        except Exception as e:
            print(f"✗ Error reading balance file: {e}. Using initial balance from config.")
            balance = config.paper_initial_balance
    
    # Return status dictionary
    status = {
        'trading_mode': trading_mode,
        'log_file': str(log_file),
        'log_exists': log_exists,
        'positions_file': str(positions_file),
        'positions_exists': positions_exists,
        'balance_file': str(balance_file),
        'balance_exists': balance_exists
    }
    
    print("Preparation complete!")
    return status


if __name__ == "__main__":
    print("Preparing live trading...")
    client = connect_binance()
    print(prepare_live_trading(client))