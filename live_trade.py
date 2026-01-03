from trader.live.connect import test_connection, get_account_info, connect_binance
from trader.live.prepare import prepare_live_trading
from trader.live.engine import run_live_trading
from trader.config import Config


def main():
    print(80 * "=")
    print("Live Trading Bot - Starting Up")
    print(80 * "=")
    print()
    
    # Test connection
    print("Testing Connection to Binance...")
    result = test_connection()
    if result:
        print("✓ Connection successful")
    else:
        print("✗ Connection failed - Exiting")
        return
    print()
    
    # Get account info
    print("Retrieving Account Info...")
    account_info = get_account_info()
    print()
    
    if account_info:
        print(80 * "=")
        print("Account Info")
        print(80 * "=")
        
        # Account Type
        print(f"Account Type: {account_info.get('account_type', 'N/A')}")
        print()
        
        # Permissions
        print("Permissions:")
        print(f"  Can Trade:    {'✓ Yes' if account_info.get('can_trade') else '✗ No'}")
        print(f"  Can Withdraw: {'✓ Yes' if account_info.get('can_withdraw') else '✗ No'}")
        print(f"  Can Deposit:  {'✓ Yes' if account_info.get('can_deposit') else '✗ No'}")
        print()
        
        # Trading Fees
        maker_comm = account_info.get('maker_commission', 0)
        taker_comm = account_info.get('taker_commission', 0)
        print("Trading Fees:")
        print(f"  Maker Commission: {maker_comm / 100:.4f}%")
        print(f"  Taker Commission: {taker_comm / 100:.4f}%")
        print()
        
        # Balances
        balances = account_info.get('balances', [])
        if balances:
            print("Balances:")
            print(f"{'Asset':<10} {'Free':<20} {'Locked':<20} {'Total':<20}")
            print("-" * 70)
            for balance in balances:
                asset = balance.get('asset', 'N/A')
                free = balance.get('free', 0)
                locked = balance.get('locked', 0)
                total = balance.get('total', 0)
                print(f"{asset:<10} {free:<20.8f} {locked:<20.8f} {total:<20.8f}")
        else:
            print("Balances: No balances found")
    else:
        print("✗ Failed to retrieve account info - Exiting")
        return
    
    print(80 * "=")
    print()
    
    # Prepare trading environment
    print("Preparing Trading Environment...")
    client = connect_binance()
    prepare_status = prepare_live_trading(client)
    print()
    
    # Show trading mode
    config = Config()
    trading_mode = config.trading_mode.lower()
    mode_display = "PAPER TRADING" if trading_mode == "paper" else "LIVE TRADING"
    
    print(80 * "=")
    print(f"Starting {mode_display} Bot")
    print(80 * "=")
    print(f"Trading Mode: {mode_display}")
    print(f"Symbol: {config.symbol}")
    
    if trading_mode == "paper":
        print(f"Initial Balance: {config.paper_initial_balance:.2f} USDT")
        print(f"Position Size: {config.paper_position_size_percent}% per trade")
        print(f"Take Profit: {config.take_profit_atr_multiplier}x ATR")
        print(f"Stop Loss: {config.stop_loss_atr_multiplier}x ATR")
    
    print()
    print("Press Ctrl+C to stop the bot gracefully")
    print(80 * "=")
    print()
    
    # Start trading bot
    try:
        run_live_trading(client)
    except KeyboardInterrupt:
        print("\n" + 80 * "=")
        print("Bot stopped by user")
        print(80 * "=")
    except Exception as e:
        print(f"\n✗ Error running trading bot: {e}")
        raise


if __name__ == "__main__":
    main()



