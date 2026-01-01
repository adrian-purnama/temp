# -*- coding: utf-8 -*-
"""
Main entry point for live trading.

Supports environment variable configuration and provides a simple interface
to start live trading on Binance Testnet or Realnet.
"""

import os
import sys
import argparse
from trader.live.config import LiveTradingConfig
from trader.live.live_trader import LiveTrader


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Live Trading Bot for Binance')
    
    # Configuration arguments
    parser.add_argument('--testnet', action='store_true', default=None,
                       help='Use Binance Testnet (default: from BINANCE_TESTNET env var)')
    parser.add_argument('--realnet', action='store_true', default=None,
                       help='Use Binance Realnet (default: from BINANCE_TESTNET env var)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Binance API key (default: from BINANCE_API_KEY env var)')
    parser.add_argument('--api-secret', type=str, default=None,
                       help='Binance API secret (default: from BINANCE_API_SECRET env var)')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Trading pair symbol (default: from TRADING_SYMBOL env var or BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default=None,
                       help='Kline timeframe (default: from TRADING_TIMEFRAME env var or 1h)')
    
    args = parser.parse_args()
    
    # Load configuration - start with defaults from config class (includes hardcoded values)
    config = LiveTradingConfig()
    
    # Override with environment variables if they exist
    testnet_env = os.getenv('BINANCE_TESTNET', '').lower()
    if testnet_env:
        config.use_testnet = testnet_env in ('true', '1', 'yes')
    
    api_key_env = os.getenv('BINANCE_API_KEY', '')
    if api_key_env:
        config.api_key = api_key_env
    
    api_secret_env = os.getenv('BINANCE_API_SECRET', '')
    if api_secret_env:
        config.api_secret = api_secret_env
    
    symbol_env = os.getenv('TRADING_SYMBOL', '')
    if symbol_env:
        config.symbol = symbol_env
    
    timeframe_env = os.getenv('TRADING_TIMEFRAME', '')
    if timeframe_env:
        config.timeframe = timeframe_env
    
    # Override with command line arguments (highest priority)
    if args.testnet is not None:
        config.use_testnet = args.testnet
    elif args.realnet is not None:
        config.use_testnet = not args.realnet
    
    if args.api_key:
        config.api_key = args.api_key
    if args.api_secret:
        config.api_secret = args.api_secret
    if args.symbol:
        config.symbol = args.symbol
    if args.timeframe:
        config.timeframe = args.timeframe
    
    # Validate configuration
    if not config.api_key or not config.api_secret:
        print("Error: API key and secret must be provided")
        print("\nYou can provide them via:")
        print("  1. Hardcoded values in trader/live/config.py (api_key and api_secret fields)")
        print("  2. Environment variables:")
        print("     BINANCE_API_KEY - Binance API key")
        print("     BINANCE_API_SECRET - Binance API secret")
        print("  3. Command line arguments:")
        print("     --api-key and --api-secret")
        print("\nOptional environment variables:")
        print("  BINANCE_TESTNET - Set to 'true' for testnet, 'false' for realnet")
        print("  TRADING_SYMBOL - Trading pair (default: BTCUSDT)")
        print("  TRADING_TIMEFRAME - Kline timeframe (default: 1h)")
        sys.exit(1)
    
    # Display configuration
    print("=" * 80)
    print("LIVE TRADING BOT")
    print("=" * 80)
    print(f"Environment: {'TESTNET' if config.use_testnet else 'REALNET'}")
    print(f"Symbol: {config.symbol}")
    print(f"Timeframe: {config.timeframe}")
    print(f"API Key: {config.api_key[:10]}...")
    print("=" * 80)
    
    if not config.use_testnet:
        print("\nWARNING: You are about to trade on REALNET with real money!")
        print("Make sure you understand the risks and have tested on Testnet first.")
        response = input("Type 'YES' to continue: ")
        if response != 'YES':
            print("Aborted.")
            sys.exit(0)
    
    # Create and start live trader
    try:
        trader = LiveTrader(config)
        trader.start()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

