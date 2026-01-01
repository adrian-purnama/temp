# -*- coding: utf-8 -*-
"""
Simple script to test Binance Testnet API connection.

Run this to verify your API keys work correctly.
"""

import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Get API keys from config
try:
    from trader.live.config import LiveTradingConfig
    config = LiveTradingConfig()
    api_key = config.api_key.strip()
    api_secret = config.api_secret.strip()
except Exception as e:
    print(f"Error loading config: {e}")
    print("\nPlease set your API keys manually:")
    api_key = input("API Key: ").strip()
    api_secret = input("API Secret: ").strip()

print("=" * 80)
print("BINANCE TESTNET API TEST")
print("=" * 80)
print(f"API Key: {api_key[:10]}...")
print(f"Using Testnet: True")
print("=" * 80)

try:
    # Initialize client
    print("\n1. Initializing client...")
    client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
    print("   ✓ Client initialized")
    
    # Test public endpoint
    print("\n2. Testing public endpoint (get_server_time)...")
    server_time = client.get_server_time()
    print(f"   ✓ Server time: {server_time['serverTime']}")
    print("   ✓ Connected to testnet.binance.vision")
    
    # Test private endpoint (requires signature)
    print("\n3. Testing private endpoint (get_account) - requires valid signature...")
    try:
        account = client.get_account()
        print("   ✓ Account access successful!")
        print(f"   ✓ API key authentication works")
        
        # Show balances
        print("\n4. Account balances:")
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                print(f"   {balance['asset']}: Free={free}, Locked={locked}")
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - Your API keys are working correctly!")
        print("=" * 80)
        
    except BinanceAPIException as e:
        if e.code == -1022:
            print("   ✗ SIGNATURE ERROR - API key authentication failed")
            print("\n   Possible causes:")
            print("   1. API key/secret mismatch")
            print("   2. IP whitelist enabled - your IP is not whitelisted")
            print("   3. API key doesn't have 'Enable Reading' permission")
            print("   4. Using mainnet keys instead of testnet keys")
            print("\n   Check your API key settings at: https://testnet.binance.vision/")
            print("   Make sure:")
            print("   - 'Enable Reading' is checked")
            print("   - IP whitelist is disabled OR your IP is whitelisted")
            sys.exit(1)
        else:
            print(f"   ✗ Error: {e}")
            sys.exit(1)
            
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)



