# -*- coding: utf-8 -*-
"""
Test script to verify buy/sell functionality on Binance Testnet.
"""

import sys
from trader.live.config import LiveTradingConfig
from trader.live.binance_client import BinanceClient
from trader.live.live_execution_engine import LiveExecutionEngine
from trader.live.order_manager import OrderManager
import time

def test_buy_sell():
    """Test buy and sell orders."""
    print("=" * 80)
    print("TESTING BUY/SELL FUNCTIONALITY")
    print("=" * 80)
    
    # Create config
    config = LiveTradingConfig()
    config.use_testnet = True
    config.symbol = "BTCUSDT"
    
    print(f"Symbol: {config.symbol}")
    print(f"Testnet: {config.use_testnet}")
    print()
    
    # Initialize client
    try:
        binance_client = BinanceClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            use_testnet=config.use_testnet
        )
        print("✓ Connected to Binance")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    # Get current price and minimum requirements
    try:
        current_price = binance_client.get_current_price(config.symbol)
        min_notional = binance_client.get_min_notional(config.symbol)
        min_quantity = binance_client.get_min_quantity(config.symbol)
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Minimum Notional: ${min_notional:.2f}")
        print(f"Minimum Quantity: {min_quantity:.8f} BTC")
        print()
    except Exception as e:
        print(f"✗ Error getting symbol info: {e}")
        return False
    
    # Calculate test order size (use minimum notional + small buffer)
    test_order_value = max(min_notional * 1.1, 11.0)  # At least $11
    test_quantity = test_order_value / current_price
    
    print(f"Test Order Value: ${test_order_value:.2f}")
    print(f"Test Quantity: {test_quantity:.8f} BTC")
    print()
    
    # Initialize order manager and execution engine
    order_manager = OrderManager(
        client=binance_client.client,
        symbol=config.symbol
    )
    
    execution_engine = LiveExecutionEngine(
        binance_client=binance_client,
        order_manager=order_manager,
        fee_pct=config.fee_pct
    )
    
    # Test BUY order
    print("-" * 80)
    print("TEST 1: BUY ORDER")
    print("-" * 80)
    try:
        buy_order = execution_engine.execute_entry_order(
            symbol=config.symbol,
            direction="long",
            quantity=test_quantity,
            trade_id=f"test_buy_{int(time.time())}",
            quote_order_qty=test_order_value
        )
        print(f"✓ BUY order placed: {buy_order['orderId']}")
        print(f"  Status: {buy_order.get('status')}")
        print(f"  Executed Qty: {buy_order.get('executedQty', 0)}")
        print(f"  Quote Qty: {buy_order.get('cumulativeQuoteQty', 0)}")
        
        # Wait for fill
        print("  Waiting for order to fill...")
        filled_order = execution_engine.wait_for_order_fill(
            buy_order['orderId'],
            timeout_seconds=30
        )
        
        if filled_order:
            print(f"✓ Order filled!")
            actual_qty = float(filled_order['raw_response'].get('executedQty', 0))
            actual_value = float(filled_order['raw_response'].get('cumulativeQuoteQty', 0))
            print(f"  Actual Quantity: {actual_qty:.8f} BTC")
            print(f"  Actual Value: ${actual_value:.2f}")
            
            # Test SELL order with the quantity we received
            print()
            print("-" * 80)
            print("TEST 2: SELL ORDER")
            print("-" * 80)
            
            if actual_qty > 0:
                sell_order = execution_engine.execute_exit_order(
                    symbol=config.symbol,
                    direction="long",
                    quantity=actual_qty,
                    trade_id=f"test_sell_{int(time.time())}"
                )
                print(f"✓ SELL order placed: {sell_order['orderId']}")
                print(f"  Status: {sell_order.get('status')}")
                
                # Wait for fill
                print("  Waiting for order to fill...")
                filled_sell = execution_engine.wait_for_order_fill(
                    sell_order['orderId'],
                    timeout_seconds=30
                )
                
                if filled_sell:
                    print(f"✓ Sell order filled!")
                    sell_qty = float(filled_sell['raw_response'].get('executedQty', 0))
                    sell_value = float(filled_sell['raw_response'].get('cumulativeQuoteQty', 0))
                    print(f"  Sold Quantity: {sell_qty:.8f} BTC")
                    print(f"  Sold Value: ${sell_value:.2f}")
                    print()
                    print("=" * 80)
                    print("✓ ALL TESTS PASSED!")
                    print("=" * 80)
                    return True
                else:
                    print("✗ Sell order did not fill within timeout")
                    return False
            else:
                print("✗ No quantity to sell")
                return False
        else:
            print("✗ Buy order did not fill within timeout")
            return False
            
    except Exception as e:
        print(f"✗ Error during buy/sell test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_buy_sell()
    sys.exit(0 if success else 1)


