#!/usr/bin/env python3
"""
Simple test trade - places a small order to verify execution
"""

import sys
import os
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

import asyncio
from brokers.topstepx_client import topstepx_client

async def simple_test():
    """Place a simple test order"""
    
    print("\n" + "="*60)
    print("🧪 SIMPLE TEST TRADE")
    print("="*60)
    
    try:
        # Connect
        print("Connecting to TopStepX...")
        await topstepx_client.connect()
        
        if not topstepx_client.connected:
            print("❌ Failed to connect")
            return False
            
        print(f"✅ Connected as: {topstepx_client.username}")
        print(f"✅ Account ID: {topstepx_client.account_id}")
        
        # Get contract ID for NQ
        print("\nGetting NQ contract...")
        contract_id = await topstepx_client._get_contract_id("NQ")
        
        if not contract_id:
            print("❌ Could not find NQ contract")
            return False
            
        print(f"✅ Contract ID: {contract_id}")
        
        # Place a simple market order
        print("\nPlacing test order...")
        print("• Account ID: 10983875")
        print("• Contract: NQ")
        print("• Side: BUY (0)")
        print("• Quantity: 1")
        print("• Type: MARKET (2)")
        
        result = await topstepx_client.submit_order(
            account_id=10983875,
            contract_id=contract_id,
            order_type=2,  # Market
            side=0,  # Buy
            size=1
        )
        
        if result:
            print(f"\n✅ ORDER PLACED!")
            print(f"Result: {result}")
            
            # Wait 5 seconds then close
            print("\nWaiting 5 seconds before closing...")
            await asyncio.sleep(5)
            
            print("Closing position...")
            close_result = await topstepx_client.submit_order(
                account_id=10983875,
                contract_id=contract_id,
                order_type=2,  # Market
                side=1,  # Sell
                size=1
            )
            
            if close_result:
                print(f"✅ POSITION CLOSED!")
                print(f"Result: {close_result}")
            else:
                print("⚠️ Failed to close - manual intervention may be needed")
        else:
            print(f"❌ Order failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    sys.exit(0 if success else 1)