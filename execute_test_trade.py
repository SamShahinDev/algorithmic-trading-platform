#!/usr/bin/env python3
"""
Execute a test trade with the NQU5 contract
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from brokers.topstepx_client import topstepx_client

async def execute_trade():
    print("\n" + "="*60)
    print("🚀 EXECUTING TEST TRADE")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print("Contract: NQU5 (CON.F.US.ENQ.U25)")
    print("Account: Practice (10983875)")
    print("="*60)
    
    # Connect
    await topstepx_client.connect()
    
    if not topstepx_client.connected:
        print("❌ Failed to connect")
        return
    
    print("✅ Connected to TopStepX")
    
    # Get current price (try market data)
    print("\n📊 Getting market data...")
    response = await topstepx_client.request('GET', '/api/MarketData/quote/CON.F.US.ENQ.U25', {})
    
    if response and response.get('success'):
        quote = response.get('quote', {})
        print(f"   Bid: {quote.get('bid')}")
        print(f"   Ask: {quote.get('ask')}")
        print(f"   Last: {quote.get('last')}")
    else:
        print("   ⚠️ No market data available")
        print("   Using approximate price: 20000")
    
    # Build order
    print("\n📝 Building order...")
    order = {
        "accountId": 10983875,  # Practice account
        "contractId": "CON.F.US.ENQ.U25",  # NQU5
        "action": "Buy",
        "orderType": "Market",
        "quantity": 1,
        "brackets": {
            "stopLoss": {
                "orderType": "Stop",
                "stopPrice": 19950.0  # 50 points stop
            },
            "takeProfit": {
                "orderType": "Limit",
                "limitPrice": 20050.0  # 50 points profit
            }
        }
    }
    
    print(f"   Direction: BUY")
    print(f"   Quantity: 1 contract")
    print(f"   Type: Market Order")
    print(f"   Stop Loss: 19950 (50 points)")
    print(f"   Take Profit: 20050 (50 points)")
    
    # Submit order
    print("\n🚀 Submitting order...")
    response = await topstepx_client.request('POST', '/api/Order/submit', order)
    
    if response and response.get('success'):
        order_id = response.get('orderId')
        print(f"   ✅ ORDER SUBMITTED!")
        print(f"   Order ID: {order_id}")
        
        # Check order status
        print("\n📋 Checking order status...")
        status_response = await topstepx_client.request('GET', f'/api/Order/{order_id}', {})
        
        if status_response and status_response.get('success'):
            order_info = status_response.get('order', {})
            print(f"   Status: {order_info.get('status')}")
            print(f"   Filled: {order_info.get('filledQuantity', 0)}/{order_info.get('quantity', 1)}")
    else:
        error = response.get('errorMessage', response) if response else 'No response'
        print(f"   ❌ Order failed: {error}")
    
    # Check positions
    print("\n📊 Checking positions...")
    response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
        "accountId": 10983875
    })
    
    if response and response.get('success'):
        positions = response.get('positions', [])
        print(f"   Open positions: {len(positions)}")
        for pos in positions:
            print(f"      - {pos.get('contractId')} | {pos.get('size')} contracts")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n⚠️ WARNING: This will attempt a REAL trade on practice account")
    response = input("Continue? (yes/no): ")
    if response.lower() == 'yes':
        asyncio.run(execute_trade())
    else:
        print("Cancelled")