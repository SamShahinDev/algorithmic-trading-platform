#!/usr/bin/env python3
"""
Test trade execution with fixed API endpoints
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from brokers.topstepx_client import topstepx_client

async def test_trade():
    print("\n" + "="*60)
    print("🚀 TESTING FIXED TRADE EXECUTION")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print("Account: 10983875 (Practice)")
    print("="*60)
    
    # Step 1: Connect
    print("\n📡 Step 1: Connecting to TopStepX...")
    await topstepx_client.connect()
    
    if not topstepx_client.connected:
        print("❌ Failed to connect")
        return
    
    print("✅ Connected successfully")
    print(f"   Session token: {topstepx_client.session_token[:20]}...")
    
    # Step 2: Get available contracts
    print("\n📊 Step 2: Getting available contracts...")
    contracts = await topstepx_client.get_available_contracts(live=True)
    
    if contracts:
        print(f"✅ Found {len(contracts)} available contracts")
        # Look for NQ
        nq_contract = None
        for contract in contracts:
            if "NQ" in contract.get("name", "") or "ENQ" in contract.get("id", ""):
                nq_contract = contract
                print(f"   Found NQ: {contract.get('name')} ({contract.get('id')})")
                break
        
        if not nq_contract:
            print("   Using default NQ contract")
            nq_contract = {"id": "CON.F.US.ENQ.U25", "name": "NQU5"}
    else:
        print("⚠️ No available contracts found, using default")
        nq_contract = {"id": "CON.F.US.ENQ.U25", "name": "NQU5"}
    
    contract_id = nq_contract.get("id")
    print(f"\n📝 Using contract: {contract_id}")
    
    # Step 3: Check current positions
    print("\n📋 Step 3: Checking current positions...")
    response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
        "accountId": 10983875
    })
    
    if response and response.get('success'):
        positions = response.get('positions', [])
        print(f"   Current open positions: {len(positions)}")
        for pos in positions:
            print(f"      - {pos.get('contractId')} | Size: {pos.get('size')}")
    
    # Step 4: Place test order
    print("\n🎯 Step 4: Placing test market order...")
    print(f"   Contract: {contract_id}")
    print("   Type: Market (2)")
    print("   Side: Buy (0)")
    print("   Size: 1 contract")
    
    result = await topstepx_client.submit_order(
        account_id=10983875,
        contract_id=contract_id,
        order_type=2,  # Market
        side=0,  # Buy
        size=1   # 1 contract
    )
    
    if result.get("success"):
        order_id = result.get("orderId")
        print(f"\n✅ SUCCESS! Order placed!")
        print(f"   Order ID: {order_id}")
        
        # Check order status
        print("\n📊 Checking order status...")
        await asyncio.sleep(2)  # Wait for order to process
        
        # Check positions again
        response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
            "accountId": 10983875
        })
        
        if response and response.get('success'):
            positions = response.get('positions', [])
            print(f"   Open positions after trade: {len(positions)}")
            for pos in positions:
                print(f"      - {pos.get('contractId')} | Size: {pos.get('size')}")
    else:
        print(f"\n❌ Order failed: {result.get('error')}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n⚠️ WARNING: This will place a REAL trade on practice account")
    response = input("Continue? (yes/no): ")
    if response.lower() == 'yes':
        asyncio.run(test_trade())
    else:
        print("Cancelled")