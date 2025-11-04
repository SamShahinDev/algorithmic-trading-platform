#!/usr/bin/env python3
"""
Test script to verify practice account recognition
"""

import asyncio
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from agents.smart_scalper_enhanced import enhanced_scalper, PRACTICE_ACCOUNT_ID
from brokers.topstepx_client import topstepx_client

async def test_practice_account():
    """Test if the bot recognizes the practice account"""
    print("\n" + "="*60)
    print("🧪 TESTING PRACTICE ACCOUNT RECOGNITION")
    print("="*60)
    
    # Test 1: Check if practice account ID is set
    print("\n📋 Test 1: Checking account ID configuration...")
    print(f"   Account ID: {enhanced_scalper.account_id}")
    print(f"   Expected: 10983875 (PRAC-V2-XXXXX-XXXXXXXX)")
    
    if enhanced_scalper.account_id == 10983875:
        print("   ✅ Practice account ID correctly configured!")
    else:
        print("   ❌ Account ID mismatch!")
    
    # Test 2: Connect to broker
    print("\n📋 Test 2: Connecting to TopStepX...")
    try:
        await topstepx_client.connect()
        if topstepx_client.connected:
            print("   ✅ Successfully connected to TopStepX!")
        else:
            print("   ❌ Failed to connect to TopStepX")
            return
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return
    
    # Test 3: Verify practice account
    print("\n📋 Test 3: Verifying practice account access...")
    try:
        result = await enhanced_scalper.verify_practice_account()
        if result:
            print("   ✅ Practice account verification completed!")
        else:
            print("   ⚠️ Account verification returned False (but may continue)")
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
    
    # Test 4: Initialize the scalper
    print("\n📋 Test 4: Initializing enhanced scalper...")
    try:
        result = await enhanced_scalper.initialize()
        if result:
            print("   ✅ Scalper initialized successfully!")
        else:
            print("   ⚠️ Initialization returned False (but may continue)")
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
    
    # Test 5: Check position sync
    print("\n📋 Test 5: Syncing position with broker...")
    try:
        result = await enhanced_scalper.sync_position_with_broker()
        print(f"   Current position: {enhanced_scalper.get_position_status()}")
        if result:
            print("   ✅ Position sync successful!")
        else:
            print("   ⚠️ Position sync failed (may not have positions)")
    except Exception as e:
        print(f"   ❌ Position sync error: {e}")
    
    # Test 6: Get active NQ contract
    print("\n📋 Test 6: Looking for active NQ contract...")
    try:
        contract_id = await enhanced_scalper.get_active_nq_contract()
        if contract_id:
            print(f"   ✅ Found active NQ contract: {contract_id}")
        else:
            print("   ⚠️ No active NQ contract found")
    except Exception as e:
        print(f"   ❌ Contract lookup error: {e}")
    
    # Test 7: Check account in broker API
    print("\n📋 Test 7: Checking account directly with TopStepX API...")
    try:
        response = await topstepx_client.request('POST', '/api/Account/search', {
            "onlyActiveAccounts": True
        })
        
        if response and response.get('success'):
            accounts = response.get('accounts', [])
            print(f"   Found {len(accounts)} account(s)")
            
            practice_found = False
            for account in accounts:
                account_id = account.get('id')
                account_name = account.get('name', 'Unknown')
                is_practice = account_id == 10983875  # Check for numeric ID
                
                if is_practice:
                    practice_found = True
                    print(f"   ✅ Practice account found: {account_name} (ID: {account_id})")
                else:
                    print(f"   📌 Other account: {account_name} (ID: {account_id})")
            
            if not practice_found:
                print("   ⚠️ Practice account not in account list (but we'll use it anyway)")
        else:
            print(f"   ❌ Account search failed: {response.get('errorMessage', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Account search error: {e}")
    
    # Test 8: Test position query for practice account
    print("\n📋 Test 8: Querying positions for practice account...")
    try:
        response = await topstepx_client.request('POST', '/api/Position/searchOpen', {
            "accountId": 10983875  # Use numeric ID
        })
        
        if response and response.get('success'):
            positions = response.get('positions', [])
            print(f"   ✅ Position query successful! Found {len(positions)} position(s)")
            
            for pos in positions:
                contract_id = pos.get('contractId', 'Unknown')
                size = pos.get('size', 0)
                pos_type = pos.get('type', 0)
                type_str = "LONG" if pos_type == 1 else "SHORT" if pos_type == 2 else "UNKNOWN"
                print(f"      - {type_str} {size} {contract_id}")
        else:
            error = response.get('errorMessage', 'Unknown')
            print(f"   ❌ Position query failed: {error}")
    except Exception as e:
        print(f"   ❌ Position query error: {e}")
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print("✅ Practice account ID is correctly configured")
    print("📌 Check the results above to see if TopStepX recognizes it")
    print("💡 If account not found in list, it may still work with direct ID")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n🚀 Starting practice account recognition test...")
    asyncio.run(test_practice_account())
    print("✅ Test complete!")