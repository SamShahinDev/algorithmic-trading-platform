#!/usr/bin/env python3
"""
Test TopStepX authentication with real credentials
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from brokers.topstepx_client import topstepx_client
import logging

logging.basicConfig(level=logging.INFO)

async def test_auth():
    print("\n" + "="*60)
    print("🔐 TESTING TOPSTEPX AUTHENTICATION")
    print("="*60)
    
    print(f"\nCredentials from .env.topstepx:")
    print(f"  Username: {topstepx_client.username}")
    print(f"  API Key: {topstepx_client.api_key[:20]}...")
    print(f"  Base URL: {topstepx_client.base_url}")
    
    print("\n📡 Attempting to connect...")
    result = await topstepx_client.connect()
    
    if result:
        print("✅ Authentication successful!")
        print(f"   Session token: {topstepx_client.session_token[:20]}..." if topstepx_client.session_token else "   No token received")
        print(f"   Account ID: {topstepx_client.account_id}")
        
        # Try to get accounts
        print("\n📋 Fetching account list...")
        response = await topstepx_client.request('POST', '/api/Account/search', {
            "onlyActiveAccounts": True
        })
        
        if response and response.get('success'):
            accounts = response.get('accounts', [])
            print(f"   Found {len(accounts)} account(s)")
            for acc in accounts[:3]:
                print(f"      - {acc.get('name')} (ID: {acc.get('id')})")
        else:
            print(f"   Failed: {response}")
            
        # Try to get NQ contract
        print("\n📊 Looking for NQ contract...")
        response = await topstepx_client.request('POST', '/api/Contract/search', {
            "searchText": "NQ",
            "live": True
        })
        
        if response and response.get('success'):
            contracts = response.get('contracts', [])
            print(f"   Found {len(contracts)} contract(s)")
            for contract in contracts[:3]:
                print(f"      - {contract.get('name')} ({contract.get('id')})")
        else:
            print(f"   Failed: {response}")
            
    else:
        print("❌ Authentication failed!")
        print("   Check credentials in .env.topstepx")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_auth())