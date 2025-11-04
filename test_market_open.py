#!/usr/bin/env python3
"""
Test if market is open and find active NQ contracts
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from brokers.topstepx_client import topstepx_client
import logging

logging.basicConfig(level=logging.INFO)

async def test_market():
    print("\n" + "="*60)
    print("🕐 MARKET STATUS CHECK")
    print("="*60)
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Tuesday)")
    print("NQ Futures Hours: Sunday 6PM - Friday 5PM ET")
    print("="*60)
    
    # Connect first
    print("\n📡 Connecting to TopStepX...")
    await topstepx_client.connect()
    
    if not topstepx_client.connected:
        print("❌ Failed to connect")
        return
    
    print("✅ Connected successfully")
    
    # Try different search methods for NQ
    print("\n🔍 METHOD 1: Search with 'NQ'...")
    response = await topstepx_client.request('POST', '/api/Contract/search', {
        "searchText": "NQ",
        "live": True
    })
    
    if response and response.get('success'):
        contracts = response.get('contracts', [])
        print(f"   Found {len(contracts)} contract(s)")
        for contract in contracts[:5]:
            print(f"      - {contract.get('name', 'Unknown')} | ID: {contract.get('id')}")
    else:
        print(f"   Response: {response}")
    
    # Try without 'live' filter
    print("\n🔍 METHOD 2: Search without 'live' filter...")
    response = await topstepx_client.request('POST', '/api/Contract/search', {
        "searchText": "NQ"
    })
    
    if response and response.get('success'):
        contracts = response.get('contracts', [])
        print(f"   Found {len(contracts)} contract(s)")
        for contract in contracts[:5]:
            print(f"      - {contract.get('name', 'Unknown')} | ID: {contract.get('id')}")
    else:
        print(f"   Response: {response}")
        
    # Try empty search to see all contracts
    print("\n🔍 METHOD 3: Get all contracts (empty search)...")
    response = await topstepx_client.request('POST', '/api/Contract/search', {})
    
    if response and response.get('success'):
        contracts = response.get('contracts', [])
        print(f"   Found {len(contracts)} total contract(s)")
        # Filter for NQ-related
        nq_contracts = [c for c in contracts if 'NQ' in c.get('name', '') or 'NQ' in c.get('id', '')]
        print(f"   NQ-related: {len(nq_contracts)}")
        for contract in nq_contracts[:5]:
            print(f"      - {contract.get('name', 'Unknown')} | ID: {contract.get('id')}")
    else:
        print(f"   Response: {response}")
    
    # Try specific contract IDs from the code
    print("\n🔍 METHOD 4: Try known contract IDs...")
    known_ids = ["CON.F.US.ENQ.U25", "NQH25", "NQM25", "NQU25", "NQZ25"]
    for contract_id in known_ids:
        response = await topstepx_client.request('POST', '/api/Contract/search', {
            "searchText": contract_id
        })
        if response and response.get('success'):
            contracts = response.get('contracts', [])
            if contracts:
                print(f"   ✅ Found: {contract_id} - {len(contracts)} match(es)")
                for c in contracts:
                    print(f"      {c.get('name')} | {c.get('id')}")
            else:
                print(f"   ❌ Not found: {contract_id}")
                
    # Check market data endpoint
    print("\n📊 METHOD 5: Check market data availability...")
    response = await topstepx_client.request('GET', '/api/MarketData/quotes', {
        "symbols": ["NQ", "ES", "YM", "RTY"]
    })
    
    if response:
        print(f"   Market data response: {response}")
    
    print("\n" + "="*60)
    print("📌 CONCLUSION")
    print("="*60)
    
    if topstepx_client.connected:
        print("✅ Authentication working")
        print("📅 Market should be OPEN (Tuesday 1:50 AM)")
        print("🔍 Check contract search results above")
    else:
        print("❌ Connection failed")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_market())