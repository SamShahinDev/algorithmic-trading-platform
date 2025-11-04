#!/usr/bin/env python3
"""
Find the correct NQ contract format for TopStepX
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from brokers.topstepx_client import topstepx_client

async def find_contracts():
    print("\n" + "="*60)
    print("🔍 FINDING ACTIVE NQ CONTRACTS")
    print("="*60)
    print(f"Time: {datetime.now()} (Market OPEN)")
    
    # Connect
    await topstepx_client.connect()
    
    if not topstepx_client.connected:
        print("❌ Failed to connect")
        return
    
    print("✅ Connected to TopStepX\n")
    
    # Try different search patterns
    search_terms = [
        "NQ",
        "MNQ",  # Micro NQ
        "NASDAQ",
        "E-mini",
        "ENQ",
        "",  # Empty search to get all
        "CON.F.US",  # Contract prefix
    ]
    
    for term in search_terms:
        print(f"🔎 Searching for: '{term}' (live=true)...")
        response = await topstepx_client.request('POST', '/api/Contract/search', {
            "searchText": term,
            "live": True
        })
        
        if response and response.get('success'):
            contracts = response.get('contracts', [])
            if contracts:
                print(f"   ✅ Found {len(contracts)} contract(s):")
                for contract in contracts[:3]:
                    print(f"      - Name: {contract.get('name')}")
                    print(f"        ID: {contract.get('id')}")
                    print(f"        Symbol: {contract.get('symbol', 'N/A')}")
            else:
                print(f"   ⚠️ No contracts found")
        else:
            if response:
                print(f"   ❌ Error: {response.get('errorMessage', response)}")
            else:
                print("   ❌ No response")
        print()
    
    # Try with live=false
    print("🔎 Searching with live=false (all contracts)...")
    response = await topstepx_client.request('POST', '/api/Contract/search', {
        "searchText": "NQ",
        "live": False
    })
    
    if response and response.get('success'):
        contracts = response.get('contracts', [])
        print(f"   Found {len(contracts)} contract(s) total")
        if contracts:
            print("   First few contracts:")
            for contract in contracts[:5]:
                print(f"      - {contract.get('name')} | {contract.get('id')}")
    
    # Try to get specific contract by ID
    print("\n🔎 Testing specific contract IDs...")
    test_ids = [
        "CON.F.US.ENQ.U25",
        "CON.F.US.ENQ.Z24",  # December 2024
        "CON.F.US.ENQ.H25",  # March 2025
        "CON.F.US.MNQ.Z24",  # Micro NQ
    ]
    
    for cid in test_ids:
        response = await topstepx_client.request('GET', f'/api/Contract/{cid}', {})
        if response and response.get('success'):
            print(f"   ✅ Found: {cid}")
        else:
            print(f"   ❌ Not found: {cid}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(find_contracts())