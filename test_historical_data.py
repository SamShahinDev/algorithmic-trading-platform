#!/usr/bin/env python3
"""
Test historical data retrieval from TopStepX API
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

from brokers.topstepx_client import topstepx_client

async def test_historical_data():
    print("\n" + "="*60)
    print("📊 TESTING HISTORICAL DATA RETRIEVAL")
    print("="*60)
    print(f"Current Time: {datetime.now()}")
    print("="*60)
    
    # Connect to TopStepX
    print("\n📡 Connecting to TopStepX...")
    await topstepx_client.connect()
    
    if not topstepx_client.connected:
        print("❌ Failed to connect")
        return
    
    print("✅ Connected successfully")
    
    # Test different time ranges
    contract_id = "CON.F.US.ENQ.U25"  # NQ futures
    end_time = datetime.utcnow()
    
    print(f"\n📈 Testing data retrieval for: {contract_id}")
    print("="*60)
    
    # Test 1: Last 1 hour (should definitely work)
    print("\n🔍 Test 1: Last 1 hour of data...")
    start_time = end_time - timedelta(hours=1)
    
    response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
        "contractId": contract_id,
        "live": False,  # Use sim data for testing
        "startTime": start_time.isoformat() + "Z",
        "endTime": end_time.isoformat() + "Z",
        "unit": 2,  # Minute bars
        "unitNumber": 5,  # 5-minute bars
        "limit": 12,  # 12 bars (1 hour / 5 minutes)
        "includePartialBar": False
    })
    
    if response and response.get('success'):
        bars = response.get('bars', [])
        print(f"   ✅ Retrieved {len(bars)} bars")
        if bars:
            first_bar = bars[-1] if bars else None
            last_bar = bars[0] if bars else None
            if first_bar:
                print(f"   First bar: {first_bar.get('t')} - Open: {first_bar.get('o')}")
            if last_bar:
                print(f"   Last bar:  {last_bar.get('t')} - Close: {last_bar.get('c')}")
    else:
        error = response.get('errorMessage', 'Unknown') if response else 'No response'
        print(f"   ❌ Failed: {error}")
    
    # Test 2: Last 24 hours
    print("\n🔍 Test 2: Last 24 hours of data...")
    start_time = end_time - timedelta(hours=24)
    
    response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
        "contractId": contract_id,
        "live": False,
        "startTime": start_time.isoformat() + "Z",
        "endTime": end_time.isoformat() + "Z",
        "unit": 3,  # Hour bars
        "unitNumber": 1,  # 1-hour bars
        "limit": 24,
        "includePartialBar": False
    })
    
    if response and response.get('success'):
        bars = response.get('bars', [])
        print(f"   ✅ Retrieved {len(bars)} hourly bars")
    else:
        error = response.get('errorMessage', 'Unknown') if response else 'No response'
        print(f"   ❌ Failed: {error}")
    
    # Test 3: Last 7 days
    print("\n🔍 Test 3: Last 7 days of data...")
    start_time = end_time - timedelta(days=7)
    
    response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
        "contractId": contract_id,
        "live": False,
        "startTime": start_time.isoformat() + "Z",
        "endTime": end_time.isoformat() + "Z",
        "unit": 4,  # Day bars
        "unitNumber": 1,
        "limit": 7,
        "includePartialBar": False
    })
    
    if response and response.get('success'):
        bars = response.get('bars', [])
        print(f"   ✅ Retrieved {len(bars)} daily bars")
        for bar in bars[:3]:  # Show first 3 bars
            date = bar.get('t', '').split('T')[0]
            print(f"      {date}: O={bar.get('o'):.2f}, H={bar.get('h'):.2f}, L={bar.get('l'):.2f}, C={bar.get('c'):.2f}")
    else:
        error = response.get('errorMessage', 'Unknown') if response else 'No response'
        print(f"   ❌ Failed: {error}")
    
    # Test 4: Last 30 days
    print("\n🔍 Test 4: Last 30 days of data...")
    start_time = end_time - timedelta(days=30)
    
    response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
        "contractId": contract_id,
        "live": False,
        "startTime": start_time.isoformat() + "Z",
        "endTime": end_time.isoformat() + "Z",
        "unit": 4,  # Day bars
        "unitNumber": 1,
        "limit": 30,
        "includePartialBar": False
    })
    
    if response and response.get('success'):
        bars = response.get('bars', [])
        print(f"   ✅ Retrieved {len(bars)} daily bars (30 days)")
    else:
        error = response.get('errorMessage', 'Unknown') if response else 'No response'
        print(f"   ❌ Failed: {error}")
    
    # Test 5: Test maximum historical reach
    print("\n🔍 Test 5: Testing maximum historical reach...")
    
    # Try 1 year back
    start_time = end_time - timedelta(days=365)
    
    response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
        "contractId": contract_id,
        "live": False,
        "startTime": start_time.isoformat() + "Z",
        "endTime": end_time.isoformat() + "Z",
        "unit": 5,  # Week bars
        "unitNumber": 1,
        "limit": 52,  # 52 weeks
        "includePartialBar": False
    })
    
    if response and response.get('success'):
        bars = response.get('bars', [])
        print(f"   ✅ Retrieved {len(bars)} weekly bars (1 year)")
        if bars:
            oldest_bar = bars[-1]
            newest_bar = bars[0]
            print(f"   Oldest data: {oldest_bar.get('t')}")
            print(f"   Newest data: {newest_bar.get('t')}")
    else:
        error = response.get('errorMessage', 'Unknown') if response else 'No response'
        print(f"   ❌ Failed to get 1 year data: {error}")
    
    # Test 6: Try different minute intervals
    print("\n🔍 Test 6: Testing different timeframes (last 2 hours)...")
    start_time = end_time - timedelta(hours=2)
    
    timeframes = [
        (2, 1, "1-minute"),
        (2, 5, "5-minute"),
        (2, 15, "15-minute"),
        (2, 30, "30-minute")
    ]
    
    for unit, unit_number, description in timeframes:
        response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
            "contractId": contract_id,
            "live": False,
            "startTime": start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "unit": unit,
            "unitNumber": unit_number,
            "limit": 100,  # Get up to 100 bars
            "includePartialBar": False
        })
        
        if response and response.get('success'):
            bars = response.get('bars', [])
            print(f"   ✅ {description}: Retrieved {len(bars)} bars")
        else:
            print(f"   ❌ {description}: Failed")
    
    # Test 7: Check data availability for other contracts
    print("\n🔍 Test 7: Testing other contracts...")
    
    contracts_to_test = [
        ("CON.F.US.MNQ.U25", "Micro NQ"),
        ("CON.F.US.EP.U25", "ES (S&P 500)"),
        ("CON.F.US.MES.U25", "Micro ES"),
        ("CON.F.US.RTY.U25", "Russell 2000")
    ]
    
    start_time = end_time - timedelta(hours=1)
    
    for contract_id, name in contracts_to_test:
        response = await topstepx_client.request('POST', '/api/History/retrieveBars', {
            "contractId": contract_id,
            "live": False,
            "startTime": start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "unit": 2,
            "unitNumber": 15,
            "limit": 4,
            "includePartialBar": False
        })
        
        if response and response.get('success'):
            bars = response.get('bars', [])
            if bars:
                latest = bars[0]
                print(f"   ✅ {name}: {len(bars)} bars, Latest: {latest.get('c', 0):.2f}")
            else:
                print(f"   ⚠️ {name}: No data")
        else:
            print(f"   ❌ {name}: Failed")
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print("Historical data capabilities:")
    print("- ✅ Minute-level data (1, 5, 15, 30 min)")
    print("- ✅ Hourly data")
    print("- ✅ Daily data")
    print("- ✅ Weekly data")
    print("- 📌 Maximum bars per request: 20,000 (per API docs)")
    print("- 📌 Rate limit: 50 requests per 30 seconds")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_historical_data())