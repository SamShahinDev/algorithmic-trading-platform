#!/usr/bin/env python3
"""Test script to debug market data retrieval"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from execution.topstep_client import TopStepXClient

async def main():
    # Load environment
    load_dotenv()

    api_key = os.getenv('TOPSTEPX_API_KEY')
    username = os.getenv('TOPSTEPX_USERNAME')
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')

    print(f"Testing TopStepX Market Data")
    print(f"Username: {username}")
    print(f"Account: {account_id}")
    print(f"Environment: DEMO")
    print("=" * 60)

    # Initialize client
    client = TopStepXClient(
        api_key=api_key,
        username=username,
        account_id=account_id,
        environment='DEMO'
    )

    # Connect
    print("\n1. Authenticating...")
    if not await client.connect():
        print("❌ Authentication failed")
        return
    print("✅ Authenticated")

    # Search for NQ contracts
    print("\n2. Searching for NQ contracts (sim data)...")
    contracts = await client.search_contracts("NQ", live=False)
    print(f"Found {len(contracts)} contracts:")
    for c in contracts[:5]:  # Show first 5
        print(f"  - {c.get('id')} | {c.get('description')} | Active: {c.get('activeContract')}")

    if not contracts:
        print("No contracts found!")
        return

    # Get available contracts
    print("\n3. Getting available contracts...")
    result = await client._request('POST', '/Contract/available', {'live': False})
    if result.get('success'):
        available = result.get('contracts', [])
        print(f"Found {len(available)} available contracts:")
        nq_contracts = [c for c in available if 'NQ' in c.get('description', '')]
        for c in nq_contracts[:5]:
            print(f"  - {c.get('id')} | {c.get('description')}")

        # Use the first available NQ contract
        if nq_contracts:
            test_contract = nq_contracts[0]['id']
            print(f"\n4. Testing retrieve_bars with contract: {test_contract}")

            # Test with UTC timestamps
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)

            print(f"   Start: {start_time.isoformat()}")
            print(f"   End: {end_time.isoformat()}")

            bars = await client.retrieve_bars(
                contract_id=test_contract,
                start_time=start_time,
                end_time=end_time,
                unit=2,  # Minute
                unit_number=1,
                limit=100,
                live=False
            )

            if bars and len(bars) > 0:
                print(f"✅ Retrieved {len(bars)} bars")
                print(f"   First bar: {bars[0]}")
                print(f"   Last bar: {bars[-1]}")
            else:
                print(f"❌ No bars returned")

    # Test with original contract
    print(f"\n5. Testing CON.F.US.ENQ.U25...")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)

    bars = await client.retrieve_bars(
        contract_id="CON.F.US.ENQ.U25",
        start_time=start_time,
        end_time=end_time,
        unit=2,
        unit_number=1,
        limit=100,
        live=False
    )

    if bars and len(bars) > 0:
        print(f"✅ Retrieved {len(bars)} bars for CON.F.US.ENQ.U25")
    else:
        print(f"❌ No bars for CON.F.US.ENQ.U25")

    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
