#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from web_platform.backend.brokers.topstepx_client import TopStepXClient

# Load environment
load_dotenv('.env.topstepx')

async def main():
    # Initialize client
    client = TopStepXClient()

    # Connect
    if await client.connect():
        print("Connected to TopStepX")

        # Get accounts
        accounts = await client.get_accounts()
        if accounts:
            print("\nAvailable accounts:")
            for acc in accounts:
                account_name = acc.get('name', '')
                account_id = acc.get('id', '')
                balance = acc.get('balance', 0)
                can_trade = acc.get('canTrade', False)
                simulated = acc.get('simulated', False)

                # Check if this account name ends with the target number
                if account_name.endswith('1794413'):
                    print(f"\n✅ FOUND TARGET ACCOUNT:")
                    print(f"  ID: {account_id}")
                    print(f"  Name: {account_name}")
                    print(f"  Balance: ${balance:,.2f}")
                    print(f"  Can Trade: {can_trade}")
                    print(f"  Type: {'Practice' if simulated else 'Live'}")
                    print(f"\n  To switch to this account, update TOPSTEPX_ACCOUNT_ID={account_id}")
                else:
                    print(f"  {account_name}: ID={account_id}, Balance=${balance:,.2f}, CanTrade={can_trade}")
        else:
            print("No accounts found")
    else:
        print("Failed to connect")

if __name__ == "__main__":
    asyncio.run(main())