#!/usr/bin/env python3
"""
Quick script to check available TopStepX accounts
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web_platform.backend.brokers.topstepx_client import TopStepXClient

async def main():
    try:
        client = TopStepXClient()
        print("📊 Getting account info...")
        accounts = await client.get_account_info()

        print("\n✅ Available accounts:")
        print("=" * 60)
        target_account_id = None
        for acc in accounts.get('accounts', []):
            status = "✅ ACTIVE" if acc['canTrade'] else "❌ INACTIVE"
            acc_type = "PRACTICE" if acc['simulated'] else "LIVE"

            # Check if this is the target account ending in 1794413
            if acc['name'].endswith('1794413'):
                print(f"🎯 TARGET: ID: {acc['id']:>10} | {acc['name']:<25} | ${acc['balance']:>8.0f} | {status} | {acc_type}")
                target_account_id = acc['id']
            else:
                print(f"ID: {acc['id']:>10} | {acc['name']:<25} | ${acc['balance']:>8.0f} | {status} | {acc_type}")

        if target_account_id:
            print(f"\n✅ Found account ending in 1794413: ID = {target_account_id}")
            print(f"To switch to this account, update TOPSTEPX_ACCOUNT_ID={target_account_id} in .env.topstepx")

        # Client cleanup handled automatically

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())