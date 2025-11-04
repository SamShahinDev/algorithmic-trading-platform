#!/usr/bin/env python3
"""
Refresh and list all TopStepX accounts
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web_platform.backend.brokers.topstepx_client import TopStepXClient

async def main():
    try:
        client = TopStepXClient()
        print("🔄 Refreshing account list...")
        print(f"⏰ Current time: {datetime.now()}")

        # Connect first
        connected = await client.connect()
        if not connected:
            print("❌ Failed to connect to TopStepX")
            return

        # Get fresh account list
        accounts = await client.get_account_info()

        print("\n✅ All Available Accounts:")
        print("=" * 80)

        new_50k_found = False
        for acc in accounts.get('accounts', []):
            status = "✅ ACTIVE" if acc['canTrade'] else "❌ INACTIVE"
            acc_type = "PRACTICE" if acc['simulated'] else "LIVE"
            balance = acc['balance']

            # Check for new 50K account indicators
            is_50k = "50K" in acc['name'] or "50KTC" in acc['name']
            is_new = balance >= 49900 and balance <= 50100  # Close to $50K

            # Highlight potential new account
            if is_50k and acc['canTrade'] and not acc['name'].endswith('13140370') and not acc['name'].endswith('56603374'):
                print(f"🎯 NEW 50K: ID: {acc['id']:>10} | {acc['name']:<30} | ${balance:>10,.2f} | {status} | {acc_type}")
                new_50k_found = True
            else:
                prefix = "   " if not (is_50k and is_new) else "⭐ "
                print(f"{prefix}ID: {acc['id']:>10} | {acc['name']:<30} | ${balance:>10,.2f} | {status} | {acc_type}")

        print("=" * 80)
        print(f"Total accounts found: {len(accounts.get('accounts', []))}")

        if not new_50k_found:
            print("\n⚠️  No new eligible 50K account found yet.")
            print("   The account may need a few moments to appear after creation.")
            print("   Try running this script again in 30 seconds.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())