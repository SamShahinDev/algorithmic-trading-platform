#!/usr/bin/env python3
"""Get the account ID for the combine account"""

import asyncio
import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')
from brokers.topstepx_client import topstepx_client

async def get_accounts():
    print("Fetching all TopStep accounts...")
    account_info = await topstepx_client.get_account_info()
    
    if account_info:
        print("\n=== ALL TOPSTEP ACCOUNTS ===")
        # The account info might be in a different format
        accounts_response = await topstepx_client._make_request('GET', '/accounts')
        
        if accounts_response and accounts_response.get('success'):
            accounts = accounts_response.get('accounts', [])
            for acc in accounts:
                print(f"\nAccount: {acc.get('name')}")
                print(f"  ID: {acc.get('id')}")
                print(f"  Balance: ${acc.get('balance', 0):,.2f}")
                print(f"  Can Trade: {acc.get('canTrade')}")
                print(f"  Simulated: {acc.get('simulated')}")
                
                # Check if this is the combine account
                if '50KTC-V2-39236-56603374' in acc.get('name', ''):
                    print(f"\n✅ FOUND COMBINE ACCOUNT!")
                    print(f"   Account ID: {acc.get('id')}")
                    return acc.get('id')
    
    return None

if __name__ == "__main__":
    combine_id = asyncio.run(get_accounts())
    if combine_id:
        print(f"\n=== USE THIS ACCOUNT ID: {combine_id} ===")
    else:
        print("\n❌ Could not find combine account 50KTC-V2-39236-56603374")