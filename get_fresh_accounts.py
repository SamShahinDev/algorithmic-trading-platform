#!/usr/bin/env python3
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv('.env.topstepx')

username = os.getenv('TOPSTEPX_USERNAME')
password = os.getenv('TOPSTEPX_PASSWORD')

# Login first
login_url = 'https://api.topstepx.com/api/v3/auth/login'
login_data = {
    'username': username,
    'password': password
}

print(f"🔄 Logging in as {username}...")
print(f"⏰ Timestamp: {datetime.now()}")

session = requests.Session()
login_response = session.post(login_url, json=login_data)

if login_response.status_code == 200:
    print("✅ Login successful")
    token = login_response.json().get('token')

    # Get accounts
    accounts_url = 'https://api.topstepx.com/api/v3/accounts'
    headers = {'Authorization': f'Bearer {token}'}

    accounts_response = session.get(accounts_url, headers=headers)

    if accounts_response.status_code == 200:
        accounts_data = accounts_response.json()
        accounts = accounts_data.get('accounts', [])

        print(f"\n📊 Found {len(accounts)} accounts:\n")
        print("=" * 90)

        # Find new 50K account
        new_50k = None
        for acc in accounts:
            balance = acc.get('balance', 0)
            name = acc.get('name', '')
            acc_id = acc.get('id', '')
            can_trade = acc.get('canTrade', False)
            simulated = acc.get('simulated', True)

            # Check if this is a 50K account
            is_50k = '50K' in name or '50KTC' in name

            # Check if it's NOT one of the known ineligible accounts
            is_known_ineligible = name.endswith('13140370') or name.endswith('56603374') or name.endswith('42686587')

            # New account should be close to 50K and tradeable
            is_new = balance >= 49800 and balance <= 50100 and can_trade

            if is_50k and not is_known_ineligible:
                prefix = "🎯 NEW" if is_new else "   "
                status = "✅ ACTIVE" if can_trade else "❌ INACTIVE"
                print(f"{prefix} ID: {acc_id:>10} | {name:<35} | ${balance:>10,.2f} | {status}")

                if is_new and not new_50k:
                    new_50k = acc
            else:
                status = "✅ ACTIVE" if can_trade else "❌ INACTIVE"
                print(f"     ID: {acc_id:>10} | {name:<35} | ${balance:>10,.2f} | {status}")

        print("=" * 90)

        if new_50k:
            print(f"\n✅ FOUND NEW 50K ACCOUNT!")
            print(f"   Account ID: {new_50k['id']}")
            print(f"   Name: {new_50k['name']}")
            print(f"   Balance: ${new_50k['balance']:,.2f}")
            print(f"\n   To use this account, update TOPSTEPX_ACCOUNT_ID={new_50k['id']} in .env.topstepx")
        else:
            print("\n⚠️  No new 50K account found. Showing all 50K accounts above.")
            print("   If you just created the account, wait 30-60 seconds and try again.")
    else:
        print(f"❌ Failed to get accounts: {accounts_response.status_code}")
        print(accounts_response.text)
else:
    print(f"❌ Login failed: {login_response.status_code}")
    print(login_response.text)