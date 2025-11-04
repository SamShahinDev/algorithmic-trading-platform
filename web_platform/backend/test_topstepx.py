#!/usr/bin/env python3
"""
Test TopStepX ProjectX Gateway Connection
Run this after adding your username to .env.topstepx
"""

import asyncio
import os
from dotenv import load_dotenv
from brokers.topstepx_client import TopStepXClient
from datetime import datetime

# Load environment
load_dotenv('.env.topstepx')

async def test_connection():
    """Test TopStepX connection and basic operations"""
    
    print("=" * 60)
    print("TopStepX ProjectX Gateway Connection Test")
    print("=" * 60)
    
    # Check configuration
    username = os.getenv('TOPSTEPX_USERNAME', '')
    api_key = os.getenv('TOPSTEPX_API_KEY', '')
    
    print(f"\n📋 Configuration:")
    print(f"  Username: {'✅ Configured' if username else '❌ Missing'}")
    print(f"  API Key: {'✅ Configured' if api_key else '❌ Missing'}")
    print(f"  API URL: {os.getenv('API_BASE_URL', 'https://api.topstepx.com/api')}")
    
    if not username:
        print("\n❌ ERROR: Username not configured!")
        print("Please add TOPSTEPX_USERNAME to .env.topstepx file")
        return False
    
    # Create client
    print(f"\n🔌 Creating TopStepX client...")
    client = TopStepXClient()
    
    # Test connection
    print(f"🔐 Attempting to connect...")
    connected = await client.connect()
    
    if connected:
        print(f"✅ Successfully connected to TopStepX!")
        
        # Get account info
        print(f"\n📊 Account Information:")
        if client.account_id:
            print(f"  Account ID: {client.account_id}")
            print(f"  Session Token: {'Active' if client.session_token else 'None'}")
            print(f"  Token Expiry: {client.token_expiry}")
        
        # Get market price
        print(f"\n📈 Testing Market Data:")
        price = await client.get_market_price("NQ")
        if price > 0:
            print(f"  NQ Current Price: ${price:,.2f}")
            print(f"  Bid: ${client.bid:,.2f}")
            print(f"  Ask: ${client.ask:,.2f}")
        else:
            print(f"  ⚠️ Market data not available (market may be closed)")
        
        # Check market hours
        if client.is_market_open():
            print(f"\n✅ Market is OPEN")
        else:
            print(f"\n⏸️ Market is CLOSED")
            print(f"  Futures open Sunday 5PM CT")
        
        # Disconnect
        await client.disconnect()
        print(f"\n👋 Disconnected from TopStepX")
        return True
        
    else:
        print(f"❌ Failed to connect to TopStepX")
        print(f"Please check:")
        print(f"  1. Username is correct")
        print(f"  2. API key is valid")
        print(f"  3. Account is active")
        return False

if __name__ == "__main__":
    print(f"Current time: {datetime.now()}")
    
    # Run test
    success = asyncio.run(test_connection())
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Connection test PASSED")
        print("System is ready for trading!")
    else:
        print("❌ Connection test FAILED")
        print("Please fix the issues above and try again")
    print("=" * 60)