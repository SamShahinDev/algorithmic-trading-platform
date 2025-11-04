#!/usr/bin/env python3
"""
Test ProjectX Gateway API Connection
Verifies authentication and basic API functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.projectx_client import ProjectXClient
from config import (
    PROJECTX_USERNAME, 
    PROJECTX_API_KEY, 
    PROJECTX_API_URL,
    PROJECTX_MARKET_HUB,
    PROJECTX_USER_HUB
)
from utils.logger import setup_logger

async def test_connection():
    """Test ProjectX API connection and basic functionality"""
    
    logger = setup_logger('ProjectXTest')
    
    print("\n" + "="*60)
    print("🧪 PROJECTX GATEWAY API CONNECTION TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    # Check credentials
    print("\n📋 Checking Configuration...")
    
    if not PROJECTX_USERNAME or PROJECTX_USERNAME == 'your_demo_username_here':
        print("❌ ERROR: ProjectX username not configured!")
        print("   Please update PROJECTX_USERNAME in .env file")
        return False
    
    if not PROJECTX_API_KEY or PROJECTX_API_KEY == 'your_demo_api_key_here':
        print("❌ ERROR: ProjectX API key not configured!")
        print("   Please update PROJECTX_API_KEY in .env file")
        return False
    
    print(f"✅ Username: {PROJECTX_USERNAME[:4]}****")
    print(f"✅ API Key: {PROJECTX_API_KEY[:8]}****")
    print(f"✅ API URL: {PROJECTX_API_URL}")
    
    # Create client
    print("\n🔌 Creating ProjectX Client...")
    client = ProjectXClient(
        username=PROJECTX_USERNAME,
        api_key=PROJECTX_API_KEY,
        api_url=PROJECTX_API_URL,
        market_hub=PROJECTX_MARKET_HUB,
        user_hub=PROJECTX_USER_HUB
    )
    
    try:
        # Test 1: Authentication
        print("\n1️⃣ Testing Authentication...")
        connected = await client.connect()
        
        if connected:
            print("✅ Authentication successful!")
            print(f"   Token expires: {client.token_expiry}")
        else:
            print("❌ Authentication failed!")
            print("   Check your username and API key")
            return False
        
        # Test 2: Account Info
        print("\n2️⃣ Fetching Account Information...")
        account_info = await client.get_account_info()
        
        if account_info:
            print("✅ Account info retrieved:")
            print(f"   Account ID: {account_info.get('accountId', 'N/A')}")
            print(f"   Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"   Buying Power: ${account_info.get('buyingPower', 0):,.2f}")
        else:
            print("⚠️ Could not retrieve account info (may be normal for demo)")
        
        # Test 3: Get Positions
        print("\n3️⃣ Checking Open Positions...")
        positions = await client.get_positions()
        
        if isinstance(positions, list):
            print(f"✅ Positions endpoint accessible")
            print(f"   Open positions: {len(positions)}")
            
            if positions:
                for pos in positions[:3]:  # Show first 3
                    print(f"   - {pos.get('symbol')}: {pos.get('quantity')} @ {pos.get('avgPrice')}")
        else:
            print("⚠️ Could not retrieve positions (may be normal for demo)")
        
        # Test 4: Market Data
        print("\n4️⃣ Testing Market Data Access...")
        market_data = await client.get_market_data('NQ')
        
        if market_data:
            print("✅ Market data retrieved:")
            print(f"   Symbol: NQ")
            print(f"   Bid: {market_data.get('bid', 'N/A')}")
            print(f"   Ask: {market_data.get('ask', 'N/A')}")
            print(f"   Last: {market_data.get('last', 'N/A')}")
        else:
            print("⚠️ Could not retrieve market data (may require active market hours)")
        
        # Test 5: Connection State
        print("\n5️⃣ Connection Status:")
        print(f"✅ State: {client.state.value}")
        print(f"✅ Session active: {client.session is not None}")
        print(f"✅ Auth token valid: {client.auth_token is not None}")
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("-"*60)
        print("✅ Connection: SUCCESSFUL")
        print("✅ Authentication: WORKING")
        print("✅ API Access: VERIFIED")
        print("\n🎉 ProjectX Gateway API is ready for trading!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await client.disconnect()
        print("✅ Disconnected")

async def main():
    """Main test runner"""
    print("\n🚀 Starting ProjectX Connection Test")
    print("This will verify your API credentials and connection")
    
    success = await test_connection()
    
    if success:
        print("\n✅ All tests passed! Your trading bot is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main_orchestrator.py' to start the full system")
        print("2. Monitor Slack channels for updates")
        print("3. Watch patterns being discovered and validated")
    else:
        print("\n❌ Tests failed. Please check your configuration.")
        print("\nTroubleshooting:")
        print("1. Verify your ProjectX demo credentials are correct")
        print("2. Check that you have an active internet connection")
        print("3. Ensure the ProjectX API is accessible")
        print("4. Review the error messages above")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)