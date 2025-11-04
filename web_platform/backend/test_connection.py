#!/usr/bin/env python3
"""
Test TopStepX connection and verify account/contract setup
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.topstepx_client import topstepx_client
from utils.logger import setup_logger

logger = setup_logger('ConnectionTest')


async def test_connection():
    """Test connection and verify account setup"""
    
    logger.info("=" * 50)
    logger.info("TESTING TOPSTEPX CONNECTION")
    logger.info("=" * 50)
    
    try:
        # Step 1: Connect
        logger.info("\n1. Connecting to TopStepX...")
        connected = await topstepx_client.connect()
        
        if not connected:
            logger.error("❌ Failed to connect")
            return False
        
        logger.info("✅ Connected successfully")
        logger.info(f"   Session Token: {topstepx_client.session_token[:20]}...")
        logger.info(f"   Account ID: {topstepx_client.account_id}")
        
        # Step 2: Get account info
        logger.info("\n2. Getting account information...")
        account_info = await topstepx_client.get_account_info()
        
        if account_info:
            logger.info("✅ Account info retrieved:")
            logger.info(f"   Balance: ${account_info.get('balance', 0)}")
            logger.info(f"   Account ID: {account_info.get('accountId')}")
            logger.info(f"   Username: {account_info.get('username')}")
        else:
            logger.warning("⚠️ Could not get account info")
        
        # Step 3: Get contract ID for NQ
        logger.info("\n3. Getting contract ID for NQ...")
        contract_id = await topstepx_client._get_contract_id("NQ")
        
        if contract_id:
            logger.info(f"✅ Contract ID for NQ: {contract_id}")
        else:
            logger.error("❌ Failed to get contract ID for NQ")
            return False
        
        # Step 4: Get current market price
        logger.info("\n4. Getting current market price...")
        current_price = await topstepx_client.get_market_price()
        
        if current_price and current_price > 0:
            logger.info(f"✅ Current NQ Price: ${current_price}")
        else:
            logger.warning("⚠️ Could not get market price")
        
        # Step 5: Check market hours
        logger.info("\n5. Checking market hours...")
        is_open = topstepx_client.is_market_open()
        
        if is_open:
            logger.info("✅ Market is OPEN")
        else:
            logger.warning("⚠️ Market is CLOSED")
            logger.info("   Market hours: Mon-Fri")
            logger.info("   - Opens: 6:00 PM ET (Sunday)")
            logger.info("   - Closes: 5:00 PM ET (Friday)")
        
        # Step 6: Get open positions
        logger.info("\n6. Checking open positions...")
        positions = await topstepx_client.get_positions()
        
        if positions:
            logger.info(f"📊 Found {len(positions)} open positions")
            for pos in positions:
                logger.info(f"   - {pos}")
        else:
            logger.info("✅ No open positions")
        
        # Step 7: Get open orders
        logger.info("\n7. Checking open orders...")
        orders = await topstepx_client.get_open_orders()
        
        if orders:
            logger.info(f"📋 Found {len(orders)} open orders")
            for order in orders:
                logger.info(f"   - {order}")
        else:
            logger.info("✅ No open orders")
        
        # Step 8: Test order placement (DRY RUN - not actually placing)
        logger.info("\n8. Testing order placement capability...")
        logger.info("   Would place: MARKET BUY 1 NQ")
        logger.info("   Account ID: " + str(topstepx_client.account_id))
        logger.info("   Contract ID: " + str(contract_id))
        
        if topstepx_client.account_id and contract_id:
            logger.info("✅ Ready to place orders")
        else:
            logger.error("❌ Missing required data for order placement")
            return False
        
        logger.info("\n" + "=" * 50)
        logger.info("CONNECTION TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection test error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        if topstepx_client.connected:
            await topstepx_client.disconnect()
            logger.info("\n🔌 Disconnected from TopStepX")


async def main():
    """Main function"""
    
    logger.info("📊 TopStepX Connection Test")
    logger.info("This will test the connection without placing any trades")
    logger.info("")
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.topstepx')
    
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID', 'Not Set')
    username = os.getenv('TOPSTEPX_USERNAME', 'Not Set')
    api_key = os.getenv('TOPSTEPX_API_KEY', 'Not Set')[:10] + "..."
    
    logger.info(f"Configuration:")
    logger.info(f"  Account ID: {account_id}")
    logger.info(f"  Username: {username}")
    logger.info(f"  API Key: {api_key}")
    logger.info("")
    
    success = await test_connection()
    
    if success:
        logger.info("\n✅ All tests passed!")
        logger.info("The connection is working properly.")
    else:
        logger.info("\n❌ Some tests failed")
        logger.info("Check the error messages above for details.")


if __name__ == "__main__":
    asyncio.run(main())