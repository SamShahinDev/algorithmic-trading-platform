#!/usr/bin/env python3
"""
Manual Test Trade Script for TopStepX
Executes a quick test trade - enters and exits within seconds
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.topstepx_client import topstepx_client, OrderType, OrderSide
from utils.logger import setup_logger

logger = setup_logger('TestTrade')


async def execute_test_trade():
    """Execute a quick test trade - buy and sell within seconds"""
    
    logger.info("=" * 50)
    logger.info("STARTING TEST TRADE EXECUTION")
    logger.info("=" * 50)
    
    try:
        # Connect to TopStepX
        logger.info("🔌 Connecting to TopStepX...")
        connected = await topstepx_client.connect()
        
        if not connected:
            logger.error("❌ Failed to connect to TopStepX")
            return False
        
        logger.info("✅ Connected to TopStepX")
        
        # Get account info
        account_info = await topstepx_client.get_account_info()
        if account_info:
            logger.info(f"📊 Account Balance: ${account_info.get('balance', 0)}")
            logger.info(f"📊 Account ID: {topstepx_client.account_id}")
        
        # Get current market price
        logger.info("\n🔍 Getting current market price...")
        current_price = await topstepx_client.get_market_price()
        
        if current_price and current_price > 0:
            logger.info(f"📈 Current NQ Price: ${current_price}")
        else:
            logger.warning("⚠️ Could not get market price - using default")
            current_price = 23500  # Fallback price
        
        # Check if market is open
        if not topstepx_client.is_market_open():
            logger.warning("⚠️ Market is closed - trade may be rejected")
            response = input("Market is closed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        logger.info("\n" + "=" * 50)
        logger.info("🎯 EXECUTING TEST TRADE")
        logger.info("=" * 50)
        
        # Step 1: Place a MARKET BUY order
        logger.info("\n📗 STEP 1: Placing BUY order...")
        logger.info(f"   Symbol: NQ")
        logger.info(f"   Side: BUY")
        logger.info(f"   Quantity: 1 contract")
        logger.info(f"   Type: MARKET")
        
        buy_result = await topstepx_client.place_order(
            symbol="NQ",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
            custom_tag="test_trade_entry"
        )
        
        if buy_result and buy_result.get('success'):
            buy_order_id = buy_result.get('orderId')
            logger.info(f"✅ BUY order placed successfully!")
            logger.info(f"   Order ID: {buy_order_id}")
            
            # Wait 3 seconds
            logger.info("\n⏰ Waiting 3 seconds before closing...")
            for i in range(3, 0, -1):
                logger.info(f"   {i}...")
                await asyncio.sleep(1)
            
            # Step 2: Close the position (MARKET SELL)
            logger.info("\n📕 STEP 2: Closing position with SELL order...")
            logger.info(f"   Symbol: NQ")
            logger.info(f"   Side: SELL")
            logger.info(f"   Quantity: 1 contract")
            logger.info(f"   Type: MARKET")
            
            sell_result = await topstepx_client.place_order(
                symbol="NQ",
                side=OrderSide.SELL,
                quantity=1,
                order_type=OrderType.MARKET,
                custom_tag="test_trade_exit"
            )
            
            if sell_result and sell_result.get('success'):
                sell_order_id = sell_result.get('orderId')
                logger.info(f"✅ SELL order placed successfully!")
                logger.info(f"   Order ID: {sell_order_id}")
                
                # Wait for settlement
                await asyncio.sleep(2)
                
                # Check final position
                positions = await topstepx_client.get_positions()
                if positions:
                    logger.info(f"\n📊 Final Positions: {len(positions)}")
                    for pos in positions:
                        logger.info(f"   {pos}")
                else:
                    logger.info("\n✅ No open positions - trade closed successfully!")
                
                logger.info("\n" + "=" * 50)
                logger.info("🎉 TEST TRADE COMPLETED SUCCESSFULLY!")
                logger.info("=" * 50)
                
                return True
            else:
                error_msg = sell_result.get('errorMessage', 'Unknown error') if sell_result else 'No response'
                logger.error(f"❌ SELL order failed: {error_msg}")
                
                # Try alternative close method
                logger.info("\n🔄 Attempting alternative position close...")
                contract_id = await topstepx_client._get_contract_id("NQ")
                if contract_id:
                    close_success = await topstepx_client.close_position(contract_id)
                    if close_success:
                        logger.info("✅ Position closed using alternative method")
                        return True
                
        else:
            error_msg = buy_result.get('errorMessage', 'Unknown error') if buy_result else 'No response'
            logger.error(f"❌ BUY order failed: {error_msg}")
            
            # Common issues and solutions
            logger.info("\n📋 Troubleshooting:")
            if "insufficient" in error_msg.lower():
                logger.info("   • Check account balance and margin requirements")
            elif "closed" in error_msg.lower():
                logger.info("   • Market is closed - wait for market hours")
            elif "auth" in error_msg.lower():
                logger.info("   • Check API key and account credentials")
            else:
                logger.info("   • Verify account ID is correct")
                logger.info("   • Ensure evaluation account is active")
                logger.info("   • Check TopStepX dashboard for any issues")
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Test trade error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Ensure we're disconnected
        if topstepx_client.connected:
            # Check for any open positions and warn
            positions = await topstepx_client.get_positions()
            if positions and len(positions) > 0:
                logger.warning("⚠️ WARNING: Open positions remain!")
                logger.warning("   Manual intervention may be required")
            
            await topstepx_client.disconnect()
            logger.info("🔌 Disconnected from TopStepX")


async def main():
    """Main function with safety confirmation"""
    
    logger.info("📊 TopStepX Test Trade Script")
    logger.info("This will execute a REAL trade on your account")
    logger.info("The trade will:")
    logger.info("  1. BUY 1 NQ contract at market")
    logger.info("  2. Wait 3 seconds")
    logger.info("  3. SELL 1 NQ contract at market")
    logger.info("")
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.topstepx')
    
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID', 'Not Set')
    username = os.getenv('TOPSTEPX_USERNAME', 'Not Set')
    
    logger.info(f"Account ID: {account_id}")
    logger.info(f"Username: {username}")
    logger.info("")
    
    response = input("⚠️  Execute test trade? (type 'yes' to confirm): ")
    
    if response.lower() == 'yes':
        success = await execute_test_trade()
        
        if success:
            logger.info("\n✅ Test completed successfully!")
            logger.info("The connection is working and orders can be executed.")
        else:
            logger.info("\n❌ Test failed - check the error messages above")
            logger.info("Verify your account credentials and that the market is open")
    else:
        logger.info("Test cancelled by user")


if __name__ == "__main__":
    asyncio.run(main())