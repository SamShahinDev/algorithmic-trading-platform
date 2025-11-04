#!/usr/bin/env python3
"""
Force Trade Test for ES Bot
Tests trade execution capability with a small position
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

from brokers.topstepx_client import TopStepXClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_es_trade():
    """Execute a test trade on ES to verify functionality"""
    
    # Initialize client
    client = TopStepXClient(
        username="exotictrades",
        api_key="86fzI3xGFVj06PWyiFYhDQGZ50QzxiJzhXW04F347h8="
    )
    client.account_id = 10983875
    
    try:
        # Connect to TopStepX
        logger.info("Connecting to TopStepX...")
        await client.connect()
        logger.info("Connected successfully")
        
        # Get current ES price
        logger.info("Getting current ES price...")
        
        # Get market price for ES
        current_price = await client.get_market_price("ES")
        if current_price:
            logger.info(f"Current ES price: ${current_price:,.2f}")
        else:
            logger.error("Could not get ES price")
            return
        
        # ES uses specific contract ID
        contract_id = "CON.F.US.EP.U25"  # ES September 2025
        
        # Check existing positions first
        logger.info("Checking existing positions...")
        positions = await client.get_positions()
        logger.info(f"Current positions: {positions}")
        
        if positions:
            logger.warning("Already have open positions. Skipping test trade.")
            for pos in positions:
                logger.info(f"Position: {pos.get('symbol')} - Size: {pos.get('size')} - P&L: ${pos.get('unrealizedPnl', 0):,.2f}")
            return
        
        # Place a small test BUY order
        logger.info("=" * 60)
        logger.info("PLACING TEST BUY ORDER FOR ES")
        logger.info("=" * 60)
        
        order_params = {
            "contractId": contract_id,
            "action": 0,  # 0 = BUY
            "orderType": 2,  # 2 = MARKET
            "quantity": 1,  # Minimum size
            "timeInForce": 0,  # Day order
            "positionEffect": 0  # Open position
        }
        
        logger.info(f"Order parameters: {order_params}")
        
        # Place the order
        from brokers.topstepx_client import OrderSide, OrderType
        order_result = await client.place_order(
            symbol="ES",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET
        )
        
        if order_result and order_result.get('success'):
            logger.info("✅ TEST BUY ORDER PLACED SUCCESSFULLY!")
            logger.info(f"Order ID: {order_result.get('orderId')}")
            
            # Wait a moment for fill
            await asyncio.sleep(2)
            
            # Check position
            positions = await client.get_positions()
            if positions:
                logger.info("Position opened:")
                for pos in positions:
                    logger.info(f"  Symbol: {pos.get('symbol')}")
                    logger.info(f"  Size: {pos.get('size')}")
                    logger.info(f"  Entry: ${pos.get('averagePrice', 0):,.2f}")
                
                # Wait 5 seconds then close
                logger.info("Waiting 5 seconds before closing...")
                await asyncio.sleep(5)
                
                # Close the position
                logger.info("Closing test position...")
                close_order = await client.place_order(
                    symbol="ES",
                    side=OrderSide.SELL,  # SELL to close
                    quantity=1,
                    order_type=OrderType.MARKET
                )
                
                if close_order and close_order.get('success'):
                    logger.info("✅ Position closed successfully!")
                    
                    # Final check
                    await asyncio.sleep(2)
                    final_positions = await client.get_positions()
                    if not final_positions:
                        logger.info("✅ All positions closed. Test complete!")
                    else:
                        logger.warning("Position still open after close attempt")
                else:
                    logger.error(f"Failed to close position: {close_order}")
            else:
                logger.warning("No position found after order placement")
        else:
            logger.error(f"Failed to place test order: {order_result}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()
        logger.info("Disconnected from TopStepX")

if __name__ == "__main__":
    logger.info("Starting ES Trade Execution Test")
    logger.info("This will place a REAL trade on your practice account")
    logger.info("=" * 60)
    
    asyncio.run(test_es_trade())