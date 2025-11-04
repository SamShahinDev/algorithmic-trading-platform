#!/usr/bin/env python3
"""
Check ES positions and orders
"""

import asyncio
import sys
import os

sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

from brokers.topstepx_client import TopStepXClient, OrderSide, OrderType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_and_close():
    """Check positions and close if needed"""
    
    client = TopStepXClient(
        username="exotictrades",
        api_key="86fzI3xGFVj06PWyiFYhDQGZ50QzxiJzhXW04F347h8="
    )
    client.account_id = 10983875
    
    try:
        await client.connect()
        logger.info("Connected to TopStepX")
        
        # Check open orders
        logger.info("\n=== CHECKING OPEN ORDERS ===")
        orders = await client.get_open_orders()
        if orders:
            logger.info(f"Found {len(orders)} open orders:")
            for order in orders:
                logger.info(f"  Order ID: {order.get('orderId')}")
                logger.info(f"  Symbol: {order.get('symbol')}")
                logger.info(f"  Side: {order.get('side')}")
                logger.info(f"  Quantity: {order.get('quantity')}")
                logger.info(f"  Status: {order.get('status')}")
                logger.info("  ---")
        else:
            logger.info("No open orders")
        
        # Check positions
        logger.info("\n=== CHECKING POSITIONS ===")
        positions = await client.get_positions()
        if positions:
            logger.info(f"Found {len(positions)} positions:")
            for pos in positions:
                logger.info(f"  Symbol: {pos.get('symbol')}")
                logger.info(f"  Size: {pos.get('size')}")
                logger.info(f"  Side: {pos.get('side')}")
                logger.info(f"  Entry Price: ${pos.get('averagePrice', 0):,.2f}")
                logger.info(f"  Unrealized P&L: ${pos.get('unrealizedPnl', 0):,.2f}")
                logger.info("  ---")
                
                # Offer to close
                if input("\nClose this position? (y/n): ").lower() == 'y':
                    logger.info("Closing position...")
                    
                    # Determine close side
                    close_side = OrderSide.SELL if pos.get('side') == 0 else OrderSide.BUY
                    
                    close_order = await client.place_order(
                        symbol="ES",
                        side=close_side,
                        quantity=abs(pos.get('size', 1)),
                        order_type=OrderType.MARKET
                    )
                    
                    if close_order and close_order.get('success'):
                        logger.info("✅ Close order placed successfully!")
                    else:
                        logger.error(f"Failed to close: {close_order}")
        else:
            logger.info("No open positions")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()
        logger.info("\nDisconnected from TopStepX")

if __name__ == "__main__":
    asyncio.run(check_and_close())