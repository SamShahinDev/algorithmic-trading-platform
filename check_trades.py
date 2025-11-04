#!/usr/bin/env python3
"""Check actual trades and P&L from TopStep broker"""

import asyncio
import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')
from brokers.topstepx_client import topstepx_client

async def check_trades():
    # Get account info
    account_info = await topstepx_client.get_account_info()
    if account_info:
        print("=== ACCOUNT STATUS ===")
        print(f"Account Balance: ${account_info.get('balance', 0):,.2f}")
        print(f"Cash Balance: ${account_info.get('cashBalance', 0):,.2f}")
        print(f"P&L Today: ${account_info.get('pnlToday', 0):,.2f}")
        print(f"Open P&L: ${account_info.get('openPnl', 0):,.2f}")
    
    # Get positions
    print("\n=== CURRENT POSITIONS ===")
    positions = await topstepx_client.get_positions()
    if positions:
        for pos in positions:
            print(f"{pos.get('symbol')}: {pos.get('netPos')} contracts @ avg ${pos.get('avgPrice', 0):.2f}")
            print(f"  Unrealized P&L: ${pos.get('unrealizedPnl', 0):,.2f}")
    else:
        print("No open positions")
    
    # Get recent orders
    print("\n=== RECENT ORDERS (Last 10) ===")
    orders = await topstepx_client.get_orders(limit=10)
    if orders:
        for order in orders[:10]:
            side = "BUY" if order.get('side') == 0 else "SELL"
            status = order.get('status', 'UNKNOWN')
            price = order.get('price', 'MARKET')
            filled_qty = order.get('filledQty', 0)
            avg_price = order.get('avgPrice', 0)
            
            print(f"Order {order.get('id')}: {side} {order.get('size')} @ {price}")
            print(f"  Status: {status}, Filled: {filled_qty} @ avg ${avg_price:.2f}")
            
            # Calculate P&L for filled orders
            if filled_qty > 0 and status in ['FILLED', 'PARTIALLY_FILLED']:
                if side == "SELL" and avg_price > 0:
                    # This was an exit order
                    print(f"  Exit Price: ${avg_price:.2f}")

if __name__ == "__main__":
    asyncio.run(check_trades())