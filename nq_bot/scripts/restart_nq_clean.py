#!/usr/bin/env python3
"""
Clean restart for NQ bot with forced position sync
"""

import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'web_platform/backend'))
from brokers.topstepx_client import topstepx_client

async def verify_no_positions():
    """Verify broker has no NQ positions"""
    print("Checking broker positions...")
    positions = await topstepx_client.get_positions()
    
    if positions:
        for pos in positions:
            if 'NQ' in str(pos.get('contractId', '')) or 'ENQ' in str(pos.get('contractId', '')):
                print(f"WARNING: Found NQ position: {pos}")
                return False
    
    print("✅ Confirmed: No NQ positions in broker")
    return True

async def main():
    # Kill any running NQ bot
    os.system("pkill -f 'nq_bot.py' 2>/dev/null")
    print("Killed any existing NQ bot process")
    
    # Clear PID file
    os.system("rm -f nq_bot.pid")
    print("Cleared PID file")
    
    # Verify no positions
    no_positions = await verify_no_positions()
    
    if not no_positions:
        print("❌ ERROR: Broker has open positions. Please close them first.")
        return
    
    # Start fresh NQ bot
    print("\nStarting fresh NQ bot...")
    os.system("cd .. && python3 nq_bot.py > logs/nq_bot_fresh.log 2>&1 &")
    print("✅ NQ bot started with clean slate")
    print("Monitor with: tail -f ../logs/nq_bot_fresh.log")

if __name__ == "__main__":
    asyncio.run(main())