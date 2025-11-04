#!/usr/bin/env python3
"""
Quick test of FVG bot with timeout
"""
import asyncio
import signal
import sys
import os

os.environ['FVG_DRY_RUN'] = 'true'

from start_fvg_bot import main


async def test_with_timeout():
    """Run FVG bot for 20 seconds then exit"""
    try:
        # Create the main task
        task = asyncio.create_task(main())

        # Wait for 20 seconds
        await asyncio.sleep(20)

        # Cancel the task
        task.cancel()

        print("\n✅ FVG bot ran successfully for 20 seconds")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == '__main__':
    success = asyncio.run(test_with_timeout())
    sys.exit(0 if success else 1)