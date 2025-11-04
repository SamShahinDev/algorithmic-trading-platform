#!/usr/bin/env python3
"""
Test script to verify FVG data reception fix
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nq_bot.utils.data_cache import DataCache
from web_platform.backend.brokers.topstepx_client import TopStepXClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_reception():
    """Test that FVG mode can receive historical and live data"""
    try:
        # Load environment
        load_dotenv('.env.topstepx')

        # Initialize broker
        broker = TopStepXClient()
        await broker.connect()
        logger.info("✅ Connected to TopStepX")

        # Initialize data cache
        contract_id = 'CON.F.US.ENQ.U25'
        data_cache = DataCache(broker, contract_id, is_live=False, logger=logger)

        # Test initial warmup
        logger.info("Testing warmup...")
        await data_cache.warmup(lookback_1m=50)

        # Check if we have data
        bars_1m = data_cache.get_bars('1m')
        if bars_1m.empty:
            logger.error("❌ No data after warmup!")
            return False
        else:
            logger.info(f"✅ Warmup successful: {len(bars_1m)} bars loaded")
            logger.info(f"   Latest bar: {bars_1m.index[-1]}")
            logger.info(f"   Close: {bars_1m['close'].iloc[-1]:.2f}")

        # Test incremental update
        logger.info("\nTesting incremental update...")
        await asyncio.sleep(2)

        success = await data_cache.update_incremental()
        if not success:
            logger.warning("⚠️ Incremental update returned False")

        # Check data again
        bars_after = data_cache.get_bars('1m')
        if bars_after.empty:
            logger.error("❌ Lost data after update!")
            return False
        else:
            logger.info(f"✅ After update: {len(bars_after)} bars")
            logger.info(f"   Latest bar: {bars_after.index[-1]}")
            logger.info(f"   Close: {bars_after['close'].iloc[-1]:.2f}")

        # Test get_bars with limit
        logger.info("\nTesting get_bars with limit...")
        bars_limited = data_cache.get_bars('1m', 5)
        if bars_limited.empty:
            logger.error("❌ get_bars with limit failed!")
            return False
        else:
            logger.info(f"✅ get_bars('1m', 5) returned {len(bars_limited)} bars")
            for idx, row in bars_limited.iterrows():
                logger.info(f"   {idx}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")

        # Test data freshness check
        logger.info("\nTesting data freshness...")
        latest = bars_after.index[-1]
        age = (datetime.now(tz=latest.tz) - latest).total_seconds()
        if age > 180:
            logger.warning(f"⚠️ Data might be stale: {age:.0f} seconds old")
        else:
            logger.info(f"✅ Data is fresh: {age:.0f} seconds old")

        logger.info("\n🎉 All tests passed! FVG data reception is working.")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'broker' in locals():
            await broker.disconnect()
            logger.info("Disconnected from broker")


async def main():
    """Main entry point"""
    success = await test_data_reception()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())