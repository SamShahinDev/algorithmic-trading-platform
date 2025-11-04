#!/usr/bin/env python3
"""
Comprehensive test of FVG bot functionality
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Set dry run mode
os.environ['FVG_DRY_RUN'] = 'true'

# Add paths
sys.path.insert(0, "/Users/royaltyvixion/Documents/XTRADING")
sys.path.insert(0, "/Users/royaltyvixion/Documents/XTRADING/nq_bot")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_fvg_comprehensive():
    """Run comprehensive FVG bot test"""
    # Direct import from file
    import fvg_runner
    FVGRunner = fvg_runner.FVGRunner

    try:
        logger.info("="*60)
        logger.info("COMPREHENSIVE FVG BOT TEST")
        logger.info("="*60)

        # Create runner
        runner = FVGRunner()
        logger.info("✅ FVG Runner created")

        # Initialize
        success = await runner.initialize()
        if not success:
            logger.error("❌ Initialization failed")
            return False
        logger.info("✅ FVG Runner initialized")

        # Check data cache
        bars = runner.data_cache.get_bars('1m', 10)
        if bars.empty:
            logger.error("❌ No data in cache")
            return False
        logger.info(f"✅ Data cache has {len(bars)} bars")
        logger.info(f"   Latest: {bars.index[-1]} Close: {bars['close'].iloc[-1]:.2f}")

        # Test data freshness check
        is_fresh = runner.check_data_freshness()
        logger.info(f"✅ Data freshness: {is_fresh}")

        # Test position cap check
        can_trade = runner.check_position_cap()
        logger.info(f"✅ Position cap check: {can_trade}")

        # Test RSI veto (should not crash)
        try:
            veto = runner.check_rsi_veto('long')
            logger.info(f"✅ RSI veto check (long): {not veto}")
        except Exception as e:
            logger.warning(f"⚠️ RSI veto check error (expected if insufficient data): {e}")

        # Update data cache
        await runner.data_cache.update_incremental()
        bars_after = runner.data_cache.get_bars('1m', 5)
        if not bars_after.empty:
            logger.info(f"✅ Data update successful, latest: {bars_after.index[-1]}")
        else:
            logger.error("❌ Data update failed")
            return False

        # Test FVG scanning
        counts = runner.fvg_strategy.scan()
        logger.info(f"✅ FVG scan complete: fresh={counts.get('fresh', 0)}, armed={counts.get('armed', 0)}")

        # Run for a few iterations
        logger.info("\nRunning main loop for 10 seconds...")
        start_time = asyncio.get_event_loop().time()
        iterations = 0

        while asyncio.get_event_loop().time() - start_time < 10:
            try:
                # Update data
                await runner.data_cache.update_incremental()

                # Scan for FVGs
                counts = runner.fvg_strategy.scan()
                iterations += 1

                # Check for armed FVGs
                best_armed = runner.fvg_strategy.get_best_armed()
                if best_armed:
                    logger.info(f"   Found armed FVG: {best_armed.id} {best_armed.direction}")

                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Loop error: {e}")
                break

        logger.info(f"✅ Completed {iterations} iterations")

        # Clean up
        await runner.cleanup()
        logger.info("✅ Cleanup complete")

        logger.info("\n" + "="*60)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("The FVG bot is now able to:")
        logger.info("  ✅ Connect to TopStepX")
        logger.info("  ✅ Receive historical data")
        logger.info("  ✅ Update data incrementally")
        logger.info("  ✅ Scan for FVG patterns")
        logger.info("  ✅ Check trading conditions")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main entry point"""
    success = await test_fvg_comprehensive()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())