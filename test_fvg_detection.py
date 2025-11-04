#!/usr/bin/env python3
"""Test FVG detection to identify why no FVGs are being found"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nq_bot'))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def test_fvg_detection():
    """Test FVG detection with debug output"""

    from dotenv import load_dotenv
    load_dotenv('.env.topstepx')

    from web_platform.backend.brokers.topstepx_client import TopStepXClient
    from nq_bot.utils.data_cache import DataCache
    from nq_bot.patterns.fvg_strategy import FVGStrategy
    from nq_bot.pattern_config import FVG

    print("\n" + "="*60)
    print("FVG DETECTION TEST - DIAGNOSTIC MODE")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print("="*60 + "\n")

    # Initialize broker
    broker = TopStepXClient()
    await broker.connect()

    # Get account info
    account_info = await broker.get_account_info()
    print(f"✓ Connected to TopStepX Practice")
    print(f"  Balance: ${account_info.get('balance', 0):,.2f}")
    print()

    # Initialize data cache
    data_cache = DataCache(broker, logger)
    # No init method needed - it initializes in constructor
    print(f"✓ Data cache initialized")

    # Initialize FVG strategy
    fvg_strategy = FVGStrategy(data_cache, logger, FVG)
    print(f"✓ FVG Strategy initialized")
    print()

    # Run a few scans
    print("Running FVG detection scans...")
    print("-" * 40)

    for i in range(5):
        # Get current data stats
        bars_1m = data_cache.get_bars('1m')

        if bars_1m is None or len(bars_1m) == 0:
            print(f"Scan {i+1}: No market data available")
        else:
            print(f"\nScan {i+1}:")
            print(f"  1m bars available: {len(bars_1m)}")
            print(f"  Latest close: {bars_1m['close'].iloc[-1]:.2f}")
            print(f"  Latest volume: {bars_1m['volume'].iloc[-1]}")

            # Check recent swings
            fvg_strategy._update_swings(bars_1m)
            print(f"  Recent swings found: {len(fvg_strategy.recent_swings)}")

            if len(fvg_strategy.recent_swings) > 0:
                for swing in fvg_strategy.recent_swings[-3:]:
                    print(f"    - {swing['type']} at {swing['level']:.2f} (bar {swing['bar_idx']})")

            # Run scan
            status = fvg_strategy.scan()
            print(f"  FVG Status: fresh={status.get('fresh', 0)}, armed={status.get('armed', 0)}")

            # Check for gaps without liquidity sweep requirement
            if len(bars_1m) >= 10:
                # Check last few bars for any price gaps
                for j in range(len(bars_1m) - 5, len(bars_1m) - 2):
                    # Bullish gap
                    if bars_1m.iloc[j]['high'] < bars_1m.iloc[j + 2]['low']:
                        gap_size = bars_1m.iloc[j + 2]['low'] - bars_1m.iloc[j]['high']
                        print(f"  → Bullish gap found at bar {j}: {gap_size:.2f} pts")
                        print(f"     Bar {j} high: {bars_1m.iloc[j]['high']:.2f}")
                        print(f"     Bar {j+2} low: {bars_1m.iloc[j + 2]['low']:.2f}")

                    # Bearish gap
                    if bars_1m.iloc[j]['low'] > bars_1m.iloc[j + 2]['high']:
                        gap_size = bars_1m.iloc[j]['low'] - bars_1m.iloc[j + 2]['high']
                        print(f"  → Bearish gap found at bar {j}: {gap_size:.2f} pts")
                        print(f"     Bar {j} low: {bars_1m.iloc[j]['low']:.2f}")
                        print(f"     Bar {j+2} high: {bars_1m.iloc[j + 2]['high']:.2f}")

        # Wait before next scan
        await asyncio.sleep(3)

    # Check FVG configuration
    print("\n" + "="*40)
    print("FVG CONFIGURATION:")
    print("-" * 40)
    print(f"Detection mode: {FVG['detection']['min_displacement_mode']}")
    print(f"Min body fraction: {FVG['detection']['min_body_frac']}")
    print(f"Min ATR multiplier: {FVG['detection']['min_atr_mult']}")
    print(f"Min volume multiplier: {FVG['detection']['min_vol_mult']}")
    print(f"Min quality score: {FVG['quality']['min_quality']}")

    if FVG['detection']['min_displacement_mode'] == 'dynamic':
        dyn = FVG['detection']['min_displacement_dyn']
        print(f"Dynamic displacement: max({dyn['base_pts']}, {dyn['atr_mult']} * ATR)")

    print("\n⚠️  KEY FINDING: FVG detection requires liquidity sweep")
    print("   - Code checks for swing high/low sweep before displacement")
    print("   - Without liquidity sweep, FVG is skipped (lines 245-247, 295-298)")
    print("   - This may be why no FVGs are detected in ranging markets")

    await broker.close()
    print("\n✓ Test complete")

if __name__ == "__main__":
    asyncio.run(test_fvg_detection())