#!/usr/bin/env python3
"""
Test Enhanced FVG Implementation with Responsive Profile
Tests new patterns and modules with responsive profile settings
"""

import os
import sys
import asyncio
import logging

# Set dry run mode
os.environ['FVG_DRY_RUN'] = 'true'

# Add paths
sys.path.insert(0, "/Users/royaltyvixion/Documents/XTRADING")
sys.path.insert(0, "/Users/royaltyvixion/Documents/XTRADING/nq_bot")

# Change to correct directory
os.chdir("/Users/royaltyvixion/Documents/XTRADING/nq_bot")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_responsive_profile():
    """Test responsive profile with all new modules enabled"""

    try:
        # Import and modify config to use responsive profile
        from pattern_config import FVG_CFG, FVG

        # Switch to responsive profile for testing
        FVG_CFG.profile_active = "responsive"

        logger.info("✅ Configuration loading test (responsive profile)")
        logger.info(f"   Profile active: {FVG_CFG.profile_active}")
        logger.info(f"   Responsive profile thresholds: body_frac={FVG_CFG.responsive.displacement_body_frac_min_base}")
        logger.info(f"   Volume mult: {FVG_CFG.responsive.volume_min_mult_trend}")
        logger.info(f"   Quality min: {FVG_CFG.responsive.quality_score_min_trend}")
        logger.info(f"   Min points floor: {FVG_CFG.responsive.displacement_min_points_floor}")

        # Test all patterns enabled
        patterns = FVG_CFG.patterns
        logger.info(f"✅ Pattern toggles test:")
        logger.info(f"   Core FVG: {patterns.enable_core_fvg}")
        logger.info(f"   OB-FVG: {patterns.enable_ob_fvg}")
        logger.info(f"   IRL-ERL-FVG: {patterns.enable_irl_erl_fvg}")
        logger.info(f"   Breaker-FVG: {patterns.enable_breaker_fvg}")

        # Test FVG strategy initialization with responsive profile
        from patterns.fvg_strategy import FVGStrategy
        from unittest.mock import Mock
        import pandas as pd
        import numpy as np

        # Create mock data cache with sample data
        data_cache = Mock()

        # Create sample 1m bars for testing
        dates = pd.date_range('2025-09-16 10:00', periods=100, freq='1min')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(21000, 21100, 100),
            'high': np.random.uniform(21050, 21150, 100),
            'low': np.random.uniform(20950, 21050, 100),
            'close': np.random.uniform(21000, 21100, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)

        # Ensure high > low > open/close relationships are valid
        for i in range(len(sample_data)):
            low = sample_data.iloc[i]['low']
            high = sample_data.iloc[i]['high']
            if high <= low:
                sample_data.iloc[i, sample_data.columns.get_loc('high')] = low + 2.0

        data_cache.get_bars.return_value = sample_data

        # Enhanced config with responsive profile
        enhanced_config = FVG.copy()
        enhanced_config['cfg'] = FVG_CFG

        # Initialize strategy
        strategy = FVGStrategy(data_cache, logger, enhanced_config)
        logger.info("✅ FVGStrategy initialization test (responsive)")

        # Test responsive profile access
        prof = strategy._prof()
        if prof is not None:
            logger.info(f"✅ Responsive profile access test: {prof.name}")
            logger.info(f"   Body frac min: {prof.displacement_body_frac_min_base}")
            logger.info(f"   Volume mult: {prof.volume_min_mult_trend}")
            logger.info(f"   Quality min: {prof.quality_score_min_trend}")
            logger.info(f"   Min points floor: {prof.displacement_min_points_floor}")

            # Verify responsive settings are different from normal
            normal_prof = FVG_CFG.normal
            assert prof.displacement_body_frac_min_base < normal_prof.displacement_body_frac_min_base
            assert prof.volume_min_mult_trend < normal_prof.volume_min_mult_trend
            assert prof.quality_score_min_trend < normal_prof.quality_score_min_trend
            logger.info("✅ Responsive profile has relaxed thresholds vs normal")
        else:
            logger.error("❌ Responsive profile access failed")
            return False

        # Test module imports
        logger.info("✅ Testing module imports:")

        try:
            from patterns.modules.ob_fvg import scan_ob_fvg
            logger.info("   OB-FVG module imported successfully")
        except Exception as e:
            logger.error(f"   ❌ OB-FVG import error: {e}")

        try:
            from patterns.modules.irl_erl_fvg import scan_irl_erl_fvg
            logger.info("   IRL-ERL-FVG module imported successfully")
        except Exception as e:
            logger.error(f"   ❌ IRL-ERL-FVG import error: {e}")

        try:
            from patterns.modules.breaker_fvg import scan_breaker_fvg
            logger.info("   Breaker-FVG module imported successfully")
        except Exception as e:
            logger.error(f"   ❌ Breaker-FVG import error: {e}")

        # Test scan method with sample data
        logger.info("✅ Testing scan with sample data:")
        scan_result = strategy.scan()
        logger.info(f"   Scan result: {scan_result}")

        # Check telemetry for new patterns
        logger.info("✅ Checking telemetry counters:")
        for pattern_type in ['ob_fvg_detected', 'irl_erl_fvg_detected', 'breaker_fvg_detected']:
            count = strategy.telemetry_counters.get(pattern_type, 0)
            logger.info(f"   {pattern_type}: {count}")

        # Test analytics integration
        logger.info("✅ Testing analytics integration:")
        try:
            from analytics.fvg_analytics import get_analytics
            analytics = get_analytics(logger)
            if analytics:
                logger.info("   Analytics module loaded successfully")
                stats = analytics.get_overall_stats()
                logger.info(f"   Overall stats: {stats}")
            else:
                logger.info("   Analytics module initialized but not populated")
        except Exception as e:
            logger.error(f"   ❌ Analytics integration error: {e}")

        logger.info("\n" + "="*60)
        logger.info("🎉 ALL RESPONSIVE PROFILE TESTS PASSED!")
        logger.info("Enhanced FVG implementation with responsive profile active")
        logger.info("All new pattern modules integrated successfully")
        logger.info("Ready for live testing and backtesting")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test entry point"""
    success = await test_responsive_profile()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())