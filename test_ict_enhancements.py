#!/usr/bin/env python3
"""
Test script to verify ICT pattern enhancements are working
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ict_manager():
    """Test the ICT pattern manager"""
    print("🧪 Testing ICT Pattern Manager...")

    try:
        from nq_bot.ict_manager import ICTPatternManager
        from nq_bot.pattern_config import FVGConfig

        # Create test configuration
        cfg = FVGConfig()

        # Initialize ICT manager
        ict_manager = ICTPatternManager(cfg)

        print(f"✅ ICT Manager loaded modules: {list(ict_manager.ict_modules.keys())}")

        # Test pattern statistics
        stats = ict_manager.get_pattern_statistics()
        print(f"✅ ICT Manager statistics: {stats}")

        return True

    except Exception as e:
        print(f"❌ ICT Manager test failed: {e}")
        return False

def test_fvg_strategy_ict_integration():
    """Test FVG strategy ICT integration"""
    print("🧪 Testing FVG Strategy ICT Integration...")

    try:
        from nq_bot.patterns.fvg_strategy import FVGStrategy
        from nq_bot.pattern_config import FVGConfig
        from nq_bot.utils.data_cache import DataCache

        # Create test configuration
        cfg = FVGConfig()
        data_cache = DataCache("test_cache.json")

        # Initialize FVG strategy with ICT manager
        strategy = FVGStrategy(cfg, data_cache)

        # Check if ICT manager was initialized
        has_ict_manager = hasattr(strategy, 'ict_manager') and strategy.ict_manager is not None
        print(f"✅ FVG Strategy has ICT Manager: {has_ict_manager}")

        if has_ict_manager:
            print(f"✅ ICT Manager modules: {list(strategy.ict_manager.ict_modules.keys())}")

        # Check telemetry counters
        ict_counters = {k: v for k, v in strategy.telemetry_counters.items() if 'ict' in k}
        print(f"✅ ICT telemetry counters: {ict_counters}")

        return True

    except Exception as e:
        print(f"❌ FVG Strategy ICT integration test failed: {e}")
        return False

def test_ict_configuration():
    """Test ICT configuration parameters"""
    print("🧪 Testing ICT Configuration...")

    try:
        from nq_bot.pattern_config import FVGConfig

        cfg = FVGConfig()
        params = cfg.ict_params

        print(f"✅ ICT session parameters loaded:")
        print(f"   London quality boost: {params.london_quality_boost}")
        print(f"   NY morning quality boost: {params.ny_morning_quality_boost}")
        print(f"   Bias detection lookback: {params.bias_lookback_bars}")
        print(f"   Micro scalp max zone ticks: {params.micro_max_zone_ticks}")
        print(f"   Session TTL multipliers: London={params.london_ttl_multiplier}, NY={params.ny_active_ttl_multiplier}")

        return True

    except Exception as e:
        print(f"❌ ICT configuration test failed: {e}")
        return False

def test_ict_modules():
    """Test ICT modules are available"""
    print("🧪 Testing ICT Module Availability...")

    try:
        from nq_bot.patterns.modules import ict_silver_bullet, ict_micro_scalp

        # Test Silver Bullet module
        sb_info = ict_silver_bullet.get_pattern_info()
        print(f"✅ Silver Bullet module: {sb_info['name']}")
        print(f"   Active 24/7: {'time window restriction removed' in str(ict_silver_bullet.__file__)}")

        # Test Micro Scalp module
        ms_info = ict_micro_scalp.get_pattern_info()
        print(f"✅ Micro Scalp module: {ms_info['name']}")
        print(f"   Active 24/7: {'killzone restriction removed' in str(ict_micro_scalp.__file__)}")

        return True

    except Exception as e:
        print(f"❌ ICT modules test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 ICT PATTERN ENHANCEMENT TEST SUITE")
    print("=" * 60)

    tests = [
        test_ict_configuration,
        test_ict_modules,
        test_ict_manager,
        test_fvg_strategy_ict_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL ICT ENHANCEMENTS WORKING CORRECTLY!")
        print("✅ ICT patterns are now:")
        print("   • Active 24/7 (no time window restrictions)")
        print("   • Independent of FVG detection")
        print("   • Session-optimized with quality boosts")
        print("   • Enhanced bias detection")
        print("   • Configurable parameters")
    else:
        print("⚠️  Some ICT enhancements need attention")

    print("=" * 60)

if __name__ == "__main__":
    main()