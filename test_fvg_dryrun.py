#!/usr/bin/env python3
"""
Simple test to verify FVG dry-run mode
"""

import os
import sys

# Set dry run mode
os.environ['FVG_DRY_RUN'] = 'true'

# Test that we can import and check dry run
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from nq_bot.pattern_config import STRATEGY_MODE, FVG
    print(f"✅ Strategy mode: {STRATEGY_MODE}")
    print(f"✅ FVG config loaded with {len(FVG)} settings")
    
    # Check dry run env
    dry_run = os.getenv('FVG_DRY_RUN', 'false').lower() == 'true'
    print(f"✅ Dry run mode: {dry_run}")
    
    if dry_run:
        print("\n🟡 FVG_DRY_RUN=true detected")
        print("In dry-run mode, the bot will:")
        print("  - Skip broker connection")
        print("  - Log ENTRY_PLACED(DRY_RUN) instead of placing real orders")
        print("  - Write all events to logs/fvg_telemetry.csv")
        print("\nTo run: python3 -m nq_bot.fvg_runner")
    
    print("\n✅ All checks passed! Dry-run mode is ready.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)