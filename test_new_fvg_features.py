#!/usr/bin/env python3
"""Test script to verify new FVG features"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nq_bot'))

from nq_bot.pattern_config import FVG
import json

print("=" * 60)
print("NEW FVG CONFIGURATION SUMMARY")
print("=" * 60)
print()

print("🔄 TREND FVG SUPPORT:")
print(f"  - Allow Trend FVGs (no sweep): {FVG.get('allow_trend_fvgs')}")
print(f"  - Sweep overshoot: {FVG.get('sweep_min_overshoot_ticks')} tick(s)")
print(f"  - Min gap size: {FVG.get('min_gap_ticks')} tick(s)")
print()

print("📊 HIGH VOLATILITY DETECTION:")
high_vol = FVG.get('high_vol', {})
print(f"  - ATR ratio threshold: {high_vol.get('atr_ratio')}")
print(f"  - Volume ratio threshold: {high_vol.get('vol_ratio')}")
print(f"  - ATR periods: fast={high_vol.get('atr_fast')}, slow={high_vol.get('atr_slow')}")
print(f"  - Volume periods: fast={high_vol.get('vol_fast')}, slow={high_vol.get('vol_slow')}")
print()

print("🎯 DYNAMIC ENTRY LEVELS:")
entry = FVG.get('entry', {})
print(f"  - Normal volatility: {entry.get('entry_pct_default', 0.50):.0%} of zone")
print(f"  - High volatility: {entry.get('entry_pct_high_vol', 0.62):.0%} of zone")
print()

print("🛡️ DISPLACEMENT REQUIREMENTS:")
detection = FVG.get('detection', {})
print(f"  - Normal body fraction: {detection.get('min_body_frac'):.0%}")
print(f"  - High vol body fraction: {detection.get('min_body_frac_high_vol'):.0%}")
print()

print("⚔️ ARMING/DEFENSE:")
lifecycle = FVG.get('lifecycle', {})
print(f"  - Max zone consumption: {lifecycle.get('invalidate_frac'):.0%}")
print(f"  - Defense line: {100 - lifecycle.get('touch_defend_inner_frac', 0.10) * 100:.0f}% of zone")
print()

print("📈 RSI FILTERS:")
rsi = FVG.get('rsi', {})
print(f"  Normal ranges:")
print(f"    - Long: {rsi.get('long_range')}")
print(f"    - Short: {rsi.get('short_range')}")
print(f"  RTH open ranges (first {rsi.get('rth_open_relax_minutes')} min):")
print(f"    - Long: {rsi.get('long_range_rth')}")
print(f"    - Short: {rsi.get('short_range_rth')}")
print()

print("✅ All new features configured!")
print()

# Verify key changes
changes_verified = []

if FVG.get('allow_trend_fvgs') == True:
    changes_verified.append("✓ Trend FVGs enabled")

if FVG.get('sweep_min_overshoot_ticks') == 1:
    changes_verified.append("✓ Sweep overshoot reduced to 1 tick")

if lifecycle.get('invalidate_frac') == 0.90:
    changes_verified.append("✓ Defense increased to 90%")

if entry.get('entry_pct_high_vol') == 0.62:
    changes_verified.append("✓ High vol entry at 62%")

if detection.get('min_body_frac_high_vol') == 0.52:
    changes_verified.append("✓ High vol body fraction reduced to 52%")

if FVG.get('min_gap_ticks') == 1:
    changes_verified.append("✓ Min gap reduced to 1 tick")

if rsi.get('long_range_rth') == [45, 85]:
    changes_verified.append("✓ RTH RSI ranges relaxed")

print("VERIFICATION:")
for check in changes_verified:
    print(f"  {check}")
print()

if len(changes_verified) == 7:
    print("🎉 ALL CHANGES SUCCESSFULLY IMPLEMENTED!")
else:
    print(f"⚠️  {len(changes_verified)}/7 changes verified")