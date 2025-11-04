#!/usr/bin/env python3
"""
Test Scalping Pattern Discovery
Find patterns that can make $100 per trade with 1 NQ contract
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.scalping_patterns import ScalpingPatternAgent
from config import SCALPING_CONFIG
from utils.logger import setup_logger

async def test_scalping():
    """Test scalping pattern discovery"""
    
    logger = setup_logger('ScalpingTest')
    
    print("\n" + "="*60)
    print("💰 SCALPING PATTERN DISCOVERY TEST")
    print("="*60)
    print(f"Target: ${SCALPING_CONFIG['target_profit']} per trade ({SCALPING_CONFIG['points_target']} NQ points)")
    print(f"Risk: ${SCALPING_CONFIG['stop_loss']} max loss")
    print(f"Timeframe: {SCALPING_CONFIG['timeframe']}")
    print(f"Min Win Rate: {SCALPING_CONFIG['min_win_rate']:.0%}")
    print("-"*60)
    
    # Initialize scalping agent
    print("\n🚀 Initializing Scalping Pattern Agent...")
    scalper = ScalpingPatternAgent()
    
    # Discover patterns
    print("\n🔍 Searching for high-probability scalping patterns...")
    print("This analyzes 5-minute bars to find quick profit opportunities...")
    
    patterns = await scalper.discover_scalping_patterns('NQ')
    
    if patterns:
        print(f"\n✅ Found {len(patterns)} scalping patterns!\n")
        
        for i, pattern in enumerate(patterns, 1):
            print(f"{i}. {pattern['name']}")
            print(f"   📊 Confidence: {pattern['confidence']:.1%}")
            
            stats = pattern.get('statistics', {})
            print(f"   📈 Win Rate: {stats.get('win_rate', 0):.1%}")
            print(f"   🎯 Occurrences: {stats.get('occurrences', 0)}")
            
            # Expected value
            ev = pattern.get('expected_value', 0)
            if ev > 0:
                print(f"   💵 Expected Value: ${ev:.2f} per trade")
                
                # Daily projection
                trades_per_day = min(10, stats.get('occurrences', 0) // 5)  # Rough estimate
                daily_profit = ev * trades_per_day
                print(f"   📅 Potential Daily: ${daily_profit:.2f} ({trades_per_day} trades)")
            
            if pattern.get('recommended'):
                print(f"   ✅ RECOMMENDED for trading")
            else:
                print(f"   ⚠️ Not recommended (EV too low)")
            
            # Entry rules
            entry = pattern.get('entry_rules', {})
            print(f"\n   Entry Rules:")
            for key, value in entry.items():
                print(f"     - {key}: {value}")
            
            # Exit rules
            exit_rules = pattern.get('exit_rules', {})
            print(f"   Exit Rules:")
            for key, value in exit_rules.items():
                print(f"     - {key}: {value}")
            
            print("-"*60)
    else:
        print("\n⚠️ No scalping patterns found meeting criteria.")
        print("This could mean:")
        print("  1. Market conditions aren't favorable for scalping")
        print("  2. Need more data to identify patterns")
        print("  3. Criteria too strict (55% win rate minimum)")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SCALPING ANALYSIS SUMMARY")
    print("-"*60)
    
    if patterns:
        # Best pattern
        best_pattern = patterns[0] if patterns else None
        if best_pattern:
            print(f"🏆 Best Pattern: {best_pattern['name']}")
            print(f"   Expected Profit: ${best_pattern.get('expected_value', 0):.2f}/trade")
            
            # Calculate potential monthly income
            trades_per_day = 4  # Conservative estimate
            trading_days = 20  # Month
            monthly = best_pattern.get('expected_value', 0) * trades_per_day * trading_days
            
            print(f"\n💰 Potential Monthly Income:")
            print(f"   Conservative (4 trades/day): ${monthly:,.2f}")
            print(f"   Moderate (6 trades/day): ${monthly * 1.5:,.2f}")
            print(f"   Aggressive (10 trades/day): ${monthly * 2.5:,.2f}")
            
            print(f"\n⚠️ Important Notes:")
            print(f"   • Requires discipline to follow rules exactly")
            print(f"   • Must respect stop losses (no revenge trading)")
            print(f"   • Account for slippage and commissions")
            print(f"   • Start with sim trading to verify patterns")
    
    print("="*60)

async def main():
    """Main runner"""
    print("\n🎯 Scalping Strategy Analyzer")
    print("Finding patterns to make $100 every 15 minutes with 1 NQ contract")
    
    await test_scalping()
    
    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("1. Review the patterns found")
    print("2. Backtest with historical data")
    print("3. Paper trade to verify win rates")
    print("4. Start with smallest position size")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()