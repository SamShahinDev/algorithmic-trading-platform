#!/usr/bin/env python3
"""
Test Pattern Discovery directly
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.pattern_discovery import PatternDiscoveryAgent
from agents.backtest_validator import BacktestValidationAgent
from agents.pattern_library import PatternLibraryManager
from utils.logger import setup_logger

async def test_discovery():
    """Test pattern discovery directly"""
    
    logger = setup_logger('PatternTest')
    
    print("\n" + "="*60)
    print("🔍 TESTING PATTERN DISCOVERY")
    print("="*60)
    
    # Initialize agents
    print("\n📚 Initializing agents...")
    discovery = PatternDiscoveryAgent()
    validator = BacktestValidationAgent()
    library = PatternLibraryManager()
    
    print("✅ Agents initialized")
    
    # Run discovery
    print("\n🔍 Running pattern discovery on NQ...")
    print("This may take a few minutes...")
    
    try:
        patterns = await discovery.discover_patterns('NQ')
        
        if patterns:
            print(f"\n✅ Found {len(patterns)} patterns!")
            
            for i, pattern in enumerate(patterns[:5], 1):
                print(f"\n{i}. {pattern.get('name', 'Unknown Pattern')}")
                print(f"   Type: {pattern.get('type')}")
                print(f"   Confidence: {pattern.get('confidence', 0):.2%}")
                
                # Validate pattern
                print(f"   Validating...")
                validation = await validator.validate(pattern)
                
                if validation and validation.get('is_valid'):
                    stats = validation.get('statistics', {})
                    print(f"   ✅ VALID!")
                    print(f"   Win Rate: {stats.get('win_rate', 0):.2%}")
                    print(f"   Profit Factor: {stats.get('profit_factor', 0):.2f}")
                    print(f"   Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
                    
                    # Store in library
                    stored = await library.add_pattern(pattern)
                    if stored:
                        print(f"   ✅ Stored in library")
                else:
                    print(f"   ❌ Not valid (didn't pass criteria)")
        else:
            print("\n⚠️ No patterns found yet. This is normal - pattern discovery")
            print("can take time to identify statistically significant patterns.")
            
    except Exception as e:
        print(f"\n❌ Error during discovery: {e}")
        import traceback
        traceback.print_exc()
    
    # Check library
    print("\n📚 Checking pattern library...")
    active = await library.get_active_patterns()
    print(f"Active patterns in library: {len(active)}")
    
    if active:
        print("\nTop patterns by confidence:")
        for p in active[:3]:
            print(f"  - {p['name']}: {p['confidence']:.2%} confidence")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    print("🚀 Starting Pattern Discovery Test")
    print("This will analyze NQ data to find trading patterns")
    
    try:
        asyncio.run(test_discovery())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()