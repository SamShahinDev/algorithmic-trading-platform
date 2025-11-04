#!/usr/bin/env python3
"""
Test script to verify the trading bot system
Run this to make sure everything is working
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.data_fetcher import DataFetcher
from agents.pattern_discovery import PatternDiscoveryAgent
from agents.pattern_library import PatternLibraryManager

async def test_data_fetcher():
    """Test data fetching functionality"""
    print("\n" + "="*50)
    print("🧪 Testing Data Fetcher...")
    print("="*50)
    
    fetcher = DataFetcher()
    
    # Try to fetch data
    print("Fetching NQ futures data (this may take a moment)...")
    data = await fetcher.get_latest_data(period='1mo', interval='1h')
    
    if not data.empty:
        print(f"✅ Successfully fetched {len(data)} bars of data")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Latest price: ${data['Close'].iloc[-1]:,.2f}")
        
        # Test support/resistance calculation
        levels = fetcher.get_support_resistance_levels()
        print(f"   Support levels: {levels['support'][:3]}")
        print(f"   Resistance levels: {levels['resistance'][:3]}")
        
        return True
    else:
        print("❌ Failed to fetch data")
        return False

async def test_pattern_discovery(data):
    """Test pattern discovery"""
    print("\n" + "="*50)
    print("🔍 Testing Pattern Discovery...")
    print("="*50)
    
    discovery = PatternDiscoveryAgent()
    
    # Initialize agent
    if not await discovery.initialize():
        print("❌ Failed to initialize Pattern Discovery")
        return False
    
    # Discover patterns
    print("Searching for patterns...")
    patterns = await discovery.discover_patterns(data)
    
    if patterns:
        print(f"✅ Found {len(patterns)} patterns:")
        for pattern in patterns[:5]:  # Show first 5
            print(f"   - {pattern.get('name', 'Unknown')}")
            print(f"     Type: {pattern.get('type', 'unknown')}")
            print(f"     Confidence: {pattern.get('confidence', 0):.1%}")
        return True
    else:
        print("⚠️ No patterns found (this may be normal)")
        return True

def test_pattern_library():
    """Test pattern library"""
    print("\n" + "="*50)
    print("📚 Testing Pattern Library...")
    print("="*50)
    
    library = PatternLibraryManager()
    
    # Test adding a pattern
    test_pattern = {
        'name': 'Test Pattern',
        'type': 'test',
        'entry_conditions': {'test': True}
    }
    
    test_stats = {
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'sample_size': 50
    }
    
    pattern_id = library.add_pattern(test_pattern, test_stats)
    
    if pattern_id:
        print(f"✅ Successfully added pattern with ID: {pattern_id}")
        
        # Test retrieval
        retrieved = library.get_pattern(pattern_id)
        if retrieved:
            print(f"✅ Successfully retrieved pattern: {retrieved['name']}")
        
        # Test statistics
        stats = library.get_statistics()
        print(f"   Total patterns in library: {stats['total_patterns']}")
        
        return True
    else:
        print("❌ Failed to add pattern")
        return False

def test_logging():
    """Test logging system"""
    print("\n" + "="*50)
    print("📝 Testing Logging System...")
    print("="*50)
    
    logger = setup_logger('TestLogger')
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    print("✅ Logging system working (check output above)")
    return True

async def main():
    """Main test function"""
    print("\n" + "🚀 TRADING BOT SYSTEM TEST 🚀")
    print("=" * 60)
    print("This will test all major components of your trading bot")
    print("=" * 60)
    
    results = []
    
    # Test logging
    results.append(("Logging", test_logging()))
    
    # Test data fetching
    data = None
    if await test_data_fetcher():
        results.append(("Data Fetcher", True))
        
        # Get data for pattern discovery
        fetcher = DataFetcher()
        data = await fetcher.get_latest_data(period='3mo', interval='1h')
    else:
        results.append(("Data Fetcher", False))
    
    # Test pattern discovery (only if we have data)
    if data is not None and not data.empty:
        success = await test_pattern_discovery(data)
        results.append(("Pattern Discovery", success))
    
    # Test pattern library
    results.append(("Pattern Library", test_pattern_library()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for component, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your bot is ready to go!")
        print("\nNext steps:")
        print("1. Add your TopStep API key to a .env file")
        print("2. Run: python main_orchestrator.py")
        print("3. Monitor the output and watch for patterns!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- No internet connection (for data fetching)")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
    
    return all_passed

if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)