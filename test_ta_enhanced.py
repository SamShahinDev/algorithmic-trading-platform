#!/usr/bin/env python3
"""Test the enhanced technical analysis scoring system"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from nq_bot.utils.technical_analysis import TechnicalAnalysisFallback
from nq_bot.utils.data_cache import DataCache

def create_test_data():
    """Create test market data"""
    # Create sample price data
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=1440, freq='1min')
    
    # Create trending price data
    base_price = 20000
    trend = np.linspace(0, 100, 1440)
    noise = np.random.randn(1440) * 5
    prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.rand(1440) * 2,
        'high': prices + np.random.rand(1440) * 3,
        'low': prices - np.random.rand(1440) * 3,
        'close': prices,
        'volume': np.random.randint(100, 1000, 1440)
    })
    
    return df

def test_scoring_system():
    """Test the TA scoring system"""
    print("Testing Enhanced Technical Analysis Scoring System")
    print("=" * 60)
    
    # Create mock broker
    class MockBroker:
        def get_bars(self, *args, **kwargs):
            return create_test_data()
    
    # Create data cache with mock broker
    data_cache = DataCache(MockBroker())
    
    # Create test data
    df = create_test_data()
    current_price = df['close'].iloc[-1]
    
    # Update data cache with test data
    data_cache.update_cache(df)
    
    # Initialize TA with data cache
    ta = TechnicalAnalysisFallback(data_cache)
    
    # Test 1: Basic analysis
    print("\n1. Testing basic analysis:")
    signal = ta.analyze(df, current_price)
    
    if signal:
        print(f"   - Action: {signal['action']}")
        print(f"   - Score: {signal['score']:.1f}")
        print(f"   - Confidence: {signal['confidence']:.2f}")
        print(f"   - Stop: ${signal['stop_price']:.2f}")
        print(f"   - Target: ${signal['target_price']:.2f}")
        print(f"   - Score components: {signal.get('score_components', {})}")
    else:
        print("   - No signal generated (score < 7.5)")
    
    # Test 2: Score calculation details
    print("\n2. Testing score calculation:")
    
    # MA Trend (2 points)
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma50 = df['close'].rolling(50).mean().iloc[-1]
    ma_aligned = (current_price > ma20 > ma50) or (current_price < ma20 < ma50)
    print(f"   - MA Trend aligned: {ma_aligned} ({2 if ma_aligned else 0} points)")
    
    # ADX (1-2 points)
    adx_value = data_cache.get_adx()
    adx_points = 0
    if adx_value:
        if 18 <= adx_value < 30:
            adx_points = 1
        elif adx_value >= 30:
            adx_points = 2
    print(f"   - ADX: {adx_value:.1f if adx_value else 'N/A'} ({adx_points} points)")
    
    # RSI (1-2 points)
    rsi_value = data_cache.get_rsi()
    print(f"   - RSI: {rsi_value:.1f if rsi_value else 'N/A'}")
    
    # Total score calculation
    print(f"\n   Total score must be >= 7.5 to generate signal")
    
    # Test 3: Hourly limit
    print("\n3. Testing hourly trade limit:")
    
    # Try to generate another signal
    signal2 = ta.analyze(df, current_price)
    if signal2:
        print("   - Second signal generated (should not happen within hour)")
    else:
        print("   - Second signal blocked (hourly limit working)")
    
    # Test 4: Confirmation close
    print("\n4. Testing confirmation close requirement:")
    if signal and 'needs_confirmation' in signal:
        print(f"   - Needs confirmation: {signal['needs_confirmation']}")
        print(f"   - Confirmation level: ${signal.get('confirmation_level', 0):.2f}")
    
    # Test 5: Level confluence
    print("\n5. Testing level confluence:")
    
    # Set some key levels
    data_cache.indicators['vwap'] = current_price - 1.25  # 5 ticks away
    data_cache.indicators['poc'] = current_price + 1.0    # 4 ticks away
    
    print(f"   - VWAP: ${data_cache.indicators['vwap']:.2f} ({abs(current_price - data_cache.indicators['vwap'])/0.25:.0f} ticks)")
    print(f"   - POC: ${data_cache.indicators['poc']:.2f} ({abs(current_price - data_cache.indicators['poc'])/0.25:.0f} ticks)")
    
    # Re-analyze with levels
    signal3 = ta.analyze(df, current_price)
    if signal3 and 'score_components' in signal3:
        level_points = signal3['score_components'].get('level_confluence', 0)
        print(f"   - Level confluence points: {level_points}")

def test_state_persistence():
    """Test state persistence"""
    print("\n" + "=" * 60)
    print("Testing State Persistence")
    print("=" * 60)
    
    # Create mock broker
    class MockBroker:
        def get_bars(self, *args, **kwargs):
            return create_test_data()
    
    # Create data cache with mock broker
    data_cache = DataCache(MockBroker())
    
    # Initialize TA
    ta = TechnicalAnalysisFallback(data_cache)
    
    # Check state loading
    print("\n1. Checking state persistence:")
    print(f"   - State file: ta_trades.json")
    print(f"   - Last TA trade time: {ta.last_ta_trade_time}")
    print(f"   - Can trade now: {ta._can_trade_hourly_limit()}")

if __name__ == "__main__":
    test_scoring_system()
    test_state_persistence()
    
    print("\n" + "=" * 60)
    print("Technical Analysis Enhancement Test Complete")
    print("=" * 60)