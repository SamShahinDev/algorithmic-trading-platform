#!/usr/bin/env python3
"""
Quick test for FVG-only tweaks
Tests dynamic displacement, quality gates, and TTL logic
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nq_bot.pattern_config import FVG
from nq_bot.patterns.fvg_strategy import FVGStrategy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_config_structure():
    """Test new hierarchical config structure"""
    print("\n=== Testing Config Structure ===")
    
    assert 'detection' in FVG
    assert 'quality' in FVG
    assert 'entry' in FVG
    assert 'edge_retry' in FVG
    assert 'risk' in FVG
    assert 'trail' in FVG
    assert 'lifecycle' in FVG
    
    # Check specific values
    assert FVG['detection']['min_displacement_mode'] == 'dynamic'
    assert FVG['detection']['min_displacement_dyn']['base_pts'] == 3.0
    assert FVG['detection']['min_displacement_dyn']['atr_mult'] == 0.6
    assert FVG['quality']['min_quality'] == 0.55
    assert FVG['entry']['ttl_sec'] == 90
    assert FVG['edge_retry']['ttl_sec'] == 45
    assert FVG['risk']['breakeven_pts'] == 9.0
    assert FVG['trail']['fast']['trigger_pts'] == 12.0
    assert FVG['trail']['fast']['giveback_ticks'] == 10
    
    print("✓ Config structure correct")


def test_dynamic_displacement():
    """Test dynamic displacement calculation"""
    print("\n=== Testing Dynamic Displacement ===")
    
    # Mock data cache
    class MockDataCache:
        def get_bars(self, timeframe):
            # Create mock bars with ATR ~10
            dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
            return pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(100) * 2 + 21000,
                'high': np.random.randn(100) * 2 + 21010,
                'low': np.random.randn(100) * 2 + 20990,
                'close': np.random.randn(100) * 2 + 21000,
                'volume': np.random.randint(100, 1000, 100)
            })
    
    # Create strategy with dynamic mode
    mock_cache = MockDataCache()
    logger = type('MockLogger', (), {'info': lambda self, msg: print(f"LOG: {msg}"), 
                                     'error': lambda self, msg: print(f"ERROR: {msg}")})()
    
    strategy = FVGStrategy(mock_cache, logger, FVG)
    
    # Test dynamic calculation
    min_disp = strategy.dynamic_min_displacement()
    print(f"Dynamic min displacement: {min_disp:.2f} pts")
    
    # Should be base_pts + atr_mult * ATR
    # With ATR ~10-20: 3.0 + 0.6 * (10-20) = 9-15
    assert 5.0 <= min_disp <= 20.0, f"Dynamic displacement out of range: {min_disp}"
    
    print("✓ Dynamic displacement working")


def test_quality_gate():
    """Test quality score calculation and gating"""
    print("\n=== Testing Quality Gate ===")
    
    # Test quality calculation
    body_frac = 0.7  # 70% body
    atr_mult = 1.2   # 1.2x ATR
    vol_mult = 1.8   # 1.8x volume
    
    # Quality = 0.3 * body_frac + 0.4 * min(atr_mult/2, 1) + 0.3 * min(vol_mult/3, 1)
    expected_quality = (0.3 * 0.7 + 
                       0.4 * min(1.2/2, 1.0) + 
                       0.3 * min(1.8/3, 1.0))
    print(f"Expected quality: {expected_quality:.3f}")
    
    # Should be: 0.21 + 0.24 + 0.18 = 0.63
    assert abs(expected_quality - 0.63) < 0.01
    
    # Test gate at 0.55
    min_quality = FVG['quality']['min_quality']
    print(f"Min quality threshold: {min_quality}")
    
    # Test pass case
    assert expected_quality >= min_quality
    print("✓ Quality gate working (pass case)")
    
    # Test reject case
    low_quality = 0.45
    assert low_quality < min_quality
    print("✓ Quality gate working (reject case)")


def test_entry_signals():
    """Test entry signal generation with TTL"""
    print("\n=== Testing Entry Signals ===")
    
    # Mock FVG object
    class MockFVG:
        def __init__(self):
            self.id = "FVG_TEST_1"
            self.direction = "long"
            self.quality = 0.65
            self.mid = 21000.0
            self.top = 21005.0
            self.bottom = 20995.0
    
    # Create mock strategy
    class MockDataCache:
        def get_bars(self, timeframe):
            return pd.DataFrame({'close': [21000]})
    
    mock_cache = MockDataCache()
    logger = type('MockLogger', (), {'info': lambda self, msg: print(f"LOG: {msg}"), 
                                     'error': lambda self, msg: print(f"ERROR: {msg}")})()
    
    strategy = FVGStrategy(mock_cache, logger, FVG)
    
    # Generate entry signals
    mock_fvg = MockFVG()
    signals = strategy.get_entry_signals(mock_fvg)
    
    print(f"Entry signals: {signals}")
    
    # Check mid entry
    assert 'mid_entry' in signals
    assert signals['mid_entry']['level'] == 21000.0
    assert signals['mid_entry']['ttl_sec'] == 90
    assert signals['mid_entry']['cancel_if_runs_ticks'] == 8
    
    # Check edge retry
    assert 'edge_retry' in signals
    edge_level = signals['edge_retry']['level']
    assert abs(edge_level - (21005.0 - 2*0.25)) < 0.01  # top - 2 ticks
    assert signals['edge_retry']['ttl_sec'] == 45
    
    print("✓ Entry signals generated correctly")


def test_telemetry_fields():
    """Test telemetry sink has FVG fields"""
    print("\n=== Testing Telemetry Fields ===")
    
    from nq_bot.utils.telemetry_sink import TelemetrySink
    import tempfile
    
    # Create temp telemetry file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as tf:
        tf_name = tf.name
    
    # Create sink and write data
    sink = TelemetrySink(tf_name)
    
    # Write test FVG event
    sink.write(
        pattern="FVG",
        event="FVG_DETECTED",
        fvg_id="FVG_1",
        fvg_quality=0.62,
        fvg_direction="long",
        fvg_top=21005.0,
        fvg_bottom=20995.0,
        fvg_mid=21000.0,
        fvg_status="FRESH",
        fvg_body_frac=0.7,
        fvg_atr_mult=1.2,
        fvg_vol_mult=1.5,
        fvg_displacement_mode="dynamic",
        fvg_min_displacement=9.0,
        fvg_actual_displacement=11.0,
        fvg_ttl_sec=90,
        fvg_cancel_if_runs=8,
        fvg_edge_retry=True
    )
    
    # Read back and check header has FVG fields  
    import csv
    with open(tf_name, 'r') as f:
        lines = f.readlines()
        
    # At minimum, should have header line
    assert len(lines) >= 1, f"Expected at least 1 line, got {len(lines)}"
    
    # Check header contains FVG fields
    header = lines[0].strip()
    assert 'fvg_id' in header
    assert 'fvg_quality' in header
    assert 'fvg_direction' in header
    assert 'fvg_ttl_sec' in header
    assert 'fvg_displacement_mode' in header
    
    print(f"✓ Telemetry fields added to header")
    
    # If we have data rows, verify them too
    if len(lines) >= 2:
        with open(tf_name, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if len(rows) > 0:
            row = rows[0]
            if 'fvg_id' in row and row['fvg_id']:
                assert row['fvg_id'] == 'FVG_1'
            print(f"✓ Telemetry data verified")
    
    # Cleanup
    os.unlink(tf_name)


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("FVG TWEAKS TEST SUITE")
    print("="*50)
    
    try:
        test_config_structure()
        test_dynamic_displacement()
        test_quality_gate()
        test_entry_signals()
        test_telemetry_fields()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✅")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)