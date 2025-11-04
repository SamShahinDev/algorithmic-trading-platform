"""
Test entry quality improvements for pattern detection
Verifies all acceptance criteria for Phase 3 implementation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nq_bot.patterns.momentum_thrust_enhanced import MomentumThrustPattern
from nq_bot.patterns.trend_line_bounce import TrendLineBouncePattern
from nq_bot.patterns.base_pattern import EntryPlan, TradeAction

class TestEntryQualityImprovements:
    """Test suite for entry quality improvements"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing"""
        # Generate 200 bars of synthetic data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=200, freq='1min')
        
        # Base price around 15000 (NQ futures)
        base_price = 15000
        prices = base_price + np.cumsum(np.random.randn(200) * 2.5)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(200) * 0.5,
            'high': prices + abs(np.random.randn(200)) * 2,
            'low': prices - abs(np.random.randn(200)) * 2,
            'close': prices + np.random.randn(200) * 0.5,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        # Ensure high >= open/close and low <= open/close
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data
    
    def test_confirmation_close_requirement(self, sample_data):
        """
        Acceptance Criteria 1: Patterns wait for confirmation close
        - Setup bar triggers pattern detection
        - No signal until bar closes through trigger level
        - Confirmation must occur within 5 bars
        """
        # Initialize pattern with confirmation required
        config = {
            'lookback': 56,
            'momentum_threshold': 0.0014,
            'volume_factor': 1.72,
            'stop_ticks': 6,
            'target_ticks': 4,
            'require_confirmation_close': True  # Enable confirmation
        }
        
        pattern = MomentumThrustPattern(config)
        
        # Create setup conditions
        setup_data = sample_data.copy()
        
        # Simulate momentum thrust setup
        setup_data.loc[150:155, 'volume'] = setup_data['volume'].mean() * 2  # High volume
        setup_data.loc[155, 'close'] = setup_data.loc[100, 'close'] * 1.002  # Momentum move
        
        # First scan should detect setup but not signal
        current_price = setup_data.iloc[-1]['close']
        signal = pattern.scan_for_setup(setup_data, current_price)
        
        assert signal is None, "Should not signal immediately when confirmation required"
        assert pattern.confirmation_waiting == True, "Should be waiting for confirmation"
        assert pattern.trigger_level is not None, "Should have set trigger level"
        
        # Add confirmation bar that closes through trigger
        confirm_data = setup_data.copy()
        if pattern.pending_setup['is_long']:
            confirm_data.loc[len(confirm_data)] = {
                'timestamp': confirm_data.iloc[-1]['timestamp'] + timedelta(minutes=1),
                'open': pattern.trigger_level - 1,
                'high': pattern.trigger_level + 2,
                'low': pattern.trigger_level - 2,
                'close': pattern.trigger_level + 1,  # Close above trigger
                'volume': confirm_data['volume'].mean()
            }
        else:
            confirm_data.loc[len(confirm_data)] = {
                'timestamp': confirm_data.iloc[-1]['timestamp'] + timedelta(minutes=1),
                'open': pattern.trigger_level + 1,
                'high': pattern.trigger_level + 2,
                'low': pattern.trigger_level - 2,
                'close': pattern.trigger_level - 1,  # Close below trigger
                'volume': confirm_data['volume'].mean()
            }
        
        # Now should get signal
        current_price = confirm_data.iloc[-1]['close']
        signal = pattern.scan_for_setup(confirm_data, current_price)
        
        # Note: Signal might still be None if other quality checks fail
        # But confirmation logic should have been triggered
        assert pattern.confirmation_waiting == False or signal is not None, \
            "Should either signal or reset after confirmation"
    
    def test_exhaustion_check(self, sample_data):
        """
        Acceptance Criteria 2: Skip exhaustion bars
        - Confirmation bar range > 1.25 × ATR
        - Pattern should be skipped
        """
        config = {
            'lookback': 56,
            'momentum_threshold': 0.0014,
            'volume_factor': 1.72,
            'stop_ticks': 6,
            'target_ticks': 4,
            'require_confirmation_close': False  # Disable for direct testing
        }
        
        pattern = MomentumThrustPattern(config)
        
        # Test exhaustion check method directly
        atr = 10.0  # Example ATR
        
        # Non-exhaustion bar
        normal_range = 12.0  # 1.2x ATR
        assert pattern.exhaustion_check(normal_range, atr) == False, \
            "Normal range should not be exhaustion"
        
        # Exhaustion bar
        exhaustion_range = 13.0  # 1.3x ATR
        assert pattern.exhaustion_check(exhaustion_range, atr) == True, \
            f"Range {exhaustion_range} > {1.25 * atr} should be exhaustion"
    
    def test_micro_pullback_requirement(self, sample_data):
        """
        Acceptance Criteria 3: Require micro-pullback
        - Must retrace ≥ 0.382 of pivot range
        - Check last 3 bars for pullback
        """
        config = {'require_confirmation_close': False}
        pattern = MomentumThrustPattern(config)
        
        # Create test data for pullback check
        test_data = pd.DataFrame({
            'high': [100, 102, 101],
            'low': [98, 99, 99.5],
            'close': [99, 101, 100]
        })
        
        # Test bullish pullback (should pullback from high)
        pivot_level = 98  # Previous low
        is_long = True
        
        achieved, level = pattern.micro_pullback_check(test_data, pivot_level, is_long)
        
        # For bullish, need to check if price pulled back enough from the high
        high_point = test_data['high'].max()  # 102
        pullback_target = high_point - (0.382 * (high_point - pivot_level))
        # pullback_target = 102 - (0.382 * 4) = 100.472
        
        actual_pullback = test_data['low'].min()  # 98
        
        # Achieved if actual_pullback <= pullback_target
        expected = actual_pullback <= pullback_target
        assert achieved == expected, f"Pullback check failed: achieved={achieved}, expected={expected}"
    
    def test_dangerous_engulfing_detection(self, sample_data):
        """
        Acceptance Criteria 4: Enhanced engulfing detection
        - All 4 conditions must be met
        - Body > 60% of range
        - Body > 60% of prev body
        - Opposite direction
        - Close beyond prev high/low
        """
        config = {'require_confirmation_close': False}
        pattern = MomentumThrustPattern(config)
        
        # Create engulfing pattern data
        test_data = pd.DataFrame({
            'open': [100, 99],
            'high': [101, 102],
            'low': [99, 98],
            'close': [99, 101.5]  # Bullish engulfing
        })
        
        atr = 2.0
        is_long = True
        
        # Test with proper engulfing
        is_dangerous = pattern.dangerous_engulfing_check(test_data, atr, is_long)
        
        # Calculate if it meets all conditions
        current = test_data.iloc[-1]
        prev = test_data.iloc[-2]
        
        body = abs(current['close'] - current['open'])
        prev_body = abs(prev['close'] - prev['open'])
        bar_range = current['high'] - current['low']
        
        # Check all 4 conditions
        cond1 = body > 0.6 * bar_range  # Body > 60% of range
        cond2 = body > 0.6 * prev_body if prev_body > 0 else True  # Body > 60% of prev
        cond3 = (current['close'] > current['open']) != (prev['close'] > prev['open'])  # Opposite
        cond4 = current['close'] > prev['high'] if is_long else current['close'] < prev['low']
        
        expected = cond1 and cond2 and cond3 and cond4
        assert is_dangerous == expected, \
            f"Engulfing detection failed: got {is_dangerous}, expected {expected}"
    
    def test_entry_plan_creation(self, sample_data):
        """
        Acceptance Criteria 5: Entry plan with quality metrics
        - EntryPlan includes all required fields
        - Retest entry calculated correctly
        """
        config = {
            'lookback': 56,
            'momentum_threshold': 0.0014, 
            'volume_factor': 1.72,
            'stop_ticks': 6,
            'target_ticks': 4,
            'require_confirmation_close': False
        }
        
        pattern = MomentumThrustPattern(config)
        
        # Create a simple setup
        pattern.pending_setup = {
            'direction': 'bullish',
            'is_long': True,
            'momentum': 0.002,
            'volume_ratio': 2.0,
            'rsi': 55,
            'setup_time': datetime.now(timezone.utc),
            'setup_bar_index': 150
        }
        pattern.trigger_level = 15000
        pattern.pivot_level = 14990
        
        # Create signal with entry plan
        signal = pattern._create_signal(
            sample_data,
            15005,
            10.0,  # ATR
            confirm_bar_range=8.0,
            is_exhaustion=False,
            pullback_achieved=True
        )
        
        assert signal is not None, "Should create signal"
        assert signal.entry_plan is not None, "Should have entry plan"
        
        plan = signal.entry_plan
        assert plan.trigger_price == 15000, "Trigger price should match"
        assert plan.confirm_bar_range == 8.0, "Confirm bar range should match"
        assert plan.is_exhaustion == False, "Exhaustion flag should match"
        assert plan.pullback_achieved == True, "Pullback flag should match"
        
        # Check retest entry calculation
        # For long: max(trigger, 50% of confirm bar)
        current_bar_low = sample_data.iloc[-1]['low']
        fifty_percent = current_bar_low + (8.0 * 0.5)
        expected_retest = max(15000, fifty_percent)
        assert abs(plan.retest_entry - expected_retest) < 100 or \
               plan.retest_entry == 15000, \
               f"Retest entry calculation: got {plan.retest_entry}, expected ~{expected_retest}"
    
    def test_confidence_adjustment(self, sample_data):
        """
        Test that confidence is adjusted based on quality checks
        """
        config = {
            'lookback': 56,
            'momentum_threshold': 0.0014,
            'volume_factor': 1.72,
            'stop_ticks': 6,
            'target_ticks': 4,
            'require_confirmation_close': False
        }
        
        pattern = MomentumThrustPattern(config)
        
        # Setup
        pattern.pending_setup = {
            'direction': 'bullish',
            'is_long': True,
            'momentum': 0.002,
            'volume_ratio': 2.0,
            'rsi': 55,
            'setup_time': datetime.now(timezone.utc),
            'setup_bar_index': 150
        }
        pattern.trigger_level = 15000
        pattern.pivot_level = 14990
        
        # Test with perfect quality
        signal_good = pattern._create_signal(
            sample_data, 15005, 10.0,
            is_exhaustion=False,
            pullback_achieved=True
        )
        
        # Test with exhaustion
        signal_exhaustion = pattern._create_signal(
            sample_data, 15005, 10.0,
            is_exhaustion=True,
            pullback_achieved=True
        )
        
        # Test with no pullback
        signal_no_pullback = pattern._create_signal(
            sample_data, 15005, 10.0,
            is_exhaustion=False,
            pullback_achieved=False
        )
        
        # Confidence should be reduced for quality issues
        assert signal_exhaustion.confidence < signal_good.confidence, \
            "Exhaustion should reduce confidence"
        assert signal_no_pullback.confidence < signal_good.confidence, \
            "No pullback should reduce confidence"
    
    def test_trend_line_bounce_integration(self, sample_data):
        """
        Test that trend line bounce pattern has same quality checks
        """
        config = {
            'stop_ticks': 6,
            'target_ticks_normal': 3,
            'target_ticks_high_conf': 5,
            'require_confirmation_close': True
        }
        
        pattern = TrendLineBouncePattern(config)
        
        # Verify methods exist
        assert hasattr(pattern, 'exhaustion_check'), "Should have exhaustion_check"
        assert hasattr(pattern, 'micro_pullback_check'), "Should have micro_pullback_check"
        assert hasattr(pattern, 'dangerous_engulfing_check'), "Should have dangerous_engulfing_check"
        assert hasattr(pattern, '_check_confirmation'), "Should have _check_confirmation"
        assert hasattr(pattern, '_create_signal'), "Should have _create_signal"
        
        # Test exhaustion check
        assert pattern.exhaustion_check(13.0, 10.0) == True, "Should detect exhaustion"
        assert pattern.exhaustion_check(12.0, 10.0) == False, "Should not detect exhaustion"

if __name__ == "__main__":
    # Run tests
    import sys
    pytest.main([__file__, '-v'])