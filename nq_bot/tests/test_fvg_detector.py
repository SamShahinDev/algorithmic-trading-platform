"""
Test harness for FVG detection and state management
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nq_bot.patterns.fvg_strategy import FVGStrategy, FVGObject
from nq_bot.pattern_config import FVG


class TestFVGDetector:
    """Test FVG detection and state transitions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock logger
        self.logger = Mock()
        self.logger.info = MagicMock()
        self.logger.error = MagicMock()
        
        # Mock data cache
        self.data_cache = Mock()
        
        # Create FVG strategy
        self.strategy = FVGStrategy(self.data_cache, self.logger, FVG)
    
    def create_synthetic_bars(self, scenario='bullish_fvg'):
        """Create synthetic bar data for testing"""
        
        if scenario == 'bullish_fvg':
            # Create bars with bullish FVG pattern
            # Bar pattern: normal → liquidity sweep → displacement → gap
            data = {
                'open': [100, 99.5, 98, 99, 102, 103.5],
                'high': [100.5, 100, 99, 104, 104.5, 104],
                'low': [99.5, 98.5, 97.5, 98.5, 101.5, 103],  # Gap: bar[2].high=99 < bar[4].low=101.5
                'close': [99.75, 98.75, 98.5, 103.5, 104, 103.75],
                'volume': [1000, 1200, 1500, 3000, 1100, 900]  # Displacement has high volume
            }
        elif scenario == 'bearish_fvg':
            # Create bars with bearish FVG pattern
            data = {
                'open': [100, 100.5, 102, 101, 98, 96.5],
                'high': [100.5, 101.5, 102.5, 101.5, 98.5, 97],  # Gap: bar[2].low=101.5 > bar[4].high=98.5
                'low': [99.5, 100, 101.5, 96, 95.5, 96],
                'close': [100.25, 101.25, 101.5, 96.5, 96, 96.25],
                'volume': [1000, 1200, 1500, 3000, 1100, 900]
            }
        elif scenario == 'touch_and_defend':
            # Create bars that touch FVG and defend
            data = {
                'open': [100, 99.5, 98, 99, 102, 103.5, 103.75, 102],
                'high': [100.5, 100, 99, 104, 104.5, 104, 104, 102.5],
                'low': [99.5, 98.5, 97.5, 98.5, 101.5, 103, 101.8, 101.9],  # Bar 6 touches gap
                'close': [99.75, 98.75, 98.5, 103.5, 104, 103.75, 102.8, 102.2],  # Closes outside inner 25%
                'volume': [1000, 1200, 1500, 3000, 1100, 900, 1000, 950]
            }
        else:
            # Normal bars without FVG
            data = {
                'open': [100, 100.2, 100.1, 100.3, 100.4, 100.2],
                'high': [100.5, 100.6, 100.4, 100.7, 100.8, 100.5],
                'low': [99.5, 99.8, 99.7, 99.9, 100, 99.8],
                'close': [100.2, 100.1, 100.3, 100.4, 100.2, 100.1],
                'volume': [1000, 1100, 1050, 1000, 1100, 1000]
            }
        
        # Create DataFrame with datetime index
        idx = pd.date_range(start='2025-01-01 09:00:00', periods=len(data['open']), freq='1min')
        return pd.DataFrame(data, index=idx)
    
    def test_bullish_fvg_detection(self):
        """Test detection of bullish FVG"""
        # Setup synthetic bars
        bars = self.create_synthetic_bars('bullish_fvg')
        
        # Mock data cache to return bars
        self.data_cache.get_bars = Mock(return_value=bars)
        
        # Add swing for liquidity sweep
        self.strategy.recent_swings = [
            {'level': 97.5, 'type': 'low', 'bar_idx': 2}
        ]
        
        # Run scan
        counts = self.strategy.scan()
        
        # Check that FVG was detected
        assert len(self.strategy.fvg_registry) > 0, "No FVG detected"
        
        # Verify FVG properties
        fvg = list(self.strategy.fvg_registry.values())[0]
        assert fvg.direction == 'long', f"Expected long FVG, got {fvg.direction}"
        assert fvg.status == 'FRESH', f"Expected FRESH status, got {fvg.status}"
        assert fvg.bottom == 99, "Incorrect FVG bottom"
        assert fvg.top == 101.5, "Incorrect FVG top"
        
        # Check logger was called
        self.logger.info.assert_any_call(
            pytest.StringContaining("FVG_DETECTED dir=long")
        )
    
    def test_bearish_fvg_detection(self):
        """Test detection of bearish FVG"""
        # Setup synthetic bars
        bars = self.create_synthetic_bars('bearish_fvg')
        
        # Mock data cache
        self.data_cache.get_bars = Mock(return_value=bars)
        
        # Add swing for liquidity sweep
        self.strategy.recent_swings = [
            {'level': 102.5, 'type': 'high', 'bar_idx': 2}
        ]
        
        # Run scan
        counts = self.strategy.scan()
        
        # Check that FVG was detected
        assert len(self.strategy.fvg_registry) > 0, "No FVG detected"
        
        # Verify FVG properties
        fvg = list(self.strategy.fvg_registry.values())[0]
        assert fvg.direction == 'short', f"Expected short FVG, got {fvg.direction}"
        assert fvg.top == 101.5, "Incorrect FVG top"
        assert fvg.bottom == 98.5, "Incorrect FVG bottom"
    
    def test_fvg_arming(self):
        """Test FVG arming when touched and defended"""
        # Create FVG first
        bars_initial = self.create_synthetic_bars('bullish_fvg')
        self.data_cache.get_bars = Mock(return_value=bars_initial)
        
        # Add swing and detect FVG
        self.strategy.recent_swings = [
            {'level': 97.5, 'type': 'low', 'bar_idx': 2}
        ]
        self.strategy.scan()
        
        # Get the FVG
        fvg_id = list(self.strategy.fvg_registry.keys())[0]
        fvg = self.strategy.fvg_registry[fvg_id]
        assert fvg.status == 'FRESH', "FVG should be FRESH initially"
        
        # Now create bars that touch and defend the gap
        bars_touch = self.create_synthetic_bars('touch_and_defend')
        self.data_cache.get_bars = Mock(return_value=bars_touch)
        
        # Run scan again
        counts = self.strategy.scan()
        
        # Check that FVG is now ARMED
        assert fvg.status == 'ARMED', f"Expected ARMED status, got {fvg.status}"
        assert fvg.armed_at is not None, "armed_at should be set"
        
        # Check logger
        self.logger.info.assert_any_call(
            pytest.StringContaining("FVG_ARMED")
        )
    
    def test_fvg_timeout_invalidation(self):
        """Test FVG expires after timeout"""
        # Create FVG
        bars = self.create_synthetic_bars('bullish_fvg')
        self.data_cache.get_bars = Mock(return_value=bars)
        self.strategy.recent_swings = [
            {'level': 97.5, 'type': 'low', 'bar_idx': 2}
        ]
        
        # Initial scan
        self.strategy.scan()
        fvg_id = list(self.strategy.fvg_registry.keys())[0]
        fvg = self.strategy.fvg_registry[fvg_id]
        
        # Set created_at to past timeout
        fvg.created_at = time.time() - 601  # 601 seconds ago
        
        # Run scan again
        self.strategy.scan()
        
        # Check FVG is expired
        assert fvg.status == 'EXPIRED', f"Expected EXPIRED status, got {fvg.status}"
        assert fvg.invalidation_reason == 'timeout', "Should be invalidated by timeout"
        
        # Check logger
        self.logger.info.assert_any_call(
            pytest.StringContaining("FVG_INVALID")
        )
    
    def test_fvg_consumption_invalidation(self):
        """Test FVG invalidated when 75% consumed"""
        # Create FVG
        bars = self.create_synthetic_bars('bullish_fvg')
        self.data_cache.get_bars = Mock(return_value=bars)
        self.strategy.recent_swings = [
            {'level': 97.5, 'type': 'low', 'bar_idx': 2}
        ]
        
        # Initial scan
        self.strategy.scan()
        fvg_id = list(self.strategy.fvg_registry.keys())[0]
        fvg = self.strategy.fvg_registry[fvg_id]
        
        # Create bars that consume 75% of the gap
        # Gap is from 99 to 101.5 (2.5 points)
        # 75% consumption means price goes to 99 + 0.25*2.5 = 99.625
        consumed_bars = bars.copy()
        consumed_bars.loc[consumed_bars.index[-1], 'low'] = 99.5  # Consume 80% of gap
        
        self.data_cache.get_bars = Mock(return_value=consumed_bars)
        
        # Run scan
        self.strategy.scan()
        
        # Check FVG is invalid
        assert fvg.status == 'INVALID', f"Expected INVALID status, got {fvg.status}"
        assert fvg.invalidation_reason == 'consumed', "Should be invalidated by consumption"
    
    def test_get_best_armed(self):
        """Test selection of best armed FVG"""
        # Create multiple FVGs manually
        fvg1 = FVGObject(
            id='FVG_1',
            direction='long',
            created_at=time.time() - 100,
            top=101,
            bottom=100,
            mid=100.5,
            quality=0.7,
            status='ARMED',
            origin_swing=99,
            armed_at=time.time() - 50
        )
        
        fvg2 = FVGObject(
            id='FVG_2',
            direction='short',
            created_at=time.time() - 80,
            top=102,
            bottom=101,
            mid=101.5,
            quality=0.9,  # Higher quality
            status='ARMED',
            origin_swing=103,
            armed_at=time.time() - 30
        )
        
        fvg3 = FVGObject(
            id='FVG_3',
            direction='long',
            created_at=time.time() - 60,
            top=100,
            bottom=99,
            mid=99.5,
            quality=0.5,
            status='FRESH',  # Not armed
            origin_swing=98
        )
        
        self.strategy.fvg_registry = {
            'FVG_1': fvg1,
            'FVG_2': fvg2,
            'FVG_3': fvg3
        }
        
        # Get best armed
        best = self.strategy.get_best_armed()
        
        # Should return FVG_2 (highest quality among armed)
        assert best is not None, "Should return an armed FVG"
        assert best.id == 'FVG_2', f"Expected FVG_2, got {best.id}"
        assert best.quality == 0.9, "Should select highest quality"
    
    def test_mark_consumed(self):
        """Test marking FVG as consumed"""
        # Create FVG
        bars = self.create_synthetic_bars('bullish_fvg')
        self.data_cache.get_bars = Mock(return_value=bars)
        self.strategy.recent_swings = [
            {'level': 97.5, 'type': 'low', 'bar_idx': 2}
        ]
        
        # Initial scan
        self.strategy.scan()
        fvg_id = list(self.strategy.fvg_registry.keys())[0]
        fvg = self.strategy.fvg_registry[fvg_id]
        
        # Mark as consumed
        self.strategy.mark_consumed(fvg_id)
        
        # Check status
        assert fvg.status == 'CONSUMED', f"Expected CONSUMED status, got {fvg.status}"
        
        # Check logger
        self.logger.info.assert_any_call(f"FVG_CONSUMED id={fvg_id}")


class StringContaining(str):
    """Helper for partial string matching in assertions"""
    def __eq__(self, other):
        return self in other


# Make pytest.StringContaining available
pytest.StringContaining = StringContaining


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])