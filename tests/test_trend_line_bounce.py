"""
Unit Tests for Trend Line Bounce Pattern
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from patterns.trend_line_bounce import TrendLineBouncePattern
from patterns.base_pattern import TradeAction, PatternSignal
from utils.trend_line_detector import TrendLine
from pattern_config import get_pattern_config

class TestTrendLineBouncePattern(unittest.TestCase):
    """Test cases for trend line bounce pattern"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = get_pattern_config('trend_line_bounce')
        self.pattern = TrendLineBouncePattern(config)
        
        # Create sample data
        self.sample_data = self._create_sample_data()
        
    def _create_sample_data(self, num_bars=200):
        """Create sample OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=num_bars, freq='1min')
        
        # Create trending data with touches
        base_price = 20000
        trend = np.linspace(0, 100, num_bars)
        noise = np.random.randn(num_bars) * 5
        
        prices = base_price + trend + noise
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices - np.random.rand(num_bars) * 2,
            'high': prices + np.random.rand(num_bars) * 3,
            'low': prices - np.random.rand(num_bars) * 3,
            'close': prices + np.random.rand(num_bars) * 2,
            'volume': np.random.randint(100, 1000, num_bars)
        })
        
        return data
    
    def test_pattern_initialization(self):
        """Test pattern is properly initialized"""
        self.assertIsNotNone(self.pattern)
        self.assertIsNotNone(self.pattern.trend_detector)
        self.assertEqual(self.pattern.stop_ticks, 6)
        self.assertEqual(self.pattern.target_ticks_normal, 3)
        self.assertEqual(self.pattern.target_ticks_high_conf, 5)
    
    def test_scan_for_setup_no_data(self):
        """Test scanning with insufficient data"""
        small_data = self.sample_data[:50]
        result = self.pattern.scan_for_setup(small_data, 20050)
        self.assertIsNone(result)
    
    def test_scan_for_setup_no_trend_lines(self):
        """Test scanning when no trend lines are detected"""
        # Create random walk data (no clear trends)
        random_data = self.sample_data.copy()
        random_data['close'] = 20000 + np.random.randn(len(random_data)) * 50
        random_data['high'] = random_data['close'] + np.random.rand(len(random_data)) * 10
        random_data['low'] = random_data['close'] - np.random.rand(len(random_data)) * 10
        
        result = self.pattern.scan_for_setup(random_data, 20000)
        self.assertIsNone(result)
    
    @patch('patterns.trend_line_bounce.TrendLineBouncePattern._is_trading_time')
    def test_time_restrictions(self, mock_is_trading_time):
        """Test time restrictions prevent trading"""
        mock_is_trading_time.return_value = False
        
        result = self.pattern.scan_for_setup(self.sample_data, 20100)
        self.assertIsNone(result)
    
    def test_dangerous_engulfing_detection(self):
        """Test dangerous engulfing candle detection"""
        data = self.sample_data.copy()
        
        # Create engulfing candle
        data.loc[data.index[-1], 'open'] = 20000
        data.loc[data.index[-1], 'close'] = 20050  # Large body
        data.loc[data.index[-1], 'volume'] = 5000  # High volume
        
        # Update previous candle
        data.loc[data.index[-2], 'high'] = 20020
        data.loc[data.index[-2], 'low'] = 19995
        
        is_engulfing = self.pattern._is_dangerous_engulfing(data)
        self.assertTrue(is_engulfing)
    
    def test_confluence_score_calculation(self):
        """Test multi-timeframe confluence score calculation"""
        # Create mock trend line
        trend_line = TrendLine(
            slope=0.5,
            intercept=20000,
            r_squared=0.96,
            touch_points=[(100, 20050), (150, 20075)],
            line_type='support',
            strength=0.8,
            angle_degrees=26.6,
            last_update=pd.Timestamp.now()
        )
        
        # Set up timeframe data
        self.pattern.timeframe_data['1m'] = self.sample_data
        self.pattern.timeframe_data['5m'] = self.sample_data.resample('5min', on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        score = self.pattern._calculate_confluence_score(20100, trend_line)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_x_zone_detection(self):
        """Test trend line intersection (X-zone) detection"""
        # Mock confluence zones
        self.pattern.trend_detector.find_confluence_zones = MagicMock(return_value=[
            {
                'index': 195,
                'price': 20100,
                'support_line': MagicMock(),
                'resistance_line': MagicMock(),
                'strength': 0.85
            }
        ])
        
        result = self.pattern._check_x_zone_setup(20098, 190)
        self.assertTrue(result)
    
    def test_signal_validation(self):
        """Test signal validation logic"""
        signal = PatternSignal(
            pattern_name="TrendLineBounce",
            action=TradeAction.BUY,
            confidence=0.75,
            entry_price=20100,
            stop_loss=20098.5,
            take_profit=20100.75,
            position_size=1,
            reason="Test signal"
        )
        
        # Test valid signal
        result = self.pattern.validate_signal(
            signal, 
            spread=0.25,
            last_tick_time=datetime.now(timezone.utc)
        )
        self.assertTrue(result)
        
        # Test with wide spread
        result = self.pattern.validate_signal(
            signal,
            spread=2.0,
            last_tick_time=datetime.now(timezone.utc)
        )
        self.assertFalse(result)
        
        # Test with stale data
        result = self.pattern.validate_signal(
            signal,
            spread=0.25,
            last_tick_time=datetime.now(timezone.utc) - timedelta(seconds=5)
        )
        self.assertFalse(result)
    
    def test_pattern_statistics_update(self):
        """Test pattern statistics are updated correctly"""
        initial_signals = self.pattern.total_signals
        
        # Simulate winning trade
        self.pattern.update_statistics(50.0, True)
        self.assertEqual(self.pattern.total_signals, initial_signals + 1)
        self.assertEqual(self.pattern.winning_signals, 1)
        self.assertEqual(self.pattern.total_pnl, 50.0)
        self.assertEqual(self.pattern.consecutive_losses, 0)
        
        # Simulate losing trade
        self.pattern.update_statistics(-25.0, False)
        self.assertEqual(self.pattern.total_signals, initial_signals + 2)
        self.assertEqual(self.pattern.winning_signals, 1)
        self.assertEqual(self.pattern.total_pnl, 25.0)
        self.assertEqual(self.pattern.consecutive_losses, 1)
    
    def test_state_persistence(self):
        """Test pattern state save and load"""
        # Set some state
        self.pattern.total_signals = 10
        self.pattern.winning_signals = 6
        self.pattern.total_pnl = 150.0
        self.pattern.consecutive_losses = 1
        
        # Save state
        state = self.pattern.get_state()
        
        # Create new pattern and load state
        new_pattern = TrendLineBouncePattern(get_pattern_config('trend_line_bounce'))
        new_pattern.load_state(state)
        
        # Verify state was loaded correctly
        self.assertEqual(new_pattern.total_signals, 10)
        self.assertEqual(new_pattern.winning_signals, 6)
        self.assertEqual(new_pattern.total_pnl, 150.0)
        self.assertEqual(new_pattern.consecutive_losses, 1)
    
    def test_circuit_breaker(self):
        """Test circuit breaker after consecutive losses"""
        # Simulate 3 consecutive losses
        self.pattern.consecutive_losses = 3
        
        signal = PatternSignal(
            pattern_name="TrendLineBounce",
            action=TradeAction.BUY,
            confidence=0.75,
            entry_price=20100,
            stop_loss=20098.5,
            take_profit=20100.75,
            position_size=1,
            reason="Test signal"
        )
        
        result = self.pattern.validate_signal(
            signal,
            spread=0.25,
            last_tick_time=datetime.now(timezone.utc)
        )
        
        self.assertFalse(result)
    
    def test_daily_trade_limit(self):
        """Test daily trade limit enforcement"""
        # Set daily trades to limit
        self.pattern.daily_trades = 20
        self.pattern.last_trade_date = datetime.now(timezone.utc).date()
        
        signal = PatternSignal(
            pattern_name="TrendLineBounce",
            action=TradeAction.BUY,
            confidence=0.75,
            entry_price=20100,
            stop_loss=20098.5,
            take_profit=20100.75,
            position_size=1,
            reason="Test signal"
        )
        
        result = self.pattern.validate_signal(
            signal,
            spread=0.25,
            last_tick_time=datetime.now(timezone.utc)
        )
        
        self.assertFalse(result)
    
    def test_position_sizing(self):
        """Test position sizing based on confidence"""
        # Mock trend line and detector
        with patch.object(self.pattern.trend_detector, 'detect_trend_lines') as mock_detect:
            with patch.object(self.pattern.trend_detector, 'find_nearest_line') as mock_nearest:
                with patch.object(self.pattern.trend_detector, 'is_third_touch') as mock_third:
                    with patch.object(self.pattern, '_is_trading_time') as mock_time:
                        with patch.object(self.pattern, '_is_dangerous_engulfing') as mock_engulfing:
                            
                            # Setup mocks
                            mock_time.return_value = True
                            mock_engulfing.return_value = False
                            mock_third.return_value = True
                            
                            trend_line = TrendLine(
                                slope=0.5,
                                intercept=20000,
                                r_squared=0.96,
                                touch_points=[(100, 20050), (150, 20075)],
                                line_type='support',
                                strength=0.8,
                                angle_degrees=26.6,
                                last_update=pd.Timestamp.now()
                            )
                            
                            mock_detect.return_value = {'support': [trend_line], 'resistance': []}
                            mock_nearest.return_value = (trend_line, 0.3)
                            
                            # Test normal confidence
                            self.pattern.current_confluence_score = 0.7
                            with patch.object(self.pattern, '_calculate_confluence_score', return_value=0.7):
                                signal = self.pattern.scan_for_setup(self.sample_data, 20100)
                                if signal:
                                    self.assertEqual(signal.position_size, 1)
                            
                            # Test high confidence
                            self.pattern.current_confluence_score = 0.9
                            with patch.object(self.pattern, '_calculate_confluence_score', return_value=0.9):
                                signal = self.pattern.scan_for_setup(self.sample_data, 20100)
                                if signal:
                                    self.assertEqual(signal.position_size, 2)


class TestTrendLineDetector(unittest.TestCase):
    """Test cases for trend line detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        from utils.trend_line_detector import TrendLineDetector
        self.detector = TrendLineDetector()
        
    def test_swing_point_detection(self):
        """Test swing high and low detection"""
        # Create data with clear peaks and troughs
        data = pd.DataFrame({
            'high': [100, 102, 105, 103, 101, 104, 106, 104, 102],
            'low': [98, 100, 103, 101, 99, 102, 104, 102, 100]
        })
        
        swings = self.detector.detect_swing_points(data, order=2)
        
        self.assertIn('highs', swings)
        self.assertIn('lows', swings)
        self.assertIsInstance(swings['highs'], np.ndarray)
        self.assertIsInstance(swings['lows'], np.ndarray)
    
    def test_trend_line_fitting(self):
        """Test RANSAC trend line fitting"""
        # Create perfect trend line data
        indices = np.array([0, 10, 20, 30, 40])
        prices = np.array([100, 102, 104, 106, 108])  # Perfect line: y = 0.2x + 100
        
        line = self.detector.fit_trend_line_ransac(indices, prices)
        
        self.assertIsNotNone(line)
        self.assertAlmostEqual(line.slope, 0.2, places=1)
        self.assertAlmostEqual(line.intercept, 100, places=0)
        self.assertGreater(line.r_squared, 0.95)
    
    def test_third_touch_detection(self):
        """Test third touch detection logic"""
        trend_line = TrendLine(
            slope=0.1,
            intercept=100,
            r_squared=0.98,
            touch_points=[(10, 101), (20, 102)],  # 2 existing touches
            line_type='support',
            strength=0.7,
            angle_degrees=5.7,
            last_update=pd.Timestamp.now()
        )
        
        # Test third touch
        is_third = self.detector.is_third_touch(trend_line, 30, 103)
        self.assertTrue(is_third)
        
        # Test when already has 3 touches
        trend_line.touch_points.append((30, 103))
        is_third = self.detector.is_third_touch(trend_line, 40, 104)
        self.assertFalse(is_third)


if __name__ == '__main__':
    unittest.main()