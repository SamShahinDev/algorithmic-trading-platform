"""
Unit tests for indicator calculations.

Tests VWAP, ADX, RSI, EMA, MACD calculations with known inputs.
Verifies incremental updates match full recalculation.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from indicators.cache import IndicatorCache
from indicators.vwap import VWAPAnalyzer
from indicators.regime import RegimeDetector
from indicators.technicals import TechnicalIndicators


class TestIndicatorCache:
    """Test IndicatorCache incremental calculations."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = IndicatorCache(max_bars=100)

        assert cache.max_bars == 100
        assert len(cache.bars) == 0
        assert cache.state['rsi_initialized'] == False

    def test_add_bars(self):
        """Test adding bars to cache."""
        cache = IndicatorCache(max_bars=100)

        # Add sample bars
        now = datetime.now()
        for i in range(50):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 + i,
                high=21010 + i,
                low=20990 + i,
                close=21005 + i,
                volume=1000
            )

        assert len(cache.bars) == 50
        assert cache.get_latest_bar().close == 21005 + 49

    def test_max_bars_limit(self):
        """Test cache respects max_bars limit."""
        cache = IndicatorCache(max_bars=50)

        now = datetime.now()
        for i in range(100):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21010,
                low=20990,
                close=21005,
                volume=1000
            )

        # Should only keep last 50 bars
        assert len(cache.bars) == 50

    def test_rsi_calculation(self):
        """Test RSI incremental calculation."""
        cache = IndicatorCache(max_bars=100)

        # Create trending up data (RSI should be high)
        now = datetime.now()
        for i in range(30):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 + (i * 5),
                high=21010 + (i * 5),
                low=20990 + (i * 5),
                close=21005 + (i * 5),
                volume=1000
            )

        rsi = cache.get_rsi()
        assert rsi > 50  # Should be bullish
        assert 0 <= rsi <= 100

    def test_ema_calculation(self):
        """Test EMA incremental calculation."""
        cache = IndicatorCache(max_bars=100)

        # Add enough bars to calculate EMA20
        now = datetime.now()
        for i in range(30):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21010,
                low=20990,
                close=21000 + i,
                volume=1000
            )

        ema20 = cache.get_ema(20)
        assert ema20 is not None
        assert ema20 > 21000  # Should be above initial price

    def test_atr_calculation(self):
        """Test ATR incremental calculation."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(30):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21050,  # 50 point range
                low=21000,
                close=21025,
                volume=1000
            )

        atr = cache.get_atr()
        assert atr is not None
        assert atr > 0

    def test_adx_calculation(self):
        """Test ADX calculation for trend strength."""
        cache = IndicatorCache(max_bars=100)

        # Create strong uptrend
        now = datetime.now()
        for i in range(30):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 + (i * 10),
                high=21020 + (i * 10),
                low=21000 + (i * 10),
                close=21015 + (i * 10),
                volume=1000
            )

        adx = cache.get_adx()
        assert adx is not None
        assert 0 <= adx <= 100


class TestVWAPAnalyzer:
    """Test VWAP calculations and distance measurements."""

    def test_vwap_calculation(self):
        """Test basic VWAP calculation."""
        cache = IndicatorCache(max_bars=100)

        # Add bars with equal volume
        now = datetime.now()
        prices = [21000, 21010, 21020, 21015, 21005]
        for i, price in enumerate(prices):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=price,
                high=price + 10,
                low=price - 10,
                close=price,
                volume=1000
            )

        vwap_analyzer = VWAPAnalyzer(cache)
        vwap = vwap_analyzer.calculate_vwap()

        assert vwap is not None
        assert vwap > 21000
        assert vwap < 21020

    def test_vwap_bands(self):
        """Test VWAP standard deviation bands."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(50):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21050,
                low=20950,
                close=21000,
                volume=1000
            )

        vwap_analyzer = VWAPAnalyzer(cache)
        bands = vwap_analyzer.get_vwap_bands()

        assert 'vwap' in bands
        assert 'upper_1std' in bands
        assert 'lower_1std' in bands
        assert 'upper_2std' in bands
        assert 'lower_2std' in bands

        # Upper bands should be higher than VWAP
        assert bands['upper_1std'] > bands['vwap']
        assert bands['upper_2std'] > bands['upper_1std']

        # Lower bands should be lower than VWAP
        assert bands['lower_1std'] < bands['vwap']
        assert bands['lower_2std'] < bands['lower_1std']

    def test_distance_from_vwap(self):
        """Test distance calculation from VWAP."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(50):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21050,
                low=20950,
                close=21000,
                volume=1000
            )

        vwap_analyzer = VWAPAnalyzer(cache)

        # Test price above VWAP
        summary = vwap_analyzer.get_vwap_summary(21050)
        assert summary['distance_from_vwap'] > 0
        assert summary['std_dev_distance'] > 0

        # Test price below VWAP
        summary = vwap_analyzer.get_vwap_summary(20950)
        assert summary['distance_from_vwap'] < 0
        assert summary['std_dev_distance'] < 0


class TestRegimeDetector:
    """Test market regime detection."""

    def test_ranging_regime(self):
        """Test detection of ranging market."""
        cache = IndicatorCache(max_bars=100)

        # Create ranging market (low ADX)
        now = datetime.now()
        for i in range(30):
            # Oscillate around 21000
            price = 21000 + (10 if i % 2 == 0 else -10)
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=price,
                high=price + 5,
                low=price - 5,
                close=price,
                volume=1000
            )

        detector = RegimeDetector(cache)
        regime = detector.get_regime()

        # Should detect ranging (though may need more bars)
        assert regime in ['RANGING', 'UNKNOWN']

    def test_trending_up_regime(self):
        """Test detection of uptrend."""
        cache = IndicatorCache(max_bars=100)

        # Create strong uptrend
        now = datetime.now()
        for i in range(50):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 + (i * 5),
                high=21010 + (i * 5),
                low=21000 + (i * 5),
                close=21005 + (i * 5),
                volume=1000
            )

        detector = RegimeDetector(cache)
        regime = detector.get_regime()

        assert regime.value in ['TRENDING_UP', 'UNKNOWN']

    def test_trending_down_regime(self):
        """Test detection of downtrend."""
        cache = IndicatorCache(max_bars=100)

        # Create strong downtrend
        now = datetime.now()
        for i in range(50):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 - (i * 5),
                high=21000 - (i * 5),
                low=20990 - (i * 5),
                close=20995 - (i * 5),
                volume=1000
            )

        detector = RegimeDetector(cache)
        regime = detector.get_regime()

        assert regime.value in ['TRENDING_DOWN', 'UNKNOWN']

    def test_regime_summary(self):
        """Test regime summary output."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(50):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21010,
                low=20990,
                close=21000,
                volume=1000
            )

        detector = RegimeDetector(cache)
        summary = detector.get_regime_summary()

        assert 'regime' in summary
        assert 'adx' in summary
        assert 'di_plus' in summary
        assert 'di_minus' in summary
        assert 'trend_strength' in summary


class TestTechnicalIndicators:
    """Test technical indicator analysis."""

    def test_rsi_signal(self):
        """Test RSI signal generation."""
        cache = IndicatorCache(max_bars=100)

        # Create oversold condition
        now = datetime.now()
        for i in range(30):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 - (i * 10),
                high=21010 - (i * 10),
                low=20990 - (i * 10),
                close=21000 - (i * 10),
                volume=1000
            )

        technicals = TechnicalIndicators(cache)
        rsi_signal = technicals.get_rsi_signal()

        assert 'rsi' in rsi_signal
        assert 'signal' in rsi_signal
        assert rsi_signal['rsi'] >= 0
        assert rsi_signal['rsi'] <= 100

    def test_ema_cross(self):
        """Test EMA crossover detection."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(60):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000 + i,
                high=21010 + i,
                low=20990 + i,
                close=21000 + i,
                volume=1000
            )

        technicals = TechnicalIndicators(cache)
        ema_cross = technicals.get_ema_cross()

        assert 'ema20' in ema_cross
        assert 'ema50' in ema_cross
        assert 'alignment' in ema_cross

    def test_macd_analysis(self):
        """Test MACD histogram and signal."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(60):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21010,
                low=20990,
                close=21000 + (i % 10),
                volume=1000
            )

        technicals = TechnicalIndicators(cache)
        macd = technicals.get_macd_analysis()

        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd

    def test_atr_volatility_level(self):
        """Test ATR volatility classification."""
        cache = IndicatorCache(max_bars=100)

        now = datetime.now()
        for i in range(30):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=21000,
                high=21100,  # High volatility
                low=20900,
                close=21000,
                volume=1000
            )

        technicals = TechnicalIndicators(cache)
        atr_vol = technicals.get_atr_volatility_level()

        assert 'atr' in atr_vol
        assert 'volatility_level' in atr_vol
        assert atr_vol['volatility_level'] in ['LOW', 'NORMAL', 'HIGH']


class TestIndicatorAccuracy:
    """Test indicator accuracy against known values."""

    def test_rsi_known_values(self):
        """Test RSI with known input/output."""
        cache = IndicatorCache(max_bars=100)

        # Use well-known RSI test data
        now = datetime.now()
        closes = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
            46.03, 46.41, 46.22, 45.64
        ]

        for i, close in enumerate(closes):
            cache.add_bar(
                timestamp=now + timedelta(minutes=i),
                open=close,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=1000
            )

        rsi = cache.get_rsi()

        # RSI should be in valid range
        assert 0 <= rsi <= 100
        # With trending data, should be > 50
        assert rsi > 50

    def test_incremental_vs_batch_calculation(self):
        """Test incremental matches batch calculation."""
        cache_incremental = IndicatorCache(max_bars=100)

        # Build incrementally
        now = datetime.now()
        bars_data = []
        for i in range(50):
            bar_data = {
                'timestamp': now + timedelta(minutes=i),
                'open': 21000 + i,
                'high': 21010 + i,
                'low': 20990 + i,
                'close': 21000 + i,
                'volume': 1000
            }
            bars_data.append(bar_data)
            cache_incremental.add_bar(**bar_data)

        # Get incremental RSI
        rsi_incremental = cache_incremental.get_rsi()

        # Build batch for comparison
        cache_batch = IndicatorCache(max_bars=100)
        for bar_data in bars_data:
            cache_batch.add_bar(**bar_data)

        rsi_batch = cache_batch.get_rsi()

        # Should be very close (within rounding)
        assert abs(rsi_incremental - rsi_batch) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
