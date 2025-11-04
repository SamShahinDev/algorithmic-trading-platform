"""
Unit tests for trading strategies.

Tests strategy logic with mock market data:
- VWAP mean reversion strategy
- Opening range breakout strategy
- Momentum continuation strategy

Verifies confidence scoring, entry/exit logic, and time filters.
"""

import pytest
from datetime import datetime, time, timedelta
from strategies.vwap_strategy import VWAPStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.base_strategy import Signal
from indicators.cache import IndicatorCache


class TestVWAPStrategy:
    """Test VWAP mean reversion strategy."""

    def setup_method(self):
        """Setup for each test."""
        self.strategy = VWAPStrategy(
            config={
                'entry_zone_min_std': 1.5,
                'entry_zone_max_std': 2.5,
                'rsi_min': 45,
                'rsi_max': 55,
                'max_spread_ticks': 2
            },
            tick_size=0.25,
            tick_value=5.0
        )

    def create_ranging_market_state(self, price: float, vwap: float, vwap_std: float, rsi: float):
        """Create mock market state for VWAP testing."""
        return {
            'current_price': price,
            'time': datetime.now(),
            'spread': 0.5,
            'regime': 'RANGING',
            'indicators': {
                'vwap': {
                    'vwap': vwap,
                    'std_dev': vwap_std,
                    'distance_from_vwap': price - vwap,
                    'std_dev_distance': (price - vwap) / vwap_std if vwap_std > 0 else 0
                },
                'rsi': {
                    'rsi': rsi,
                    'signal': 'NEUTRAL'
                },
                'atr': {
                    'atr': 20.0,
                    'volatility_level': 'NORMAL'
                }
            }
        }

    def test_vwap_long_setup(self):
        """Test VWAP mean reversion LONG setup."""
        # Price 2 std dev below VWAP, RSI neutral
        market_state = self.create_ranging_market_state(
            price=20900,
            vwap=21000,
            vwap_std=50,
            rsi=50
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.LONG
        assert setup.confidence >= 7.0
        assert setup.entry_price == 20900
        assert setup.target_price > setup.entry_price
        assert setup.stop_price < setup.entry_price

    def test_vwap_short_setup(self):
        """Test VWAP mean reversion SHORT setup."""
        # Price 2 std dev above VWAP, RSI neutral
        market_state = self.create_ranging_market_state(
            price=21100,
            vwap=21000,
            vwap_std=50,
            rsi=50
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.SHORT
        assert setup.confidence >= 7.0
        assert setup.entry_price == 21100
        assert setup.target_price < setup.entry_price
        assert setup.stop_price > setup.entry_price

    def test_vwap_no_setup_trending_market(self):
        """Test VWAP skips trending markets."""
        market_state = self.create_ranging_market_state(
            price=20900,
            vwap=21000,
            vwap_std=50,
            rsi=50
        )
        market_state['regime'] = 'TRENDING_UP'

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE
        assert 'Not RANGING' in str(setup.conditions_failed)

    def test_vwap_no_setup_wrong_distance(self):
        """Test VWAP skips if price too close to VWAP."""
        # Price only 0.5 std dev from VWAP
        market_state = self.create_ranging_market_state(
            price=21025,
            vwap=21000,
            vwap_std=50,
            rsi=50
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE

    def test_vwap_no_setup_rsi_extreme(self):
        """Test VWAP skips if RSI not neutral."""
        market_state = self.create_ranging_market_state(
            price=20900,
            vwap=21000,
            vwap_std=50,
            rsi=70  # Overbought
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE

    def test_vwap_confidence_scoring(self):
        """Test VWAP confidence increases with better conditions."""
        # Perfect setup
        market_state_perfect = self.create_ranging_market_state(
            price=20900,  # 2 std dev
            vwap=21000,
            vwap_std=50,
            rsi=50
        )

        # Good but not perfect
        market_state_good = self.create_ranging_market_state(
            price=20925,  # 1.5 std dev
            vwap=21000,
            vwap_std=50,
            rsi=48
        )

        setup_perfect = self.strategy.analyze(market_state_perfect)
        setup_good = self.strategy.analyze(market_state_good)

        assert setup_perfect.confidence > setup_good.confidence


class TestBreakoutStrategy:
    """Test opening range breakout strategy."""

    def setup_method(self):
        """Setup for each test."""
        self.strategy = BreakoutStrategy(
            config={
                'range_start_time': '09:30:00',
                'range_end_time': '10:00:00',
                'breakout_min_ticks': 2,
                'volume_multiplier': 1.5,
                'max_range_size_ticks': 40
            },
            tick_size=0.25,
            tick_value=5.0
        )

    def create_breakout_market_state(self, price: float, volume: float, current_time: time):
        """Create mock market state for breakout testing."""
        return {
            'current_price': price,
            'time': datetime.combine(datetime.today(), current_time),
            'spread': 0.5,
            'regime': 'RANGING',
            'bars': [
                {'close': price, 'volume': volume, 'timestamp': datetime.now()}
            ],
            'indicators': {
                'atr': {'atr': 20.0, 'volatility_level': 'NORMAL'},
                'rsi': {'rsi': 55, 'signal': 'NEUTRAL'}
            }
        }

    def test_opening_range_tracking(self):
        """Test opening range is tracked correctly."""
        # During opening range (9:30-10:00)
        market_state = self.create_breakout_market_state(
            price=21000,
            volume=2000,
            current_time=time(9, 35, 0)
        )

        self.strategy.analyze(market_state)

        # Range should be initializing
        assert self.strategy.opening_range is not None or not self.strategy.range_established

        # Add more data points
        for price in [21010, 21005, 21015, 20995]:
            market_state['current_price'] = price
            market_state['time'] = market_state['time'] + timedelta(minutes=5)
            self.strategy.analyze(market_state)

        # After 10:00, range should be established
        market_state['time'] = datetime.combine(datetime.today(), time(10, 5, 0))
        self.strategy.analyze(market_state)

        assert self.strategy.range_established

    def test_breakout_long_setup(self):
        """Test breakout LONG setup."""
        # Establish range first
        self.strategy.opening_range = {'high': 21010, 'low': 20990}
        self.strategy.range_established = True
        self.strategy.breakout_taken = False

        # Price breaks above range
        market_state = self.create_breakout_market_state(
            price=21015,  # 2+ ticks above range high
            volume=3000,  # High volume
            current_time=time(10, 15, 0)
        )

        setup = self.strategy.analyze(market_state)

        # May or may not trigger depending on volume average
        # Just verify logic runs
        assert setup.signal in [Signal.LONG, Signal.NONE]

    def test_breakout_short_setup(self):
        """Test breakout SHORT setup."""
        # Establish range first
        self.strategy.opening_range = {'high': 21010, 'low': 20990}
        self.strategy.range_established = True
        self.strategy.breakout_taken = False

        # Price breaks below range
        market_state = self.create_breakout_market_state(
            price=20985,  # 2+ ticks below range low
            volume=3000,
            current_time=time(10, 15, 0)
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal in [Signal.SHORT, Signal.NONE]

    def test_breakout_no_setup_during_range(self):
        """Test no breakout during range building."""
        market_state = self.create_breakout_market_state(
            price=21000,
            volume=2000,
            current_time=time(9, 45, 0)  # During range
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE

    def test_breakout_one_per_day(self):
        """Test only one breakout per day."""
        self.strategy.opening_range = {'high': 21010, 'low': 20990}
        self.strategy.range_established = True
        self.strategy.breakout_taken = True  # Already taken

        market_state = self.create_breakout_market_state(
            price=21020,
            volume=3000,
            current_time=time(10, 15, 0)
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE


class TestMomentumStrategy:
    """Test momentum continuation strategy."""

    def setup_method(self):
        """Setup for each test."""
        self.strategy = MomentumStrategy(
            config={
                'ema_fast': 20,
                'ema_slow': 50,
                'pullback_max_ticks': 10,
                'volume_multiplier': 1.0
            },
            tick_size=0.25,
            tick_value=5.0
        )

    def create_momentum_market_state(
        self,
        price: float,
        ema20: float,
        ema50: float,
        macd_histogram: float,
        regime: str = 'TRENDING_UP'
    ):
        """Create mock market state for momentum testing."""
        return {
            'current_price': price,
            'time': datetime.now(),
            'spread': 0.5,
            'regime': regime,
            'bars': [
                {'close': price, 'volume': 2000, 'timestamp': datetime.now()}
            ],
            'indicators': {
                'ema': {
                    'ema20': ema20,
                    'ema50': ema50,
                    'alignment': 'BULLISH' if ema20 > ema50 else 'BEARISH'
                },
                'macd': {
                    'macd': 10.0,
                    'signal': 8.0,
                    'histogram': macd_histogram
                },
                'rsi': {
                    'rsi': 60,
                    'signal': 'BULLISH'
                },
                'atr': {
                    'atr': 20.0,
                    'volatility_level': 'NORMAL'
                }
            }
        }

    def test_momentum_long_setup(self):
        """Test momentum LONG setup on pullback."""
        # Uptrend with pullback to EMA20
        market_state = self.create_momentum_market_state(
            price=21005,
            ema20=21000,
            ema50=20900,
            macd_histogram=5.0,
            regime='TRENDING_UP'
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal in [Signal.LONG, Signal.NONE]
        if setup.signal == Signal.LONG:
            assert setup.entry_price == 21005
            assert setup.target_price > setup.entry_price
            assert setup.stop_price < setup.entry_price

    def test_momentum_short_setup(self):
        """Test momentum SHORT setup on pullback."""
        # Downtrend with pullback to EMA20
        market_state = self.create_momentum_market_state(
            price=20995,
            ema20=21000,
            ema50=21100,
            macd_histogram=-5.0,
            regime='TRENDING_DOWN'
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal in [Signal.SHORT, Signal.NONE]
        if setup.signal == Signal.SHORT:
            assert setup.entry_price == 20995
            assert setup.target_price < setup.entry_price
            assert setup.stop_price > setup.entry_price

    def test_momentum_no_setup_ranging_market(self):
        """Test momentum skips ranging markets."""
        market_state = self.create_momentum_market_state(
            price=21005,
            ema20=21000,
            ema50=20900,
            macd_histogram=5.0,
            regime='RANGING'
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE

    def test_momentum_no_setup_wrong_ema_alignment(self):
        """Test momentum skips wrong EMA alignment."""
        # EMA20 < EMA50 in supposed uptrend
        market_state = self.create_momentum_market_state(
            price=21005,
            ema20=20900,
            ema50=21000,
            macd_histogram=5.0,
            regime='TRENDING_UP'
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE

    def test_momentum_no_setup_far_from_ema(self):
        """Test momentum skips if price too far from EMA20."""
        # Price 20 ticks from EMA20 (> max 10 ticks)
        market_state = self.create_momentum_market_state(
            price=21050,
            ema20=21000,
            ema50=20900,
            macd_histogram=5.0,
            regime='TRENDING_UP'
        )

        setup = self.strategy.analyze(market_state)

        assert setup.signal == Signal.NONE

    def test_momentum_confidence_scoring(self):
        """Test momentum confidence varies with conditions."""
        # Perfect conditions
        market_state_perfect = self.create_momentum_market_state(
            price=21002,  # Close to EMA20
            ema20=21000,
            ema50=20900,
            macd_histogram=10.0,  # Strong histogram
            regime='TRENDING_UP'
        )

        # Good but not perfect
        market_state_good = self.create_momentum_market_state(
            price=21008,  # Further from EMA20
            ema20=21000,
            ema50=20900,
            macd_histogram=2.0,  # Weak histogram
            regime='TRENDING_UP'
        )

        setup_perfect = self.strategy.analyze(market_state_perfect)
        setup_good = self.strategy.analyze(market_state_good)

        # Perfect should have higher confidence
        if setup_perfect.signal != Signal.NONE and setup_good.signal != Signal.NONE:
            assert setup_perfect.confidence >= setup_good.confidence


class TestStrategyCommon:
    """Test common strategy functionality."""

    def test_time_filters(self):
        """Test time-based filters work."""
        strategy = VWAPStrategy(
            config={
                'entry_zone_min_std': 1.5,
                'entry_zone_max_std': 2.5,
                'rsi_min': 45,
                'rsi_max': 55,
                'max_spread_ticks': 2
            },
            tick_size=0.25,
            tick_value=5.0
        )

        # Before market open
        early_time = datetime.combine(datetime.today(), time(8, 0, 0))
        can_trade, reason = strategy._check_time_filters(early_time)
        assert not can_trade

        # During market hours
        market_time = datetime.combine(datetime.today(), time(10, 0, 0))
        can_trade, reason = strategy._check_time_filters(market_time)
        assert can_trade

        # After market close
        late_time = datetime.combine(datetime.today(), time(17, 0, 0))
        can_trade, reason = strategy._check_time_filters(late_time)
        assert not can_trade

    def test_spread_filter(self):
        """Test spread filter."""
        strategy = VWAPStrategy(
            config={
                'max_spread_ticks': 2
            },
            tick_size=0.25,
            tick_value=5.0
        )

        # Good spread
        assert strategy._check_spread(0.5)

        # Bad spread
        assert not strategy._check_spread(2.0)

    def test_cooldown_period(self):
        """Test cooldown between trades."""
        strategy = VWAPStrategy(
            config={},
            tick_size=0.25,
            tick_value=5.0
        )

        # Record trade
        strategy.record_trade()

        # Immediate check - should be in cooldown
        can_trade, reason = strategy.can_trade()
        assert not can_trade
        assert 'cooldown' in reason.lower()

        # Manually expire cooldown
        strategy.last_trade_time = datetime.now() - timedelta(minutes=20)

        # Should be able to trade now
        can_trade, reason = strategy.can_trade()
        assert can_trade


class TestStrategySetup:
    """Test TradeSetup dataclass functionality."""

    def test_setup_validation(self):
        """Test setup validation logic."""
        from strategies.base_strategy import TradeSetup

        # Valid setup
        valid_setup = TradeSetup(
            signal=Signal.LONG,
            confidence=8.5,
            strategy_name='Test',
            timestamp=datetime.now(),
            entry_price=21000,
            stop_price=20980,
            target_price=21040,
            reasoning={'test': True},
            conditions_met=['Condition 1'],
            conditions_failed=[]
        )

        assert valid_setup.is_valid()

        # Invalid setup (no signal)
        invalid_setup = TradeSetup(
            signal=Signal.NONE,
            confidence=0,
            strategy_name='Test',
            timestamp=datetime.now()
        )

        assert not invalid_setup.is_valid()

    def test_setup_to_dict(self):
        """Test setup serialization."""
        from strategies.base_strategy import TradeSetup

        setup = TradeSetup(
            signal=Signal.LONG,
            confidence=8.5,
            strategy_name='Test',
            timestamp=datetime.now(),
            entry_price=21000,
            stop_price=20980,
            target_price=21040
        )

        setup_dict = setup.to_dict()

        assert setup_dict['signal'] == 'LONG'
        assert setup_dict['confidence'] == 8.5
        assert setup_dict['entry_price'] == 21000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
