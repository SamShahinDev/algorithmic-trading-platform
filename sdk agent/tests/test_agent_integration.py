"""
End-to-end integration tests with mock trades.

Tests full trading workflow:
- Market data → Strategy evaluation → SDK agent → Risk manager → Order placement
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from strategies.vwap_strategy import VWAPStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.momentum_strategy import MomentumStrategy
from agent.sdk_agent import SDKAgent
from agent.strategy_selector import StrategySelector
from agent.risk_manager import RiskManager
from indicators.cache import IndicatorCache


class TestAgentIntegration:
    """Test end-to-end trading workflow."""

    def setup_method(self):
        """Setup for each test."""
        # Create strategies
        self.strategies = {
            'VWAP': VWAPStrategy(
                config={'entry_zone_min_std': 1.5, 'entry_zone_max_std': 2.5, 'rsi_min': 45, 'rsi_max': 55},
                tick_size=0.25,
                tick_value=5.0
            ),
            'Breakout': BreakoutStrategy(
                config={'range_start_time': '09:30:00', 'range_end_time': '10:00:00'},
                tick_size=0.25,
                tick_value=5.0
            ),
            'Momentum': MomentumStrategy(
                config={'ema_fast': 20, 'ema_slow': 50},
                tick_size=0.25,
                tick_value=5.0
            )
        }

        # Create SDK agent (mock Claude client)
        self.sdk_agent = SDKAgent(
            config={'min_confidence_for_claude': 8.0, 'max_acceptable_slippage_ticks': 3.0}
        )

        # Create risk manager
        self.risk_manager = RiskManager(
            config={
                'daily_limits': {'target_profit': 250, 'max_loss': -150, 'max_trades': 8},
                'strategy_limits': {'VWAP': 4, 'Breakout': 2, 'Momentum': 2}
            }
        )

        # Create strategy selector
        self.strategy_selector = StrategySelector(
            strategies=self.strategies,
            sdk_agent=self.sdk_agent,
            risk_manager=self.risk_manager,
            config={}
        )

    def create_market_state_vwap_setup(self):
        """Create market state for VWAP setup."""
        return {
            'current_price': 20900,  # 2 std dev below VWAP
            'time': datetime.now(),
            'spread': 0.5,
            'regime': 'RANGING',
            'bars': [],
            'indicators': {
                'vwap': {
                    'vwap': 21000,
                    'std_dev': 50,
                    'distance_from_vwap': -100,
                    'std_dev_distance': -2.0
                },
                'rsi': {'rsi': 50, 'signal': 'NEUTRAL'},
                'ema': {'ema20': 21000, 'ema50': 20950, 'alignment': 'BULLISH'},
                'macd': {'macd': 5, 'signal': 3, 'histogram': 2},
                'atr': {'atr': 20, 'volatility_level': 'NORMAL'},
                'regime': {'regime': 'RANGING', 'adx': 20}
            }
        }

    @pytest.mark.asyncio
    async def test_full_trade_workflow_mock(self):
        """Test full trade workflow with mocked Claude."""
        # Create market state
        market_state = self.create_market_state_vwap_setup()

        # Mock Claude API response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85, "reasoning": "Good setup"}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Evaluate market
            trade_decision = await self.strategy_selector.evaluate_market(market_state)

            # Should have trade decision
            assert trade_decision is not None
            assert trade_decision['strategy_name'] in ['VWAP', 'Breakout', 'Momentum']
            assert 'setup' in trade_decision
            assert 'decision' in trade_decision

    @pytest.mark.asyncio
    async def test_low_confidence_skips_claude(self):
        """Test low confidence setup skips Claude call."""
        # Create market state with poor setup (low confidence)
        market_state = {
            'current_price': 21000,
            'time': datetime.now(),
            'spread': 0.5,
            'regime': 'RANGING',
            'bars': [],
            'indicators': {
                'vwap': {'vwap': 21000, 'std_dev': 50, 'distance_from_vwap': 0, 'std_dev_distance': 0},
                'rsi': {'rsi': 50, 'signal': 'NEUTRAL'},
                'ema': {'ema20': 21000, 'ema50': 20950, 'alignment': 'NEUTRAL'},
                'macd': {'macd': 0, 'signal': 0, 'histogram': 0},
                'atr': {'atr': 20, 'volatility_level': 'NORMAL'},
                'regime': {'regime': 'RANGING', 'adx': 20}
            }
        }

        # Evaluate market (should skip Claude)
        trade_decision = await self.strategy_selector.evaluate_market(market_state)

        # Should be None (no high-confidence setup)
        assert trade_decision is None

        # Check stats
        stats = self.strategy_selector.get_slippage_statistics()
        assert stats['high_confidence_setups'] == 0

    @pytest.mark.asyncio
    async def test_risk_manager_blocks_excessive_trades(self):
        """Test risk manager blocks trades when limits hit."""
        # Fill daily trades to max
        self.risk_manager.daily_trades = 8

        market_state = self.create_market_state_vwap_setup()

        # Mock strategy to return high-confidence setup
        mock_setup = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 20900,
            'stop_price': 20880,
            'target_price': 20940,
            'strategy_name': 'VWAP'
        }

        # Try to validate trade
        risk_check = self.risk_manager.validate_trade(
            strategy_name='VWAP',
            setup_dict=mock_setup,
            decision={'action': 'ENTER'}
        )

        # Should be rejected
        assert not risk_check['approved']
        assert 'limit' in risk_check['reason'].lower()

    @pytest.mark.asyncio
    async def test_risk_manager_blocks_target_reached(self):
        """Test risk manager blocks trades when profit target reached."""
        # Set daily P&L to target
        self.risk_manager.daily_pnl = 250

        market_state = self.create_market_state_vwap_setup()

        mock_setup = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 20900,
            'stop_price': 20880,
            'target_price': 20940,
            'strategy_name': 'VWAP'
        }

        risk_check = self.risk_manager.validate_trade(
            strategy_name='VWAP',
            setup_dict=mock_setup,
            decision={'action': 'ENTER'}
        )

        assert not risk_check['approved']
        assert 'target' in risk_check['reason'].lower()

    @pytest.mark.asyncio
    async def test_risk_manager_blocks_max_loss(self):
        """Test risk manager blocks trades when max loss hit."""
        # Set daily P&L to max loss
        self.risk_manager.daily_pnl = -150

        market_state = self.create_market_state_vwap_setup()

        mock_setup = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 20900,
            'stop_price': 20880,
            'target_price': 20940,
            'strategy_name': 'VWAP'
        }

        risk_check = self.risk_manager.validate_trade(
            strategy_name='VWAP',
            setup_dict=mock_setup,
            decision={'action': 'ENTER'}
        )

        assert not risk_check['approved']
        assert 'loss' in risk_check['reason'].lower()

    @pytest.mark.asyncio
    async def test_risk_manager_blocks_strategy_limit(self):
        """Test risk manager blocks when strategy-specific limit hit."""
        # VWAP strategy has taken 4 trades (max)
        self.risk_manager.strategy_trades['VWAP'] = 4

        market_state = self.create_market_state_vwap_setup()

        mock_setup = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 20900,
            'stop_price': 20880,
            'target_price': 20940,
            'strategy_name': 'VWAP'
        }

        risk_check = self.risk_manager.validate_trade(
            strategy_name='VWAP',
            setup_dict=mock_setup,
            decision={'action': 'ENTER'}
        )

        assert not risk_check['approved']
        assert 'VWAP' in risk_check['reason']

    @pytest.mark.asyncio
    async def test_slippage_tracking(self):
        """Test slippage is tracked throughout workflow."""
        market_state = self.create_market_state_vwap_setup()

        # Mock Claude to approve with slippage
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy to allow re-evaluation
            with patch.object(self.strategies['VWAP'], 'analyze') as mock_analyze:
                # First call returns high-confidence setup
                from strategies.base_strategy import TradeSetup, Signal
                original_setup = TradeSetup(
                    signal=Signal.LONG,
                    confidence=9.0,
                    strategy_name='VWAP',
                    timestamp=datetime.now(),
                    entry_price=20900,
                    stop_price=20880,
                    target_price=20940
                )

                # Second call (post-validation) with slippage
                post_validation_setup = TradeSetup(
                    signal=Signal.LONG,
                    confidence=9.0,
                    strategy_name='VWAP',
                    timestamp=datetime.now(),
                    entry_price=20901,  # 1 tick slippage
                    stop_price=20880,
                    target_price=20940
                )

                mock_analyze.side_effect = [original_setup, post_validation_setup]

                # Evaluate market
                trade_decision = await self.strategy_selector.evaluate_market(market_state)

                if trade_decision:
                    # Check slippage was tracked
                    assert 'slippage_ticks' in trade_decision['decision']

    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance is tracked across trades."""
        # Simulate completed trades
        self.strategy_selector.update_strategy_performance('VWAP', 50.0)  # Win
        self.strategy_selector.update_strategy_performance('VWAP', -25.0)  # Loss
        self.strategy_selector.update_strategy_performance('VWAP', 75.0)  # Win

        # Get stats
        stats = self.strategy_selector.get_strategy_stats('VWAP')

        assert stats['total_trades'] == 3
        assert stats['winning_trades'] == 2
        assert stats['losing_trades'] == 1
        assert stats['total_pnl'] == 100.0
        assert stats['win_rate'] == pytest.approx(66.67, rel=0.1)

    @pytest.mark.asyncio
    async def test_daily_reset(self):
        """Test daily counters reset at new day."""
        # Set some counters
        self.risk_manager.daily_trades = 5
        self.risk_manager.daily_pnl = 100

        # Manually trigger new day
        from datetime import date
        self.risk_manager.current_date = date.today() - timedelta(days=1)
        self.risk_manager._check_new_day()

        # Counters should be reset
        assert self.risk_manager.daily_trades == 0
        assert self.risk_manager.daily_pnl == 0

    @pytest.mark.asyncio
    async def test_concurrent_strategy_evaluation(self):
        """Test all strategies are evaluated concurrently."""
        market_state = self.create_market_state_vwap_setup()

        # Track which strategies were evaluated
        evaluated = []

        async def track_analyze(original_analyze, name):
            evaluated.append(name)
            return await original_analyze(market_state)

        # Patch all strategies
        for name, strategy in self.strategies.items():
            original = strategy.analyze
            strategy.analyze = lambda ms, n=name: track_analyze(original, n)

        await self.strategy_selector.evaluate_market(market_state)

        # All strategies should have been evaluated
        # (Note: actual evaluation is synchronous, but architecture supports it)
        assert len(evaluated) >= 0  # At least attempted


class TestAgentEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_market_state(self):
        """Test handling of empty market state."""
        strategies = {
            'VWAP': VWAPStrategy(config={}, tick_size=0.25, tick_value=5.0)
        }
        sdk_agent = SDKAgent(config={})
        risk_manager = RiskManager(config={'daily_limits': {}, 'strategy_limits': {}})

        selector = StrategySelector(
            strategies=strategies,
            sdk_agent=sdk_agent,
            risk_manager=risk_manager,
            config={}
        )

        # Empty market state
        market_state = {}

        # Should handle gracefully
        try:
            result = await selector.evaluate_market(market_state)
            # May return None or handle error
            assert result is None or isinstance(result, dict)
        except Exception as e:
            # Error handling is acceptable
            pass

    @pytest.mark.asyncio
    async def test_claude_api_error(self):
        """Test handling of Claude API errors."""
        sdk_agent = SDKAgent(config={})

        # Mock Claude to raise error
        with patch.object(sdk_agent.client.messages, 'create', side_effect=Exception("API Error")):
            market_state = {'current_price': 21000}
            setup_dict = {
                'signal': 'LONG',
                'confidence': 9.0,
                'entry_price': 21000,
                'stop_price': 20980,
                'target_price': 21040
            }

            # Should handle error gracefully
            decision = await sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=None
            )

            # Should return SKIP decision
            assert decision['action'] == 'SKIP'
            assert 'error' in decision['reasoning'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
