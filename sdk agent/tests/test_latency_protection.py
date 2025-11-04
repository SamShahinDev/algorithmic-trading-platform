"""
Tests for latency protection mechanisms.

Tests pre-filter logic, post-validation, and slippage handling:
- Pre-filter: Score < 8 → Skip Claude
- Post-validation: Re-check setup after Claude response
- Slippage: Max 3 ticks acceptable
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch
from agent.sdk_agent import SDKAgent
from strategies.vwap_strategy import VWAPStrategy
from strategies.base_strategy import TradeSetup, Signal


class TestPreFilter:
    """Test pre-filter logic that skips low-confidence setups."""

    def setup_method(self):
        """Setup for each test."""
        self.sdk_agent = SDKAgent(
            config={
                'min_confidence_for_claude': 8.0,
                'max_acceptable_slippage_ticks': 3.0
            }
        )

    @pytest.mark.asyncio
    async def test_prefilter_low_confidence_skips_claude(self):
        """Test setup with score < 8 skips Claude call."""
        # Low confidence setup (7/10)
        setup_dict = {
            'signal': 'LONG',
            'confidence': 7.0,  # Below threshold
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Evaluate (should skip Claude)
        decision = await self.sdk_agent.evaluate_setup(
            strategy_name='VWAP',
            setup_dict=setup_dict,
            market_state=market_state,
            performance_today={},
            strategy_instance=None
        )

        # Should be skipped
        assert decision['action'] == 'SKIP'
        assert 'pre-filter' in decision['reasoning'].lower()

        # Claude should NOT have been called
        stats = self.sdk_agent.get_statistics()
        assert stats['pre_filtered'] == 1
        assert stats['claude_calls'] == 0

    @pytest.mark.asyncio
    async def test_prefilter_high_confidence_calls_claude(self):
        """Test setup with score >= 8 calls Claude."""
        # High confidence setup (9/10)
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,  # Above threshold
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy instance for post-validation
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000,
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Claude should have been called
            stats = self.sdk_agent.get_statistics()
            assert stats['claude_calls'] >= 1

    @pytest.mark.asyncio
    async def test_prefilter_threshold_boundary(self):
        """Test pre-filter at exact threshold (8.0)."""
        # Exactly at threshold
        setup_dict = {
            'signal': 'LONG',
            'confidence': 8.0,  # Exactly at threshold
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=8.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000,
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Should call Claude (>= threshold)
            stats = self.sdk_agent.get_statistics()
            assert stats['claude_calls'] >= 1


class TestPostValidation:
    """Test post-validation after Claude response."""

    def setup_method(self):
        """Setup for each test."""
        self.sdk_agent = SDKAgent(
            config={
                'min_confidence_for_claude': 8.0,
                'max_acceptable_slippage_ticks': 3.0
            }
        )

    @pytest.mark.asyncio
    async def test_postvalidation_setup_still_valid(self):
        """Test post-validation passes when setup still valid."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude to approve
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy to return same setup on re-evaluation
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000.25,  # Minor slippage (1 tick)
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Should pass validation
            assert decision['action'] == 'ENTER'
            assert decision.get('validation_passed') == True
            assert 'slippage_ticks' in decision

    @pytest.mark.asyncio
    async def test_postvalidation_setup_degraded(self):
        """Test post-validation fails when setup degraded."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude to approve
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy to return NO setup on re-evaluation (degraded)
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.NONE,  # Setup degraded
                confidence=0,
                strategy_name='VWAP',
                timestamp=datetime.now()
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Should skip due to degradation
            assert decision['action'] == 'SKIP'
            assert 'degraded' in decision['reasoning'].lower() or 'invalid' in decision['reasoning'].lower()
            assert decision.get('validation_failed') == True

    @pytest.mark.asyncio
    async def test_postvalidation_excessive_slippage(self):
        """Test post-validation fails with excessive slippage (>3 ticks)."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude to approve
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy to return setup with excessive slippage
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21001.0,  # 4 ticks slippage (> 3 tick limit)
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Should skip due to excessive slippage
            assert decision['action'] == 'SKIP'
            assert 'slippage' in decision['reasoning'].lower()
            assert decision.get('slippage_ticks', 0) > 3

    @pytest.mark.asyncio
    async def test_postvalidation_acceptable_slippage(self):
        """Test post-validation passes with acceptable slippage (<=3 ticks)."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude to approve
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy with acceptable slippage (2 ticks)
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000.50,  # 2 ticks slippage (acceptable)
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Should pass
            assert decision['action'] == 'ENTER'
            assert decision.get('validation_passed') == True
            assert decision.get('slippage_ticks', 0) <= 3


class TestLatencyMeasurement:
    """Test latency measurement and tracking."""

    def setup_method(self):
        """Setup for each test."""
        self.sdk_agent = SDKAgent(
            config={
                'min_confidence_for_claude': 8.0,
                'max_acceptable_slippage_ticks': 3.0
            }
        )

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test Claude API latency is measured."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude with artificial delay
        async def mock_create_with_delay(*args, **kwargs):
            await asyncio.sleep(0.4)  # 400ms delay
            mock_response = Mock()
            mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]
            return mock_response

        with patch.object(self.sdk_agent.client.messages, 'create', side_effect=mock_create_with_delay):
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000,
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Latency should be measured
            assert 'latency_ms' in decision
            assert decision['latency_ms'] >= 400  # At least 400ms

    @pytest.mark.asyncio
    async def test_latency_statistics(self):
        """Test latency statistics are tracked."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000,
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            # Make multiple calls
            for i in range(3):
                await self.sdk_agent.evaluate_setup(
                    strategy_name='VWAP',
                    setup_dict=setup_dict,
                    market_state=market_state,
                    performance_today={},
                    strategy_instance=mock_strategy
                )

            # Check stats
            stats = self.sdk_agent.get_statistics()
            assert stats['claude_calls'] == 3
            assert stats['avg_latency_ms'] > 0


class TestSlippageCalculation:
    """Test slippage calculation accuracy."""

    def setup_method(self):
        """Setup for each test."""
        self.sdk_agent = SDKAgent(
            config={
                'min_confidence_for_claude': 8.0,
                'max_acceptable_slippage_ticks': 3.0
            }
        )
        self.sdk_agent.tick_size = 0.25

    @pytest.mark.asyncio
    async def test_slippage_calculation_long(self):
        """Test slippage calculation for LONG position."""
        original_entry = 21000.00
        current_entry = 21000.75  # 3 ticks higher

        slippage_ticks = (current_entry - original_entry) / self.sdk_agent.tick_size

        assert slippage_ticks == 3.0

    @pytest.mark.asyncio
    async def test_slippage_calculation_short(self):
        """Test slippage calculation for SHORT position."""
        original_entry = 21000.00
        current_entry = 20999.25  # 3 ticks lower

        # For SHORT, favorable slippage is negative
        slippage_ticks = abs(current_entry - original_entry) / self.sdk_agent.tick_size

        assert slippage_ticks == 3.0

    @pytest.mark.asyncio
    async def test_slippage_zero(self):
        """Test zero slippage."""
        original_entry = 21000.00
        current_entry = 21000.00  # No change

        slippage_ticks = abs(current_entry - original_entry) / self.sdk_agent.tick_size

        assert slippage_ticks == 0.0

    @pytest.mark.asyncio
    async def test_slippage_fractional_tick(self):
        """Test slippage with fractional tick."""
        original_entry = 21000.00
        current_entry = 21000.60  # 2.4 ticks

        slippage_ticks = abs(current_entry - original_entry) / self.sdk_agent.tick_size

        assert slippage_ticks == pytest.approx(2.4, rel=0.01)


class TestValidationStatistics:
    """Test validation success/failure tracking."""

    def setup_method(self):
        """Setup for each test."""
        self.sdk_agent = SDKAgent(
            config={
                'min_confidence_for_claude': 8.0,
                'max_acceptable_slippage_ticks': 3.0
            }
        )

    @pytest.mark.asyncio
    async def test_validation_success_tracking(self):
        """Test successful validations are tracked."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.LONG,
                confidence=9.0,
                strategy_name='VWAP',
                timestamp=datetime.now(),
                entry_price=21000.25,  # Minor slippage
                stop_price=20980,
                target_price=21040
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Check stats
            stats = self.sdk_agent.get_statistics()
            assert stats['validation_successes'] >= 1

    @pytest.mark.asyncio
    async def test_validation_failure_tracking(self):
        """Test failed validations are tracked."""
        setup_dict = {
            'signal': 'LONG',
            'confidence': 9.0,
            'entry_price': 21000,
            'stop_price': 20980,
            'target_price': 21040,
            'strategy_name': 'VWAP'
        }

        market_state = {'current_price': 21000}

        # Mock Claude
        mock_response = Mock()
        mock_response.content = [Mock(text='{"action": "ENTER", "confidence": 0.85}')]

        with patch.object(self.sdk_agent.client.messages, 'create', return_value=mock_response):
            # Mock strategy to fail validation
            mock_strategy = Mock()
            mock_setup = TradeSetup(
                signal=Signal.NONE,  # Degraded
                confidence=0,
                strategy_name='VWAP',
                timestamp=datetime.now()
            )
            mock_strategy.analyze = Mock(return_value=mock_setup)

            decision = await self.sdk_agent.evaluate_setup(
                strategy_name='VWAP',
                setup_dict=setup_dict,
                market_state=market_state,
                performance_today={},
                strategy_instance=mock_strategy
            )

            # Check stats
            stats = self.sdk_agent.get_statistics()
            assert stats['validation_failures'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
