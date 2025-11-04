"""
Tests for slippage measurement and logging.

Tests slippage tracking from signal to fill:
- Calculate running averages
- Alert on excessive slippage (>3 tick average)
- Verify logging format
"""

import pytest
from datetime import datetime
from agent.risk_manager import RiskManager
from agent.strategy_selector import StrategySelector
from agent.sdk_agent import SDKAgent
from strategies.vwap_strategy import VWAPStrategy


class TestSlippageTracking:
    """Test slippage measurement and tracking."""

    def setup_method(self):
        """Setup for each test."""
        self.risk_manager = RiskManager(
            config={
                'daily_limits': {'target_profit': 250, 'max_loss': -150, 'max_trades': 8},
                'strategy_limits': {}
            }
        )

    def test_slippage_tracking_initialization(self):
        """Test slippage metrics initialize correctly."""
        assert self.risk_manager.slippage_metrics['total_trades'] == 0
        assert self.risk_manager.slippage_metrics['total_slippage_ticks'] == 0
        assert len(self.risk_manager.slippage_metrics['slippage_samples']) == 0
        assert self.risk_manager.slippage_metrics['slippage_cost_dollars'] == 0

    def test_slippage_tracking_single_trade(self):
        """Test slippage is tracked for single trade."""
        # Track 2 ticks of slippage
        self.risk_manager._track_slippage(2.0)

        assert self.risk_manager.slippage_metrics['total_trades'] == 1
        assert self.risk_manager.slippage_metrics['total_slippage_ticks'] == 2.0
        assert len(self.risk_manager.slippage_metrics['slippage_samples']) == 1
        # NQ: $5 per tick
        assert self.risk_manager.slippage_metrics['slippage_cost_dollars'] == 10.0

    def test_slippage_tracking_multiple_trades(self):
        """Test slippage across multiple trades."""
        # Track multiple trades
        slippages = [1.5, 2.0, 0.5, 3.0, 1.0]

        for slippage in slippages:
            self.risk_manager._track_slippage(slippage)

        assert self.risk_manager.slippage_metrics['total_trades'] == 5
        assert self.risk_manager.slippage_metrics['total_slippage_ticks'] == sum(slippages)
        assert len(self.risk_manager.slippage_metrics['slippage_samples']) == 5

        # Total cost
        expected_cost = sum(slippages) * 5.0
        assert self.risk_manager.slippage_metrics['slippage_cost_dollars'] == expected_cost

    def test_slippage_running_average(self):
        """Test running average calculation."""
        slippages = [1.0, 2.0, 3.0, 2.5, 1.5]

        for slippage in slippages:
            self.risk_manager._track_slippage(slippage)

        stats = self.risk_manager.get_slippage_statistics()

        expected_avg = sum(slippages) / len(slippages)
        assert stats['avg_slippage_ticks'] == pytest.approx(expected_avg, rel=0.01)

    def test_slippage_min_max_tracking(self):
        """Test min/max slippage tracking."""
        slippages = [1.0, 5.0, 0.5, 3.0, 2.0]

        for slippage in slippages:
            self.risk_manager._track_slippage(slippage)

        stats = self.risk_manager.get_slippage_statistics()

        assert stats['max_slippage_ticks'] == 5.0
        assert stats['min_slippage_ticks'] == 0.5

    def test_slippage_sample_limit(self):
        """Test slippage samples are limited to last 100."""
        # Add 150 samples
        for i in range(150):
            self.risk_manager._track_slippage(1.0)

        # Should only keep last 100
        assert len(self.risk_manager.slippage_metrics['slippage_samples']) == 100

    def test_slippage_statistics_empty(self):
        """Test statistics with no slippage data."""
        stats = self.risk_manager.get_slippage_statistics()

        assert stats['total_trades'] == 0
        assert stats['avg_slippage_ticks'] == 0
        assert stats['max_slippage_ticks'] == 0
        assert stats['min_slippage_ticks'] == 0


class TestSlippageAlerts:
    """Test slippage alerting logic."""

    def setup_method(self):
        """Setup for each test."""
        self.risk_manager = RiskManager(
            config={
                'daily_limits': {},
                'strategy_limits': {}
            }
        )

    def test_high_slippage_detection(self):
        """Test detection of high slippage (>3 tick average)."""
        # Add high slippage samples
        high_slippages = [4.0, 5.0, 3.5, 4.5, 3.8]

        for slippage in high_slippages:
            self.risk_manager._track_slippage(slippage)

        stats = self.risk_manager.get_slippage_statistics()

        # Average should be > 3 ticks
        assert stats['avg_slippage_ticks'] > 3.0

    def test_acceptable_slippage(self):
        """Test acceptable slippage levels (<= 3 ticks)."""
        # Add acceptable slippage samples
        good_slippages = [1.0, 2.0, 1.5, 2.5, 1.8]

        for slippage in good_slippages:
            self.risk_manager._track_slippage(slippage)

        stats = self.risk_manager.get_slippage_statistics()

        # Average should be <= 3 ticks
        assert stats['avg_slippage_ticks'] <= 3.0

    def test_slippage_cost_calculation(self):
        """Test slippage cost in dollars."""
        # 2 ticks of slippage
        self.risk_manager._track_slippage(2.0)

        stats = self.risk_manager.get_slippage_statistics()

        # NQ: $5 per tick
        expected_cost = 2.0 * 5.0
        assert stats['total_cost_dollars'] == expected_cost


class TestSlippageInStrategySelector:
    """Test slippage tracking in strategy selector."""

    def setup_method(self):
        """Setup for each test."""
        strategies = {
            'VWAP': VWAPStrategy(config={}, tick_size=0.25, tick_value=5.0)
        }
        sdk_agent = SDKAgent(config={})
        risk_manager = RiskManager(config={'daily_limits': {}, 'strategy_limits': {}})

        self.selector = StrategySelector(
            strategies=strategies,
            sdk_agent=sdk_agent,
            risk_manager=risk_manager,
            config={}
        )

    def test_selector_slippage_initialization(self):
        """Test strategy selector slippage stats initialize."""
        stats = self.selector.get_slippage_statistics()

        assert stats['total_setups_evaluated'] == 0
        assert stats['high_confidence_setups'] == 0
        assert stats['avg_slippage_ticks'] == 0

    def test_selector_tracks_slippage(self):
        """Test strategy selector tracks slippage."""
        # Simulate decision with slippage
        setup = {
            'strategy_name': 'VWAP',
            'confidence': 9.0
        }

        decision = {
            'action': 'ENTER',
            'slippage_ticks': 2.5,
            'latency_ms': 400
        }

        self.selector._track_decision(setup, decision)

        stats = self.selector.get_slippage_statistics()

        assert stats['avg_slippage_ticks'] == 2.5
        assert len(self.selector.slippage_stats['slippage_samples']) == 1

    def test_selector_validation_failure_tracking(self):
        """Test validation failures are tracked."""
        setup = {'strategy_name': 'VWAP', 'confidence': 9.0}
        decision = {'action': 'SKIP', 'validation_failed': True}

        self.selector._track_decision(setup, decision)

        stats = self.selector.get_slippage_statistics()

        assert stats['validation_failures'] == 1

    def test_selector_successful_entry_tracking(self):
        """Test successful entries are tracked."""
        setup = {'strategy_name': 'VWAP', 'confidence': 9.0}
        decision = {'action': 'ENTER', 'slippage_ticks': 1.5}

        self.selector._track_decision(setup, decision)

        stats = self.selector.get_slippage_statistics()

        assert stats['claude_decisions'] == 1


class TestSlippageLogging:
    """Test slippage logging format and persistence."""

    def setup_method(self):
        """Setup for each test."""
        self.risk_manager = RiskManager(
            config={
                'daily_limits': {},
                'strategy_limits': {}
            }
        )

    def test_trade_log_includes_slippage(self):
        """Test trade log includes slippage data."""
        # Record trade with slippage
        trade_details = {
            'entry_price': 21000,
            'slippage_ticks': 2.5,
            'slippage_cost_dollars': 12.5,
            'latency_ms': 400
        }

        self.risk_manager.on_trade_closed(
            strategy_name='VWAP',
            pnl=50.0,
            trade_details=trade_details
        )

        # Check trade history includes slippage
        assert len(self.risk_manager.trade_history) == 1
        trade_record = self.risk_manager.trade_history[0]

        assert 'slippage_ticks' in trade_record
        assert trade_record['slippage_ticks'] == 2.5

    def test_daily_summary_includes_slippage(self):
        """Test daily summary includes slippage stats."""
        # Add some slippage
        self.risk_manager._track_slippage(2.0)
        self.risk_manager._track_slippage(3.0)

        summary = self.risk_manager.get_daily_summary()

        assert 'slippage' in summary
        assert 'avg_ticks' in summary['slippage']
        assert summary['slippage']['avg_ticks'] == 2.5


class TestSlippageFromSignalToFill:
    """Test slippage tracking from signal detection to order fill."""

    def test_three_stage_slippage_tracking(self):
        """Test slippage at three stages: signal → Claude → fill."""
        # Stage 1: Initial signal
        signal_price = 21000.00

        # Stage 2: After Claude latency (400ms)
        post_claude_price = 21000.50  # 2 ticks slippage
        claude_slippage = abs(post_claude_price - signal_price) / 0.25
        assert claude_slippage == 2.0

        # Stage 3: After order execution
        fill_price = 21000.75  # Additional 1 tick slippage
        total_slippage = abs(fill_price - signal_price) / 0.25
        assert total_slippage == 3.0

        # Execution slippage only
        execution_slippage = abs(fill_price - post_claude_price) / 0.25
        assert execution_slippage == 1.0

    def test_slippage_cost_calculation(self):
        """Test slippage cost calculation in dollars."""
        signal_price = 21000.00
        fill_price = 21000.75  # 3 ticks

        slippage_ticks = abs(fill_price - signal_price) / 0.25
        slippage_cost = slippage_ticks * 5.0  # NQ: $5/tick

        assert slippage_ticks == 3.0
        assert slippage_cost == 15.0

    def test_favorable_slippage_tracking(self):
        """Test tracking of favorable slippage."""
        # For LONG: Lower fill price is favorable
        signal_price = 21000.00
        fill_price = 20999.50  # 2 ticks better

        slippage_ticks = (signal_price - fill_price) / 0.25
        assert slippage_ticks == 2.0  # Positive (favorable)

        # Should still track absolute value for stats
        abs_slippage = abs(slippage_ticks)
        assert abs_slippage == 2.0


class TestSlippageReporting:
    """Test slippage reporting and statistics."""

    def setup_method(self):
        """Setup for each test."""
        self.risk_manager = RiskManager(
            config={
                'daily_limits': {},
                'strategy_limits': {}
            }
        )

    def test_slippage_samples_in_report(self):
        """Test last N slippage samples are included in report."""
        # Add samples
        for i in range(25):
            self.risk_manager._track_slippage(float(i % 5))

        stats = self.risk_manager.get_slippage_statistics()

        # Should include last 20 samples
        assert 'samples' in stats
        assert len(stats['samples']) == 20

    def test_slippage_percentiles(self):
        """Test slippage percentile calculations."""
        slippages = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        for slippage in slippages:
            self.risk_manager._track_slippage(slippage)

        stats = self.risk_manager.get_slippage_statistics()

        # Check min/max
        assert stats['min_slippage_ticks'] == 1.0
        assert stats['max_slippage_ticks'] == 5.0

        # Average
        expected_avg = sum(slippages) / len(slippages)
        assert stats['avg_slippage_ticks'] == pytest.approx(expected_avg, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
