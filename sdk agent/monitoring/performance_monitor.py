"""
Performance Monitor - Real-time performance tracking and analysis.

Monitors:
- Trading performance metrics
- Slippage trends
- Latency patterns
- Risk metrics
- Strategy performance
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from loggers.decision_logger import DecisionLogger
from loggers.trade_logger import TradeLogger
from loggers.slippage_logger import SlippageLogger


class PerformanceMonitor:
    """
    Real-time performance monitoring and analysis.

    Tracks and analyzes:
    - P&L and trade statistics
    - Slippage trends and patterns
    - Claude API latency
    - Strategy performance comparison
    - Risk limit adherence
    """

    def __init__(
        self,
        log_dir: Path = None,
        slippage_threshold: float = 3.0,
        latency_threshold_ms: float = 500.0
    ):
        """
        Initialize performance monitor.

        Args:
            log_dir: Directory containing log files
            slippage_threshold: Alert threshold for slippage (ticks)
            latency_threshold_ms: Alert threshold for latency (milliseconds)
        """
        self.log_dir = log_dir or Path('logs')
        self.slippage_threshold = slippage_threshold
        self.latency_threshold = latency_threshold_ms

        # Initialize loggers
        self.decision_logger = DecisionLogger(log_dir=self.log_dir)
        self.trade_logger = TradeLogger(log_dir=self.log_dir)
        self.slippage_logger = SlippageLogger(
            log_dir=self.log_dir,
            alert_threshold_ticks=slippage_threshold
        )

        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.session_start = datetime.now()
        self.alerts: List[Dict[str, Any]] = []

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get current real-time metrics.

        Returns:
            Dictionary with current performance metrics
        """
        # Get daily trade stats
        trade_stats = self.trade_logger.get_daily_statistics()

        # Get decision stats
        decision_stats = self.decision_logger.get_statistics()

        # Get slippage stats
        slippage_stats = self.slippage_logger.get_statistics()

        # Get open trades
        open_trades = self.trade_logger.get_open_trades()

        return {
            'session_duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60,
            'trading': {
                'total_trades': trade_stats['total_trades'],
                'winning_trades': trade_stats['winning_trades'],
                'losing_trades': trade_stats['losing_trades'],
                'win_rate': trade_stats['win_rate'],
                'total_pnl': trade_stats['total_pnl'],
                'avg_pnl': trade_stats['avg_pnl'],
                'largest_winner': trade_stats['avg_winner'],
                'largest_loser': trade_stats['avg_loser'],
                'open_trades_count': len(open_trades)
            },
            'decisions': {
                'total_decisions': decision_stats['total_decisions'],
                'pre_filter_pass_rate': decision_stats['pre_filter_pass_rate'],
                'claude_approval_rate': decision_stats['claude_approval_rate'],
                'post_validation_pass_rate': decision_stats['post_validation_pass_rate'],
                'total_claude_calls': decision_stats['total_claude_calls'],
                'total_executions': decision_stats['total_executions']
            },
            'latency': {
                'avg_claude_latency_ms': decision_stats['avg_claude_latency_ms'],
                'threshold_ms': self.latency_threshold,
                'threshold_exceeded': decision_stats['avg_claude_latency_ms'] > self.latency_threshold
            },
            'slippage': {
                'avg_slippage_ticks': slippage_stats['avg_slippage_ticks'],
                'max_slippage_ticks': slippage_stats['max_slippage_ticks'],
                'min_slippage_ticks': slippage_stats['min_slippage_ticks'],
                'total_cost_dollars': slippage_stats['total_cost_dollars'],
                'running_avg_20': slippage_stats['running_avg_20'],
                'threshold_ticks': slippage_stats['alert_threshold'],
                'threshold_exceeded': slippage_stats['threshold_exceeded']
            },
            'alerts': self.get_active_alerts()
        }

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-strategy performance comparison.

        Returns:
            Dictionary with performance by strategy
        """
        # Get all recent trades
        trades = self.trade_logger.get_recent_trades(count=1000)

        strategy_stats = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_slippage_ticks': 0.0,
            'total_duration_seconds': 0
        })

        # Aggregate by strategy
        for trade in trades:
            if trade.get('status') != 'CLOSED':
                continue

            strategy = trade.get('strategy', 'Unknown')
            stats = strategy_stats[strategy]

            stats['total_trades'] += 1

            pnl = trade.get('pnl_net', 0)
            if pnl > 0:
                stats['winning_trades'] += 1
            elif pnl < 0:
                stats['losing_trades'] += 1

            stats['total_pnl'] += pnl

            if 'entry_slippage_ticks' in trade:
                stats['total_slippage_ticks'] += trade['entry_slippage_ticks']
            if 'exit_slippage_ticks' in trade:
                stats['total_slippage_ticks'] += trade['exit_slippage_ticks']

            if 'duration_seconds' in trade:
                stats['total_duration_seconds'] += trade['duration_seconds']

        # Calculate averages and percentages
        result = {}
        for strategy, stats in strategy_stats.items():
            total = stats['total_trades']
            if total > 0:
                result[strategy] = {
                    'total_trades': total,
                    'win_rate': round(stats['winning_trades'] / total * 100, 2),
                    'total_pnl': round(stats['total_pnl'], 2),
                    'avg_pnl_per_trade': round(stats['total_pnl'] / total, 2),
                    'avg_slippage_ticks': round(stats['total_slippage_ticks'] / total, 2),
                    'avg_duration_minutes': round(stats['total_duration_seconds'] / total / 60, 1)
                }

        return result

    def get_slippage_analysis(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Analyze slippage patterns over a time window.

        Args:
            window_minutes: Time window to analyze

        Returns:
            Dictionary with slippage analysis
        """
        events = self.slippage_logger.get_recent_slippage_events(count=1000)

        if not events:
            return {
                'total_events': 0,
                'avg_slippage': 0,
                'trend': 'UNKNOWN',
                'by_type': {}
            }

        # Filter to time window
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_events = [
            e for e in events
            if datetime.fromisoformat(e['timestamp']) >= cutoff_time
        ]

        if not recent_events:
            return {
                'total_events': 0,
                'avg_slippage': 0,
                'trend': 'UNKNOWN',
                'by_type': {}
            }

        # Calculate metrics
        slippages = [e['slippage_ticks'] for e in recent_events]
        avg_slippage = sum(slippages) / len(slippages)

        # Analyze by type
        by_type = defaultdict(list)
        for event in recent_events:
            by_type[event['type']].append(event['slippage_ticks'])

        type_stats = {
            type_name: {
                'count': len(values),
                'avg': round(sum(values) / len(values), 2),
                'max': round(max(values), 2)
            }
            for type_name, values in by_type.items()
        }

        # Detect trend
        if len(slippages) >= 10:
            first_half = slippages[:len(slippages)//2]
            second_half = slippages[len(slippages)//2:]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second > avg_first * 1.2:
                trend = 'INCREASING'
            elif avg_second < avg_first * 0.8:
                trend = 'DECREASING'
            else:
                trend = 'STABLE'
        else:
            trend = 'INSUFFICIENT_DATA'

        return {
            'window_minutes': window_minutes,
            'total_events': len(recent_events),
            'avg_slippage': round(avg_slippage, 2),
            'max_slippage': round(max(slippages), 2),
            'min_slippage': round(min(slippages), 2),
            'trend': trend,
            'by_type': type_stats
        }

    def get_latency_analysis(self) -> Dict[str, Any]:
        """
        Analyze Claude API latency patterns.

        Returns:
            Dictionary with latency analysis
        """
        decisions = self.decision_logger.get_recent_decisions(count=1000)

        if not decisions:
            return {
                'total_calls': 0,
                'avg_latency_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'threshold_violations': 0
            }

        # Extract latency values
        latencies = [
            d['claude_latency_ms']
            for d in decisions
            if d.get('claude_called') and d.get('claude_latency_ms')
        ]

        if not latencies:
            return {
                'total_calls': 0,
                'avg_latency_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'threshold_violations': 0
            }

        violations = sum(1 for lat in latencies if lat > self.latency_threshold)

        return {
            'total_calls': len(latencies),
            'avg_latency_ms': round(sum(latencies) / len(latencies), 1),
            'max_latency_ms': round(max(latencies), 1),
            'min_latency_ms': round(min(latencies), 1),
            'threshold_ms': self.latency_threshold,
            'threshold_violations': violations,
            'violation_rate': round(violations / len(latencies) * 100, 2) if latencies else 0
        }

    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for performance alerts.

        Returns:
            List of active alerts
        """
        alerts = []

        # Check slippage
        slippage_stats = self.slippage_logger.get_statistics()
        if slippage_stats['threshold_exceeded']:
            alerts.append({
                'type': 'HIGH_SLIPPAGE',
                'severity': 'WARNING',
                'message': f"Running average slippage ({slippage_stats['running_avg_20']:.2f} ticks) "
                          f"exceeds threshold ({self.slippage_threshold} ticks)",
                'timestamp': datetime.now().isoformat()
            })

        # Check latency
        latency_analysis = self.get_latency_analysis()
        if latency_analysis['avg_latency_ms'] > self.latency_threshold:
            alerts.append({
                'type': 'HIGH_LATENCY',
                'severity': 'WARNING',
                'message': f"Average Claude latency ({latency_analysis['avg_latency_ms']:.1f}ms) "
                          f"exceeds threshold ({self.latency_threshold}ms)",
                'timestamp': datetime.now().isoformat()
            })

        # Check win rate
        trade_stats = self.trade_logger.get_daily_statistics()
        if trade_stats['total_trades'] >= 5 and trade_stats['win_rate'] < 40:
            alerts.append({
                'type': 'LOW_WIN_RATE',
                'severity': 'WARNING',
                'message': f"Win rate ({trade_stats['win_rate']:.1f}%) is below 40% "
                          f"with {trade_stats['total_trades']} trades",
                'timestamp': datetime.now().isoformat()
            })

        # Store alerts
        self.alerts = alerts

        return alerts

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get currently active alerts.

        Returns:
            List of active alerts
        """
        return self.alerts

    def get_performance_summary(self) -> str:
        """
        Get a formatted performance summary.

        Returns:
            Formatted performance summary string
        """
        metrics = self.get_real_time_metrics()

        summary = []
        summary.append("=" * 60)
        summary.append("PERFORMANCE SUMMARY")
        summary.append("=" * 60)

        # Trading performance
        summary.append("\nTrading Performance:")
        summary.append(f"  Total Trades: {metrics['trading']['total_trades']}")
        summary.append(f"  Win Rate: {metrics['trading']['win_rate']:.1f}%")
        summary.append(f"  Total P&L: ${metrics['trading']['total_pnl']:.2f}")
        summary.append(f"  Avg P&L: ${metrics['trading']['avg_pnl']:.2f}")

        # Decision statistics
        summary.append("\nDecision Statistics:")
        summary.append(f"  Total Decisions: {metrics['decisions']['total_decisions']}")
        summary.append(f"  Pre-Filter Pass Rate: {metrics['decisions']['pre_filter_pass_rate']:.1f}%")
        summary.append(f"  Claude Approval Rate: {metrics['decisions']['claude_approval_rate']:.1f}%")
        summary.append(f"  Post-Validation Pass Rate: {metrics['decisions']['post_validation_pass_rate']:.1f}%")

        # Latency
        summary.append("\nLatency:")
        summary.append(f"  Avg Claude Latency: {metrics['latency']['avg_claude_latency_ms']:.1f}ms")

        # Slippage
        summary.append("\nSlippage:")
        summary.append(f"  Avg Slippage: {metrics['slippage']['avg_slippage_ticks']:.2f} ticks")
        summary.append(f"  Running Avg (20): {metrics['slippage']['running_avg_20']:.2f} ticks")
        summary.append(f"  Total Cost: ${metrics['slippage']['total_cost_dollars']:.2f}")

        # Alerts
        if metrics['alerts']:
            summary.append("\nActive Alerts:")
            for alert in metrics['alerts']:
                summary.append(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")

        summary.append("=" * 60)

        return "\n".join(summary)
