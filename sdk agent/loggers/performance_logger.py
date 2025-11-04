"""
Performance Logger - Logs daily performance summaries.

Logs to: logs/performance.jsonl
Format: Line-delimited JSON (JSONL)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class PerformanceLogger:
    """
    Logs daily performance summaries.

    Tracks:
    - Daily P&L
    - Trade counts and win rates
    - Strategy performance
    - Risk metrics
    - Latency and slippage statistics
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize performance logger.

        Args:
            log_dir: Directory for log files (default: logs/)
        """
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'performance.jsonl'
        self.logger = logging.getLogger(__name__)

    def log_daily_summary(
        self,
        date: datetime,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        gross_pnl: float,
        commissions: float,
        largest_winner: float,
        largest_loser: float,
        avg_winner: float,
        avg_loser: float,
        win_rate: float,
        profit_factor: float,
        avg_trade_duration_minutes: float,
        strategies: Dict[str, Dict[str, Any]],
        latency_stats: Dict[str, float],
        slippage_stats: Dict[str, float],
        risk_metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log daily performance summary.

        Args:
            date: Date of summary
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total P&L (net)
            gross_pnl: Gross P&L (before commissions)
            commissions: Total commissions paid
            largest_winner: Largest winning trade
            largest_loser: Largest losing trade
            avg_winner: Average winning trade
            avg_loser: Average losing trade
            win_rate: Win rate percentage
            profit_factor: Profit factor
            avg_trade_duration_minutes: Average trade duration
            strategies: Per-strategy statistics
            latency_stats: Latency statistics
            slippage_stats: Slippage statistics
            risk_metrics: Risk metrics
            metadata: Additional metadata
        """
        summary_record = {
            'date': date.strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'trading': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'gross_pnl': round(gross_pnl, 2),
                'commissions': round(commissions, 2),
                'largest_winner': round(largest_winner, 2),
                'largest_loser': round(largest_loser, 2),
                'avg_winner': round(avg_winner, 2),
                'avg_loser': round(avg_loser, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_trade_duration_minutes': round(avg_trade_duration_minutes, 1)
            },
            'strategies': strategies,
            'latency': {
                'avg_claude_latency_ms': round(latency_stats.get('avg_claude_latency_ms', 0), 1),
                'max_claude_latency_ms': round(latency_stats.get('max_claude_latency_ms', 0), 1),
                'min_claude_latency_ms': round(latency_stats.get('min_claude_latency_ms', 0), 1),
                'total_claude_calls': latency_stats.get('total_claude_calls', 0)
            },
            'slippage': {
                'avg_slippage_ticks': round(slippage_stats.get('avg_slippage_ticks', 0), 2),
                'max_slippage_ticks': round(slippage_stats.get('max_slippage_ticks', 0), 2),
                'min_slippage_ticks': round(slippage_stats.get('min_slippage_ticks', 0), 2),
                'total_slippage_cost': round(slippage_stats.get('total_slippage_cost', 0), 2)
            },
            'risk': risk_metrics
        }

        if metadata:
            summary_record['metadata'] = metadata

        # Write to JSONL file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(summary_record) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write performance log: {e}")

    def get_recent_summaries(self, count: int = 30) -> list:
        """
        Get the most recent N daily summaries.

        Args:
            count: Number of recent summaries to retrieve

        Returns:
            List of daily summary records
        """
        if not self.log_file.exists():
            return []

        try:
            summaries = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    summaries.append(json.loads(line.strip()))

            # Return last N summaries
            return summaries[-count:] if len(summaries) > count else summaries

        except Exception as e:
            self.logger.error(f"Failed to read performance log: {e}")
            return []

    def get_summary_for_date(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for a specific date.

        Args:
            date: Date to get summary for

        Returns:
            Daily summary record or None if not found
        """
        date_str = date.strftime('%Y-%m-%d')

        summaries = self.get_recent_summaries(count=365)
        for summary in summaries:
            if summary.get('date') == date_str:
                return summary

        return None

    def get_period_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get statistics for the last N days.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with period statistics
        """
        summaries = self.get_recent_summaries(count=days)

        if not summaries:
            return {
                'period_days': days,
                'total_trades': 0,
                'total_pnl': 0,
                'avg_daily_pnl': 0,
                'win_rate': 0,
                'avg_profit_factor': 0,
                'total_commissions': 0,
                'avg_slippage_ticks': 0
            }

        total_trades = sum(s['trading']['total_trades'] for s in summaries)
        total_pnl = sum(s['trading']['total_pnl'] for s in summaries)
        total_winning = sum(s['trading']['winning_trades'] for s in summaries)
        total_commissions = sum(s['trading']['commissions'] for s in summaries)

        # Average metrics
        avg_win_rate = sum(s['trading']['win_rate'] for s in summaries) / len(summaries)
        avg_profit_factor = sum(s['trading']['profit_factor'] for s in summaries) / len(summaries)
        avg_slippage = sum(s['slippage']['avg_slippage_ticks'] for s in summaries) / len(summaries)

        return {
            'period_days': len(summaries),
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'avg_daily_pnl': round(total_pnl / len(summaries), 2),
            'win_rate': round(avg_win_rate, 2),
            'avg_profit_factor': round(avg_profit_factor, 2),
            'total_commissions': round(total_commissions, 2),
            'avg_slippage_ticks': round(avg_slippage, 2),
            'winning_days': sum(1 for s in summaries if s['trading']['total_pnl'] > 0),
            'losing_days': sum(1 for s in summaries if s['trading']['total_pnl'] < 0),
            'breakeven_days': sum(1 for s in summaries if s['trading']['total_pnl'] == 0)
        }

    def get_best_and_worst_days(self, count: int = 5) -> Dict[str, list]:
        """
        Get best and worst trading days.

        Args:
            count: Number of days to return for each

        Returns:
            Dictionary with 'best' and 'worst' day lists
        """
        summaries = self.get_recent_summaries(count=365)

        if not summaries:
            return {'best': [], 'worst': []}

        # Sort by total P&L
        sorted_summaries = sorted(summaries, key=lambda s: s['trading']['total_pnl'], reverse=True)

        return {
            'best': sorted_summaries[:count],
            'worst': sorted_summaries[-count:][::-1]  # Reverse to show worst first
        }

    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance across strategies.

        Returns:
            Dictionary with per-strategy statistics
        """
        summaries = self.get_recent_summaries(count=30)

        if not summaries:
            return {}

        strategy_stats = {}

        for summary in summaries:
            for strategy_name, strategy_data in summary.get('strategies', {}).items():
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = {
                        'total_trades': 0,
                        'total_pnl': 0,
                        'winning_trades': 0,
                        'losing_trades': 0
                    }

                stats = strategy_stats[strategy_name]
                stats['total_trades'] += strategy_data.get('total_trades', 0)
                stats['total_pnl'] += strategy_data.get('total_pnl', 0)
                stats['winning_trades'] += strategy_data.get('winning_trades', 0)
                stats['losing_trades'] += strategy_data.get('losing_trades', 0)

        # Calculate win rates
        for strategy_name, stats in strategy_stats.items():
            total = stats['total_trades']
            if total > 0:
                stats['win_rate'] = round(stats['winning_trades'] / total * 100, 2)
                stats['avg_pnl_per_trade'] = round(stats['total_pnl'] / total, 2)
            else:
                stats['win_rate'] = 0
                stats['avg_pnl_per_trade'] = 0

        return strategy_stats
