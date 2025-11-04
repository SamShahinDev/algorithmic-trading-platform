#!/usr/bin/env python3
"""
Log Analysis Script - Analyze trading logs and generate reports.

Usage:
    python scripts/analyze_logs.py --date 2025-01-07
    python scripts/analyze_logs.py --days 7
    python scripts/analyze_logs.py --strategy VWAP
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loggers.decision_logger import DecisionLogger
from loggers.trade_logger import TradeLogger
from loggers.slippage_logger import SlippageLogger
from loggers.performance_logger import PerformanceLogger


def analyze_decisions(log_dir: Path, date: datetime = None) -> dict:
    """
    Analyze decision log.

    Args:
        log_dir: Log directory
        date: Date to analyze (None for all)

    Returns:
        Dictionary with decision analysis
    """
    logger = DecisionLogger(log_dir=log_dir)
    decisions = logger.get_recent_decisions(count=10000)

    if date:
        date_str = date.strftime('%Y-%m-%d')
        decisions = [d for d in decisions if d.get('timestamp', '').startswith(date_str)]

    if not decisions:
        return {'total': 0}

    # Analyze decisions
    total = len(decisions)
    pre_filter_pass = sum(1 for d in decisions if d.get('pre_filter') == 'PASS')
    claude_called = sum(1 for d in decisions if d.get('claude_called'))
    claude_approved = sum(1 for d in decisions if d.get('claude_decision') == 'ENTER')
    post_validation_pass = sum(1 for d in decisions if d.get('post_validation') == 'PASS')
    executions = sum(1 for d in decisions if d.get('execution') == 'FILLED')

    # Latency statistics
    latencies = [d['claude_latency_ms'] for d in decisions if d.get('claude_latency_ms')]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Slippage statistics
    validation_slippages = [d['validation_slippage_ticks'] for d in decisions if d.get('validation_slippage_ticks')]
    total_slippages = [d['total_slippage_ticks'] for d in decisions if d.get('total_slippage_ticks')]

    avg_validation_slippage = sum(validation_slippages) / len(validation_slippages) if validation_slippages else 0
    avg_total_slippage = sum(total_slippages) / len(total_slippages) if total_slippages else 0

    # By strategy
    by_strategy = defaultdict(int)
    for d in decisions:
        strategy = d.get('strategy', 'Unknown')
        by_strategy[strategy] += 1

    return {
        'total': total,
        'pre_filter_pass': pre_filter_pass,
        'pre_filter_pass_rate': round(pre_filter_pass / total * 100, 2) if total > 0 else 0,
        'claude_called': claude_called,
        'claude_approved': claude_approved,
        'claude_approval_rate': round(claude_approved / claude_called * 100, 2) if claude_called > 0 else 0,
        'post_validation_pass': post_validation_pass,
        'post_validation_pass_rate': round(post_validation_pass / pre_filter_pass * 100, 2) if pre_filter_pass > 0 else 0,
        'executions': executions,
        'avg_latency_ms': round(avg_latency, 1),
        'avg_validation_slippage_ticks': round(avg_validation_slippage, 2),
        'avg_total_slippage_ticks': round(avg_total_slippage, 2),
        'by_strategy': dict(by_strategy)
    }


def analyze_trades(log_dir: Path, date: datetime = None, strategy: str = None) -> dict:
    """
    Analyze trade log.

    Args:
        log_dir: Log directory
        date: Date to analyze (None for all)
        strategy: Strategy to filter by (None for all)

    Returns:
        Dictionary with trade analysis
    """
    logger = TradeLogger(log_dir=log_dir)

    if date:
        stats = logger.get_daily_statistics(date=date)
    else:
        stats = logger.get_daily_statistics()

    # Get recent trades for additional analysis
    trades = logger.get_recent_trades(count=10000)

    if date:
        date_str = date.strftime('%Y-%m-%d')
        trades = [t for t in trades if t.get('timestamp', '').startswith(date_str)]

    if strategy:
        trades = [t for t in trades if t.get('strategy') == strategy]

    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    # Trade durations
    durations = [t['duration_seconds'] for t in closed_trades if 'duration_seconds' in t]
    avg_duration_minutes = sum(durations) / len(durations) / 60 if durations else 0

    # Exit reasons
    exit_reasons = defaultdict(int)
    for trade in closed_trades:
        reason = trade.get('exit_reason', 'Unknown')
        exit_reasons[reason] += 1

    return {
        **stats,
        'avg_duration_minutes': round(avg_duration_minutes, 1),
        'exit_reasons': dict(exit_reasons)
    }


def analyze_slippage(log_dir: Path, date: datetime = None) -> dict:
    """
    Analyze slippage log.

    Args:
        log_dir: Log directory
        date: Date to analyze (None for all)

    Returns:
        Dictionary with slippage analysis
    """
    logger = SlippageLogger(log_dir=log_dir)

    if date:
        summary = logger.get_daily_summary(date=date)
    else:
        summary = logger.get_statistics()

    # Get events for detailed analysis
    events = logger.get_recent_slippage_events(count=10000)

    if date:
        date_str = date.strftime('%Y-%m-%d')
        events = [e for e in events if e.get('date') == date_str]

    # Analyze by type
    by_type = defaultdict(list)
    for event in events:
        event_type = event.get('type', 'Unknown')
        by_type[event_type].append(event['slippage_ticks'])

    type_stats = {}
    for event_type, slippages in by_type.items():
        type_stats[event_type] = {
            'count': len(slippages),
            'avg': round(sum(slippages) / len(slippages), 2),
            'max': round(max(slippages), 2),
            'min': round(min(slippages), 2)
        }

    return {
        **summary,
        'by_type': type_stats
    }


def print_report(decisions: dict, trades: dict, slippage: dict) -> None:
    """
    Print formatted analysis report.

    Args:
        decisions: Decision analysis
        trades: Trade analysis
        slippage: Slippage analysis
    """
    print("\n" + "=" * 80)
    print("TRADING LOG ANALYSIS REPORT")
    print("=" * 80)

    # Decisions
    print("\n📊 DECISION ANALYSIS")
    print("-" * 80)
    if decisions.get('total', 0) > 0:
        print(f"Total Decisions: {decisions['total']}")
        print(f"Pre-Filter Pass Rate: {decisions['pre_filter_pass_rate']:.1f}% ({decisions['pre_filter_pass']}/{decisions['total']})")
        print(f"Claude Called: {decisions['claude_called']}")
        print(f"Claude Approval Rate: {decisions['claude_approval_rate']:.1f}% ({decisions['claude_approved']}/{decisions['claude_called']})")
        print(f"Post-Validation Pass Rate: {decisions['post_validation_pass_rate']:.1f}% ({decisions['post_validation_pass']}/{decisions['pre_filter_pass']})")
        print(f"Executions: {decisions['executions']}")
        print(f"Avg Claude Latency: {decisions['avg_latency_ms']:.1f}ms")
        print(f"Avg Validation Slippage: {decisions['avg_validation_slippage_ticks']:.2f} ticks")
        print(f"Avg Total Slippage: {decisions['avg_total_slippage_ticks']:.2f} ticks")

        if decisions['by_strategy']:
            print("\nBy Strategy:")
            for strategy, count in decisions['by_strategy'].items():
                print(f"  {strategy}: {count}")
    else:
        print("No decisions found")

    # Trades
    print("\n📈 TRADE ANALYSIS")
    print("-" * 80)
    if trades.get('total_trades', 0) > 0:
        print(f"Total Trades: {trades['total_trades']}")
        print(f"Win Rate: {trades['win_rate']:.1f}% ({trades['winning_trades']}W / {trades['losing_trades']}L)")
        print(f"Total P&L: ${trades['total_pnl']:.2f}")
        print(f"Avg P&L: ${trades['avg_pnl']:.2f}")
        print(f"Avg Winner: ${trades['avg_winner']:.2f}")
        print(f"Avg Loser: ${trades['avg_loser']:.2f}")
        print(f"Avg Duration: {trades['avg_duration_minutes']:.1f} minutes")
        print(f"Avg Entry Slippage: {trades['avg_entry_slippage_ticks']:.2f} ticks")
        print(f"Avg Exit Slippage: {trades['avg_exit_slippage_ticks']:.2f} ticks")
        print(f"Total Commission: ${trades['total_commission']:.2f}")

        if trades.get('exit_reasons'):
            print("\nExit Reasons:")
            for reason, count in trades['exit_reasons'].items():
                print(f"  {reason}: {count}")
    else:
        print("No trades found")

    # Slippage
    print("\n⚡ SLIPPAGE ANALYSIS")
    print("-" * 80)
    if slippage.get('total_events', 0) > 0:
        print(f"Total Events: {slippage['total_events']}")
        print(f"Avg Slippage: {slippage['avg_slippage_ticks']:.2f} ticks")
        print(f"Max Slippage: {slippage['max_slippage_ticks']:.2f} ticks")
        print(f"Min Slippage: {slippage['min_slippage_ticks']:.2f} ticks")
        print(f"Total Cost: ${slippage['total_cost_dollars']:.2f}")

        if slippage.get('by_type'):
            print("\nBy Type:")
            for event_type, stats in slippage['by_type'].items():
                print(f"  {event_type}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg: {stats['avg']:.2f} ticks")
                print(f"    Max: {stats['max']:.2f} ticks")
    else:
        print("No slippage events found")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze trading logs')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory (default: logs)')
    parser.add_argument('--date', type=str, help='Date to analyze (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Number of recent days to analyze')
    parser.add_argument('--strategy', type=str, help='Filter by strategy name')
    parser.add_argument('--export', type=str, help='Export results to JSON file')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    # Determine date range
    date = None
    if args.date:
        try:
            date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    elif args.days:
        date = datetime.now() - timedelta(days=args.days)

    # Analyze logs
    print(f"Analyzing logs in {log_dir}...")

    decisions = analyze_decisions(log_dir, date)
    trades = analyze_trades(log_dir, date, args.strategy)
    slippage = analyze_slippage(log_dir, date)

    # Print report
    print_report(decisions, trades, slippage)

    # Export if requested
    if args.export:
        export_data = {
            'analysis_date': datetime.now().isoformat(),
            'filter_date': date.strftime('%Y-%m-%d') if date else None,
            'filter_strategy': args.strategy,
            'decisions': decisions,
            'trades': trades,
            'slippage': slippage
        }

        export_path = Path(args.export)
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Results exported to: {export_path}")


if __name__ == '__main__':
    main()
