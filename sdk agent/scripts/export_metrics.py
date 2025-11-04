#!/usr/bin/env python3
"""
Metrics Export Script - Export trading metrics to CSV/JSON.

Usage:
    python scripts/export_metrics.py --format csv --output metrics.csv
    python scripts/export_metrics.py --format json --output metrics.json --days 7
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loggers.decision_logger import DecisionLogger
from loggers.trade_logger import TradeLogger
from loggers.slippage_logger import SlippageLogger


def export_decisions_csv(log_dir: Path, output_path: Path, days: int = None) -> None:
    """
    Export decisions to CSV.

    Args:
        log_dir: Log directory
        output_path: Output file path
        days: Number of recent days to export
    """
    logger = DecisionLogger(log_dir=log_dir)
    decisions = logger.get_recent_decisions(count=10000)

    if days:
        cutoff = datetime.now() - timedelta(days=days)
        decisions = [
            d for d in decisions
            if datetime.fromisoformat(d['timestamp']) >= cutoff
        ]

    if not decisions:
        print("No decisions to export")
        return

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'strategy', 'setup_score', 'pre_filter',
            'claude_called', 'claude_decision', 'claude_confidence',
            'claude_latency_ms', 'post_validation', 'validation_slippage_ticks',
            'execution', 'fill_slippage_ticks', 'total_slippage_ticks',
            'entry_price'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision)

    print(f"✓ Exported {len(decisions)} decisions to {output_path}")


def export_trades_csv(log_dir: Path, output_path: Path, days: int = None) -> None:
    """
    Export trades to CSV.

    Args:
        log_dir: Log directory
        output_path: Output file path
        days: Number of recent days to export
    """
    logger = TradeLogger(log_dir=log_dir)
    trades = logger.get_recent_trades(count=10000)

    if days:
        cutoff = datetime.now() - timedelta(days=days)
        trades = [
            t for t in trades
            if datetime.fromisoformat(t['timestamp']) >= cutoff
        ]

    if not trades:
        print("No trades to export")
        return

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'trade_id', 'timestamp', 'strategy', 'signal', 'entry_price',
            'exit_price', 'stop_price', 'target_price', 'entry_slippage_ticks',
            'exit_slippage_ticks', 'duration_seconds', 'pnl_gross', 'pnl_net',
            'commission', 'exit_reason', 'status'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)

    print(f"✓ Exported {len(trades)} trades to {output_path}")


def export_slippage_csv(log_dir: Path, output_path: Path, days: int = None) -> None:
    """
    Export slippage events to CSV.

    Args:
        log_dir: Log directory
        output_path: Output file path
        days: Number of recent days to export
    """
    logger = SlippageLogger(log_dir=log_dir)
    events = logger.get_recent_slippage_events(count=10000)

    if days:
        cutoff = datetime.now() - timedelta(days=days)
        events = [
            e for e in events
            if datetime.fromisoformat(e['timestamp']) >= cutoff
        ]

    if not events:
        print("No slippage events to export")
        return

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'date', 'strategy', 'type', 'slippage_ticks',
            'slippage_cost_dollars', 'price_from', 'price_to', 'trade_id'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

        writer.writeheader()
        for event in events:
            writer.writerow(event)

    print(f"✓ Exported {len(events)} slippage events to {output_path}")


def export_combined_json(log_dir: Path, output_path: Path, days: int = None) -> None:
    """
    Export all metrics to JSON.

    Args:
        log_dir: Log directory
        output_path: Output file path
        days: Number of recent days to export
    """
    decision_logger = DecisionLogger(log_dir=log_dir)
    trade_logger = TradeLogger(log_dir=log_dir)
    slippage_logger = SlippageLogger(log_dir=log_dir)

    # Get data
    decisions = decision_logger.get_recent_decisions(count=10000)
    trades = trade_logger.get_recent_trades(count=10000)
    slippage_events = slippage_logger.get_recent_slippage_events(count=10000)

    # Filter by days
    if days:
        cutoff = datetime.now() - timedelta(days=days)

        decisions = [
            d for d in decisions
            if datetime.fromisoformat(d['timestamp']) >= cutoff
        ]
        trades = [
            t for t in trades
            if datetime.fromisoformat(t['timestamp']) >= cutoff
        ]
        slippage_events = [
            e for e in slippage_events
            if datetime.fromisoformat(e['timestamp']) >= cutoff
        ]

    # Get statistics
    decision_stats = decision_logger.get_statistics()
    slippage_stats = slippage_logger.get_statistics()

    # Combine into single export
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'days_included': days,
        'statistics': {
            'decisions': decision_stats,
            'slippage': slippage_stats
        },
        'data': {
            'decisions': decisions,
            'trades': trades,
            'slippage_events': slippage_events
        }
    }

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"✓ Exported combined metrics to {output_path}")
    print(f"  - {len(decisions)} decisions")
    print(f"  - {len(trades)} trades")
    print(f"  - {len(slippage_events)} slippage events")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Export trading metrics')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory (default: logs)')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv', help='Export format')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--type', type=str, choices=['decisions', 'trades', 'slippage', 'all'],
                       default='all', help='Data type to export')
    parser.add_argument('--days', type=int, help='Export last N days only')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_path = Path(args.output)

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting metrics from {log_dir}...")

    if args.format == 'csv':
        if args.type == 'decisions' or args.type == 'all':
            decisions_path = output_path if args.type == 'decisions' else output_path.parent / 'decisions.csv'
            export_decisions_csv(log_dir, decisions_path, args.days)

        if args.type == 'trades' or args.type == 'all':
            trades_path = output_path if args.type == 'trades' else output_path.parent / 'trades.csv'
            export_trades_csv(log_dir, trades_path, args.days)

        if args.type == 'slippage' or args.type == 'all':
            slippage_path = output_path if args.type == 'slippage' else output_path.parent / 'slippage.csv'
            export_slippage_csv(log_dir, slippage_path, args.days)

    elif args.format == 'json':
        export_combined_json(log_dir, output_path, args.days)

    print("\n✓ Export complete!")


if __name__ == '__main__':
    main()
