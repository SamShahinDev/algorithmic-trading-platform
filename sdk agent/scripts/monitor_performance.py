#!/usr/bin/env python3
"""
Performance Monitoring Script - Real-time performance monitoring.

Usage:
    python scripts/monitor_performance.py
    python scripts/monitor_performance.py --interval 30
    python scripts/monitor_performance.py --alerts-only
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.performance_monitor import PerformanceMonitor
from monitoring.alert_system import AlertSystem, AlertSeverity


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def print_metrics(metrics: dict, show_timestamp: bool = True) -> None:
    """
    Print formatted metrics.

    Args:
        metrics: Metrics dictionary
        show_timestamp: Whether to show timestamp
    """
    clear_screen()

    print("=" * 80)
    print("SDK TRADING AGENT - REAL-TIME PERFORMANCE MONITOR")
    print("=" * 80)

    if show_timestamp:
        print(f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nSession Duration: {metrics['session_duration_minutes']:.1f} minutes")

    # Trading Performance
    trading = metrics['trading']
    print("\n📈 TRADING PERFORMANCE")
    print("-" * 80)
    print(f"Total Trades: {trading['total_trades']} (W: {trading['winning_trades']} / L: {trading['losing_trades']})")
    print(f"Win Rate: {trading['win_rate']:.1f}%")
    print(f"Total P&L: ${trading['total_pnl']:.2f}")
    print(f"Avg P&L: ${trading['avg_pnl']:.2f}")
    print(f"Open Positions: {trading['open_trades_count']}")

    # Decision Statistics
    decisions = metrics['decisions']
    print("\n🎯 DECISION STATISTICS")
    print("-" * 80)
    print(f"Total Decisions: {decisions['total_decisions']}")
    print(f"Pre-Filter Pass Rate: {decisions['pre_filter_pass_rate']:.1f}%")
    print(f"Claude Approval Rate: {decisions['claude_approval_rate']:.1f}%")
    print(f"Post-Validation Pass Rate: {decisions['post_validation_pass_rate']:.1f}%")
    print(f"Total Claude Calls: {decisions['total_claude_calls']}")
    print(f"Total Executions: {decisions['total_executions']}")

    # Latency
    latency = metrics['latency']
    latency_status = "⚠️  HIGH" if latency['threshold_exceeded'] else "✓ OK"
    print("\n🕐 LATENCY")
    print("-" * 80)
    print(f"Avg Claude Latency: {latency['avg_claude_latency_ms']:.1f}ms [{latency_status}]")
    print(f"Threshold: {latency['threshold_ms']:.0f}ms")

    # Slippage
    slippage = metrics['slippage']
    slippage_status = "⚠️  HIGH" if slippage['threshold_exceeded'] else "✓ OK"
    print("\n⚡ SLIPPAGE")
    print("-" * 80)
    print(f"Avg Slippage: {slippage['avg_slippage_ticks']:.2f} ticks")
    print(f"Running Avg (20): {slippage['running_avg_20']:.2f} ticks [{slippage_status}]")
    print(f"Max Slippage: {slippage['max_slippage_ticks']:.2f} ticks")
    print(f"Total Cost: ${slippage['total_cost_dollars']:.2f}")
    print(f"Threshold: {slippage['threshold_ticks']:.1f} ticks")

    # Alerts
    alerts = metrics['alerts']
    if alerts:
        print("\n⚠️  ACTIVE ALERTS")
        print("-" * 80)
        for alert in alerts:
            severity_icon = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'CRITICAL': '🚨'
            }.get(alert['severity'], '•')

            print(f"{severity_icon} [{alert['severity']}] {alert['type']}")
            print(f"   {alert['message']}")
    else:
        print("\n✓ No Active Alerts")

    print("\n" + "=" * 80)


def print_alerts_only(alerts: list) -> None:
    """
    Print alerts in compact format.

    Args:
        alerts: List of alerts
    """
    if not alerts:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No alerts")
        return

    for alert in alerts:
        severity_icon = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }.get(alert['severity'], '•')

        timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
        print(f"[{timestamp}] {severity_icon} [{alert['severity']}] {alert['type']}: {alert['message']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Monitor trading performance in real-time')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory (default: logs)')
    parser.add_argument('--interval', type=int, default=10, help='Update interval in seconds (default: 10)')
    parser.add_argument('--alerts-only', action='store_true', help='Show only alerts (compact mode)')
    parser.add_argument('--slippage-threshold', type=float, default=3.0, help='Slippage threshold in ticks')
    parser.add_argument('--latency-threshold', type=float, default=500.0, help='Latency threshold in ms')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        print(f"Creating log directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize monitors
    performance_monitor = PerformanceMonitor(
        log_dir=log_dir,
        slippage_threshold=args.slippage_threshold,
        latency_threshold_ms=args.latency_threshold
    )

    alert_system = AlertSystem(
        log_dir=log_dir,
        slippage_threshold_ticks=args.slippage_threshold,
        latency_threshold_ms=args.latency_threshold
    )

    print(f"Starting performance monitor (update interval: {args.interval}s)")
    print(f"Log directory: {log_dir}")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            # Get metrics
            metrics = performance_monitor.get_real_time_metrics()

            # Check for alerts
            new_alerts = alert_system.check_all_alerts()

            if args.alerts_only:
                # Compact mode - only show new alerts
                if new_alerts:
                    print_alerts_only(new_alerts)
            else:
                # Full dashboard mode
                print_metrics(metrics)

            # Wait for next update
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopping performance monitor...")

        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL SESSION SUMMARY")
        print("=" * 80)

        alert_summary = alert_system.get_alert_summary()
        print(f"\nTotal Alerts: {alert_summary['total_alerts']}")

        if alert_summary['by_severity']:
            print("\nBy Severity:")
            for severity, count in alert_summary['by_severity'].items():
                print(f"  {severity}: {count}")

        if alert_summary['by_type']:
            print("\nBy Type:")
            for alert_type, count in alert_summary['by_type'].items():
                print(f"  {alert_type}: {count}")

        print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
