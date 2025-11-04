"""
Slippage Logger - Logs daily slippage statistics.

Logs to: logs/slippage.jsonl
Format: Line-delimited JSON (JSONL)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class SlippageLogger:
    """
    Logs slippage statistics and alerts.

    Tracks:
    - Daily slippage averages
    - Running slippage samples
    - Slippage cost in dollars
    - Alerts for excessive slippage
    """

    def __init__(self, log_dir: Path = None, alert_threshold_ticks: float = 3.0):
        """
        Initialize slippage logger.

        Args:
            log_dir: Directory for log files (default: logs/)
            alert_threshold_ticks: Threshold for slippage alerts (default: 3.0)
        """
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'slippage.jsonl'
        self.logger = logging.getLogger(__name__)
        self.alert_threshold = alert_threshold_ticks

        # In-memory tracking
        self.daily_samples: List[float] = []
        self.last_log_date: Optional[str] = None

    def track_slippage(
        self,
        timestamp: datetime,
        strategy: str,
        slippage_type: str,
        slippage_ticks: float,
        price_moved_from: float,
        price_moved_to: float,
        trade_id: Optional[str] = None
    ) -> None:
        """
        Track a slippage event.

        Args:
            timestamp: Event timestamp
            strategy: Strategy name
            slippage_type: Type of slippage ('VALIDATION', 'FILL', 'TOTAL')
            slippage_ticks: Slippage in ticks
            price_moved_from: Starting price
            price_moved_to: Ending price
            trade_id: Associated trade ID
        """
        date_str = timestamp.strftime('%Y-%m-%d')

        # Reset daily samples if new day
        if self.last_log_date != date_str:
            if self.last_log_date is not None:
                # Log summary for previous day
                self._log_daily_summary(self.last_log_date)
            self.daily_samples = []
            self.last_log_date = date_str

        # Add to daily samples
        self.daily_samples.append(slippage_ticks)

        # Log individual slippage event
        slippage_record = {
            'timestamp': timestamp.isoformat(),
            'date': date_str,
            'strategy': strategy,
            'type': slippage_type,
            'slippage_ticks': round(slippage_ticks, 2),
            'slippage_cost_dollars': round(slippage_ticks * 5.0, 2),  # NQ: $5/tick
            'price_from': round(price_moved_from, 2),
            'price_to': round(price_moved_to, 2),
            'trade_id': trade_id
        }

        # Check for alert
        if slippage_ticks > self.alert_threshold:
            slippage_record['alert'] = 'HIGH_SLIPPAGE'
            self.logger.warning(
                f"⚠️  HIGH SLIPPAGE ALERT: {slippage_ticks:.2f} ticks "
                f"(threshold: {self.alert_threshold})"
            )

        # Write to JSONL file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(slippage_record) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write slippage log: {e}")

    def _log_daily_summary(self, date_str: str) -> None:
        """
        Log daily slippage summary.

        Args:
            date_str: Date string (YYYY-MM-DD)
        """
        if not self.daily_samples:
            return

        avg_slippage = sum(self.daily_samples) / len(self.daily_samples)
        max_slippage = max(self.daily_samples)
        min_slippage = min(self.daily_samples)
        total_cost = sum(self.daily_samples) * 5.0  # NQ: $5/tick

        summary_record = {
            'date': date_str,
            'type': 'DAILY_SUMMARY',
            'total_events': len(self.daily_samples),
            'avg_slippage_ticks': round(avg_slippage, 2),
            'max_slippage_ticks': round(max_slippage, 2),
            'min_slippage_ticks': round(min_slippage, 2),
            'total_cost_dollars': round(total_cost, 2)
        }

        # Alert if average exceeds threshold
        if avg_slippage > self.alert_threshold:
            summary_record['alert'] = 'DAILY_AVG_EXCEEDS_THRESHOLD'
            self.logger.warning(
                f"⚠️  DAILY SLIPPAGE ALERT for {date_str}: "
                f"Average {avg_slippage:.2f} ticks exceeds threshold {self.alert_threshold}"
            )

        # Write summary
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(summary_record) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write daily slippage summary: {e}")

    def get_recent_slippage_events(self, count: int = 100) -> list:
        """
        Get the most recent N slippage events.

        Args:
            count: Number of recent events to retrieve

        Returns:
            List of slippage records
        """
        if not self.log_file.exists():
            return []

        try:
            events = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get('type') != 'DAILY_SUMMARY':
                        events.append(record)

            # Return last N events
            return events[-count:] if len(events) > count else events

        except Exception as e:
            self.logger.error(f"Failed to read slippage log: {e}")
            return []

    def get_daily_summary(self, date: datetime = None) -> Dict[str, Any]:
        """
        Get slippage summary for a specific day.

        Args:
            date: Date to get summary for (default: today)

        Returns:
            Dictionary with daily slippage summary
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')

        if not self.log_file.exists():
            return {
                'date': date_str,
                'total_events': 0,
                'avg_slippage_ticks': 0,
                'max_slippage_ticks': 0,
                'min_slippage_ticks': 0,
                'total_cost_dollars': 0
            }

        try:
            # Look for existing summary
            with open(self.log_file, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get('date') == date_str and record.get('type') == 'DAILY_SUMMARY':
                        return record

            # If no summary found, calculate from events
            events = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get('date') == date_str and record.get('type') != 'DAILY_SUMMARY':
                        events.append(record)

            if not events:
                return {
                    'date': date_str,
                    'total_events': 0,
                    'avg_slippage_ticks': 0,
                    'max_slippage_ticks': 0,
                    'min_slippage_ticks': 0,
                    'total_cost_dollars': 0
                }

            slippages = [e['slippage_ticks'] for e in events]
            return {
                'date': date_str,
                'total_events': len(events),
                'avg_slippage_ticks': round(sum(slippages) / len(slippages), 2),
                'max_slippage_ticks': round(max(slippages), 2),
                'min_slippage_ticks': round(min(slippages), 2),
                'total_cost_dollars': round(sum(slippages) * 5.0, 2)
            }

        except Exception as e:
            self.logger.error(f"Failed to get daily slippage summary: {e}")
            return {
                'date': date_str,
                'total_events': 0,
                'avg_slippage_ticks': 0,
                'max_slippage_ticks': 0,
                'min_slippage_ticks': 0,
                'total_cost_dollars': 0
            }

    def get_running_average(self, window: int = 20) -> float:
        """
        Get running average of recent slippage.

        Args:
            window: Number of recent events to average

        Returns:
            Average slippage in ticks
        """
        events = self.get_recent_slippage_events(count=window)
        if not events:
            return 0.0

        slippages = [e['slippage_ticks'] for e in events]
        return round(sum(slippages) / len(slippages), 2)

    def check_alert_threshold(self, window: int = 5) -> bool:
        """
        Check if running average exceeds alert threshold.

        Args:
            window: Number of recent events to check

        Returns:
            True if threshold exceeded, False otherwise
        """
        running_avg = self.get_running_average(window=window)
        return running_avg > self.alert_threshold

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall slippage statistics.

        Returns:
            Dictionary with overall statistics
        """
        events = self.get_recent_slippage_events(count=1000)

        if not events:
            return {
                'total_events': 0,
                'avg_slippage_ticks': 0,
                'max_slippage_ticks': 0,
                'min_slippage_ticks': 0,
                'total_cost_dollars': 0,
                'running_avg_20': 0,
                'alert_threshold': self.alert_threshold,
                'threshold_exceeded': False
            }

        slippages = [e['slippage_ticks'] for e in events]

        return {
            'total_events': len(events),
            'avg_slippage_ticks': round(sum(slippages) / len(slippages), 2),
            'max_slippage_ticks': round(max(slippages), 2),
            'min_slippage_ticks': round(min(slippages), 2),
            'total_cost_dollars': round(sum(slippages) * 5.0, 2),
            'running_avg_20': self.get_running_average(window=20),
            'alert_threshold': self.alert_threshold,
            'threshold_exceeded': self.check_alert_threshold(window=5)
        }
