"""
Alert System - Monitors metrics and sends alerts for anomalies.

Monitors:
- Slippage thresholds
- Latency thresholds
- Win rate degradation
- P&L limits
- Risk violations
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from loggers.slippage_logger import SlippageLogger
from loggers.decision_logger import DecisionLogger
from loggers.trade_logger import TradeLogger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Alert types."""
    HIGH_SLIPPAGE = "HIGH_SLIPPAGE"
    HIGH_LATENCY = "HIGH_LATENCY"
    LOW_WIN_RATE = "LOW_WIN_RATE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DAILY_PROFIT_TARGET = "DAILY_PROFIT_TARGET"
    MAX_TRADES_REACHED = "MAX_TRADES_REACHED"
    VALIDATION_FAILURE_RATE = "VALIDATION_FAILURE_RATE"
    SLIPPAGE_TREND_INCREASING = "SLIPPAGE_TREND_INCREASING"


class Alert:
    """Represents a single alert."""

    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        timestamp: datetime = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            timestamp: Alert timestamp (default: now)
            metadata: Additional metadata
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.severity.value}] {self.alert_type.value}: {self.message}"


class AlertSystem:
    """
    Monitors metrics and generates alerts.

    Checks:
    - Slippage thresholds and trends
    - Latency performance
    - Win rate degradation
    - P&L limits
    - Risk violations
    """

    def __init__(
        self,
        log_dir: Path = None,
        slippage_threshold_ticks: float = 3.0,
        latency_threshold_ms: float = 500.0,
        min_win_rate: float = 40.0,
        daily_loss_limit: float = -150.0,
        daily_profit_target: float = 250.0,
        max_trades_per_day: int = 8,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize alert system.

        Args:
            log_dir: Directory containing log files
            slippage_threshold_ticks: Slippage alert threshold
            latency_threshold_ms: Latency alert threshold
            min_win_rate: Minimum acceptable win rate
            daily_loss_limit: Daily loss limit (negative value)
            daily_profit_target: Daily profit target
            max_trades_per_day: Maximum trades per day
            alert_callback: Optional callback function for alerts
        """
        self.log_dir = log_dir or Path('logs')

        # Thresholds
        self.slippage_threshold = slippage_threshold_ticks
        self.latency_threshold = latency_threshold_ms
        self.min_win_rate = min_win_rate
        self.daily_loss_limit = daily_loss_limit
        self.daily_profit_target = daily_profit_target
        self.max_trades_per_day = max_trades_per_day

        # Alert callback
        self.alert_callback = alert_callback

        # Initialize loggers
        self.slippage_logger = SlippageLogger(
            log_dir=self.log_dir,
            alert_threshold_ticks=slippage_threshold_ticks
        )
        self.decision_logger = DecisionLogger(log_dir=self.log_dir)
        self.trade_logger = TradeLogger(log_dir=self.log_dir)

        self.logger = logging.getLogger(__name__)

        # Alert history
        self.alert_history: List[Alert] = []
        self.last_check_time = datetime.now()

    def check_all_alerts(self) -> List[Alert]:
        """
        Check all alert conditions.

        Returns:
            List of triggered alerts
        """
        alerts = []

        # Check slippage
        alerts.extend(self.check_slippage_alerts())

        # Check latency
        alerts.extend(self.check_latency_alerts())

        # Check win rate
        alerts.extend(self.check_win_rate_alerts())

        # Check P&L limits
        alerts.extend(self.check_pnl_alerts())

        # Check trade count
        alerts.extend(self.check_trade_count_alerts())

        # Check validation failures
        alerts.extend(self.check_validation_alerts())

        # Store alerts and trigger callbacks
        for alert in alerts:
            self.alert_history.append(alert)
            self.logger.warning(str(alert))

            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")

        self.last_check_time = datetime.now()

        return alerts

    def check_slippage_alerts(self) -> List[Alert]:
        """Check for slippage-related alerts."""
        alerts = []

        slippage_stats = self.slippage_logger.get_statistics()

        # Check running average
        if slippage_stats['running_avg_20'] > self.slippage_threshold:
            alerts.append(Alert(
                alert_type=AlertType.HIGH_SLIPPAGE,
                severity=AlertSeverity.WARNING,
                message=f"Running average slippage ({slippage_stats['running_avg_20']:.2f} ticks) "
                       f"exceeds threshold ({self.slippage_threshold} ticks)",
                metadata={
                    'running_avg': slippage_stats['running_avg_20'],
                    'threshold': self.slippage_threshold,
                    'total_cost': slippage_stats['total_cost_dollars']
                }
            ))

        # Check for increasing trend
        if slippage_stats['total_events'] >= 20:
            events = self.slippage_logger.get_recent_slippage_events(count=20)
            slippages = [e['slippage_ticks'] for e in events]

            first_half = slippages[:10]
            second_half = slippages[10:]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second > avg_first * 1.3:  # 30% increase
                alerts.append(Alert(
                    alert_type=AlertType.SLIPPAGE_TREND_INCREASING,
                    severity=AlertSeverity.WARNING,
                    message=f"Slippage trend increasing: {avg_first:.2f} → {avg_second:.2f} ticks",
                    metadata={
                        'avg_first_half': round(avg_first, 2),
                        'avg_second_half': round(avg_second, 2),
                        'increase_pct': round((avg_second - avg_first) / avg_first * 100, 1)
                    }
                ))

        return alerts

    def check_latency_alerts(self) -> List[Alert]:
        """Check for latency-related alerts."""
        alerts = []

        decision_stats = self.decision_logger.get_statistics()

        if decision_stats['avg_claude_latency_ms'] > self.latency_threshold:
            alerts.append(Alert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                message=f"Average Claude latency ({decision_stats['avg_claude_latency_ms']:.1f}ms) "
                       f"exceeds threshold ({self.latency_threshold}ms)",
                metadata={
                    'avg_latency_ms': decision_stats['avg_claude_latency_ms'],
                    'threshold_ms': self.latency_threshold,
                    'total_calls': decision_stats['total_claude_calls']
                }
            ))

        return alerts

    def check_win_rate_alerts(self) -> List[Alert]:
        """Check for win rate alerts."""
        alerts = []

        trade_stats = self.trade_logger.get_daily_statistics()

        # Only check if we have enough trades
        if trade_stats['total_trades'] >= 5:
            if trade_stats['win_rate'] < self.min_win_rate:
                alerts.append(Alert(
                    alert_type=AlertType.LOW_WIN_RATE,
                    severity=AlertSeverity.WARNING,
                    message=f"Win rate ({trade_stats['win_rate']:.1f}%) below minimum ({self.min_win_rate}%) "
                           f"with {trade_stats['total_trades']} trades",
                    metadata={
                        'win_rate': trade_stats['win_rate'],
                        'min_win_rate': self.min_win_rate,
                        'total_trades': trade_stats['total_trades'],
                        'winning_trades': trade_stats['winning_trades']
                    }
                ))

        return alerts

    def check_pnl_alerts(self) -> List[Alert]:
        """Check for P&L limit alerts."""
        alerts = []

        trade_stats = self.trade_logger.get_daily_statistics()
        total_pnl = trade_stats['total_pnl']

        # Check loss limit
        if total_pnl <= self.daily_loss_limit:
            alerts.append(Alert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                severity=AlertSeverity.CRITICAL,
                message=f"Daily loss limit reached: ${total_pnl:.2f} (limit: ${self.daily_loss_limit:.2f})",
                metadata={
                    'total_pnl': total_pnl,
                    'loss_limit': self.daily_loss_limit,
                    'total_trades': trade_stats['total_trades']
                }
            ))

        # Check profit target
        if total_pnl >= self.daily_profit_target:
            alerts.append(Alert(
                alert_type=AlertType.DAILY_PROFIT_TARGET,
                severity=AlertSeverity.INFO,
                message=f"Daily profit target reached: ${total_pnl:.2f} (target: ${self.daily_profit_target:.2f})",
                metadata={
                    'total_pnl': total_pnl,
                    'profit_target': self.daily_profit_target,
                    'total_trades': trade_stats['total_trades']
                }
            ))

        return alerts

    def check_trade_count_alerts(self) -> List[Alert]:
        """Check for trade count alerts."""
        alerts = []

        trade_stats = self.trade_logger.get_daily_statistics()

        if trade_stats['total_trades'] >= self.max_trades_per_day:
            alerts.append(Alert(
                alert_type=AlertType.MAX_TRADES_REACHED,
                severity=AlertSeverity.WARNING,
                message=f"Maximum daily trades reached: {trade_stats['total_trades']} "
                       f"(limit: {self.max_trades_per_day})",
                metadata={
                    'total_trades': trade_stats['total_trades'],
                    'max_trades': self.max_trades_per_day,
                    'total_pnl': trade_stats['total_pnl']
                }
            ))

        return alerts

    def check_validation_alerts(self) -> List[Alert]:
        """Check for validation failure alerts."""
        alerts = []

        decision_stats = self.decision_logger.get_statistics()

        # Check if post-validation pass rate is low
        if decision_stats['total_decisions'] >= 10:
            if decision_stats['post_validation_pass_rate'] < 70:
                alerts.append(Alert(
                    alert_type=AlertType.VALIDATION_FAILURE_RATE,
                    severity=AlertSeverity.WARNING,
                    message=f"Post-validation pass rate ({decision_stats['post_validation_pass_rate']:.1f}%) "
                           f"is below 70%",
                    metadata={
                        'post_validation_pass_rate': decision_stats['post_validation_pass_rate'],
                        'total_decisions': decision_stats['total_decisions']
                    }
                ))

        return alerts

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """
        Get recent alerts.

        Args:
            count: Number of recent alerts to return

        Returns:
            List of recent alerts
        """
        return self.alert_history[-count:] if len(self.alert_history) > count else self.alert_history

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """
        Get alerts by severity.

        Args:
            severity: Alert severity to filter by

        Returns:
            List of alerts with specified severity
        """
        return [alert for alert in self.alert_history if alert.severity == severity]

    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """
        Get alerts by type.

        Args:
            alert_type: Alert type to filter by

        Returns:
            List of alerts with specified type
        """
        return [alert for alert in self.alert_history if alert.alert_type == alert_type]

    def clear_alert_history(self) -> None:
        """Clear alert history."""
        self.alert_history = []
        self.logger.info("Alert history cleared")

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get alert summary statistics.

        Returns:
            Dictionary with alert statistics
        """
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'last_alert': None
            }

        by_severity = {}
        by_type = {}

        for alert in self.alert_history:
            severity = alert.severity.value
            alert_type = alert.alert_type.value

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[alert_type] = by_type.get(alert_type, 0) + 1

        return {
            'total_alerts': len(self.alert_history),
            'by_severity': by_severity,
            'by_type': by_type,
            'last_alert': self.alert_history[-1].to_dict() if self.alert_history else None
        }
