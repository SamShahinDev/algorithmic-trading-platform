"""
Error Logger - Logs exceptions and errors.

Logs to: logs/errors.log
Format: Standard log format with full stack traces
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ErrorLogger:
    """
    Logs exceptions and errors with full context.

    Tracks:
    - Exception details and stack traces
    - Error context (what was being attempted)
    - Recovery actions
    - Error frequency
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize error logger.

        Args:
            log_dir: Directory for log files (default: logs/)
        """
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'errors.log'

        # Setup file handler
        self.logger = logging.getLogger('error_logger')
        self.logger.setLevel(logging.ERROR)

        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.ERROR)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # Error frequency tracking
        self.error_counts: Dict[str, int] = {}

    def log_error(
        self,
        error: Exception,
        context: str,
        severity: str = 'ERROR',
        recovery_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error with full context.

        Args:
            error: The exception that occurred
            context: What was being attempted when error occurred
            severity: Error severity ('ERROR', 'CRITICAL', 'WARNING')
            recovery_action: Action taken to recover (if any)
            metadata: Additional error metadata
        """
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()

        # Track error frequency
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Build log message
        log_message = f"\n{'=' * 80}\n"
        log_message += f"CONTEXT: {context}\n"
        log_message += f"ERROR TYPE: {error_type}\n"
        log_message += f"ERROR MESSAGE: {error_message}\n"
        log_message += f"FREQUENCY: {self.error_counts[error_type]} occurrence(s)\n"

        if recovery_action:
            log_message += f"RECOVERY ACTION: {recovery_action}\n"

        if metadata:
            log_message += f"METADATA: {metadata}\n"

        log_message += f"\nSTACK TRACE:\n{stack_trace}"
        log_message += f"{'=' * 80}\n"

        # Log with appropriate severity
        if severity == 'CRITICAL':
            self.logger.critical(log_message)
        elif severity == 'WARNING':
            self.logger.warning(log_message)
        else:
            self.logger.error(log_message)

    def log_api_error(
        self,
        api_name: str,
        error: Exception,
        request_details: Optional[Dict[str, Any]] = None,
        recovery_action: Optional[str] = None
    ) -> None:
        """
        Log an API error.

        Args:
            api_name: Name of the API (e.g., 'TopStepX', 'Claude')
            error: The exception that occurred
            request_details: Details about the failed request
            recovery_action: Action taken to recover
        """
        context = f"{api_name} API call failed"
        metadata = {'api': api_name}

        if request_details:
            metadata['request_details'] = request_details

        self.log_error(
            error=error,
            context=context,
            severity='ERROR',
            recovery_action=recovery_action,
            metadata=metadata
        )

    def log_strategy_error(
        self,
        strategy_name: str,
        error: Exception,
        market_data: Optional[Dict[str, Any]] = None,
        recovery_action: Optional[str] = None
    ) -> None:
        """
        Log a strategy error.

        Args:
            strategy_name: Name of the strategy
            error: The exception that occurred
            market_data: Market data at time of error
            recovery_action: Action taken to recover
        """
        context = f"Strategy '{strategy_name}' encountered error"
        metadata = {'strategy': strategy_name}

        if market_data:
            metadata['market_data'] = market_data

        self.log_error(
            error=error,
            context=context,
            severity='ERROR',
            recovery_action=recovery_action,
            metadata=metadata
        )

    def log_order_error(
        self,
        error: Exception,
        order_details: Dict[str, Any],
        recovery_action: Optional[str] = None
    ) -> None:
        """
        Log an order execution error.

        Args:
            error: The exception that occurred
            order_details: Details about the failed order
            recovery_action: Action taken to recover
        """
        context = "Order execution failed"
        metadata = {'order_details': order_details}

        self.log_error(
            error=error,
            context=context,
            severity='CRITICAL',  # Order errors are critical
            recovery_action=recovery_action,
            metadata=metadata
        )

    def log_websocket_error(
        self,
        error: Exception,
        connection_state: str,
        recovery_action: Optional[str] = None
    ) -> None:
        """
        Log a WebSocket error.

        Args:
            error: The exception that occurred
            connection_state: WebSocket connection state
            recovery_action: Action taken to recover
        """
        context = f"WebSocket error (state: {connection_state})"
        metadata = {'connection_state': connection_state}

        self.log_error(
            error=error,
            context=context,
            severity='ERROR',
            recovery_action=recovery_action,
            metadata=metadata
        )

    def log_risk_violation(
        self,
        violation_type: str,
        details: Dict[str, Any],
        action_taken: str
    ) -> None:
        """
        Log a risk limit violation.

        Args:
            violation_type: Type of violation (e.g., 'MAX_LOSS', 'MAX_TRADES')
            details: Details about the violation
            action_taken: Action taken in response
        """
        context = f"Risk violation: {violation_type}"
        metadata = {
            'violation_type': violation_type,
            'details': details
        }

        # Create pseudo-exception for logging
        error = Exception(f"Risk violation: {violation_type}")

        self.log_error(
            error=error,
            context=context,
            severity='WARNING',
            recovery_action=action_taken,
            metadata=metadata
        )

    def get_error_statistics(self) -> Dict[str, int]:
        """
        Get error frequency statistics.

        Returns:
            Dictionary with error counts by type
        """
        return self.error_counts.copy()

    def get_most_frequent_errors(self, count: int = 5) -> list:
        """
        Get the most frequently occurring errors.

        Args:
            count: Number of top errors to return

        Returns:
            List of (error_type, count) tuples
        """
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_errors[:count]
