"""
Decision Logger - Logs every decision with full context.

Logs to: logs/decisions.jsonl
Format: Line-delimited JSON (JSONL)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class DecisionLogger:
    """
    Logs every trading decision with full context.

    Tracks:
    - Strategy setup scores
    - Pre-filter results
    - Claude API calls and latency
    - Post-validation results
    - Execution outcomes
    - Slippage at each stage
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize decision logger.

        Args:
            log_dir: Directory for log files (default: logs/)
        """
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'decisions.jsonl'
        self.logger = logging.getLogger(__name__)

    def log_decision(
        self,
        timestamp: datetime,
        strategy: str,
        setup_score: float,
        pre_filter: str,
        claude_called: bool,
        claude_decision: Optional[str] = None,
        claude_confidence: Optional[float] = None,
        claude_latency_ms: Optional[float] = None,
        post_validation: Optional[str] = None,
        validation_slippage_ticks: Optional[float] = None,
        execution: Optional[str] = None,
        fill_slippage_ticks: Optional[float] = None,
        total_slippage_ticks: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trading decision.

        Args:
            timestamp: Decision timestamp
            strategy: Strategy name (e.g., 'vwap', 'breakout')
            setup_score: Initial setup confidence (0-10)
            pre_filter: Pre-filter result ('PASS', 'SKIP')
            claude_called: Whether Claude API was called
            claude_decision: Claude's decision ('ENTER', 'SKIP', 'WAIT')
            claude_confidence: Claude's confidence (0.0-1.0)
            claude_latency_ms: Claude API latency in milliseconds
            post_validation: Post-validation result ('PASS', 'FAIL')
            validation_slippage_ticks: Slippage from signal to post-validation
            execution: Execution result ('FILLED', 'PARTIAL', 'REJECTED')
            fill_slippage_ticks: Slippage from post-validation to fill
            total_slippage_ticks: Total slippage from signal to fill
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Target price
            reasoning: Decision reasoning
            metadata: Additional metadata
        """
        decision_record = {
            'timestamp': timestamp.isoformat(),
            'strategy': strategy,
            'setup_score': round(setup_score, 2),
            'pre_filter': pre_filter,
            'claude_called': claude_called,
        }

        # Add Claude details if called
        if claude_called:
            decision_record.update({
                'claude_decision': claude_decision,
                'claude_confidence': round(claude_confidence, 3) if claude_confidence else None,
                'claude_latency_ms': round(claude_latency_ms, 1) if claude_latency_ms else None,
            })

        # Add post-validation details
        if post_validation:
            decision_record.update({
                'post_validation': post_validation,
                'validation_slippage_ticks': round(validation_slippage_ticks, 2) if validation_slippage_ticks else None,
            })

        # Add execution details
        if execution:
            decision_record.update({
                'execution': execution,
                'fill_slippage_ticks': round(fill_slippage_ticks, 2) if fill_slippage_ticks else None,
                'total_slippage_ticks': round(total_slippage_ticks, 2) if total_slippage_ticks else None,
                'entry_price': round(entry_price, 2) if entry_price else None,
                'stop_price': round(stop_price, 2) if stop_price else None,
                'target_price': round(target_price, 2) if target_price else None,
            })

        # Add reasoning and metadata
        if reasoning:
            decision_record['reasoning'] = reasoning

        if metadata:
            decision_record['metadata'] = metadata

        # Write to JSONL file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(decision_record) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write decision log: {e}")

    def get_recent_decisions(self, count: int = 20) -> list:
        """
        Get the most recent N decisions.

        Args:
            count: Number of recent decisions to retrieve

        Returns:
            List of decision records
        """
        if not self.log_file.exists():
            return []

        try:
            decisions = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    decisions.append(json.loads(line.strip()))

            # Return last N decisions
            return decisions[-count:] if len(decisions) > count else decisions

        except Exception as e:
            self.logger.error(f"Failed to read decision log: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from decision log.

        Returns:
            Dictionary with decision statistics
        """
        decisions = self.get_recent_decisions(count=1000)  # Last 1000 decisions

        if not decisions:
            return {
                'total_decisions': 0,
                'pre_filter_pass_rate': 0,
                'claude_approval_rate': 0,
                'post_validation_pass_rate': 0,
                'avg_claude_latency_ms': 0,
                'avg_total_slippage_ticks': 0
            }

        total = len(decisions)
        pre_filter_pass = sum(1 for d in decisions if d.get('pre_filter') == 'PASS')
        claude_called = sum(1 for d in decisions if d.get('claude_called'))
        claude_approved = sum(1 for d in decisions if d.get('claude_decision') == 'ENTER')
        post_validation_pass = sum(1 for d in decisions if d.get('post_validation') == 'PASS')

        # Average latency (only for decisions that called Claude)
        claude_latencies = [d['claude_latency_ms'] for d in decisions if d.get('claude_latency_ms')]
        avg_latency = sum(claude_latencies) / len(claude_latencies) if claude_latencies else 0

        # Average total slippage (only for executed trades)
        total_slippages = [d['total_slippage_ticks'] for d in decisions if d.get('total_slippage_ticks')]
        avg_slippage = sum(total_slippages) / len(total_slippages) if total_slippages else 0

        return {
            'total_decisions': total,
            'pre_filter_pass_rate': round(pre_filter_pass / total * 100, 1) if total > 0 else 0,
            'claude_approval_rate': round(claude_approved / claude_called * 100, 1) if claude_called > 0 else 0,
            'post_validation_pass_rate': round(post_validation_pass / pre_filter_pass * 100, 1) if pre_filter_pass > 0 else 0,
            'avg_claude_latency_ms': round(avg_latency, 1),
            'avg_total_slippage_ticks': round(avg_slippage, 2),
            'total_claude_calls': claude_called,
            'total_executions': len(total_slippages)
        }
