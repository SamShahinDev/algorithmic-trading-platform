"""
Trade Logger - Logs all trade entries and exits with slippage.

Logs to: logs/trades.jsonl
Format: Line-delimited JSON (JSONL)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class TradeLogger:
    """
    Logs all trade entries and exits with full details.

    Tracks:
    - Entry and exit prices
    - Slippage at entry and exit
    - Trade duration
    - P&L (gross and net)
    - Strategy performance
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize trade logger.

        Args:
            log_dir: Directory for log files (default: logs/)
        """
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / 'trades.jsonl'
        self.logger = logging.getLogger(__name__)

    def log_trade(
        self,
        trade_id: str,
        timestamp: datetime,
        strategy: str,
        signal: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        entry_slippage_ticks: Optional[float] = None,
        exit_slippage_ticks: Optional[float] = None,
        duration_seconds: Optional[int] = None,
        pnl_gross: Optional[float] = None,
        pnl_net: Optional[float] = None,
        commission: Optional[float] = None,
        exit_reason: Optional[str] = None,
        status: str = 'OPEN',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trade.

        Args:
            trade_id: Unique trade identifier
            timestamp: Trade timestamp
            strategy: Strategy name
            signal: Signal type ('LONG', 'SHORT')
            entry_price: Entry price
            exit_price: Exit price (None if still open)
            stop_price: Stop loss price
            target_price: Target price
            entry_slippage_ticks: Slippage at entry in ticks
            exit_slippage_ticks: Slippage at exit in ticks
            duration_seconds: Trade duration in seconds
            pnl_gross: Gross P&L
            pnl_net: Net P&L (after commissions)
            commission: Commission paid
            exit_reason: Reason for exit ('TARGET', 'STOP', 'MANUAL', 'TIMEOUT')
            status: Trade status ('OPEN', 'CLOSED', 'CANCELLED')
            metadata: Additional metadata
        """
        trade_record = {
            'trade_id': trade_id,
            'timestamp': timestamp.isoformat(),
            'strategy': strategy,
            'signal': signal,
            'entry_price': round(entry_price, 2),
            'stop_price': round(stop_price, 2) if stop_price else None,
            'target_price': round(target_price, 2) if target_price else None,
            'status': status,
        }

        # Add entry slippage
        if entry_slippage_ticks is not None:
            trade_record['entry_slippage_ticks'] = round(entry_slippage_ticks, 2)
            trade_record['entry_slippage_cost'] = round(entry_slippage_ticks * 5.0, 2)  # NQ: $5/tick

        # Add exit details if closed
        if status == 'CLOSED' and exit_price:
            trade_record.update({
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'duration_seconds': duration_seconds,
            })

            if exit_slippage_ticks is not None:
                trade_record['exit_slippage_ticks'] = round(exit_slippage_ticks, 2)
                trade_record['exit_slippage_cost'] = round(exit_slippage_ticks * 5.0, 2)

            if pnl_gross is not None:
                trade_record['pnl_gross'] = round(pnl_gross, 2)

            if pnl_net is not None:
                trade_record['pnl_net'] = round(pnl_net, 2)

            if commission is not None:
                trade_record['commission'] = round(commission, 2)

        # Add metadata
        if metadata:
            trade_record['metadata'] = metadata

        # Write to JSONL file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write trade log: {e}")

    def get_recent_trades(self, count: int = 50) -> list:
        """
        Get the most recent N trades.

        Args:
            count: Number of recent trades to retrieve

        Returns:
            List of trade records
        """
        if not self.log_file.exists():
            return []

        try:
            trades = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    trades.append(json.loads(line.strip()))

            # Return last N trades
            return trades[-count:] if len(trades) > count else trades

        except Exception as e:
            self.logger.error(f"Failed to read trade log: {e}")
            return []

    def get_open_trades(self) -> list:
        """
        Get all currently open trades.

        Returns:
            List of open trade records
        """
        all_trades = self.get_recent_trades(count=1000)
        return [t for t in all_trades if t.get('status') == 'OPEN']

    def get_daily_statistics(self, date: datetime = None) -> Dict[str, Any]:
        """
        Get statistics for a specific day.

        Args:
            date: Date to get statistics for (default: today)

        Returns:
            Dictionary with daily trade statistics
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')

        trades = self.get_recent_trades(count=1000)
        daily_trades = [
            t for t in trades
            if t.get('timestamp', '').startswith(date_str) and t.get('status') == 'CLOSED'
        ]

        if not daily_trades:
            return {
                'date': date_str,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'avg_entry_slippage_ticks': 0,
                'avg_exit_slippage_ticks': 0,
                'total_commission': 0
            }

        total = len(daily_trades)
        winning = [t for t in daily_trades if t.get('pnl_net', 0) > 0]
        losing = [t for t in daily_trades if t.get('pnl_net', 0) <= 0]

        total_pnl = sum(t.get('pnl_net', 0) for t in daily_trades)
        avg_winner = sum(t.get('pnl_net', 0) for t in winning) / len(winning) if winning else 0
        avg_loser = sum(t.get('pnl_net', 0) for t in losing) / len(losing) if losing else 0

        # Slippage statistics
        entry_slippages = [t['entry_slippage_ticks'] for t in daily_trades if 'entry_slippage_ticks' in t]
        exit_slippages = [t['exit_slippage_ticks'] for t in daily_trades if 'exit_slippage_ticks' in t]

        avg_entry_slippage = sum(entry_slippages) / len(entry_slippages) if entry_slippages else 0
        avg_exit_slippage = sum(exit_slippages) / len(exit_slippages) if exit_slippages else 0

        total_commission = sum(t.get('commission', 0) for t in daily_trades)

        return {
            'date': date_str,
            'total_trades': total,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': round(len(winning) / total * 100, 1) if total > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(total_pnl / total, 2) if total > 0 else 0,
            'avg_winner': round(avg_winner, 2),
            'avg_loser': round(avg_loser, 2),
            'avg_entry_slippage_ticks': round(avg_entry_slippage, 2),
            'avg_exit_slippage_ticks': round(avg_exit_slippage, 2),
            'total_commission': round(total_commission, 2)
        }

    def get_strategy_statistics(self, strategy: str) -> Dict[str, Any]:
        """
        Get statistics for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            Dictionary with strategy statistics
        """
        trades = self.get_recent_trades(count=1000)
        strategy_trades = [
            t for t in trades
            if t.get('strategy') == strategy and t.get('status') == 'CLOSED'
        ]

        if not strategy_trades:
            return {
                'strategy': strategy,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            }

        total = len(strategy_trades)
        winning = [t for t in strategy_trades if t.get('pnl_net', 0) > 0]
        total_pnl = sum(t.get('pnl_net', 0) for t in strategy_trades)

        return {
            'strategy': strategy,
            'total_trades': total,
            'win_rate': round(len(winning) / total * 100, 1) if total > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(total_pnl / total, 2) if total > 0 else 0
        }
