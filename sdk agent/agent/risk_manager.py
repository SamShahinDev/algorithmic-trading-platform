"""
Risk management module with P&L and slippage tracking.

This module handles all risk management functions:
- Trade validation against daily limits
- Position sizing (fixed 1 contract for this system)
- Daily P&L tracking with profit target and max loss
- Slippage metrics tracking
- Trade history logging
- Circuit breaker protection
"""

from typing import Dict, Optional, Any, List
from datetime import datetime, date
import logging
import json

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system with comprehensive tracking.

    Enforces risk limits, tracks P&L, and monitors slippage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.

        Args:
            config: Risk management configuration from settings.yaml
        """
        self.config = config

        # Daily limits from config
        daily_limits = config.get('daily_limits', {})
        self.target_profit = daily_limits.get('target_profit', 250)
        self.max_loss = abs(daily_limits.get('max_loss', -150))
        self.max_trades = daily_limits.get('max_trades', 8)

        # Strategy limits
        strategy_limits = config.get('strategy_limits', {})
        self.strategy_max_trades = strategy_limits

        # Position sizing (fixed 1 contract)
        self.position_size = 1

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.strategy_trades: Dict[str, int] = {}
        self.current_date = date.today()
        self.current_positions = 0

        # Trade history
        self.trade_history: List[Dict[str, Any]] = []

        # Slippage tracking
        self.slippage_metrics = {
            'total_trades': 0,
            'total_slippage_ticks': 0,
            'slippage_samples': [],
            'slippage_cost_dollars': 0.0
        }

        logger.info(
            f"RiskManager initialized - Target: ${self.target_profit}, "
            f"Max Loss: -${self.max_loss}, Max Trades: {self.max_trades}"
        )

    def validate_trade(
        self,
        strategy_name: str,
        setup_dict: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate trade against all risk rules.

        Args:
            strategy_name: Strategy name
            setup_dict: Setup details
            decision: SDK agent decision

        Returns:
            Dict with approved (bool) and reason
        """
        # Reset daily counters if new day
        self._check_new_day()

        # 1. Check if profit target reached
        if self.daily_pnl >= self.target_profit:
            logger.info(
                f"[RISK REJECT] Profit target reached: ${self.daily_pnl:.2f} >= ${self.target_profit}"
            )
            return {
                'approved': False,
                'reason': f'Daily profit target reached (${self.daily_pnl:.2f})'
            }

        # 2. Check if max loss reached
        if self.daily_pnl <= -self.max_loss:
            logger.warning(
                f"[RISK REJECT] Max loss reached: ${self.daily_pnl:.2f} <= -${self.max_loss}"
            )
            return {
                'approved': False,
                'reason': f'Daily loss limit reached (${self.daily_pnl:.2f})'
            }

        # 3. Check daily trade limit
        if self.daily_trades >= self.max_trades:
            logger.info(
                f"[RISK REJECT] Max trades reached: {self.daily_trades} >= {self.max_trades}"
            )
            return {
                'approved': False,
                'reason': f'Daily trade limit reached ({self.daily_trades}/{self.max_trades})'
            }

        # 4. Check strategy-specific trade limit
        strategy_trades = self.strategy_trades.get(strategy_name, 0)
        strategy_max = self.strategy_max_trades.get(strategy_name, 10)

        if strategy_trades >= strategy_max:
            logger.info(
                f"[RISK REJECT] {strategy_name} trade limit reached: "
                f"{strategy_trades} >= {strategy_max}"
            )
            return {
                'approved': False,
                'reason': f'{strategy_name} daily limit reached ({strategy_trades}/{strategy_max})'
            }

        # 5. Check risk/reward ratio
        entry = setup_dict.get('entry_price')
        stop = setup_dict.get('stop_price')
        target = setup_dict.get('target_price')

        if entry and stop and target:
            signal = setup_dict.get('signal')

            if signal == 'LONG':
                risk = entry - stop
                reward = target - entry
            else:
                risk = stop - entry
                reward = entry - target

            if risk > 0:
                rr_ratio = reward / risk

                if rr_ratio < 1.0:
                    logger.warning(
                        f"[RISK REJECT] Poor risk/reward: {rr_ratio:.2f} < 1.0"
                    )
                    return {
                        'approved': False,
                        'reason': f'Risk/reward too low ({rr_ratio:.2f})'
                    }

        # 6. Check max positions
        if self.current_positions >= 1:  # Only 1 position at a time
            logger.info(
                f"[RISK REJECT] Position already open"
            )
            return {
                'approved': False,
                'reason': 'Position already open'
            }

        # All checks passed
        logger.info(
            f"[RISK APPROVED] {strategy_name} trade validated. "
            f"Daily: ${self.daily_pnl:.2f}, Trades: {self.daily_trades}/{self.max_trades}"
        )

        return {
            'approved': True,
            'position_size': self.position_size,
            'entry_price': setup_dict.get('entry_price'),
            'stop_price': setup_dict.get('stop_price'),
            'target_price': setup_dict.get('target_price')
        }

    def on_trade_opened(self, strategy_name: str, trade_details: Dict[str, Any]) -> None:
        """
        Update tracking when trade opens.

        Args:
            strategy_name: Strategy name
            trade_details: Trade details dict
        """
        self._check_new_day()

        self.current_positions += 1
        self.daily_trades += 1

        # Update strategy-specific counter
        if strategy_name not in self.strategy_trades:
            self.strategy_trades[strategy_name] = 0
        self.strategy_trades[strategy_name] += 1

        # Track slippage if available
        slippage_ticks = trade_details.get('slippage_ticks', 0)
        if slippage_ticks:
            self._track_slippage(slippage_ticks)

        logger.info(
            f"[TRADE OPENED] {strategy_name} - "
            f"Daily trades: {self.daily_trades}/{self.max_trades}, "
            f"Strategy trades: {self.strategy_trades[strategy_name]}"
        )

    def on_trade_closed(
        self,
        strategy_name: str,
        pnl: float,
        trade_details: Dict[str, Any]
    ) -> None:
        """
        Update tracking when trade closes.

        Args:
            strategy_name: Strategy name
            pnl: Realized P&L
            trade_details: Trade details dict
        """
        self._check_new_day()

        self.daily_pnl += pnl
        self.current_positions = max(0, self.current_positions - 1)

        # Add to trade history
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'pnl': pnl,
            'daily_pnl': self.daily_pnl,
            'trade_number': self.daily_trades,
            **trade_details
        }

        self.trade_history.append(trade_record)

        # Log to trade journal
        self._log_trade(trade_record)

        logger.info(
            f"[TRADE CLOSED] {strategy_name} - P&L: ${pnl:.2f}, "
            f"Daily P&L: ${self.daily_pnl:.2f}, "
            f"Target: ${self.target_profit}, Max Loss: -${self.max_loss}"
        )

        # Check if limits hit
        if self.daily_pnl >= self.target_profit:
            logger.info(
                f"🎯 [PROFIT TARGET HIT] ${self.daily_pnl:.2f} >= ${self.target_profit}. "
                f"Stop trading for today!"
            )

        if self.daily_pnl <= -self.max_loss:
            logger.warning(
                f"⛔ [MAX LOSS HIT] ${self.daily_pnl:.2f} <= -${self.max_loss}. "
                f"Stop trading for today!"
            )

    def _track_slippage(self, slippage_ticks: float) -> None:
        """
        Track slippage metrics.

        Args:
            slippage_ticks: Slippage in ticks
        """
        self.slippage_metrics['total_trades'] += 1
        self.slippage_metrics['total_slippage_ticks'] += slippage_ticks
        self.slippage_metrics['slippage_samples'].append(slippage_ticks)

        # Calculate cost (NQ: $5 per tick)
        slippage_cost = slippage_ticks * 5.0
        self.slippage_metrics['slippage_cost_dollars'] += slippage_cost

        # Keep only recent samples
        if len(self.slippage_metrics['slippage_samples']) > 100:
            self.slippage_metrics['slippage_samples'] = \
                self.slippage_metrics['slippage_samples'][-100:]

        logger.debug(f"Slippage tracked: {slippage_ticks:.2f} ticks (${slippage_cost:.2f})")

    def _log_trade(self, trade_record: Dict[str, Any]) -> None:
        """
        Log trade to JSON file.

        Args:
            trade_record: Trade details dict
        """
        try:
            with open('logs/trades.jsonl', 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
        except Exception as e:
            logger.error(f"Failed to write trade log: {e}")

    def _check_new_day(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self.current_date:
            logger.info(
                f"[NEW DAY] Resetting counters. Previous day: {self.current_date}, "
                f"Final P&L: ${self.daily_pnl:.2f}, Trades: {self.daily_trades}"
            )

            self.current_date = today
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.strategy_trades = {}
            self.slippage_metrics = {
                'total_trades': 0,
                'total_slippage_ticks': 0,
                'slippage_samples': [],
                'slippage_cost_dollars': 0.0
            }

    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive daily summary.

        Returns:
            Dict with all daily metrics
        """
        self._check_new_day()

        # Calculate slippage stats
        slippage_samples = self.slippage_metrics['slippage_samples']
        avg_slippage = (
            sum(slippage_samples) / len(slippage_samples)
            if slippage_samples else 0
        )

        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]

        return {
            'date': self.current_date.isoformat(),
            'daily_pnl': self.daily_pnl,
            'target_profit': self.target_profit,
            'max_loss': -self.max_loss,
            'remaining_to_target': self.target_profit - self.daily_pnl,
            'remaining_to_max_loss': self.max_loss + self.daily_pnl,
            'total_trades': self.daily_trades,
            'max_trades': self.max_trades,
            'remaining_trades': self.max_trades - self.daily_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / self.daily_trades * 100 if self.daily_trades > 0 else 0,
            'current_positions': self.current_positions,
            'strategy_trades': self.strategy_trades,
            'slippage': {
                'avg_ticks': avg_slippage,
                'total_cost_dollars': self.slippage_metrics['slippage_cost_dollars'],
                'sample_count': len(slippage_samples)
            },
            'target_reached': self.daily_pnl >= self.target_profit,
            'max_loss_hit': self.daily_pnl <= -self.max_loss,
            'can_continue_trading': (
                self.daily_pnl < self.target_profit and
                self.daily_pnl > -self.max_loss and
                self.daily_trades < self.max_trades
            )
        }

    def get_slippage_statistics(self) -> Dict[str, Any]:
        """
        Get detailed slippage statistics.

        Returns:
            Dict with slippage metrics
        """
        samples = self.slippage_metrics['slippage_samples']

        if not samples:
            return {
                'total_trades': 0,
                'avg_slippage_ticks': 0,
                'max_slippage_ticks': 0,
                'min_slippage_ticks': 0,
                'total_cost_dollars': 0
            }

        return {
            'total_trades': self.slippage_metrics['total_trades'],
            'avg_slippage_ticks': sum(samples) / len(samples),
            'max_slippage_ticks': max(samples),
            'min_slippage_ticks': min(samples),
            'total_cost_dollars': self.slippage_metrics['slippage_cost_dollars'],
            'samples': samples[-20:]  # Last 20 samples
        }

    def reset_daily(self) -> None:
        """Manually reset daily counters (for testing)."""
        logger.info("Manually resetting daily counters")
        self.current_date = date.today()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.strategy_trades = {}
        self.trade_history = []
        self.slippage_metrics = {
            'total_trades': 0,
            'total_slippage_ticks': 0,
            'slippage_samples': [],
            'slippage_cost_dollars': 0.0
        }
