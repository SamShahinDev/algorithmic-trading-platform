"""
Strategy selection and orchestration module.

This module orchestrates all strategies:
- Polls all 3 strategies every analysis interval
- Filters setups by confidence score (8+)
- Sends high-confidence setups to SDK agent
- Handles post-validation before execution
- Tracks slippage statistics across all trades
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Main orchestrator for strategy evaluation and selection.

    Coordinates all strategies, SDK agent, and risk manager
    to find and execute the best trading opportunities.
    """

    def __init__(
        self,
        strategies: Dict[str, Any],
        sdk_agent: Any,
        risk_manager: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize strategy selector.

        Args:
            strategies: Dict of strategy name -> strategy instance
            sdk_agent: SDKAgent instance
            risk_manager: RiskManager instance
            config: Configuration dict
        """
        self.strategies = strategies
        self.sdk_agent = sdk_agent
        self.risk_manager = risk_manager
        self.config = config

        # Tracking
        self.performance_history: Dict[str, List[float]] = {
            name: [] for name in strategies.keys()
        }
        self.setup_history: List[Dict[str, Any]] = []

        # Slippage statistics
        self.slippage_stats = {
            'total_setups_evaluated': 0,
            'high_confidence_setups': 0,
            'claude_decisions': 0,
            'validation_failures': 0,
            'successful_entries': 0,
            'total_slippage_ticks': 0,
            'slippage_samples': []
        }

        logger.info(f"StrategySelector initialized with {len(strategies)} strategies")

    async def evaluate_market(self, market_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main orchestrator - evaluate all strategies and decide if we should trade.

        Flow:
        1. Call all 3 strategies to score setups
        2. Find highest-scoring setup
        3. If score >= 8/10, send to SDK agent
        4. SDK agent handles pre-filter, Claude call, and post-validation
        5. If SDK agent says ENTER, return trade decision

        Args:
            market_state: Current market conditions

        Returns:
            Trade decision dict or None
        """
        self.slippage_stats['total_setups_evaluated'] += 1

        # Get performance for SDK agent
        performance_today = self.risk_manager.get_daily_summary()

        # Poll all strategies
        all_setups = []

        for strategy_name, strategy in self.strategies.items():
            try:
                # Check if strategy can trade
                can_trade, reason = strategy.can_trade()
                if not can_trade:
                    logger.info(f"[{strategy_name}] Cannot trade: {reason}")
                    continue

                # Analyze market
                setup = strategy.analyze(market_state)

                if setup.signal.value != 'NONE' and setup.is_valid():
                    setup_dict = setup.to_dict()
                    setup_dict['strategy_name'] = strategy_name
                    # Don't include strategy_instance - not JSON serializable

                    all_setups.append(setup_dict)

                    logger.info(
                        f"✅ [{strategy_name}] Setup: {setup.signal.value} "
                        f"@ {setup.entry_price:.2f}, confidence: {setup.confidence:.1f}/10"
                    )
                else:
                    # Show why no setup (INFO level, not DEBUG)
                    if len(setup.conditions_failed) > 0:
                        # Show up to 3 failed conditions
                        reasons = "; ".join(setup.conditions_failed[:3])
                        logger.info(f"⏭️ [{strategy_name}] Skip: {reasons}")
                    else:
                        logger.info(f"⏭️ [{strategy_name}] No signal")

            except Exception as e:
                logger.error(f"[{strategy_name}] Error analyzing: {e}", exc_info=True)
                continue

        # No setups found
        if not all_setups:
            logger.info("📍 No setups detected from any strategy")
            return None

        # Find highest confidence setup
        best_setup = max(all_setups, key=lambda s: s.get('confidence', 0))

        logger.info(
            f"[BEST SETUP] {best_setup['strategy_name']}: "
            f"{best_setup['signal']} @ {best_setup['entry_price']:.2f}, "
            f"confidence: {best_setup['confidence']:.1f}/10"
        )

        # Check if high enough confidence for SDK agent
        if best_setup['confidence'] >= 8.0:
            self.slippage_stats['high_confidence_setups'] += 1

            logger.info(
                f"[HIGH CONFIDENCE] Setup score {best_setup['confidence']:.1f}/10 >= 8.0. "
                f"Sending to SDK agent..."
            )

            # Send to SDK agent for evaluation (with latency protection)
            decision = await self.sdk_agent.evaluate_setup(
                strategy_name=best_setup['strategy_name'],
                setup_dict=best_setup,
                market_state=market_state,
                performance_today=performance_today
            )

            # Track decision
            self._track_decision(best_setup, decision)

            # Check if SDK agent said ENTER and validation passed
            if decision.get('action') == 'ENTER':
                # Verify with risk manager
                risk_check = self.risk_manager.validate_trade(
                    strategy_name=best_setup['strategy_name'],
                    setup_dict=best_setup,
                    decision=decision
                )

                if risk_check.get('approved'):
                    self.slippage_stats['successful_entries'] += 1

                    logger.info(
                        f"[TRADE APPROVED] {best_setup['strategy_name']} "
                        f"{best_setup['signal']} @ {best_setup['entry_price']:.2f}"
                    )

                    # Return trade decision
                    return {
                        'strategy_name': best_setup['strategy_name'],
                        'setup': best_setup,
                        'decision': decision,
                        'risk_check': risk_check
                    }
                else:
                    logger.warning(
                        f"[RISK MANAGER REJECT] {risk_check.get('reason')}"
                    )
                    return None
            else:
                logger.info(
                    f"[SDK AGENT SKIP] {decision.get('reasoning')}"
                )
                return None
        else:
            logger.info(
                f"[LOW CONFIDENCE] Setup score {best_setup['confidence']:.1f}/10 < 8.0. "
                f"Skipping SDK agent evaluation."
            )
            return None

    def _track_decision(self, setup: Dict[str, Any], decision: Dict[str, Any]) -> None:
        """
        Track decision and slippage metrics.

        Args:
            setup: Original setup dict
            decision: SDK agent decision dict
        """
        # Track slippage if available
        if 'slippage_ticks' in decision and decision['slippage_ticks'] is not None:
            slippage = decision['slippage_ticks']
            self.slippage_stats['total_slippage_ticks'] += slippage
            self.slippage_stats['slippage_samples'].append(slippage)

            # Keep only recent samples
            if len(self.slippage_stats['slippage_samples']) > 100:
                self.slippage_stats['slippage_samples'] = \
                    self.slippage_stats['slippage_samples'][-100:]

        # Track validation results
        if decision.get('validation_failed'):
            self.slippage_stats['validation_failures'] += 1

        if decision.get('action') == 'ENTER':
            self.slippage_stats['claude_decisions'] += 1

        # Store in history
        self.setup_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': setup.get('strategy_name'),
            'confidence': setup.get('confidence'),
            'decision': decision.get('action'),
            'slippage_ticks': decision.get('slippage_ticks'),
            'latency_ms': decision.get('latency_ms')
        })

        # Keep only recent history
        if len(self.setup_history) > 1000:
            self.setup_history = self.setup_history[-1000:]

    def update_strategy_performance(self, strategy_name: str, pnl: float) -> None:
        """
        Update strategy performance after trade completion.

        Args:
            strategy_name: Strategy name
            pnl: P&L from completed trade
        """
        if strategy_name in self.performance_history:
            self.performance_history[strategy_name].append(pnl)

            # Keep only recent history
            if len(self.performance_history[strategy_name]) > 100:
                self.performance_history[strategy_name] = \
                    self.performance_history[strategy_name][-100:]

            logger.info(
                f"[PERFORMANCE UPDATE] {strategy_name}: ${pnl:.2f} "
                f"(total trades: {len(self.performance_history[strategy_name])})"
            )

    def get_strategy_stats(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Dict with performance metrics
        """
        if strategy_name not in self.performance_history:
            return {}

        results = self.performance_history[strategy_name]
        if not results:
            return {}

        winning_trades = [r for r in results if r > 0]
        losing_trades = [r for r in results if r < 0]

        return {
            'total_trades': len(results),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_pnl': sum(results),
            'avg_pnl': sum(results) / len(results),
            'avg_win': sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            'win_rate': len(winning_trades) / len(results) * 100 if results else 0,
            'best_trade': max(results) if results else 0,
            'worst_trade': min(results) if results else 0,
            'recent_10_pnl': sum(results[-10:]) if len(results) >= 10 else sum(results)
        }

    def get_slippage_statistics(self) -> Dict[str, Any]:
        """
        Get slippage statistics.

        Returns:
            Dict with slippage metrics
        """
        samples = self.slippage_stats['slippage_samples']

        avg_slippage = (
            sum(samples) / len(samples) if samples else 0
        )

        max_slippage = max(samples) if samples else 0
        min_slippage = min(samples) if samples else 0

        return {
            'total_setups_evaluated': self.slippage_stats['total_setups_evaluated'],
            'high_confidence_setups': self.slippage_stats['high_confidence_setups'],
            'high_confidence_rate': (
                self.slippage_stats['high_confidence_setups'] /
                self.slippage_stats['total_setups_evaluated']
                if self.slippage_stats['total_setups_evaluated'] > 0 else 0
            ),
            'claude_decisions': self.slippage_stats['claude_decisions'],
            'validation_failures': self.slippage_stats['validation_failures'],
            'successful_entries': self.slippage_stats['successful_entries'],
            'avg_slippage_ticks': avg_slippage,
            'max_slippage_ticks': max_slippage,
            'min_slippage_ticks': min_slippage,
            'slippage_samples_count': len(samples)
        }

    def get_all_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all strategies.

        Returns:
            Dict mapping strategy name to stats
        """
        return {
            strategy_name: self.get_strategy_stats(strategy_name)
            for strategy_name in self.strategies.keys()
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.performance_history = {name: [] for name in self.strategies.keys()}
        self.setup_history = []
        self.slippage_stats = {
            'total_setups_evaluated': 0,
            'high_confidence_setups': 0,
            'claude_decisions': 0,
            'validation_failures': 0,
            'successful_entries': 0,
            'total_slippage_ticks': 0,
            'slippage_samples': []
        }
        logger.info("StrategySelector daily stats reset")
