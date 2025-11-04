"""
Claude SDK Agent with latency protection mechanisms.

This module implements the AI-powered trading agent using Claude Sonnet 4.5:
- High-confidence pre-filter (only call Claude on 8+ score setups)
- Post-Claude validation (verify setup still qualified after latency)
- Decision caching infrastructure (for future optimization)
- Comprehensive latency and slippage tracking
- Structured JSON logging for all decisions
"""

from anthropic import Anthropic
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


def _serialize_for_json(obj):
    """
    Recursively convert datetime objects to strings for JSON serialization.

    Args:
        obj: Any Python object

    Returns:
        Object with datetime instances converted to ISO strings
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    else:
        return obj


class SDKAgent:
    """
    Claude-powered trading agent with latency protection.

    Evaluates high-confidence trade setups using Claude Sonnet 4.5
    with pre-filtering and post-validation to handle API latency.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SDK agent.

        Args:
            config: Agent configuration from settings.yaml
        """
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.config = config

        # Decision cache for future optimization
        self.decision_cache: List[Dict[str, Any]] = []
        self.max_cache_size = 100

        # Latency protection settings
        self.min_confidence_for_claude = 8.0  # Only call Claude on excellent setups
        self.max_setup_age_seconds = 2.0      # Setup must still be valid
        self.max_acceptable_slippage_ticks = 3  # Max price movement during latency

        # Statistics tracking
        self.stats = {
            'total_evaluations': 0,
            'pre_filtered': 0,
            'claude_calls': 0,
            'validation_failures': 0,
            'validation_successes': 0,
            'total_latency_ms': 0,
            'total_slippage_ticks': 0,
            'decisions': []
        }

        logger.info(f"SDKAgent initialized with model: {self.model}")
        logger.info(f"Latency protection: min_confidence={self.min_confidence_for_claude}, "
                   f"max_slippage={self.max_acceptable_slippage_ticks} ticks")

    async def evaluate_setup(
        self,
        strategy_name: str,
        setup_dict: Dict[str, Any],
        market_state: Dict[str, Any],
        performance_today: Dict[str, Any],
        strategy_instance: Any = None
    ) -> Dict[str, Any]:
        """
        Evaluate if we should take a trade with latency protection.

        Flow:
        1. Check setup confidence score (pre-filter)
        2. If >= 8/10, call Claude API
        3. After Claude responds, re-validate setup still qualified
        4. Return final decision with slippage data

        Args:
            strategy_name: Name of strategy generating setup
            setup_dict: Setup details from strategy.analyze()
            market_state: Current market conditions
            performance_today: Today's P&L and trade stats
            strategy_instance: Strategy object for re-validation

        Returns:
            Dict with action, reasoning, confidence, and latency metrics
        """
        self.stats['total_evaluations'] += 1

        # PRE-FILTER: Only call Claude on high-confidence setups
        setup_score = setup_dict.get('confidence', 0)

        if setup_score < self.min_confidence_for_claude:
            self.stats['pre_filtered'] += 1

            decision = {
                'action': 'SKIP',
                'reasoning': f'Pre-filter: Setup score {setup_score:.1f}/10 below threshold ({self.min_confidence_for_claude})',
                'confidence': 0.0,
                'pre_filtered': True,
                'setup_score': setup_score,
                'timestamp': datetime.now().isoformat()
            }

            self._log_decision(strategy_name, decision, setup_dict)

            logger.info(
                f"[PRE-FILTER SKIP] {strategy_name}: Score {setup_score:.1f}/10 "
                f"< {self.min_confidence_for_claude}. Not calling Claude."
            )

            return decision

        # High-confidence setup, call Claude
        logger.info(
            f"[PRE-FILTER PASS] {strategy_name}: Score {setup_score:.1f}/10 "
            f">= {self.min_confidence_for_claude}. Calling Claude..."
        )

        start_time = datetime.now()

        # Build comprehensive prompt
        prompt = self._build_decision_prompt(
            strategy_name,
            setup_dict,
            market_state,
            performance_today
        )

        # Call Claude API
        try:
            self.stats['claude_calls'] += 1

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse Claude's decision
            decision_text = response.content[0].text

            # Extract JSON from response
            try:
                # Try to find JSON in response
                start_idx = decision_text.find('{')
                end_idx = decision_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = decision_text[start_idx:end_idx]
                    claude_decision = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Claude response as JSON: {e}")
                logger.debug(f"Response text: {decision_text}")

                # Fallback: assume SKIP
                claude_decision = {
                    'action': 'SKIP',
                    'confidence': 0.0,
                    'reasoning': f'Failed to parse Claude response: {str(e)}'
                }

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.stats['total_latency_ms'] += latency_ms

            logger.info(
                f"[CLAUDE RESPONSE] {strategy_name} in {latency_ms:.0f}ms: "
                f"{claude_decision.get('action', 'UNKNOWN')} "
                f"(confidence: {claude_decision.get('confidence', 0):.2f})"
            )

            # POST-VALIDATION: Re-check setup still qualified
            validation_result = await self._validate_setup_still_qualified(
                strategy_name,
                setup_dict,
                claude_decision,
                latency_ms,
                market_state,
                strategy_instance
            )

            self._log_decision(strategy_name, validation_result, setup_dict)

            # Cache decision for future optimization
            self._cache_decision(strategy_name, setup_dict, validation_result)

            return validation_result

        except Exception as e:
            logger.error(f"[CLAUDE API ERROR] {strategy_name}: {e}", exc_info=True)

            decision = {
                'action': 'SKIP',
                'reasoning': f'API error: {str(e)}',
                'confidence': 0.0,
                'error': True,
                'timestamp': datetime.now().isoformat()
            }

            self._log_decision(strategy_name, decision, setup_dict)

            return decision

    async def _validate_setup_still_qualified(
        self,
        strategy_name: str,
        original_setup: Dict[str, Any],
        claude_decision: Dict[str, Any],
        latency_ms: float,
        market_state: Dict[str, Any],
        strategy_instance: Any
    ) -> Dict[str, Any]:
        """
        POST-CLAUDE VALIDATION: Verify setup hasn't degraded during latency.

        Args:
            strategy_name: Strategy name
            original_setup: Original setup dict before Claude call
            claude_decision: Claude's decision
            latency_ms: API call latency in milliseconds
            market_state: Current market state
            strategy_instance: Strategy object for re-validation

        Returns:
            Dict with final decision and validation metrics
        """
        if claude_decision.get('action') == 'SKIP':
            # Claude said skip anyway, no need to validate
            logger.info(f"[POST-VALIDATION SKIP] Claude said SKIP, no validation needed")
            return {
                **claude_decision,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat()
            }

        # Claude said ENTER - verify setup still valid
        logger.info(f"[POST-VALIDATION START] Claude said ENTER, re-checking setup...")

        if strategy_instance is None:
            logger.warning(f"[POST-VALIDATION SKIP] No strategy instance provided, cannot re-validate")
            return {
                **claude_decision,
                'latency_ms': latency_ms,
                'validation_skipped': True,
                'timestamp': datetime.now().isoformat()
            }

        # Re-analyze current market conditions
        try:
            current_setup_obj = strategy_instance.analyze(market_state)

            # Check if setup is still valid
            if current_setup_obj.signal.value == 'NONE' or not current_setup_obj.is_valid():
                self.stats['validation_failures'] += 1

                logger.warning(
                    f"[VALIDATION FAILED] Setup no longer qualified after {latency_ms:.0f}ms. "
                    f"Original signal: {original_setup.get('signal')}, "
                    f"Current signal: {current_setup_obj.signal.value}"
                )

                return {
                    'action': 'SKIP',
                    'reasoning': (
                        f"Setup degraded during Claude latency ({latency_ms:.0f}ms). "
                        f"No longer meets entry criteria. "
                        f"Original: {original_setup.get('signal')}, Current: {current_setup_obj.signal.value}"
                    ),
                    'confidence': 0.0,
                    'validation_failed': True,
                    'latency_ms': latency_ms,
                    'slippage_ticks': None,
                    'original_claude_decision': claude_decision.get('action'),
                    'timestamp': datetime.now().isoformat()
                }

            # Calculate slippage
            original_entry = original_setup.get('entry_price')
            current_entry = current_setup_obj.entry_price

            if original_entry and current_entry:
                tick_size = 0.25  # NQ tick size
                slippage_ticks = abs(current_entry - original_entry) / tick_size

                self.stats['total_slippage_ticks'] += slippage_ticks

                # Check if slippage acceptable
                if slippage_ticks > self.max_acceptable_slippage_ticks:
                    self.stats['validation_failures'] += 1

                    logger.warning(
                        f"[VALIDATION FAILED] Excessive slippage: {slippage_ticks:.1f} ticks "
                        f"(max: {self.max_acceptable_slippage_ticks}). "
                        f"Original: {original_entry:.2f}, Current: {current_entry:.2f}"
                    )

                    return {
                        'action': 'SKIP',
                        'reasoning': f"Excessive slippage ({slippage_ticks:.1f} ticks) during latency",
                        'confidence': 0.0,
                        'validation_failed': True,
                        'latency_ms': latency_ms,
                        'slippage_ticks': slippage_ticks,
                        'original_entry': original_entry,
                        'current_entry': current_entry,
                        'original_claude_decision': claude_decision.get('action'),
                        'timestamp': datetime.now().isoformat()
                    }

                # Validation passed - setup still good
                self.stats['validation_successes'] += 1

                logger.info(
                    f"[VALIDATION PASSED] Setup still qualified with {slippage_ticks:.1f} ticks slippage. "
                    f"Original: {original_entry:.2f}, Current: {current_entry:.2f}"
                )

                return {
                    **claude_decision,
                    'validation_passed': True,
                    'latency_ms': latency_ms,
                    'slippage_ticks': slippage_ticks,
                    'original_entry': original_entry,
                    'current_entry': current_entry,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # No price info, assume valid
                self.stats['validation_successes'] += 1

                logger.info(f"[VALIDATION PASSED] Setup still qualified (no price comparison)")

                return {
                    **claude_decision,
                    'validation_passed': True,
                    'latency_ms': latency_ms,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"[VALIDATION ERROR] {e}", exc_info=True)

            # On error, skip to be safe
            self.stats['validation_failures'] += 1

            return {
                'action': 'SKIP',
                'reasoning': f'Validation error: {str(e)}',
                'confidence': 0.0,
                'validation_error': True,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat()
            }

    def _build_decision_prompt(
        self,
        strategy_name: str,
        setup_dict: Dict[str, Any],
        market_state: Dict[str, Any],
        performance_today: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive prompt for Claude with all context.

        Args:
            strategy_name: Strategy name
            setup_dict: Setup details
            market_state: Current market conditions
            performance_today: Today's performance metrics

        Returns:
            Formatted prompt string
        """
        # Convert datetime objects to strings for JSON serialization
        setup_json = _serialize_for_json(setup_dict)
        market_json = _serialize_for_json(market_state)

        prompt = f"""You are a professional NQ futures trader with one goal: make $250/day consistently.

A {strategy_name} setup has been DETECTED and scored {setup_dict.get('confidence', 0):.1f}/10:

SETUP DETAILS:
{json.dumps(setup_json, indent=2)}

MARKET STATE:
{json.dumps(market_json, indent=2)}

TODAY'S PERFORMANCE:
{json.dumps(performance_today, indent=2)}

CRITICAL TIMING NOTE:
This decision will take ~400ms. By the time we execute, price may have moved 1-3 ticks.
Consider whether this setup will STILL be valid after a small price movement.

Should you take this trade?

Consider:
1. Is this setup high probability RIGHT NOW?
2. Will it still be valid after 400ms latency?
3. How does it compare to recent trades?
4. Are we on track for $250 target?
5. Any reasons to skip despite high score?

Respond with JSON only (no other text):
{{
  "action": "ENTER" or "SKIP",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation"
}}"""

        return prompt

    def _log_decision(
        self,
        strategy_name: str,
        decision: Dict[str, Any],
        setup_dict: Dict[str, Any]
    ) -> None:
        """
        Log decision to structured JSON file.

        Args:
            strategy_name: Strategy name
            decision: Final decision dict
            setup_dict: Original setup details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'decision': decision,
            'setup_score': setup_dict.get('confidence', 0),
            'setup_type': setup_dict.get('setup_type', 'unknown')
        }

        self.stats['decisions'].append(log_entry)

        # Write to decisions.jsonl file
        try:
            with open('logs/decisions.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write decision log: {e}")

    def _cache_decision(
        self,
        strategy_name: str,
        setup_dict: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> None:
        """
        Cache decision for future optimization.

        Args:
            strategy_name: Strategy name
            setup_dict: Setup details
            decision: Final decision
        """
        cache_entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'setup_characteristics': {
                'confidence': setup_dict.get('confidence'),
                'regime': setup_dict.get('market_regime'),
                'volatility': setup_dict.get('volatility_level'),
                'signal': setup_dict.get('signal')
            },
            'decision': decision.get('action'),
            'claude_confidence': decision.get('confidence')
        }

        self.decision_cache.append(cache_entry)

        # Limit cache size
        if len(self.decision_cache) > self.max_cache_size:
            self.decision_cache = self.decision_cache[-self.max_cache_size:]

        logger.debug(f"Cached decision: {cache_entry}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dict with performance metrics
        """
        avg_latency = (
            self.stats['total_latency_ms'] / self.stats['claude_calls']
            if self.stats['claude_calls'] > 0 else 0
        )

        avg_slippage = (
            self.stats['total_slippage_ticks'] / self.stats['validation_successes']
            if self.stats['validation_successes'] > 0 else 0
        )

        validation_rate = (
            self.stats['validation_successes'] /
            (self.stats['validation_successes'] + self.stats['validation_failures'])
            if (self.stats['validation_successes'] + self.stats['validation_failures']) > 0
            else 0
        )

        return {
            'total_evaluations': self.stats['total_evaluations'],
            'pre_filtered': self.stats['pre_filtered'],
            'pre_filter_rate': self.stats['pre_filtered'] / self.stats['total_evaluations']
                if self.stats['total_evaluations'] > 0 else 0,
            'claude_calls': self.stats['claude_calls'],
            'avg_latency_ms': avg_latency,
            'validation_successes': self.stats['validation_successes'],
            'validation_failures': self.stats['validation_failures'],
            'validation_success_rate': validation_rate,
            'avg_slippage_ticks': avg_slippage,
            'cache_size': len(self.decision_cache)
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.stats = {
            'total_evaluations': 0,
            'pre_filtered': 0,
            'claude_calls': 0,
            'validation_failures': 0,
            'validation_successes': 0,
            'total_latency_ms': 0,
            'total_slippage_ticks': 0,
            'decisions': []
        }
        logger.info("Daily statistics reset")
