"""
Base strategy class for all trading strategies.

This module provides an abstract base class that all strategies must implement.
Defines the interface for strategy analysis, setup qualification, and risk management.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class StrategyState(Enum):
    """Strategy execution state."""
    WAITING = "WAITING"  # No setup
    SETUP_DETECTED = "SETUP_DETECTED"  # Valid setup found
    POSITION_OPEN = "POSITION_OPEN"  # Trade active
    COOLDOWN = "COOLDOWN"  # Waiting after trade


@dataclass
class TradeSetup:
    """
    Represents a complete trade setup.

    Contains all information needed for the SDK agent to evaluate
    and execute a trading opportunity.
    """
    # Core setup information
    signal: Signal
    confidence: float  # 0-10 score
    strategy_name: str
    timestamp: datetime

    # Price levels
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None

    # Setup characteristics
    setup_type: Optional[str] = None  # e.g., "VWAP mean reversion", "Opening range breakout"

    # Detailed reasoning for AI agent
    reasoning: Dict[str, Any] = None  # Detailed analysis
    conditions_met: list = None  # List of condition descriptions
    conditions_failed: list = None  # List of failed conditions

    # Risk metrics
    risk_reward_ratio: Optional[float] = None
    risk_dollars: Optional[float] = None
    reward_dollars: Optional[float] = None

    # Market context
    market_regime: Optional[str] = None
    current_price: Optional[float] = None
    volatility_level: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.reasoning is None:
            self.reasoning = {}
        if self.conditions_met is None:
            self.conditions_met = []
        if self.conditions_failed is None:
            self.conditions_failed = []

    def is_valid(self) -> bool:
        """Check if setup is valid for trading."""
        return (
            self.signal != Signal.NONE and
            self.confidence > 0 and
            self.entry_price is not None and
            self.stop_price is not None and
            self.target_price is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert setup to dictionary for logging/API."""
        return {
            'signal': self.signal.value,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'setup_type': self.setup_type,
            'reasoning': self.reasoning,
            'conditions_met': self.conditions_met,
            'conditions_failed': self.conditions_failed,
            'risk_reward_ratio': self.risk_reward_ratio,
            'risk_dollars': self.risk_dollars,
            'reward_dollars': self.reward_dollars,
            'market_regime': self.market_regime,
            'current_price': self.current_price,
            'volatility_level': self.volatility_level
        }


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement this interface to integrate with
    the SDK agent and order management system.
    """

    def __init__(self, name: str, config: Dict[str, Any], tick_size: float = 0.25, tick_value: float = 5.0):
        """
        Initialize base strategy.

        Args:
            name: Strategy name
            config: Strategy-specific configuration from settings.yaml
            tick_size: Instrument tick size (default 0.25 for NQ)
            tick_value: Dollar value per tick (default $5 for NQ)
        """
        self.name = name
        self.config = config
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.state = StrategyState.WAITING

        # Track trades for strategy limits
        self.trades_today = 0
        self.max_trades = config.get('max_trades', 10)

        logger.info(f"{self.name} strategy initialized with config: {config}")

    @abstractmethod
    def analyze(self, market_state: Dict[str, Any]) -> TradeSetup:
        """
        Analyze current market state and return setup if found.

        This is the main entry point called by the SDK agent.

        Args:
            market_state: Dictionary containing:
                - current_price: Current market price
                - bars: Recent OHLCV bars
                - indicators: All indicator values
                - regime: Current market regime
                - time: Current timestamp

        Returns:
            TradeSetup object (signal=NONE if no setup)
        """
        pass

    @abstractmethod
    def qualify_setup(self, market_state: Dict[str, Any]) -> Tuple[bool, Signal, float]:
        """
        Check if current market conditions qualify for a setup.

        Args:
            market_state: Current market data

        Returns:
            Tuple of (qualified: bool, signal: Signal, confidence: float)
        """
        pass

    @abstractmethod
    def calculate_stops_targets(
        self,
        entry_price: float,
        signal: Signal,
        market_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and target levels.

        Args:
            entry_price: Intended entry price
            signal: Trade direction (LONG or SHORT)
            market_state: Current market data

        Returns:
            Tuple of (stop_price: float, target_price: float)
        """
        pass

    @abstractmethod
    def get_setup_description(self, setup: TradeSetup) -> str:
        """
        Get human-readable description of the setup.

        Used for logging and Discord notifications.

        Args:
            setup: TradeSetup object

        Returns:
            Formatted description string
        """
        pass

    # Helper methods for all strategies

    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """
        Check if strategy can take new trades.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        if not self.config.get('enabled', True):
            return False, f"{self.name} strategy is disabled"

        if self.trades_today >= self.max_trades:
            return False, f"{self.name} daily trade limit reached ({self.trades_today}/{self.max_trades})"

        if self.state == StrategyState.POSITION_OPEN:
            return False, f"{self.name} already has an open position"

        if self.state == StrategyState.COOLDOWN:
            return False, f"{self.name} in cooldown period"

        return True, None

    def calculate_risk_reward(
        self,
        entry: float,
        stop: float,
        target: float,
        signal: Signal
    ) -> Dict[str, float]:
        """
        Calculate risk/reward metrics.

        Args:
            entry: Entry price
            stop: Stop loss price
            target: Target price
            signal: Trade direction

        Returns:
            Dict with risk, reward, and ratio
        """
        if signal == Signal.LONG:
            risk_ticks = (entry - stop) / self.tick_size
            reward_ticks = (target - entry) / self.tick_size
        else:
            risk_ticks = (stop - entry) / self.tick_size
            reward_ticks = (entry - target) / self.tick_size

        risk_dollars = risk_ticks * self.tick_value
        reward_dollars = reward_ticks * self.tick_value
        ratio = reward_ticks / risk_ticks if risk_ticks > 0 else 0

        return {
            'risk_ticks': risk_ticks,
            'reward_ticks': reward_ticks,
            'risk_dollars': risk_dollars,
            'reward_dollars': reward_dollars,
            'ratio': ratio
        }

    def is_in_trading_hours(self, current_time: datetime, time_filters: Dict[str, str]) -> bool:
        """
        Check if current time is within allowed trading hours.

        For futures (24/5 markets), checks for weekend closure period.
        For equities, checks regular market hours.

        Args:
            current_time: Current timestamp
            time_filters: Time filter config from settings

        Returns:
            True if in trading hours
        """
        market_open = time_filters.get('market_open', '09:30')
        market_close = time_filters.get('market_close', '15:00')

        # Futures market (nearly 24/5) - check for weekend closure and daily break
        # NQ futures: Sunday 5 PM → Friday 3:10 PM with daily 3:10 PM - 5:00 PM break
        if market_open == '17:00' and market_close == '15:10':
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            hour = current_time.hour
            minute = current_time.minute

            # Weekend closure: Friday 3:10 PM → Sunday 5:00 PM
            # Friday after 3:10 PM
            if weekday == 4:  # Friday
                if hour > 15 or (hour == 15 and minute >= 10):
                    return False

            # All day Saturday
            if weekday == 5:  # Saturday
                return False

            # Sunday before 5:00 PM
            if weekday == 6:  # Sunday
                if hour < 17:
                    return False

            # Daily break: 3:10 PM - 5:00 PM (Monday-Friday)
            if weekday < 5:  # Monday-Friday
                # After 3:10 PM but before 5:00 PM = daily break
                if (hour == 15 and minute >= 10) or (hour == 16) or (hour == 17 and minute == 0):
                    return False

            return True

        # Regular equity market hours
        current_time_str = current_time.strftime("%H:%M")

        # Check if in market hours
        if current_time_str < market_open or current_time_str >= market_close:
            return False

        # Check if in lunch skip period
        lunch_start = time_filters.get('lunch_skip_start')
        lunch_end = time_filters.get('lunch_skip_end')

        if lunch_start and lunch_end:
            if lunch_start <= current_time_str < lunch_end:
                return False

        return True

    def record_trade(self) -> None:
        """Record that a trade was taken."""
        self.trades_today += 1
        self.state = StrategyState.POSITION_OPEN
        logger.info(f"{self.name} trade recorded ({self.trades_today}/{self.max_trades})")

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        self.trades_today = 0
        self.state = StrategyState.WAITING
        logger.info(f"{self.name} daily counters reset")

    def set_state(self, state: StrategyState) -> None:
        """Update strategy state."""
        old_state = self.state
        self.state = state
        logger.debug(f"{self.name} state changed: {old_state.value} -> {state.value}")
