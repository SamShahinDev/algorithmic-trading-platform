"""
VWAP Mean Reversion strategy.

This strategy trades mean reversion to VWAP in ranging markets:
- Long when price is 1.5-2.5σ below VWAP
- Short when price is 1.5-2.5σ above VWAP
- Only in RANGING regime
- RSI must be neutral (45-55)
- Checks spread and time session filters
"""

from typing import Dict, Any, Tuple, List
from datetime import datetime
from .base_strategy import BaseStrategy, Signal, TradeSetup, StrategyState
import logging

logger = logging.getLogger(__name__)


class VWAPStrategy(BaseStrategy):
    """
    VWAP Mean Reversion strategy.

    Trades pullbacks to VWAP using standard deviation bands.
    Best in ranging, choppy markets.
    """

    def __init__(self, config: Dict[str, Any], tick_size: float = 0.25, tick_value: float = 5.0):
        """
        Initialize VWAP strategy.

        Args:
            config: Strategy configuration from settings.yaml
            tick_size: Instrument tick size
            tick_value: Dollar value per tick
        """
        super().__init__("VWAP", config, tick_size, tick_value)

        # Strategy parameters
        self.entry_std_dev_min = config.get('entry_std_dev_min', 1.5)
        self.entry_std_dev_max = config.get('entry_std_dev_max', 2.5)
        self.rsi_min = config.get('rsi_min', 45)
        self.rsi_max = config.get('rsi_max', 55)
        self.max_spread_ticks = config.get('max_spread_ticks', 1)
        self.target_ticks = config.get('target_ticks', 12)
        self.stop_ticks = config.get('stop_ticks', 8)

    def analyze(self, market_state: Dict[str, Any]) -> TradeSetup:
        """
        Analyze market for VWAP mean reversion setup.

        Args:
            market_state: Dictionary with current_price, indicators, regime, time, etc.

        Returns:
            TradeSetup object
        """
        current_time = market_state.get('time', datetime.now())

        # Create empty setup
        setup = TradeSetup(
            signal=Signal.NONE,
            confidence=0.0,
            strategy_name=self.name,
            timestamp=current_time,
            setup_type="VWAP Mean Reversion"
        )

        # Check if strategy can trade
        can_trade, reason = self.can_trade()
        if not can_trade:
            setup.conditions_failed.append(reason)
            return setup

        # Qualify the setup
        qualified, signal, confidence, conditions_failed = self.qualify_setup(market_state)

        if not qualified:
            # Populate conditions_failed before returning
            setup.conditions_failed.extend(conditions_failed)
            return setup

        # We have a qualified setup
        setup.signal = signal
        setup.confidence = confidence
        current_price = market_state.get('current_price')
        setup.current_price = current_price

        # Calculate stops and targets
        stop_price, target_price = self.calculate_stops_targets(current_price, signal, market_state)
        setup.entry_price = current_price
        setup.stop_price = stop_price
        setup.target_price = target_price

        # Calculate risk/reward
        rr = self.calculate_risk_reward(current_price, stop_price, target_price, signal)
        setup.risk_reward_ratio = rr['ratio']
        setup.risk_dollars = rr['risk_dollars']
        setup.reward_dollars = rr['reward_dollars']

        # Add market context
        indicators = market_state.get('indicators', {})
        setup.market_regime = market_state.get('regime', 'UNKNOWN')
        setup.volatility_level = indicators.get('atr', {}).get('level', 'UNKNOWN')

        # Build detailed reasoning
        setup.reasoning = self._build_reasoning(market_state, signal)

        return setup

    def qualify_setup(self, market_state: Dict[str, Any]) -> Tuple[bool, Signal, float, List[str]]:
        """
        Check all conditions for VWAP mean reversion setup.

        Returns:
            (qualified, signal, confidence score 0-10, conditions_failed list)
        """
        indicators = market_state.get('indicators', {})
        current_price = market_state.get('current_price')
        current_time = market_state.get('time', datetime.now())
        time_filters = market_state.get('time_filters', {})
        regime = market_state.get('regime', 'UNKNOWN')

        conditions_met = []
        conditions_failed = []
        confidence = 0.0
        signal = Signal.NONE

        # 1. Check market regime (RANGING only)
        if regime != 'RANGING':
            conditions_failed.append(f"Market regime is {regime}, need RANGING")
            return False, Signal.NONE, 0.0, conditions_failed

        conditions_met.append(f"Market regime: {regime} (ideal for VWAP)")
        confidence += 2.0

        # 2. Check time session
        if not self.is_in_trading_hours(current_time, time_filters):
            conditions_failed.append("Outside trading hours")
            return False, Signal.NONE, 0.0, conditions_failed

        conditions_met.append("Within trading hours")
        confidence += 1.0

        # 3. Check VWAP indicators
        vwap_data = indicators.get('vwap', {})
        vwap = vwap_data.get('vwap')
        vwap_std = vwap_data.get('std')

        if vwap is None or vwap_std is None:
            conditions_failed.append("VWAP data not available")
            return False, Signal.NONE, 0.0, conditions_failed

        # Calculate distance from VWAP in standard deviations
        distance_dollars = current_price - vwap
        std_dev_distance = distance_dollars / vwap_std if vwap_std > 0 else 0

        # 4. Check if in entry zone (1.5σ to 2.5σ)
        abs_std_distance = abs(std_dev_distance)

        if abs_std_distance < self.entry_std_dev_min:
            conditions_failed.append(f"Too close to VWAP ({abs_std_distance:.2f}σ < {self.entry_std_dev_min}σ)")
            return False, Signal.NONE, 0.0, conditions_failed

        if abs_std_distance > self.entry_std_dev_max:
            conditions_failed.append(f"Too far from VWAP ({abs_std_distance:.2f}σ > {self.entry_std_dev_max}σ)")
            return False, Signal.NONE, 0.0, conditions_failed

        conditions_met.append(f"In VWAP entry zone ({abs_std_distance:.2f}σ)")
        confidence += 2.5

        # Determine direction based on VWAP position
        if std_dev_distance < 0:
            # Price below VWAP -> LONG (mean reversion up)
            signal = Signal.LONG
            conditions_met.append(f"Price {abs_std_distance:.2f}σ below VWAP (LONG setup)")
        else:
            # Price above VWAP -> SHORT (mean reversion down)
            signal = Signal.SHORT
            conditions_met.append(f"Price {abs_std_distance:.2f}σ above VWAP (SHORT setup)")

        confidence += 1.5

        # 5. Check RSI (neutral zone 45-55)
        rsi_data = indicators.get('rsi', {})
        rsi = rsi_data.get('rsi')

        if rsi is None:
            conditions_failed.append("RSI not available")
            return False, Signal.NONE, 0.0, conditions_failed

        if rsi < self.rsi_min or rsi > self.rsi_max:
            conditions_failed.append(f"RSI out of neutral zone ({rsi:.1f}, need {self.rsi_min}-{self.rsi_max})")
            return False, Signal.NONE, 0.0, conditions_failed

        conditions_met.append(f"RSI neutral ({rsi:.1f})")
        confidence += 1.5

        # 6. Check spread
        spread = market_state.get('spread', 0)
        spread_ticks = spread / self.tick_size if self.tick_size > 0 else 0

        if spread_ticks > self.max_spread_ticks:
            conditions_failed.append(f"Spread too wide ({spread_ticks:.1f} ticks > {self.max_spread_ticks})")
            return False, Signal.NONE, 0.0, conditions_failed

        conditions_met.append(f"Spread acceptable ({spread_ticks:.1f} ticks)")
        confidence += 1.0

        # 7. Check VWAP slope (prefer flat VWAP in ranging market)
        vwap_slope = vwap_data.get('slope', 0)
        if abs(vwap_slope) < 0.5:
            conditions_met.append("VWAP relatively flat (good for mean reversion)")
            confidence += 0.5
        else:
            conditions_met.append("VWAP has some slope (acceptable)")

        # All conditions met
        return True, signal, round(confidence, 1), conditions_failed

    def calculate_stops_targets(
        self,
        entry_price: float,
        signal: Signal,
        market_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Calculate stop and target based on fixed tick values.

        For VWAP strategy, we use fixed stop/target from config.
        """
        if signal == Signal.LONG:
            stop = entry_price - (self.stop_ticks * self.tick_size)
            target = entry_price + (self.target_ticks * self.tick_size)
        else:  # SHORT
            stop = entry_price + (self.stop_ticks * self.tick_size)
            target = entry_price - (self.target_ticks * self.tick_size)

        return stop, target

    def get_setup_description(self, setup: TradeSetup) -> str:
        """
        Get human-readable setup description.

        Returns:
            Formatted string for logging/notifications
        """
        if setup.signal == Signal.NONE:
            return "No VWAP setup detected"

        direction = "LONG" if setup.signal == Signal.LONG else "SHORT"
        entry = setup.entry_price
        stop = setup.stop_price
        target = setup.target_price
        confidence = setup.confidence

        description = f"""
🎯 VWAP Mean Reversion Setup ({direction})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Confidence: {confidence}/10
Entry: {entry:.2f}
Stop: {stop:.2f}
Target: {target:.2f}
Risk/Reward: {setup.risk_reward_ratio:.2f}

Market Context:
• Regime: {setup.market_regime}
• Volatility: {setup.volatility_level}
• Current Price: {setup.current_price:.2f}

Conditions Met:
"""
        for condition in setup.conditions_met:
            description += f"✓ {condition}\n"

        if setup.conditions_failed:
            description += "\nConditions Failed:\n"
            for condition in setup.conditions_failed:
                description += f"✗ {condition}\n"

        return description.strip()

    def _build_reasoning(self, market_state: Dict[str, Any], signal: Signal) -> Dict[str, Any]:
        """
        Build detailed reasoning for SDK agent.

        Returns:
            Dictionary with analysis details
        """
        indicators = market_state.get('indicators', {})
        vwap_data = indicators.get('vwap', {})
        rsi_data = indicators.get('rsi', {})
        current_price = market_state.get('current_price')

        vwap = vwap_data.get('vwap', 0)
        vwap_std = vwap_data.get('std', 0)
        distance_dollars = current_price - vwap
        std_dev_distance = distance_dollars / vwap_std if vwap_std > 0 else 0

        reasoning = {
            'strategy': 'VWAP Mean Reversion',
            'direction': signal.value,
            'entry_logic': f"Price is {abs(std_dev_distance):.2f}σ from VWAP, within entry zone ({self.entry_std_dev_min}σ to {self.entry_std_dev_max}σ)",
            'vwap_analysis': {
                'vwap_price': vwap,
                'current_price': current_price,
                'distance_dollars': distance_dollars,
                'distance_std_dev': std_dev_distance,
                'vwap_std': vwap_std,
                'vwap_slope': vwap_data.get('slope', 0)
            },
            'rsi_analysis': {
                'rsi': rsi_data.get('rsi', 0),
                'condition': rsi_data.get('condition', 'UNKNOWN'),
                'in_neutral_zone': True
            },
            'regime_analysis': {
                'regime': market_state.get('regime', 'UNKNOWN'),
                'adx': indicators.get('regime', {}).get('adx', 0),
                'is_ranging': market_state.get('regime') == 'RANGING'
            },
            'risk_management': {
                'stop_ticks': self.stop_ticks,
                'target_ticks': self.target_ticks,
                'risk_reward_ratio': self.target_ticks / self.stop_ticks
            }
        }

        return reasoning
