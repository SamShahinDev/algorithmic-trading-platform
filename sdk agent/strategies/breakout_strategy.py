"""
Opening Range Breakout strategy.

This strategy trades breakouts from the opening 30-minute range:
- Tracks 9:30-10:00 CT range (high/low)
- Only triggers after 10:00 CT
- Requires clean breakout (2+ ticks) with volume confirmation
- Maximum range size of 40 ticks
- Uses fixed stop/target from config
"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime, time
from .base_strategy import BaseStrategy, Signal, TradeSetup, StrategyState
import logging

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout strategy.

    Trades clean breakouts from the first 30 minutes of trading
    with volume confirmation.
    """

    def __init__(self, config: Dict[str, Any], tick_size: float = 0.25, tick_value: float = 5.0):
        """
        Initialize breakout strategy.

        Args:
            config: Strategy configuration from settings.yaml
            tick_size: Instrument tick size
            tick_value: Dollar value per tick
        """
        super().__init__("Breakout", config, tick_size, tick_value)

        # Strategy parameters
        self.range_start_time = config.get('range_start_time', '09:30')
        self.range_end_time = config.get('range_end_time', '10:00')
        self.min_breakout_ticks = config.get('min_breakout_ticks', 2)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.max_range_size_ticks = config.get('max_range_size_ticks', 40)
        self.target_ticks = config.get('target_ticks', 30)
        self.stop_ticks = config.get('stop_ticks', 12)

        # Opening range tracking
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        self.range_established = False
        self.breakout_taken = False

    def analyze(self, market_state: Dict[str, Any]) -> TradeSetup:
        """
        Analyze market for opening range breakout setup.

        Args:
            market_state: Dictionary with current_price, indicators, regime, time, bars, etc.

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
            setup_type="Opening Range Breakout"
        )

        # Check if strategy can trade
        can_trade, reason = self.can_trade()
        if not can_trade:
            setup.conditions_failed.append(reason)
            return setup

        # Update opening range tracking
        self._update_opening_range(market_state)

        # Qualify the setup
        qualified, signal, confidence = self.qualify_setup(market_state)

        if not qualified:
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

    def qualify_setup(self, market_state: Dict[str, Any]) -> Tuple[bool, Signal, float]:
        """
        Check all conditions for opening range breakout setup.

        Returns:
            (qualified, signal, confidence score 0-10)
        """
        current_price = market_state.get('current_price')
        current_time = market_state.get('time', datetime.now())
        bars = market_state.get('bars', [])

        conditions_met = []
        conditions_failed = []
        confidence = 0.0
        signal = Signal.NONE

        # 1. Check if range is established
        if not self.range_established:
            conditions_failed.append("Opening range not yet established")
            return False, Signal.NONE, 0.0

        conditions_met.append(f"Opening range established: {self.range_low:.2f} - {self.range_high:.2f}")
        confidence += 1.5

        # 2. Check if after 10:00 CT (range end time)
        current_time_str = current_time.strftime("%H:%M")
        if current_time_str < self.range_end_time:
            conditions_failed.append(f"Before {self.range_end_time} CT (currently {current_time_str})")
            return False, Signal.NONE, 0.0

        conditions_met.append(f"After {self.range_end_time} CT")
        confidence += 1.0

        # 3. Check if breakout already taken
        if self.breakout_taken:
            conditions_failed.append("Breakout already taken today")
            return False, Signal.NONE, 0.0

        # 4. Check range size (not too wide)
        range_size = self.range_high - self.range_low
        range_size_ticks = range_size / self.tick_size

        if range_size_ticks > self.max_range_size_ticks:
            conditions_failed.append(f"Range too wide ({range_size_ticks:.0f} ticks > {self.max_range_size_ticks})")
            return False, Signal.NONE, 0.0

        conditions_met.append(f"Range size acceptable ({range_size_ticks:.0f} ticks)")
        confidence += 1.5

        # 5. Check for clean breakout (2+ ticks above/below)
        breakout_threshold_long = self.range_high + (self.min_breakout_ticks * self.tick_size)
        breakout_threshold_short = self.range_low - (self.min_breakout_ticks * self.tick_size)

        if current_price >= breakout_threshold_long:
            signal = Signal.LONG
            breakout_distance_ticks = (current_price - self.range_high) / self.tick_size
            conditions_met.append(f"Clean breakout above range ({breakout_distance_ticks:.1f} ticks)")
            confidence += 2.5
        elif current_price <= breakout_threshold_short:
            signal = Signal.SHORT
            breakout_distance_ticks = (self.range_low - current_price) / self.tick_size
            conditions_met.append(f"Clean breakout below range ({breakout_distance_ticks:.1f} ticks)")
            confidence += 2.5
        else:
            conditions_failed.append(f"No clean breakout (price {current_price:.2f} within range)")
            return False, Signal.NONE, 0.0

        # 6. Check volume confirmation
        if len(bars) > 0:
            current_bar = bars[-1]
            current_volume = current_bar.get('volume', 0)

            # Calculate average volume from recent bars
            recent_volumes = [bar.get('volume', 0) for bar in bars[-20:] if bar.get('volume', 0) > 0]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < self.volume_multiplier:
                conditions_failed.append(f"Insufficient volume ({volume_ratio:.2f}x < {self.volume_multiplier}x)")
                return False, Signal.NONE, 0.0

            conditions_met.append(f"Volume confirmation ({volume_ratio:.2f}x average)")
            confidence += 2.0

        # 7. Check momentum (price continuing in breakout direction)
        if len(bars) >= 2:
            prev_close = bars[-2].get('close', current_price)
            current_close = bars[-1].get('close', current_price)

            if signal == Signal.LONG and current_close > prev_close:
                conditions_met.append("Momentum confirming (price rising)")
                confidence += 1.0
            elif signal == Signal.SHORT and current_close < prev_close:
                conditions_met.append("Momentum confirming (price falling)")
                confidence += 1.0
            else:
                conditions_met.append("Momentum neutral")
                confidence += 0.5

        # All conditions met
        return True, signal, round(confidence, 1)

    def calculate_stops_targets(
        self,
        entry_price: float,
        signal: Signal,
        market_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Calculate stop and target based on opening range.

        For breakout strategy, stop is placed at opposite side of range,
        target is fixed ticks from config.
        """
        if signal == Signal.LONG:
            # Stop below range low
            stop = self.range_low - (self.tick_size * 2)  # 2 ticks buffer
            target = entry_price + (self.target_ticks * self.tick_size)
        else:  # SHORT
            # Stop above range high
            stop = self.range_high + (self.tick_size * 2)  # 2 ticks buffer
            target = entry_price - (self.target_ticks * self.tick_size)

        return stop, target

    def get_setup_description(self, setup: TradeSetup) -> str:
        """
        Get human-readable setup description.

        Returns:
            Formatted string for logging/notifications
        """
        if setup.signal == Signal.NONE:
            return "No opening range breakout detected"

        direction = "LONG" if setup.signal == Signal.LONG else "SHORT"
        entry = setup.entry_price
        stop = setup.stop_price
        target = setup.target_price
        confidence = setup.confidence

        description = f"""
📈 Opening Range Breakout ({direction})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Confidence: {confidence}/10
Entry: {entry:.2f}
Stop: {stop:.2f}
Target: {target:.2f}
Risk/Reward: {setup.risk_reward_ratio:.2f}

Opening Range: {self.range_low:.2f} - {self.range_high:.2f}
Range Size: {((self.range_high - self.range_low) / self.tick_size):.0f} ticks

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

    def _update_opening_range(self, market_state: Dict[str, Any]) -> None:
        """
        Update opening range high/low during range period.

        Args:
            market_state: Current market data
        """
        current_time = market_state.get('time', datetime.now())
        current_time_str = current_time.strftime("%H:%M")
        bars = market_state.get('bars', [])

        # Reset at start of new day (before range start)
        if current_time_str < self.range_start_time:
            self.range_high = None
            self.range_low = None
            self.range_established = False
            self.breakout_taken = False
            return

        # During range period, track high/low
        if self.range_start_time <= current_time_str < self.range_end_time:
            if len(bars) > 0:
                # Get bars within opening range period
                for bar in bars:
                    bar_time = bar.get('timestamp', datetime.now())
                    bar_time_str = bar_time.strftime("%H:%M")

                    if self.range_start_time <= bar_time_str < self.range_end_time:
                        bar_high = bar.get('high')
                        bar_low = bar.get('low')

                        if bar_high and bar_low:
                            if self.range_high is None or bar_high > self.range_high:
                                self.range_high = bar_high
                            if self.range_low is None or bar_low < self.range_low:
                                self.range_low = bar_low

        # After range period, mark as established
        if current_time_str >= self.range_end_time and self.range_high and self.range_low:
            if not self.range_established:
                self.range_established = True
                logger.info(f"Opening range established: {self.range_low:.2f} - {self.range_high:.2f}")

    def _build_reasoning(self, market_state: Dict[str, Any], signal: Signal) -> Dict[str, Any]:
        """
        Build detailed reasoning for SDK agent.

        Returns:
            Dictionary with analysis details
        """
        current_price = market_state.get('current_price')
        bars = market_state.get('bars', [])

        # Volume analysis
        current_volume = 0
        avg_volume = 0
        volume_ratio = 0

        if len(bars) > 0:
            current_volume = bars[-1].get('volume', 0)
            recent_volumes = [bar.get('volume', 0) for bar in bars[-20:] if bar.get('volume', 0) > 0]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        range_size = self.range_high - self.range_low if self.range_high and self.range_low else 0
        range_size_ticks = range_size / self.tick_size

        reasoning = {
            'strategy': 'Opening Range Breakout',
            'direction': signal.value,
            'entry_logic': f"Price broke {signal.value.lower()} of opening range ({self.range_start_time}-{self.range_end_time} CT)",
            'range_analysis': {
                'range_high': self.range_high,
                'range_low': self.range_low,
                'range_size_dollars': range_size,
                'range_size_ticks': range_size_ticks,
                'current_price': current_price,
                'breakout_distance_ticks': (current_price - self.range_high) / self.tick_size if signal == Signal.LONG else (self.range_low - current_price) / self.tick_size
            },
            'volume_analysis': {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_threshold': self.volume_multiplier,
                'volume_confirmed': volume_ratio >= self.volume_multiplier
            },
            'regime_analysis': {
                'regime': market_state.get('regime', 'UNKNOWN'),
                'suitable_for_breakout': market_state.get('regime') in ['VOLATILE', 'RANGING']
            },
            'risk_management': {
                'stop_ticks': self.stop_ticks,
                'target_ticks': self.target_ticks,
                'stop_at_range_edge': True,
                'risk_reward_ratio': self.target_ticks / self.stop_ticks
            }
        }

        return reasoning

    def reset_daily(self) -> None:
        """Reset daily counters and opening range tracking."""
        super().reset_daily()
        self.range_high = None
        self.range_low = None
        self.range_established = False
        self.breakout_taken = False
        logger.info(f"{self.name} opening range tracking reset")

    def record_trade(self) -> None:
        """Record that a breakout trade was taken."""
        super().record_trade()
        self.breakout_taken = True
        logger.info(f"{self.name} breakout trade recorded")
