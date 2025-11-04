"""
Momentum Continuation strategy.

This strategy trades pullbacks in trending markets:
- Only in TRENDING_UP or TRENDING_DOWN regime
- Price must pullback to EMA (within 10 ticks)
- EMA 20 > EMA 50 for longs, EMA 20 < EMA 50 for shorts
- MACD histogram must be positive for longs, negative for shorts
- Volume at or above average
- Uses fixed stop/target from config
"""

from typing import Dict, Any, Tuple
from datetime import datetime
from .base_strategy import BaseStrategy, Signal, TradeSetup, StrategyState
import logging

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum Continuation strategy.

    Trades pullbacks to EMA in trending markets.
    Enters on momentum continuation signals.
    """

    def __init__(self, config: Dict[str, Any], tick_size: float = 0.25, tick_value: float = 5.0):
        """
        Initialize momentum strategy.

        Args:
            config: Strategy configuration from settings.yaml
            tick_size: Instrument tick size
            tick_value: Dollar value per tick
        """
        super().__init__("Momentum", config, tick_size, tick_value)

        # Strategy parameters
        self.ema_fast = config.get('ema_fast', 20)
        self.ema_slow = config.get('ema_slow', 50)
        self.pullback_max_ticks = config.get('pullback_max_ticks', 10)
        self.macd_histogram_threshold = config.get('macd_histogram_threshold', 0)
        self.volume_multiplier = config.get('volume_multiplier', 1.0)
        self.target_ticks = config.get('target_ticks', 20)
        self.stop_ticks = config.get('stop_ticks', 10)

    def analyze(self, market_state: Dict[str, Any]) -> TradeSetup:
        """
        Analyze market for momentum continuation setup.

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
            setup_type="Momentum Continuation"
        )

        # Check if strategy can trade
        can_trade, reason = self.can_trade()
        if not can_trade:
            setup.conditions_failed.append(reason)
            return setup

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
        Check all conditions for momentum continuation setup.

        Returns:
            (qualified, signal, confidence score 0-10)
        """
        indicators = market_state.get('indicators', {})
        current_price = market_state.get('current_price')
        current_time = market_state.get('time', datetime.now())
        time_filters = market_state.get('time_filters', {})
        regime = market_state.get('regime', 'UNKNOWN')
        bars = market_state.get('bars', [])

        conditions_met = []
        conditions_failed = []
        confidence = 0.0
        signal = Signal.NONE

        # 1. Check market regime (TRENDING only)
        if regime not in ['TRENDING_UP', 'TRENDING_DOWN']:
            conditions_failed.append(f"Market regime is {regime}, need TRENDING")
            return False, Signal.NONE, 0.0

        conditions_met.append(f"Market regime: {regime} (trending)")
        confidence += 2.0

        # Determine signal based on trend direction
        if regime == 'TRENDING_UP':
            signal = Signal.LONG
        else:
            signal = Signal.SHORT

        # 2. Check time session
        if not self.is_in_trading_hours(current_time, time_filters):
            conditions_failed.append("Outside trading hours")
            return False, Signal.NONE, 0.0

        conditions_met.append("Within trading hours")
        confidence += 1.0

        # 3. Check EMA alignment
        ema_data = indicators.get('ema', {})
        ema_20 = ema_data.get('ema_20')
        ema_50 = ema_data.get('ema_50')

        if ema_20 is None or ema_50 is None:
            conditions_failed.append("EMA data not available")
            return False, Signal.NONE, 0.0

        if signal == Signal.LONG:
            # For longs, need EMA 20 > EMA 50
            if ema_20 <= ema_50:
                conditions_failed.append(f"EMA not aligned for LONG (EMA20 {ema_20:.2f} <= EMA50 {ema_50:.2f})")
                return False, Signal.NONE, 0.0
            conditions_met.append(f"EMA aligned for LONG (EMA20 {ema_20:.2f} > EMA50 {ema_50:.2f})")
        else:
            # For shorts, need EMA 20 < EMA 50
            if ema_20 >= ema_50:
                conditions_failed.append(f"EMA not aligned for SHORT (EMA20 {ema_20:.2f} >= EMA50 {ema_50:.2f})")
                return False, Signal.NONE, 0.0
            conditions_met.append(f"EMA aligned for SHORT (EMA20 {ema_20:.2f} < EMA50 {ema_50:.2f})")

        confidence += 2.0

        # 4. Check pullback to EMA (within max ticks)
        # Use EMA 20 for pullback reference
        ema_reference = ema_20
        distance_to_ema = abs(current_price - ema_reference)
        distance_ticks = distance_to_ema / self.tick_size

        if distance_ticks > self.pullback_max_ticks:
            conditions_failed.append(f"Too far from EMA ({distance_ticks:.1f} ticks > {self.pullback_max_ticks})")
            return False, Signal.NONE, 0.0

        conditions_met.append(f"Pullback to EMA20 ({distance_ticks:.1f} ticks)")
        confidence += 2.5

        # Bonus if price is very close to EMA (ideal entry)
        if distance_ticks <= 3:
            conditions_met.append("Price very close to EMA (ideal entry)")
            confidence += 1.0

        # 5. Check MACD histogram
        macd_data = indicators.get('macd', {})
        macd_histogram = macd_data.get('histogram')

        if macd_histogram is None:
            conditions_failed.append("MACD data not available")
            return False, Signal.NONE, 0.0

        if signal == Signal.LONG:
            # For longs, need positive histogram
            if macd_histogram <= self.macd_histogram_threshold:
                conditions_failed.append(f"MACD histogram not positive ({macd_histogram:.2f})")
                return False, Signal.NONE, 0.0
            conditions_met.append(f"MACD histogram positive ({macd_histogram:.2f})")
        else:
            # For shorts, need negative histogram
            if macd_histogram >= -self.macd_histogram_threshold:
                conditions_failed.append(f"MACD histogram not negative ({macd_histogram:.2f})")
                return False, Signal.NONE, 0.0
            conditions_met.append(f"MACD histogram negative ({macd_histogram:.2f})")

        confidence += 1.5

        # 6. Check volume
        if len(bars) > 0:
            current_bar = bars[-1]
            current_volume = current_bar.get('volume', 0)

            # Calculate average volume
            recent_volumes = [bar.get('volume', 0) for bar in bars[-20:] if bar.get('volume', 0) > 0]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < self.volume_multiplier:
                conditions_failed.append(f"Insufficient volume ({volume_ratio:.2f}x < {self.volume_multiplier}x)")
                return False, Signal.NONE, 0.0

            conditions_met.append(f"Volume acceptable ({volume_ratio:.2f}x average)")
            confidence += 1.0

        # 7. Check price action (momentum resuming)
        if len(bars) >= 3:
            # Look at last 3 bars
            bars_recent = bars[-3:]
            closes = [bar.get('close', current_price) for bar in bars_recent]

            if signal == Signal.LONG:
                # Check if recent bars show upward momentum resuming
                if closes[-1] > closes[-2]:
                    conditions_met.append("Momentum resuming (price rising)")
                    confidence += 1.0
                else:
                    conditions_met.append("Momentum neutral")
            else:
                # Check if recent bars show downward momentum resuming
                if closes[-1] < closes[-2]:
                    conditions_met.append("Momentum resuming (price falling)")
                    confidence += 1.0
                else:
                    conditions_met.append("Momentum neutral")

        # All conditions met
        return True, signal, round(confidence, 1)

    def calculate_stops_targets(
        self,
        entry_price: float,
        signal: Signal,
        market_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Calculate stop and target based on fixed tick values.

        For momentum strategy, we use fixed stop/target from config.
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
            return "No momentum setup detected"

        direction = "LONG" if setup.signal == Signal.LONG else "SHORT"
        entry = setup.entry_price
        stop = setup.stop_price
        target = setup.target_price
        confidence = setup.confidence

        description = f"""
🚀 Momentum Continuation Setup ({direction})
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
        ema_data = indicators.get('ema', {})
        macd_data = indicators.get('macd', {})
        current_price = market_state.get('current_price')
        bars = market_state.get('bars', [])

        ema_20 = ema_data.get('ema_20', 0)
        ema_50 = ema_data.get('ema_50', 0)
        distance_to_ema = abs(current_price - ema_20)
        distance_ticks = distance_to_ema / self.tick_size

        # Volume analysis
        current_volume = 0
        avg_volume = 0
        volume_ratio = 0

        if len(bars) > 0:
            current_volume = bars[-1].get('volume', 0)
            recent_volumes = [bar.get('volume', 0) for bar in bars[-20:] if bar.get('volume', 0) > 0]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        reasoning = {
            'strategy': 'Momentum Continuation',
            'direction': signal.value,
            'entry_logic': f"Pullback to EMA20 in {market_state.get('regime', 'UNKNOWN')} market",
            'ema_analysis': {
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_aligned': (ema_20 > ema_50) if signal == Signal.LONG else (ema_20 < ema_50),
                'ema_distance': ema_data.get('distance', 0),
                'current_price': current_price,
                'distance_to_ema20_ticks': distance_ticks,
                'pullback_max_ticks': self.pullback_max_ticks
            },
            'macd_analysis': {
                'macd': macd_data.get('macd', 0),
                'macd_signal': macd_data.get('signal', 0),
                'histogram': macd_data.get('histogram', 0),
                'trend': macd_data.get('trend', 'UNKNOWN'),
                'histogram_positive': macd_data.get('histogram', 0) > 0
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
                'is_trending': market_state.get('regime') in ['TRENDING_UP', 'TRENDING_DOWN'],
                'trend_direction': 'UP' if signal == Signal.LONG else 'DOWN'
            },
            'risk_management': {
                'stop_ticks': self.stop_ticks,
                'target_ticks': self.target_ticks,
                'risk_reward_ratio': self.target_ticks / self.stop_ticks
            }
        }

        return reasoning
