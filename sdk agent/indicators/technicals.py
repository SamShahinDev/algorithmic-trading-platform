"""
Technical indicators module.

This module provides access to standard technical indicators:
- RSI (Relative Strength Index)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)

All indicators are calculated incrementally in the IndicatorCache.
This module provides convenient accessor methods and analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .cache import IndicatorCache


class TechnicalIndicators:
    """
    Technical indicators analyzer.

    Provides access to and analysis of standard technical indicators.
    """

    def __init__(self, cache: IndicatorCache):
        """
        Initialize technical indicators analyzer.

        Args:
            cache: IndicatorCache instance
        """
        self.cache = cache

    # RSI Methods

    def get_rsi(self) -> Optional[float]:
        """
        Get current RSI value.

        Returns:
            RSI value (0-100) or None
        """
        return self.cache.get_latest('rsi')

    def is_rsi_neutral(
        self,
        rsi_min: float = 45.0,
        rsi_max: float = 55.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if RSI is in neutral zone.

        Args:
            rsi_min: Minimum RSI for neutral (default 45)
            rsi_max: Maximum RSI for neutral (default 55)

        Returns:
            Tuple of (is_neutral: bool, reason: str)
        """
        rsi = self.get_rsi()

        if rsi is None:
            return False, "RSI not available"

        if rsi < rsi_min:
            return False, f"RSI oversold ({rsi:.1f} < {rsi_min})"

        if rsi > rsi_max:
            return False, f"RSI overbought ({rsi:.1f} > {rsi_max})"

        return True, f"RSI neutral ({rsi:.1f})"

    def get_rsi_signal(self) -> Dict[str, any]:
        """
        Get RSI trading signal.

        Returns:
            Dict with RSI analysis
        """
        rsi = self.get_rsi()

        if rsi is None:
            return {
                'rsi': None,
                'signal': 'NONE',
                'condition': 'NO_DATA'
            }

        # Determine signal
        if rsi < 30:
            signal = 'LONG'
            condition = 'OVERSOLD'
        elif rsi > 70:
            signal = 'SHORT'
            condition = 'OVERBOUGHT'
        elif 45 <= rsi <= 55:
            signal = 'NEUTRAL'
            condition = 'NEUTRAL'
        else:
            signal = 'NONE'
            condition = 'NORMAL'

        return {
            'rsi': rsi,
            'signal': signal,
            'condition': condition
        }

    # EMA Methods

    def get_ema_20(self) -> Optional[float]:
        """Get current EMA 20 value."""
        return self.cache.get_latest('ema_20')

    def get_ema_50(self) -> Optional[float]:
        """Get current EMA 50 value."""
        return self.cache.get_latest('ema_50')

    def get_ema_cross(self) -> Dict[str, any]:
        """
        Check for EMA crossover.

        Returns:
            Dict with EMA crossover analysis
        """
        ema_20_array = self.cache.get_array('ema_20', count=2)
        ema_50_array = self.cache.get_array('ema_50', count=2)

        if len(ema_20_array) < 2 or len(ema_50_array) < 2:
            return {
                'ema_20': None,
                'ema_50': None,
                'cross': False,
                'direction': None,
                'signal': 'NONE'
            }

        ema_20_current = ema_20_array[-1]
        ema_20_prev = ema_20_array[-2]
        ema_50_current = ema_50_array[-1]
        ema_50_prev = ema_50_array[-2]

        # Check for crossover
        cross_up = (ema_20_prev <= ema_50_prev) and (ema_20_current > ema_50_current)
        cross_down = (ema_20_prev >= ema_50_prev) and (ema_20_current < ema_50_current)

        if cross_up:
            direction = 'UP'
            signal = 'LONG'
        elif cross_down:
            direction = 'DOWN'
            signal = 'SHORT'
        else:
            direction = 'UP' if ema_20_current > ema_50_current else 'DOWN'
            signal = 'NONE'

        return {
            'ema_20': ema_20_current,
            'ema_50': ema_50_current,
            'cross': cross_up or cross_down,
            'direction': direction,
            'signal': signal,
            'distance': ema_20_current - ema_50_current
        }

    def is_price_above_ema(
        self,
        current_price: float,
        ema_period: int = 20,
        max_distance_ticks: int = 10,
        tick_size: float = 0.25
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if price is within acceptable distance above EMA.

        Args:
            current_price: Current market price
            ema_period: EMA period (20 or 50)
            max_distance_ticks: Maximum pullback distance in ticks
            tick_size: Instrument tick size

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        if ema_period == 20:
            ema = self.get_ema_20()
        elif ema_period == 50:
            ema = self.get_ema_50()
        else:
            return False, f"Invalid EMA period: {ema_period}"

        if ema is None:
            return False, "EMA not available"

        distance = current_price - ema
        distance_ticks = distance / tick_size

        if distance_ticks < 0:
            return False, f"Price below EMA{ema_period} ({abs(distance_ticks):.1f} ticks)"

        if distance_ticks > max_distance_ticks:
            return False, f"Price too far above EMA{ema_period} ({distance_ticks:.1f} > {max_distance_ticks} ticks)"

        return True, f"Price acceptable distance above EMA{ema_period} ({distance_ticks:.1f} ticks)"

    # MACD Methods

    def get_macd(self) -> Optional[float]:
        """Get current MACD line value."""
        return self.cache.get_latest('macd')

    def get_macd_signal(self) -> Optional[float]:
        """Get current MACD signal line value."""
        return self.cache.get_latest('macd_signal')

    def get_macd_histogram(self) -> Optional[float]:
        """Get current MACD histogram value."""
        return self.cache.get_latest('macd_histogram')

    def get_macd_analysis(self) -> Dict[str, any]:
        """
        Get MACD trading signal.

        Returns:
            Dict with MACD analysis
        """
        macd = self.get_macd()
        signal = self.get_macd_signal()
        histogram = self.get_macd_histogram()

        if macd is None or signal is None or histogram is None:
            return {
                'macd': None,
                'signal': None,
                'histogram': None,
                'trend': 'UNKNOWN',
                'cross': False,
                'trade_signal': 'NONE'
            }

        # Check for crossover
        macd_array = self.cache.get_array('macd', count=2)
        signal_array = self.cache.get_array('macd_signal', count=2)

        cross = False
        trade_signal = 'NONE'

        if len(macd_array) >= 2 and len(signal_array) >= 2:
            cross_up = (macd_array[-2] <= signal_array[-2]) and (macd_array[-1] > signal_array[-1])
            cross_down = (macd_array[-2] >= signal_array[-2]) and (macd_array[-1] < signal_array[-1])

            if cross_up:
                cross = True
                trade_signal = 'LONG'
            elif cross_down:
                cross = True
                trade_signal = 'SHORT'

        # Determine trend
        if histogram > 0:
            trend = 'BULLISH'
        elif histogram < 0:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'

        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram,
            'trend': trend,
            'cross': cross,
            'trade_signal': trade_signal
        }

    # ATR Methods

    def get_atr(self) -> Optional[float]:
        """Get current ATR value."""
        return self.cache.get_latest('atr')

    def get_atr_volatility_level(self) -> Dict[str, any]:
        """
        Determine volatility level based on ATR.

        Returns:
            Dict with volatility classification
        """
        atr = self.get_atr()

        if atr is None:
            return {
                'atr': None,
                'level': 'UNKNOWN',
                'stop_ticks': None,
                'target_ticks': None
            }

        # Define ATR-based volatility levels (for NQ)
        if atr < 20:
            level = 'LOW'
            stop_ticks = 6
            target_ticks = 10
        elif atr < 35:
            level = 'NORMAL'
            stop_ticks = 8
            target_ticks = 12
        else:
            level = 'HIGH'
            stop_ticks = 12
            target_ticks = 18

        return {
            'atr': atr,
            'level': level,
            'stop_ticks': stop_ticks,
            'target_ticks': target_ticks
        }

    def calculate_atr_stop_target(
        self,
        entry_price: float,
        direction: str,
        tick_size: float = 0.25,
        atr_multiplier_stop: float = 1.5,
        atr_multiplier_target: float = 2.5
    ) -> Dict[str, float]:
        """
        Calculate ATR-based stop and target.

        Args:
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            tick_size: Instrument tick size
            atr_multiplier_stop: ATR multiplier for stop
            atr_multiplier_target: ATR multiplier for target

        Returns:
            Dict with stop and target prices
        """
        atr = self.get_atr()

        if atr is None:
            # Fallback to fixed ticks
            atr = 20 * tick_size

        stop_distance = atr * atr_multiplier_stop
        target_distance = atr * atr_multiplier_target

        if direction.lower() == 'long':
            stop = entry_price - stop_distance
            target = entry_price + target_distance
        else:
            stop = entry_price + stop_distance
            target = entry_price - target_distance

        return {
            'stop': stop,
            'target': target,
            'atr': atr,
            'risk_reward': atr_multiplier_target / atr_multiplier_stop
        }

    # Summary Methods

    def get_all_indicators(self) -> Dict[str, Optional[float]]:
        """
        Get current values for all indicators.

        Returns:
            Dict of indicator name -> current value
        """
        return {
            'rsi': self.get_rsi(),
            'ema_20': self.get_ema_20(),
            'ema_50': self.get_ema_50(),
            'macd': self.get_macd(),
            'macd_signal': self.get_macd_signal(),
            'macd_histogram': self.get_macd_histogram(),
            'atr': self.get_atr()
        }

    def get_indicator_summary(self) -> Dict[str, any]:
        """
        Get comprehensive technical indicator summary.

        Returns:
            Dict with all indicator analyses
        """
        return {
            'values': self.get_all_indicators(),
            'rsi': self.get_rsi_signal(),
            'ema': self.get_ema_cross(),
            'macd': self.get_macd_analysis(),
            'atr': self.get_atr_volatility_level()
        }
