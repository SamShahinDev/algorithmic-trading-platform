"""
VWAP (Volume Weighted Average Price) indicator module.

This module provides VWAP-specific calculations and analysis:
- VWAP calculation from bars
- VWAP standard deviation bands (1.5σ and 2.5σ)
- Distance from VWAP in ticks
- Entry zone identification
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .cache import IndicatorCache


class VWAPAnalyzer:
    """
    VWAP analyzer for trading signals.

    Provides VWAP-based metrics and entry zone identification.
    """

    def __init__(self, cache: IndicatorCache, tick_size: float = 0.25):
        """
        Initialize VWAP analyzer.

        Args:
            cache: IndicatorCache instance
            tick_size: Instrument tick size (default 0.25 for NQ)
        """
        self.cache = cache
        self.tick_size = tick_size

    def get_vwap(self) -> Optional[float]:
        """
        Get current VWAP value.

        Returns:
            Current VWAP or None
        """
        return self.cache.get_latest('vwap')

    def get_vwap_std(self) -> Optional[float]:
        """
        Get current VWAP standard deviation.

        Returns:
            Current standard deviation or None
        """
        return self.cache.get_latest('vwap_std')

    def get_vwap_bands(
        self,
        std_dev_1: float = 1.5,
        std_dev_2: float = 2.5
    ) -> Dict[str, Optional[float]]:
        """
        Calculate VWAP bands at specified standard deviations.

        Args:
            std_dev_1: First standard deviation multiplier (default 1.5)
            std_dev_2: Second standard deviation multiplier (default 2.5)

        Returns:
            Dict with band levels:
                - vwap: VWAP value
                - upper_1: Upper band at std_dev_1
                - lower_1: Lower band at std_dev_1
                - upper_2: Upper band at std_dev_2
                - lower_2: Lower band at std_dev_2
                - std: Standard deviation
        """
        vwap = self.get_vwap()
        vwap_std = self.get_vwap_std()

        if vwap is None or vwap_std is None:
            return {
                'vwap': None,
                'upper_1': None,
                'lower_1': None,
                'upper_2': None,
                'lower_2': None,
                'std': None
            }

        return {
            'vwap': vwap,
            'upper_1': vwap + (std_dev_1 * vwap_std),
            'lower_1': vwap - (std_dev_1 * vwap_std),
            'upper_2': vwap + (std_dev_2 * vwap_std),
            'lower_2': vwap - (std_dev_2 * vwap_std),
            'std': vwap_std
        }

    def get_distance_from_vwap(self, current_price: float) -> Dict[str, float]:
        """
        Calculate distance from VWAP in various units.

        Args:
            current_price: Current market price

        Returns:
            Dict with:
                - dollars: Distance in dollars
                - ticks: Distance in ticks
                - std_dev: Distance in standard deviations
                - pct: Distance as percentage
        """
        vwap = self.get_vwap()
        vwap_std = self.get_vwap_std()

        if vwap is None:
            return {
                'dollars': 0.0,
                'ticks': 0.0,
                'std_dev': 0.0,
                'pct': 0.0
            }

        # Calculate distance
        distance_dollars = current_price - vwap
        distance_ticks = distance_dollars / self.tick_size

        # Calculate in standard deviations
        if vwap_std and vwap_std > 0:
            distance_std_dev = distance_dollars / vwap_std
        else:
            distance_std_dev = 0.0

        # Calculate percentage
        distance_pct = (distance_dollars / vwap) * 100 if vwap > 0 else 0.0

        return {
            'dollars': distance_dollars,
            'ticks': distance_ticks,
            'std_dev': distance_std_dev,
            'pct': distance_pct
        }

    def is_in_entry_zone(
        self,
        current_price: float,
        std_dev_min: float = 1.5,
        std_dev_max: float = 2.5,
        direction: str = 'long'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if price is in VWAP entry zone.

        Args:
            current_price: Current market price
            std_dev_min: Minimum standard deviations from VWAP
            std_dev_max: Maximum standard deviations from VWAP
            direction: Trade direction ('long' or 'short')

        Returns:
            Tuple of (in_zone: bool, reason: str)
        """
        vwap = self.get_vwap()
        vwap_std = self.get_vwap_std()

        if vwap is None or vwap_std is None:
            return False, "VWAP not available"

        distance = self.get_distance_from_vwap(current_price)
        std_dev_distance = abs(distance['std_dev'])

        # Check if in range
        if std_dev_distance < std_dev_min:
            return False, f"Too close to VWAP ({std_dev_distance:.2f}σ < {std_dev_min}σ)"

        if std_dev_distance > std_dev_max:
            return False, f"Too far from VWAP ({std_dev_distance:.2f}σ > {std_dev_max}σ)"

        # Check correct side for direction
        if direction.lower() == 'long':
            if current_price > vwap:
                return False, "Price above VWAP (need pullback below for long)"
            return True, f"In long entry zone ({std_dev_distance:.2f}σ below VWAP)"

        elif direction.lower() == 'short':
            if current_price < vwap:
                return False, "Price below VWAP (need rally above for short)"
            return True, f"In short entry zone ({std_dev_distance:.2f}σ above VWAP)"

        return False, "Invalid direction"

    def get_vwap_slope(self, lookback_bars: int = 30) -> Optional[float]:
        """
        Calculate VWAP slope over lookback period.

        Args:
            lookback_bars: Number of bars for slope calculation

        Returns:
            Slope value or None (positive = uptrend, negative = downtrend)
        """
        vwap_array = self.cache.get_array('vwap', count=lookback_bars)

        if len(vwap_array) < 2:
            return None

        # Simple linear regression slope
        x = np.arange(len(vwap_array))
        slope = np.polyfit(x, vwap_array, 1)[0]

        return float(slope)

    def get_vwap_summary(self, current_price: float) -> Dict[str, Any]:
        """
        Get comprehensive VWAP summary.

        Args:
            current_price: Current market price

        Returns:
            Dict with all VWAP metrics
        """
        vwap = self.get_vwap()
        vwap_std = self.get_vwap_std()
        bands = self.get_vwap_bands()
        distance = self.get_distance_from_vwap(current_price)
        slope = self.get_vwap_slope()

        in_long_zone, long_reason = self.is_in_entry_zone(current_price, direction='long')
        in_short_zone, short_reason = self.is_in_entry_zone(current_price, direction='short')

        return {
            'vwap': vwap,
            'std': vwap_std,
            'bands': bands,
            'distance': distance,
            'slope': slope,
            'current_price': current_price,
            'long_entry_zone': in_long_zone,
            'long_entry_reason': long_reason,
            'short_entry_zone': in_short_zone,
            'short_entry_reason': short_reason
        }

    def calculate_target_stop(
        self,
        entry_price: float,
        direction: str,
        target_ticks: int = 12,
        stop_ticks: int = 8
    ) -> Dict[str, float]:
        """
        Calculate target and stop based on VWAP entry.

        Args:
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            target_ticks: Target in ticks
            stop_ticks: Stop in ticks

        Returns:
            Dict with target and stop prices
        """
        if direction.lower() == 'long':
            target = entry_price + (target_ticks * self.tick_size)
            stop = entry_price - (stop_ticks * self.tick_size)
        else:
            target = entry_price - (target_ticks * self.tick_size)
            stop = entry_price + (stop_ticks * self.tick_size)

        return {
            'target': target,
            'stop': stop,
            'risk_reward': target_ticks / stop_ticks
        }
