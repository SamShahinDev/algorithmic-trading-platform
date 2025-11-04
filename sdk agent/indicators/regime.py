"""
Market regime detection module.

This module identifies current market conditions:
- ADX-based trending vs ranging detection
- VWAP slope for trend direction
- Regime classification for strategy selection
"""

import numpy as np
from typing import Optional, Dict
from enum import Enum
from .cache import IndicatorCache


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Market regime detection system.

    Uses ADX and VWAP slope to classify market conditions.
    """

    def __init__(
        self,
        cache: IndicatorCache,
        adx_trending_threshold: float = 25.0,
        vwap_slope_lookback: int = 30
    ):
        """
        Initialize regime detector.

        Args:
            cache: IndicatorCache instance
            adx_trending_threshold: ADX threshold for trending market (default 25)
            vwap_slope_lookback: Lookback period for VWAP slope (default 30 bars)
        """
        self.cache = cache
        self.adx_trending_threshold = adx_trending_threshold
        self.vwap_slope_lookback = vwap_slope_lookback

    def get_regime(self) -> MarketRegime:
        """
        Determine current market regime.

        Returns:
            MarketRegime enum value
        """
        adx = self.cache.get_latest('adx')

        if adx is None:
            return MarketRegime.UNKNOWN

        # Check if trending or ranging based on ADX
        if adx >= self.adx_trending_threshold:
            # Trending market - determine direction
            vwap_slope = self.get_vwap_slope()

            if vwap_slope is None:
                return MarketRegime.UNKNOWN

            # Positive slope = uptrend, negative = downtrend
            if vwap_slope > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        else:
            # Ranging market (ADX below threshold)
            # Could add volatility check here
            atr = self.cache.get_latest('atr')

            if atr is not None:
                # Get ATR average over recent bars
                atr_array = self.cache.get_array('atr', count=20)
                if len(atr_array) > 0:
                    atr_avg = np.mean(atr_array)
                    # If current ATR is significantly above average, it's volatile
                    if atr > atr_avg * 1.5:
                        return MarketRegime.VOLATILE

            return MarketRegime.RANGING

    def get_adx(self) -> Optional[float]:
        """
        Get current ADX value.

        Returns:
            ADX value or None
        """
        return self.cache.get_latest('adx')

    def get_vwap_slope(self) -> Optional[float]:
        """
        Calculate VWAP slope.

        Returns:
            Slope value or None (positive = uptrend, negative = downtrend)
        """
        vwap_array = self.cache.get_array('vwap', count=self.vwap_slope_lookback)

        if len(vwap_array) < 2:
            return None

        # Linear regression slope
        x = np.arange(len(vwap_array))
        slope = np.polyfit(x, vwap_array, 1)[0]

        return float(slope)

    def is_trending(self) -> bool:
        """
        Check if market is currently trending.

        Returns:
            True if trending (up or down)
        """
        regime = self.get_regime()
        return regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def is_ranging(self) -> bool:
        """
        Check if market is currently ranging.

        Returns:
            True if ranging
        """
        return self.get_regime() == MarketRegime.RANGING

    def is_volatile(self) -> bool:
        """
        Check if market is currently volatile.

        Returns:
            True if volatile
        """
        return self.get_regime() == MarketRegime.VOLATILE

    def get_trend_strength(self) -> Optional[float]:
        """
        Get trend strength (0-100).

        Returns:
            Trend strength where ADX is the measure (higher = stronger trend)
        """
        adx = self.get_adx()
        if adx is None:
            return None

        # Normalize ADX to 0-100 scale (cap at 100)
        return min(adx, 100.0)

    def get_regime_summary(self) -> Dict[str, any]:
        """
        Get comprehensive regime analysis.

        Returns:
            Dict with regime metrics
        """
        regime = self.get_regime()
        adx = self.get_adx()
        vwap_slope = self.get_vwap_slope()
        trend_strength = self.get_trend_strength()

        # Get additional context
        atr = self.cache.get_latest('atr')
        atr_array = self.cache.get_array('atr', count=20)
        atr_avg = np.mean(atr_array) if len(atr_array) > 0 else None

        return {
            'regime': regime.value,
            'is_trending': self.is_trending(),
            'is_ranging': self.is_ranging(),
            'is_volatile': self.is_volatile(),
            'adx': adx,
            'adx_threshold': self.adx_trending_threshold,
            'vwap_slope': vwap_slope,
            'trend_strength': trend_strength,
            'atr': atr,
            'atr_avg': atr_avg,
            'atr_ratio': atr / atr_avg if atr and atr_avg and atr_avg > 0 else None
        }

    def should_use_strategy(self, strategy_name: str) -> tuple[bool, Optional[str]]:
        """
        Determine if given strategy is appropriate for current regime.

        Args:
            strategy_name: Strategy name ('vwap', 'breakout', 'momentum')

        Returns:
            Tuple of (should_use: bool, reason: str)
        """
        regime = self.get_regime()

        if regime == MarketRegime.UNKNOWN:
            return False, "Market regime unknown - insufficient data"

        # Strategy-regime matching
        if strategy_name.lower() == 'vwap':
            # VWAP works best in ranging markets
            if regime == MarketRegime.RANGING:
                return True, "Ranging market ideal for VWAP mean reversion"
            elif regime == MarketRegime.VOLATILE:
                return False, "Too volatile for VWAP strategy"
            elif self.is_trending():
                adx = self.get_adx()
                if adx and adx > 35:
                    return False, f"Too trending for VWAP (ADX={adx:.1f})"
                return True, "Mild trend acceptable for VWAP"

        elif strategy_name.lower() == 'breakout':
            # Breakout works in volatile/transitioning markets
            if regime == MarketRegime.VOLATILE:
                return True, "Volatile market good for breakouts"
            elif regime == MarketRegime.RANGING:
                # Check if consolidating
                atr_array = self.cache.get_array('atr', count=20)
                if len(atr_array) > 5:
                    recent_atr = np.mean(atr_array[-5:])
                    avg_atr = np.mean(atr_array)
                    if recent_atr < avg_atr * 0.8:
                        return True, "Consolidating range - good for breakout"
                return False, "Range too stable for breakouts"
            elif self.is_trending():
                return False, "Already trending - not ideal for breakouts"

        elif strategy_name.lower() == 'momentum':
            # Momentum works best in trending markets
            if self.is_trending():
                return True, f"Trending market ({regime.value}) ideal for momentum"
            elif regime == MarketRegime.RANGING:
                return False, "Ranging market not suitable for momentum"
            elif regime == MarketRegime.VOLATILE:
                # Check for directional bias
                vwap_slope = self.get_vwap_slope()
                if vwap_slope and abs(vwap_slope) > 0.1:
                    return True, "Volatile with directional bias"
                return False, "Volatile but no clear direction"

        return False, f"No rule for strategy '{strategy_name}'"
