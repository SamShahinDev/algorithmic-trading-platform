"""
ICT Liquidity Sweep + Order Block Module

Detects liquidity sweeps of equal highs/lows followed by micro structure shifts,
then marks the order block (last counter candle before displacement) as entry zone.

Pattern:
1. Equal highs/lows formation
2. Sweep beyond equal level by minimum ticks
3. Micro structure shift in opposite direction
4. Order block from last counter candle becomes entry zone
"""

from typing import List, Dict, Any
from .shared import (
    has_equal_highs, has_equal_lows, swept_above, swept_below,
    micro_mss_bull, micro_mss_bear, get_order_block
)


def generate(bars: List[Any], i: int, cfg, tick: float) -> List[Dict[str, Any]]:
    """
    Generate liquidity sweep + order block entry candidates

    Args:
        bars: List of bar objects with OHLC data
        i: Current bar index
        cfg: Configuration object with ict_params
        tick: Tick size for calculations

    Returns:
        List of zone candidates with dir, lower, upper, tag
    """
    candidates = []

    try:
        if i < 3:  # Need at least 4 bars for pattern
            return candidates

        params = cfg.ict_params

        # Bullish pattern: Sweep sell-side liquidity (equal lows) then MSS up
        if _detect_bullish_liquidity_ob(bars, i, params, tick):
            ob = get_order_block(bars, i, "bullish")
            if ob:
                candidates.append({
                    'dir': 'long',
                    'lower': ob['body_lower'],  # Order block body as entry zone
                    'upper': ob['body_upper'],
                    'tag': 'ict_liquidity_ob',
                    'subtype': 'bullish_sweep_ob',
                    'quality_boost': 0.1  # Slight quality boost for liquidity concepts
                })

        # Bearish pattern: Sweep buy-side liquidity (equal highs) then MSS down
        if _detect_bearish_liquidity_ob(bars, i, params, tick):
            ob = get_order_block(bars, i, "bearish")
            if ob:
                candidates.append({
                    'dir': 'short',
                    'lower': ob['body_lower'],
                    'upper': ob['body_upper'],
                    'tag': 'ict_liquidity_ob',
                    'subtype': 'bearish_sweep_ob',
                    'quality_boost': 0.1
                })

    except Exception as e:
        # Fail silently to avoid disrupting main strategy
        pass

    return candidates


def _detect_bullish_liquidity_ob(bars: List[Any], i: int, params, tick: float) -> bool:
    """
    Detect bullish liquidity sweep + order block pattern

    Pattern:
    1. Equal lows in recent bars (sell-side liquidity)
    2. Current bar sweeps below those lows
    3. Micro structure shift bullish (close above recent high)

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        True if pattern detected
    """
    try:
        # 1. Check for equal lows in previous bar
        if not has_equal_lows(bars, i - 1, params.eqh_eql_window, tick):
            return False

        # 2. Check if current bar swept below the equal lows
        prev_low = bars[i - 1].low
        current_low = bars[i].low
        if not swept_below(prev_low, current_low, params.sweep_min_ticks, tick):
            return False

        # 3. Check for bullish micro structure shift
        if not micro_mss_bull(bars, i, params.mss_lookback):
            return False

        # Additional validation: ensure we have a real displacement candle
        current_bar = bars[i]
        body_size = abs(current_bar.close - current_bar.open)
        bar_range = current_bar.high - current_bar.low

        # Require meaningful body (at least 30% of range) for valid displacement
        if bar_range > 0 and (body_size / bar_range) < 0.3:
            return False

        return True

    except (AttributeError, IndexError, ZeroDivisionError):
        return False


def _detect_bearish_liquidity_ob(bars: List[Any], i: int, params, tick: float) -> bool:
    """
    Detect bearish liquidity sweep + order block pattern

    Pattern:
    1. Equal highs in recent bars (buy-side liquidity)
    2. Current bar sweeps above those highs
    3. Micro structure shift bearish (close below recent low)

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        True if pattern detected
    """
    try:
        # 1. Check for equal highs in previous bar
        if not has_equal_highs(bars, i - 1, params.eqh_eql_window, tick):
            return False

        # 2. Check if current bar swept above the equal highs
        prev_high = bars[i - 1].high
        current_high = bars[i].high
        if not swept_above(prev_high, current_high, params.sweep_min_ticks, tick):
            return False

        # 3. Check for bearish micro structure shift
        if not micro_mss_bear(bars, i, params.mss_lookback):
            return False

        # Additional validation: ensure we have a real displacement candle
        current_bar = bars[i]
        body_size = abs(current_bar.close - current_bar.open)
        bar_range = current_bar.high - current_bar.low

        # Require meaningful body (at least 30% of range) for valid displacement
        if bar_range > 0 and (body_size / bar_range) < 0.3:
            return False

        return True

    except (AttributeError, IndexError, ZeroDivisionError):
        return False


def get_pattern_info() -> Dict[str, Any]:
    """
    Get information about this pattern module

    Returns:
        Dict with pattern metadata
    """
    return {
        'name': 'ICT Liquidity Sweep + Order Block',
        'description': 'Detects liquidity sweeps of equal highs/lows with MSS and order block entry',
        'timeframes': ['1m', '5m'],
        'requirements': [
            'Equal highs/lows formation',
            'Liquidity sweep (min 1 tick)',
            'Micro structure shift opposite direction',
            'Valid displacement candle (30%+ body)'
        ],
        'entry_zones': 'Order block body (last counter candle)',
        'quality_factors': [
            'Displacement candle body fraction',
            'Sweep distance beyond equal level',
            'MSS strength and follow-through'
        ]
    }