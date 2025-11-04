"""
Shared ICT Helper Functions

Common utilities for ICT pattern detection including:
- Equal highs/lows detection
- Liquidity sweep detection
- Micro structure shifts (MSS)
- Change of character (CHOCH) detection
"""

from typing import List, Any, Union


def _get_bar_value(bar: Any, attr: str) -> float:
    """
    Helper function to safely extract values from bar objects

    Args:
        bar: Bar object (could be pandas Series, dict, or object with attributes)
        attr: Attribute name ('high', 'low', 'open', 'close')

    Returns:
        Float value or 0.0 if not found
    """
    try:
        if hasattr(bar, attr):
            return float(getattr(bar, attr))
        elif hasattr(bar, 'get') and callable(getattr(bar, 'get')):
            return float(bar.get(attr, 0.0))
        elif hasattr(bar, '__getitem__'):
            return float(bar[attr])
        else:
            return 0.0
    except (KeyError, TypeError, ValueError, AttributeError):
        return 0.0


def has_equal_highs(bars: List[Any], i: int, lookback: int, tick: float, tol_ticks: int = 1) -> bool:
    """
    Check if current bar has equal highs with any bar in lookback period with enhanced tolerance

    Args:
        bars: List of bar objects with .high attribute
        i: Current bar index
        lookback: Number of bars to look back
        tick: Tick size for comparison
        tol_ticks: Tolerance in ticks (enhanced default: 2)

    Returns:
        True if equal highs found within tolerance
    """
    try:
        if i < 1 or i >= len(bars):
            return False

        # Enhanced tolerance for better pattern recognition
        tolerance_ticks = max(tol_ticks, 2)  # Minimum 2 tick tolerance
        current_high = _get_bar_value(bars[i], 'high')
        start_idx = max(0, i - lookback)

        for j in range(i - 1, start_idx - 1, -1):
            bar_high = _get_bar_value(bars[j], 'high')
            if abs(bar_high - current_high) <= tolerance_ticks * tick:
                return True

        return False
    except (AttributeError, IndexError, TypeError):
        return False


def has_equal_lows(bars: List[Any], i: int, lookback: int, tick: float, tol_ticks: int = 1) -> bool:
    """
    Check if current bar has equal lows with any bar in lookback period with enhanced tolerance

    Args:
        bars: List of bar objects with .low attribute
        i: Current bar index
        lookback: Number of bars to look back
        tick: Tick size for comparison
        tol_ticks: Tolerance in ticks (enhanced default: 2)

    Returns:
        True if equal lows found within tolerance
    """
    try:
        if i < 1 or i >= len(bars):
            return False

        # Enhanced tolerance for better pattern recognition
        tolerance_ticks = max(tol_ticks, 2)  # Minimum 2 tick tolerance
        current_low = _get_bar_value(bars[i], 'low')
        start_idx = max(0, i - lookback)

        for j in range(i - 1, start_idx - 1, -1):
            bar_low = _get_bar_value(bars[j], 'low')
            if abs(bar_low - current_low) <= tolerance_ticks * tick:
                return True

        return False
    except (AttributeError, IndexError, TypeError):
        return False


def swept_above(prev_hi: float, bar_hi: float, min_ticks: int, tick: float) -> bool:
    """
    Check if bar swept above previous high by minimum ticks

    Args:
        prev_hi: Previous high level
        bar_hi: Current bar high
        min_ticks: Minimum ticks required for sweep
        tick: Tick size

    Returns:
        True if swept above by min_ticks or more
    """
    return (bar_hi - prev_hi) >= (min_ticks * tick)


def swept_below(prev_lo: float, bar_lo: float, min_ticks: int, tick: float) -> bool:
    """
    Check if bar swept below previous low by minimum ticks

    Args:
        prev_lo: Previous low level
        bar_lo: Current bar low
        min_ticks: Minimum ticks required for sweep
        tick: Tick size

    Returns:
        True if swept below by min_ticks or more
    """
    return (prev_lo - bar_lo) >= (min_ticks * tick)


def micro_mss_bull(bars: List[Any], i: int, lookback: int) -> bool:
    """
    Detect bullish micro structure shift (price closes above recent high)

    Args:
        bars: List of bar objects with .high and .close attributes
        i: Current bar index
        lookback: Number of bars to check for previous high

    Returns:
        True if current close is above recent high (bullish MSS)
    """
    try:
        if i < lookback or i >= len(bars):
            return False

        # Find highest high in lookback period (excluding current bar)
        start_idx = max(0, i - lookback)
        prev_high = max(bars[j].high for j in range(start_idx, i))

        # Check if current close breaks above previous high
        return bars[i].close > prev_high

    except (AttributeError, IndexError, ValueError):
        return False


def micro_mss_bear(bars: List[Any], i: int, lookback: int) -> bool:
    """
    Detect bearish micro structure shift (price closes below recent low)

    Args:
        bars: List of bar objects with .low and .close attributes
        i: Current bar index
        lookback: Number of bars to check for previous low

    Returns:
        True if current close is below recent low (bearish MSS)
    """
    try:
        if i < lookback or i >= len(bars):
            return False

        # Find lowest low in lookback period (excluding current bar)
        start_idx = max(0, i - lookback)
        prev_low = min(bars[j].low for j in range(start_idx, i))

        # Check if current close breaks below previous low
        return bars[i].close < prev_low

    except (AttributeError, IndexError, ValueError):
        return False


def get_order_block(bars: List[Any], i: int, direction: str) -> dict:
    """
    Get order block (last counter-trend candle before displacement)

    Args:
        bars: List of bar objects
        i: Current bar index
        direction: "bullish" or "bearish" for the expected displacement

    Returns:
        Dict with 'lower', 'upper', 'body_lower', 'body_upper' of the order block
    """
    try:
        if i < 1 or i >= len(bars):
            return {}

        # For bullish displacement, find last bearish candle
        # For bearish displacement, find last bullish candle
        ob_bar = bars[i - 1]  # Simple: use previous bar as order block

        body_lower = min(ob_bar.open, ob_bar.close)
        body_upper = max(ob_bar.open, ob_bar.close)

        return {
            'lower': ob_bar.low,
            'upper': ob_bar.high,
            'body_lower': body_lower,
            'body_upper': body_upper,
            'bar': ob_bar
        }

    except (AttributeError, IndexError):
        return {}


def check_fvg_simple(bars: List[Any], i: int, direction: str) -> bool:
    """
    Simple FVG check for 3-bar pattern

    Args:
        bars: List of bar objects
        i: Current bar index (should be the 3rd bar of potential FVG)
        direction: "bullish" or "bearish"

    Returns:
        True if valid FVG pattern detected
    """
    try:
        if i < 2 or i >= len(bars):
            return False

        bar_before = bars[i - 2]  # First bar
        bar_middle = bars[i - 1]  # Middle bar (creates gap)
        bar_current = bars[i]     # Current bar

        if direction == "bullish":
            # Bullish FVG: middle bar low > before bar high, current bar confirms gap
            return (bar_middle.low > bar_before.high and
                    bar_current.low > bar_before.high)
        elif direction == "bearish":
            # Bearish FVG: middle bar high < before bar low, current bar confirms gap
            return (bar_middle.high < bar_before.low and
                    bar_current.high < bar_before.low)
        else:
            return False

    except (AttributeError, IndexError):
        return False


def get_fvg_bounds(bars: List[Any], i: int, direction: str) -> dict:
    """
    Get FVG zone boundaries from 3-bar pattern

    Args:
        bars: List of bar objects
        i: Current bar index
        direction: "bullish" or "bearish"

    Returns:
        Dict with 'lower', 'upper' bounds of the FVG
    """
    try:
        if i < 2 or i >= len(bars):
            return {}

        bar_before = bars[i - 2]
        bar_middle = bars[i - 1]

        if direction == "bullish":
            return {
                'lower': bar_before.high,  # Bottom of gap
                'upper': bar_middle.low    # Top of gap
            }
        elif direction == "bearish":
            return {
                'lower': bar_middle.high,  # Bottom of gap
                'upper': bar_before.low    # Top of gap
            }
        else:
            return {}

    except (AttributeError, IndexError):
        return {}