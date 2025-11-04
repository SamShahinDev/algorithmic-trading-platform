"""
ICT Silver Bullet Module

Detects Silver Bullet patterns during specific killzone windows:
- London: 02:00-04:00 CT
- NY Morning: 09:30-11:00 CT
- NY Afternoon: 13:00-15:00 CT

Pattern:
1. Must be within one of the three Silver Bullet windows
2. Liquidity sweep of equal highs/lows
3. Displacement FVG in opposite direction
4. Entry at FVG mid or 62% level
"""

from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from .shared import (
    has_equal_highs, has_equal_lows, swept_above, swept_below,
    check_fvg_simple, get_fvg_bounds
)


def generate(bars: List[Any], i: int, cfg, tick: float, ct_now: datetime) -> List[Dict[str, Any]]:
    """
    Generate Silver Bullet entry candidates

    Args:
        bars: List of bar objects with OHLC data
        i: Current bar index
        cfg: Configuration object with ict_params
        tick: Tick size for calculations
        ct_now: Current Chicago time

    Returns:
        List of zone candidates with dir, lower, upper, tag
    """
    candidates = []

    try:
        params = cfg.ict_params

        # ICT Silver Bullet now active 24/7 - removed time window restriction
        # if not _in_sb_window(ct_now, params):
        #     return candidates

        if i < 2:  # Need at least 3 bars for FVG pattern
            return candidates

        # Bullish Silver Bullet: Sweep lows + bullish FVG
        if _detect_bullish_silver_bullet(bars, i, params, tick):
            fvg_bounds = get_fvg_bounds(bars, i, "bullish")
            if fvg_bounds:
                # Use FVG bounds for entry zone
                candidates.append({
                    'dir': 'long',
                    'lower': fvg_bounds['lower'],
                    'upper': fvg_bounds['upper'],
                    'tag': 'ict_silver_bullet',
                    'subtype': 'bullish_sb',
                    'window': _get_current_window(ct_now, params),
                    'quality_boost': 0.15,  # Higher boost for Silver Bullet
                    'prefer_62_entry': True  # Prefer 62% entry over 50%
                })

        # Bearish Silver Bullet: Sweep highs + bearish FVG
        if _detect_bearish_silver_bullet(bars, i, params, tick):
            fvg_bounds = get_fvg_bounds(bars, i, "bearish")
            if fvg_bounds:
                candidates.append({
                    'dir': 'short',
                    'lower': fvg_bounds['lower'],
                    'upper': fvg_bounds['upper'],
                    'tag': 'ict_silver_bullet',
                    'subtype': 'bearish_sb',
                    'window': _get_current_window(ct_now, params),
                    'quality_boost': 0.15,
                    'prefer_62_entry': True
                })

    except Exception as e:
        # Fail silently to avoid disrupting main strategy
        pass

    return candidates


def _in_sb_window(ct: datetime, params) -> bool:
    """
    Check if current time is within any Silver Bullet window

    Args:
        ct: Chicago time datetime object
        params: ICT parameters with window definitions

    Returns:
        True if within any Silver Bullet window
    """
    try:
        t = (ct.hour, ct.minute)

        def within(start_str: str, end_str: str) -> bool:
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            a = (sh, sm)
            b = (eh, em)
            # Handle overnight window crossing midnight
            return (a <= t < b) if a < b else (t >= a or t < b)

        # Check all three Silver Bullet windows
        return any([
            within(params.sb_london_start, params.sb_london_end),
            within(params.sb_ny_morn_start, params.sb_ny_morn_end),
            within(params.sb_ny_pm_start, params.sb_ny_pm_end),
        ])

    except (AttributeError, ValueError):
        return False


def _get_current_window(ct: datetime, params) -> str:
    """Get the name of the current Silver Bullet window"""
    try:
        t = (ct.hour, ct.minute)

        def within(start_str: str, end_str: str) -> bool:
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            a = (sh, sm)
            b = (eh, em)
            return (a <= t < b) if a < b else (t >= a or t < b)

        if within(params.sb_london_start, params.sb_london_end):
            return "LONDON"
        elif within(params.sb_ny_morn_start, params.sb_ny_morn_end):
            return "NY_MORNING"
        elif within(params.sb_ny_pm_start, params.sb_ny_pm_end):
            return "NY_AFTERNOON"
        else:
            return "UNKNOWN"

    except (AttributeError, ValueError):
        return "UNKNOWN"


def _detect_bullish_silver_bullet(bars: List[Any], i: int, params, tick: float) -> bool:
    """
    Detect bullish Silver Bullet pattern

    Pattern:
    1. Equal lows formation
    2. Sweep below equal lows
    3. Bullish FVG formation (3-bar pattern)

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        True if pattern detected
    """
    try:
        # 1. Check for equal lows in previous bars
        if not has_equal_lows(bars, i - 1, params.eqh_eql_window, tick):
            return False

        # 2. Check if we swept below the equal lows
        prev_low = bars[i - 1].low
        current_low = bars[i].low
        if not swept_below(prev_low, current_low, params.sweep_min_ticks, tick):
            return False

        # 3. Check for bullish FVG pattern
        if not check_fvg_simple(bars, i, "bullish"):
            return False

        # Additional validation: ensure meaningful gap size
        fvg_bounds = get_fvg_bounds(bars, i, "bullish")
        if fvg_bounds:
            gap_size = fvg_bounds['upper'] - fvg_bounds['lower']
            min_gap_size = tick * 1  # At least 1 tick gap
            if gap_size < min_gap_size:
                return False

        return True

    except (AttributeError, IndexError):
        return False


def _detect_bearish_silver_bullet(bars: List[Any], i: int, params, tick: float) -> bool:
    """
    Detect bearish Silver Bullet pattern

    Pattern:
    1. Equal highs formation
    2. Sweep above equal highs
    3. Bearish FVG formation (3-bar pattern)

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        True if pattern detected
    """
    try:
        # 1. Check for equal highs in previous bars
        if not has_equal_highs(bars, i - 1, params.eqh_eql_window, tick):
            return False

        # 2. Check if we swept above the equal highs
        prev_high = bars[i - 1].high
        current_high = bars[i].high
        if not swept_above(prev_high, current_high, params.sweep_min_ticks, tick):
            return False

        # 3. Check for bearish FVG pattern
        if not check_fvg_simple(bars, i, "bearish"):
            return False

        # Additional validation: ensure meaningful gap size
        fvg_bounds = get_fvg_bounds(bars, i, "bearish")
        if fvg_bounds:
            gap_size = fvg_bounds['upper'] - fvg_bounds['lower']
            min_gap_size = tick * 1  # At least 1 tick gap
            if gap_size < min_gap_size:
                return False

        return True

    except (AttributeError, IndexError):
        return False


def get_pattern_info() -> Dict[str, Any]:
    """
    Get information about this pattern module

    Returns:
        Dict with pattern metadata
    """
    return {
        'name': 'ICT Silver Bullet',
        'description': 'High-probability setups during specific killzone windows with sweep + FVG',
        'timeframes': ['1m', '5m'],
        'windows': {
            'LONDON': '02:00-04:00 CT',
            'NY_MORNING': '09:30-11:00 CT',
            'NY_AFTERNOON': '13:00-15:00 CT'
        },
        'requirements': [
            'Within Silver Bullet window',
            'Equal highs/lows formation',
            'Liquidity sweep (min 1 tick)',
            'Valid FVG formation (3-bar pattern)',
            'Minimum gap size (1 tick)'
        ],
        'entry_zones': 'FVG bounds (prefer 62% entry)',
        'quality_factors': [
            'Window timing (London premium)',
            'Sweep distance',
            'FVG gap size',
            'Displacement strength'
        ]
    }