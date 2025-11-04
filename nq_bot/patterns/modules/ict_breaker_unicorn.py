"""
ICT Breaker + FVG "Unicorn" Module

Detects the rare "Unicorn" pattern where a failed order block becomes a breaker
and overlaps with an FVG in the new direction.

Pattern:
1. Order block formation (counter-trend candle)
2. Order block gets violated (close through OB body) = becomes breaker
3. FVG forms in new direction
4. FVG overlaps with the breaker zone
5. Entry at the overlap zone (highest probability area)
"""

from typing import List, Dict, Any, Optional
from .shared import (
    get_order_block, check_fvg_simple, get_fvg_bounds
)


def generate(bars: List[Any], i: int, cfg, tick: float) -> List[Dict[str, Any]]:
    """
    Generate Breaker + FVG "Unicorn" entry candidates

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
        if i < 5:  # Need sufficient bars for pattern development
            return candidates

        # Look for breaker + FVG overlap patterns
        bullish_unicorn = _detect_bullish_unicorn(bars, i, cfg, tick)
        if bullish_unicorn:
            candidates.append(bullish_unicorn)

        bearish_unicorn = _detect_bearish_unicorn(bars, i, cfg, tick)
        if bearish_unicorn:
            candidates.append(bearish_unicorn)

    except Exception as e:
        # Fail silently to avoid disrupting main strategy
        pass

    return candidates


def _detect_bullish_unicorn(bars: List[Any], i: int, cfg, tick: float) -> Optional[Dict[str, Any]]:
    """
    Detect bullish Breaker + FVG "Unicorn" pattern

    Pattern:
    1. Find recent bearish order block that got violated (became breaker)
    2. Current bars form bullish FVG
    3. FVG overlaps with breaker zone
    4. Entry at overlap area

    Args:
        bars: Bar data
        i: Current index
        cfg: Configuration
        tick: Tick size

    Returns:
        Zone candidate dict or None
    """
    try:
        # 1. Check for current bullish FVG
        if not check_fvg_simple(bars, i, "bullish"):
            return None

        fvg_bounds = get_fvg_bounds(bars, i, "bullish")
        if not fvg_bounds:
            return None

        # 2. Look for recent breaker (failed bearish OB) in the lookback period
        lookback = min(10, i - 2)  # Look back up to 10 bars
        for j in range(i - 3, i - lookback - 1, -1):  # Start from 3 bars back
            if j < 1:
                break

            # Check if this could have been a bearish order block
            potential_ob = get_order_block(bars, j, "bearish")
            if not potential_ob:
                continue

            # Check if the order block got violated (price closed above OB body)
            violated = _check_ob_violation(bars, j, i, potential_ob, "bearish")
            if not violated:
                continue

            # 3. Check for overlap between breaker (failed OB) and current FVG
            overlap = _calculate_overlap(potential_ob, fvg_bounds)
            if not overlap or overlap['size'] < tick * 0.5:  # Minimum overlap
                continue

            # Found a Unicorn pattern!
            return {
                'dir': 'long',
                'lower': overlap['lower'],
                'upper': overlap['upper'],
                'tag': 'ict_unicorn',
                'subtype': 'bullish_breaker_fvg',
                'breaker_bar': j,
                'fvg_bar': i,
                'overlap_size': overlap['size'],
                'quality_boost': 0.2,  # High boost for rare Unicorn pattern
                'prefer_62_entry': True
            }

        return None

    except (AttributeError, IndexError):
        return None


def _detect_bearish_unicorn(bars: List[Any], i: int, cfg, tick: float) -> Optional[Dict[str, Any]]:
    """
    Detect bearish Breaker + FVG "Unicorn" pattern

    Pattern:
    1. Find recent bullish order block that got violated (became breaker)
    2. Current bars form bearish FVG
    3. FVG overlaps with breaker zone
    4. Entry at overlap area

    Args:
        bars: Bar data
        i: Current index
        cfg: Configuration
        tick: Tick size

    Returns:
        Zone candidate dict or None
    """
    try:
        # 1. Check for current bearish FVG
        if not check_fvg_simple(bars, i, "bearish"):
            return None

        fvg_bounds = get_fvg_bounds(bars, i, "bearish")
        if not fvg_bounds:
            return None

        # 2. Look for recent breaker (failed bullish OB) in the lookback period
        lookback = min(10, i - 2)  # Look back up to 10 bars
        for j in range(i - 3, i - lookback - 1, -1):  # Start from 3 bars back
            if j < 1:
                break

            # Check if this could have been a bullish order block
            potential_ob = get_order_block(bars, j, "bullish")
            if not potential_ob:
                continue

            # Check if the order block got violated (price closed below OB body)
            violated = _check_ob_violation(bars, j, i, potential_ob, "bullish")
            if not violated:
                continue

            # 3. Check for overlap between breaker (failed OB) and current FVG
            overlap = _calculate_overlap(potential_ob, fvg_bounds)
            if not overlap or overlap['size'] < tick * 0.5:  # Minimum overlap
                continue

            # Found a Unicorn pattern!
            return {
                'dir': 'short',
                'lower': overlap['lower'],
                'upper': overlap['upper'],
                'tag': 'ict_unicorn',
                'subtype': 'bearish_breaker_fvg',
                'breaker_bar': j,
                'fvg_bar': i,
                'overlap_size': overlap['size'],
                'quality_boost': 0.2,  # High boost for rare Unicorn pattern
                'prefer_62_entry': True
            }

        return None

    except (AttributeError, IndexError):
        return None


def _check_ob_violation(bars: List[Any], ob_bar_idx: int, current_idx: int,
                       ob_data: Dict[str, Any], ob_direction: str) -> bool:
    """
    Check if an order block got violated (became a breaker)

    Args:
        bars: Bar data
        ob_bar_idx: Index where OB was formed
        current_idx: Current bar index
        ob_data: Order block data with bounds
        ob_direction: "bullish" or "bearish" OB

    Returns:
        True if OB was violated (became breaker)
    """
    try:
        if ob_direction == "bearish":
            # Bearish OB violated when price closes above OB body
            violation_level = ob_data['body_upper']
            for k in range(ob_bar_idx + 1, current_idx):
                if k >= len(bars):
                    break
                if bars[k].close > violation_level:
                    return True
        else:  # bullish
            # Bullish OB violated when price closes below OB body
            violation_level = ob_data['body_lower']
            for k in range(ob_bar_idx + 1, current_idx):
                if k >= len(bars):
                    break
                if bars[k].close < violation_level:
                    return True

        return False

    except (AttributeError, IndexError, KeyError):
        return False


def _calculate_overlap(ob_data: Dict[str, Any], fvg_bounds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calculate overlap between order block and FVG

    Args:
        ob_data: Order block data with body bounds
        fvg_bounds: FVG bounds

    Returns:
        Dict with overlap lower, upper, size or None if no overlap
    """
    try:
        # Use OB body for overlap calculation (more precise than full OB range)
        ob_lower = ob_data['body_lower']
        ob_upper = ob_data['body_upper']

        fvg_lower = fvg_bounds['lower']
        fvg_upper = fvg_bounds['upper']

        # Calculate overlap
        overlap_lower = max(ob_lower, fvg_lower)
        overlap_upper = min(ob_upper, fvg_upper)

        # Check if there's actual overlap
        if overlap_lower >= overlap_upper:
            return None

        overlap_size = overlap_upper - overlap_lower

        return {
            'lower': overlap_lower,
            'upper': overlap_upper,
            'size': overlap_size
        }

    except (KeyError, TypeError):
        return None


def get_pattern_info() -> Dict[str, Any]:
    """
    Get information about this pattern module

    Returns:
        Dict with pattern metadata
    """
    return {
        'name': 'ICT Breaker + FVG "Unicorn"',
        'description': 'Rare high-probability pattern where failed OB becomes breaker and overlaps with FVG',
        'timeframes': ['1m', '5m'],
        'rarity': 'Very Rare (Unicorn)',
        'requirements': [
            'Failed order block (becomes breaker)',
            'Valid FVG formation in new direction',
            'Meaningful overlap between breaker and FVG',
            'Minimum overlap size (0.5 ticks)'
        ],
        'entry_zones': 'Overlap area between breaker and FVG',
        'quality_factors': [
            'Overlap size and percentage',
            'FVG gap quality',
            'Breaker strength (how decisively OB failed)',
            'Time proximity between breaker and FVG'
        ],
        'notes': [
            'Very rare pattern - expect few occurrences',
            'High quality boost when detected',
            'Prefer 62% entry within overlap zone',
            'Strong confluence of two ICT concepts'
        ]
    }