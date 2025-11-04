"""
ICT Killzone Micro-Scalp Module

Detects micro-scalping opportunities during active killzones with bias alignment.
Focuses on tiny, high-probability setups with fast execution.

Pattern:
1. Must be in session killzone AND bias aligned
2. Micro sweep of local swing (1-2 ticks)
3. Micro MSS in bias direction
4. Tiny OB/FVG zone (≤ micro_max_zone_ticks)
5. Fast-path entry with short TTL
6. Per-session trade count limits
"""

from typing import List, Dict, Any, Optional
from .shared import (
    has_equal_highs, has_equal_lows, swept_above, swept_below,
    micro_mss_bull, micro_mss_bear, get_order_block
)


def generate(bars: List[Any], i: int, cfg, ict, tick: float, used_count: int) -> List[Dict[str, Any]]:
    """
    Generate killzone micro-scalp entry candidates

    Args:
        bars: List of bar objects with OHLC data
        i: Current bar index
        cfg: Configuration object with ict_params
        ict: ICT context for bias and killzone status
        tick: Tick size for calculations
        used_count: Number of micro-scalp trades already taken this session

    Returns:
        List of zone candidates with dir, lower, upper, tag
    """
    candidates = []

    try:
        if i < 3:  # Need minimum bars for pattern
            return candidates

        params = cfg.ict_params

        # ICT Micro Scalp now active 24/7 - removed killzone restriction
        # if not ict.session_killzone:
        #     return candidates

        if ict.bias_dir not in ("long", "short"):
            return candidates

        # Check session trade limit
        if used_count >= params.micro_max_trades_session:
            return candidates

        # Only scalp in bias direction
        if ict.bias_dir == "long":
            bullish_micro = _detect_bullish_micro_scalp(bars, i, params, tick)
            if bullish_micro:
                candidates.append(bullish_micro)

        elif ict.bias_dir == "short":
            bearish_micro = _detect_bearish_micro_scalp(bars, i, params, tick)
            if bearish_micro:
                candidates.append(bearish_micro)

    except Exception as e:
        # Fail silently to avoid disrupting main strategy
        pass

    return candidates


def _detect_bullish_micro_scalp(bars: List[Any], i: int, params, tick: float) -> Optional[Dict[str, Any]]:
    """
    Detect bullish micro-scalp pattern

    Pattern:
    1. Micro sweep below local lows (1-2 ticks)
    2. Micro MSS bullish (close above recent high)
    3. Tiny order block ≤ max zone ticks
    4. Fast-path entry preferred

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        Zone candidate dict or None
    """
    try:
        # 1. Check for micro sweep of local lows
        if not _detect_micro_sweep_below(bars, i, params, tick):
            return None

        # 2. Check for bullish micro MSS
        if not micro_mss_bull(bars, i, params.mss_lookback):
            return None

        # 3. Get order block from displacement
        ob = get_order_block(bars, i, "bullish")
        if not ob:
            return None

        # 4. Validate zone size (must be micro)
        zone_size = ob['body_upper'] - ob['body_lower']
        max_zone_size = params.micro_max_zone_ticks * tick

        if zone_size > max_zone_size:
            return None

        # 5. Additional quality checks for micro-scalp
        if not _validate_micro_quality(bars, i, "bullish", tick):
            return None

        return {
            'dir': 'long',
            'lower': ob['body_lower'],
            'upper': ob['body_upper'],
            'tag': 'ict_micro',
            'subtype': 'bullish_micro_scalp',
            'zone_size_ticks': zone_size / tick,
            'quality_boost': 0.08,  # Moderate boost for micro-scalp
            'fastpath': True,  # Enable fast-path entry
            'ttl_override': params.micro_fastpath_ttl_s,  # Short TTL
            'prefer_edge_entry': True  # Enter at zone edge ±1 tick
        }

    except (AttributeError, IndexError, ZeroDivisionError):
        return None


def _detect_bearish_micro_scalp(bars: List[Any], i: int, params, tick: float) -> Optional[Dict[str, Any]]:
    """
    Detect bearish micro-scalp pattern

    Pattern:
    1. Micro sweep above local highs (1-2 ticks)
    2. Micro MSS bearish (close below recent low)
    3. Tiny order block ≤ max zone ticks
    4. Fast-path entry preferred

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        Zone candidate dict or None
    """
    try:
        # 1. Check for micro sweep of local highs
        if not _detect_micro_sweep_above(bars, i, params, tick):
            return None

        # 2. Check for bearish micro MSS
        if not micro_mss_bear(bars, i, params.mss_lookback):
            return None

        # 3. Get order block from displacement
        ob = get_order_block(bars, i, "bearish")
        if not ob:
            return None

        # 4. Validate zone size (must be micro)
        zone_size = ob['body_upper'] - ob['body_lower']
        max_zone_size = params.micro_max_zone_ticks * tick

        if zone_size > max_zone_size:
            return None

        # 5. Additional quality checks for micro-scalp
        if not _validate_micro_quality(bars, i, "bearish", tick):
            return None

        return {
            'dir': 'short',
            'lower': ob['body_lower'],
            'upper': ob['body_upper'],
            'tag': 'ict_micro',
            'subtype': 'bearish_micro_scalp',
            'zone_size_ticks': zone_size / tick,
            'quality_boost': 0.08,  # Moderate boost for micro-scalp
            'fastpath': True,  # Enable fast-path entry
            'ttl_override': params.micro_fastpath_ttl_s,  # Short TTL
            'prefer_edge_entry': True  # Enter at zone edge ±1 tick
        }

    except (AttributeError, IndexError, ZeroDivisionError):
        return None


def _detect_micro_sweep_below(bars: List[Any], i: int, params, tick: float) -> bool:
    """
    Detect micro sweep below local lows

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        True if micro sweep detected
    """
    try:
        # Look for equal lows in very recent bars (micro timeframe)
        micro_window = min(5, params.eqh_eql_window)  # Smaller window for micro

        if not has_equal_lows(bars, i - 1, micro_window, tick, tol_ticks=1):
            return False

        # Check for micro sweep (1-3 ticks below)
        prev_low = bars[i - 1].low
        current_low = bars[i].low
        sweep_distance = prev_low - current_low

        return 1 * tick <= sweep_distance <= 3 * tick

    except (AttributeError, IndexError):
        return False


def _detect_micro_sweep_above(bars: List[Any], i: int, params, tick: float) -> bool:
    """
    Detect micro sweep above local highs

    Args:
        bars: Bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        True if micro sweep detected
    """
    try:
        # Look for equal highs in very recent bars (micro timeframe)
        micro_window = min(5, params.eqh_eql_window)  # Smaller window for micro

        if not has_equal_highs(bars, i - 1, micro_window, tick, tol_ticks=1):
            return False

        # Check for micro sweep (1-3 ticks above)
        prev_high = bars[i - 1].high
        current_high = bars[i].high
        sweep_distance = current_high - prev_high

        return 1 * tick <= sweep_distance <= 3 * tick

    except (AttributeError, IndexError):
        return False


def _validate_micro_quality(bars: List[Any], i: int, direction: str, tick: float) -> bool:
    """
    Validate quality for micro-scalp entry

    Args:
        bars: Bar data
        i: Current index
        direction: "bullish" or "bearish"
        tick: Tick size

    Returns:
        True if quality is sufficient for micro-scalp
    """
    try:
        current_bar = bars[i]

        # 1. Require meaningful displacement body
        body_size = abs(current_bar.close - current_bar.open)
        bar_range = current_bar.high - current_bar.low

        if bar_range <= 0:
            return False

        body_fraction = body_size / bar_range

        # Micro-scalp requires strong displacement (≥50% body)
        if body_fraction < 0.5:
            return False

        # 2. Check direction alignment
        if direction == "bullish":
            # Must close in upper half of bar for bullish
            close_position = (current_bar.close - current_bar.low) / bar_range
            if close_position < 0.6:  # Close in top 40%
                return False
        else:  # bearish
            # Must close in lower half of bar for bearish
            close_position = (current_bar.close - current_bar.low) / bar_range
            if close_position > 0.4:  # Close in bottom 40%
                return False

        # 3. Minimum bar range for micro-scalp (avoid noise)
        min_range = tick * 1.5  # At least 1.5 ticks range
        if bar_range < min_range:
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
        'name': 'ICT Killzone Micro-Scalp',
        'description': 'High-frequency micro-scalping during active killzones with bias alignment',
        'timeframes': ['1m only'],
        'requirements': [
            'Active session killzone',
            'Clear ICT bias alignment',
            'Micro sweep of local swing (1-3 ticks)',
            'Micro MSS in bias direction',
            'Tiny zone ≤ max zone ticks',
            'Strong displacement (≥50% body)',
            'Per-session trade limits'
        ],
        'entry_zones': 'Micro order block (≤2 ticks typically)',
        'execution': {
            'fast_path': True,
            'ttl_seconds': 30,
            'entry_style': 'Edge ±1 tick',
            'session_limit': 4
        },
        'quality_factors': [
            'Displacement strength (body fraction)',
            'Sweep precision (1-3 ticks ideal)',
            'MSS follow-through',
            'Zone size (smaller = better)',
            'Killzone timing'
        ],
        'notes': [
            'High-frequency pattern - many opportunities',
            'Requires active monitoring and fast execution',
            'Session limits prevent over-trading',
            'Best during high-volume killzones',
            'Conservative position sizing recommended'
        ]
    }