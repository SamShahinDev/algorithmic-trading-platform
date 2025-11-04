"""
ICT FVG Continuation Module

Detects trend continuation setups using higher timeframe bias and FVG retracements.

Pattern:
1. Clear HTF bias (from ICT context or 15m swing analysis)
2. Recent impulse move ≥ minimum points threshold
3. HTF FVG formation (simulated via larger retracement analysis)
4. LTF price retraces into HTF FVG area
5. Entry at FVG levels for continuation in bias direction
"""

from typing import List, Dict, Any, Optional
from .shared import check_fvg_simple, get_fvg_bounds


def generate(htf_ctx, bars_ltf: List[Any], i_ltf: int, cfg, tick: float) -> List[Dict[str, Any]]:
    """
    Generate FVG continuation entry candidates

    Args:
        htf_ctx: ICT context with bias and continuation data
        bars_ltf: LTF (1m) bar data
        i_ltf: Current LTF bar index
        cfg: Configuration object with ict_params
        tick: Tick size for calculations

    Returns:
        List of zone candidates with dir, lower, upper, tag
    """
    candidates = []

    try:
        if i_ltf < 10:  # Need sufficient bars for analysis
            return candidates

        params = cfg.ict_params

        # Must have clear HTF bias from ICT context
        bias_dir = getattr(htf_ctx, "bias_dir", "neutral")
        if bias_dir not in ("long", "short"):
            return candidates

        # Check for continuation opportunities
        if bias_dir == "long":
            bullish_cont = _detect_bullish_continuation(htf_ctx, bars_ltf, i_ltf, params, tick)
            if bullish_cont:
                candidates.append(bullish_cont)

        elif bias_dir == "short":
            bearish_cont = _detect_bearish_continuation(htf_ctx, bars_ltf, i_ltf, params, tick)
            if bearish_cont:
                candidates.append(bearish_cont)

    except Exception as e:
        # Fail silently to avoid disrupting main strategy
        pass

    return candidates


def _detect_bullish_continuation(htf_ctx, bars: List[Any], i: int, params, tick: float) -> Optional[Dict[str, Any]]:
    """
    Detect bullish FVG continuation pattern

    Pattern:
    1. HTF bullish bias confirmed
    2. Recent bullish impulse ≥ minimum points
    3. Current retracement into continuation zone
    4. LTF FVG or order block for entry

    Args:
        htf_ctx: ICT context
        bars: LTF bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        Zone candidate dict or None
    """
    try:
        # 1. Check for recent bullish impulse
        impulse_bars = min(20, i)  # Look back up to 20 bars
        impulse_start_price = None
        impulse_end_price = None

        # Find recent significant move up
        for lookback in range(5, impulse_bars + 1):
            if i - lookback < 0:
                break

            start_price = bars[i - lookback].low
            current_price = bars[i].high
            impulse_size = current_price - start_price

            if impulse_size >= params.cont_min_impulse_pts:
                impulse_start_price = start_price
                impulse_end_price = current_price
                break

        if impulse_start_price is None:
            return None

        # 2. Check if we're in a retracement (price pulled back from highs)
        recent_high = max(bar.high for bar in bars[max(0, i-5):i+1])
        current_price = bars[i].close

        # Must be retracing but not too deep (stay above 78.6% of impulse)
        retracement_level = impulse_end_price - (impulse_size * 0.786)
        if current_price < retracement_level:
            return None

        # 3. Look for continuation zone (simulated HTF FVG via retracement analysis)
        # Use 62-79% retracement of recent impulse as continuation zone
        cont_upper = impulse_end_price - (impulse_size * 0.618)  # 61.8% retracement
        cont_lower = impulse_end_price - (impulse_size * 0.786)  # 78.6% retracement

        # 4. Check if current price is near continuation zone
        current_in_zone = cont_lower <= current_price <= cont_upper
        approaching_zone = abs(current_price - cont_upper) <= tick * 5  # Within 5 ticks

        if not (current_in_zone or approaching_zone):
            return None

        # 5. Look for LTF entry structure (simple FVG or order block simulation)
        entry_zone = _find_ltf_entry_structure(bars, i, "bullish", tick)
        if not entry_zone:
            # Use continuation zone as entry if no specific LTF structure
            entry_zone = {'lower': cont_lower, 'upper': cont_upper}

        return {
            'dir': 'long',
            'lower': entry_zone['lower'],
            'upper': entry_zone['upper'],
            'tag': 'ict_fvg_cont',
            'subtype': 'bullish_continuation',
            'impulse_size': impulse_size,
            'continuation_level': (cont_upper + cont_lower) / 2,
            'quality_boost': 0.12,
            'prefer_62_entry': True
        }

    except (AttributeError, IndexError, ZeroDivisionError):
        return None


def _detect_bearish_continuation(htf_ctx, bars: List[Any], i: int, params, tick: float) -> Optional[Dict[str, Any]]:
    """
    Detect bearish FVG continuation pattern

    Pattern:
    1. HTF bearish bias confirmed
    2. Recent bearish impulse ≥ minimum points
    3. Current retracement into continuation zone
    4. LTF FVG or order block for entry

    Args:
        htf_ctx: ICT context
        bars: LTF bar data
        i: Current index
        params: ICT parameters
        tick: Tick size

    Returns:
        Zone candidate dict or None
    """
    try:
        # 1. Check for recent bearish impulse
        impulse_bars = min(20, i)  # Look back up to 20 bars
        impulse_start_price = None
        impulse_end_price = None

        # Find recent significant move down
        for lookback in range(5, impulse_bars + 1):
            if i - lookback < 0:
                break

            start_price = bars[i - lookback].high
            current_price = bars[i].low
            impulse_size = start_price - current_price

            if impulse_size >= params.cont_min_impulse_pts:
                impulse_start_price = start_price
                impulse_end_price = current_price
                break

        if impulse_start_price is None:
            return None

        # 2. Check if we're in a retracement (price bounced up from lows)
        recent_low = min(bar.low for bar in bars[max(0, i-5):i+1])
        current_price = bars[i].close

        # Must be retracing but not too deep (stay below 78.6% of impulse)
        retracement_level = impulse_end_price + (impulse_size * 0.786)
        if current_price > retracement_level:
            return None

        # 3. Look for continuation zone (simulated HTF FVG via retracement analysis)
        # Use 62-79% retracement of recent impulse as continuation zone
        cont_lower = impulse_end_price + (impulse_size * 0.618)  # 61.8% retracement
        cont_upper = impulse_end_price + (impulse_size * 0.786)  # 78.6% retracement

        # 4. Check if current price is near continuation zone
        current_in_zone = cont_lower <= current_price <= cont_upper
        approaching_zone = abs(current_price - cont_lower) <= tick * 5  # Within 5 ticks

        if not (current_in_zone or approaching_zone):
            return None

        # 5. Look for LTF entry structure (simple FVG or order block simulation)
        entry_zone = _find_ltf_entry_structure(bars, i, "bearish", tick)
        if not entry_zone:
            # Use continuation zone as entry if no specific LTF structure
            entry_zone = {'lower': cont_lower, 'upper': cont_upper}

        return {
            'dir': 'short',
            'lower': entry_zone['lower'],
            'upper': entry_zone['upper'],
            'tag': 'ict_fvg_cont',
            'subtype': 'bearish_continuation',
            'impulse_size': impulse_size,
            'continuation_level': (cont_upper + cont_lower) / 2,
            'quality_boost': 0.12,
            'prefer_62_entry': True
        }

    except (AttributeError, IndexError, ZeroDivisionError):
        return None


def _find_ltf_entry_structure(bars: List[Any], i: int, direction: str, tick: float) -> Optional[Dict[str, Any]]:
    """
    Find LTF entry structure (FVG or order block) for continuation entry

    Args:
        bars: Bar data
        i: Current index
        direction: "bullish" or "bearish"
        tick: Tick size

    Returns:
        Dict with lower/upper bounds or None
    """
    try:
        # Look for recent FVG in the last few bars
        for lookback in range(0, min(3, i-1)):
            check_idx = i - lookback
            if check_idx < 2:
                continue

            if check_fvg_simple(bars, check_idx, direction):
                fvg_bounds = get_fvg_bounds(bars, check_idx, direction)
                if fvg_bounds:
                    # Validate gap size
                    gap_size = abs(fvg_bounds['upper'] - fvg_bounds['lower'])
                    if gap_size >= tick * 0.5:  # Minimum gap size
                        return fvg_bounds

        # If no FVG found, look for order block (simplified)
        if i >= 1:
            prev_bar = bars[i - 1]
            body_lower = min(prev_bar.open, prev_bar.close)
            body_upper = max(prev_bar.open, prev_bar.close)

            # Only use if body has meaningful size
            body_size = body_upper - body_lower
            if body_size >= tick * 1:  # At least 1 tick body
                return {
                    'lower': body_lower,
                    'upper': body_upper
                }

        return None

    except (AttributeError, IndexError):
        return None


def get_pattern_info() -> Dict[str, Any]:
    """
    Get information about this pattern module

    Returns:
        Dict with pattern metadata
    """
    return {
        'name': 'ICT FVG Continuation',
        'description': 'Trend continuation setups using HTF bias and FVG retracements',
        'timeframes': ['1m with HTF bias context'],
        'requirements': [
            'Clear HTF bias (long/short)',
            'Recent impulse ≥ minimum points threshold',
            'Price retracing into continuation zone (62-79%)',
            'LTF entry structure (FVG or order block)',
            'Not retraced beyond 78.6% of impulse'
        ],
        'entry_zones': 'LTF FVG or order block within continuation zone',
        'quality_factors': [
            'HTF bias strength and clarity',
            'Impulse size and momentum',
            'Retracement depth (ideal 62-78.6%)',
            'LTF entry structure quality',
            'Time proximity to continuation zone'
        ],
        'notes': [
            'Requires clear HTF bias from ICT context',
            'Uses retracement analysis as HTF FVG proxy',
            'Prefer 62% entry within identified zones',
            'Conservative approach - avoids deep retracements'
        ]
    }