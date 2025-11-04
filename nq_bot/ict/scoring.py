"""
ICT Confluence Scoring

Provides confluence scoring for FVG zones based on ICT concepts:
- Bias alignment
- Premium/discount location and OTE overlap
- Recent raid activity
- Session/killzone timing
- Optional SMT support
"""

from typing import Dict, Tuple, Any, Optional
import logging


def confluence_score(
    ict,
    zone,
    session_name: str,
    weights: Tuple[float, float, float, float, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate confluence score for a FVG zone based on ICT analysis

    Args:
        ict: ICTContext instance
        zone: FVG zone object with directional and level information
        session_name: Current session name
        weights: (bias, location, raid, session, smt) weight tuple

    Returns:
        Tuple of (total_score, component_scores_dict)
    """
    try:
        w_bias, w_loc, w_raid, w_sess, w_smt = weights

        # 1. Bias alignment score
        s_bias = _score_bias_alignment(ict, zone)

        # 2. Location preference score (premium/discount + OTE)
        s_loc = _score_location_preference(ict, zone)

        # 3. Raid recency score
        s_raid = _score_raid_recency(ict)

        # 4. Session/killzone score
        s_session = _score_session_timing(ict)

        # 5. SMT support score
        s_smt = _score_smt_support(ict)

        # Calculate weighted total
        score = (w_bias * s_bias +
                w_loc * s_loc +
                w_raid * s_raid +
                w_sess * s_session +
                w_smt * s_smt)

        component_scores = {
            "bias": s_bias,
            "loc": s_loc,
            "raid": s_raid,
            "sess": s_session,
            "smt": s_smt
        }

        return round(score, 3), component_scores

    except Exception as e:
        # Log error and return neutral score
        logger = logging.getLogger(__name__)
        logger.error(f"Error calculating confluence score: {e}")
        return 0.5, {"bias": 0.5, "loc": 0.5, "raid": 0.0, "sess": 0.5, "smt": 0.5}


def _score_bias_alignment(ict, zone) -> float:
    """
    Score bias alignment between ICT bias and zone direction

    Args:
        ict: ICTContext instance
        zone: FVG zone object

    Returns:
        Score: 1.0 (aligned), 0.5 (neutral bias), 0.0 (opposing)
    """
    try:
        # Determine zone direction
        zone_is_bullish = _is_zone_bullish(zone)

        if zone_is_bullish is None:
            return 0.5  # Unknown direction

        # Check bias alignment
        bias_dir = ict.bias_dir.lower()

        if bias_dir == "neutral":
            return 0.5
        elif bias_dir == "long" and zone_is_bullish:
            return 1.0  # Bullish bias + bullish zone
        elif bias_dir == "short" and not zone_is_bullish:
            return 1.0  # Bearish bias + bearish zone
        else:
            return 0.0  # Opposing bias

    except Exception:
        return 0.5


def _score_location_preference(ict, zone) -> float:
    """
    Score location preference (premium/discount + OTE) with enhanced logic

    Args:
        ict: ICTContext instance
        zone: FVG zone object

    Returns:
        Score: 1.0 (ideal), 0.8 (strong), 0.6 (partial), 0.3 (weak), 0.0 (poor location)
    """
    try:
        zone_is_bullish = _is_zone_bullish(zone)

        if zone_is_bullish is None:
            return 0.0

        # Check premium/discount preference with enhanced context
        # Bullish zones preferred in discount, bearish zones in premium
        location_context = getattr(ict, 'location_context_str', 'equilibrium')
        price_position_pct = getattr(ict, 'price_position_percentage', 50.0)

        # Enhanced location scoring based on position percentage
        location_score = 0.0
        if zone_is_bullish:
            # Bullish zones: prefer discount area (lower percentage)
            if location_context == 'discount':
                location_score = 1.0
            elif price_position_pct < 50.0:  # Below equilibrium
                location_score = 0.6
            elif location_context == 'equilibrium':
                location_score = 0.3
            else:  # Premium area
                location_score = 0.0
        else:
            # Bearish zones: prefer premium area (higher percentage)
            if location_context == 'premium':
                location_score = 1.0
            elif price_position_pct > 50.0:  # Above equilibrium
                location_score = 0.6
            elif location_context == 'equilibrium':
                location_score = 0.3
            else:  # Discount area
                location_score = 0.0

        # Check OTE overlap with enhanced tolerance
        try:
            ote_overlap = ict.ote_overlap(zone)
            ote_score = 1.0 if ote_overlap else 0.0
        except Exception:
            ote_score = 0.0

        # Combined scoring with weighted components
        if location_score >= 0.6 and ote_score > 0:
            return min(1.0, location_score + 0.2)  # Bonus for OTE + good location
        elif location_score >= 0.6:
            return location_score
        elif ote_score > 0:
            return max(0.6, location_score + 0.3)  # OTE can partially compensate
        else:
            return location_score

    except Exception:
        return 0.0


def _score_raid_recency(ict) -> float:
    """
    Score recent raid activity

    Args:
        ict: ICTContext instance

    Returns:
        Score: 1.0 (recent raid), 0.0 (no recent raid)
    """
    try:
        return 1.0 if ict.raid_recent else 0.0
    except Exception:
        return 0.0


def _score_session_timing(ict) -> float:
    """
    Score session/killzone timing with enhanced session awareness

    Args:
        ict: ICTContext instance

    Returns:
        Score: 1.0 (premium killzone), 0.8 (standard killzone), 0.6 (active session), 0.3 (off-hours)
    """
    try:
        # Get enhanced session information
        in_killzone = getattr(ict, 'session_killzone', False)
        session_name = getattr(ict, 'session_name', 'OTHER')

        if in_killzone:
            # Score different killzones
            if session_name == 'LONDON':
                return 1.0  # London killzone is premium for FVG patterns
            elif session_name in ['NY_AM', 'NY_PM']:
                return 0.8  # NY killzones are strong
            else:
                return 0.8  # Other killzones (should not happen but safe fallback)
        else:
            # Score non-killzone sessions
            if session_name in ['LONDON', 'NY_AM', 'NY_PM']:
                return 0.6  # Active session but not in killzone
            elif session_name == 'ASIAN':
                return 0.3  # Asian session (lower volatility)
            else:
                return 0.3  # Other off-hours

    except Exception:
        return 0.5


def _score_smt_support(ict) -> float:
    """
    Score SMT (Smart Money Technique) support

    Args:
        ict: ICTContext instance

    Returns:
        Score: 1.0 (supporting), 0.5 (neutral/disabled), 0.0 (opposing)
    """
    try:
        if ict.smt_support is True:
            return 1.0
        elif ict.smt_support is None:
            return 0.5  # SMT disabled or neutral
        else:
            return 0.0  # SMT opposing
    except Exception:
        return 0.5


def _is_zone_bullish(zone) -> Optional[bool]:
    """
    Determine if a zone is bullish or bearish

    Args:
        zone: FVG zone object

    Returns:
        True (bullish), False (bearish), None (unknown)
    """
    try:
        # Check various attributes that might indicate direction
        if hasattr(zone, 'is_bullish'):
            return zone.is_bullish
        elif hasattr(zone, 'direction'):
            direction = zone.direction.lower()
            if direction in ['long', 'bullish', 'up']:
                return True
            elif direction in ['short', 'bearish', 'down']:
                return False
        elif hasattr(zone, 'type'):
            zone_type = zone.type.lower()
            if 'bullish' in zone_type or 'long' in zone_type:
                return True
            elif 'bearish' in zone_type or 'short' in zone_type:
                return False

        # If no clear direction attributes, return None
        return None

    except Exception:
        return None


def format_confluence_log(score: float, parts: Dict[str, float], zone_id: str = "") -> str:
    """
    Format confluence score for logging

    Args:
        score: Total confluence score
        parts: Component score breakdown
        zone_id: Optional zone identifier

    Returns:
        Formatted log string
    """
    try:
        zone_prefix = f"zone={zone_id} " if zone_id else ""
        components = " ".join([f"{k}={v:.2f}" for k, v in parts.items()])
        return f"{zone_prefix}score={score:.3f} {components}"
    except Exception:
        return f"score={score:.3f}"