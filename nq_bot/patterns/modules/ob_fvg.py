"""
Order-Block + Fair Value Gap Pattern Detection
Detects FVG patterns that form adjacent to order blocks with liquidity sweeps
"""

import time
from typing import Optional
import pandas as pd
import numpy as np


def _last_bearish_before(bars: pd.DataFrame, i: int) -> int:
    """Find the last bearish candle before index i"""
    j = i - 1
    while j >= 0:
        if bars.iloc[j]['close'] <= bars.iloc[j]['open']:  # bearish
            return j
        j -= 1
    return max(0, i - 1)


def _last_bullish_before(bars: pd.DataFrame, i: int) -> int:
    """Find the last bullish candle before index i"""
    j = i - 1
    while j >= 0:
        if bars.iloc[j]['close'] >= bars.iloc[j]['open']:  # bullish
            return j
        j -= 1
    return max(0, i - 1)


def _swept_prior_low(fvg_strategy, ob_i: int, bars: pd.DataFrame) -> bool:
    """Check if order block swept a prior swing low"""
    if ob_i < 2:
        return False

    ob_bar = bars.iloc[ob_i]
    ob_low = ob_bar['low']

    # Look for recent swing lows that were swept
    for swing in fvg_strategy.recent_swings:
        if (swing['type'] == 'low' and
            swing['bar_idx'] < ob_i and
            ob_i - swing['bar_idx'] <= 10):  # Within 10 bars

            swing_level = swing['level']
            # Check if OB wick swept below swing low by at least 1 tick
            if ob_low <= swing_level - fvg_strategy.TICK_SIZE:
                # Check if OB closed back above swing
                if ob_bar['close'] > swing_level:
                    return True
    return False


def _swept_prior_high(fvg_strategy, ob_i: int, bars: pd.DataFrame) -> bool:
    """Check if order block swept a prior swing high"""
    if ob_i < 2:
        return False

    ob_bar = bars.iloc[ob_i]
    ob_high = ob_bar['high']

    # Look for recent swing highs that were swept
    for swing in fvg_strategy.recent_swings:
        if (swing['type'] == 'high' and
            swing['bar_idx'] < ob_i and
            ob_i - swing['bar_idx'] <= 10):  # Within 10 bars

            swing_level = swing['level']
            # Check if OB wick swept above swing high by at least 1 tick
            if ob_high >= swing_level + fvg_strategy.TICK_SIZE:
                # Check if OB closed back below swing
                if ob_bar['close'] < swing_level:
                    return True
    return False


def scan_ob_fvg(fvg_strategy, bars: pd.DataFrame, i: int, now_dt, prof):
    """
    Scan for Order-Block + FVG patterns

    Args:
        fvg_strategy: FVGStrategy instance
        bars: Market data DataFrame
        i: Current bar index (displacement bar)
        now_dt: Current datetime
        prof: FVGProfile instance
    """

    # Need at least 3 bars for pattern detection
    if i < 2 or i >= len(bars) - 1:
        return

    # Check for bullish OB-FVG pattern
    _scan_bullish_ob_fvg(fvg_strategy, bars, i, now_dt, prof)

    # Check for bearish OB-FVG pattern
    _scan_bearish_ob_fvg(fvg_strategy, bars, i, now_dt, prof)


def _scan_bullish_ob_fvg(fvg_strategy, bars: pd.DataFrame, i: int, now_dt, prof):
    """Scan for bullish Order-Block + FVG pattern"""

    # 1) Check for bullish FVG gap first
    if i > 0 and i < len(bars) - 1:
        wick_gap = fvg_strategy._gap_ticks_bullish(bars.iloc[i-1], bars.iloc[i+1])
        body_gap = fvg_strategy._body_gap_ticks_bullish(bars.iloc[i-1], bars.iloc[i+1])
        gap_ticks = wick_gap if wick_gap > 0 else body_gap
    else:
        return

    if gap_ticks < fvg_strategy.config.get('min_gap_ticks', 1):
        return

    # 2) Find Order Block - last bearish candle before impulse
    ob_i = _last_bearish_before(bars, i)
    if ob_i < 1 or ob_i >= i:
        return

    ob_bar = bars.iloc[ob_i]
    ob_low, ob_high = ob_bar['low'], ob_bar['high']

    # 3) Sweep confirmation - OB must have swept prior swing low
    if not _swept_prior_low(fvg_strategy, ob_i, bars):
        return

    # 4) FVG imbalance near OB - require ≥25% overlap with OB height
    fvg_low = bars.iloc[i-1]['high']  # Bottom of bullish FVG
    fvg_high = bars.iloc[i+1]['low']  # Top of bullish FVG

    if fvg_high <= fvg_low:  # Invalid FVG
        return

    ob_height = ob_high - ob_low
    if ob_height <= 0:
        return

    # Calculate overlap between FVG and OB
    overlap_low = max(fvg_low, ob_low)
    overlap_high = min(fvg_high, ob_high)
    overlap = max(0.0, overlap_high - overlap_low)

    if overlap / ob_height < 0.25:  # Require 25% overlap
        return

    # 5) Quality gates using profile thresholds
    displacement_bar = bars.iloc[i]
    body_frac = fvg_strategy._body_fraction(displacement_bar) if hasattr(fvg_strategy, '_body_fraction') else abs(displacement_bar['close'] - displacement_bar['open']) / (displacement_bar['high'] - displacement_bar['low'])

    # Check body fraction
    min_body_frac = prof.displacement_body_frac_min_high_vol if fvg_strategy._is_high_vol(bars, i) else prof.displacement_body_frac_min_base
    if body_frac < min_body_frac:
        return

    # Check volume
    avg_volume = bars['volume'].rolling(20).mean().iloc[i]
    if pd.isna(avg_volume) or avg_volume <= 0:
        return

    vol_mult = displacement_bar['volume'] / avg_volume
    if vol_mult < prof.volume_min_mult_trend:
        return

    # Check displacement range
    bar_range = displacement_bar['high'] - displacement_bar['low']
    atr_value = fvg_strategy._calculate_atr(bars, 14).iloc[i] if len(bars) > 14 else 10.0
    min_range = max(prof.displacement_min_points_floor, prof.displacement_atr_multiple * atr_value)

    if bar_range < min_range:
        return

    # 6) Calculate quality score
    atr_mult = bar_range / atr_value if atr_value > 0 else 0
    quality = (body_frac * 0.3 +
              min(atr_mult / 2, 1.0) * 0.4 +
              min(vol_mult / 3, 1.0) * 0.3)

    if quality < prof.quality_score_min_trend:
        return

    # 7) Create FVG zone object and register
    fvg_id = f"OB_FVG_{fvg_strategy.next_id}"
    fvg_strategy.next_id += 1

    # Use FVG boundaries for the zone
    from nq_bot.patterns.fvg_strategy import FVGObject

    fvg = FVGObject(
        id=fvg_id,
        direction='long',
        created_at=time.time(),
        top=fvg_high,
        bottom=fvg_low,
        mid=(fvg_high + fvg_low) / 2,
        quality=quality,
        status='FRESH',
        origin_swing=None,  # OB pattern doesn't need specific swing
        body_frac=body_frac,
        range_pts=bar_range,
        vol_mult=vol_mult,
        atr_mult=atr_mult
    )

    fvg_strategy.fvg_registry[fvg_id] = fvg

    # Increment telemetry
    if hasattr(fvg_strategy, 'telemetry_counters'):
        fvg_strategy.telemetry_counters['ob_fvg_detected'] = fvg_strategy.telemetry_counters.get('ob_fvg_detected', 0) + 1

    fvg_strategy.logger.info(f"OB_FVG_DETECTED type=BULLISH dir=long top={fvg_high:.2f} bottom={fvg_low:.2f} "
                            f"gap_ticks={gap_ticks:.1f} quality={quality:.3f} ob_overlap={overlap/ob_height:.1%} "
                            f"body_frac={body_frac:.3f} vol_mult={vol_mult:.2f}")


def _scan_bearish_ob_fvg(fvg_strategy, bars: pd.DataFrame, i: int, now_dt, prof):
    """Scan for bearish Order-Block + FVG pattern"""

    # 1) Check for bearish FVG gap first
    if i > 0 and i < len(bars) - 1:
        wick_gap = fvg_strategy._gap_ticks_bearish(bars.iloc[i-1], bars.iloc[i+1])
        body_gap = fvg_strategy._body_gap_ticks_bearish(bars.iloc[i-1], bars.iloc[i+1])
        gap_ticks = wick_gap if wick_gap > 0 else body_gap
    else:
        return

    if gap_ticks < fvg_strategy.config.get('min_gap_ticks', 1):
        return

    # 2) Find Order Block - last bullish candle before impulse
    ob_i = _last_bullish_before(bars, i)
    if ob_i < 1 or ob_i >= i:
        return

    ob_bar = bars.iloc[ob_i]
    ob_low, ob_high = ob_bar['low'], ob_bar['high']

    # 3) Sweep confirmation - OB must have swept prior swing high
    if not _swept_prior_high(fvg_strategy, ob_i, bars):
        return

    # 4) FVG imbalance near OB - require ≥25% overlap with OB height
    fvg_high = bars.iloc[i-1]['low']   # Top of bearish FVG
    fvg_low = bars.iloc[i+1]['high']   # Bottom of bearish FVG

    if fvg_low >= fvg_high:  # Invalid FVG
        return

    ob_height = ob_high - ob_low
    if ob_height <= 0:
        return

    # Calculate overlap between FVG and OB
    overlap_low = max(fvg_low, ob_low)
    overlap_high = min(fvg_high, ob_high)
    overlap = max(0.0, overlap_high - overlap_low)

    if overlap / ob_height < 0.25:  # Require 25% overlap
        return

    # 5) Quality gates using profile thresholds
    displacement_bar = bars.iloc[i]
    body_frac = fvg_strategy._body_fraction(displacement_bar) if hasattr(fvg_strategy, '_body_fraction') else abs(displacement_bar['close'] - displacement_bar['open']) / (displacement_bar['high'] - displacement_bar['low'])

    # Check body fraction
    min_body_frac = prof.displacement_body_frac_min_high_vol if fvg_strategy._is_high_vol(bars, i) else prof.displacement_body_frac_min_base
    if body_frac < min_body_frac:
        return

    # Check volume
    avg_volume = bars['volume'].rolling(20).mean().iloc[i]
    if pd.isna(avg_volume) or avg_volume <= 0:
        return

    vol_mult = displacement_bar['volume'] / avg_volume
    if vol_mult < prof.volume_min_mult_trend:
        return

    # Check displacement range
    bar_range = displacement_bar['high'] - displacement_bar['low']
    atr_value = fvg_strategy._calculate_atr(bars, 14).iloc[i] if len(bars) > 14 else 10.0
    min_range = max(prof.displacement_min_points_floor, prof.displacement_atr_multiple * atr_value)

    if bar_range < min_range:
        return

    # 6) Calculate quality score
    atr_mult = bar_range / atr_value if atr_value > 0 else 0
    quality = (body_frac * 0.3 +
              min(atr_mult / 2, 1.0) * 0.4 +
              min(vol_mult / 3, 1.0) * 0.3)

    if quality < prof.quality_score_min_trend:
        return

    # 7) Create FVG zone object and register
    fvg_id = f"OB_FVG_{fvg_strategy.next_id}"
    fvg_strategy.next_id += 1

    # Use FVG boundaries for the zone
    from nq_bot.patterns.fvg_strategy import FVGObject

    fvg = FVGObject(
        id=fvg_id,
        direction='short',
        created_at=time.time(),
        top=fvg_high,
        bottom=fvg_low,
        mid=(fvg_high + fvg_low) / 2,
        quality=quality,
        status='FRESH',
        origin_swing=None,  # OB pattern doesn't need specific swing
        body_frac=body_frac,
        range_pts=bar_range,
        vol_mult=vol_mult,
        atr_mult=atr_mult
    )

    fvg_strategy.fvg_registry[fvg_id] = fvg

    # Increment telemetry
    if hasattr(fvg_strategy, 'telemetry_counters'):
        fvg_strategy.telemetry_counters['ob_fvg_detected'] = fvg_strategy.telemetry_counters.get('ob_fvg_detected', 0) + 1

    fvg_strategy.logger.info(f"OB_FVG_DETECTED type=BEARISH dir=short top={fvg_high:.2f} bottom={fvg_low:.2f} "
                            f"gap_ticks={gap_ticks:.1f} quality={quality:.3f} ob_overlap={overlap/ob_height:.1%} "
                            f"body_frac={body_frac:.3f} vol_mult={vol_mult:.2f}")