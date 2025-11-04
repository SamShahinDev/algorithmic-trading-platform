"""
Breaker + Fair Value Gap Pattern Detection
Detects FVG patterns that form after Order Block invalidation (breaker) on retest
"""

import time
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class OrderBlockTracker:
    """Tracks order blocks and their invalidation (breaker formation)"""

    def __init__(self):
        self.order_blocks = []  # Active order blocks
        self.recent_breakers = []  # Recently formed breakers

    def update_order_blocks(self, bars: pd.DataFrame, current_idx: int):
        """Update order block tracking"""
        if current_idx < 10:
            return

        # Find new order blocks - significant bearish/bullish candles
        lookback = min(20, current_idx - 5)
        start_idx = max(5, current_idx - lookback)

        for i in range(start_idx, current_idx - 2):
            bar = bars.iloc[i]
            body_size = abs(bar['close'] - bar['open'])
            range_size = bar['high'] - bar['low']

            if range_size <= 0:
                continue

            body_frac = body_size / range_size

            # Strong bearish OB (potential resistance)
            if (bar['close'] < bar['open'] and
                body_frac > 0.7 and
                range_size > 2.0):

                ob = {
                    'id': f"OB_{i}_{bar['high']:.2f}",
                    'type': 'bearish',
                    'high': bar['high'],
                    'low': bar['low'],
                    'idx': i,
                    'invalidated': False,
                    'breaker_direction': None
                }

                # Check if already exists
                if not any(existing['id'] == ob['id'] for existing in self.order_blocks):
                    self.order_blocks.append(ob)

            # Strong bullish OB (potential support)
            elif (bar['close'] > bar['open'] and
                  body_frac > 0.7 and
                  range_size > 2.0):

                ob = {
                    'id': f"OB_{i}_{bar['low']:.2f}",
                    'type': 'bullish',
                    'high': bar['high'],
                    'low': bar['low'],
                    'idx': i,
                    'invalidated': False,
                    'breaker_direction': None
                }

                # Check if already exists
                if not any(existing['id'] == ob['id'] for existing in self.order_blocks):
                    self.order_blocks.append(ob)

        # Check for OB invalidations (breaker formation)
        self._check_invalidations(bars, current_idx)

        # Clean up old order blocks
        self.order_blocks = [ob for ob in self.order_blocks if current_idx - ob['idx'] < 100]
        self.recent_breakers = [br for br in self.recent_breakers if current_idx - br['break_idx'] < 20]

    def _check_invalidations(self, bars: pd.DataFrame, current_idx: int):
        """Check for order block invalidations"""
        for ob in self.order_blocks:
            if ob['invalidated']:
                continue

            # Check recent bars for invalidation
            check_start = max(ob['idx'] + 1, current_idx - 10)
            for i in range(check_start, current_idx + 1):
                if i >= len(bars):
                    continue

                bar = bars.iloc[i]

                if ob['type'] == 'bearish':
                    # Bearish OB invalidated by close above its high
                    if bar['close'] > ob['high']:
                        ob['invalidated'] = True
                        ob['breaker_direction'] = 'long'  # Now expect bullish continuation

                        breaker = {
                            'ob_id': ob['id'],
                            'direction': 'long',
                            'break_idx': i,
                            'break_level': ob['high'],
                            'original_ob': ob
                        }
                        self.recent_breakers.append(breaker)
                        break

                elif ob['type'] == 'bullish':
                    # Bullish OB invalidated by close below its low
                    if bar['close'] < ob['low']:
                        ob['invalidated'] = True
                        ob['breaker_direction'] = 'short'  # Now expect bearish continuation

                        breaker = {
                            'ob_id': ob['id'],
                            'direction': 'short',
                            'break_idx': i,
                            'break_level': ob['low'],
                            'original_ob': ob
                        }
                        self.recent_breakers.append(breaker)
                        break

    def find_new_breaker(self, current_idx: int) -> Optional[Dict]:
        """Find recently formed breaker for the current bar"""
        for breaker in self.recent_breakers:
            # Look for breakers formed within last 5 bars
            if current_idx - breaker['break_idx'] <= 5:
                return breaker
        return None


# Global order block tracker instance
_ob_tracker = OrderBlockTracker()


def _is_retest_of_breaker(breaker: Dict, bars: pd.DataFrame, current_idx: int) -> bool:
    """Check if current price action is retesting the breaker level"""
    if current_idx < breaker['break_idx'] + 2:
        return False

    break_level = breaker['break_level']
    current_bar = bars.iloc[current_idx]

    # Allow some tolerance for retest (within 1-2 ticks)
    tolerance = 0.50  # 2 ticks

    if breaker['direction'] == 'long':
        # Bullish breaker - look for retest from above
        return (current_bar['low'] <= break_level + tolerance and
                current_bar['high'] >= break_level - tolerance)
    else:
        # Bearish breaker - look for retest from below
        return (current_bar['high'] >= break_level - tolerance and
                current_bar['low'] <= break_level + tolerance)


def scan_breaker_fvg(fvg_strategy, bars: pd.DataFrame, i: int, now_dt, prof):
    """
    Scan for Breaker + FVG patterns

    Args:
        fvg_strategy: FVGStrategy instance
        bars: Market data DataFrame
        i: Current bar index (displacement bar)
        now_dt: Current datetime
        prof: FVGProfile instance
    """

    # Need sufficient bars for pattern detection
    if i < 10 or i >= len(bars) - 1:
        return

    # Update order block tracker
    _ob_tracker.update_order_blocks(bars, i)

    # 1) Look for recently formed breaker
    breaker = _ob_tracker.find_new_breaker(i)
    if not breaker:
        return

    # 2) Check if we're retesting the breaker level
    if not _is_retest_of_breaker(breaker, bars, i):
        return

    direction = breaker['direction']

    # 3) Look for FVG in the breaker direction
    if direction == 'long':
        if i > 0 and i < len(bars) - 1:
            wick_gap = fvg_strategy._gap_ticks_bullish(bars.iloc[i-1], bars.iloc[i+1])
            body_gap = fvg_strategy._body_gap_ticks_bullish(bars.iloc[i-1], bars.iloc[i+1])
            gap_ticks = wick_gap if wick_gap > 0 else body_gap
        else:
            return
    else:
        if i > 0 and i < len(bars) - 1:
            wick_gap = fvg_strategy._gap_ticks_bearish(bars.iloc[i-1], bars.iloc[i+1])
            body_gap = fvg_strategy._body_gap_ticks_bearish(bars.iloc[i-1], bars.iloc[i+1])
            gap_ticks = wick_gap if wick_gap > 0 else body_gap
        else:
            return

    if gap_ticks < fvg_strategy.config.get('min_gap_ticks', 1):
        return

    # 4) Quality gates using profile thresholds
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

    # 5) Calculate quality score with breaker bonus
    atr_mult = bar_range / atr_value if atr_value > 0 else 0
    base_quality = (body_frac * 0.3 +
                   min(atr_mult / 2, 1.0) * 0.4 +
                   min(vol_mult / 3, 1.0) * 0.3)

    # Add small bonus for breaker retest context
    quality = min(base_quality + 0.05, 1.0)

    if quality < prof.quality_score_min_trend:
        return

    # 6) Create FVG zone object and register
    fvg_id = f"BREAKER_FVG_{fvg_strategy.next_id}"
    fvg_strategy.next_id += 1

    # Calculate FVG boundaries
    if direction == 'long':
        fvg_bottom = bars.iloc[i-1]['high']
        fvg_top = bars.iloc[i+1]['low']
    else:
        fvg_top = bars.iloc[i-1]['low']
        fvg_bottom = bars.iloc[i+1]['high']

    # Ensure valid zone
    if direction == 'long' and fvg_top <= fvg_bottom:
        return
    if direction == 'short' and fvg_bottom >= fvg_top:
        return

    # Use consistent top/bottom ordering
    zone_top = max(fvg_top, fvg_bottom)
    zone_bottom = min(fvg_top, fvg_bottom)

    from nq_bot.patterns.fvg_strategy import FVGObject

    fvg = FVGObject(
        id=fvg_id,
        direction=direction,
        created_at=time.time(),
        top=zone_top,
        bottom=zone_bottom,
        mid=(zone_top + zone_bottom) / 2,
        quality=quality,
        status='FRESH',
        origin_swing=None,  # Breaker pattern has its own context
        body_frac=body_frac,
        range_pts=bar_range,
        vol_mult=vol_mult,
        atr_mult=atr_mult
    )

    fvg_strategy.fvg_registry[fvg_id] = fvg

    # Increment telemetry
    if hasattr(fvg_strategy, 'telemetry_counters'):
        fvg_strategy.telemetry_counters['breaker_fvg_detected'] = fvg_strategy.telemetry_counters.get('breaker_fvg_detected', 0) + 1

    fvg_strategy.logger.info(f"BREAKER_FVG_DETECTED dir={direction} top={zone_top:.2f} bottom={zone_bottom:.2f} "
                            f"gap_ticks={gap_ticks:.1f} quality={quality:.3f} breaker_level={breaker['break_level']:.2f} "
                            f"ob_type={breaker['original_ob']['type']} body_frac={body_frac:.3f} vol_mult={vol_mult:.2f}")