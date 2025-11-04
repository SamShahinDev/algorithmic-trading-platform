"""
Internal Range Liquidity → External Range Liquidity + FVG Pattern Detection
Detects FVG patterns that form after IRL raids with direction toward ERL
"""

import time
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class SwingTracker:
    """Simple swing tracker for IRL/ERL detection"""

    def __init__(self):
        self.internal_swings = []  # Recent internal liquidity levels
        self.external_swings = []  # External range targets

    def update_swings(self, bars: pd.DataFrame, current_idx: int):
        """Update swing levels from recent price action"""
        if len(bars) < 20:
            return

        # Simple swing detection - look for local highs/lows in recent 20 bars
        lookback = min(20, current_idx)
        start_idx = max(0, current_idx - lookback)

        # Find swing highs and lows
        for i in range(start_idx + 2, current_idx - 2):
            if i < 2 or i >= len(bars) - 2:
                continue

            # Swing high
            if (bars.iloc[i]['high'] > bars.iloc[i-1]['high'] and
                bars.iloc[i]['high'] > bars.iloc[i-2]['high'] and
                bars.iloc[i]['high'] > bars.iloc[i+1]['high'] and
                bars.iloc[i]['high'] > bars.iloc[i+2]['high']):

                swing = {
                    'level': bars.iloc[i]['high'],
                    'type': 'high',
                    'idx': i,
                    'time': bars.index[i] if hasattr(bars.index[i], 'timestamp') else i
                }

                # Classify as internal (recent) or external (significant)
                bar_range = bars.iloc[i]['high'] - bars.iloc[i]['low']
                if bar_range > 3.0:  # Significant level
                    self.external_swings.append(swing)
                else:
                    self.internal_swings.append(swing)

            # Swing low
            if (bars.iloc[i]['low'] < bars.iloc[i-1]['low'] and
                bars.iloc[i]['low'] < bars.iloc[i-2]['low'] and
                bars.iloc[i]['low'] < bars.iloc[i+1]['low'] and
                bars.iloc[i]['low'] < bars.iloc[i+2]['low']):

                swing = {
                    'level': bars.iloc[i]['low'],
                    'type': 'low',
                    'idx': i,
                    'time': bars.index[i] if hasattr(bars.index[i], 'timestamp') else i
                }

                # Classify as internal (recent) or external (significant)
                bar_range = bars.iloc[i]['high'] - bars.iloc[i]['low']
                if bar_range > 3.0:  # Significant level
                    self.external_swings.append(swing)
                else:
                    self.internal_swings.append(swing)

        # Keep only recent swings
        self.internal_swings = self.internal_swings[-10:]  # Last 10 internal levels
        self.external_swings = self.external_swings[-5:]   # Last 5 external levels

    def nearest_internal(self, current_price: float) -> Optional[Dict]:
        """Find nearest internal liquidity level"""
        if not self.internal_swings:
            return None

        nearest = None
        min_distance = float('inf')

        for swing in self.internal_swings:
            distance = abs(swing['level'] - current_price)
            if distance < min_distance:
                min_distance = distance
                nearest = swing

        return nearest

    def nearest_external(self, current_price: float) -> Optional[Dict]:
        """Find nearest external range target"""
        if not self.external_swings:
            return None

        nearest = None
        min_distance = float('inf')

        for swing in self.external_swings:
            distance = abs(swing['level'] - current_price)
            if distance < min_distance:
                min_distance = distance
                nearest = swing

        return nearest


# Global swing tracker instance
_swing_tracker = SwingTracker()


def _recent_raid_of(fvg_strategy, irl_level: Dict, bars: pd.DataFrame, current_idx: int, lookback: int = 20) -> bool:
    """Check if there was a recent raid of the IRL level"""
    if current_idx < lookback:
        return False

    irl_price = irl_level['level']
    irl_type = irl_level['type']

    # Look for raid in recent bars
    start_idx = max(0, current_idx - lookback)

    for i in range(start_idx, current_idx):
        bar = bars.iloc[i]

        if irl_type == 'high':
            # Look for wick above IRL high that closed back below
            if (bar['high'] >= irl_price + fvg_strategy.TICK_SIZE and
                bar['close'] < irl_price):
                return True
        else:  # irl_type == 'low'
            # Look for wick below IRL low that closed back above
            if (bar['low'] <= irl_price - fvg_strategy.TICK_SIZE and
                bar['close'] > irl_price):
                return True

    return False


def _mini_structure_shift(fvg_strategy, bars: pd.DataFrame, current_idx: int, direction: str) -> bool:
    """Check for mini structure shift (CHOCH/CISD)"""
    if current_idx < 5:
        return False

    # Simple structure shift detection - look for change in recent highs/lows
    recent_bars = bars.iloc[max(0, current_idx-5):current_idx+1]

    if len(recent_bars) < 3:
        return False

    if direction == 'long':
        # Look for higher low formation
        lows = recent_bars['low'].values
        return len(lows) >= 2 and lows[-1] > lows[-2]
    else:
        # Look for lower high formation
        highs = recent_bars['high'].values
        return len(highs) >= 2 and highs[-1] < highs[-2]


def scan_irl_erl_fvg(fvg_strategy, bars: pd.DataFrame, i: int, now_dt, prof):
    """
    Scan for IRL→ERL + FVG patterns

    Args:
        fvg_strategy: FVGStrategy instance
        bars: Market data DataFrame
        i: Current bar index (displacement bar)
        now_dt: Current datetime
        prof: FVGProfile instance
    """

    # Need sufficient bars for pattern detection
    if i < 5 or i >= len(bars) - 1:
        return

    # Update swing tracker
    _swing_tracker.update_swings(bars, i)

    current_price = bars.iloc[i]['close']

    # 1) Identify current internal swing (IRL) and external target (ERL)
    irl = _swing_tracker.nearest_internal(current_price)
    erl = _swing_tracker.nearest_external(current_price)

    if not irl or not erl:
        return

    # 2) Check if there was a recent raid of IRL
    if not _recent_raid_of(fvg_strategy, irl, bars, i, lookback=20):
        return

    # 3) Determine direction toward ERL
    direction = 'long' if erl['level'] > current_price else 'short'

    # 4) Check for fresh FVG in the direction toward ERL
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

    # 5) Optional micro structure shift (CHOCH/CISD)
    if not _mini_structure_shift(fvg_strategy, bars, i, direction):
        return

    # 6) Quality gates using profile thresholds
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

    # 7) Calculate quality score
    atr_mult = bar_range / atr_value if atr_value > 0 else 0
    quality = (body_frac * 0.3 +
              min(atr_mult / 2, 1.0) * 0.4 +
              min(vol_mult / 3, 1.0) * 0.3)

    if quality < prof.quality_score_min_trend:
        return

    # 8) Create FVG zone object and register
    fvg_id = f"IRL_ERL_FVG_{fvg_strategy.next_id}"
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
        origin_swing=None,  # IRL-ERL pattern has its own context
        body_frac=body_frac,
        range_pts=bar_range,
        vol_mult=vol_mult,
        atr_mult=atr_mult
    )

    fvg_strategy.fvg_registry[fvg_id] = fvg

    # Increment telemetry
    if hasattr(fvg_strategy, 'telemetry_counters'):
        fvg_strategy.telemetry_counters['irl_erl_fvg_detected'] = fvg_strategy.telemetry_counters.get('irl_erl_fvg_detected', 0) + 1

    fvg_strategy.logger.info(f"IRL_ERL_FVG_DETECTED dir={direction} top={zone_top:.2f} bottom={zone_bottom:.2f} "
                            f"gap_ticks={gap_ticks:.1f} quality={quality:.3f} irl_level={irl['level']:.2f} "
                            f"erl_target={erl['level']:.2f} body_frac={body_frac:.3f} vol_mult={vol_mult:.2f}")