"""
Candlestick Guard Utility for NQ Trading Bot
Detects key reversal candles and provides veto/boost decisions
"""

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class CandleDetect:
    match: bool
    strength: float        # 0..1
    body_frac: float
    wick_up_frac: float
    wick_dn_frac: float
    is_dangerous: bool     # range/volume/body extreme flags
    kind: str              # 'hammer'|'shooting_star'|'bull_engulf'|...

class CandlestickGuard:
    """
    Detects reversal candlestick patterns and provides trading decisions
    """
    
    def __init__(self):
        # Store recent detections for scope_bars window
        self.recent_detections = {}  # key: (symbol, timestamp) -> CandleDetect
        self.max_cache_age = timedelta(minutes=5)
    
    def detect_hammer(self, o: float, h: float, l: float, c: float, 
                     atr: float, vol: float, avg_vol: float, tick_size: float) -> CandleDetect:
        """
        Detect bullish hammer pattern (veto shorts / boost longs)
        - lower_wick ≥ 2.0 × body
        - upper_wick ≤ 0.3 × body
        - close in top 30% of range
        - body ≤ 0.6 × ATR(14)
        """
        body = abs(c - o)
        range_hl = h - l
        
        if range_hl < tick_size:
            return CandleDetect(False, 0, 0, 0, 0, False, 'hammer')
        
        # Calculate wicks
        if c >= o:  # Bullish body
            upper_wick = h - c
            lower_wick = o - l
        else:  # Bearish body  
            upper_wick = h - o
            lower_wick = c - l
        
        body_frac = body / range_hl if range_hl > 0 else 0
        wick_up_frac = upper_wick / range_hl if range_hl > 0 else 0
        wick_dn_frac = lower_wick / range_hl if range_hl > 0 else 0
        
        # Hammer conditions
        lower_wick_ok = lower_wick >= 2.0 * body if body > 0 else False
        upper_wick_ok = upper_wick <= 0.3 * body if body > 0 else True
        close_position = (c - l) / range_hl if range_hl > 0 else 0
        close_in_top = close_position >= 0.7
        body_size_ok = body <= 0.6 * atr
        
        match = lower_wick_ok and upper_wick_ok and close_in_top and body_size_ok
        
        # Calculate strength (0-1)
        strength = 0.0
        if match:
            strength = min(1.0, (lower_wick / body - 2.0) / 2.0) * 0.5  # Lower wick ratio
            strength += (1.0 - upper_wick / (0.3 * body)) * 0.3 if body > 0 else 0.3  # Upper wick smallness
            strength += close_position * 0.2  # Close position
        
        # Check if dangerous
        is_dangerous = (range_hl >= 1.25 * atr or vol >= 1.5 * avg_vol) if match else False
        
        return CandleDetect(match, strength, body_frac, wick_up_frac, wick_dn_frac, is_dangerous, 'hammer')
    
    def detect_shooting_star(self, o: float, h: float, l: float, c: float,
                            atr: float, vol: float, avg_vol: float, tick_size: float) -> CandleDetect:
        """
        Detect bearish shooting star (veto longs / boost shorts)
        Mirror of hammer
        """
        body = abs(c - o)
        range_hl = h - l
        
        if range_hl < tick_size:
            return CandleDetect(False, 0, 0, 0, 0, False, 'shooting_star')
        
        # Calculate wicks
        if c >= o:  # Bullish body
            upper_wick = h - c
            lower_wick = o - l
        else:  # Bearish body
            upper_wick = h - o
            lower_wick = c - l
        
        body_frac = body / range_hl if range_hl > 0 else 0
        wick_up_frac = upper_wick / range_hl if range_hl > 0 else 0
        wick_dn_frac = lower_wick / range_hl if range_hl > 0 else 0
        
        # Shooting star conditions (mirror of hammer)
        upper_wick_ok = upper_wick >= 2.0 * body if body > 0 else False
        lower_wick_ok = lower_wick <= 0.3 * body if body > 0 else True
        close_position = (c - l) / range_hl if range_hl > 0 else 0
        close_in_bottom = close_position <= 0.3
        body_size_ok = body <= 0.6 * atr
        
        match = upper_wick_ok and lower_wick_ok and close_in_bottom and body_size_ok
        
        # Calculate strength
        strength = 0.0
        if match:
            strength = min(1.0, (upper_wick / body - 2.0) / 2.0) * 0.5  # Upper wick ratio
            strength += (1.0 - lower_wick / (0.3 * body)) * 0.3 if body > 0 else 0.3  # Lower wick smallness
            strength += (1.0 - close_position) * 0.2  # Close position (inverted)
        
        # Check if dangerous
        is_dangerous = (range_hl >= 1.25 * atr or vol >= 1.5 * avg_vol) if match else False
        
        return CandleDetect(match, strength, body_frac, wick_up_frac, wick_dn_frac, is_dangerous, 'shooting_star')
    
    def detect_bullish_engulfing(self, o: float, h: float, l: float, c: float,
                                 prev_o: float, prev_h: float, prev_l: float, prev_c: float,
                                 atr: float, vol: float, avg_vol: float, tick_size: float) -> CandleDetect:
        """
        Detect bullish engulfing pattern
        - body0 ≥ 1.1 × body1 (true body engulf)
        - Current candle bullish, previous bearish
        """
        curr_body = abs(c - o)
        prev_body = abs(prev_c - prev_o)
        range_hl = h - l
        
        if range_hl < tick_size or prev_body < tick_size:
            return CandleDetect(False, 0, 0, 0, 0, False, 'bull_engulf')
        
        # Check if current is bullish and previous is bearish
        is_bullish = c > o
        prev_is_bearish = prev_c < prev_o
        
        # Body engulfing condition
        body_engulf = curr_body >= 1.1 * prev_body
        
        # True engulfing: current body covers previous body
        true_engulf = is_bullish and prev_is_bearish and body_engulf and o <= prev_c and c >= prev_o
        
        body_frac = curr_body / range_hl if range_hl > 0 else 0
        wick_up_frac = (h - c) / range_hl if range_hl > 0 else 0
        wick_dn_frac = (o - l) / range_hl if range_hl > 0 else 0
        
        match = true_engulf
        
        # Calculate strength
        strength = 0.0
        if match:
            strength = min(1.0, (curr_body / prev_body - 1.1) / 0.5) * 0.6  # Engulfing ratio
            strength += body_frac * 0.4  # Body dominance
        
        # Check if dangerous
        is_dangerous = (body_frac >= 0.6 and range_hl >= 1.25 * atr and vol >= 1.5 * avg_vol) if match else False
        
        return CandleDetect(match, strength, body_frac, wick_up_frac, wick_dn_frac, is_dangerous, 'bull_engulf')
    
    def detect_bearish_engulfing(self, o: float, h: float, l: float, c: float,
                                 prev_o: float, prev_h: float, prev_l: float, prev_c: float,
                                 atr: float, vol: float, avg_vol: float, tick_size: float) -> CandleDetect:
        """
        Detect bearish engulfing pattern
        Mirror of bullish engulfing
        """
        curr_body = abs(c - o)
        prev_body = abs(prev_c - prev_o)
        range_hl = h - l
        
        if range_hl < tick_size or prev_body < tick_size:
            return CandleDetect(False, 0, 0, 0, 0, False, 'bear_engulf')
        
        # Check if current is bearish and previous is bullish
        is_bearish = c < o
        prev_is_bullish = prev_c > prev_o
        
        # Body engulfing condition
        body_engulf = curr_body >= 1.1 * prev_body
        
        # True engulfing: current body covers previous body
        true_engulf = is_bearish and prev_is_bullish and body_engulf and o >= prev_c and c <= prev_o
        
        body_frac = curr_body / range_hl if range_hl > 0 else 0
        wick_up_frac = (h - o) / range_hl if range_hl > 0 else 0
        wick_dn_frac = (c - l) / range_hl if range_hl > 0 else 0
        
        match = true_engulf
        
        # Calculate strength
        strength = 0.0
        if match:
            strength = min(1.0, (curr_body / prev_body - 1.1) / 0.5) * 0.6  # Engulfing ratio
            strength += body_frac * 0.4  # Body dominance
        
        # Check if dangerous
        is_dangerous = (body_frac >= 0.6 and range_hl >= 1.25 * atr and vol >= 1.5 * avg_vol) if match else False
        
        return CandleDetect(match, strength, body_frac, wick_up_frac, wick_dn_frac, is_dangerous, 'bear_engulf')
    
    def detect_all_patterns(self, bars: list, atr: float, avg_vol: float, tick_size: float) -> Optional[CandleDetect]:
        """
        Check all patterns on the last closed bar
        bars: list of dicts with 'open', 'high', 'low', 'close', 'volume'
        Returns the strongest matching pattern or None
        """
        if len(bars) < 2:
            return None
        
        # Last closed bar
        last = bars[-1]
        prev = bars[-2] if len(bars) > 1 else None
        
        o, h, l, c, vol = last['open'], last['high'], last['low'], last['close'], last['volume']
        
        detections = []
        
        # Single bar patterns
        hammer = self.detect_hammer(o, h, l, c, atr, vol, avg_vol, tick_size)
        if hammer.match:
            detections.append(hammer)
        
        shooting_star = self.detect_shooting_star(o, h, l, c, atr, vol, avg_vol, tick_size)
        if shooting_star.match:
            detections.append(shooting_star)
        
        # Two-bar patterns
        if prev:
            prev_o, prev_h, prev_l, prev_c = prev['open'], prev['high'], prev['low'], prev['close']
            
            bull_engulf = self.detect_bullish_engulfing(o, h, l, c, prev_o, prev_h, prev_l, prev_c,
                                                        atr, vol, avg_vol, tick_size)
            if bull_engulf.match:
                detections.append(bull_engulf)
            
            bear_engulf = self.detect_bearish_engulfing(o, h, l, c, prev_o, prev_h, prev_l, prev_c,
                                                        atr, vol, avg_vol, tick_size)
            if bear_engulf.match:
                detections.append(bear_engulf)
        
        # Return the strongest detection
        if detections:
            return max(detections, key=lambda d: d.strength)
        
        return None
    
    def guard_decision(self, side: Literal['long', 'short'], near_level: bool, 
                      params: dict, context: dict) -> dict:
        """
        Make veto/boost decision based on candlestick patterns
        
        Args:
            side: intended trade side ('long' or 'short')
            near_level: within near_level_ticks of key level or TL touch
            params: CANDLES config block
            context: dict with {atr, vol_ratio, tick_size, bars, symbol, timestamp}
        
        Returns:
            {"hard_veto": bool, "soft_bonus": float, "candle": CandleDetect|None}
        """
        if not params.get('enable', True):
            return {"hard_veto": False, "soft_bonus": 0.0, "candle": None}
        
        # Check for recent detection in cache
        symbol = context.get('symbol', 'NQ')
        timestamp = context.get('timestamp', datetime.now())
        
        # Clean old cache entries
        self._clean_cache()
        
        # Check if we have a recent detection within scope_bars
        candle = None
        for (cached_symbol, cached_time), cached_candle in self.recent_detections.items():
            if cached_symbol == symbol:
                bars_ago = (timestamp - cached_time).total_seconds() / 60  # Assume 1-min bars
                if bars_ago <= params.get('scope_bars', 2):
                    candle = cached_candle
                    break
        
        # If no cached detection, check current bar
        if not candle and 'bars' in context:
            bars = context['bars']
            atr = context.get('atr', 5.0)
            avg_vol = context.get('avg_vol', 1000)
            tick_size = context.get('tick_size', 0.25)
            
            candle = self.detect_all_patterns(bars, atr, avg_vol, tick_size)
            
            if candle:
                # Cache the detection
                self.recent_detections[(symbol, timestamp)] = candle
        
        if not candle:
            return {"hard_veto": False, "soft_bonus": 0.0, "candle": None}
        
        # Determine if candle is against intended side
        is_against = False
        if side == 'long':
            # Bearish patterns are against longs
            is_against = candle.kind in ['shooting_star', 'bear_engulf']
        else:  # short
            # Bullish patterns are against shorts
            is_against = candle.kind in ['hammer', 'bull_engulf']
        
        # Hard veto if against side and (near level or dangerous)
        hard_veto = is_against and (near_level or candle.is_dangerous)
        
        # Soft bonus/penalty
        soft_bonus = 0.0
        if not hard_veto:
            boost_range = params.get('soft_boost', [0.05, 0.10])
            if is_against:
                # Penalty for going against candle
                soft_bonus = -min(boost_range[1], boost_range[0] * candle.strength)
            else:
                # Bonus for aligned candle
                soft_bonus = min(boost_range[1], boost_range[0] * candle.strength)
        
        return {
            "hard_veto": hard_veto,
            "soft_bonus": soft_bonus,
            "candle": candle
        }
    
    def _clean_cache(self):
        """Remove old cache entries"""
        now = datetime.now()
        to_remove = []
        for key, _ in self.recent_detections.items():
            _, timestamp = key
            if now - timestamp > self.max_cache_age:
                to_remove.append(key)
        
        for key in to_remove:
            del self.recent_detections[key]

# Global instance
_candle_guard = None

def get_candle_guard() -> CandlestickGuard:
    """Get or create the global candlestick guard instance"""
    global _candle_guard
    if _candle_guard is None:
        _candle_guard = CandlestickGuard()
    return _candle_guard