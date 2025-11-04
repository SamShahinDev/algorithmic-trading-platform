"""
Market Regime Detection and Filtering
Controls when patterns are allowed to trade based on market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, time, timedelta
from typing import Dict, Tuple, Optional, List
import logging
from collections import deque

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects market regimes and determines when patterns can trade
    """
    
    # Constants for Momentum Thrust
    MT_ADX_MIN = 18
    MT_ATR_BAND_LOW = 1.2  # × 24h median
    MT_ATR_BAND_HIGH = 2.5
    MT_TIME_START = time(19, 30)  # 19:30 CT
    MT_TIME_END = time(23, 30)    # 23:30 CT
    
    # Constants for news and open blocks
    NEWS_BLOCK_TIME = time(8, 30)  # 08:30 CT
    NEWS_BLOCK_WINDOW_MINUTES = 3
    OPEN_BLOCK_START = time(9, 30)  # 09:30 CT
    OPEN_BLOCK_END = time(9, 35)    # 09:35 CT
    
    # Constants for Trend Line Bounce
    LEVEL_PROXIMITY_MIN_TICKS = 4
    LEVEL_PROXIMITY_MAX_TICKS = 6
    TICK_SIZE = 0.25
    
    def __init__(self, data_cache=None):
        """
        Initialize market regime detector
        
        Args:
            data_cache: DataCache instance for accessing indicators
        """
        self.data_cache = data_cache
        
        # ATR median tracking (24 hours = 1440 1-minute bars)
        self.atr_history = deque(maxlen=1440)
        self.atr_median_24h = None
        self.last_atr_update = None
        
        # Cache for key levels
        self.vwap = None
        self.onh = None  # Overnight high
        self.onl = None  # Overnight low
        self.poc = None  # Point of control
        self.levels_last_update = None
        
        # RANSAC R² tracking for trend quality
        self.ransac_r2 = None
        self.ransac_last_update = None
    
    def update_atr_history(self, current_atr: float) -> None:
        """
        Update ATR history and calculate 24h median
        
        Args:
            current_atr: Current ATR value
        """
        self.atr_history.append(current_atr)
        
        # Import baseline config
        from ..pattern_config import ATR_BASELINE_MIN_FALLBACK
        
        if len(self.atr_history) >= ATR_BASELINE_MIN_FALLBACK:
            # Use all available data for median, up to 24h
            self.atr_median_24h = np.median(list(self.atr_history))
            self.last_atr_update = datetime.now(timezone.utc)
            
            # Log median update every hour
            if len(self.atr_history) % 60 == 0:
                logger.debug(f"24h ATR median updated: {self.atr_median_24h:.2f} "
                           f"(samples: {len(self.atr_history)})")
    
    def get_atr_band(self, current_atr: float) -> float:
        """
        Calculate ATR band ratio (current ATR / median baseline)
        
        Args:
            current_atr: Current ATR value
            
        Returns:
            ATR band ratio, never 0.00, defaults to 1.00 if baseline unavailable
        """
        if self.atr_median_24h and self.atr_median_24h > 0:
            atr_band = current_atr / self.atr_median_24h if current_atr > 0 else 1.00
            return round(atr_band, 2)
        else:
            # Baseline not ready, return 1.00 and log
            logger.debug('REGIME_INFO atr_band=1.00 baseline="not_ready"')
            return 1.00
    
    def get_current_time_ct(self) -> time:
        """
        Get current time in Central Time
        
        Returns:
            Current time as time object in CT
        """
        try:
            from zoneinfo import ZoneInfo
            ct_tz = ZoneInfo('America/Chicago')
        except ImportError:
            import pytz
            ct_tz = pytz.timezone('America/Chicago')
        
        now_ct = datetime.now(timezone.utc).astimezone(ct_tz)
        return now_ct.time()
    
    def is_news_block(self) -> Tuple[bool, str]:
        """
        Check if we're in news block window
        
        Returns:
            Tuple of (is_blocked, reason)
        """
        current_time = self.get_current_time_ct()
        
        # Create time window around news
        news_start = (datetime.combine(datetime.today(), self.NEWS_BLOCK_TIME) - 
                     timedelta(minutes=self.NEWS_BLOCK_WINDOW_MINUTES)).time()
        news_end = (datetime.combine(datetime.today(), self.NEWS_BLOCK_TIME) + 
                   timedelta(minutes=self.NEWS_BLOCK_WINDOW_MINUTES)).time()
        
        if news_start <= current_time <= news_end:
            return True, f"News block: {current_time.strftime('%H:%M:%S')} CT (08:27:00-08:33:00)"
        
        return False, ""
    
    def is_open_block(self) -> Tuple[bool, str]:
        """
        Check if we're in market open block window
        
        Returns:
            Tuple of (is_blocked, reason)
        """
        current_time = self.get_current_time_ct()
        
        if self.OPEN_BLOCK_START <= current_time <= self.OPEN_BLOCK_END:
            return True, f"Open block: {current_time.strftime('%H:%M:%S')} CT (09:30:00-09:35:00)"
        
        return False, ""
    
    def check_mt_regime(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if Momentum Thrust pattern is allowed
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        reasons = []
        
        # Check time window (19:30:00-23:30:00 CT inclusive)
        current_time = self.get_current_time_ct()
        if not (self.MT_TIME_START <= current_time <= self.MT_TIME_END):
            reasons.append(f"time {current_time.strftime('%H:%M:%S')}")
        
        # Check news and open blocks
        is_news, news_reason = self.is_news_block()
        if is_news:
            reasons.append(news_reason)
        
        is_open, open_reason = self.is_open_block()
        if is_open:
            reasons.append(open_reason)
        
        # Check ADX requirement
        if self.data_cache:
            adx = self.data_cache.get_indicator('adx', '1m')
            if adx is not None and adx < self.MT_ADX_MIN:
                reasons.append(f"ADX {adx:.1f} < {self.MT_ADX_MIN}")
        
        # Check ATR bands
        if self.data_cache and self.atr_median_24h:
            current_atr = self.data_cache.get_indicator('atr', '1m')
            if current_atr:
                self.update_atr_history(current_atr)
                
                atr_low = self.atr_median_24h * self.MT_ATR_BAND_LOW
                atr_high = self.atr_median_24h * self.MT_ATR_BAND_HIGH
                
                if not (atr_low <= current_atr <= atr_high):
                    reasons.append(f"ATR {current_atr:.2f} outside [{atr_low:.2f}, {atr_high:.2f}]")
        
        # Return result
        if reasons:
            return False, f"MT blocked: {', '.join(reasons)}"
        
        return True, "MT regime OK"
    
    def update_key_levels(self, data: pd.DataFrame) -> None:
        """
        Update key price levels (VWAP, ONH, ONL, POC)
        
        Args:
            data: OHLCV data
        """
        if len(data) < 10:
            return
        
        # Calculate VWAP
        if 'volume' in data.columns:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            cumulative_tpv = (typical_price * data['volume']).cumsum()
            cumulative_volume = data['volume'].cumsum()
            self.vwap = cumulative_tpv.iloc[-1] / cumulative_volume.iloc[-1] if cumulative_volume.iloc[-1] > 0 else None
        
        # Calculate overnight high/low (using last 390 bars for regular session)
        if len(data) >= 390:
            overnight_data = data.iloc[-780:-390]  # Previous session
            if len(overnight_data) > 0:
                self.onh = overnight_data['high'].max()
                self.onl = overnight_data['low'].min()
        
        # Simple POC calculation (highest volume price level)
        if 'volume' in data.columns:
            # Create price bins
            price_range = data['high'].max() - data['low'].min()
            if price_range > 0:
                bins = 50
                price_bins = pd.cut(data['close'], bins=bins)
                volume_profile = data.groupby(price_bins)['volume'].sum()
                if len(volume_profile) > 0:
                    poc_bin = volume_profile.idxmax()
                    if poc_bin is not pd.NaT and poc_bin is not None:
                        self.poc = poc_bin.mid
        
        self.levels_last_update = datetime.now(timezone.utc)
    
    def check_tlb_regime(self, current_price: float, data: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Check if Trend Line Bounce pattern is allowed
        
        Args:
            current_price: Current market price
            data: OHLCV data for level calculation
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        reasons = []
        
        # Check news and open blocks
        is_news, news_reason = self.is_news_block()
        if is_news:
            reasons.append(news_reason)
        
        is_open, open_reason = self.is_open_block()
        if is_open:
            reasons.append(open_reason)
        
        # Update levels if data provided
        if data is not None:
            self.update_key_levels(data)
        
        # Check proximity to key levels
        near_level = False
        level_distances = []
        
        min_distance = self.LEVEL_PROXIMITY_MIN_TICKS * self.TICK_SIZE
        max_distance = self.LEVEL_PROXIMITY_MAX_TICKS * self.TICK_SIZE
        
        if self.vwap:
            dist = abs(current_price - self.vwap)
            if min_distance <= dist <= max_distance:
                near_level = True
                level_distances.append(f"VWAP {dist/self.TICK_SIZE:.1f} ticks")
        
        if self.onh:
            dist = abs(current_price - self.onh)
            if min_distance <= dist <= max_distance:
                near_level = True
                level_distances.append(f"ONH {dist/self.TICK_SIZE:.1f} ticks")
        
        if self.onl:
            dist = abs(current_price - self.onl)
            if min_distance <= dist <= max_distance:
                near_level = True
                level_distances.append(f"ONL {dist/self.TICK_SIZE:.1f} ticks")
        
        if self.poc:
            dist = abs(current_price - self.poc)
            if min_distance <= dist <= max_distance:
                near_level = True
                level_distances.append(f"POC {dist/self.TICK_SIZE:.1f} ticks")
        
        if not near_level and (self.vwap or self.onh or self.onl or self.poc):
            reasons.append(f"not near key levels (4-6 ticks)")
        
        # Check RANSAC R² if available
        if self.ransac_r2 is not None and self.ransac_r2 < 0.85:
            reasons.append(f"RANSAC R² {self.ransac_r2:.3f} < 0.85")
        
        # Return result
        if reasons:
            return False, f"TLB blocked: {', '.join(reasons)}"
        
        if level_distances:
            return True, f"TLB regime OK: near {', '.join(level_distances)}"
        
        return True, "TLB regime OK"
    
    def update_ransac_r2(self, r2_value: float) -> None:
        """
        Update RANSAC R² value for trend quality
        
        Args:
            r2_value: R² value from RANSAC regression
        """
        self.ransac_r2 = r2_value
        self.ransac_last_update = datetime.now(timezone.utc)
    
    def get_regime_status(self, current_price: float, data: pd.DataFrame = None) -> Dict:
        """
        Get comprehensive regime status for all patterns
        
        Args:
            current_price: Current market price
            data: OHLCV data for calculations
            
        Returns:
            Dictionary with regime status for each pattern
        """
        mt_allowed, mt_reason = self.check_mt_regime(current_price)
        tlb_allowed, tlb_reason = self.check_tlb_regime(current_price, data)
        
        # Check global blocks
        is_news, news_reason = self.is_news_block()
        is_open, open_reason = self.is_open_block()
        
        return {
            'timestamp': datetime.now(timezone.utc),
            'current_time_ct': self.get_current_time_ct().strftime('%H:%M:%S'),
            'global_blocks': {
                'news_block': is_news,
                'news_reason': news_reason,
                'open_block': is_open,
                'open_reason': open_reason
            },
            'patterns': {
                'momentum_thrust': {
                    'allowed': mt_allowed and not is_news and not is_open,
                    'reason': mt_reason
                },
                'trend_line_bounce': {
                    'allowed': tlb_allowed and not is_news and not is_open,
                    'reason': tlb_reason
                }
            },
            'metrics': {
                'atr_median_24h': self.atr_median_24h,
                'current_atr': self.data_cache.get_indicator('atr', '1m') if self.data_cache else None,
                'adx': self.data_cache.get_indicator('adx', '1m') if self.data_cache else None,
                'vwap': self.vwap,
                'ransac_r2': self.ransac_r2
            }
        }
    
    def should_scan_pattern(self, pattern_name: str, current_price: float, data: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Determine if a specific pattern should be scanned
        
        Args:
            pattern_name: Name of the pattern
            current_price: Current market price
            data: OHLCV data for calculations
            
        Returns:
            Tuple of (should_scan, reason)
        """
        # Check discovery mode bypass first
        try:
            from ..pattern_config import DISCOVERY_MODE, DISABLE_REGIME_GATING, DISABLE_TIME_BLOCKS
            if DISCOVERY_MODE and DISABLE_REGIME_GATING:
                return True, "discovery_mode_regime_bypass"
        except ImportError:
            pass
        
        pattern_name_lower = pattern_name.lower()
        
        # Check global blocks (unless bypassed by discovery mode)
        try:
            from ..pattern_config import DISCOVERY_MODE, DISABLE_TIME_BLOCKS
            if not (DISCOVERY_MODE and DISABLE_TIME_BLOCKS):
                is_news, news_reason = self.is_news_block()
                if is_news:
                    return False, news_reason
                
                is_open, open_reason = self.is_open_block()
                if is_open:
                    return False, open_reason
        except ImportError:
            # Fallback to original behavior if config not available
            is_news, news_reason = self.is_news_block()
            if is_news:
                return False, news_reason
            
            is_open, open_reason = self.is_open_block()
            if is_open:
                return False, open_reason
        
        # Check pattern-specific regime
        if 'momentum' in pattern_name_lower or 'thrust' in pattern_name_lower:
            return self.check_mt_regime(current_price)
        elif 'trend' in pattern_name_lower or 'bounce' in pattern_name_lower:
            return self.check_tlb_regime(current_price, data)
        else:
            # Unknown pattern, allow by default
            return True, f"{pattern_name} not regime-filtered"

def in_mt_window(now_ct) -> bool:
    """Check if current CT time is within MT window"""
    from ..pattern_config import MT_WINDOW_CT
    start_time = time(int(MT_WINDOW_CT[0].split(':')[0]), int(MT_WINDOW_CT[0].split(':')[1]))
    end_time = time(int(MT_WINDOW_CT[1].split(':')[0]), int(MT_WINDOW_CT[1].split(':')[1]))
    return start_time <= now_ct <= end_time

def mt_min_confidence(now_ct):
    """Get minimum confidence threshold for MT based on time window"""
    from ..pattern_config import MT_MIN_CONFIDENCE_WINDOW, MT_MIN_CONFIDENCE_DEFAULT
    return MT_MIN_CONFIDENCE_WINDOW if in_mt_window(now_ct) else MT_MIN_CONFIDENCE_DEFAULT

def mt_exhaustion_threshold(adx, now_ct):
    """Get exhaustion threshold based on ADX and time window"""
    from ..pattern_config import (MT_EXHAUSTION_MULT_ATR, MT_EXHAUSTION_MULT_ATR_DEFAULT, 
                                 MT_ADX_TREND_GUARD)
    return (MT_EXHAUSTION_MULT_ATR if in_mt_window(now_ct) and adx >= MT_ADX_TREND_GUARD 
            else MT_EXHAUSTION_MULT_ATR_DEFAULT)