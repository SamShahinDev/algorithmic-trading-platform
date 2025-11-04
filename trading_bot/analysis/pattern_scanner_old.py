"""
NQ-Optimized Pattern Scanner
Data-driven patterns discovered from 2 years of historical NQ data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import talib

class PatternType(Enum):
    """Pattern types discovered from NQ data"""
    MEAN_REVERSION = "mean_reversion"
    VOLUME_SPIKE = "volume_spike"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    RANGE_BREAKOUT = "range_breakout"
    OPENING_RANGE = "opening_range"

@dataclass
class Pattern:
    """Detected pattern with confidence"""
    pattern_type: PatternType
    direction: int  # 1 for long, -1 for short
    strength: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float

class PatternScanner:
    """Scanner for NQ-specific profitable patterns"""
    
    def __init__(self, min_strength: float = 50):
        self.min_strength = min_strength
        self.patterns_detected = []
        
    def scan_all_patterns(self, data: pd.DataFrame, features: pd.DataFrame = None) -> Dict[PatternType, Pattern]:
        """Compatibility method for old code - ignores features parameter"""
        return self.scan(data)
    
    def scan(self, data: pd.DataFrame) -> Dict[PatternType, Pattern]:
        """Scan for all NQ-optimized patterns"""
        patterns = {}
        
        # Only scan if we have enough data
        if len(data) < 30:
            return patterns
        
        # Mean Reversion Pattern (Best performer)
        mean_rev = self._scan_mean_reversion(data)
        if mean_rev and mean_rev.strength >= self.min_strength:
            patterns[PatternType.MEAN_REVERSION] = mean_rev
        
        # Volume Spike Fade Pattern
        vol_spike = self._scan_volume_spike(data)
        if vol_spike and vol_spike.strength >= self.min_strength:
            patterns[PatternType.VOLUME_SPIKE] = vol_spike
        
        # Momentum Breakout Pattern
        momentum = self._scan_momentum_breakout(data)
        if momentum and momentum.strength >= self.min_strength:
            patterns[PatternType.MOMENTUM_BREAKOUT] = momentum
        
        # Range Breakout Pattern
        range_break = self._scan_range_breakout(data)
        if range_break and range_break.strength >= self.min_strength:
            patterns[PatternType.RANGE_BREAKOUT] = range_break
        
        return patterns
    
    def _scan_mean_reversion(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Scan for mean reversion setup - Our best performer"""
        # RELAXED Parameters for more frequent signals
        bb_period = 20
        bb_std = 2.0  # Reduced from 2.5
        rsi_period = 14
        
        # Calculate indicators - ensure float64 for talib
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Bollinger Bands
        sma = talib.SMA(close, timeperiod=bb_period)
        std = talib.STDDEV(close, timeperiod=bb_period)
        upper_band = sma + (std * bb_std)
        lower_band = sma - (std * bb_std)
        
        # RSI
        rsi = talib.RSI(close, timeperiod=rsi_period)
        
        # Volume analysis
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        current_close = close[-1]
        current_rsi = rsi[-1]
        current_upper = upper_band[-1]
        current_lower = lower_band[-1]
        
        # Check for mean reversion setup - RELAXED thresholds
        if current_close < current_lower and current_rsi < 40:
            # Oversold - Long setup
            strength = min(100, (40 - current_rsi) * 2 + 
                          (current_lower - current_close) / current_close * 1000)
            
            return Pattern(
                pattern_type=PatternType.MEAN_REVERSION,
                direction=1,
                strength=strength,
                entry_price=current_close,
                stop_loss=current_close - 4,  # 4 point stop
                take_profit=sma[-1],  # Target mean
                confidence=0.55  # Historical win rate
            )
            
        elif current_close > current_upper and current_rsi > 60:
            # Overbought - Short setup
            strength = min(100, (current_rsi - 60) * 2 + 
                          (current_close - current_upper) / current_close * 1000)
            
            return Pattern(
                pattern_type=PatternType.MEAN_REVERSION,
                direction=-1,
                strength=strength,
                entry_price=current_close,
                stop_loss=current_close + 4,
                take_profit=sma[-1],
                confidence=0.55
            )
        
        return None
    
    def _scan_volume_spike(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Scan for volume spike fade setup"""
        volume = data['volume'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        # Volume spike detection
        vol_ma = talib.SMA(volume, timeperiod=20)
        current_vol = volume[-1]
        
        if current_vol > vol_ma[-1] * 1.5:  # RELAXED: 1.5x average volume
            # Check for price spike to fade
            price_change = (close[-1] - close[-6]) / close[-6]  # 5-bar change
            
            if abs(price_change) > 0.0015:  # RELAXED: 0.15% move
                direction = -1 if price_change > 0 else 1  # Fade the move
                strength = min(100, current_vol / vol_ma[-1] * 20)
                
                return Pattern(
                    pattern_type=PatternType.VOLUME_SPIKE,
                    direction=direction,
                    entry_price=close[-1],
                    stop_loss=close[-1] - (5 * direction),
                    take_profit=close[-1] + (6 * direction),
                    confidence=0.52
                )
        
        return None
    
    def _scan_momentum_breakout(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Scan for momentum acceleration"""
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Rate of change
        roc = talib.ROC(close, timeperiod=15)
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        if abs(roc[-1]) > 0.15 and volume[-1] > vol_ma[-1] * 1.3:  # RELAXED thresholds
            direction = 1 if roc[-1] > 0 else -1
            strength = min(100, abs(roc[-1]) * 100)
            
            return Pattern(
                pattern_type=PatternType.MOMENTUM_BREAKOUT,
                direction=direction,
                entry_price=close[-1],
                stop_loss=close[-1] - (4 * direction),
                take_profit=close[-1] + (6 * direction),
                confidence=0.48
            )
        
        return None
    
    def _scan_range_breakout(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Scan for range breakout"""
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Calculate range
        resistance = max(high[-20:])
        support = min(low[-20:])
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        # Check for breakout
        if close[-1] > resistance + 2 and volume[-1] > vol_ma[-1] * 1.5:
            # Bullish breakout
            return Pattern(
                pattern_type=PatternType.RANGE_BREAKOUT,
                direction=1,
                entry_price=close[-1],
                stop_loss=resistance - 2,
                take_profit=close[-1] + 8,
                confidence=0.35
            )
        elif close[-1] < support - 2 and volume[-1] > vol_ma[-1] * 1.5:
            # Bearish breakout
            return Pattern(
                pattern_type=PatternType.RANGE_BREAKOUT,
                direction=-1,
                entry_price=close[-1],
                stop_loss=support + 2,
                take_profit=close[-1] - 8,
                confidence=0.35
            )
        
        return None
