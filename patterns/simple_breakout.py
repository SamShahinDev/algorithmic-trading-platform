"""
Simple Breakout Pattern - Triggers on any 5-minute high/low break
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import logging
from patterns.base_pattern import BasePattern, PatternSignal
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SimpleBreakoutPattern(BasePattern):
    """Simple breakout pattern that triggers on 5-minute high/low breaks"""
    
    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.lookback_bars = self.config.get('lookback_bars', 10)  # Look at last 10 bars
        self.min_range = self.config.get('min_range', 2.0)  # Minimum range in points
        
    def _initialize(self):
        """Initialize pattern-specific parameters"""
        self.lookback_bars = self.config.get('lookback_bars', 10)
        self.min_range = self.config.get('min_range', 2.0)
        
    def scan_for_setup(self, data: pd.DataFrame, current_price: float, **kwargs) -> Optional[PatternSignal]:
        """Scan for simple breakout patterns"""
        
        if len(data) < self.lookback_bars + 1:
            return None
            
        # Get recent high and low
        recent_data = data.tail(self.lookback_bars)
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        
        # Check if range is sufficient
        range_points = recent_high - recent_low
        if range_points < self.min_range:
            return None
            
        # Check for breakout
        last_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        
        signal = None
        
        # Bullish breakout
        if last_close > recent_high and prev_close <= recent_high:
            logger.info(f"BULLISH BREAKOUT: Price {last_close} broke above {recent_high}")
            signal = PatternSignal(
                pattern_name="SimpleBreakout",
                action=TradeAction.BUY,
                confidence=0.5,  # Fixed confidence
                entry_price=current_price,
                stop_loss=recent_low,
                take_profit=current_price + (range_points * 2),  # 2:1 RR
                position_size=1,
                reason=f"Breakout above {recent_high:.2f}",
                direction=1
            )
            
        # Bearish breakout
        elif last_close < recent_low and prev_close >= recent_low:
            logger.info(f"BEARISH BREAKOUT: Price {last_close} broke below {recent_low}")
            signal = PatternSignal(
                pattern_name="SimpleBreakout",
                action=TradeAction.SELL,
                confidence=0.5,  # Fixed confidence
                entry_price=current_price,
                stop_loss=recent_high,
                take_profit=current_price - (range_points * 2),  # 2:1 RR
                position_size=1,
                reason=f"Breakout below {recent_low:.2f}",
                direction=-1
            )
            
        return signal
        
    def calculate_confidence(self, data: pd.DataFrame, signal: PatternSignal) -> float:
        """Return fixed confidence for simple pattern"""
        return 0.5  # 50% confidence for all signals