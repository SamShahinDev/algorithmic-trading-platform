"""
Optimized NQ Pattern Scanner - Data-Driven Patterns
Implements patterns discovered from 2 years of historical NQ data
All patterns validated for 1:2 risk/reward (5 point stop, 10 point target)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import talib
import logging

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Pattern types discovered from NQ data"""
    BOLLINGER_SQUEEZE = "bollinger_squeeze"  # DISABLED - Low validation win rate (35.8%)
    VOLUME_CLIMAX = "volume_climax"  # DISABLED - Marginal profitability (36.2%)
    MOMENTUM_THRUST = "momentum_thrust"  # ACTIVE - Solid profitability (44.5%)
    # RANGE_EXPANSION was tested but failed profitability criteria (no configurations met 34% win rate)

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
    expected_win_rate: float

class OptimizedPatternScanner:
    """Scanner using data-driven NQ patterns with proven profitability"""
    
    def __init__(self, min_strength: float = 0):  # Set to 0 to allow all patterns
        self.min_strength = min_strength
        self.patterns_detected = []
        
        # Pattern enable/disable configuration
        self.ENABLED_PATTERNS = {
            'momentum_thrust': True,      # ENABLED - This pattern made $340 profit on Friday!
            'bollinger_squeeze': False,    # 35.8% validation win rate - TOO RISKY
            'volume_climax': False,        # 36.2% validation win rate - MARGINAL
        }
        
        # Best performing pattern configurations from discovery
        self.best_patterns = {
            'bollinger_squeeze_1': {
                'bb_period': 20, 'bb_std': 2.5, 'squeeze_threshold': 0.4,
                'validation_win_rate': 0.358, 'training_win_rate': 0.538
            },
            'bollinger_squeeze_2': {
                'bb_period': 25, 'bb_std': 2.0, 'squeeze_threshold': 0.6,
                'validation_win_rate': 0.380, 'training_win_rate': 0.496
            },
            'momentum_thrust': {
                'roc_period': 10, 'roc_threshold': 0.01,  # Reduced to 0.01 (1%) for maximum sensitivity
                'validation_win_rate': 0.445, 'training_win_rate': 0.453
            },
            'volume_climax': {
                'vol_mult': 2.0, 'price_move': 0.002,
                'validation_win_rate': 0.362, 'training_win_rate': 0.407
            }
        }
    
    def scan_all_patterns(self, data: pd.DataFrame, features: pd.DataFrame = None) -> Dict[PatternType, Pattern]:
        """Compatibility method for old code"""
        return self.scan(data)
    
    def scan(self, data: pd.DataFrame) -> Dict[PatternType, Pattern]:
        """Scan ONLY for profitable patterns (Momentum Thrust)"""
        patterns = {}
        
        if len(data) < 30:
            return patterns
        
        # Log active patterns on first scan
        if not hasattr(self, '_logged_config'):
            logger.info("=" * 60)
            logger.info("PATTERN CONFIGURATION - TRADING WITH MOMENTUM THRUST ONLY")
            logger.info(f"Active patterns: {[k for k, v in self.ENABLED_PATTERNS.items() if v]}")
            logger.info(f"Disabled patterns: {[k for k, v in self.ENABLED_PATTERNS.items() if not v]}")
            logger.info("=" * 60)
            self._logged_config = True
        
        # DISABLED: Bollinger Squeeze patterns (35.8-38% win rate too close to breakeven)
        # if self.ENABLED_PATTERNS.get('bollinger_squeeze', False):
        #     # Best Bollinger Squeeze (53.8% training, 35.8% validation)
        #     squeeze1 = self._scan_bollinger_squeeze(...)
        
        # ACTIVE: Momentum Thrust (45.3% training, 44.5% validation) - ONLY PROFITABLE PATTERN
        if self.ENABLED_PATTERNS.get('momentum_thrust', True):
            momentum = self._scan_momentum_thrust(
                data,
                **{k: v for k, v in self.best_patterns['momentum_thrust'].items() 
                   if k not in ['validation_win_rate', 'training_win_rate']}
            )
            if momentum:  # Accept all momentum patterns regardless of strength
                patterns[PatternType.MOMENTUM_THRUST] = momentum
                logger.info(f"Momentum Thrust added to patterns: strength={momentum.strength:.1f}%, direction={momentum.direction}")
        
        # DISABLED: Volume Climax (36.2% win rate marginally profitable)
        # if self.ENABLED_PATTERNS.get('volume_climax', False):
        #     volume = self._scan_volume_climax(...)
        
        return patterns
    
    def _scan_bollinger_squeeze(self, data: pd.DataFrame, bb_period: int = 20,
                                bb_std: float = 2.5, squeeze_threshold: float = 0.4) -> Optional[Pattern]:
        """Bollinger Band squeeze breakout pattern"""
        if len(data) < bb_period:
            return None
        
        close = data['close'].values.astype(np.float64)
        
        sma = talib.SMA(close, timeperiod=bb_period)
        std = talib.STDDEV(close, timeperiod=bb_period)
        upper = sma + (std * bb_std)
        lower = sma - (std * bb_std)
        
        # Check for squeeze
        band_width = upper[-1] - lower[-1]
        avg_width = np.mean(upper[-20:] - lower[-20:])
        
        if band_width < avg_width * squeeze_threshold:
            current_close = close[-1]
            
            # Breakout detection
            if current_close > upper[-1]:
                strength = min(100, 50 + (current_close - upper[-1]) / upper[-1] * 500)
                return Pattern(
                    pattern_type=PatternType.BOLLINGER_SQUEEZE,
                    direction=1,
                    strength=strength,
                    entry_price=current_close,
                    stop_loss=current_close - 5,
                    take_profit=current_close + 10,
                    confidence=0.36,  # Conservative validation win rate
                    expected_win_rate=0.36
                )
            elif current_close < lower[-1]:
                strength = min(100, 50 + (lower[-1] - current_close) / lower[-1] * 500)
                return Pattern(
                    pattern_type=PatternType.BOLLINGER_SQUEEZE,
                    direction=-1,
                    strength=strength,
                    entry_price=current_close,
                    stop_loss=current_close + 5,
                    take_profit=current_close - 10,
                    confidence=0.36,
                    expected_win_rate=0.36
                )
        
        return None
    
    def _scan_momentum_thrust(self, data: pd.DataFrame, roc_period: int = 10,
                              roc_threshold: float = 1.0) -> Optional[Pattern]:  # Changed to 1.0% threshold
        """Momentum thrust continuation pattern"""
        if len(data) < roc_period + 20:
            return None
        
        # Import normalized indicators
        import sys
        sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')
        from indicators.normalized_indicators import NormalizedIndicators
        
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Use normalized ROC (returns percentage)
        roc_val = NormalizedIndicators.roc_percent(close, roc_period)
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        # Check if ROC crosses threshold (comparing percentages)
        logger.info(f"ROC: {abs(roc_val):.2f}% vs threshold {roc_threshold}%, Volume ratio: {volume[-1]/vol_ma[-1]:.2f}")
        if NormalizedIndicators.crosses_roc_threshold(close, roc_period, roc_threshold) and volume[-1] > vol_ma[-1] * 0.5:
            logger.info("MOMENTUM THRUST PATTERN DETECTED!")
            current_close = close[-1]
            direction = 1 if roc_val > 0 else -1
            strength = min(100, abs(roc_val))  # roc_val is already in percent
            logger.info(f"Direction: {direction}, Strength: {strength}%")
            
            return Pattern(
                pattern_type=PatternType.MOMENTUM_THRUST,
                direction=direction,
                strength=strength,
                entry_price=current_close,
                stop_loss=current_close - (5 * direction),
                take_profit=current_close + (10 * direction),
                confidence=0.44,  # Strong validation win rate
                expected_win_rate=0.44
            )
        
        return None
    
    def _scan_volume_climax(self, data: pd.DataFrame, vol_mult: float = 2.0,
                            price_move: float = 0.002) -> Optional[Pattern]:
        """Volume climax reversal pattern"""
        if len(data) < 20:
            return None
        
        volume = data['volume'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        vol_avg = talib.SMA(volume, timeperiod=20)
        
        if volume[-1] > vol_avg[-1] * vol_mult:
            # Check for exhaustion move
            move = (close[-1] - close[-5]) / close[-5]
            
            if abs(move) > price_move:
                current_close = close[-1]
                # Fade the move
                direction = -1 if move > 0 else 1
                strength = min(100, volume[-1] / vol_avg[-1] * 20)
                
                return Pattern(
                    pattern_type=PatternType.VOLUME_CLIMAX,
                    direction=direction,
                    strength=strength,
                    entry_price=current_close,
                    stop_loss=current_close - (5 * direction),
                    take_profit=current_close + (10 * direction),
                    confidence=0.36,
                    expected_win_rate=0.36
                )
        
        return None
    
    def get_pattern_stats(self) -> Dict:
        """Get statistics about discovered patterns"""
        return {
            'patterns_config': self.best_patterns,
            'min_win_rate': 0.34,  # Minimum for profitability with 1:2 R/R
            'commission': 2.52,
            'stop_loss': 5,
            'take_profit': 10,
            'expected_daily_setups': 8  # Based on discovery results
        }