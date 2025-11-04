# File: trading_bot/indicators/normalized_indicators.py
import talib
import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class NormalizedIndicators:
    """Normalized technical indicators with consistent unit contracts"""
    
    @staticmethod
    def roc_percent(close: np.ndarray, period: int) -> float:
        """
        Calculate Rate of Change in percent units
        
        Args:
            close: Close prices array
            period: Lookback period
            
        Returns:
            ROC in percent (1.0 = 1% change)
        """
        if len(close) < period + 1:
            return 0.0
            
        roc = talib.ROC(close, timeperiod=period)
        val = float(roc[-1]) if not np.isnan(roc[-1]) else 0.0
        
        # Sanity check - NQ rarely moves >10% intraday
        if abs(val) > 10:
            logger.warning(f"ROC suspiciously large: {val}%, capping at 10%")
            val = np.sign(val) * 10
            
        return val
    
    @staticmethod
    def crosses_roc_threshold(close: np.ndarray, period: int, threshold_pct: float) -> bool:
        """
        Check if ROC crosses threshold
        
        Args:
            close: Close prices
            period: ROC period
            threshold_pct: Threshold in percent (1.0 = 1%)
            
        Returns:
            True if threshold crossed
        """
        val = NormalizedIndicators.roc_percent(close, period)
        return abs(val) >= threshold_pct
    
    @staticmethod
    def get_all_normalized_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all indicators with normalized units"""
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        
        indicators = {
            'roc_5': NormalizedIndicators.roc_percent(close, 5),
            'roc_10': NormalizedIndicators.roc_percent(close, 10),
            'roc_20': NormalizedIndicators.roc_percent(close, 20),
            'rsi': float(talib.RSI(close, 14)[-1]),
            'atr': float(talib.ATR(high, low, close, 14)[-1]),
            'adx': float(talib.ADX(high, low, close, 14)[-1]),
            'volume_ratio': float(volume[-1] / np.mean(volume[-20:]))
        }
        
        return indicators