"""
NQ Behavior Analyzer - Understanding 10-point moves
Analyzes market conditions that precede significant moves in NQ futures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time, timedelta
import logging
import json
from dataclasses import dataclass
import talib

logger = logging.getLogger(__name__)

@dataclass
class MarketCondition:
    """Market conditions before a significant move"""
    timestamp: datetime
    move_direction: str  # 'up' or 'down'
    move_points: float
    
    # Pre-move conditions
    volume_ratio: float  # Current vs average
    volatility: float  # ATR
    trend_strength: float  # ADX
    momentum: float  # RSI
    
    # Price structure
    distance_from_vwap: float
    distance_from_daily_high: float
    distance_from_daily_low: float
    
    # Time factors
    time_of_day: str  # 'opening', 'midday', 'closing'
    minutes_from_open: int
    day_of_week: str

class NQBehaviorAnalyzer:
    """Analyze NQ behavior patterns for 10-point moves"""
    
    # NQ session times (Chicago time)
    OPEN_TIME = time(8, 30)
    CLOSE_TIME = time(15, 0)
    
    # Move thresholds
    SIGNIFICANT_MOVE = 10  # points
    ANALYSIS_WINDOW = 30  # bars to analyze before move
    OUTCOME_WINDOW = 60  # bars to check for move completion
    
    def __init__(self):
        self.conditions_database = []
        self.pattern_statistics = {}
        
    def analyze_live_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze current market conditions for 10-point move potential
        
        Args:
            data: Recent price data (at least 100 bars)
        
        Returns:
            Analysis with move probability and direction
        """
        if len(data) < 100:
            return {'probability': 0, 'direction': 'neutral', 'confidence': 0}
        
        # Calculate current market metrics
        current_condition = self._calculate_current_conditions(data)
        
        # Compare with historical patterns
        similar_conditions = self._find_similar_historical(current_condition)
        
        # Calculate probabilities
        if similar_conditions:
            up_moves = sum(1 for c in similar_conditions if c['direction'] == 'up')
            down_moves = sum(1 for c in similar_conditions if c['direction'] == 'down')
            total = len(similar_conditions)
            
            if up_moves > down_moves * 1.5:
                return {
                    'probability': up_moves / total,
                    'direction': 'up',
                    'confidence': self._calculate_confidence(similar_conditions),
                    'expected_points': 10,
                    'conditions': current_condition
                }
            elif down_moves > up_moves * 1.5:
                return {
                    'probability': down_moves / total,
                    'direction': 'down',
                    'confidence': self._calculate_confidence(similar_conditions),
                    'expected_points': 10,
                    'conditions': current_condition
                }
        
        return {'probability': 0, 'direction': 'neutral', 'confidence': 0}
    
    def _calculate_current_conditions(self, data: pd.DataFrame) -> Dict:
        """Calculate current market conditions"""
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Technical indicators
        atr = talib.ATR(high, low, close, timeperiod=14)
        adx = talib.ADX(high, low, close, timeperiod=14)
        rsi = talib.RSI(close, timeperiod=14)
        
        # Volume analysis
        vol_ma = talib.SMA(volume, timeperiod=20)
        volume_ratio = volume[-1] / vol_ma[-1] if vol_ma[-1] > 0 else 1
        
        # VWAP calculation (simplified)
        typical_price = (high + low + close) / 3
        vwap = np.sum(typical_price[-20:] * volume[-20:]) / np.sum(volume[-20:])
        
        # Daily high/low (from available data)
        daily_high = np.max(high[-390:])  # ~6.5 hours of 1-min bars
        daily_low = np.min(low[-390:])
        
        # Time analysis
        current_time = data.index[-1]
        minutes_from_open = self._calculate_minutes_from_open(current_time)
        
        return {
            'volume_ratio': volume_ratio,
            'volatility': atr[-1] if not np.isnan(atr[-1]) else 0,
            'trend_strength': adx[-1] if not np.isnan(adx[-1]) else 0,
            'momentum': rsi[-1] if not np.isnan(rsi[-1]) else 50,
            'distance_from_vwap': (close[-1] - vwap) / close[-1] * 100,
            'distance_from_daily_high': (daily_high - close[-1]),
            'distance_from_daily_low': (close[-1] - daily_low),
            'time_of_day': self._get_session_period(current_time),
            'minutes_from_open': minutes_from_open
        }
    
    def _calculate_minutes_from_open(self, timestamp: datetime) -> int:
        """Calculate minutes from market open"""
        market_open = timestamp.replace(hour=8, minute=30, second=0)
        if timestamp.time() < self.OPEN_TIME:
            return 0
        return int((timestamp - market_open).total_seconds() / 60)
    
    def _get_session_period(self, timestamp: datetime) -> str:
        """Determine session period"""
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Convert to minutes from midnight
        total_minutes = hour * 60 + minute
        open_minutes = 8 * 60 + 30  # 8:30 AM
        
        if total_minutes < open_minutes + 90:  # First 90 minutes
            return 'opening'
        elif total_minutes < open_minutes + 240:  # Next 2.5 hours
            return 'midday'
        else:
            return 'closing'
    
    def _find_similar_historical(self, current: Dict) -> List[Dict]:
        """Find historically similar conditions"""
        # This would normally query a database of historical patterns
        # For now, return empty list as we don't have historical data loaded
        return []
    
    def _calculate_confidence(self, similar_conditions: List[Dict]) -> float:
        """Calculate confidence based on historical similarity"""
        if not similar_conditions:
            return 0
        
        # Calculate based on consistency of outcomes
        outcomes = [c['outcome'] for c in similar_conditions]
        most_common = max(set(outcomes), key=outcomes.count)
        consistency = outcomes.count(most_common) / len(outcomes)
        
        return consistency
    
    def identify_10point_setups(self, data: pd.DataFrame) -> List[Dict]:
        """
        Identify setups that could lead to 10-point moves
        
        Returns:
            List of potential setups with entry/stop/target
        """
        setups = []
        
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Pattern 1: Volume Expansion Breakout
        vol_ma = talib.SMA(volume, timeperiod=20)
        if volume[-1] > vol_ma[-1] * 2.5:
            recent_range = high[-20:].max() - low[-20:].min()
            if recent_range < 10:  # Tight range before breakout
                setup = {
                    'pattern': 'volume_expansion_breakout',
                    'entry': close[-1],
                    'direction': 1 if close[-1] > close[-2] else -1,
                    'stop': 5,  # Fixed 5-point stop
                    'target': 10,  # Fixed 10-point target
                    'confidence': 0.45
                }
                setups.append(setup)
        
        # Pattern 2: Momentum Exhaustion Reversal
        rsi = talib.RSI(close, timeperiod=9)
        if rsi[-1] < 20 or rsi[-1] > 80:
            momentum_change = close[-1] - close[-5]
            if abs(momentum_change) > 5:  # Strong momentum
                setup = {
                    'pattern': 'momentum_exhaustion',
                    'entry': close[-1],
                    'direction': -1 if rsi[-1] > 80 else 1,
                    'stop': 5,
                    'target': 10,
                    'confidence': 0.40
                }
                setups.append(setup)
        
        # Pattern 3: Range Extension
        atr = talib.ATR(high, low, close, timeperiod=14)
        current_range = high[-1] - low[-1]
        if current_range > atr[-1] * 2:  # Extended range
            mid_point = (high[-1] + low[-1]) / 2
            if close[-1] > mid_point + 2:  # Near high
                setup = {
                    'pattern': 'range_extension_fade',
                    'entry': close[-1],
                    'direction': -1,
                    'stop': 5,
                    'target': 10,
                    'confidence': 0.38
                }
                setups.append(setup)
            elif close[-1] < mid_point - 2:  # Near low
                setup = {
                    'pattern': 'range_extension_fade',
                    'entry': close[-1],
                    'direction': 1,
                    'stop': 5,
                    'target': 10,
                    'confidence': 0.38
                }
                setups.append(setup)
        
        return setups

class TimeOfDayPatterns:
    """Time-specific patterns for NQ futures"""
    
    @staticmethod
    def opening_drive(data: pd.DataFrame) -> Optional[Dict]:
        """
        Opening drive pattern - First 30 minutes momentum
        Strong directional move at open that continues
        """
        current_time = data.index[-1]
        if current_time.time() < time(9, 0) or current_time.time() > time(9, 30):
            return None
        
        # Calculate opening range
        open_price = data.iloc[0]['open']
        current_price = data.iloc[-1]['close']
        
        move = current_price - open_price
        
        if abs(move) > 5:  # Already 5+ points from open
            return {
                'pattern': 'opening_drive',
                'direction': 1 if move > 0 else -1,
                'entry': current_price,
                'stop': 5,
                'target': 10,
                'confidence': 0.42,
                'time_window': '8:30-9:30'
            }
        
        return None
    
    @staticmethod
    def midday_reversal(data: pd.DataFrame) -> Optional[Dict]:
        """
        Midday reversal pattern - 11:30-12:30 reversal zone
        Market often reverses during lunch hour
        """
        current_time = data.index[-1]
        if current_time.time() < time(11, 30) or current_time.time() > time(12, 30):
            return None
        
        # Check for morning trend exhaustion
        morning_data = data[data.index.time < time(11, 30)]
        if len(morning_data) < 60:
            return None
        
        morning_move = morning_data.iloc[-1]['close'] - morning_data.iloc[0]['open']
        
        if abs(morning_move) > 15:  # Strong morning move
            close = data['close'].values.astype(np.float64)
            rsi = talib.RSI(close, timeperiod=14)
            
            if (morning_move > 0 and rsi[-1] > 65) or (morning_move < 0 and rsi[-1] < 35):
                return {
                    'pattern': 'midday_reversal',
                    'direction': -1 if morning_move > 0 else 1,
                    'entry': data.iloc[-1]['close'],
                    'stop': 5,
                    'target': 10,
                    'confidence': 0.41,
                    'time_window': '11:30-12:30'
                }
        
        return None
    
    @staticmethod
    def power_hour(data: pd.DataFrame) -> Optional[Dict]:
        """
        Power hour pattern - Final hour momentum
        Strong moves in the last hour of trading
        """
        current_time = data.index[-1]
        if current_time.time() < time(14, 0) or current_time.time() > time(15, 0):
            return None
        
        # Check for afternoon range break
        afternoon_data = data[data.index.time > time(13, 0)]
        if len(afternoon_data) < 30:
            return None
        
        high = afternoon_data['high'].max()
        low = afternoon_data['low'].min()
        current = data.iloc[-1]['close']
        
        if current > high - 1:  # Breaking afternoon high
            return {
                'pattern': 'power_hour_breakout',
                'direction': 1,
                'entry': current,
                'stop': 5,
                'target': 10,
                'confidence': 0.39,
                'time_window': '14:00-15:00'
            }
        elif current < low + 1:  # Breaking afternoon low
            return {
                'pattern': 'power_hour_breakdown',
                'direction': -1,
                'entry': current,
                'stop': 5,
                'target': 10,
                'confidence': 0.39,
                'time_window': '14:00-15:00'
            }
        
        return None