"""
Scalping Pattern Discovery Agent
Identifies high-probability 5-point scalping opportunities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf

from agents.base_agent import BaseAgent
from utils.logger import setup_logger
from config import SCALPING_CONFIG

class ScalpingPatternAgent(BaseAgent):
    """
    Discovers scalping patterns for quick 5-point profits
    Target: $100 per trade with 1 NQ contract
    """
    
    def __init__(self):
        """Initialize scalping pattern agent"""
        super().__init__('ScalpingPatterns')
        self.logger = setup_logger('ScalpingPatterns')
        
        # Scalping parameters
        self.target_points = SCALPING_CONFIG['points_target']
        self.stop_points = SCALPING_CONFIG['points_stop']
        self.timeframe = SCALPING_CONFIG['timeframe']
        
        self.logger.info(f"💨 Scalping Pattern Agent initialized")
        self.logger.info(f"   Target: {self.target_points} points (${self.target_points * 20})")
        self.logger.info(f"   Timeframe: {self.timeframe}")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        return True
    
    async def execute(self, data: Dict) -> Dict:
        """Execute scalping pattern discovery"""
        patterns = await self.discover_scalping_patterns(data.get('symbol', 'NQ'))
        return {'patterns': patterns, 'success': True}
    
    async def discover_scalping_patterns(self, symbol: str = 'NQ') -> List[Dict]:
        """
        Discover high-probability scalping patterns
        
        Returns:
            List of scalping patterns with entry/exit rules
        """
        patterns = []
        
        try:
            # Fetch recent data
            data = await self.fetch_scalping_data(symbol)
            
            if data is None or data.empty:
                self.logger.warning("No data available for scalping analysis")
                return patterns
            
            self.logger.info(f"Analyzing {len(data)} bars for scalping patterns...")
            
            # 1. VWAP Bounce Pattern
            vwap_pattern = self.find_vwap_bounces(data)
            if vwap_pattern:
                patterns.append(vwap_pattern)
            
            # 2. Opening Range Breakout
            orb_pattern = self.find_opening_range_breakouts(data)
            if orb_pattern:
                patterns.append(orb_pattern)
            
            # 3. Quick Mean Reversion
            reversion_pattern = self.find_mean_reversions(data)
            if reversion_pattern:
                patterns.append(reversion_pattern)
            
            # 4. Momentum Scalps
            momentum_pattern = self.find_momentum_scalps(data)
            if momentum_pattern:
                patterns.append(momentum_pattern)
            
            # 5. Support/Resistance Quick Trades
            sr_pattern = self.find_support_resistance_scalps(data)
            if sr_pattern:
                patterns.append(sr_pattern)
            
            self.logger.info(f"✅ Found {len(patterns)} scalping patterns")
            
            # Rank patterns by expected profitability
            patterns = self.rank_patterns(patterns)
            
            self.record_success()
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error discovering scalping patterns: {e}")
            self.record_error(e)
            return patterns
    
    async def fetch_scalping_data(self, symbol: str) -> pd.DataFrame:
        """Fetch intraday data for scalping analysis"""
        try:
            ticker = yf.Ticker(f"{symbol}=F")  # Futures symbol
            
            # Get 5-minute data for last 5 days
            data = ticker.history(period="5d", interval="5m")
            
            if not data.empty:
                # Add technical indicators
                data = self.add_scalping_indicators(data)
                self.logger.info(f"Fetched {len(data)} 5-minute bars")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching scalping data: {e}")
            return pd.DataFrame()
    
    def add_scalping_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for scalping"""
        
        # VWAP
        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # Moving averages
        data['SMA_9'] = data['Close'].rolling(9).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['EMA_5'] = data['Close'].ewm(span=5).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume analysis
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # ATR for volatility
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        data['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        return data
    
    def find_vwap_bounces(self, data: pd.DataFrame) -> Optional[Dict]:
        """Find VWAP bounce patterns"""
        
        if 'VWAP' not in data.columns:
            return None
        
        # Look for bounces off VWAP
        touches = 0
        successful_bounces = 0
        
        for i in range(20, len(data) - 5):
            # Check if price touched VWAP and bounced
            if (data['Low'].iloc[i] <= data['VWAP'].iloc[i] <= data['High'].iloc[i]):
                touches += 1
                
                # Check if it moved 5+ points within next 5 bars
                future_high = data['High'].iloc[i+1:i+6].max()
                if future_high - data['VWAP'].iloc[i] >= self.target_points:
                    successful_bounces += 1
        
        if touches > 0:
            win_rate = successful_bounces / touches
            
            if win_rate >= SCALPING_CONFIG['min_win_rate']:
                return {
                    'name': 'VWAP_Bounce_Scalp',
                    'type': 'scalping',
                    'confidence': win_rate,
                    'entry_rules': {
                        'condition': 'price_touches_vwap',
                        'confirmation': 'bullish_candle',
                        'volume': 'above_average'
                    },
                    'exit_rules': {
                        'take_profit': self.target_points,
                        'stop_loss': self.stop_points,
                        'time_limit': '15_minutes'
                    },
                    'statistics': {
                        'occurrences': touches,
                        'win_rate': win_rate,
                        'avg_time_to_target': '8_minutes'
                    }
                }
        
        return None
    
    def find_opening_range_breakouts(self, data: pd.DataFrame) -> Optional[Dict]:
        """Find opening range breakout patterns"""
        
        # Group by date and find opening range (first 30 minutes)
        data['Date'] = data.index.date
        data['Time'] = data.index.time
        
        breakout_opportunities = 0
        successful_breakouts = 0
        
        for date in data['Date'].unique():
            day_data = data[data['Date'] == date]
            
            if len(day_data) < 10:
                continue
            
            # First 30 minutes (6 x 5-minute bars)
            opening_range = day_data.iloc[:6]
            range_high = opening_range['High'].max()
            range_low = opening_range['Low'].min()
            
            # Look for breakouts in rest of day
            rest_of_day = day_data.iloc[6:]
            
            for i in range(len(rest_of_day) - 3):
                # Breakout above range
                if rest_of_day['Close'].iloc[i] > range_high:
                    breakout_opportunities += 1
                    
                    # Check if hit target
                    future_high = rest_of_day['High'].iloc[i+1:i+4].max()
                    if future_high - rest_of_day['Close'].iloc[i] >= self.target_points:
                        successful_breakouts += 1
        
        if breakout_opportunities > 0:
            win_rate = successful_breakouts / breakout_opportunities
            
            if win_rate >= SCALPING_CONFIG['min_win_rate']:
                return {
                    'name': 'Opening_Range_Breakout',
                    'type': 'scalping',
                    'confidence': win_rate,
                    'entry_rules': {
                        'condition': 'break_above_30min_high',
                        'confirmation': 'volume_surge',
                        'time_window': '9:30-10:00_ET'
                    },
                    'exit_rules': {
                        'take_profit': self.target_points,
                        'stop_loss': self.stop_points,
                        'trail_after': 3  # Trail stop after 3 points
                    },
                    'statistics': {
                        'occurrences': breakout_opportunities,
                        'win_rate': win_rate,
                        'best_time': '9:45-10:15_ET'
                    }
                }
        
        return None
    
    def find_mean_reversions(self, data: pd.DataFrame) -> Optional[Dict]:
        """Find quick mean reversion patterns"""
        
        if 'BB_Upper' not in data.columns or 'BB_Lower' not in data.columns:
            return None
        
        reversions = 0
        successful = 0
        
        for i in range(20, len(data) - 5):
            # Oversold bounce
            if data['Close'].iloc[i] <= data['BB_Lower'].iloc[i] and data['RSI'].iloc[i] < 30:
                reversions += 1
                
                # Check for 5-point bounce
                future_high = data['High'].iloc[i+1:i+6].max()
                if future_high - data['Close'].iloc[i] >= self.target_points:
                    successful += 1
            
            # Overbought reversal
            elif data['Close'].iloc[i] >= data['BB_Upper'].iloc[i] and data['RSI'].iloc[i] > 70:
                reversions += 1
                
                # Check for 5-point drop
                future_low = data['Low'].iloc[i+1:i+6].min()
                if data['Close'].iloc[i] - future_low >= self.target_points:
                    successful += 1
        
        if reversions > 0:
            win_rate = successful / reversions
            
            if win_rate >= SCALPING_CONFIG['min_win_rate']:
                return {
                    'name': 'Bollinger_Mean_Reversion',
                    'type': 'scalping',
                    'confidence': win_rate,
                    'entry_rules': {
                        'long': 'close_below_BB_lower_and_RSI<30',
                        'short': 'close_above_BB_upper_and_RSI>70',
                        'confirmation': 'reversal_candle'
                    },
                    'exit_rules': {
                        'take_profit': self.target_points,
                        'stop_loss': self.stop_points,
                        'target': 'middle_band'
                    },
                    'statistics': {
                        'occurrences': reversions,
                        'win_rate': win_rate,
                        'avg_reversal_time': '10_minutes'
                    }
                }
        
        return None
    
    def find_momentum_scalps(self, data: pd.DataFrame) -> Optional[Dict]:
        """Find momentum continuation patterns"""
        
        if 'EMA_5' not in data.columns:
            return None
        
        momentum_setups = 0
        successful = 0
        
        for i in range(20, len(data) - 5):
            # Strong momentum up
            if (data['Close'].iloc[i] > data['EMA_5'].iloc[i] and
                data['Close'].iloc[i] > data['Close'].iloc[i-1] and
                data['Volume'].iloc[i] > data['Volume_SMA'].iloc[i] * 1.5):
                
                momentum_setups += 1
                
                # Check continuation
                future_high = data['High'].iloc[i+1:i+4].max()
                if future_high - data['Close'].iloc[i] >= self.target_points:
                    successful += 1
        
        if momentum_setups > 0:
            win_rate = successful / momentum_setups
            
            if win_rate >= SCALPING_CONFIG['min_win_rate']:
                return {
                    'name': 'Momentum_Continuation_Scalp',
                    'type': 'scalping',
                    'confidence': win_rate,
                    'entry_rules': {
                        'condition': 'strong_momentum',
                        'indicators': 'price>EMA5_and_volume>1.5x_avg',
                        'confirmation': 'pullback_to_EMA5'
                    },
                    'exit_rules': {
                        'take_profit': self.target_points,
                        'stop_loss': self.stop_points,
                        'time_stop': '10_minutes'
                    },
                    'statistics': {
                        'occurrences': momentum_setups,
                        'win_rate': win_rate,
                        'best_performance': 'trending_days'
                    }
                }
        
        return None
    
    def find_support_resistance_scalps(self, data: pd.DataFrame) -> Optional[Dict]:
        """Find support/resistance bounce patterns"""
        
        # Identify key levels
        recent_data = data.tail(100)
        
        # Find pivot points
        highs = recent_data['High'].rolling(5).max()
        lows = recent_data['Low'].rolling(5).min()
        
        # Key levels (simplified)
        resistance_levels = highs.nlargest(3).values
        support_levels = lows.nsmallest(3).values
        
        bounces = 0
        successful = 0
        
        for i in range(20, len(data) - 5):
            price = data['Close'].iloc[i]
            
            # Check support bounces
            for support in support_levels:
                if abs(data['Low'].iloc[i] - support) <= 2:  # Within 2 points
                    bounces += 1
                    
                    future_high = data['High'].iloc[i+1:i+6].max()
                    if future_high - support >= self.target_points:
                        successful += 1
                    break
        
        if bounces > 0:
            win_rate = successful / bounces
            
            if win_rate >= SCALPING_CONFIG['min_win_rate']:
                return {
                    'name': 'Support_Resistance_Bounce',
                    'type': 'scalping',
                    'confidence': win_rate,
                    'entry_rules': {
                        'condition': 'touch_key_level',
                        'confirmation': 'rejection_candle',
                        'volume': 'increasing'
                    },
                    'exit_rules': {
                        'take_profit': self.target_points,
                        'stop_loss': self.stop_points,
                        'break_level_stop': True
                    },
                    'statistics': {
                        'occurrences': bounces,
                        'win_rate': win_rate,
                        'strongest_levels': 'round_numbers'
                    }
                }
        
        return None
    
    def rank_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Rank patterns by expected profitability"""
        
        for pattern in patterns:
            # Calculate expected value
            win_rate = pattern.get('statistics', {}).get('win_rate', 0.5)
            
            # Expected profit per trade (accounting for commission)
            gross_profit = self.target_points * 20  # $20 per point
            gross_loss = self.stop_points * 20
            commission = SCALPING_CONFIG['commission_per_trade'] * 2  # Round trip
            
            expected_value = (win_rate * gross_profit) - ((1 - win_rate) * gross_loss) - commission
            
            pattern['expected_value'] = expected_value
            pattern['recommended'] = expected_value > 20  # At least $20 expected profit
        
        # Sort by expected value
        patterns.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        return patterns