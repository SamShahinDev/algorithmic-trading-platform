"""
Pattern Discovery Agent
Discovers new trading patterns in market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from utils.logger import setup_logger
from scipy import signal
from scipy.stats import linregress

class PatternDiscoveryAgent(BaseAgent):
    """
    Discovers trading patterns in market data
    This is your pattern-finding detective
    """
    
    def __init__(self):
        """Initialize pattern discovery agent"""
        super().__init__('PatternDiscovery')
        self.logger = setup_logger('PatternDiscovery')
        
        # Pattern discovery settings
        self.min_pattern_occurrences = 5  # Minimum times pattern must appear
        self.lookback_periods = [20, 50, 100, 200]  # Different timeframes to check
        
        # Discovered patterns
        self.discovered_patterns = []
        
        self.logger.info("🔍 Pattern Discovery Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.logger.info("Initializing pattern discovery systems...")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, data: pd.DataFrame) -> List[Dict]:
        """
        Main execution - discover patterns
        
        Args:
            data: Market data to analyze
        
        Returns:
            List[Dict]: Discovered patterns
        """
        return await self.discover_patterns(data)
    
    async def discover_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Discover trading patterns in market data
        
        Args:
            data: OHLCV market data
        
        Returns:
            List[Dict]: List of discovered patterns
        """
        if data is None or data.empty:
            self.logger.warning("No data available for pattern discovery")
            return []
        
        self.logger.info(f"🔎 Analyzing {len(data)} bars for patterns...")
        
        patterns = []
        
        try:
            # 1. Trend Line Bounce Pattern (your strategy)
            trend_patterns = await self.find_trend_line_bounces(data)
            patterns.extend(trend_patterns)
            
            # 2. Support/Resistance Bounce Patterns
            sr_patterns = await self.find_support_resistance_bounces(data)
            patterns.extend(sr_patterns)
            
            # 3. Moving Average Bounce Patterns
            ma_patterns = await self.find_ma_bounces(data)
            patterns.extend(ma_patterns)
            
            # 4. Volume Spike Reversal Patterns
            volume_patterns = await self.find_volume_reversals(data)
            patterns.extend(volume_patterns)
            
            # 5. Opening Range Breakout Patterns
            orb_patterns = await self.find_opening_range_breakouts(data)
            patterns.extend(orb_patterns)
            
            # 6. Failed Breakout Reversal Patterns
            reversal_patterns = await self.find_failed_breakouts(data)
            patterns.extend(reversal_patterns)
            
            self.logger.info(f"✨ Discovered {len(patterns)} potential patterns")
            
            # Record success
            self.record_success()
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern discovery: {e}")
            self.record_error(e)
            return []
    
    async def find_trend_line_bounces(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find trend line bounce patterns (your main strategy)
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: Trend line bounce patterns
        """
        patterns = []
        
        try:
            for lookback in self.lookback_periods:
                if len(data) < lookback:
                    continue
                
                # Calculate trend line using swing lows
                trend_line = self.calculate_trend_line_from_lows(data, lookback)
                
                if trend_line is None:
                    continue
                
                # Find bounces off the trend line
                bounces = self.find_trend_bounces(data, trend_line, lookback)
                
                if len(bounces) >= self.min_pattern_occurrences:
                    # Calculate pattern statistics
                    win_rate = self.calculate_bounce_success_rate(data, bounces)
                    
                    if win_rate > 0.5:  # Only keep profitable patterns
                        pattern = {
                            'name': f'Trend Line Bounce ({lookback} bars)',
                            'type': 'trend_bounce',
                            'entry_conditions': {
                                'trend_line_touch': True,
                                'trend_line_slope': trend_line['slope'],
                                'lookback_period': lookback,
                                'min_touches': 3
                            },
                            'exit_conditions': {
                                'target_type': 'resistance',
                                'stop_type': 'below_trend_line',
                                'risk_reward': 2.0
                            },
                            'filters': {
                                'min_trend_angle': 15,  # degrees
                                'max_trend_angle': 45,
                                'volume_confirmation': True,
                                'rsi_oversold': 40
                            },
                            'statistics': {
                                'occurrences': len(bounces),
                                'preliminary_win_rate': win_rate
                            },
                            'confidence': win_rate * 0.5 + (len(bounces) / 100) * 0.5
                        }
                        
                        patterns.append(pattern)
                        self.logger.info(f"  📈 Found trend bounce pattern: {pattern['name']}")
        
        except Exception as e:
            self.logger.error(f"Error finding trend line bounces: {e}")
        
        return patterns
    
    def calculate_trend_line_from_lows(self, data: pd.DataFrame, lookback: int) -> Optional[Dict]:
        """
        Calculate ascending trend line from swing lows
        
        Args:
            data: Market data
            lookback: Number of bars to analyze
        
        Returns:
            Optional[Dict]: Trend line parameters
        """
        try:
            recent_data = data.tail(lookback)
            
            # Find local minima (swing lows)
            lows = recent_data['Low'].values
            
            # Find peaks (inverted for minima)
            distance = max(5, lookback // 20)  # Dynamic distance based on lookback
            peaks, _ = signal.find_peaks(-lows, distance=distance)
            
            if len(peaks) < 2:
                return None
            
            # Get swing low points
            swing_lows = [(i, lows[i]) for i in peaks]
            
            # Fit line through swing lows
            x_points = [p[0] for p in swing_lows]
            y_points = [p[1] for p in swing_lows]
            
            if len(x_points) < 2:
                return None
            
            # Linear regression
            slope, intercept, r_value, _, _ = linregress(x_points, y_points)
            
            # Only keep ascending trend lines with good fit
            if slope > 0 and abs(r_value) > 0.7:
                return {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'points': swing_lows,
                    'start_index': recent_data.index[0],
                    'end_index': recent_data.index[-1]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating trend line: {e}")
            return None
    
    def find_trend_bounces(self, data: pd.DataFrame, trend_line: Dict, lookback: int) -> List[Dict]:
        """
        Find where price bounced off trend line
        
        Args:
            data: Market data
            trend_line: Trend line parameters
            lookback: Lookback period
        
        Returns:
            List[Dict]: Bounce points
        """
        bounces = []
        
        try:
            recent_data = data.tail(lookback)
            
            for i in range(20, len(recent_data) - 5):  # Need future data to confirm bounce
                # Calculate trend line value at this point
                trend_value = trend_line['intercept'] + trend_line['slope'] * i
                
                # Check if low came within 0.2% of trend line
                low = recent_data.iloc[i]['Low']
                distance_pct = abs(low - trend_value) / trend_value
                
                if distance_pct < 0.002:  # Within 0.2% of trend line
                    # Check if price bounced up after
                    future_high = recent_data.iloc[i+1:i+6]['High'].max()
                    bounce_pct = (future_high - low) / low
                    
                    if bounce_pct > 0.002:  # At least 0.2% bounce
                        bounces.append({
                            'index': i,
                            'date': recent_data.index[i],
                            'trend_value': trend_value,
                            'low': low,
                            'bounce_size': bounce_pct
                        })
        
        except Exception as e:
            self.logger.error(f"Error finding trend bounces: {e}")
        
        return bounces
    
    def calculate_bounce_success_rate(self, data: pd.DataFrame, bounces: List[Dict]) -> float:
        """
        Calculate success rate of bounce patterns
        
        Args:
            data: Market data
            bounces: List of bounce points
        
        Returns:
            float: Success rate (0-1)
        """
        if not bounces:
            return 0
        
        successful = 0
        
        for bounce in bounces:
            # Define success as price moving up at least 0.5% after bounce
            if bounce.get('bounce_size', 0) > 0.005:
                successful += 1
        
        return successful / len(bounces) if bounces else 0
    
    async def find_support_resistance_bounces(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find support and resistance bounce patterns
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: S/R bounce patterns
        """
        patterns = []
        
        try:
            # Find key S/R levels
            levels = self.find_key_levels(data)
            
            for level in levels:
                # Find bounces off this level
                bounces = self.find_level_bounces(data, level)
                
                if len(bounces) >= 3:  # Minimum 3 bounces to validate level
                    pattern = {
                        'name': f'S/R Bounce at {level:.2f}',
                        'type': 'sr_bounce',
                        'entry_conditions': {
                            'level': level,
                            'tolerance': 5,  # Points tolerance
                            'min_touches': 3
                        },
                        'exit_conditions': {
                            'target': 'next_resistance' if level < data['Close'].iloc[-1] else 'next_support',
                            'stop': level - 10 if level < data['Close'].iloc[-1] else level + 10
                        },
                        'statistics': {
                            'touches': len(bounces),
                            'strength': len(bounces) / 10  # Normalize strength
                        }
                    }
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error finding S/R bounces: {e}")
        
        return patterns
    
    def find_key_levels(self, data: pd.DataFrame, min_touches: int = 2) -> List[float]:
        """
        Find key support/resistance levels
        
        Args:
            data: Market data
            min_touches: Minimum touches to qualify as key level
        
        Returns:
            List[float]: Key price levels
        """
        levels = []
        
        try:
            # Use recent 200 bars
            recent_data = data.tail(200)
            
            # Find local maxima and minima
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            
            # Find peaks
            high_peaks, _ = signal.find_peaks(highs, distance=10)
            low_peaks, _ = signal.find_peaks(-lows, distance=10)
            
            # Collect all significant levels
            significant_prices = []
            
            for idx in high_peaks:
                significant_prices.append(highs[idx])
            
            for idx in low_peaks:
                significant_prices.append(lows[idx])
            
            # Cluster nearby levels
            if significant_prices:
                significant_prices = sorted(significant_prices)
                clustered = []
                current_cluster = [significant_prices[0]]
                
                for price in significant_prices[1:]:
                    if price - current_cluster[-1] < 10:  # Within 10 points
                        current_cluster.append(price)
                    else:
                        if len(current_cluster) >= min_touches:
                            clustered.append(np.mean(current_cluster))
                        current_cluster = [price]
                
                if len(current_cluster) >= min_touches:
                    clustered.append(np.mean(current_cluster))
                
                levels = clustered[:10]  # Top 10 levels
        
        except Exception as e:
            self.logger.error(f"Error finding key levels: {e}")
        
        return levels
    
    def find_level_bounces(self, data: pd.DataFrame, level: float, tolerance: float = 10) -> List[Dict]:
        """
        Find bounces off a specific level
        
        Args:
            data: Market data
            level: Price level
            tolerance: Tolerance in points
        
        Returns:
            List[Dict]: Bounces off the level
        """
        bounces = []
        
        try:
            recent_data = data.tail(200)
            
            for i in range(len(recent_data) - 5):
                low = recent_data.iloc[i]['Low']
                high = recent_data.iloc[i]['High']
                
                # Check if price touched the level
                if abs(low - level) < tolerance or abs(high - level) < tolerance:
                    # Check for reversal
                    next_bars = recent_data.iloc[i+1:i+6]
                    
                    if low < level < high:  # Level is within bar
                        continue
                    
                    if low > level:  # Potential support bounce
                        if next_bars['High'].max() > high * 1.002:
                            bounces.append({
                                'index': i,
                                'type': 'support',
                                'level': level,
                                'bounce_from': low
                            })
                    else:  # Potential resistance bounce
                        if next_bars['Low'].min() < low * 0.998:
                            bounces.append({
                                'index': i,
                                'type': 'resistance',
                                'level': level,
                                'bounce_from': high
                            })
        
        except Exception as e:
            self.logger.error(f"Error finding level bounces: {e}")
        
        return bounces
    
    async def find_ma_bounces(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find moving average bounce patterns
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: MA bounce patterns
        """
        patterns = []
        
        try:
            # Check different MA periods
            ma_periods = [20, 50, 200]
            
            for period in ma_periods:
                if f'SMA_{period}' not in data.columns:
                    continue
                
                bounces = []
                
                for i in range(period, len(data) - 5):
                    ma_value = data[f'SMA_{period}'].iloc[i]
                    low = data['Low'].iloc[i]
                    
                    # Check if price touched MA
                    if abs(low - ma_value) / ma_value < 0.002:  # Within 0.2%
                        # Check for bounce
                        future_high = data['High'].iloc[i+1:i+6].max()
                        if (future_high - low) / low > 0.003:  # 0.3% bounce
                            bounces.append({
                                'index': i,
                                'ma_value': ma_value,
                                'bounce_size': (future_high - low) / low
                            })
                
                if len(bounces) >= 5:
                    pattern = {
                        'name': f'MA{period} Bounce',
                        'type': 'ma_bounce',
                        'entry_conditions': {
                            'ma_period': period,
                            'touch_tolerance': 0.002,
                            'trend_filter': 'above_ma'
                        },
                        'statistics': {
                            'bounces': len(bounces),
                            'avg_bounce': np.mean([b['bounce_size'] for b in bounces])
                        }
                    }
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error finding MA bounces: {e}")
        
        return patterns
    
    async def find_volume_reversals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find volume spike reversal patterns
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: Volume reversal patterns
        """
        patterns = []
        
        try:
            if 'Volume' not in data.columns or 'Volume_MA' not in data.columns:
                return patterns
            
            reversals = []
            
            for i in range(20, len(data) - 5):
                volume = data['Volume'].iloc[i]
                volume_ma = data['Volume_MA'].iloc[i]
                
                # Check for volume spike (1.5x average)
                if volume > volume_ma * 1.5:
                    # Check if this was a reversal point
                    prev_trend = data['Close'].iloc[i-5:i].pct_change().mean()
                    future_trend = data['Close'].iloc[i:i+5].pct_change().mean()
                    
                    # Reversal if trend changes direction
                    if prev_trend * future_trend < 0:
                        reversals.append({
                            'index': i,
                            'volume_ratio': volume / volume_ma,
                            'reversal_strength': abs(future_trend)
                        })
            
            if len(reversals) >= 5:
                pattern = {
                    'name': 'Volume Spike Reversal',
                    'type': 'volume_reversal',
                    'entry_conditions': {
                        'volume_spike': 1.5,
                        'confirm_reversal': True
                    },
                    'statistics': {
                        'occurrences': len(reversals),
                        'avg_volume_ratio': np.mean([r['volume_ratio'] for r in reversals])
                    }
                }
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error finding volume reversals: {e}")
        
        return patterns
    
    async def find_opening_range_breakouts(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find opening range breakout patterns
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: ORB patterns
        """
        patterns = []
        
        try:
            # For futures, we'll use the first hour as opening range
            # This is simplified - real implementation would use actual session times
            
            breakouts = []
            
            # Group by date (simplified)
            for i in range(0, len(data) - 30, 24):  # Assuming hourly data, 24 bars per day
                if i + 30 > len(data):
                    break
                
                # First hour range
                opening_range_high = data['High'].iloc[i:i+1].max()
                opening_range_low = data['Low'].iloc[i:i+1].min()
                
                # Check for breakout in next hours
                for j in range(i+1, min(i+8, len(data))):
                    if data['High'].iloc[j] > opening_range_high * 1.001:
                        # Upside breakout
                        breakouts.append({
                            'type': 'long',
                            'range_size': (opening_range_high - opening_range_low) / opening_range_low
                        })
                        break
                    elif data['Low'].iloc[j] < opening_range_low * 0.999:
                        # Downside breakout
                        breakouts.append({
                            'type': 'short',
                            'range_size': (opening_range_high - opening_range_low) / opening_range_low
                        })
                        break
            
            if len(breakouts) >= 5:
                pattern = {
                    'name': 'Opening Range Breakout',
                    'type': 'orb',
                    'entry_conditions': {
                        'range_period': 60,  # minutes
                        'breakout_buffer': 0.001
                    },
                    'statistics': {
                        'occurrences': len(breakouts)
                    }
                }
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error finding ORB patterns: {e}")
        
        return patterns
    
    async def find_failed_breakouts(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find failed breakout reversal patterns
        
        Args:
            data: Market data
        
        Returns:
            List[Dict]: Failed breakout patterns
        """
        patterns = []
        
        try:
            # Find recent highs/lows that were broken but failed
            failed_breakouts = []
            
            for i in range(20, len(data) - 10):
                # Check for recent high
                recent_high = data['High'].iloc[i-20:i].max()
                current_high = data['High'].iloc[i]
                
                if current_high > recent_high * 1.001:  # Breakout attempt
                    # Check if it failed (price came back down)
                    future_lows = data['Low'].iloc[i+1:i+6]
                    if future_lows.min() < recent_high:
                        failed_breakouts.append({
                            'type': 'failed_resistance',
                            'level': recent_high,
                            'failure_depth': (recent_high - future_lows.min()) / recent_high
                        })
            
            if len(failed_breakouts) >= 5:
                pattern = {
                    'name': 'Failed Breakout Reversal',
                    'type': 'failed_breakout',
                    'entry_conditions': {
                        'breakout_threshold': 0.001,
                        'failure_confirmation': 'close_below_level'
                    },
                    'statistics': {
                        'occurrences': len(failed_breakouts)
                    }
                }
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error finding failed breakouts: {e}")
        
        return patterns