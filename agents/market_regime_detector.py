"""
Market Regime Detection Module
Identifies current market conditions (trending, ranging, volatile) for pattern optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from agents.base_agent import BaseAgent
from utils.logger import setup_logger
from utils.slack_notifier import slack_notifier

class MarketRegimeDetector(BaseAgent):
    """
    Detects and classifies market regimes for better pattern performance
    """
    
    def __init__(self):
        """Initialize market regime detector"""
        super().__init__('MarketRegimeDetector')
        self.logger = setup_logger('MarketRegimeDetector')
        
        # Regime definitions
        self.regimes = {
            'trending_up': {'id': 0, 'color': '🟢', 'description': 'Strong uptrend'},
            'trending_down': {'id': 1, 'color': '🔴', 'description': 'Strong downtrend'},
            'ranging': {'id': 2, 'color': '🟡', 'description': 'Sideways/Range-bound'},
            'volatile': {'id': 3, 'color': '🟣', 'description': 'High volatility'},
            'quiet': {'id': 4, 'color': '⚪', 'description': 'Low volatility'}
        }
        
        # Detection parameters
        self.lookback_period = 20
        self.volatility_threshold = 1.5  # ATR multiplier
        self.trend_threshold = 0.002  # 0.2% per day trend
        
        # Cache
        self.current_regime = None
        self.regime_history = []
        
        self.logger.info("📊 Market Regime Detector initialized")
    
    async def initialize(self) -> bool:
        """Initialize the detector"""
        try:
            self.logger.info("Market regime detection ready")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime
        
        Args:
            data: Market data
        
        Returns:
            Dict: Current regime and statistics
        """
        return self.detect_regime(data)
    
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect the current market regime
        
        Args:
            data: Recent market data (at least 50 bars)
        
        Returns:
            Dict: Regime classification and metrics
        """
        if len(data) < 50:
            self.logger.warning("Insufficient data for regime detection")
            return {'regime': 'unknown', 'confidence': 0}
        
        try:
            # Calculate regime indicators
            indicators = self.calculate_regime_indicators(data)
            
            # Classify regime
            regime = self.classify_regime(indicators)
            
            # Calculate regime strength/confidence
            confidence = self.calculate_regime_confidence(indicators, regime)
            
            # Store in history
            self.current_regime = regime
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': confidence,
                'indicators': indicators
            })
            
            # Log regime change
            if len(self.regime_history) > 1:
                if self.regime_history[-2]['regime'] != regime:
                    self.logger.info(f"🔄 Regime change: {self.regime_history[-2]['regime']} → {regime}")
                    
                    # Send Slack notification for regime change
                    import asyncio
                    asyncio.create_task(slack_notifier.regime_change(
                        self.regime_history[-2]['regime'],
                        regime,
                        confidence
                    ))
            
            return {
                'regime': regime,
                'confidence': confidence,
                'indicators': indicators,
                'description': self.regimes[regime]['description'],
                'color': self.regimes[regime]['color']
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return {'regime': 'unknown', 'confidence': 0}
    
    def calculate_regime_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate various indicators for regime detection
        
        Args:
            data: Market data
        
        Returns:
            Dict: Calculated indicators
        """
        indicators = {}
        
        # 1. Trend Indicators
        indicators['trend_strength'] = self.calculate_trend_strength(data)
        indicators['trend_direction'] = self.calculate_trend_direction(data)
        indicators['trend_consistency'] = self.calculate_trend_consistency(data)
        
        # 2. Volatility Indicators
        indicators['volatility'] = self.calculate_volatility(data)
        indicators['volatility_regime'] = self.classify_volatility(indicators['volatility'])
        indicators['volatility_expanding'] = self.is_volatility_expanding(data)
        
        # 3. Range Indicators
        indicators['range_bound'] = self.calculate_range_percentage(data)
        indicators['breakout_potential'] = self.calculate_breakout_potential(data)
        
        # 4. Momentum Indicators
        indicators['momentum'] = self.calculate_momentum(data)
        indicators['momentum_divergence'] = self.detect_momentum_divergence(data)
        
        # 5. Volume Indicators (if available)
        if 'Volume' in data.columns:
            indicators['volume_trend'] = self.calculate_volume_trend(data)
            indicators['volume_confirmation'] = self.check_volume_confirmation(data)
        
        # 6. Market Structure
        indicators['higher_highs'] = self.count_higher_highs(data)
        indicators['lower_lows'] = self.count_lower_lows(data)
        indicators['swing_count'] = self.count_swings(data)
        
        # 7. Statistical Properties
        indicators['skewness'] = self.calculate_return_skewness(data)
        indicators['kurtosis'] = self.calculate_return_kurtosis(data)
        indicators['autocorrelation'] = self.calculate_autocorrelation(data)
        
        return indicators
    
    def classify_regime(self, indicators: Dict) -> str:
        """
        Classify market regime based on indicators
        
        Args:
            indicators: Calculated indicators
        
        Returns:
            str: Regime classification
        """
        # Rule-based classification
        trend_strength = indicators['trend_strength']
        trend_direction = indicators['trend_direction']
        volatility_regime = indicators['volatility_regime']
        range_bound = indicators['range_bound']
        
        # Strong trending market
        if abs(trend_strength) > 0.7:
            if trend_direction > 0:
                return 'trending_up'
            else:
                return 'trending_down'
        
        # High volatility market
        if volatility_regime == 'high':
            if abs(trend_strength) < 0.3:
                return 'volatile'
            elif trend_direction > 0:
                return 'trending_up'
            else:
                return 'trending_down'
        
        # Range-bound market
        if range_bound > 0.7:
            return 'ranging'
        
        # Low volatility/quiet market
        if volatility_regime == 'low':
            return 'quiet'
        
        # Moderate trending
        if abs(trend_strength) > 0.3:
            if trend_direction > 0:
                return 'trending_up'
            else:
                return 'trending_down'
        
        # Default to ranging
        return 'ranging'
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength using linear regression
        
        Returns:
            float: Trend strength (-1 to 1)
        """
        closes = data['Close'].values[-self.lookback_period:]
        x = np.arange(len(closes))
        
        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x, closes)
        
        # Normalize slope
        normalized_slope = slope / np.mean(closes)
        
        # Use R-squared as strength measure
        trend_strength = r_value ** 2 * np.sign(slope)
        
        return np.clip(trend_strength, -1, 1)
    
    def calculate_trend_direction(self, data: pd.DataFrame) -> float:
        """
        Calculate trend direction
        
        Returns:
            float: Direction (-1 for down, 1 for up)
        """
        # Simple moving average comparison
        sma_short = data['Close'].rolling(10).mean().iloc[-1]
        sma_long = data['Close'].rolling(30).mean().iloc[-1]
        
        if pd.isna(sma_short) or pd.isna(sma_long):
            return 0
        
        return 1 if sma_short > sma_long else -1
    
    def calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """
        Calculate how consistent the trend is
        
        Returns:
            float: Consistency score (0 to 1)
        """
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 2:
            return 0
        
        # Count consecutive same-direction moves
        consecutive = 0
        max_consecutive = 0
        prev_sign = 0
        
        for ret in returns:
            curr_sign = np.sign(ret)
            if curr_sign == prev_sign and curr_sign != 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
            prev_sign = curr_sign
        
        return min(max_consecutive / len(returns), 1)
    
    def calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate current volatility level
        
        Returns:
            float: Volatility measure
        """
        if 'ATR' in data.columns:
            current_atr = data['ATR'].iloc[-1]
            avg_atr = data['ATR'].rolling(20).mean().iloc[-1]
            return current_atr / avg_atr if avg_atr > 0 else 1
        else:
            # Calculate using returns
            returns = data['Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)
    
    def classify_volatility(self, volatility: float) -> str:
        """
        Classify volatility level
        
        Args:
            volatility: Volatility measure
        
        Returns:
            str: 'low', 'normal', or 'high'
        """
        if volatility < 0.7:
            return 'low'
        elif volatility > 1.3:
            return 'high'
        else:
            return 'normal'
    
    def is_volatility_expanding(self, data: pd.DataFrame) -> bool:
        """
        Check if volatility is expanding
        
        Returns:
            bool: True if volatility is increasing
        """
        if 'ATR' not in data.columns:
            return False
        
        recent_atr = data['ATR'].iloc[-5:].mean()
        previous_atr = data['ATR'].iloc[-10:-5].mean()
        
        return recent_atr > previous_atr * 1.1
    
    def calculate_range_percentage(self, data: pd.DataFrame) -> float:
        """
        Calculate percentage of time spent in range
        
        Returns:
            float: Range-bound percentage (0 to 1)
        """
        highs = data['High'].iloc[-self.lookback_period:]
        lows = data['Low'].iloc[-self.lookback_period:]
        
        range_high = highs.max()
        range_low = lows.min()
        range_size = range_high - range_low
        
        if range_size == 0:
            return 1
        
        # Count bars within middle 60% of range
        middle_high = range_low + range_size * 0.7
        middle_low = range_low + range_size * 0.3
        
        in_range = 0
        for i in range(len(data) - self.lookback_period, len(data)):
            if middle_low <= data['Close'].iloc[i] <= middle_high:
                in_range += 1
        
        return in_range / self.lookback_period
    
    def calculate_breakout_potential(self, data: pd.DataFrame) -> float:
        """
        Calculate potential for breakout
        
        Returns:
            float: Breakout potential (0 to 1)
        """
        # Bollinger Band squeeze
        if 'Close' not in data.columns:
            return 0
        
        sma = data['Close'].rolling(20).mean()
        std = data['Close'].rolling(20).std()
        
        if len(sma) < 20 or pd.isna(std.iloc[-1]):
            return 0
        
        # Current band width
        current_width = (std.iloc[-1] * 2) / sma.iloc[-1] if sma.iloc[-1] > 0 else 0
        
        # Historical band width
        historical_widths = (std * 2) / sma
        avg_width = historical_widths.mean()
        
        if avg_width == 0:
            return 0
        
        # Squeeze = low current width relative to average
        squeeze = 1 - (current_width / avg_width)
        
        return np.clip(squeeze, 0, 1)
    
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        Calculate market momentum
        
        Returns:
            float: Momentum (-1 to 1)
        """
        if 'RSI' in data.columns:
            current_rsi = data['RSI'].iloc[-1]
            return (current_rsi - 50) / 50
        else:
            # Rate of change
            roc = (data['Close'].iloc[-1] / data['Close'].iloc[-self.lookback_period] - 1)
            return np.clip(roc * 10, -1, 1)
    
    def detect_momentum_divergence(self, data: pd.DataFrame) -> bool:
        """
        Detect price/momentum divergence
        
        Returns:
            bool: True if divergence detected
        """
        if 'RSI' not in data.columns:
            return False
        
        # Price trend
        price_trend = np.sign(data['Close'].iloc[-1] - data['Close'].iloc[-10])
        
        # RSI trend
        rsi_trend = np.sign(data['RSI'].iloc[-1] - data['RSI'].iloc[-10])
        
        # Divergence if opposite signs
        return price_trend != rsi_trend and price_trend != 0 and rsi_trend != 0
    
    def calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """
        Calculate volume trend
        
        Returns:
            float: Volume trend (-1 to 1)
        """
        if 'Volume' not in data.columns:
            return 0
        
        recent_volume = data['Volume'].iloc[-5:].mean()
        previous_volume = data['Volume'].iloc[-10:-5].mean()
        
        if previous_volume == 0:
            return 0
        
        return np.clip((recent_volume / previous_volume - 1), -1, 1)
    
    def check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """
        Check if volume confirms price movement
        
        Returns:
            bool: True if volume confirms trend
        """
        if 'Volume' not in data.columns:
            return True  # Assume confirmation if no volume data
        
        # Price change
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-5]
        
        # Volume change
        recent_volume = data['Volume'].iloc[-5:].mean()
        previous_volume = data['Volume'].iloc[-10:-5].mean()
        
        # Confirmation: up move with higher volume, down move with lower volume
        if price_change > 0:
            return recent_volume > previous_volume
        else:
            return recent_volume <= previous_volume
    
    def count_higher_highs(self, data: pd.DataFrame) -> int:
        """Count number of higher highs"""
        highs = data['High'].iloc[-self.lookback_period:]
        count = 0
        prev_high = highs.iloc[0]
        
        for high in highs[1:]:
            if high > prev_high:
                count += 1
            prev_high = high
        
        return count
    
    def count_lower_lows(self, data: pd.DataFrame) -> int:
        """Count number of lower lows"""
        lows = data['Low'].iloc[-self.lookback_period:]
        count = 0
        prev_low = lows.iloc[0]
        
        for low in lows[1:]:
            if low < prev_low:
                count += 1
            prev_low = low
        
        return count
    
    def count_swings(self, data: pd.DataFrame) -> int:
        """Count number of price swings"""
        closes = data['Close'].iloc[-self.lookback_period:]
        
        # Detect turning points
        swings = 0
        for i in range(1, len(closes) - 1):
            # Local maximum
            if closes.iloc[i] > closes.iloc[i-1] and closes.iloc[i] > closes.iloc[i+1]:
                swings += 1
            # Local minimum
            elif closes.iloc[i] < closes.iloc[i-1] and closes.iloc[i] < closes.iloc[i+1]:
                swings += 1
        
        return swings
    
    def calculate_return_skewness(self, data: pd.DataFrame) -> float:
        """Calculate skewness of returns"""
        returns = data['Close'].pct_change().dropna()
        return stats.skew(returns) if len(returns) > 2 else 0
    
    def calculate_return_kurtosis(self, data: pd.DataFrame) -> float:
        """Calculate kurtosis of returns (fat tails)"""
        returns = data['Close'].pct_change().dropna()
        return stats.kurtosis(returns) if len(returns) > 3 else 0
    
    def calculate_autocorrelation(self, data: pd.DataFrame) -> float:
        """Calculate autocorrelation of returns"""
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 2:
            return 0
        
        return returns.autocorr(lag=1)
    
    def calculate_regime_confidence(self, indicators: Dict, regime: str) -> float:
        """
        Calculate confidence in regime classification
        
        Args:
            indicators: Calculated indicators
            regime: Classified regime
        
        Returns:
            float: Confidence score (0 to 1)
        """
        confidence = 0
        factors = 0
        
        # Check indicator alignment with regime
        if regime == 'trending_up':
            if indicators['trend_direction'] > 0:
                confidence += 1
                factors += 1
            if indicators['trend_strength'] > 0.5:
                confidence += 1
                factors += 1
            if indicators['higher_highs'] > indicators['lower_lows']:
                confidence += 1
                factors += 1
            if indicators.get('volume_confirmation', False):
                confidence += 1
                factors += 1
                
        elif regime == 'trending_down':
            if indicators['trend_direction'] < 0:
                confidence += 1
                factors += 1
            if indicators['trend_strength'] < -0.5:
                confidence += 1
                factors += 1
            if indicators['lower_lows'] > indicators['higher_highs']:
                confidence += 1
                factors += 1
                
        elif regime == 'ranging':
            if indicators['range_bound'] > 0.6:
                confidence += 1
                factors += 1
            if abs(indicators['trend_strength']) < 0.3:
                confidence += 1
                factors += 1
            if indicators['swing_count'] > 3:
                confidence += 1
                factors += 1
                
        elif regime == 'volatile':
            if indicators['volatility_regime'] == 'high':
                confidence += 1
                factors += 1
            if indicators['volatility_expanding']:
                confidence += 1
                factors += 1
            if abs(indicators['kurtosis']) > 1:
                confidence += 1
                factors += 1
        
        return confidence / factors if factors > 0 else 0.5
    
    def get_regime_recommendations(self, regime: str) -> Dict:
        """
        Get trading recommendations for current regime
        
        Args:
            regime: Current market regime
        
        Returns:
            Dict: Trading recommendations
        """
        recommendations = {
            'trending_up': {
                'preferred_patterns': ['trend_bounce', 'ma_bounce', 'breakout'],
                'avoid_patterns': ['mean_reversion', 'fade'],
                'position_sizing': 'normal',
                'stop_placement': 'wider',
                'target_placement': 'extended',
                'notes': 'Favor momentum strategies, trail stops'
            },
            'trending_down': {
                'preferred_patterns': ['failed_breakout', 'resistance_short'],
                'avoid_patterns': ['trend_bounce', 'support_bounce'],
                'position_sizing': 'reduced',
                'stop_placement': 'tight',
                'target_placement': 'conservative',
                'notes': 'Be cautious with longs, consider shorts'
            },
            'ranging': {
                'preferred_patterns': ['sr_bounce', 'range_fade'],
                'avoid_patterns': ['breakout', 'trend_following'],
                'position_sizing': 'normal',
                'stop_placement': 'outside_range',
                'target_placement': 'opposite_range',
                'notes': 'Buy support, sell resistance'
            },
            'volatile': {
                'preferred_patterns': ['volatility_breakout', 'momentum'],
                'avoid_patterns': ['tight_stops', 'mean_reversion'],
                'position_sizing': 'reduced',
                'stop_placement': 'very_wide',
                'target_placement': 'aggressive',
                'notes': 'Reduce size, widen stops, quick profits'
            },
            'quiet': {
                'preferred_patterns': ['breakout_anticipation'],
                'avoid_patterns': ['momentum', 'volatility_based'],
                'position_sizing': 'reduced',
                'stop_placement': 'tight',
                'target_placement': 'modest',
                'notes': 'Wait for volatility expansion'
            }
        }
        
        return recommendations.get(regime, {
            'preferred_patterns': [],
            'avoid_patterns': [],
            'position_sizing': 'minimal',
            'notes': 'Unknown regime - trade cautiously'
        })
    
    def detect_regime_change(self, data: pd.DataFrame) -> bool:
        """
        Detect if regime is changing
        
        Args:
            data: Market data
        
        Returns:
            bool: True if regime change detected
        """
        if len(self.regime_history) < 2:
            return False
        
        current = self.detect_regime(data)
        previous = self.regime_history[-2]['regime']
        
        return current['regime'] != previous