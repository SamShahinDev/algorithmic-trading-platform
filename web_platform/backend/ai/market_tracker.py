"""
Market Tracker - Market conditions monitoring
Tracks volatility, trends, and support/resistance levels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
# import yfinance as yf  # DISABLED - Using TopStepX data only

class MarketTracker:
    """Tracks and analyzes market conditions"""
    
    def __init__(self):
        """Initialize market tracker"""
        self.current_conditions = {}
        self.last_update = None
        self.update_interval = 300  # 5 minutes
        
    async def calculate_volatility(self) -> Dict:
        """
        Calculate market volatility metrics
        
        Returns:
            Dict with volatility indicators (ATR, standard deviation, etc.)
        """
        try:
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="1mo", interval="1h")
            
            if data.empty:
                return {'atr': 0, 'std_dev': 0, 'volatility_level': 'unknown'}
            
            # Calculate ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate standard deviation
            std_dev = data['Close'].rolling(window=20).std().iloc[-1]
            
            # Determine volatility level
            recent_price = data['Close'].iloc[-1]
            volatility_percentage = (atr / recent_price) * 100
            
            if volatility_percentage < 0.5:
                volatility_level = 'very_low'
            elif volatility_percentage < 1.0:
                volatility_level = 'low'
            elif volatility_percentage < 2.0:
                volatility_level = 'medium'
            elif volatility_percentage < 3.0:
                volatility_level = 'high'
            else:
                volatility_level = 'very_high'
            
            return {
                'atr': float(atr),
                'std_dev': float(std_dev),
                'volatility_percentage': float(volatility_percentage),
                'volatility_level': volatility_level
            }
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return {'atr': 0, 'std_dev': 0, 'volatility_level': 'unknown'}
    
    def identify_trend(self) -> Dict:
        """
        Identify current market trend
        
        Returns:
            Dict with trend indicators (direction, strength, etc.)
        """
        try:
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="1mo", interval="1d")
            
            if data.empty or len(data) < 20:
                return {'direction': 'unknown', 'strength': 0}
            
            closes = data['Close'].values
            
            # Calculate moving averages
            sma_5 = np.mean(closes[-5:])
            sma_10 = np.mean(closes[-10:])
            sma_20 = np.mean(closes[-20:])
            
            # Calculate ADX for trend strength (simplified)
            high = data['High'].values
            low = data['Low'].values
            close = closes
            
            # Directional indicators
            plus_dm = np.diff(high)
            minus_dm = -np.diff(low)
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Trend determination
            if sma_5 > sma_10 > sma_20:
                direction = 'strong_up'
                strength = min(100, ((sma_5 / sma_20 - 1) * 100) * 10)
            elif sma_5 > sma_20:
                direction = 'up'
                strength = min(100, ((sma_5 / sma_20 - 1) * 100) * 5)
            elif sma_5 < sma_10 < sma_20:
                direction = 'strong_down'
                strength = min(100, ((sma_20 / sma_5 - 1) * 100) * 10)
            elif sma_5 < sma_20:
                direction = 'down'
                strength = min(100, ((sma_20 / sma_5 - 1) * 100) * 5)
            else:
                direction = 'ranging'
                strength = 20  # Low trend strength for ranging
            
            # Support and resistance from recent highs/lows
            recent_high = max(closes[-10:])
            recent_low = min(closes[-10:])
            current_price = closes[-1]
            
            return {
                'direction': direction,
                'strength': float(strength),
                'sma_5': float(sma_5),
                'sma_10': float(sma_10),
                'sma_20': float(sma_20),
                'recent_high': float(recent_high),
                'recent_low': float(recent_low),
                'current_price': float(current_price)
            }
            
        except Exception as e:
            print(f"Error identifying trend: {e}")
            return {'direction': 'unknown', 'strength': 0}
    
    async def analyze_volume_profile(self) -> Dict:
        """
        Analyze volume profile at different price levels
        
        Returns:
            Dict with volume profile analysis
        """
        try:
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="5d", interval="1h")
            
            if data.empty:
                return {'volume_trend': 'unknown', 'high_volume_nodes': []}
            
            # Calculate volume-weighted average price (VWAP)
            vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
            
            # Identify high volume nodes (price levels with high volume)
            price_bins = pd.cut(data['Close'], bins=20)
            volume_profile = data.groupby(price_bins)['Volume'].sum().sort_values(ascending=False)
            
            high_volume_nodes = []
            if len(volume_profile) > 0:
                top_nodes = volume_profile.head(3)
                for interval in top_nodes.index:
                    high_volume_nodes.append({
                        'price_range': f"{interval.left:.2f}-{interval.right:.2f}",
                        'volume': float(top_nodes[interval])
                    })
            
            # Determine volume trend
            recent_volume = data['Volume'].iloc[-24:].mean()  # Last 24 hours
            older_volume = data['Volume'].iloc[-48:-24].mean() if len(data) > 48 else recent_volume
            
            if recent_volume > older_volume * 1.2:
                volume_trend = 'increasing'
            elif recent_volume < older_volume * 0.8:
                volume_trend = 'decreasing'
            else:
                volume_trend = 'stable'
            
            return {
                'volume_trend': volume_trend,
                'current_vwap': float(vwap.iloc[-1]) if len(vwap) > 0 else 0,
                'high_volume_nodes': high_volume_nodes,
                'recent_volume': float(recent_volume),
                'volume_ratio': float(recent_volume / older_volume) if older_volume > 0 else 1
            }
            
        except Exception as e:
            print(f"Error analyzing volume profile: {e}")
            return {'volume_trend': 'unknown', 'high_volume_nodes': []}
    
    async def update_support_resistance(self) -> Dict:
        """
        Update dynamic support and resistance levels
        
        Returns:
            Dict with current S/R levels
        """
        try:
            ticker = yf.Ticker("NQ=F")
            data = ticker.history(period="1mo", interval="1h")
            
            if data.empty:
                return {'support': [], 'resistance': []}
            
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            current_price = closes[-1]
            
            # Find swing points
            support_levels = []
            resistance_levels = []
            
            # Look for local minima and maxima
            for i in range(10, len(closes) - 10):
                # Check for swing low (support)
                if lows[i] == min(lows[i-10:i+10]):
                    support_levels.append(float(lows[i]))
                
                # Check for swing high (resistance)
                if highs[i] == max(highs[i-10:i+10]):
                    resistance_levels.append(float(highs[i]))
            
            # Cluster nearby levels
            support_levels = self._cluster_levels(support_levels)
            resistance_levels = self._cluster_levels(resistance_levels)
            
            # Filter to relevant levels (within 2% of current price)
            price_range = current_price * 0.02
            
            relevant_support = [s for s in support_levels if s < current_price and s > current_price - price_range]
            relevant_resistance = [r for r in resistance_levels if r > current_price and r < current_price + price_range]
            
            # Sort by proximity to current price
            relevant_support.sort(reverse=True)  # Highest support first
            relevant_resistance.sort()  # Lowest resistance first
            
            return {
                'support': relevant_support[:5],  # Top 5 support levels
                'resistance': relevant_resistance[:5],  # Top 5 resistance levels
                'current_price': float(current_price)
            }
            
        except Exception as e:
            print(f"Error updating support/resistance: {e}")
            return {'support': [], 'resistance': []}
    
    async def track_market_conditions(self) -> Dict:
        """
        Track and store current market conditions
        
        Returns:
            Dict with comprehensive market conditions
        """
        # Update all market metrics
        volatility = await self.calculate_volatility()
        trend = self.identify_trend()
        volume = await self.analyze_volume_profile()
        sr_levels = await self.update_support_resistance()
        
        # Combine all conditions
        market_conditions = {
            'timestamp': datetime.now().isoformat(),
            'volatility': volatility,
            'trend': trend,
            'volume': volume,
            'support_resistance': sr_levels,
            'market_regime': self._determine_market_regime(volatility, trend)
        }
        
        # Store in memory
        self.current_conditions = market_conditions
        self.last_update = datetime.now()
        
        # TODO: Store in database table market_conditions
        
        return market_conditions
    
    async def get_current_conditions(self) -> Dict:
        """Get current market conditions, updating if necessary"""
        if (not self.last_update or 
            datetime.now() - self.last_update > timedelta(seconds=self.update_interval)):
            await self.track_market_conditions()
        
        return self.current_conditions
    
    async def get_sr_levels(self) -> Dict:
        """Get current support and resistance levels"""
        conditions = await self.get_current_conditions()
        return conditions.get('support_resistance', {'support': [], 'resistance': []})
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.001) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] < current_cluster[-1] * threshold:
                current_cluster.append(level)
            else:
                # Add average of cluster
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Add last cluster
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _determine_market_regime(self, volatility: Dict, trend: Dict) -> str:
        """Determine overall market regime"""
        vol_level = volatility.get('volatility_level', 'unknown')
        trend_direction = trend.get('direction', 'unknown')
        trend_strength = trend.get('strength', 0)
        
        if 'strong' in trend_direction:
            if vol_level in ['low', 'very_low']:
                return 'steady_trend'
            else:
                return 'volatile_trend'
        elif trend_direction == 'ranging':
            if vol_level in ['low', 'very_low']:
                return 'tight_range'
            else:
                return 'wide_range'
        else:
            return 'mixed'

# Global instance
market_tracker = MarketTracker()