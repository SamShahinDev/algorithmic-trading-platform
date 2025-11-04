"""
Market Microstructure Analysis
Analyzes order flow, volume profiles, and market microstructure patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class MicrostructureRegime(Enum):
    """Market microstructure regimes"""
    BALANCED = "balanced"
    BUYER_DOMINATED = "buyer_dominated" 
    SELLER_DOMINATED = "seller_dominated"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    THIN_LIQUIDITY = "thin_liquidity"
    HIGH_LIQUIDITY = "high_liquidity"


@dataclass
class VolumeProfile:
    """Volume profile data"""
    point_of_control: float  # Price with highest volume
    value_area_high: float    # Upper bound of 70% volume area
    value_area_low: float     # Lower bound of 70% volume area
    volume_nodes: Dict[float, float]  # Price -> Volume mapping
    total_volume: float
    buy_volume: float
    sell_volume: float
    delta: float  # Buy volume - Sell volume


@dataclass
class OrderFlowMetrics:
    """Order flow imbalance metrics"""
    buy_pressure: float  # 0-100
    sell_pressure: float  # 0-100
    net_pressure: float  # -100 to 100
    cumulative_delta: float
    delta_divergence: bool
    absorption: bool  # Large volume absorbed without price movement
    exhaustion: bool  # Volume drying up at extremes


class MicrostructureAnalyzer:
    """Analyze order flow and market microstructure"""
    
    def __init__(self, tick_size: float = 0.25):
        """
        Initialize microstructure analyzer
        
        Args:
            tick_size: Minimum price increment for the instrument
        """
        self.tick_size = tick_size
        self.volume_profile_cache = {}
        self.order_flow_history = []
        
    def analyze_current_state(self, df: pd.DataFrame) -> Dict:
        """
        Determine current market microstructure
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Complete microstructure analysis
        """
        analysis = {}
        
        # Tick Analysis
        tick_distribution = self._analyze_tick_distribution(df)
        analysis['tick_behavior'] = tick_distribution
        
        # Volume Profile
        volume_profile = self._build_volume_profile(df)
        analysis['volume_profile'] = volume_profile
        analysis['poc'] = volume_profile.point_of_control
        analysis['value_area'] = (volume_profile.value_area_low, volume_profile.value_area_high)
        
        # Order Flow Imbalance
        order_flow = self._calculate_order_flow_imbalance(df)
        analysis['order_flow'] = order_flow
        analysis['buy_pressure'] = order_flow.buy_pressure
        analysis['sell_pressure'] = order_flow.sell_pressure
        analysis['net_pressure'] = order_flow.net_pressure
        
        # Market Depth Analysis (using volume as proxy)
        depth_metrics = self._analyze_market_depth(df)
        analysis['depth_metrics'] = depth_metrics
        
        # Microstructure Regime
        regime = self._classify_microstructure_regime(analysis)
        analysis['regime'] = regime
        
        # Liquidity Analysis
        liquidity = self._analyze_liquidity(df)
        analysis['liquidity'] = liquidity
        
        # Trade Size Distribution
        trade_sizes = self._analyze_trade_sizes(df)
        analysis['trade_sizes'] = trade_sizes
        
        return analysis
    
    def _analyze_tick_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze tick-by-tick price movements
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Tick distribution analysis
        """
        # Calculate tick changes
        tick_changes = df['close'].diff() / self.tick_size
        
        # Remove NaN values
        tick_changes = tick_changes.dropna()
        
        if len(tick_changes) == 0:
            return {
                'upticks': 0,
                'downticks': 0,
                'unchanged': 0,
                'tick_imbalance': 0,
                'avg_tick_size': 0
            }
        
        upticks = (tick_changes > 0).sum()
        downticks = (tick_changes < 0).sum()
        unchanged = (tick_changes == 0).sum()
        
        total_ticks = len(tick_changes)
        tick_imbalance = (upticks - downticks) / (total_ticks + 1e-10)
        
        # Calculate tick velocity (rate of change)
        tick_velocity = tick_changes.rolling(10).mean().iloc[-1] if len(tick_changes) > 10 else 0
        
        return {
            'upticks': upticks,
            'downticks': downticks,
            'unchanged': unchanged,
            'tick_imbalance': tick_imbalance,
            'avg_tick_size': abs(tick_changes).mean(),
            'tick_velocity': tick_velocity,
            'tick_acceleration': tick_changes.rolling(10).mean().diff().iloc[-1] if len(tick_changes) > 10 else 0
        }
    
    def _build_volume_profile(self, df: pd.DataFrame, bins: int = 30) -> VolumeProfile:
        """
        Build volume profile from OHLCV data
        
        Args:
            df: OHLCV dataframe
            bins: Number of price bins for volume profile
            
        Returns:
            Volume profile object
        """
        if len(df) < 2:
            # Return empty profile for insufficient data
            return VolumeProfile(
                point_of_control=df['close'].iloc[-1] if len(df) > 0 else 0,
                value_area_high=df['high'].iloc[-1] if len(df) > 0 else 0,
                value_area_low=df['low'].iloc[-1] if len(df) > 0 else 0,
                volume_nodes={},
                total_volume=0,
                buy_volume=0,
                sell_volume=0,
                delta=0
            )
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            # All prices are the same
            return VolumeProfile(
                point_of_control=df['close'].iloc[-1],
                value_area_high=df['close'].iloc[-1],
                value_area_low=df['close'].iloc[-1],
                volume_nodes={df['close'].iloc[-1]: df['volume'].sum()},
                total_volume=df['volume'].sum(),
                buy_volume=df['volume'].sum() / 2,
                sell_volume=df['volume'].sum() / 2,
                delta=0
            )
        
        price_bins = np.linspace(price_min, price_max, bins)
        volume_nodes = {}
        
        # Distribute volume across price bins using VWAP logic
        for _, row in df.iterrows():
            # Find the bin for this bar's average price
            avg_price = (row['high'] + row['low'] + row['close']) / 3
            bin_idx = np.searchsorted(price_bins, avg_price)
            bin_idx = min(bin_idx, len(price_bins) - 1)
            
            bin_price = price_bins[bin_idx]
            if bin_price not in volume_nodes:
                volume_nodes[bin_price] = 0
            volume_nodes[bin_price] += row['volume']
        
        # Find Point of Control (price with highest volume)
        if volume_nodes:
            poc = max(volume_nodes.keys(), key=lambda x: volume_nodes[x])
        else:
            poc = df['close'].iloc[-1]
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_nodes.values())
        target_volume = total_volume * 0.7
        
        # Sort nodes by volume
        sorted_nodes = sorted(volume_nodes.items(), key=lambda x: x[1], reverse=True)
        
        accumulated_volume = 0
        value_area_prices = []
        
        for price, volume in sorted_nodes:
            accumulated_volume += volume
            value_area_prices.append(price)
            if accumulated_volume >= target_volume:
                break
        
        if value_area_prices:
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
        else:
            value_area_high = poc
            value_area_low = poc
        
        # Calculate buy/sell volume (using close position in bar as proxy)
        buy_volume = 0
        sell_volume = 0
        
        for _, row in df.iterrows():
            # If close is in upper half of bar, consider it buying pressure
            mid_price = (row['high'] + row['low']) / 2
            if row['close'] > mid_price:
                buy_volume += row['volume'] * ((row['close'] - mid_price) / (row['high'] - mid_price + 1e-10))
                sell_volume += row['volume'] * (1 - (row['close'] - mid_price) / (row['high'] - mid_price + 1e-10))
            else:
                sell_volume += row['volume'] * ((mid_price - row['close']) / (mid_price - row['low'] + 1e-10))
                buy_volume += row['volume'] * (1 - (mid_price - row['close']) / (mid_price - row['low'] + 1e-10))
        
        delta = buy_volume - sell_volume
        
        return VolumeProfile(
            point_of_control=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            volume_nodes=volume_nodes,
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            delta=delta
        )
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> OrderFlowMetrics:
        """
        Calculate order flow imbalance metrics
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Order flow metrics
        """
        if len(df) < 2:
            return OrderFlowMetrics(
                buy_pressure=50,
                sell_pressure=50,
                net_pressure=0,
                cumulative_delta=0,
                delta_divergence=False,
                absorption=False,
                exhaustion=False
            )
        
        # Calculate buy/sell pressure for each bar
        buy_pressures = []
        sell_pressures = []
        deltas = []
        
        for _, row in df.iterrows():
            # Calculate pressure based on close position in range
            range_size = row['high'] - row['low']
            if range_size > 0:
                buy_pressure = ((row['close'] - row['low']) / range_size) * 100
                sell_pressure = ((row['high'] - row['close']) / range_size) * 100
            else:
                buy_pressure = 50
                sell_pressure = 50
            
            buy_pressures.append(buy_pressure)
            sell_pressures.append(sell_pressure)
            
            # Calculate delta (buy volume - sell volume proxy)
            buy_vol = row['volume'] * (buy_pressure / 100)
            sell_vol = row['volume'] * (sell_pressure / 100)
            deltas.append(buy_vol - sell_vol)
        
        # Recent metrics (last 20 bars)
        recent_buy_pressure = np.mean(buy_pressures[-20:]) if len(buy_pressures) > 0 else 50
        recent_sell_pressure = np.mean(sell_pressures[-20:]) if len(sell_pressures) > 0 else 50
        net_pressure = recent_buy_pressure - recent_sell_pressure
        
        # Cumulative delta
        cumulative_delta = sum(deltas)
        
        # Check for delta divergence (price up but delta down or vice versa)
        if len(df) >= 20:
            price_change = df['close'].iloc[-1] - df['close'].iloc[-20]
            delta_change = sum(deltas[-20:])
            delta_divergence = (price_change > 0 and delta_change < 0) or (price_change < 0 and delta_change > 0)
        else:
            delta_divergence = False
        
        # Check for absorption (high volume but small price movement)
        if len(df) >= 10:
            recent_volume = df['volume'].iloc[-10:].mean()
            avg_volume = df['volume'].mean()
            price_range = df['high'].iloc[-10:].max() - df['low'].iloc[-10:].min()
            avg_range = (df['high'] - df['low']).mean()
            
            absorption = (recent_volume > avg_volume * 1.5) and (price_range < avg_range * 0.7)
        else:
            absorption = False
        
        # Check for exhaustion (decreasing volume at extremes)
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            at_high = df['close'].iloc[-1] > recent_high * 0.99
            at_low = df['close'].iloc[-1] < recent_low * 1.01
            
            volume_decreasing = df['volume'].iloc[-5:].mean() < df['volume'].iloc[-20:-5].mean()
            
            exhaustion = (at_high or at_low) and volume_decreasing
        else:
            exhaustion = False
        
        return OrderFlowMetrics(
            buy_pressure=recent_buy_pressure,
            sell_pressure=recent_sell_pressure,
            net_pressure=net_pressure,
            cumulative_delta=cumulative_delta,
            delta_divergence=delta_divergence,
            absorption=absorption,
            exhaustion=exhaustion
        )
    
    def _analyze_market_depth(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market depth using volume and price action as proxy
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Market depth metrics
        """
        if len(df) < 20:
            return {
                'bid_depth': 0,
                'ask_depth': 0,
                'depth_imbalance': 0,
                'liquidity_score': 50
            }
        
        # Use volume at different price levels as depth proxy
        recent_df = df.iloc[-20:]
        
        # Calculate average volume at different price levels
        price_levels = np.linspace(recent_df['low'].min(), recent_df['high'].max(), 10)
        volume_at_levels = []
        
        for i in range(len(price_levels) - 1):
            level_low = price_levels[i]
            level_high = price_levels[i + 1]
            
            # Find bars that traded in this price range
            mask = (recent_df['low'] <= level_high) & (recent_df['high'] >= level_low)
            level_volume = recent_df.loc[mask, 'volume'].sum()
            volume_at_levels.append(level_volume)
        
        # Current price position
        current_price = df['close'].iloc[-1]
        price_position = (current_price - recent_df['low'].min()) / (recent_df['high'].max() - recent_df['low'].min() + 1e-10)
        
        # Estimate bid/ask depth based on volume distribution
        mid_point = len(volume_at_levels) // 2
        bid_depth = sum(volume_at_levels[:mid_point])
        ask_depth = sum(volume_at_levels[mid_point:])
        
        total_depth = bid_depth + ask_depth
        depth_imbalance = (bid_depth - ask_depth) / (total_depth + 1e-10)
        
        # Liquidity score based on volume consistency
        volume_std = np.std(volume_at_levels)
        volume_mean = np.mean(volume_at_levels)
        liquidity_score = min(100, max(0, 100 * (1 - volume_std / (volume_mean + 1e-10))))
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_imbalance': depth_imbalance,
            'liquidity_score': liquidity_score,
            'price_position': price_position
        }
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market liquidity conditions
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Liquidity metrics
        """
        if len(df) < 20:
            return {
                'spread_proxy': 0,
                'liquidity_ratio': 1,
                'volume_stability': 50,
                'price_impact': 0
            }
        
        # Spread proxy (using high-low range)
        spread_proxy = ((df['high'] - df['low']) / df['close']).mean()
        
        # Liquidity ratio (volume / price range)
        price_ranges = df['high'] - df['low']
        liquidity_ratios = df['volume'] / (price_ranges + 1e-10)
        liquidity_ratio = liquidity_ratios.mean()
        
        # Volume stability (inverse of volume volatility)
        volume_stability = 100 * (1 - df['volume'].pct_change().std())
        volume_stability = max(0, min(100, volume_stability))
        
        # Price impact (how much price moves per unit volume)
        price_changes = df['close'].pct_change().abs()
        price_impact = (price_changes / (df['volume'] + 1e-10)).mean()
        
        return {
            'spread_proxy': spread_proxy,
            'liquidity_ratio': liquidity_ratio,
            'volume_stability': volume_stability,
            'price_impact': price_impact
        }
    
    def _analyze_trade_sizes(self, df: pd.DataFrame) -> Dict:
        """
        Analyze trade size distribution
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Trade size metrics
        """
        if len(df) < 10:
            return {
                'avg_trade_size': 0,
                'large_trade_ratio': 0,
                'trade_size_trend': 0,
                'institutional_activity': 0
            }
        
        volumes = df['volume'].values
        
        # Average trade size (assuming each bar represents multiple trades)
        avg_trade_size = volumes.mean()
        
        # Large trade ratio (trades > 2x average)
        large_trades = volumes > (avg_trade_size * 2)
        large_trade_ratio = large_trades.sum() / len(volumes)
        
        # Trade size trend
        if len(volumes) >= 20:
            recent_avg = volumes[-10:].mean()
            prior_avg = volumes[-20:-10].mean()
            trade_size_trend = (recent_avg - prior_avg) / (prior_avg + 1e-10)
        else:
            trade_size_trend = 0
        
        # Institutional activity proxy (large trades with small price impact)
        institutional_activity = 0
        for i in range(1, min(10, len(df))):
            if df['volume'].iloc[-i] > avg_trade_size * 2:
                price_impact = abs(df['close'].iloc[-i] - df['close'].iloc[-i-1]) / df['close'].iloc[-i-1]
                if price_impact < 0.001:  # Large volume, small price move
                    institutional_activity += 1
        
        institutional_activity = (institutional_activity / 10) * 100
        
        return {
            'avg_trade_size': avg_trade_size,
            'large_trade_ratio': large_trade_ratio,
            'trade_size_trend': trade_size_trend,
            'institutional_activity': institutional_activity
        }
    
    def _classify_microstructure_regime(self, analysis: Dict) -> MicrostructureRegime:
        """
        Classify the current microstructure regime
        
        Args:
            analysis: Complete microstructure analysis
            
        Returns:
            Microstructure regime classification
        """
        order_flow = analysis.get('order_flow', None)
        liquidity = analysis.get('liquidity', {})
        depth_metrics = analysis.get('depth_metrics', {})
        
        if not order_flow:
            return MicrostructureRegime.BALANCED
        
        # Check liquidity conditions first
        if liquidity.get('liquidity_score', 50) < 30:
            return MicrostructureRegime.THIN_LIQUIDITY
        elif liquidity.get('liquidity_score', 50) > 70:
            return MicrostructureRegime.HIGH_LIQUIDITY
        
        # Check order flow dominance
        net_pressure = order_flow.net_pressure
        
        if abs(net_pressure) < 10:
            return MicrostructureRegime.BALANCED
        elif net_pressure > 30:
            # Check if accumulation or just buyer dominated
            if order_flow.absorption and not order_flow.exhaustion:
                return MicrostructureRegime.ACCUMULATION
            else:
                return MicrostructureRegime.BUYER_DOMINATED
        elif net_pressure < -30:
            # Check if distribution or just seller dominated
            if order_flow.absorption and not order_flow.exhaustion:
                return MicrostructureRegime.DISTRIBUTION
            else:
                return MicrostructureRegime.SELLER_DOMINATED
        else:
            return MicrostructureRegime.BALANCED
    
    def get_trade_signals(self, analysis: Dict) -> Dict[str, float]:
        """
        Generate trading signals from microstructure analysis
        
        Args:
            analysis: Microstructure analysis dictionary
            
        Returns:
            Trading signals with confidence scores
        """
        signals = {}
        
        regime = analysis.get('regime', MicrostructureRegime.BALANCED)
        order_flow = analysis.get('order_flow', None)
        
        if not order_flow:
            return signals
        
        # Accumulation signal
        if regime == MicrostructureRegime.ACCUMULATION:
            signals['long_accumulation'] = 80
        
        # Distribution signal
        elif regime == MicrostructureRegime.DISTRIBUTION:
            signals['short_distribution'] = 80
        
        # Delta divergence signal
        if order_flow.delta_divergence:
            if order_flow.net_pressure > 0:
                signals['long_divergence'] = 70
            else:
                signals['short_divergence'] = 70
        
        # Exhaustion signal
        if order_flow.exhaustion:
            if analysis.get('tick_behavior', {}).get('tick_imbalance', 0) > 0:
                signals['short_exhaustion'] = 75
            else:
                signals['long_exhaustion'] = 75
        
        # Absorption signal
        if order_flow.absorption:
            if order_flow.net_pressure > 20:
                signals['long_absorption'] = 65
            elif order_flow.net_pressure < -20:
                signals['short_absorption'] = 65
        
        return signals