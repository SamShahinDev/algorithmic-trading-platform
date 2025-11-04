"""
Pattern Scanner for Identifying Profitable Trading Patterns
Detects momentum bursts, mean reversion, breakouts, and other patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of trading patterns"""
    MOMENTUM_BURST = "momentum_burst"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    FADE_EXTREME = "fade_extreme"
    VOLUME_CLIMAX = "volume_climax"
    SQUEEZE = "squeeze"
    TREND_CONTINUATION = "trend_continuation"
    REVERSAL = "reversal"


@dataclass
class PatternSignal:
    """Data class for pattern signals"""
    pattern_type: PatternType
    triggered: bool
    strength: float  # 0-100
    direction: int  # 1 for long, -1 for short
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    features: Dict[str, float]
    timestamp: pd.Timestamp


class PatternScanner:
    """Identify profitable trading patterns in historical data"""
    
    def __init__(self, min_strength: float = 50):
        """
        Initialize pattern scanner
        
        Args:
            min_strength: Minimum pattern strength to trigger signal (0-100)
        """
        self.min_strength = min_strength
        self.pattern_detectors = {
            PatternType.MOMENTUM_BURST: self._detect_momentum_burst,
            PatternType.MEAN_REVERSION: self._detect_mean_reversion,
            PatternType.BREAKOUT: self._detect_breakout,
            PatternType.FADE_EXTREME: self._detect_fade_extreme,
            PatternType.VOLUME_CLIMAX: self._detect_volume_climax,
            PatternType.SQUEEZE: self._detect_squeeze,
            PatternType.TREND_CONTINUATION: self._detect_trend_continuation,
            PatternType.REVERSAL: self._detect_reversal
        }
        self.pattern_history = []
        
    def scan_all_patterns(self, df: pd.DataFrame, features: pd.DataFrame) -> Dict[PatternType, PatternSignal]:
        """
        Run all pattern detectors and score opportunities
        
        Args:
            df: OHLCV data
            features: Calculated features from FeatureEngineer
            
        Returns:
            Dictionary of detected patterns with signals
        """
        import logging
        logger = logging.getLogger(__name__)
        
        signals = {}
        
        for pattern_type, detector in self.pattern_detectors.items():
            signal = detector(df, features)
            if signal and signal.triggered and signal.strength >= self.min_strength:
                signals[pattern_type] = signal
                self.pattern_history.append(signal)
                # Debug logging
                logger.debug(f"Pattern detected: {pattern_type.value}")
                logger.debug(f"  Strength: {signal.strength:.2f}%")
                logger.debug(f"  Direction: {'BUY' if signal.direction == 1 else 'SELL'}")
                logger.debug(f"  Confidence: {signal.confidence:.2f}%")
                logger.debug(f"  Entry: {signal.entry_price:.2f}, Stop: {signal.stop_loss:.2f}, Target: {signal.take_profit:.2f}")
        
        logger.debug(f"Pattern scanner output structure: {list(signals.keys()) if signals else 'No patterns detected'}")
        
        return signals
    
    def _detect_momentum_burst(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect acceleration patterns indicating strong momentum
        
        Conditions:
        - Increasing momentum over short period
        - Volume expansion
        - Price breaking recent range
        """
        if len(df) < 20:
            return None
            
        # Calculate momentum metrics
        momentum_5 = features['returns'].rolling(5).mean()
        momentum_10 = features['returns'].rolling(10).mean()
        acceleration = momentum_5 - momentum_5.shift(5)
        
        # Volume confirmation
        volume_ratio = features['volume_ratio_20'] if 'volume_ratio_20' in features else 1.0
        
        # Current values
        current_momentum = momentum_5.iloc[-1]
        current_acceleration = acceleration.iloc[-1]
        current_volume_ratio = volume_ratio.iloc[-1] if isinstance(volume_ratio, pd.Series) else volume_ratio
        
        # Check for momentum burst
        momentum_percentile = (momentum_5 > momentum_5.quantile(0.8)).iloc[-1]
        acceleration_percentile = (acceleration > acceleration.quantile(0.9)).iloc[-1]
        volume_confirmed = current_volume_ratio > 1.5
        
        triggered = momentum_percentile and acceleration_percentile and volume_confirmed
        
        if triggered:
            # Calculate pattern strength
            strength = min(100, (
                (current_momentum / momentum_5.std()) * 20 +
                (current_acceleration / acceleration.std()) * 30 +
                (current_volume_ratio - 1) * 20 +
                30  # Base strength for pattern detection
            ))
            
            # Determine direction
            direction = 1 if current_momentum > 0 else -1
            
            # Calculate entry, stop, and target
            current_price = df['close'].iloc[-1]
            atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
            
            entry_price = current_price
            stop_loss = current_price - (direction * atr * 1.5)
            take_profit = current_price + (direction * atr * 3)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_pattern_confidence(
                strength=strength,
                volume_confirmation=volume_confirmed,
                trend_alignment=features['trend_regime'].iloc[-1] == direction if 'trend_regime' in features else True
            )
            
            return PatternSignal(
                pattern_type=PatternType.MOMENTUM_BURST,
                triggered=True,
                strength=strength,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                features={
                    'momentum': current_momentum,
                    'acceleration': current_acceleration,
                    'volume_ratio': current_volume_ratio
                },
                timestamp=df.index[-1]
            )
        
        return None
    
    def _detect_mean_reversion(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect mean reversion opportunities
        
        Conditions:
        - Price extended from mean
        - RSI in extreme territory
        - Decreasing momentum
        """
        if len(df) < 50:
            return None
            
        # Calculate extension from mean
        ma_20 = features['ma_20'] if 'ma_20' in features else df['close'].rolling(20).mean()
        std_20 = features['std_20'] if 'std_20' in features else df['close'].rolling(20).std()
        z_score = (df['close'] - ma_20) / std_20
        
        current_z_score = z_score.iloc[-1]
        current_rsi = features['rsi_14'].iloc[-1] if 'rsi_14' in features else 50
        
        # Check for extreme conditions
        oversold = current_z_score < -2 and current_rsi < 30
        overbought = current_z_score > 2 and current_rsi > 70
        
        triggered = oversold or overbought
        
        if triggered:
            # Momentum should be decreasing for mean reversion
            momentum_decreasing = features['returns'].rolling(5).mean().diff().iloc[-1] < 0 if overbought else features['returns'].rolling(5).mean().diff().iloc[-1] > 0
            
            if momentum_decreasing:
                direction = 1 if oversold else -1
                
                strength = min(100, (
                    abs(current_z_score) * 25 +
                    (abs(current_rsi - 50) / 50) * 100 * 25 +
                    30 if momentum_decreasing else 0 +
                    20  # Base strength
                ))
                
                current_price = df['close'].iloc[-1]
                atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
                
                entry_price = current_price
                stop_loss = current_price - (direction * atr * 2)
                take_profit = ma_20.iloc[-1]  # Target is mean
                
                confidence = self._calculate_pattern_confidence(
                    strength=strength,
                    volume_confirmation=True,
                    trend_alignment=False  # Mean reversion goes against trend
                )
                
                return PatternSignal(
                    pattern_type=PatternType.MEAN_REVERSION,
                    triggered=True,
                    strength=strength,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    features={
                        'z_score': current_z_score,
                        'rsi': current_rsi,
                        'distance_from_mean': abs(current_price - ma_20.iloc[-1])
                    },
                    timestamp=df.index[-1]
                )
        
        return None
    
    def _detect_breakout(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect breakout patterns
        
        Conditions:
        - Price breaking key resistance/support
        - Volume expansion
        - Momentum confirmation
        """
        if len(df) < 50:
            return None
            
        # Calculate recent high/low
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        
        current_price = df['close'].iloc[-1]
        previous_high = high_20.iloc[-2]
        previous_low = low_20.iloc[-2]
        
        # Check for breakout
        bullish_breakout = current_price > previous_high
        bearish_breakout = current_price < previous_low
        
        triggered = bullish_breakout or bearish_breakout
        
        if triggered:
            direction = 1 if bullish_breakout else -1
            
            # Volume confirmation
            volume_ratio = features['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in features else 1.0
            volume_confirmed = volume_ratio > 1.3
            
            # Momentum confirmation
            momentum = features['returns'].rolling(5).mean().iloc[-1]
            momentum_confirmed = (momentum > 0 and direction == 1) or (momentum < 0 and direction == -1)
            
            if volume_confirmed and momentum_confirmed:
                strength = min(100, (
                    30 if volume_confirmed else 0 +
                    30 if momentum_confirmed else 0 +
                    20 * abs((current_price - (previous_high if bullish_breakout else previous_low)) / current_price) +
                    20  # Base strength
                ))
                
                atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
                
                entry_price = current_price
                stop_loss = previous_high - atr if bullish_breakout else previous_low + atr
                take_profit = current_price + (direction * atr * 3)
                
                confidence = self._calculate_pattern_confidence(
                    strength=strength,
                    volume_confirmation=volume_confirmed,
                    trend_alignment=True
                )
                
                return PatternSignal(
                    pattern_type=PatternType.BREAKOUT,
                    triggered=True,
                    strength=strength,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    features={
                        'breakout_level': previous_high if bullish_breakout else previous_low,
                        'volume_ratio': volume_ratio,
                        'momentum': momentum
                    },
                    timestamp=df.index[-1]
                )
        
        return None
    
    def _detect_fade_extreme(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect extreme moves to fade
        
        Conditions:
        - Extreme price move in single bar
        - Volume spike
        - Extended from VWAP or mean
        """
        if len(df) < 20:
            return None
            
        # Calculate extreme move
        current_return = features['returns'].iloc[-1]
        return_std = features['returns'].rolling(20).std().iloc[-1]
        
        # Check for extreme move (3+ standard deviations)
        extreme_move = abs(current_return) > (return_std * 3)
        
        if extreme_move:
            direction = -1 if current_return > 0 else 1  # Fade the move
            
            # Volume spike confirmation
            volume_ratio = features['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in features else 1.0
            volume_spike = volume_ratio > 2.0
            
            # Check extension from mean
            ma_20 = features['ma_20'].iloc[-1] if 'ma_20' in features else df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            extension = abs((current_price - ma_20) / ma_20)
            
            if volume_spike and extension > 0.02:  # 2% extension
                strength = min(100, (
                    (abs(current_return) / return_std) * 20 +
                    (volume_ratio - 1) * 20 +
                    extension * 1000 +
                    20  # Base strength
                ))
                
                atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
                
                entry_price = current_price
                stop_loss = current_price - (direction * atr * 1)  # Tight stop for fade
                take_profit = ma_20  # Target is mean
                
                confidence = self._calculate_pattern_confidence(
                    strength=strength,
                    volume_confirmation=volume_spike,
                    trend_alignment=False
                )
                
                return PatternSignal(
                    pattern_type=PatternType.FADE_EXTREME,
                    triggered=True,
                    strength=strength,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    features={
                        'extreme_return': current_return,
                        'std_moves': abs(current_return) / return_std,
                        'extension': extension
                    },
                    timestamp=df.index[-1]
                )
        
        return None
    
    def _detect_volume_climax(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect volume climax patterns
        
        Conditions:
        - Extreme volume spike
        - Price reversal signs
        - Exhaustion patterns
        """
        if len(df) < 50:
            return None
            
        # Volume analysis
        volume_ratio = features['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in features else 1.0
        volume_ma = features['volume_ma_20'].iloc[-1] if 'volume_ma_20' in features else df['volume'].rolling(20).mean().iloc[-1]
        
        # Check for volume climax (3x average volume)
        volume_climax = volume_ratio > 3.0
        
        if volume_climax:
            # Check for exhaustion signs
            current_return = features['returns'].iloc[-1]
            prev_return = features['returns'].iloc[-2]
            
            # Reversal pattern: big move with volume, then opposite move
            reversal = (current_return * prev_return) < 0  # Opposite signs
            
            if reversal:
                direction = 1 if current_return > 0 else -1
                
                # Check for exhaustion gap
                gap_size = abs(df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                exhaustion_gap = gap_size > 0.005  # 0.5% gap
                
                strength = min(100, (
                    (volume_ratio - 1) * 15 +
                    30 if reversal else 0 +
                    20 if exhaustion_gap else 0 +
                    gap_size * 2000 +
                    15  # Base strength
                ))
                
                current_price = df['close'].iloc[-1]
                atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
                
                entry_price = current_price
                stop_loss = df['low'].iloc[-1] - atr if direction == 1 else df['high'].iloc[-1] + atr
                take_profit = current_price + (direction * atr * 2.5)
                
                confidence = self._calculate_pattern_confidence(
                    strength=strength,
                    volume_confirmation=True,
                    trend_alignment=False
                )
                
                return PatternSignal(
                    pattern_type=PatternType.VOLUME_CLIMAX,
                    triggered=True,
                    strength=strength,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    features={
                        'volume_ratio': volume_ratio,
                        'gap_size': gap_size,
                        'reversal': reversal
                    },
                    timestamp=df.index[-1]
                )
        
        return None
    
    def _detect_squeeze(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect volatility squeeze patterns (Bollinger Band squeeze)
        
        Conditions:
        - Bollinger Bands inside Keltner Channels
        - Decreasing volatility
        - Potential breakout setup
        """
        if len(df) < 50:
            return None
            
        # Calculate Bollinger Bands
        ma_20 = features['ma_20'] if 'ma_20' in features else df['close'].rolling(20).mean()
        std_20 = features['std_20'] if 'std_20' in features else df['close'].rolling(20).std()
        
        bb_upper = ma_20 + (std_20 * 2)
        bb_lower = ma_20 - (std_20 * 2)
        bb_width = bb_upper - bb_lower
        
        # Calculate Keltner Channels
        atr_20 = features['atr_20'] if 'atr_20' in features else self._calculate_atr(df, 20)
        kc_upper = ma_20 + (atr_20 * 1.5)
        kc_lower = ma_20 - (atr_20 * 1.5)
        
        # Check for squeeze
        squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
        
        # Check if squeeze is releasing
        prev_squeeze = (bb_upper.iloc[-2] < kc_upper.iloc[-2]) and (bb_lower.iloc[-2] > kc_lower.iloc[-2])
        squeeze_release = prev_squeeze and not squeeze
        
        if squeeze or squeeze_release:
            # Determine direction based on momentum
            momentum = features['returns'].rolling(10).mean().iloc[-1]
            direction = 1 if momentum > 0 else -1
            
            # Calculate squeeze duration
            squeeze_duration = 0
            for i in range(1, min(20, len(df))):
                if (bb_upper.iloc[-i] < kc_upper.iloc[-i]) and (bb_lower.iloc[-i] > kc_lower.iloc[-i]):
                    squeeze_duration += 1
                else:
                    break
            
            strength = min(100, (
                squeeze_duration * 5 +
                30 if squeeze_release else 20 +
                abs(momentum) * 1000 +
                20  # Base strength
            ))
            
            current_price = df['close'].iloc[-1]
            
            entry_price = current_price
            stop_loss = current_price - (direction * atr_20.iloc[-1] * 1.5)
            take_profit = current_price + (direction * atr_20.iloc[-1] * 3)
            
            confidence = self._calculate_pattern_confidence(
                strength=strength,
                volume_confirmation=True,
                trend_alignment=True
            )
            
            return PatternSignal(
                pattern_type=PatternType.SQUEEZE,
                triggered=True,
                strength=strength,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                features={
                    'squeeze_duration': squeeze_duration,
                    'bb_width': bb_width.iloc[-1],
                    'momentum': momentum
                },
                timestamp=df.index[-1]
            )
        
        return None
    
    def _detect_trend_continuation(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect trend continuation patterns (flags, pennants)
        
        Conditions:
        - Strong existing trend
        - Brief consolidation
        - Momentum resumption
        """
        if len(df) < 50:
            return None
            
        # Identify trend
        ma_20 = features['ma_20'] if 'ma_20' in features else df['close'].rolling(20).mean()
        ma_50 = features['ma_50'] if 'ma_50' in features else df['close'].rolling(50).mean()
        
        uptrend = ma_20.iloc[-1] > ma_50.iloc[-1]
        downtrend = ma_20.iloc[-1] < ma_50.iloc[-1]
        
        if uptrend or downtrend:
            direction = 1 if uptrend else -1
            
            # Check for consolidation
            recent_range = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            prior_range = df['high'].rolling(10).max().shift(10) - df['low'].rolling(10).min().shift(10)
            
            consolidation = recent_range.iloc[-1] < (prior_range.iloc[-1] * 0.7)
            
            # Check for momentum resumption
            current_momentum = features['returns'].rolling(5).mean().iloc[-1]
            momentum_direction = (current_momentum > 0 and uptrend) or (current_momentum < 0 and downtrend)
            
            if consolidation and momentum_direction:
                # Calculate trend strength
                trend_strength = abs(ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]
                
                strength = min(100, (
                    trend_strength * 2000 +
                    30 if consolidation else 0 +
                    abs(current_momentum) * 1000 +
                    20  # Base strength
                ))
                
                current_price = df['close'].iloc[-1]
                atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
                
                entry_price = current_price
                stop_loss = df['low'].rolling(10).min().iloc[-1] if uptrend else df['high'].rolling(10).max().iloc[-1]
                take_profit = current_price + (direction * atr * 3)
                
                confidence = self._calculate_pattern_confidence(
                    strength=strength,
                    volume_confirmation=True,
                    trend_alignment=True
                )
                
                return PatternSignal(
                    pattern_type=PatternType.TREND_CONTINUATION,
                    triggered=True,
                    strength=strength,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    features={
                        'trend_strength': trend_strength,
                        'consolidation_ratio': recent_range.iloc[-1] / prior_range.iloc[-1],
                        'momentum': current_momentum
                    },
                    timestamp=df.index[-1]
                )
        
        return None
    
    def _detect_reversal(self, df: pd.DataFrame, features: pd.DataFrame) -> Optional[PatternSignal]:
        """
        Detect reversal patterns (double top/bottom, head and shoulders)
        
        Conditions:
        - Multiple tests of support/resistance
        - Divergence in momentum
        - Volume confirmation
        """
        if len(df) < 100:
            return None
            
        # Find recent peaks and troughs
        peaks = self._find_peaks(df['high'], window=20)
        troughs = self._find_troughs(df['low'], window=20)
        
        # Check for double top/bottom
        double_top = False
        double_bottom = False
        
        if len(peaks) >= 2:
            # Check if recent peaks are at similar levels (within 1%)
            last_two_peaks = peaks[-2:]
            if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.01:
                double_top = True
        
        if len(troughs) >= 2:
            # Check if recent troughs are at similar levels (within 1%)
            last_two_troughs = troughs[-2:]
            if abs(last_two_troughs[0] - last_two_troughs[1]) / last_two_troughs[0] < 0.01:
                double_bottom = True
        
        if double_top or double_bottom:
            direction = -1 if double_top else 1
            
            # Check for momentum divergence
            rsi = features['rsi_20'] if 'rsi_20' in features else 50
            price_trend = df['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]).iloc[-1]
            rsi_trend = rsi.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]).iloc[-1] if isinstance(rsi, pd.Series) else 0
            
            divergence = (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0)
            
            if divergence:
                # Volume confirmation
                volume_ratio = features['volume_ratio_20'].iloc[-1] if 'volume_ratio_20' in features else 1.0
                
                strength = min(100, (
                    40 if divergence else 0 +
                    20 if double_top or double_bottom else 0 +
                    (volume_ratio - 1) * 20 +
                    20  # Base strength
                ))
                
                current_price = df['close'].iloc[-1]
                atr = features['atr_20'].iloc[-1] if 'atr_20' in features else df['close'].pct_change().std() * current_price
                
                entry_price = current_price
                
                if double_top:
                    stop_loss = max(peaks[-2:]) + atr
                    take_profit = current_price - (atr * 3)
                else:  # double_bottom
                    stop_loss = min(troughs[-2:]) - atr
                    take_profit = current_price + (atr * 3)
                
                confidence = self._calculate_pattern_confidence(
                    strength=strength,
                    volume_confirmation=volume_ratio > 1.2,
                    trend_alignment=False  # Reversal goes against trend
                )
                
                return PatternSignal(
                    pattern_type=PatternType.REVERSAL,
                    triggered=True,
                    strength=strength,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    features={
                        'pattern': 'double_top' if double_top else 'double_bottom',
                        'divergence': divergence,
                        'volume_ratio': volume_ratio
                    },
                    timestamp=df.index[-1]
                )
        
        return None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean()
    
    def _find_peaks(self, series: pd.Series, window: int = 20) -> List[float]:
        """Find local peaks in price series"""
        peaks = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                peaks.append(series.iloc[i])
        return peaks
    
    def _find_troughs(self, series: pd.Series, window: int = 20) -> List[float]:
        """Find local troughs in price series"""
        troughs = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                troughs.append(series.iloc[i])
        return troughs
    
    def _calculate_pattern_confidence(self, strength: float, volume_confirmation: bool, 
                                     trend_alignment: bool) -> float:
        """
        Calculate overall confidence for a pattern
        
        Args:
            strength: Pattern strength (0-100)
            volume_confirmation: Whether volume confirms the pattern
            trend_alignment: Whether pattern aligns with trend
            
        Returns:
            Confidence score (0-100)
        """
        confidence = strength * 0.6  # Base confidence from strength
        
        if volume_confirmation:
            confidence += 20
        
        if trend_alignment:
            confidence += 20
        
        return min(100, confidence)
    
    def get_pattern_statistics(self) -> Dict[PatternType, Dict[str, float]]:
        """
        Get statistics for all detected patterns
        
        Returns:
            Dictionary with pattern statistics
        """
        stats = {}
        
        for pattern_type in PatternType:
            pattern_signals = [s for s in self.pattern_history if s.pattern_type == pattern_type]
            
            if pattern_signals:
                stats[pattern_type] = {
                    'count': len(pattern_signals),
                    'avg_strength': np.mean([s.strength for s in pattern_signals]),
                    'avg_confidence': np.mean([s.confidence for s in pattern_signals]),
                    'long_ratio': len([s for s in pattern_signals if s.direction == 1]) / len(pattern_signals)
                }
        
        return stats