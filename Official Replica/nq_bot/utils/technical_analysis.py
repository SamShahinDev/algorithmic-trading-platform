"""
Enhanced Technical Analysis Fallback Module for NQ Trading
Provides fallback signals with precise scoring system (0-10 points, require ≥7.5)
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import logging
from enum import Enum
import json
from pathlib import Path
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)

# Scoring Constants
MA_TREND_POINTS = 2
ADX_MODERATE_POINTS = 1  # 18-30
ADX_STRONG_POINTS = 2    # ≥30
RSI_ZONE1_POINTS = 1     # 48-62 long, 38-52 short
RSI_ZONE2_POINTS = 2     # 62-72 long, 28-38 short
BOLLINGER_POINTS = 1
ATR_PERCENTILE_POINTS = 1
STOCH_CROSS_POINTS = 1
VOLUME_ZSCORE_POINTS = 1
LEVEL_CONFLUENCE_POINTS = 2
MIN_SCORE_THRESHOLD = 7.5
MAX_TRADES_PER_HOUR = 10  # Increased for discovery mode testing

class SignalStrength(Enum):
    """Signal strength levels based on score"""
    STRONG = "strong"       # Score >= 9.0
    MODERATE = "moderate"   # Score >= 7.5
    WEAK = "weak"          # Score < 7.5 (not used)
    NEUTRAL = "neutral"

class TechnicalAnalysisFallback:
    """
    Enhanced technical analysis fallback system with precise scoring
    Generates trading signals based on 0-10 point scoring system
    """
    
    def __init__(self, data_cache=None):
        """Initialize enhanced technical analysis system
        
        Args:
            data_cache: DataCache instance for accessing indicators
        """
        self.data_cache = data_cache
        
        # Indicator periods
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ma_fast = 20
        self.ma_medium = 50
        self.ma_slow = 200
        self.bb_period = 20
        self.bb_std = 2.0
        self.atr_period = 14
        self.adx_period = 14
        self.stoch_period = 14
        
        # NQ specifics
        self.tick_size = 0.25
        self.point_value = 20
        
        # Risk parameters
        self.default_stop_ticks = 8
        self.default_target_ticks = 6
        
        # Support/Resistance tracking
        self.support_levels = []
        self.resistance_levels = []
        self.pivot_points = {}
        self.vwap = None
        self.onh = None  # Overnight high
        self.onl = None  # Overnight low
        self.poc = None  # Point of control
        
        # ATR percentile tracking (for 1000 bars)
        self.atr_history = deque(maxlen=1000)
        
        # Hourly trade limit enforcement
        self.ta_trades_this_hour = []
        self.state_file = Path('nq_bot/state/ta_fallback_state.json')
        self.state_file.parent.mkdir(exist_ok=True, parents=True)
        self._load_state()
        
        # Confirmation tracking
        self.pending_setup = None
        self.confirmation_waiting = False
        
        # Performance tracking
        self.signals_generated = 0
        self.successful_signals = 0
    
    def analyze(self, data: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Perform enhanced technical analysis with scoring system
        
        Args:
            data: OHLCV data (1-minute bars)
            current_price: Current market price
            
        Returns:
            Trading signal with score ≥7.5, or None
        """
        if len(data) < 200:
            logger.debug("Insufficient data for technical analysis")
            return None
        
        # Check hourly trade limit
        if not self._check_hourly_limit():
            logger.info("TA fallback blocked: hourly trade limit reached")
            return None
        
        # Calculate all indicators
        indicators = self._calculate_indicators(data)
        
        # Update support/resistance and levels
        self._update_levels(data, indicators)
        
        # Calculate precise score (0-10 scale)
        score, score_breakdown = self._calculate_score(indicators, current_price)
        
        # Only proceed if score >= 7.5
        if score < MIN_SCORE_THRESHOLD:
            logger.debug(f"TA score {score:.1f} below threshold {MIN_SCORE_THRESHOLD}")
            return None
        
        # Determine trade direction
        direction = self._determine_direction(indicators, score_breakdown)
        
        if direction is None:
            return None
        
        # Check for confirmation if required
        if self.confirmation_waiting and self.pending_setup:
            return self._check_confirmation(data, current_price, indicators)
        
        # Create signal
        signal = self._create_enhanced_signal(
            direction, 
            current_price, 
            score,
            score_breakdown,
            indicators
        )
        
        if signal:
            # Set up for confirmation
            self.pending_setup = signal
            self.confirmation_waiting = True
            logger.info(f"TA signal setup: {direction} with score {score:.1f}, waiting for confirmation")
            
            # Record trade time
            self.ta_trades_this_hour.append(datetime.now(timezone.utc))
            self._save_state()
            self.signals_generated += 1
        
        return None  # Return None until confirmation
    
    def _calculate_score(self, indicators: Dict, current_price: float) -> Tuple[float, Dict]:
        """
        Calculate precise 0-10 score based on exact point system
        
        Args:
            indicators: Technical indicators
            current_price: Current market price
            
        Returns:
            Tuple of (total_score, score_breakdown)
        """
        score = 0.0
        breakdown = {}
        
        # 1. MA trend breadth (20>50>200): +2 points
        if indicators['ma_fast'] > indicators['ma_medium'] > indicators['ma_slow']:
            score += MA_TREND_POINTS
            breakdown['ma_trend'] = {'points': MA_TREND_POINTS, 'desc': 'Bullish MA alignment'}
        elif indicators['ma_fast'] < indicators['ma_medium'] < indicators['ma_slow']:
            score += MA_TREND_POINTS
            breakdown['ma_trend'] = {'points': MA_TREND_POINTS, 'desc': 'Bearish MA alignment'}
        
        # 2. ADX regime: +1 (18-30), +2 (≥30)
        adx = indicators.get('adx', 0)
        if adx >= 30:
            score += ADX_STRONG_POINTS
            breakdown['adx'] = {'points': ADX_STRONG_POINTS, 'desc': f'Strong trend (ADX={adx:.1f})'}
        elif adx >= 18:
            score += ADX_MODERATE_POINTS
            breakdown['adx'] = {'points': ADX_MODERATE_POINTS, 'desc': f'Moderate trend (ADX={adx:.1f})'}
        
        # 3. RSI zones
        rsi = indicators.get('rsi', 50)
        # For longs: 48-62 +1, 62-72 +2 (skip >72)
        # For shorts: 38-52 +1, 28-38 +2 (skip <28)
        
        # Determine likely direction based on other indicators
        is_bullish = indicators['ma_fast'] > indicators['ma_medium']
        
        if is_bullish:
            if 62 <= rsi <= 72:
                score += RSI_ZONE2_POINTS
                breakdown['rsi'] = {'points': RSI_ZONE2_POINTS, 'desc': f'RSI zone 2 long ({rsi:.1f})'}
            elif 48 <= rsi <= 62:
                score += RSI_ZONE1_POINTS
                breakdown['rsi'] = {'points': RSI_ZONE1_POINTS, 'desc': f'RSI zone 1 long ({rsi:.1f})'}
            elif rsi > 72:
                breakdown['rsi'] = {'points': 0, 'desc': f'RSI too high for long ({rsi:.1f})'}
        else:
            if 28 <= rsi <= 38:
                score += RSI_ZONE2_POINTS
                breakdown['rsi'] = {'points': RSI_ZONE2_POINTS, 'desc': f'RSI zone 2 short ({rsi:.1f})'}
            elif 38 <= rsi <= 52:
                score += RSI_ZONE1_POINTS
                breakdown['rsi'] = {'points': RSI_ZONE1_POINTS, 'desc': f'RSI zone 1 short ({rsi:.1f})'}
            elif rsi < 28:
                breakdown['rsi'] = {'points': 0, 'desc': f'RSI too low for short ({rsi:.1f})'}
        
        # 4. Bollinger alignment: +1
        bb_position = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        if is_bullish and 0.2 <= bb_position <= 0.5:
            score += BOLLINGER_POINTS
            breakdown['bollinger'] = {'points': BOLLINGER_POINTS, 'desc': 'Good BB position for long'}
        elif not is_bullish and 0.5 <= bb_position <= 0.8:
            score += BOLLINGER_POINTS
            breakdown['bollinger'] = {'points': BOLLINGER_POINTS, 'desc': 'Good BB position for short'}
        
        # 5. ATR percentile 30-80th: +1
        atr_percentile = self._get_atr_percentile(indicators['atr'])
        if 30 <= atr_percentile <= 80:
            score += ATR_PERCENTILE_POINTS
            breakdown['atr_percentile'] = {'points': ATR_PERCENTILE_POINTS, 'desc': f'ATR in range ({atr_percentile:.0f}th percentile)'}
        
        # 6. Stoch cross aligned: +1
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if is_bullish and stoch_k > stoch_d and stoch_k < 80:
            score += STOCH_CROSS_POINTS
            breakdown['stoch'] = {'points': STOCH_CROSS_POINTS, 'desc': 'Bullish stoch cross'}
        elif not is_bullish and stoch_k < stoch_d and stoch_k > 20:
            score += STOCH_CROSS_POINTS
            breakdown['stoch'] = {'points': STOCH_CROSS_POINTS, 'desc': 'Bearish stoch cross'}
        
        # 7. Volume z-score ≥ +1: +1
        volume_zscore = self._calculate_volume_zscore(indicators.get('volume_data', []))
        if volume_zscore >= 1.0:
            score += VOLUME_ZSCORE_POINTS
            breakdown['volume'] = {'points': VOLUME_ZSCORE_POINTS, 'desc': f'High volume (z={volume_zscore:.1f})'}
        
        # 8. Level confluence (4-6 ticks): +2
        confluence_found = self._check_level_confluence(current_price)
        if confluence_found:
            score += LEVEL_CONFLUENCE_POINTS
            breakdown['confluence'] = {'points': LEVEL_CONFLUENCE_POINTS, 'desc': confluence_found}
        
        return score, breakdown
    
    def _check_level_confluence(self, current_price: float) -> Optional[str]:
        """
        Check if price is within 4-6 ticks of key levels
        
        Args:
            current_price: Current market price
            
        Returns:
            Description of confluence if found, None otherwise
        """
        min_distance = 4 * self.tick_size  # 4 ticks
        max_distance = 6 * self.tick_size  # 6 ticks
        
        levels = []
        
        # Check VWAP
        if self.vwap:
            dist = abs(current_price - self.vwap)
            if min_distance <= dist <= max_distance:
                levels.append(f"VWAP ({dist/self.tick_size:.1f} ticks)")
        
        # Check overnight high/low
        if self.onh:
            dist = abs(current_price - self.onh)
            if min_distance <= dist <= max_distance:
                levels.append(f"ONH ({dist/self.tick_size:.1f} ticks)")
        
        if self.onl:
            dist = abs(current_price - self.onl)
            if min_distance <= dist <= max_distance:
                levels.append(f"ONL ({dist/self.tick_size:.1f} ticks)")
        
        # Check POC
        if self.poc:
            dist = abs(current_price - self.poc)
            if min_distance <= dist <= max_distance:
                levels.append(f"POC ({dist/self.tick_size:.1f} ticks)")
        
        # Check pivot points
        if self.pivot_points:
            for level_name, level_value in self.pivot_points.items():
                dist = abs(current_price - level_value)
                if min_distance <= dist <= max_distance:
                    levels.append(f"{level_name.upper()} ({dist/self.tick_size:.1f} ticks)")
        
        if levels:
            return f"Near {', '.join(levels)}"
        
        return None
    
    def _get_atr_percentile(self, current_atr: float) -> float:
        """
        Calculate ATR percentile rank
        
        Args:
            current_atr: Current ATR value
            
        Returns:
            Percentile rank (0-100)
        """
        self.atr_history.append(current_atr)
        
        if len(self.atr_history) < 100:
            return 50  # Default to middle if not enough history
        
        return stats.percentileofscore(list(self.atr_history), current_atr)
    
    def _calculate_volume_zscore(self, volume_data: List[float]) -> float:
        """
        Calculate volume z-score
        
        Args:
            volume_data: Recent volume data
            
        Returns:
            Z-score of current volume
        """
        if len(volume_data) < 20:
            return 0
        
        volume_array = np.array(volume_data)
        current_volume = volume_array[-1]
        mean_volume = np.mean(volume_array[:-1])
        std_volume = np.std(volume_array[:-1])
        
        if std_volume == 0:
            return 0
        
        return (current_volume - mean_volume) / std_volume
    
    def _determine_direction(self, indicators: Dict, score_breakdown: Dict) -> Optional[str]:
        """
        Determine trade direction based on indicators and score
        
        Args:
            indicators: Technical indicators
            score_breakdown: Score breakdown
            
        Returns:
            'BUY', 'SELL', or None
        """
        # Check RSI extremes
        rsi = indicators.get('rsi', 50)
        if rsi > 72:
            return None  # Skip longs when RSI > 72
        if rsi < 28:
            return None  # Skip shorts when RSI < 28
        
        # Determine based on MA alignment and other factors
        if indicators['ma_fast'] > indicators['ma_medium']:
            if 'ma_trend' in score_breakdown or rsi >= 48:
                return 'BUY'
        elif indicators['ma_fast'] < indicators['ma_medium']:
            if 'ma_trend' in score_breakdown or rsi <= 52:
                return 'SELL'
        
        return None
    
    def _check_hourly_limit(self) -> bool:
        """
        Check if we can trade based on hourly limit
        
        Returns:
            True if trading allowed, False if limit reached
        """
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        # Filter trades within the last hour
        self.ta_trades_this_hour = [
            trade_time for trade_time in self.ta_trades_this_hour 
            if trade_time > hour_ago
        ]
        
        return len(self.ta_trades_this_hour) < MAX_TRADES_PER_HOUR
    
    def _check_confirmation(self, data: pd.DataFrame, current_price: float, 
                           indicators: Dict) -> Optional[Dict]:
        """
        Check for confirmation close
        
        Args:
            data: OHLCV data
            current_price: Current market price
            indicators: Technical indicators
            
        Returns:
            Signal if confirmed, None otherwise
        """
        if not self.pending_setup:
            return None
        
        # Check if confirmation bar closed through trigger
        current_bar = data.iloc[-1]
        direction = self.pending_setup['action']
        trigger_price = self.pending_setup.get('trigger_price', current_price)
        
        confirmed = False
        if direction == 'BUY':
            confirmed = current_bar['close'] > trigger_price
        else:
            confirmed = current_bar['close'] < trigger_price
        
        if confirmed:
            logger.info(f"TA confirmation complete at {current_price:.2f}")
            signal = self.pending_setup
            self.pending_setup = None
            self.confirmation_waiting = False
            return signal
        
        # Check timeout (5 bars)
        # This would need bar counting logic in production
        
        return None
    
    def _create_enhanced_signal(self, action: str, price: float, score: float,
                               score_breakdown: Dict, indicators: Dict) -> Dict:
        """
        Create an enhanced trading signal with scoring details
        
        Args:
            action: 'BUY' or 'SELL'
            price: Entry price
            score: Total score (0-10)
            score_breakdown: Detailed score breakdown
            indicators: Technical indicators
            
        Returns:
            Signal dictionary
        """
        # Calculate dynamic stop and target based on ATR
        atr = indicators.get('atr', 10)
        stop_distance = max(8 * self.tick_size, atr * 1.5)
        target_distance = max(6 * self.tick_size, atr * 2.0)
        
        if action == 'BUY':
            stop_loss = price - stop_distance
            take_profit = price + target_distance
            trigger_price = price + self.tick_size  # Trigger level for confirmation
        else:
            stop_loss = price + stop_distance
            take_profit = price - target_distance
            trigger_price = price - self.tick_size
        
        # Determine signal strength based on score
        if score >= 9.0:
            strength = SignalStrength.STRONG
        elif score >= MIN_SCORE_THRESHOLD:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Build reason string
        reasons = []
        for component, details in score_breakdown.items():
            if details['points'] > 0:
                reasons.append(f"{details['desc']} (+{details['points']})")
        
        return {
            'source': 'technical_analysis_fallback',
            'action': action,
            'entry_price': price,
            'trigger_price': trigger_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': min(score / 10.0, 1.0),
            'strength': strength.value,
            'score': round(score, 1),
            'score_breakdown': score_breakdown,
            'reasons': reasons,
            'indicators': {
                'rsi': round(indicators['rsi'], 1),
                'adx': round(indicators.get('adx', 0), 1),
                'ma_fast': round(indicators['ma_fast'], 2),
                'ma_medium': round(indicators['ma_medium'], 2),
                'ma_slow': round(indicators['ma_slow'], 2)
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'uses_confirmation': True
        }
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary of indicators
        """
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        indicators = {}
        
        # RSI
        indicators['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)[-1]
        
        # Moving Averages
        indicators['ma_fast'] = talib.SMA(close, timeperiod=self.ma_fast)[-1]
        indicators['ma_medium'] = talib.SMA(close, timeperiod=self.ma_medium)[-1]
        indicators['ma_slow'] = talib.SMA(close, timeperiod=self.ma_slow)[-1] if len(close) >= self.ma_slow else indicators['ma_medium']
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        indicators['bb_upper'] = upper[-1]
        indicators['bb_middle'] = middle[-1]
        indicators['bb_lower'] = lower[-1]
        
        # ATR
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)[-1]
        
        # ADX
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)[-1]
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, 
                                   fastk_period=self.stoch_period,
                                   slowk_period=3,
                                   slowd_period=3)
        indicators['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50
        indicators['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else 50
        
        # Volume
        volume_ma = talib.SMA(volume, timeperiod=20)
        indicators['volume_ratio'] = volume[-1] / (volume_ma[-1] + 1e-10)
        indicators['volume_data'] = volume[-20:].tolist() if len(volume) >= 20 else volume.tolist()
        
        # Current price
        indicators['current_price'] = close[-1]
        
        return indicators
    
    def _update_levels(self, data: pd.DataFrame, indicators: Dict):
        """
        Update key price levels
        
        Args:
            data: OHLCV data
            indicators: Technical indicators
        """
        # Calculate VWAP
        if 'volume' in data.columns and len(data) > 0:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            cumulative_tpv = (typical_price * data['volume']).cumsum()
            cumulative_volume = data['volume'].cumsum()
            self.vwap = cumulative_tpv.iloc[-1] / cumulative_volume.iloc[-1] if cumulative_volume.iloc[-1] > 0 else None
        
        # Calculate overnight high/low (simplified)
        if len(data) >= 390:
            overnight_data = data.iloc[-780:-390] if len(data) >= 780 else data.iloc[:len(data)//2]
            if len(overnight_data) > 0:
                self.onh = overnight_data['high'].max()
                self.onl = overnight_data['low'].min()
        
        # Simple POC (highest volume price)
        if 'volume' in data.columns and len(data) > 50:
            price_levels = pd.cut(data['close'], bins=20)
            volume_profile = data.groupby(price_levels)['volume'].sum()
            if len(volume_profile) > 0:
                max_volume_bin = volume_profile.idxmax()
                if max_volume_bin is not pd.NaT and max_volume_bin is not None:
                    self.poc = max_volume_bin.mid
        
        # Calculate pivot points
        if len(data) >= 2:
            prev_high = data['high'].iloc[-2]
            prev_low = data['low'].iloc[-2]
            prev_close = data['close'].iloc[-2]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            self.pivot_points = {
                'pivot': pivot,
                'r1': 2 * pivot - prev_low,
                's1': 2 * pivot - prev_high
            }
    
    def _save_state(self):
        """Save state to file"""
        state = {
            'ta_trades_this_hour': [t.isoformat() for t in self.ta_trades_this_hour],
            'signals_generated': self.signals_generated,
            'successful_signals': self.successful_signals
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Parse trade times
                self.ta_trades_this_hour = [
                    datetime.fromisoformat(t) for t in state.get('ta_trades_this_hour', [])
                ]
                
                # Clean old trades
                self._check_hourly_limit()
                
                self.signals_generated = state.get('signals_generated', 0)
                self.successful_signals = state.get('successful_signals', 0)
                
            except Exception as e:
                logger.error(f"Error loading TA state: {e}")
    
    def get_market_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get a summary of current market conditions with scoring
        
        Args:
            data: OHLCV data
            
        Returns:
            Market summary with score
        """
        if len(data) < 200:
            return {'status': 'insufficient_data'}
        
        indicators = self._calculate_indicators(data)
        self._update_levels(data, indicators)
        
        current_price = data['close'].iloc[-1]
        score, breakdown = self._calculate_score(indicators, current_price)
        
        return {
            'score': round(score, 1),
            'threshold': MIN_SCORE_THRESHOLD,
            'can_trade': score >= MIN_SCORE_THRESHOLD,
            'trades_this_hour': len(self.ta_trades_this_hour),
            'max_trades_per_hour': MAX_TRADES_PER_HOUR,
            'score_breakdown': breakdown,
            'indicators': {
                'rsi': round(indicators['rsi'], 1),
                'adx': round(indicators.get('adx', 0), 1),
                'ma_trend': 'bullish' if indicators['ma_fast'] > indicators['ma_medium'] > indicators['ma_slow'] else 'bearish' if indicators['ma_fast'] < indicators['ma_medium'] < indicators['ma_slow'] else 'neutral'
            },
            'levels': {
                'vwap': round(self.vwap, 2) if self.vwap else None,
                'onh': round(self.onh, 2) if self.onh else None,
                'onl': round(self.onl, 2) if self.onl else None,
                'poc': round(self.poc, 2) if self.poc else None
            }
        }
    
    def update_signal_result(self, was_successful: bool):
        """
        Update signal performance tracking
        
        Args:
            was_successful: Whether the signal was profitable
        """
        if was_successful:
            self.successful_signals += 1
        
        self._save_state()
        
        success_rate = (self.successful_signals / self.signals_generated * 100 
                       if self.signals_generated > 0 else 0)
        
        logger.info(f"TA Fallback Performance - Signals: {self.signals_generated}, "
                   f"Success Rate: {success_rate:.1f}%")