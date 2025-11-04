"""
Trend Line Bounce Scalping Pattern for NQ
Sophisticated pattern that scalps 3-5 ticks on trend line bounces
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, time
import logging
import json
from pathlib import Path

import sys
import os
# Add parent directory to path for accessing utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base_pattern import BasePattern, PatternSignal, TradeAction, EntryPlan
from ..utils.trend_line_detector import TrendLineDetector, TrendLine
from ..utils.candles import get_candle_guard
from ..pattern_config import CANDLES, EXECUTE_ON_SCORE_ONLY, MIN_PASS_SCORE_DISCOVERY, FORCE_IMMEDIATE_MARKET

logger = logging.getLogger(__name__)

class TrendLineBouncePattern(BasePattern):
    """
    Trend line bounce scalping pattern
    Trades the 3rd touch of validated trend lines with multi-timeframe confluence
    """
    
    def _initialize(self):
        """Initialize pattern-specific components"""
        # Extract target tick values from config
        self.target_ticks_normal = self.config.get('target_ticks_normal', 3)
        self.target_ticks_high_conf = self.config.get('target_ticks_high_conf', 5)

        # Trend line detector
        self.trend_detector = TrendLineDetector(
            min_touches=2,
            max_lines_per_direction=3,
            min_r_squared=0.95,
            max_angle_degrees=75,
            touch_tolerance_pct=0.001
        )

        # State tracking for confirmation
        self.pending_setup = None
        self.confirmation_waiting = False
        self.trigger_level = None
        self.pivot_level = None
        
        # Multi-timeframe data storage
        self.timeframe_data = {
            '1m': pd.DataFrame(),
            '5m': pd.DataFrame(),
            '1h': pd.DataFrame()
        }
        
        # Pattern-specific configuration
        self.tick_size = 0.25  # NQ tick size
        self.point_value = 20  # NQ point value
        
        # Pattern-specific risk parameters - OPTIMIZED FOR TOPSTEPX $150 TARGET
        self.TLB_STOP_MIN_TICKS = 25  # $125 stop = 25 ticks
        self.TLB_STOP_ATR_MULTIPLIER = 0.5
        self.TLB_T1_TICKS = 30  # $150 target = 30 ticks (primary target)
        self.TLB_T2_TICKS = 30  # Single target approach for TopStepX brackets
        
        # Override with config if provided
        self.stop_min_ticks = self.config.get('stop_min_ticks', self.TLB_STOP_MIN_TICKS)
        self.stop_atr_multiplier = self.config.get('stop_atr_multiplier', self.TLB_STOP_ATR_MULTIPLIER)
        self.target1_ticks = self.config.get('target1_ticks', self.TLB_T1_TICKS)
        self.target2_ticks = self.config.get('target2_ticks', self.TLB_T2_TICKS)
        
        # Engulfing candle parameters
        self.atr_period = 14
        self.engulfing_atr_multiplier = 1.5
        self.engulfing_volume_multiplier = 2.0
        
        # Time restrictions (ET)
        self.no_trade_start = time(8, 30)  # 8:30 AM ET
        self.no_trade_end = time(9, 15)    # 9:15 AM ET
        
        # State tracking
        self.active_trend_lines = {'support': [], 'resistance': []}
        self.nearest_line_distance = float('inf')
        self.current_confluence_score = 0.0
        self.x_zones = []  # Trend line intersections
        
        # Performance tracking
        self.pattern_trades = []
        self.consecutive_wins = 0
        
        # Load saved trend lines if they exist
        self._load_trend_lines()
    
    def update_timeframe_data(self, timeframe: str, data: pd.DataFrame):
        """
        Update data for a specific timeframe
        
        Args:
            timeframe: '1m', '5m', or '1h'
            data: OHLCV DataFrame for that timeframe
        """
        if timeframe in self.timeframe_data:
            self.timeframe_data[timeframe] = data.copy()
            logger.debug(f"Updated {timeframe} data: {len(data)} bars")
    
    def scan_for_setup(self, data: pd.DataFrame, current_price: float) -> Optional[PatternSignal]:
        """
        Enhanced scan with confirmation and quality checks
        
        Args:
            data: 1-minute OHLCV data
            current_price: Current market price
            
        Returns:
            PatternSignal if pattern detected with quality confirmation, None otherwise
        """
        if len(data) < 100:
            return None
        
        # Update 1m data
        self.timeframe_data['1m'] = data.copy()
        
        # Check time restrictions
        if not self._is_trading_time():
            logger.debug("Outside trading hours for pattern")
            return None
        
        # Get ATR from data cache if available
        if self.data_cache:
            atr_value = self.data_cache.get_indicator('atr', '1m')
        else:
            # Calculate ATR if no cache
            high = data['high'].values.astype(np.float64)
            low = data['low'].values.astype(np.float64)
            close = data['close'].values.astype(np.float64)
            atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
            atr_value = atr[-1] if len(atr) > 0 else 0
        
        # Check if we're waiting for confirmation
        if self.confirmation_waiting and self.pending_setup:
            return self._check_confirmation(data, current_price, atr_value)
        
        # Look for new setup
        return self._scan_for_new_setup(data, current_price, atr_value)
        
    def _scan_for_new_setup(self, data: pd.DataFrame, current_price: float, atr: float) -> Optional[PatternSignal]:
        """Scan for new trend line bounce setup"""
        
        # Write EVAL telemetry at start of scan
        try:
            from ..utils.telemetry_sink import get_telemetry_sink
            sink = get_telemetry_sink()
            if sink:
                sink.write_telemetry('EVAL', {
                    'pattern': 'TLB',
                    'current_price': current_price,
                    'atr': atr
                })
        except Exception as e:
            logger.debug(f"TLB telemetry write failed: {e}")
        
        # Detect trend lines (cached for 10 seconds)
        lines = self.trend_detector.detect_trend_lines(data)
        self.active_trend_lines = lines
        
        # Find nearest trend line
        current_index = len(data) - 1
        nearest = self.trend_detector.find_nearest_line(current_index, current_price)
        
        if not nearest:
            return None
        
        nearest_line, distance = nearest
        self.nearest_line_distance = distance
        
        # Check if price is close enough to trend line (within 2 ticks)
        if distance > self.tick_size * 2:
            return None
        
        # Check if this is the 2nd or 3rd touch (relaxed from 3rd only)
        touch_count = len(nearest_line.touch_points)
        if touch_count < 2:
            logger.debug(f"Not enough touches: line has {touch_count} touches")
            return None
        
        # Accept both 2nd and 3rd touch for more frequent signals
        if not (self.trend_detector.is_second_touch(nearest_line, current_index, current_price) or 
                self.trend_detector.is_third_touch(nearest_line, current_index, current_price)):
            logger.debug(f"Not valid touch: line has {touch_count} touches")
            return None
        
        # Store pending setup for confirmation
        if nearest_line.line_type == 'support':
            self.trigger_level = data.iloc[-1]['high']
            self.pivot_level = data.iloc[-2]['low']
            direction = "support"
            is_long = True
        else:  # resistance
            self.trigger_level = data.iloc[-1]['low']
            self.pivot_level = data.iloc[-2]['high']
            direction = "resistance"
            is_long = False
        
        # Calculate multi-timeframe confluence
        confluence_score = self._calculate_confluence_score(current_price, nearest_line)
        self.current_confluence_score = confluence_score
        
        # Check for X-zone (trend line intersection)
        x_zone_setup = self._check_x_zone_setup(current_price, current_index)
        
        # Store pending setup
        self.pending_setup = {
            'direction': direction,
            'is_long': is_long,
            'nearest_line': nearest_line,
            'distance': distance,
            'confluence_score': confluence_score,
            'x_zone_setup': x_zone_setup,
            'setup_time': datetime.now(timezone.utc),
            'setup_bar_index': current_index,
            'touch_count': touch_count
        }
        
        # Check for score-only immediate execution (Conservative for TLB: require higher threshold)
        tlb_score_threshold = MIN_PASS_SCORE_DISCOVERY * 1.5 if MIN_PASS_SCORE_DISCOVERY else 0.525
        if (EXECUTE_ON_SCORE_ONLY and FORCE_IMMEDIATE_MARKET and 
            confluence_score >= tlb_score_threshold):
            # Skip confirmation requirement for high-scoring setups
            logger.info(f"TLB score-only PASS: score={confluence_score:.3f} >= {tlb_score_threshold:.3f}, "
                       f"direction={direction}, touch={touch_count}")
            
            # Write PATTERN_PASS telemetry
            try:
                from ..utils.telemetry_sink import get_telemetry_sink
                sink = get_telemetry_sink()
                if sink:
                    sink.write_telemetry('PATTERN_PASS', {
                        'pattern': 'TLB',
                        'score': confluence_score,
                        'threshold': tlb_score_threshold,
                        'direction': direction,
                        'touch_count': touch_count,
                        'score_only': True
                    })
            except Exception as e:
                logger.debug(f"TLB PATTERN_PASS telemetry failed: {e}")
            
            self.confirmation_waiting = False
            return self._create_signal(data, current_price, atr, score_only=True)
        
        # If confirmation is required, wait for it
        if self.require_confirmation_close:
            self.confirmation_waiting = True
            logger.info(f"Trend line bounce setup detected ({direction}), waiting for confirmation close through {self.trigger_level:.2f}")
            return None
        else:
            # No confirmation required, signal immediately
            return self._create_signal(data, current_price, atr)
        
    def _check_confirmation(self, data: pd.DataFrame, current_price: float, atr: float) -> Optional[PatternSignal]:
        """Check if confirmation requirements are met"""
        
        if not self.pending_setup or not self.confirmation_waiting:
            return None
        
        current_bar = data.iloc[-1]
        is_long = self.pending_setup['is_long']
        
        # Check if too much time has passed (e.g., 5 bars)
        bars_since_setup = len(data) - 1 - self.pending_setup['setup_bar_index']
        if bars_since_setup > 5:
            logger.info("Setup expired without confirmation")
            self._reset_setup()
            return None
        
        # Check for confirmation close
        confirmed = False
        if is_long:
            confirmed = current_bar['close'] > self.trigger_level
        else:
            confirmed = current_bar['close'] < self.trigger_level
        
        if confirmed:
            # Check if confirmation bar is exhaustion
            confirm_bar_range = current_bar['high'] - current_bar['low']
            is_exhaustion = self.exhaustion_check(confirm_bar_range, atr)
            
            if is_exhaustion:
                logger.warning(f"Confirmation bar shows exhaustion (range={confirm_bar_range:.2f}, ATR={atr:.2f})")
                self._reset_setup()
                return None
            
            # Check for micro-pullback
            pullback_achieved, pullback_level = self.micro_pullback_check(
                data.iloc[-3:], self.pivot_level, is_long
            )
            
            if not pullback_achieved:
                logger.info(f"Waiting for micro-pullback to {pullback_level:.2f}")
                return None  # Keep waiting
            
            # Check for dangerous engulfing
            if self.dangerous_engulfing_check(data, atr, is_long):
                logger.warning("Dangerous engulfing detected, skipping signal")
                self._reset_setup()
                return None
            
            # Section 6.5: CANDLE GUARD - Apply veto/boost
            if CANDLES.get('enable', False):
                candle_guard = get_candle_guard()
                
                # Prepare context for guard decision
                is_long = self.pending_setup['is_long']
                side = 'long' if is_long else 'short'
                near_level = self.pending_setup['distance'] <= CANDLES.get('near_level_ticks', 6) * self.tick_size
                
                # Get volume info
                avg_vol = data['volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else data['volume'].mean()
                vol_ratio = data.iloc[-1]['volume'] / avg_vol if avg_vol > 0 else 1.0
                
                context = {
                    'atr': atr,
                    'vol_ratio': vol_ratio,
                    'tick_size': self.tick_size,
                    'bars': data.tail(10).to_dict('records'),
                    'symbol': 'NQ',
                    'timestamp': datetime.now(),
                    'avg_vol': avg_vol
                }
                
                decision = candle_guard.guard_decision(side, near_level, CANDLES, context)
                
                # Check for hard veto
                if decision['hard_veto']:
                    candle = decision.get('candle')
                    if candle:
                        logger.info(f"CANDLE_VETO pattern=trend_line_bounce type={candle.kind} "
                                  f"strength={candle.strength:.2f} near_level={near_level} dangerous={candle.is_dangerous}")
                    self._reset_setup()
                    return None
                
                # Apply soft bonus/penalty to confluence score
                soft_bonus = decision.get('soft_bonus', 0.0)
                if soft_bonus != 0:
                    orig_score = self.pending_setup['confluence_score']
                    new_score = max(0.0, min(1.0, orig_score + soft_bonus))
                    self.pending_setup['confluence_score'] = new_score
                    
                    candle = decision.get('candle')
                    if candle:
                        logger.info(f"CANDLE_ADJUST pattern=trend_line_bounce type={candle.kind} "
                                  f"bonus={soft_bonus:+.2f} score={orig_score:.2f}->{new_score:.2f}")
            
            # All checks passed, create signal
            logger.info(f"Confirmation complete with quality checks passed")
            signal = self._create_signal(data, current_price, atr,
                                        confirm_bar_range=confirm_bar_range,
                                        is_exhaustion=is_exhaustion,
                                        pullback_achieved=pullback_achieved)
            
            self._reset_setup()
            return signal
        
        return None
    
    def _create_signal(self, data: pd.DataFrame, current_price: float, atr: float,
                      confirm_bar_range: float = None, is_exhaustion: bool = False,
                      pullback_achieved: bool = True, score_only: bool = False) -> PatternSignal:
        """Create pattern signal with entry plan"""
        
        setup = self.pending_setup
        is_long = setup['is_long']
        nearest_line = setup['nearest_line']
        distance = setup['distance']
        confluence_score = setup['confluence_score']
        x_zone_setup = setup['x_zone_setup']
        
        # Calculate entry levels
        if confirm_bar_range:
            # Retest entry at trigger or 50% of confirmation bar
            current_bar = data.iloc[-1]
            if is_long:
                fifty_percent = current_bar['low'] + (confirm_bar_range * 0.5)
                retest_entry = max(self.trigger_level, fifty_percent)
            else:
                fifty_percent = current_bar['high'] - (confirm_bar_range * 0.5)
                retest_entry = min(self.trigger_level, fifty_percent)
        else:
            retest_entry = self.trigger_level
        
        # Set entry price
        entry_price = retest_entry
        
        # Find swing level for stop placement
        swing_level = self._find_swing_level(data, is_long, nearest_line)
        
        # Calculate ATR-based stop
        atr_value = self._get_current_atr(data)
        atr_stop_distance = atr_value * self.stop_atr_multiplier
        min_stop_distance = self.stop_min_ticks * self.tick_size
        
        # Use max of minimum ticks or ATR-based stop
        stop_distance = max(min_stop_distance, atr_stop_distance)
        
        # Determine trade direction and stops
        if is_long:
            action = TradeAction.BUY
            # Stop beyond swing or using calculated distance
            stop_loss = min(swing_level - (2 * self.tick_size), entry_price - stop_distance)
        else:
            action = TradeAction.SELL
            # Stop beyond swing or using calculated distance
            stop_loss = max(swing_level + (2 * self.tick_size), entry_price + stop_distance)
        
        # Determine targets based on market conditions
        clean_trend = self._is_clean_trend(data, is_long)
        
        if clean_trend and confluence_score >= 0.85:
            # Single target at T2 in clean trends
            position_size = 2
            target1 = entry_price + (self.target2_ticks * self.tick_size) if is_long else entry_price - (self.target2_ticks * self.tick_size)
            target2 = target1  # Same as T1 for single target
            mode = "score_only" if score_only else "clean"
            reason = f"TLB {mode} ({setup['touch_count']} touches)"
        else:
            # Standard two-target approach
            position_size = 1
            if is_long:
                target1 = entry_price + (self.target1_ticks * self.tick_size)
                target2 = entry_price + (self.target2_ticks * self.tick_size)
            else:
                target1 = entry_price - (self.target1_ticks * self.tick_size)
                target2 = entry_price - (self.target2_ticks * self.tick_size)
            mode = "score_only" if score_only else "standard"
            reason = f"TLB {mode} ({setup['touch_count']} touches)"
        
        # Use T2 as primary target
        take_profit = target2
        
        # Adjust confidence based on quality checks
        if is_exhaustion:
            confluence_score *= 0.8  # Reduce if exhaustion
        if not pullback_achieved:
            confluence_score *= 0.9  # Reduce if no pullback
        
        # Create entry plan
        entry_plan = EntryPlan(
            trigger_price=self.trigger_level,
            confirm_price=data.iloc[-1]['close'],
            retest_entry=retest_entry,
            confirm_bar_range=confirm_bar_range or 0,
            is_exhaustion=is_exhaustion,
            pullback_achieved=pullback_achieved
        )
        
        # Create signal with target levels
        signal = PatternSignal(
            pattern_name="TrendLineBounce",
            action=action,
            confidence=confluence_score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reason=reason,
            confluence_score=confluence_score,
            timeframe_signals=self._get_timeframe_signals(current_price, nearest_line),
            entry_plan=entry_plan,
            # Add target levels for position management
            target1=target1,
            target2=target2,
            swing_level=swing_level
        )
        
        # Log detailed pattern trigger
        self._log_pattern_trigger(signal, nearest_line, distance)
        
        return signal
    
    def _reset_setup(self):
        """Reset pending setup state"""
        self.pending_setup = None
        self.confirmation_waiting = False
        self.trigger_level = None
        self.pivot_level = None
    
    def calculate_confidence(self, data: pd.DataFrame, signal: PatternSignal) -> float:
        """
        Calculate confidence score for pattern signal
        
        Args:
            data: Price data
            signal: Pattern signal
            
        Returns:
            Confidence score (0-1)
        """
        # Confidence is already calculated in scan_for_setup via confluence score
        return signal.confluence_score
    
    def _is_trading_time(self) -> bool:
        """Check if current time is within trading hours"""
        # Convert to ET (assume system is in UTC)
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(timezone.utc).astimezone(ZoneInfo('America/New_York'))
        except ImportError:
            # Fallback to pytz if zoneinfo not available
            import pytz
            now_et = datetime.now(timezone.utc).astimezone(pytz.timezone('America/New_York'))
        current_time = now_et.time()
        
        # Check if in restricted period
        if self.no_trade_start <= current_time <= self.no_trade_end:
            return False
        
        return True
    
    def _is_dangerous_engulfing(self, data: pd.DataFrame) -> bool:
        """
        DEPRECATED: Use dangerous_engulfing_check from base class instead
        Kept for backward compatibility
        """
        if len(data) < self.atr_period + 2:
            return False
        
        # Calculate ATR
        atr = talib.ATR(data['high'].values, 
                       data['low'].values, 
                       data['close'].values, 
                       timeperiod=self.atr_period)
        
        if np.isnan(atr[-1]):
            return False
        
        # Use base class method with enhanced checks
        is_long = self.pending_setup['is_long'] if self.pending_setup else True
        return self.dangerous_engulfing_check(data, atr[-1], is_long)
    
    def _calculate_confluence_score(self, current_price: float, trend_line: TrendLine) -> float:
        """
        Calculate multi-timeframe confluence score
        
        Args:
            current_price: Current market price
            trend_line: The trend line being tested
            
        Returns:
            Confluence score (0-1)
        """
        score = 0.0
        weights = {
            '1m': 0.4,
            '5m': 0.3,
            '1h': 0.3
        }
        
        for timeframe, weight in weights.items():
            if timeframe not in self.timeframe_data or self.timeframe_data[timeframe].empty:
                continue
            
            tf_data = self.timeframe_data[timeframe]
            
            # Check trend direction
            if len(tf_data) >= 20:
                sma_20 = tf_data['close'].rolling(20).mean().iloc[-1]
                ema_9 = tf_data['close'].ewm(span=9).mean().iloc[-1]
                
                if trend_line.line_type == 'support':
                    # For support, we want price above moving averages
                    if current_price > sma_20 and current_price > ema_9:
                        score += weight * 0.5
                else:
                    # For resistance, we want price below moving averages
                    if current_price < sma_20 and current_price < ema_9:
                        score += weight * 0.5
            
            # Check momentum
            if len(tf_data) >= 14:
                rsi = talib.RSI(tf_data['close'].values, timeperiod=14)
                if not np.isnan(rsi[-1]):
                    if trend_line.line_type == 'support':
                        # For support bounce, RSI should not be extremely oversold
                        if 30 < rsi[-1] < 50:
                            score += weight * 0.3
                    else:
                        # For resistance bounce, RSI should not be extremely overbought
                        if 50 < rsi[-1] < 70:
                            score += weight * 0.3
            
            # Check volume
            if len(tf_data) >= 20:
                current_vol = tf_data['volume'].iloc[-1]
                avg_vol = tf_data['volume'].rolling(20).mean().iloc[-1]
                if current_vol > avg_vol * 0.8:  # Decent volume
                    score += weight * 0.2
        
        # Add trend line strength to score
        score += trend_line.strength * 0.2
        
        # Cap at 1.0
        return min(1.0, score)
    
    def _check_x_zone_setup(self, current_price: float, current_index: int) -> bool:
        """
        Check if we're at a trend line intersection (X-zone)
        
        Args:
            current_price: Current market price
            current_index: Current bar index
            
        Returns:
            True if X-zone setup is valid
        """
        # Find confluence zones
        self.x_zones = self.trend_detector.find_confluence_zones()
        
        if not self.x_zones:
            return False
        
        # Check nearest X-zone
        for zone in self.x_zones:
            # Check if we're near the intersection
            if abs(zone['index'] - current_index) < 10:  # Within 10 bars
                if abs(zone['price'] - current_price) < self.tick_size * 5:  # Within 5 ticks
                    logger.info(f"X-zone detected at {zone['price']:.2f}")
                    return True
        
        return False
    
    def _get_timeframe_signals(self, current_price: float, trend_line: TrendLine) -> Dict[str, bool]:
        """
        Get signal confirmation from each timeframe
        
        Args:
            current_price: Current price
            trend_line: Trend line being tested
            
        Returns:
            Dictionary of timeframe confirmations
        """
        signals = {}
        
        for timeframe in ['1m', '5m', '1h']:
            if timeframe not in self.timeframe_data or self.timeframe_data[timeframe].empty:
                signals[timeframe] = False
                continue
            
            tf_data = self.timeframe_data[timeframe]
            
            if len(tf_data) >= 20:
                sma_20 = tf_data['close'].rolling(20).mean().iloc[-1]
                
                if trend_line.line_type == 'support':
                    signals[timeframe] = current_price > sma_20
                else:
                    signals[timeframe] = current_price < sma_20
            else:
                signals[timeframe] = False
        
        return signals
    
    def _log_pattern_trigger(self, signal: PatternSignal, trend_line: TrendLine, distance: float):
        """Log detailed pattern trigger information"""
        log_msg = f"""
TrendLineBounce pattern triggered
- Confidence: {signal.confidence:.2f} ({'HIGH' if signal.confidence >= 0.85 else 'NORMAL'})
- Trend line: {trend_line.line_type.capitalize()} @ {trend_line.get_price_at_index(len(self.timeframe_data['1m'])-1):.2f} ({len(trend_line.touch_points)} touches)
- Distance: {distance/self.tick_size:.1f} ticks
- Multi-TF confluence: 1m={signal.timeframe_signals.get('1m', False)}, 5m={signal.timeframe_signals.get('5m', False)}, 1h={signal.timeframe_signals.get('1h', False)}
- Entry: {signal.action.value} {signal.position_size} contracts @ {signal.entry_price:.2f}
- Stop: {signal.stop_loss:.2f} ({self.stop_ticks} ticks)
- Target: {signal.take_profit:.2f} ({self.target_ticks_high_conf if signal.position_size == 2 else self.target_ticks_normal} ticks)
        """
        logger.info(log_msg)
    
    def _save_trend_lines(self):
        """Save current trend lines to file"""
        state_file = Path('state/trend_lines.json')
        state_file.parent.mkdir(exist_ok=True)
        
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'support_lines': [self._serialize_trend_line(line) for line in self.active_trend_lines.get('support', [])],
            'resistance_lines': [self._serialize_trend_line(line) for line in self.active_trend_lines.get('resistance', [])]
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_trend_lines(self):
        """Load saved trend lines from file"""
        state_file = Path('state/trend_lines.json')
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Check if state is recent (within 1 hour)
            # Parse timestamp ensuring it's timezone-aware
            timestamp_str = state['timestamp']
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            
            # Use fromisoformat with explicit timezone handling
            if '+' in timestamp_str or '-' in timestamp_str[-6:]:
                # Already has timezone info
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                # Add UTC timezone if missing
                timestamp = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            
            if age < 3600:  # 1 hour
                logger.info(f"Loading trend lines from {age:.0f} seconds ago")
                # Note: Would need to deserialize trend lines here
                # For now, we'll let them be recalculated
        except Exception as e:
            logger.error(f"Error loading trend lines: {e}")
    
    def _serialize_trend_line(self, line: TrendLine) -> Dict:
        """Serialize trend line for storage"""
        return {
            'slope': line.slope,
            'intercept': line.intercept,
            'r_squared': line.r_squared,
            'touch_points': line.touch_points,
            'line_type': line.line_type,
            'strength': line.strength,
            'angle_degrees': line.angle_degrees
        }
    
    def get_pattern_metrics(self) -> Dict:
        """Get pattern-specific metrics for monitoring"""
        return {
            'pattern_active': 'trend_line_bounce',
            'lines_detected': len(self.active_trend_lines.get('support', [])) + len(self.active_trend_lines.get('resistance', [])),
            'distance_to_nearest': self.nearest_line_distance,
            'confluence_score': self.current_confluence_score,
            'x_zones': len(self.x_zones),
            'daily_trades': self.daily_trades,
            'win_rate': (self.winning_signals / self.total_signals * 100) if self.total_signals > 0 else 0,
            'total_pnl': self.total_pnl
        }
    
    def _find_swing_level(self, data: pd.DataFrame, is_long: bool, trend_line) -> float:
        """
        Find swing level near the trend line for stop placement
        
        Args:
            data: OHLCV DataFrame
            is_long: True for long positions, False for short
            trend_line: The trend line being traded
            
        Returns:
            Swing price level
        """
        # Look at last 10 bars for swing
        lookback = min(10, len(data))
        recent_bars = data.iloc[-lookback:]
        
        if is_long:
            # Find swing low near support line
            swing = recent_bars['low'].min()
        else:
            # Find swing high near resistance line
            swing = recent_bars['high'].max()
        
        return swing
    
    def _get_current_atr(self, data: pd.DataFrame) -> float:
        """
        Get current ATR value
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Current ATR value
        """
        if self.data_cache:
            return self.data_cache.get_indicator('atr', '1m')
        else:
            # Calculate ATR if no cache
            high = data['high'].values.astype(np.float64)
            low = data['low'].values.astype(np.float64)
            close = data['close'].values.astype(np.float64)
            atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
            return atr[-1] if len(atr) > 0 else 5.0  # Default to 5 if calculation fails
    
    def _is_clean_trend(self, data: pd.DataFrame, is_long: bool) -> bool:
        """
        Determine if market is in a clean trend
        
        Args:
            data: OHLCV DataFrame
            is_long: Direction of trade
            
        Returns:
            True if clean trend detected
        """
        if len(data) < 50:
            return False
        
        # Use simple moving average alignment
        close = data['close'].values
        sma_10 = talib.SMA(close, timeperiod=10)
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        
        if is_long:
            # Clean uptrend: price > SMA10 > SMA20 > SMA50
            clean = (close[-1] > sma_10[-1] and 
                    sma_10[-1] > sma_20[-1] and 
                    sma_20[-1] > sma_50[-1])
        else:
            # Clean downtrend: price < SMA10 < SMA20 < SMA50
            clean = (close[-1] < sma_10[-1] and 
                    sma_10[-1] < sma_20[-1] and 
                    sma_20[-1] < sma_50[-1])
        
        return clean