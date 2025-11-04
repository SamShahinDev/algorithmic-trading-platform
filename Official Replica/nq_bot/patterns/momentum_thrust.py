"""
Momentum Thrust Pattern for NQ - Enhanced with Dual Confirmation
High-velocity price movements with volume confirmation
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, time
import logging
import json
from pathlib import Path

from .base_pattern import BasePattern, PatternSignal, TradeAction

logger = logging.getLogger(__name__)

class MomentumThrustPattern(BasePattern):
    """
    Momentum thrust pattern - captures strong directional moves
    Enhanced with thrust confirmation mode for discovery
    """
    
    def _initialize(self):
        """Initialize pattern-specific components"""
        # Pattern configuration from discovery
        self.lookback = self.config.get('lookback', 56)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.0014)
        self.volume_factor = self.config.get('volume_factor', 1.72)
        self.min_strength = self.config.get('min_strength', 40)
        
        # Pattern-specific risk parameters
        self.MT_STOP_MIN_TICKS = 10
        self.MT_STOP_MAX_TICKS = 12
        self.MT_T1_TICKS = 5
        self.MT_T2_TICKS = 10
        self.MT_TRAIL_MIN_TICKS = 6
        self.MT_TRAIL_MAX_TICKS = 10
        self.MT_TRAIL_ADX_THRESHOLD = 22
        
        # Override with config if provided
        self.stop_min_ticks = self.config.get('stop_min_ticks', self.MT_STOP_MIN_TICKS)
        self.stop_max_ticks = self.config.get('stop_max_ticks', self.MT_STOP_MAX_TICKS)
        self.target1_ticks = self.config.get('target1_ticks', self.MT_T1_TICKS)
        self.target2_ticks = self.config.get('target2_ticks', self.MT_T2_TICKS)
        
        # NQ specifics
        self.tick_size = 0.25
        self.point_value = 20
        
        # Technical indicators periods
        self.rsi_period = 14
        self.volume_ma_period = 20
        self.atr_period = 14
        self.adx_period = 14
        
        # State tracking
        self.last_signal_bar = -1
        self.consecutive_signals = 0
        self.pattern_trades = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
    
    def scan_for_setup(self, data: pd.DataFrame, current_price: float) -> Optional[PatternSignal]:
        """
        Scan for momentum thrust pattern with dual confirmation modes
        Always computes scores and logs telemetry
        
        Args:
            data: OHLCV data (1-minute bars)
            current_price: Current market price
            
        Returns:
            PatternSignal if pattern detected, None otherwise
        """
        # Check minimum data requirement
        if len(data) < max(self.lookback, 100):
            # Log insufficient data
            try:
                from ..utils.telemetry_sink import get_telemetry_sink
                sink = get_telemetry_sink()
                sink.write(
                    pattern='momentum_thrust',
                    event='EVAL',
                    price=current_price,
                    score=0,
                    min_score=0,
                    exec_reason='insufficient_data'
                )
            except:
                pass
            return None
        
        # Import configuration
        try:
            from ..pattern_config import (
                DISCOVERY_MODE, USE_GLOBAL_MIN_CONF_WHEN_DISCOVERY,
                GLOBAL_MIN_CONFIDENCE, PATTERN_MIN_CONFIDENCE,
                MT_MOMENTUM_THRESHOLD, MT_VOLUME_FACTOR_MIN,
                MT_RSI_LONG_MIN, MT_RSI_LONG_MAX,
                MT_RSI_SHORT_MIN, MT_RSI_SHORT_MAX,
                MT_CONFIRM_MODE, MT_BREAKOUT_LOOKBACK,
                MT_THRUST_BODY_MIN, MT_THRUST_RANGE_ATR_MIN,
                MT_THRUST_MOM_MIN, MT_THRUST_VOL_MIN,
                TELEMETRY
            )
        except ImportError:
            # Fallback if new config not available
            DISCOVERY_MODE = True
            USE_GLOBAL_MIN_CONF_WHEN_DISCOVERY = True
            GLOBAL_MIN_CONFIDENCE = 0.35
            PATTERN_MIN_CONFIDENCE = 0.60
            MT_MOMENTUM_THRESHOLD = 0.0005
            MT_VOLUME_FACTOR_MIN = 1.2
            MT_RSI_LONG_MIN = 45
            MT_RSI_LONG_MAX = 72
            MT_RSI_SHORT_MIN = 28
            MT_RSI_SHORT_MAX = 55
            MT_CONFIRM_MODE = "dual"
            MT_BREAKOUT_LOOKBACK = 20
            MT_THRUST_BODY_MIN = 0.60
            MT_THRUST_RANGE_ATR_MIN = 0.80
            MT_THRUST_MOM_MIN = 0.0003
            MT_THRUST_VOL_MIN = 1.15
            TELEMETRY = {"csv_eval_all": True}
        
        # Calculate all indicators first - convert to float64 for TA-Lib
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        open_price = data['open'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Core indicators
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        volume_ma = talib.SMA(volume, timeperiod=self.volume_ma_period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        roc = talib.ROC(close, timeperiod=10)
        
        # Current values
        current_idx = len(data) - 1
        current_close = close[-1]
        current_high = high[-1]
        current_low = low[-1]
        current_open = open_price[-1]
        current_volume = volume[-1]
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 1.0
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 20.0
        current_roc = roc[-1] if not np.isnan(roc[-1]) else 0.0
        volume_avg = volume_ma[-1] if not np.isnan(volume_ma[-1]) else 1.0
        
        # Raw inputs calculation
        momentum_raw = (current_close - close[-self.lookback]) / close[-self.lookback] if self.lookback < len(close) else 0.0
        volume_ratio = current_volume / max(volume_avg, 1.0)
        
        # Current bar metrics for thrust confirmation
        bar_range = current_high - current_low
        body_size = abs(current_close - current_open)
        body_fraction = body_size / max(bar_range, 0.0001)
        confirm_range_atr = bar_range / max(current_atr, 0.0001)
        
        # 1) Confirmation paths
        # Breakout confirmation
        if MT_BREAKOUT_LOOKBACK < len(data):
            recent_highs = high[-MT_BREAKOUT_LOOKBACK:-1]
            recent_lows = low[-MT_BREAKOUT_LOOKBACK:-1]
            breakout_up = current_close > np.max(recent_highs) if len(recent_highs) > 0 else False
            breakout_down = current_close < np.min(recent_lows) if len(recent_lows) > 0 else False
        else:
            breakout_up = False
            breakout_down = False
        
        # Thrust confirmation
        thrust_up = (body_fraction >= MT_THRUST_BODY_MIN and
                    confirm_range_atr >= MT_THRUST_RANGE_ATR_MIN and
                    momentum_raw >= MT_THRUST_MOM_MIN and
                    volume_ratio >= MT_THRUST_VOL_MIN)
        
        thrust_down = (body_fraction >= MT_THRUST_BODY_MIN and
                      confirm_range_atr >= MT_THRUST_RANGE_ATR_MIN and
                      momentum_raw <= -MT_THRUST_MOM_MIN and
                      volume_ratio >= MT_THRUST_VOL_MIN)
        
        # Select confirmation based on mode
        if MT_CONFIRM_MODE == "breakout":
            confirm_long, confirm_short = breakout_up, breakout_down
        elif MT_CONFIRM_MODE == "thrust":
            confirm_long, confirm_short = thrust_up, thrust_down
        else:  # "dual"
            confirm_long = breakout_up or thrust_up
            confirm_short = breakout_down or thrust_down
        
        # Determine confirmation type for telemetry
        if breakout_up or breakout_down:
            confirmation_type = "breakout"
        elif thrust_up or thrust_down:
            confirmation_type = "thrust"
        else:
            confirmation_type = "none"
        
        # 2) Safety filters
        exhaustion_ok = confirm_range_atr <= 1.25  # Not too extended
        pullback_ok = True  # Simplified for now
        safety_ok = exhaustion_ok and pullback_ok
        
        # 3) Helper functions for scoring
        def normalize(value, floor, target):
            """Normalize value to 0..1 range"""
            if value <= floor:
                return 0.0
            if value >= target:
                return 1.0
            return (value - floor) / (target - floor)
        
        def rsi_component(rsi_val, is_long):
            """Calculate RSI contribution to score"""
            if is_long:
                if rsi_val < MT_RSI_LONG_MIN or rsi_val > MT_RSI_LONG_MAX:
                    return 0.0
                midpoint = (MT_RSI_LONG_MIN + MT_RSI_LONG_MAX) / 2
                if rsi_val <= midpoint:
                    return (rsi_val - MT_RSI_LONG_MIN) / (midpoint - MT_RSI_LONG_MIN)
                else:
                    return 1.0 - (rsi_val - midpoint) / (MT_RSI_LONG_MAX - midpoint) * 0.5
            else:
                if rsi_val < MT_RSI_SHORT_MIN or rsi_val > MT_RSI_SHORT_MAX:
                    return 0.0
                midpoint = (MT_RSI_SHORT_MIN + MT_RSI_SHORT_MAX) / 2
                if rsi_val >= midpoint:
                    return (MT_RSI_SHORT_MAX - rsi_val) / (MT_RSI_SHORT_MAX - midpoint)
                else:
                    return 1.0 - (midpoint - rsi_val) / (midpoint - MT_RSI_SHORT_MIN) * 0.5
        
        # 4) Compute scores for both directions
        # Long side scoring
        mom_score_long = normalize(momentum_raw, 0.0001, MT_MOMENTUM_THRESHOLD * 2.0) if momentum_raw > 0 else 0.0
        vol_score_long = normalize(volume_ratio, 1.05, max(1.5, MT_VOLUME_FACTOR_MIN + 0.3))
        rsi_score_long = rsi_component(current_rsi, is_long=True)
        roc_score_long = normalize(current_roc, 0.01, 0.10) if current_roc > 0 else 0.0
        
        # Short side scoring
        mom_score_short = normalize(abs(momentum_raw), 0.0001, MT_MOMENTUM_THRESHOLD * 2.0) if momentum_raw < 0 else 0.0
        vol_score_short = normalize(volume_ratio, 1.05, max(1.5, MT_VOLUME_FACTOR_MIN + 0.3))
        rsi_score_short = rsi_component(current_rsi, is_long=False)
        roc_score_short = normalize(abs(current_roc), 0.01, 0.10) if current_roc < 0 else 0.0
        
        # Weighted scores
        score_long = 0.45 * mom_score_long + 0.30 * vol_score_long + 0.15 * rsi_score_long + 0.10 * roc_score_long
        score_short = 0.45 * mom_score_short + 0.30 * vol_score_short + 0.15 * rsi_score_short + 0.10 * roc_score_short
        
        # Add confirmation bonus
        if confirm_long:
            score_long = min(score_long * 1.2, 1.0)
        if confirm_short:
            score_short = min(score_short * 1.2, 1.0)
        
        # Pick best side
        if score_long >= score_short:
            best_side = "long"
            best_score = score_long
            mom_score = mom_score_long
            vol_score = vol_score_long
            rsi_score = rsi_score_long
            roc_score = roc_score_long
            has_confirmation = confirm_long
        else:
            best_side = "short"
            best_score = score_short
            mom_score = mom_score_short
            vol_score = vol_score_short
            rsi_score = rsi_score_short
            roc_score = roc_score_short
            has_confirmation = confirm_short
        
        # Determine minimum confidence
        min_conf = (GLOBAL_MIN_CONFIDENCE if DISCOVERY_MODE and USE_GLOBAL_MIN_CONF_WHEN_DISCOVERY
                   else self.config.get('min_confidence', PATTERN_MIN_CONFIDENCE))
        
        # Check if passes
        passes_score = best_score >= min_conf
        
        # Determine exec reason
        if not safety_ok:
            exec_reason = "safety_filter_block"
        elif not has_confirmation:
            exec_reason = "no_confirmation"
        elif not passes_score:
            exec_reason = "score_below_min"
        else:
            exec_reason = "pass"
        
        # 5) ALWAYS write telemetry with real scores
        if TELEMETRY.get('csv_eval_all', False):
            try:
                from ..utils.telemetry_sink import get_telemetry_sink
                sink = get_telemetry_sink()
                
                # Extended telemetry with raw values
                sink.write(
                    pattern='momentum_thrust',
                    event='EVAL',
                    price=current_price,
                    score=best_score,
                    min_score=min_conf,
                    adx=current_adx,
                    atr=current_atr,
                    rsi=current_rsi,
                    pullback_pct=0,  # Placeholder
                    confirm_range_atr=confirm_range_atr,
                    exec_reason=exec_reason,
                    mom_score=mom_score,
                    vol_score=vol_score,
                    rsi_score=rsi_score,
                    roc_score=roc_score,
                    # Additional raw metrics
                    momentum_raw=momentum_raw,
                    roc_raw=current_roc,
                    volume_ratio=volume_ratio,
                    body_fraction=body_fraction,
                    confirmation=confirmation_type,
                    direction=best_side
                )
            except Exception as e:
                logger.debug(f"Telemetry write failed: {e}")
        
        # Log meaningful scores
        if best_score > 0.1:  # Log any non-trivial score
            logger.debug(f"MT EVAL: {best_side} score={best_score:.3f} (mom={mom_score:.2f}, vol={vol_score:.2f}, "
                        f"rsi={rsi_score:.2f}, roc={roc_score:.2f}) confirm={confirmation_type} reason={exec_reason}")
        
        # 6) Only return signal if passes all checks
        if not safety_ok or not passes_score:
            return None
        
        # 7) Build signal with entry plan
        signal = None
        
        if best_side == "long" and (confirm_long or best_score >= min_conf * 1.2):
            # Find micro swing for stop
            swing_low = self._find_micro_swing(data, is_long=True)
            
            # Calculate entry and exit levels
            entry_price = current_price
            
            # Stop: 10-12 ticks beyond micro swing
            stop_offset = np.random.randint(self.stop_min_ticks, self.stop_max_ticks + 1)
            stop_loss = swing_low - (stop_offset * self.tick_size)
            
            # Targets
            target1 = entry_price + (self.target1_ticks * self.tick_size)
            target2 = entry_price + (self.target2_ticks * self.tick_size)
            
            signal = PatternSignal(
                pattern_name='momentum_thrust',
                action=TradeAction.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=target2,
                position_size=1,
                reason=f"Bullish MT: score={best_score:.3f} confirm={confirmation_type}",
                confidence=best_score,
                target1=target1,
                target2=target2,
                swing_level=swing_low
            )
            
            logger.info(f"Bullish MT PASS: score={best_score:.3f} >= {min_conf:.2f}, "
                       f"momentum={momentum_raw:.4f}, vol_ratio={volume_ratio:.2f}, confirm={confirmation_type}")
        
        elif best_side == "short" and (confirm_short or best_score >= min_conf * 1.2):
            # Find micro swing for stop
            swing_high = self._find_micro_swing(data, is_long=False)
            
            # Calculate entry and exit levels
            entry_price = current_price
            
            # Stop: 10-12 ticks beyond micro swing
            stop_offset = np.random.randint(self.stop_min_ticks, self.stop_max_ticks + 1)
            stop_loss = swing_high + (stop_offset * self.tick_size)
            
            # Targets
            target1 = entry_price - (self.target1_ticks * self.tick_size)
            target2 = entry_price - (self.target2_ticks * self.tick_size)
            
            signal = PatternSignal(
                pattern_name='momentum_thrust',
                action=TradeAction.SELL,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=target2,
                position_size=1,
                reason=f"Bearish MT: score={best_score:.3f} confirm={confirmation_type}",
                confidence=best_score,
                target1=target1,
                target2=target2,
                swing_level=swing_high
            )
            
            logger.info(f"Bearish MT PASS: score={best_score:.3f} >= {min_conf:.2f}, "
                       f"momentum={momentum_raw:.4f}, vol_ratio={volume_ratio:.2f}, confirm={confirmation_type}")
        
        # Update state tracking
        if signal:
            self.last_signal_bar = current_idx
            self.consecutive_signals += 1
            
            # Write PASS telemetry
            if TELEMETRY.get('csv_eval_all', False):
                try:
                    from ..utils.telemetry_sink import get_telemetry_sink
                    sink = get_telemetry_sink()
                    sink.write(
                        pattern='momentum_thrust',
                        event='PASS',
                        price=current_price,
                        score=best_score,
                        min_score=min_conf,
                        direction=best_side,
                        stop_ticks=stop_offset,
                        t1_ticks=self.target1_ticks,
                        t2_ticks=self.target2_ticks
                    )
                except:
                    pass
        else:
            if current_idx - self.last_signal_bar > 10:
                self.consecutive_signals = 0
        
        return signal
    
    def calculate_confidence(self, data: pd.DataFrame, signal: PatternSignal) -> float:
        """
        Calculate confidence score for pattern signal
        
        Args:
            data: OHLCV DataFrame
            signal: Pattern signal
            
        Returns:
            Confidence score between 0 and 1
        """
        # Signal already has computed confidence
        return signal.confidence
    
    def validate_signal(self, signal: PatternSignal, spread: float, 
                       last_tick_time: datetime) -> bool:
        """
        Validate signal before execution
        
        Args:
            signal: The pattern signal to validate
            spread: Current bid-ask spread
            last_tick_time: Time of last market tick
            
        Returns:
            True if signal is valid for execution
        """
        # Check spread
        max_spread = self.config.get('max_spread_ticks', 1) * self.tick_size
        if spread > max_spread:
            logger.debug(f"Momentum thrust signal rejected: spread {spread:.2f} > max {max_spread:.2f}")
            return False
        
        # Check data staleness
        if last_tick_time:
            staleness = (datetime.now(timezone.utc) - last_tick_time).total_seconds()
            max_staleness = self.config.get('max_data_staleness_seconds', 2)
            if staleness > max_staleness:
                logger.debug(f"Momentum thrust signal rejected: data stale by {staleness:.1f}s")
                return False
        
        # Check daily trade limit
        if self.daily_trades >= self.config.get('max_daily_trades', 10):
            logger.debug(f"Momentum thrust signal rejected: daily limit reached ({self.daily_trades})")
            return False
        
        # Check consecutive signals
        if self.consecutive_signals > 3:
            logger.debug(f"Momentum thrust signal rejected: too many consecutive signals")
            return False
        
        return True
    
    def update_statistics(self, pnl: float, is_win: bool):
        """Update pattern statistics after trade completion"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if is_win:
            self.winning_trades += 1
        
        # Update daily stats
        super().update_statistics(pnl, is_win)
        
        # Log performance
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logger.info(f"Momentum Thrust Stats - Trades: {self.total_trades}, "
                   f"Win Rate: {win_rate:.1f}%, Total P&L: ${self.total_pnl:.2f}")
    
    def get_pattern_metrics(self) -> Dict:
        """Get current pattern metrics"""
        metrics = super().get_pattern_metrics()
        
        # Add momentum thrust specific metrics
        metrics.update({
            'lookback_period': self.lookback,
            'momentum_threshold': self.momentum_threshold,
            'volume_factor': self.volume_factor,
            'consecutive_signals': self.consecutive_signals,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_pnl': self.total_pnl
        })
        
        return metrics
    
    def get_state(self) -> Dict:
        """Get current pattern state for persistence"""
        state = super().get_state()
        state.update({
            'last_signal_bar': self.last_signal_bar,
            'consecutive_signals': self.consecutive_signals,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl
        })
        return state
    
    def load_state(self, state: Dict):
        """Load pattern state from persistence"""
        super().load_state(state)
        self.last_signal_bar = state.get('last_signal_bar', -1)
        self.consecutive_signals = state.get('consecutive_signals', 0)
        self.total_trades = state.get('total_trades', 0)
        self.winning_trades = state.get('winning_trades', 0)
        self.total_pnl = state.get('total_pnl', 0.0)
    
    def _find_micro_swing(self, data: pd.DataFrame, is_long: bool, lookback: int = 10) -> float:
        """Find the micro swing level for stop placement"""
        if len(data) < lookback:
            # Fallback to simple calculation
            if is_long:
                return data['low'].iloc[-1] - (2 * self.tick_size)
            else:
                return data['high'].iloc[-1] + (2 * self.tick_size)
        
        # Look for swing in recent bars
        recent_bars = data.iloc[-lookback:]
        
        if is_long:
            # Find swing low
            swing = recent_bars['low'].min()
        else:
            # Find swing high
            swing = recent_bars['high'].max()
        
        return swing
    
    def get_trail_parameters(self, adx_value: float) -> Dict:
        """Get trailing stop parameters based on ADX"""
        if adx_value >= self.MT_TRAIL_ADX_THRESHOLD:
            return {
                'should_trail': True,
                'trail_distance_min': self.MT_TRAIL_MIN_TICKS,
                'trail_distance_max': self.MT_TRAIL_MAX_TICKS,
                'trail_distance': np.random.randint(self.MT_TRAIL_MIN_TICKS, self.MT_TRAIL_MAX_TICKS + 1)
            }
        else:
            return {
                'should_trail': False,
                'reason': f'ADX {adx_value:.1f} below threshold {self.MT_TRAIL_ADX_THRESHOLD}'
            }