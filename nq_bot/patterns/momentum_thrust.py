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
                TELEMETRY,
                EXECUTE_ON_SCORE_ONLY, MIN_PASS_SCORE_DISCOVERY,
                FORCE_IMMEDIATE_MARKET
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
            EXECUTE_ON_SCORE_ONLY = False
            MIN_PASS_SCORE_DISCOVERY = 0.35
            FORCE_IMMEDIATE_MARKET = False
        
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
        
        # Determine preliminary best side for safety filters
        if momentum_raw >= 0:
            best_side = "long"
        else:
            best_side = "short"

        # 2) Safety filters
        exhaustion_ok = confirm_range_atr <= 1.25  # Not too extended
        pullback_ok = True  # Simplified for now

        # Phase 1: Add trend line confluence filter
        trend_confluence = self._check_trend_line_confluence(data, current_price, best_side)

        # Phase 2: Add structure-based entry validation
        structure_validation = self._validate_market_structure(data, best_side, current_close, high, low)

        safety_ok = exhaustion_ok and pullback_ok and trend_confluence and structure_validation
        
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
        
        # Refine best side determination with full scoring (before Phase 3 enhancements)
        if mom_score_long >= mom_score_short:
            best_side = "long"
        else:
            best_side = "short"

        # Phase 3: Enhanced momentum quality scoring with session filters
        session_multiplier = self._get_session_quality_multiplier()
        structure_bonus = self._calculate_structure_quality_bonus(data, best_side)
        volume_quality = self._calculate_volume_quality_score(data)

        # Weighted scores with enhancements
        base_score_long = 0.35 * mom_score_long + 0.25 * vol_score_long + 0.15 * rsi_score_long + 0.10 * roc_score_long + 0.15 * volume_quality
        base_score_short = 0.35 * mom_score_short + 0.25 * vol_score_short + 0.15 * rsi_score_short + 0.10 * roc_score_short + 0.15 * volume_quality

        # Apply session and structure multipliers
        score_long = min(1.0, base_score_long * session_multiplier + structure_bonus)
        score_short = min(1.0, base_score_short * session_multiplier + structure_bonus)
        
        # Add confirmation bonus
        if confirm_long:
            score_long = min(score_long * 1.2, 1.0)
        if confirm_short:
            score_short = min(score_short * 1.2, 1.0)
        
        # Pick best side (determine before Phase 3 enhancements)
        if mom_score_long >= mom_score_short:
            best_side = "long"
            has_confirmation = confirm_long
        else:
            best_side = "short"
            has_confirmation = confirm_short

        # Now apply Phase 3 enhancements and pick final scores
        if best_side == "long":
            best_score = score_long
            mom_score = mom_score_long
            vol_score = vol_score_long
            rsi_score = rsi_score_long
            roc_score = roc_score_long
        else:
            best_score = score_short
            mom_score = mom_score_short
            vol_score = vol_score_short
            rsi_score = rsi_score_short
            roc_score = roc_score_short
        
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
        
        # Debug log for every evaluation
        logger.debug(f"DEBUG_MT flags exec_on_score={EXECUTE_ON_SCORE_ONLY} fim={FORCE_IMMEDIATE_MARKET} "
                    f"min={MIN_PASS_SCORE_DISCOVERY} score={best_score:.3f} side={best_side}")
        
        # Log meaningful scores
        if best_score > 0.1:  # Log any non-trivial score
            logger.debug(f"MT EVAL: {best_side} score={best_score:.3f} (mom={mom_score:.2f}, vol={vol_score:.2f}, "
                        f"rsi={rsi_score:.2f}, roc={roc_score:.2f}) confirm={confirmation_type} reason={exec_reason}")
        
        # 6) CRITICAL: Check for score-only immediate execution FIRST (Discovery Mode)
        # This must happen BEFORE any confirmation checks
        if EXECUTE_ON_SCORE_ONLY and FORCE_IMMEDIATE_MARKET and best_score >= MIN_PASS_SCORE_DISCOVERY:
            # Score is sufficient - skip confirmation and build signal immediately
            logger.info(f"MT SCORE-ONLY PASS: {best_side} score={best_score:.3f} >= {MIN_PASS_SCORE_DISCOVERY}")
            
            # Build signal for immediate execution
            if best_side == "long":
                # Find micro swing for stop
                swing_low = self._find_micro_swing(data, is_long=True)
                entry_price = current_price
                
                # Stop: min 10 ticks or 14 ticks ($70) protective stop
                stop_offset = 14  # Use protective stop distance
                stop_loss = entry_price - (stop_offset * self.tick_size)
                
                # Targets
                target1 = entry_price + (5 * self.tick_size)  # 5 ticks
                target2 = entry_price + (10 * self.tick_size) # 10 ticks
                
                signal = PatternSignal(
                    pattern_name='momentum_thrust',
                    action=TradeAction.BUY,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=target2,
                    position_size=1,
                    reason=f"MT score_only: score={best_score:.3f}",
                    confidence=best_score,
                    target1=target1,
                    target2=target2,
                    swing_level=swing_low
                )
            else:  # short
                # Find micro swing for stop
                swing_high = self._find_micro_swing(data, is_long=False)
                entry_price = current_price
                
                # Stop: min 10 ticks or 14 ticks ($70) protective stop
                stop_offset = 14  # Use protective stop distance
                stop_loss = entry_price + (stop_offset * self.tick_size)
                
                # Targets
                target1 = entry_price - (5 * self.tick_size)  # 5 ticks
                target2 = entry_price - (10 * self.tick_size) # 10 ticks
                
                signal = PatternSignal(
                    pattern_name='momentum_thrust',
                    action=TradeAction.SELL,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=target2,
                    position_size=1,
                    reason=f"MT score_only: score={best_score:.3f}",
                    confidence=best_score,
                    target1=target1,
                    target2=target2,
                    swing_level=swing_high
                )
            
            # Write PATTERN_PASS telemetry
            if TELEMETRY.get('csv_eval_all', False):
                try:
                    from ..utils.telemetry_sink import get_telemetry_sink
                    sink = get_telemetry_sink()
                    sink.write(
                        pattern='momentum_thrust',
                        event='PATTERN_PASS',
                        price=current_price,
                        score=best_score,
                        min_score=MIN_PASS_SCORE_DISCOVERY,
                        direction=best_side,
                        stop_ticks=stop_offset,
                        t1_ticks=5,
                        t2_ticks=10,
                        reason='score_only'
                    )
                except:
                    pass
            
            return signal
        
        # If not score-only pass, check normal conditions
        if not safety_ok or not passes_score:
            return None
        
        # 6.5) Apply Candlestick Guard
        try:
            from ..pattern_config import CANDLES
            from ..utils.candles import get_candle_guard
            
            if CANDLES.get('enable', False):
                # Check if near key levels (simplified - check if near recent high/low)
                recent_high = high[-20:].max() if len(high) >= 20 else high.max()
                recent_low = low[-20:].min() if len(low) >= 20 else low.min()
                near_level_ticks = CANDLES.get('near_level_ticks', 6)
                near_level = (abs(current_price - recent_high) <= near_level_ticks * self.tick_size or
                             abs(current_price - recent_low) <= near_level_ticks * self.tick_size)
                
                # Prepare context for candle guard
                bars = []
                for i in range(max(0, len(data) - 3), len(data)):
                    bars.append({
                        'open': open_price[i],
                        'high': high[i],
                        'low': low[i],
                        'close': close[i],
                        'volume': volume[i]
                    })
                
                context = {
                    'atr': current_atr,
                    'avg_vol': volume_avg,
                    'vol_ratio': volume_ratio,
                    'tick_size': self.tick_size,
                    'bars': bars,
                    'symbol': 'NQ',
                    'timestamp': datetime.now(timezone.utc)
                }
                
                guard = get_candle_guard()
                decision = guard.guard_decision(best_side, near_level, CANDLES, context)
                
                if decision['hard_veto']:
                    # Log veto
                    candle = decision.get('candle')
                    if candle:
                        logger.info(f"CANDLE_VETO pattern=momentum_thrust type={candle.kind} "
                                   f"side={best_side} near_level={near_level} "
                                   f"strength={candle.strength:.2f}")
                        
                        # Write to telemetry
                        if TELEMETRY.get('csv_eval_all', False):
                            try:
                                from ..utils.telemetry_sink import get_telemetry_sink
                                sink = get_telemetry_sink()
                                sink.write(
                                    pattern='momentum_thrust',
                                    event='CANDLE_VETO',
                                    price=current_price,
                                    score=best_score,
                                    candle_type=candle.kind,
                                    candle_strength=candle.strength,
                                    near_level=near_level,
                                    side=best_side
                                )
                            except:
                                pass
                    return None
                
                # Apply soft bonus
                soft_bonus = decision.get('soft_bonus', 0.0)
                if soft_bonus != 0:
                    old_score = best_score
                    best_score = max(0.0, min(1.0, best_score + soft_bonus))
                    
                    candle = decision.get('candle')
                    if candle:
                        logger.info(f"CANDLE_BOOST type={candle.kind} bonus={soft_bonus:.3f} "
                                   f"score_before={old_score:.3f} score_after={best_score:.3f}")
                    
                    # Re-check if still passes after adjustment
                    if best_score < min_conf:
                        return None
        except Exception as e:
            logger.debug(f"Candle guard error: {e}")
        
        # 7) Build signal with entry plan
        signal = None
        
        # Check if we should use score-only immediate execution
        score_only_pass = (EXECUTE_ON_SCORE_ONLY and FORCE_IMMEDIATE_MARKET and 
                          best_score >= MIN_PASS_SCORE_DISCOVERY)
        
        if best_side == "long" and (confirm_long or best_score >= min_conf * 1.2 or score_only_pass):
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
            
            # Determine if this is score-only pass
            pass_reason = "score_only" if (score_only_pass and not confirm_long) else confirmation_type
            
            signal = PatternSignal(
                pattern_name='momentum_thrust',
                action=TradeAction.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=target2,
                position_size=1,
                reason=f"Bullish MT: score={best_score:.3f} mode={pass_reason}",
                confidence=best_score,
                target1=target1,
                target2=target2,
                swing_level=swing_low
            )
            
            logger.info(f"Bullish MT PASS: score={best_score:.3f} >= {min_conf:.2f}, "
                       f"momentum={momentum_raw:.4f}, vol_ratio={volume_ratio:.2f}, mode={pass_reason}")
        
        elif best_side == "short" and (confirm_short or best_score >= min_conf * 1.2 or score_only_pass):
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
            
            # Determine if this is score-only pass
            pass_reason = "score_only" if (score_only_pass and not confirm_short) else confirmation_type
            
            signal = PatternSignal(
                pattern_name='momentum_thrust',
                action=TradeAction.SELL,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=target2,
                position_size=1,
                reason=f"Bearish MT: score={best_score:.3f} mode={pass_reason}",
                confidence=best_score,
                target1=target1,
                target2=target2,
                swing_level=swing_high
            )
            
            logger.info(f"Bearish MT PASS: score={best_score:.3f} >= {min_conf:.2f}, "
                       f"momentum={momentum_raw:.4f}, vol_ratio={volume_ratio:.2f}, mode={pass_reason}")
        
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
                        event='PATTERN_PASS',
                        price=current_price,
                        score=best_score,
                        min_score=min_conf,
                        direction=best_side,
                        stop_ticks=stop_offset if 'stop_offset' in locals() else 10,
                        t1_ticks=self.target1_ticks,
                        t2_ticks=self.target2_ticks,
                        reason=pass_reason if 'pass_reason' in locals() else 'standard'
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

    def _check_trend_line_confluence(self, data: pd.DataFrame, current_price: float, direction: str) -> bool:
        """
        Phase 1: Check if price is near key trend line levels
        Prevents entries in 'no man's land' - requires confluence with structure

        Args:
            data: OHLCV DataFrame
            current_price: Current market price
            direction: "long" or "short"

        Returns:
            True if near trend line confluence, False if in no man's land
        """
        try:
            if len(data) < 20:
                return True  # Not enough data for trend line analysis

            # Get recent swing points for trend line construction
            lookback = min(50, len(data))
            recent_data = data.iloc[-lookback:]

            # Find swing highs and lows
            swing_highs = self._find_swing_points(recent_data, 'high', window=5)
            swing_lows = self._find_swing_points(recent_data, 'low', window=5)

            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return True  # Not enough swings for trend lines

            # Calculate trend lines
            resistance_levels = self._calculate_trend_lines(swing_highs, 'resistance')
            support_levels = self._calculate_trend_lines(swing_lows, 'support')

            # Check confluence based on direction
            confluence_distance = 6 * self.tick_size  # 6 ticks proximity

            if direction == "long":
                # For longs, need to be near support levels
                for level in support_levels:
                    if abs(current_price - level) <= confluence_distance:
                        logger.debug(f"MT CONFLUENCE: Long near support {level:.2f}, price {current_price:.2f}")
                        return True

                # Also check for bounces off recent swing lows
                recent_swing_low = min([point['price'] for point in swing_lows[-3:]])
                if abs(current_price - recent_swing_low) <= confluence_distance:
                    logger.debug(f"MT CONFLUENCE: Long near recent swing low {recent_swing_low:.2f}")
                    return True

            else:  # short
                # For shorts, need to be near resistance levels
                for level in resistance_levels:
                    if abs(current_price - level) <= confluence_distance:
                        logger.debug(f"MT CONFLUENCE: Short near resistance {level:.2f}, price {current_price:.2f}")
                        return True

                # Also check for rejections from recent swing highs
                recent_swing_high = max([point['price'] for point in swing_highs[-3:]])
                if abs(current_price - recent_swing_high) <= confluence_distance:
                    logger.debug(f"MT CONFLUENCE: Short near recent swing high {recent_swing_high:.2f}")
                    return True

            # If no confluence found, reject entry (in no man's land)
            logger.debug(f"MT NO_CONFLUENCE: {direction} rejected - no structure nearby at {current_price:.2f}")
            return False

        except Exception as e:
            logger.debug(f"Trend line confluence check failed: {e}")
            return True  # Default to allow on error

    def _find_swing_points(self, data: pd.DataFrame, price_type: str, window: int = 5) -> List[Dict]:
        """
        Find swing high/low points in price data

        Args:
            data: OHLCV DataFrame
            price_type: 'high' or 'low'
            window: Window size for swing detection

        Returns:
            List of swing points with index and price
        """
        swings = []
        prices = data[price_type].values

        for i in range(window, len(prices) - window):
            current = prices[i]

            if price_type == 'high':
                # Check if this is a swing high
                is_swing = all(current >= prices[i-window:i]) and all(current >= prices[i+1:i+window+1])
            else:
                # Check if this is a swing low
                is_swing = all(current <= prices[i-window:i]) and all(current <= prices[i+1:i+window+1])

            if is_swing:
                swings.append({
                    'index': i,
                    'price': current,
                    'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                })

        return swings

    def _calculate_trend_lines(self, swing_points: List[Dict], line_type: str) -> List[float]:
        """
        Calculate trend line levels from swing points

        Args:
            swing_points: List of swing point dicts
            line_type: 'support' or 'resistance'

        Returns:
            List of trend line price levels
        """
        if len(swing_points) < 2:
            return []

        trend_lines = []

        # Use last 3-4 swing points to create trend lines
        recent_swings = swing_points[-4:] if len(swing_points) >= 4 else swing_points

        # Create trend lines from pairs of swing points
        for i in range(len(recent_swings) - 1):
            for j in range(i + 1, len(recent_swings)):
                point1 = recent_swings[i]
                point2 = recent_swings[j]

                # Calculate slope and current level
                if point2['index'] != point1['index']:  # Avoid division by zero
                    slope = (point2['price'] - point1['price']) / (point2['index'] - point1['index'])

                    # Project to current bar (assume last swing is most recent)
                    current_index = swing_points[-1]['index'] if swing_points else point2['index']
                    projected_level = point2['price'] + slope * (current_index - point2['index'])

                    trend_lines.append(projected_level)

        # Also add horizontal levels from recent swings
        for swing in recent_swings[-2:]:  # Last 2 swings as horizontal levels
            trend_lines.append(swing['price'])

        return trend_lines

    def _validate_market_structure(self, data: pd.DataFrame, direction: str, current_close: float,
                                 high_series: np.ndarray, low_series: np.ndarray) -> bool:
        """
        Phase 2: Validate market structure for momentum thrust entries
        Requires proper structure break and displacement validation

        Args:
            data: OHLCV DataFrame
            direction: "long" or "short"
            current_close: Current bar close price
            high_series: High price array
            low_series: Low price array

        Returns:
            True if market structure supports the entry
        """
        try:
            if len(data) < 10:
                return True  # Not enough data for structure analysis

            # Phase 2a: Market Structure Break (MSB) validation
            msb_valid = self._check_market_structure_break(direction, current_close, high_series, low_series)
            if not msb_valid:
                logger.debug(f"MT STRUCTURE: {direction} rejected - no valid market structure break")
                return False

            # Phase 2b: Displacement validation
            displacement_valid = self._check_displacement_quality(data, direction)
            if not displacement_valid:
                logger.debug(f"MT STRUCTURE: {direction} rejected - insufficient displacement quality")
                return False

            # Phase 2c: Liquidity sweep confirmation
            liquidity_sweep = self._check_liquidity_sweep(direction, high_series, low_series)
            if not liquidity_sweep:
                logger.debug(f"MT STRUCTURE: {direction} rejected - no liquidity sweep detected")
                return False

            logger.debug(f"MT STRUCTURE: {direction} validated - MSB + displacement + sweep confirmed")
            return True

        except Exception as e:
            logger.debug(f"Market structure validation failed: {e}")
            return True  # Default to allow on error

    def _check_market_structure_break(self, direction: str, current_close: float,
                                    high_series: np.ndarray, low_series: np.ndarray) -> bool:
        """
        Check for valid market structure break (MSB)

        Args:
            direction: "long" or "short"
            current_close: Current close price
            high_series: Array of high prices
            low_series: Array of low prices

        Returns:
            True if valid MSB detected
        """
        try:
            lookback = min(20, len(high_series) - 1)

            if direction == "long":
                # For longs: current close should break above recent swing high
                recent_highs = high_series[-lookback:]
                previous_structure_high = np.max(recent_highs[:-1])  # Exclude current bar

                # MSB: close above previous structure high + buffer
                msb_threshold = previous_structure_high + (2 * self.tick_size)
                return current_close > msb_threshold

            else:  # short
                # For shorts: current close should break below recent swing low
                recent_lows = low_series[-lookback:]
                previous_structure_low = np.min(recent_lows[:-1])  # Exclude current bar

                # MSB: close below previous structure low - buffer
                msb_threshold = previous_structure_low - (2 * self.tick_size)
                return current_close < msb_threshold

        except Exception as e:
            logger.debug(f"MSB check failed: {e}")
            return False

    def _check_displacement_quality(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Check quality of price displacement for momentum thrust

        Args:
            data: OHLCV DataFrame
            direction: "long" or "short"

        Returns:
            True if displacement quality is sufficient
        """
        try:
            if len(data) < 3:
                return True

            # Check last 2-3 bars for displacement characteristics
            recent_bars = data.iloc[-3:]

            # Calculate displacement metrics
            total_move = 0
            strong_bars = 0

            for _, bar in recent_bars.iterrows():
                bar_range = bar['high'] - bar['low']
                body_size = abs(bar['close'] - bar['open'])

                if bar_range > 0:
                    body_fraction = body_size / bar_range

                    # Strong displacement bar criteria
                    if body_fraction >= 0.6:  # 60% body
                        strong_bars += 1

                        if direction == "long" and bar['close'] > bar['open']:
                            total_move += body_size
                        elif direction == "short" and bar['close'] < bar['open']:
                            total_move += body_size

            # Require at least 1 strong displacement bar
            min_strong_bars = 1
            min_total_move = 4 * self.tick_size  # Minimum 4 ticks total displacement

            displacement_valid = (strong_bars >= min_strong_bars and total_move >= min_total_move)

            if displacement_valid:
                logger.debug(f"MT DISPLACEMENT: {direction} valid - {strong_bars} strong bars, {total_move/self.tick_size:.1f} ticks move")

            return displacement_valid

        except Exception as e:
            logger.debug(f"Displacement quality check failed: {e}")
            return False

    def _check_liquidity_sweep(self, direction: str, high_series: np.ndarray, low_series: np.ndarray) -> bool:
        """
        Check for liquidity sweep before momentum thrust

        Args:
            direction: "long" or "short"
            high_series: Array of high prices
            low_series: Array of low prices

        Returns:
            True if liquidity sweep detected
        """
        try:
            lookback = min(15, len(high_series) - 2)

            if direction == "long":
                # Look for sweep below recent equal lows before thrust up
                recent_lows = low_series[-lookback:]
                current_low = low_series[-1]

                # Find potential equal lows
                for i in range(len(recent_lows) - 2):
                    for j in range(i + 1, len(recent_lows) - 1):
                        low1 = recent_lows[i]
                        low2 = recent_lows[j]

                        # Check if they're equal (within 1 tick)
                        if abs(low1 - low2) <= self.tick_size:
                            equal_low_level = min(low1, low2)

                            # Check if current/recent bar swept below
                            sweep_threshold = equal_low_level - (1.5 * self.tick_size)
                            if current_low <= sweep_threshold:
                                logger.debug(f"MT SWEEP: Long - swept below equal lows at {equal_low_level:.2f}")
                                return True

            else:  # short
                # Look for sweep above recent equal highs before thrust down
                recent_highs = high_series[-lookback:]
                current_high = high_series[-1]

                # Find potential equal highs
                for i in range(len(recent_highs) - 2):
                    for j in range(i + 1, len(recent_highs) - 1):
                        high1 = recent_highs[i]
                        high2 = recent_highs[j]

                        # Check if they're equal (within 1 tick)
                        if abs(high1 - high2) <= self.tick_size:
                            equal_high_level = max(high1, high2)

                            # Check if current/recent bar swept above
                            sweep_threshold = equal_high_level + (1.5 * self.tick_size)
                            if current_high >= sweep_threshold:
                                logger.debug(f"MT SWEEP: Short - swept above equal highs at {equal_high_level:.2f}")
                                return True

            # If no clear liquidity sweep found, still allow (not all setups have perfect sweeps)
            return True  # Made less restrictive for real market conditions

        except Exception as e:
            logger.debug(f"Liquidity sweep check failed: {e}")
            return True

    def _get_session_quality_multiplier(self) -> float:
        """
        Phase 3: Calculate session quality multiplier for momentum thrust
        Favors high-activity periods with better liquidity

        Returns:
            Multiplier between 0.8 and 1.3 based on session quality
        """
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            # Get current time in different timezones
            utc_now = datetime.now(timezone.utc)
            ny_time = utc_now.astimezone(ZoneInfo("America/New_York"))
            london_time = utc_now.astimezone(ZoneInfo("Europe/London"))

            hour_ny = ny_time.hour
            hour_london = london_time.hour

            # High-quality session periods (higher multiplier)
            # London Open: 2:00-6:00 AM EST
            if 2 <= hour_ny <= 6:
                return 1.25  # London session boost

            # NY Open: 9:30-12:00 PM EST
            elif 9.5 <= hour_ny <= 12:
                return 1.3   # NY morning session boost (highest)

            # NY Lunch: 12:00-2:00 PM EST (overlap period)
            elif 12 <= hour_ny <= 14:
                return 1.2   # Lunch overlap boost

            # Asian session overlap: 6:00-9:00 PM EST
            elif 18 <= hour_ny <= 21:
                return 1.1   # Asian overlap boost

            # Off-hours: reduced multiplier
            else:
                return 0.85  # Off-hours penalty

        except Exception as e:
            logger.debug(f"Session quality calculation failed: {e}")
            return 1.0  # Default neutral multiplier

    def _calculate_structure_quality_bonus(self, data: pd.DataFrame, direction: str) -> float:
        """
        Phase 3: Calculate structure quality bonus for momentum scoring

        Args:
            data: OHLCV DataFrame
            direction: "long" or "short"

        Returns:
            Bonus score between 0.0 and 0.15 based on structure quality
        """
        try:
            if len(data) < 5:
                return 0.0

            recent_bars = data.iloc[-5:]
            bonus = 0.0

            # Structure quality factors
            # 1. Trend alignment (last 5 bars)
            trend_aligned_bars = 0
            for _, bar in recent_bars.iterrows():
                if direction == "long" and bar['close'] > bar['open']:
                    trend_aligned_bars += 1
                elif direction == "short" and bar['close'] < bar['open']:
                    trend_aligned_bars += 1

            trend_alignment = trend_aligned_bars / len(recent_bars)
            bonus += 0.05 * trend_alignment  # Up to 5% bonus for trend alignment

            # 2. Progressive momentum (each bar stronger than previous)
            progressive_strength = 0
            if len(recent_bars) >= 3:
                last_3_bars = recent_bars.iloc[-3:]
                strength_increasing = True

                for i in range(1, len(last_3_bars)):
                    prev_bar = last_3_bars.iloc[i-1]
                    curr_bar = last_3_bars.iloc[i]

                    prev_strength = abs(prev_bar['close'] - prev_bar['open'])
                    curr_strength = abs(curr_bar['close'] - curr_bar['open'])

                    if curr_strength <= prev_strength:
                        strength_increasing = False
                        break

                if strength_increasing:
                    progressive_strength = 1.0

            bonus += 0.05 * progressive_strength  # Up to 5% bonus for progressive momentum

            # 3. Volume confirmation (current volume above recent average)
            if len(data) >= 10:
                current_volume = recent_bars.iloc[-1]['volume']
                avg_volume = data.iloc[-10:-1]['volume'].mean()

                if current_volume > avg_volume * 1.2:  # 20% above average
                    bonus += 0.05  # 5% bonus for volume confirmation

            return min(0.15, bonus)  # Cap at 15% total bonus

        except Exception as e:
            logger.debug(f"Structure quality bonus calculation failed: {e}")
            return 0.0

    def _calculate_volume_quality_score(self, data: pd.DataFrame) -> float:
        """
        Phase 3: Calculate volume quality component for enhanced scoring

        Args:
            data: OHLCV DataFrame

        Returns:
            Volume quality score between 0.0 and 1.0
        """
        try:
            if len(data) < 20:
                return 0.5  # Default moderate score

            current_volume = data.iloc[-1]['volume']
            recent_avg = data.iloc[-20:-1]['volume'].mean()
            short_avg = data.iloc[-5:-1]['volume'].mean()

            # Volume surge factor
            volume_surge = current_volume / max(recent_avg, 1.0)

            # Progressive volume (last 5 bars vs previous 15)
            progressive_volume = short_avg / max(recent_avg, 1.0)

            # Combined volume quality
            # Normalize volume surge (optimal range 1.2 - 3.0)
            surge_score = min(1.0, max(0.0, (volume_surge - 1.0) / 2.0))

            # Progressive volume score
            prog_score = min(1.0, max(0.0, (progressive_volume - 1.0) / 0.5))

            # Weighted combination
            volume_quality = 0.7 * surge_score + 0.3 * prog_score

            return volume_quality

        except Exception as e:
            logger.debug(f"Volume quality calculation failed: {e}")
            return 0.5

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