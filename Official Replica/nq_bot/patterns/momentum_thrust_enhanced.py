"""
Enhanced Momentum Thrust Pattern with Entry Quality Improvements
Includes confirmation close, exhaustion checks, and micro-pullback requirements
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, time
import logging

from .base_pattern import BasePattern, PatternSignal, TradeAction, EntryPlan

logger = logging.getLogger(__name__)

class MomentumThrustPattern(BasePattern):
    """
    Enhanced momentum thrust pattern with entry quality improvements
    - Waits for confirmation close through trigger level
    - Skips exhaustion bars (range > 1.25 × ATR)
    - Requires micro-pullback (≥0.382 retrace)
    """
    
    def _initialize(self):
        """Initialize pattern-specific components"""
        # Pattern configuration from discovery
        self.lookback = self.config.get('lookback', 56)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.0014)
        self.volume_factor = self.config.get('volume_factor', 1.72)
        self.min_strength = self.config.get('min_strength', 40)
        
        # MT window specific configuration
        from ..pattern_config import MT_CANCEL_IF_RUNS_TICKS, MT_VOLUME_FACTOR_DYNAMIC
        self.cancel_if_runs_ticks = MT_CANCEL_IF_RUNS_TICKS
        self.volume_factor_dynamic = MT_VOLUME_FACTOR_DYNAMIC
        
        # Risk parameters
        self.stop_ticks = self.config.get('stop_ticks', 6)
        self.target_ticks = self.config.get('target_ticks', 4)
        
        # NQ specifics
        self.tick_size = 0.25
        self.point_value = 20
        
        # Technical indicators periods
        self.rsi_period = 14
        self.volume_ma_period = 20
        self.atr_period = 14
        
        # State tracking for confirmation
        self.pending_setup = None
        self.confirmation_waiting = False
        self.trigger_level = None
        self.pivot_level = None
    
    def scan_for_setup(self, data: pd.DataFrame, current_price: float) -> Optional[PatternSignal]:
        """
        Enhanced scan with confirmation and quality checks
        
        Args:
            data: OHLCV data (1-minute bars)
            current_price: Current market price
            
        Returns:
            PatternSignal if pattern detected with quality confirmation, None otherwise
        """
        if len(data) < max(self.lookback, 100):
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
        """Scan for new momentum thrust setup"""
        
        # Calculate indicators
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # RSI for momentum confirmation
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        
        # Volume analysis
        volume_ma = talib.SMA(volume, timeperiod=self.volume_ma_period)
        volume_ratio = volume / (volume_ma + 1e-10)
        
        # Rate of change for momentum
        roc = talib.ROC(close, timeperiod=10)
        
        # Momentum calculation over lookback period
        momentum = (close[-1] - close[-self.lookback]) / close[-self.lookback]
        
        # Current bar analysis
        current_idx = len(data) - 1
        current_volume_ratio = volume_ratio[current_idx]
        current_rsi = rsi[current_idx]
        current_roc = roc[current_idx]
        
        # Calculate dynamic volume factor if enabled
        volume_threshold = self.volume_factor
        if self.volume_factor_dynamic:
            try:
                # Calculate volume z-score over last 200 bars
                volume_values = data['volume'].iloc[-200:] if len(data) >= 200 else data['volume']
                if len(volume_values) > 20:
                    vol_mean = volume_values.mean()
                    vol_std = volume_values.std()
                    if vol_std > 0:
                        vol_z = (data['volume'].iloc[-1] - vol_mean) / vol_std
                        # Dynamic factor: 1.4 to 2.5 based on z-score
                        volume_threshold = max(1.4, min(2.5, 1.2 + vol_z))
            except Exception:
                pass  # Fall back to static factor
        
        # Check for momentum thrust conditions
        is_bullish = (momentum > self.momentum_threshold and 
                     current_volume_ratio > volume_threshold and
                     current_rsi > 50 and current_rsi < 80 and
                     current_roc > 0)
        
        is_bearish = (momentum < -self.momentum_threshold and 
                     current_volume_ratio > volume_threshold and
                     current_rsi < 50 and current_rsi > 20 and
                     current_roc < 0)
        
        if is_bullish or is_bearish:
            # Set trigger level as current high (for long) or low (for short)
            if is_bullish:
                self.trigger_level = data.iloc[-1]['high']
                self.pivot_level = data.iloc[-2]['low']  # Previous low as pivot
                direction = "bullish"
                is_long = True
            else:
                self.trigger_level = data.iloc[-1]['low']
                self.pivot_level = data.iloc[-2]['high']  # Previous high as pivot
                direction = "bearish"
                is_long = False
            
            # Store pending setup
            self.pending_setup = {
                'direction': direction,
                'is_long': is_long,
                'momentum': momentum,
                'volume_ratio': current_volume_ratio,
                'rsi': current_rsi,
                'setup_time': datetime.now(timezone.utc),
                'setup_bar_index': current_idx
            }
            
            # If confirmation is required, wait for it
            if self.require_confirmation_close:
                self.confirmation_waiting = True
                logger.info(f"Momentum thrust setup detected ({direction}), waiting for confirmation close through {self.trigger_level:.2f}")
                return None
            else:
                # No confirmation required, signal immediately
                return self._create_signal(data, current_price, atr)
    
        return None
    
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
                      pullback_achieved: bool = True) -> PatternSignal:
        """Create pattern signal with entry plan"""
        
        setup = self.pending_setup
        is_long = setup['is_long']
        
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
        
        # Calculate stops and targets
        if is_long:
            stop_loss = entry_price - (self.stop_ticks * self.tick_size)
            take_profit = entry_price + (self.target_ticks * self.tick_size)
            action = TradeAction.BUY
        else:
            stop_loss = entry_price + (self.stop_ticks * self.tick_size)
            take_profit = entry_price - (self.target_ticks * self.tick_size)
            action = TradeAction.SELL
        
        # Calculate confidence
        base_confidence = self._calculate_confidence(
            setup['momentum'], setup['volume_ratio'], setup['rsi'], 0
        )
        
        # Adjust confidence based on quality checks
        if is_exhaustion:
            base_confidence *= 0.8  # Reduce if exhaustion
        if not pullback_achieved:
            base_confidence *= 0.9  # Reduce if no pullback
        
        # Create entry plan
        entry_plan = EntryPlan(
            trigger_price=self.trigger_level,
            confirm_price=data.iloc[-1]['close'],
            retest_entry=retest_entry,
            confirm_bar_range=confirm_bar_range or 0,
            is_exhaustion=is_exhaustion,
            pullback_achieved=pullback_achieved
        )
        
        signal = PatternSignal(
            pattern_name='momentum_thrust_enhanced',
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=1,
            reason=f"{'Bullish' if is_long else 'Bearish'} momentum thrust with confirmation: "
                   f"momentum={setup['momentum']:.4f}, vol_ratio={setup['volume_ratio']:.2f}, "
                   f"rsi={setup['rsi']:.1f}, exhaustion={is_exhaustion}, pullback={pullback_achieved}",
            confidence=base_confidence,
            entry_plan=entry_plan
        )
        
        logger.info(f"Momentum thrust signal generated: {action.value} at {entry_price:.2f}, "
                   f"confidence={base_confidence:.2f}")
        
        return signal
    
    def _calculate_confidence(self, momentum: float, volume_ratio: float, 
                             rsi: float, roc: float) -> float:
        """Calculate confidence score based on multiple factors"""
        
        # Base confidence from momentum strength
        momentum_score = min(abs(momentum) / (self.momentum_threshold * 2), 1.0) * 0.3
        
        # Volume confirmation
        volume_score = min(volume_ratio / (self.volume_factor * 1.5), 1.0) * 0.3
        
        # RSI positioning (best between 30-70)
        if 30 <= rsi <= 70:
            rsi_score = 0.3
        elif 20 <= rsi <= 80:
            rsi_score = 0.2
        else:
            rsi_score = 0.1
        
        # ROC confirmation
        roc_score = min(abs(roc) / 2, 1.0) * 0.1
        
        # Total confidence
        confidence = momentum_score + volume_score + rsi_score + roc_score
        
        return min(confidence, 1.0)
    
    def _reset_setup(self):
        """Reset pending setup state"""
        self.pending_setup = None
        self.confirmation_waiting = False
        self.trigger_level = None
        self.pivot_level = None
    
    def calculate_confidence(self, data: pd.DataFrame, signal: PatternSignal) -> float:
        """Required override from base class"""
        return signal.confidence