"""
Base Pattern Class for Trading Patterns
Provides common functionality for all pattern implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

# Entry quality constants
EXHAUSTION_ATR_MULTIPLIER = 1.25
MICRO_PULLBACK_RATIO = 0.382
ENGULFING_BODY_FRACTION = 0.6
ENGULFING_VOLUME_MULTIPLIER = 1.5

class TradeAction(Enum):
    """Trade action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class EntryPlan:
    """Enhanced entry plan with retest levels"""
    trigger_price: float  # Original trigger level
    confirm_price: float  # Confirmation bar close
    retest_entry: float   # Retest entry level (trigger or 50% of confirm bar)
    confirm_bar_range: float  # Range of confirmation bar
    is_exhaustion: bool  # True if confirmation bar is exhaustion
    pullback_achieved: bool  # True if micro-pullback achieved
    
@dataclass
class PatternSignal:
    """Pattern signal with entry parameters"""
    pattern_name: str
    action: TradeAction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    reason: str
    confluence_score: float = 0.0
    timeframe_signals: Dict[str, bool] = None
    entry_plan: Optional[EntryPlan] = None  # Enhanced entry plan
    target1: Optional[float] = None  # First target level
    target2: Optional[float] = None  # Second target level
    swing_level: Optional[float] = None  # Swing high/low reference
    
    def __post_init__(self):
        if self.timeframe_signals is None:
            self.timeframe_signals = {}

class BasePattern(ABC):
    """Base class for all trading patterns"""
    
    def __init__(self, config: Dict = None, **kwargs):
        """
        Initialize base pattern with flexible kwargs
        
        Args:
            config: Pattern configuration dictionary
            **kwargs: Additional parameters including stop_ticks, target_ticks, target1, target2
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Handle exit parameters flexibly
        self.stop_ticks = kwargs.get('stop_ticks', self.config.get('stop_ticks'))
        self.target1 = kwargs.get('target1')
        self.target2 = kwargs.get('target2')
        
        # Legacy support for target_ticks
        if self.target1 is None:
            target_ticks = kwargs.get('target_ticks', self.config.get('target_ticks'))
            if isinstance(target_ticks, (int, float)):
                self.target1 = target_ticks
        
        # Store extra kwargs for subclass access
        self.extra_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['stop_ticks', 'target1', 'target2', 'target_ticks']}
        
        # Pattern statistics
        self.total_signals = 0
        self.winning_signals = 0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.last_signal_time = None
        
        # State management
        self.pattern_state = {}
        self.is_enabled = self.config.get('enabled', True)
        
        # Risk parameters
        self.max_daily_trades = self.config.get('max_daily_trades', 20)
        self.daily_trades = 0
        self.last_trade_date = None
        
        # Entry quality flags
        self.require_confirmation_close = self.config.get('require_confirmation_close', True)
        self.data_cache = None  # Will be set by pattern manager
        
        # Log initialization once
        if not hasattr(BasePattern, '_init_logged'):
            BasePattern._init_logged = set()
        if self.__class__.__name__ not in BasePattern._init_logged:
            self.logger.info(f"PATTERN_INIT name={self.__class__.__name__} stop={self.stop_ticks} t1={self.target1} t2={self.target2}")
            BasePattern._init_logged.add(self.__class__.__name__)
        
        # Initialize pattern-specific components
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize pattern-specific components"""
        pass
    
    @abstractmethod
    def scan_for_setup(self, data: pd.DataFrame, current_price: float) -> Optional[PatternSignal]:
        """
        Scan for pattern setup
        
        Args:
            data: Price data (OHLCV)
            current_price: Current market price
            
        Returns:
            PatternSignal if pattern detected, None otherwise
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, data: pd.DataFrame, signal: PatternSignal) -> float:
        """
        Calculate confidence score for pattern signal
        
        Args:
            data: Price data
            signal: Pattern signal
            
        Returns:
            Confidence score (0-1)
        """
        pass
    
    def validate_signal(self, signal: PatternSignal, spread: float, last_tick_time: datetime) -> bool:
        """
        Validate pattern signal before execution
        
        Args:
            signal: Pattern signal to validate
            spread: Current bid-ask spread in ticks
            last_tick_time: Timestamp of last market tick
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Check if pattern is enabled
        if not self.is_enabled:
            self.logger.debug("Pattern is disabled")
            return False
        
        # Check daily trade limit
        current_date = datetime.now(timezone.utc).date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
        
        if self.daily_trades >= self.max_daily_trades:
            self.logger.warning(f"Daily trade limit reached ({self.max_daily_trades})")
            return False
        
        # Check spread
        max_spread = self.config.get('max_spread_ticks', 1)
        if spread > max_spread:
            self.logger.debug(f"Spread too wide: {spread} > {max_spread}")
            return False
        
        # Check market data freshness
        max_staleness = self.config.get('max_data_staleness_seconds', 2)
        data_age = (datetime.now(timezone.utc) - last_tick_time).total_seconds()
        if data_age > max_staleness:
            self.logger.debug(f"Market data stale: {data_age:.1f}s old")
            return False
        
        # Check minimum confidence (dynamic for MT pattern)
        min_confidence = self.config.get('min_confidence', 0.6)
        
        # Use dynamic confidence for momentum_thrust pattern in MT window
        if 'momentum' in self.__class__.__name__.lower():
            try:
                from ..utils.market_regime import mt_min_confidence
                from datetime import datetime
                try:
                    from zoneinfo import ZoneInfo
                    ct_tz = ZoneInfo('America/Chicago')
                except ImportError:
                    import pytz
                    ct_tz = pytz.timezone('America/Chicago')
                
                now_ct = datetime.now().astimezone(ct_tz).time()
                min_confidence = mt_min_confidence(now_ct)
            except Exception:
                pass  # Fall back to default if helper fails
        
        if signal.confidence < min_confidence:
            self.logger.debug(f"Confidence too low: {signal.confidence:.2f} < {min_confidence}")
            return False
        
        # Check consecutive losses circuit breaker
        max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        if self.consecutive_losses >= max_consecutive_losses:
            self.logger.warning(f"Circuit breaker: {self.consecutive_losses} consecutive losses")
            return False
        
        return True
    
    def exhaustion_check(self, bar_range: float, atr: float) -> bool:
        """
        Check if bar shows exhaustion characteristics
        
        Args:
            bar_range: Range of the bar (high - low)
            atr: Average True Range value
            
        Returns:
            True if bar shows exhaustion (range > threshold * ATR)
        """
        if atr <= 0:
            return False
        
        multiplier = EXHAUSTION_ATR_MULTIPLIER
        
        # Use dynamic multiplier for momentum_thrust pattern in MT window
        if 'momentum' in self.__class__.__name__.lower():
            try:
                from ..utils.market_regime import mt_exhaustion_threshold
                from datetime import datetime
                try:
                    from zoneinfo import ZoneInfo
                    ct_tz = ZoneInfo('America/Chicago')
                except ImportError:
                    import pytz
                    ct_tz = pytz.timezone('America/Chicago')
                
                now_ct = datetime.now().astimezone(ct_tz).time()
                # Get ADX for exhaustion threshold calculation
                adx = 20  # Default fallback
                if hasattr(self, 'data_cache') and self.data_cache:
                    adx = self.data_cache.get_indicator('adx', '1m') or 20
                
                multiplier = mt_exhaustion_threshold(adx, now_ct)
            except Exception:
                pass  # Fall back to default if helper fails
        
        return bar_range > (multiplier * atr)
    
    def micro_pullback_check(self, bars: pd.DataFrame, pivot_price: float, is_long: bool) -> Tuple[bool, float]:
        """
        Check if micro-pullback requirement is met (0.382 retrace)
        
        Args:
            bars: Recent price bars
            pivot_price: Pivot point to measure pullback from
            is_long: True for long setup, False for short
            
        Returns:
            (pullback_achieved, pullback_level): Whether pullback occurred and at what level
        """
        if len(bars) < 2:
            return False, 0.0
        
        last_bar = bars.iloc[-1]
        prev_bar = bars.iloc[-2]
        
        if is_long:
            # For long: need pullback from high
            high_point = max(prev_bar['high'], last_bar['high'])
            pullback_target = high_point - ((high_point - pivot_price) * MICRO_PULLBACK_RATIO)
            pullback_achieved = last_bar['low'] <= pullback_target
            return pullback_achieved, pullback_target
        else:
            # For short: need pullback from low  
            low_point = min(prev_bar['low'], last_bar['low'])
            pullback_target = low_point + ((pivot_price - low_point) * MICRO_PULLBACK_RATIO)
            pullback_achieved = last_bar['high'] >= pullback_target
            return pullback_achieved, pullback_target
    
    def dangerous_engulfing_check(self, bars: pd.DataFrame, atr: float, is_long: bool) -> bool:
        """
        Check for dangerous engulfing pattern (all conditions must be met)
        
        Args:
            bars: Price bars (need at least 2)
            atr: ATR(14) value
            is_long: True for long setup, False for short
            
        Returns:
            True if dangerous engulfing detected
        """
        if len(bars) < 50:  # Need enough bars for volume average
            return False
        
        current_bar = bars.iloc[-1]
        prev_bar = bars.iloc[-2]
        
        # Calculate bar metrics
        current_range = current_bar['high'] - current_bar['low']
        body_size = abs(current_bar['close'] - current_bar['open'])
        body_fraction = body_size / (current_range + 0.0001)
        
        # Volume check
        avg_volume = bars['volume'].iloc[-50:].mean()
        volume_ratio = current_bar['volume'] / (avg_volume + 1)
        
        # All conditions for dangerous engulfing
        if is_long:
            # For long setup: bearish engulfing is dangerous
            condition1 = current_bar['close'] < prev_bar['low']  # Close below previous low
            condition2 = current_range > (EXHAUSTION_ATR_MULTIPLIER * atr)  # Large range
            condition3 = body_fraction >= ENGULFING_BODY_FRACTION  # Strong body
            condition4 = volume_ratio >= ENGULFING_VOLUME_MULTIPLIER  # High volume
        else:
            # For short setup: bullish engulfing is dangerous
            condition1 = current_bar['close'] > prev_bar['high']  # Close above previous high
            condition2 = current_range > (EXHAUSTION_ATR_MULTIPLIER * atr)  # Large range
            condition3 = body_fraction >= ENGULFING_BODY_FRACTION  # Strong body
            condition4 = volume_ratio >= ENGULFING_VOLUME_MULTIPLIER  # High volume
        
        # All conditions must be met
        is_dangerous = condition1 and condition2 and condition3 and condition4
        
        if is_dangerous:
            self.logger.warning(f"Dangerous engulfing detected: close_vs_prev={'below' if is_long else 'above'}, "
                              f"range/atr={current_range/atr:.2f}, body_fraction={body_fraction:.2f}, "
                              f"volume_ratio={volume_ratio:.2f}")
        
        return is_dangerous
    
    def update_statistics(self, pnl: float, is_win: bool):
        """
        Update pattern statistics after trade
        
        Args:
            pnl: Trade P&L
            is_win: Whether trade was profitable
        """
        self.total_signals += 1
        self.total_pnl += pnl
        
        if is_win:
            self.winning_signals += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        self.daily_trades += 1
        
        # Calculate win rate
        win_rate = (self.winning_signals / self.total_signals * 100) if self.total_signals > 0 else 0
        
        self.logger.info(
            f"Pattern stats - Trades: {self.total_signals}, "
            f"Win rate: {win_rate:.1f}%, PnL: ${self.total_pnl:.2f}, "
            f"Consecutive losses: {self.consecutive_losses}"
        )
    
    def get_state(self) -> Dict:
        """Get pattern state for persistence"""
        return {
            'pattern_name': self.__class__.__name__,
            'total_signals': self.total_signals,
            'winning_signals': self.winning_signals,
            'total_pnl': self.total_pnl,
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': self.daily_trades,
            'last_trade_date': self.last_trade_date.isoformat() if self.last_trade_date else None,
            'pattern_state': self.pattern_state,
            'is_enabled': self.is_enabled
        }
    
    def load_state(self, state: Dict):
        """Load pattern state from persistence"""
        self.total_signals = state.get('total_signals', 0)
        self.winning_signals = state.get('winning_signals', 0)
        self.total_pnl = state.get('total_pnl', 0.0)
        self.consecutive_losses = state.get('consecutive_losses', 0)
        self.daily_trades = state.get('daily_trades', 0)
        self.pattern_state = state.get('pattern_state', {})
        self.is_enabled = state.get('is_enabled', True)
        
        last_trade_date = state.get('last_trade_date')
        if last_trade_date:
            from datetime import date
            self.last_trade_date = date.fromisoformat(last_trade_date)
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trades = 0
        self.last_trade_date = datetime.now(timezone.utc).date()
    
    def disable(self):
        """Disable pattern"""
        self.is_enabled = False
        self.logger.info(f"Pattern {self.__class__.__name__} disabled")
    
    def enable(self):
        """Enable pattern"""
        self.is_enabled = True
        self.consecutive_losses = 0  # Reset circuit breaker
        self.logger.info(f"Pattern {self.__class__.__name__} enabled")
    
    def set_data_cache(self, data_cache):
        """Set reference to data cache for indicator access"""
        self.data_cache = data_cache