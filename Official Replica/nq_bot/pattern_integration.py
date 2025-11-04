"""
Pattern Integration Module for NQ Bot
Integrates new patterns with existing bot infrastructure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timezone
import logging

# Ensure patterns are importable
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add parent directory to path for accessing utils modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from .patterns.trend_line_bounce import TrendLineBouncePattern
from .patterns.momentum_thrust import MomentumThrustPattern
from .pattern_config import PATTERN_CONFIG, get_pattern_config, get_all_enabled_patterns
from .utils.technical_analysis import TechnicalAnalysisFallback
from .utils.market_regime import MarketRegimeDetector

logger = logging.getLogger(__name__)

class PatternManager:
    """
    Manages multiple trading patterns and coordinates signals
    """
    
    def __init__(self, data_cache=None):
        """Initialize pattern manager
        
        Args:
            data_cache: DataCache instance for market regime detection
        """
        self.patterns = {}
        self.pattern_stats = {}
        self.active_signal = None
        self.last_signal_time = None
        self.data_cache = data_cache
        self.last_fill_time = None  # Track for probe trades
        
        # Initialize technical analysis fallback with data_cache
        self.technical_analysis = TechnicalAnalysisFallback(data_cache)
        
        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector(data_cache)
        
        # Initialize enabled patterns
        self._initialize_patterns()
        
        logger.info(f"PatternManager initialized with {len(self.patterns)} patterns + TA fallback + regime filtering")
    
    def _initialize_patterns(self):
        """Initialize all enabled patterns"""
        enabled_patterns = get_all_enabled_patterns()
        
        for pattern_name in enabled_patterns:
            config = get_pattern_config(pattern_name)
            
            if pattern_name == 'trend_line_bounce':
                self.patterns[pattern_name] = TrendLineBouncePattern(config)
                self.pattern_stats[pattern_name] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0,
                    'last_signal': None
                }
                logger.info(f"Initialized TrendLineBounce pattern")
            
            elif pattern_name == 'momentum_thrust':
                self.patterns[pattern_name] = MomentumThrustPattern(config)
                self.pattern_stats[pattern_name] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0,
                    'last_signal': None
                }
                logger.info(f"Initialized MomentumThrust pattern")
    
    def scan_all_patterns(self, data: pd.DataFrame, current_price: float, 
                          spread: float = 0.25, last_tick_time: datetime = None) -> Optional[Dict]:
        """
        Scan all patterns for trading signals with regime filtering
        
        Args:
            data: OHLCV data
            current_price: Current market price
            spread: Current bid-ask spread
            last_tick_time: Time of last market tick
            
        Returns:
            Best signal if found, None otherwise
        """
        if last_tick_time is None:
            last_tick_time = datetime.now(timezone.utc)
        
        signals = []
        
        # Scan each pattern with regime filtering
        for pattern_name, pattern in self.patterns.items():
            try:
                # Check if pattern is allowed by regime
                should_scan, regime_reason = self.regime_detector.should_scan_pattern(
                    pattern_name, current_price, data
                )
                
                # Add regime telemetry
                if hasattr(self, 'data_cache') and self.data_cache:
                    adx = self.data_cache.get_indicator('adx', '1m') or 0
                    atr = self.data_cache.get_indicator('atr', '1m') or 0
                    atr_band = self.regime_detector.get_atr_band(atr) if atr > 0 else 1.00
                    time_ct = self.regime_detector.get_current_time_ct()
                    
                    from .pattern_config import TRACE
                    if TRACE.get('regime', False):
                        if should_scan:
                            logger.info(f"REGIME_PASS pattern={pattern_name} adx={adx:.2f} atr_band={atr_band:.2f} time_ct={time_ct}")
                        else:
                            logger.info(f"REGIME_BLOCK pattern={pattern_name} reason=\"{regime_reason}\" adx={adx:.2f} atr_band={atr_band:.2f} time_ct={time_ct}")
                
                if not should_scan:
                    continue
                
                # Scan for setup
                signal = pattern.scan_for_setup(data, current_price)
                
                # Always write PATTERN_EVAL to CSV for discovery mode
                from .pattern_config import TELEMETRY
                if TELEMETRY.get('csv_eval_all', False):
                    try:
                        from .utils.telemetry_sink import get_telemetry_sink
                        sink = get_telemetry_sink()
                        
                        # Get current indicators
                        adx_val = self.data_cache.get_indicator('adx', '1m') if self.data_cache else 0
                        atr_val = self.data_cache.get_indicator('atr', '1m') if self.data_cache else 0
                        rsi_val = self.data_cache.get_indicator('rsi', '1m') if self.data_cache else 50
                        
                        # Determine exec_reason
                        min_conf = get_pattern_config(pattern_name).get('min_confidence', 0.60)
                        if signal:
                            score = getattr(signal, 'confidence', 0)
                            if score >= min_conf:
                                exec_reason = "pass"
                            else:
                                exec_reason = "score_below_min"
                        else:
                            exec_reason = "no_setup"
                        
                        # Write CSV row
                        sink.write(
                            pattern=pattern_name,
                            event="EVAL",
                            price=current_price,
                            score=getattr(signal, 'confidence', 0) if signal else 0,
                            min_score=min_conf,
                            adx=adx_val,
                            atr=atr_val,
                            rsi=rsi_val,
                            exec_reason=exec_reason
                        )
                    except Exception as e:
                        logger.debug(f"CSV telemetry write failed: {e}")
                
                # Add pattern evaluation telemetry
                from .pattern_config import TRACE
                if TRACE.get('pattern', False) and signal:
                    bar_ts_utc = data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if len(data) > 0 else 'unknown'
                    min_score = get_pattern_config(pattern_name).get('min_confidence', 0.60)
                    
                    # Get signal attributes safely
                    score = getattr(signal, 'confidence', 0)
                    trigger_price = getattr(signal, 'entry_price', current_price)
                    
                    # Mock filter values - these would come from actual pattern analysis
                    filters = {
                        'exhaustion': True,
                        'pullback': True, 
                        'rsi_zone': True,
                        'vol_ok': True
                    }
                    
                    confirm_range_atr = 1.0  # Mock value
                    pullback_pct = 0.5      # Mock value
                    passes = score >= min_score
                    
                    logger.info(f"PATTERN_EVAL name={pattern_name} ts={bar_ts_utc} score={score:.2f} min={min_score:.2f} pass={passes} "
                               f"filters={filters} trigger={trigger_price} confirm_range_atr={confirm_range_atr:.2f} pullback_pct={pullback_pct:.2f}")
                    
                    # Write to CSV telemetry
                    try:
                        from .utils.telemetry_sink import get_telemetry_sink
                        sink = get_telemetry_sink()
                        sink.write(
                            pattern=pattern_name,
                            event="PATTERN_EVAL",
                            price=trigger_price,
                            score=score,
                            min_score=min_score,
                            adx=adx,
                            atr=atr,
                            confirm_range_atr=confirm_range_atr,
                            pullback_pct=pullback_pct
                        )
                    except Exception as e:
                        logger.debug(f"Telemetry write failed: {e}")
                    
                    # Log near misses
                    if not passes and (min_score - TRACE.get('near_miss_margin', 0.05)) <= score < min_score:
                        logger.info(f"PATTERN_EVAL name={pattern_name} ts={bar_ts_utc} score={score:.2f} min={min_score:.2f} pass=NEAR_MISS")
                
                if signal:
                    # Validate signal
                    if pattern.validate_signal(signal, spread, last_tick_time):
                        # Add pattern pass telemetry
                        from .pattern_config import TRACE
                        if TRACE.get('pattern', False):
                            entry_plan = f"{signal.action.value}@{signal.entry_price:.2f}"
                            t1_ticks = getattr(signal, 'target1_ticks', 5)
                            t2_ticks = getattr(signal, 'target2_ticks', 10)
                            stop_ticks = getattr(signal, 'stop_ticks', 6)
                            
                            logger.info(f"PATTERN_PASS name={pattern_name} entry={entry_plan} t1={t1_ticks} t2={t2_ticks} stop={stop_ticks}")
                            
                            # Write to CSV telemetry
                            try:
                                from .utils.telemetry_sink import get_telemetry_sink
                                sink = get_telemetry_sink()
                                sink.write(
                                    pattern=pattern_name,
                                    event="PATTERN_PASS",
                                    price=signal.entry_price,
                                    score=signal.confidence,
                                    entry_plan=entry_plan,
                                    t1_ticks=t1_ticks,
                                    t2_ticks=t2_ticks,
                                    stop_ticks=stop_ticks
                                )
                            except Exception as e:
                                logger.debug(f"Telemetry write failed: {e}")
                        
                        # Execute or explain why not
                        from .pattern_config import PATTERN_CONFIG
                        min_conf_for_pattern = PATTERN_CONFIG.get(pattern_name, {}).get('min_confidence', 0.60)
                        exec_result = self.execute_or_explain(
                            pattern_name, signal, signal.confidence, 
                            min_conf_for_pattern
                        )
                        
                        if exec_result.get('will_execute', False):
                            signals.append({
                                'pattern_name': pattern_name,
                                'signal': signal,
                                'priority': PATTERN_CONFIG.get(pattern_name, {}).get('priority', 0)
                            })
                
            except Exception as e:
                logger.error(f"Error scanning pattern {pattern_name}: {e}")
        
        # Return highest priority signal if multiple found
        if signals:
            signals.sort(key=lambda x: (x['signal'].confidence, x['priority']), reverse=True)
            best_signal = signals[0]
            
            self.active_signal = best_signal['signal']
            self.last_signal_time = datetime.now(timezone.utc)
            
            return best_signal
        
        # If no pattern signals, try technical analysis fallback
        ta_signal = self.technical_analysis.analyze(data, current_price)
        if ta_signal:
            logger.info(f"Using technical analysis fallback: {ta_signal['action']} "
                       f"with {ta_signal['confidence']:.2f} confidence")
            return {
                'pattern_name': 'technical_analysis_fallback',
                'signal': ta_signal,
                'priority': 0  # Lowest priority
            }
        
        # Check for probe trade on idle (discovery mode only)
        try:
            from .pattern_config import DISCOVERY_MODE, PROBE
            if DISCOVERY_MODE and PROBE.get('enabled', False):
                # Check if we've been idle too long
                idle_minutes = PROBE.get('idle_minutes', 10)
                now = datetime.now(timezone.utc)
                
                if self.last_fill_time is None:
                    # No fills yet, check against startup time (use last_signal_time as proxy)
                    idle_since = self.last_signal_time or now
                else:
                    idle_since = self.last_fill_time
                
                idle_duration = (now - idle_since).total_seconds() / 60 if idle_since else idle_minutes + 1
                
                if idle_duration >= idle_minutes:
                    # Check data freshness and ATR
                    if self.data_cache:
                        atr = self.data_cache.get_indicator('atr', '1m')
                        if atr and atr > 0 and len(data) > 5:
                            # Determine direction based on simple momentum
                            close_price = data['close'].iloc[-1]
                            ma_5 = data['close'].tail(5).mean()
                            
                            # Create probe signal
                            from .patterns.base_pattern import TradingSignal, SignalAction
                            
                            is_long = close_price > ma_5
                            action = SignalAction.BUY if is_long else SignalAction.SELL
                            
                            # Calculate entry as limit order near current price
                            entry_offset = 1 * 0.25  # 1 tick offset for limit order
                            entry_price = current_price - entry_offset if is_long else current_price + entry_offset
                            
                            # Set probe targets
                            t1_ticks = PROBE.get('t1_ticks', 4)
                            stop_ticks = PROBE.get('stop_ticks', 8)
                            
                            if is_long:
                                take_profit = entry_price + (t1_ticks * 0.25)
                                stop_loss = entry_price - (stop_ticks * 0.25)
                            else:
                                take_profit = entry_price - (t1_ticks * 0.25)
                                stop_loss = entry_price + (stop_ticks * 0.25)
                            
                            probe_signal = TradingSignal(
                                action=action,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                confidence=0.30,  # Low confidence for probe
                                pattern_name='probe_trade',
                                contracts=PROBE.get('size', 1)
                            )
                            
                            # Log probe activation
                            logger.info(f"PROBE_ARMED reason=\"idle\" idle_min={idle_duration:.1f} atr={atr:.2f}")
                            
                            # Write to CSV
                            from .utils.telemetry_sink import get_telemetry_sink
                            sink = get_telemetry_sink()
                            sink.write(
                                pattern='probe_trade',
                                event='PROBE',
                                price=current_price,
                                score=0.30,
                                atr=atr,
                                exec_reason='idle_probe'
                            )
                            
                            # Update last signal time to prevent rapid probes
                            self.last_signal_time = now
                            
                            return {
                                'pattern_name': 'probe_trade',
                                'signal': probe_signal,
                                'priority': -1  # Lower than TA fallback
                            }
        except Exception as e:
            logger.debug(f"Probe trade check failed: {e}")
        
        return None
    
    def execute_or_explain(self, pattern_name: str, signal, score: float, min_conf: float) -> Dict:
        """
        Determine if signal will execute and log detailed reason if not
        
        Args:
            pattern_name: Name of pattern generating signal
            signal: The trading signal object
            score: Signal confidence score
            min_conf: Minimum confidence threshold
            
        Returns:
            Dict with 'will_execute' bool and 'reason' if not executing
        """
        from .pattern_config import TELEMETRY
        
        try:
            from .utils.telemetry_sink import get_telemetry_sink
            sink = get_telemetry_sink()
        except:
            sink = None
        
        # Check 1: Position flat?
        if hasattr(self, 'position_manager') and self.position_manager:
            if self.position_manager.has_position():
                reason = "not_flat"
                logger.info(f"DECISION_TRACE pattern={pattern_name} reason={reason} score={score:.3f}")
                if sink and TELEMETRY.get('csv_eval_all', False):
                    sink.write(
                        pattern=pattern_name,
                        event='DECISION_TRACE',
                        price=getattr(signal, 'entry_price', 0),
                        score=score,
                        exec_reason=reason
                    )
                return {'will_execute': False, 'reason': reason}
        
        # Check 2: Cooldown active?
        now = datetime.now(timezone.utc)
        if self.last_signal_time and (now - self.last_signal_time).total_seconds() < 60:
            reason = f"cooldown_{int((now - self.last_signal_time).total_seconds())}s"
            logger.info(f"DECISION_TRACE pattern={pattern_name} reason={reason} score={score:.3f}")
            if sink and TELEMETRY.get('csv_eval_all', False):
                sink.write(
                    pattern=pattern_name,
                    event='DECISION_TRACE',
                    price=getattr(signal, 'entry_price', 0),
                    score=score,
                    exec_reason=reason
                )
            return {'will_execute': False, 'reason': reason}
        
        # Check 3: Risk manager allows?
        if hasattr(self, 'risk_manager') and self.risk_manager:
            allowed, risk_reason = self.risk_manager.allow_new_trade()
            if not allowed:
                reason = f"risk_block:{risk_reason}"
                logger.info(f"RISK_BLOCK pattern={pattern_name} reason={risk_reason} score={score:.3f}")
                if sink and TELEMETRY.get('csv_eval_all', False):
                    sink.write(
                        pattern=pattern_name,
                        event='RISK_BLOCK',
                        price=getattr(signal, 'entry_price', 0),
                        score=score,
                        exec_reason=risk_reason
                    )
                return {'will_execute': False, 'reason': reason}
        
        # Signal passes all checks
        logger.info(f"EXEC_ATTEMPT pattern={pattern_name} score={score:.3f} action={getattr(signal, 'action', 'UNKNOWN')} "
                   f"entry={getattr(signal, 'entry_price', 0):.2f}")
        
        if sink and TELEMETRY.get('csv_exec', False):
            sink.write(
                pattern=pattern_name,
                event='EXEC_ATTEMPT',
                price=getattr(signal, 'entry_price', 0),
                score=score,
                exec_reason='pass_all_checks'
            )
        
        return {'will_execute': True, 'reason': 'pass_all_checks'}
    
    def update_pattern_data(self, pattern_name: str, timeframe: str, data: pd.DataFrame):
        """
        Update timeframe data for specific pattern
        
        Args:
            pattern_name: Name of pattern
            timeframe: Timeframe ('1m', '5m', '1h')
            data: OHLCV data for timeframe
        """
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            if hasattr(pattern, 'update_timeframe_data'):
                pattern.update_timeframe_data(timeframe, data)
    
    def update_all_patterns_data(self, timeframe_data: Dict[str, pd.DataFrame]):
        """
        Update all patterns with multi-timeframe data
        
        Args:
            timeframe_data: Dictionary of timeframe -> DataFrame
        """
        for pattern_name, pattern in self.patterns.items():
            if hasattr(pattern, 'update_timeframe_data'):
                for timeframe, data in timeframe_data.items():
                    pattern.update_timeframe_data(timeframe, data)
    
    def on_trade_filled(self):
        """Update last fill time for probe trade tracking"""
        self.last_fill_time = datetime.now(timezone.utc)
        logger.debug(f"Updated last_fill_time to {self.last_fill_time}")
    
    def on_trade_result(self, pattern_name: str, pnl: float, is_win: bool):
        """
        Update pattern statistics after trade completion
        
        Args:
            pattern_name: Name of pattern that generated the trade
            pnl: Trade P&L
            is_win: Whether trade was profitable
        """
        if pattern_name in self.patterns:
            # Update pattern's internal stats
            self.patterns[pattern_name].update_statistics(pnl, is_win)
            
            # Update manager's stats
            stats = self.pattern_stats[pattern_name]
            stats['trades'] += 1
            stats['total_pnl'] += pnl
            
            if is_win:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            
            logger.info(f"{pattern_name} trade result - PnL: ${pnl:.2f}, "
                       f"Total trades: {stats['trades']}, Win rate: {win_rate:.1f}%")
    
    def get_pattern_metrics(self) -> Dict:
        """
        Get metrics for all patterns
        
        Returns:
            Dictionary of pattern metrics
        """
        metrics = {}
        
        for pattern_name, pattern in self.patterns.items():
            if hasattr(pattern, 'get_pattern_metrics'):
                metrics[pattern_name] = pattern.get_pattern_metrics()
            else:
                # Basic metrics
                stats = self.pattern_stats[pattern_name]
                metrics[pattern_name] = {
                    'enabled': pattern.is_enabled,
                    'trades': stats['trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'win_rate': (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0,
                    'total_pnl': stats['total_pnl']
                }
        
        return metrics
    
    def save_state(self) -> Dict:
        """
        Save state of all patterns
        
        Returns:
            State dictionary
        """
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'patterns': {},
            'pattern_stats': self.pattern_stats
        }
        
        for pattern_name, pattern in self.patterns.items():
            state['patterns'][pattern_name] = pattern.get_state()
        
        return state
    
    def load_state(self, state: Dict):
        """
        Load state for all patterns
        
        Args:
            state: State dictionary
        """
        if 'patterns' in state:
            for pattern_name, pattern_state in state['patterns'].items():
                if pattern_name in self.patterns:
                    self.patterns[pattern_name].load_state(pattern_state)
        
        if 'pattern_stats' in state:
            self.pattern_stats = state['pattern_stats']
    
    def reset_daily_stats(self):
        """Reset daily statistics for all patterns"""
        for pattern in self.patterns.values():
            pattern.reset_daily_stats()
    
    def get_regime_status(self, current_price: float, data: pd.DataFrame = None) -> Dict:
        """
        Get current market regime status
        
        Args:
            current_price: Current market price
            data: OHLCV data for calculations
            
        Returns:
            Regime status dictionary
        """
        return self.regime_detector.get_regime_status(current_price, data)
    
    def update_atr_history(self, atr_value: float):
        """
        Update ATR history for regime detection
        
        Args:
            atr_value: Current ATR value
        """
        self.regime_detector.update_atr_history(atr_value)
    
    def update_ransac_r2(self, r2_value: float):
        """
        Update RANSAC R² for trend quality
        
        Args:
            r2_value: R² value from RANSAC regression
        """
        self.regime_detector.update_ransac_r2(r2_value)
    
    def disable_pattern(self, pattern_name: str):
        """
        Disable a specific pattern
        
        Args:
            pattern_name: Name of pattern to disable
        """
        if pattern_name in self.patterns:
            self.patterns[pattern_name].disable()
            logger.info(f"Pattern {pattern_name} disabled")
    
    def enable_pattern(self, pattern_name: str):
        """
        Enable a specific pattern
        
        Args:
            pattern_name: Name of pattern to enable
        """
        if pattern_name in self.patterns:
            self.patterns[pattern_name].enable()
            logger.info(f"Pattern {pattern_name} enabled")
    
    def get_active_patterns(self) -> List[str]:
        """
        Get list of currently active patterns
        
        Returns:
            List of active pattern names
        """
        return [name for name, pattern in self.patterns.items() if pattern.is_enabled]


def integrate_with_nq_bot(bot_instance):
    """
    Helper function to integrate pattern manager with existing NQ bot
    
    Args:
        bot_instance: Instance of the NQ trading bot
    """
    # Add pattern manager to bot
    bot_instance.pattern_manager = PatternManager()
    
    # Add method to scan patterns
    def scan_patterns(self, data, current_price):
        """Scan all patterns for signals"""
        if hasattr(self, 'pattern_manager'):
            # Get current spread from broker
            spread = getattr(self, 'current_spread', 0.25)
            last_tick_time = getattr(self, 'last_tick_time', datetime.now(timezone.utc))
            
            signal_data = self.pattern_manager.scan_all_patterns(
                data, current_price, spread, last_tick_time
            )
            
            if signal_data:
                return signal_data['signal']
        return None
    
    # Bind method to bot instance
    import types
    bot_instance.scan_patterns = types.MethodType(scan_patterns, bot_instance)
    
    # Add method to update pattern metrics in heartbeat
    original_heartbeat = getattr(bot_instance, 'write_heartbeat', None)
    
    def enhanced_heartbeat(self, **kwargs):
        """Enhanced heartbeat with pattern metrics"""
        if hasattr(self, 'pattern_manager'):
            pattern_metrics = self.pattern_manager.get_pattern_metrics()
            kwargs['pattern_metrics'] = pattern_metrics
        
        if original_heartbeat:
            original_heartbeat(**kwargs)
    
    bot_instance.write_heartbeat = types.MethodType(enhanced_heartbeat, bot_instance)
    
    logger.info("Pattern manager integrated with NQ bot")
    
    return bot_instance