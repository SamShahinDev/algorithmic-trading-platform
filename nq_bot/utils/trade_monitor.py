"""
Trade Monitor for NQ Trading Bot
Monitors MAE (Maximum Adverse Excursion) and time-based exits
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

# MAE and time thresholds
MAE_THRESHOLD_TICKS = 8
MAE_WINDOW_SECONDS = 30
REVERSE_THRESHOLD_TICKS = 3
REVERSE_WINDOW_SECONDS = 15
TIME_TO_PROFIT_SECONDS = 90
PROFIT_THRESHOLD_TICKS = 5

# NQ constants
NQ_TICK = 0.25
USD_PER_TICK = 5.0

class TradeMonitor:
    """
    Monitors active trades for MAE violations and time-based exits
    - MAE ≥ 8 ticks within 30s → flatten
    - Reverse > 3 ticks within 15s → flatten
    - No +5 ticks by 90s → exit/trail
    """
    
    def __init__(self):
        """Initialize TradeMonitor"""
        self.active_trade = None
        self.mae_tracking = {}
        self.exit_signal = None
        
        logger.info("TradeMonitor initialized with MAE/time monitoring")
    
    def start_monitoring(self, fill_price: float, is_long: bool, pattern: str = None):
        """
        Start monitoring a new trade
        
        Args:
            fill_price: Actual fill price (not signal price)
            is_long: True for long, False for short
            pattern: Pattern that generated the trade
        """
        self.active_trade = {
            'fill_price': fill_price,
            'is_long': is_long,
            'pattern': pattern,
            'entry_time': datetime.now(timezone.utc),
            'entry_timestamp': time.time(),
            'mae': 0.0,  # Maximum Adverse Excursion
            'mae_ticks': 0.0,
            'mfe': 0.0,  # Maximum Favorable Excursion
            'mfe_ticks': 0.0,
            'max_price': fill_price,
            'min_price': fill_price,
            'first_reverse': None,
            'profit_reached': False,
            'profit_reached_time': None
        }
        
        self.exit_signal = None
        
        logger.info(f"Started monitoring {'LONG' if is_long else 'SHORT'} trade")
        logger.info(f"  Fill price: {fill_price:.2f}")
        logger.info(f"  Pattern: {pattern}")
        logger.info(f"  MAE threshold: {MAE_THRESHOLD_TICKS} ticks in {MAE_WINDOW_SECONDS}s")
        logger.info(f"  Reverse threshold: {REVERSE_THRESHOLD_TICKS} ticks in {REVERSE_WINDOW_SECONDS}s")
        logger.info(f"  Time limit: {TIME_TO_PROFIT_SECONDS}s to reach +{PROFIT_THRESHOLD_TICKS} ticks")
    
    def check_exit_conditions(self, current_price: float) -> Optional[Dict]:
        """
        Check if any exit conditions are met
        
        Args:
            current_price: Current market price
            
        Returns:
            Exit signal dict or None
        """
        if not self.active_trade:
            return None
        
        fill_price = self.active_trade['fill_price']
        is_long = self.active_trade['is_long']
        entry_time = self.active_trade['entry_time']
        now = datetime.now(timezone.utc)
        elapsed_seconds = (now - entry_time).total_seconds()
        
        # Calculate current P&L in ticks
        if is_long:
            pnl_ticks = (current_price - fill_price) / NQ_TICK
            adverse_move = fill_price - current_price
        else:
            pnl_ticks = (fill_price - current_price) / NQ_TICK
            adverse_move = current_price - fill_price
        
        # Update max/min prices
        self.active_trade['max_price'] = max(self.active_trade['max_price'], current_price)
        self.active_trade['min_price'] = min(self.active_trade['min_price'], current_price)
        
        # Update MAE (Maximum Adverse Excursion)
        if is_long:
            mae_price = fill_price - self.active_trade['min_price']
            mfe_price = self.active_trade['max_price'] - fill_price
        else:
            mae_price = self.active_trade['max_price'] - fill_price
            mfe_price = fill_price - self.active_trade['min_price']
        
        mae_ticks = mae_price / NQ_TICK
        mfe_ticks = mfe_price / NQ_TICK
        
        self.active_trade['mae'] = mae_price
        self.active_trade['mae_ticks'] = mae_ticks
        self.active_trade['mfe'] = mfe_price
        self.active_trade['mfe_ticks'] = mfe_ticks
        
        # Check profit threshold reached
        if pnl_ticks >= PROFIT_THRESHOLD_TICKS and not self.active_trade['profit_reached']:
            self.active_trade['profit_reached'] = True
            self.active_trade['profit_reached_time'] = now
            logger.info(f"✅ Profit threshold reached: +{pnl_ticks:.1f} ticks at {elapsed_seconds:.1f}s")
        
        # 1. MAE Check: ≥8 ticks adverse within 30s
        if mae_ticks >= MAE_THRESHOLD_TICKS and elapsed_seconds <= MAE_WINDOW_SECONDS:
            logger.warning(f"⚠️ MAE VIOLATION: {mae_ticks:.1f} ticks in {elapsed_seconds:.1f}s")
            return {
                'action': 'FLATTEN',
                'reason': f'MAE_{MAE_THRESHOLD_TICKS}_{MAE_WINDOW_SECONDS}s',
                'mae_ticks': mae_ticks,
                'elapsed_seconds': elapsed_seconds,
                'current_pnl': pnl_ticks * NQ_TICK * USD_PER_TICK
            }
        
        # 2. Reverse Check: >3 ticks adverse within 15s
        if pnl_ticks <= -REVERSE_THRESHOLD_TICKS and elapsed_seconds <= REVERSE_WINDOW_SECONDS:
            logger.warning(f"⚠️ QUICK REVERSE: {-pnl_ticks:.1f} ticks against in {elapsed_seconds:.1f}s")
            return {
                'action': 'FLATTEN',
                'reason': f'REV_{REVERSE_THRESHOLD_TICKS}_{REVERSE_WINDOW_SECONDS}s',
                'reverse_ticks': -pnl_ticks,
                'elapsed_seconds': elapsed_seconds,
                'current_pnl': pnl_ticks * NQ_TICK * USD_PER_TICK
            }
        
        # 3. Time Check: Not +5 ticks by 90s
        if elapsed_seconds >= TIME_TO_PROFIT_SECONDS and not self.active_trade['profit_reached']:
            logger.warning(f"⚠️ TIME LIMIT: Failed to reach +{PROFIT_THRESHOLD_TICKS} ticks in {TIME_TO_PROFIT_SECONDS}s")
            
            # Decide between exit or trail based on current P&L
            if pnl_ticks > 0:
                # In profit but not enough - trail stop
                return {
                    'action': 'TRAIL',
                    'reason': f'TIME_{TIME_TO_PROFIT_SECONDS}s',
                    'current_pnl_ticks': pnl_ticks,
                    'elapsed_seconds': elapsed_seconds,
                    'trail_distance': 2  # Trail by 2 ticks
                }
            else:
                # Not in profit - exit
                return {
                    'action': 'FLATTEN',
                    'reason': f'TIME_{TIME_TO_PROFIT_SECONDS}s',
                    'current_pnl_ticks': pnl_ticks,
                    'elapsed_seconds': elapsed_seconds,
                    'current_pnl': pnl_ticks * NQ_TICK * USD_PER_TICK
                }
        
        # Log monitoring status every 10 seconds
        if int(elapsed_seconds) % 10 == 0 and int(elapsed_seconds) > 0:
            logger.debug(f"Trade monitor: {elapsed_seconds:.0f}s, P&L: {pnl_ticks:.1f} ticks, "
                        f"MAE: {mae_ticks:.1f} ticks, MFE: {mfe_ticks:.1f} ticks")
        
        return None
    
    def stop_monitoring(self, exit_reason: str = None):
        """
        Stop monitoring the current trade
        
        Args:
            exit_reason: Reason for stopping (e.g., "STOP", "TARGET", "MAE_8_30s")
        """
        if self.active_trade:
            elapsed = (datetime.now(timezone.utc) - self.active_trade['entry_time']).total_seconds()
            mae = self.active_trade['mae_ticks']
            mfe = self.active_trade['mfe_ticks']
            
            logger.info(f"Stopped monitoring trade after {elapsed:.1f}s")
            logger.info(f"  Exit reason: {exit_reason}")
            logger.info(f"  MAE: {mae:.1f} ticks, MFE: {mfe:.1f} ticks")
            
            if self.active_trade['profit_reached']:
                time_to_profit = (self.active_trade['profit_reached_time'] - 
                                self.active_trade['entry_time']).total_seconds()
                logger.info(f"  Time to +{PROFIT_THRESHOLD_TICKS} ticks: {time_to_profit:.1f}s")
        
        self.active_trade = None
        self.exit_signal = None
    
    def get_trade_metrics(self) -> Optional[Dict]:
        """
        Get current trade metrics
        
        Returns:
            Dict with trade metrics or None if no active trade
        """
        if not self.active_trade:
            return None
        
        elapsed = (datetime.now(timezone.utc) - self.active_trade['entry_time']).total_seconds()
        
        return {
            'fill_price': self.active_trade['fill_price'],
            'is_long': self.active_trade['is_long'],
            'pattern': self.active_trade['pattern'],
            'elapsed_seconds': elapsed,
            'mae_ticks': self.active_trade['mae_ticks'],
            'mfe_ticks': self.active_trade['mfe_ticks'],
            'profit_reached': self.active_trade['profit_reached'],
            'max_price': self.active_trade['max_price'],
            'min_price': self.active_trade['min_price']
        }
    
    def is_monitoring(self) -> bool:
        """Check if currently monitoring a trade"""
        return self.active_trade is not None