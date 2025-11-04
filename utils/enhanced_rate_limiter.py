"""
Enhanced Rate Limiting with Per-Pattern Controls
Prevents excessive trading and ensures safe practice
"""

from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedRateLimiter:
    """Advanced rate limiting with per-pattern and global controls"""
    
    def __init__(self):
        # Pattern-specific limits
        self.pattern_daily_trades = {}  # pattern -> count
        self.pattern_trade_timestamps = {}  # pattern -> deque of timestamps
        self.pattern_consecutive_losses = {}  # pattern -> loss count
        
        # Configurable limits
        self.MAX_TRADES_PER_PATTERN_PER_DAY = 5
        self.MAX_TRADES_PER_PATTERN_PER_HOUR = 3
        self.MAX_CONSECUTIVE_LOSSES_PER_PATTERN = 2
        
        # Global limits
        self.MAX_TOTAL_TRADES_PER_DAY = 15
        self.MAX_TOTAL_TRADES_PER_HOUR = 8
        self.MIN_SECONDS_BETWEEN_TRADES = 30
        
        # Tracking
        self.total_daily_trades = 0
        self.all_trade_timestamps = deque(maxlen=100)
        self.last_trade_time = None
        self.last_reset_date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"Rate limiter initialized with limits: "
                   f"{self.MAX_TRADES_PER_PATTERN_PER_DAY}/day per pattern, "
                   f"{self.MAX_TOTAL_TRADES_PER_DAY}/day total")
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().strftime("%Y%m%d")
        if self.last_reset_date != today:
            logger.info(f"New trading day {today}, resetting counters")
            self.pattern_daily_trades = {}
            self.total_daily_trades = 0
            self.last_reset_date = today
            self.pattern_consecutive_losses = {}
    
    def can_trade_pattern(self, pattern_name: str) -> Tuple[bool, str]:
        """Check if pattern can trade"""
        self._check_daily_reset()
        now = datetime.now()
        
        # Check consecutive losses
        losses = self.pattern_consecutive_losses.get(pattern_name, 0)
        if losses >= self.MAX_CONSECUTIVE_LOSSES_PER_PATTERN:
            return False, f"Pattern {pattern_name} has {losses} consecutive losses (max: {self.MAX_CONSECUTIVE_LOSSES_PER_PATTERN})"
        
        # Check pattern daily limit
        daily_count = self.pattern_daily_trades.get(pattern_name, 0)
        if daily_count >= self.MAX_TRADES_PER_PATTERN_PER_DAY:
            return False, f"Pattern {pattern_name} hit daily limit ({daily_count}/{self.MAX_TRADES_PER_PATTERN_PER_DAY})"
        
        # Check pattern hourly rate
        timestamps = self.pattern_trade_timestamps.get(pattern_name, deque())
        hour_ago = now.timestamp() - 3600
        recent_trades = sum(1 for ts in timestamps if ts > hour_ago)
        
        if recent_trades >= self.MAX_TRADES_PER_PATTERN_PER_HOUR:
            return False, f"Pattern {pattern_name} hit hourly limit ({recent_trades}/{self.MAX_TRADES_PER_PATTERN_PER_HOUR})"
        
        return True, "OK"
    
    def can_trade_global(self) -> Tuple[bool, str]:
        """Check global trading limits"""
        self._check_daily_reset()
        now = datetime.now()
        
        # Check minimum time between trades
        if self.last_trade_time:
            seconds_since_last = (now - self.last_trade_time).total_seconds()
            if seconds_since_last < self.MIN_SECONDS_BETWEEN_TRADES:
                wait_time = self.MIN_SECONDS_BETWEEN_TRADES - seconds_since_last
                return False, f"Too soon after last trade (wait {wait_time:.0f}s)"
        
        # Check global daily limit
        if self.total_daily_trades >= self.MAX_TOTAL_TRADES_PER_DAY:
            return False, f"Hit daily trade limit ({self.total_daily_trades}/{self.MAX_TOTAL_TRADES_PER_DAY})"
        
        # Check global hourly rate
        hour_ago = now.timestamp() - 3600
        recent_trades = sum(1 for ts in self.all_trade_timestamps if ts > hour_ago)
        
        if recent_trades >= self.MAX_TOTAL_TRADES_PER_HOUR:
            return False, f"Hit hourly trade limit ({recent_trades}/{self.MAX_TOTAL_TRADES_PER_HOUR})"
        
        return True, "OK"
    
    def record_pattern_trade(self, pattern_name: str):
        """Record a pattern trade"""
        now = datetime.now()
        
        # Update pattern tracking
        self.pattern_daily_trades[pattern_name] = self.pattern_daily_trades.get(pattern_name, 0) + 1
        
        if pattern_name not in self.pattern_trade_timestamps:
            self.pattern_trade_timestamps[pattern_name] = deque(maxlen=100)
        self.pattern_trade_timestamps[pattern_name].append(now.timestamp())
        
        # Update global tracking
        self.total_daily_trades += 1
        self.all_trade_timestamps.append(now.timestamp())
        self.last_trade_time = now
        
        logger.info(f"Recorded trade for {pattern_name}: "
                   f"pattern trades today={self.pattern_daily_trades[pattern_name]}, "
                   f"total trades today={self.total_daily_trades}")
    
    def record_trade_result(self, pattern_name: str, is_win: bool, pnl: float):
        """Record trade result for consecutive loss tracking"""
        if is_win:
            # Reset consecutive losses on win
            self.pattern_consecutive_losses[pattern_name] = 0
            logger.info(f"Pattern {pattern_name} WIN (${pnl:.2f}), losses reset")
        else:
            # Increment consecutive losses
            self.pattern_consecutive_losses[pattern_name] = \
                self.pattern_consecutive_losses.get(pattern_name, 0) + 1
            losses = self.pattern_consecutive_losses[pattern_name]
            logger.warning(f"Pattern {pattern_name} LOSS (${pnl:.2f}), "
                          f"consecutive losses: {losses}")
            
            if losses >= self.MAX_CONSECUTIVE_LOSSES_PER_PATTERN:
                logger.warning(f"⚠️ Pattern {pattern_name} disabled after {losses} consecutive losses")
    
    def get_status(self) -> Dict:
        """Get current rate limit status"""
        self._check_daily_reset()
        now = datetime.now()
        hour_ago = now.timestamp() - 3600
        
        return {
            "date": self.last_reset_date,
            "total_trades_today": self.total_daily_trades,
            "total_trades_last_hour": sum(1 for ts in self.all_trade_timestamps if ts > hour_ago),
            "pattern_trades": self.pattern_daily_trades.copy(),
            "pattern_losses": self.pattern_consecutive_losses.copy(),
            "limits": {
                "per_pattern_daily": self.MAX_TRADES_PER_PATTERN_PER_DAY,
                "per_pattern_hourly": self.MAX_TRADES_PER_PATTERN_PER_HOUR,
                "total_daily": self.MAX_TOTAL_TRADES_PER_DAY,
                "total_hourly": self.MAX_TOTAL_TRADES_PER_HOUR,
                "min_seconds_between": self.MIN_SECONDS_BETWEEN_TRADES
            }
        }
    
    def reset_pattern_losses(self, pattern_name: str):
        """Manually reset consecutive losses for a pattern"""
        self.pattern_consecutive_losses[pattern_name] = 0
        logger.info(f"Manually reset losses for pattern {pattern_name}")