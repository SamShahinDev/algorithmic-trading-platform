# File: trading_bot/risk/direction_lockout.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ExitReason(Enum):
    """Reasons for position exit"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    RISK_LIMIT = "risk_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    PHANTOM_POSITION = "phantom_position"
    UNKNOWN = "unknown"

@dataclass
class TradeExit:
    """Record of trade exit"""
    timestamp: datetime
    direction: str  # 'LONG' or 'SHORT'
    exit_reason: ExitReason
    pnl: float
    entry_price: float
    exit_price: float

class DirectionLockout:
    """
    Prevents whipsaw trades after stop losses
    Implements intelligent direction lockout based on exit reasons
    """
    
    def __init__(self, 
                 stop_loss_lockout_minutes: int = 5,
                 max_same_direction_stops: int = 2,
                 lockout_decay_minutes: int = 15):
        """
        Args:
            stop_loss_lockout_minutes: Minutes to lock direction after stop loss
            max_same_direction_stops: Max stops in same direction before extended lockout
            lockout_decay_minutes: Time for lockout history to decay
        """
        self.stop_loss_lockout_minutes = stop_loss_lockout_minutes
        self.max_same_direction_stops = max_same_direction_stops
        self.lockout_decay_minutes = lockout_decay_minutes
        
        # Tracking
        self.exit_history: List[TradeExit] = []
        self.locked_directions: Dict[str, datetime] = {}  # 'LONG'/'SHORT' -> lockout_until
        self.direction_stop_counts: Dict[str, int] = {'LONG': 0, 'SHORT': 0}
        
        # Statistics
        self.stats = {
            'total_lockouts': 0,
            'long_lockouts': 0,
            'short_lockouts': 0,
            'trades_prevented': 0,
            'extended_lockouts': 0
        }
    
    def can_trade_direction(self, direction: str) -> Tuple[bool, Optional[str]]:
        """
        Check if trading in a specific direction is allowed
        
        Args:
            direction: 'LONG' or 'SHORT'
            
        Returns:
            (can_trade, reason)
        """
        # Clean expired lockouts
        self._clean_expired_lockouts()
        
        # Check if direction is locked
        if direction in self.locked_directions:
            lockout_until = self.locked_directions[direction]
            if datetime.now() < lockout_until:
                remaining = (lockout_until - datetime.now()).total_seconds()
                self.stats['trades_prevented'] += 1
                
                logger.warning(f"⛔ Direction {direction} locked for {remaining:.0f}s")
                return False, f"Direction locked for {remaining:.0f}s after stop loss"
        
        return True, None
    
    def record_exit(self, direction: str, exit_reason: str, 
                   pnl: float = 0, entry_price: float = 0, exit_price: float = 0):
        """
        Record position exit and apply lockouts if needed
        
        Args:
            direction: 'LONG' or 'SHORT'
            exit_reason: Reason for exit (stop_loss, take_profit, etc.)
            pnl: Profit/loss of the trade
            entry_price: Entry price
            exit_price: Exit price
        """
        # Parse exit reason
        try:
            reason = ExitReason(exit_reason.lower())
        except ValueError:
            reason = ExitReason.UNKNOWN
        
        # Record exit
        exit_record = TradeExit(
            timestamp=datetime.now(),
            direction=direction,
            exit_reason=reason,
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exit_price
        )
        self.exit_history.append(exit_record)
        
        # Apply lockout for stop losses
        if reason == ExitReason.STOP_LOSS:
            self._apply_stop_loss_lockout(direction, pnl)
        
        # Clean old history
        self._clean_old_history()
        
        logger.info(f"📝 Exit recorded: {direction} {reason.value} P&L: ${pnl:.2f}")
    
    def _apply_stop_loss_lockout(self, direction: str, pnl: float):
        """Apply lockout after stop loss"""
        # Increment stop count for this direction
        self.direction_stop_counts[direction] += 1
        
        # Check recent stops in same direction
        recent_stops = self._count_recent_stops(direction, minutes=30)
        
        # Determine lockout duration
        if recent_stops >= self.max_same_direction_stops:
            # Extended lockout for repeated stops
            lockout_minutes = self.stop_loss_lockout_minutes * 3
            self.stats['extended_lockouts'] += 1
            logger.warning(f"🔒 EXTENDED LOCKOUT: {direction} locked for {lockout_minutes}min after {recent_stops} stops")
        else:
            # Normal lockout
            lockout_minutes = self.stop_loss_lockout_minutes
            logger.info(f"🔒 Direction lockout: {direction} locked for {lockout_minutes}min")
        
        # Apply lockout
        self.locked_directions[direction] = datetime.now() + timedelta(minutes=lockout_minutes)
        
        # Update stats
        self.stats['total_lockouts'] += 1
        if direction == 'LONG':
            self.stats['long_lockouts'] += 1
        else:
            self.stats['short_lockouts'] += 1
    
    def _count_recent_stops(self, direction: str, minutes: int = 30) -> int:
        """Count recent stop losses in same direction"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        count = 0
        for exit in self.exit_history:
            if (exit.timestamp > cutoff and 
                exit.direction == direction and 
                exit.exit_reason == ExitReason.STOP_LOSS):
                count += 1
        
        return count
    
    def _clean_expired_lockouts(self):
        """Remove expired lockouts"""
        now = datetime.now()
        expired = []
        
        for direction, lockout_until in self.locked_directions.items():
            if now >= lockout_until:
                expired.append(direction)
        
        for direction in expired:
            del self.locked_directions[direction]
            # Reset stop count when lockout expires
            self.direction_stop_counts[direction] = max(0, self.direction_stop_counts[direction] - 1)
            logger.info(f"✅ Direction {direction} unlocked")
    
    def _clean_old_history(self):
        """Remove old exit records"""
        cutoff = datetime.now() - timedelta(minutes=self.lockout_decay_minutes)
        self.exit_history = [e for e in self.exit_history if e.timestamp > cutoff]
    
    def get_lockout_status(self) -> Dict:
        """Get current lockout status"""
        self._clean_expired_lockouts()
        
        status = {
            'locked_directions': {},
            'stop_counts': self.direction_stop_counts.copy(),
            'recent_exits': [],
            'stats': self.stats.copy()
        }
        
        # Add lockout times
        for direction, until in self.locked_directions.items():
            remaining = (until - datetime.now()).total_seconds()
            if remaining > 0:
                status['locked_directions'][direction] = {
                    'locked_until': until.isoformat(),
                    'remaining_seconds': remaining
                }
        
        # Add recent exits
        for exit in self.exit_history[-5:]:  # Last 5 exits
            status['recent_exits'].append({
                'timestamp': exit.timestamp.isoformat(),
                'direction': exit.direction,
                'reason': exit.exit_reason.value,
                'pnl': exit.pnl
            })
        
        return status
    
    def reset_lockouts(self):
        """Reset all lockouts (emergency use only)"""
        logger.warning("🔓 All direction lockouts reset")
        self.locked_directions.clear()
        self.direction_stop_counts = {'LONG': 0, 'SHORT': 0}
    
    def should_skip_pattern(self, pattern_direction: str) -> bool:
        """
        Check if a pattern should be skipped due to lockouts
        
        Args:
            pattern_direction: 1 for LONG, -1 for SHORT
            
        Returns:
            True if pattern should be skipped
        """
        direction = 'LONG' if pattern_direction == 1 else 'SHORT'
        can_trade, _ = self.can_trade_direction(direction)
        return not can_trade