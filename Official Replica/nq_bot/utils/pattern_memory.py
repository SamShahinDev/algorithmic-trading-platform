"""
Pattern Memory - Pattern tracking with state validation
Prevents using same patterns for entry and exit
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading state machine states"""
    IDLE = "IDLE"               # No position
    ENTERING = "ENTERING"       # Placing entry order
    OPEN = "OPEN"              # Position active
    EXITING = "EXITING"        # Placing exit order
    CLOSED = "CLOSED"          # Position closed


class PatternMemory:
    """Pattern tracking with state validation"""
    
    def __init__(self, cooldown_bars: int = 15, reuse_threshold_bars: int = 20):
        self.entry_patterns = []
        self.pattern_cooldowns = {}
        self._state = TradingState.IDLE
        self._state_history = []
        
        # Configuration
        self.cooldown_bars = cooldown_bars
        self.reuse_threshold_bars = reuse_threshold_bars
        
        # Tracking
        self.pattern_stats = {
            'patterns_recorded': 0,
            'patterns_blocked': 0,
            'cooldown_blocks': 0,
            'state_blocks': 0
        }
        
    def record_entry_pattern(self, pattern: Dict, bar_index: int) -> bool:
        """
        Record pattern at entry with state check
        
        Returns:
            True if pattern was recorded, False if blocked
        """
        
        # State validation
        if self._state not in [TradingState.IDLE, TradingState.ENTERING]:
            logger.warning(f"Invalid state for entry pattern: {self._state}")
            self.pattern_stats['state_blocks'] += 1
            return False
        
        # Check if pattern type is on cooldown
        if self._is_pattern_on_cooldown(pattern['pattern_type'], bar_index):
            logger.debug(f"Pattern {pattern['pattern_type']} on cooldown")
            self.pattern_stats['cooldown_blocks'] += 1
            return False
        
        # Record the pattern
        pattern_record = {
            'pattern_type': pattern.get('pattern_type'),
            'direction': pattern.get('direction'),
            'strength': pattern.get('strength', 0),
            'bar_index': bar_index,
            'timestamp': datetime.now(),
            'correlation_id': str(uuid.uuid4())[:8],
            'metadata': pattern.get('metadata', {})
        }
        
        self.entry_patterns.append(pattern_record)
        
        # Set cooldown for this pattern type
        self.pattern_cooldowns[pattern['pattern_type']] = bar_index + self.cooldown_bars
        
        # Update state
        self._transition_state(TradingState.ENTERING)
        
        self.pattern_stats['patterns_recorded'] += 1
        
        logger.info(f"Entry pattern recorded: {pattern['pattern_type']} at bar {bar_index} (ID: {pattern_record['correlation_id']})")
        
        return True
    
    def can_use_for_exit(self, pattern: Dict, current_bar_index: int) -> bool:
        """
        Validate pattern for exit with state check
        
        Returns:
            True if pattern can be used for exit
        """
        
        # State validation
        if self._state != TradingState.OPEN:
            logger.debug(f"Invalid state for exit pattern: {self._state}")
            return False
        
        # Check cooldown
        if self._is_pattern_on_cooldown(pattern.get('pattern_type'), current_bar_index):
            logger.debug(f"Pattern {pattern.get('pattern_type')} still on cooldown")
            return False
        
        # Prevent using same pattern as entry
        if self._is_same_as_entry_pattern(pattern, current_bar_index):
            logger.debug(f"Pattern {pattern.get('pattern_type')} too similar to entry")
            self.pattern_stats['patterns_blocked'] += 1
            return False
        
        return True
    
    def _is_pattern_on_cooldown(self, pattern_type: str, current_bar: int) -> bool:
        """Check if pattern type is on cooldown"""
        cooldown_bar = self.pattern_cooldowns.get(pattern_type, 0)
        return current_bar < cooldown_bar
    
    def _is_same_as_entry_pattern(self, pattern: Dict, current_bar: int) -> bool:
        """Check if pattern is too similar to recent entry patterns"""
        
        # Check last 3 entry patterns
        for entry_pattern in self.entry_patterns[-3:]:
            # Same type and direction
            if (pattern.get('pattern_type') == entry_pattern['pattern_type'] and
                pattern.get('direction') == entry_pattern['direction']):
                
                # Check time proximity
                bars_since = current_bar - entry_pattern['bar_index']
                if bars_since < self.reuse_threshold_bars:
                    logger.debug(
                        f"Pattern {pattern.get('pattern_type')} too close to entry "
                        f"({bars_since} bars, threshold: {self.reuse_threshold_bars})"
                    )
                    return True
        
        return False
    
    def _transition_state(self, new_state: TradingState):
        """Transition to new state with validation"""
        
        # Record state change
        self._state_history.append({
            'from': self._state,
            'to': new_state,
            'timestamp': datetime.now()
        })
        
        old_state = self._state
        self._state = new_state
        
        logger.debug(f"State transition: {old_state} -> {new_state}")
    
    def transition_to_open(self):
        """State transition after entry fill"""
        if self._state == TradingState.ENTERING:
            self._transition_state(TradingState.OPEN)
        else:
            logger.warning(f"Invalid transition to OPEN from {self._state}")
    
    def transition_to_exiting(self):
        """State transition when exit starts"""
        if self._state == TradingState.OPEN:
            self._transition_state(TradingState.EXITING)
        else:
            logger.warning(f"Invalid transition to EXITING from {self._state}")
    
    def transition_to_closed(self):
        """State transition after exit fill"""
        if self._state in [TradingState.EXITING, TradingState.OPEN]:
            self._transition_state(TradingState.CLOSED)
        else:
            logger.warning(f"Invalid transition to CLOSED from {self._state}")
    
    def reset(self):
        """Reset for new position cycle"""
        
        logger.info(f"Pattern memory reset (had {len(self.entry_patterns)} patterns)")
        
        # Clear patterns but keep some history
        if len(self.entry_patterns) > 10:
            # Keep last 10 for analysis
            self.entry_patterns = self.entry_patterns[-10:]
        
        # Clear cooldowns
        self.pattern_cooldowns.clear()
        
        # Reset state
        self._transition_state(TradingState.IDLE)
    
    def get_current_state(self) -> TradingState:
        """Get current trading state"""
        return self._state
    
    def is_state_valid_for_entry(self) -> bool:
        """Check if current state allows entry"""
        return self._state in [TradingState.IDLE, TradingState.CLOSED]
    
    def is_state_valid_for_exit(self) -> bool:
        """Check if current state allows exit"""
        return self._state == TradingState.OPEN
    
    def get_entry_pattern_summary(self) -> List[Dict]:
        """Get summary of entry patterns"""
        return [
            {
                'type': p['pattern_type'],
                'direction': p['direction'],
                'bar': p['bar_index'],
                'id': p['correlation_id']
            }
            for p in self.entry_patterns[-5:]  # Last 5 patterns
        ]
    
    def get_statistics(self) -> Dict:
        """Get pattern memory statistics"""
        return {
            'current_state': self._state.value,
            'patterns_in_memory': len(self.entry_patterns),
            'active_cooldowns': len(self.pattern_cooldowns),
            'stats': self.pattern_stats.copy(),
            'state_history': len(self._state_history)
        }


class PatternValidator:
    """Validate patterns for quality and reliability"""
    
    def __init__(self, min_strength: float = 0.5):
        self.min_strength = min_strength
        self.validation_stats = {
            'validated': 0,
            'rejected_weak': 0,
            'rejected_invalid': 0
        }
    
    def validate_pattern(self, pattern: Dict) -> Tuple[bool, str]:
        """
        Validate a pattern for use
        
        Returns:
            (is_valid, reason)
        """
        
        # Check required fields
        if not pattern.get('pattern_type'):
            self.validation_stats['rejected_invalid'] += 1
            return False, "Missing pattern type"
        
        if not pattern.get('direction'):
            self.validation_stats['rejected_invalid'] += 1
            return False, "Missing direction"
        
        # Check strength
        strength = pattern.get('strength', 0)
        if strength < self.min_strength:
            self.validation_stats['rejected_weak'] += 1
            return False, f"Pattern too weak: {strength:.2f} < {self.min_strength}"
        
        # Additional validation rules
        if pattern.get('pattern_type') == 'reversal':
            # Reversal patterns need higher strength
            if strength < 0.7:
                self.validation_stats['rejected_weak'] += 1
                return False, f"Reversal pattern needs strength > 0.7, got {strength:.2f}"
        
        self.validation_stats['validated'] += 1
        return True, "Valid"
    
    def get_statistics(self) -> Dict:
        """Get validation statistics"""
        total = sum(self.validation_stats.values())
        
        stats = {
            'total_patterns': total,
            'stats': self.validation_stats.copy()
        }
        
        if total > 0:
            stats['acceptance_rate'] = (self.validation_stats['validated'] / total) * 100
        
        return stats