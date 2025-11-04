# File: trading_bot/execution/order_gate.py
import asyncio
import time
import hashlib
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class OrderSignal:
    """Represents an order signal"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    pattern: str
    size: int = 1
    stop_loss: float = 0
    take_profit: float = 0

class OrderGate:
    """
    Prevents duplicate orders through:
    1. Time-based cooldown
    2. Content fingerprinting
    3. Pattern-specific deduplication
    """
    
    def __init__(self, 
                 cooldown_secs: float = 5.0,
                 fingerprint_ttl: float = 30.0,
                 pattern_cooldown: float = 60.0):
        self.cooldown_secs = cooldown_secs
        self.fingerprint_ttl = fingerprint_ttl
        self.pattern_cooldown = pattern_cooldown
        
        self.last_order_time = 0.0
        self.recent_fingerprints: Dict[str, float] = {}
        self.pattern_last_fired: Dict[str, float] = {}
        self.lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'total_signals': 0,
            'blocked_cooldown': 0,
            'blocked_duplicate': 0,
            'blocked_pattern': 0,
            'approved': 0
        }
        
    def _fingerprint(self, signal: OrderSignal) -> str:
        """Create unique fingerprint for order signal"""
        # Round price to nearest tick (0.25 for NQ)
        tick_size = 0.25
        rounded_price = round(signal.entry_price / tick_size) * tick_size
        rounded_stop = round(signal.stop_loss / tick_size) * tick_size if signal.stop_loss else 0
        
        # Include key order parameters
        raw = f"{signal.symbol}|{signal.side}|{rounded_price:.2f}|{rounded_stop:.2f}|{signal.pattern}|{signal.size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    async def can_place_order(self, signal: OrderSignal) -> Tuple[bool, str, Dict]:
        """
        Check if order can be placed
        
        Returns:
            (can_place, reason, details)
        """
        async with self.lock:
            now = time.time()
            self.stats['total_signals'] += 1
            
            # 1. Check global cooldown
            time_since_last = now - self.last_order_time
            if time_since_last < self.cooldown_secs:
                remaining = self.cooldown_secs - time_since_last
                self.stats['blocked_cooldown'] += 1
                return False, f"cooldown_{remaining:.1f}s", {
                    'type': 'cooldown',
                    'remaining_seconds': remaining
                }
            
            # 2. Check pattern-specific cooldown
            if signal.pattern in self.pattern_last_fired:
                pattern_age = now - self.pattern_last_fired[signal.pattern]
                if pattern_age < self.pattern_cooldown:
                    remaining = self.pattern_cooldown - pattern_age
                    self.stats['blocked_pattern'] += 1
                    return False, f"pattern_cooldown_{remaining:.1f}s", {
                        'type': 'pattern_cooldown',
                        'pattern': signal.pattern,
                        'remaining_seconds': remaining
                    }
            
            # 3. Clean expired fingerprints
            self._cleanup_old_fingerprints(now)
            
            # 4. Check for duplicate signal
            fp = self._fingerprint(signal)
            if fp in self.recent_fingerprints:
                age = now - self.recent_fingerprints[fp]
                self.stats['blocked_duplicate'] += 1
                return False, "duplicate_signal", {
                    'type': 'duplicate',
                    'fingerprint': fp,
                    'age_seconds': age
                }
            
            # 5. Approved - register the order
            self.recent_fingerprints[fp] = now
            self.pattern_last_fired[signal.pattern] = now
            self.last_order_time = now
            self.stats['approved'] += 1
            
            return True, "approved", {
                'type': 'approved',
                'fingerprint': fp,
                'stats': self.get_stats()
            }
    
    def _cleanup_old_fingerprints(self, now: float):
        """Remove expired fingerprints"""
        expired = [fp for fp, timestamp in self.recent_fingerprints.items() 
                  if now - timestamp > self.fingerprint_ttl]
        for fp in expired:
            del self.recent_fingerprints[fp]
            
        # Also clean old pattern fires
        expired_patterns = [p for p, t in self.pattern_last_fired.items()
                          if now - t > self.pattern_cooldown * 2]
        for p in expired_patterns:
            del self.pattern_last_fired[p]
    
    def get_stats(self) -> Dict:
        """Get gate statistics"""
        total = self.stats['total_signals']
        if total == 0:
            return self.stats
            
        return {
            **self.stats,
            'approval_rate': (self.stats['approved'] / total) * 100,
            'cooldown_rate': (self.stats['blocked_cooldown'] / total) * 100,
            'duplicate_rate': (self.stats['blocked_duplicate'] / total) * 100,
            'pattern_block_rate': (self.stats['blocked_pattern'] / total) * 100
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {k: 0 for k in self.stats}