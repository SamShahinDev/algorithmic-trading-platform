"""
Comprehensive API Rate Limit Safety Guard System
Prevents rate limit violations and manages API usage across all trading bots
"""

import time
import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Setup logging for rate limit events
rate_limit_logger = logging.getLogger('rate_limit')
rate_limit_logger.setLevel(logging.INFO)

# Create file handler for rate limit logs
handler = logging.FileHandler('logs/rate_limits.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
rate_limit_logger.addHandler(handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - RATE LIMIT - %(levelname)s - %(message)s'))
rate_limit_logger.addHandler(console_handler)

class RateLimiter:
    """Centralized rate limiter with intelligent throttling and safety guards"""
    
    def __init__(self, max_requests: int = 200, time_window: int = 60, name: str = "General"):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
            name: Name of this rate limiter (for logging)
        """
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.name = name
        self.requests = deque()  # Track timestamps and types of requests
        
        # Thresholds
        self.warning_threshold = 0.8  # Warn at 80% usage
        self.critical_threshold = 0.9  # Emergency stop at 90%
        self.safety_buffer = 0.95  # Hard stop at 95%
        
        # Statistics
        self.total_requests = 0
        self.rejected_requests = 0
        self.warning_count = 0
        self.critical_count = 0
        
        rate_limit_logger.info(f"Rate limiter '{name}' initialized: {max_requests} requests per {time_window}s")
        
    async def check_and_track(self, request_type: str = "general") -> Tuple[bool, Dict]:
        """
        Check if request is safe and track it
        
        Args:
            request_type: Type of request for tracking
            
        Returns:
            Tuple of (is_safe, usage_stats)
        """
        # Clean up old requests first
        await self._cleanup_old_requests()
        
        current_count = len(self.requests)
        usage_percent = current_count / self.max_requests
        
        # Get current stats
        stats = self.get_current_usage()
        
        # Safety buffer - absolute hard stop
        if usage_percent >= self.safety_buffer:
            self.rejected_requests += 1
            rate_limit_logger.critical(
                f"🚨 RATE LIMIT SAFETY BUFFER [{self.name}]: {current_count}/{self.max_requests} "
                f"({usage_percent:.0%}) - REQUEST REJECTED!"
            )
            return False, stats
            
        # Emergency stop at critical threshold
        if usage_percent >= self.critical_threshold:
            self.critical_count += 1
            self.rejected_requests += 1
            rate_limit_logger.error(
                f"🛑 RATE LIMIT CRITICAL [{self.name}]: {current_count}/{self.max_requests} "
                f"({usage_percent:.0%}) - REQUEST REJECTED!"
            )
            return False, stats
            
        # Warning at warning threshold
        if usage_percent >= self.warning_threshold:
            self.warning_count += 1
            rate_limit_logger.warning(
                f"⚠️ RATE LIMIT WARNING [{self.name}]: {current_count}/{self.max_requests} "
                f"({usage_percent:.0%}) - Approaching limit!"
            )
            
        # Track the request
        self.requests.append({
            'timestamp': time.time(),
            'type': request_type,
            'usage_at_time': usage_percent
        })
        self.total_requests += 1
        
        # Log every 10th request in debug mode
        if self.total_requests % 10 == 0:
            rate_limit_logger.debug(
                f"[{self.name}] Request #{self.total_requests}: {current_count}/{self.max_requests} "
                f"({usage_percent:.0%})"
            )
        
        return True, stats
        
    async def _cleanup_old_requests(self):
        """Remove requests older than time window"""
        cutoff_time = time.time() - self.time_window
        
        removed_count = 0
        while self.requests and self.requests[0]['timestamp'] < cutoff_time:
            self.requests.popleft()
            removed_count += 1
            
        if removed_count > 0:
            rate_limit_logger.debug(f"[{self.name}] Cleaned up {removed_count} old requests")
            
    def get_current_usage(self) -> Dict:
        """Get current API usage statistics"""
        # Synchronous cleanup
        cutoff_time = time.time() - self.time_window
        while self.requests and self.requests[0]['timestamp'] < cutoff_time:
            self.requests.popleft()
            
        count = len(self.requests)
        percentage = (count / self.max_requests) * 100 if self.max_requests > 0 else 0
        
        return {
            'count': count,
            'limit': self.max_requests,
            'percentage': percentage,
            'remaining': max(0, self.max_requests - count),
            'status': self._get_status(percentage),
            'total_requests': self.total_requests,
            'rejected_requests': self.rejected_requests,
            'warning_count': self.warning_count,
            'critical_count': self.critical_count
        }
        
    def _get_status(self, percentage: float) -> str:
        """Get status emoji based on usage percentage"""
        if percentage < 50:
            return "🟢"  # Green - Safe
        elif percentage < 70:
            return "🟡"  # Yellow - Caution
        elif percentage < 85:
            return "🟠"  # Orange - Warning
        else:
            return "🔴"  # Red - Critical
            
    async def wait_if_needed(self) -> bool:
        """
        Wait if rate limit is approaching, returns True if waited
        """
        stats = self.get_current_usage()
        
        if stats['percentage'] >= 85:
            # Calculate wait time based on oldest request
            if self.requests:
                oldest_time = self.requests[0]['timestamp']
                wait_time = max(0, (oldest_time + self.time_window) - time.time() + 1)
                
                if wait_time > 0:
                    rate_limit_logger.info(
                        f"[{self.name}] Auto-throttling: waiting {wait_time:.1f}s "
                        f"(usage at {stats['percentage']:.0f}%)"
                    )
                    await asyncio.sleep(wait_time)
                    return True
                    
        return False
        
    def reset(self):
        """Reset rate limiter (for testing or emergency reset)"""
        self.requests.clear()
        rate_limit_logger.info(f"[{self.name}] Rate limiter reset")


class SmartThrottle:
    """Dynamically adjust bot speed based on rate limit usage"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        
    def get_sleep_time(self, base_sleep: float = 5) -> float:
        """
        Calculate optimal sleep time based on current rate limit usage
        
        Args:
            base_sleep: Base sleep time in seconds
            
        Returns:
            Adjusted sleep time
        """
        usage = self.rate_limiter.get_current_usage()['percentage']
        
        if usage < 50:
            multiplier = 1.0  # Normal speed
        elif usage < 70:
            multiplier = 1.5  # 50% slower
        elif usage < 85:
            multiplier = 2.0  # 100% slower
        elif usage < 90:
            multiplier = 3.0  # 200% slower
        else:
            multiplier = 5.0  # Emergency slow (400% slower)
            
        adjusted_sleep = base_sleep * multiplier
        
        # Log significant changes
        if multiplier > 1.5:
            rate_limit_logger.info(
                f"[{self.rate_limiter.name}] Throttling: {multiplier}x slower "
                f"(usage at {usage:.0f}%)"
            )
            
        return adjusted_sleep


class RateLimitDashboard:
    """Display real-time rate limit status"""
    
    def __init__(self, rate_limiters: Dict[str, RateLimiter]):
        self.rate_limiters = rate_limiters
        self.display_interval = 5  # seconds
        
    async def display_status(self):
        """Show real-time rate limit usage"""
        while True:
            lines = []
            lines.append("╔══════════════════════════════════════════════╗")
            lines.append("║           API RATE LIMIT STATUS              ║")
            lines.append("╠══════════════════════════════════════════════╣")
            
            for name, limiter in self.rate_limiters.items():
                stats = limiter.get_current_usage()
                line = f"║ {name:12s}: {stats['count']:3d}/{stats['limit']:3d} ({stats['percentage']:3.0f}%) {stats['status']} "
                
                # Add warning indicators
                if stats['rejected_requests'] > 0:
                    line += f"[REJ:{stats['rejected_requests']}] "
                    
                # Pad to consistent length
                line = line[:47] + "║"
                lines.append(line)
                
            # Add summary line
            lines.append("╠══════════════════════════════════════════════╣")
            
            # Calculate total usage across all limiters
            total_used = sum(l.get_current_usage()['count'] for l in self.rate_limiters.values())
            total_limit = sum(l.max_requests for l in self.rate_limiters.values())
            total_percentage = (total_used / total_limit * 100) if total_limit > 0 else 0
            
            lines.append(f"║ Total Usage: {total_used:4d}/{total_limit:4d} ({total_percentage:3.0f}%)             ║")
            lines.append("╚══════════════════════════════════════════════╝")
            
            # Clear screen and print (optional, can be removed if you don't want clearing)
            # print("\033[2J\033[H")  # Clear screen
            print("\n".join(lines))
            
            await asyncio.sleep(self.display_interval)


# Global rate limiters (singleton pattern)
_general_limiter = None
_historical_limiter = None
_order_limiter = None

def get_general_limiter() -> RateLimiter:
    """Get or create general API rate limiter"""
    global _general_limiter
    if _general_limiter is None:
        _general_limiter = RateLimiter(max_requests=180, time_window=60, name="General API")
    return _general_limiter

def get_historical_limiter() -> RateLimiter:
    """Get or create historical data rate limiter"""
    global _historical_limiter
    if _historical_limiter is None:
        _historical_limiter = RateLimiter(max_requests=50, time_window=30, name="Historical")
    return _historical_limiter

def get_order_limiter() -> RateLimiter:
    """Get or create order placement rate limiter"""
    global _order_limiter
    if _order_limiter is None:
        _order_limiter = RateLimiter(max_requests=20, time_window=60, name="Orders")
    return _order_limiter

def get_all_limiters() -> Dict[str, RateLimiter]:
    """Get all rate limiters for monitoring"""
    return {
        "General API": get_general_limiter(),
        "Historical": get_historical_limiter(),
        "Orders": get_order_limiter()
    }