"""
Time Synchronization Manager - Ensure reliable timestamps
Validates time sync between bot and broker
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TimeSyncManager:
    """Ensure reliable timestamps and time synchronization"""
    
    def __init__(self, broker_client, max_drift_seconds: float = 2.0):
        self.broker = broker_client
        self.max_drift = max_drift_seconds
        
        # Sync state
        self.last_sync_check: Optional[datetime] = None
        self.is_synced = True
        self.current_drift = 0.0
        self.sync_check_interval = 300  # 5 minutes
        
        # Statistics
        self.sync_stats = {
            'checks_performed': 0,
            'sync_failures': 0,
            'max_drift_seen': 0.0,
            'total_drift': 0.0
        }
    
    async def validate_sync(self) -> bool:
        """
        Check time sync with broker
        
        Returns:
            True if synchronized within tolerance
        """
        
        try:
            # Get broker server time
            broker_time = await self._get_broker_time()
            if not broker_time:
                logger.error("Failed to get broker time")
                self.is_synced = False
                return False
            
            # Get local time in UTC
            local_time = datetime.now(timezone.utc)
            
            # Calculate drift
            drift = abs((broker_time - local_time).total_seconds())
            self.current_drift = drift
            
            # Update statistics
            self.sync_stats['checks_performed'] += 1
            self.sync_stats['total_drift'] += drift
            if drift > self.sync_stats['max_drift_seen']:
                self.sync_stats['max_drift_seen'] = drift
            
            # Check if within tolerance
            self.is_synced = drift <= self.max_drift
            
            if not self.is_synced:
                logger.warning(
                    f"TIME DRIFT DETECTED: {drift:.1f}s "
                    f"(max allowed: {self.max_drift}s) - pattern exits disabled"
                )
                self.sync_stats['sync_failures'] += 1
            else:
                logger.debug(f"Time sync OK: drift={drift:.2f}s")
            
            self.last_sync_check = datetime.now()
            return self.is_synced
            
        except Exception as e:
            logger.error(f"Time sync check failed: {e}")
            self.is_synced = False
            self.sync_stats['sync_failures'] += 1
            return False
    
    async def _get_broker_time(self) -> Optional[datetime]:
        """Get current time from broker server"""
        
        try:
            # Try different methods based on broker API
            
            # Method 1: Direct server time endpoint
            if hasattr(self.broker, 'get_server_time'):
                return await self.broker.get_server_time()
            
            # Method 2: From market data timestamp
            if hasattr(self.broker, 'get_market_data'):
                market_data = await self.broker.get_market_data('NQ')
                if market_data and 'timestamp' in market_data:
                    return datetime.fromisoformat(market_data['timestamp'])
            
            # Method 3: From order timestamp
            if hasattr(self.broker, 'get_recent_orders'):
                orders = await self.broker.get_recent_orders(limit=1)
                if orders and 'timestamp' in orders[0]:
                    return datetime.fromisoformat(orders[0]['timestamp'])
            
            # Fallback: assume synchronized
            logger.warning("No broker time source available - assuming synchronized")
            return datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error getting broker time: {e}")
            return None
    
    def should_check_sync(self) -> bool:
        """
        Check if sync validation is needed
        
        Returns:
            True if sync check is due
        """
        
        if not self.last_sync_check:
            return True
        
        age = (datetime.now() - self.last_sync_check).total_seconds()
        return age >= self.sync_check_interval
    
    def is_time_sensitive_allowed(self) -> bool:
        """
        Check if time-sensitive operations are allowed
        
        Returns:
            True if time sync is good and recent
        """
        
        # Check if sync is current
        if self.should_check_sync():
            logger.debug("Time sync check is overdue")
            return False
        
        return self.is_synced
    
    async def continuous_sync_monitor(self):
        """Background task to monitor time sync"""
        
        logger.info("Starting continuous time sync monitoring")
        
        while True:
            try:
                # Check sync
                await self.validate_sync()
                
                # Adjust check interval based on drift
                if self.current_drift > self.max_drift * 0.5:
                    # Check more frequently if drift is high
                    self.sync_check_interval = 60  # 1 minute
                else:
                    # Normal interval
                    self.sync_check_interval = 300  # 5 minutes
                
                # Sleep until next check
                await asyncio.sleep(self.sync_check_interval)
                
            except Exception as e:
                logger.error(f"Time sync monitor error: {e}")
                await asyncio.sleep(60)
    
    def get_adjusted_time(self, adjustment_seconds: float = 0) -> datetime:
        """
        Get current time with optional adjustment
        
        Args:
            adjustment_seconds: Seconds to adjust (positive = future)
            
        Returns:
            Adjusted datetime
        """
        
        return datetime.now(timezone.utc) + timedelta(seconds=adjustment_seconds)
    
    def format_timestamp(self, dt: Optional[datetime] = None) -> str:
        """
        Format timestamp consistently
        
        Args:
            dt: Datetime to format (default: now)
            
        Returns:
            ISO format timestamp string
        """
        
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        return dt.isoformat()
    
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp string safely
        
        Args:
            timestamp_str: Timestamp string to parse
            
        Returns:
            Datetime object or None if invalid
        """
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str)
        except:
            try:
                # Try common formats
                for fmt in [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f"
                ]:
                    return datetime.strptime(timestamp_str, fmt)
            except:
                logger.error(f"Failed to parse timestamp: {timestamp_str}")
                return None
    
    def get_statistics(self) -> Dict:
        """Get time sync statistics"""
        
        stats = self.sync_stats.copy()
        stats['is_synced'] = self.is_synced
        stats['current_drift'] = self.current_drift
        stats['last_check'] = self.last_sync_check
        
        if stats['checks_performed'] > 0:
            stats['average_drift'] = stats['total_drift'] / stats['checks_performed']
            stats['sync_success_rate'] = (
                (stats['checks_performed'] - stats['sync_failures']) / 
                stats['checks_performed'] * 100
            )
        
        return stats


class MarketTimeManager:
    """Manage market hours and trading sessions"""
    
    def __init__(self):
        # NQ futures market hours (CT)
        self.market_open = {
            'sunday': (17, 0),    # 5:00 PM
            'monday': (0, 0),     # Continuous
            'tuesday': (0, 0),
            'wednesday': (0, 0),
            'thursday': (0, 0),
            'friday': (0, 0)      # Until 4:00 PM
        }
        
        self.market_close = {
            'friday': (16, 0),    # 4:00 PM
            'saturday': None,     # Closed
            'sunday': (23, 59)    # Until Monday
        }
        
        # Daily maintenance (CT)
        self.maintenance_start = (16, 0)  # 4:00 PM
        self.maintenance_end = (17, 0)    # 5:00 PM
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if market is open
        
        Args:
            check_time: Time to check (default: now)
            
        Returns:
            True if market is open
        """
        
        if check_time is None:
            check_time = datetime.now()
        
        # Convert to CT (Chicago Time)
        # This is simplified - production would use proper timezone handling
        
        weekday = check_time.weekday()
        current_hour = check_time.hour
        current_minute = check_time.minute
        
        # Saturday is closed
        if weekday == 5:
            return False
        
        # Check maintenance window (weekdays)
        if weekday < 5:  # Monday-Friday
            if (current_hour == 16 and current_minute >= 0) or \
               (current_hour == 16 and current_minute < 60):
                return False
        
        return True
    
    def time_until_open(self) -> Optional[timedelta]:
        """Get time until market opens"""
        
        if self.is_market_open():
            return timedelta(0)
        
        now = datetime.now()
        
        # Find next open time
        # Simplified - production would be more sophisticated
        if now.weekday() == 5:  # Saturday
            # Market opens Sunday 5 PM
            days_until_sunday = 6 - now.weekday()
            next_open = now + timedelta(days=days_until_sunday)
            next_open = next_open.replace(hour=17, minute=0, second=0)
        else:
            # Market opens after maintenance
            next_open = now.replace(hour=17, minute=0, second=0)
            if now.hour >= 17:
                next_open += timedelta(days=1)
        
        return next_open - now