"""
Market Hours Utility
Checks if markets are open for trading
"""

from datetime import datetime, time, timedelta
import pytz
from typing import Tuple, Optional


class MarketHours:
    """Check market hours for futures trading"""
    
    def __init__(self, timezone_str: str = 'US/Central'):
        self.timezone = pytz.timezone(timezone_str)
        
        # NQ Futures market hours (CT)
        # Sunday 5:00 PM - Friday 4:00 PM with daily break 4:00 PM - 5:00 PM
        self.futures_sessions = {
            'sunday_open': time(17, 0),     # 5:00 PM
            'monday_break_start': time(16, 0),  # 4:00 PM
            'monday_break_end': time(17, 0),    # 5:00 PM
            'friday_close': time(16, 0),    # 4:00 PM
            'daily_break_start': time(16, 0),  # 4:00 PM
            'daily_break_end': time(17, 0),    # 5:00 PM
        }
    
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if the futures market is open
        
        Args:
            dt: Datetime to check (default: now)
            
        Returns:
            True if market is open, False otherwise
        """
        
        if dt is None:
            dt = datetime.now(self.timezone)
        else:
            if dt.tzinfo is None:
                dt = self.timezone.localize(dt)
            else:
                dt = dt.astimezone(self.timezone)
        
        weekday = dt.weekday()
        current_time = dt.time()
        
        # Saturday - market closed
        if weekday == 5:
            return False
        
        # Sunday - opens at 5:00 PM
        if weekday == 6:
            return current_time >= self.futures_sessions['sunday_open']
        
        # Monday to Thursday
        if weekday in [0, 1, 2, 3]:
            # Check for daily break (4:00 PM - 5:00 PM)
            if self.futures_sessions['daily_break_start'] <= current_time < self.futures_sessions['daily_break_end']:
                return False
            return True
        
        # Friday - closes at 4:00 PM
        if weekday == 4:
            if current_time >= self.futures_sessions['friday_close']:
                return False
            # Check for break if before 4 PM
            if self.futures_sessions['daily_break_start'] <= current_time < self.futures_sessions['daily_break_end']:
                return False
            return True
        
        return False
    
    def get_next_open(self, dt: Optional[datetime] = None) -> datetime:
        """
        Get the next market open time
        
        Args:
            dt: Current datetime (default: now)
            
        Returns:
            Next market open datetime
        """
        
        if dt is None:
            dt = datetime.now(self.timezone)
        else:
            if dt.tzinfo is None:
                dt = self.timezone.localize(dt)
            else:
                dt = dt.astimezone(self.timezone)
        
        # If market is open, return current time
        if self.is_market_open(dt):
            return dt
        
        weekday = dt.weekday()
        current_time = dt.time()
        
        # During daily break
        if self.futures_sessions['daily_break_start'] <= current_time < self.futures_sessions['daily_break_end']:
            # Market reopens at 5:00 PM same day
            return dt.replace(hour=17, minute=0, second=0, microsecond=0)
        
        # Saturday or Friday after close
        if weekday == 5 or (weekday == 4 and current_time >= self.futures_sessions['friday_close']):
            # Next open is Sunday 5:00 PM
            days_until_sunday = (6 - weekday) % 7
            if days_until_sunday == 0 and weekday == 5:
                days_until_sunday = 1
            next_open = dt + timedelta(days=days_until_sunday)
            next_open = next_open.replace(hour=17, minute=0, second=0, microsecond=0)
            return next_open
        
        # Should not reach here
        return dt
    
    def get_session_times(self, dt: Optional[datetime] = None) -> Tuple[datetime, datetime]:
        """
        Get current or next session start and end times
        
        Args:
            dt: Reference datetime (default: now)
            
        Returns:
            Tuple of (session_start, session_end)
        """
        
        if dt is None:
            dt = datetime.now(self.timezone)
        else:
            if dt.tzinfo is None:
                dt = self.timezone.localize(dt)
            else:
                dt = dt.astimezone(self.timezone)
        
        weekday = dt.weekday()
        current_time = dt.time()
        
        # Determine session based on day and time
        if weekday == 6 and current_time >= self.futures_sessions['sunday_open']:
            # Sunday evening session
            session_start = dt.replace(hour=17, minute=0, second=0, microsecond=0)
            session_end = (dt + timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
        
        elif weekday in [0, 1, 2, 3]:  # Monday to Thursday
            if current_time < self.futures_sessions['daily_break_start']:
                # Morning/day session
                session_start = dt.replace(hour=17, minute=0, second=0, microsecond=0) - timedelta(days=1)
                session_end = dt.replace(hour=16, minute=0, second=0, microsecond=0)
            else:
                # Evening session
                session_start = dt.replace(hour=17, minute=0, second=0, microsecond=0)
                session_end = (dt + timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)
        
        elif weekday == 4:  # Friday
            if current_time < self.futures_sessions['friday_close']:
                # Friday session
                session_start = (dt - timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
                session_end = dt.replace(hour=16, minute=0, second=0, microsecond=0)
            else:
                # Next session is Sunday
                session_start = (dt + timedelta(days=2)).replace(hour=17, minute=0, second=0, microsecond=0)
                session_end = (dt + timedelta(days=3)).replace(hour=16, minute=0, second=0, microsecond=0)
        
        else:
            # Saturday - next session is Sunday
            session_start = (dt + timedelta(days=(6 - weekday) % 7)).replace(hour=17, minute=0, second=0, microsecond=0)
            session_end = session_start + timedelta(days=1) - timedelta(hours=1)
        
        return session_start, session_end
    
    def is_asian_session(self, dt: Optional[datetime] = None) -> bool:
        """Check if currently in Asian trading session (7 PM - 3 AM CT)"""
        if dt is None:
            dt = datetime.now(self.timezone)
        hour = dt.hour
        return hour >= 19 or hour < 3
    
    def is_european_session(self, dt: Optional[datetime] = None) -> bool:
        """Check if currently in European trading session (2 AM - 8 AM CT)"""
        if dt is None:
            dt = datetime.now(self.timezone)
        hour = dt.hour
        return 2 <= hour < 8
    
    def is_us_session(self, dt: Optional[datetime] = None) -> bool:
        """Check if currently in US trading session (8 AM - 4 PM CT)"""
        if dt is None:
            dt = datetime.now(self.timezone)
        hour = dt.hour
        return 8 <= hour < 16


# Create default instance
market_hours = MarketHours()

# Convenience functions
def is_market_open() -> bool:
    """Quick check if market is open"""
    return market_hours.is_market_open()


def get_next_open() -> datetime:
    """Get next market open time"""
    return market_hours.get_next_open()


def get_market_info() -> dict:
    """Get market information"""
    return {
        'is_open': market_hours.is_market_open(),
        'next_open': market_hours.get_next_open() if not market_hours.is_market_open() else None
    }