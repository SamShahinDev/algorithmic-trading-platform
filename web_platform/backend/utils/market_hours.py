"""
Market Hours Validation for Futures Trading
Handles CME Globex hours for NQ futures
"""

from datetime import datetime, time, timezone
import pytz
from typing import Dict, Tuple, Optional
from enum import Enum

class MarketSession(Enum):
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    OVERNIGHT = "overnight"

class MarketHours:
    """
    Futures market hours checker for NQ (E-mini Nasdaq-100)
    All times in Central Time (Chicago)
    """
    
    def __init__(self):
        self.ct_tz = pytz.timezone('America/Chicago')
        
        # CME Globex hours for equity index futures (CT)
        # Sunday: Opens at 5:00 PM
        # Monday-Thursday: 5:00 PM previous day to 4:00 PM
        # Friday: Closes at 4:00 PM
        # Daily break: 4:00 PM - 5:00 PM
        
        self.schedule = {
            0: {  # Monday
                'open': time(17, 0),  # 5:00 PM Sunday
                'close': time(16, 0),  # 4:00 PM Monday
                'session': MarketSession.REGULAR
            },
            1: {  # Tuesday
                'open': time(17, 0),  # 5:00 PM Monday
                'close': time(16, 0),  # 4:00 PM Tuesday
                'session': MarketSession.REGULAR
            },
            2: {  # Wednesday
                'open': time(17, 0),  # 5:00 PM Tuesday
                'close': time(16, 0),  # 4:00 PM Wednesday
                'session': MarketSession.REGULAR
            },
            3: {  # Thursday
                'open': time(17, 0),  # 5:00 PM Wednesday
                'close': time(16, 0),  # 4:00 PM Thursday
                'session': MarketSession.REGULAR
            },
            4: {  # Friday
                'open': time(17, 0),  # 5:00 PM Thursday
                'close': time(16, 0),  # 4:00 PM Friday
                'session': MarketSession.REGULAR
            },
            5: {  # Saturday
                'open': None,  # Closed
                'close': None,
                'session': MarketSession.CLOSED
            },
            6: {  # Sunday
                'open': time(17, 0),  # 5:00 PM Sunday
                'close': time(23, 59),  # Continues to Monday
                'session': MarketSession.REGULAR
            }
        }
        
        # US Market holidays (2024) - Futures may have reduced hours
        self.holidays = [
            datetime(2024, 1, 1),   # New Year's Day
            datetime(2024, 1, 15),  # MLK Day
            datetime(2024, 2, 19),  # President's Day
            datetime(2024, 3, 29),  # Good Friday
            datetime(2024, 5, 27),  # Memorial Day
            datetime(2024, 7, 4),   # Independence Day
            datetime(2024, 9, 2),   # Labor Day
            datetime(2024, 11, 28), # Thanksgiving
            datetime(2024, 12, 25), # Christmas
        ]
        
        # Best trading hours (highest liquidity)
        self.best_hours = {
            'start': time(8, 30),   # 8:30 AM CT
            'end': time(15, 0)      # 3:00 PM CT
        }
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if market is currently open"""
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        # Convert to CT
        ct_time = check_time.astimezone(self.ct_tz)
        
        # Check if holiday
        if self.is_holiday(ct_time):
            return False
        
        # Get day of week (0=Monday, 6=Sunday)
        weekday = ct_time.weekday()
        
        # Saturday is always closed
        if weekday == 5:
            return False
        
        # Get current time
        current_time = ct_time.time()
        
        # Special handling for Sunday-Monday transition
        if weekday == 6:  # Sunday
            # Market opens at 5 PM on Sunday
            return current_time >= time(17, 0)
        
        # For weekdays, check if we're in the daily break (4-5 PM)
        if weekday in [0, 1, 2, 3, 4]:  # Monday-Friday
            # Closed between 4 PM and 5 PM
            if time(16, 0) <= current_time < time(17, 0):
                return False
            
            # Friday closes at 4 PM and doesn't reopen
            if weekday == 4 and current_time >= time(16, 0):
                return False
            
            return True
        
        return False
    
    def get_market_session(self, check_time: Optional[datetime] = None) -> MarketSession:
        """Get current market session type"""
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        if not self.is_market_open(check_time):
            return MarketSession.CLOSED
        
        ct_time = check_time.astimezone(self.ct_tz)
        current_time = ct_time.time()
        
        # Best trading hours (regular session equivalent)
        if self.best_hours['start'] <= current_time <= self.best_hours['end']:
            return MarketSession.REGULAR
        
        # Pre-market (before 8:30 AM)
        if current_time < self.best_hours['start']:
            return MarketSession.PRE_MARKET
        
        # After hours (after 3:00 PM but before close)
        if current_time > self.best_hours['end']:
            return MarketSession.AFTER_HOURS
        
        return MarketSession.OVERNIGHT
    
    def is_holiday(self, check_date: datetime) -> bool:
        """Check if given date is a market holiday"""
        check_date_only = check_date.date()
        for holiday in self.holidays:
            if holiday.date() == check_date_only:
                return True
        return False
    
    def get_next_open(self) -> datetime:
        """Get next market open time"""
        now = datetime.now(timezone.utc)
        ct_now = now.astimezone(self.ct_tz)
        
        # If market is open, return current time
        if self.is_market_open(now):
            return ct_now
        
        # If Saturday, next open is Sunday 5 PM
        if ct_now.weekday() == 5:
            days_ahead = 1  # Sunday
            next_open = ct_now.replace(hour=17, minute=0, second=0, microsecond=0)
            next_open = next_open.replace(day=ct_now.day + days_ahead)
            return next_open
        
        # If in daily break (4-5 PM), next open is 5 PM same day
        if time(16, 0) <= ct_now.time() < time(17, 0) and ct_now.weekday() != 4:
            return ct_now.replace(hour=17, minute=0, second=0, microsecond=0)
        
        # If Friday after 4 PM, next open is Sunday 5 PM
        if ct_now.weekday() == 4 and ct_now.time() >= time(16, 0):
            days_ahead = 2  # Sunday
            next_open = ct_now.replace(hour=17, minute=0, second=0, microsecond=0)
            next_open = next_open.replace(day=ct_now.day + days_ahead)
            return next_open
        
        # Otherwise, next open is 5 PM today
        return ct_now.replace(hour=17, minute=0, second=0, microsecond=0)
    
    def get_next_close(self) -> datetime:
        """Get next market close time"""
        now = datetime.now(timezone.utc)
        ct_now = now.astimezone(self.ct_tz)
        
        # If market is closed, find next close after next open
        if not self.is_market_open(now):
            next_open = self.get_next_open()
            # Next close is 4 PM on the next weekday
            if next_open.weekday() == 6:  # Sunday opens, Monday closes
                return next_open.replace(day=next_open.day + 1, hour=16, minute=0)
            else:
                return next_open.replace(hour=16, minute=0)
        
        # If market is open, next close is 4 PM today
        return ct_now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    def get_trading_hours_info(self) -> Dict:
        """Get comprehensive trading hours information"""
        now = datetime.now(timezone.utc)
        ct_now = now.astimezone(self.ct_tz)
        
        return {
            'current_time_ct': ct_now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_open': self.is_market_open(),
            'session': self.get_market_session().value,
            'next_open': self.get_next_open().strftime('%Y-%m-%d %H:%M:%S %Z'),
            'next_close': self.get_next_close().strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_holiday': self.is_holiday(ct_now),
            'best_trading_hours': f"{self.best_hours['start'].strftime('%H:%M')} - {self.best_hours['end'].strftime('%H:%M')} CT"
        }
    
    def should_trade_now(self) -> Tuple[bool, str]:
        """
        Determine if trading should be active now
        Returns (should_trade, reason)
        """
        if not self.is_market_open():
            return False, "Market is closed"
        
        session = self.get_market_session()
        
        if session == MarketSession.CLOSED:
            return False, "Market is closed"
        
        if session == MarketSession.PRE_MARKET:
            return True, "Pre-market session - lower liquidity"
        
        if session == MarketSession.REGULAR:
            return True, "Regular trading hours - optimal liquidity"
        
        if session == MarketSession.AFTER_HOURS:
            return True, "After-hours session - lower liquidity"
        
        if session == MarketSession.OVERNIGHT:
            return True, "Overnight session - minimal liquidity"
        
        return False, "Unknown market condition"

# Global instance
market_hours = MarketHours()

# Convenience functions
def is_market_open() -> bool:
    """Quick check if market is open"""
    return market_hours.is_market_open()

def should_trade() -> bool:
    """Quick check if we should trade"""
    should, _ = market_hours.should_trade_now()
    return should

def get_market_info() -> Dict:
    """Get current market information"""
    return market_hours.get_trading_hours_info()