"""
Symbol Mapper
Maps between different symbol formats and handles futures contract rollovers
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SymbolMapper:
    """Maps between different symbol formats and handles contract rollovers"""
    
    def __init__(self, symbology_path: Optional[str] = None):
        """
        Initialize symbol mapper
        
        Args:
            symbology_path: Optional path to symbology CSV file
        """
        self.symbology = None
        if symbology_path:
            try:
                self.symbology = pd.read_csv(symbology_path)
                logger.info(f"Loaded symbology from {symbology_path}")
            except Exception as e:
                logger.warning(f"Could not load symbology: {e}")
                
        # Futures month codes
        self.futures_months = {
            'F': 1,  # January
            'G': 2,  # February  
            'H': 3,  # March
            'J': 4,  # April
            'K': 5,  # May
            'M': 6,  # June
            'N': 7,  # July
            'Q': 8,  # August
            'U': 9,  # September
            'V': 10, # October
            'X': 11, # November
            'Z': 12  # December
        }
        
        # Reverse mapping
        self.month_codes = {v: k for k, v in self.futures_months.items()}
        
        # Contract specifications
        self.contract_specs = {
            'ES': {
                'name': 'E-mini S&P 500',
                'tick_size': 0.25,
                'tick_value': 12.50,
                'point_value': 50,
                'roll_months': [3, 6, 9, 12],  # Quarterly
                'roll_day': 8,  # Second Friday (approximate)
                'exchange': 'CME'
            },
            'NQ': {
                'name': 'E-mini Nasdaq 100',
                'tick_size': 0.25,
                'tick_value': 5.00,
                'point_value': 20,
                'roll_months': [3, 6, 9, 12],  # Quarterly
                'roll_day': 8,
                'exchange': 'CME'
            },
            'CL': {
                'name': 'Crude Oil',
                'tick_size': 0.01,
                'tick_value': 10.00,
                'point_value': 1000,
                'roll_months': list(range(1, 13)),  # Monthly
                'roll_day': 20,  # Around 20th of month before expiry
                'exchange': 'NYMEX'
            },
            'MES': {
                'name': 'Micro E-mini S&P 500',
                'tick_size': 0.25,
                'tick_value': 1.25,
                'point_value': 5,
                'roll_months': [3, 6, 9, 12],  # Quarterly
                'roll_day': 8,
                'exchange': 'CME'
            },
            'MNQ': {
                'name': 'Micro E-mini Nasdaq 100',
                'tick_size': 0.25,
                'tick_value': 0.50,
                'point_value': 2,
                'roll_months': [3, 6, 9, 12],  # Quarterly
                'roll_day': 8,
                'exchange': 'CME'
            },
            'MCL': {
                'name': 'Micro Crude Oil',
                'tick_size': 0.01,
                'tick_value': 1.00,
                'point_value': 100,
                'roll_months': list(range(1, 13)),  # Monthly
                'roll_day': 20,
                'exchange': 'NYMEX'
            }
        }
        
    def get_front_month(self, base_symbol: str, date: datetime) -> str:
        """
        Get the front month contract for a given date
        
        Args:
            base_symbol: Base symbol (e.g., 'ES', 'CL', 'NQ')
            date: Date to get front month for
            
        Returns:
            Front month contract symbol (e.g., 'ESH25', 'CLG25')
        """
        if base_symbol not in self.contract_specs:
            logger.warning(f"Unknown symbol: {base_symbol}")
            return base_symbol
            
        specs = self.contract_specs[base_symbol]
        roll_months = specs['roll_months']
        roll_day = specs['roll_day']
        
        year = date.year
        month = date.month
        day = date.day
        
        # Find next contract month
        next_contract_month = None
        for roll_month in roll_months:
            if month < roll_month:
                next_contract_month = roll_month
                break
            elif month == roll_month:
                # Check if we've rolled yet
                if day < roll_day:
                    next_contract_month = roll_month
                    break
                    
        # If no contract found, use first contract of next year
        if next_contract_month is None:
            next_contract_month = roll_months[0]
            year += 1
            
        # Format symbol
        month_code = self.month_codes[next_contract_month]
        year_code = str(year)[-2:]  # Last 2 digits
        
        return f"{base_symbol}{month_code}{year_code}"
        
    def get_contract_chain(self, base_symbol: str, date: datetime, num_contracts: int = 3) -> List[str]:
        """
        Get chain of futures contracts (front month and subsequent)
        
        Args:
            base_symbol: Base symbol
            date: Reference date
            num_contracts: Number of contracts to return
            
        Returns:
            List of contract symbols
        """
        if base_symbol not in self.contract_specs:
            return [base_symbol]
            
        specs = self.contract_specs[base_symbol]
        roll_months = specs['roll_months']
        
        contracts = []
        current_date = date
        
        for _ in range(num_contracts):
            contract = self.get_front_month(base_symbol, current_date)
            
            if contract not in contracts:
                contracts.append(contract)
                
            # Move to next month
            current_month_idx = roll_months.index(current_date.month) if current_date.month in roll_months else 0
            
            if current_month_idx < len(roll_months) - 1:
                next_month = roll_months[current_month_idx + 1]
                current_date = current_date.replace(month=next_month)
            else:
                # Move to next year
                current_date = current_date.replace(year=current_date.year + 1, month=roll_months[0])
                
        return contracts
        
    def databento_to_topstep(self, databento_symbol: str, date: datetime) -> str:
        """
        Convert Databento symbol format to TopStep/Tradovate format
        
        Args:
            databento_symbol: Symbol from Databento data
            date: Date for contract mapping
            
        Returns:
            TopStep/Tradovate formatted symbol
        """
        # Extract base symbol from Databento format
        # Databento format examples: ESH5, CLG5, NQM5
        
        # Common mappings
        symbol_mappings = {
            'ES': 'ES',    # E-mini S&P 500
            'NQ': 'NQ',    # E-mini Nasdaq
            'CL': 'CL',    # Crude Oil
            'GC': 'GC',    # Gold
            'SI': 'SI',    # Silver
            'ZB': 'ZB',    # 30-Year Treasury Bond
            'ZN': 'ZN',    # 10-Year Treasury Note
            'ZC': 'ZC',    # Corn
            'ZS': 'ZS',    # Soybeans
            'ZW': 'ZW',    # Wheat
            'NG': 'NG',    # Natural Gas
            '6E': '6E',    # Euro FX
            '6B': '6B',    # British Pound
            '6J': '6J',    # Japanese Yen
        }
        
        # Check for micro contracts
        if databento_symbol.startswith('MES'):
            return 'M' + self.get_front_month('ES', date)
        elif databento_symbol.startswith('MNQ'):
            return 'M' + self.get_front_month('NQ', date)
        elif databento_symbol.startswith('MCL'):
            return 'M' + self.get_front_month('CL', date)
            
        # Extract base symbol (first 2-3 characters that are letters)
        base = ''
        for char in databento_symbol:
            if char.isalpha():
                base += char
            else:
                break
                
        if base in symbol_mappings:
            return self.get_front_month(symbol_mappings[base], date)
            
        # Return original if no mapping found
        return databento_symbol
        
    def parse_futures_symbol(self, symbol: str) -> Tuple[str, str, int]:
        """
        Parse a futures symbol into components
        
        Args:
            symbol: Futures symbol (e.g., 'ESH25', 'CLG25')
            
        Returns:
            Tuple of (base_symbol, month_code, year)
        """
        if len(symbol) < 4:
            return symbol, '', 0
            
        # Find where letters end and month/year begin
        base_end = 0
        for i, char in enumerate(symbol):
            if char in self.futures_months:
                base_end = i
                break
                
        if base_end == 0:
            return symbol, '', 0
            
        base = symbol[:base_end]
        month_code = symbol[base_end]
        year_str = symbol[base_end + 1:]
        
        # Parse year (could be 2 or 4 digits)
        try:
            year = int(year_str)
            if year < 100:
                # Convert 2-digit year to 4-digit
                current_century = datetime.now().year // 100 * 100
                year = current_century + year
                
                # Handle year wrap (e.g., 98 could be 2098 or 1998)
                if year > datetime.now().year + 10:
                    year -= 100
                    
        except ValueError:
            year = 0
            
        return base, month_code, year
        
    def get_expiry_date(self, symbol: str) -> Optional[datetime]:
        """
        Get expiry date for a futures contract
        
        Args:
            symbol: Futures symbol
            
        Returns:
            Expiry date or None if cannot determine
        """
        base, month_code, year = self.parse_futures_symbol(symbol)
        
        if not month_code or year == 0:
            return None
            
        if base not in self.contract_specs:
            return None
            
        month = self.futures_months.get(month_code)
        if not month:
            return None
            
        specs = self.contract_specs[base]
        
        # Calculate expiry (simplified - actual dates vary)
        # Most futures expire on 3rd Friday of expiry month
        expiry = datetime(year, month, 1)
        
        # Find third Friday
        first_day = expiry.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        
        return third_friday
        
    def should_roll(self, symbol: str, date: datetime, days_before: int = 5) -> bool:
        """
        Check if a contract should be rolled
        
        Args:
            symbol: Current contract symbol
            date: Current date
            days_before: Days before expiry to roll
            
        Returns:
            True if should roll to next contract
        """
        expiry = self.get_expiry_date(symbol)
        
        if not expiry:
            return False
            
        days_to_expiry = (expiry - date).days
        
        return days_to_expiry <= days_before
        
    def get_next_contract(self, symbol: str) -> Optional[str]:
        """
        Get the next contract in the chain
        
        Args:
            symbol: Current contract symbol
            
        Returns:
            Next contract symbol or None
        """
        base, month_code, year = self.parse_futures_symbol(symbol)
        
        if not base or not month_code:
            return None
            
        if base not in self.contract_specs:
            return None
            
        specs = self.contract_specs[base]
        roll_months = specs['roll_months']
        
        current_month = self.futures_months.get(month_code)
        if not current_month:
            return None
            
        # Find next month in roll schedule
        try:
            current_idx = roll_months.index(current_month)
            
            if current_idx < len(roll_months) - 1:
                # Next month in same year
                next_month = roll_months[current_idx + 1]
                next_month_code = self.month_codes[next_month]
                return f"{base}{next_month_code}{str(year)[-2:]}"
            else:
                # First month of next year
                next_month = roll_months[0]
                next_month_code = self.month_codes[next_month]
                next_year = year + 1
                return f"{base}{next_month_code}{str(next_year)[-2:]}"
                
        except ValueError:
            return None
            
    def get_contract_specs(self, base_symbol: str) -> Optional[Dict]:
        """
        Get contract specifications
        
        Args:
            base_symbol: Base symbol (e.g., 'ES', 'CL')
            
        Returns:
            Contract specifications dictionary or None
        """
        return self.contract_specs.get(base_symbol)
        
    def is_trading_hours(self, symbol: str, timestamp: datetime) -> bool:
        """
        Check if market is open for a given symbol
        
        Args:
            symbol: Trading symbol
            timestamp: Time to check
            
        Returns:
            True if market is open
        """
        # Extract base symbol
        base = symbol[:2] if len(symbol) >= 2 else symbol
        
        # Simplified trading hours (actual hours are more complex)
        trading_hours = {
            'ES': [(18, 0, 17, 0)],  # 6 PM - 5 PM ET (almost 24 hours)
            'NQ': [(18, 0, 17, 0)],
            'CL': [(18, 0, 17, 0)],
            'GC': [(18, 0, 17, 0)],
        }
        
        hours = trading_hours.get(base, [(9, 30, 16, 0)])  # Default to regular market hours
        
        current_time = timestamp.time()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        for start_hour, start_min, end_hour, end_min in hours:
            # Handle overnight sessions
            if start_hour > end_hour:
                # Session crosses midnight
                if current_hour >= start_hour or current_hour < end_hour:
                    return True
                elif current_hour == end_hour and current_minute <= end_min:
                    return True
            else:
                # Normal session
                if start_hour <= current_hour <= end_hour:
                    if current_hour == start_hour and current_minute < start_min:
                        return False
                    if current_hour == end_hour and current_minute > end_min:
                        return False
                    return True
                    
        return False


if __name__ == "__main__":
    # Test symbol mapper
    mapper = SymbolMapper()
    
    # Test dates
    test_date = datetime(2025, 1, 15)
    
    print("Testing Symbol Mapper")
    print("=" * 50)
    
    # Test front month contracts
    for symbol in ['ES', 'NQ', 'CL']:
        front_month = mapper.get_front_month(symbol, test_date)
        print(f"{symbol} front month on {test_date.date()}: {front_month}")
        
        # Get contract chain
        chain = mapper.get_contract_chain(symbol, test_date, 3)
        print(f"{symbol} contract chain: {chain}")
        
        # Get expiry
        expiry = mapper.get_expiry_date(front_month)
        print(f"{front_month} expires on: {expiry.date() if expiry else 'Unknown'}")
        
        # Check if should roll
        should_roll = mapper.should_roll(front_month, test_date, days_before=10)
        print(f"Should roll {front_month}? {should_roll}")
        
        print()
        
    # Test parsing
    print("\nTesting symbol parsing:")
    test_symbols = ['ESH25', 'CLG25', 'NQM25', 'MESH25']
    
    for symbol in test_symbols:
        base, month, year = mapper.parse_futures_symbol(symbol)
        print(f"{symbol} -> Base: {base}, Month: {month}, Year: {year}")