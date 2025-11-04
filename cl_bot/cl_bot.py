"""
CL (Crude Oil) Trading Bot
Implements CL-specific trading strategies based on discovered patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_bot import BaseBot, OrderType, OrderSide, PositionStatus
from shared.symbol_mapper import SymbolMapper
from shared.data_loader import DatabentoDailyLoader
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load TopStepX credentials
load_dotenv('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend/.env.topstepx')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLBot(BaseBot):
    """CL Trading Bot with discovered patterns"""
    
    def __init__(self, config_path: str = None):
        """Initialize CL bot with patterns"""
        
        # Default configuration for CL
        default_config = {
            'symbol': 'CL',
            'tick_size': 0.01,
            'tick_value': 10.00,
            'max_position_size': 1,
            'risk_per_trade': 50,
            'daily_loss_limit': -500,
            'max_consecutive_losses': 3,
            'min_trade_interval_seconds': 60,
            'patterns': [],
            'topstepx_account_id': int(os.getenv('TOPSTEPX_ACCOUNT_ID', '10983875'))
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        super().__init__(default_config)
        
        # Initialize TopStepX client
        self.topstepx_client = None
        self._init_topstepx()
        
        # Load discovered patterns
        self.load_patterns()
        
        # CL-specific settings
        self.symbol_mapper = SymbolMapper()
        self.current_contract = None
        
        # Technical indicators buffer
        self.price_buffer = []
        self.volume_buffer = []
        self.buffer_size = 100
        
        # Pattern tracking
        self.active_pattern = None
        self.pattern_entry_time = None
        
        # CL-specific market events
        self.inventory_day = 2  # Wednesday
        self.inventory_time = time(10, 30)  # 10:30 AM ET
        self.avoid_inventory_window = 30  # Minutes before/after to avoid
        
        # Asian session tracking (important for crude)
        self.asian_session_start = time(18, 0)  # 6 PM ET
        self.asian_session_end = time(3, 0)    # 3 AM ET
        
        # London open (high volatility period)
        self.london_open = time(3, 0)  # 3 AM ET
        self.london_close = time(11, 30)  # 11:30 AM ET
        
        logger.info(f"CL Bot initialized with {len(self.patterns)} patterns")
        
    def _init_topstepx(self):
        """Initialize TopStepX client connection"""
        try:
            from web_platform.backend.brokers.topstepx_client import TopStepXClient
            self.topstepx_client = TopStepXClient()
            logger.info("TopStepX client initialized for CL bot")
        except Exception as e:
            logger.error(f"Failed to initialize TopStepX client: {e}")
            
    async def connect_to_topstepx(self):
        """Connect to TopStepX"""
        if self.topstepx_client:
            connected = await self.topstepx_client.connect()
            if connected:
                logger.info("CL Bot connected to TopStepX")
                return True
        return False
        
    def load_patterns(self):
        """Load CL-specific patterns"""
        patterns_file = Path(__file__).parent / 'cl_patterns.json'
        
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                self.patterns = json.load(f)
                logger.info(f"Loaded {len(self.patterns)} CL patterns")
        else:
            # Use discovered patterns
            self.patterns = [
                {
                    'name': 'CL_volume_surge_down_hold10',
                    'win_rate': 0.754,
                    'samples': 2985,
                    'hold_minutes': 10,
                    'entry': 'volume > 2x average & direction down',
                    'stop_loss_pct': 0.002,
                    'target_pct': 0.003,
                    'enabled': True
                },
                {
                    'name': 'CL_volume_surge_down_hold5',
                    'win_rate': 0.739,
                    'samples': 2985,
                    'hold_minutes': 5,
                    'entry': 'volume > 2x average & direction down',
                    'stop_loss_pct': 0.002,
                    'target_pct': 0.003,
                    'enabled': True
                }
            ]
            
    async def on_market_data(self, data: Dict):
        """Process incoming CL market data"""
        await self.update_market_data(data)
        
        # Update buffers
        self.update_buffers(data)
        
        # Update contract if needed
        if self.should_update_contract(data.get('timestamp')):
            self.update_contract(data.get('timestamp'))
        
        # Check if we should avoid trading (inventory report)
        if self.should_avoid_trading():
            return
        
        # Check for pattern signals only when flat
        if self.position == 0 and self.can_trade():
            pattern = await self.check_patterns()
            if pattern:
                await self.execute_pattern_trade(pattern)
        
        # Manage existing position
        elif self.position != 0:
            await self.manage_cl_position()
            
    def update_buffers(self, data: Dict):
        """Update price and volume buffers for technical indicators"""
        self.price_buffer.append(data.get('close', self.current_price))
        self.volume_buffer.append(data.get('volume', 0))
        
        # Keep buffer size limited
        if len(self.price_buffer) > self.buffer_size:
            self.price_buffer.pop(0)
        if len(self.volume_buffer) > self.buffer_size:
            self.volume_buffer.pop(0)
            
    def should_update_contract(self, timestamp):
        """Check if we need to roll to new CL contract"""
        if not timestamp:
            return False
            
        current_date = pd.Timestamp(timestamp)
        
        # Update contract if not set
        if not self.current_contract:
            return True
            
        # CL has monthly contracts, check if should roll (10 days before expiry)
        if self.symbol_mapper.should_roll(self.current_contract, current_date, days_before=10):
            return True
            
        return False
        
    def update_contract(self, timestamp):
        """Update to current CL contract"""
        current_date = pd.Timestamp(timestamp)
        self.current_contract = self.symbol_mapper.get_front_month('CL', current_date)
        logger.info(f"Updated CL contract to: {self.current_contract}")
        
    def should_avoid_trading(self) -> bool:
        """Check if we should avoid trading due to market events"""
        if not self.current_time:
            return False
            
        # Check for inventory report window (Wednesdays)
        if self.current_time.weekday() == self.inventory_day:
            current_time = self.current_time.time()
            inventory_start = (datetime.combine(datetime.today(), self.inventory_time) - 
                             timedelta(minutes=self.avoid_inventory_window)).time()
            inventory_end = (datetime.combine(datetime.today(), self.inventory_time) + 
                           timedelta(minutes=self.avoid_inventory_window)).time()
            
            if inventory_start <= current_time <= inventory_end:
                logger.debug("Avoiding trading during inventory report window")
                return True
                
        return False
        
    def is_high_volatility_period(self) -> bool:
        """Check if we're in a high volatility period for CL"""
        if not self.current_time:
            return False
            
        current_time = self.current_time.time()
        
        # London open is high volatility
        if self.london_open <= current_time <= time(5, 0):
            return True
            
        # US open overlap with London
        if time(9, 0) <= current_time <= time(11, 0):
            return True
            
        return False
        
    async def check_patterns(self) -> Optional[Dict]:
        """Check for CL pattern signals"""
        
        for pattern in self.patterns:
            if not pattern.get('enabled', True):
                continue
                
            # Check volume surge patterns (main CL patterns)
            if 'volume_surge' in pattern['name']:
                if self.check_volume_surge_pattern(pattern):
                    logger.info(f"CL volume surge pattern triggered: {pattern['name']}")
                    return pattern
                    
        return None
        
    def check_volume_surge_pattern(self, pattern: Dict) -> bool:
        """Check for volume surge pattern specific to CL"""
        if len(self.volume_buffer) < 50:
            return False
            
        # Calculate average volume
        avg_volume = np.mean(self.volume_buffer[-50:-1])
        current_volume = self.volume_buffer[-1]
        
        # Check for volume surge
        if current_volume > avg_volume * 2:
            # CL patterns are specifically for down moves
            if 'down' in pattern['name']:
                if len(self.price_buffer) >= 2:
                    # Check for downward price movement
                    price_down = self.price_buffer[-1] < self.price_buffer[-2]
                    
                    # Additional filter: Check if we're not in extreme volatility
                    if not self.is_high_volatility_period() or pattern.get('trade_volatile', False):
                        return price_down
                        
        return False
        
    def calculate_position_size(self, stop_distance: float) -> int:
        """Calculate CL position size based on risk"""
        # CL specific: Each tick (0.01) = $10
        risk_per_contract = stop_distance * self.tick_value / self.tick_size
        
        if risk_per_contract <= 0:
            return 0
            
        # Calculate contracts based on risk
        contracts = int(self.risk_per_trade / risk_per_contract)
        
        # Limit to max position size
        return min(contracts, self.max_position_size)
        
    async def execute_pattern_trade(self, pattern: Dict):
        """Execute trade based on CL pattern signal"""
        logger.info(f"Executing CL trade for pattern: {pattern['name']}")
        
        # CL patterns are bearish (volume surge down)
        side = OrderSide.SELL
        
        # Calculate position size
        stop_distance = pattern['stop_loss_pct'] * self.current_price / self.tick_size
        position_size = self.calculate_position_size(stop_distance)
        
        if position_size == 0:
            logger.warning("Position size is 0, skipping trade")
            return
            
        # Calculate stop loss and take profit
        stop_loss = self.current_price + (pattern['stop_loss_pct'] * self.current_price)
        take_profit = self.current_price - (pattern['target_pct'] * self.current_price)
        
        # Adjust for CL volatility
        if self.is_high_volatility_period():
            # Wider stops during volatile periods
            stop_loss = self.current_price + (pattern['stop_loss_pct'] * self.current_price * 1.5)
            take_profit = self.current_price - (pattern['target_pct'] * self.current_price * 1.5)
            
        # Place order
        order = await self.place_order(
            side=side,
            size=position_size,
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order:
            self.active_pattern = pattern
            self.pattern_entry_time = datetime.now()
            logger.info(f"CL order placed: {order}")
            
    async def manage_cl_position(self):
        """Manage existing CL position"""
        if not self.active_pattern or not self.pattern_entry_time:
            return
            
        # Check hold time
        hold_time = (datetime.now() - self.pattern_entry_time).total_seconds() / 60
        max_hold = self.active_pattern.get('hold_minutes', 60)
        
        # CL specific: Exit earlier if approaching inventory report
        if self.should_avoid_trading() and hold_time > 2:
            logger.info("Closing CL position before inventory report")
            await self.close_position("inventory_report_approaching")
            self.active_pattern = None
            self.pattern_entry_time = None
            return
            
        if hold_time >= max_hold:
            logger.info(f"CL position hold time reached ({hold_time:.1f} min), closing position")
            await self.close_position("hold_time_reached")
            self.active_pattern = None
            self.pattern_entry_time = None
            
    def get_cl_status(self) -> Dict:
        """Get CL-specific status"""
        status = self.get_status()
        
        # Add CL-specific info
        status.update({
            'current_contract': self.current_contract,
            'active_pattern': self.active_pattern['name'] if self.active_pattern else None,
            'buffer_size': len(self.price_buffer),
            'is_inventory_day': self.current_time.weekday() == self.inventory_day if self.current_time else False,
            'is_high_volatility': self.is_high_volatility_period(),
            'patterns_enabled': sum(1 for p in self.patterns if p.get('enabled', True))
        })
        
        return status
        
    def get_session_info(self) -> str:
        """Get current trading session info"""
        if not self.current_time:
            return "Unknown"
            
        current_time = self.current_time.time()
        
        if self.asian_session_start <= current_time or current_time <= self.asian_session_end:
            return "Asian Session"
        elif self.london_open <= current_time <= self.london_close:
            return "London Session"
        elif time(9, 0) <= current_time <= time(16, 0):
            return "US Session"
        else:
            return "Transition Period"


if __name__ == "__main__":
    # Test CL bot initialization
    bot = CLBot()
    print(f"CL Bot initialized with {len(bot.patterns)} patterns")
    print(f"Patterns: {[p['name'] for p in bot.patterns]}")
    print(f"Status: {bot.get_cl_status()}")