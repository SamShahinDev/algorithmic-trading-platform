"""
ES Trading Bot
Implements ES-specific trading strategies based on discovered patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_bot import BaseBot, OrderType, OrderSide, PositionStatus
from shared.symbol_mapper import SymbolMapper
from shared.data_loader import DatabentoDailyLoader
from datetime import datetime, timedelta
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

class ESBot(BaseBot):
    """ES Trading Bot with discovered patterns"""
    
    def __init__(self, config_path: str = None):
        """Initialize ES bot with patterns"""
        
        # Default configuration for ES
        default_config = {
            'symbol': 'ES',
            'tick_size': 0.25,
            'tick_value': 12.50,
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
        
        # ES-specific settings
        self.symbol_mapper = SymbolMapper()
        self.current_contract = None
        
        # Technical indicators buffer
        self.price_buffer = []
        self.volume_buffer = []
        self.buffer_size = 100
        
        # Pattern tracking
        self.active_pattern = None
        self.pattern_entry_time = None
        
        # Market hours (ES trades almost 24 hours)
        self.market_open_time = (18, 0)  # 6 PM ET Sunday
        self.market_close_time = (17, 0)  # 5 PM ET Friday
        
        # RSI and Bollinger Bands for reversal patterns
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        
        logger.info(f"ES Bot initialized with {len(self.patterns)} patterns")
        
    def _init_topstepx(self):
        """Initialize TopStepX client connection"""
        try:
            from web_platform.backend.brokers.topstepx_client import TopStepXClient
            self.topstepx_client = TopStepXClient()
            logger.info("TopStepX client initialized for ES bot")
        except Exception as e:
            logger.error(f"Failed to initialize TopStepX client: {e}")
            
    async def connect_to_topstepx(self):
        """Connect to TopStepX"""
        if self.topstepx_client:
            connected = await self.topstepx_client.connect()
            if connected:
                logger.info("ES Bot connected to TopStepX")
                return True
        return False
        
    def load_patterns(self):
        """Load ES-specific patterns"""
        patterns_file = Path(__file__).parent / 'es_patterns.json'
        
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                self.patterns = json.load(f)
                logger.info(f"Loaded {len(self.patterns)} ES patterns")
        else:
            # Use discovered patterns
            self.patterns = [
                {
                    'name': 'ES_volume_surge_up_hold5',
                    'win_rate': 0.724,
                    'samples': 1198,
                    'hold_minutes': 5,
                    'entry': 'volume > 2x average',
                    'stop_loss_pct': 0.002,
                    'target_pct': 0.003,
                    'enabled': True
                },
                {
                    'name': 'ES_volume_surge_up_hold10',
                    'win_rate': 0.709,
                    'samples': 1198,
                    'hold_minutes': 10,
                    'entry': 'volume > 2x average',
                    'stop_loss_pct': 0.002,
                    'target_pct': 0.003,
                    'enabled': True
                },
                {
                    'name': 'ES_oversold_bounce_hold30',
                    'win_rate': 0.647,
                    'samples': 34,
                    'hold_minutes': 30,
                    'entry': 'RSI < 30 & price < lower_bb',
                    'stop_loss_pct': 0.003,
                    'target_pct': 0.005,
                    'enabled': True
                },
                {
                    'name': 'ES_power_hour_bearish',
                    'win_rate': 0.520,
                    'samples': 1890,
                    'hold_minutes': 15,
                    'entry': 'time >= 14:30',
                    'stop_loss_pct': 0.002,
                    'target_pct': 0.003,
                    'enabled': False  # Disabled by default due to lower win rate
                }
            ]
            
    async def on_market_data(self, data: Dict):
        """Process incoming ES market data"""
        await self.update_market_data(data)
        
        # Update buffers
        self.update_buffers(data)
        
        # Update contract if needed
        if self.should_update_contract(data.get('timestamp')):
            self.update_contract(data.get('timestamp'))
        
        # Check for pattern signals only when flat
        if self.position == 0 and self.can_trade():
            pattern = await self.check_patterns()
            if pattern:
                await self.execute_pattern_trade(pattern)
        
        # Manage existing position
        elif self.position != 0:
            await self.manage_es_position()
            
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
        """Check if we need to roll to new contract"""
        if not timestamp:
            return False
            
        current_date = pd.Timestamp(timestamp)
        
        # Update contract monthly or if not set
        if not self.current_contract:
            return True
            
        # Check if we should roll (5 days before expiry)
        if self.symbol_mapper.should_roll(self.current_contract, current_date, days_before=5):
            return True
            
        return False
        
    def update_contract(self, timestamp):
        """Update to current ES contract"""
        current_date = pd.Timestamp(timestamp)
        self.current_contract = self.symbol_mapper.get_front_month('ES', current_date)
        logger.info(f"Updated ES contract to: {self.current_contract}")
        
    async def check_patterns(self) -> Optional[Dict]:
        """Check for ES pattern signals"""
        
        for pattern in self.patterns:
            if not pattern.get('enabled', True):
                continue
                
            # Check volume surge patterns
            if 'volume_surge' in pattern['name']:
                if self.check_volume_surge_pattern(pattern):
                    logger.info(f"Volume surge pattern triggered: {pattern['name']}")
                    return pattern
                    
            # Check oversold bounce pattern
            elif 'oversold_bounce' in pattern['name']:
                if self.check_oversold_pattern(pattern):
                    logger.info(f"Oversold bounce pattern triggered: {pattern['name']}")
                    return pattern
                    
            # Check power hour pattern
            elif 'power_hour' in pattern['name']:
                if self.check_power_hour_pattern(pattern):
                    logger.info(f"Power hour pattern triggered: {pattern['name']}")
                    return pattern
                    
        return None
        
    def check_volume_surge_pattern(self, pattern: Dict) -> bool:
        """Check for volume surge pattern"""
        if len(self.volume_buffer) < 50:
            return False
            
        # Calculate average volume
        avg_volume = np.mean(self.volume_buffer[-50:-1])
        current_volume = self.volume_buffer[-1]
        
        # Check for volume surge
        if current_volume > avg_volume * 2:
            # Check direction for "up" patterns
            if 'up' in pattern['name']:
                if len(self.price_buffer) >= 2:
                    return self.price_buffer[-1] > self.price_buffer[-2]
            return True
            
        return False
        
    def check_oversold_pattern(self, pattern: Dict) -> bool:
        """Check for oversold bounce pattern"""
        if len(self.price_buffer) < max(self.rsi_period, self.bb_period):
            return False
            
        # Calculate RSI
        rsi = self.calculate_rsi()
        
        # Calculate Bollinger Bands
        prices = pd.Series(self.price_buffer)
        bb_middle = prices.rolling(self.bb_period).mean().iloc[-1]
        bb_std = prices.rolling(self.bb_period).std().iloc[-1]
        bb_lower = bb_middle - (bb_std * self.bb_std)
        
        current_price = self.price_buffer[-1]
        
        # Check if oversold
        if rsi < 30 and current_price < bb_lower:
            # Check for increased volume
            if len(self.volume_buffer) >= 20:
                avg_volume = np.mean(self.volume_buffer[-20:])
                if self.volume_buffer[-1] > avg_volume * 1.2:
                    return True
                    
        return False
        
    def check_power_hour_pattern(self, pattern: Dict) -> bool:
        """Check for power hour pattern"""
        if not self.current_time:
            return False
            
        hour = self.current_time.hour
        minute = self.current_time.minute
        
        # Power hour is 2:30 PM to 4:00 PM ET (14:30 to 16:00)
        if hour == 14 and minute >= 30:
            return True
        elif hour == 15:
            return True
        elif hour == 16 and minute == 0:
            return True
            
        return False
        
    def calculate_rsi(self) -> float:
        """Calculate RSI from price buffer"""
        if len(self.price_buffer) < self.rsi_period + 1:
            return 50  # Neutral RSI
            
        prices = pd.Series(self.price_buffer)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
        
    def calculate_position_size(self, stop_distance: float) -> int:
        """Calculate ES position size based on risk"""
        # ES specific: Each point = $50, each tick (0.25) = $12.50
        risk_per_contract = stop_distance * self.tick_value / self.tick_size
        
        if risk_per_contract <= 0:
            return 0
            
        # Calculate contracts based on risk
        contracts = int(self.risk_per_trade / risk_per_contract)
        
        # Limit to max position size
        return min(contracts, self.max_position_size)
        
    async def execute_pattern_trade(self, pattern: Dict):
        """Execute trade based on pattern signal"""
        logger.info(f"Executing ES trade for pattern: {pattern['name']}")
        
        # Determine trade direction
        if 'bearish' in pattern['name'] or 'down' in pattern['name']:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY
            
        # Calculate position size
        stop_distance = pattern['stop_loss_pct'] * self.current_price / self.tick_size
        position_size = self.calculate_position_size(stop_distance)
        
        if position_size == 0:
            logger.warning("Position size is 0, skipping trade")
            return
            
        # Calculate stop loss and take profit
        if side == OrderSide.BUY:
            stop_loss = self.current_price - (pattern['stop_loss_pct'] * self.current_price)
            take_profit = self.current_price + (pattern['target_pct'] * self.current_price)
        else:
            stop_loss = self.current_price + (pattern['stop_loss_pct'] * self.current_price)
            take_profit = self.current_price - (pattern['target_pct'] * self.current_price)
            
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
            logger.info(f"ES order placed: {order}")
            
    async def manage_es_position(self):
        """Manage existing ES position"""
        if not self.active_pattern or not self.pattern_entry_time:
            return
            
        # Check hold time
        hold_time = (datetime.now() - self.pattern_entry_time).total_seconds() / 60
        max_hold = self.active_pattern.get('hold_minutes', 60)
        
        if hold_time >= max_hold:
            logger.info(f"ES position hold time reached ({hold_time:.1f} min), closing position")
            await self.close_position("hold_time_reached")
            self.active_pattern = None
            self.pattern_entry_time = None
            
    def get_es_status(self) -> Dict:
        """Get ES-specific status"""
        status = self.get_status()
        
        # Add ES-specific info
        status.update({
            'current_contract': self.current_contract,
            'active_pattern': self.active_pattern['name'] if self.active_pattern else None,
            'buffer_size': len(self.price_buffer),
            'current_rsi': self.calculate_rsi() if len(self.price_buffer) > self.rsi_period else None,
            'patterns_enabled': sum(1 for p in self.patterns if p.get('enabled', True))
        })
        
        return status


if __name__ == "__main__":
    # Test ES bot initialization
    bot = ESBot()
    print(f"ES Bot initialized with {len(bot.patterns)} patterns")
    print(f"Patterns: {[p['name'] for p in bot.patterns]}")
    print(f"Status: {bot.get_es_status()}")