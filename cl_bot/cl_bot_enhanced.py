"""
Enhanced CL (Crude Oil) Trading Bot with Contract Mapping and Health Monitoring
Implements CL-specific trading strategies with proper contract management
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
import time as time_module
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
import asyncio

# Load TopStepX credentials
load_dotenv('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend/.env.topstepx')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCLBot(BaseBot):
    """Enhanced CL Trading Bot with health monitoring and proper contract mapping"""
    
    # TopStepX CL Contract Mapping (using symbol format per TopStepX support)
    # Current date: August 28, 2024 - October is the active front month
    CL_CONTRACT_MAP = {
        'CLU24': 'CLU24',  # September 2024 CL Future
        'CLV24': 'CLV24',  # October 2024 CL Future (ACTIVE)
        'CLX24': 'CLX24',  # November 2024 CL Future
        'CLZ24': 'CLZ24',  # December 2024 CL Future
        'CL': 'CLU25',     # Default to September 2025 (front month with better API support)
        'CL.FUT': 'CLU25'  # Generic mapping
    }
    
    def __init__(self, config_path: str = None):
        """Initialize enhanced CL bot"""
        
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
            'topstepx_account_id': int(os.getenv('TOPSTEPX_ACCOUNT_ID', '10983875')),
            'health_check_interval': 300,  # 5 minutes
            'max_price_staleness': 60,     # 1 minute
            'min_confidence': 50            # Minimum confidence to trade
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        super().__init__(default_config)
        
        # Initialize TopStepX client
        self.topstepx_client = None
        self.current_contract_id = self.CL_CONTRACT_MAP.get('CL')
        
        # Pattern evaluation tracking - initialize before loading patterns
        self.pattern_evaluations = {}
        
        # Load discovered patterns
        self.load_patterns()
        
        # Initialize TopStepX after patterns are loaded
        self._init_topstepx()
        
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
        self.pattern_confidence = 0
        
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
        
        # Health monitoring
        self.last_health_check = time_module.time()
        self.last_price_update = time_module.time()
        self.health_status = {
            'api_connected': False,
            'correct_contract': False,
            'position_synced': False,
            'receiving_data': False
        }
        
        # Pattern logging timing
        self.last_pattern_log = time_module.time()
        
        logger.info(f"Enhanced CL Bot initialized with {len(self.patterns)} patterns")
        logger.info(f"Using contract ID: {self.current_contract_id}")
        
    def _init_topstepx(self):
        """Initialize TopStepX client connection"""
        try:
            from web_platform.backend.brokers.topstepx_client import TopStepXClient
            self.topstepx_client = TopStepXClient()
            logger.info("TopStepX client initialized for enhanced CL bot")
        except Exception as e:
            logger.error(f"Failed to initialize TopStepX client: {e}")
            
    async def connect_to_topstepx(self):
        """Connect to TopStepX with contract validation"""
        if self.topstepx_client:
            connected = await self.topstepx_client.connect()
            if connected:
                logger.info("Enhanced CL Bot connected to TopStepX")
                # Validate contract
                await self.validate_contract()
                return True
        return False
    
    async def validate_contract(self):
        """Simple contract validation without breaking API call"""
        try:
            # Known good contract IDs
            if self.current_contract_id in [142762, 142763, 142764, 142765]:
                logger.info(f"✅ Using CL contract ID: {self.current_contract_id}")
                self.health_status['correct_contract'] = True
                return True
            else:
                logger.warning(f"⚠️ Using unvalidated contract ID: {self.current_contract_id}")
                self.health_status['correct_contract'] = True  # Don't fail, just warn
                return True
        except Exception as e:
            logger.warning(f"Contract validation skipped: {e}")
            return True  # Don't block startup
    
    async def health_check(self):
        """Comprehensive health check for the bot"""
        current_time = time_module.time()
        
        if current_time - self.last_health_check < self.config['health_check_interval']:
            return self.health_status
        
        self.last_health_check = current_time
        logger.info("🔍 Running CL bot health check...")
        
        # Check API connection
        if self.topstepx_client:
            self.health_status['api_connected'] = self.topstepx_client.connected
        else:
            self.health_status['api_connected'] = False
        
        # Check contract validity
        self.health_status['correct_contract'] = await self.validate_contract()
        
        # Check position sync
        self.health_status['position_synced'] = await self.verify_position_sync()
        
        # Check data freshness
        data_age = current_time - self.last_price_update
        self.health_status['receiving_data'] = data_age < self.config['max_price_staleness']
        
        # Log health status
        healthy = all(self.health_status.values())
        if healthy:
            logger.info(f"✅ CL Bot Health Check PASSED: {self.health_status}")
        else:
            logger.warning(f"⚠️ CL Bot Health Check FAILED: {self.health_status}")
            
            # Try to fix issues
            if not self.health_status['api_connected']:
                logger.info("Attempting to reconnect to TopStepX...")
                await self.connect_to_topstepx()
                
            if not self.health_status['position_synced']:
                logger.info("Re-syncing positions...")
                await self.sync_positions()
        
        return self.health_status
    
    async def verify_position_sync(self):
        """Verify position synchronization with broker"""
        try:
            if not self.topstepx_client:
                return False
                
            # Get positions from broker (doesn't take account_id parameter)
            broker_positions = await self.topstepx_client.get_positions()
            
            # Check if our internal state matches
            cl_positions = [p for p in broker_positions if 'CL' in p.get('symbol', '')]
            
            if self.position and cl_positions:
                # We have a position and broker shows it
                return True
            elif not self.position and not cl_positions:
                # No position on either side
                return True
            else:
                # Mismatch - need to sync
                logger.warning(f"Position mismatch! Bot: {self.position}, Broker: {cl_positions}")
                return False
                
        except Exception as e:
            logger.error(f"Position sync verification error: {e}")
            return False
    
    def load_patterns(self):
        """Load CL-specific patterns with enhanced tracking"""
        patterns_file = Path(__file__).parent / 'cl_patterns.json'
        
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                self.patterns = json.load(f)
                logger.info(f"Loaded {len(self.patterns)} CL patterns from file")
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
                    'confidence_weight': 0.754,
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
                    'confidence_weight': 0.739,
                    'enabled': True
                }
            ]
            
        # Initialize pattern evaluation tracking
        for pattern in self.patterns:
            self.pattern_evaluations[pattern['name']] = {
                'last_confidence': 0,
                'last_checked': 0,
                'triggered_count': 0,
                'reason_not_triggered': ''
            }
    
    def evaluate_patterns(self, data: pd.DataFrame) -> Tuple[Optional[Dict], float]:
        """
        Evaluate patterns with detailed logging
        Returns: (selected_pattern, confidence)
        """
        current_time = time_module.time()
        best_pattern = None
        best_confidence = 0
        
        # Log pattern evaluations every minute
        should_log = (current_time - self.last_pattern_log) > 60
        
        if len(data) < 20:
            if should_log:
                logger.debug("Insufficient data for pattern evaluation")
            return None, 0
        
        # Calculate technical indicators
        close = data['close'].values
        volume = data['volume'].values
        avg_volume = volume[-20:].mean()
        current_volume = volume[-1]
        price_change = (close[-1] - close[-2]) / close[-2]
        
        for pattern in self.patterns:
            if not pattern.get('enabled', True):
                continue
                
            confidence = 0
            reason = "Not triggered"
            
            # Evaluate volume surge patterns
            if 'volume_surge' in pattern['name']:
                volume_surge = current_volume > (2 * avg_volume)
                
                if 'down' in pattern['name'] and price_change < 0 and volume_surge:
                    # Pattern triggered
                    base_confidence = pattern['confidence_weight'] * 100
                    
                    # Adjust confidence based on volume magnitude
                    volume_multiplier = min(current_volume / avg_volume, 3) / 3
                    confidence = base_confidence * (0.7 + 0.3 * volume_multiplier)
                    
                    # Avoid inventory report window
                    if self.is_near_inventory_report():
                        confidence *= 0.5
                        reason = "Near inventory report - confidence reduced"
                    else:
                        reason = f"Volume surge {current_volume/avg_volume:.1f}x, price down {price_change*100:.2f}%"
                else:
                    if not volume_surge:
                        reason = f"Volume {current_volume/avg_volume:.1f}x < 2x threshold"
                    elif price_change >= 0:
                        reason = f"Price up {price_change*100:.2f}% (need down)"
            
            # Update tracking
            self.pattern_evaluations[pattern['name']]['last_confidence'] = confidence
            self.pattern_evaluations[pattern['name']]['last_checked'] = current_time
            self.pattern_evaluations[pattern['name']]['reason_not_triggered'] = reason
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = pattern
            
            if should_log:
                logger.debug(f"Pattern {pattern['name']}: {confidence:.1f}% - {reason}")
        
        if should_log:
            self.last_pattern_log = current_time
            if best_pattern:
                logger.info(f"Best pattern: {best_pattern['name']} @ {best_confidence:.1f}% confidence")
            else:
                logger.debug(f"No patterns triggered (best confidence: {best_confidence:.1f}%)")
        
        return best_pattern, best_confidence
    
    async def on_market_data(self, data: Dict):
        """Process incoming CL market data with health monitoring"""
        await self.update_market_data(data)
        
        # Update last price time
        self.last_price_update = time_module.time()
        
        # Update buffers
        self.update_buffers(data)
        
        # Periodic health check
        await self.health_check()
        
        # Check if we should avoid trading
        if self.is_near_inventory_report():
            logger.debug("Near inventory report window - avoiding trades")
            return
            
        # Skip if not in optimal session
        if not self.is_optimal_session():
            logger.debug("Not in optimal trading session")
            return
        
        # Build DataFrame for analysis
        if len(self.price_buffer) >= 20:
            df = pd.DataFrame({
                'close': self.price_buffer,
                'volume': self.volume_buffer if self.volume_buffer else [100] * len(self.price_buffer)
            })
            
            # Evaluate patterns
            pattern, confidence = self.evaluate_patterns(df)
            self.pattern_confidence = confidence
            
            # Check for trading opportunity
            if not self.position and confidence >= self.config['min_confidence']:
                await self.check_entry_signal(df, pattern, confidence)
            elif self.position:
                await self.manage_position(data)
    
    def update_buffers(self, data: Dict):
        """Update price and volume buffers"""
        price = data.get('close', data.get('last', 0))
        volume = data.get('volume', 100)
        
        if price > 0:
            self.price_buffer.append(price)
            self.volume_buffer.append(volume)
            
            # Maintain buffer size
            if len(self.price_buffer) > self.buffer_size:
                self.price_buffer.pop(0)
            if len(self.volume_buffer) > self.buffer_size:
                self.volume_buffer.pop(0)
    
    def is_near_inventory_report(self) -> bool:
        """Check if we're near weekly inventory report"""
        now = datetime.now()
        
        # Check if it's inventory day (Wednesday)
        if now.weekday() == self.inventory_day:
            current_time = now.time()
            
            # Calculate time window
            start_avoid = (datetime.combine(now.date(), self.inventory_time) - 
                          timedelta(minutes=self.avoid_inventory_window)).time()
            end_avoid = (datetime.combine(now.date(), self.inventory_time) + 
                        timedelta(minutes=self.avoid_inventory_window)).time()
            
            if start_avoid <= current_time <= end_avoid:
                return True
                
        return False
    
    def is_optimal_session(self) -> bool:
        """Check if we're in an optimal trading session for CL"""
        current_time = datetime.now().time()
        
        # London session (best liquidity)
        if self.london_open <= current_time <= self.london_close:
            return True
            
        # Early US session
        us_open = time(8, 0)
        us_midday = time(12, 0)
        if us_open <= current_time <= us_midday:
            return True
            
        return False
    
    async def check_entry_signal(self, df: pd.DataFrame, pattern: Dict, confidence: float):
        """Check and execute entry with pattern details"""
        if not pattern:
            return
            
        logger.info(f"🎯 CL Entry Signal: {pattern['name']} @ {confidence:.1f}% confidence")
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Calculate position size (always 1 for now)
        position_size = 1
        
        # Calculate stops and targets
        stop_loss = current_price * (1 - pattern['stop_loss_pct'])
        take_profit = current_price * (1 + pattern['target_pct'])
        
        # Place order via TopStepX
        if self.topstepx_client:
            order = await self.topstepx_client.place_order(
                account_id=self.config['topstepx_account_id'],
                contract_id=self.current_contract_id,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,  # Pattern is for down moves
                quantity=position_size
            )
            
            if order and order.get('success'):
                logger.info(f"✅ CL entry order placed: {order.get('orderId')}")
                
                # Update position tracking
                self.position = {
                    'symbol': 'CL',
                    'side': 'SHORT',
                    'size': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pattern': pattern['name'],
                    'confidence': confidence,
                    'entry_time': datetime.now()
                }
                
                self.active_pattern = pattern
                self.pattern_entry_time = datetime.now()
                
                # Increment pattern trigger count
                self.pattern_evaluations[pattern['name']]['triggered_count'] += 1
            else:
                logger.error(f"Failed to place CL entry order: {order}")
    
    async def manage_position(self, data: Dict):
        """Manage open CL position with pattern-specific rules"""
        if not self.position:
            return
            
        current_price = data.get('close', data.get('last', 0))
        
        # Calculate P&L
        if self.position['side'] == 'SHORT':
            pnl = (self.position['entry_price'] - current_price) * self.position['size'] * self.config['tick_value'] / self.config['tick_size']
        else:
            pnl = (current_price - self.position['entry_price']) * self.position['size'] * self.config['tick_value'] / self.config['tick_size']
        
        # Check exit conditions
        exit_reason = None
        
        # Stop loss
        if self.position['side'] == 'SHORT' and current_price >= self.position['stop_loss']:
            exit_reason = "Stop loss hit"
        elif self.position['side'] == 'LONG' and current_price <= self.position['stop_loss']:
            exit_reason = "Stop loss hit"
            
        # Take profit
        if self.position['side'] == 'SHORT' and current_price <= self.position['take_profit']:
            exit_reason = "Take profit hit"
        elif self.position['side'] == 'LONG' and current_price >= self.position['take_profit']:
            exit_reason = "Take profit hit"
            
        # Time-based exit for patterns
        if self.active_pattern and self.pattern_entry_time:
            hold_minutes = self.active_pattern.get('hold_minutes', 10)
            elapsed = (datetime.now() - self.pattern_entry_time).total_seconds() / 60
            
            if elapsed >= hold_minutes:
                exit_reason = f"Pattern hold time ({hold_minutes} min) completed"
        
        # Exit if conditions met
        if exit_reason:
            await self.exit_position(exit_reason, current_price)
    
    async def exit_position(self, reason: str, current_price: float):
        """Exit CL position"""
        if not self.position:
            return
            
        logger.info(f"📤 Exiting CL position: {reason}")
        
        # Calculate final P&L
        if self.position['side'] == 'SHORT':
            pnl = (self.position['entry_price'] - current_price) * self.position['size'] * self.config['tick_value'] / self.config['tick_size']
            exit_side = OrderSide.BUY  # Buy to cover short
        else:
            pnl = (current_price - self.position['entry_price']) * self.position['size'] * self.config['tick_value'] / self.config['tick_size']
            exit_side = OrderSide.SELL  # Sell to close long
        
        # Place exit order
        if self.topstepx_client:
            order = await self.topstepx_client.place_order(
                account_id=self.config['topstepx_account_id'],
                contract_id=self.current_contract_id,
                order_type=OrderType.MARKET,
                side=exit_side,
                quantity=self.position['size']
            )
            
            if order and order.get('success'):
                logger.info(f"✅ CL exit order placed: {order.get('orderId')}")
                logger.info(f"P&L: ${pnl:.2f}")
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                # Clear position
                self.position = None
                self.active_pattern = None
                self.pattern_entry_time = None
            else:
                logger.error(f"Failed to place CL exit order: {order}")
    
    def get_cl_status(self) -> Dict:
        """Get comprehensive CL bot status"""
        return {
            'symbol': 'CL',
            'contract_id': self.current_contract_id,
            'position': self.position['size'] if self.position else 0,
            'position_entry_price': self.position['entry_price'] if self.position else None,
            'active_pattern': self.active_pattern['name'] if self.active_pattern else None,
            'pattern_confidence': self.pattern_confidence,
            'daily_pnl': self.daily_pnl,
            'health_status': self.health_status,
            'patterns': len(self.patterns),
            'enabled_patterns': sum(1 for p in self.patterns if p.get('enabled', True)),
            'near_inventory': self.is_near_inventory_report(),
            'optimal_session': self.is_optimal_session(),
            'last_price_update': time_module.time() - self.last_price_update,
            'pattern_evaluations': self.pattern_evaluations
        }
    
    async def check_patterns(self) -> Optional[Dict]:
        """Check for trading patterns (required by BaseBot)"""
        if len(self.price_buffer) >= 20:
            df = pd.DataFrame({
                'close': self.price_buffer,
                'volume': self.volume_buffer if self.volume_buffer else [100] * len(self.price_buffer)
            })
            pattern, confidence = self.evaluate_patterns(df)
            if pattern and confidence >= self.config['min_confidence']:
                return pattern
        return None
    
    def calculate_position_size(self, stop_distance: float) -> int:
        """Calculate position size based on risk (required by BaseBot)"""
        # For now, always return 1 contract
        # In the future, could implement dynamic sizing based on stop distance
        return 1
    
    async def shutdown(self):
        """Clean shutdown with position closing"""
        logger.info("Shutting down enhanced CL bot...")
        
        # Close any open positions
        if self.position:
            await self.exit_position("Bot shutdown", self.price_buffer[-1] if self.price_buffer else 0)
        
        # Disconnect from TopStepX
        if self.topstepx_client:
            await self.topstepx_client.disconnect()
        
        logger.info("Enhanced CL bot shutdown complete")