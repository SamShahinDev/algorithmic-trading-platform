"""
Base Bot Class
Abstract base class for all trading bots with common functionality
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Optional, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionStatus(Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"

class BaseBot(ABC):
    """Abstract base class for all trading bots"""
    
    def __init__(self, config: Dict):
        """
        Initialize bot with configuration
        
        Args:
            config: Bot configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ES', 'CL')
                - tick_size: Minimum price increment
                - tick_value: Dollar value per tick
                - max_position_size: Maximum contracts
                - risk_per_trade: Max risk in dollars per trade
                - daily_loss_limit: Maximum daily loss
                - patterns: List of validated patterns to trade
        """
        self.config = config
        self.symbol = config['symbol']
        self.tick_size = config['tick_size']
        self.tick_value = config['tick_value']
        
        # Position tracking
        self.position = 0  # Current position size
        self.position_side = PositionStatus.FLAT
        self.avg_entry_price = 0
        self.position_pnl = 0
        
        # Order management
        self.orders = {}
        self.pending_orders = {}
        self.order_counter = 0
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 1)
        self.risk_per_trade = config.get('risk_per_trade', 50)
        self.daily_loss_limit = config.get('daily_loss_limit', -500)
        
        # Performance tracking
        self.trades_today = []
        self.daily_pnl = 0
        self.total_pnl = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        
        # Market data
        self.current_price = 0
        self.current_bid = 0
        self.current_ask = 0
        self.current_time = None
        self.market_data_buffer = []  # Store recent market data
        
        # Pattern configuration
        self.patterns = config.get('patterns', [])
        self.active_pattern = None
        
        # Bot state
        self.is_running = False
        self.is_trading_enabled = True
        self.last_trade_time = None
        self.min_trade_interval = config.get('min_trade_interval_seconds', 60)
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.symbol}")
        
        self.logger.info(f"Initialized {self.symbol} bot with {len(self.patterns)} patterns")
        
    @abstractmethod
    async def on_market_data(self, data: Dict):
        """
        Process incoming market data
        Must be implemented by subclasses
        
        Args:
            data: Market data dictionary with keys:
                - timestamp: Data timestamp
                - open: Open price
                - high: High price  
                - low: Low price
                - close: Close price
                - volume: Volume
                - bid: Current bid
                - ask: Current ask
        """
        pass
        
    @abstractmethod
    async def check_patterns(self) -> Optional[Dict]:
        """
        Check for trading patterns
        Must be implemented by subclasses
        
        Returns:
            Pattern dictionary if signal found, None otherwise
        """
        pass
        
    @abstractmethod
    def calculate_position_size(self, stop_distance: float) -> int:
        """
        Calculate position size based on risk
        Must be implemented by subclasses
        
        Args:
            stop_distance: Distance to stop loss in points
            
        Returns:
            Number of contracts to trade
        """
        pass
        
    async def update_market_data(self, data: Dict):
        """Update internal market data state"""
        self.current_price = data.get('close', self.current_price)
        self.current_bid = data.get('bid', self.current_price)
        self.current_ask = data.get('ask', self.current_price)
        self.current_time = pd.Timestamp(data.get('timestamp', datetime.now()))
        
        # Add to buffer for pattern analysis
        self.market_data_buffer.append(data)
        
        # Keep only recent data (e.g., last 100 bars)
        if len(self.market_data_buffer) > 100:
            self.market_data_buffer.pop(0)
            
    async def place_order(self, 
                         side: OrderSide, 
                         size: int, 
                         order_type: OrderType = OrderType.MARKET,
                         limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict:
        """
        Place an order
        
        Args:
            side: Buy or sell
            size: Number of contracts
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            stop_loss: Stop loss price for brackets
            take_profit: Take profit price for brackets
            
        Returns:
            Order dictionary
        """
        # Check if trading is enabled
        if not self.is_trading_enabled:
            self.logger.warning("Trading disabled, order rejected")
            return {}
            
        # Check position limits
        if self.position != 0 and size > self.max_position_size:
            self.logger.warning(f"Position size {size} exceeds max {self.max_position_size}")
            size = self.max_position_size
            
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            self.logger.error(f"Daily loss limit reached: {self.daily_pnl}")
            self.is_trading_enabled = False
            return {}
            
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            self.is_trading_enabled = False
            return {}
            
        # Generate order
        order = {
            'id': self.generate_order_id(),
            'symbol': self.symbol,
            'side': side.value,
            'size': size,
            'type': order_type.value,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        
        self.pending_orders[order['id']] = order
        self.logger.info(f"Order placed: {order}")
        
        # Here you would integrate with actual broker API
        # For now, we'll simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            await self.simulate_order_fill(order)
            
        return order
        
    async def simulate_order_fill(self, order: Dict):
        """Simulate order fill for testing"""
        # Simulate market order fill
        if order['type'] == 'market':
            fill_price = self.current_ask if order['side'] == 'buy' else self.current_bid
            
            # Update position
            if order['side'] == 'buy':
                if self.position < 0:
                    # Closing short
                    pnl = (self.avg_entry_price - fill_price) * abs(self.position) * self.tick_value / self.tick_size
                    self.record_trade(order, fill_price, pnl)
                    
                self.position += order['size']
                if self.position > 0:
                    self.position_side = PositionStatus.LONG
                    self.avg_entry_price = fill_price
                    
            else:  # sell
                if self.position > 0:
                    # Closing long
                    pnl = (fill_price - self.avg_entry_price) * self.position * self.tick_value / self.tick_size
                    self.record_trade(order, fill_price, pnl)
                    
                self.position -= order['size']
                if self.position < 0:
                    self.position_side = PositionStatus.SHORT
                    self.avg_entry_price = fill_price
                    
            if self.position == 0:
                self.position_side = PositionStatus.FLAT
                self.avg_entry_price = 0
                
            # Move from pending to filled
            order['status'] = 'filled'
            order['fill_price'] = fill_price
            order['fill_time'] = datetime.now()
            self.orders[order['id']] = order
            del self.pending_orders[order['id']]
            
            self.logger.info(f"Order filled: {order['id']} at {fill_price}")
            
    def record_trade(self, order: Dict, fill_price: float, pnl: float):
        """Record completed trade"""
        trade = {
            'order_id': order['id'],
            'symbol': self.symbol,
            'side': order['side'],
            'size': order['size'],
            'entry_price': self.avg_entry_price,
            'exit_price': fill_price,
            'pnl': pnl,
            'timestamp': datetime.now()
        }
        
        self.trades_today.append(trade)
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        self.logger.info(f"Trade recorded: P&L ${pnl:.2f}, Daily P&L: ${self.daily_pnl:.2f}")
        
    async def close_position(self, reason: str = "manual"):
        """Close current position"""
        if self.position == 0:
            self.logger.info("No position to close")
            return
            
        side = OrderSide.SELL if self.position > 0 else OrderSide.BUY
        size = abs(self.position)
        
        self.logger.info(f"Closing position: {self.position} contracts, reason: {reason}")
        
        await self.place_order(
            side=side,
            size=size,
            order_type=OrderType.MARKET
        )
        
    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.order_counter}"
        
    def can_trade(self) -> bool:
        """Check if bot can place new trades"""
        # Check if trading is enabled
        if not self.is_trading_enabled:
            return False
            
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
            
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            return False
            
        # Check minimum time between trades
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False
                
        return True
        
    async def start(self):
        """Start the bot"""
        self.is_running = True
        self.logger.info(f"{self.symbol} bot started")
        
        while self.is_running:
            try:
                # Main trading loop
                if self.can_trade() and self.position == 0:
                    # Check for patterns only when flat
                    pattern = await self.check_patterns()
                    
                    if pattern:
                        await self.enter_trade(pattern)
                        
                elif self.position != 0:
                    # Manage existing position
                    await self.manage_position()
                    
                await asyncio.sleep(1)  # Main loop delay
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)
                
    async def enter_trade(self, pattern: Dict):
        """Enter a trade based on pattern signal"""
        self.logger.info(f"Entering trade for pattern: {pattern['name']}")
        
        # Calculate position size
        stop_distance = pattern.get('stop_distance', 10)  # Default 10 ticks
        position_size = self.calculate_position_size(stop_distance)
        
        # Determine side
        side = OrderSide.BUY if pattern['direction'] == 'bullish' else OrderSide.SELL
        
        # Calculate stop loss and take profit
        if side == OrderSide.BUY:
            stop_loss = self.current_price - (stop_distance * self.tick_size)
            take_profit = self.current_price + (pattern.get('target_distance', 20) * self.tick_size)
        else:
            stop_loss = self.current_price + (stop_distance * self.tick_size)
            take_profit = self.current_price - (pattern.get('target_distance', 20) * self.tick_size)
            
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
            self.last_trade_time = datetime.now()
            
    async def manage_position(self):
        """Manage existing position"""
        if not self.active_pattern:
            return
            
        # Calculate current P&L
        if self.position > 0:
            current_pnl = (self.current_bid - self.avg_entry_price) * self.position * self.tick_value / self.tick_size
        elif self.position < 0:
            current_pnl = (self.avg_entry_price - self.current_ask) * abs(self.position) * self.tick_value / self.tick_size
        else:
            current_pnl = 0
            
        self.position_pnl = current_pnl
        
        # Check for exit conditions
        # Time-based exit
        if self.last_trade_time:
            hold_time = (datetime.now() - self.last_trade_time).total_seconds() / 60
            max_hold_time = self.active_pattern.get('max_hold_minutes', 60)
            
            if hold_time >= max_hold_time:
                self.logger.info(f"Time exit triggered: {hold_time:.1f} minutes")
                await self.close_position("time_exit")
                return
                
    async def stop(self):
        """Stop the bot"""
        self.is_running = False
        
        # Close any open positions
        if self.position != 0:
            await self.close_position("bot_stopped")
            
        self.logger.info(f"{self.symbol} bot stopped")
        
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'symbol': self.symbol,
            'is_running': self.is_running,
            'position': self.position,
            'position_side': self.position_side.value,
            'avg_entry_price': self.avg_entry_price,
            'position_pnl': self.position_pnl,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'trades_today': len(self.trades_today),
            'consecutive_losses': self.consecutive_losses,
            'is_trading_enabled': self.is_trading_enabled,
            'active_pattern': self.active_pattern.get('name') if self.active_pattern else None
        }
        
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)"""
        self.trades_today = []
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.is_trading_enabled = True
        
        self.logger.info("Daily statistics reset")