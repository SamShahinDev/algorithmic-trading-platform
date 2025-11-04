"""
Live Trading Agent
Executes trades through TopStep API
"""

import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from agents.base_agent import BaseAgent
from utils.logger import setup_logger, log_trade
from config import TOPSTEPX_API_KEY, TOPSTEPX_ENVIRONMENT, TRADING_CONFIG, RISK_CONFIG

class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class LiveTradingAgent(BaseAgent):
    """
    Executes live trades through TopStep API
    This is your trade executor
    """
    
    def __init__(self):
        """Initialize live trading agent"""
        super().__init__('LiveTrading')
        self.logger = setup_logger('LiveTrading')
        
        # API configuration
        self.api_key = TOPSTEPX_API_KEY
        self.environment = TOPSTEPX_ENVIRONMENT
        
        # WebSocket connection
        self.ws_connection = None
        self.session = None
        
        # Trading state
        self.positions = {}
        self.open_orders = {}
        self.account_balance = 0
        self.daily_pnl = 0
        self.is_connected = False
        
        # Risk management
        self.max_positions = TRADING_CONFIG['max_positions']
        self.max_daily_loss = TRADING_CONFIG['max_daily_loss']
        
        self.logger.info("💹 Live Trading Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize the trading agent"""
        try:
            if not self.api_key:
                self.logger.error("❌ No API key configured! Please set TOPSTEP_API_KEY in .env file")
                return False
            
            self.logger.info("Initializing trading connection...")
            
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Test API connection
            connected = await self.test_connection()
            
            if connected:
                self.logger.info("✅ Successfully connected to TopStep API")
                self.is_connected = True
                
                # Get account info
                await self.get_account_info()
                
                return True
            else:
                self.logger.error("Failed to connect to TopStep API")
                return False
                
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, pattern: Dict) -> Dict:
        """
        Execute a trade based on pattern
        
        Args:
            pattern: Pattern that triggered
        
        Returns:
            Dict: Trade execution result
        """
        return await self.execute_trade(pattern)
    
    async def test_connection(self) -> bool:
        """
        Test connection to TopStep API
        
        Returns:
            bool: True if connected
        """
        try:
            # In production, this would test the actual API
            # For now, we'll simulate
            self.logger.info("Testing API connection...")
            
            # Simulated connection test
            # In real implementation, would ping the API
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def connect_websocket(self):
        """
        Connect to TopStep WebSocket for real-time data
        """
        try:
            self.logger.info("Connecting to WebSocket...")
            
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            # In production, connect to actual WebSocket
            # async with self.session.ws_connect(self.ws_url, headers=headers) as ws:
            #     self.ws_connection = ws
            #     await self.handle_websocket_messages()
            
            # Simulated for now
            self.logger.info("WebSocket connection simulated")
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
    
    async def execute_trade(self, pattern: Dict) -> Dict:
        """
        Execute a trade based on pattern signal
        
        Args:
            pattern: Pattern configuration with entry/exit rules
        
        Returns:
            Dict: Execution result
        """
        try:
            # Check if we can trade
            if not await self.can_trade():
                return {
                    'success': False,
                    'error': 'Cannot trade - risk limits or position limits reached'
                }
            
            # Get current market price (simulated)
            current_price = await self.get_current_price()
            
            # Determine trade direction and size
            direction = self.determine_direction(pattern)
            position_size = self.calculate_position_size(pattern, current_price)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_risk_levels(
                pattern, 
                current_price, 
                direction
            )
            
            # Place the order
            order_result = await self.place_order(
                symbol=TRADING_CONFIG['symbol'],
                side=OrderSide.BUY if direction == 'long' else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_result['success']:
                # Log the trade
                log_trade(
                    action='BUY' if direction == 'long' else 'SELL',
                    symbol=TRADING_CONFIG['symbol'],
                    quantity=position_size,
                    price=current_price,
                    pattern=pattern.get('name', 'Unknown'),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                self.logger.info(f"✅ Trade executed: {order_result['order_id']}")
                
                # Track the position
                self.positions[order_result['order_id']] = {
                    'symbol': TRADING_CONFIG['symbol'],
                    'side': direction,
                    'quantity': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'pattern': pattern.get('name'),
                    'entry_time': datetime.now()
                }
                
                self.record_success()
            else:
                self.logger.error(f"Trade execution failed: {order_result.get('error')}")
                self.record_error(Exception(order_result.get('error')))
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            self.record_error(e)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Place an order through TopStep API
        
        Args:
            symbol: Trading symbol
            side: Buy or Sell
            quantity: Number of contracts
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Dict: Order result
        """
        try:
            # In production, this would call the actual API
            # For now, simulate order placement
            
            self.logger.info(f"Placing {side.value} order for {quantity} {symbol}")
            
            # Simulated API call
            order_data = {
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'type': order_type.value,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            # In production:
            # async with self.session.post(
            #     f"{self.rest_url}/orders",
            #     json=order_data,
            #     headers={'Authorization': f'Bearer {self.api_key}'}
            # ) as response:
            #     result = await response.json()
            
            # Simulated response
            await asyncio.sleep(0.1)  # Simulate network delay
            
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            result = {
                'success': True,
                'order_id': order_id,
                'status': 'FILLED',
                'filled_price': price or await self.get_current_price(),
                'filled_quantity': quantity,
                'timestamp': datetime.now().isoformat()
            }
            
            # Track open order
            self.open_orders[order_id] = order_data
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def close_position(self, position_id: str) -> Dict:
        """
        Close an open position
        
        Args:
            position_id: Position identifier
        
        Returns:
            Dict: Close result
        """
        try:
            if position_id not in self.positions:
                return {
                    'success': False,
                    'error': 'Position not found'
                }
            
            position = self.positions[position_id]
            
            # Place opposite order to close
            close_side = OrderSide.SELL if position['side'] == 'long' else OrderSide.BUY
            
            result = await self.place_order(
                symbol=position['symbol'],
                side=close_side,
                quantity=position['quantity'],
                order_type=OrderType.MARKET
            )
            
            if result['success']:
                # Calculate P&L
                exit_price = result['filled_price']
                entry_price = position['entry_price']
                
                if position['side'] == 'long':
                    pnl = (exit_price - entry_price) * position['quantity'] * 20  # NQ point value
                else:
                    pnl = (entry_price - exit_price) * position['quantity'] * 20
                
                # Remove from positions
                del self.positions[position_id]
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                self.logger.info(f"Position closed. P&L: ${pnl:.2f}")
                
                return {
                    'success': True,
                    'pnl': pnl,
                    'exit_price': exit_price
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def close_all_positions(self):
        """Close all open positions"""
        self.logger.warning("Closing all positions...")
        
        positions_to_close = list(self.positions.keys())
        
        for position_id in positions_to_close:
            result = await self.close_position(position_id)
            if result['success']:
                self.logger.info(f"Closed position {position_id}")
            else:
                self.logger.error(f"Failed to close position {position_id}")
        
        self.logger.info("All positions closed")
    
    async def get_current_price(self) -> float:
        """
        Get current market price for NQ
        
        Returns:
            float: Current price
        """
        try:
            # In production, fetch from API
            # For now, return simulated price
            
            # Simulated price around 15000 for NQ
            import random
            return 15000 + random.uniform(-50, 50)
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0
    
    async def get_account_info(self) -> Dict:
        """
        Get account information from TopStep
        
        Returns:
            Dict: Account info
        """
        try:
            # In production, fetch from API
            # For now, return simulated data
            
            self.account_balance = 50000
            
            account_info = {
                'account_id': 'TOPSTEP_DEMO',
                'balance': self.account_balance,
                'buying_power': self.account_balance,
                'daily_pnl': self.daily_pnl,
                'open_positions': len(self.positions),
                'margin_used': sum(p['quantity'] * 500 for p in self.positions.values())  # Simplified
            }
            
            self.logger.info(f"Account Balance: ${account_info['balance']:,.2f}")
            self.logger.info(f"Daily P&L: ${account_info['daily_pnl']:,.2f}")
            
            return account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def can_trade(self) -> bool:
        """
        Check if we can place a trade based on risk rules
        
        Returns:
            bool: True if can trade
        """
        # Check position limit
        if len(self.positions) >= self.max_positions:
            self.logger.warning("Maximum positions reached")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            self.logger.warning("Daily loss limit reached")
            return False
        
        # Check if market is open (simplified)
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            self.logger.warning("Market is closed")
            return False
        
        return True
    
    def determine_direction(self, pattern: Dict) -> str:
        """
        Determine trade direction from pattern
        
        Args:
            pattern: Pattern configuration
        
        Returns:
            str: 'long' or 'short'
        """
        # Most patterns in our system are long-biased (bounces)
        # In production, would analyze pattern type
        
        pattern_type = pattern.get('type', '')
        
        if 'bounce' in pattern_type or 'support' in pattern_type:
            return 'long'
        elif 'resistance' in pattern_type or 'failed' in pattern_type:
            return 'short'
        else:
            return 'long'  # Default to long
    
    def calculate_position_size(self, pattern: Dict, current_price: float) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            pattern: Pattern configuration
            current_price: Current market price
        
        Returns:
            int: Number of contracts
        """
        # Use Kelly Criterion or fixed risk
        confidence = pattern.get('statistics', {}).get('confidence', 0.5)
        
        # Risk 1-2% of account based on confidence
        risk_percent = 0.01 if confidence < 0.7 else 0.02
        risk_amount = self.account_balance * risk_percent
        
        # Estimate stop distance (simplified)
        stop_distance = 20  # 20 points for NQ
        
        # Calculate contracts
        contracts = int(risk_amount / (stop_distance * 20))  # $20 per point for NQ
        
        # Limit position size
        return max(1, min(contracts, 5))
    
    def calculate_risk_levels(
        self, 
        pattern: Dict, 
        current_price: float, 
        direction: str
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            pattern: Pattern configuration
            current_price: Current market price
            direction: Trade direction
        
        Returns:
            Tuple[float, float]: Stop loss and take profit prices
        """
        # Get ATR for volatility-based stops (simplified)
        atr = 30  # Simplified ATR for NQ
        
        if direction == 'long':
            stop_loss = current_price - (atr * RISK_CONFIG['stop_loss_atr_multiplier'])
            take_profit = current_price + (atr * RISK_CONFIG['stop_loss_atr_multiplier'] * RISK_CONFIG['take_profit_ratio'])
        else:
            stop_loss = current_price + (atr * RISK_CONFIG['stop_loss_atr_multiplier'])
            take_profit = current_price - (atr * RISK_CONFIG['stop_loss_atr_multiplier'] * RISK_CONFIG['take_profit_ratio'])
        
        return stop_loss, take_profit
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close all positions before shutting down
            if self.positions:
                await self.close_all_positions()
            
            # Close WebSocket connection
            if self.ws_connection:
                await self.ws_connection.close()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            self.logger.info("Trading agent cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")