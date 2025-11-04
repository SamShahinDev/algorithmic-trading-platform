"""
ProjectX Gateway API Client
Handles authentication, market data, and order execution
"""

import asyncio
import aiohttp
import json
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

class ProjectXClient:
    """
    ProjectX Gateway API Client
    Manages authentication, WebSocket connections, and API calls
    """
    
    def __init__(self, username: str, api_key: str, api_url: str, market_hub: str, user_hub: str):
        """
        Initialize ProjectX client
        
        Args:
            username: ProjectX demo username
            api_key: ProjectX demo API key
            api_url: Base API URL
            market_hub: Market data hub URL
            user_hub: User/trading hub URL
        """
        self.username = username
        self.api_key = api_key
        self.api_url = api_url
        self.auth_url = f"{api_url}/api/Auth/loginKey"
        self.market_hub_url = market_hub
        self.user_hub_url = user_hub
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        
        # WebSocket connections
        self.market_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.user_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        
        # Callbacks
        self.on_market_data: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        self.on_account_update: Optional[Callable] = None
        
        # Logger
        self.logger = logging.getLogger('ProjectXClient')
        
        # Reconnection settings
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        self.reconnect_attempts = 0
    
    async def connect(self) -> bool:
        """
        Connect to ProjectX API
        
        Returns:
            bool: True if connected successfully
        """
        try:
            self.logger.info("Connecting to ProjectX API...")
            self.state = ConnectionState.CONNECTING
            
            # Create session
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Authenticate
            if await self.authenticate():
                self.state = ConnectionState.AUTHENTICATED
                
                # Connect to WebSocket hubs
                await self.connect_websockets()
                
                self.state = ConnectionState.CONNECTED
                self.logger.info("✅ Successfully connected to ProjectX API")
                return True
            else:
                self.state = ConnectionState.ERROR
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.state = ConnectionState.ERROR
            return False
    
    async def authenticate(self) -> bool:
        """
        Authenticate with ProjectX API using JWT
        
        Returns:
            bool: True if authenticated successfully
        """
        try:
            # Check if we have valid credentials
            if not self.username or not self.api_key:
                self.logger.error("Missing username or API key")
                return False
            
            # Check if token is still valid
            if self.auth_token and self.token_expiry and datetime.now() < self.token_expiry:
                self.logger.info("Using existing valid token")
                return True
            
            self.logger.info("Authenticating with ProjectX...")
            
            # Prepare authentication request
            auth_data = {
                "userName": self.username,
                "apiKey": self.api_key
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'text/plain'
            }
            
            # Send authentication request
            async with self.session.post(
                self.auth_url,
                json=auth_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('success'):
                        self.auth_token = result.get('token')
                        # Token typically valid for 24 hours
                        self.token_expiry = datetime.now() + timedelta(hours=23)
                        
                        self.logger.info("✅ Authentication successful")
                        return True
                    else:
                        error_msg = result.get('errorMessage', 'Unknown error')
                        self.logger.error(f"Authentication failed: {error_msg}")
                        return False
                else:
                    self.logger.error(f"Authentication failed with status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    async def connect_websockets(self):
        """Connect to WebSocket hubs for real-time data"""
        try:
            if not self.auth_token:
                self.logger.error("No auth token available for WebSocket connection")
                return
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}'
            }
            
            # Connect to market data hub
            self.logger.info("Connecting to market data hub...")
            # Note: This is a simplified connection. ProjectX uses SignalR which requires special handling
            # In production, you'd use a SignalR client library like signalrcore
            
            # For now, we'll note that WebSocket connections require SignalR protocol
            self.logger.warning("WebSocket connections require SignalR protocol implementation")
            
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
    
    async def get_account_info(self) -> Dict:
        """
        Get account information
        
        Returns:
            Dict: Account information
        """
        try:
            if not self.auth_token:
                await self.authenticate()
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}'
            }
            
            async with self.session.get(
                f"{self.api_url}/api/Account",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to get account info: {response.status}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def place_order(self, order_data: Dict) -> Dict:
        """
        Place an order
        
        Args:
            order_data: Order details
        
        Returns:
            Dict: Order result
        """
        try:
            if not self.auth_token:
                await self.authenticate()
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}',
                'Content-Type': 'application/json'
            }
            
            # ProjectX specific order format
            projectx_order = {
                "symbol": order_data.get('symbol'),
                "side": order_data.get('side'),  # BUY or SELL
                "quantity": order_data.get('quantity'),
                "orderType": order_data.get('order_type', 'MARKET'),
                "price": order_data.get('price'),
                "stopPrice": order_data.get('stop_loss'),
                "takeProfitPrice": order_data.get('take_profit')
            }
            
            async with self.session.post(
                f"{self.api_url}/api/Orders",
                json=projectx_order,
                headers=headers
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    self.logger.info(f"Order placed successfully: {result.get('orderId')}")
                    return {
                        'success': True,
                        'order_id': result.get('orderId'),
                        'status': result.get('status'),
                        **result
                    }
                else:
                    self.logger.error(f"Order placement failed: {result}")
                    return {
                        'success': False,
                        'error': result.get('message', 'Order failed')
                    }
                    
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_positions(self) -> list:
        """
        Get open positions
        
        Returns:
            list: List of open positions
        """
        try:
            if not self.auth_token:
                await self.authenticate()
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}'
            }
            
            async with self.session.get(
                f"{self.api_url}/api/Positions",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to get positions: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_market_data(self, symbol: str) -> Dict:
        """
        Get current market data for a symbol
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict: Market data
        """
        try:
            if not self.auth_token:
                await self.authenticate()
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}'
            }
            
            async with self.session.get(
                f"{self.api_url}/api/MarketData/{symbol}",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to get market data: {response.status}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    async def close_position(self, position_id: str) -> Dict:
        """
        Close a position
        
        Args:
            position_id: Position ID to close
        
        Returns:
            Dict: Close result
        """
        try:
            if not self.auth_token:
                await self.authenticate()
            
            headers = {
                'Authorization': f'Bearer {self.auth_token}'
            }
            
            async with self.session.delete(
                f"{self.api_url}/api/Positions/{position_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.info(f"Position closed: {position_id}")
                    return {
                        'success': True,
                        **result
                    }
                else:
                    self.logger.error(f"Failed to close position: {response.status}")
                    return {
                        'success': False,
                        'error': f'Failed with status {response.status}'
                    }
                    
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def disconnect(self):
        """Disconnect from ProjectX API"""
        try:
            self.logger.info("Disconnecting from ProjectX API...")
            
            # Close WebSocket connections
            if self.market_ws:
                await self.market_ws.close()
            if self.user_ws:
                await self.user_ws.close()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            self.state = ConnectionState.DISCONNECTED
            self.auth_token = None
            self.token_expiry = None
            
            self.logger.info("Disconnected from ProjectX API")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def keep_alive(self):
        """Keep connection alive with periodic pings"""
        while self.state == ConnectionState.CONNECTED:
            try:
                # Refresh token if needed
                if self.token_expiry and datetime.now() > self.token_expiry - timedelta(hours=1):
                    await self.authenticate()
                
                # Send keepalive ping (implementation depends on ProjectX requirements)
                await asyncio.sleep(30)  # Ping every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Keep-alive error: {e}")
                await self.reconnect()
    
    async def reconnect(self):
        """Attempt to reconnect on connection loss"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        self.logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        await asyncio.sleep(self.reconnect_delay)
        
        if await self.connect():
            self.reconnect_attempts = 0
            self.logger.info("Reconnection successful")
        else:
            await self.reconnect()