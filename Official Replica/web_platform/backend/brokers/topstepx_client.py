"""
TopStepX/Tradovate API Client
Handles order execution and market data for TopStepX funded accounts
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import logging
from enum import Enum
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import time
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import rate limiter
try:
    from shared.rate_limiter import (
        get_general_limiter, get_historical_limiter, get_order_limiter,
        SmartThrottle, RateLimitDashboard
    )
except ImportError:
    logger.warning("Rate limiter not available - running without rate limit protection")
    get_general_limiter = lambda: None
    get_historical_limiter = lambda: None
    get_order_limiter = lambda: None

# Load TopStepX environment variables
load_dotenv('.env.topstepx')

logger = logging.getLogger(__name__)

class OrderType(Enum):
    LIMIT = 1
    MARKET = 2
    STOP = 4
    TRAILING_STOP = 5
    JOIN_BID = 6
    JOIN_ASK = 7

class OrderSide(Enum):
    BUY = 0  # Bid
    SELL = 1  # Ask

class OrderStatus(Enum):
    PENDING = "Pending"
    WORKING = "Working"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"

class TopStepXClient:
    """
    TopStepX API client for order execution and market data
    """
    
    def __init__(self, api_key: str = None, username: str = None):
        """Initialize TopStepX client"""
        # Load from environment variables if not provided
        self.api_key = api_key or os.getenv('TOPSTEPX_API_KEY', '86fzI3xGFVj06PWyiFYhDQGZ50QzxiJzhXW04F347h8=')
        self.username = username or os.getenv('TOPSTEPX_USERNAME', '')
        
        # TopStepX ProjectX Gateway API URLs from environment
        self.base_url = os.getenv('API_BASE_URL', 'https://api.topstepx.com/api')
        self.user_hub = os.getenv('USER_HUB_URL', 'https://rtc.topstepx.com/hubs/user')
        self.market_hub = os.getenv('MARKET_HUB_URL', 'https://rtc.topstepx.com/hubs/market')
        
        # Connection state
        self.connected = False
        self.session = None
        self.session_token = None
        self.token_expiry = None
        self.account_id = None
        
        # Account info
        self.account_balance = 0
        self.daily_pnl = 0
        self.open_positions = []
        
        # Market data
        self.current_price = 0
        self.bid = 0
        self.ask = 0
        
        # Risk settings (from TopStepX account)
        self.use_brackets = os.getenv('USE_AUTO_BRACKETS', 'true').lower() == 'true'
        
        # Heartbeat state
        self._heartbeat_running = False
        self._heartbeat_task = None
        self.bracket_profit = float(os.getenv('BRACKET_PROFIT_TARGET', '100'))
        self.bracket_stop = float(os.getenv('BRACKET_STOP_LOSS', '100'))
        self.default_contract_size = int(os.getenv('DEFAULT_CONTRACT_SIZE', '1'))
        
        # Initialize rate limiters
        self.general_limiter = get_general_limiter() if get_general_limiter else None
        self.historical_limiter = get_historical_limiter() if get_historical_limiter else None
        self.order_limiter = get_order_limiter() if get_order_limiter else None
        self.smart_throttle = SmartThrottle(self.general_limiter) if self.general_limiter else None
        
        if self.username:
            logger.info(f"🔌 TopStepX Client initialized for user: {self.username}")
        else:
            logger.warning("⚠️ TopStepX username not configured - please add to .env.topstepx")
            logger.info("🔌 TopStepX Client initialized (no username yet)")
    
    async def _make_request(self, method: str, endpoint: str, data: Dict = None, request_type: str = "general") -> Dict:
        """
        Rate-limited wrapper for API requests
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request payload
            request_type: Type of request for rate limiting
            
        Returns:
            API response or error
        """
        # Determine which rate limiter to use
        if "history" in endpoint.lower() or "bars" in endpoint.lower():
            limiter = self.historical_limiter
        elif "order" in endpoint.lower() or "trade" in endpoint.lower():
            limiter = self.order_limiter
        else:
            limiter = self.general_limiter
            
        # Check rate limit if available
        if limiter:
            is_safe, stats = await limiter.check_and_track(request_type)
            
            if not is_safe:
                error_msg = f"Rate limit protection triggered! Usage: {stats['percentage']:.0f}% ({stats['count']}/{stats['limit']})"
                logger.error(f"🚨 {error_msg}")
                
                # Wait if needed before rejecting
                await limiter.wait_if_needed()
                
                # Try again after waiting
                is_safe, stats = await limiter.check_and_track(request_type)
                if not is_safe:
                    raise Exception(error_msg)
                    
            # Log high usage
            if stats['percentage'] > 70:
                logger.warning(f"Rate limit at {stats['percentage']:.0f}% for {limiter.name}")
        
        # Make the actual request
        return await self.request(method, endpoint, data)
    
    async def request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """
        Generic request method for TopStepX API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/api/Account/search')
            data: Request data/payload
        
        Returns:
            API response as dictionary
        """
        if not self.connected:
            logger.warning("Not connected to TopStepX, attempting to connect...")
            await self.connect()
            
        if not self.session_token:
            logger.error("No session token available")
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        # Handle endpoint properly - remove leading /api if base_url already has it
        if endpoint.startswith('/api/'):
            endpoint = endpoint[4:]  # Remove '/api' prefix
        elif endpoint.startswith('api/'):
            endpoint = endpoint[3:]  # Remove 'api' prefix
        url = f"{self.base_url}{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {
            'Authorization': f'Bearer {self.session_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            async with self.session.request(method, url, headers=headers, json=data) as response:
                result = await response.json()
                
                if response.status == 200:
                    return result
                else:
                    logger.error(f"API request failed: {response.status} - {result}")
                    return result
                    
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    async def connect(self) -> bool:
        """Connect to TopStepX ProjectX Gateway API"""
        try:
            if not self.username:
                logger.error("❌ Cannot connect: TopStepX username not configured")
                logger.info("ℹ️ Please add TOPSTEPX_USERNAME to .env.topstepx file")
                return False
                
            self.session = aiohttp.ClientSession()
            
            # Authenticate with API key using correct endpoint
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json"
            }
            
            auth_data = {
                "userName": self.username,
                "apiKey": self.api_key
            }
            
            logger.info(f"🔐 Attempting to authenticate user: {self.username}")
            
            # TopStepX authentication endpoint
            async with self.session.post(
                f"{self.base_url}/Auth/loginKey",
                headers=headers,
                json=auth_data
            ) as response:
                if response.status == 200:
                    # Response includes session token
                    result = await response.json()
                    self.session_token = result.get("token", result.get("your_session_token_here"))
                    self.token_expiry = datetime.now() + timedelta(hours=24)
                    
                    # Set account ID from environment if available
                    env_account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
                    if env_account_id:
                        self.account_id = int(env_account_id)
                        logger.info(f"Using configured account ID: {self.account_id}")
                    
                    # Get account information
                    await self._get_account_info()
                    
                    self.connected = True
                    logger.info(f"✅ Connected to TopStepX ProjectX Gateway")
                    
                    # Disabled aggressive market data streaming to avoid rate limits
                    # asyncio.create_task(self.stream_market_data())
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to connect: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def validate_session(self) -> bool:
        """Validate session token (required every 24 hours)"""
        if not self.session_token:
            return False
        
        try:
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Auth/validate",
                headers=headers
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def validate_token(self) -> bool:
        """Validate current authentication token"""
        try:
            # Check token expiry first
            if self.token_expiry and datetime.now() > self.token_expiry:
                logger.info("TOKEN_EXPIRED refreshing")
                return False
            
            # Validate with API
            valid = await self.validate_session()
            if valid:
                logger.debug("TOKEN_VALID ok")
            else:
                logger.warning("TOKEN_INVALID")
            return valid
        except Exception as e:
            logger.error(f"TOKEN_VALIDATE_ERROR: {e}")
            return False
    
    async def refresh_token(self) -> bool:
        """Refresh authentication token"""
        try:
            logger.info("TOKEN_REFRESH starting")
            
            # Close existing session
            if self.session:
                await self.session.close()
                self.session = None
            
            # Re-authenticate
            success = await self.connect()
            
            if success:
                logger.info("TOKEN_REFRESH ok")
            else:
                logger.error("TOKEN_REFRESH failed")
            
            return success
        except Exception as e:
            logger.error(f"TOKEN_REFRESH_ERROR: {e}")
            return False
    
    def start_token_heartbeat(self, interval_sec: int = 1500):
        """Start token validation heartbeat to prevent silent stalls"""
        async def heartbeat_loop():
            while self._heartbeat_running:
                try:
                    await asyncio.sleep(interval_sec)
                    if self._heartbeat_running:  # Check again after sleep
                        valid = await self.validate_session()
                        if not valid:
                            logger.warning("Token validation failed, reconnecting...")
                            await self.connect()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self._heartbeat_running:
                        logger.error(f"Token heartbeat error: {e}")
        
        self._heartbeat_running = True
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        logger.info(f"Started token heartbeat every {interval_sec}s")
    
    def stop_token_heartbeat(self):
        """Stop the token heartbeat"""
        self._heartbeat_running = False
        if hasattr(self, '_heartbeat_task') and self._heartbeat_task:
            self._heartbeat_task.cancel()
            logger.info("Token heartbeat stopped")
    
    async def _get_account_info(self):
        """Get account information after authentication"""
        try:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            # Search for active accounts
            async with self.session.post(
                f"{self.base_url}/Account/search",
                headers=headers,
                json={"onlyActiveAccounts": True}
            ) as response:
                if response.status == 200:
                    accounts = await response.json()
                    logger.info(f"Account search returned: {accounts}")
                    if accounts and len(accounts) > 0:
                        self.account_id = accounts[0].get("id", 1)
                        logger.info(f"✅ Account ID: {self.account_id}")
                    else:
                        logger.warning("⚠️ No accounts found - using default ID")
                        self.account_id = 1
                else:
                    logger.error(f"Account search failed: {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
    
    async def get_available_contracts(self, live: bool = True) -> List[Dict]:
        """Get list of available tradeable contracts"""
        if not self.connected:
            await self.connect()
            
        try:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Contract/available",
                headers=headers,
                json={"live": live}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        contracts = result.get("contracts", [])
                        logger.info(f"Found {len(contracts)} available contracts")
                        return contracts
                else:
                    logger.error(f"Failed to get available contracts: {response.status}")
        except Exception as e:
            logger.error(f"Error getting available contracts: {e}")
        
        return []
    
    async def _get_contract_id(self, symbol: str = "NQ") -> Optional[str]:
        """Get contract ID for trading symbol"""
        # First try to get from available contracts
        contracts = await self.get_available_contracts(live=True)
        
        # Search for matching symbol in available contracts
        for contract in contracts:
            contract_name = contract.get("name", "")
            contract_id = contract.get("id", "")
            
            # Check for exact match or contains
            if symbol.upper() in contract_name or symbol.upper() in contract_id:
                logger.info(f"Found available contract for {symbol}: {contract_id} ({contract_name})")
                return contract_id
        
        # Fall back to direct mapping if not found in available
        contract_map = {
            "NQ": "CON.F.US.ENQ.U25",  # E-mini NASDAQ-100 (NQU5)
            "MNQ": "CON.F.US.MNQ.U25",  # Micro E-mini NASDAQ-100 (MNQU5)
            "ES": "CON.F.US.EP.U25",    # E-mini S&P 500 (ESU5)
            "MES": "CON.F.US.MES.U25",  # Micro E-mini S&P 500 (MESU5)
            "CL": "CLU25",               # Crude Oil September 2025 (front month with better API support)
            "MCL": "MCLU25",             # Micro Crude Oil September 2025
        }
        
        if symbol.upper() in contract_map:
            contract_id = contract_map[symbol.upper()]
            logger.info(f"Using mapped contract ID for {symbol}: {contract_id}")
            return contract_id
        
        # Default to E-mini NASDAQ-100
        logger.warning(f"No contract found for {symbol}, using default NQ")
        return "CON.F.US.ENQ.U25"
    
    async def disconnect(self):
        """Disconnect from API"""
        if self.session:
            await self.session.close()
        self.connected = False
        logger.info("Disconnected from TopStepX")
    
    async def place_order(self, 
                          symbol: str = "NQ",
                          side: OrderSide = OrderSide.BUY,
                          quantity: int = 1,
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          custom_tag: Optional[str] = None) -> Dict:
        """
        Place an order through TopStepX ProjectX Gateway
        
        IMPORTANT: TopStepX automatic brackets will be applied based on account settings
        """
        if not self.connected:
            return {"success": False, "error": "Not connected"}
        
        if not self.account_id:
            logger.warning("No account ID, attempting to get account info")
            # Try to get from environment
            env_account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
            if env_account_id:
                self.account_id = int(env_account_id)
                logger.info(f"Using account ID from environment: {self.account_id}")
            else:
                await self._get_account_info()
                if not self.account_id:
                    logger.error("No account ID available - cannot place order")
                    return {"success": False, "error": "No account ID configured"}
        
        try:
            # Get the contract ID for the symbol
            contract_id = await self._get_contract_id(symbol)
            if not contract_id:
                return {"success": False, "error": "Failed to get contract ID"}
            
            logger.info(f"📝 Placing {order_type.name} {side.name} order for {quantity} {symbol} @ {contract_id}")
            
            # Build order data according to API spec
            order_data = {
                "accountId": self.account_id,
                "contractId": contract_id,
                "type": order_type.value,  # Already correct enum values (1=Limit, 2=Market, etc)
                "side": side.value,  # Already correct enum values (0=Buy, 1=Sell)
                "size": quantity,
                "limitPrice": None,
                "stopPrice": None,
                "trailPrice": None,
                "customTag": custom_tag,
                "linkedOrderId": None
            }
            
            # Add price for limit orders
            if order_type == OrderType.LIMIT and price:
                order_data["limitPrice"] = float(price)
                logger.info(f"  Limit price: ${price:,.2f}")
            
            # Add stop price for stop orders
            if order_type == OrderType.STOP and stop_price:
                order_data["stopPrice"] = float(stop_price)
                logger.info(f"  Stop price: ${stop_price:,.2f}")
            
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Order/place",
                headers=headers,
                json=order_data
            ) as response:
                result_text = await response.text()
                if response.status == 200:
                    try:
                        result = json.loads(result_text)
                        if result.get("success", False):
                            order_id = result.get("orderId")
                            logger.info(f"✅ Order placed successfully! Order ID: {order_id}")
                            return {
                                "success": True,
                                "order_id": order_id,
                                "data": result
                            }
                        else:
                            error_msg = result.get("errorMessage", "Unknown error")
                            error_code = result.get("errorCode", "N/A")
                            logger.error(f"Order rejected - Code: {error_code}, Message: {error_msg}")
                            logger.error(f"Full response: {result}")
                            return {
                                "success": False,
                                "error": error_msg,
                                "error_code": error_code,
                                "full_response": result
                            }
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response: {result_text}")
                        return {
                            "success": False,
                            "error": f"Invalid response: {result_text}"
                        }
                else:
                    logger.error(f"Order failed ({response.status}): {result_text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {result_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order"""
        if not self.connected:
            return False
        
        if not self.account_id:
            logger.error("No account ID for order cancellation")
            return False
        
        try:
            cancel_data = {
                "accountId": self.account_id,
                "orderId": order_id
            }
            
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Order/cancel",
                headers=headers,
                json=cancel_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        logger.info(f"✅ Order {order_id} cancelled")
                        return True
                    else:
                        error_msg = result.get("errorMessage", "Unknown error")
                        logger.error(f"Failed to cancel order {order_id}: {error_msg}")
                        return False
                else:
                    error = await response.text()
                    logger.error(f"Cancel request failed ({response.status}): {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False
    
    async def modify_order(self, order_id: int, 
                          size: Optional[int] = None,
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          trail_price: Optional[float] = None) -> bool:
        """Modify an open order"""
        if not self.connected or not self.account_id:
            return False
        
        try:
            modify_data = {
                "accountId": self.account_id,
                "orderId": order_id,
                "size": size,
                "limitPrice": limit_price,
                "stopPrice": stop_price,
                "trailPrice": trail_price
            }
            
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Order/modify",
                headers=headers,
                json=modify_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        logger.info(f"✅ Order {order_id} modified successfully")
                        return True
                    else:
                        error_msg = result.get("errorMessage", "Unknown error")
                        logger.error(f"Failed to modify order {order_id}: {error_msg}")
                        return False
                else:
                    error = await response.text()
                    logger.error(f"Modify request failed ({response.status}): {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Modify error: {e}")
            return False
    
    async def close_position(self, contract_id: str) -> bool:
        """Close a specific position by contract ID"""
        if not self.connected or not self.account_id:
            return False
        
        try:
            close_data = {
                "accountId": self.account_id,
                "contractId": contract_id
            }
            
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Position/closeContract",
                headers=headers,
                json=close_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        logger.info(f"✅ Position {contract_id} closed successfully")
                        return True
                    else:
                        error_msg = result.get("errorMessage", "Unknown error")
                        logger.error(f"Failed to close position {contract_id}: {error_msg}")
                        return False
                else:
                    error = await response.text()
                    logger.error(f"Close position failed ({response.status}): {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return False
    
    async def partial_close_position(self, contract_id: str, size: int) -> bool:
        """Partially close a position"""
        if not self.connected or not self.account_id:
            return False
        
        try:
            partial_close_data = {
                "accountId": self.account_id,
                "contractId": contract_id,
                "size": size
            }
            
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            async with self.session.post(
                f"{self.base_url}/Position/partialCloseContract",
                headers=headers,
                json=partial_close_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        logger.info(f"✅ Partially closed {size} contracts of {contract_id}")
                        return True
                    else:
                        error_msg = result.get("errorMessage", "Unknown error")
                        logger.error(f"Failed to partial close {contract_id}: {error_msg}")
                        return False
                else:
                    error = await response.text()
                    logger.error(f"Partial close failed ({response.status}): {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Partial close error: {e}")
            return False
    
    async def submit_order(self, account_id: int, contract_id: str, 
                          order_type: int, side: int, size: int) -> Dict:
        """
        Submit a simple order using exact API format
        
        Args:
            account_id: Account ID (e.g., 10983875)
            contract_id: Contract ID (e.g., "CON.F.US.ENQ.U25")
            order_type: 1=Limit, 2=Market, 4=Stop
            side: 0=Bid(Buy), 1=Ask(Sell)
            size: Number of contracts
        
        Returns:
            Dict with success status and order ID if successful
        """
        if not self.connected:
            await self.connect()
            
        try:
            order_data = {
                "accountId": account_id,
                "contractId": contract_id,
                "type": order_type,
                "side": side,
                "size": size
            }
            
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            logger.info(f"📤 Submitting order: {order_data}")
            
            async with self.session.post(
                f"{self.base_url}/Order/place",
                headers=headers,
                json=order_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        order_id = result.get("orderId")
                        logger.info(f"✅ Order placed! ID: {order_id}")
                        return {"success": True, "orderId": order_id, "response": result}
                    else:
                        error = result.get("errorMessage", "Unknown error")
                        logger.error(f"❌ Order failed: {error}")
                        return {"success": False, "error": error}
                else:
                    text = await response.text()
                    logger.error(f"❌ HTTP {response.status}: {text}")
                    return {"success": False, "error": f"HTTP {response.status}: {text}"}
                    
        except Exception as e:
            logger.error(f"❌ Order submission error: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_all_positions(self) -> bool:
        """Emergency close all open positions using TopStepX Position API"""
        if not self.connected:
            return False
        
        try:
            # Get the NQ contract ID
            contract_id = await self._get_contract_id("NQ")
            
            if contract_id:
                # Attempt to close the position for this contract
                success = await self.close_position(contract_id)
                if success:
                    logger.warning("🛑 All NQ positions closed (Emergency)")
                    return True
                else:
                    logger.warning("No open positions to close or close failed")
            
            return False
            
        except Exception as e:
            logger.error(f"Emergency close error: {e}")
            return False
    
    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        if not self.connected or not self.account_id:
            return []
        
        try:
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            request_data = {
                "accountId": self.account_id
            }
            
            async with self.session.post(
                f"{self.base_url}/Order/searchOpen",
                headers=headers,
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        orders = result.get("orders", [])
                        logger.info(f"Found {len(orders)} open orders")
                        return orders
                    else:
                        logger.error(f"Failed to get open orders: {result.get('errorMessage')}")
                        return []
                else:
                    error = await response.text()
                    logger.error(f"Get open orders failed ({response.status}): {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Get open orders error: {e}")
            return []
    
    async def search_trades(self, start_timestamp: datetime, end_timestamp: datetime = None) -> Dict:
        """Search for trades within a date range
        
        Args:
            start_timestamp: Start of the search period
            end_timestamp: End of the search period (optional)
            
        Returns:
            Dict containing trades list and metadata
        """
        if not self.connected or not self.account_id:
            return {"trades": [], "success": False, "errorMessage": "Not connected"}
        
        try:
            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            request_data = {
                "accountId": self.account_id,
                "startTimestamp": start_timestamp.isoformat() + "Z" if not start_timestamp.tzinfo else start_timestamp.isoformat(),
                "endTimestamp": end_timestamp.isoformat() + "Z" if end_timestamp else None
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            async with self.session.post(
                f"{self.base_url}/Trade/search",
                headers=headers,
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        trades = result.get("trades", [])
                        logger.info(f"Found {len(trades)} trades in search period")
                        return result
                    else:
                        error_msg = result.get("errorMessage", "Unknown error")
                        logger.error(f"Trade search failed: {error_msg}")
                        return result
                else:
                    error = await response.text()
                    logger.error(f"Trade search failed ({response.status}): {error}")
                    return {"trades": [], "success": False, "errorMessage": error}
                    
        except Exception as e:
            logger.error(f"Trade search error: {e}")
            return {"trades": [], "success": False, "errorMessage": str(e)}
    
    async def get_daily_trades(self) -> List[Dict]:
        """Get all trades for today"""
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.search_trades(today_start)
        return result.get("trades", [])
    
    async def calculate_daily_pnl(self) -> float:
        """Calculate today's total P&L from trades"""
        trades = await self.get_daily_trades()
        total_pnl = 0.0
        total_fees = 0.0
        
        for trade in trades:
            pnl = trade.get("profitAndLoss")
            if pnl is not None:  # None indicates half-turn trade
                total_pnl += float(pnl)
            fees = trade.get("fees", 0)
            total_fees += float(fees)
        
        net_pnl = total_pnl - total_fees
        logger.info(f"Daily P&L: ${net_pnl:.2f} (Gross: ${total_pnl:.2f}, Fees: ${total_fees:.2f})")
        return net_pnl
    
    async def get_trade_statistics(self, days: int = 7) -> Dict:
        """Get trade statistics for the specified number of days
        
        Args:
            days: Number of days to analyze (default: 7)
            
        Returns:
            Dict with statistics including win rate, avg win/loss, total P&L
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.search_trades(start_date)
        trades = result.get("trades", [])
        
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "total_fees": 0,
                "net_pnl": 0
            }
        
        wins = []
        losses = []
        total_fees = 0.0
        
        for trade in trades:
            pnl = trade.get("profitAndLoss")
            if pnl is not None:  # Skip half-turn trades
                pnl_value = float(pnl)
                if pnl_value > 0:
                    wins.append(pnl_value)
                elif pnl_value < 0:
                    losses.append(abs(pnl_value))
            
            fees = trade.get("fees", 0)
            total_fees += float(fees)
        
        total_trades = len(wins) + len(losses)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(wins) - sum(losses)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = (sum(wins) / sum(losses)) if losses else float('inf') if wins else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(total_pnl - total_fees, 2),
            "wins": len(wins),
            "losses": len(losses)
        }
    
    async def get_positions(self) -> List[Dict]:
        """Get all open positions
        
        Note: TopStepX may track positions differently than orders.
        This method returns open orders as a proxy for positions.
        """
        if not self.connected:
            return []
        
        # For TopStepX, positions might be tracked via open orders
        # We'll need to determine the exact endpoint for positions
        open_orders = await self.get_open_orders()
        
        # Convert orders to position format
        positions = []
        for order in open_orders:
            if order.get("filledPrice"):  # Only filled orders are positions
                positions.append({
                    "symbol": "NQ",  # Extract from contractId if needed
                    "side": "Buy" if order.get("side") == 0 else "Sell",
                    "quantity": order.get("size", 0),
                    "entry_price": order.get("filledPrice", 0),
                    "order_id": order.get("id")
                })
        
        self.open_positions = positions
        return positions
    
    async def get_account_info(self) -> Dict:
        """Get account balance and P&L"""
        if not self.connected:
            return {}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.session_token}",
                "Content-Type": "application/json"
            }
            
            # Use the account search endpoint to get account details
            async with self.session.post(
                f"{self.base_url}/Account/search",
                headers=headers,
                json={"onlyActiveAccounts": True}
            ) as response:
                if response.status == 200:
                    accounts = await response.json()
                    if accounts and len(accounts) > 0:
                        # Find the account matching our configured ID
                        for account in accounts:
                            if str(account.get("id")) == str(self.account_id):
                                self.account_balance = account.get("balance", 0)
                                self.daily_pnl = account.get("dailyPnL", 0)
                                return account
                        # If no match, return first account
                        account = accounts[0]
                        self.account_balance = account.get("balance", 0)
                        self.daily_pnl = account.get("dailyPnL", 0)
                        return account
                    else:
                        return {}
                else:
                    logger.error(f"Failed to get account info: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Get account error: {e}")
            return {}
    
    async def get_market_price(self, symbol: str = "NQ") -> float:
        """Get current market price for symbol using History API"""
        if not self.connected:
            return 0
        
        try:
            # Get the contract ID
            contract_id = await self._get_contract_id(symbol)
            if not contract_id:
                logger.error(f"No contract ID for {symbol}")
                return 0
            
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.session_token}"
            }
            
            # Get the last bar of data (1 minute)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            
            request_data = {
                "contractId": contract_id,
                "live": False,  # Use delayed data if available
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat(),
                "unit": 2,  # 2 = Minute
                "unitNumber": 1,  # 1 minute bars
                "limit": 1,  # Just get the last bar
                "includePartialBar": True
            }
            
            async with self.session.post(
                f"{self.base_url}/History/retrieveBars",
                headers=headers,
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    bars = result.get("bars", [])
                    if bars:
                        # Get the most recent bar
                        last_bar = bars[-1]
                        price = float(last_bar.get("c", 0))  # Close price
                        bid = float(last_bar.get("l", 0))  # Low as proxy for bid
                        ask = float(last_bar.get("h", 0))  # High as proxy for ask
                        logger.info(f"{symbol} Price: ${price:,.2f} (Bid: ${bid:,.2f}, Ask: ${ask:,.2f})")
                        
                        # Only update self values if this is NQ
                        if symbol.upper() == "NQ":
                            self.current_price = price
                            self.bid = bid
                            self.ask = ask
                        
                        return float(price)
                    else:
                        logger.warning(f"No price bars returned for {symbol}, trying quote...")
                        
                        # Try to get quote data as fallback with timeout
                        quote_url = f"{self.base_url}/Quote/retrieveQuote"
                        try:
                            quote_response = await asyncio.wait_for(
                                self.session.post(
                                    quote_url,
                                    headers=headers,
                                    json={"contractId": contract_id}
                                ),
                                timeout=5.0
                            )
                            
                            async with quote_response:
                                if quote_response.status == 200:
                                    quote_data = await quote_response.json()
                                    bid = float(quote_data.get("bid", 0))
                                    ask = float(quote_data.get("ask", 0))
                                    last = float(quote_data.get("last", 0))
                                    
                                    # Use mid-point if no last price
                                    price = last if last > 0 else (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                                    
                                    if price > 0:
                                        logger.info(f"{symbol} Quote Price: ${price:,.2f} (Bid: ${bid:,.2f}, Ask: ${ask:,.2f})")
                                        
                                        # Only update self values if this is NQ
                                        if symbol.upper() == "NQ":
                                            self.current_price = price
                                            self.bid = bid
                                            self.ask = ask
                                        
                                        return float(price)
                                    else:
                                        logger.warning(f"No quote available for {symbol}")
                        except asyncio.TimeoutError:
                            logger.warning(f"Quote timeout for {symbol}, trying historical bars...")
                            # Try historical bars as final fallback
                            return await self._get_price_from_bars(symbol, contract_id, headers)
                        except Exception as quote_error:
                            logger.error(f"Quote error for {symbol}: {quote_error}")
                else:
                    error_text = await response.text()
                    logger.error(f"Price request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Get price error: {e}")
        
        return 0
    
    async def _get_price_from_bars(self, symbol: str, contract_id: str, headers: dict) -> float:
        """Get price using historical bars as final fallback"""
        try:
            # Try different time ranges for bars
            end_time = datetime.now(timezone.utc)
            for minutes_back in [1, 5, 15, 30]:
                start_time = end_time - timedelta(minutes=minutes_back)
                
                request_data = {
                    "contractId": contract_id,
                    "live": True,
                    "startTime": start_time.isoformat(),
                    "endTime": end_time.isoformat(),
                    "unit": 2,  # Minute bars
                    "unitNumber": 1,
                    "limit": 10,
                    "includePartialBar": True
                }
                
                async with self.session.post(
                    f"{self.base_url}/History/retrieveBars",
                    headers=headers,
                    json=request_data,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        bars = result.get("bars", [])
                        if bars:
                            # Get the most recent bar
                            latest_bar = bars[-1]
                            price = float(latest_bar.get("c", 0))
                            if price > 0:
                                logger.info(f"{symbol} Historical Price: ${price:,.2f} (from {minutes_back}min ago)")
                                return price
            
            logger.warning(f"No historical bars available for {symbol}")
            return 0
            
        except Exception as e:
            logger.error(f"Historical bars error for {symbol}: {e}")
            return 0
    
    async def retrieve_bars(
        self,
        contract_id: str,
        start: Optional[pd.Timestamp] = None,
        *,
        unit: int = 2,            # 2 = Minute
        unit_number: int = 1,     # 1-minute
        limit: int = 200,
        include_partial: bool = True,
        live: bool = False        # False for practice/sim, True for live
    ) -> pd.DataFrame:
        """
        Retrieve bars with proper UTC timestamp handling for incremental updates
        """
        if not self.connected:
            await self.connect()
            
        try:
            # Calculate time range - API requires both startTime and endTime
            end_time = pd.Timestamp.now(tz="UTC")
            if start is not None:
                # Convert to pandas Timestamp first
                start_ts = pd.Timestamp(start)
                # Check if it already has timezone info
                if start_ts.tz is not None:
                    # Already has timezone, just convert to UTC
                    start_time = start_ts.tz_convert("UTC")
                else:
                    # No timezone, localize to UTC
                    start_time = start_ts.tz_localize("UTC")
            else:
                # Default lookback based on limit and unit
                if unit == 2:  # Minutes
                    start_time = end_time - pd.Timedelta(minutes=unit_number * limit)
                elif unit == 3:  # Hours
                    start_time = end_time - pd.Timedelta(hours=unit_number * limit)
                else:
                    start_time = end_time - pd.Timedelta(minutes=limit)
            
            payload = {
                "contractId": contract_id,
                "live": live,
                "unit": unit,
                "unitNumber": unit_number,
                "limit": limit,
                "includePartialBar": include_partial,
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat()
            }
            
            headers = {
                "Authorization": f"Bearer {self.session_token}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.base_url}/History/retrieveBars",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    res = await response.json()
                    bars = res.get("bars", [])
                    
                    if not bars:
                        return pd.DataFrame(columns=["open","high","low","close","volume"])
                    
                    df = pd.DataFrame(bars)
                    # Map t,o,h,l,c,v to standard columns
                    df = df.rename(columns={
                        "t": "timestamp",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                    })
                    
                    # Parse timestamp with UTC
                    idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df = df.assign(timestamp=idx).set_index("timestamp").sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                    df = df[~df.index.isna()]  # Remove NaT values
                    
                    return df[["open","high","low","close","volume"]]
                else:
                    error_text = await response.text()
                    logger.error(f"retrieve_bars failed: {response.status} - {error_text}")
                    return pd.DataFrame(columns=["open","high","low","close","volume"])
                    
        except Exception as e:
            logger.error(f"retrieve_bars error: {e}")
            return pd.DataFrame(columns=["open","high","low","close","volume"])
    
    async def get_historical_data(self, symbol: str = "NQ", interval: str = "1m", bars: int = 200) -> pd.DataFrame:
        """Get historical market data for pattern analysis
        
        Args:
            symbol: Symbol to get data for (default: NQ)
            interval: Time interval (1m, 5m, 15m, etc.)
            bars: Number of bars to retrieve
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Get contract ID for symbol
            contract_id = await self._get_contract_id(symbol)
            if not contract_id:
                logger.error(f"Could not find contract ID for {symbol}")
                return pd.DataFrame()
            
            # Parse interval to unit and unitNumber
            interval_map = {
                "1m": (2, 1),   # 1 minute
                "5m": (2, 5),   # 5 minutes
                "15m": (2, 15), # 15 minutes
                "30m": (2, 30), # 30 minutes
                "1h": (3, 1),   # 1 hour
                "4h": (3, 4),   # 4 hours
                "1d": (4, 1),   # 1 day
            }
            
            unit, unit_number = interval_map.get(interval, (2, 1))
            
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            # Calculate start time based on bars and interval
            if unit == 2:  # Minutes
                start_time = end_time - timedelta(minutes=unit_number * bars)
            elif unit == 3:  # Hours
                start_time = end_time - timedelta(hours=unit_number * bars)
            elif unit == 4:  # Days
                start_time = end_time - timedelta(days=unit_number * bars)
            else:
                start_time = end_time - timedelta(minutes=bars)
            
            # Make request
            headers = {
                "Authorization": f"Bearer {self.session_token}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "contractId": contract_id,
                "live": False,  # Use sim data for historical
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat(),
                "unit": unit,
                "unitNumber": unit_number,
                "limit": bars,
                "includePartialBar": False
            }
            
            async with self.session.post(
                f"{self.base_url}/History/retrieveBars",
                headers=headers,
                json=request_data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    bars_data = result.get("bars", [])
                    
                    if bars_data:
                        # Convert to DataFrame
                        df = pd.DataFrame(bars_data)
                        # Rename columns to match expected format
                        df = df.rename(columns={
                            't': 'timestamp',
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume'
                        })
                        
                        # Convert timestamp to datetime with UTC timezone
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                        
                        # Remove any rows with invalid timestamps
                        df = df[~df['timestamp'].isna()]
                        
                        # Sort by timestamp and set as index
                        df = df.sort_values('timestamp')
                        df = df.set_index('timestamp')
                        
                        # Remove duplicates keeping last
                        df = df[~df.index.duplicated(keep='last')]
                        
                        logger.info(f"Retrieved {len(df)} bars for {symbol}")
                        return df
                    else:
                        logger.warning(f"No historical data received for {symbol}")
                        return pd.DataFrame()
                else:
                    error_text = await response.text()
                    logger.error(f"Historical data request failed: {response.status} - {error_text}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def stream_market_data(self):
        """Stream real-time market data"""
        while self.connected:
            try:
                # Update market price every 30 seconds to avoid rate limits
                await self.get_market_price()
                await self.get_positions()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(5)
    
    def is_market_open(self) -> bool:
        """Check if futures market is open"""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        
        # Futures closed on Saturday
        if weekday == 5:  # Saturday
            return False
        
        # Get current time in CT (UTC-6 or UTC-5 for DST)
        ct_hour = (now.hour - 6) % 24
        
        # Sunday opens at 5 PM CT (23:00 UTC)
        if weekday == 6:  # Sunday
            return ct_hour >= 17
        
        # Friday closes at 4 PM CT (22:00 UTC)
        if weekday == 4:  # Friday
            return ct_hour < 16
        
        # Monday-Thursday: Check for daily break (4-5 PM CT)
        if 0 <= weekday <= 3:
            return ct_hour < 16 or ct_hour >= 17
        
        return True

# Global instance
topstepx_client = TopStepXClient()