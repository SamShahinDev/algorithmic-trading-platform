"""
TopStepX REST API Client.

This module provides a clean wrapper for TopStepX ProjectX Gateway API:
- Authentication with API key
- Account search and management
- Order placement (market, limit, stop)
- Order cancellation and modification
- Position queries
- Rate limit handling (200 requests/60 seconds)
"""

import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for TopStepX API."""
    LIMIT = 1
    MARKET = 2
    STOP_LIMIT = 3
    STOP = 4
    TRAILING_STOP = 5
    JOIN_BID = 6
    JOIN_ASK = 7


class OrderSide(Enum):
    """Order sides."""
    BID = 0  # Buy
    ASK = 1  # Sell


class OrderStatus(Enum):
    """Order status values."""
    NONE = 0
    OPEN = 1
    FILLED = 2
    CANCELLED = 3
    EXPIRED = 4
    REJECTED = 5
    PENDING = 6


class PositionType(Enum):
    """Position types."""
    UNDEFINED = 0
    LONG = 1
    SHORT = 2


class RateLimiter:
    """Simple rate limiter for TopStepX API."""

    def __init__(self, max_requests: int = 200, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window (200 for general endpoints)
            window_seconds: Time window in seconds (60 for general endpoints)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []  # List of request timestamps

    async def acquire(self) -> None:
        """Wait if necessary to acquire rate limit permission."""
        now = time.time()

        # Remove old requests outside the window
        self.requests = [ts for ts in self.requests if now - ts < self.window_seconds]

        # Check if we've hit the limit
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait
            oldest = self.requests[0]
            wait_time = self.window_seconds - (now - oldest) + 0.1  # Add small buffer

            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

            # Clean up again after waiting
            now = time.time()
            self.requests = [ts for ts in self.requests if now - ts < self.window_seconds]

        # Record this request
        self.requests.append(now)

    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limit statistics."""
        now = time.time()
        self.requests = [ts for ts in self.requests if now - ts < self.window_seconds]

        return {
            'count': len(self.requests),
            'limit': self.max_requests,
            'window': self.window_seconds,
            'percentage': (len(self.requests) / self.max_requests) * 100
        }


class TopStepXClient:
    """
    TopStepX REST API client.

    Provides clean interface for order execution and account management.
    """

    def __init__(self, api_key: str, username: str, account_id: str, environment: str = 'DEMO'):
        """
        Initialize TopStepX client.

        Args:
            api_key: TopStepX API key
            username: TopStepX username/email
            account_id: TopStepX account ID (e.g., PRAC-V2-XXXXX-XXXXXXXX)
            environment: Trading environment ('DEMO' or 'LIVE')
        """
        self.api_key = api_key
        self.username = username
        self.account_id = account_id
        self.environment = environment
        self.base_url = "https://api.topstepx.com/api"

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

        # Rate limiters
        self.general_limiter = RateLimiter(max_requests=200, window_seconds=60)
        self.historical_limiter = RateLimiter(max_requests=50, window_seconds=30)

        logger.info(f"TopStepX client initialized - Username: {username}, Account: {account_id}, Environment: {environment}")

    async def connect(self) -> bool:
        """
        Authenticate with TopStepX API.

        Returns:
            bool: True if successful
        """
        try:
            self.session = aiohttp.ClientSession()

            # Authenticate using API key
            auth_data = {
                "userName": self.username,
                "apiKey": self.api_key
            }

            headers = {
                "accept": "text/plain",
                "Content-Type": "application/json"
            }

            logger.info(f"Authenticating user: {self.username}")

            async with self.session.post(
                f"{self.base_url}/Auth/loginKey",
                json=auth_data,
                headers=headers
            ) as response:
                response_text = await response.text()
                logger.debug(f"Auth response status: {response.status}")
                logger.debug(f"Auth response body: {response_text}")

                if response.status == 200:
                    try:
                        result = await response.json()
                    except:
                        # Try parsing the text we already read
                        import json
                        result = json.loads(response_text)

                    if result.get('success'):
                        self.session_token = result.get('token')
                        self.token_expiry = datetime.now() + timedelta(hours=24)

                        logger.info("✅ Successfully authenticated")

                        # Get account information
                        accounts = await self.search_accounts(only_active=True)
                        if accounts and len(accounts) > 0:
                            self.account_id = accounts[0]['id']
                            logger.info(f"Using account ID: {self.account_id}")

                        return True
                    else:
                        logger.error(f"Authentication failed - Success: {result.get('success')}, ErrorCode: {result.get('errorCode')}, ErrorMessage: {result.get('errorMessage')}")
                        logger.error(f"Full response: {result}")
                        return False
                else:
                    logger.error(f"Authentication failed - HTTP {response.status}: {response_text}")
                    return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Close the API session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from TopStepX")

    async def validate_token(self) -> bool:
        """
        Validate session token.

        Returns:
            bool: True if token is valid
        """
        # Check expiry first
        if self.token_expiry and datetime.now() > self.token_expiry:
            logger.warning("Token expired, reconnecting...")
            return await self.connect()

        try:
            async with self.session.post(
                f"{self.base_url}/Auth/validate",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('success', False)
                return False
        except:
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth token."""
        return {
            "Authorization": f"Bearer {self.session_token}",
            "Content-Type": "application/json",
            "accept": "text/plain"
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        use_historical_limiter: bool = False
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request payload
            use_historical_limiter: Use historical data rate limiter

        Returns:
            API response
        """
        # Validate token - auto-reconnect if expired
        if not await self.validate_token():
            logger.warning("Session token expired - attempting to re-authenticate")
            try:
                await self.authenticate()
                logger.info("✅ Successfully re-authenticated")
            except Exception as e:
                logger.error(f"Failed to re-authenticate: {e}")
                raise Exception("Invalid or expired session token")

        # Apply rate limiting
        limiter = self.historical_limiter if use_historical_limiter else self.general_limiter
        await limiter.acquire()

        # Make request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            async with self.session.request(
                method,
                url,
                json=data,
                headers=self._get_headers()
            ) as response:
                result = await response.json()

                if response.status == 429:
                    logger.error("Rate limit exceeded (429)")
                    raise Exception("Rate limit exceeded")

                return result

        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    # Account Methods

    async def search_accounts(self, only_active: bool = True) -> List[Dict[str, Any]]:
        """
        Search for accounts.

        Args:
            only_active: Filter only active accounts

        Returns:
            List of account dictionaries
        """
        result = await self._request(
            'POST',
            '/Account/search',
            {'onlyActiveAccounts': only_active}
        )

        if result.get('success'):
            return result.get('accounts', [])
        else:
            logger.error(f"Failed to search accounts: {result.get('errorMessage')}")
            return []

    async def get_account_balance(self, account_id: int = None) -> Optional[float]:
        """
        Get account balance.

        Args:
            account_id: Account ID (uses default if None)

        Returns:
            Account balance or None
        """
        account_id = account_id or self.account_id
        accounts = await self.search_accounts(only_active=True)

        for account in accounts:
            if account['id'] == account_id:
                return account.get('balance')

        return None

    # Order Methods

    async def place_order(
        self,
        contract_id: str,
        side: OrderSide,
        size: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        account_id: Optional[int] = None,
        custom_tag: Optional[str] = None
    ) -> Optional[int]:
        """
        Place an order.

        Args:
            contract_id: Contract ID (e.g., "CON.F.US.ENQ.Z25")
            side: Order side (BID or ASK)
            size: Number of contracts
            order_type: Order type (MARKET, LIMIT, STOP)
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            account_id: Account ID (uses default if None)
            custom_tag: Optional custom tag

        Returns:
            Order ID or None if failed
        """
        account_id = account_id or self.account_id

        order_data = {
            'accountId': account_id,
            'contractId': contract_id,
            'type': order_type.value,
            'side': side.value,
            'size': size,
            'limitPrice': limit_price,
            'stopPrice': stop_price,
            'customTag': custom_tag
        }

        result = await self._request('POST', '/Order/place', order_data)

        if result.get('success'):
            order_id = result.get('orderId')
            logger.info(f"✅ Order placed: {order_id} - {side.name} {size} {contract_id}")
            return order_id
        else:
            logger.error(f"Failed to place order: {result.get('errorMessage')}")
            return None

    async def cancel_order(self, order_id: int, account_id: Optional[int] = None) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            account_id: Account ID (uses default if None)

        Returns:
            True if successful
        """
        account_id = account_id or self.account_id

        result = await self._request(
            'POST',
            '/Order/cancel',
            {'accountId': account_id, 'orderId': order_id}
        )

        if result.get('success'):
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
        else:
            logger.error(f"Failed to cancel order: {result.get('errorMessage')}")
            return False

    async def search_open_orders(self, account_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for open orders.

        Args:
            account_id: Account ID (uses default if None)

        Returns:
            List of open orders
        """
        account_id = account_id or self.account_id

        result = await self._request(
            'POST',
            '/Order/searchOpen',
            {'accountId': account_id}
        )

        if result.get('success'):
            return result.get('orders', [])
        else:
            return []

    # Position Methods

    async def search_open_positions(self, account_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for open positions.

        Args:
            account_id: Account ID (uses default if None)

        Returns:
            List of open positions
        """
        account_id = account_id or self.account_id

        result = await self._request(
            'POST',
            '/Position/searchOpen',
            {'accountId': account_id}
        )

        if result.get('success'):
            return result.get('positions', [])
        else:
            return []

    async def close_position(
        self,
        contract_id: str,
        account_id: Optional[int] = None
    ) -> bool:
        """
        Close entire position for contract.

        Args:
            contract_id: Contract ID
            account_id: Account ID (uses default if None)

        Returns:
            True if successful
        """
        account_id = account_id or self.account_id

        result = await self._request(
            'POST',
            '/Position/closeContract',
            {'accountId': account_id, 'contractId': contract_id}
        )

        if result.get('success'):
            logger.info(f"✅ Position closed: {contract_id}")
            return True
        else:
            logger.error(f"Failed to close position: {result.get('errorMessage')}")
            return False

    async def partial_close_position(
        self,
        contract_id: str,
        size: int,
        account_id: Optional[int] = None
    ) -> bool:
        """
        Partially close a position.

        Args:
            contract_id: Contract ID
            size: Number of contracts to close
            account_id: Account ID (uses default if None)

        Returns:
            True if successful
        """
        account_id = account_id or self.account_id

        result = await self._request(
            'POST',
            '/Position/partialCloseContract',
            {
                'accountId': account_id,
                'contractId': contract_id,
                'size': size
            }
        )

        if result.get('success'):
            logger.info(f"✅ Partial close: {size} contracts of {contract_id}")
            return True
        else:
            logger.error(f"Failed to partial close: {result.get('errorMessage')}")
            return False

    # Contract Methods

    async def search_contracts(self, search_text: str, live: bool = True) -> List[Dict[str, Any]]:
        """
        Search for contracts.

        Args:
            search_text: Search text (e.g., "NQ")
            live: Use live data (True) or sim data (False)

        Returns:
            List of matching contracts
        """
        result = await self._request(
            'POST',
            '/Contract/search',
            {'searchText': search_text, 'live': live}
        )

        if result.get('success'):
            return result.get('contracts', [])
        else:
            return []

    async def get_contract_by_id(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """
        Get contract details by ID.

        Args:
            contract_id: Contract ID

        Returns:
            Contract details or None
        """
        result = await self._request(
            'POST',
            '/Contract/searchById',
            {'contractId': contract_id}
        )

        if result.get('success'):
            return result.get('contract')
        else:
            return None

    async def retrieve_bars(
        self,
        contract_id: str,
        start_time: datetime,
        end_time: datetime,
        unit: int = 2,  # 2 = Minute
        unit_number: int = 1,
        limit: int = 100,
        include_partial_bar: bool = True,
        live: bool = False  # False for DEMO/practice, True for live
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical bars for a contract.

        Args:
            contract_id: Contract ID (e.g., "CON.F.US.ENQ.U25")
            start_time: Start time for bars
            end_time: End time for bars
            unit: Time unit (1=Second, 2=Minute, 3=Hour, 4=Day, 5=Week, 6=Month)
            unit_number: Number of units to aggregate
            limit: Maximum number of bars to retrieve (max 20,000)
            include_partial_bar: Whether to include current partial bar
            live: Use live data (True) or sim data (False) - should be False for DEMO accounts

        Returns:
            List of bar dictionaries with t, o, h, l, c, v fields
        """
        bar_data = {
            'contractId': contract_id,
            'live': live,
            'startTime': start_time.isoformat(),
            'endTime': end_time.isoformat(),
            'unit': unit,
            'unitNumber': unit_number,
            'limit': limit,
            'includePartialBar': include_partial_bar
        }

        logger.info(f"retrieve_bars request: contract={contract_id}, live={live}, start={start_time.isoformat()}, end={end_time.isoformat()}, limit={limit}")

        result = await self._request(
            'POST',
            '/History/retrieveBars',
            bar_data,
            use_historical_limiter=True
        )

        logger.debug(f"retrieve_bars full response: {result}")

        if result.get('success'):
            bars = result.get('bars', [])
            logger.info(f"Retrieved {len(bars)} bars for {contract_id}")
            return bars
        else:
            logger.error(f"Failed to retrieve bars - Success: {result.get('success')}, ErrorCode: {result.get('errorCode')}, ErrorMessage: {result.get('errorMessage')}")
            logger.error(f"Full response: {result}")
            return []

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get rate limit statistics.

        Returns:
            Dict with general and historical rate limit stats
        """
        return {
            'general': self.general_limiter.get_stats(),
            'historical': self.historical_limiter.get_stats()
        }
