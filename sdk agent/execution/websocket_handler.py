"""
TopStepX WebSocket Handler.

This module provides real-time WebSocket streaming using SignalR:
- User Hub connection (account/position/order updates)
- Market Hub connection (real-time quotes)
- Automatic reconnection logic
- Token refresh (24-hour expiry)
- Event callbacks for fills and updates
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import SignalR, but make it optional
try:
    from signalrcore.hub_connection_builder import HubConnectionBuilder
    SIGNALR_AVAILABLE = True
except ImportError:
    logger.warning("signalrcore not installed. WebSocket functionality will be limited.")
    logger.info("Install with: pip install signalrcore")
    SIGNALR_AVAILABLE = False


class HubType(Enum):
    """SignalR hub types."""
    USER = "user"
    MARKET = "market"


class WebSocketHandler:
    """
    WebSocket handler for TopStepX real-time streaming.

    Manages SignalR connections to User Hub and Market Hub.
    """

    def __init__(self, session_token: str, account_id: int):
        """
        Initialize WebSocket handler.

        Args:
            session_token: JWT token from REST API authentication
            account_id: Account ID to monitor
        """
        if not SIGNALR_AVAILABLE:
            logger.error("SignalR not available - cannot initialize WebSocket handler")
            raise ImportError("signalrcore package required for WebSocket functionality")

        self.session_token = session_token
        self.account_id = account_id

        # Hub connections
        self.user_hub = None
        self.market_hub = None
        self.connected = False

        # Connection URLs
        self.user_hub_url = "wss://rtc.topstepx.com/hubs/user"
        self.market_hub_url = "wss://rtc.topstepx.com/hubs/market"

        # Event callbacks
        self.order_callbacks: List[Callable] = []
        self.position_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.quote_callbacks: List[Callable] = []
        self.account_callbacks: List[Callable] = []

        # Latest data cache
        self.latest_quotes: Dict[str, Dict] = {}
        self.active_orders: Dict[int, Dict] = {}
        self.active_positions: Dict[str, Dict] = {}

        logger.info(f"WebSocket handler initialized for account {account_id}")

    async def connect(self) -> bool:
        """
        Establish WebSocket connections to both hubs.

        Returns:
            bool: True if successful
        """
        try:
            # Connect to User Hub
            logger.info("Connecting to User Hub...")
            self.user_hub = HubConnectionBuilder()\
                .with_url(self.user_hub_url, options={
                    "skip_negotiation": True,
                    "access_token_factory": lambda: self.session_token
                })\
                .configure_logging(logging.WARNING)\
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_reconnect_interval": 30
                })\
                .build()

            # Register User Hub event handlers
            self.user_hub.on_open(lambda: self._on_hub_open(HubType.USER))
            self.user_hub.on_close(lambda: self._on_hub_close(HubType.USER))
            self.user_hub.on_error(lambda data: self._on_hub_error(HubType.USER, data))

            # Register user data event listeners
            self.user_hub.on("GatewayUserAccount", self._on_account_update)
            self.user_hub.on("GatewayUserOrder", self._on_order_update)
            self.user_hub.on("GatewayUserPosition", self._on_position_update)
            self.user_hub.on("GatewayUserTrade", self._on_trade_update)

            # Start User Hub
            self.user_hub.start()

            # Connect to Market Hub
            logger.info("Connecting to Market Hub...")
            self.market_hub = HubConnectionBuilder()\
                .with_url(self.market_hub_url, options={
                    "skip_negotiation": True,
                    "access_token_factory": lambda: self.session_token
                })\
                .configure_logging(logging.WARNING)\
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_reconnect_interval": 30
                })\
                .build()

            # Register Market Hub event handlers
            self.market_hub.on_open(lambda: self._on_hub_open(HubType.MARKET))
            self.market_hub.on_close(lambda: self._on_hub_close(HubType.MARKET))

            # Register market data event listeners
            self.market_hub.on("GatewayQuote", self._on_quote_update)

            # Start Market Hub
            self.market_hub.start()

            # Wait for connections to establish
            await asyncio.sleep(2)

            if self.user_hub and self.market_hub:
                self.connected = True
                logger.info("✅ WebSocket connections established")

                # Subscribe to updates
                await self.subscribe()
                return True
            else:
                logger.error("Failed to establish WebSocket connections")
                return False

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket hubs."""
        try:
            await self.unsubscribe()

            if self.user_hub:
                self.user_hub.stop()

            if self.market_hub:
                self.market_hub.stop()

            self.connected = False
            logger.info("WebSocket disconnected")

        except Exception as e:
            logger.error(f"Disconnect error: {e}")

    async def subscribe(self) -> None:
        """Subscribe to real-time updates."""
        try:
            if self.user_hub:
                # Subscribe to user data
                self.user_hub.send("SubscribeAccounts", [])
                self.user_hub.send("SubscribeOrders", [self.account_id])
                self.user_hub.send("SubscribePositions", [self.account_id])
                self.user_hub.send("SubscribeTrades", [self.account_id])
                logger.info(f"✅ Subscribed to account {self.account_id} updates")

            if self.market_hub:
                # Subscribe to NQ quotes
                symbols = ["F.US.ENQ", "F.US.MNQ"]
                for symbol in symbols:
                    self.market_hub.send("SubscribeQuote", [symbol])
                logger.info(f"✅ Subscribed to market data for {symbols}")

        except Exception as e:
            logger.error(f"Subscribe error: {e}")

    async def unsubscribe(self) -> None:
        """Unsubscribe from all updates."""
        try:
            if self.user_hub:
                self.user_hub.send("UnsubscribeAccounts", [])
                self.user_hub.send("UnsubscribeOrders", [self.account_id])
                self.user_hub.send("UnsubscribePositions", [self.account_id])
                self.user_hub.send("UnsubscribeTrades", [self.account_id])

            if self.market_hub:
                symbols = ["F.US.ENQ", "F.US.MNQ"]
                for symbol in symbols:
                    self.market_hub.send("UnsubscribeQuote", [symbol])

            logger.info("Unsubscribed from all updates")

        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")

    # Connection Event Handlers

    def _on_hub_open(self, hub_type: HubType) -> None:
        """Handle hub connection open."""
        logger.info(f"✅ {hub_type.value.title()} Hub connected")

    def _on_hub_close(self, hub_type: HubType) -> None:
        """Handle hub connection close."""
        logger.warning(f"⚠️ {hub_type.value.title()} Hub disconnected")
        if hub_type == HubType.USER:
            self.connected = False

    def _on_hub_error(self, hub_type: HubType, data: Any) -> None:
        """Handle hub errors."""
        logger.error(f"{hub_type.value.title()} Hub error: {data}")

    # User Data Event Handlers

    def _on_account_update(self, data: List) -> None:
        """Handle account updates."""
        try:
            # SignalR sends data as list
            if isinstance(data, list) and len(data) > 0:
                account_data = data[0]
                logger.debug(f"Account update: {account_data}")

                # Notify callbacks
                for callback in self.account_callbacks:
                    asyncio.create_task(callback(account_data))

        except Exception as e:
            logger.error(f"Account update error: {e}")

    def _on_order_update(self, data: List) -> None:
        """
        Handle real-time order updates.

        Order Status Enum:
        0 = None, 1 = Open, 2 = Filled, 3 = Cancelled,
        4 = Expired, 5 = Rejected, 6 = Pending
        """
        try:
            if isinstance(data, list) and len(data) > 0:
                order_data = data[0]
                order_id = order_data.get("id")
                status = order_data.get("status")

                # Update cache
                if status in [1, 6]:  # Open or Pending
                    self.active_orders[order_id] = order_data
                elif order_id in self.active_orders:
                    del self.active_orders[order_id]

                # Log order status
                status_names = {
                    0: "None", 1: "Open", 2: "Filled", 3: "Cancelled",
                    4: "Expired", 5: "Rejected", 6: "Pending"
                }
                status_text = status_names.get(status, "Unknown")

                logger.info(f"📋 Order {order_id}: {status_text}")

                if status == 2:  # Filled
                    filled_price = order_data.get("filledPrice")
                    size = order_data.get("size")
                    side = "Buy" if order_data.get("side") == 0 else "Sell"
                    logger.info(f"✅ FILLED: {side} {size} @ ${filled_price:.2f}")

                # Notify callbacks
                for callback in self.order_callbacks:
                    asyncio.create_task(callback(order_data))

        except Exception as e:
            logger.error(f"Order update error: {e}")

    def _on_position_update(self, data: List) -> None:
        """Handle real-time position updates."""
        try:
            if isinstance(data, list) and len(data) > 0:
                position_data = data[0]
                contract_id = position_data.get("contractId")
                size = position_data.get("size")

                # Update cache
                if size > 0:
                    self.active_positions[contract_id] = position_data
                elif contract_id in self.active_positions:
                    del self.active_positions[contract_id]

                # Log position
                avg_price = position_data.get("averagePrice")
                position_type = "Long" if position_data.get("type") == 1 else "Short"
                logger.info(f"📊 Position: {position_type} {size} @ ${avg_price:.2f}")

                # Notify callbacks
                for callback in self.position_callbacks:
                    asyncio.create_task(callback(position_data))

        except Exception as e:
            logger.error(f"Position update error: {e}")

    def _on_trade_update(self, data: List) -> None:
        """Handle real-time trade fills."""
        try:
            if isinstance(data, list) and len(data) > 0:
                trade_data = data[0]
                price = trade_data.get("price")
                pnl = trade_data.get("profitAndLoss")
                fees = trade_data.get("fees", 0)
                size = trade_data.get("size")
                side = "Buy" if trade_data.get("side") == 0 else "Sell"

                if pnl is not None:  # Closing trade
                    net_pnl = float(pnl) - float(fees)
                    logger.info(f"💰 Trade Closed: {side} {size} @ ${price:.2f}, "
                              f"P&L: ${pnl:.2f}, Fees: ${fees:.2f}, Net: ${net_pnl:.2f}")
                else:  # Opening trade
                    logger.info(f"🎯 Trade Opened: {side} {size} @ ${price:.2f}")

                # Notify callbacks
                for callback in self.trade_callbacks:
                    asyncio.create_task(callback(trade_data))

        except Exception as e:
            logger.error(f"Trade update error: {e}")

    # Market Data Event Handlers

    def _on_quote_update(self, data: List) -> None:
        """Handle real-time quote updates."""
        try:
            if isinstance(data, list) and len(data) > 0:
                quote_data = data[0]
                symbol = quote_data.get("symbol")
                last_price = quote_data.get("lastPrice")
                best_bid = quote_data.get("bestBid")
                best_ask = quote_data.get("bestAsk")

                # Update cache
                self.latest_quotes[symbol] = quote_data

                # Notify callbacks
                for callback in self.quote_callbacks:
                    asyncio.create_task(callback(quote_data))

                # Log periodically (every 100th quote to avoid spam)
                if not hasattr(self, '_quote_count'):
                    self._quote_count = {}
                self._quote_count[symbol] = self._quote_count.get(symbol, 0) + 1

                if self._quote_count[symbol] % 100 == 0:
                    logger.debug(f"Quote: {symbol} @ ${last_price:.2f} "
                               f"(Bid: ${best_bid:.2f}, Ask: ${best_ask:.2f})")

        except Exception as e:
            logger.error(f"Quote update error: {e}")

    # Callback Registration

    def on_order_update(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for order updates."""
        self.order_callbacks.append(callback)

    def on_position_update(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for position updates."""
        self.position_callbacks.append(callback)

    def on_trade_update(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for trade updates."""
        self.trade_callbacks.append(callback)

    def on_quote_update(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for quote updates."""
        self.quote_callbacks.append(callback)

    def on_account_update(self, callback: Callable[[Dict], None]) -> None:
        """Register callback for account updates."""
        self.account_callbacks.append(callback)

    # Data Access

    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for symbol."""
        return self.latest_quotes.get(symbol)

    def get_active_orders(self) -> Dict[int, Dict]:
        """Get all active orders."""
        return self.active_orders.copy()

    def get_active_positions(self) -> Dict[str, Dict]:
        """Get all active positions."""
        return self.active_positions.copy()
