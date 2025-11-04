"""
TopStepX Real-Time SignalR Client
Handles real-time WebSocket connections for live market data and order updates
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
from signalrcore.hub_connection_builder import HubConnectionBuilder
from signalrcore.protocol.messagepack_protocol import MessagePackHubProtocol
from signalrcore.subject import Subject

logger = logging.getLogger(__name__)


class TopStepXRealTimeClient:
    """Real-time SignalR client for TopStepX market data and order updates"""
    
    def __init__(self, jwt_token: str, account_id: int):
        """Initialize the real-time client
        
        Args:
            jwt_token: Bearer token for authentication
            account_id: Account ID to monitor
        """
        self.jwt_token = jwt_token
        self.account_id = account_id
        self.user_hub = None
        self.market_hub = None
        self.connected = False
        
        # Callbacks for events
        self.order_callbacks = []
        self.position_callbacks = []
        self.trade_callbacks = []
        self.quote_callbacks = []
        self.account_callbacks = []
        
        # Latest data cache
        self.latest_quotes = {}
        self.active_orders = {}
        self.active_positions = {}
        
        # Connection URLs - WebSocket URLs for SignalR
        self.user_hub_url = f"wss://rtc.topstepx.com/hubs/user"
        self.market_hub_url = "wss://rtc.topstepx.com/hubs/market"
    
    async def connect(self) -> bool:
        """Establish SignalR connections to both hubs"""
        try:
            # Connect to User Hub
            logger.info("🔌 Connecting to TopStepX User Hub...")
            # WebSocket URL already has wss:// scheme
            user_url_with_token = self.user_hub_url
            self.user_hub = HubConnectionBuilder()\
                .with_url(user_url_with_token, options={
                    "skip_negotiation": True,
                    "access_token_factory": lambda: self.jwt_token
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
            self.user_hub.on_open(self.on_user_hub_open)
            self.user_hub.on_close(self.on_user_hub_close)
            self.user_hub.on_error(self.on_hub_error)
            
            # Register event listeners
            self.user_hub.on("GatewayUserAccount", self.on_account_update)
            self.user_hub.on("GatewayUserOrder", self.on_order_update)
            self.user_hub.on("GatewayUserPosition", self.on_position_update)
            self.user_hub.on("GatewayUserTrade", self.on_trade_update)
            
            # Start User Hub connection
            self.user_hub.start()
            
            # Connect to Market Hub
            logger.info("🔌 Connecting to TopStepX Market Hub...")
            self.market_hub = HubConnectionBuilder()\
                .with_url(self.market_hub_url)\
                .configure_logging(logging.WARNING)\
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_reconnect_interval": 30
                })\
                .build()
            
            # Register Market Hub event handlers
            self.market_hub.on_open(self.on_market_hub_open)
            self.market_hub.on_close(self.on_market_hub_close)
            
            # Register market event listeners
            self.market_hub.on("GatewayQuote", self.on_quote_update)
            self.market_hub.on("GatewayDepth", self.on_depth_update)
            self.market_hub.on("GatewayTrade", self.on_market_trade)
            
            # Start Market Hub connection
            self.market_hub.start()
            
            # Wait for connections to establish
            await asyncio.sleep(2)
            
            if self.user_hub and self.market_hub:
                self.connected = True
                logger.info("✅ Successfully connected to TopStepX real-time feeds")
                
                # Subscribe to updates
                await self.subscribe_to_updates()
                return True
            else:
                logger.error("Failed to establish SignalR connections")
                return False
                
        except Exception as e:
            logger.error(f"SignalR connection error: {e}")
            return False
    
    async def subscribe_to_updates(self):
        """Subscribe to real-time updates for the account"""
        try:
            if self.user_hub:
                # Subscribe to user data
                self.user_hub.send("SubscribeAccounts", [])
                self.user_hub.send("SubscribeOrders", [self.account_id])
                self.user_hub.send("SubscribePositions", [self.account_id])
                self.user_hub.send("SubscribeTrades", [self.account_id])
                logger.info(f"📊 Subscribed to account {self.account_id} updates")
            
            if self.market_hub:
                # Subscribe to NQ futures quotes
                symbols = ["F.US.ENQ", "F.US.MNQ"]  # E-mini and Micro NQ
                for symbol in symbols:
                    self.market_hub.send("SubscribeQuote", [symbol])
                    self.market_hub.send("SubscribeDepth", [symbol])
                logger.info(f"📈 Subscribed to market data for NQ futures")
                
        except Exception as e:
            logger.error(f"Subscription error: {e}")
    
    async def unsubscribe_from_updates(self):
        """Unsubscribe from all real-time updates"""
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
                    self.market_hub.send("UnsubscribeDepth", [symbol])
                    
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
    
    # Connection Event Handlers
    def on_user_hub_open(self):
        """Handle user hub connection open"""
        logger.info("✅ User Hub connected")
    
    def on_user_hub_close(self):
        """Handle user hub connection close"""
        logger.warning("⚠️ User Hub disconnected")
        self.connected = False
    
    def on_market_hub_open(self):
        """Handle market hub connection open"""
        logger.info("✅ Market Hub connected")
    
    def on_market_hub_close(self):
        """Handle market hub connection close"""
        logger.warning("⚠️ Market Hub disconnected")
    
    def on_hub_error(self, data):
        """Handle hub errors"""
        logger.error(f"Hub error: {data}")
    
    # User Data Event Handlers
    def on_account_update(self, data):
        """Handle account updates"""
        try:
            logger.info(f"Account Update: {data}")
            
            # Notify callbacks
            for callback in self.account_callbacks:
                asyncio.create_task(callback(data))
                
        except Exception as e:
            logger.error(f"Account update error: {e}")
    
    def on_order_update(self, data):
        """Handle real-time order updates"""
        try:
            order_id = data.get("id")
            status = data.get("status")
            
            # Update cache
            if status in [1, 6]:  # Open or Pending
                self.active_orders[order_id] = data
            elif order_id in self.active_orders:
                del self.active_orders[order_id]
            
            # Log order status
            status_map = {0: "None", 1: "Open", 2: "Filled", 3: "Cancelled", 
                         4: "Expired", 5: "Rejected", 6: "Pending"}
            status_text = status_map.get(status, "Unknown")
            
            logger.info(f"📋 Order {order_id} Status: {status_text}")
            
            if status == 2:  # Filled
                filled_price = data.get("filledPrice")
                size = data.get("size")
                side = "Buy" if data.get("side") == 0 else "Sell"
                logger.info(f"✅ Order FILLED: {side} {size} @ ${filled_price}")
            
            # Notify callbacks
            for callback in self.order_callbacks:
                asyncio.create_task(callback(data))
                
        except Exception as e:
            logger.error(f"Order update error: {e}")
    
    def on_position_update(self, data):
        """Handle real-time position updates"""
        try:
            position_id = data.get("id")
            contract_id = data.get("contractId")
            size = data.get("size")
            
            # Update cache
            if size > 0:
                self.active_positions[contract_id] = data
            elif contract_id in self.active_positions:
                del self.active_positions[contract_id]
            
            # Calculate unrealized P&L if we have quotes
            avg_price = data.get("averagePrice")
            position_type = data.get("type")  # 1=Long, 2=Short
            
            if contract_id in self.latest_quotes and avg_price:
                current_price = self.latest_quotes[contract_id].get("lastPrice", avg_price)
                
                if position_type == 1:  # Long
                    unrealized_pnl = (current_price - avg_price) * size * 20  # NQ=$20/point
                else:  # Short
                    unrealized_pnl = (avg_price - current_price) * size * 20
                
                data["unrealizedPnl"] = unrealized_pnl
                logger.info(f"📊 Position Update: {size} contracts @ ${avg_price:.2f}, "
                          f"Unrealized P&L: ${unrealized_pnl:.2f}")
            
            # Notify callbacks
            for callback in self.position_callbacks:
                asyncio.create_task(callback(data))
                
        except Exception as e:
            logger.error(f"Position update error: {e}")
    
    def on_trade_update(self, data):
        """Handle real-time trade fills"""
        try:
            trade_id = data.get("id")
            price = data.get("price")
            pnl = data.get("profitAndLoss")
            fees = data.get("fees", 0)
            size = data.get("size")
            side = "Buy" if data.get("side") == 0 else "Sell"
            
            if pnl is not None:  # Full turn (closing trade)
                net_pnl = float(pnl) - float(fees)
                logger.info(f"💰 Trade Closed: {side} {size} @ ${price:.2f}, "
                          f"P&L: ${pnl:.2f}, Fees: ${fees:.2f}, Net: ${net_pnl:.2f}")
            else:  # Half turn (opening trade)
                logger.info(f"🎯 Trade Opened: {side} {size} @ ${price:.2f}")
            
            # Notify callbacks
            for callback in self.trade_callbacks:
                asyncio.create_task(callback(data))
                
        except Exception as e:
            logger.error(f"Trade update error: {e}")
    
    # Market Data Event Handlers
    def on_quote_update(self, data):
        """Handle real-time quote updates"""
        try:
            symbol = data.get("symbol")
            last_price = data.get("lastPrice")
            bid = data.get("bestBid")
            ask = data.get("bestAsk")
            volume = data.get("volume")
            
            # Update cache
            self.latest_quotes[symbol] = data
            
            # Map symbol to contract IDs for position P&L
            symbol_to_contract = {
                "F.US.ENQ": "CON.F.US.ENQ.U25",
                "F.US.MNQ": "CON.F.US.MNQ.U25"
            }
            
            if symbol in symbol_to_contract:
                contract_id = symbol_to_contract[symbol]
                self.latest_quotes[contract_id] = data
            
            logger.debug(f"Quote: {symbol} Last: ${last_price:.2f} "
                        f"Bid: ${bid:.2f} Ask: ${ask:.2f} Vol: {volume}")
            
            # Update position P&L with new price
            if symbol in symbol_to_contract:
                contract_id = symbol_to_contract[symbol]
                if contract_id in self.active_positions:
                    position = self.active_positions[contract_id]
                    position_copy = position.copy()
                    self.on_position_update(position_copy)
            
            # Notify callbacks
            for callback in self.quote_callbacks:
                asyncio.create_task(callback(data))
                
        except Exception as e:
            logger.error(f"Quote update error: {e}")
    
    def on_depth_update(self, data):
        """Handle depth of market updates"""
        try:
            # DOM updates for order book
            dom_type = data.get("type")
            price = data.get("price")
            volume = data.get("volume")
            
            # Types: 1=Ask, 2=Bid, 3=BestAsk, 4=BestBid
            if dom_type in [3, 4]:  # Best bid/ask updates
                logger.debug(f"DOM Update: {'BestBid' if dom_type == 4 else 'BestAsk'} "
                           f"@ ${price:.2f} x {volume}")
                           
        except Exception as e:
            logger.error(f"Depth update error: {e}")
    
    def on_market_trade(self, data):
        """Handle market trade events"""
        try:
            symbol = data.get("symbolId")
            price = data.get("price")
            volume = data.get("volume")
            trade_type = data.get("type")  # 0=Buy, 1=Sell
            
            logger.debug(f"Market Trade: {symbol} {'Buy' if trade_type == 0 else 'Sell'} "
                        f"{volume} @ ${price:.2f}")
                        
        except Exception as e:
            logger.error(f"Market trade error: {e}")
    
    # Callback Registration
    def on_order(self, callback: Callable):
        """Register callback for order updates"""
        self.order_callbacks.append(callback)
    
    def on_position(self, callback: Callable):
        """Register callback for position updates"""
        self.position_callbacks.append(callback)
    
    def on_trade(self, callback: Callable):
        """Register callback for trade updates"""
        self.trade_callbacks.append(callback)
    
    def on_quote(self, callback: Callable):
        """Register callback for quote updates"""
        self.quote_callbacks.append(callback)
    
    def on_account(self, callback: Callable):
        """Register callback for account updates"""
        self.account_callbacks.append(callback)
    
    # Data Access Methods
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get the latest quote for a symbol"""
        return self.latest_quotes.get(symbol)
    
    def get_active_orders(self) -> Dict[int, Dict]:
        """Get all active orders"""
        return self.active_orders.copy()
    
    def get_active_positions(self) -> Dict[str, Dict]:
        """Get all active positions"""
        return self.active_positions.copy()
    
    # Connection Management
    async def disconnect(self):
        """Disconnect from SignalR hubs"""
        try:
            await self.unsubscribe_from_updates()
            
            if self.user_hub:
                self.user_hub.stop()
            
            if self.market_hub:
                self.market_hub.stop()
            
            self.connected = False
            logger.info("🔌 Disconnected from TopStepX real-time feeds")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    async def reconnect(self):
        """Reconnect to SignalR hubs"""
        await self.disconnect()
        await asyncio.sleep(2)
        return await self.connect()


# Example usage and testing
async def test_realtime_client():
    """Test the real-time client"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv('.env.topstepx')
    
    # Get credentials (would come from TopStepX client in production)
    jwt_token = "test_token"  # Would get from authentication
    account_id = 123  # Would get from account search
    
    # Create client
    client = TopStepXRealTimeClient(jwt_token, account_id)
    
    # Register callbacks
    async def on_order(data):
        print(f"Order Update: {data}")
    
    async def on_position(data):
        print(f"Position Update: {data}")
    
    async def on_quote(data):
        print(f"Quote: {data.get('symbol')} @ ${data.get('lastPrice')}")
    
    client.on_order(on_order)
    client.on_position(on_position)
    client.on_quote(on_quote)
    
    # Connect
    if await client.connect():
        print("Connected! Listening for updates...")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            await client.disconnect()
    else:
        print("Failed to connect")


if __name__ == "__main__":
    asyncio.run(test_realtime_client())