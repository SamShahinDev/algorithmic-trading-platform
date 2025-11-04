"""
Realtime Position Sync - WebSocket + Polling Hybrid
Ensures positions stay synchronized in real-time
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class RealtimePositionSync:
    """
    WebSocket + polling hybrid approach for real-time position updates.
    WebSocket provides instant updates, polling provides reliability.
    """
    
    def __init__(self, position_manager, broker_client, config: Dict[str, Any] = None):
        self.position_manager = position_manager
        self.broker = broker_client
        self.config = config or {}
        
        # WebSocket state
        self.ws_connected = False
        self.ws_connection = None
        self.ws_reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Polling configuration
        self.polling_interval = self.config.get('polling_interval', 30)  # seconds
        self.fast_polling_interval = 5  # When WebSocket is down
        
        # Control flags
        self.running = False
        self.tasks = []
        
        # Metrics
        self.metrics = {
            'ws_connections': 0,
            'ws_disconnections': 0,
            'ws_messages_received': 0,
            'polling_syncs': 0,
            'last_ws_message': None,
            'last_poll': None
        }
    
    async def start(self):
        """Start both realtime and polling sync"""
        if self.running:
            logger.warning("Realtime sync already running")
            return
        
        self.running = True
        logger.info("Starting realtime position sync...")
        
        # Start WebSocket listener (if URL configured)
        if self.config.get('websocket_url'):
            self.tasks.append(asyncio.create_task(self._websocket_listener()))
        else:
            logger.info("No WebSocket URL configured, using polling only")
        
        # Always start polling as backup
        self.tasks.append(asyncio.create_task(self._polling_loop()))
        
        # Start health monitor
        self.tasks.append(asyncio.create_task(self._health_monitor()))
        
        logger.info("✅ Realtime position sync started")
    
    async def stop(self):
        """Stop all sync tasks"""
        logger.info("Stopping realtime position sync...")
        self.running = False
        
        # Close WebSocket if connected
        if self.ws_connection:
            await self.ws_connection.close()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        
        logger.info("Realtime position sync stopped")
    
    async def _websocket_listener(self):
        """Real-time position updates via WebSocket"""
        ws_url = self.config.get('websocket_url')
        
        while self.running:
            try:
                logger.info(f"Connecting to WebSocket: {ws_url}")
                
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connection = websocket
                    self.ws_connected = True
                    self.ws_reconnect_attempts = 0
                    self.metrics['ws_connections'] += 1
                    
                    logger.info("✅ WebSocket connected for position updates")
                    
                    # Subscribe to position updates
                    await self._subscribe_to_updates(websocket)
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            await self._handle_websocket_message(message)
                        except Exception as e:
                            logger.error(f"Error handling WebSocket message: {e}")
                    
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            finally:
                self.ws_connected = False
                self.ws_connection = None
                self.metrics['ws_disconnections'] += 1
                
                if self.running:
                    # Exponential backoff for reconnection
                    self.ws_reconnect_attempts += 1
                    
                    if self.ws_reconnect_attempts > self.max_reconnect_attempts:
                        logger.error(f"Max WebSocket reconnection attempts ({self.max_reconnect_attempts}) reached")
                        break
                    
                    wait_time = min(2 ** self.ws_reconnect_attempts, 60)
                    logger.info(f"Reconnecting WebSocket in {wait_time}s (attempt {self.ws_reconnect_attempts})")
                    await asyncio.sleep(wait_time)
    
    async def _subscribe_to_updates(self, websocket):
        """Subscribe to position and order updates"""
        # Subscribe message format depends on broker
        subscribe_msg = {
            'action': 'subscribe',
            'channels': ['positions', 'orders', 'fills']
        }
        
        await websocket.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to position updates")
    
    async def _handle_websocket_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            self.metrics['ws_messages_received'] += 1
            self.metrics['last_ws_message'] = datetime.now()
            
            msg_type = data.get('type', data.get('channel'))
            
            if msg_type in ['position_update', 'positions']:
                logger.info(f"Position update received: {data}")
                # Trigger immediate sync
                await self.position_manager.sync_with_broker("websocket_position")
                
            elif msg_type in ['order_update', 'orders']:
                logger.info(f"Order update received: {data}")
                # Trigger immediate sync
                await self.position_manager.sync_with_broker("websocket_order")
                
            elif msg_type in ['fill', 'fills', 'execution']:
                logger.info(f"Fill received: {data}")
                # Critical - position has changed
                await self.position_manager.sync_with_broker("websocket_fill")
                
            elif msg_type == 'heartbeat':
                # Just update last message time
                pass
                
            else:
                logger.debug(f"Unknown WebSocket message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid WebSocket message format: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _polling_loop(self):
        """Backup polling for reliability"""
        
        while self.running:
            try:
                # Use faster polling if WebSocket is down
                if self.ws_connected:
                    interval = self.polling_interval
                    source = "polling_backup"
                else:
                    interval = self.fast_polling_interval
                    source = "polling_primary"
                
                await asyncio.sleep(interval)
                
                if not self.running:
                    break
                
                # Sync with broker
                logger.debug(f"Polling sync (WebSocket {'up' if self.ws_connected else 'down'})")
                await self.position_manager.sync_with_broker(source)
                
                self.metrics['polling_syncs'] += 1
                self.metrics['last_poll'] = datetime.now()
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitor(self):
        """Monitor sync health and alert on issues"""
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.running:
                    break
                
                # Check WebSocket health
                if self.config.get('websocket_url') and not self.ws_connected:
                    ws_downtime = None
                    if self.metrics['last_ws_message']:
                        ws_downtime = (datetime.now() - self.metrics['last_ws_message']).total_seconds()
                    
                    if ws_downtime and ws_downtime > 300:  # 5 minutes
                        logger.warning(f"WebSocket down for {ws_downtime:.0f}s - relying on polling")
                
                # Check sync freshness
                sync_age = await self.position_manager.get_sync_age()
                if sync_age > 120:  # 2 minutes
                    logger.error(f"Position sync is stale: {sync_age:.0f}s old")
                
                # Check for phantom detections
                health = self.position_manager.get_health_status()
                if health['metrics']['phantom_detections'] > 0:
                    logger.warning(f"Phantom positions detected: {health['metrics']['phantom_detections']}")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def get_metrics(self) -> Dict:
        """Get sync metrics"""
        return {
            'websocket': {
                'connected': self.ws_connected,
                'connections': self.metrics['ws_connections'],
                'disconnections': self.metrics['ws_disconnections'],
                'messages': self.metrics['ws_messages_received'],
                'last_message': self.metrics['last_ws_message']
            },
            'polling': {
                'syncs': self.metrics['polling_syncs'],
                'last_poll': self.metrics['last_poll'],
                'interval': self.polling_interval if self.ws_connected else self.fast_polling_interval
            }
        }


class OrderReconciliation:
    """Track and reconcile bracket orders with positions"""
    
    def __init__(self, broker_client, position_manager):
        self.broker = broker_client
        self.position_manager = position_manager
        self.reconciliation_interval = 30  # seconds
        self.running = False
        
    async def start(self):
        """Start order reconciliation loop"""
        self.running = True
        asyncio.create_task(self._reconciliation_loop())
        logger.info("Order reconciliation started")
    
    async def stop(self):
        """Stop order reconciliation"""
        self.running = False
        logger.info("Order reconciliation stopped")
    
    async def _reconciliation_loop(self):
        """Continuous order reconciliation"""
        
        while self.running:
            try:
                await asyncio.sleep(self.reconciliation_interval)
                
                if not self.running:
                    break
                
                await self.reconcile_all_brackets()
                
            except Exception as e:
                logger.error(f"Order reconciliation error: {e}")
    
    async def reconcile_all_brackets(self):
        """Reconcile bracket orders for all positions"""
        
        # Get current positions
        positions = await self.position_manager.get_all_positions()
        orders = await self.position_manager.get_all_orders()
        
        for instrument, position in positions.items():
            await self.reconcile_brackets(instrument, position, orders)
        
        # Check for orphaned brackets (no position)
        await self._check_orphaned_brackets(positions, orders)
    
    async def reconcile_brackets(self, instrument: str, position: Dict, all_orders: Dict):
        """Ensure stops/targets match position"""
        
        if not position or position.get('quantity', 0) == 0:
            # No position = should have no brackets
            orphaned = self._find_brackets_for_instrument(instrument, all_orders)
            if orphaned:
                logger.warning(f"Found {len(orphaned)} orphaned brackets for {instrument}")
                await self._cancel_orders(orphaned)
            return
        
        # Position exists - verify protection
        qty = position['quantity']
        entry_price = position.get('average_price', 0)
        
        # Calculate expected stop/target
        if qty > 0:  # Long position
            expected_stop = entry_price - self._get_stop_distance(instrument)
            expected_target = entry_price + self._get_target_distance(instrument)
        else:  # Short position
            expected_stop = entry_price + self._get_stop_distance(instrument)
            expected_target = entry_price - self._get_target_distance(instrument)
        
        # Find actual orders
        brackets = self._find_brackets_for_instrument(instrument, all_orders)
        actual_stop = self._find_stop_order(brackets, qty)
        actual_target = self._find_target_order(brackets, qty)
        
        # Reconcile stop
        if not actual_stop:
            logger.error(f"MISSING STOP ORDER for {instrument} position {qty}")
            await self._place_emergency_stop(instrument, qty, expected_stop)
        elif abs(actual_stop['price'] - expected_stop) > self._get_price_tolerance(instrument):
            logger.warning(f"Stop price mismatch for {instrument}: {actual_stop['price']} vs {expected_stop}")
            await self._modify_stop(actual_stop, expected_stop)
        
        # Reconcile target
        if not actual_target:
            logger.warning(f"Missing target order for {instrument}")
            await self._place_target(instrument, qty, expected_target)
        elif abs(actual_target['price'] - expected_target) > self._get_price_tolerance(instrument):
            logger.warning(f"Target price mismatch for {instrument}: {actual_target['price']} vs {expected_target}")
            await self._modify_target(actual_target, expected_target)
    
    def _find_brackets_for_instrument(self, instrument: str, orders: Dict) -> List[Dict]:
        """Find bracket orders for specific instrument"""
        brackets = []
        for order in orders.values():
            if order.get('instrument') == instrument:
                if order.get('order_type') in ['stop', 'limit', 'stop_limit']:
                    brackets.append(order)
        return brackets
    
    def _find_stop_order(self, orders: List[Dict], position_qty: int) -> Optional[Dict]:
        """Find stop order protecting position"""
        for order in orders:
            if order.get('order_type') in ['stop', 'stop_limit']:
                # Stop order should be opposite side of position
                order_qty = order.get('quantity', 0)
                if (position_qty > 0 and order_qty < 0) or (position_qty < 0 and order_qty > 0):
                    return order
        return None
    
    def _find_target_order(self, orders: List[Dict], position_qty: int) -> Optional[Dict]:
        """Find target order for position"""
        for order in orders:
            if order.get('order_type') == 'limit':
                # Target order should be opposite side of position
                order_qty = order.get('quantity', 0)
                if (position_qty > 0 and order_qty < 0) or (position_qty < 0 and order_qty > 0):
                    return order
        return None
    
    async def _check_orphaned_brackets(self, positions: Dict, orders: Dict):
        """Check for brackets without positions"""
        
        instruments_with_positions = set(positions.keys())
        
        for order in orders.values():
            instrument = order.get('instrument')
            if instrument and instrument not in instruments_with_positions:
                if order.get('order_type') in ['stop', 'limit', 'stop_limit']:
                    logger.warning(f"Orphaned bracket order found: {order['order_id']} for {instrument}")
                    await self.broker.cancel_order(order['order_id'])
    
    async def _place_emergency_stop(self, instrument: str, qty: int, stop_price: float):
        """Place emergency stop order"""
        logger.critical(f"PLACING EMERGENCY STOP for {instrument}")
        
        try:
            order = await self.broker.place_stop_order(
                instrument=instrument,
                quantity=-qty,  # Opposite side to close
                stop_price=stop_price,
                text="EMERGENCY STOP"
            )
            
            if order:
                logger.info(f"Emergency stop placed: {order.get('order_id')}")
            else:
                logger.error("Failed to place emergency stop!")
                
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
    
    async def _place_target(self, instrument: str, qty: int, target_price: float):
        """Place target order"""
        try:
            order = await self.broker.place_limit_order(
                instrument=instrument,
                quantity=-qty,  # Opposite side to close
                limit_price=target_price,
                text="Target order"
            )
            
            if order:
                logger.info(f"Target placed: {order.get('order_id')}")
                
        except Exception as e:
            logger.error(f"Target order error: {e}")
    
    async def _modify_stop(self, stop_order: Dict, new_price: float):
        """Modify existing stop order"""
        try:
            await self.broker.modify_order(
                order_id=stop_order['order_id'],
                stop_price=new_price
            )
            logger.info(f"Stop modified to {new_price}")
            
        except Exception as e:
            logger.error(f"Stop modification error: {e}")
    
    async def _modify_target(self, target_order: Dict, new_price: float):
        """Modify existing target order"""
        try:
            await self.broker.modify_order(
                order_id=target_order['order_id'],
                limit_price=new_price
            )
            logger.info(f"Target modified to {new_price}")
            
        except Exception as e:
            logger.error(f"Target modification error: {e}")
    
    async def _cancel_orders(self, orders: List[Dict]):
        """Cancel multiple orders"""
        for order in orders:
            try:
                await self.broker.cancel_order(order['order_id'])
                logger.info(f"Cancelled order: {order['order_id']}")
            except Exception as e:
                logger.error(f"Cancel error for {order['order_id']}: {e}")
    
    def _get_stop_distance(self, instrument: str) -> float:
        """Get stop distance for instrument"""
        # This should come from config
        defaults = {
            'NQ': 20,  # 20 points
            'ES': 5,   # 5 points
            'CL': 0.50  # 50 cents
        }
        return defaults.get(instrument, 10)
    
    def _get_target_distance(self, instrument: str) -> float:
        """Get target distance for instrument"""
        defaults = {
            'NQ': 40,  # 40 points
            'ES': 10,  # 10 points
            'CL': 1.00  # $1
        }
        return defaults.get(instrument, 20)
    
    def _get_price_tolerance(self, instrument: str) -> float:
        """Get acceptable price tolerance"""
        defaults = {
            'NQ': 2,    # 2 points
            'ES': 0.5,  # 0.5 points
            'CL': 0.05  # 5 cents
        }
        return defaults.get(instrument, 1)