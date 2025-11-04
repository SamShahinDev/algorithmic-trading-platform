"""
Real-time Manual Detection - Layer 5: WebSocket/Streaming
Real-time detection with fallback to polling
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Any, AsyncIterator
from enum import Enum

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Available streaming types"""
    WEBSOCKET = "websocket"
    SSE = "sse"
    LONG_POLL = "long_poll"
    POLLING = "polling"


class RealtimeManualDetection:
    """WebSocket/streaming detection for instant manual exit detection"""
    
    def __init__(self, position_manager, broker_client):
        self.position_manager = position_manager
        self.broker = broker_client
        
        # Streaming state
        self.stream_client = None
        self.stream_type = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.fallback_to_polling = False
        
        # Control
        self.running = False
        
        # Callbacks
        self.manual_detection_callback = None
        
        # Metrics
        self.stream_metrics = {
            'messages_received': 0,
            'manual_detections': 0,
            'avg_latency_ms': 0,
            'connection_failures': 0,
            'stream_type': None
        }
    
    async def initialize_streaming(self) -> bool:
        """Try to establish streaming connection"""
        
        logger.info("Initializing real-time streaming...")
        
        # Try streaming methods in order of preference
        methods = [
            (StreamType.WEBSOCKET, self._connect_websocket),
            (StreamType.SSE, self._connect_sse),
            (StreamType.LONG_POLL, self._connect_long_poll)
        ]
        
        for stream_type, connect_func in methods:
            try:
                logger.info(f"Trying {stream_type.value} connection...")
                
                if await connect_func():
                    self.stream_type = stream_type
                    self.stream_metrics['stream_type'] = stream_type.value
                    logger.info(f"✅ Streaming connected via {stream_type.value}")
                    return True
                    
            except Exception as e:
                logger.warning(f"{stream_type.value} failed: {e}")
                self.stream_metrics['connection_failures'] += 1
        
        # Fallback to polling
        logger.warning("No streaming available - using fast polling")
        self.fallback_to_polling = True
        self.stream_type = StreamType.POLLING
        self.stream_metrics['stream_type'] = 'polling'
        return False
    
    async def _connect_websocket(self) -> bool:
        """Connect via WebSocket"""
        
        # Check if broker supports WebSocket
        if not hasattr(self.broker, 'connect_websocket'):
            return False
        
        try:
            # Connect to WebSocket
            self.stream_client = await self.broker.connect_websocket()
            
            if self.stream_client:
                # Subscribe to position updates
                await self._subscribe_to_position_updates()
                return True
                
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        
        return False
    
    async def _connect_sse(self) -> bool:
        """Connect via Server-Sent Events"""
        
        # Check if broker supports SSE
        if not hasattr(self.broker, 'connect_sse'):
            return False
        
        try:
            # Connect to SSE
            self.stream_client = await self.broker.connect_sse()
            return self.stream_client is not None
            
        except Exception as e:
            logger.error(f"SSE connection error: {e}")
        
        return False
    
    async def _connect_long_poll(self) -> bool:
        """Connect via long polling"""
        
        # Check if broker supports long polling
        if not hasattr(self.broker, 'long_poll'):
            return False
        
        try:
            # Setup long polling
            self.stream_client = self._create_long_poll_stream()
            return True
            
        except Exception as e:
            logger.error(f"Long poll setup error: {e}")
        
        return False
    
    async def _subscribe_to_position_updates(self):
        """Subscribe to position update events"""
        
        if self.stream_type == StreamType.WEBSOCKET and self.stream_client:
            subscribe_msg = {
                'action': 'subscribe',
                'channels': ['positions', 'fills', 'orders']
            }
            
            await self.stream_client.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to position updates")
    
    def _create_long_poll_stream(self) -> AsyncIterator:
        """Create async iterator for long polling"""
        
        async def poll_generator():
            while self.running:
                try:
                    # Long poll for updates
                    updates = await self.broker.long_poll(timeout=30)
                    
                    if updates:
                        for update in updates:
                            yield update
                            
                except Exception as e:
                    logger.error(f"Long poll error: {e}")
                    await asyncio.sleep(1)
        
        return poll_generator()
    
    async def start(self):
        """Start real-time detection"""
        
        self.running = True
        
        # Initialize streaming
        streaming_available = await self.initialize_streaming()
        
        if streaming_available:
            # Start stream listener
            asyncio.create_task(self.stream_listener())
        else:
            # Start fast polling
            asyncio.create_task(self.fast_polling_loop())
    
    async def stop(self):
        """Stop real-time detection"""
        
        self.running = False
        
        # Close stream
        if self.stream_client:
            try:
                if self.stream_type == StreamType.WEBSOCKET:
                    await self.stream_client.close()
            except:
                pass
    
    async def stream_listener(self):
        """Process real-time updates with reconnection"""
        
        while self.running and not self.fallback_to_polling:
            try:
                # Process stream events
                async for event in self.stream_client:
                    if not self.running:
                        break
                    
                    await self._process_stream_event(event)
                    
            except Exception as e:
                logger.error(f"Stream error: {e}")
                self.reconnect_attempts += 1
                self.stream_metrics['connection_failures'] += 1
                
                if self.reconnect_attempts > self.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached - falling back to polling")
                    self.fallback_to_polling = True
                    asyncio.create_task(self.fast_polling_loop())
                    break
                
                # Exponential backoff
                wait_time = min(300, 2 ** self.reconnect_attempts)
                logger.info(f"Reconnecting in {wait_time}s (attempt {self.reconnect_attempts})")
                await asyncio.sleep(wait_time)
                
                # Try to reconnect
                if await self.initialize_streaming():
                    self.reconnect_attempts = 0
                else:
                    break
    
    async def _process_stream_event(self, event: Dict):
        """Handle streaming position update"""
        
        # Track metrics
        self.stream_metrics['messages_received'] += 1
        
        # Parse event
        if isinstance(event, str):
            try:
                event = json.loads(event)
            except:
                return
        
        event_type = event.get('type', event.get('channel'))
        
        if event_type not in ['position_update', 'fill', 'order_update']:
            return
        
        # Get timing info
        detection_start = time.time()
        event_timestamp = event.get('timestamp')
        
        # Get states
        old_state = self.position_manager.get_current_state()
        new_state = event.get('data', {})
        
        # Check for manual intervention
        manual = await self._detect_manual_change(old_state, new_state, event_type)
        
        if manual:
            # Add latency info
            detection_latency = (time.time() - detection_start) * 1000
            manual['detection_latency_ms'] = detection_latency
            
            # Calculate stream latency if timestamp available
            if event_timestamp:
                try:
                    if isinstance(event_timestamp, str):
                        event_time = datetime.fromisoformat(event_timestamp)
                    else:
                        event_time = event_timestamp
                    
                    stream_latency = (datetime.now() - event_time).total_seconds() * 1000
                    manual['stream_latency_ms'] = stream_latency
                    
                    # Update average
                    self._update_latency_metric(stream_latency)
                    
                except:
                    pass
            
            # Track detection
            self.stream_metrics['manual_detections'] += 1
            
            # Handle immediately
            await self.handle_manual_intervention(manual)
        
        # Update position manager state
        self.position_manager.update_from_stream(new_state)
    
    async def _detect_manual_change(self, old_state: Dict, new_state: Dict, 
                                   event_type: str) -> Optional[Dict]:
        """Detect manual intervention from state change"""
        
        # Position disappeared
        if event_type == 'position_update':
            old_pos = old_state.get('position')
            new_pos = new_state.get('position')
            
            if old_pos and not new_pos:
                # Check if bot initiated
                if not self._is_bot_initiated():
                    return {
                        'type': 'manual_full_exit',
                        'method': 'realtime_stream',
                        'confidence': 0.98,
                        'position_before': old_pos,
                        'event_type': event_type,
                        'timestamp': datetime.now()
                    }
        
        # Unexpected fill
        elif event_type == 'fill':
            # Check if fill was expected
            if not self._is_expected_fill(new_state):
                return {
                    'type': 'manual_fill',
                    'method': 'realtime_stream',
                    'confidence': 0.90,
                    'fill_data': new_state,
                    'event_type': event_type,
                    'timestamp': datetime.now()
                }
        
        return None
    
    def _is_bot_initiated(self) -> bool:
        """Check if change was bot-initiated"""
        
        # Check bot flags
        if hasattr(self.position_manager, 'bot'):
            bot = self.position_manager.bot
            
            if getattr(bot, 'is_exiting', False):
                return True
            
            if getattr(bot, 'is_modifying', False):
                return True
            
            # Check recent bot actions
            if last_action := getattr(bot, 'last_action_time', None):
                elapsed = (datetime.now() - last_action).total_seconds()
                if elapsed < 5:  # Within 5 seconds
                    return True
        
        return False
    
    def _is_expected_fill(self, fill_data: Dict) -> bool:
        """Check if fill was expected by bot"""
        
        # This would check against expected fills
        # For now, simplified logic
        return False
    
    async def fast_polling_loop(self):
        """Fast polling fallback (5 second intervals)"""
        
        logger.info("Starting fast polling loop (5s intervals)")
        
        poll_interval = 5  # 5 seconds for fast polling
        last_state = None
        
        while self.running:
            try:
                # Get current state
                current_state = await self._get_current_broker_state()
                
                if last_state is not None:
                    # Check for manual intervention
                    manual = await self._detect_manual_change(
                        last_state, 
                        current_state,
                        'polling'
                    )
                    
                    if manual:
                        manual['detection_method'] = 'fast_polling'
                        await self.handle_manual_intervention(manual)
                
                last_state = current_state
                
            except Exception as e:
                logger.error(f"Fast polling error: {e}")
            
            await asyncio.sleep(poll_interval)
    
    async def _get_current_broker_state(self) -> Dict:
        """Get current broker state"""
        
        positions = await self.broker.get_open_positions()
        orders = await self.broker.get_working_orders()
        
        return {
            'position': positions[0] if positions else None,
            'orders': orders,
            'timestamp': datetime.now()
        }
    
    async def handle_manual_intervention(self, detection: Dict):
        """Handle detected manual intervention"""
        
        logger.warning(f"REALTIME MANUAL DETECTION: {detection['type']}")
        logger.info(f"Detection latency: {detection.get('detection_latency_ms', 0):.1f}ms")
        
        if stream_latency := detection.get('stream_latency_ms'):
            logger.info(f"Stream latency: {stream_latency:.1f}ms")
        
        # Call registered callback
        if self.manual_detection_callback:
            await self.manual_detection_callback(detection)
    
    def _update_latency_metric(self, latency_ms: float):
        """Update average latency metric"""
        
        # Simple moving average
        alpha = 0.1  # Smoothing factor
        
        if self.stream_metrics['avg_latency_ms'] == 0:
            self.stream_metrics['avg_latency_ms'] = latency_ms
        else:
            self.stream_metrics['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.stream_metrics['avg_latency_ms']
            )
    
    def set_manual_detection_callback(self, callback):
        """Set callback for manual detection"""
        self.manual_detection_callback = callback
    
    def get_metrics(self) -> Dict:
        """Get streaming metrics"""
        return self.stream_metrics.copy()