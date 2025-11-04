# File: trading_bot/execution/position_tracker.py
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
import time

logger = logging.getLogger(__name__)

class PositionSource(Enum):
    """Source of position update"""
    BOT_ORDER = "bot_order"
    STOP_HIT = "stop_hit"
    MANUAL_TRADE = "manual_trade"
    BROKER_SYNC = "broker_sync"
    WEBSOCKET = "websocket"
    UNKNOWN = "unknown"

@dataclass
class PositionEvent:
    """Position change event"""
    timestamp: datetime
    source: PositionSource
    old_position: Optional[Dict]
    new_position: Optional[Dict]
    details: Dict

class PositionTracker:
    """
    Comprehensive position tracking through:
    1. Periodic polling (backup)
    2. WebSocket real-time updates (primary)
    3. Event-driven reconciliation
    """
    
    def __init__(self, bot, broker_client, account_id: int):
        self.bot = bot
        self.broker = broker_client
        self.account_id = account_id
        
        # State tracking
        self.last_known_position: Optional[Dict] = None
        self.last_sync_time: Optional[datetime] = None
        self.position_listeners: List[Callable] = []
        self.position_history: List[PositionEvent] = []
        
        # Configuration
        self.sync_interval_seconds = 30
        self.reconciliation_enabled = True
        self.max_sync_failures = 3
        self.sync_failures = 0
        
        # SignalR connection
        self.hub_connection = None
        self.websocket_connected = False
        self.reconnect_attempts = 0
        
        # Health metrics
        self.metrics = {
            'syncs_performed': 0,
            'websocket_updates': 0,
            'reconciliations': 0,
            'position_changes': 0,
            'sync_errors': 0
        }
    
    async def start(self):
        """Start all position tracking systems"""
        logger.info("Starting comprehensive position tracking...")
        
        try:
            # 1. Initial position sync
            await self.force_sync(PositionSource.BROKER_SYNC)
            
            # 2. Start periodic sync loop
            asyncio.create_task(self._periodic_sync_loop())
            
            # 3. Start WebSocket connection (disabled for now as SignalR needs specific library)
            # asyncio.create_task(self._maintain_websocket_connection())
            
            # 4. Start health monitoring
            asyncio.create_task(self._monitor_health())
            
            logger.info("Position tracking systems started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start position tracking: {e}")
            raise
    
    async def force_sync(self, source: PositionSource = PositionSource.BROKER_SYNC) -> bool:
        """Force immediate position synchronization"""
        try:
            logger.debug("Forcing position sync...")
            
            # Get positions from broker
            response = await self.broker.request('POST', '/api/Position/searchOpen', {
                "accountId": self.account_id
            })
            
            if not response or not response.get('success'):
                self.sync_failures += 1
                self.metrics['sync_errors'] += 1
                logger.error(f"Position sync failed: {response}")
                return False
            
            # Find NQ position
            positions = response.get('positions', [])
            nq_position = self._find_nq_position(positions)
            
            # Process the update
            await self._process_position_update(nq_position, source)
            
            # Update metrics
            self.last_sync_time = datetime.now()
            self.sync_failures = 0
            self.metrics['syncs_performed'] += 1
            
            return True
            
        except Exception as e:
            self.sync_failures += 1
            self.metrics['sync_errors'] += 1
            logger.error(f"Position sync exception: {e}")
            return False
    
    async def _periodic_sync_loop(self):
        """Backup sync loop - runs continuously"""
        await asyncio.sleep(10)  # Initial delay
        
        while True:
            try:
                # Check if we need emergency sync
                if self._needs_emergency_sync():
                    logger.warning("Emergency sync triggered")
                    await self.force_sync(PositionSource.BROKER_SYNC)
                
                # Regular periodic sync
                elif self._should_sync():
                    await self.force_sync(PositionSource.BROKER_SYNC)
                
                await asyncio.sleep(self.sync_interval_seconds)
                
            except Exception as e:
                logger.error(f"Periodic sync error: {e}")
                await asyncio.sleep(5)
    
    async def _process_position_update(self, broker_position: Optional[Dict], 
                                      source: PositionSource):
        """Process position update from any source"""
        try:
            # Validate position
            from .position_validator import PositionValidator
            
            if broker_position:
                is_valid, error = PositionValidator.is_valid_position(broker_position)
                if not is_valid:
                    logger.error(f"Invalid position received: {error}")
                    return
            
            # Normalize position data
            normalized_position = self._normalize_position(broker_position)
            
            # Check if position actually changed
            if self._has_position_changed(self.last_known_position, normalized_position):
                # Record event
                event = PositionEvent(
                    timestamp=datetime.now(),
                    source=source,
                    old_position=self.last_known_position,
                    new_position=normalized_position,
                    details={'raw_data': broker_position}
                )
                self.position_history.append(event)
                self.metrics['position_changes'] += 1
                
                # Log the change
                old_summary = self._position_summary(self.last_known_position)
                new_summary = self._position_summary(normalized_position)
                logger.warning(f"📍 POSITION CHANGE [{source.value}]: {old_summary} → {new_summary}")
                
                # Update state
                old_position = self.last_known_position
                self.last_known_position = normalized_position
                
                # Reconcile with bot state
                if self.reconciliation_enabled:
                    await self._reconcile_bot_state(old_position, normalized_position, source)
                
                # Notify listeners
                await self._notify_listeners(old_position, normalized_position, source)
                
        except Exception as e:
            logger.error(f"Error processing position update: {e}")
    
    async def _reconcile_bot_state(self, old_pos: Optional[Dict], 
                                  new_pos: Optional[Dict], 
                                  source: PositionSource):
        """Reconcile bot's internal state with broker reality"""
        self.metrics['reconciliations'] += 1
        
        bot_has_position = self.bot.current_position is not None
        broker_has_position = new_pos is not None
        
        # Case 1: States match - verify details
        if bot_has_position == broker_has_position:
            if broker_has_position:
                # Verify sizes match
                bot_size = self.bot.current_position_size
                broker_size = abs(new_pos['size'])
                
                if bot_size != broker_size:
                    logger.error(f"🔴 SIZE MISMATCH: Bot={bot_size}, Broker={broker_size}")
                    await self._adopt_broker_position(new_pos, source)
            return
        
        # Case 2: Bot has position, broker doesn't (phantom position)
        if bot_has_position and not broker_has_position:
            logger.error("🔴 PHANTOM POSITION DETECTED - Bot has position, broker doesn't")
            
            # Clear bot position
            self.bot.current_position = None
            self.bot.current_position_size = 0
            self.bot.current_position_type = None
            from ..intelligent_trading_bot_fixed_v2 import BotState
            self.bot.state = BotState.READY
            
            # Record phantom close
            if old_pos and hasattr(self.bot, 'direction_lockout'):
                self.bot.direction_lockout.record_exit("phantom_position_cleared")
            
            return
        
        # Case 3: Broker has position, bot doesn't (unknown position)
        if not bot_has_position and broker_has_position:
            logger.error("🔴 UNKNOWN POSITION DETECTED - Broker has position, bot doesn't")
            
            # Determine source
            if source == PositionSource.WEBSOCKET:
                # Real-time detection - likely external trade
                logger.warning("Position likely opened externally (manual trade?)")
            
            # Adopt the position
            await self._adopt_broker_position(new_pos, source)
    
    async def _adopt_broker_position(self, broker_pos: Dict, source: PositionSource):
        """Make bot adopt broker's position"""
        from ..intelligent_trading_bot_fixed_v2 import Position
        
        logger.warning(f"🔄 ADOPTING BROKER POSITION: {self._position_summary(broker_pos)}")
        
        # Create position object
        self.bot.current_position = Position(
            symbol=self.bot.symbol,
            side=0 if broker_pos['side'] == 'LONG' else 1,
            position_type=1 if broker_pos['side'] == 'LONG' else 2,
            size=abs(broker_pos['size']),
            entry_price=broker_pos['avg_price'],
            entry_time=datetime.now(),
            stop_loss=broker_pos['avg_price'] - 20 if broker_pos['side'] == 'LONG' else broker_pos['avg_price'] + 20,
            take_profit=broker_pos['avg_price'] + 20 if broker_pos['side'] == 'LONG' else broker_pos['avg_price'] - 20,
            pattern=None,
            confidence=0,
            order_id=str(broker_pos.get('id', 'adopted'))
        )
        
        # Update bot state
        self.bot.current_position_size = abs(broker_pos['size'])
        self.bot.current_position_type = 1 if broker_pos['side'] == 'LONG' else 2
        from ..intelligent_trading_bot_fixed_v2 import BotState
        self.bot.state = BotState.POSITION_OPEN
        
        # Log adoption
        logger.info(f"✅ Position adopted from source: {source.value}")
    
    def _normalize_position(self, broker_pos: Optional[Dict]) -> Optional[Dict]:
        """Convert broker position to standard format"""
        if not broker_pos or broker_pos.get('size', 0) == 0:
            return None
        
        return {
            'size': broker_pos.get('size', 0),
            'side': 'LONG' if broker_pos.get('type') == 1 else 'SHORT',
            'avg_price': broker_pos.get('averagePrice', 0),
            'contract_id': broker_pos.get('contractId', ''),
            'id': broker_pos.get('id'),
            'timestamp': datetime.now(),
            'raw': broker_pos
        }
    
    def _has_position_changed(self, old_pos: Optional[Dict], new_pos: Optional[Dict]) -> bool:
        """Check if position actually changed"""
        # One is None, other isn't
        if (old_pos is None) != (new_pos is None):
            return True
        
        # Both None
        if old_pos is None and new_pos is None:
            return False
        
        # Compare key fields
        return (old_pos['size'] != new_pos['size'] or 
                old_pos['side'] != new_pos['side'] or
                abs(old_pos['avg_price'] - new_pos['avg_price']) > 0.01)
    
    def _find_nq_position(self, positions: List[Dict]) -> Optional[Dict]:
        """Find NQ position from list"""
        for pos in positions:
            if self._is_nq_contract(pos.get('contractId', '')):
                return pos
        return None
    
    def _is_nq_contract(self, contract_id: str) -> bool:
        """Check if contract is NQ"""
        return any(symbol in contract_id for symbol in ['NQ', 'ENQ', 'MNQ'])
    
    def _position_summary(self, pos: Optional[Dict]) -> str:
        """Human-readable position summary"""
        if not pos:
            return "FLAT"
        return f"{abs(pos['size'])} {pos['side']} @ {pos['avg_price']:.2f}"
    
    def _should_sync(self) -> bool:
        """Check if periodic sync is needed"""
        if not self.last_sync_time:
            return True
        
        age = (datetime.now() - self.last_sync_time).total_seconds()
        return age >= self.sync_interval_seconds
    
    def _needs_emergency_sync(self) -> bool:
        """Check if emergency sync is needed"""
        # Too many sync failures
        if self.sync_failures >= self.max_sync_failures:
            return True
        
        # No sync for too long
        if self.last_sync_time:
            age = (datetime.now() - self.last_sync_time).total_seconds()
            if age > self.sync_interval_seconds * 3:  # 3x normal interval
                return True
        
        return False
    
    async def _notify_listeners(self, old_pos: Optional[Dict], 
                               new_pos: Optional[Dict], 
                               source: PositionSource):
        """Notify registered listeners of position change"""
        for listener in self.position_listeners:
            try:
                await listener(old_pos, new_pos, source)
            except Exception as e:
                logger.error(f"Position listener error: {e}")
    
    async def _monitor_health(self):
        """Monitor position tracking health"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Log health metrics
                sync_age = (datetime.now() - self.last_sync_time).total_seconds() if self.last_sync_time else -1
                
                logger.info(f"""
                === Position Tracking Health ===
                WebSocket Connected: {self.websocket_connected}
                Last Sync: {sync_age:.0f}s ago
                Sync Failures: {self.sync_failures}
                Total Syncs: {self.metrics['syncs_performed']}
                Position Changes: {self.metrics['position_changes']}
                ==============================
                """)
                
                # Alert if unhealthy
                if sync_age > 300:  # 5 minutes
                    logger.error("⚠️ Position sync is stale!")
                    
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def add_listener(self, callback: Callable):
        """Add position change listener"""
        self.position_listeners.append(callback)
    
    def get_position_age(self) -> Optional[float]:
        """Get age of last position sync in seconds"""
        if not self.last_sync_time:
            return None
        return (datetime.now() - self.last_sync_time).total_seconds()
    
    def get_metrics(self) -> Dict:
        """Get tracking metrics"""
        return {
            **self.metrics,
            'websocket_connected': self.websocket_connected,
            'position_age': self.get_position_age(),
            'sync_failures': self.sync_failures,
            'history_length': len(self.position_history)
        }