"""
Unified Position Manager - Single Source of Truth for Position State
Prevents phantom positions through broker-authoritative state management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Immutable position state at a point in time"""
    instrument: str
    quantity: int
    average_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    sync_id: int
    source: str


class UnifiedPositionManager:
    """
    Single authority for ALL position state.
    Broker is the ONLY source of truth.
    """
    
    def __init__(self, broker_client, instruments: List[str] = None):
        self.broker = broker_client
        self.instruments = instruments or ['NQ', 'ES', 'CL']
        
        # Broker is ONLY truth
        self._broker_state = {
            'positions': {},      # instrument -> position
            'orders': {},         # order_id -> order
            'last_sync': None,
            'sync_id': 0,
            'sync_source': None
        }
        
        # Lock for thread safety
        self._state_lock = asyncio.Lock()
        
        # Health tracking
        self.health = {
            'sync_failures': 0,
            'phantom_detections': 0,
            'last_phantom': None,
            'consecutive_failures': 0,
            'last_successful_sync': None
        }
        
        # State persistence
        self.state_file = Path('logs/position_state.json')
        self.state_file.parent.mkdir(exist_ok=True)
        
        # Audit log
        self.audit_file = Path('logs/position_audit.jsonl')
        
    async def get_position(self, instrument: str) -> Optional[Dict]:
        """Read-only access to positions"""
        async with self._state_lock:
            return self._broker_state['positions'].get(instrument)
    
    async def get_all_positions(self) -> Dict[str, Any]:
        """Get all positions"""
        async with self._state_lock:
            return self._broker_state['positions'].copy()
    
    async def get_all_orders(self) -> Dict[str, Any]:
        """Get all working orders"""
        async with self._state_lock:
            return self._broker_state['orders'].copy()
    
    async def get_sync_age(self) -> float:
        """Get seconds since last sync"""
        async with self._state_lock:
            if not self._broker_state['last_sync']:
                return float('inf')
            return (datetime.now() - self._broker_state['last_sync']).total_seconds()
    
    async def sync_with_broker(self, source: str = "scheduled") -> bool:
        """
        ONLY method that updates position state.
        This is the critical synchronization point.
        """
        async with self._state_lock:
            try:
                logger.info(f"Starting position sync from {source}")
                
                # 1. Get broker positions with retry
                positions = await self._fetch_broker_positions_with_retry()
                
                # 2. Get working orders with retry
                orders = await self._fetch_broker_orders_with_retry()
                
                # 3. Validate data
                valid_positions = self._validate_positions(positions)
                valid_orders = self._validate_orders(orders)
                
                # 4. Detect phantoms BEFORE update
                phantoms = self._detect_phantoms(valid_positions)
                if phantoms:
                    await self._handle_phantom_detection(phantoms, source)
                
                # 5. Save old state for comparison
                old_state = {
                    'positions': self._broker_state['positions'].copy(),
                    'orders': self._broker_state['orders'].copy(),
                    'sync_id': self._broker_state['sync_id']
                }
                
                # 6. Update state atomically
                self._broker_state = {
                    'positions': self._organize_by_instrument(valid_positions),
                    'orders': {o['order_id']: o for o in valid_orders if 'order_id' in o},
                    'last_sync': datetime.now(),
                    'sync_id': self._broker_state['sync_id'] + 1,
                    'sync_source': source
                }
                
                # 7. Log changes
                self._log_state_changes(old_state, self._broker_state, source)
                
                # 8. Persist to disk
                await self._persist_state()
                
                # 9. Update health metrics
                self.health['consecutive_failures'] = 0
                self.health['last_successful_sync'] = datetime.now()
                
                logger.info(f"✅ Position sync complete from {source} (sync_id: {self._broker_state['sync_id']})")
                return True
                
            except Exception as e:
                logger.error(f"Position sync failed from {source}: {e}")
                logger.error(traceback.format_exc())
                
                self.health['sync_failures'] += 1
                self.health['consecutive_failures'] += 1
                
                # Alert if multiple consecutive failures
                if self.health['consecutive_failures'] >= 3:
                    logger.critical(f"CRITICAL: {self.health['consecutive_failures']} consecutive sync failures!")
                
                return False
    
    async def _fetch_broker_positions_with_retry(self, max_retries: int = 3) -> List[Dict]:
        """Fetch positions from broker with retry logic"""
        for attempt in range(max_retries):
            try:
                # Get all positions for all instruments
                all_positions = []
                for instrument in self.instruments:
                    positions = await self.broker.get_positions(instrument)
                    if positions:
                        all_positions.extend(positions)
                
                return all_positions
                
            except Exception as e:
                logger.warning(f"Position fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return []
    
    async def _fetch_broker_orders_with_retry(self, max_retries: int = 3) -> List[Dict]:
        """Fetch orders from broker with retry logic"""
        for attempt in range(max_retries):
            try:
                orders = await self.broker.get_working_orders()
                return orders or []
                
            except Exception as e:
                logger.warning(f"Order fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        return []
    
    def _validate_positions(self, positions: List[Dict]) -> List[Dict]:
        """Validate and clean position data"""
        valid = []
        for pos in positions:
            # Skip if essential fields missing
            if not pos.get('instrument'):
                logger.warning(f"Position missing instrument: {pos}")
                continue
            
            # Ensure numeric fields
            try:
                pos['quantity'] = int(pos.get('quantity', 0))
                pos['average_price'] = float(pos.get('average_price', 0))
                
                # Only include non-flat positions
                if pos['quantity'] != 0:
                    valid.append(pos)
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid position data: {pos}, error: {e}")
        
        return valid
    
    def _validate_orders(self, orders: List[Dict]) -> List[Dict]:
        """Validate and clean order data"""
        valid = []
        for order in orders:
            # Skip if essential fields missing
            if not order.get('order_id'):
                logger.warning(f"Order missing order_id: {order}")
                continue
            
            # Ensure order is actually working
            if order.get('status') in ['filled', 'cancelled', 'rejected']:
                continue
            
            valid.append(order)
        
        return valid
    
    def _organize_by_instrument(self, positions: List[Dict]) -> Dict[str, Dict]:
        """Organize positions by instrument"""
        by_instrument = {}
        for pos in positions:
            instrument = pos['instrument']
            if instrument in by_instrument:
                logger.warning(f"Multiple positions for {instrument}, using latest")
            by_instrument[instrument] = pos
        return by_instrument
    
    def _detect_phantoms(self, broker_positions: List[Dict]) -> List[Dict]:
        """Detect phantom positions by comparing with current state"""
        phantoms = []
        
        # Check each instrument we track
        for instrument in self.instruments:
            current = self._broker_state['positions'].get(instrument)
            broker = next((p for p in broker_positions if p['instrument'] == instrument), None)
            
            # Phantom: We think we have position but broker doesn't
            if current and current.get('quantity', 0) != 0 and not broker:
                phantoms.append({
                    'instrument': instrument,
                    'bot_position': current,
                    'broker_position': None,
                    'type': 'bot_only'
                })
            
            # Mismatch: Position quantities don't match
            elif current and broker and current.get('quantity') != broker.get('quantity'):
                phantoms.append({
                    'instrument': instrument,
                    'bot_position': current,
                    'broker_position': broker,
                    'type': 'mismatch'
                })
        
        return phantoms
    
    async def _handle_phantom_detection(self, phantoms: List[Dict], source: str):
        """Handle detected phantom positions"""
        for phantom in phantoms:
            logger.critical("=" * 80)
            logger.critical(f"PHANTOM POSITION DETECTED - {phantom['type'].upper()}")
            logger.critical(f"Instrument: {phantom['instrument']}")
            logger.critical(f"Bot thinks: {phantom['bot_position']}")
            logger.critical(f"Broker has: {phantom['broker_position']}")
            logger.critical(f"Detection source: {source}")
            logger.critical("=" * 80)
            
            # Update health metrics
            self.health['phantom_detections'] += 1
            self.health['last_phantom'] = {
                'time': datetime.now().isoformat(),
                'details': phantom,
                'source': source
            }
            
            # Audit log
            self._write_audit_log({
                'event': 'phantom_detection',
                'phantom': phantom,
                'source': source,
                'timestamp': datetime.now().isoformat()
            })
    
    def _log_state_changes(self, old_state: Dict, new_state: Dict, source: str):
        """Log any state changes"""
        # Check position changes
        for instrument in self.instruments:
            old_pos = old_state['positions'].get(instrument)
            new_pos = new_state['positions'].get(instrument)
            
            if old_pos != new_pos:
                old_qty = old_pos.get('quantity', 0) if old_pos else 0
                new_qty = new_pos.get('quantity', 0) if new_pos else 0
                
                if old_qty != new_qty:
                    logger.info(f"Position change - {instrument}: {old_qty} -> {new_qty} (source: {source})")
        
        # Check order changes
        old_order_ids = set(old_state['orders'].keys())
        new_order_ids = set(new_state['orders'].keys())
        
        added = new_order_ids - old_order_ids
        removed = old_order_ids - new_order_ids
        
        if added:
            logger.info(f"New orders: {added} (source: {source})")
        if removed:
            logger.info(f"Removed orders: {removed} (source: {source})")
    
    async def _persist_state(self):
        """Persist current state to disk"""
        try:
            state_data = {
                'positions': self._broker_state['positions'],
                'orders': self._broker_state['orders'],
                'last_sync': self._broker_state['last_sync'].isoformat() if self._broker_state['last_sync'] else None,
                'sync_id': self._broker_state['sync_id'],
                'sync_source': self._broker_state['sync_source'],
                'health': self.health
            }
            
            # Write atomically
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            temp_file.replace(self.state_file)
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
    
    def _write_audit_log(self, event: Dict):
        """Write to audit log"""
        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(event, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    async def force_reconciliation(self, reason: str = "manual"):
        """Force immediate reconciliation"""
        logger.warning(f"Force reconciliation requested: {reason}")
        return await self.sync_with_broker(f"force_{reason}")
    
    def get_health_status(self) -> Dict:
        """Get current health status"""
        return {
            'healthy': self.health['consecutive_failures'] < 3,
            'metrics': self.health,
            'last_sync': self._broker_state['last_sync'],
            'sync_id': self._broker_state['sync_id']
        }