# File: /Users/royaltyvixion/Documents/XTRADING/trading_bot/position_state_manager.py

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class PositionStateManager:
    """
    Manages position state with broker reconciliation
    Features dynamic sync intervals for scalping and standardized field names
    """
    
    # Standard position schema used throughout the bot
    STANDARD_SCHEMA = {
        'entry': 'Entry price',
        'stop': 'Stop loss price',
        'target': 'Take profit price',
        'size': 'Number of contracts',
        'side': 'BUY or SELL',
        'pattern': 'Source pattern',
        'order_id': 'Broker order ID',
        'timestamp': 'Entry timestamp'
    }
    
    def __init__(self, broker, account_id: int):
        self.broker = broker
        self.account_id = account_id
        
        # Single source of truth
        self._position = None
        self._last_sync = None
        self._sync_failures = 0
        
        # Position details
        self.order_id = None
        self.entry_time = None
        self.pattern_source = None
        
        # Dynamic sync intervals for scalping
        self.sync_interval_in_position = 10  # 10s when holding position
        self.sync_interval_flat = 30  # 30s when flat
        self.sync_interval_new_position = 5  # 5s for first 30s after entry
        
    @property
    def has_position(self) -> bool:
        """Check if we have an active position"""
        return self._position is not None
    
    @property
    def position(self) -> Optional[Dict]:
        """Get current position with standard field names"""
        return self._position
    
    def get_sync_interval(self) -> int:
        """
        Get dynamic sync interval based on position state
        
        Returns:
            Sync interval in seconds
        """
        if not self.has_position:
            # No position - slower sync
            return self.sync_interval_flat
        
        # We have a position - check how old it is
        position_age = (datetime.now(timezone.utc) - self.entry_time).total_seconds()
        
        if position_age < 30:
            # New position - fastest sync for scalping
            return self.sync_interval_new_position
        else:
            # Established position - medium sync
            return self.sync_interval_in_position
    
    def should_sync(self) -> Tuple[bool, str]:
        """
        Check if we should sync based on dynamic intervals
        
        Returns:
            (should_sync, reason)
        """
        if not self._last_sync:
            return True, "No previous sync"
        
        time_since_sync = (datetime.now(timezone.utc) - self._last_sync).total_seconds()
        required_interval = self.get_sync_interval()
        
        if time_since_sync >= required_interval:
            if self.has_position:
                position_age = (datetime.now(timezone.utc) - self.entry_time).total_seconds()
                if position_age < 30:
                    return True, f"New position sync ({self.sync_interval_new_position}s interval)"
                else:
                    return True, f"Position sync ({self.sync_interval_in_position}s interval)"
            else:
                return True, f"Flat sync ({self.sync_interval_flat}s interval)"
        
        return False, f"Too soon (last sync {time_since_sync:.1f}s ago, need {required_interval}s)"
    
    async def sync_with_broker(self, force: bool = False) -> bool:
        """
        Sync with broker using dynamic intervals
        """
        try:
            # Check if we should sync
            should_sync, reason = self.should_sync()
            
            if not force and not should_sync:
                logger.debug(f"Skipping sync: {reason}")
                return True
            
            logger.debug(f"Syncing positions: {reason}")
            
            # Get positions from broker
            response = await self.broker.request('POST', '/api/Position/searchOpen', {
                "accountId": self.account_id
            })
            
            if not response or not response.get('success'):
                self._sync_failures += 1
                logger.warning(f"Position sync failed: {response}")
                return False
            
            broker_positions = response.get('positions', [])
            
            # Find NQ position
            nq_position = None
            for pos in broker_positions:
                contract_id = pos.get('contractId', '')
                if 'NQ' in contract_id or 'ENQ' in contract_id:
                    nq_position = pos
                    break
            
            # Smart position reconciliation
            if nq_position:
                # Broker has position - adopt it with standardized fields
                self._adopt_broker_position(nq_position)
                logger.info(f"Position confirmed: {self._position['size']} @ {self._position['entry']}")
                
            elif self._position and self.order_id:
                # We have position but broker doesn't
                position_age = (datetime.now(timezone.utc) - self.entry_time).total_seconds()
                
                if position_age < 5:
                    # Very new position - likely propagation delay
                    logger.debug(f"Position {self.order_id} not in broker yet ({position_age:.1f}s old)")
                    return True
                    
                elif position_age < 30:
                    # Recent position - verify order status
                    logger.warning(f"Position {self.order_id} missing from broker ({position_age:.1f}s old)")
                    if await self._verify_order_status():
                        logger.info("Order still active - keeping position")
                        return True
                    else:
                        logger.warning("Order not found - position likely closed")
                        self._clear_position()
                        
                else:
                    # Old phantom position
                    logger.error(f"Phantom position detected ({position_age:.1f}s old) - clearing")
                    self._clear_position()
                    
            else:
                # No position anywhere - we're flat
                if self._position:
                    logger.info("Position closed - now flat")
                self._position = None
            
            self._last_sync = datetime.now(timezone.utc)
            self._sync_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"Position sync error: {e}")
            self._sync_failures += 1
            return False
    
    def record_entry(self, position_data: Dict, order_id: str):
        """Record a new position entry with standardized field names"""
        # Convert to standard schema
        self._position = {
            'entry': position_data.get('entry'),
            'stop': position_data.get('stop'),
            'target': position_data.get('target'),
            'size': position_data.get('contracts'),
            'side': position_data.get('action'),
            'pattern': position_data.get('pattern'),
            'order_id': order_id,
            'timestamp': position_data.get('timestamp', datetime.now(timezone.utc)),
            'confidence': position_data.get('confidence')
        }
        
        self.order_id = order_id
        self.entry_time = datetime.now(timezone.utc)
        self.pattern_source = position_data.get('pattern')
        
        logger.info(f"Position entered: {self._position['size']} {self._position['side']} @ {self._position['entry']}")
        logger.info(f"Stop: {self._position['stop']}, Target: {self._position['target']}")
        logger.info(f"Using fast sync interval ({self.sync_interval_new_position}s) for new position")
    
    def _standardize_broker_position(self, broker_pos: Dict) -> Dict:
        """Convert broker format to standard schema"""
        # Broker uses different field names - standardize them
        side = 'BUY' if broker_pos.get('type') == 1 else 'SELL'
        
        standardized = {
            'entry': broker_pos.get('averagePrice', 0),
            'stop': None,  # Broker doesn't provide - will need to infer
            'target': None,  # Broker doesn't provide - will need to infer
            'size': broker_pos.get('size', 0),
            'side': side,
            'pattern': self.pattern_source,  # Use stored pattern source
            'order_id': broker_pos.get('id'),
            'timestamp': broker_pos.get('creationTimestamp'),
            'confidence': None  # Not available from broker
        }
        
        # Try to infer stop/target if we have pattern source
        if self._position and self._position.get('stop'):
            # Keep existing stop/target if we have them
            standardized['stop'] = self._position.get('stop')
            standardized['target'] = self._position.get('target')
        
        return standardized
    
    def _adopt_broker_position(self, broker_pos: Dict):
        """Adopt position from broker using standard schema"""
        self._position = self._standardize_broker_position(broker_pos)
        
        # Update entry time if available
        if broker_pos.get('creationTimestamp'):
            try:
                self.entry_time = datetime.fromisoformat(
                    broker_pos['creationTimestamp'].replace('Z', '+00:00')
                )
            except:
                self.entry_time = datetime.now(timezone.utc)
    
    def _clear_position(self):
        """Clear position state"""
        was_in_position = self.has_position
        
        self._position = None
        self.order_id = None
        self.entry_time = None
        self.pattern_source = None
        
        if was_in_position:
            logger.info("Position cleared - now flat")
            logger.info(f"Switching to flat sync interval ({self.sync_interval_flat}s)")
    
    async def _verify_order_status(self) -> bool:
        """Check if our order still exists"""
        try:
            # Check open orders
            response = await self.broker.request('POST', '/api/Order/searchOpen', {
                "accountId": self.account_id
            })
            
            if response and response.get('success'):
                orders = response.get('orders', [])
                for order in orders:
                    if str(order.get('id')) == str(self.order_id):
                        logger.debug(f"Order {self.order_id} still active")
                        return True
            
            # Check recent filled orders
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            
            response = await self.broker.request('POST', '/api/Order/search', {
                "accountId": self.account_id,
                "startTimestamp": start_time.isoformat(),
                "endTimestamp": end_time.isoformat()
            })
            
            if response and response.get('success'):
                orders = response.get('orders', [])
                for order in orders:
                    if str(order.get('id')) == str(self.order_id):
                        if order.get('status') == 2:  # Filled
                            logger.debug(f"Order {self.order_id} was filled")
                            return True
            
            logger.debug(f"Order {self.order_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Order verification error: {e}")
            return False
    
    def get_position_summary(self) -> str:
        """Get human-readable position summary"""
        if not self.has_position:
            return "No position"
        
        pos = self._position
        age = (datetime.now(timezone.utc) - self.entry_time).total_seconds() if self.entry_time else 0
        
        return (f"{pos['size']} {pos.get('side', 'UNKNOWN')} "
                f"@ {pos.get('entry', 0):.2f} "
                f"(age: {age:.0f}s)")