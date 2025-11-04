"""
Broker Outage Manager - Graceful degradation during broker unavailability
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class OutageLevel(Enum):
    """Outage severity levels"""
    NORMAL = "normal"
    BRIEF = "brief"          # < 30s
    EXTENDED = "extended"    # 30s - 5min
    CRITICAL = "critical"    # > 5min
    RECOVERED = "recovered"


class BrokerOutageManager:
    """
    Manages graceful degradation during broker outages.
    Protects positions and prevents phantom trades.
    """
    
    def __init__(self, position_manager, broker_client, alert_system=None):
        self.position_manager = position_manager
        self.broker = broker_client
        self.alert_system = alert_system
        
        # Outage tracking
        self.outage_start: Optional[datetime] = None
        self.outage_level = OutageLevel.NORMAL
        self.consecutive_failures = 0
        self.last_known_state = None
        
        # Emergency mode
        self.emergency_mode = False
        self.trading_suspended = False
        
        # Configuration
        self.brief_threshold = 30  # seconds
        self.extended_threshold = 300  # 5 minutes
        self.max_failures_before_outage = 3
        
        # Recovery
        self.recovery_in_progress = False
        self.recovery_attempts = 0
        
    async def handle_sync_failure(self, error: Exception):
        """
        React to broker sync failure.
        Implements escalating response based on outage duration.
        """
        
        self.consecutive_failures += 1
        
        # First failure - might be transient
        if self.consecutive_failures < self.max_failures_before_outage:
            logger.warning(f"Sync failure #{self.consecutive_failures}: {error}")
            return
        
        # Multiple failures - declare outage
        if not self.outage_start:
            await self._declare_outage()
        
        # Calculate outage duration
        outage_duration = (datetime.now() - self.outage_start).total_seconds()
        
        # Escalate response based on duration
        if outage_duration < self.brief_threshold:
            await self._handle_brief_outage(outage_duration)
            
        elif outage_duration < self.extended_threshold:
            await self._handle_extended_outage(outage_duration)
            
        else:
            await self._handle_critical_outage(outage_duration)
    
    async def _declare_outage(self):
        """Declare broker outage and snapshot state"""
        self.outage_start = datetime.now()
        self.outage_level = OutageLevel.BRIEF
        
        logger.error("=" * 80)
        logger.error("BROKER OUTAGE DETECTED")
        logger.error("=" * 80)
        
        # Snapshot current state
        self.last_known_state = await self._snapshot_current_state()
        
        # Alert
        if self.alert_system:
            await self.alert_system.broker_outage_started({
                'time': self.outage_start,
                'last_state': self.last_known_state
            })
    
    async def _snapshot_current_state(self) -> Dict:
        """Capture current position state"""
        try:
            positions = await self.position_manager.get_all_positions()
            orders = await self.position_manager.get_all_orders()
            
            return {
                'timestamp': datetime.now(),
                'positions': positions,
                'orders': orders,
                'sync_age': await self.position_manager.get_sync_age()
            }
        except Exception as e:
            logger.error(f"Failed to snapshot state: {e}")
            return {}
    
    async def _handle_brief_outage(self, duration: float):
        """Handle brief outage (<30s) - continue with cached state"""
        
        if self.outage_level != OutageLevel.BRIEF:
            return
        
        logger.warning(f"Brief outage ({duration:.0f}s) - using cached state")
        logger.warning("⚠️ Position data may be stale")
        
        # Continue trading but warn about stale data
        if self.last_known_state:
            sync_age = (datetime.now() - self.last_known_state['timestamp']).total_seconds()
            logger.warning(f"Using position data {sync_age:.0f}s old")
    
    async def _handle_extended_outage(self, duration: float):
        """Handle extended outage (30s-5min) - suspend new trades"""
        
        # Escalate level
        if self.outage_level == OutageLevel.BRIEF:
            self.outage_level = OutageLevel.EXTENDED
            logger.error(f"Outage escalated to EXTENDED ({duration:.0f}s)")
            
            # Suspend new trading
            self.trading_suspended = True
            self.emergency_mode = True
            
            logger.critical("=" * 80)
            logger.critical("TRADING SUSPENDED - BROKER OUTAGE")
            logger.critical(f"Duration: {duration:.0f}s")
            logger.critical("Only position management allowed")
            logger.critical("=" * 80)
            
            # Alert
            if self.alert_system:
                await self.alert_system.trading_suspended({
                    'duration': duration,
                    'level': 'extended'
                })
    
    async def _handle_critical_outage(self, duration: float):
        """Handle critical outage (>5min) - emergency shutdown"""
        
        # Escalate to critical
        if self.outage_level != OutageLevel.CRITICAL:
            self.outage_level = OutageLevel.CRITICAL
            
            logger.critical("=" * 80)
            logger.critical(f"CRITICAL BROKER OUTAGE - {duration:.0f}s")
            logger.critical("INITIATING EMERGENCY SHUTDOWN")
            logger.critical("=" * 80)
            
            # Try emergency flatten if possible
            await self._attempt_emergency_flatten()
            
            # Alert
            if self.alert_system:
                await self.alert_system.critical_outage({
                    'duration': duration,
                    'action': 'emergency_shutdown'
                })
            
            # Shutdown after delay
            logger.critical("System will shutdown in 10 seconds...")
            await asyncio.sleep(10)
            sys.exit(1)
    
    async def _attempt_emergency_flatten(self):
        """Attempt to flatten all positions in emergency"""
        logger.critical("Attempting emergency position flatten...")
        
        if not self.last_known_state:
            logger.error("No known positions to flatten")
            return
        
        positions = self.last_known_state.get('positions', {})
        
        for instrument, position in positions.items():
            qty = position.get('quantity', 0)
            if qty != 0:
                try:
                    logger.critical(f"Flattening {instrument}: {qty} contracts")
                    
                    # Try multiple times with short timeout
                    for attempt in range(3):
                        try:
                            order = await asyncio.wait_for(
                                self.broker.place_market_order(
                                    instrument=instrument,
                                    quantity=-qty,
                                    text="EMERGENCY FLATTEN"
                                ),
                                timeout=5
                            )
                            
                            if order:
                                logger.info(f"Emergency flatten order placed: {order.get('order_id')}")
                                break
                        except asyncio.TimeoutError:
                            logger.error(f"Flatten attempt {attempt + 1} timed out")
                        except Exception as e:
                            logger.error(f"Flatten attempt {attempt + 1} failed: {e}")
                        
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Failed to flatten {instrument}: {e}")
    
    async def handle_sync_success(self):
        """Handle successful sync - check for recovery"""
        
        # Reset failure counter
        self.consecutive_failures = 0
        
        # Check if we're recovering from outage
        if self.outage_start and not self.recovery_in_progress:
            await self.recover_from_outage()
    
    async def recover_from_outage(self):
        """Recovery sequence after broker returns"""
        
        if self.recovery_in_progress:
            return
        
        self.recovery_in_progress = True
        outage_duration = (datetime.now() - self.outage_start).total_seconds()
        
        logger.warning("=" * 80)
        logger.warning("BROKER CONNECTION RESTORED")
        logger.warning(f"Outage duration: {outage_duration:.0f}s")
        logger.warning("Starting recovery sequence...")
        logger.warning("=" * 80)
        
        try:
            # Step 1: Full reconciliation
            logger.info("1. Running full position reconciliation...")
            if not await self.position_manager.sync_with_broker("outage_recovery"):
                logger.error("Recovery sync failed - retrying...")
                await asyncio.sleep(2)
                if not await self.position_manager.sync_with_broker("outage_recovery_retry"):
                    raise Exception("Failed to sync after outage")
            
            # Step 2: Verify position integrity
            logger.info("2. Verifying position integrity...")
            await self._verify_position_integrity()
            
            # Step 3: Check for position changes during outage
            logger.info("3. Checking for position changes during outage...")
            await self._check_position_changes()
            
            # Step 4: Re-verify all protective orders
            logger.info("4. Verifying all bracket orders...")
            await self._verify_all_brackets()
            
            # Step 5: Resume trading
            logger.info("5. Resuming normal operations...")
            self.emergency_mode = False
            self.trading_suspended = False
            self.outage_level = OutageLevel.RECOVERED
            
            logger.info("=" * 80)
            logger.info("✅ RECOVERY COMPLETE - TRADING RESUMED")
            logger.info("=" * 80)
            
            # Alert
            if self.alert_system:
                await self.alert_system.broker_recovered({
                    'outage_duration': outage_duration,
                    'recovery_time': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self.recovery_attempts += 1
            
            if self.recovery_attempts < 3:
                logger.info(f"Retrying recovery (attempt {self.recovery_attempts})...")
                await asyncio.sleep(5)
                await self.recover_from_outage()
            else:
                logger.critical("Recovery failed after 3 attempts - manual intervention required")
                if self.alert_system:
                    await self.alert_system.recovery_failed({
                        'attempts': self.recovery_attempts,
                        'error': str(e)
                    })
        
        finally:
            self.recovery_in_progress = False
            self.recovery_attempts = 0
            self.outage_start = None
    
    async def _verify_position_integrity(self):
        """Verify positions are consistent after outage"""
        
        current_positions = await self.position_manager.get_all_positions()
        
        if not self.last_known_state:
            logger.warning("No pre-outage state to compare")
            return
        
        old_positions = self.last_known_state.get('positions', {})
        
        # Check for unexpected changes
        for instrument in set(list(current_positions.keys()) + list(old_positions.keys())):
            old_pos = old_positions.get(instrument, {})
            new_pos = current_positions.get(instrument, {})
            
            old_qty = old_pos.get('quantity', 0)
            new_qty = new_pos.get('quantity', 0)
            
            if old_qty != new_qty:
                logger.warning(f"Position change during outage - {instrument}: {old_qty} -> {new_qty}")
                
                # This could indicate fills during outage
                if self.alert_system:
                    await self.alert_system.position_change_during_outage({
                        'instrument': instrument,
                        'old_quantity': old_qty,
                        'new_quantity': new_qty
                    })
    
    async def _check_position_changes(self):
        """Check for any position changes during outage"""
        
        # Get recent fills from broker if available
        try:
            if hasattr(self.broker, 'get_recent_fills'):
                fills = await self.broker.get_recent_fills(
                    since=self.outage_start
                )
                
                if fills:
                    logger.warning(f"Found {len(fills)} fills during outage:")
                    for fill in fills:
                        logger.warning(f"  - {fill}")
                        
        except Exception as e:
            logger.error(f"Failed to check fills: {e}")
    
    async def _verify_all_brackets(self):
        """Verify all bracket orders are in place"""
        
        positions = await self.position_manager.get_all_positions()
        orders = await self.position_manager.get_all_orders()
        
        for instrument, position in positions.items():
            qty = position.get('quantity', 0)
            if qty != 0:
                # Check for stop order
                has_stop = any(
                    o.get('instrument') == instrument and 
                    o.get('order_type') in ['stop', 'stop_limit']
                    for o in orders.values()
                )
                
                if not has_stop:
                    logger.error(f"Missing stop order for {instrument} after recovery")
                    # OrderReconciliation will handle placing it
    
    def is_trading_allowed(self) -> bool:
        """Check if new trading is allowed"""
        return not self.trading_suspended and not self.emergency_mode
    
    def get_status(self) -> Dict:
        """Get current outage status"""
        status = {
            'level': self.outage_level.value,
            'trading_allowed': self.is_trading_allowed(),
            'emergency_mode': self.emergency_mode,
            'consecutive_failures': self.consecutive_failures
        }
        
        if self.outage_start:
            duration = (datetime.now() - self.outage_start).total_seconds()
            status['outage_duration'] = duration
            status['outage_start'] = self.outage_start
        
        return status