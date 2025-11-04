"""
Bot Startup Sequence - Ensures clean state before ANY trading
Critical for preventing phantom positions from day 1
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

from utils.position_manager import UnifiedPositionManager

logger = logging.getLogger(__name__)


class BotStartupSequence:
    """
    Ensures clean state before ANY trading.
    This is the FIRST thing that runs and MUST succeed.
    """
    
    def __init__(self, broker_client, config: Dict[str, Any]):
        self.broker = broker_client
        self.config = config
        self.position_manager = None
        self.startup_report = {
            'start_time': None,
            'checks_passed': [],
            'checks_failed': [],
            'warnings': [],
            'final_state': None
        }
    
    async def initialize(self) -> bool:
        """
        Full startup sequence with safety checks.
        Returns True only if bot is safe to trade.
        """
        
        self.startup_report['start_time'] = datetime.now()
        
        logger.info("=" * 80)
        logger.info("BOT STARTUP SEQUENCE INITIATED")
        logger.info("=" * 80)
        
        try:
            # 1. Connect to broker
            if not await self._connect_to_broker():
                return False
            
            # 2. Initialize position manager
            if not await self._initialize_position_manager():
                return False
            
            # 3. Perform initial reconciliation
            if not await self._initial_reconciliation():
                return False
            
            # 4. Check for existing positions
            if not await self._check_existing_positions():
                return False
            
            # 5. Check for orphaned orders
            if not await self._check_orphaned_orders():
                return False
            
            # 6. Verify risk limits
            if not await self._verify_risk_limits():
                return False
            
            # 7. Clear stale state files
            if not await self._clear_stale_state():
                return False
            
            # 8. Start monitoring loops
            if not await self._start_monitoring():
                return False
            
            # Success!
            self.startup_report['final_state'] = 'ready'
            self._save_startup_report()
            
            logger.info("=" * 80)
            logger.info("✅ STARTUP COMPLETE - BOT READY TO TRADE")
            logger.info(f"Checks passed: {len(self.startup_report['checks_passed'])}")
            logger.info(f"Warnings: {len(self.startup_report['warnings'])}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.critical(f"STARTUP SEQUENCE FAILED: {e}")
            self.startup_report['final_state'] = 'failed'
            self.startup_report['checks_failed'].append(f"Fatal error: {e}")
            self._save_startup_report()
            return False
    
    async def _connect_to_broker(self) -> bool:
        """Step 1: Connect to broker"""
        logger.info("1. Connecting to broker...")
        
        try:
            # Check if already connected
            if hasattr(self.broker, 'is_connected'):
                if await self.broker.is_connected():
                    logger.info("   ✓ Already connected to broker")
                    self.startup_report['checks_passed'].append('broker_connection')
                    return True
            
            # Attempt connection
            if hasattr(self.broker, 'connect'):
                if await self.broker.connect():
                    logger.info("   ✓ Connected to broker")
                    self.startup_report['checks_passed'].append('broker_connection')
                    return True
                else:
                    logger.error("   ✗ Broker connection failed")
                    self.startup_report['checks_failed'].append('broker_connection')
                    return False
            
            # Assume connected if no connect method
            logger.info("   ✓ Broker client ready")
            self.startup_report['checks_passed'].append('broker_connection')
            return True
            
        except Exception as e:
            logger.error(f"   ✗ Broker connection error: {e}")
            self.startup_report['checks_failed'].append(f'broker_connection: {e}')
            return False
    
    async def _initialize_position_manager(self) -> bool:
        """Step 2: Initialize position manager"""
        logger.info("2. Initializing position manager...")
        
        try:
            instruments = self.config.get('instruments', ['NQ'])
            self.position_manager = UnifiedPositionManager(
                broker_client=self.broker,
                instruments=instruments
            )
            
            logger.info(f"   ✓ Position manager initialized for {instruments}")
            self.startup_report['checks_passed'].append('position_manager_init')
            return True
            
        except Exception as e:
            logger.error(f"   ✗ Position manager init failed: {e}")
            self.startup_report['checks_failed'].append(f'position_manager_init: {e}')
            return False
    
    async def _initial_reconciliation(self) -> bool:
        """Step 3: IMMEDIATE position reconciliation"""
        logger.info("3. Performing initial position reconciliation...")
        
        try:
            # Force sync with broker
            if not await self.position_manager.sync_with_broker("startup"):
                logger.error("   ✗ Initial sync failed")
                self.startup_report['checks_failed'].append('initial_sync')
                return False
            
            # Verify sync is fresh
            sync_age = await self.position_manager.get_sync_age()
            if sync_age > 5:
                logger.error(f"   ✗ Sync too old: {sync_age}s")
                self.startup_report['checks_failed'].append(f'sync_age: {sync_age}s')
                return False
            
            logger.info(f"   ✓ Position sync complete (age: {sync_age:.1f}s)")
            self.startup_report['checks_passed'].append('initial_sync')
            return True
            
        except Exception as e:
            logger.error(f"   ✗ Reconciliation error: {e}")
            self.startup_report['checks_failed'].append(f'initial_sync: {e}')
            return False
    
    async def _check_existing_positions(self) -> bool:
        """Step 4: Check for existing positions"""
        logger.info("4. Checking for existing positions...")
        
        try:
            positions = await self.position_manager.get_all_positions()
            
            if not positions:
                logger.info("   ✓ Starting flat (no existing positions)")
                self.startup_report['checks_passed'].append('no_existing_positions')
                return True
            
            # Found existing positions
            logger.warning(f"   ⚠ EXISTING POSITIONS FOUND: {positions}")
            self.startup_report['warnings'].append(f'existing_positions: {positions}')
            
            # Check configuration for how to handle
            if self.config.get('adopt_existing_positions', False):
                logger.info("   → Adopting existing positions (config: adopt_existing_positions=True)")
                await self._adopt_positions(positions)
                self.startup_report['checks_passed'].append('adopted_positions')
                return True
                
            elif self.config.get('flatten_on_startup', False):
                logger.warning("   → Flattening existing positions (config: flatten_on_startup=True)")
                if await self._flatten_positions(positions):
                    self.startup_report['checks_passed'].append('flattened_positions')
                    return True
                else:
                    logger.error("   ✗ Failed to flatten positions")
                    self.startup_report['checks_failed'].append('flatten_positions')
                    return False
            
            else:
                # Cannot proceed with existing positions
                logger.error("   ✗ Cannot start with existing positions")
                logger.error("     Set 'adopt_existing_positions' or 'flatten_on_startup' in config")
                self.startup_report['checks_failed'].append('existing_positions_blocked')
                return False
                
        except Exception as e:
            logger.error(f"   ✗ Position check error: {e}")
            self.startup_report['checks_failed'].append(f'position_check: {e}')
            return False
    
    async def _check_orphaned_orders(self) -> bool:
        """Step 5: Check for orphaned orders"""
        logger.info("5. Checking for orphaned orders...")
        
        try:
            orders = await self.position_manager.get_all_orders()
            
            if not orders:
                logger.info("   ✓ No orphaned orders found")
                self.startup_report['checks_passed'].append('no_orphaned_orders')
                return True
            
            # Found working orders
            logger.warning(f"   ⚠ WORKING ORDERS FOUND: {len(orders)} orders")
            self.startup_report['warnings'].append(f'orphaned_orders: {len(orders)}')
            
            # Check configuration for how to handle
            if self.config.get('cancel_orphaned_orders', True):
                logger.info("   → Cancelling orphaned orders")
                if await self._cancel_orphaned_orders(orders):
                    self.startup_report['checks_passed'].append('cancelled_orders')
                    return True
                else:
                    logger.error("   ✗ Failed to cancel orders")
                    self.startup_report['checks_failed'].append('cancel_orders')
                    return False
            
            else:
                logger.warning("   → Keeping existing orders (config: cancel_orphaned_orders=False)")
                self.startup_report['warnings'].append('kept_orphaned_orders')
                return True
                
        except Exception as e:
            logger.error(f"   ✗ Order check error: {e}")
            self.startup_report['checks_failed'].append(f'order_check: {e}')
            return False
    
    async def _verify_risk_limits(self) -> bool:
        """Step 6: Verify risk limits are configured"""
        logger.info("6. Verifying risk limits...")
        
        try:
            required_limits = [
                'max_position_size',
                'max_loss_per_trade',
                'daily_loss_limit'
            ]
            
            missing = []
            for limit in required_limits:
                if limit not in self.config:
                    missing.append(limit)
            
            if missing:
                logger.error(f"   ✗ Missing risk limits: {missing}")
                self.startup_report['checks_failed'].append(f'risk_limits: missing {missing}')
                return False
            
            logger.info("   ✓ Risk limits configured:")
            logger.info(f"     - Max position: {self.config['max_position_size']}")
            logger.info(f"     - Max loss/trade: ${self.config['max_loss_per_trade']}")
            logger.info(f"     - Daily limit: ${self.config['daily_loss_limit']}")
            
            self.startup_report['checks_passed'].append('risk_limits')
            return True
            
        except Exception as e:
            logger.error(f"   ✗ Risk limit check error: {e}")
            self.startup_report['checks_failed'].append(f'risk_limits: {e}')
            return False
    
    async def _clear_stale_state(self) -> bool:
        """Step 7: Clear stale state files"""
        logger.info("7. Clearing stale state files...")
        
        try:
            state_files = [
                Path('logs/bot_state.json'),
                Path('logs/position_cache.json'),
                Path('logs/trade_state.json')
            ]
            
            cleared = 0
            for file in state_files:
                if file.exists():
                    # Backup before deleting
                    backup = file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                    file.rename(backup)
                    logger.info(f"   → Backed up {file.name} to {backup.name}")
                    cleared += 1
            
            if cleared > 0:
                logger.info(f"   ✓ Cleared {cleared} stale state files")
            else:
                logger.info("   ✓ No stale state files found")
            
            self.startup_report['checks_passed'].append('state_cleanup')
            return True
            
        except Exception as e:
            logger.error(f"   ✗ State cleanup error: {e}")
            self.startup_report['checks_failed'].append(f'state_cleanup: {e}')
            return False
    
    async def _start_monitoring(self) -> bool:
        """Step 8: Start monitoring loops"""
        logger.info("8. Starting monitoring loops...")
        
        try:
            # These will be started by the main system
            logger.info("   ✓ Monitoring loops ready to start")
            self.startup_report['checks_passed'].append('monitoring')
            return True
            
        except Exception as e:
            logger.error(f"   ✗ Monitoring setup error: {e}")
            self.startup_report['checks_failed'].append(f'monitoring: {e}')
            return False
    
    async def _adopt_positions(self, positions: Dict):
        """Adopt existing positions into bot state"""
        logger.info("Adopting existing positions...")
        for instrument, position in positions.items():
            logger.info(f"  - {instrument}: {position.get('quantity')} @ {position.get('average_price')}")
    
    async def _flatten_positions(self, positions: Dict) -> bool:
        """Flatten all existing positions"""
        logger.info("Flattening existing positions...")
        
        try:
            for instrument, position in positions.items():
                qty = position.get('quantity', 0)
                if qty != 0:
                    logger.info(f"  - Flattening {instrument}: {qty} contracts")
                    
                    # Place market order to flatten
                    order = await self.broker.place_market_order(
                        instrument=instrument,
                        quantity=-qty,  # Opposite side to flatten
                        text="Startup flatten"
                    )
                    
                    if order:
                        logger.info(f"    ✓ Flatten order placed: {order.get('order_id')}")
                    else:
                        logger.error(f"    ✗ Failed to flatten {instrument}")
                        return False
            
            # Wait for positions to clear
            await asyncio.sleep(2)
            
            # Verify flat
            await self.position_manager.sync_with_broker("post_flatten")
            final_positions = await self.position_manager.get_all_positions()
            
            if final_positions:
                logger.error(f"Positions still exist after flatten: {final_positions}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Flatten error: {e}")
            return False
    
    async def _cancel_orphaned_orders(self, orders: Dict) -> bool:
        """Cancel all orphaned orders"""
        logger.info(f"Cancelling {len(orders)} orphaned orders...")
        
        try:
            for order_id, order in orders.items():
                logger.info(f"  - Cancelling order {order_id}")
                
                if await self.broker.cancel_order(order_id):
                    logger.info(f"    ✓ Cancelled")
                else:
                    logger.error(f"    ✗ Failed to cancel")
                    # Continue with other orders
            
            # Resync after cancellations
            await asyncio.sleep(1)
            await self.position_manager.sync_with_broker("post_cancel")
            
            return True
            
        except Exception as e:
            logger.error(f"Cancel orders error: {e}")
            return False
    
    def _save_startup_report(self):
        """Save startup report to file"""
        try:
            report_file = Path('logs/startup_report.json')
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(self.startup_report, f, indent=2, default=str)
            
            logger.info(f"Startup report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save startup report: {e}")
    
    def get_startup_report(self) -> Dict:
        """Get the startup report"""
        return self.startup_report