"""
Enhanced Exit Flow - Production-ready exit system with all safety features
Integrates all exit components for reliable, predictable exits
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from utils.exit_manager import ExitManager
from utils.pattern_memory import PatternMemory, TradingState
from utils.exit_labeler import ExitLabeler, FillTracker
from utils.dual_stop_manager import DualStopManager
from utils.time_sync_manager import TimeSyncManager
from utils.partial_fill_handler import PartialFillHandler
from utils.restart_recovery import RestartRecovery

logger = logging.getLogger(__name__)


# Feature flags for safe rollout
ENHANCED_EXIT_FEATURES = {
    # Core fixes
    "exit_priority_system": True,
    "idempotent_exits": True,
    "pattern_persistence": True,
    "honest_labeling": True,
    
    # Safety features
    "dual_stops": True,
    "safe_cancel_timeout": True,
    "tick_rounding": True,
    "time_sync_check": True,
    
    # Advanced features
    "partial_fill_handling": True,
    "restart_recovery": True,
    "state_machine_validation": True,
    "structured_logging": True,
    
    # Integration
    "manual_exit_detection": True,
    "phantom_protection": True,
    "rate_limit_priority": True,
    
    # Rollout control
    "enhanced_exits": True,  # Master switch
    "legacy_fallback": True  # Keep old code path
}


class EnhancedExitFlow:
    """
    Complete production-ready exit system with all safety features.
    Fixes the 7-second exit bug and ensures reliable exits.
    """
    
    def __init__(self, broker_client, config: Dict[str, Any] = None):
        self.broker = broker_client
        self.config = config or {}
        self.features = ENHANCED_EXIT_FEATURES.copy()
        
        # Override features from config if provided
        if 'features' in self.config:
            self.features.update(self.config['features'])
        
        # Core components
        self.exit_manager = ExitManager(config)
        self.pattern_memory = PatternMemory()
        self.exit_labeler = ExitLabeler()
        self.fill_tracker = FillTracker()
        self.dual_stops = DualStopManager(broker_client, config)
        self.time_sync = TimeSyncManager(broker_client)
        self.partial_handler = PartialFillHandler(broker_client, self.dual_stops)
        self.restart_recovery = RestartRecovery(broker_client, self.dual_stops)
        
        # State tracking
        self.current_position = None
        self.current_market_data = {}
        self.active_brackets = {}
        
        # Logging
        self.exit_log_file = Path('logs/enhanced_exits.jsonl')
        self.exit_log_file.parent.mkdir(exist_ok=True)
        
        # Statistics
        self.exit_stats = {
            'total_exits': 0,
            'successful_exits': 0,
            'failed_exits': 0,
            'timeout_exits': 0,
            'emergency_exits': 0
        }
    
    async def initialize(self) -> bool:
        """Initialize enhanced exit system"""
        
        logger.info("Initializing enhanced exit flow...")
        
        try:
            # Check feature flags
            if not self.features.get('enhanced_exits', True):
                logger.info("Enhanced exits disabled - using legacy flow")
                return False
            
            # Perform restart recovery
            if self.features.get('restart_recovery', True):
                recovery_report = await self.restart_recovery.recover_on_startup()
                
                if recovery_report['success']:
                    logger.info(f"Recovery complete: {recovery_report['stats']}")
                    
                    # Adopt recovered positions
                    recovered_positions = self.restart_recovery.get_recovered_positions()
                    if recovered_positions:
                        self.current_position = recovered_positions[0]  # Take first
                        self.pattern_memory._state = TradingState.OPEN
                        logger.info(f"Adopted position: {self.current_position}")
            
            # Start time sync monitoring
            if self.features.get('time_sync_check', True):
                asyncio.create_task(self.time_sync.continuous_sync_monitor())
            
            # Validate initial time sync
            await self.time_sync.validate_sync()
            
            logger.info("✅ Enhanced exit flow initialized")
            return True
            
        except Exception as e:
            logger.error(f"Exit flow initialization failed: {e}")
            return False
    
    async def execute_exit(self, position: Dict, market_data: Dict) -> Dict:
        """
        Production-ready exit system with all safety features
        
        Args:
            position: Current position dict
            market_data: Current market data dict
            
        Returns:
            Exit result dict with details
        """
        
        self.current_position = position
        self.current_market_data = market_data
        
        exit_result = {
            'success': False,
            'exit_type': None,
            'exit_reason': None,
            'exit_price': None,
            'actual_price': None,
            'pnl': None,
            'slippage': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 0. Check feature flags
            if not self.features.get('enhanced_exits', True):
                logger.info("Using legacy exit flow")
                return await self._legacy_exit_flow(position, market_data)
            
            # 1. Time sync check (non-blocking for hard stops)
            time_sync_ok = True
            if self.features.get('time_sync_check', True):
                if self.time_sync.should_check_sync():
                    time_sync_ok = await self.time_sync.validate_sync()
            
            # 2. State machine validation
            if self.features.get('state_machine_validation', True):
                if not self.pattern_memory.is_state_valid_for_exit():
                    logger.debug(f"Invalid state for exit: {self.pattern_memory.get_current_state()}")
                    exit_result['error'] = 'invalid_state'
                    return exit_result
            
            # 3. Evaluate exits with priority and idempotency
            should_exit, exit_type, reason, exit_price = await self.exit_manager.evaluate_exits(
                position,
                market_data,
                time_sync_ok
            )
            
            if not should_exit:
                exit_result['error'] = 'no_exit_triggered'
                return exit_result
            
            # Record exit decision
            exit_result['exit_type'] = exit_type
            exit_result['exit_reason'] = reason
            exit_result['exit_price'] = exit_price
            
            # 4. State transition
            if self.features.get('state_machine_validation', True):
                self.pattern_memory.transition_to_exiting()
            
            # 5. Cancel broker brackets safely
            if self.features.get('dual_stops', True):
                bracket_info = self.dual_stops.get_active_bracket(position.get('id'))
                if bracket_info:
                    await self.dual_stops.safe_cancel_brackets(position, bracket_info)
            
            # 6. Execute market exit with rate limit priority
            exit_order = await self._execute_market_exit(position, exit_type)
            
            if not exit_order:
                logger.error("Failed to place exit order")
                exit_result['error'] = 'order_placement_failed'
                return exit_result
            
            # 7. Wait for fill with timeout
            actual_exit_price = await self._wait_for_exit_fill(
                exit_order, 
                timeout=self.config.get('exit_fill_timeout', 5.0)
            )
            
            if not actual_exit_price:
                logger.error("EXIT FILL TIMEOUT - position may be orphaned!")
                actual_exit_price = market_data.get('last_price', exit_price)
                exit_result['timeout'] = True
                self.exit_stats['timeout_exits'] += 1
            
            exit_result['actual_price'] = actual_exit_price
            
            # 8. Record fill with slippage
            if self.features.get('honest_labeling', True):
                fill_record = self.fill_tracker.record_fill(
                    order_id=exit_order.get('order_id'),
                    expected_price=exit_price,
                    actual_price=actual_exit_price,
                    size=abs(position.get('quantity', 0)),
                    side='exit'
                )
                
                exit_result['slippage'] = fill_record['slippage']
            
            # 9. Honest labeling
            if self.features.get('honest_labeling', True):
                real_reason, actual_pnl, pnl_points = self.exit_labeler.label_exit(
                    position,
                    actual_exit_price,
                    exit_type
                )
                
                exit_result['real_reason'] = real_reason
                exit_result['pnl'] = actual_pnl
                exit_result['pnl_points'] = pnl_points
                exit_result['reason_match'] = (exit_type == real_reason)
            
            # 10. Structured logging with correlation ID
            if self.features.get('structured_logging', True):
                await self._log_exit(exit_result)
            
            # 11. State cleanup
            await self.exit_manager.complete_exit()
            
            if self.features.get('pattern_persistence', True):
                self.pattern_memory.transition_to_closed()
                self.pattern_memory.reset()
            
            self.current_position = None
            
            # 12. Verify no orphaned orders
            await self._verify_no_orphaned_orders()
            
            # Success!
            exit_result['success'] = True
            self.exit_stats['successful_exits'] += 1
            self.exit_stats['total_exits'] += 1
            
            logger.info(f"✅ Exit complete: {exit_type} at {actual_exit_price:.2f} (PnL: ${actual_pnl:.2f})")
            
        except Exception as e:
            logger.error(f"Exit flow error: {e}", exc_info=True)
            exit_result['error'] = str(e)
            self.exit_stats['failed_exits'] += 1
            
            # Emergency cleanup
            await self._emergency_position_cleanup()
        
        return exit_result
    
    async def _execute_market_exit(self, position: Dict, exit_type: str) -> Optional[Dict]:
        """Execute market exit order with priority"""
        
        try:
            quantity = position.get('quantity', 0)
            
            if quantity == 0:
                logger.warning("No position to exit")
                return None
            
            # Place market order with priority
            order_params = {
                'instrument': position.get('instrument', 'NQ'),
                'quantity': -quantity,  # Opposite side to close
                'text': f"Exit: {exit_type}"
            }
            
            # Add priority for critical exits
            if self.features.get('rate_limit_priority', True):
                if exit_type in ['hard_stop', 'max_drawdown']:
                    order_params['priority'] = 'critical'
            
            order = await self.broker.place_market_order(**order_params)
            
            if order:
                logger.info(f"Exit order placed: {order.get('order_id')}")
            else:
                logger.error("Failed to place exit order")
            
            return order
            
        except Exception as e:
            logger.error(f"Exit order error: {e}")
            return None
    
    async def _wait_for_exit_fill(self, order: Dict, timeout: float) -> Optional[float]:
        """Wait for exit order fill with timeout"""
        
        start_time = datetime.now()
        order_id = order.get('order_id')
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                # Check order status
                status = await self.broker.get_order_status(order_id)
                
                if status.get('status') == 'filled':
                    return status.get('avg_fill_price')
                elif status.get('status') in ['cancelled', 'rejected']:
                    logger.error(f"Exit order {order_id} was {status.get('status')}")
                    return None
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error checking exit fill: {e}")
                await asyncio.sleep(0.1)
        
        logger.warning(f"Exit fill timeout for order {order_id}")
        return None
    
    async def _verify_no_orphaned_orders(self):
        """Verify no orders remain after exit"""
        
        try:
            # Get all working orders
            orders = await self.broker.get_working_orders()
            
            if orders:
                # Check for orders related to exited position
                for order in orders:
                    if order.get('instrument') == self.current_position.get('instrument'):
                        logger.warning(f"Orphaned order found: {order.get('order_id')}")
                        
                        # Cancel it
                        try:
                            await self.broker.cancel_order(order.get('order_id'))
                            logger.info(f"Cancelled orphaned order: {order.get('order_id')}")
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error verifying orphaned orders: {e}")
    
    async def _emergency_position_cleanup(self):
        """Emergency cleanup after exit failure"""
        
        logger.critical("EMERGENCY POSITION CLEANUP")
        self.exit_stats['emergency_exits'] += 1
        
        try:
            # Try to flatten position
            if self.current_position:
                quantity = self.current_position.get('quantity', 0)
                
                if quantity != 0:
                    logger.critical(f"Attempting emergency flatten: {quantity} contracts")
                    
                    # Try multiple times
                    for attempt in range(3):
                        try:
                            order = await self.broker.place_market_order(
                                instrument=self.current_position.get('instrument', 'NQ'),
                                quantity=-quantity,
                                text="EMERGENCY FLATTEN"
                            )
                            
                            if order:
                                logger.info(f"Emergency order placed: {order.get('order_id')}")
                                break
                        except Exception as e:
                            logger.error(f"Emergency attempt {attempt + 1} failed: {e}")
                            await asyncio.sleep(1)
            
            # Reset state
            await self.exit_manager.exit_once.reset_for_new_position()
            self.pattern_memory.reset()
            
        except Exception as e:
            logger.critical(f"Emergency cleanup failed: {e}")
    
    async def _log_exit(self, exit_result: Dict):
        """Log exit with structured format"""
        
        exit_log = {
            'timestamp': exit_result['timestamp'],
            'position_id': self.current_position.get('id') if self.current_position else None,
            'instrument': self.current_position.get('instrument') if self.current_position else None,
            'quantity': self.current_position.get('quantity') if self.current_position else None,
            'entry_price': self.current_position.get('average_price') if self.current_position else None,
            'exit_type': exit_result['exit_type'],
            'exit_reason': exit_result['exit_reason'],
            'real_reason': exit_result.get('real_reason'),
            'reason_match': exit_result.get('reason_match'),
            'expected_price': exit_result['exit_price'],
            'actual_price': exit_result['actual_price'],
            'slippage': exit_result.get('slippage'),
            'pnl': exit_result.get('pnl'),
            'pnl_points': exit_result.get('pnl_points'),
            'timeout': exit_result.get('timeout', False),
            'success': exit_result['success']
        }
        
        # Write to file
        try:
            with open(self.exit_log_file, 'a') as f:
                f.write(json.dumps(exit_log) + '\n')
        except Exception as e:
            logger.error(f"Failed to write exit log: {e}")
        
        # Log to console
        logger.info(f"EXIT LOG: {json.dumps(exit_log, indent=2)}")
    
    async def _legacy_exit_flow(self, position: Dict, market_data: Dict) -> Dict:
        """Legacy exit flow for fallback"""
        
        logger.info("Using legacy exit flow")
        
        # Simple market exit without enhancements
        exit_result = {
            'success': False,
            'exit_type': 'legacy',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            quantity = position.get('quantity', 0)
            if quantity != 0:
                order = await self.broker.place_market_order(
                    instrument=position.get('instrument', 'NQ'),
                    quantity=-quantity,
                    text="Legacy exit"
                )
                
                if order:
                    exit_result['success'] = True
                    exit_result['order_id'] = order.get('order_id')
                    
        except Exception as e:
            logger.error(f"Legacy exit error: {e}")
            exit_result['error'] = str(e)
        
        return exit_result
    
    async def handle_manual_exit(self):
        """Handle manual exit request"""
        
        if self.features.get('manual_exit_detection', True):
            logger.warning("Manual exit requested")
            
            # Force exit through exit manager
            if await self.exit_manager.force_exit("Manual override"):
                # Execute exit flow
                if self.current_position:
                    return await self.execute_exit(
                        self.current_position,
                        self.current_market_data
                    )
        
        return {'success': False, 'error': 'no_position'}
    
    def get_statistics(self) -> Dict:
        """Get comprehensive exit statistics"""
        
        stats = {
            'exit_flow': self.exit_stats.copy(),
            'exit_manager': self.exit_manager.get_statistics(),
            'pattern_memory': self.pattern_memory.get_statistics(),
            'exit_labels': self.exit_labeler.get_statistics(),
            'fill_tracking': self.fill_tracker.get_statistics(),
            'dual_stops': self.dual_stops.get_statistics(),
            'time_sync': self.time_sync.get_statistics(),
            'partial_fills': self.partial_handler.get_statistics(),
            'recovery': self.restart_recovery.get_statistics()
        }
        
        return stats