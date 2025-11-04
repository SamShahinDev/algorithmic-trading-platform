"""
Restart Recovery - Recover gracefully from bot restarts
Detects and validates existing positions and orders
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class RestartRecovery:
    """Recover gracefully from restarts with position and order validation"""
    
    def __init__(self, broker_client, dual_stop_manager):
        self.broker = broker_client
        self.dual_stops = dual_stop_manager
        
        # State files
        self.state_dir = Path('logs/state')
        self.state_dir.mkdir(exist_ok=True)
        
        self.position_state_file = self.state_dir / 'last_position_state.json'
        self.order_state_file = self.state_dir / 'last_order_state.json'
        
        # Recovery state
        self.recovery_performed = False
        self.recovered_positions = []
        self.recovered_orders = []
        
        # Statistics
        self.recovery_stats = {
            'positions_found': 0,
            'positions_adopted': 0,
            'orders_found': 0,
            'orders_validated': 0,
            'emergency_stops_placed': 0,
            'orphaned_orders_cancelled': 0
        }
    
    async def recover_on_startup(self) -> Dict:
        """
        Detect and validate existing orders on startup
        
        Returns:
            Recovery report dict
        """
        
        logger.info("=" * 60)
        logger.info("RESTART RECOVERY CHECK")
        logger.info("=" * 60)
        
        recovery_report = {
            'timestamp': datetime.now().isoformat(),
            'positions': [],
            'orders': [],
            'actions_taken': [],
            'success': False
        }
        
        try:
            # Load previous state
            previous_state = self._load_previous_state()
            
            # Get current state from broker
            logger.info("Checking for existing positions and orders...")
            positions = await self.broker.get_open_positions()
            orders = await self.broker.get_working_orders()
            
            self.recovery_stats['positions_found'] = len(positions) if positions else 0
            self.recovery_stats['orders_found'] = len(orders) if orders else 0
            
            # Process positions
            if positions:
                logger.warning(f"Found {len(positions)} existing position(s)")
                for position in positions:
                    await self._process_existing_position(position, orders, recovery_report)
            else:
                logger.info("✓ No existing positions found")
            
            # Process orphaned orders
            orphaned_orders = await self._find_orphaned_orders(positions, orders)
            if orphaned_orders:
                await self._handle_orphaned_orders(orphaned_orders, recovery_report)
            
            # Save current state for next restart
            self._save_current_state(positions, orders)
            
            # Mark recovery complete
            self.recovery_performed = True
            recovery_report['success'] = True
            recovery_report['stats'] = self.recovery_stats.copy()
            
            logger.info("=" * 60)
            logger.info(f"RECOVERY COMPLETE - Adopted: {self.recovery_stats['positions_adopted']} positions")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            recovery_report['error'] = str(e)
            recovery_report['success'] = False
        
        return recovery_report
    
    async def _process_existing_position(self, position: Dict, 
                                        all_orders: List[Dict],
                                        recovery_report: Dict):
        """Process an existing position found on startup"""
        
        logger.info(f"Processing position: {position}")
        
        # Find protective orders
        stop_order = self._find_stop_order(all_orders, position)
        target_order = self._find_target_order(all_orders, position)
        
        position_info = {
            'instrument': position.get('instrument'),
            'quantity': position.get('quantity'),
            'average_price': position.get('average_price'),
            'has_stop': stop_order is not None,
            'has_target': target_order is not None
        }
        
        recovery_report['positions'].append(position_info)
        
        # Validate protection
        if self._validate_bracket_integrity(position, stop_order, target_order):
            logger.info("✅ Position has valid protection")
            
            # Adopt the position and brackets
            self._adopt_existing_bracket(position, stop_order, target_order)
            self.recovery_stats['positions_adopted'] += 1
            
            recovery_report['actions_taken'].append({
                'action': 'adopt_position',
                'position_id': position.get('id'),
                'reason': 'valid_brackets'
            })
            
        else:
            logger.warning("⚠️ Position missing/invalid protection!")
            
            # Place emergency protection
            await self._place_emergency_protection(position)
            self.recovery_stats['emergency_stops_placed'] += 1
            
            recovery_report['actions_taken'].append({
                'action': 'emergency_protection',
                'position_id': position.get('id'),
                'reason': 'invalid_brackets'
            })
    
    def _validate_bracket_integrity(self, position: Dict, 
                                   stop: Optional[Dict], 
                                   target: Optional[Dict]) -> bool:
        """Ensure brackets are properly configured"""
        
        if not stop:
            logger.warning("Missing stop order")
            return False
        
        # Target is optional but recommended
        if not target:
            logger.warning("Missing target order (will continue with stop only)")
        
        # Check quantities match
        position_qty = abs(position.get('quantity', 0))
        stop_qty = abs(stop.get('quantity', 0))
        
        if stop_qty != position_qty:
            logger.warning(f"Stop quantity mismatch: {stop_qty} vs {position_qty}")
            return False
        
        if target:
            target_qty = abs(target.get('quantity', 0))
            if target_qty != position_qty:
                logger.warning(f"Target quantity mismatch: {target_qty} vs {position_qty}")
                return False
        
        # Check sides are opposite of position
        if position.get('quantity', 0) > 0:  # Long position
            if stop.get('side') != 'SELL':
                logger.warning("Stop side incorrect for long position")
                return False
            if target and target.get('side') != 'SELL':
                logger.warning("Target side incorrect for long position")
                return False
        else:  # Short position
            if stop.get('side') != 'BUY':
                logger.warning("Stop side incorrect for short position")
                return False
            if target and target.get('side') != 'BUY':
                logger.warning("Target side incorrect for short position")
                return False
        
        # Check prices make sense
        entry_price = position.get('average_price', 0)
        stop_price = stop.get('stop_price', stop.get('price', 0))
        
        if position.get('quantity', 0) > 0:  # Long
            if stop_price >= entry_price:
                logger.warning(f"Stop price invalid for long: {stop_price} >= {entry_price}")
                return False
            if target:
                target_price = target.get('limit_price', target.get('price', 0))
                if target_price <= entry_price:
                    logger.warning(f"Target price invalid for long: {target_price} <= {entry_price}")
                    return False
        else:  # Short
            if stop_price <= entry_price:
                logger.warning(f"Stop price invalid for short: {stop_price} <= {entry_price}")
                return False
            if target:
                target_price = target.get('limit_price', target.get('price', 0))
                if target_price >= entry_price:
                    logger.warning(f"Target price invalid for short: {target_price} >= {entry_price}")
                    return False
        
        logger.info("Bracket validation passed")
        return True
    
    def _adopt_existing_bracket(self, position: Dict, 
                               stop_order: Optional[Dict], 
                               target_order: Optional[Dict]):
        """Adopt existing position and brackets into bot state"""
        
        # Create bracket info for dual stop manager
        bracket_info = {
            'position_id': position.get('id'),
            'stop_order_id': stop_order.get('order_id') if stop_order else None,
            'target_order_id': target_order.get('order_id') if target_order else None,
            'stop_price': stop_order.get('stop_price', stop_order.get('price')) if stop_order else None,
            'target_price': target_order.get('limit_price', target_order.get('price')) if target_order else None,
            'quantity': abs(position.get('quantity', 0)),
            'side': 'SELL' if position.get('quantity', 0) > 0 else 'BUY',
            'adopted': True,
            'created_at': datetime.now()
        }
        
        # Register with dual stop manager
        self.dual_stops.active_brackets[position.get('id')] = bracket_info
        
        # Track recovery
        self.recovered_positions.append(position)
        self.recovered_orders.extend([o for o in [stop_order, target_order] if o])
        
        logger.info(f"Adopted position {position.get('id')} with brackets")
    
    async def _place_emergency_protection(self, position: Dict):
        """Place emergency stop for unprotected position"""
        
        logger.critical(f"PLACING EMERGENCY PROTECTION for position")
        
        quantity = position.get('quantity', 0)
        entry_price = position.get('average_price', 0)
        
        if quantity == 0:
            return
        
        try:
            # Calculate emergency stop (wider than normal)
            if quantity > 0:  # Long
                stop_price = entry_price - 30  # 30 points for emergency
                side = 'SELL'
            else:  # Short
                stop_price = entry_price + 30
                side = 'BUY'
            
            stop_order = await self.broker.place_stop_order(
                instrument=position.get('instrument', 'NQ'),
                quantity=abs(quantity),
                side=side,
                stop_price=stop_price,
                text="RECOVERY EMERGENCY STOP"
            )
            
            if stop_order:
                logger.info(f"Emergency stop placed at {stop_price}")
                
                # Register with dual stop manager
                bracket_info = {
                    'position_id': position.get('id'),
                    'stop_order_id': stop_order.get('order_id'),
                    'target_order_id': None,
                    'stop_price': stop_price,
                    'is_emergency': True,
                    'created_at': datetime.now()
                }
                self.dual_stops.active_brackets[position.get('id')] = bracket_info
                
            else:
                logger.critical("FAILED TO PLACE EMERGENCY STOP!")
                
        except Exception as e:
            logger.critical(f"Emergency stop error: {e}")
    
    def _find_stop_order(self, orders: List[Dict], position: Dict) -> Optional[Dict]:
        """Find stop order protecting position"""
        
        position_qty = position.get('quantity', 0)
        
        for order in orders:
            # Check if it's a stop order
            if order.get('order_type') not in ['stop', 'stop_limit']:
                continue
            
            # Check instrument matches
            if order.get('instrument') != position.get('instrument'):
                continue
            
            # Check side is opposite of position
            order_qty = order.get('quantity', 0)
            
            if position_qty > 0:  # Long position needs sell stop
                if order.get('side') == 'SELL' or order_qty < 0:
                    return order
            else:  # Short position needs buy stop
                if order.get('side') == 'BUY' or order_qty > 0:
                    return order
        
        return None
    
    def _find_target_order(self, orders: List[Dict], position: Dict) -> Optional[Dict]:
        """Find target order for position"""
        
        position_qty = position.get('quantity', 0)
        
        for order in orders:
            # Check if it's a limit order
            if order.get('order_type') != 'limit':
                continue
            
            # Check instrument matches
            if order.get('instrument') != position.get('instrument'):
                continue
            
            # Check side is opposite of position
            order_qty = order.get('quantity', 0)
            
            if position_qty > 0:  # Long position needs sell target
                if order.get('side') == 'SELL' or order_qty < 0:
                    return order
            else:  # Short position needs buy target
                if order.get('side') == 'BUY' or order_qty > 0:
                    return order
        
        return None
    
    async def _find_orphaned_orders(self, positions: List[Dict], 
                                   orders: List[Dict]) -> List[Dict]:
        """Find orders without corresponding positions"""
        
        if not orders:
            return []
        
        # Get list of instruments with positions
        position_instruments = set()
        if positions:
            position_instruments = {p.get('instrument') for p in positions}
        
        orphaned = []
        
        for order in orders:
            instrument = order.get('instrument')
            
            # Check if order is for an instrument without position
            if instrument not in position_instruments:
                orphaned.append(order)
                logger.warning(f"Orphaned order found: {order.get('order_id')} for {instrument}")
        
        return orphaned
    
    async def _handle_orphaned_orders(self, orphaned_orders: List[Dict], 
                                     recovery_report: Dict):
        """Handle orders without positions"""
        
        logger.warning(f"Found {len(orphaned_orders)} orphaned orders")
        
        for order in orphaned_orders:
            try:
                # Cancel orphaned order
                result = await self.broker.cancel_order(order.get('order_id'))
                
                if result:
                    logger.info(f"Cancelled orphaned order: {order.get('order_id')}")
                    self.recovery_stats['orphaned_orders_cancelled'] += 1
                    
                    recovery_report['actions_taken'].append({
                        'action': 'cancel_orphaned_order',
                        'order_id': order.get('order_id'),
                        'order_type': order.get('order_type')
                    })
                    
            except Exception as e:
                logger.error(f"Failed to cancel orphaned order: {e}")
    
    def _load_previous_state(self) -> Dict:
        """Load previous state from files"""
        
        state = {
            'positions': [],
            'orders': [],
            'timestamp': None
        }
        
        try:
            if self.position_state_file.exists():
                with open(self.position_state_file) as f:
                    position_data = json.load(f)
                    state['positions'] = position_data.get('positions', [])
                    state['timestamp'] = position_data.get('timestamp')
                    logger.info(f"Loaded previous position state from {state['timestamp']}")
            
            if self.order_state_file.exists():
                with open(self.order_state_file) as f:
                    order_data = json.load(f)
                    state['orders'] = order_data.get('orders', [])
                    
        except Exception as e:
            logger.error(f"Failed to load previous state: {e}")
        
        return state
    
    def _save_current_state(self, positions: List[Dict], orders: List[Dict]):
        """Save current state for next restart"""
        
        try:
            # Save position state
            position_data = {
                'positions': positions if positions else [],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.position_state_file, 'w') as f:
                json.dump(position_data, f, indent=2, default=str)
            
            # Save order state
            order_data = {
                'orders': orders if orders else [],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.order_state_file, 'w') as f:
                json.dump(order_data, f, indent=2, default=str)
            
            logger.debug("State saved for next restart")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_recovered_positions(self) -> List[Dict]:
        """Get list of recovered positions"""
        return self.recovered_positions.copy()
    
    def get_statistics(self) -> Dict:
        """Get recovery statistics"""
        return self.recovery_stats.copy()