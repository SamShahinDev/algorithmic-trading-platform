"""
Dual Stop Manager - Broker + bot stops with timeout protection
Manages bracket orders safely with proper cancellation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class DualStopManager:
    """Broker + bot stops with timeout protection"""
    
    def __init__(self, broker_client, config: Dict[str, Any] = None):
        self.broker = broker_client
        self.config = config or {}
        
        # Configuration
        self.fill_timeout = self.config.get('fill_timeout', 5.0)
        self.cancel_timeout = self.config.get('cancel_timeout', 1.5)
        self.emergency_stop_offset = self.config.get('emergency_stop_offset', 25)
        
        # Tracking
        self.active_brackets = {}
        self.orphaned_orders = []
        self.bracket_stats = {
            'brackets_placed': 0,
            'brackets_cancelled': 0,
            'cancel_timeouts': 0,
            'emergency_stops': 0
        }
    
    async def place_bracket_order(self, entry_order: Dict, position: Dict) -> Optional[Dict]:
        """
        Place OCO bracket with fill validation
        
        Returns:
            Bracket info dict or None if failed
        """
        
        logger.info(f"Placing bracket order for position {position.get('id')}")
        
        # Wait for entry fill
        fill_price = await self._wait_for_fill(entry_order, timeout=self.fill_timeout)
        
        if not fill_price:
            logger.error("Entry fill timeout - aborting bracket")
            return None
        
        # Calculate tick-safe stops
        fill_price = self._round_tick(fill_price)
        quantity = position.get('quantity', 0)
        
        if quantity > 0:  # LONG
            stop_price = self._round_tick(fill_price - self.config.get('stop_distance', 20))
            target_price = self._round_tick(fill_price + self.config.get('target_distance', 40))
            bracket_side = 'SELL'
        else:  # SHORT
            stop_price = self._round_tick(fill_price + self.config.get('stop_distance', 20))
            target_price = self._round_tick(fill_price - self.config.get('target_distance', 40))
            bracket_side = 'BUY'
        
        # Place OCO bracket
        try:
            bracket_result = await self._place_oco_bracket(
                quantity=abs(quantity),
                side=bracket_side,
                stop_price=stop_price,
                target_price=target_price,
                parent_order_id=entry_order.get('order_id')
            )
            
            if bracket_result:
                bracket_info = {
                    'position_id': position.get('id'),
                    'stop_order_id': bracket_result.get('stop_order_id'),
                    'target_order_id': bracket_result.get('target_order_id'),
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'quantity': abs(quantity),
                    'side': bracket_side,
                    'created_at': datetime.now()
                }
                
                # Store bracket info
                self.active_brackets[position.get('id')] = bracket_info
                self.bracket_stats['brackets_placed'] += 1
                
                logger.info(
                    f"Bracket placed successfully: "
                    f"Stop={stop_price:.2f}, Target={target_price:.2f}"
                )
                
                return bracket_info
            
        except Exception as e:
            logger.error(f"Bracket placement failed: {e}")
            
        # If bracket fails, place emergency stop at minimum
        await self._place_emergency_stop(position, stop_price, bracket_side)
        return None
    
    async def _place_oco_bracket(self, quantity: int, side: str, 
                                 stop_price: float, target_price: float,
                                 parent_order_id: Optional[str] = None) -> Dict:
        """Place OCO (One Cancels Other) bracket order"""
        
        try:
            # Place stop order
            stop_order = await self.broker.place_stop_order(
                instrument='NQ',
                quantity=quantity,
                side=side,
                stop_price=stop_price,
                text=f"Stop for {parent_order_id}"
            )
            
            if not stop_order:
                raise Exception("Failed to place stop order")
            
            # Place target order
            target_order = await self.broker.place_limit_order(
                instrument='NQ',
                quantity=quantity,
                side=side,
                limit_price=target_price,
                text=f"Target for {parent_order_id}"
            )
            
            if not target_order:
                # Cancel stop if target fails
                await self.broker.cancel_order(stop_order['order_id'])
                raise Exception("Failed to place target order")
            
            # Link orders as OCO if broker supports it
            if hasattr(self.broker, 'link_oco_orders'):
                await self.broker.link_oco_orders(
                    stop_order['order_id'],
                    target_order['order_id']
                )
            
            return {
                'stop_order_id': stop_order['order_id'],
                'target_order_id': target_order['order_id'],
                'stop_order': stop_order,
                'target_order': target_order
            }
            
        except Exception as e:
            logger.error(f"OCO bracket placement error: {e}")
            return None
    
    async def safe_cancel_brackets(self, position: Dict, bracket_info: Optional[Dict] = None):
        """
        Cancel brackets with timeout protection
        
        Args:
            position: Position dict
            bracket_info: Optional bracket info (will look up if not provided)
        """
        
        # Get bracket info if not provided
        if not bracket_info:
            bracket_info = self.active_brackets.get(position.get('id'))
        
        if not bracket_info:
            logger.debug(f"No brackets found for position {position.get('id')}")
            return
        
        logger.info(f"Cancelling brackets for position {position.get('id')}")
        
        try:
            # Try to cancel with short timeout
            await asyncio.wait_for(
                self._cancel_bracket_orders(bracket_info),
                timeout=self.cancel_timeout
            )
            
            logger.info("Brackets cancelled successfully")
            self.bracket_stats['brackets_cancelled'] += 1
            
            # Remove from active brackets
            self.active_brackets.pop(position.get('id'), None)
            
        except asyncio.TimeoutError:
            logger.warning("Cancel timeout - proceeding with exit anyway")
            self.bracket_stats['cancel_timeouts'] += 1
            
            # Mark orders as potentially orphaned
            self._mark_orders_orphaned(bracket_info)
            
        except Exception as e:
            logger.error(f"Cancel error: {e} - proceeding with exit")
            self._mark_orders_orphaned(bracket_info)
    
    async def _cancel_bracket_orders(self, bracket_info: Dict):
        """Actual cancellation logic"""
        
        tasks = []
        
        if bracket_info.get('stop_order_id'):
            tasks.append(self.broker.cancel_order(bracket_info['stop_order_id']))
        
        if bracket_info.get('target_order_id'):
            tasks.append(self.broker.cancel_order(bracket_info['target_order_id']))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to cancel order: {result}")
    
    def _mark_orders_orphaned(self, bracket_info: Dict):
        """Mark orders as potentially orphaned for later cleanup"""
        
        orphaned = {
            'timestamp': datetime.now(),
            'bracket_info': bracket_info,
            'reason': 'cancel_timeout'
        }
        
        self.orphaned_orders.append(orphaned)
        
        logger.warning(
            f"Marked orders as orphaned: "
            f"Stop={bracket_info.get('stop_order_id')}, "
            f"Target={bracket_info.get('target_order_id')}"
        )
    
    async def _place_emergency_stop(self, position: Dict, stop_price: float, side: str):
        """Place emergency stop order when bracket fails"""
        
        logger.critical(f"PLACING EMERGENCY STOP for position {position.get('id')}")
        
        try:
            # Use wider stop for emergency
            if position.get('quantity', 0) > 0:  # Long
                emergency_price = stop_price - 5  # 5 points wider
            else:  # Short
                emergency_price = stop_price + 5
            
            emergency_stop = await self.broker.place_stop_order(
                instrument='NQ',
                quantity=abs(position.get('quantity', 0)),
                side=side,
                stop_price=self._round_tick(emergency_price),
                text="EMERGENCY STOP"
            )
            
            if emergency_stop:
                logger.info(f"Emergency stop placed at {emergency_price:.2f}")
                self.bracket_stats['emergency_stops'] += 1
                
                # Track as minimal bracket
                self.active_brackets[position.get('id')] = {
                    'position_id': position.get('id'),
                    'stop_order_id': emergency_stop['order_id'],
                    'target_order_id': None,
                    'stop_price': emergency_price,
                    'is_emergency': True
                }
            else:
                logger.critical("FAILED TO PLACE EMERGENCY STOP - POSITION UNPROTECTED!")
                
        except Exception as e:
            logger.critical(f"Emergency stop error: {e} - POSITION UNPROTECTED!")
    
    async def _wait_for_fill(self, order: Dict, timeout: float) -> Optional[float]:
        """Wait for order fill with timeout"""
        
        start_time = datetime.now()
        order_id = order.get('order_id')
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                # Check order status
                status = await self.broker.get_order_status(order_id)
                
                if status.get('status') == 'filled':
                    return status.get('avg_fill_price')
                elif status.get('status') in ['cancelled', 'rejected']:
                    logger.error(f"Order {order_id} was {status.get('status')}")
                    return None
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
                await asyncio.sleep(0.1)
        
        logger.warning(f"Fill timeout for order {order_id}")
        return None
    
    async def cleanup_orphaned_orders(self):
        """Clean up any orphaned orders"""
        
        if not self.orphaned_orders:
            return
        
        logger.info(f"Cleaning up {len(self.orphaned_orders)} orphaned order sets")
        
        for orphaned in self.orphaned_orders[:]:  # Copy list to iterate
            # Check if orders still exist
            bracket_info = orphaned['bracket_info']
            
            try:
                # Try to cancel again
                await self._cancel_bracket_orders(bracket_info)
                self.orphaned_orders.remove(orphaned)
                
            except Exception as e:
                logger.error(f"Failed to cleanup orphaned orders: {e}")
                
                # Remove if old enough
                age = (datetime.now() - orphaned['timestamp']).total_seconds()
                if age > 300:  # 5 minutes
                    self.orphaned_orders.remove(orphaned)
    
    def _round_tick(self, price: float) -> float:
        """Round to valid NQ tick"""
        TICK = 0.25
        return round(price / TICK) * TICK
    
    def get_active_bracket(self, position_id: str) -> Optional[Dict]:
        """Get active bracket for position"""
        return self.active_brackets.get(position_id)
    
    def get_statistics(self) -> Dict:
        """Get bracket management statistics"""
        
        stats = self.bracket_stats.copy()
        stats['active_brackets'] = len(self.active_brackets)
        stats['orphaned_orders'] = len(self.orphaned_orders)
        
        if stats['brackets_placed'] > 0:
            stats['cancel_success_rate'] = (
                (stats['brackets_cancelled'] / stats['brackets_placed']) * 100
            )
            stats['timeout_rate'] = (
                (stats['cancel_timeouts'] / stats['brackets_placed']) * 100
            )
        
        return stats