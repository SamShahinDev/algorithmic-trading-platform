"""
Partial Fill Handler - Handle partial fills correctly
Manages position and bracket adjustments for partial fills
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)


class PartialFillHandler:
    """Handle partial fills with position and bracket adjustments"""
    
    def __init__(self, broker_client, dual_stop_manager):
        self.broker = broker_client
        self.dual_stops = dual_stop_manager
        
        # Tracking
        self.partial_positions = {}
        self.fill_history = []
        
        # Statistics
        self.partial_stats = {
            'total_partials': 0,
            'fully_filled': 0,
            'cancelled_remainders': 0,
            'bracket_resizes': 0
        }
    
    async def handle_partial_fill(self, order_event: Dict, position: Dict) -> Dict:
        """
        Handle partial fill event
        
        Args:
            order_event: Fill event from broker
            position: Current position dict
            
        Returns:
            Updated position dict
        """
        
        filled_qty = order_event.get('filled_quantity', 0)
        total_qty = order_event.get('total_quantity', 0)
        order_id = order_event.get('order_id')
        
        if filled_qty >= total_qty:
            # Full fill - no special handling needed
            logger.debug(f"Full fill: {filled_qty}/{total_qty}")
            return position
        
        logger.warning(
            f"PARTIAL FILL DETECTED: {filled_qty}/{total_qty} "
            f"({(filled_qty/total_qty*100):.1f}% filled)"
        )
        
        self.partial_stats['total_partials'] += 1
        
        # Record the partial fill
        partial_info = {
            'order_id': order_id,
            'filled_quantity': filled_qty,
            'total_quantity': total_qty,
            'fill_percentage': (filled_qty / total_qty) * 100,
            'timestamp': datetime.now(),
            'avg_fill_price': order_event.get('avg_fill_price')
        }
        
        self.partial_positions[position.get('id')] = partial_info
        self.fill_history.append(partial_info)
        
        # Update position size
        old_size = position.get('quantity', 0)
        position['quantity'] = filled_qty if position['quantity'] > 0 else -filled_qty
        
        logger.info(f"Position size adjusted: {old_size} -> {position['quantity']}")
        
        # Resize brackets if they exist
        await self._adjust_brackets_for_partial(position, old_size, filled_qty)
        
        # Handle remainder
        remaining_qty = total_qty - filled_qty
        if remaining_qty > 0:
            await self._handle_remainder(order_event, remaining_qty)
        
        return position
    
    async def _adjust_brackets_for_partial(self, position: Dict, 
                                           old_size: int, new_size: int):
        """Adjust bracket orders for partial fill"""
        
        # Get active brackets
        bracket_info = self.dual_stops.get_active_bracket(position.get('id'))
        
        if not bracket_info:
            logger.debug("No brackets to adjust")
            return
        
        logger.info(f"Resizing brackets from {abs(old_size)} to {abs(new_size)}")
        
        try:
            # Cancel old brackets
            await self.dual_stops.safe_cancel_brackets(position, bracket_info)
            
            # Place new brackets with correct size
            new_position = position.copy()
            new_position['quantity'] = new_size
            
            # Create dummy entry order for bracket placement
            entry_order = {
                'order_id': f"partial_{position.get('id')}",
                'avg_fill_price': position.get('average_price')
            }
            
            await self.dual_stops.place_bracket_order(entry_order, new_position)
            
            self.partial_stats['bracket_resizes'] += 1
            logger.info("Brackets resized successfully")
            
        except Exception as e:
            logger.error(f"Failed to resize brackets: {e}")
            # Place emergency stop at minimum
            await self._place_emergency_protection(position)
    
    async def _handle_remainder(self, order_event: Dict, remaining_qty: int):
        """Handle the unfilled remainder of an order"""
        
        order_id = order_event.get('order_id')
        
        logger.info(f"Handling remainder: {remaining_qty} units unfilled")
        
        try:
            # Cancel the remainder
            cancel_result = await self.broker.cancel_order(order_id)
            
            if cancel_result:
                logger.info(f"Remainder cancelled for order {order_id}")
                self.partial_stats['cancelled_remainders'] += 1
            else:
                logger.warning(f"Failed to cancel remainder for order {order_id}")
            
        except Exception as e:
            logger.error(f"Error cancelling remainder: {e}")
    
    async def check_for_additional_fills(self, position: Dict) -> bool:
        """
        Check if partial position has received additional fills
        
        Returns:
            True if position is now fully filled
        """
        
        partial_info = self.partial_positions.get(position.get('id'))
        
        if not partial_info:
            return True  # Not a partial position
        
        try:
            # Get current order status
            order_status = await self.broker.get_order_status(
                partial_info['order_id']
            )
            
            current_filled = order_status.get('filled_quantity', 0)
            total_qty = partial_info['total_quantity']
            
            if current_filled > partial_info['filled_quantity']:
                # Additional fills received
                logger.info(
                    f"Additional fills: {partial_info['filled_quantity']} -> "
                    f"{current_filled}/{total_qty}"
                )
                
                # Update tracking
                partial_info['filled_quantity'] = current_filled
                partial_info['fill_percentage'] = (current_filled / total_qty) * 100
                
                # Check if now fully filled
                if current_filled >= total_qty:
                    logger.info("Partial position now FULLY FILLED")
                    self.partial_stats['fully_filled'] += 1
                    del self.partial_positions[position.get('id')]
                    return True
                
                # Still partial
                return False
            
            return current_filled >= total_qty
            
        except Exception as e:
            logger.error(f"Error checking for additional fills: {e}")
            return False
    
    async def _place_emergency_protection(self, position: Dict):
        """Place emergency stop for partial position"""
        
        logger.warning("Placing emergency protection for partial position")
        
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return
        
        try:
            # Calculate emergency stop
            entry_price = position.get('average_price', 0)
            
            if quantity > 0:  # Long
                stop_price = entry_price - 25  # Wider stop for safety
                side = 'SELL'
            else:  # Short
                stop_price = entry_price + 25
                side = 'BUY'
            
            stop_order = await self.broker.place_stop_order(
                instrument='NQ',
                quantity=abs(quantity),
                side=side,
                stop_price=stop_price,
                text="Emergency stop (partial)"
            )
            
            if stop_order:
                logger.info(f"Emergency stop placed at {stop_price}")
            else:
                logger.critical("FAILED TO PLACE EMERGENCY STOP FOR PARTIAL!")
                
        except Exception as e:
            logger.critical(f"Emergency protection error: {e}")
    
    def is_partial_position(self, position_id: str) -> bool:
        """Check if position is partial"""
        return position_id in self.partial_positions
    
    def get_partial_info(self, position_id: str) -> Optional[Dict]:
        """Get partial fill information for position"""
        return self.partial_positions.get(position_id)
    
    def get_statistics(self) -> Dict:
        """Get partial fill statistics"""
        
        stats = self.partial_stats.copy()
        stats['active_partials'] = len(self.partial_positions)
        stats['fill_history_count'] = len(self.fill_history)
        
        if self.fill_history:
            # Calculate average fill percentage
            fill_percentages = [f['fill_percentage'] for f in self.fill_history]
            stats['average_fill_percentage'] = sum(fill_percentages) / len(fill_percentages)
            
            # Find worst partial
            stats['worst_partial'] = min(fill_percentages)
        
        return stats


class FillMonitor:
    """Monitor fills and detect issues"""
    
    def __init__(self, broker_client):
        self.broker = broker_client
        self.expected_fills = {}
        self.received_fills = {}
        self.fill_issues = []
    
    def expect_fill(self, order_id: str, expected_qty: int, expected_price: float):
        """Register an expected fill"""
        
        self.expected_fills[order_id] = {
            'quantity': expected_qty,
            'price': expected_price,
            'timestamp': datetime.now(),
            'received': False
        }
    
    def record_fill(self, order_id: str, filled_qty: int, fill_price: float):
        """Record an actual fill"""
        
        self.received_fills[order_id] = {
            'quantity': filled_qty,
            'price': fill_price,
            'timestamp': datetime.now()
        }
        
        # Check against expected
        if order_id in self.expected_fills:
            expected = self.expected_fills[order_id]
            expected['received'] = True
            
            # Check for issues
            if filled_qty != expected['quantity']:
                issue = {
                    'order_id': order_id,
                    'type': 'quantity_mismatch',
                    'expected': expected['quantity'],
                    'received': filled_qty,
                    'timestamp': datetime.now()
                }
                self.fill_issues.append(issue)
                logger.warning(f"Fill quantity mismatch: {issue}")
            
            # Check price deviation
            price_diff = abs(fill_price - expected['price'])
            if price_diff > 0.5:  # More than 2 ticks
                issue = {
                    'order_id': order_id,
                    'type': 'price_deviation',
                    'expected': expected['price'],
                    'received': fill_price,
                    'deviation': price_diff,
                    'timestamp': datetime.now()
                }
                self.fill_issues.append(issue)
                logger.warning(f"Fill price deviation: {issue}")
    
    def check_missing_fills(self, timeout_seconds: float = 10) -> List[str]:
        """Check for expected fills that haven't arrived"""
        
        missing = []
        now = datetime.now()
        
        for order_id, expected in self.expected_fills.items():
            if not expected['received']:
                age = (now - expected['timestamp']).total_seconds()
                if age > timeout_seconds:
                    missing.append(order_id)
                    logger.warning(f"Missing fill for order {order_id} (age: {age:.1f}s)")
        
        return missing
    
    def get_fill_issues(self) -> List[Dict]:
        """Get list of fill issues"""
        return self.fill_issues.copy()