"""
Exit Manager - Priority-based exit system with idempotent protection
Prevents double exits and race conditions
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List

logger = logging.getLogger(__name__)


class ExitOnce:
    """Prevent double exits and race conditions"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._armed = True
        self._exit_in_progress = False
        self._last_exit_time = None
        self._exit_count = 0
        
    async def try_arm(self) -> bool:
        """Attempt to arm for exit - returns True if allowed"""
        async with self._lock:
            if not self._armed or self._exit_in_progress:
                logger.debug(f"Exit blocked - armed: {self._armed}, in_progress: {self._exit_in_progress}")
                return False
            self._exit_in_progress = True
            self._exit_count += 1
            logger.info(f"Exit armed (attempt #{self._exit_count})")
            return True
            
    async def complete_exit(self):
        """Mark exit as complete"""
        async with self._lock:
            self._armed = False
            self._exit_in_progress = False
            self._last_exit_time = datetime.now()
            logger.info(f"Exit completed at {self._last_exit_time}")
            
    async def reset_for_new_position(self):
        """Reset for a new position"""
        async with self._lock:
            # Ensure minimum time between positions
            if self._last_exit_time:
                time_since_exit = (datetime.now() - self._last_exit_time).total_seconds()
                if time_since_exit < 2.0:
                    logger.warning(f"Reset too soon after exit: {time_since_exit:.1f}s")
                    return False
                    
            self._armed = True
            self._exit_in_progress = False
            self._exit_count = 0
            logger.info("Exit gate reset for new position")
            return True
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'armed': self._armed,
            'in_progress': self._exit_in_progress,
            'exit_count': self._exit_count,
            'last_exit': self._last_exit_time
        }


class ExitManager:
    """Centralized exit system with strict priority and idempotency"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.exit_once = ExitOnce()
        
        # Exit priorities (highest to lowest)
        self.exit_priorities = [
            ('hard_stop', self._check_hard_stop, True),           # Always process
            ('take_profit', self._check_take_profit, True),       # Always process
            ('trailing_stop', self._check_trailing_stop, True),   # Always process
            ('reversal_pattern', self._check_reversal_pattern, False),  # Skip if time sync bad
            ('time_stop', self._check_time_stop, False),          # Skip if time sync bad
            ('max_drawdown', self._check_max_drawdown, True)      # Always process
        ]
        
        # Configuration
        self.stop_loss_points = self.config.get('stop_loss_points', 20)
        self.take_profit_points = self.config.get('take_profit_points', 40)
        self.trailing_stop_points = self.config.get('trailing_stop_points', 15)
        self.max_drawdown_points = self.config.get('max_drawdown_points', 30)
        self.time_stop_minutes = self.config.get('time_stop_minutes', 30)
        
        # Tracking
        self.exit_stats = {
            'hard_stop': 0,
            'take_profit': 0,
            'trailing_stop': 0,
            'reversal_pattern': 0,
            'time_stop': 0,
            'max_drawdown': 0
        }
        
    async def evaluate_exits(self, position: Dict, market_data: Dict, 
                            time_sync_ok: bool = True) -> Tuple[bool, Optional[str], Optional[str], Optional[float]]:
        """
        Check exits in priority order with idempotent protection
        
        Returns:
            (should_exit, exit_type, reason, exit_price)
        """
        
        # Prevent double exits
        if not await self.exit_once.try_arm():
            logger.debug("Exit already in progress - skipping evaluation")
            return False, None, None, None
            
        try:
            # Get current price
            current_price = self._round_tick(market_data.get('last_price', 0))
            
            if current_price <= 0:
                logger.error(f"Invalid price for exit evaluation: {current_price}")
                await self.exit_once.reset_for_new_position()
                return False, None, None, None
            
            # Check each exit condition in priority order
            for exit_name, check_func, always_check in self.exit_priorities:
                # Skip time-sensitive checks if sync is bad
                if not always_check and not time_sync_ok:
                    logger.debug(f"Skipping {exit_name} - time sync issue")
                    continue
                    
                should_exit, reason = await check_func(position, current_price, market_data)
                
                if should_exit:
                    logger.info(f"EXIT TRIGGERED: {exit_name} - {reason}")
                    self.exit_stats[exit_name] += 1
                    return True, exit_name, reason, current_price
                    
            # No exit triggered - reset gate
            await self.exit_once.reset_for_new_position()
            return False, None, None, None
            
        except Exception as e:
            logger.error(f"Exit evaluation error: {e}")
            # Reset on error to prevent lockup
            await self.exit_once.reset_for_new_position()
            return False, None, None, None
    
    async def _check_hard_stop(self, position: Dict, current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Check if hard stop loss is hit"""
        
        entry_price = self._round_tick(position.get('average_price', 0))
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return False, "No position"
        
        # Calculate stop level
        if quantity > 0:  # Long
            stop_price = entry_price - self.stop_loss_points
            if current_price <= stop_price:
                loss = (current_price - entry_price) * quantity
                return True, f"Hard stop hit at {current_price} (loss: {loss:.2f} points)"
        else:  # Short
            stop_price = entry_price + self.stop_loss_points
            if current_price >= stop_price:
                loss = (entry_price - current_price) * abs(quantity)
                return True, f"Hard stop hit at {current_price} (loss: {loss:.2f} points)"
        
        return False, ""
    
    async def _check_take_profit(self, position: Dict, current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Check if take profit target is hit"""
        
        entry_price = self._round_tick(position.get('average_price', 0))
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return False, "No position"
        
        # Calculate target level
        if quantity > 0:  # Long
            target_price = entry_price + self.take_profit_points
            if current_price >= target_price:
                profit = (current_price - entry_price) * quantity
                return True, f"Take profit hit at {current_price} (profit: {profit:.2f} points)"
        else:  # Short
            target_price = entry_price - self.take_profit_points
            if current_price <= target_price:
                profit = (entry_price - current_price) * abs(quantity)
                return True, f"Take profit hit at {current_price} (profit: {profit:.2f} points)"
        
        return False, ""
    
    async def _check_trailing_stop(self, position: Dict, current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Check if trailing stop is hit"""
        
        # Get position high water mark
        high_water_mark = position.get('high_water_mark', position.get('average_price', 0))
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return False, "No position"
        
        # Update high water mark
        if quantity > 0:  # Long
            if current_price > high_water_mark:
                position['high_water_mark'] = current_price
                high_water_mark = current_price
            
            # Check trailing stop
            trailing_stop = high_water_mark - self.trailing_stop_points
            if current_price <= trailing_stop:
                return True, f"Trailing stop hit at {current_price} (high: {high_water_mark})"
                
        else:  # Short
            if current_price < high_water_mark:
                position['high_water_mark'] = current_price
                high_water_mark = current_price
            
            # Check trailing stop
            trailing_stop = high_water_mark + self.trailing_stop_points
            if current_price >= trailing_stop:
                return True, f"Trailing stop hit at {current_price} (low: {high_water_mark})"
        
        return False, ""
    
    async def _check_reversal_pattern(self, position: Dict, current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Check for reversal patterns (time-sensitive)"""
        
        # This would integrate with pattern detection
        # For now, placeholder logic
        patterns = market_data.get('patterns', [])
        
        for pattern in patterns:
            if pattern.get('type') == 'reversal' and pattern.get('strength', 0) > 0.7:
                return True, f"Reversal pattern detected: {pattern.get('name')}"
        
        return False, ""
    
    async def _check_time_stop(self, position: Dict, current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Check if position has been open too long (time-sensitive)"""
        
        entry_time = position.get('entry_time')
        if not entry_time:
            return False, ""
        
        # Convert string to datetime if needed
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        time_in_position = (datetime.now() - entry_time).total_seconds() / 60  # minutes
        
        if time_in_position >= self.time_stop_minutes:
            return True, f"Time stop triggered after {time_in_position:.1f} minutes"
        
        return False, ""
    
    async def _check_max_drawdown(self, position: Dict, current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Check if maximum drawdown is exceeded"""
        
        entry_price = self._round_tick(position.get('average_price', 0))
        quantity = position.get('quantity', 0)
        
        if quantity == 0:
            return False, "No position"
        
        # Calculate current drawdown
        if quantity > 0:  # Long
            drawdown = entry_price - current_price
            if drawdown >= self.max_drawdown_points:
                return True, f"Max drawdown exceeded: {drawdown:.2f} points"
        else:  # Short
            drawdown = current_price - entry_price
            if drawdown >= self.max_drawdown_points:
                return True, f"Max drawdown exceeded: {drawdown:.2f} points"
        
        return False, ""
    
    def _round_tick(self, price: float) -> float:
        """Round to valid NQ tick (0.25)"""
        TICK = 0.25
        return round(price / TICK) * TICK
    
    async def force_exit(self, reason: str = "Manual"):
        """Force an immediate exit"""
        if await self.exit_once.try_arm():
            logger.warning(f"FORCED EXIT: {reason}")
            return True
        return False
    
    async def complete_exit(self):
        """Mark exit as complete"""
        await self.exit_once.complete_exit()
    
    async def reset_for_new_position(self):
        """Reset for new position"""
        return await self.exit_once.reset_for_new_position()
    
    def get_statistics(self) -> Dict:
        """Get exit statistics"""
        total_exits = sum(self.exit_stats.values())
        
        stats = {
            'total_exits': total_exits,
            'exit_breakdown': self.exit_stats.copy(),
            'gate_status': self.exit_once.get_status()
        }
        
        if total_exits > 0:
            stats['exit_percentages'] = {
                k: (v / total_exits * 100) for k, v in self.exit_stats.items()
            }
        
        return stats