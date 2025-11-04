"""
ExecutionManager - Advanced order execution for NQ trading bot
Handles STOP-LIMIT orders, retest entries, and OCO bracket management
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Constants
NQ_TICK = 0.25
USD_PER_TICK = 5.0
LIMIT_OFFSET_TICKS = 2
CANCEL_IF_RUNS_TICKS = 4
MAX_SLIPPAGE_TICKS = 2

class OrderState(Enum):
    """Order states"""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ExecutionManager:
    """
    Advanced execution manager for NQ trading
    - STOP-LIMIT orders with controlled slippage
    - Retest entry at trigger or 50% retracement
    - Cancel-if-runs logic
    - OCO bracket order management
    - TopStepX auto-bracket integration
    """
    
    def __init__(self, broker, data_cache, account_id: int):
        """
        Initialize ExecutionManager
        
        Args:
            broker: TopStepX client for order execution
            data_cache: DataCache for market data
            account_id: Trading account ID
        """
        self.broker = broker
        self.data_cache = data_cache
        self.account_id = account_id
        
        # Order tracking
        self.active_orders = {}
        self.pending_entries = {}
        self.filled_orders = {}
        
        # Position tracking
        self.current_position = None
        self.entry_order_id = None
        self.stop_order_id = None
        self.target_order_id = None
        
        # Timing metrics
        self.order_timings = {}
        
        logger.info(f"ExecutionManager initialized for account {account_id}")
    
    async def place_entry(self, signal: Dict) -> Optional[Dict]:
        """
        Place entry order based on signal
        
        Args:
            signal: Trading signal with entry, stop, target, action
            
        Returns:
            Order details or None if failed
        """
        try:
            start_time = time.time()
            
            # Extract signal parameters
            action = signal.get('action', 'BUY')
            entry_price = float(signal.get('entry_price'))
            stop_price = float(signal.get('stop_loss'))
            target_price = float(signal.get('take_profit'))
            contracts = signal.get('contracts', 1)
            pattern = signal.get('pattern_name', 'unknown')
            
            # Determine order side
            is_long = action in ['BUY', 'Buy', 'buy']
            
            # Get current market price
            current_price = self.data_cache.get_current_price()
            if not current_price:
                logger.error("No current price available")
                return None
            
            # Calculate STOP-LIMIT order prices
            if is_long:
                # Buy entry
                stop_trigger = entry_price
                limit_price = entry_price + (LIMIT_OFFSET_TICKS * NQ_TICK)
                cancel_threshold = entry_price + (CANCEL_IF_RUNS_TICKS * NQ_TICK)
                retest_level = entry_price - ((entry_price - stop_price) * 0.5)  # 50% retracement
            else:
                # Sell entry
                stop_trigger = entry_price
                limit_price = entry_price - (LIMIT_OFFSET_TICKS * NQ_TICK)
                cancel_threshold = entry_price - (CANCEL_IF_RUNS_TICKS * NQ_TICK)
                retest_level = entry_price + ((stop_price - entry_price) * 0.5)  # 50% retracement
            
            logger.info(f"ExecutionManager: Placing {action} STOP-LIMIT order")
            logger.info(f"  Pattern: {pattern}")
            logger.info(f"  Stop Trigger: {stop_trigger:.2f}")
            logger.info(f"  Limit Price: {limit_price:.2f}")
            logger.info(f"  Cancel if runs to: {cancel_threshold:.2f}")
            logger.info(f"  Retest level: {retest_level:.2f}")
            
            # Check if we should wait for retest
            if await self._should_wait_for_retest(current_price, entry_price, retest_level, is_long):
                logger.info("Waiting for retest of entry level...")
                # Store pending entry for monitoring
                self.pending_entries[pattern] = {
                    'signal': signal,
                    'stop_trigger': stop_trigger,
                    'limit_price': limit_price,
                    'cancel_threshold': cancel_threshold,
                    'retest_level': retest_level,
                    'is_long': is_long,
                    'timestamp': datetime.now(timezone.utc)
                }
                return {'status': 'pending_retest', 'pattern': pattern}
            
            # Place STOP-LIMIT order
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            order_side = OrderSide.BUY if is_long else OrderSide.SELL
            
            # Create STOP-LIMIT order
            order_params = {
                'symbol': 'NQ',
                'side': order_side,
                'quantity': contracts,
                'order_type': OrderType.STOP,  # STOP order with limit
                'stop_price': stop_trigger,
                'price': limit_price  # Changed from limit_price to price
            }
            
            # Place the order
            order = await self.broker.place_order(**order_params)
            
            if order:
                order_id = order.get('id')
                self.entry_order_id = order_id
                
                # Track order for cancel-if-runs monitoring
                self.active_orders[order_id] = {
                    'type': 'entry',
                    'cancel_threshold': cancel_threshold,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'is_long': is_long,
                    'placed_at': datetime.now(timezone.utc),
                    'pattern': pattern
                }
                
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"✅ Entry order placed in {elapsed:.1f}ms - Order ID: {order_id}")
                
                # Start monitoring for cancel-if-runs
                asyncio.create_task(self._monitor_cancel_if_runs(order_id))
                
                return order
            else:
                logger.error("Failed to place entry order")
                return None
                
        except Exception as e:
            logger.error(f"Error placing entry order: {e}")
            return None
    
    async def _should_wait_for_retest(self, current: float, entry: float, retest: float, is_long: bool) -> bool:
        """
        Check if we should wait for a retest of entry level
        
        Args:
            current: Current price
            entry: Entry trigger price
            retest: Retest level (50% retracement)
            is_long: True for long, False for short
            
        Returns:
            bool: True if should wait for retest
        """
        if is_long:
            # For longs, if price is above entry, wait for pullback
            if current > entry + (2 * NQ_TICK):
                return True
        else:
            # For shorts, if price is below entry, wait for bounce
            if current < entry - (2 * NQ_TICK):
                return True
        
        return False
    
    async def _monitor_cancel_if_runs(self, order_id: str):
        """
        Monitor order for cancel-if-runs condition
        
        Args:
            order_id: Order ID to monitor
        """
        try:
            order_info = self.active_orders.get(order_id)
            if not order_info:
                return
            
            cancel_threshold = order_info['cancel_threshold']
            is_long = order_info['is_long']
            
            logger.info(f"Monitoring order {order_id} for cancel-if-runs")
            
            while order_id in self.active_orders:
                # Get current price
                current = self.data_cache.get_current_price()
                if not current:
                    await asyncio.sleep(0.5)
                    continue
                
                # Check cancel condition
                if is_long:
                    if current >= cancel_threshold:
                        run_ticks = int((current - order_info['trigger_price']) / NQ_TICK)
                        limit_offset = int((cancel_threshold - order_info['trigger_price']) / NQ_TICK)
                        
                        # Add execution telemetry
                        try:
                            from ..pattern_config import TRACE
                            if TRACE.get('exec', False):
                                logger.info(f"EXEC_SKIP reason=\"cancel_if_runs\" run_ticks={run_ticks} limit_offset={limit_offset} confirm_close={current:.2f}")
                        except Exception:
                            pass
                        
                        logger.warning(f"Cancel-if-runs triggered! Price {current:.2f} >= {cancel_threshold:.2f}")
                        await self.cancel_order(order_id)
                        break
                else:
                    if current <= cancel_threshold:
                        run_ticks = int((order_info['trigger_price'] - current) / NQ_TICK) 
                        limit_offset = int((order_info['trigger_price'] - cancel_threshold) / NQ_TICK)
                        
                        # Add execution telemetry
                        try:
                            from ..pattern_config import TRACE
                            if TRACE.get('exec', False):
                                logger.info(f"EXEC_SKIP reason=\"cancel_if_runs\" run_ticks={run_ticks} limit_offset={limit_offset} confirm_close={current:.2f}")
                        except Exception:
                            pass
                        
                        logger.warning(f"Cancel-if-runs triggered! Price {current:.2f} <= {cancel_threshold:.2f}")
                        await self.cancel_order(order_id)
                        break
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
        except Exception as e:
            logger.error(f"Error monitoring cancel-if-runs: {e}")
    
    async def place_bracket_orders(self, fill_price: float, order_info: Dict) -> bool:
        """
        Place OCO bracket orders (stop loss and take profit)
        
        Args:
            fill_price: Entry fill price
            order_info: Original order information
            
        Returns:
            bool: Success status
        """
        try:
            start_time = time.time()
            
            stop_price = order_info['stop_price']
            target_price = order_info['target_price']
            is_long = order_info['is_long']
            
            logger.info(f"Placing OCO bracket orders...")
            logger.info(f"  Fill: {fill_price:.2f}")
            logger.info(f"  Stop: {stop_price:.2f}")
            logger.info(f"  Target: {target_price:.2f}")
            
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            # Calculate stop order side (opposite of position)
            stop_side = OrderSide.SELL if is_long else OrderSide.BUY
            target_side = stop_side  # Same side for exit
            
            # Place stop loss order
            stop_order = await self.broker.place_order(
                symbol='NQ',
                side=stop_side,
                quantity=1,
                order_type=OrderType.STOP,
                stop_price=stop_price,
                custom_tag='SL'  # Tag for identification
            )
            
            if stop_order:
                self.stop_order_id = stop_order.get('id')
                logger.info(f"✅ Stop loss placed - Order ID: {self.stop_order_id}")
            
            # Place take profit order
            target_order = await self.broker.place_order(
                symbol='NQ',
                side=target_side,
                quantity=1,
                order_type=OrderType.LIMIT,
                price=target_price,
                custom_tag='TP'  # Tag for identification
            )
            
            if target_order:
                self.target_order_id = target_order.get('id')
                logger.info(f"✅ Take profit placed - Order ID: {self.target_order_id}")
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✅ OCO brackets placed in {elapsed:.1f}ms")
            
            # Check TopStepX auto-bracket status
            if self.broker.use_brackets:
                logger.info(f"⚠️  TopStepX auto-brackets active (${self.broker.bracket_stop} stop, ${self.broker.bracket_profit} target)")
                logger.info("    These act as outer failsafe beyond our OCO orders")
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing bracket orders: {e}")
            return False
    
    async def handle_fill(self, order_id: str, fill_price: float):
        """
        Handle order fill event
        
        Args:
            order_id: Filled order ID
            fill_price: Fill price
        """
        try:
            if order_id == self.entry_order_id:
                # Entry filled - place brackets
                order_info = self.active_orders.get(order_id)
                if order_info:
                    # Add execution fill telemetry
                    try:
                        from ..pattern_config import TRACE
                        if TRACE.get('exec', False):
                            side = "long" if order_info['is_long'] else "short"
                            qty = order_info.get('quantity', 1)
                            stop_price = order_info['stop_price']
                            target_price = order_info['target_price'] 
                            target2_price = order_info.get('target2_price', target_price)
                            
                            logger.info(f"EXEC_FILL order={order_id} side={side} qty={qty} entry={fill_price:.2f} "
                                       f"oco(stop={stop_price:.2f},t1={target_price:.2f},t2={target2_price:.2f})")
                        
                        # Write to CSV telemetry
                        from ..utils.telemetry_sink import get_telemetry_sink
                        sink = get_telemetry_sink()
                        sink.write(
                            pattern=order_info.get('pattern', 'unknown'),
                            event="EXEC_FILL",
                            price=fill_price,
                            exec_reason="entry_fill",
                            stop_ticks=int(abs(stop_price - fill_price) / NQ_TICK),
                            t1_ticks=int(abs(target_price - fill_price) / NQ_TICK),
                            t2_ticks=int(abs(target2_price - fill_price) / NQ_TICK) if target2_price != target_price else None
                        )
                    except Exception:
                        pass
                    
                    logger.info(f"Entry filled at {fill_price:.2f}")
                    
                    # Place OCO brackets immediately
                    await self.place_bracket_orders(fill_price, order_info)
                    
                    # Update position tracking
                    self.current_position = {
                        'entry_price': fill_price,
                        'stop_price': order_info['stop_price'],
                        'target_price': order_info['target_price'],
                        'is_long': order_info['is_long'],
                        'pattern': order_info['pattern'],
                        'entry_time': datetime.now(timezone.utc)
                    }
                    
                    # Move to filled orders
                    self.filled_orders[order_id] = order_info
                    del self.active_orders[order_id]
                    
            elif order_id == self.stop_order_id:
                # Stop loss hit
                logger.info(f"Stop loss filled at {fill_price:.2f}")
                await self.cancel_order(self.target_order_id)  # Cancel take profit
                
                # Write EXIT telemetry
                try:
                    from ..pattern_config import TELEMETRY
                    if TELEMETRY.get('csv_exec', False) and self.current_position:
                        from ..utils.telemetry_sink import get_telemetry_sink
                        sink = get_telemetry_sink()
                        
                        # Calculate P&L and MAE
                        entry_price = self.current_position.get('entry_price', fill_price)
                        is_long = self.current_position.get('is_long', True)
                        pnl_ticks = (fill_price - entry_price) / NQ_TICK if is_long else (entry_price - fill_price) / NQ_TICK
                        
                        sink.write(
                            pattern=self.current_position.get('pattern', 'unknown'),
                            event="EXIT",
                            price=fill_price,
                            exec_reason="stop_loss",
                            slippage_ticks=0,  # TODO: calculate actual slippage
                            mae_30s=pnl_ticks  # Negative since stop hit
                        )
                except Exception as e:
                    logger.debug(f"Exit telemetry write failed: {e}")
                
                self.current_position = None
                
            elif order_id == self.target_order_id:
                # Take profit hit
                logger.info(f"Take profit filled at {fill_price:.2f}")
                await self.cancel_order(self.stop_order_id)  # Cancel stop loss
                
                # Write EXIT telemetry
                try:
                    from ..pattern_config import TELEMETRY
                    if TELEMETRY.get('csv_exec', False) and self.current_position:
                        from ..utils.telemetry_sink import get_telemetry_sink
                        sink = get_telemetry_sink()
                        
                        # Calculate time to target
                        entry_time = self.current_position.get('entry_time')
                        if entry_time:
                            time_to_t1 = (datetime.now(timezone.utc) - entry_time).total_seconds()
                        else:
                            time_to_t1 = None
                        
                        sink.write(
                            pattern=self.current_position.get('pattern', 'unknown'),
                            event="EXIT",
                            price=fill_price,
                            exec_reason="take_profit",
                            time_to_t1_s=time_to_t1
                        )
                except Exception as e:
                    logger.debug(f"Exit telemetry write failed: {e}")
                
                self.current_position = None
                
        except Exception as e:
            logger.error(f"Error handling fill: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: Success status
        """
        try:
            if not order_id:
                return False
            
            result = await self.broker.cancel_order(order_id)
            
            if result:
                logger.info(f"Order {order_id} cancelled")
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def check_pending_retests(self):
        """Check pending entries for retest conditions"""
        for pattern, pending in list(self.pending_entries.items()):
            current = self.data_cache.get_current_price()
            if not current:
                continue
            
            retest_level = pending['retest_level']
            is_long = pending['is_long']
            
            # Check if retest level reached
            if is_long:
                if current <= retest_level:
                    logger.info(f"Retest level reached for {pattern} at {current:.2f}")
                    # Place the order
                    await self.place_entry(pending['signal'])
                    del self.pending_entries[pattern]
            else:
                if current >= retest_level:
                    logger.info(f"Retest level reached for {pattern} at {current:.2f}")
                    # Place the order
                    await self.place_entry(pending['signal'])
                    del self.pending_entries[pattern]
            
            # Check timeout (5 minutes)
            age = (datetime.now(timezone.utc) - pending['timestamp']).total_seconds()
            if age > 300:
                logger.info(f"Pending entry for {pattern} timed out")
                del self.pending_entries[pattern]
    
    async def close_position(self):
        """Close current position at market"""
        try:
            if not self.current_position:
                logger.warning("No position to close")
                return False
            
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            # Determine close side
            is_long = self.current_position['is_long']
            close_side = OrderSide.SELL if is_long else OrderSide.BUY
            
            # Cancel pending orders
            if self.stop_order_id:
                await self.cancel_order(self.stop_order_id)
            if self.target_order_id:
                await self.cancel_order(self.target_order_id)
            
            # Place market order to close
            close_order = await self.broker.place_order(
                symbol='NQ',
                side=close_side,
                quantity=1,
                order_type=OrderType.MARKET,
                custom_tag='EXIT'
            )
            
            if close_order:
                logger.info(f"✅ Position closed - Order ID: {close_order.get('id')}")
                self.current_position = None
                self.stop_order_id = None
                self.target_order_id = None
                return True
            else:
                logger.error("Failed to close position")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_position_info(self) -> Optional[Dict]:
        """Get current position information"""
        return self.current_position
    
    def get_active_orders(self) -> Dict:
        """Get all active orders"""
        return self.active_orders.copy()
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            'active_orders': len(self.active_orders),
            'pending_retests': len(self.pending_entries),
            'filled_orders': len(self.filled_orders),
            'has_position': self.current_position is not None,
            'order_timings': self.order_timings
        }