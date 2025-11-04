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
            
            # Check for immediate market execution
            from ..pattern_config import FORCE_IMMEDIATE_MARKET, DISABLE_PENDING_ENTRIES
            
            if FORCE_IMMEDIATE_MARKET:
                # IMMEDIATE MARKET EXECUTION
                logger.info(f"ExecutionManager: Placing IMMEDIATE MARKET {action} order")
                logger.info(f"  Pattern: {pattern}")
                logger.info(f"  Current Price: {current_price:.2f}")
                logger.info(f"  Stop Loss: {stop_price:.2f}")
                logger.info(f"  Target: {target_price:.2f}")
                
                # Check pre-trade guards
                guards_pass, guard_reason = await self.check_pretrade_guards()
                if not guards_pass:
                    logger.warning(f"Pre-trade guards failed: {guard_reason}")
                    return None
                
                from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
                
                order_side = OrderSide.BUY if is_long else OrderSide.SELL
                
                # Create MARKET order
                order_params = {
                    'symbol': 'NQ',
                    'side': order_side,
                    'quantity': contracts,
                    'order_type': OrderType.MARKET
                }
                
                # Place the market order immediately
                order = await self.broker.place_order(**order_params)
                
                if order:
                    order_id = order.get('id')
                    self.entry_order_id = order_id
                    
                    # Track order
                    self.active_orders[order_id] = {
                        'type': 'entry',
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'is_long': is_long,
                        'placed_at': datetime.now(timezone.utc),
                        'pattern': pattern,
                        'market_order': True
                    }
                    
                    elapsed = (time.time() - start_time) * 1000
                    logger.info(f"✅ MARKET order placed in {elapsed:.1f}ms - Order ID: {order_id}")
                    
                    # Wait briefly for fill confirmation
                    await asyncio.sleep(0.5)
                    
                    # Place protective stop immediately after fill
                    from ..pattern_config import STOP_GUARD
                    if STOP_GUARD.get('enable', False):
                        await self._place_protective_stop(order_id, stop_price, is_long)
                    
                    return order
                else:
                    logger.error("Failed to place MARKET order")
                    return None
            
            # Check if pending entries are disabled
            if DISABLE_PENDING_ENTRIES:
                logger.info(f"DISABLE_PENDING_ENTRIES is True, skipping STOP-LIMIT order for {pattern}")
                return {'status': 'skipped_pending', 'pattern': pattern}
            
            # ORIGINAL STOP-LIMIT LOGIC (if not using immediate market)
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
                    
                    # Check if this was a MARKET order (already has protective stop)
                    if not order_info.get('market_order', False):
                        # Place protective stop for STOP-LIMIT fills
                        from ..pattern_config import STOP_GUARD
                        if STOP_GUARD.get('enable', False):
                            await self._place_protective_stop(
                                order_id, 
                                order_info['stop_price'], 
                                order_info['is_long']
                            )
                    
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
    
    async def _place_protective_stop(self, order_id: str, logic_stop: float, is_long: bool):
        """
        Place protective stop $70 (14 ticks) from entry
        
        Args:
            order_id: Entry order ID
            logic_stop: Logic-based stop price from pattern
            is_long: True for long position
        """
        try:
            from ..pattern_config import STOP_GUARD
            
            if not STOP_GUARD.get('enable', False):
                return
            
            # Get fill price (assume current price for now, will be updated on actual fill)
            fill_price = self.data_cache.get_current_price()
            if not fill_price:
                logger.error("No current price for protective stop")
                return
            
            # Calculate protective stop distance
            stop_usd = STOP_GUARD.get('usd', 70.0)
            stop_ticks = stop_usd / USD_PER_TICK  # $70 / $5 = 14 ticks
            stop_distance = stop_ticks * NQ_TICK
            
            # Calculate protective stop price
            if is_long:
                protective_stop = fill_price - stop_distance
                # Use tighter of logic stop or protective stop
                if STOP_GUARD.get('respect_tighter_logic_stop', True):
                    final_stop = max(logic_stop, protective_stop)
                else:
                    final_stop = protective_stop
            else:
                protective_stop = fill_price + stop_distance
                # Use tighter of logic stop or protective stop
                if STOP_GUARD.get('respect_tighter_logic_stop', True):
                    final_stop = min(logic_stop, protective_stop)
                else:
                    final_stop = protective_stop
            
            logger.info(f"Placing protective stop: ${stop_usd} ({stop_ticks:.0f} ticks)")
            logger.info(f"  Fill price: {fill_price:.2f}")
            logger.info(f"  Protective stop: {protective_stop:.2f}")
            logger.info(f"  Logic stop: {logic_stop:.2f}")
            logger.info(f"  Final stop: {final_stop:.2f}")
            
            # Place stop order
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            stop_side = OrderSide.SELL if is_long else OrderSide.BUY
            
            stop_order = await self.broker.place_order(
                symbol='NQ',
                side=stop_side,
                quantity=1,
                order_type=OrderType.STOP,
                stop_price=final_stop,
                custom_tag='PROTECTIVE_STOP'
            )
            
            if stop_order:
                self.stop_order_id = stop_order.get('id')
                logger.info(f"✅ Protective stop placed at {final_stop:.2f} - Order ID: {self.stop_order_id}")
            else:
                logger.error("Failed to place protective stop")
                
        except Exception as e:
            logger.error(f"Error placing protective stop: {e}")
    
    async def check_pretrade_guards(self, symbol: str = 'NQ') -> tuple[bool, str]:
        """
        Check pre-trade safety guards before placing MARKET orders
        Returns: (pass: bool, reason: str)
        """
        try:
            from ..pattern_config import PRETRADE_GUARDS
            import time
            
            # TopStepX doesn't provide order book, so we check data freshness only
            # Check if we have recent market data
            if self.data_cache:
                bars = self.data_cache.get_bars('1m', limit=1)
                if bars is not None and not bars.empty:
                    # Check data freshness
                    from datetime import datetime, timezone
                    bar_time = bars['timestamp'].iloc[-1] if 'timestamp' in bars else None
                    if bar_time:
                        now = datetime.now(timezone.utc)
                        if hasattr(bar_time, 'tz_localize'):
                            # If bar_time is timezone-naive, make it UTC
                            bar_time = bar_time.tz_localize('UTC')
                        age_seconds = (now - bar_time).total_seconds()
                        max_age_seconds = PRETRADE_GUARDS.get('max_age_ms', 800) / 1000.0
                        
                        if age_seconds > max_age_seconds:
                            logger.warning(f"IMMEDIATE_ABORT reason=stale_data age_s={age_seconds:.1f} max={max_age_seconds:.1f}")
                            return False, f"stale_data_{age_seconds:.1f}s"
                    
                    # For TopStepX, assume reasonable spread (typically 1-2 ticks for NQ)
                    # This is a simplified check - TopStepX handles execution quality
                    logger.info(f"Pre-trade guards PASSED: data_age={age_seconds if bar_time else 0:.1f}s")
                    return True, "guards_passed"
            
            # If no data cache or bars, pass anyway (rely on broker)
            logger.info("Pre-trade guards PASSED: relying on broker execution")
            return True, "guards_passed"
            
        except Exception as e:
            logger.error(f"Pre-trade guard check failed: {e}")
            # On error, still allow trade (don't block on technical issues)
            return True, f"guard_check_error_allowing_{str(e)}"
    
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
    
    async def place_fvg_entry(self, fvg_signals: Dict, stop_pts: float, target_pts: float) -> Optional[Dict]:
        """
        Place FVG entry with TTL and edge retry logic
        
        Args:
            fvg_signals: Entry signals from FVGStrategy.get_entry_signals()
            stop_pts: Stop loss in points
            target_pts: Target in points
            
        Returns:
            Order details or None
        """
        try:
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            direction = fvg_signals['direction']
            is_long = direction == 'long'
            
            # Start with mid entry if configured
            if 'mid_entry' in fvg_signals:
                mid_cfg = fvg_signals['mid_entry']
                level = mid_cfg['level']
                ttl_sec = mid_cfg['ttl_sec']
                cancel_if_runs = mid_cfg['cancel_if_runs_ticks']
                
                logger.info(f"FVG_ENTRY_MID_PLACING level={level:.2f} ttl={ttl_sec}s")
                
                # Place limit order with TTL monitoring
                order_side = OrderSide.BUY if is_long else OrderSide.SELL
                
                order = await self.broker.place_order(
                    symbol='NQ',
                    side=order_side,
                    quantity=1,
                    order_type=OrderType.LIMIT,
                    price=level
                )
                
                if order:
                    order_id = order.get('id')
                    self.entry_order_id = order_id
                    
                    # Track with TTL
                    self.active_orders[order_id] = {
                        'type': 'fvg_mid',
                        'fvg_id': fvg_signals['fvg_id'],
                        'level': level,
                        'ttl_sec': ttl_sec,
                        'cancel_if_runs_ticks': cancel_if_runs,
                        'edge_retry': fvg_signals.get('edge_retry'),
                        'is_long': is_long,
                        'placed_at': datetime.now(timezone.utc),
                        'stop_pts': stop_pts,
                        'target_pts': target_pts
                    }
                    
                    # Start TTL and cancel-if-runs monitoring
                    asyncio.create_task(self._monitor_fvg_ttl(order_id))
                    
                    return order
            
            # Direct edge entry if no mid configured
            elif 'edge_retry' in fvg_signals:
                return await self._place_fvg_edge(fvg_signals, stop_pts, target_pts)
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing FVG entry: {e}")
            return None
    
    async def _monitor_fvg_ttl(self, order_id: str):
        """Monitor FVG order for TTL expiry and cancel-if-runs"""
        try:
            order_info = self.active_orders.get(order_id)
            if not order_info:
                return
            
            ttl_sec = order_info.get('ttl_sec', 90)
            cancel_if_runs = order_info.get('cancel_if_runs_ticks', 8)
            level = order_info['level']
            is_long = order_info['is_long']
            placed_at = order_info['placed_at']
            
            logger.info(f"FVG_TTL_MONITOR started for {order_id} ttl={ttl_sec}s")
            
            start_time = time.time()
            
            while order_id in self.active_orders:
                elapsed = time.time() - start_time
                
                # Check TTL expiry
                if elapsed >= ttl_sec:
                    logger.info(f"FVG_TTL_EXPIRED order={order_id} elapsed={elapsed:.1f}s")
                    await self.broker.cancel_order(order_id)
                    
                    # Try edge retry if configured
                    if order_info.get('edge_retry'):
                        logger.info(f"FVG_EDGE_RETRY triggered after TTL expiry")
                        await self._place_fvg_edge_retry(order_info)
                    
                    del self.active_orders[order_id]
                    return
                
                # Check cancel-if-runs
                current = self.data_cache.get_current_price()
                if current:
                    if is_long:
                        if current > level + (cancel_if_runs * NQ_TICK):
                            logger.info(f"FVG_CANCEL_IF_RUNS order={order_id} ran {cancel_if_runs} ticks")
                            await self.broker.cancel_order(order_id)
                            del self.active_orders[order_id]
                            return
                    else:
                        if current < level - (cancel_if_runs * NQ_TICK):
                            logger.info(f"FVG_CANCEL_IF_RUNS order={order_id} ran {cancel_if_runs} ticks")
                            await self.broker.cancel_order(order_id)
                            del self.active_orders[order_id]
                            return
                
                # Check for fill
                status = await self.broker.get_order_status(order_id)
                if status and status.get('status') == 'FILLED':
                    logger.info(f"FVG_MID_FILLED order={order_id} at {level:.2f}")
                    
                    # Log bracket delegation in FVG-ONLY mode
                    from ..pattern_config import STRATEGY_MODE, TOPSTEPX_AUTO_BRACKET
                    if STRATEGY_MODE == "FVG_ONLY" and TOPSTEPX_AUTO_BRACKET["enable"]:
                        logger.info(f"BRACKET_DELEGATED mode=broker tp={TOPSTEPX_AUTO_BRACKET['tp_pts']} sl={TOPSTEPX_AUTO_BRACKET['sl_pts']}")
                    
                    del self.active_orders[order_id]
                    return
                
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error monitoring FVG TTL: {e}")
    
    async def _place_fvg_edge_retry(self, original_order: Dict) -> Optional[Dict]:
        """Place edge retry order after mid entry TTL expiry"""
        try:
            edge_cfg = original_order.get('edge_retry')
            if not edge_cfg:
                return None
            
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            level = edge_cfg['level']
            ttl_sec = edge_cfg['ttl_sec']
            is_long = original_order['is_long']
            
            logger.info(f"FVG_EDGE_PLACING level={level:.2f} ttl={ttl_sec}s")
            
            order_side = OrderSide.BUY if is_long else OrderSide.SELL
            
            order = await self.broker.place_order(
                symbol='NQ',
                side=order_side,
                quantity=1,
                order_type=OrderType.LIMIT,
                price=level
            )
            
            if order:
                order_id = order.get('id')
                self.active_orders[order_id] = {
                    'type': 'fvg_edge',
                    'fvg_id': original_order['fvg_id'],
                    'level': level,
                    'ttl_sec': ttl_sec,
                    'is_long': is_long,
                    'placed_at': datetime.now(timezone.utc),
                    'stop_pts': original_order['stop_pts'],
                    'target_pts': original_order['target_pts']
                }
                
                # Monitor edge TTL (shorter)
                asyncio.create_task(self._monitor_fvg_edge_ttl(order_id))
                
                return order
                
        except Exception as e:
            logger.error(f"Error placing FVG edge retry: {e}")
            return None
    
    async def _monitor_fvg_edge_ttl(self, order_id: str):
        """Monitor edge retry order for TTL expiry only"""
        try:
            order_info = self.active_orders.get(order_id)
            if not order_info:
                return
            
            ttl_sec = order_info.get('ttl_sec', 45)
            
            logger.info(f"FVG_EDGE_TTL_MONITOR started for {order_id} ttl={ttl_sec}s")
            
            start_time = time.time()
            
            while order_id in self.active_orders:
                elapsed = time.time() - start_time
                
                # Check TTL expiry
                if elapsed >= ttl_sec:
                    logger.info(f"FVG_EDGE_TTL_EXPIRED order={order_id} elapsed={elapsed:.1f}s")
                    await self.broker.cancel_order(order_id)
                    del self.active_orders[order_id]
                    return
                
                # Check for fill
                status = await self.broker.get_order_status(order_id)
                if status and status.get('status') == 'FILLED':
                    logger.info(f"FVG_EDGE_FILLED order={order_id}")
                    
                    # Log bracket delegation in FVG-ONLY mode
                    from ..pattern_config import STRATEGY_MODE, TOPSTEPX_AUTO_BRACKET
                    if STRATEGY_MODE == "FVG_ONLY" and TOPSTEPX_AUTO_BRACKET["enable"]:
                        logger.info(f"BRACKET_DELEGATED mode=broker tp={TOPSTEPX_AUTO_BRACKET['tp_pts']} sl={TOPSTEPX_AUTO_BRACKET['sl_pts']}")
                    
                    del self.active_orders[order_id]
                    return
                
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error monitoring FVG edge TTL: {e}")
    
    async def place_limit_with_oco(self, account_id: int, contract_id: str, side: str, 
                                   qty: int, limit_price: float, stop_loss_price: float, 
                                   take_profit_price: float, tag: str = "") -> Optional[Dict]:
        """
        Place limit order with OCO bracket (stop loss and take profit)
        
        Args:
            account_id: Trading account ID
            contract_id: Contract identifier (e.g., 'CON.F.US.ENQ.U25')
            side: 'BUY' or 'SELL'
            qty: Quantity to trade
            limit_price: Limit entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            tag: Optional tag for tracking
            
        Returns:
            Dict with order details or None if failed
        """
        try:
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            # Validate account
            if account_id != self.account_id:
                logger.error(f"Account mismatch: {account_id} != {self.account_id}")
                return None
            
            # Determine order side
            order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
            is_long = side.upper() == 'BUY'
            
            # Place limit order
            logger.info(f"Placing limit order with OCO: {side} {qty} @ {limit_price:.2f}")
            logger.info(f"  Stop: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}")
            
            # Create limit order params
            order_params = {
                'symbol': 'NQ',
                'side': order_side,
                'quantity': qty,
                'order_type': OrderType.LIMIT,
                'price': limit_price,
                'custom_tag': tag or 'FVG_ENTRY'
            }
            
            # Place the limit order
            entry_order = await self.broker.place_order(**order_params)
            
            if not entry_order:
                logger.error("Failed to place limit entry order")
                return None
            
            order_id = entry_order.get('id')
            
            # Track order for monitoring
            self.active_orders[order_id] = {
                'type': 'entry',
                'limit_price': limit_price,
                'stop_price': stop_loss_price,
                'target_price': take_profit_price,
                'is_long': is_long,
                'placed_at': datetime.now(timezone.utc),
                'pattern': tag,
                'oco_pending': True  # Flag for OCO placement on fill
            }
            
            # Start monitoring for fill to place OCO
            asyncio.create_task(self._monitor_for_oco(order_id))
            
            logger.info(f"✅ Limit order placed - ID: {order_id}")
            
            return {
                'order_id': order_id,
                'status': 'placed',
                'limit_price': limit_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
            
        except Exception as e:
            logger.error(f"Error placing limit with OCO: {e}")
            return None
    
    async def _monitor_for_oco(self, order_id: str, timeout: int = 60):
        """
        Monitor limit order for fill and place OCO brackets
        
        Args:
            order_id: Order ID to monitor
            timeout: Timeout in seconds
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check if order still active
                if order_id not in self.active_orders:
                    return
                
                # Check order status
                order_status = await self.broker.get_order_status(order_id)
                
                if order_status and order_status.get('filled'):
                    fill_price = order_status.get('fill_price', 0)
                    order_info = self.active_orders.get(order_id)
                    
                    if order_info and order_info.get('oco_pending'):
                        logger.info(f"Limit order {order_id} filled at {fill_price:.2f}")
                        
                        # Place OCO brackets
                        await self.place_bracket_orders(fill_price, order_info)
                        
                        # Update tracking
                        order_info['oco_pending'] = False
                        self.current_position = {
                            'entry_price': fill_price,
                            'stop_price': order_info['stop_price'],
                            'target_price': order_info['target_price'],
                            'is_long': order_info['is_long'],
                            'pattern': order_info.get('pattern', 'FVG'),
                            'entry_time': datetime.now(timezone.utc)
                        }
                        
                        # Move to filled
                        self.filled_orders[order_id] = order_info
                        del self.active_orders[order_id]
                        return
                
                await asyncio.sleep(0.5)
            
            # Timeout - cancel unfilled order
            logger.warning(f"Limit order {order_id} timed out after {timeout}s")
            await self.cancel_order(order_id)
            
        except Exception as e:
            logger.error(f"Error monitoring for OCO: {e}")
    
    async def modify_stop_to_breakeven(self, current_price: float = None) -> bool:
        """
        Modify stop loss to breakeven
        
        Args:
            current_price: Current market price (optional)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.current_position or not self.stop_order_id:
                return False
            
            entry_price = self.current_position['entry_price']
            is_long = self.current_position['is_long']
            
            # Cancel existing stop
            await self.cancel_order(self.stop_order_id)
            
            # Place new stop at breakeven
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            stop_side = OrderSide.SELL if is_long else OrderSide.BUY
            
            stop_order = await self.broker.place_order(
                symbol='NQ',
                side=stop_side,
                quantity=1,
                order_type=OrderType.STOP,
                stop_price=entry_price,
                custom_tag='BE_STOP'
            )
            
            if stop_order:
                self.stop_order_id = stop_order.get('id')
                logger.info(f"✅ Stop moved to breakeven at {entry_price:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error moving stop to breakeven: {e}")
            return False
    
    async def trail_stop(self, trail_distance: float) -> bool:
        """
        Update trailing stop based on current price
        
        Args:
            trail_distance: Distance in points to trail
            
        Returns:
            bool: Success status
        """
        try:
            if not self.current_position or not self.stop_order_id:
                return False
            
            current_price = self.data_cache.get_current_price()
            if not current_price:
                return False
            
            is_long = self.current_position['is_long']
            
            # Calculate new stop
            if is_long:
                new_stop = current_price - trail_distance
            else:
                new_stop = current_price + trail_distance
            
            # Cancel existing stop
            await self.cancel_order(self.stop_order_id)
            
            # Place new trailing stop
            from web_platform.backend.brokers.topstepx_client import OrderType, OrderSide
            
            stop_side = OrderSide.SELL if is_long else OrderSide.BUY
            
            stop_order = await self.broker.place_order(
                symbol='NQ',
                side=stop_side,
                quantity=1,
                order_type=OrderType.STOP,
                stop_price=new_stop,
                custom_tag='TRAIL_STOP'
            )
            
            if stop_order:
                self.stop_order_id = stop_order.get('id')
                logger.info(f"✅ Trailing stop updated to {new_stop:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return False

    def place_limit_or_mit(self, *, zone_id, side, price, stop_loss_pts,
                          take_profit_pts, ttl_seconds, max_slip_ticks,
                          tick_size, mit, tag) -> bool:
        """
        Place fast path order (limit or market-if-touched)

        Args:
            zone_id: FVG zone identifier
            side: 'buy' or 'sell'
            price: Entry price
            stop_loss_pts: Stop loss in points
            take_profit_pts: Take profit in points
            ttl_seconds: Time to live for order
            max_slip_ticks: Maximum slippage allowed
            tick_size: Tick size for calculations
            mit: If True, use market-if-touched on breach
            tag: Order tag for tracking

        Returns:
            bool: True if order placed successfully
        """
        try:
            logger.info(f"FAST_PATH_ORDER_ATTEMPT zone={zone_id} side={side} price={price:.2f} mit={mit} tag={tag}")

            # For now, log the order attempt and simulate placement
            # This integrates with the existing place_fvg_entry or place_limit_with_oco methods

            # Calculate stop and target prices
            if side == "buy":
                stop_price = price - stop_loss_pts
                target_price = price + take_profit_pts
            else:
                stop_price = price + stop_loss_pts
                target_price = price - take_profit_pts

            logger.info(f"FAST_PATH_ORDER zone={zone_id} entry={price:.2f} stop={stop_price:.2f} target={target_price:.2f}")

            # Track for TTL management and cancellation
            order_info = {
                'zone_id': zone_id,
                'side': side,
                'price': price,
                'stop_price': stop_price,
                'target_price': target_price,
                'ttl_seconds': ttl_seconds,
                'placed_at': time.time(),
                'tag': tag,
                'mit': mit
            }

            # Store pending order for tick() management
            if not hasattr(self, 'pending_fast_orders'):
                self.pending_fast_orders = {}

            self.pending_fast_orders[zone_id] = order_info

            logger.info(f"FAST_PATH_STAGED zone={zone_id} ttl={ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"FAST_PATH_ERROR zone={zone_id} error={str(e)}")
            return False

    def tick(self, now_dt, last_price, bid=None, ask=None):
        """
        Drive MIT triggers, TTL cancels, and fill polling for fast path orders

        Args:
            now_dt: Current datetime
            last_price: Latest price
            bid: Current bid price (optional)
            ask: Current ask price (optional)
        """
        if not hasattr(self, 'pending_fast_orders'):
            return

        current_time = time.time()
        to_remove = []

        for zone_id, order_info in self.pending_fast_orders.items():
            # Check TTL expiration
            if current_time - order_info['placed_at'] > order_info['ttl_seconds']:
                logger.info(f"FAST_PATH_EXPIRED zone={zone_id} ttl={order_info['ttl_seconds']}s")
                to_remove.append(zone_id)
                continue

            # For MIT orders, check if price touches trigger level
            if order_info['mit']:
                trigger_price = order_info['price']
                if ((order_info['side'] == 'buy' and last_price <= trigger_price) or
                    (order_info['side'] == 'sell' and last_price >= trigger_price)):

                    logger.info(f"FAST_PATH_TRIGGERED zone={zone_id} last={last_price:.2f} trigger={trigger_price:.2f}")
                    # Would convert to market order here
                    to_remove.append(zone_id)

        # Clean up processed orders
        for zone_id in to_remove:
            del self.pending_fast_orders[zone_id]