# File: trading_bot/execution/atomic_orders.py
import asyncio
import uuid
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class OrderState(Enum):
    """Order lifecycle states"""
    CREATED = "created"
    VALIDATING = "validating"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order types"""
    ENTRY = "entry"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    FLATTEN = "flatten"

@dataclass
class OrderRequest:
    """Atomic order request"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # BUY/SELL
    size: int = 1
    order_type: OrderType = OrderType.ENTRY
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    pattern: str = ""
    confidence: float = 0
    max_slippage: float = 0.5
    timeout_seconds: int = 10
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    state: OrderState
    broker_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    slippage: float = 0
    commission: float = 0
    rejection_reason: Optional[str] = None
    retry_count: int = 0
    execution_time_ms: float = 0

class AtomicOrderManager:
    """
    Ensures atomic order execution with:
    1. State machine management
    2. Rollback on failure
    3. Idempotent retries
    4. Order lifecycle tracking
    """
    
    def __init__(self, bot, broker_client, order_gate):
        self.bot = bot
        self.broker = broker_client
        self.order_gate = order_gate
        
        # Order tracking
        self.active_orders: Dict[str, OrderResult] = {}
        self.order_history: List[OrderResult] = []
        self.pending_orders: Dict[str, OrderRequest] = {}
        
        # State machine
        self.current_order: Optional[OrderRequest] = None
        self.lock = asyncio.Lock()
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.enable_partial_fills = False
        
        # Metrics
        self.metrics = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'orders_cancelled': 0,
            'total_slippage': 0,
            'total_commission': 0,
            'avg_execution_time': 0
        }
    
    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit order with atomic guarantees
        
        Process:
        1. Validate order
        2. Check gate permissions
        3. Submit to broker
        4. Monitor fill
        5. Handle success/failure atomically
        """
        async with self.lock:
            start_time = time.time()
            
            # Initialize result
            result = OrderResult(
                order_id=request.order_id,
                state=OrderState.CREATED
            )
            
            try:
                logger.info(f"📝 ATOMIC ORDER: {request.order_id[:8]} - {request.side} {request.size} @ {request.entry_price:.2f}")
                
                # 1. Validate order
                result.state = OrderState.VALIDATING
                is_valid, error = await self._validate_order(request)
                if not is_valid:
                    result.state = OrderState.REJECTED
                    result.rejection_reason = error
                    logger.error(f"❌ Order validation failed: {error}")
                    return result
                
                # 2. Check order gate
                from .order_gate import OrderSignal
                signal = OrderSignal(
                    symbol=request.symbol,
                    side=request.side,
                    entry_price=request.entry_price,
                    pattern=request.pattern,
                    size=request.size,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit
                )
                
                can_place, reason, details = await self.order_gate.can_place_order(signal)
                if not can_place:
                    result.state = OrderState.REJECTED
                    result.rejection_reason = f"Gate blocked: {reason}"
                    logger.warning(f"⛔ Order blocked by gate: {reason}")
                    return result
                
                # 3. Submit order with retries
                result.state = OrderState.SUBMITTING
                self.pending_orders[request.order_id] = request
                
                for attempt in range(self.max_retries):
                    try:
                        # Submit to broker
                        broker_result = await self._submit_to_broker(request)
                        
                        if broker_result['success']:
                            result.state = OrderState.SUBMITTED
                            result.broker_order_id = broker_result.get('order_id')
                            
                            # 4. Monitor fill
                            fill_result = await self._monitor_fill(
                                request, 
                                result.broker_order_id,
                                timeout=request.timeout_seconds
                            )
                            
                            if fill_result['filled']:
                                result.state = OrderState.FILLED
                                result.fill_price = fill_result['fill_price']
                                result.fill_time = datetime.now()
                                result.slippage = abs(result.fill_price - request.entry_price)
                                result.commission = 2.52  # NQ commission
                                
                                # Update metrics
                                self._update_metrics(result, time.time() - start_time)
                                
                                # 5. Update position tracking
                                await self._update_position_state(request, result)
                                
                                logger.info(f"✅ ORDER FILLED: {request.order_id[:8]} @ {result.fill_price:.2f} (slippage: {result.slippage:.2f})")
                                break
                            else:
                                # Order not filled in time
                                await self._cancel_order(result.broker_order_id)
                                result.state = OrderState.EXPIRED
                                result.rejection_reason = "Fill timeout"
                                logger.warning(f"⏱️ Order expired: {request.order_id[:8]}")
                        else:
                            # Broker rejection
                            result.state = OrderState.REJECTED
                            result.rejection_reason = broker_result.get('error', 'Unknown')
                            result.retry_count = attempt + 1
                            
                            if attempt < self.max_retries - 1:
                                logger.warning(f"🔄 Retry {attempt + 1}/{self.max_retries} for order {request.order_id[:8]}")
                                await asyncio.sleep(self.retry_delay)
                            else:
                                logger.error(f"❌ Order failed after {self.max_retries} attempts: {result.rejection_reason}")
                                break
                    
                    except Exception as e:
                        logger.error(f"Order submission exception: {e}")
                        result.state = OrderState.FAILED
                        result.rejection_reason = str(e)
                        result.retry_count = attempt + 1
                        
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                
                # Clean up pending
                if request.order_id in self.pending_orders:
                    del self.pending_orders[request.order_id]
                
                # Record result
                result.execution_time_ms = (time.time() - start_time) * 1000
                self.order_history.append(result)
                
                # Handle failure atomically
                if result.state not in [OrderState.FILLED, OrderState.PARTIAL]:
                    await self._rollback_order(request, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Atomic order exception: {e}")
                result.state = OrderState.FAILED
                result.rejection_reason = str(e)
                result.execution_time_ms = (time.time() - start_time) * 1000
                self.order_history.append(result)
                
                # Ensure cleanup
                await self._rollback_order(request, result)
                return result
    
    async def _validate_order(self, request: OrderRequest) -> Tuple[bool, Optional[str]]:
        """Comprehensive order validation"""
        # Check required fields
        if not request.symbol or not request.side:
            return False, "Missing required fields"
        
        if request.side not in ['BUY', 'SELL']:
            return False, f"Invalid side: {request.side}"
        
        if request.size <= 0:
            return False, f"Invalid size: {request.size}"
        
        # Price validation
        if request.entry_price <= 0:
            return False, f"Invalid entry price: {request.entry_price}"
        
        # Risk validation
        if request.order_type == OrderType.ENTRY:
            if request.stop_loss <= 0:
                return False, "Entry order requires stop loss"
            
            # Validate stop distance
            stop_distance = abs(request.entry_price - request.stop_loss)
            if stop_distance < 1 or stop_distance > 50:  # NQ specific
                return False, f"Invalid stop distance: {stop_distance:.2f}"
        
        # Check position conflicts
        if self.bot.current_position and request.order_type == OrderType.ENTRY:
            return False, "Already in position"
        
        # Check for duplicate orders
        for order_id, pending in self.pending_orders.items():
            if (pending.symbol == request.symbol and 
                pending.side == request.side and
                abs(pending.entry_price - request.entry_price) < 0.5):
                return False, f"Duplicate order detected: {order_id[:8]}"
        
        return True, None
    
    async def _submit_to_broker(self, request: OrderRequest) -> Dict:
        """Submit order to broker with proper formatting"""
        try:
            # Determine order side (0=BUY, 1=SELL for TopstepX)
            order_side = 0 if request.side == 'BUY' else 1
            
            # Build order payload
            order_data = {
                "accountId": self.bot.account_id,
                "contractId": self.bot.contract_id,
                "orderType": 1,  # Market order
                "side": order_side,
                "quantity": request.size,
                "stopLoss": request.stop_loss,
                "takeProfit": request.take_profit
            }
            
            # Submit order
            response = await self.broker.request('POST', '/api/Order/placeOrder', order_data)
            
            if response and response.get('success'):
                return {
                    'success': True,
                    'order_id': response.get('orderId', str(uuid.uuid4()))
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Order submission failed')
                }
                
        except Exception as e:
            logger.error(f"Broker submission error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _monitor_fill(self, request: OrderRequest, broker_order_id: str, 
                           timeout: int = 10) -> Dict:
        """Monitor order fill status"""
        start_time = time.time()
        check_interval = 0.5
        
        while (time.time() - start_time) < timeout:
            try:
                # Check order status
                response = await self.broker.request('GET', f'/api/Order/status/{broker_order_id}')
                
                if response and response.get('success'):
                    status = response.get('status', '').upper()
                    
                    if status == 'FILLED':
                        return {
                            'filled': True,
                            'fill_price': response.get('fillPrice', request.entry_price),
                            'fill_time': response.get('fillTime')
                        }
                    elif status in ['CANCELLED', 'REJECTED', 'EXPIRED']:
                        return {
                            'filled': False,
                            'reason': status
                        }
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Fill monitoring error: {e}")
                await asyncio.sleep(check_interval)
        
        # Timeout reached
        return {
            'filled': False,
            'reason': 'timeout'
        }
    
    async def _cancel_order(self, broker_order_id: str) -> bool:
        """Cancel pending order"""
        try:
            response = await self.broker.request('POST', f'/api/Order/cancel/{broker_order_id}')
            return response and response.get('success', False)
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    async def _update_position_state(self, request: OrderRequest, result: OrderResult):
        """Update bot position state after successful fill"""
        if result.state != OrderState.FILLED:
            return
        
        if request.order_type == OrderType.ENTRY:
            # Create new position
            from ..intelligent_trading_bot_fixed_v2 import Position, BotState
            
            self.bot.current_position = Position(
                symbol=request.symbol,
                side=0 if request.side == 'BUY' else 1,
                position_type=1 if request.side == 'BUY' else 2,
                size=request.size,
                entry_price=result.fill_price,
                entry_time=result.fill_time,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                pattern=request.pattern,
                confidence=request.confidence,
                order_id=result.broker_order_id
            )
            
            self.bot.current_position_size = request.size
            self.bot.current_position_type = 1 if request.side == 'BUY' else 2
            self.bot.state = BotState.POSITION_OPEN
            
            logger.info(f"📊 Position opened: {request.side} {request.size} @ {result.fill_price:.2f}")
            
        elif request.order_type == OrderType.FLATTEN:
            # Clear position
            self.bot.current_position = None
            self.bot.current_position_size = 0
            self.bot.current_position_type = None
            from ..intelligent_trading_bot_fixed_v2 import BotState
            self.bot.state = BotState.READY
            
            logger.info(f"📊 Position closed @ {result.fill_price:.2f}")
    
    async def _rollback_order(self, request: OrderRequest, result: OrderResult):
        """Rollback failed order atomically"""
        logger.warning(f"🔄 Rolling back order {request.order_id[:8]}")
        
        # Cancel any pending broker orders
        if result.broker_order_id:
            await self._cancel_order(result.broker_order_id)
        
        # Reset bot state if needed
        if request.order_type == OrderType.ENTRY:
            from ..intelligent_trading_bot_fixed_v2 import BotState
            if self.bot.state == BotState.ORDER_PENDING:
                self.bot.state = BotState.READY
        
        # Clean up tracking
        if request.order_id in self.pending_orders:
            del self.pending_orders[request.order_id]
    
    def _update_metrics(self, result: OrderResult, execution_time: float):
        """Update execution metrics"""
        self.metrics['orders_submitted'] += 1
        
        if result.state == OrderState.FILLED:
            self.metrics['orders_filled'] += 1
            self.metrics['total_slippage'] += result.slippage
            self.metrics['total_commission'] += result.commission
        elif result.state == OrderState.REJECTED:
            self.metrics['orders_rejected'] += 1
        elif result.state == OrderState.CANCELLED:
            self.metrics['orders_cancelled'] += 1
        
        # Update average execution time
        avg = self.metrics['avg_execution_time']
        count = self.metrics['orders_submitted']
        self.metrics['avg_execution_time'] = ((avg * (count - 1)) + execution_time * 1000) / count
    
    async def flatten_position(self, reason: str = "manual") -> OrderResult:
        """Emergency position flatten"""
        if not self.bot.current_position:
            logger.warning("No position to flatten")
            return OrderResult(
                order_id=str(uuid.uuid4()),
                state=OrderState.REJECTED,
                rejection_reason="No position"
            )
        
        # Create flatten order
        request = OrderRequest(
            symbol=self.bot.symbol,
            side='SELL' if self.bot.current_position_type == 1 else 'BUY',
            size=self.bot.current_position_size,
            order_type=OrderType.FLATTEN,
            entry_price=0,  # Market order
            pattern=f"flatten_{reason}",
            timeout_seconds=5
        )
        
        logger.warning(f"🚨 EMERGENCY FLATTEN: {reason}")
        return await self.submit_order(request)
    
    def get_metrics(self) -> Dict:
        """Get execution metrics"""
        filled = self.metrics['orders_filled']
        if filled > 0:
            return {
                **self.metrics,
                'avg_slippage': self.metrics['total_slippage'] / filled,
                'avg_commission': self.metrics['total_commission'] / filled,
                'fill_rate': (filled / self.metrics['orders_submitted']) * 100 if self.metrics['orders_submitted'] > 0 else 0
            }
        return self.metrics
    
    def get_pending_orders(self) -> List[OrderRequest]:
        """Get list of pending orders"""
        return list(self.pending_orders.values())
    
    def has_pending_orders(self) -> bool:
        """Check if any orders are pending"""
        return len(self.pending_orders) > 0