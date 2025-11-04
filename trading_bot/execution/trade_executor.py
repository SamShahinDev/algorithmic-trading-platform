"""
Trade Executor for Order Management
Handles order placement, modification, cancellation, and execution tracking
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

from brokers.topstepx_client import topstepx_client


class OrderType(Enum):
    """Order types"""
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4
    TRAILING_STOP = 5


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side"""
    BUY = 0   # TopStep format
    SELL = 1  # TopStep format


@dataclass
class Order:
    """Order details"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    
    # Execution details
    filled_quantity: int = 0
    average_fill_price: float = 0
    fill_time: Optional[datetime] = None
    
    # Metadata
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    
    # Risk management
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    
    # Tracking
    tags: Dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class ExecutionReport:
    """Execution report for filled orders"""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    execution_time: datetime
    liquidity_indicator: str  # maker/taker
    exchange: str
    notes: str = ""


class TradeExecutor:
    """Advanced trade execution and order management system"""
    
    def __init__(self, 
                 account_id: int = 10983875,
                 mode: str = "paper",
                 max_slippage: float = 0.5,
                 retry_attempts: int = 3):
        """
        Initialize trade executor
        
        Args:
            account_id: TopStep account ID
            mode: Trading mode (paper/live)
            max_slippage: Maximum allowed slippage in points
            retry_attempts: Number of retry attempts for failed orders
        """
        self.account_id = account_id
        self.mode = mode
        self.max_slippage = max_slippage
        self.retry_attempts = retry_attempts
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.executions = []  # List of ExecutionReport
        self.pending_orders = {}  # Orders waiting to be submitted
        self.working_orders = {}  # Active orders in market
        
        # Position tracking
        self.positions = {}  # symbol -> position details
        self.position_version = 0  # Version tracking for position updates
        
        # Execution callbacks
        self.on_order_filled: Optional[Callable] = None
        self.on_order_rejected: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        
        # Performance metrics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.cancelled_orders = 0
        self.total_slippage = 0
        self.total_commission = 0
        
    async def submit_order(self,
                          symbol: str,
                          side: str,
                          quantity: int,
                          order_type: str = "market",
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          time_in_force: str = "day",
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          tags: Optional[Dict] = None) -> str:
        """
        Submit an order with advanced features
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            order_type: Order type (market, limit, stop, stop_limit)
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order time in force (day, gtc, ioc, fok)
            stop_loss: Stop loss price for bracket order
            take_profit: Take profit price for bracket order
            tags: Custom tags for tracking
            
        Returns:
            Order ID
        """
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Map order parameters
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        order_type_enum = self._map_order_type(order_type)
        
        # Create order object
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=order_side,
            order_type=order_type_enum,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            tags=tags or {}
        )
        
        # Store order
        self.orders[order_id] = order
        self.pending_orders[order_id] = order
        self.total_orders += 1
        
        try:
            # Submit to broker
            if order_type == "market":
                result = await self._submit_market_order(order)
            elif order_type == "limit":
                result = await self._submit_limit_order(order)
            elif order_type == "stop":
                result = await self._submit_stop_order(order)
            else:
                result = await self._submit_market_order(order)  # Default to market
            
            if result.get('success'):
                # Update order status
                order.status = OrderStatus.SUBMITTED
                order.updated_time = datetime.now()
                
                # Move to working orders
                del self.pending_orders[order_id]
                self.working_orders[order_id] = order
                
                # Submit bracket orders if specified
                if stop_loss or take_profit:
                    await self._submit_bracket_orders(order, stop_loss, take_profit)
                
                # Store broker order ID
                broker_order_id = result.get('orderId')
                order.tags['broker_order_id'] = broker_order_id
                
                return order_id
            else:
                # Order rejected
                order.status = OrderStatus.REJECTED
                order.notes = result.get('error', 'Unknown error')
                self.rejected_orders += 1
                
                if self.on_order_rejected:
                    await self.on_order_rejected(order)
                
                raise Exception(f"Order rejected: {order.notes}")
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.notes = str(e)
            self.rejected_orders += 1
            raise
    
    async def _submit_market_order(self, order: Order) -> Dict:
        """Submit market order to broker"""
        contract_id = self._get_contract_id(order.symbol)
        
        result = await topstepx_client.submit_order(
            account_id=self.account_id,
            contract_id=contract_id,
            order_type=2,  # Market order in TopStep API
            side=order.side.value,
            size=order.quantity
        )
        
        return result
    
    async def _submit_limit_order(self, order: Order) -> Dict:
        """Submit limit order to broker"""
        contract_id = self._get_contract_id(order.symbol)
        
        # TopStep API format for limit order
        result = await topstepx_client.request('POST', '/api/Order/place', {
            "accountId": self.account_id,
            "contractId": contract_id,
            "type": 1,  # Limit order
            "side": order.side.value,
            "size": order.quantity,
            "price": order.price
        })
        
        return result
    
    async def _submit_stop_order(self, order: Order) -> Dict:
        """Submit stop order to broker"""
        contract_id = self._get_contract_id(order.symbol)
        
        # TopStep API format for stop order
        result = await topstepx_client.request('POST', '/api/Order/place', {
            "accountId": self.account_id,
            "contractId": contract_id,
            "type": 4,  # Stop order
            "side": order.side.value,
            "size": order.quantity,
            "stopPrice": order.stop_price
        })
        
        return result
    
    async def _submit_bracket_orders(self, parent_order: Order, 
                                    stop_loss: Optional[float],
                                    take_profit: Optional[float]):
        """Submit bracket orders (stop loss and take profit)"""
        if stop_loss:
            # Submit stop loss order
            sl_side = "sell" if parent_order.side == OrderSide.BUY else "buy"
            sl_order_id = await self.submit_order(
                symbol=parent_order.symbol,
                side=sl_side,
                quantity=parent_order.quantity,
                order_type="stop",
                stop_price=stop_loss,
                tags={'parent_order': parent_order.order_id, 'order_type': 'stop_loss'}
            )
            
            parent_order.stop_loss_order_id = sl_order_id
            self.orders[sl_order_id].parent_order_id = parent_order.order_id
        
        if take_profit:
            # Submit take profit order
            tp_side = "sell" if parent_order.side == OrderSide.BUY else "buy"
            tp_order_id = await self.submit_order(
                symbol=parent_order.symbol,
                side=tp_side,
                quantity=parent_order.quantity,
                order_type="limit",
                price=take_profit,
                tags={'parent_order': parent_order.order_id, 'order_type': 'take_profit'}
            )
            
            parent_order.take_profit_order_id = tp_order_id
            self.orders[tp_order_id].parent_order_id = parent_order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Success status
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
            return False
        
        try:
            # Get broker order ID
            broker_order_id = order.tags.get('broker_order_id')
            
            if broker_order_id:
                # Submit cancellation to broker
                result = await topstepx_client.request('POST', '/api/Order/cancel', {
                    "accountId": self.account_id,
                    "orderId": broker_order_id
                })
                
                if result.get('success'):
                    order.status = OrderStatus.CANCELLED
                    order.updated_time = datetime.now()
                    self.cancelled_orders += 1
                    
                    # Remove from working orders
                    if order_id in self.working_orders:
                        del self.working_orders[order_id]
                    
                    # Cancel related bracket orders
                    await self._cancel_bracket_orders(order)
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    async def _cancel_bracket_orders(self, order: Order):
        """Cancel related bracket orders"""
        if order.stop_loss_order_id:
            await self.cancel_order(order.stop_loss_order_id)
        
        if order.take_profit_order_id:
            await self.cancel_order(order.take_profit_order_id)
    
    async def modify_order(self, 
                          order_id: str,
                          price: Optional[float] = None,
                          quantity: Optional[int] = None,
                          stop_price: Optional[float] = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            price: New limit price
            quantity: New quantity
            stop_price: New stop price
            
        Returns:
            Success status
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            return False
        
        try:
            broker_order_id = order.tags.get('broker_order_id')
            
            if broker_order_id:
                # Build modification request
                modify_request = {
                    "accountId": self.account_id,
                    "orderId": broker_order_id
                }
                
                if price is not None:
                    modify_request["price"] = price
                if quantity is not None:
                    modify_request["size"] = quantity
                if stop_price is not None:
                    modify_request["stopPrice"] = stop_price
                
                # Submit modification
                result = await topstepx_client.request('POST', '/api/Order/modify', modify_request)
                
                if result.get('success'):
                    # Update order object
                    if price is not None:
                        order.price = price
                    if quantity is not None:
                        order.quantity = quantity
                    if stop_price is not None:
                        order.stop_price = stop_price
                    
                    order.updated_time = datetime.now()
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error modifying order: {e}")
            return False
    
    async def update_order_status(self):
        """Update status of all working orders"""
        for order_id, order in list(self.working_orders.items()):
            broker_order_id = order.tags.get('broker_order_id')
            
            if broker_order_id:
                try:
                    # Query order status from broker
                    result = await topstepx_client.request('POST', '/api/Order/status', {
                        "accountId": self.account_id,
                        "orderId": broker_order_id
                    })
                    
                    if result.get('success'):
                        status_data = result.get('order', {})
                        
                        # Update order based on broker status
                        broker_status = status_data.get('status', '').lower()
                        
                        if broker_status == 'filled':
                            await self._handle_order_fill(order, status_data)
                        elif broker_status == 'partial':
                            order.status = OrderStatus.PARTIAL
                            order.filled_quantity = status_data.get('filledQuantity', 0)
                        elif broker_status == 'cancelled':
                            order.status = OrderStatus.CANCELLED
                            del self.working_orders[order_id]
                        elif broker_status == 'rejected':
                            order.status = OrderStatus.REJECTED
                            del self.working_orders[order_id]
                        
                        order.updated_time = datetime.now()
                        
                except Exception as e:
                    print(f"Error updating order {order_id}: {e}")
    
    async def _handle_order_fill(self, order: Order, fill_data: Dict):
        """Handle order fill"""
        order.status = OrderStatus.FILLED
        order.filled_quantity = fill_data.get('filledQuantity', order.quantity)
        order.average_fill_price = fill_data.get('averagePrice', order.price or 0)
        order.fill_time = datetime.now()
        
        # Calculate slippage
        if order.order_type == OrderType.MARKET and order.price:
            slippage = abs(order.average_fill_price - order.price)
            self.total_slippage += slippage
        
        # Create execution report
        execution = ExecutionReport(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.average_fill_price,
            commission=2.52,  # TopStep commission
            execution_time=order.fill_time,
            liquidity_indicator="taker" if order.order_type == OrderType.MARKET else "maker",
            exchange="TopStep",
            notes=f"Filled via {order.order_type.name} order"
        )
        
        self.executions.append(execution)
        self.filled_orders += 1
        self.total_commission += execution.commission
        
        # Remove from working orders
        if order.order_id in self.working_orders:
            del self.working_orders[order.order_id]
        
        # Update position
        await self._update_position(order, execution)
        
        # Trigger callback
        if self.on_order_filled:
            await self.on_order_filled(order, execution)
        
        # Handle OCO (One-Cancels-Other) for bracket orders
        if order.parent_order_id:
            parent_order = self.orders.get(order.parent_order_id)
            if parent_order:
                # Cancel the other bracket order
                if order.tags.get('order_type') == 'stop_loss' and parent_order.take_profit_order_id:
                    await self.cancel_order(parent_order.take_profit_order_id)
                elif order.tags.get('order_type') == 'take_profit' and parent_order.stop_loss_order_id:
                    await self.cancel_order(parent_order.stop_loss_order_id)
    
    async def _update_position(self, order: Order, execution: ExecutionReport):
        """Update position tracking"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'average_price': 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'commission': 0
            }
        
        position = self.positions[symbol]
        
        # Update position based on side
        if order.side == OrderSide.BUY:
            # Adding to position
            new_quantity = position['quantity'] + execution.quantity
            if new_quantity != 0:
                position['average_price'] = (
                    (position['average_price'] * position['quantity'] + 
                     execution.price * execution.quantity) / new_quantity
                )
            position['quantity'] = new_quantity
        else:
            # Reducing or reversing position
            if position['quantity'] > 0:
                # Closing long position
                closed_quantity = min(position['quantity'], execution.quantity)
                realized_pnl = (execution.price - position['average_price']) * closed_quantity * 20
                position['realized_pnl'] += realized_pnl
                position['quantity'] -= execution.quantity
                
                if position['quantity'] < 0:
                    # Reversed to short
                    position['average_price'] = execution.price
            else:
                # Adding to short position
                new_quantity = position['quantity'] - execution.quantity
                if new_quantity != 0:
                    position['average_price'] = (
                        (position['average_price'] * abs(position['quantity']) + 
                         execution.price * execution.quantity) / abs(new_quantity)
                    )
                position['quantity'] = new_quantity
        
        position['commission'] += execution.commission
        self.position_version += 1
        
        # Trigger callback
        if self.on_position_update:
            await self.on_position_update(symbol, position)
    
    async def close_position(self, symbol: str, quantity: Optional[int] = None) -> str:
        """
        Close a position
        
        Args:
            symbol: Symbol to close
            quantity: Quantity to close (None for full position)
            
        Returns:
            Order ID of closing order
        """
        if symbol not in self.positions:
            raise Exception(f"No position in {symbol}")
        
        position = self.positions[symbol]
        
        if position['quantity'] == 0:
            raise Exception(f"No open position in {symbol}")
        
        # Determine closing side and quantity
        if position['quantity'] > 0:
            side = "sell"
            close_quantity = quantity or position['quantity']
        else:
            side = "buy"
            close_quantity = quantity or abs(position['quantity'])
        
        # Submit closing order
        order_id = await self.submit_order(
            symbol=symbol,
            side=side,
            quantity=close_quantity,
            order_type="market",
            tags={'action': 'close_position'}
        )
        
        return order_id
    
    async def close_all_positions(self) -> List[str]:
        """
        Close all open positions
        
        Returns:
            List of order IDs
        """
        order_ids = []
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if position['quantity'] != 0:
                try:
                    order_id = await self.close_position(symbol)
                    order_ids.append(order_id)
                except Exception as e:
                    print(f"Error closing position in {symbol}: {e}")
        
        return order_ids
    
    async def cancel_all_orders(self) -> int:
        """
        Cancel all working orders
        
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        
        for order_id in list(self.working_orders.keys()):
            if await self.cancel_order(order_id):
                cancelled += 1
        
        return cancelled
    
    def _map_order_type(self, order_type: str) -> OrderType:
        """Map string order type to enum"""
        mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT,
            'trailing_stop': OrderType.TRAILING_STOP
        }
        return mapping.get(order_type.lower(), OrderType.MARKET)
    
    def _get_contract_id(self, symbol: str) -> str:
        """Get TopStep contract ID for symbol"""
        # This should be dynamically determined based on current contract
        contract_map = {
            'NQ.FUT': 'CON.F.US.ENQ.U25',
            'MNQ.FUT': 'CON.F.US.MNQ.U25',
            'ES.FUT': 'CON.F.US.EP.U25',
            'MES.FUT': 'CON.F.US.MES.U25'
        }
        return contract_map.get(symbol, symbol)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_working_orders(self) -> List[Order]:
        """Get all working orders"""
        return list(self.working_orders.values())
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions"""
        return self.positions.copy()
    
    def get_executions(self, 
                      symbol: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> List[ExecutionReport]:
        """
        Get execution reports
        
        Args:
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of execution reports
        """
        executions = self.executions
        
        if symbol:
            executions = [e for e in executions if e.symbol == symbol]
        
        if start_time:
            executions = [e for e in executions if e.execution_time >= start_time]
        
        if end_time:
            executions = [e for e in executions if e.execution_time <= end_time]
        
        return executions
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'cancelled_orders': self.cancelled_orders,
            'fill_rate': self.filled_orders / self.total_orders if self.total_orders > 0 else 0,
            'total_slippage': self.total_slippage,
            'avg_slippage': self.total_slippage / self.filled_orders if self.filled_orders > 0 else 0,
            'total_commission': self.total_commission,
            'working_orders': len(self.working_orders),
            'open_positions': len([p for p in self.positions.values() if p['quantity'] != 0])
        }
    
    def export_executions(self, filename: str):
        """Export executions to file"""
        export_data = {
            'stats': self.get_execution_stats(),
            'executions': [
                {
                    'execution_id': e.execution_id,
                    'order_id': e.order_id,
                    'symbol': e.symbol,
                    'side': e.side.name,
                    'quantity': e.quantity,
                    'price': e.price,
                    'commission': e.commission,
                    'time': e.execution_time.isoformat(),
                    'notes': e.notes
                }
                for e in self.executions
            ],
            'positions': self.positions
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Executions exported to {filename}")