"""
Order Management System.

This module provides high-level order management:
- Bracket orders (entry + stop loss + take profit)
- Position tracking
- Fill handling via WebSocket
- Automated stop/target placement
- Order state management
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .topstep_client import TopStepXClient, OrderSide, OrderType, OrderStatus
from .websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class BracketOrder:
    """Bracket order containing entry, stop, and target."""
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    target_order_id: Optional[int] = None
    contract_id: str = ""
    side: PositionSide = PositionSide.LONG
    size: int = 1
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    filled: bool = False
    closed: bool = False
    pnl: Optional[float] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Position:
    """Active position."""
    contract_id: str
    side: PositionSide
    size: int
    entry_price: float
    stop_order_id: Optional[int] = None
    target_order_id: Optional[int] = None
    unrealized_pnl: float = 0.0
    opened_at: datetime = None

    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()


class OrderManager:
    """
    High-level order management system.

    Manages bracket orders, position tracking, and fill handling.
    """

    def __init__(
        self,
        client: TopStepXClient,
        websocket: WebSocketHandler,
        config: Dict[str, Any]
    ):
        """
        Initialize order manager.

        Args:
            client: TopStepX REST API client
            websocket: WebSocket handler for real-time updates
            config: Configuration dict
        """
        self.client = client
        self.websocket = websocket
        self.config = config

        # Active orders and positions
        self.bracket_orders: Dict[int, BracketOrder] = {}  # entry_order_id -> BracketOrder
        self.active_positions: Dict[str, Position] = {}  # contract_id -> Position

        # Register WebSocket callbacks
        self.websocket.on_order_update(self._handle_order_update)
        self.websocket.on_trade_update(self._handle_trade_update)
        self.websocket.on_position_update(self._handle_position_update)

        logger.info("Order manager initialized")

    async def place_bracket_order(
        self,
        contract_id: str,
        side: PositionSide,
        size: int,
        entry_price: Optional[float] = None,
        stop_loss_ticks: int = 8,
        take_profit_ticks: int = 12,
        tick_size: float = 0.25,
        tick_value: float = 5.0
    ) -> Optional[BracketOrder]:
        """
        Place bracket order (entry + stop + target).

        Args:
            contract_id: Contract ID
            side: Position side (LONG or SHORT)
            size: Number of contracts
            entry_price: Entry price (None for market order)
            stop_loss_ticks: Stop loss in ticks
            take_profit_ticks: Take profit in ticks
            tick_size: Contract tick size
            tick_value: Contract tick value

        Returns:
            BracketOrder if successful, None otherwise
        """
        try:
            # Create bracket order object
            bracket = BracketOrder(
                contract_id=contract_id,
                side=side,
                size=size,
                entry_price=entry_price
            )

            # Determine order side for entry
            order_side = OrderSide.BID if side == PositionSide.LONG else OrderSide.ASK

            # Place entry order
            if entry_price is None:
                # Market order
                logger.info(f"Placing MARKET {side.value} order: {size} {contract_id}")
                entry_order_id = await self.client.place_order(
                    contract_id=contract_id,
                    side=order_side,
                    size=size,
                    order_type=OrderType.MARKET
                )
            else:
                # Limit order
                logger.info(f"Placing LIMIT {side.value} order: {size} @ ${entry_price:.2f}")
                entry_order_id = await self.client.place_order(
                    contract_id=contract_id,
                    side=order_side,
                    size=size,
                    order_type=OrderType.LIMIT,
                    limit_price=entry_price
                )

            if not entry_order_id:
                logger.error("Failed to place entry order")
                return None

            bracket.entry_order_id = entry_order_id

            # Calculate stop and target prices
            if entry_price:
                if side == PositionSide.LONG:
                    bracket.stop_price = entry_price - (stop_loss_ticks * tick_size)
                    bracket.target_price = entry_price + (take_profit_ticks * tick_size)
                else:
                    bracket.stop_price = entry_price + (stop_loss_ticks * tick_size)
                    bracket.target_price = entry_price - (take_profit_ticks * tick_size)

            # Store bracket order
            self.bracket_orders[entry_order_id] = bracket

            logger.info(f"✅ Bracket order placed: Entry={entry_order_id}")
            return bracket

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            return None

    async def _place_bracket_legs(
        self,
        bracket: BracketOrder,
        entry_fill_price: float
    ) -> bool:
        """
        Place stop and target orders after entry fill.

        Args:
            bracket: Bracket order
            entry_fill_price: Filled entry price

        Returns:
            True if successful
        """
        try:
            # Calculate stop and target if not already set
            if bracket.stop_price is None or bracket.target_price is None:
                tick_size = self.config.get('tick_size', 0.25)
                stop_ticks = self.config.get('stop_ticks', 8)
                target_ticks = self.config.get('target_ticks', 12)

                if bracket.side == PositionSide.LONG:
                    bracket.stop_price = entry_fill_price - (stop_ticks * tick_size)
                    bracket.target_price = entry_fill_price + (target_ticks * tick_size)
                else:
                    bracket.stop_price = entry_fill_price + (stop_ticks * tick_size)
                    bracket.target_price = entry_fill_price - (target_ticks * tick_size)

            # Place stop loss (opposite side of entry)
            stop_side = OrderSide.ASK if bracket.side == PositionSide.LONG else OrderSide.BID

            logger.info(f"Placing STOP @ ${bracket.stop_price:.2f}")
            stop_order_id = await self.client.place_order(
                contract_id=bracket.contract_id,
                side=stop_side,
                size=bracket.size,
                order_type=OrderType.STOP,
                stop_price=bracket.stop_price
            )

            if stop_order_id:
                bracket.stop_order_id = stop_order_id
            else:
                logger.error("Failed to place stop loss order")
                return False

            # Place take profit (opposite side of entry)
            target_side = OrderSide.ASK if bracket.side == PositionSide.LONG else OrderSide.BID

            logger.info(f"Placing TARGET @ ${bracket.target_price:.2f}")
            target_order_id = await self.client.place_order(
                contract_id=bracket.contract_id,
                side=target_side,
                size=bracket.size,
                order_type=OrderType.LIMIT,
                limit_price=bracket.target_price
            )

            if target_order_id:
                bracket.target_order_id = target_order_id
            else:
                logger.error("Failed to place target order")
                # Cancel stop if target fails
                if stop_order_id:
                    await self.client.cancel_order(stop_order_id)
                return False

            logger.info(f"✅ Bracket legs placed: Stop={stop_order_id}, Target={target_order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to place bracket legs: {e}")
            return False

    async def cancel_bracket(self, entry_order_id: int) -> bool:
        """
        Cancel bracket order.

        Args:
            entry_order_id: Entry order ID

        Returns:
            True if successful
        """
        if entry_order_id not in self.bracket_orders:
            logger.warning(f"Bracket order {entry_order_id} not found")
            return False

        bracket = self.bracket_orders[entry_order_id]

        try:
            # Cancel all active orders
            cancelled = []

            if bracket.entry_order_id and not bracket.filled:
                if await self.client.cancel_order(bracket.entry_order_id):
                    cancelled.append(f"Entry={bracket.entry_order_id}")

            if bracket.stop_order_id:
                if await self.client.cancel_order(bracket.stop_order_id):
                    cancelled.append(f"Stop={bracket.stop_order_id}")

            if bracket.target_order_id:
                if await self.client.cancel_order(bracket.target_order_id):
                    cancelled.append(f"Target={bracket.target_order_id}")

            if cancelled:
                logger.info(f"✅ Cancelled bracket orders: {', '.join(cancelled)}")

            # Remove from tracking
            del self.bracket_orders[entry_order_id]

            return True

        except Exception as e:
            logger.error(f"Failed to cancel bracket: {e}")
            return False

    async def close_position(self, contract_id: str) -> bool:
        """
        Close position for contract.

        Args:
            contract_id: Contract ID

        Returns:
            True if successful
        """
        if contract_id not in self.active_positions:
            logger.warning(f"No active position for {contract_id}")
            return False

        try:
            # Use TopStepX close position API
            success = await self.client.close_position(contract_id)

            if success:
                # Remove from tracking
                del self.active_positions[contract_id]
                logger.info(f"✅ Position closed: {contract_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    # WebSocket Event Handlers

    async def _handle_order_update(self, order_data: Dict) -> None:
        """Handle order update from WebSocket."""
        try:
            order_id = order_data.get("id")
            status = order_data.get("status")

            # Check if this is a bracket order entry
            if order_id in self.bracket_orders:
                bracket = self.bracket_orders[order_id]

                if status == 2:  # Filled
                    filled_price = order_data.get("filledPrice")
                    bracket.entry_price = filled_price
                    bracket.filled = True

                    logger.info(f"📊 Entry filled @ ${filled_price:.2f}")

                    # Place stop and target
                    await self._place_bracket_legs(bracket, filled_price)

                    # Create position
                    position = Position(
                        contract_id=bracket.contract_id,
                        side=bracket.side,
                        size=bracket.size,
                        entry_price=filled_price,
                        stop_order_id=bracket.stop_order_id,
                        target_order_id=bracket.target_order_id
                    )
                    self.active_positions[bracket.contract_id] = position

                elif status in [3, 4, 5]:  # Cancelled, Expired, Rejected
                    logger.warning(f"Entry order {order_id} status: {status}")
                    if order_id in self.bracket_orders:
                        del self.bracket_orders[order_id]

        except Exception as e:
            logger.error(f"Error handling order update: {e}")

    async def _handle_trade_update(self, trade_data: Dict) -> None:
        """Handle trade update from WebSocket."""
        try:
            contract_id = trade_data.get("contractId")
            pnl = trade_data.get("profitAndLoss")

            if pnl is not None:  # Closing trade
                # Find and close associated bracket
                for entry_id, bracket in list(self.bracket_orders.items()):
                    if bracket.contract_id == contract_id and bracket.filled:
                        bracket.pnl = pnl
                        bracket.closed = True

                        logger.info(f"💰 Bracket closed: P&L ${pnl:.2f}")

                        # Remove from tracking
                        del self.bracket_orders[entry_id]
                        break

                # Remove position
                if contract_id in self.active_positions:
                    del self.active_positions[contract_id]

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    async def _handle_position_update(self, position_data: Dict) -> None:
        """Handle position update from WebSocket."""
        try:
            contract_id = position_data.get("contractId")
            size = position_data.get("size")

            if size == 0 and contract_id in self.active_positions:
                # Position closed
                logger.info(f"Position closed: {contract_id}")
                del self.active_positions[contract_id]

        except Exception as e:
            logger.error(f"Error handling position update: {e}")

    # Status and Queries

    def get_active_positions(self) -> Dict[str, Position]:
        """Get all active positions."""
        return self.active_positions.copy()

    def get_active_brackets(self) -> Dict[int, BracketOrder]:
        """Get all active bracket orders."""
        return self.bracket_orders.copy()

    def has_position(self, contract_id: str) -> bool:
        """Check if position exists for contract."""
        return contract_id in self.active_positions

    def get_position_count(self) -> int:
        """Get count of active positions."""
        return len(self.active_positions)

    async def get_total_unrealized_pnl(self) -> float:
        """
        Calculate total unrealized P&L across all positions.

        Returns:
            Total unrealized P&L
        """
        total = 0.0

        for contract_id, position in self.active_positions.items():
            # Get latest quote
            symbol = contract_id.split('.')[-2]  # Extract symbol from contract ID
            quote = self.websocket.get_latest_quote(f"F.US.{symbol}")

            if quote:
                current_price = quote.get("lastPrice")
                if current_price:
                    tick_value = self.config.get('tick_value', 5.0)
                    tick_size = self.config.get('tick_size', 0.25)

                    if position.side == PositionSide.LONG:
                        pnl = ((current_price - position.entry_price) / tick_size) * tick_value * position.size
                    else:
                        pnl = ((position.entry_price - current_price) / tick_size) * tick_value * position.size

                    total += pnl

        return total
