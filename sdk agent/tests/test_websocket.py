"""
Tests for WebSocket connection and data handling.

Tests WebSocket connectivity, message parsing, and callback execution.
"""

import pytest
import asyncio
from datetime import datetime
from execution.websocket_handler import WebSocketHandler


# Skip if no credentials
pytestmark = pytest.mark.skipif(
    not pytest.config.getoption("--run-websocket", default=False),
    reason="WebSocket tests require --run-websocket flag"
)


class TestWebSocketConnection:
    """Test WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_websocket_initialization(self):
        """Test WebSocket handler initialization."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        assert handler.session_token == "test_token"
        assert handler.account_id == "test_account"
        assert handler.user_hub is None
        assert handler.market_hub is None

    @pytest.mark.asyncio
    async def test_connect_user_hub(self):
        """Test connecting to user hub (requires real credentials)."""
        # This test requires real session token
        # Skip in automated tests
        pytest.skip("Requires real session token")

    @pytest.mark.asyncio
    async def test_connect_market_hub(self):
        """Test connecting to market data hub (requires real credentials)."""
        # This test requires real session token
        # Skip in automated tests
        pytest.skip("Requires real session token")

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting from WebSocket."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        # Disconnect without connecting (should not error)
        await handler.disconnect()

        assert handler.user_hub is None
        assert handler.market_hub is None


class TestWebSocketCallbacks:
    """Test WebSocket callback registration and execution."""

    @pytest.mark.asyncio
    async def test_register_bar_callback(self):
        """Test registering bar update callback."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        callback_called = False
        received_bar = None

        async def test_callback(bar):
            nonlocal callback_called, received_bar
            callback_called = True
            received_bar = bar

        handler.register_bar_callback(test_callback)

        # Simulate bar update
        test_bar = {
            'timestamp': datetime.now(),
            'open': 21000,
            'high': 21010,
            'low': 20990,
            'close': 21005,
            'volume': 1000
        }

        await handler._on_bar_update(test_bar)

        assert callback_called
        assert received_bar == test_bar

    @pytest.mark.asyncio
    async def test_register_order_callback(self):
        """Test registering order update callback."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        callback_called = False
        received_order = None

        async def test_callback(order):
            nonlocal callback_called, received_order
            callback_called = True
            received_order = order

        handler.register_order_callback(test_callback)

        # Simulate order update
        test_order = {
            'order_id': '123',
            'status': 'FILLED',
            'fill_price': 21005
        }

        await handler._on_order_update(test_order)

        assert callback_called
        assert received_order == test_order

    @pytest.mark.asyncio
    async def test_register_position_callback(self):
        """Test registering position update callback."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        callback_called = False
        received_position = None

        async def test_callback(position):
            nonlocal callback_called, received_position
            callback_called = True
            received_position = position

        handler.register_position_callback(test_callback)

        # Simulate position update
        test_position = {
            'contract_id': 1,
            'quantity': 1,
            'avg_price': 21000,
            'pnl': 25.0
        }

        await handler._on_position_update(test_position)

        assert callback_called
        assert received_position == test_position

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Test multiple callbacks for same event."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        callback1_called = False
        callback2_called = False

        async def callback1(bar):
            nonlocal callback1_called
            callback1_called = True

        async def callback2(bar):
            nonlocal callback2_called
            callback2_called = True

        handler.register_bar_callback(callback1)
        handler.register_bar_callback(callback2)

        test_bar = {
            'timestamp': datetime.now(),
            'open': 21000,
            'high': 21010,
            'low': 20990,
            'close': 21005,
            'volume': 1000
        }

        await handler._on_bar_update(test_bar)

        assert callback1_called
        assert callback2_called


class TestWebSocketDataParsing:
    """Test WebSocket message parsing."""

    @pytest.mark.asyncio
    async def test_parse_bar_message(self):
        """Test parsing bar update message."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        # Mock WebSocket message
        raw_message = {
            'type': 'bar',
            'data': {
                'timestamp': '2025-01-01T10:00:00',
                'open': 21000.0,
                'high': 21010.0,
                'low': 20990.0,
                'close': 21005.0,
                'volume': 1000
            }
        }

        bar = handler._parse_bar_message(raw_message)

        assert bar is not None
        assert bar['open'] == 21000.0
        assert bar['close'] == 21005.0

    @pytest.mark.asyncio
    async def test_parse_order_message(self):
        """Test parsing order update message."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        raw_message = {
            'type': 'order',
            'data': {
                'order_id': '123',
                'status': 'FILLED',
                'fill_price': 21005.0,
                'quantity': 1
            }
        }

        order = handler._parse_order_message(raw_message)

        assert order is not None
        assert order['order_id'] == '123'
        assert order['status'] == 'FILLED'

    @pytest.mark.asyncio
    async def test_parse_invalid_message(self):
        """Test handling invalid message."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        # Invalid message
        raw_message = {'invalid': 'data'}

        # Should not raise exception
        try:
            handler._parse_bar_message(raw_message)
        except Exception:
            pytest.fail("Should handle invalid message gracefully")


class TestWebSocketReconnection:
    """Test WebSocket reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_on_disconnect(self):
        """Test automatic reconnection on disconnect."""
        # This would require mocking WebSocket connection
        # Skip for now
        pytest.skip("Requires WebSocket mocking")

    @pytest.mark.asyncio
    async def test_max_reconnect_attempts(self):
        """Test max reconnection attempts."""
        # This would require mocking WebSocket connection
        # Skip for now
        pytest.skip("Requires WebSocket mocking")


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics."""

    @pytest.mark.asyncio
    async def test_message_latency(self):
        """Test message processing latency."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        latencies = []

        async def measure_callback(bar):
            # Measure time from bar timestamp to callback
            now = datetime.now()
            bar_time = bar['timestamp']
            latency = (now - bar_time).total_seconds() * 1000
            latencies.append(latency)

        handler.register_bar_callback(measure_callback)

        # Simulate multiple bar updates
        for i in range(10):
            test_bar = {
                'timestamp': datetime.now(),
                'open': 21000,
                'high': 21010,
                'low': 20990,
                'close': 21005,
                'volume': 1000
            }
            await handler._on_bar_update(test_bar)
            await asyncio.sleep(0.01)

        # Check latencies are reasonable (< 100ms)
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            assert avg_latency < 100  # Less than 100ms

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self):
        """Test handling concurrent messages."""
        handler = WebSocketHandler(
            session_token="test_token",
            account_id="test_account"
        )

        message_count = 0

        async def count_callback(bar):
            nonlocal message_count
            message_count += 1
            await asyncio.sleep(0.01)  # Simulate processing

        handler.register_bar_callback(count_callback)

        # Send multiple messages concurrently
        tasks = []
        for i in range(20):
            test_bar = {
                'timestamp': datetime.now(),
                'open': 21000 + i,
                'high': 21010 + i,
                'low': 20990 + i,
                'close': 21005 + i,
                'volume': 1000
            }
            task = asyncio.create_task(handler._on_bar_update(test_bar))
            tasks.append(task)

        await asyncio.gather(*tasks)

        assert message_count == 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
