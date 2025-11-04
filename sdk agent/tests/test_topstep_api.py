"""
Tests for TopStepX API integration.

Tests API calls using practice account (not live trading).
Verifies authentication, order placement, position management.
"""

import pytest
import os
from execution.topstep_client import TopStepXClient


# Skip tests if no credentials available
pytestmark = pytest.mark.skipif(
    not os.getenv('TOPSTEPX_EMAIL') or not os.getenv('TOPSTEPX_PASSWORD'),
    reason="TopStepX credentials not available"
)


class TestTopStepXAuthentication:
    """Test authentication with TopStepX API."""

    @pytest.mark.asyncio
    async def test_authentication_success(self):
        """Test successful authentication."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            assert client.session_token is not None
            assert client.account_id is not None
            assert client.is_authenticated

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_authentication_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        client = TopStepXClient()

        # Override with invalid credentials
        client.email = "invalid@example.com"
        client.password = "wrongpassword"

        with pytest.raises(Exception):
            await client.authenticate()

        await client.close()


class TestTopStepXAccountInfo:
    """Test account information retrieval."""

    @pytest.mark.asyncio
    async def test_get_account_info(self):
        """Test retrieving account information."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            account_info = await client.get_account_info()

            assert account_info is not None
            assert 'balance' in account_info or 'equity' in account_info
            assert 'account_id' in account_info

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_positions(self):
        """Test retrieving current positions."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            positions = await client.get_positions()

            assert isinstance(positions, list)
            # Positions list may be empty if no open positions

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_orders(self):
        """Test retrieving orders."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            orders = await client.get_orders()

            assert isinstance(orders, list)
            # Orders list may be empty if no active orders

        finally:
            await client.close()


class TestTopStepXMarketData:
    """Test market data retrieval."""

    @pytest.mark.asyncio
    async def test_get_contract_info(self):
        """Test retrieving contract information."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            # Get NQ contract info (contract ID from config)
            contract_id = 1  # NQ futures
            contract_info = await client.get_contract_info(contract_id)

            assert contract_info is not None
            assert 'symbol' in contract_info or 'name' in contract_info

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_quote(self):
        """Test retrieving current quote."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            contract_id = 1  # NQ futures
            quote = await client.get_quote(contract_id)

            assert quote is not None
            assert 'bid' in quote or 'ask' in quote or 'last' in quote

        finally:
            await client.close()


@pytest.mark.skip(reason="Only run manually on practice account")
class TestTopStepXOrderPlacement:
    """Test order placement (PRACTICE ACCOUNT ONLY)."""

    @pytest.mark.asyncio
    async def test_place_limit_order(self):
        """Test placing a limit order."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            # Get current quote
            contract_id = 1
            quote = await client.get_quote(contract_id)

            # Place limit order well away from market
            limit_price = quote['bid'] - 100  # Far below market

            order = await client.place_order(
                contract_id=contract_id,
                side='BUY',
                quantity=1,
                order_type='LIMIT',
                limit_price=limit_price
            )

            assert order is not None
            assert 'order_id' in order

            # Cancel the order
            if 'order_id' in order:
                await client.cancel_order(order['order_id'])

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_place_bracket_order(self):
        """Test placing bracket order (entry + stop + target)."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            contract_id = 1
            quote = await client.get_quote(contract_id)

            # Bracket order far from market
            entry_price = quote['bid'] - 100
            stop_price = entry_price - 20
            target_price = entry_price + 40

            result = await client.place_bracket_order(
                contract_id=contract_id,
                side='BUY',
                quantity=1,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price
            )

            assert result is not None
            assert 'entry_order_id' in result

            # Cancel all orders
            if 'entry_order_id' in result:
                await client.cancel_order(result['entry_order_id'])
            if 'stop_order_id' in result:
                await client.cancel_order(result['stop_order_id'])
            if 'target_order_id' in result:
                await client.cancel_order(result['target_order_id'])

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """Test canceling an order."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            contract_id = 1
            quote = await client.get_quote(contract_id)

            # Place order to cancel
            limit_price = quote['bid'] - 100

            order = await client.place_order(
                contract_id=contract_id,
                side='BUY',
                quantity=1,
                order_type='LIMIT',
                limit_price=limit_price
            )

            assert 'order_id' in order

            # Cancel it
            cancel_result = await client.cancel_order(order['order_id'])

            assert cancel_result is not None

        finally:
            await client.close()


class TestTopStepXErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request(self):
        """Test request without authentication fails."""
        client = TopStepXClient()

        # Try to get account info without authenticating
        with pytest.raises(Exception):
            await client.get_account_info()

        await client.close()

    @pytest.mark.asyncio
    async def test_invalid_contract_id(self):
        """Test request with invalid contract ID."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            # Try invalid contract ID
            with pytest.raises(Exception):
                await client.get_contract_info(999999)

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting handling."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            # Make many rapid requests (may hit rate limit)
            for i in range(20):
                try:
                    await client.get_account_info()
                except Exception as e:
                    # Check if rate limit error
                    if '429' in str(e) or 'rate' in str(e).lower():
                        # Expected rate limit
                        pass
                    else:
                        raise

        finally:
            await client.close()


class TestTopStepXSession:
    """Test session management."""

    @pytest.mark.asyncio
    async def test_session_reuse(self):
        """Test session can be reused for multiple requests."""
        client = TopStepXClient()

        try:
            await client.authenticate()

            # Make multiple requests with same session
            account1 = await client.get_account_info()
            account2 = await client.get_account_info()

            assert account1 == account2

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_session_close(self):
        """Test session cleanup."""
        client = TopStepXClient()

        await client.authenticate()
        assert client.session is not None

        await client.close()

        # Session should be closed
        assert client.session is None or client.session.closed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
