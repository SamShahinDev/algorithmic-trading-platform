#!/usr/bin/env python3
"""
Test script for TopStepX Real-Time Integration
Tests SignalR connection, trade search, and analytics
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.topstepx_client import topstepx_client
from brokers.topstepx_realtime import TopStepXRealTimeClient
from services.trade_analytics import TradeAnalyticsService
from utils.logger import setup_logger

logger = setup_logger('RealtimeTest')


async def test_trade_search():
    """Test the trade search API"""
    logger.info("=" * 50)
    logger.info("Testing Trade Search API")
    logger.info("=" * 50)
    
    try:
        # Connect to TopStepX
        if not topstepx_client.connected:
            connected = await topstepx_client.connect()
            if not connected:
                logger.error("Failed to connect to TopStepX")
                return False
        
        # Search for trades in the last 7 days
        logger.info("Searching for trades in the last 7 days...")
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        result = await topstepx_client.search_trades(start_date)
        
        if result.get("success"):
            trades = result.get("trades", [])
            logger.info(f"✅ Found {len(trades)} trades")
            
            # Display first few trades
            for i, trade in enumerate(trades[:3]):
                logger.info(f"  Trade {i+1}:")
                logger.info(f"    - ID: {trade.get('id')}")
                logger.info(f"    - Price: ${trade.get('price')}")
                logger.info(f"    - P&L: ${trade.get('profitAndLoss')}")
                logger.info(f"    - Side: {'Buy' if trade.get('side') == 0 else 'Sell'}")
        else:
            logger.warning(f"Trade search failed: {result.get('errorMessage')}")
            
        # Get daily trades
        logger.info("\nGetting today's trades...")
        daily_trades = await topstepx_client.get_daily_trades()
        logger.info(f"✅ Today's trades: {len(daily_trades)}")
        
        # Calculate daily P&L
        logger.info("\nCalculating daily P&L...")
        daily_pnl = await topstepx_client.calculate_daily_pnl()
        logger.info(f"✅ Daily P&L: ${daily_pnl:.2f}")
        
        # Get trade statistics
        logger.info("\nGetting 7-day trade statistics...")
        stats = await topstepx_client.get_trade_statistics(days=7)
        logger.info(f"✅ Trade Statistics:")
        logger.info(f"  - Total trades: {stats.get('total_trades')}")
        logger.info(f"  - Win rate: {stats.get('win_rate')}%")
        logger.info(f"  - Total P&L: ${stats.get('total_pnl')}")
        logger.info(f"  - Profit factor: {stats.get('profit_factor')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Trade search test error: {e}")
        return False


async def test_trade_analytics():
    """Test the trade analytics service"""
    logger.info("=" * 50)
    logger.info("Testing Trade Analytics Service")
    logger.info("=" * 50)
    
    try:
        # Initialize analytics service
        analytics = TradeAnalyticsService(topstepx_client)
        
        # Get performance summary
        logger.info("Getting 7-day performance summary...")
        summary = await analytics.get_performance_summary(days=7)
        
        if summary:
            logger.info("✅ Performance Summary Retrieved")
            
            # Display basic stats
            basic = summary.get("basic_stats", {})
            if basic:
                logger.info(f"  Basic Stats:")
                logger.info(f"    - Win rate: {basic.get('win_rate')}%")
                logger.info(f"    - Net P&L: ${basic.get('net_pnl')}")
                logger.info(f"    - Avg win: ${basic.get('avg_win')}")
                logger.info(f"    - Avg loss: ${basic.get('avg_loss')}")
            
            # Display risk metrics
            risk = summary.get("risk_metrics", {})
            if risk:
                logger.info(f"  Risk Metrics:")
                logger.info(f"    - Sharpe ratio: {risk.get('sharpe_ratio')}")
                logger.info(f"    - Max drawdown: ${risk.get('max_drawdown')}")
                logger.info(f"    - VaR (95%): ${risk.get('value_at_risk_95')}")
        
        # Generate insights
        logger.info("\nGenerating trading insights...")
        insights = await analytics.generate_insights(days=7)
        
        if insights:
            logger.info("✅ Trading Insights:")
            for insight in insights[:5]:  # Show first 5 insights
                logger.info(f"  - {insight}")
        
        return True
        
    except Exception as e:
        logger.error(f"Analytics test error: {e}")
        return False


async def test_signalr_connection():
    """Test SignalR real-time connection"""
    logger.info("=" * 50)
    logger.info("Testing SignalR Real-Time Connection")
    logger.info("=" * 50)
    
    try:
        # Connect to TopStepX first to get session token
        if not topstepx_client.connected:
            connected = await topstepx_client.connect()
            if not connected:
                logger.error("Failed to connect to TopStepX")
                return False
        
        # Check if we have required credentials
        if not topstepx_client.session_token:
            logger.warning("No session token available - SignalR requires authentication")
            return False
        
        if not topstepx_client.account_id:
            logger.warning("No account ID available - SignalR requires account selection")
            return False
        
        logger.info(f"Session token: {topstepx_client.session_token[:20]}...")
        logger.info(f"Account ID: {topstepx_client.account_id}")
        
        # Create SignalR client
        realtime_client = TopStepXRealTimeClient(
            jwt_token=topstepx_client.session_token,
            account_id=topstepx_client.account_id
        )
        
        # Register test callbacks
        order_updates = []
        position_updates = []
        quote_updates = []
        
        async def on_order(data):
            order_updates.append(data)
            logger.info(f"📋 Order Update: {data}")
        
        async def on_position(data):
            position_updates.append(data)
            logger.info(f"📊 Position Update: {data}")
        
        async def on_quote(data):
            quote_updates.append(data)
            symbol = data.get('symbol')
            price = data.get('lastPrice')
            logger.info(f"💹 Quote: {symbol} @ ${price}")
        
        realtime_client.on_order(on_order)
        realtime_client.on_position(on_position)
        realtime_client.on_quote(on_quote)
        
        # Connect to SignalR
        logger.info("Connecting to SignalR hubs...")
        connected = await realtime_client.connect()
        
        if connected:
            logger.info("✅ Successfully connected to SignalR")
            
            # Listen for updates for 10 seconds
            logger.info("Listening for real-time updates for 10 seconds...")
            await asyncio.sleep(10)
            
            # Check what we received
            logger.info(f"\nReceived Updates:")
            logger.info(f"  - Orders: {len(order_updates)}")
            logger.info(f"  - Positions: {len(position_updates)}")
            logger.info(f"  - Quotes: {len(quote_updates)}")
            
            # Get cached data
            active_orders = realtime_client.get_active_orders()
            active_positions = realtime_client.get_active_positions()
            latest_quotes = realtime_client.latest_quotes
            
            logger.info(f"\nCached Data:")
            logger.info(f"  - Active orders: {len(active_orders)}")
            logger.info(f"  - Active positions: {len(active_positions)}")
            logger.info(f"  - Quotes tracked: {len(latest_quotes)}")
            
            # Disconnect
            await realtime_client.disconnect()
            logger.info("🔌 Disconnected from SignalR")
            
            return True
        else:
            logger.error("Failed to connect to SignalR")
            return False
            
    except Exception as e:
        logger.error(f"SignalR test error: {e}")
        return False


async def test_api_endpoints():
    """Test the new trade API endpoints"""
    logger.info("=" * 50)
    logger.info("Testing Trade API Endpoints")
    logger.info("=" * 50)
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            # Test trade search endpoint
            logger.info("Testing /api/trades/search...")
            async with session.get(f"{base_url}/api/trades/search?days=7") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Trade search: {data.get('summary', {}).get('total_trades')} trades found")
                else:
                    logger.error(f"Trade search failed: {resp.status}")
            
            # Test daily trades endpoint
            logger.info("Testing /api/trades/daily...")
            async with session.get(f"{base_url}/api/trades/daily") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Daily trades: {len(data.get('trades', []))} trades today")
                else:
                    logger.error(f"Daily trades failed: {resp.status}")
            
            # Test statistics endpoint
            logger.info("Testing /api/trades/statistics/7...")
            async with session.get(f"{base_url}/api/trades/statistics/7") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    stats = data.get('statistics', {})
                    logger.info(f"✅ Statistics: Win rate {stats.get('win_rate')}%")
                else:
                    logger.error(f"Statistics failed: {resp.status}")
            
            # Test performance analysis endpoint
            logger.info("Testing /api/trades/performance/7...")
            async with session.get(f"{base_url}/api/trades/performance/7") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Performance analysis: {len(data.get('insights', []))} insights generated")
                else:
                    logger.error(f"Performance analysis failed: {resp.status}")
            
            # Test insights endpoint
            logger.info("Testing /api/trades/insights...")
            async with session.get(f"{base_url}/api/trades/insights?days=7") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    insights = data.get('insights', [])
                    logger.info(f"✅ Insights: {len(insights)} insights")
                    for insight in insights[:3]:
                        logger.info(f"  - {insight}")
                else:
                    logger.error(f"Insights failed: {resp.status}")
        
        return True
        
    except Exception as e:
        logger.error(f"API endpoint test error: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("🚀 Starting TopStepX Real-Time Integration Tests")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: Trade Search
    logger.info("\n📝 Test 1: Trade Search API")
    results['trade_search'] = await test_trade_search()
    await asyncio.sleep(1)
    
    # Test 2: Trade Analytics
    logger.info("\n📊 Test 2: Trade Analytics Service")
    results['analytics'] = await test_trade_analytics()
    await asyncio.sleep(1)
    
    # Test 3: SignalR Connection
    logger.info("\n🔌 Test 3: SignalR Real-Time Connection")
    results['signalr'] = await test_signalr_connection()
    await asyncio.sleep(1)
    
    # Test 4: API Endpoints (only if server is running)
    logger.info("\n🌐 Test 4: API Endpoints")
    try:
        results['api'] = await test_api_endpoints()
    except Exception as e:
        logger.warning(f"API test skipped (server not running): {e}")
        results['api'] = None
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    logger.info("\n" + "=" * 50)
    logger.info(f"OVERALL: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        logger.info("🎉 All tests passed successfully!")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())