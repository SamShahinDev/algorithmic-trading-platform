#!/usr/bin/env python3
"""
Trade Reconciliation Script
Compares bot trade records with broker trade history
"""

import sys
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

from trading_bot.utils.trade_logger import TradeLogger
from brokers.topstepx_client import topstepx_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_broker_trades(account_id: str, date: str = None):
    """
    Fetch trade history from broker
    
    Args:
        account_id: TopStep account ID
        date: Date to fetch trades for (YYYY-MM-DD format)
    
    Returns:
        List of broker trade records
    """
    try:
        # Connect to broker
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        # Fetch today's trades if no date specified
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Get fills/trades for the day
        fills = await topstepx_client.get_fills(account_id, date)
        
        if not fills:
            logger.info(f"No broker trades found for {date}")
            return []
        
        # Transform to standard format
        broker_trades = []
        for fill in fills:
            trade = {
                'order_id': fill.get('orderId'),
                'timestamp': fill.get('timestamp'),
                'symbol': fill.get('symbol'),
                'side': fill.get('side'),
                'quantity': fill.get('quantity'),
                'price': fill.get('price'),
                'pnl': fill.get('pnl', 0),
                'fees': fill.get('fees', 0)
            }
            broker_trades.append(trade)
        
        logger.info(f"Found {len(broker_trades)} broker trades")
        return broker_trades
        
    except Exception as e:
        logger.error(f"Failed to fetch broker trades: {e}")
        return []


async def reconcile_trades(bot_name: str = "nq_bot", account_id: str = None):
    """
    Main reconciliation function
    
    Args:
        bot_name: Name of the bot to reconcile
        account_id: TopStep account ID
    """
    print("=" * 60)
    print("TRADE RECONCILIATION REPORT")
    print("=" * 60)
    print(f"Bot: {bot_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize trade logger
    trade_logger = TradeLogger(bot_name=bot_name)
    
    # Get bot trades
    bot_trades = trade_logger.get_recent_trades(limit=100)
    print(f"Bot trades found: {len(bot_trades)}")
    
    # Get broker trades
    if account_id:
        broker_trades = await fetch_broker_trades(account_id)
        print(f"Broker trades found: {len(broker_trades)}")
        
        # Perform reconciliation
        report = trade_logger.reconcile_with_broker(broker_trades)
        
        # Display results
        print("\n" + "=" * 60)
        print("RECONCILIATION RESULTS")
        print("=" * 60)
        
        if report.get('reconciled'):
            print("✅ All trades reconciled successfully!")
        else:
            print("⚠️ Discrepancies found:")
            
            missing_in_bot = report.get('missing_in_bot', [])
            if missing_in_bot:
                print(f"\n❌ Missing in bot records: {len(missing_in_bot)}")
                for trade in missing_in_bot[:5]:  # Show first 5
                    print(f"  - Order {trade.get('order_id')}: "
                          f"{trade.get('symbol')} {trade.get('quantity')} @ {trade.get('price')}")
            
            missing_in_broker = report.get('missing_in_broker', [])
            if missing_in_broker:
                print(f"\n❌ Missing in broker records: {len(missing_in_broker)}")
                for trade in missing_in_broker[:5]:  # Show first 5
                    print(f"  - Trade {trade.get('trade_id')}: "
                          f"{trade.get('symbol')} {trade.get('size')} @ {trade.get('exit_price')}")
            
            pnl_mismatches = report.get('pnl_mismatches', [])
            if pnl_mismatches:
                print(f"\n⚠️ P&L mismatches: {len(pnl_mismatches)}")
                for mismatch in pnl_mismatches[:5]:  # Show first 5
                    print(f"  - Order {mismatch.get('order_id')}: "
                          f"Bot=${mismatch.get('bot_pnl'):.2f} vs "
                          f"Broker=${mismatch.get('broker_pnl'):.2f} "
                          f"(Δ ${mismatch.get('difference'):.2f})")
    else:
        print("\n⚠️ No account ID provided - showing bot trades only")
    
    # Show recent bot trades
    print("\n" + "=" * 60)
    print("RECENT BOT TRADES")
    print("=" * 60)
    
    if bot_trades:
        print(f"{'Time':<20} {'Type':<6} {'Size':<4} {'Entry':<8} {'Exit':<8} {'P&L':<10} {'Reason':<20}")
        print("-" * 80)
        
        for trade in bot_trades[:10]:  # Show last 10
            timestamp = trade['timestamp'][:19] if trade.get('timestamp') else 'Unknown'
            position_type = trade.get('position_type', 'UNK')[:6]
            size = trade.get('size', 0)
            entry = trade.get('entry_price', 0)
            exit = trade.get('exit_price', 0)
            pnl = trade.get('net_pnl', 0)
            reason = trade.get('exit_reason', 'Unknown')[:20]
            
            print(f"{timestamp:<20} {position_type:<6} {size:<4} "
                  f"{entry:<8.2f} {exit:<8.2f} ${pnl:<9.2f} {reason:<20}")
    else:
        print("No bot trades found")
    
    # Show daily summary
    print("\n" + "=" * 60)
    print("DAILY SUMMARY")
    print("=" * 60)
    
    summary = trade_logger.daily_stats
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Winning Trades: {summary['winning_trades']}")
    print(f"Losing Trades: {summary['losing_trades']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Gross P&L: ${summary['gross_pnl']:.2f}")
    print(f"Total Fees: ${summary['fees']:.2f}")
    print(f"Net P&L: ${summary['net_pnl']:.2f}")
    print(f"Largest Win: ${summary['largest_win']:.2f}")
    print(f"Largest Loss: ${summary['largest_loss']:.2f}")
    print(f"Expectancy: ${summary['expectancy']:.2f}")
    
    print("\n" + "=" * 60)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconcile bot trades with broker')
    parser.add_argument('--bot', default='nq_bot', help='Bot name (default: nq_bot)')
    parser.add_argument('--account', help='TopStep account ID for broker reconciliation')
    parser.add_argument('--date', help='Date to reconcile (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        await reconcile_trades(
            bot_name=args.bot,
            account_id=args.account
        )
    finally:
        if topstepx_client.connected:
            await topstepx_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())