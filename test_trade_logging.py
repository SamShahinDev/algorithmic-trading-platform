#!/usr/bin/env python3
"""
Test script for trade logging system
Creates mock trades to verify logging works correctly
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from trading_bot.utils.trade_logger import TradeLogger
from trading_bot.analysis.optimized_pattern_scanner import PatternType


@dataclass
class MockPosition:
    """Mock position for testing"""
    symbol: str
    side: int  # 0=BUY/LONG, 1=SELL/SHORT
    position_type: int  # 1=LONG, 2=SHORT
    size: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    pattern: Optional[PatternType]
    confidence: float
    order_id: Optional[str] = None
    pnl: float = 0
    status: str = "open"
    max_profit: float = 0
    max_loss: float = 0


def test_trade_logging():
    """Test the trade logging system with mock trades"""
    print("=" * 60)
    print("TESTING TRADE LOGGING SYSTEM")
    print("=" * 60)
    
    # Initialize trade logger
    trade_logger = TradeLogger(bot_name="test_bot")
    print("✓ Trade logger initialized")
    
    # Create test trades
    test_trades = [
        # Winning long trade
        {
            'position': MockPosition(
                symbol="NQZ4",
                side=0,
                position_type=1,  # LONG
                size=1,
                entry_price=20425.00,
                entry_time=datetime.now(timezone.utc) - timedelta(minutes=30),
                stop_loss=20420.00,
                take_profit=20435.00,
                pattern=None,  # PatternType.BULL_FLAG if available
                confidence=0.75,
                order_id="TEST001"
            ),
            'exit_price': 20435.00,
            'exit_reason': "Take profit hit"
        },
        # Losing long trade
        {
            'position': MockPosition(
                symbol="NQZ4",
                side=0,
                position_type=1,  # LONG
                size=2,
                entry_price=20450.00,
                entry_time=datetime.now(timezone.utc) - timedelta(minutes=20),
                stop_loss=20445.00,
                take_profit=20460.00,
                pattern=None,  # PatternType.ASCENDING_TRIANGLE if available
                confidence=0.65,
                order_id="TEST002"
            ),
            'exit_price': 20445.00,
            'exit_reason': "Stop loss hit"
        },
        # Winning short trade
        {
            'position': MockPosition(
                symbol="NQZ4",
                side=1,
                position_type=2,  # SHORT
                size=1,
                entry_price=20480.00,
                entry_time=datetime.now(timezone.utc) - timedelta(minutes=15),
                stop_loss=20485.00,
                take_profit=20470.00,
                pattern=None,  # PatternType.BEAR_FLAG if available
                confidence=0.80,
                order_id="TEST003"
            ),
            'exit_price': 20470.00,
            'exit_reason': "Take profit hit"
        },
        # Losing short trade
        {
            'position': MockPosition(
                symbol="NQZ4",
                side=1,
                position_type=2,  # SHORT
                size=1,
                entry_price=20465.00,
                entry_time=datetime.now(timezone.utc) - timedelta(minutes=10),
                stop_loss=20470.00,
                take_profit=20455.00,
                pattern=None,
                confidence=0.55,
                order_id="TEST004"
            ),
            'exit_price': 20470.00,
            'exit_reason': "Stop loss hit"
        }
    ]
    
    print(f"\nRecording {len(test_trades)} test trades...")
    print("-" * 60)
    
    # Record each trade
    for i, trade_data in enumerate(test_trades, 1):
        position = trade_data['position']
        exit_price = trade_data['exit_price']
        exit_reason = trade_data['exit_reason']
        
        print(f"\nTrade {i}:")
        print(f"  Type: {'LONG' if position.position_type == 1 else 'SHORT'}")
        print(f"  Entry: {position.entry_price:.2f}")
        print(f"  Exit: {exit_price:.2f}")
        print(f"  Size: {position.size}")
        
        # Record the trade
        trade_record = trade_logger.record_trade(
            position=position,
            exit_price=exit_price,
            exit_reason=exit_reason
        )
        
        if trade_record:
            print(f"  ✓ Trade recorded: ID {trade_record['trade_id'][:8]}...")
            print(f"  Net P&L: ${trade_record['net_pnl']:.2f}")
        else:
            print(f"  ✗ Failed to record trade")
    
    # Save daily summary
    print("\n" + "=" * 60)
    print("DAILY SUMMARY")
    print("=" * 60)
    
    summary = trade_logger.save_daily_summary()
    
    # Verify files were created
    print("\n" + "=" * 60)
    print("FILE VERIFICATION")
    print("=" * 60)
    
    files_to_check = [
        trade_logger.json_log_file,
        trade_logger.csv_log_file,
        trade_logger.db_file,
        trade_logger.daily_summary_file
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {file_path.name}: {size} bytes")
        else:
            print(f"✗ {file_path.name}: NOT FOUND")
    
    # Test database query
    print("\n" + "=" * 60)
    print("DATABASE QUERY TEST")
    print("=" * 60)
    
    recent_trades = trade_logger.get_recent_trades(limit=5)
    print(f"Retrieved {len(recent_trades)} trades from database")
    
    if recent_trades:
        print("\nRecent trades:")
        for trade in recent_trades:
            print(f"  - {trade['position_type']} {trade['size']} @ {trade['entry_price']:.2f} "
                  f"-> {trade['exit_price']:.2f} | P&L: ${trade['net_pnl']:.2f}")
    
    # Test reconciliation (mock)
    print("\n" + "=" * 60)
    print("RECONCILIATION TEST")
    print("=" * 60)
    
    mock_broker_trades = [
        {'order_id': 'TEST001', 'pnl': 195},  # Should match
        {'order_id': 'TEST002', 'pnl': -210},  # Should match
        {'order_id': 'TEST005', 'pnl': 100},  # Missing in bot
    ]
    
    report = trade_logger.reconcile_with_broker(mock_broker_trades)
    
    if report.get('reconciled'):
        print("✅ All trades reconciled")
    else:
        print("⚠️ Reconciliation issues found:")
        print(f"  Missing in bot: {len(report.get('missing_in_bot', []))}")
        print(f"  Missing in broker: {len(report.get('missing_in_broker', []))}")
        print(f"  P&L mismatches: {len(report.get('pnl_mismatches', []))}")
    
    print("\n" + "=" * 60)
    print("✅ TRADE LOGGING SYSTEM TEST COMPLETE")
    print("=" * 60)
    
    # Note about cleanup
    print("\nTest files created in logs/trades/")
    print("To clean up test files, manually delete files starting with 'test_bot'")
    
    return True  # Test successful


if __name__ == "__main__":
    test_trade_logging()