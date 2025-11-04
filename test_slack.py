#!/usr/bin/env python3
"""
Test Slack integration for Trading Bot
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.slack_notifier import slack_notifier, ChannelType, MessagePriority

async def test_slack_notifications():
    """Test all types of Slack notifications"""
    
    print("🧪 Testing Trading Bot Slack Integration")
    print("=" * 50)
    
    # Test 1: System Status
    print("\n1️⃣ Testing system status...")
    await slack_notifier.system_status(
        "Test Mode Active",
        "Running integration tests for all notification types"
    )
    await asyncio.sleep(1)
    
    # Test 2: Pattern Discovery
    print("2️⃣ Testing pattern discovery...")
    await slack_notifier.pattern_discovered(
        "Test Trend Line Bounce",
        "trend_bounce",
        0.753,
        {
            'win_rate': 0.682,
            'occurrences': 42,
            'profit_factor': 2.1
        }
    )
    await asyncio.sleep(1)
    
    # Test 3: Backtest Complete
    print("3️⃣ Testing backtest results...")
    await slack_notifier.backtest_complete(
        "Test Pattern",
        {
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.08,
            'total_trades': 150,
            'sample_size': 150
        }
    )
    await asyncio.sleep(1)
    
    # Test 4: Trade Execution
    print("4️⃣ Testing trade notifications...")
    # Entry
    await slack_notifier.trade_executed({
        'action': 'Entry',
        'pattern_name': 'Test Pattern',
        'direction': 'long',
        'price': 23456.50,
        'quantity': 2,
        'trade_id': 'TEST001'
    })
    await asyncio.sleep(1)
    
    # Exit
    await slack_notifier.trade_executed({
        'action': 'Exit',
        'pattern_name': 'Test Pattern',
        'direction': 'long',
        'price': 23556.50,
        'quantity': 2,
        'pnl': 400.00,
        'trade_id': 'TEST001'
    })
    await asyncio.sleep(1)
    
    # Test 5: Risk Alert
    print("5️⃣ Testing risk alerts...")
    await slack_notifier.risk_alert(
        'drawdown',
        {
            'message': 'Portfolio drawdown exceeding threshold',
            'metrics': {
                'Current Drawdown': '6.2%',
                'Threshold': '5%',
                'Peak Equity': '$51,000',
                'Current Equity': '$47,838'
            }
        }
    )
    await asyncio.sleep(1)
    
    # Test 6: Performance Update
    print("6️⃣ Testing performance updates...")
    await slack_notifier.performance_update({
        'daily_pnl': 1250.00,
        'total_trades': 8,
        'win_rate': 0.625,
        'active_patterns': 5,
        'best_pattern': 'Trend Line Bounce',
        'worst_pattern': 'Failed Breakout'
    })
    await asyncio.sleep(1)
    
    # Test 7: Market Regime Change
    print("7️⃣ Testing regime changes...")
    await slack_notifier.regime_change(
        'ranging',
        'trending_up',
        0.82
    )
    await asyncio.sleep(1)
    
    # Test 8: ML Pattern
    print("8️⃣ Testing ML pattern discovery...")
    await slack_notifier.ml_pattern_found(
        'kmeans_cluster',
        7,
        {
            'occurrences': 28,
            'win_rate': 0.714,
            'confidence': 0.68
        }
    )
    await asyncio.sleep(1)
    
    # Test 9: Monte Carlo Results
    print("9️⃣ Testing Monte Carlo results...")
    await slack_notifier.monte_carlo_complete(
        "Test Pattern",
        {
            'iterations': 1000,
            'median_win_rate': 0.65,
            'win_rate_std': 0.08,
            'robustness_score': 0.72
        }
    )
    
    # Final message
    await asyncio.sleep(2)
    await slack_notifier.system_status(
        "Test Complete",
        "✅ All notification types tested successfully!\n\nCheck each channel for the test messages."
    )
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nCheck your Slack channels:")
    print("  • #trading-orchestrator - System messages")
    print("  • #trading-patterns - Pattern discovery")
    print("  • #trading-backtest - Backtest results")
    print("  • #trading-live - Trade notifications")
    print("  • #trading-risk - Risk alerts")
    print("  • #trading-performance - Performance updates")
    print("  • #trading-regime - Market regime changes")
    print("  • #trading-ml - ML discoveries")

if __name__ == "__main__":
    print("🚀 Trading Bot Slack Test Suite")
    print("This will send test messages to all channels")
    print()
    
    try:
        asyncio.run(test_slack_notifications())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)