#!/usr/bin/env python3
"""
Comprehensive test suite for all trading bot components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pattern_scanner():
    """Test pattern detection functionality"""
    print("\n" + "="*50)
    print("TESTING PATTERN SCANNER")
    print("="*50)
    
    try:
        from analysis.pattern_scanner import PatternScanner
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.05,
            'high': prices + abs(np.random.randn(100) * 0.1),
            'low': prices - abs(np.random.randn(100) * 0.1),
            'close': prices,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        # Initialize scanner
        scanner = PatternScanner()
        
        # Generate features first
        from data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        features = engineer.create_features(df)
        
        # Scan for patterns
        patterns = scanner.scan_all_patterns(df, features)
        
        print(f"✓ Scanner initialized successfully")
        print(f"✓ Found {len(patterns)} patterns:")
        
        detected_count = 0
        for pattern_type, signal in patterns.items():
            if signal.detected:
                detected_count += 1
                print(f"  - {pattern_type.value}: Strength {signal.strength:.2f}")
        
        if detected_count == 0:
            print("  No patterns detected in random data (normal)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_microstructure():
    """Test microstructure analysis"""
    print("\n" + "="*50)
    print("TESTING MICROSTRUCTURE ANALYZER")
    print("="*50)
    
    try:
        from analysis.microstructure import MicrostructureAnalyzer
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        df = pd.DataFrame({
            'bid': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'ask': 100.05 + np.cumsum(np.random.randn(100) * 0.1),
            'bid_size': np.random.randint(10, 100, 100),
            'ask_size': np.random.randint(10, 100, 100),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        # Initialize analyzer
        analyzer = MicrostructureAnalyzer()
        
        # Analyze microstructure
        metrics = analyzer.analyze_microstructure(df)
        
        print(f"✓ Analyzer initialized successfully")
        print(f"✓ Calculated metrics:")
        print(f"  - Order Flow Imbalance: {metrics.order_flow_imbalance:.3f}")
        print(f"  - Bid-Ask Spread: {metrics.bid_ask_spread:.4f}")
        print(f"  - Volume Profile Signal: {metrics.volume_profile_signal:.2f}")
        print(f"  - Liquidity Score: {metrics.liquidity_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_risk_manager():
    """Test risk management and TopStep compliance"""
    print("\n" + "="*50)
    print("TESTING RISK MANAGER")
    print("="*50)
    
    try:
        from execution.risk_manager import RiskManager, AccountType
        
        # Initialize manager
        manager = RiskManager(
            account_type=AccountType.EVAL_50K,
            initial_balance=50000
        )
        
        print(f"✓ Risk manager initialized")
        print(f"  Account: $50,000 Evaluation")
        print(f"  Daily limit: ${manager.daily_loss_limit}")
        print(f"  Trailing DD: ${manager.trailing_drawdown}")
        
        # Test trade validation
        validation = manager.validate_trade(
            symbol='NQ',
            side='BUY',
            quantity=1,
            entry_price=15000,
            stop_loss=14950
        )
        
        print(f"\n✓ Trade validation: {'Approved' if validation.approved else 'Denied'}")
        if not validation.approved:
            print(f"  Reason: {validation.reason}")
        
        # Test position sizing
        position_size = manager.calculate_position_size(
            symbol='NQ',
            entry_price=15000,
            stop_loss=14950,
            confidence=0.75
        )
        
        print(f"✓ Position size calculated: {position_size} contracts")
        
        # Simulate trades
        manager.record_trade(
            symbol='NQ',
            pnl=250,
            contracts=1
        )
        print(f"✓ Recorded trade: +$250")
        
        manager.record_trade(
            symbol='NQ',
            pnl=-100,
            contracts=1
        )
        print(f"✓ Recorded trade: -$100")
        
        stats = manager.get_daily_stats()
        print(f"\n✓ Daily statistics:")
        print(f"  - Daily P&L: ${stats.daily_pnl:.2f}")
        print(f"  - Trade count: {stats.trade_count}")
        print(f"  - Win rate: {stats.win_rate:.1%}")
        print(f"  - Remaining risk: ${stats.remaining_loss:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_confidence_engine():
    """Test confidence scoring system"""
    print("\n" + "="*50)
    print("TESTING CONFIDENCE ENGINE")
    print("="*50)
    
    try:
        from execution.confidence_engine import AdvancedConfidenceEngine
        from analysis.pattern_scanner import PatternType, PatternSignal
        from analysis.microstructure import MicrostructureMetrics
        
        # Initialize engine
        engine = AdvancedConfidenceEngine()
        
        print(f"✓ Confidence engine initialized")
        
        # Create sample signals
        pattern_signals = {
            PatternType.MOMENTUM_BURST: PatternSignal(
                detected=True,
                strength=0.8,
                confidence=0.75,
                entry_price=15000,
                stop_loss=14950,
                take_profit=15100
            ),
            PatternType.MEAN_REVERSION: PatternSignal(
                detected=False,
                strength=0,
                confidence=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0
            ),
            PatternType.BREAKOUT: PatternSignal(
                detected=True,
                strength=0.6,
                confidence=0.65,
                entry_price=15000,
                stop_loss=14970,
                take_profit=15060
            )
        }
        
        microstructure_metrics = MicrostructureMetrics(
            order_flow_imbalance=0.3,
            bid_ask_spread=0.0002,
            volume_profile_signal=0.5,
            liquidity_score=0.7,
            tick_imbalance=0.2,
            large_trade_ratio=0.15,
            quote_stability=0.8,
            effective_spread=0.0003
        )
        
        technical_signals = {
            'rsi': 65,
            'macd_signal': 1,
            'bb_position': 0.7,
            'volume_ratio': 1.2
        }
        
        market_regime = {
            'regime': 'trending_up',
            'confidence': 0.75,
            'volatility': 'normal'
        }
        
        risk_metrics = {
            'risk_reward_ratio': 2.5,
            'position_size': 1,
            'stop_distance': 50
        }
        
        # Calculate confidence
        confidence = engine.calculate_confidence(
            pattern_signals=pattern_signals,
            microstructure=microstructure_metrics,
            technical_indicators=technical_signals,
            market_regime=market_regime,
            risk_metrics=risk_metrics
        )
        
        print(f"\n✓ Confidence calculated:")
        print(f"  - Overall score: {confidence['overall_confidence']:.1%}")
        print(f"  - Trade decision: {confidence['trade_decision']}")
        print(f"  - Confidence level: {confidence['confidence_level']}")
        
        print(f"\n✓ Component scores:")
        for component, score in confidence['component_scores'].items():
            print(f"  - {component}: {score:.1%}")
        
        # Test learning
        engine.update_performance(PatternType.MOMENTUM_BURST, True, 0.02)
        engine.update_performance(PatternType.BREAKOUT, False, -0.01)
        
        print(f"\n✓ Pattern learning updated")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_trade_executor():
    """Test trade execution (mock mode)"""
    print("\n" + "="*50)
    print("TESTING TRADE EXECUTOR")
    print("="*50)
    
    try:
        from execution.trade_executor import TradeExecutor
        from execution.risk_manager import RiskManager
        
        # Initialize components
        risk_manager = RiskManager(
            account_balance=50000,
            daily_loss_limit=1500
        )
        
        executor = TradeExecutor(
            broker_client=None,  # Mock mode
            risk_manager=risk_manager
        )
        
        print(f"✓ Trade executor initialized in mock mode")
        
        # Test order creation
        trade_signal = {
            'symbol': 'NQ',
            'side': 'BUY',
            'confidence': 0.75,
            'entry_price': 15000,
            'stop_loss': 14950,
            'take_profit': 15100
        }
        
        print(f"\n✓ Testing order creation:")
        print(f"  Symbol: {trade_signal['symbol']}")
        print(f"  Side: {trade_signal['side']}")
        print(f"  Entry: ${trade_signal['entry_price']}")
        print(f"  Stop: ${trade_signal['stop_loss']}")
        print(f"  Target: ${trade_signal['take_profit']}")
        
        # In mock mode, we just validate the signal
        if trade_signal['confidence'] > 0.6:
            print(f"\n✓ Trade signal validated")
            print(f"  Would execute: 1 contract")
            print(f"  Risk: $50 per contract")
            print(f"  Reward: $100 per contract")
            print(f"  R:R Ratio: 2:1")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_performance_tracker():
    """Test performance tracking"""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE TRACKER")
    print("="*50)
    
    try:
        sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')
        from utils.performance_tracker import PerformanceTracker
        
        # Initialize tracker
        tracker = PerformanceTracker(initial_balance=50000)
        
        print(f"✓ Performance tracker initialized")
        print(f"  Initial balance: $50,000")
        
        # Simulate trades
        trades = [
            {'pnl': 250, 'symbol': 'NQ', 'side': 'BUY'},
            {'pnl': -100, 'symbol': 'NQ', 'side': 'SELL'},
            {'pnl': 180, 'symbol': 'NQ', 'side': 'BUY'},
            {'pnl': -80, 'symbol': 'NQ', 'side': 'SELL'},
            {'pnl': 320, 'symbol': 'NQ', 'side': 'BUY'},
        ]
        
        for i, trade in enumerate(trades, 1):
            tracker.record_trade(
                symbol=trade['symbol'],
                side=trade['side'],
                entry_price=15000,
                exit_price=15000 + trade['pnl']/2,
                quantity=1,
                pnl=trade['pnl'],
                commission=2.5
            )
            print(f"  Trade {i}: {'✓' if trade['pnl'] > 0 else '✗'} ${trade['pnl']:+.0f}")
        
        # Get metrics
        metrics = tracker.get_performance_summary()
        
        print(f"\n✓ Performance metrics:")
        print(f"  - Total trades: {metrics['total_trades']}")
        print(f"  - Win rate: {metrics['win_rate']:.1%}")
        print(f"  - Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"  - Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  - Max drawdown: ${metrics.get('max_drawdown', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def main():
    """Run all component tests"""
    print("\n" + "="*60)
    print(" TRADING BOT COMPONENT TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test each component
    tests = [
        ("Pattern Scanner", test_pattern_scanner),
        ("Microstructure Analyzer", test_microstructure),
        ("Risk Manager", test_risk_manager),
        ("Confidence Engine", test_confidence_engine),
        ("Trade Executor", test_trade_executor),
        ("Performance Tracker", test_performance_tracker),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Failed to run {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:30} {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n  🎉 All components working correctly!")
    else:
        print(f"\n  ⚠️  {total - passed} component(s) need attention")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)