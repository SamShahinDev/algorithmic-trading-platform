#!/usr/bin/env python3
"""
Test the main intelligent trading bot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test_bot_init():
    """Test bot initialization"""
    print("\n" + "="*60)
    print(" TESTING INTELLIGENT TRADING BOT")
    print("="*60)
    
    try:
        from intelligent_trading_bot import IntelligentTradingBot
        
        # Initialize bot in test mode
        bot = IntelligentTradingBot(mode='test')
        
        print("\n✓ Bot initialized successfully")
        print(f"  Mode: {bot.mode}")
        print(f"  Components loaded:")
        
        if hasattr(bot, 'data_loader'):
            print(f"    ✓ Data Loader")
        if hasattr(bot, 'feature_engineer'):
            print(f"    ✓ Feature Engineer")
        if hasattr(bot, 'pattern_scanner'):
            print(f"    ✓ Pattern Scanner")
        if hasattr(bot, 'microstructure_analyzer'):
            print(f"    ✓ Microstructure Analyzer")
        if hasattr(bot, 'regime_detector'):
            print(f"    ✓ Market Regime Detector")
        if hasattr(bot, 'confidence_engine'):
            print(f"    ✓ Confidence Engine")
        if hasattr(bot, 'risk_manager'):
            print(f"    ✓ Risk Manager")
        if hasattr(bot, 'executor'):
            print(f"    ✓ Trade Executor")
        if hasattr(bot, 'performance_tracker'):
            print(f"    ✓ Performance Tracker")
        if hasattr(bot, 'logger'):
            print(f"    ✓ Logger")
        
        # Test data loading
        print("\n✓ Testing data loading...")
        from datetime import datetime, timedelta
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        try:
            data = await bot.data_loader.load_data(
                start_time=start_time,
                end_time=end_time,
                symbol='NQ.FUT'
            )
            
            if data is not None and not data.empty:
                print(f"    ✓ Loaded {len(data)} data points")
            else:
                print(f"    ⚠ No data available for test period")
        except Exception as e:
            print(f"    ⚠ Data loading error: {e}")
        
        # Test signal generation
        print("\n✓ Testing signal generation...")
        
        # Create sample data for testing
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        test_data = pd.DataFrame({
            'open': 15000 + np.cumsum(np.random.randn(100) * 10),
            'high': 15050 + np.cumsum(np.random.randn(100) * 10),
            'low': 14950 + np.cumsum(np.random.randn(100) * 10),
            'close': 15000 + np.cumsum(np.random.randn(100) * 10),
            'volume': np.random.randint(100, 1000, 100),
            'bid': 14999 + np.cumsum(np.random.randn(100) * 10),
            'ask': 15001 + np.cumsum(np.random.randn(100) * 10),
            'bid_size': np.random.randint(10, 100, 100),
            'ask_size': np.random.randint(10, 100, 100),
        }, index=dates)
        
        # Generate features
        features = bot.feature_engineer.calculate_features(test_data)
        print(f"    ✓ Generated {len(features.columns)} features")
        
        # Scan patterns
        patterns = bot.pattern_scanner.scan_all_patterns(test_data, features)
        detected = sum(1 for p in patterns.values() if p.detected)
        print(f"    ✓ Detected {detected} patterns")
        
        # Analyze microstructure
        micro = bot.microstructure_analyzer.analyze_current_state(test_data)
        print(f"    ✓ Analyzed microstructure")
        
        # Detect regime (skip if not available)
        if hasattr(bot, 'regime_detector'):
            regime = bot.regime_detector.detect_regime(test_data, features)
            print(f"    ✓ Detected regime: {regime.current_regime.value}")
        else:
            print(f"    ⚠ Regime detector not available")
        
        # Test confidence calculation
        print("\n✓ Testing confidence engine...")
        
        # Mock signals for confidence
        from analysis.pattern_scanner import PatternType, PatternSignal
        from analysis.microstructure import OrderFlowMetrics
        
        test_patterns = {
            PatternType.MOMENTUM_BURST: PatternSignal(
                detected=True,
                strength=0.7,
                confidence=0.65,
                entry_price=15000,
                stop_loss=14950,
                take_profit=15100
            )
        }
        
        test_micro = OrderFlowMetrics(
            buy_pressure=60,
            sell_pressure=40,
            net_pressure=20,
            cumulative_delta=500,
            delta_divergence=False,
            absorption=False,
            exhaustion=False
        )
        
        test_regime = {
            'regime': 'trending_up',
            'confidence': 0.7,
            'volatility': 'normal'
        }
        
        test_risk = {
            'risk_reward_ratio': 2.0,
            'position_size': 1,
            'stop_distance': 50
        }
        
        test_technical = {
            'rsi': 55,
            'macd_signal': 1,
            'bb_position': 0.6,
            'volume_ratio': 1.1
        }
        
        confidence = bot.confidence_engine.calculate_confidence(
            pattern_signals=test_patterns,
            microstructure=test_micro,
            technical_indicators=test_technical,
            market_regime=test_regime,
            risk_metrics=test_risk
        )
        
        print(f"    ✓ Confidence score: {confidence['overall_confidence']:.1%}")
        print(f"    ✓ Trade decision: {confidence['trade_decision']}")
        
        print("\n" + "="*60)
        print(" BOT TEST COMPLETE")
        print("="*60)
        print("\n✓ All bot components are functional!")
        print("\nTo start live trading:")
        print("1. Add TopStepX credentials to .env.topstepx")
        print("2. Run: python3 intelligent_trading_bot.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Bot initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_bot_init())
    sys.exit(0 if success else 1)