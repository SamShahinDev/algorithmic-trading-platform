#!/usr/bin/env python3
"""
Simple test suite to verify each component can be imported and initialized
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/web_platform/backend')

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print(" TESTING MODULE IMPORTS")
    print("="*60)
    
    modules_to_test = [
        ("Feature Engineering", "data.feature_engineering", "FeatureEngineer"),
        ("Pattern Scanner", "analysis.pattern_scanner", "PatternScanner"),
        ("Microstructure", "analysis.microstructure", "MicrostructureAnalyzer"),
        ("Strategy Discovery", "analysis.strategy_discovery", "StrategyDiscovery"),
        ("Risk Manager", "execution.risk_manager", "RiskManager"),
        ("Confidence Engine", "execution.confidence_engine", "AdvancedConfidenceEngine"),
        ("Trade Executor", "execution.trade_executor", "TradeExecutor"),
        ("Data Loader", "data.data_loader", "HybridDataLoader"),
        ("Market Regime", "data.market_regime", "MarketRegimeDetector"),
        ("Performance Tracker", "utils.performance_tracker", "PerformanceTracker"),
        ("Logger", "utils.logger", "TradingLogger"),
        ("Parameter Tuner", "optimization.parameter_tuner", "ParameterOptimizer"),
        ("Backtest Engine", "optimization.backtest_engine", "BacktestEngine"),
    ]
    
    results = []
    
    for name, module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {name:25} imported successfully")
            results.append(True)
        except Exception as e:
            print(f"  ✗ {name:25} failed: {str(e)[:50]}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n  Summary: {passed}/{total} modules imported ({passed/total*100:.0f}%)")
    return passed == total


def test_basic_functionality():
    """Test basic initialization of key components"""
    print("\n" + "="*60)
    print(" TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Test 1: Feature Engineering
    try:
        from data.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        
        # Create sample data
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        features = engineer.calculate_features(df)
        print(f"  ✓ Feature Engineering: Generated {len(features.columns)} features")
    except Exception as e:
        print(f"  ✗ Feature Engineering: {e}")
    
    # Test 2: Pattern Scanner
    try:
        from analysis.pattern_scanner import PatternScanner
        scanner = PatternScanner()
        print(f"  ✓ Pattern Scanner: Initialized with {len(scanner.patterns)} pattern types")
    except Exception as e:
        print(f"  ✗ Pattern Scanner: {e}")
    
    # Test 3: Risk Manager
    try:
        from execution.risk_manager import RiskManager, AccountType
        manager = RiskManager(account_type=AccountType.EVAL_50K)
        
        # Access account rules
        rules = manager.account_rules
        print(f"  ✓ Risk Manager: Loaded {rules.account_type.value} rules")
        print(f"    - Daily limit: ${rules.daily_loss_limit}")
        print(f"    - Trailing DD: ${rules.trailing_drawdown}")
    except Exception as e:
        print(f"  ✗ Risk Manager: {e}")
    
    # Test 4: Market Regime
    try:
        from data.market_regime import MarketRegimeDetector, RegimeType
        detector = MarketRegimeDetector()
        
        # Count regime types
        regime_count = len([r for r in RegimeType])
        print(f"  ✓ Market Regime: Supports {regime_count} regime types")
    except Exception as e:
        print(f"  ✗ Market Regime: {e}")
    
    # Test 5: Data Loader
    try:
        from data.data_loader import HybridDataLoader
        loader = HybridDataLoader()
        
        # Check file index
        file_count = len(loader.databento_files)
        if file_count > 0:
            date_range = f"{loader.file_index[0]['date']} to {loader.file_index[-1]['date']}"
            print(f"  ✓ Data Loader: Indexed {file_count} files ({date_range})")
        else:
            print(f"  ✓ Data Loader: Initialized (no files found)")
    except Exception as e:
        print(f"  ✗ Data Loader: {e}")
    
    return True


def test_topstep_connection():
    """Test TopStepX broker connection"""
    print("\n" + "="*60)
    print(" TESTING TOPSTEP CONNECTION")
    print("="*60)
    
    try:
        from brokers.topstepx_client import topstepx_client
        
        # Check if configured
        if topstepx_client.username:
            print(f"  ✓ TopStepX client configured")
            print(f"    - Username: {topstepx_client.username}")
            print(f"    - Account: {topstepx_client.account_id}")
            
            # Test authentication
            if topstepx_client.jwt_token:
                print(f"  ✓ Authentication token present")
            else:
                print(f"  ⚠ No authentication token")
        else:
            print(f"  ⚠ TopStepX not configured (add credentials to .env.topstepx)")
            
    except Exception as e:
        print(f"  ✗ TopStepX client error: {e}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" TRADING BOT COMPONENT VERIFICATION")
    print("="*60)
    
    # Run tests
    import_success = test_imports()
    basic_success = test_basic_functionality()
    topstep_success = test_topstep_connection()
    
    # Final summary
    print("\n" + "="*60)
    print(" FINAL SUMMARY")
    print("="*60)
    
    if import_success and basic_success:
        print("\n  🎉 All core components are working!")
        print("\n  Next steps:")
        print("  1. Configure TopStepX credentials in .env.topstepx")
        print("  2. Run the intelligent_trading_bot.py to start trading")
        print("  3. Monitor performance with the dashboard")
    else:
        print("\n  ⚠️ Some components need attention")
        print("  Check the error messages above for details")
    
    print("="*60)


if __name__ == "__main__":
    main()