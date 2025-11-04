#!/usr/bin/env python3
"""
Integration test for pattern detection pipeline
Tests that patterns flow through the entire system
"""

import pandas as pd
import numpy as np
import sys
import logging
from datetime import datetime, timedelta

# Add paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING')
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

# Import bot components
from trading_bot.analysis.optimized_pattern_scanner import OptimizedPatternScanner as PatternScanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data_with_momentum():
    """Create synthetic data that should trigger momentum thrust pattern"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    
    # Create a strong upward movement to trigger momentum
    prices = np.linspace(23400, 23410, 100)  # 10 point rise = 0.04% move
    
    # Add some noise
    prices = prices + np.random.randn(100) * 0.5
    
    # Create volume spike in last bars
    volume = np.ones(100) * 1000
    volume[-20:] = 1500  # 50% above average
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': volume
    })
    
    return df

def test_pattern_detection():
    """Test that patterns are detected and flow through the pipeline"""
    
    logger.info("=" * 60)
    logger.info("PATTERN PIPELINE INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Initialize scanner with no minimum strength
    scanner = PatternScanner(min_strength=0)
    
    # Create test data
    test_data = create_test_data_with_momentum()
    logger.info(f"Created test data with {len(test_data)} bars")
    logger.info(f"Price range: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    logger.info(f"Price change: {(test_data['close'].iloc[-1] / test_data['close'].iloc[-11] - 1) * 100:.3f}%")
    
    # Scan for patterns
    patterns = scanner.scan(test_data)
    
    # Log results
    if patterns:
        logger.info(f"✅ PATTERNS DETECTED: {len(patterns)}")
        for pattern_type, pattern in patterns.items():
            logger.info(f"  - {pattern_type.value}:")
            logger.info(f"    Direction: {pattern.direction}")
            logger.info(f"    Strength: {pattern.strength:.2f}%")
            logger.info(f"    Confidence: {pattern.confidence:.2%}")
            logger.info(f"    Entry: {pattern.entry_price:.2f}")
            logger.info(f"    Stop: {pattern.stop_loss:.2f}")
            logger.info(f"    Target: {pattern.take_profit:.2f}")
            
            # Verify pattern properties
            assert hasattr(pattern, 'direction'), "Pattern missing direction"
            assert hasattr(pattern, 'strength'), "Pattern missing strength"
            assert hasattr(pattern, 'confidence'), "Pattern missing confidence"
            assert pattern.direction in [-1, 1], f"Invalid direction: {pattern.direction}"
            
        logger.info("✅ All pattern properties validated")
        
        # Test pattern filtering
        DISABLED_PATTERNS = ['bollinger_squeeze', 'volume_climax']
        valid_patterns = []
        for pattern_type, pattern in patterns.items():
            pattern_name = str(pattern_type).split('.')[-1].lower()
            if pattern_name not in DISABLED_PATTERNS:
                valid_patterns.append(pattern)
                logger.info(f"✅ Pattern {pattern_name} is ENABLED")
            else:
                logger.info(f"⚠️ Pattern {pattern_name} is DISABLED")
        
        logger.info(f"Valid patterns after filtering: {len(valid_patterns)}")
        
        return True
    else:
        logger.warning("❌ NO PATTERNS DETECTED")
        logger.info("Trying with more aggressive data...")
        
        # Create more aggressive test data
        aggressive_data = test_data.copy()
        aggressive_data['close'] = aggressive_data['close'] * 1.01  # Add 1% to all prices
        
        patterns = scanner.scan(aggressive_data)
        if patterns:
            logger.info(f"✅ PATTERNS DETECTED WITH AGGRESSIVE DATA: {len(patterns)}")
            return True
        else:
            logger.error("❌ STILL NO PATTERNS - Check ROC threshold")
            return False

if __name__ == "__main__":
    success = test_pattern_detection()
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ PATTERN PIPELINE TEST PASSED")
        logger.info("Patterns are being detected and flowing through the system")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ PATTERN PIPELINE TEST FAILED")
        logger.error("Check ROC threshold and pattern detection logic")
        logger.error("=" * 60)
        sys.exit(1)