"""
Feature Flags for Safe Trading Bot Operation
Controls execution behavior and safety checks
"""

import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureFlags:
    """Central configuration for bot behavior"""
    
    def __init__(self):
        # Trading mode flags
        self.PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        self.DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'
        
        # Confidence thresholds
        self.SKIP_CONFIDENCE_CHECK = os.getenv('SKIP_CONFIDENCE_CHECK', 'false').lower() == 'true'
        self.MIN_CONFIDENCE_OVERRIDE = float(os.getenv('MIN_CONFIDENCE_OVERRIDE', '30'))  # Lower threshold for testing
        self.NORMAL_MIN_CONFIDENCE = 50  # Normal production threshold
        
        # Position limits
        self.MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '1'))
        self.MAX_POSITION_SIZE = int(os.getenv('MAX_POSITION_SIZE', '1'))
        
        # Risk limits
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '500'))
        self.MAX_TRADE_LOSS = float(os.getenv('MAX_TRADE_LOSS', '150'))
        
        # Pattern controls
        self.ENABLE_MOMENTUM_THRUST = os.getenv('ENABLE_MOMENTUM_THRUST', 'true').lower() == 'true'
        self.ENABLE_TREND_LINE_BOUNCE = os.getenv('ENABLE_TREND_LINE_BOUNCE', 'true').lower() == 'true'
        self.ENABLE_AGGRESSIVE_PATTERNS = os.getenv('ENABLE_AGGRESSIVE_PATTERNS', 'false').lower() == 'true'
        
        # Timing controls
        self.TRADE_DURING_NEWS = os.getenv('TRADE_DURING_NEWS', 'false').lower() == 'true'
        self.TRADE_PREMARKET = os.getenv('TRADE_PREMARKET', 'false').lower() == 'true'
        self.TRADE_AFTERHOURS = os.getenv('TRADE_AFTERHOURS', 'false').lower() == 'true'
        
        # Logging
        self.VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'
        self.LOG_ALL_SIGNALS = os.getenv('LOG_ALL_SIGNALS', 'true').lower() == 'true'
        
        self._log_configuration()
    
    def _log_configuration(self):
        """Log current configuration"""
        config = {
            "Mode": "PAPER" if self.PAPER_TRADING else "LIVE",
            "Dry Run": self.DRY_RUN,
            "Skip Confidence": self.SKIP_CONFIDENCE_CHECK,
            "Min Confidence": self.get_min_confidence(),
            "Max Positions": self.MAX_OPEN_POSITIONS,
            "Max Daily Loss": self.MAX_DAILY_LOSS,
            "Patterns": {
                "Momentum": self.ENABLE_MOMENTUM_THRUST,
                "Trend Line": self.ENABLE_TREND_LINE_BOUNCE,
                "Aggressive": self.ENABLE_AGGRESSIVE_PATTERNS
            }
        }
        
        logger.info("=" * 60)
        logger.info("FEATURE FLAGS CONFIGURATION")
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  - {k}: {v}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 60)
        
        # Safety warnings
        if not self.PAPER_TRADING:
            logger.warning("⚠️ LIVE TRADING MODE ENABLED ⚠️")
        
        if self.SKIP_CONFIDENCE_CHECK:
            logger.warning("⚠️ CONFIDENCE CHECKS DISABLED - USE WITH CAUTION ⚠️")
        
        if self.DRY_RUN:
            logger.info("DRY RUN MODE - Orders will be logged but not sent")
    
    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold based on mode"""
        if self.SKIP_CONFIDENCE_CHECK:
            return 0  # No minimum
        elif self.MIN_CONFIDENCE_OVERRIDE < self.NORMAL_MIN_CONFIDENCE:
            logger.warning(f"Using reduced confidence threshold: {self.MIN_CONFIDENCE_OVERRIDE}%")
            return self.MIN_CONFIDENCE_OVERRIDE
        else:
            return self.NORMAL_MIN_CONFIDENCE
    
    def can_execute_trade(self, confidence: float) -> bool:
        """Check if trade can be executed based on confidence"""
        min_conf = self.get_min_confidence()
        
        if self.SKIP_CONFIDENCE_CHECK:
            logger.debug(f"Confidence check skipped (confidence: {confidence:.1f}%)")
            return True
        
        return confidence >= min_conf
    
    def get_position_size(self, base_size: int = 1) -> int:
        """Get allowed position size"""
        return min(base_size, self.MAX_POSITION_SIZE)
    
    def should_log_signal(self, confidence: float) -> bool:
        """Determine if signal should be logged"""
        if self.LOG_ALL_SIGNALS:
            return True
        return confidence >= self.get_min_confidence() * 0.8  # Log signals close to threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "paper_trading": self.PAPER_TRADING,
            "dry_run": self.DRY_RUN,
            "skip_confidence_check": self.SKIP_CONFIDENCE_CHECK,
            "min_confidence": self.get_min_confidence(),
            "max_open_positions": self.MAX_OPEN_POSITIONS,
            "max_position_size": self.MAX_POSITION_SIZE,
            "max_daily_loss": self.MAX_DAILY_LOSS,
            "patterns_enabled": {
                "momentum_thrust": self.ENABLE_MOMENTUM_THRUST,
                "trend_line_bounce": self.ENABLE_TREND_LINE_BOUNCE,
                "aggressive": self.ENABLE_AGGRESSIVE_PATTERNS
            }
        }