"""
Pattern Configuration for Trading Bot
Centralized configuration for all trading patterns
"""

from datetime import time

# Pattern-specific configurations
PATTERN_CONFIG = {
    'trend_line_bounce': {
        'enabled': True,
        'max_daily_trades': 20,
        'risk_per_trade': 0.001,  # 0.1% of account
        'min_confidence': 0.60,  # Increased to 60%
        'max_spread_ticks': 1,
        'max_data_staleness_seconds': 2,
        'max_consecutive_losses': 3,
        
        # Pattern-specific parameters
        'stop_ticks': 6,
        'target_ticks_normal': 3,
        'target_ticks_high_conf': 5,
        
        # Time restrictions (ET)
        'time_restrictions': {
            'no_trade_start': '08:30',
            'no_trade_end': '09:15',
            'timezone': 'America/New_York'
        },
        
        # Multi-timeframe configuration
        'multi_timeframe': {
            '1m': {'weight': 0.4, 'enabled': True},
            '5m': {'weight': 0.3, 'enabled': True},
            '1h': {'weight': 0.3, 'enabled': True}
        },
        
        # Trend line detection parameters
        'trend_line': {
            'min_touches': 2,
            'max_lines_per_direction': 3,
            'min_r_squared': 0.95,
            'max_angle_degrees': 75,
            'touch_tolerance_pct': 0.001,
            'update_interval_seconds': 10
        },
        
        # Engulfing candle filter
        'engulfing_filter': {
            'atr_period': 14,
            'atr_multiplier': 1.5,
            'volume_multiplier': 2.0
        },
        
        # Position sizing
        'position_sizing': {
            'normal_confidence_contracts': 1,
            'high_confidence_contracts': 2,
            'high_confidence_threshold': 0.85
        },
        
        # Risk management
        'risk_management': {
            'max_loss_per_day': 1000,  # $1000
            'max_trades_per_hour': 5,
            'pause_after_consecutive_losses': 3,
            'pause_duration_minutes': 30
        },
        
        # Market regime requirements
        'regime_filter': {
            'enabled': True,
            'require_near_levels': True,
            'level_proximity_min_ticks': 4,
            'level_proximity_max_ticks': 6,
            'min_ransac_r2': 0.85,
            'blocked_times': [
                {'start': '08:27', 'end': '08:33', 'reason': 'news'},
                {'start': '09:30', 'end': '09:35', 'reason': 'open'}
            ]
        }
    },
    
    # Existing momentum thrust pattern configuration
    'momentum_thrust': {
        'enabled': True,
        'max_daily_trades': 10,
        'risk_per_trade': 0.001,
        'min_confidence': 0.60,  # Increased to 60%
        'max_spread_ticks': 1,
        'max_data_staleness_seconds': 2,
        'max_consecutive_losses': 3,
        
        # Pattern-specific parameters from discovery (relaxed for realistic trading)
        'lookback': 56,
        'momentum_threshold': 0.0005,  # was 0.0014 - realistic intraday level
        'volume_factor': 1.2,  # was 1.72 - more achievable volume spike
        'min_strength': 20,  # was 40 - lower ADX requirement
        
        # Market regime requirements
        'regime_filter': {
            'enabled': True,
            'min_adx': 18,
            'atr_band_low_multiplier': 1.2,  # × 24h median
            'atr_band_high_multiplier': 2.5,
            'allowed_time_start': '19:30',  # CT
            'allowed_time_end': '23:30',     # CT
            'time_zone': 'America/Chicago',
            'blocked_times': [
                {'start': '08:27', 'end': '08:33', 'reason': 'news'},
                {'start': '09:30', 'end': '09:35', 'reason': 'open'}
            ]
        }
    }
}

# Market data configuration
LIVE_MARKET_DATA = False   # False for TopStep practice/sim, True for funded live
CONTRACT_ID = "CON.F.US.ENQ.U25"  # Update on rollover

# Global trading configuration
GLOBAL_CONFIG = {
    'max_total_daily_trades': 50,
    'max_concurrent_positions': 1,  # Only 1 position at a time for NQ
    'emergency_stop_loss': 20,  # Emergency stop in ticks
    'daily_loss_limit': 2000,  # Maximum daily loss
    'profit_target': 5000,  # Daily profit target
    
    # Market hours (futures)
    'market_hours': {
        'sunday_open': '18:00',  # 6 PM ET Sunday
        'friday_close': '17:00',  # 5 PM ET Friday
        'timezone': 'America/New_York'
    },
    
    # API rate limiting
    'rate_limits': {
        'max_requests_per_minute': 180,
        'max_orders_per_minute': 20,
        'throttle_at_percentage': 80
    },
    
    # Logging and monitoring
    'monitoring': {
        'log_level': 'INFO',
        'metrics_interval_seconds': 60,
        'heartbeat_interval_seconds': 10,
        'save_state_interval_seconds': 300
    },
    
    # Market regime detection
    'regime_detection': {
        'enabled': True,
        'atr_history_bars': 1440,  # 24 hours of 1-minute bars
        'news_block_minutes': 3,    # ±3 minutes around news
        'news_time': '08:30',       # CT
        'open_block_start': '09:30', # CT
        'open_block_end': '09:35',   # CT
        'time_zone': 'America/Chicago'
    }
}

# Pattern priority (higher number = higher priority)
PATTERN_PRIORITY = {
    'trend_line_bounce': 10,  # Highest priority
    'momentum_thrust': 5
}

# Pattern combinations allowed
PATTERN_COMBINATIONS = {
    # Patterns that can trigger together
    'compatible': [
        ['trend_line_bounce', 'momentum_thrust']
    ],
    # Patterns that exclude each other
    'exclusive': []
}

def get_pattern_config(pattern_name: str) -> dict:
    """
    Get configuration for a specific pattern
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        Pattern configuration dictionary
    """
    return PATTERN_CONFIG.get(pattern_name, {})

def is_pattern_enabled(pattern_name: str) -> bool:
    """
    Check if a pattern is enabled
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        True if pattern is enabled
    """
    config = get_pattern_config(pattern_name)
    return config.get('enabled', False)

def get_all_enabled_patterns() -> list:
    """
    Get list of all enabled patterns
    
    Returns:
        List of enabled pattern names
    """
    return [name for name in PATTERN_CONFIG.keys() 
            if PATTERN_CONFIG[name].get('enabled', False)]

def validate_pattern_config():
    """
    Validate pattern configuration for consistency
    
    Raises:
        ValueError if configuration is invalid
    """
    total_max_trades = sum(config.get('max_daily_trades', 0) 
                          for config in PATTERN_CONFIG.values() 
                          if config.get('enabled', False))
    
    if total_max_trades > GLOBAL_CONFIG['max_total_daily_trades']:
        import logging
        logging.warning(
            f"Sum of pattern max daily trades ({total_max_trades}) exceeds "
            f"global limit ({GLOBAL_CONFIG['max_total_daily_trades']})"
        )
    
    # Check pattern priorities are unique
    priorities = [PATTERN_PRIORITY.get(name, 0) for name in PATTERN_CONFIG.keys()]
    if len(priorities) != len(set(priorities)):
        import logging
        logging.warning("Pattern priorities are not unique")

# Decision telemetry configuration
TRACE = {
    "regime": True, 
    "pattern": True, 
    "risk": True, 
    "exec": True, 
    "data": True,
    "near_miss_margin": 0.05  # also log PATTERN_EVAL where min-0.05 <= score < min
}

# MT Window Configuration
MT_WINDOW_CT = ("19:30", "23:30")
MT_MIN_CONFIDENCE_DEFAULT = 0.60
MT_MIN_CONFIDENCE_WINDOW = 0.55       # use this only in MT window
MT_CANCEL_IF_RUNS_TICKS = 6           # was 4
MT_EXHAUSTION_MULT_ATR = 1.40         # was 1.25, guarded by ADX >= 25
MT_EXHAUSTION_MULT_ATR_DEFAULT = 1.25
MT_ADX_TREND_GUARD = 25
MT_SECOND_CHANCE_BARS = 2             # watch next 2 bars after skip
MT_PULLBACK_MIN = 0.382               # unchanged
MT_VOLUME_FACTOR_DYNAMIC = True

# Practice/sim feed only
LIVE_MARKET_DATA = False
CONTRACT_ID = "CON.F.US.ENQ.U25"   # keep current

# Discovery Mode (24/7 practice trading)
DISCOVERY_MODE = True
ALLOW_24H_TRADING = True
DISABLE_REGIME_GATING = True
DISABLE_TIME_BLOCKS = True
DISABLE_RISK_THROTTLES = True

# --- Momentum Thrust discovery defaults (realistic intraday levels) ---
MT_MOMENTUM_THRESHOLD = 0.0005   # was 0.0014
MT_VOLUME_FACTOR_MIN  = 1.2      # was 1.72
MT_MIN_STRENGTH       = 20       # was 40

# RSI "healthy" zones (more permissive but still sane)
MT_RSI_LONG_MIN  = 45
MT_RSI_LONG_MAX  = 72
MT_RSI_SHORT_MIN = 28
MT_RSI_SHORT_MAX = 55

# --- Dual confirmation for MT ---
MT_CONFIRM_MODE = "dual"          # "breakout" | "thrust" | "dual"
MT_BREAKOUT_LOOKBACK = 20         # N-bar high/low for breakout

# Thrust confirm thresholds (tuned for typical NQ tape)
MT_THRUST_BODY_MIN        = 0.60  # real body / range on confirm bar
MT_THRUST_RANGE_ATR_MIN   = 0.80  # confirm range vs ATR(14)
MT_THRUST_MOM_MIN         = 0.0003
MT_THRUST_VOL_MIN         = 1.15

# Confidence control
PATTERN_MIN_CONFIDENCE = 0.60    # production default
GLOBAL_MIN_CONFIDENCE  = 0.35    # slightly lower to see first passes

# In discovery, use GLOBAL_MIN_CONFIDENCE for all patterns
USE_GLOBAL_MIN_CONF_WHEN_DISCOVERY = True

# Retest / execution sanity
CANCEL_IF_RUNS_BYPASS_DISCOVERY = True
RETEST_LIMIT_OFFSET_TICKS = 2

# Keep essential protections
MAX_SLIPPAGE_TICKS = 2
OUTER_FAILSAFE_TICKS = 20        # $100 outer guard
POSITION_MAX_CONTRACTS = 1       # cap size while discovering
SIGNAL_COOLDOWN_SEC = 60         # avoid spam entries

# ATR Baseline Configuration
ATR_BASELINE_LOOKBACK_BARS = 1500   # ~24-25h of 1m bars
ATR_BASELINE_MIN_FALLBACK = 200     # use this if 1500 not available

# Telemetry Configuration
TELEMETRY = {
    "csv_eval_all": True,   # write every PATTERN_EVAL to CSV
    "csv_exec": True        # write fills/exits too
}

# Probe Trade Configuration (discovery/practice only)
PROBE = {
    "enabled": True,               # discovery/practice only
    "idle_minutes": 10,            # if no fills in X minutes, consider a probe
    "size": 1,
    "t1_ticks": 4,                 # small target
    "stop_ticks": 8,               # small stop
    "max_slippage_ticks": 2
}

# Validate configuration on import
validate_pattern_config()