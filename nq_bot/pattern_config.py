"""
Pattern Configuration for Trading Bot
Centralized configuration for all trading patterns
"""

from datetime import time
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class FVGProfile:
    name: str
    displacement_body_frac_min_base: float
    displacement_body_frac_min_high_vol: float
    displacement_atr_multiple: float
    displacement_min_points_floor: float
    volume_min_mult_trend: float
    volume_min_mult_sweep: float
    quality_score_min_trend: float
    defense_max_fill_pct: float

@dataclass
class ResponsiveArming:
    arm_on_wick: bool = True          # arm as soon as price touches zone
    arm_close_outside_pct: float = 0.10  # 10% instead of 25% for close-based arming
    micro_fvg_max_ticks: int = 2      # zones ≤2 ticks = micro
    micro_front_run_ticks: int = 1    # enter edge±1 tick for micro zones
    micro_rsi_relax: bool = True
    micro_rsi_relax_points: int = 5   # widen RSI bounds by ±5 on micro

@dataclass
class SessionProfile:
    name: str
    displacement_body_frac_min_base: float
    displacement_body_frac_min_high_vol: float  # Add missing attribute
    displacement_min_points_floor: float
    displacement_atr_multiple: float
    volume_min_mult_trend: float
    volume_min_mult_sweep: float
    quality_score_min_trend: float
    defense_max_fill_pct: float
    # Optional: small RSI widen for non-NY open (keeps your RSI logic the same interface)
    rsi_relax_points: int = 0   # e.g., 0 in NY, 5 in INTL

@dataclass
class SessionSchedule:
    exchange_tz: str = "America/Chicago"
    # CT windows (24h). Adjust if you prefer different cuts.
    tokyo_start: str = "19:00"
    tokyo_end: str = "01:00"
    london_start: str = "01:00"
    london_end: str = "07:00"
    ny_rth_start: str = "08:30"
    ny_rth_end: str = "15:00"
    # Optional: keep your responsive killzones in NY
    use_responsive_killzones: bool = True
    kz1_start: str = "08:30"
    kz1_end: str = "10:30"
    kz2_start: str = "13:00"
    kz2_end: str = "14:30"

@dataclass
class FastPathConfig:
    """Fast order placement on ARMED zones to capture before invalidation"""
    enable_tokyo: bool = False
    enable_london: bool = True
    enable_ny: bool = True

    # When to activate the fast path
    max_zone_ticks_for_edge: int = 2       # micro-zones (<= 2 ticks) → edge front-run
    min_body_frac_for_fast: float = 0.65   # impulse must be real
    min_tape_range_pts: float = 2.00       # current 1m bar range ≥ this
    max_spread_ticks: int = 1              # don't fast-fire in wide spreads

    # Order placement rules
    front_run_ticks: int = 1               # edge ±1 tick for micro-zones
    prefer_mid_vs_62: str = "62"           # "50" or "62" for non-micro fast orders
    use_mit_on_touch: bool = True          # Market-If-Touched at chosen price
    ttl_seconds: int = 30                  # cancel if not filled quickly
    protect_max_slip_ticks: int = 2        # safety cap for MIT/marketability

    # Defense coordination
    arm_defense_cap: float = 0.94          # allow a bit deeper fill while fast path active (NY/London)
    revert_defense_cap: float = 0.90       # revert after order or timeout

@dataclass
class PatternToggles:
    enable_core_fvg: bool = True
    enable_ob_fvg: bool = True
    enable_irl_erl_fvg: bool = True
    enable_breaker_fvg: bool = True

@dataclass
class ICTContextConfig:
    enabled: bool = True
    htf_timeframes: Tuple[str, ...] = ("1h","4h","1d")
    raid_lookback_bars: int = 20
    ote_range: Tuple[float,float] = (0.62,0.79)
    use_smt: bool = False
    smt_symbol: str = "ES"
    refresh_seconds: int = 30
    track_pdh_pdl: bool = True
    track_week_high_low: bool = True
    track_midnight_open: bool = True
    track_830_open: bool = True

@dataclass
class ICTSessionConfig:
    score_gate_enabled: bool = False  # DISABLED to allow all ICT patterns through
    min_score_gate: float = 0.25  # Significantly relaxed for more opportunities
    weights_bias_loc_raid_sess_smt: Tuple[float,float,float,float,float] = (0.25,0.25,0.15,0.20,0.15)  # Increased bias/location weight
    gate_sessions: Tuple[str, ...] = ()  # Empty = all sessions allowed (24/7 ICT)

@dataclass
class ICTModuleToggles:
    enable_liquidity_ob: bool = True
    enable_silver_bullet: bool = True
    enable_breaker_unicorn: bool = True
    enable_fvg_continuation: bool = True
    enable_micro_scalp: bool = True

@dataclass
class ICTModuleParams:
    # generic
    sweep_min_ticks: int = 1  # Already optimal
    mss_lookback: int = 3          # Reduced from 5 for faster detection
    eqh_eql_window: int = 10       # Reduced from 15 for more responsive detection

    # silver bullet windows (America/Chicago)
    sb_london_start: str = "02:00"
    sb_london_end: str = "04:00"
    sb_ny_morn_start: str = "09:30"
    sb_ny_morn_end: str = "11:00"
    sb_ny_pm_start: str = "13:00"
    sb_ny_pm_end: str = "15:00"

    # continuation
    cont_htf: str = "15m"          # HTF for trend/FVG
    cont_min_impulse_pts: float = 4.0

    # micro scalp
    micro_max_zone_ticks: int = 3  # Increased from 2 for more opportunities
    micro_fastpath_ttl_s: int = 15  # Reduced from 30 for faster execution
    micro_max_trades_session: int = 6  # Increased from 4 for more trades

    # Session-specific optimization
    london_quality_boost: float = 0.15    # Extra quality for London session
    ny_morning_quality_boost: float = 0.12  # Extra quality for NY morning
    ny_afternoon_quality_boost: float = 0.08  # Extra quality for NY afternoon

    # Bias detection parameters
    bias_lookback_bars: int = 20    # Bars for bias calculation
    bias_slope_threshold: float = 2.0  # Points threshold for bias direction

    # Session-specific TTL adjustments
    london_ttl_multiplier: float = 1.5    # Longer TTL during London
    ny_active_ttl_multiplier: float = 0.7  # Shorter TTL during NY active hours
    off_hours_ttl_multiplier: float = 2.0  # Longer TTL during off hours

    # Quality thresholds by session
    london_min_quality: float = 0.40    # Lower threshold for high-quality London
    ny_active_min_quality: float = 0.45  # Slightly higher for NY active
    off_hours_min_quality: float = 0.55  # Higher threshold for off hours

    # Zone size optimization by session
    london_max_zone_multiplier: float = 1.2  # Allow larger zones in London
    ny_active_max_zone_multiplier: float = 1.0  # Standard zones in NY active
    off_hours_max_zone_multiplier: float = 0.8  # Smaller zones off hours

@dataclass
class FVGConfig:
    # Profile selection
    profile_active: str = "responsive"   # EMERGENCY: Switched to aggressive responsive profile

    # Profile configurations
    normal: FVGProfile = None
    responsive: FVGProfile = None

    # Session-aware profiles
    schedule: SessionSchedule = None
    profile_tokyo: SessionProfile = None
    profile_london: SessionProfile = None
    profile_ny: SessionProfile = None

    # Pattern toggles
    patterns: PatternToggles = None

    # Fast-arm + micro-FVG features (responsive only)
    responsive_arming: ResponsiveArming = None

    # Fast path configuration for armed zones
    fast_path: FastPathConfig = None

    # ICT Context integration
    ict_context: ICTContextConfig = ICTContextConfig()
    ict_session: ICTSessionConfig = ICTSessionConfig()
    ict_modules: ICTModuleToggles = ICTModuleToggles()
    ict_params: ICTModuleParams = ICTModuleParams()

    # Analytics and guards
    analytics: 'AnalyticsConfig' = None
    ict_guards: 'ICTGuards' = None

    # Legacy FVG settings (preserved for backward compatibility)
    sweep_min_overshoot_ticks: int = 1
    allow_trend_fvgs: bool = True
    min_gap_ticks: int = 1

    def __post_init__(self):
        if self.normal is None:
            self.normal = FVGProfile(
                name="normal",
                displacement_body_frac_min_base=0.50,  # Relaxed from 0.60 to allow ICT
                displacement_body_frac_min_high_vol=0.45,  # Relaxed from 0.52
                displacement_atr_multiple=0.50,  # Relaxed from 0.60
                displacement_min_points_floor=2.0,  # Relaxed from 3.0
                volume_min_mult_trend=1.00,  # Relaxed from 1.20
                volume_min_mult_sweep=1.10,  # Relaxed from 1.20
                quality_score_min_trend=0.45,  # Relaxed from 0.55
                defense_max_fill_pct=0.90,
            )

        if self.responsive is None:
            self.responsive = FVGProfile(
                name="responsive",
                displacement_body_frac_min_base=0.15,  # EMERGENCY: Relaxed from 0.25 to 0.15
                displacement_body_frac_min_high_vol=0.15,  # EMERGENCY: Relaxed from 0.25 to 0.15
                displacement_atr_multiple=0.35,  # EMERGENCY: Relaxed from 0.45 to 0.35
                displacement_min_points_floor=1.00,  # EMERGENCY: Relaxed from 1.50 to 1.00
                volume_min_mult_trend=0.40,  # EMERGENCY: Relaxed from 0.55 to 0.40
                volume_min_mult_sweep=0.80,  # EMERGENCY: Relaxed from 1.15 to 0.80
                quality_score_min_trend=0.15,  # EMERGENCY: Relaxed from 0.25 to 0.15
                defense_max_fill_pct=0.95,  # EMERGENCY: Increased from 0.92 to 0.95
            )

        if self.patterns is None:
            self.patterns = PatternToggles()

        if self.responsive_arming is None:
            self.responsive_arming = ResponsiveArming()

        if self.schedule is None:
            self.schedule = SessionSchedule()

        if self.profile_tokyo is None:
            self.profile_tokyo = SessionProfile(
                name="INTL_TOKYO",
                displacement_body_frac_min_base=0.35,
                displacement_body_frac_min_high_vol=0.30,
                displacement_min_points_floor=1.00,
                displacement_atr_multiple=0.35,
                volume_min_mult_trend=0.60,
                volume_min_mult_sweep=0.80,
                quality_score_min_trend=0.25,
                defense_max_fill_pct=0.93,
                rsi_relax_points=5,
            )

        if self.profile_london is None:
            self.profile_london = SessionProfile(
                name="INTL_LONDON",
                displacement_body_frac_min_base=0.25,
                displacement_body_frac_min_high_vol=0.25,
                displacement_min_points_floor=1.25,
                displacement_atr_multiple=0.40,
                volume_min_mult_trend=0.55,
                volume_min_mult_sweep=0.90,
                quality_score_min_trend=0.40,
                defense_max_fill_pct=0.93,
                rsi_relax_points=3,
            )

        if self.profile_ny is None:
            self.profile_ny = SessionProfile(
                name="NY",
                displacement_body_frac_min_base=0.30,  # Relaxed from 0.55
                displacement_body_frac_min_high_vol=0.25,  # Relaxed from 0.50
                displacement_min_points_floor=1.25,  # Lowered from 1.50
                displacement_atr_multiple=0.40,  # Lowered from 0.45
                volume_min_mult_trend=0.60,  # Relaxed from 1.00
                volume_min_mult_sweep=0.80,  # Relaxed from 1.15
                quality_score_min_trend=0.25,  # Relaxed from 0.45
                defense_max_fill_pct=0.92,
                rsi_relax_points=0,
            )

        if self.fast_path is None:
            self.fast_path = FastPathConfig()

# Strategy mode selection
STRATEGY_MODE = "FULL"   # valid: "FVG_ONLY" | "FULL" - Using FULL mode for FVG + ICT patterns

# FVG Strategy Configuration - Modern object-oriented config
FVG_CFG = FVGConfig(
    profile_active="responsive",  # Switch to responsive for enhanced opportunities
    sweep_min_overshoot_ticks=1,
    allow_trend_fvgs=True,
    min_gap_ticks=1
)

# Legacy FVG configuration (for backward compatibility)
FVG = {
    # Sweep / Trend FVG
    "sweep_min_overshoot_ticks": 1,               # Reduced from 2 ticks
    "allow_trend_fvgs": True,                     # Allow FVGs without sweep
    "min_gap_ticks": 1,                           # Lowered from 2 ticks

    "detection": {
        # Dynamic displacement mode
        "min_displacement_mode": "dynamic",        # "dynamic" | "fixed"
        "min_displacement_pts": 4.0,               # used only if mode="fixed"
        "min_displacement_dyn": {"base_pts": 2.0, "atr_mult": 0.4, "mode": "dynamic"},  # max(base, atr_mult*ATR) - RELAXED

        "min_body_frac": 0.40,                    # body >= 40% of range (normal vol) - RELAXED
        "min_body_frac_high_vol": 0.35,           # body >= 35% of range (high vol) - RELAXED
        "min_atr_mult": 0.6,                      # range >= 0.6 * ATR(14,1m) - RELAXED
        "min_vol_mult": 1.0,                      # volume >= 1.0x 20-bar avg - RELAXED
    },

    # High volatility detection
    "high_vol": {
        "atr_fast": 14,
        "atr_slow": 50,
        "vol_fast": 20,
        "vol_slow": 60,
        "atr_ratio": 1.30,                        # ATR_fast/ATR_slow >= 1.30
        "vol_ratio": 1.25,                        # Vol_fast/Vol_slow >= 1.25
    },

    "quality": {
        # Gate trades by quality score
        "min_quality": 0.55                        # NEW (may bump to 0.60 later)
    },

    "entry": {
        "use_mid_entry": True,                    # primary entry at mid-gap
        "entry_pct_default": 0.50,                # 50% midline (normal vol)
        "entry_pct_high_vol": 0.62,               # 62% midline (high vol)
        "use_edge_retry": True,                   # optional second limit at near edge
        "edge_offset_ticks": 2,                   # NEW
        "ttl_sec": 90,                            # NEW: give mid-entry 90s to fill
        "cancel_if_runs_ticks": 8                 # NEW: cancel if runs away
    },

    "edge_retry": {
        "enable": True,                            # NEW
        "ttl_sec": 45,                            # NEW: shorter TTL for edge retry
    },

    "risk": {
        "stop_pts": 7.5,                          # hard cap
        "max_stop_pts": 12.0,                     # max stop loss width (Option A fix)
        "tp_pts": 17.5,
        "breakeven_pts": 9.0,                     # CHANGED from 10.0
    },

    "trail": {
        "fast": {
            "trigger_pts": 12.0,                  # CHANGED (was 10-ish)
            "giveback_ticks": 10                  # NEW (10–12 ticks acceptable)
        }
    },

    "lifecycle": {
        "invalidate_frac": 0.90,                  # CHANGED: invalid if 90% consumed (was 75%)
        "touch_defend_inner_frac": 0.10,          # CHANGED: defend outer 90% (was 25%)
        "arm_timeout_sec": 600,
        "cooldown_secs": 60,
        "one_and_done_per_fvg": True,
        "lookback_bars": 300
    },

    # RSI filters
    "rsi": {
        "long_range": [50, 80],                   # Normal RSI range for longs
        "short_range": [20, 50],                  # Normal RSI range for shorts
        "long_range_rth": [45, 85],               # Relaxed RSI for RTH open (±5)
        "short_range_rth": [15, 55],              # Relaxed RSI for RTH open (±5)
        "rth_open_relax_minutes": 45,             # First 45 min of RTH
        "exchange_tz": "America/Chicago",         # CME timezone
    },

    # Risk guardrails
    "burst_guard_seconds": 120,                   # Min seconds between entries per direction
    "daily_trade_cap": 12,                        # Max trades per day
    "max_consecutive_losses": 3,                  # Stop after N consecutive losses
    "daily_loss_limit": 1000,                     # Daily loss limit in dollars

    # Legacy compatibility (will remove in future)
    "max_concurrent": 1
}

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
        'enabled': False,  # TEMPORARILY DISABLED - Testing other patterns without MT bleeding
        'max_daily_trades': 5,  # Reduced from 10 to prevent overtrading
        'risk_per_trade': 0.001,
        'min_confidence': 0.75,  # Increased from 0.60 for higher quality signals
        'max_spread_ticks': 1,
        'max_data_staleness_seconds': 2,
        'max_consecutive_losses': 3,

        # Cooldown and circuit breaker settings
        'min_minutes_between_trades': 15,    # Prevent rapid-fire trades
        'cooldown_after_loss_minutes': 30,   # Extra cooldown after loss
        'daily_loss_limit': 5,               # Max losses per day

        # Market regime filters
        'min_adx_for_trade': 25,             # Require trending market (was 20)
        'max_atr_multiplier': 2.0,           # Skip extreme volatility
        'min_volume_ratio': 0.8,             # Skip low volume periods

        # Trend alignment settings
        'require_trend_alignment': True,      # Only trade WITH the trend
        'trend_sma_period': 50,              # Use 50-period SMA for trend direction
        'trend_buffer_ticks': 2,             # Must be 2+ ticks away from SMA

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
CONTRACT_ID = "CON.F.US.ENQ.Z25"  # December 2025 NQ contract (active front month)

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
CONTRACT_ID = "CON.F.US.ENQ.Z25"   # December 2025 NQ contract (active front month)

# Discovery Mode (24/7 practice trading)
DISCOVERY_MODE = True
ALLOW_24H_TRADING = True
DISABLE_REGIME_GATING = True
DISABLE_TIME_BLOCKS = True
DISABLE_RISK_THROTTLES = True

# --- Momentum Thrust discovery defaults (realistic intraday levels) ---
MT_MOMENTUM_THRESHOLD = 0.002   # Increased from 0.0005 - requires stronger momentum
MT_VOLUME_FACTOR_MIN  = 1.2      # was 1.72
MT_MIN_STRENGTH       = 20       # was 40

# RSI "healthy" zones (more permissive but still sane)
MT_RSI_LONG_MIN  = 45
MT_RSI_LONG_MAX  = 72
MT_RSI_SHORT_MIN = 28
MT_RSI_SHORT_MAX = 55

# --- Dual confirmation for MT ---
MT_CONFIRM_MODE = "thrust"       # Changed from "dual" - now requires stronger thrust confirmation
MT_BREAKOUT_LOOKBACK = 20         # N-bar high/low for breakout

# Thrust confirm thresholds (tuned for typical NQ tape)
MT_THRUST_BODY_MIN        = 0.70  # Increased from 0.60 - requires stronger candles
MT_THRUST_RANGE_ATR_MIN   = 1.20  # Increased from 0.80 - requires more significant moves
MT_THRUST_MOM_MIN         = 0.001   # Increased from 0.0003 - stronger momentum required
MT_THRUST_VOL_MIN         = 1.30  # Increased from 1.15 - needs higher volume confirmation

# Confidence control
PATTERN_MIN_CONFIDENCE = 0.60    # production default
GLOBAL_MIN_CONFIDENCE  = 0.60    # Increased from 0.35 to filter weak signals

# In discovery, use GLOBAL_MIN_CONFIDENCE for all patterns
USE_GLOBAL_MIN_CONF_WHEN_DISCOVERY = True

# Retest / execution sanity
CANCEL_IF_RUNS_BYPASS_DISCOVERY = True
RETEST_LIMIT_OFFSET_TICKS = 2

# Keep essential protections
MAX_SLIPPAGE_TICKS = 2
OUTER_FAILSAFE_TICKS = 20        # $100 outer guard
POSITION_MAX_CONTRACTS = 1       # cap size while discovering (1 trade at a time)
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

# Force immediate market execution (no pending STOP/LIMIT)
# For FVG strategy, we use limit orders at specific levels
FORCE_IMMEDIATE_MARKET = False if STRATEGY_MODE == "FVG_ONLY" else True
DISABLE_PENDING_ENTRIES = False if STRATEGY_MODE == "FVG_ONLY" else True   # guard to bypass any legacy stop/limit flows

# TopStepX Auto-Bracket Configuration (FVG-ONLY mode)
TOPSTEPX_AUTO_BRACKET = {
    "enable": True,  # Enable for ALL strategies - align with TopStepX $150 TP / $125 SL
    "tp_pts": 30.0,  # $150 target = 30 ticks ($5 per tick)
    "sl_pts": 25.0   # $125 stop = 25 ticks ($5 per tick)
}

# Protective stop (server-side), dollars → ticks via contract tickValue
STOP_GUARD = {
    "enable": False,  # DISABLED: TopStepX provides automatic SL $150 / TP $300 brackets
    "usd": 70.0,                       # $70 → 14 ticks on NQ (tickValue = $5)
    "respect_tighter_logic_stop": True # keep tighter strategy stop if closer
}

# Internal OCO Configuration
INTERNAL_OCO = {
    "enable": False  # DISABLED: TopStepX provides automatic bracket orders (SL $150 / TP $300)
}

# Candlestick guard
CANDLES = {
    "enable": True,
    "near_level_ticks": 6,        # key level proximity (VWAP/ONH/ONL/POC/TL)
    "danger_range_atr_mult": 1.25,
    "danger_body_frac": 0.60,
    "danger_vol_mult": 1.5,
    "soft_boost": [0.05, 0.10],   # add/subtract to confidence
    "scope_bars": 2               # effect window after detection (closed bars)
}

# Discovery scoring/exec - enable score-only immediate execution
EXECUTE_ON_SCORE_ONLY = True          # Enable immediate pass on score in discovery
MIN_PASS_SCORE_DISCOVERY = 0.65       # Increased from 0.35 for quality control

# Pre-trade safety guards for MARKET entries
PRETRADE_GUARDS = {
    "max_spread_ticks": 2,            # Abort if bid/ask spread > 2 ticks
    "max_age_ms": 3500                # Relaxed from 800ms to 3500ms for better fill rates
}

# Analytics Configuration
@dataclass
class AnalyticsConfig:
    """Configuration for ICT analytics and rollup reporting"""
    ict_rollups_enabled: bool = True
    rollup_period_seconds: int = 300  # 5 minutes
    ict_score_buckets: tuple = (0.4, 0.6, 0.8)  # Score bucket thresholds

# ICT Guards Configuration
@dataclass
class ICTGuards:
    """ICT per-session caps and performance kill-switch configuration"""
    micro_max_trades_per_session: int = 4
    silver_max_trades_per_window: int = 2
    tag_killswitch_window_trades: int = 20
    tag_disable_if_win_lt: float = 0.38
    tag_disable_if_avgR_lt: float = 0.70

# Initialize analytics and guards if not already set
if FVG_CFG.analytics is None:
    FVG_CFG.analytics = AnalyticsConfig()
if FVG_CFG.ict_guards is None:
    FVG_CFG.ict_guards = ICTGuards()

# Validate configuration on import
validate_pattern_config()