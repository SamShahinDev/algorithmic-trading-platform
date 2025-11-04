"""
FVG-Only Trading Bot Runner
Runs the bot in Fair Value Gap only mode without other patterns
"""

import asyncio
import logging
import sys
import os
import time
import csv
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add nq_bot directory

from pattern_config import STRATEGY_MODE, FVG, FVG_CFG
from patterns.fvg_strategy import FVGStrategy, FVGObject
from ict import ICTContext
try:
    from utils.data_cache import DataCache
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nq_bot.utils.data_cache import DataCache
from unittest.mock import Mock
try:
    from utils.execution_manager import ExecutionManager
except ImportError:
    try:
        from nq_bot.utils.execution_manager import ExecutionManager
    except ImportError:
        # Allow dry run without full dependencies
        ExecutionManager = None
from web_platform.backend.brokers.topstepx_client import TopStepXClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Track trade performance metrics"""
    fills: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0
    total_mae: float = 0
    total_mfe: float = 0
    total_duration: float = 0
    last_rollup: float = 0


class FVGRunner:
    """Main runner for FVG-only strategy"""
    
    def __init__(self):
        """Initialize FVG runner"""
        
        # Verify strategy mode
        if STRATEGY_MODE != "FVG_ONLY":
            raise ValueError(f"Wrong strategy mode: {STRATEGY_MODE}. Expected FVG_ONLY")
        
        # Check for dry-run mode
        self.dry_run = os.getenv('FVG_DRY_RUN', 'false').lower() == 'true'
        
        # Print banner
        print("\n" + "="*60)
        if self.dry_run:
            print("🟡 FVG-ONLY DRY RUN MODE (NO REAL ORDERS)")
        else:
            print("🟢 FVG-ONLY PRACTICE MODE")
        print("="*60)
        print(f"Time: {datetime.now()}")
        print("="*60 + "\n")
        
        mode_str = "DRY RUN" if self.dry_run else "PRACTICE"
        logger.info(f"🟢 FVG-ONLY {mode_str} MODE STARTED")
        
        # Load environment from parent directory
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env.topstepx')
        load_dotenv(env_path)
        logger.info(f"Loading environment from: {env_path}")
        
        # Initialize components
        self.broker = TopStepXClient()
        self.data_cache = None
        self.exec_manager = None
        self.fvg_strategy = None
        self.ict = None  # ICT context service
        
        # Trade tracking
        self.current_position = None
        self.current_fvg_id = None
        self.last_trade_time = 0
        self.last_trade_time_by_direction = {'long': 0, 'short': 0}  # Burst guard
        self.daily_trades = 0
        self.metrics = TradeMetrics(last_rollup=time.time())

        # Telemetry tracking
        self.telemetry = {
            'bars_seen': 0,
            'sweep_fvg_detected': 0,
            'trend_fvg_detected': 0,
            'fresh': 0,
            'armed': 0,
            'orders_placed': 0,
            'fills': 0,
            'entries_50pct': 0,
            'entries_62pct_highvol': 0,
            'blocked': {
                'displacement_body': 0,
                'gap_min': 0,
                'defense_overfill': 0,
                'rsi_range': 0,
                'cooldown': 0,
                'burst_guard': 0,
                'daily_trade_cap': 0,
                'risk_limits': 0
            },
            'high_vol_checks': 0,
            'high_vol_true': 0,
            'rth_open_relax_hits': 0
        }
        self.last_telemetry_time = time.time()

        # Session tracking
        self._current_session = "OTHER"
        self._effective_profile = None
        self.logger = logger

        # CSV telemetry
        self.csv_file = 'logs/fvg_telemetry.csv'
        self._init_csv()
        
    def _init_csv(self):
        """Initialize CSV telemetry file"""
        try:
            # Create logs directory if needed
            os.makedirs('logs', exist_ok=True)
            
            # Write header if file doesn't exist
            if not os.path.exists(self.csv_file):
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'event', 'id', 'direction', 'entry', 'stop', 'tp',
                        'mae', 'mfe', 'pnl', 'duration', 'details'
                    ])
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}")
    
    def _write_csv(self, event: str, **kwargs):
        """Write event to CSV telemetry"""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    event,
                    kwargs.get('id', ''),
                    kwargs.get('direction', ''),
                    kwargs.get('entry', ''),
                    kwargs.get('stop', ''),
                    kwargs.get('tp', ''),
                    kwargs.get('mae', ''),
                    kwargs.get('mfe', ''),
                    kwargs.get('pnl', ''),
                    kwargs.get('duration', ''),
                    kwargs.get('details', '')
                ])
        except Exception as e:
            logger.error(f"CSV write error: {e}")

    def _within(self, hhmm_now, start_hhmm, end_hhmm):
        """Check if time is within session window, supports crossing midnight"""
        try:
            sh, sm = map(int, start_hhmm.split(":")); eh, em = map(int, end_hhmm.split(":"))
            t = hhmm_now
            a = (sh, sm); b = (eh, em)
            return (a <= t < b) if a < b else (t >= a or t < b)
        except Exception as e:
            self.logger.error(f"_within error: {e}, start_hhmm={start_hhmm}, end_hhmm={end_hhmm}")
            return False

    def _active_session(self, now_dt):
        """Determine active trading session from current time"""
        try:
            ct = now_dt.astimezone(ZoneInfo(FVG_CFG.schedule.exchange_tz))
            t = (ct.hour, ct.minute)

            self.logger.debug(f"Session check: CT time={ct}, tuple={t}")
            self.logger.debug(f"Tokyo: {FVG_CFG.schedule.tokyo_start} - {FVG_CFG.schedule.tokyo_end}")

            if self._within(t, FVG_CFG.schedule.tokyo_start, FVG_CFG.schedule.tokyo_end):
                return "TOKYO"
            if self._within(t, FVG_CFG.schedule.london_start, FVG_CFG.schedule.london_end):
                return "LONDON"
            if self._within(t, FVG_CFG.schedule.ny_rth_start, FVG_CFG.schedule.ny_rth_end):
                return "NY_RTH"
            return "OTHER"
        except Exception as e:
            self.logger.error(f"_active_session error: {e}")
            return "OTHER"

    def _choose_session_profile(self, session, now_dt):
        """Choose appropriate session profile and store for strategy use"""
        # Base: keep your normal/responsive toggle logic as-is
        # If you already auto-switch responsive in NY killzones, leave it unchanged.
        if session == "TOKYO":
            prof = FVG_CFG.profile_tokyo
        elif session == "LONDON":
            prof = FVG_CFG.profile_london
        else:
            prof = FVG_CFG.profile_ny  # NY_RTH + OTHER default to NY

        # Store for strategy use; detectors read only this resolved profile
        # Note: session already stored in self._current_session in main loop
        self._effective_profile = prof
        self.logger.info(f"SESSION: {{\"name\": \"{session}\", \"profile\": \"{prof.name}\"}}")

    def _echo_config(self):
        """Echo active FVG configuration on startup with CONFIG_ECHO tag for triage"""
        logger.info("="*60)
        logger.info("CONFIG_ECHO: FVG CONFIGURATION ACTIVE")

        # Enhanced Profile Configuration
        logger.info(f"CONFIG_ECHO: profile_active: {FVG_CFG.profile_active}")

        # Session information
        if self._effective_profile:
            logger.info(f"CONFIG_ECHO: active_session: {self._active_session}")
            logger.info(f"CONFIG_ECHO: session_profile: {self._effective_profile.name}")
            prof = self._effective_profile
        else:
            # Fallback to legacy profile selection
            prof = FVG_CFG.responsive if FVG_CFG.profile_active == "responsive" else FVG_CFG.normal
        logger.info(f"CONFIG_ECHO: defense_max_fill_pct: {prof.defense_max_fill_pct}")
        logger.info(f"CONFIG_ECHO: volume_min_mult_trend: {prof.volume_min_mult_trend}")
        logger.info(f"CONFIG_ECHO: quality_score_min_trend: {prof.quality_score_min_trend}")
        logger.info(f"CONFIG_ECHO: displacement_body_frac_min_base: {prof.displacement_body_frac_min_base}")
        logger.info(f"CONFIG_ECHO: displacement_body_frac_min_high_vol: {prof.displacement_body_frac_min_high_vol}")
        logger.info(f"CONFIG_ECHO: displacement_atr_multiple: {prof.displacement_atr_multiple}")
        logger.info(f"CONFIG_ECHO: displacement_min_points_floor: {prof.displacement_min_points_floor}")

        # Pattern toggles
        patterns = FVG_CFG.patterns
        logger.info(f"CONFIG_ECHO: patterns: {{'enable_core_fvg': {patterns.enable_core_fvg}, 'enable_ob_fvg': {patterns.enable_ob_fvg}, 'enable_irl_erl_fvg': {patterns.enable_irl_erl_fvg}, 'enable_breaker_fvg': {patterns.enable_breaker_fvg}}}")

        # Legacy configuration (backward compatibility)
        logger.info(f"CONFIG_ECHO: allow_trend_fvgs: {FVG.get('allow_trend_fvgs')}")
        logger.info(f"CONFIG_ECHO: sweep_min_overshoot_ticks: {FVG.get('sweep_min_overshoot_ticks')}")
        logger.info(f"CONFIG_ECHO: min_gap_ticks: {FVG.get('min_gap_ticks')}")

        detection = FVG.get('detection', {})
        logger.info(f"CONFIG_ECHO: min_displacement_mode: {detection.get('min_displacement_mode')}")
        logger.info(f"CONFIG_ECHO: min_displacement_pts: {detection.get('min_displacement_pts')}")
        logger.info(f"CONFIG_ECHO: min_displacement_dyn: {detection.get('min_displacement_dyn')}")

        high_vol = FVG.get('high_vol', {})
        logger.info(f"CONFIG_ECHO: high_vol_atr_ratio: {high_vol.get('atr_ratio')}")
        logger.info(f"CONFIG_ECHO: high_vol_vol_ratio: {high_vol.get('vol_ratio')}")

        lifecycle = FVG.get('lifecycle', {})
        logger.info(f"CONFIG_ECHO: invalidate_frac: {lifecycle.get('invalidate_frac')}")
        logger.info(f"CONFIG_ECHO: touch_defend_inner_frac: {lifecycle.get('touch_defend_inner_frac')}")
        logger.info(f"CONFIG_ECHO: arm_timeout_sec: {lifecycle.get('arm_timeout_sec')}")

        entry = FVG.get('entry', {})
        logger.info(f"CONFIG_ECHO: entry_pct_default: {entry.get('entry_pct_default')}")
        logger.info(f"CONFIG_ECHO: entry_pct_high_vol: {entry.get('entry_pct_high_vol')}")
        logger.info(f"CONFIG_ECHO: ttl_sec: {entry.get('ttl_sec')}")

        risk = FVG.get('risk', {})
        logger.info(f"CONFIG_ECHO: stop_pts: {risk.get('stop_pts')}")
        logger.info(f"CONFIG_ECHO: tp_pts: {risk.get('tp_pts')}")
        logger.info(f"CONFIG_ECHO: breakeven_pts: {risk.get('breakeven_pts')}")

        rsi = FVG.get('rsi', {})
        logger.info(f"CONFIG_ECHO: rsi_long_range: {rsi.get('long_range')}")
        logger.info(f"CONFIG_ECHO: rsi_short_range: {rsi.get('short_range')}")
        logger.info(f"CONFIG_ECHO: exchange_tz: {rsi.get('exchange_tz')}")

        # Guardrails
        logger.info(f"CONFIG_ECHO: burst_guard_seconds: {FVG.get('burst_guard_seconds', 120)}")
        logger.info(f"CONFIG_ECHO: daily_trade_cap: {FVG.get('daily_trade_cap', 12)}")
        logger.info(f"CONFIG_ECHO: max_consecutive_losses: {FVG.get('max_consecutive_losses', 3)}")
        logger.info(f"CONFIG_ECHO: daily_loss_limit: {FVG.get('daily_loss_limit', 1000)}")

        # ICT Guards
        ict_guards = FVG_CFG.ict_guards
        logger.info(f"CONFIG_ECHO: ict_micro_max_trades_per_session: {ict_guards.micro_max_trades_per_session}")
        logger.info(f"CONFIG_ECHO: ict_silver_max_trades_per_window: {ict_guards.silver_max_trades_per_window}")
        logger.info(f"CONFIG_ECHO: ict_tag_killswitch_window_trades: {ict_guards.tag_killswitch_window_trades}")
        logger.info(f"CONFIG_ECHO: ict_tag_disable_if_win_lt: {ict_guards.tag_disable_if_win_lt}")
        logger.info(f"CONFIG_ECHO: ict_tag_disable_if_avgR_lt: {ict_guards.tag_disable_if_avgR_lt}")

        # Data feed configuration
        from pattern_config import CONTRACT_ID, LIVE_MARKET_DATA
        logger.info(f"CONFIG_ECHO: contract_id: {CONTRACT_ID}")
        logger.info(f"CONFIG_ECHO: live_market_data: {LIVE_MARKET_DATA}")
        logger.info(f"CONFIG_ECHO: tick_size: 0.25")
        logger.info("="*60)

    def _log_telemetry(self):
        """Log telemetry snapshot as JSON"""
        current_time = time.time()

        # Calculate rates
        high_vol_rate = 0
        if self.telemetry['high_vol_checks'] > 0:
            high_vol_rate = self.telemetry['high_vol_true'] / self.telemetry['high_vol_checks']

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'session': self._active_session,
            'session_profile': self._effective_profile.name if self._effective_profile else 'none',
            'bars_seen': self.telemetry['bars_seen'],
            'sweep_fvg_detected': self.telemetry['sweep_fvg_detected'],
            'trend_fvg_detected': self.telemetry['trend_fvg_detected'],
            'fresh': self.telemetry['fresh'],
            'armed': self.telemetry['armed'],
            'orders_placed': self.telemetry['orders_placed'],
            'fills': self.telemetry['fills'],
            'entries_50pct': self.telemetry['entries_50pct'],
            'entries_62pct_highvol': self.telemetry['entries_62pct_highvol'],
            'blocked': self.telemetry['blocked'],
            'high_vol_true_rate': f"{high_vol_rate:.2%}",
            'rth_open_relax_hits': self.telemetry['rth_open_relax_hits']
        }

        logger.info(f"TELEMETRY: {json.dumps(snapshot, separators=(',', ':'))}")
        self.last_telemetry_time = current_time
    
    async def initialize(self):
        """Initialize async components"""
        try:
            # Connect to broker (skip in dry run)
            if not self.dry_run:
                await self.broker.connect()
                logger.info("✅ Connected to TopStepX")
            else:
                logger.info("✅ Dry run mode - skipping broker connection")
            
            # Initialize data cache with correct parameters
            contract_id = 'CON.F.US.ENQ.Z25'  # December 2025 NQ contract (active front month)
            # Use practice feed for practice accounts (live=False)
            self.data_cache = DataCache(self.broker, contract_id, is_live=False, logger=logger)
            await self.data_cache.initialize()
            logger.info("✅ DataCache initialized")
            
            # Initialize execution manager (mock in dry run)
            if not self.dry_run:
                account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
                self.exec_manager = ExecutionManager(self.broker, self.data_cache, account_id)
                logger.info("✅ ExecutionManager initialized")
            else:
                # Create mock execution manager for dry run
                self.exec_manager = Mock()
                logger.info("✅ Mock ExecutionManager for dry run")
            
            # Initialize ICT context service
            self.ict = ICTContext(self.data_cache, FVG_CFG.ict_context, logger)
            logger.info("✅ ICT Context initialized")

            # Initialize FVG strategy with enhanced config support
            # Inject new config into legacy config for compatibility
            enhanced_config = FVG.copy()
            enhanced_config['cfg'] = FVG_CFG
            self.fvg_strategy = FVGStrategy(self.data_cache, logger, enhanced_config)
            self.fvg_strategy.runner = self  # Give strategy access to runner for session profiles
            self.fvg_strategy.execution = self.exec_manager  # Inject execution manager for fast path orders
            self.fvg_strategy.ict = self.ict  # Attach ICT context to strategy
            logger.info("✅ FVG Strategy initialized")

            # Echo configuration on startup
            self._echo_config()
            
            # Start data cache updates with much longer interval to avoid rate limits
            asyncio.create_task(self.data_cache.start_auto_update(interval_seconds=30))
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def check_data_freshness(self, max_age_ms: int = 3500) -> bool:
        """Check if market data is fresh enough"""
        try:
            latest_bar = self.data_cache.get_bars('1m', 1)
            if latest_bar is None or latest_bar.empty:
                return False
            
            bar_time = latest_bar.index[-1]
            # Handle timezone-aware bar_time vs timezone-naive datetime.now()
            if bar_time.tz is not None:
                # bar_time is timezone-aware, make current time timezone-aware too
                from datetime import timezone
                current_time = datetime.now(timezone.utc)
                if bar_time.tz != timezone.utc:
                    # Convert bar_time to UTC for comparison
                    bar_time = bar_time.astimezone(timezone.utc)
            else:
                # bar_time is timezone-naive, use naive current time
                current_time = datetime.now()

            age_ms = (current_time - bar_time).total_seconds() * 1000
            
            if age_ms > max_age_ms:
                logger.warning(f"Data too stale: {age_ms:.0f}ms > {max_age_ms}ms")
                return False
            
            logger.debug(f"Data freshness OK: {age_ms:.0f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Data freshness check error: {e}")
            return False
    
    def check_position_cap(self, max_open: int = 1) -> bool:
        """Check if position limit reached"""
        return self.current_position is None
    
    def _is_rth_open_window(self) -> bool:
        """Check if we're in the RTH open window for relaxed RSI"""
        try:
            from datetime import datetime
            import pytz

            rsi_cfg = FVG.get('rsi', {})
            exchange_tz = pytz.timezone(rsi_cfg.get('exchange_tz', 'America/Chicago'))
            now = datetime.now(exchange_tz)

            # Check if it's a trading day (Mon-Fri)
            if now.weekday() > 4:  # Saturday or Sunday
                return False

            # RTH open for NQ futures is 08:30 CT
            rth_open_hour = 8
            rth_open_minute = 30
            relax_minutes = rsi_cfg.get('rth_open_relax_minutes', 45)

            # Calculate minutes since RTH open
            minutes_since_midnight = now.hour * 60 + now.minute
            rth_open_minutes = rth_open_hour * 60 + rth_open_minute
            minutes_since_open = minutes_since_midnight - rth_open_minutes

            return 0 <= minutes_since_open <= relax_minutes

        except Exception as e:
            logger.debug(f"RTH open window check error: {e}")
            return False

    def check_rsi_veto(self, direction: str) -> bool:
        """Check RSI veto conditions with RTH open relaxation"""
        try:
            # Get RSI from data cache
            bars = self.data_cache.get_bars('1m', 14)
            if bars is None or len(bars) < 14:
                return True  # No veto if insufficient data

            # Calculate RSI
            close_prices = bars['close']
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Get RSI ranges based on RTH open window
            rsi_cfg = FVG.get('rsi', {})
            if self._is_rth_open_window():
                # Use relaxed ranges during RTH open
                long_range = list(rsi_cfg.get('long_range_rth', [45, 85]))
                short_range = list(rsi_cfg.get('short_range_rth', [15, 55]))
                logger.debug(f"Using RTH open RSI ranges: long {long_range}, short {short_range}")
            else:
                # Use normal ranges
                long_range = list(rsi_cfg.get('long_range', [50, 80]))
                short_range = list(rsi_cfg.get('short_range', [20, 50]))

            # Apply session-aware RSI relaxation
            if self._effective_profile and self._effective_profile.rsi_relax_points > 0:
                relax_points = self._effective_profile.rsi_relax_points
                long_range[0] = max(0, long_range[0] - relax_points)
                long_range[1] = min(100, long_range[1] + relax_points)
                short_range[0] = max(0, short_range[0] - relax_points)
                short_range[1] = min(100, short_range[1] + relax_points)
                logger.debug(f"Session RSI relaxation (+/-{relax_points}): long {long_range}, short {short_range}")

            # Apply veto based on direction and ranges
            if direction == 'long':
                if current_rsi < long_range[0] or current_rsi > long_range[1]:
                    logger.warning(f"RSI veto: Long trade blocked at RSI {current_rsi:.1f} (range {long_range})")
                    return False
            elif direction == 'short':
                if current_rsi < short_range[0] or current_rsi > short_range[1]:
                    logger.warning(f"RSI veto: Short trade blocked at RSI {current_rsi:.1f} (range {short_range})")
                    return False

            return True
            
        except Exception as e:
            logger.error(f"RSI veto check error: {e}")
            return True  # No veto on error
    
    def _check_pretrade_guards(self, direction: str = None) -> bool:
        """Check all pre-trade conditions"""
        try:
            # Check position cap
            if not self.check_position_cap(1):
                logger.debug("Position cap reached")
                return False
            
            # Check data freshness
            if not self.check_data_freshness(3500):
                return False
            
            # Check cooldown
            if time.time() - self.last_trade_time < FVG.get('lifecycle', {}).get('cooldown_secs', 60):
                logger.debug("Trade cooldown active")
                self.telemetry['blocked']['cooldown'] += 1
                return False
            
            # Check RSI veto if direction provided
            if direction and not self.check_rsi_veto(direction):
                self.telemetry['blocked']['rsi_range'] += 1
                return False

            # Check burst guard (min time between entries per direction)
            if direction:
                burst_guard_seconds = FVG.get('burst_guard_seconds', 120)
                time_since_last = time.time() - self.last_trade_time_by_direction.get(direction, 0)
                if time_since_last < burst_guard_seconds:
                    logger.debug(f"Burst guard: {direction} trade blocked, only {time_since_last:.0f}s since last (need {burst_guard_seconds}s)")
                    self.telemetry['blocked']['burst_guard'] += 1
                    return False

            # Check daily trade cap
            daily_cap = FVG.get('daily_trade_cap', 12)
            if self.daily_trades >= daily_cap:
                logger.warning(f"Daily trade cap reached: {self.daily_trades}/{daily_cap}")
                self.telemetry['blocked']['daily_trade_cap'] += 1
                return False

            # TODO: Add candlestick veto if enabled
            # if FVG.get('candle_veto_enabled', False):
            #     if not self.check_candle_veto(direction):
            #         return False

            return True
            
        except Exception as e:
            logger.error(f"Pre-trade guard error: {e}")
            return False
    
    def _build_signal(self, fvg: FVGObject) -> Optional[Dict]:
        """Build trading signal from FVG with dynamic entry levels"""
        try:
            tick_size = 0.25

            # Get entry signals from FVG strategy (includes high vol check)
            entry_signals = self.fvg_strategy.get_entry_signals(fvg)

            # Use the calculated entry level
            if 'mid_entry' in entry_signals:
                entry = entry_signals['mid_entry']['level']
            else:
                # Fallback to mid if no entry signal
                entry = fvg.mid
            
            # Calculate stop loss
            if fvg.direction == 'long':
                stop = fvg.bottom - (2 * tick_size)
            else:
                stop = fvg.top + (2 * tick_size)
            
            # Check max stop
            stop_pts = abs(entry - stop)
            if stop_pts > FVG.get('max_stop_pts', 7.5):
                logger.warning(f"Stop too wide: {stop_pts:.2f} pts > {FVG.get('max_stop_pts')} pts")
                return None
            
            # Calculate target
            tp_pts = FVG.get('tp_pts', 17.5)
            if fvg.direction == 'long':
                tp = entry + tp_pts
            else:
                tp = entry - tp_pts
            
            signal = {
                'action': 'BUY' if fvg.direction == 'long' else 'SELL',
                'entry_price': entry,
                'stop_loss': stop,
                'take_profit': tp,
                'confidence': fvg.quality,
                'pattern_name': 'fvg',
                'fvg_id': fvg.id
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal build error: {e}")
            return None
    
    async def execute_trade(self, signal: Dict, fvg: FVGObject):
        """Execute FVG trade with OCO bracket"""
        try:
            # Check if dry run
            if self.dry_run:
                # Log dry run entry
                logger.info(f"ENTRY_PLACED(DRY_RUN) id={fvg.id} dir={fvg.direction} "
                           f"entry={signal['entry_price']:.2f} stop={signal['stop_loss']:.2f} "
                           f"tp={signal['take_profit']:.2f}")
                
                self._write_csv('ENTRY_PLACED(DRY_RUN)',
                              id=fvg.id,
                              direction=fvg.direction,
                              entry=signal['entry_price'],
                              stop=signal['stop_loss'],
                              tp=signal['take_profit'])
                
                # Mark FVG as consumed in dry run
                self.fvg_strategy.mark_consumed(fvg.id)
                self.last_trade_time = time.time()
                return
            
            # Real execution
            account_id = int(os.getenv('TOPSTEPX_ACCOUNT_ID'))
            contract_id = 'CON.F.US.ENQ.Z25'  # December 2025 NQ contract (active front month)
            
            result = await self.exec_manager.place_limit_with_oco(
                account_id=account_id,
                contract_id=contract_id,
                side=signal['action'],
                qty=1,
                limit_price=signal['entry_price'],
                stop_loss_price=signal['stop_loss'],
                take_profit_price=signal['take_profit'],
                tag=f"FVG_{fvg.id}"
            )
            
            if not result:
                logger.error("Failed to place limit order with OCO")
                return
            
            order_id = result.get('order_id')
            
            # Log and telemetry
            logger.info(f"ENTRY_PLACED id={fvg.id} dir={fvg.direction} "
                       f"entry={signal['entry_price']:.2f} stop={signal['stop_loss']:.2f} "
                       f"tp={signal['take_profit']:.2f}")
            
            self._write_csv('ENTRY_PLACED',
                          id=fvg.id,
                          direction=fvg.direction,
                          entry=signal['entry_price'],
                          stop=signal['stop_loss'],
                          tp=signal['take_profit'])
            
            # Track order
            self.current_fvg_id = fvg.id
            
            # Wait for fill (with timeout)
            fill_price = await self._wait_for_fill(order_id, timeout=30)
            
            if fill_price:
                # Mark FVG as consumed
                self.fvg_strategy.mark_consumed(fvg.id)
                
                # Place OCO bracket
                await self._place_oco_bracket(signal, fill_price)
                
                # Update tracking
                self.current_position = {
                    'side': signal['action'],
                    'entry': fill_price,
                    'stop': signal['stop_loss'],
                    'tp': signal['take_profit'],
                    'entry_time': time.time()
                }
                
                self.last_trade_time = time.time()
                
                # Log fill
                logger.info(f"EXEC_FILL id={fvg.id} price={fill_price:.2f}")
                self._write_csv('EXEC_FILL', id=fvg.id, entry=fill_price)
                
                # Start position monitoring
                asyncio.create_task(self._monitor_position())
                
            else:
                # Cancel unfilled order
                await self.exec_manager.cancel_order(order_id)
                logger.info(f"CANCEL id={fvg.id} - order timeout")
                self._write_csv('CANCEL', id=fvg.id, details='timeout')
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def _wait_for_fill(self, order_id: str, timeout: int = 30) -> Optional[float]:
        """Wait for order fill with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check order status
            status = await self.exec_manager.get_order_status(order_id)
            
            if status and status.get('filled'):
                return status.get('fill_price')
            
            await asyncio.sleep(0.5)
        
        return None
    
    async def _place_oco_bracket(self, signal: Dict, fill_price: float):
        """Place OCO bracket orders"""
        try:
            # Place stop loss
            stop_side = 'SELL' if signal['action'] == 'BUY' else 'BUY'
            stop_id = await self.exec_manager.place_stop_order(
                side=stop_side,
                stop_price=signal['stop_loss'],
                quantity=1
            )
            
            # Place take profit
            tp_id = await self.exec_manager.place_limit_order(
                side=stop_side,
                price=signal['take_profit'],
                quantity=1
            )
            
            if stop_id and tp_id:
                logger.info(f"OCO bracket placed: stop={signal['stop_loss']:.2f} tp={signal['take_profit']:.2f}")
            
        except Exception as e:
            logger.error(f"OCO bracket error: {e}")
    
    async def _monitor_position(self):
        """Monitor open position for trailing and breakeven"""
        if not self.current_position:
            return
        
        try:
            entry = self.current_position['entry']
            side = self.current_position['side']
            entry_time = self.current_position['entry_time']
            
            be_moved = False
            trail_active = False
            
            while self.current_position:
                # Get current price
                bars = self.data_cache.get_bars('1m', 1)
                if bars is None or bars.empty:
                    await asyncio.sleep(1)
                    continue
                
                current_price = bars['close'].iloc[-1]
                
                # Calculate PnL in points
                if side == 'BUY':
                    pnl_pts = current_price - entry
                else:
                    pnl_pts = entry - current_price
                
                # Breakeven management
                if not be_moved and pnl_pts >= FVG.get('breakeven_pts', 10):
                    # Move stop to breakeven
                    await self._move_stop_to_breakeven()
                    be_moved = True
                    logger.info("MOVE_BE - Stop moved to breakeven")
                    self._write_csv('MOVE_BE', details=f"pnl={pnl_pts:.2f}")
                
                # Fast trail activation
                if (not trail_active and 
                    FVG.get('fast_trail_enable', True) and
                    pnl_pts >= 10 and 
                    time.time() - entry_time <= FVG.get('fast_trail_arming_secs', 180)):
                    
                    trail_active = True
                    logger.info(f"TRAIL_ON - Fast trail activated at +{pnl_pts:.2f} pts")
                    self._write_csv('TRAIL_ON', details=f"pnl={pnl_pts:.2f}")
                    
                    # Start trailing
                    asyncio.create_task(self._trail_stop(current_price))
                
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Position monitor error: {e}")
    
    async def _move_stop_to_breakeven(self):
        """Move stop to breakeven"""
        if not self.current_position:
            return
        
        try:
            # Use ExecutionManager's breakeven method
            success = await self.exec_manager.modify_stop_to_breakeven()
            if success:
                logger.info("MOVE_BE - Stop successfully moved to breakeven")
                self._write_csv('MOVE_BE', 
                              details=f"entry={self.current_position['entry']:.2f}")
            else:
                logger.error("Failed to move stop to breakeven")
        except Exception as e:
            logger.error(f"Breakeven move error: {e}")
    
    async def _trail_stop(self, start_price: float):
        """Trail stop loss behind swing extremes"""
        if not self.current_position:
            return
        
        try:
            trail_ticks = FVG.get('fast_trail_ticks', 8)
            trail_distance = trail_ticks * 0.25
            side = self.current_position['side']
            best_price = start_price
            
            while self.current_position:
                bars = self.data_cache.get_bars('1m', 5)  # Get last 5 bars for swing
                if bars is None or bars.empty:
                    await asyncio.sleep(1)
                    continue
                
                current_price = bars['close'].iloc[-1]
                
                # Track best price
                if side == 'BUY':
                    if current_price > best_price:
                        best_price = current_price
                        # Update trailing stop
                        success = await self.exec_manager.trail_stop(trail_distance)
                        if success:
                            logger.info(f"TRAIL_UPDATE at {current_price:.2f}, stop at {current_price - trail_distance:.2f}")
                            self._write_csv('TRAIL_UPDATE', 
                                          details=f"price={current_price:.2f},trail={trail_distance:.2f}")
                else:  # SELL
                    if current_price < best_price:
                        best_price = current_price
                        # Update trailing stop
                        success = await self.exec_manager.trail_stop(trail_distance)
                        if success:
                            logger.info(f"TRAIL_UPDATE at {current_price:.2f}, stop at {current_price + trail_distance:.2f}")
                            self._write_csv('TRAIL_UPDATE',
                                          details=f"price={current_price:.2f},trail={trail_distance:.2f}")
                
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Trail stop error: {e}")
    
    def _generate_rollup(self):
        """Generate 5-minute performance rollup"""
        try:
            # Get FVG counts
            counts = self.fvg_strategy.scan()
            
            # Calculate metrics
            win_rate = (self.metrics.wins / self.metrics.fills * 100) if self.metrics.fills > 0 else 0
            avg_mae = self.metrics.total_mae / self.metrics.fills if self.metrics.fills > 0 else 0
            avg_mfe = self.metrics.total_mfe / self.metrics.fills if self.metrics.fills > 0 else 0
            avg_duration = self.metrics.total_duration / self.metrics.fills if self.metrics.fills > 0 else 0
            avg_r = self.metrics.total_pnl / self.metrics.fills if self.metrics.fills > 0 else 0
            
            logger.info(f"ROLLUP 5m fresh={counts['fresh']} armed={counts['armed']} "
                       f"fills={self.metrics.fills} win_rate={win_rate:.1f}% "
                       f"avg_R={avg_r:.2f} avg_MAE={avg_mae:.2f} avg_MFE={avg_mfe:.2f} "
                       f"time_to_fill={avg_duration:.1f}s")
            
            self._write_csv('ROLLUP_5M',
                          details=f"fresh={counts['fresh']},armed={counts['armed']},"
                                 f"fills={self.metrics.fills},win_rate={win_rate:.1f}")
            
            self.metrics.last_rollup = time.time()
            
        except Exception as e:
            logger.error(f"Rollup error: {e}")
    
    async def run(self):
        """Main trading loop"""
        try:
            # Initialize components
            if not await self.initialize():
                logger.error("Initialization failed")
                return
            
            logger.info("Starting main loop...")
            
            while True:
                try:
                    # Refresh data
                    await self.data_cache.update_incremental()

                    # Tick execution manager for fast path order management
                    if self.exec_manager and hasattr(self.exec_manager, 'tick'):
                        current_price = self.data_cache.get_current_price()
                        if current_price:
                            self.exec_manager.tick(datetime.now(), current_price)

                    # Update session profile based on current time
                    try:
                        now_dt = datetime.now()
                        session = self._active_session(now_dt)
                        self._current_session = session
                        self._choose_session_profile(session, now_dt)
                    except Exception as e:
                        logger.error(f"Session detection error: {e}")
                        # Fallback to default session
                        self._current_session = "OTHER"
                        self._effective_profile = FVG_CFG.profile_ny

                    # Refresh ICT context (periodic updates)
                    if self.ict:
                        self.ict.refresh(now_dt)

                    # Scan for FVGs
                    counts = self.fvg_strategy.scan()
                    self.telemetry['bars_seen'] += 1

                    # Update telemetry counts
                    if 'fresh' in counts:
                        self.telemetry['fresh'] = counts['fresh']
                    if 'armed' in counts:
                        self.telemetry['armed'] = counts['armed']

                    # Check for armed FVGs
                    best_armed = self.fvg_strategy.get_best_armed()

                    if best_armed and self._check_pretrade_guards(best_armed.direction):
                        # Build signal
                        signal = self._build_signal(best_armed)

                        if signal:
                            # Track entry percentage
                            if 'entry_pct' in signal:
                                if signal['entry_pct'] == 0.62:
                                    self.telemetry['entries_62pct_highvol'] += 1
                                else:
                                    self.telemetry['entries_50pct'] += 1

                            # Execute trade
                            await self.execute_trade(signal, best_armed)
                            self.telemetry['orders_placed'] += 1
                            self.daily_trades += 1
                            self.last_trade_time_by_direction[best_armed.direction] = time.time()
                    
                    # Check for position close (skip in dry run)
                    if self.current_position and not self.dry_run:
                        # Check if position closed
                        positions = await self.broker.get_positions()
                        if not positions:
                            # Position closed
                            logger.info("Position closed")
                            self.current_position = None
                            self.current_fvg_id = None
                    
                    # Log telemetry every 5 minutes
                    if time.time() - self.last_telemetry_time >= 300:
                        self._log_telemetry()

                    # Generate rollup every 5 minutes
                    if time.time() - self.metrics.last_rollup >= 300:
                        self._generate_rollup()
                    
                    # Sleep before next iteration
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Loop iteration error: {e}")
                    await asyncio.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up...")
            
            # Cancel any open orders
            if self.exec_manager:
                await self.exec_manager.cancel_all_orders()
            
            # Disconnect broker
            if self.broker:
                await self.broker.disconnect()
            
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main entry point"""
    runner = FVGRunner()
    await runner.run()


if __name__ == '__main__':
    asyncio.run(main())