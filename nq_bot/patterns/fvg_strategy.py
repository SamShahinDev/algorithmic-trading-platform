"""
Fair Value Gap (FVG) Strategy Module
Detects and manages FVG trading opportunities
"""

import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np

try:
    from ict.scoring import confluence_score
except ImportError:
    # Fallback if ICT module not available
    def confluence_score(ict, zone, session_name, weights):
        return 0.5, {"bias": 0.5, "loc": 0.5, "raid": 0.0, "sess": 0.5, "smt": 0.5}

# Import ICT modules with fallback handling
try:
    from .modules import ict_liquidity_ob, ict_silver_bullet, ict_breaker_unicorn, ict_fvg_continuation, ict_micro_scalp
    ICT_MODULES_AVAILABLE = True
except ImportError:
    # Graceful fallback if modules not available
    ICT_MODULES_AVAILABLE = False
    class MockModule:
        @staticmethod
        def generate(*args, **kwargs):
            return []
    ict_liquidity_ob = MockModule()
    ict_silver_bullet = MockModule()
    ict_breaker_unicorn = MockModule()
    ict_fvg_continuation = MockModule()
    ict_micro_scalp = MockModule()

# Import the new independent ICT manager
try:
    from ..ict_manager import ICTPatternManager
    ICT_MANAGER_AVAILABLE = True
except ImportError:
    ICT_MANAGER_AVAILABLE = False
    ICTPatternManager = None


@dataclass
class FVGObject:
    """Represents a single Fair Value Gap"""
    id: str
    direction: str  # 'long' or 'short'
    created_at: float  # timestamp
    top: float  # upper boundary of gap
    bottom: float  # lower boundary of gap
    mid: float  # midpoint for entry
    quality: float  # quality score based on displacement metrics
    status: str  # FRESH, ARMED, CONSUMED, INVALID, EXPIRED
    origin_swing: float  # swing point that was swept before gap
    armed_at: Optional[float] = None
    last_touch_at: Optional[float] = None
    invalidation_reason: Optional[str] = None
    
    # Displacement bar metrics
    body_frac: float = 0.0
    range_pts: float = 0.0
    vol_mult: float = 0.0
    atr_mult: float = 0.0

    # ICT confluence scoring
    ict_score: Optional[float] = None
    ict_parts: Optional[Dict[str, float]] = None

    # ICT module metadata
    source_module: Optional[str] = None  # e.g., 'ict_silver_bullet', 'ict_micro'
    module_subtype: Optional[str] = None  # e.g., 'bullish_sb', 'micro_scalp'
    module_meta: Optional[Dict[str, Any]] = field(default_factory=dict)  # Module-specific data

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish FVG zone"""
        return self.direction == 'long'


class FVGStrategy:
    """Fair Value Gap detection and management strategy"""

    # NQ tick size for integer comparisons
    TICK_SIZE = 0.25

    def __init__(self, data_cache, logger, config):
        """
        Initialize FVG Strategy

        Args:
            data_cache: DataCache instance for market data
            logger: Logger instance
            config: FVG configuration dict from pattern_config
        """
        self.data_cache = data_cache
        self.logger = logger
        self.config = config

        # FVG registry
        self.fvg_registry: Dict[str, FVGObject] = {}
        self.next_id = 1

        # Track recent swings for liquidity sweep detection
        self.recent_swings: List[Dict] = []  # {'level': price, 'type': 'high'/'low', 'bar_idx': idx}
        self.swing_lookback = 20  # bars to look for swings

        # Telemetry counters for debugging (expanded for new patterns and fast-arm)
        self.telemetry_counters = {
            'trend_candidates_considered': 0,
            'sweep_candidates_considered': 0,
            'displacement_pass': 0,
            'gap_pass': 0,
            'score_pass': 0,
            'near_miss_gap_min': 0,
            'near_miss_body_frac': 0,
            'near_miss_score': 0,
            'ob_fvg_detected': 0,
            'irl_erl_fvg_detected': 0,
            'breaker_fvg_detected': 0,
            'armed_on_wick': 0,
            'armed_on_close': 0,
            'entry_micro_edge': 0,
            # Fast path telemetry
            'fast_order_placed': 0,
            'fast_order_reject': 0,
            'fast_order_filled': 0,
            'armed_invalidate_fast': 0,
            'armed_invalidate_timeout': 0,
            'armed_invalidate_defense': 0
        }

        # Armed lifetime tracking
        self.armed_lifetimes = []

        # Support for new configuration format
        if isinstance(config, dict) and 'cfg' in config:
            self.cfg = config['cfg']
        else:
            self.cfg = getattr(config, 'cfg', None) if hasattr(config, 'cfg') else None

        # Configuration for arming & continuation improvements
        self.arming_touch_frac = getattr(self.cfg, 'arming_touch_frac', 0.05)  # 5% wick-touch tolerance
        self.structure_break_grace_bars = getattr(self.cfg, 'structure_break_grace_bars', 2)  # Structure break grace period

        # Track structure breaks for grace period
        self.structure_break_tracker = {}  # {zone_id: {'bar_count': int, 'price_below/above': float}}

        # ICT context (injected by runner)
        self.ict = None  # Legacy - now using ict_manager

        # Initialize independent ICT manager
        self.ict_manager = None
        if ICT_MANAGER_AVAILABLE and self.cfg:
            try:
                self.ict_manager = ICTPatternManager(self.cfg, self.logger)
                self.logger.info("Independent ICT Pattern Manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize ICT manager: {e}")

        # Add ICT telemetry counters
        self.telemetry_counters['ict_score_calculated'] = 0
        self.telemetry_counters['ict_liquidity_ob_detected'] = 0
        self.telemetry_counters['ict_independent_evaluated'] = 0
        self.telemetry_counters['ict_independent_zones_created'] = 0
        self.telemetry_counters['ict_silver_bullet_detected'] = 0
        self.telemetry_counters['ict_unicorn_detected'] = 0
        self.telemetry_counters['ict_fvg_cont_detected'] = 0
        self.telemetry_counters['ict_micro_detected'] = 0

        # ICT Guards: Session-based counters and performance tracking
        self.session_tag_counts = {}  # (session, tag) -> count
        self.disabled_tags = set()  # Tags disabled by kill-switch
        self.tag_performance_history = {}  # tag -> deque of recent results
        self.silver_bullet_window_counts = {}  # window_id -> count
        self.last_session_reset = time.time()

        # Initialize performance history for ICT tags
        ict_tags = ['ict_liquidity_ob', 'ict_silver_bullet', 'ict_unicorn', 'ict_fvg_cont', 'ict_micro']
        cfg_obj = getattr(config, 'cfg', None) if hasattr(config, 'cfg') else config.get('cfg') if isinstance(config, dict) else None
        if cfg_obj and hasattr(cfg_obj, 'ict_guards'):
            max_trades = cfg_obj.ict_guards.tag_killswitch_window_trades
        else:
            max_trades = 20  # Default fallback

        for tag in ict_tags:
            self.tag_performance_history[tag] = deque(maxlen=max_trades)

        # Legacy session counters (keep for backwards compatibility)
        self.session_counters = {
            'ict_micro': 0,  # Count of micro-scalp trades this session
            'last_session_reset': time.time()
        }

    def _attach_ict_score(self, zone: FVGObject) -> None:
        """Attach ICT confluence score to a fresh zone"""
        try:
            self.logger.info(f"ICT_SCORE_ENTRY zone={zone.id} method_called=True")

            # Debug logging to identify which condition is failing
            # Fix reference: ICT manager is injected as ict_manager, not ict
            has_ict = bool(getattr(self, 'ict_manager', None))
            has_cfg = bool(self.cfg)
            ict_context_enabled = bool(self.cfg and hasattr(self.cfg, 'ict_context') and self.cfg.ict_context.enabled)

            self.logger.info(f"ICT_SCORE_DEBUG zone={zone.id} has_ict={has_ict} has_cfg={has_cfg} ict_context_enabled={ict_context_enabled}")

            # Use ict_manager instead of ict
            if not getattr(self, 'ict_manager', None) or not self.cfg or not (hasattr(self.cfg, 'ict_context') and self.cfg.ict_context.enabled):
                self.logger.info(f"ICT_SCORE_SKIPPED zone={zone.id} reason=missing_requirements")
                return

            # Get current session name
            session_name = getattr(self, '_active_session', 'OTHER')

            # Calculate confluence score
            score, parts = confluence_score(
                self.ict_manager,
                zone,
                session_name,
                self.cfg.ict_session.weights_bias_loc_raid_sess_smt
            )

            # Attach to zone
            zone.ict_score = score
            zone.ict_parts = parts

            # Update telemetry
            self.telemetry_counters['ict_score_calculated'] += 1

            # Log ICT score
            self.logger.info(f"ICT_SCORE zone={zone.id} score={score:.3f} "
                           f"bias={parts.get('bias', 0):.2f} "
                           f"loc={parts.get('loc', 0):.2f} "
                           f"raid={parts.get('raid', 0):.2f} "
                           f"sess={parts.get('sess', 0):.2f} "
                           f"smt={parts.get('smt', 0):.2f}")

        except Exception as e:
            self.logger.error(f"Error calculating ICT score for zone {zone.id}: {e}")

    def _check_ict_guards(self, zone: FVGObject, session: str) -> bool:
        """
        Check ICT guards before allowing zone execution

        Args:
            zone: FVG zone object
            session: Current trading session

        Returns:
            True if zone is allowed, False if blocked by guards
        """
        try:
            # Get ICT guards configuration
            cfg_obj = getattr(self.config, 'cfg', None) if hasattr(self.config, 'cfg') else self.config.get('cfg') if isinstance(self.config, dict) else None
            if cfg_obj and hasattr(cfg_obj, 'ict_guards'):
                guards = cfg_obj.ict_guards
            else:
                # Return True if no guards configured
                return True

            tag = getattr(zone, 'source_module', None) or 'core_fvg'

            # Check if tag is disabled by kill-switch
            if tag in self.disabled_tags:
                self.logger.info(f"ICT_KILLSWITCH_BLOCK zone={zone.id} tag={tag} "
                               f"reason=performance_below_threshold")
                return False

            # Per-session caps enforcement
            session_tag_key = (session, tag)
            current_count = self.session_tag_counts.get(session_tag_key, 0)

            # Micro-scalp per-session cap
            if tag == 'ict_micro' and current_count >= guards.micro_max_trades_per_session:
                self.logger.info(f"ICT_CAP_BLOCK zone={zone.id} tag={tag} "
                               f"session={session} count={current_count} "
                               f"max={guards.micro_max_trades_per_session}")
                return False

            # Silver Bullet per-window cap
            if tag == 'ict_silver_bullet':
                window_id = self._get_current_sb_window()
                if window_id:
                    window_count = self.silver_bullet_window_counts.get(window_id, 0)
                    if window_count >= guards.silver_max_trades_per_window:
                        self.logger.info(f"ICT_CAP_BLOCK zone={zone.id} tag={tag} "
                                       f"window={window_id} count={window_count} "
                                       f"max={guards.silver_max_trades_per_window}")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking ICT guards for zone {zone.id}: {e}")
            return True  # Allow by default on error

    def _get_current_sb_window(self) -> Optional[str]:
        """Get current Silver Bullet window identifier"""
        try:
            now = datetime.now()
            hour = now.hour

            # Silver Bullet windows: 3-4am, 10-11am, 2-3pm (example windows)
            if 3 <= hour < 4:
                return "3-4am"
            elif 10 <= hour < 11:
                return "10-11am"
            elif 14 <= hour < 15:  # 2-3pm
                return "2-3pm"

            return None

        except Exception as e:
            self.logger.error(f"Error determining SB window: {e}")
            return None

    def _increment_session_tag_count(self, session: str, tag: str):
        """Increment session tag counter"""
        session_tag_key = (session, tag)
        if session_tag_key not in self.session_tag_counts:
            self.session_tag_counts[session_tag_key] = 0
        self.session_tag_counts[session_tag_key] += 1

        # Increment Silver Bullet window count
        if tag == 'ict_silver_bullet':
            window_id = self._get_current_sb_window()
            if window_id:
                if window_id not in self.silver_bullet_window_counts:
                    self.silver_bullet_window_counts[window_id] = 0
                self.silver_bullet_window_counts[window_id] += 1

    def _update_tag_performance(self, tag: str, result: str, r_multiple: float):
        """Update performance history for kill-switch evaluation"""
        try:
            if tag not in self.tag_performance_history:
                # Get ICT guards configuration
                cfg_obj = getattr(self.config, 'cfg', None) if hasattr(self.config, 'cfg') else self.config.get('cfg') if isinstance(self.config, dict) else None
                if cfg_obj and hasattr(cfg_obj, 'ict_guards'):
                    max_trades = cfg_obj.ict_guards.tag_killswitch_window_trades
                else:
                    max_trades = 20  # Default fallback
                self.tag_performance_history[tag] = deque(maxlen=max_trades)

            # Record result
            is_win = result == 'win'
            self.tag_performance_history[tag].append({
                'win': is_win,
                'r_multiple': r_multiple,
                'timestamp': time.time()
            })

            # Check kill-switch criteria
            self._evaluate_killswitch(tag)

        except Exception as e:
            self.logger.error(f"Error updating performance for tag {tag}: {e}")

    def _evaluate_killswitch(self, tag: str):
        """Evaluate if tag should be disabled by kill-switch"""
        try:
            # Get ICT guards configuration
            cfg_obj = getattr(self.config, 'cfg', None) if hasattr(self.config, 'cfg') else self.config.get('cfg') if isinstance(self.config, dict) else None
            if not cfg_obj or not hasattr(cfg_obj, 'ict_guards'):
                return
            guards = cfg_obj.ict_guards
            history = self.tag_performance_history.get(tag, [])

            # Need minimum trades to evaluate
            if len(history) < guards.tag_killswitch_window_trades:
                return

            # Calculate recent performance
            wins = sum(1 for trade in history if trade['win'])
            total_trades = len(history)
            win_rate = wins / total_trades if total_trades > 0 else 0

            total_r = sum(trade['r_multiple'] for trade in history)
            avg_r = total_r / total_trades if total_trades > 0 else 0

            # Check kill-switch criteria
            should_disable = (win_rate < guards.tag_disable_if_win_lt or
                             avg_r < guards.tag_disable_if_avgR_lt)

            if should_disable and tag not in self.disabled_tags:
                self.disabled_tags.add(tag)
                self.logger.info(f"ICT_KILLSWITCH_TRIGGERED tag={tag} "
                               f"win_rate={win_rate:.3f} avg_r={avg_r:.3f} "
                               f"trades={total_trades} disabled_until_next_session")
            elif not should_disable and tag in self.disabled_tags:
                self.disabled_tags.remove(tag)
                self.logger.info(f"ICT_KILLSWITCH_RESTORED tag={tag} "
                               f"win_rate={win_rate:.3f} avg_r={avg_r:.3f}")

        except Exception as e:
            self.logger.error(f"Error evaluating kill-switch for tag {tag}: {e}")

    def _reset_session_counts(self, new_session: str):
        """Reset session-based counts when session changes"""
        try:
            # Clear session tag counts for previous sessions
            current_time = time.time()
            keys_to_remove = []

            for (session, tag), count in self.session_tag_counts.items():
                if session != new_session:
                    keys_to_remove.append((session, tag))

            for key in keys_to_remove:
                del self.session_tag_counts[key]

            # Clear disabled tags at session change
            if self.disabled_tags:
                self.logger.info(f"ICT_KILLSWITCH_SESSION_RESET "
                               f"restored_tags={list(self.disabled_tags)} new_session={new_session}")
                self.disabled_tags.clear()

            # Clear Silver Bullet window counts
            self.silver_bullet_window_counts.clear()

            self.last_session_reset = current_time

        except Exception as e:
            self.logger.error(f"Error resetting session counts: {e}")

    def record_trade_completion(self, zone_id: str, result: str, r_multiple: float):
        """
        Record trade completion for performance tracking

        Args:
            zone_id: Zone ID that was traded
            result: 'win' or 'loss'
            r_multiple: R-multiple of the trade
        """
        try:
            # Find the zone to get its tag
            zone = self.fvg_registry.get(zone_id)
            if not zone:
                return

            tag = getattr(zone, 'source_module', None) or 'core_fvg'

            # Update performance tracking
            self._update_tag_performance(tag, result, r_multiple)

            self.logger.info(f"ICT_TRADE_COMPLETE zone={zone_id} tag={tag} "
                           f"result={result} r_multiple={r_multiple:.2f}")

        except Exception as e:
            self.logger.error(f"Error recording trade completion for zone {zone_id}: {e}")

    def check_session_reset(self):
        """Check if session has changed and reset counters if needed"""
        try:
            if not hasattr(self, 'ict_context') or not self.ict_context:
                return

            current_session = getattr(self.ict_context, 'session_name', 'OTHER')

            # If we don't have a previous session recorded, store it
            if not hasattr(self, '_last_known_session'):
                self._last_known_session = current_session
                return

            # Check if session changed
            if current_session != self._last_known_session:
                self.logger.info(f"ICT_SESSION_CHANGE from={self._last_known_session} to={current_session}")
                self._reset_session_counts(current_session)
                self._last_known_session = current_session

        except Exception as e:
            self.logger.error(f"Error checking session reset: {e}")

    def _collect_ict_candidates(self, bars, i: int, now_dt: datetime) -> List[Dict[str, Any]]:
        """
        Collect ICT pattern candidates from all enabled modules

        Args:
            bars: Bar data
            i: Current bar index
            now_dt: Current datetime

        Returns:
            List of zone candidates from ICT modules
        """
        candidates = []

        try:
            if not ICT_MODULES_AVAILABLE or not self.cfg or not getattr(self, 'ict_manager', None):
                return candidates

            # Convert datetime to Chicago time for Silver Bullet
            try:
                from zoneinfo import ZoneInfo
                ct_now = now_dt.astimezone(ZoneInfo("America/Chicago"))
            except Exception:
                ct_now = now_dt  # Fallback

            tick_size = getattr(self.cfg, 'tick_size', 0.25)

            # Reset session counters if needed (simple daily reset)
            self._reset_session_counters_if_needed()

            # Collect from each enabled module
            if self.cfg.ict_modules.enable_liquidity_ob:
                try:
                    cands = ict_liquidity_ob.generate(bars, i, self.cfg, tick_size)
                    candidates.extend(cands)
                except Exception as e:
                    self.logger.debug(f"Error in liquidity_ob module: {e}")

            if self.cfg.ict_modules.enable_silver_bullet:
                try:
                    cands = ict_silver_bullet.generate(bars, i, self.cfg, tick_size, ct_now)
                    candidates.extend(cands)
                except Exception as e:
                    self.logger.debug(f"Error in silver_bullet module: {e}")

            if self.cfg.ict_modules.enable_breaker_unicorn:
                try:
                    cands = ict_breaker_unicorn.generate(bars, i, self.cfg, tick_size)
                    candidates.extend(cands)
                except Exception as e:
                    self.logger.debug(f"Error in breaker_unicorn module: {e}")

            if self.cfg.ict_modules.enable_fvg_continuation:
                try:
                    cands = ict_fvg_continuation.generate(self.ict_manager, bars, i, self.cfg, tick_size)
                    candidates.extend(cands)
                except Exception as e:
                    self.logger.debug(f"Error in fvg_continuation module: {e}")

            if self.cfg.ict_modules.enable_micro_scalp:
                try:
                    used_count = self.session_counters.get('ict_micro', 0)
                    cands = ict_micro_scalp.generate(bars, i, self.cfg, self.ict_manager, tick_size, used_count)
                    candidates.extend(cands)
                except Exception as e:
                    self.logger.debug(f"Error in micro_scalp module: {e}")

        except Exception as e:
            self.logger.error(f"Error collecting ICT candidates: {e}")

        return candidates

    def _emit_from_candidates(self, candidates: List[Dict[str, Any]], now_dt: datetime) -> int:
        """
        Create FVG zones from ICT module candidates

        Args:
            candidates: List of zone candidates from ICT modules
            now_dt: Current datetime

        Returns:
            Number of zones created
        """
        zones_created = 0

        try:
            for cand in candidates:
                # Create FVG zone from candidate
                zone = self._create_zone_from_candidate(cand)
                if not zone:
                    continue

                # Attach ICT score
                self._attach_ict_score(zone)

                # Register as fresh zone
                self.fvg_registry[zone.id] = zone

                # Update telemetry
                tag = cand.get('tag', 'ict_generic')
                counter_key = f"{tag}_detected"
                if counter_key in self.telemetry_counters:
                    self.telemetry_counters[counter_key] += 1

                # Update session counters for micro-scalp
                if tag == 'ict_micro':
                    self.session_counters['ict_micro'] += 1

                # Log zone creation with ICT fields
                subtype = cand.get('subtype', 'unknown')
                quality_boost = cand.get('quality_boost', 0.0)

                # Get ICT context fields
                ict_bias = getattr(self.ict_context, 'bias_dir', 'neutral') if hasattr(self, 'ict_context') else 'neutral'
                ict_draw_target = getattr(self.ict_context, 'draw_target', None) if hasattr(self, 'ict_context') else None
                ict_premium_discount = getattr(self.ict_context, 'premium_discount', 'neutral') if hasattr(self, 'ict_context') else 'neutral'
                ict_ote_overlap = getattr(self.ict_context, 'ote_overlap', False) if hasattr(self, 'ict_context') else False
                ict_raid_recent = getattr(self.ict_context, 'raid_recent', False) if hasattr(self, 'ict_context') else False
                session = getattr(self.ict_context, 'session_name', 'OTHER') if hasattr(self, 'ict_context') else 'OTHER'

                # Make sure ict_draw_target is safe for formatting
                ict_draw_target_safe = ict_draw_target if ict_draw_target is not None else 'none'

                # Make sure ict_draw_target is safe for formatting
                ict_draw_target_safe = ict_draw_target if ict_draw_target is not None else 'none'

                self.logger.info(f"ICT_ZONE_CREATED tag={tag} subtype={subtype} "
                               f"zone={zone.id} dir={zone.direction} "
                               f"bounds={zone.bottom:.2f}-{zone.top:.2f} "
                               f"quality_boost={quality_boost:.3f} "
                               f"ict_score={zone.ict_score:.3f} "
                               f"pattern_tag={zone.source_module or tag} "
                               f"ict_bias={ict_bias} ict_draw_target={ict_draw_target_safe} "
                               f"ict_premium_discount={ict_premium_discount} "
                               f"ict_ote_overlap={ict_ote_overlap} ict_raid_recent={ict_raid_recent} "
                               f"session={session}")

                zones_created += 1

        except Exception as e:
            self.logger.error(f"Error emitting ICT candidates: {e}")

        return zones_created

    def _create_zone_from_candidate(self, cand: Dict[str, Any]) -> Optional[FVGObject]:
        """
        Create FVG zone object from ICT module candidate

        Args:
            cand: Candidate dict from ICT module

        Returns:
            FVGObject or None if creation failed
        """
        try:
            direction = 'long' if cand['dir'] == 'long' else 'short'
            lower = cand['lower']
            upper = cand['upper']

            # Validate bounds
            if lower >= upper:
                return None

            # Create zone ID
            fvg_id = f"ICT_{self.next_id}"
            self.next_id += 1

            # Calculate quality boost
            base_quality = 0.5  # Base quality for ICT zones
            quality_boost = cand.get('quality_boost', 0.0)
            quality = min(1.0, base_quality + quality_boost)

            # Create zone
            zone = FVGObject(
                id=fvg_id,
                direction=direction,
                created_at=time.time(),
                top=upper,
                bottom=lower,
                mid=(upper + lower) / 2,
                quality=quality,
                status='FRESH',
                origin_swing=None,  # ICT zones don't use swing detection
                body_frac=1.0,      # Default values for ICT zones
                range_pts=upper - lower,
                vol_mult=1.0,
                atr_mult=1.0,
                source_module=cand.get('tag', 'ict_generic'),
                module_subtype=cand.get('subtype', 'unknown'),
                module_meta=cand.get('meta', {})
            )

            return zone

        except (KeyError, TypeError, ValueError) as e:
            self.logger.debug(f"Error creating zone from candidate: {e}")
            return None

    def _reset_session_counters_if_needed(self):
        """Reset session counters at start of new session (simplified daily reset)"""
        try:
            current_time = time.time()
            last_reset = self.session_counters.get('last_session_reset', 0)

            # Reset daily (24 hours = 86400 seconds)
            if current_time - last_reset > 86400:
                self.session_counters['ict_micro'] = 0
                self.session_counters['last_session_reset'] = current_time

        except Exception as e:
            self.logger.debug(f"Error resetting session counters: {e}")

    def _prof(self):
        """Get effective FVG profile (session-aware or fallback to normal/responsive)"""
        # First priority: session-aware profile from runner
        if hasattr(self, 'runner') and hasattr(self.runner, '_effective_profile') and self.runner._effective_profile:
            return self.runner._effective_profile

        # Second priority: normal/responsive toggle
        if self.cfg is not None:
            return self.cfg.responsive if self.cfg.profile_active == "responsive" else self.cfg.normal

        # Fallback to legacy config behavior
        return None

    def _body_fraction(self, bar: pd.Series) -> float:
        """Calculate body fraction of a bar"""
        bar_range = bar['high'] - bar['low']
        if bar_range <= 0:
            return 0.0
        body_size = abs(bar['close'] - bar['open'])
        return body_size / bar_range

    def _get_session_name(self, now_dt: datetime) -> str:
        """Get current session name with robust string/callable handling"""
        sess_attr = getattr(self.runner, "_active_session", None) if hasattr(self, 'runner') else None
        # If older runners expose a callable, call it; otherwise accept the string value.
        if callable(sess_attr):
            try:
                return sess_attr(now_dt) or "NY_RTH"
            except Exception:
                return "NY_RTH"
        if isinstance(sess_attr, str) and sess_attr:
            return sess_attr
        return "NY_RTH"

    def _session_allows_fast(self, now_dt: datetime) -> bool:
        """Check if current session allows fast path orders"""
        if not self.cfg or not self.cfg.fast_path:
            return False

        sess = self._get_session_name(now_dt)
        fp = self.cfg.fast_path
        return ((sess == "TOKYO" and fp.enable_tokyo) or
                (sess == "LONDON" and fp.enable_london) or
                (sess in ("NY_RTH", "OTHER") and fp.enable_ny))

    def _maybe_fast_order(self, zone: FVGObject, current_bar: pd.Series, current_time: datetime) -> bool:
        """Try to place a fast order on armed zone"""
        if not self._session_allows_fast(current_time):
            return False

        if not self.cfg or not self.cfg.fast_path:
            return False

        fp = self.cfg.fast_path

        # Tape checks
        body = self._body_fraction(current_bar)
        rng = max(current_bar['high'] - current_bar['low'], self.TICK_SIZE)

        # For now, assume spread is 1 tick (would need market data for actual spread)
        spread_ticks = 1

        if body < fp.min_body_frac_for_fast:
            return False
        if rng < fp.min_tape_range_pts:
            return False
        if spread_ticks > fp.max_spread_ticks:
            return False

        # Zone geometry
        zone_ticks = int(round((zone.top - zone.bottom) / self.TICK_SIZE))
        is_micro = zone_ticks <= fp.max_zone_ticks_for_edge

        # Choose price
        if is_micro:
            # Edge front-run for micro zones
            if zone.direction == 'long':
                px = zone.bottom + fp.front_run_ticks * self.TICK_SIZE
            else:
                px = zone.top - fp.front_run_ticks * self.TICK_SIZE
            entry_kind = "edge_front_run"
        else:
            # Mid or 62% for non-micro zones
            mid = zone.bottom + 0.50 * (zone.top - zone.bottom)
            if fp.prefer_mid_vs_62 == "62":
                if zone.direction == 'long':
                    p62 = zone.bottom + 0.62 * (zone.top - zone.bottom)
                else:
                    p62 = zone.top - 0.62 * (zone.top - zone.bottom)
                px = p62
            else:
                px = mid
            entry_kind = "mid_or_62"

        # Store previous defense cap for restoration
        zone._prev_defense = getattr(zone, "_prev_defense", None)
        if zone._prev_defense is None:
            zone._prev_defense = self._current_defense_cap()

        # Temporarily relax defense
        self._temp_defense_cap = fp.arm_defense_cap

        # Submit fast order (would need execution manager)
        ok = self._submit_fast_child(zone, px, current_time,
                                    mit=fp.use_mit_on_touch,
                                    ttl=fp.ttl_seconds,
                                    max_slip_ticks=fp.protect_max_slip_ticks,
                                    tag=f"fast:{entry_kind}:{zone.id}")

        if ok:
            self.telemetry_counters['fast_order_placed'] += 1
            self.logger.info(f"FAST_ORDER zone={zone.id} price={px:.2f} kind={entry_kind} ticks={zone_ticks}")
            return True
        else:
            # Restore defense if order not placed
            self._restore_defense(zone)
            self.telemetry_counters['fast_order_reject'] += 1
            return False

    def _current_defense_cap(self) -> float:
        """Get current defense cap (temporary or profile)"""
        if hasattr(self, "_temp_defense_cap") and self._temp_defense_cap is not None:
            return self._temp_defense_cap

        prof = self._prof()
        if prof:
            return prof.defense_max_fill_pct

        # Legacy fallback
        lifecycle = self.config.get('lifecycle', {})
        return lifecycle.get('invalidate_frac', 0.90)

    def _restore_defense(self, zone: FVGObject):
        """Restore defense cap after fast order completes"""
        if hasattr(self, "_temp_defense_cap"):
            self._temp_defense_cap = None
        # Previous defense stored on zone will be used

    def _track_invalidation(self, fvg: FVGObject, reason: str, current_time: float):
        """Track invalidation and record armed lifetime if applicable"""
        if fvg.armed_at:
            # Calculate and record armed lifetime
            lifetime_ms = int((current_time - fvg.armed_at) * 1000)
            self.armed_lifetimes.append(lifetime_ms)

            # Track invalidation reason
            if 'fast' in reason:
                self.telemetry_counters['armed_invalidate_fast'] += 1
            elif 'timeout' in reason:
                self.telemetry_counters['armed_invalidate_timeout'] += 1
            elif 'defense' in reason or 'consumed' in reason:
                self.telemetry_counters['armed_invalidate_defense'] += 1

    def _submit_fast_child(self, zone: FVGObject, price: float, now_dt: datetime,
                          mit: bool, ttl: int, max_slip_ticks: int, tag: str) -> bool:
        """Submit fast order through execution manager"""
        # ICT Guards check
        session = getattr(self.ict_context, 'session_name', 'OTHER') if hasattr(self, 'ict_context') else 'OTHER'
        if not self._check_ict_guards(zone, session):
            self.telemetry_counters['fast_order_reject'] += 1
            return False

        # Respect existing guards
        if not self._rsi_ok_for_zone(zone, now_dt):
            self.telemetry_counters['fast_order_reject'] += 1
            self.logger.info(f"FAST_ORDER_REJECT zone={zone.id} reason=rsi_block")
            return False

        if self._burst_blocked(zone):
            self.telemetry_counters['fast_order_reject'] += 1
            self.logger.info(f"FAST_ORDER_REJECT zone={zone.id} reason=burst_block")
            return False

        if self._daily_cap_blocked(zone):
            self.telemetry_counters['fast_order_reject'] += 1
            self.logger.info(f"FAST_ORDER_REJECT zone={zone.id} reason=daily_cap_block")
            return False

        side = "buy" if zone.direction == "long" else "sell"

        # Ensure we have an execution manager
        if not hasattr(self, "execution") or self.execution is None:
            self.logger.error("FAST_ORDER_ERROR: no_execution_manager")
            self.telemetry_counters['fast_order_reject'] += 1
            return False

        try:
            # Get configuration values
            stop_pts = getattr(self.cfg, 'stop_pts', 7.5) if self.cfg else 7.5
            tp_pts = getattr(self.cfg, 'tp_pts', 17.5) if self.cfg else 17.5
            tick_size = getattr(self.cfg, 'tick_size', 0.25) if self.cfg else 0.25

            ok = self.execution.place_limit_or_mit(
                zone_id=zone.id,
                side=side,
                price=price,
                stop_loss_pts=stop_pts,
                take_profit_pts=tp_pts,
                ttl_seconds=ttl,
                max_slip_ticks=max_slip_ticks,
                tick_size=tick_size,
                mit=mit,
                tag=tag
            )

            if ok:
                self.telemetry_counters['fast_order_placed'] += 1
                self.logger.info(f"FAST_ORDER_PLACED zone={zone.id} side={side} price={price:.2f} ttl={ttl}s tag={tag}")

                # Increment session tag counters for ICT guards
                tag_name = getattr(zone, 'source_module', None) or 'core_fvg'
                session = getattr(self.ict_context, 'session_name', 'OTHER') if hasattr(self, 'ict_context') else 'OTHER'
                self._increment_session_tag_count(session, tag_name)
            else:
                self.telemetry_counters['fast_order_reject'] += 1
                self.logger.info(f"FAST_ORDER_REJECT zone={zone.id} reason=execution_failed")

            return bool(ok)

        except Exception as e:
            self.logger.error(f"FAST_ORDER_ERROR zone={zone.id} error={str(e)}")
            self.telemetry_counters['fast_order_reject'] += 1
            return False

    def _rsi_ok_for_zone(self, zone: FVGObject, now_dt: datetime) -> bool:
        """Check if RSI allows trade in this direction"""
        # Placeholder for RSI checks - can integrate with existing RSI logic
        return True

    def _burst_blocked(self, zone: FVGObject) -> bool:
        """Check if burst guard blocks this trade"""
        # Placeholder for burst protection logic
        return False

    def _daily_cap_blocked(self, zone: FVGObject) -> bool:
        """Check if daily trade cap blocks this trade"""
        # Placeholder for daily cap logic
        return False

    def scan(self) -> Dict[str, int]:
        """
        Scan for FVG opportunities

        Returns:
            Dict with counts by status: {"fresh": N, "armed": M, "consumed": K, "invalid": J}
        """
        try:
            # Check for session changes and reset ICT guards if needed
            self.check_session_reset()

            # Get recent 1m bars
            bars_1m = self.data_cache.get_bars('1m')
            if bars_1m is None or len(bars_1m) < 30:
                self.logger.info(f"FVG_SCAN: Insufficient bars. Got {len(bars_1m) if bars_1m is not None else 0} bars, need 30+")
                return {"fresh": 0, "armed": 0, "consumed": 0, "invalid": 0}
            
            # Limit to lookback period
            lookback = self.config.get('lifecycle', {}).get('lookback_bars', 300)
            if len(bars_1m) > lookback:
                bars_1m = bars_1m.iloc[-lookback:]
            
            # Calculate indicators
            atr_14 = self._calculate_atr(bars_1m, 14)
            avg_volume_20 = bars_1m['volume'].rolling(20).mean()
            
            # Update recent swings
            self._update_swings(bars_1m)
            
            # Detect new FVGs (core patterns)
            self._detect_fvgs(bars_1m, atr_14, avg_volume_20)

            # Detect new advanced patterns if enabled and profile configured
            prof = self._prof()
            now_dt = datetime.now()

            # Define default scan index for ICT patterns (second-to-last bar)
            i = len(bars_1m) - 2 if len(bars_1m) >= 2 else 0

            if prof is not None:
                # Check last few bars for new patterns
                for scan_i in range(max(1, len(bars_1m) - 5), len(bars_1m) - 1):
                    if self.cfg and self.cfg.patterns.enable_ob_fvg:
                        try:
                            from .modules.ob_fvg import scan_ob_fvg
                            scan_ob_fvg(self, bars_1m, scan_i, now_dt, prof)
                        except Exception as e:
                            self.logger.debug(f"OB-FVG scan error: {e}")

                    if self.cfg and self.cfg.patterns.enable_irl_erl_fvg:
                        try:
                            from .modules.irl_erl_fvg import scan_irl_erl_fvg
                            scan_irl_erl_fvg(self, bars_1m, scan_i, now_dt, prof)
                        except Exception as e:
                            self.logger.debug(f"IRL-ERL-FVG scan error: {e}")

                    if self.cfg and self.cfg.patterns.enable_breaker_fvg:
                        try:
                            from .modules.breaker_fvg import scan_breaker_fvg
                            scan_breaker_fvg(self, bars_1m, scan_i, now_dt, prof)
                        except Exception as e:
                            self.logger.debug(f"Breaker-FVG scan error: {e}")

            # ICT Pattern Scanning - Independent of FVG detection and profile status
            if self.ict_manager:
                try:
                    self.logger.info(f"DEBUG: Starting ICT manager scan at bar {i}, prof={prof}")
                    self._scan_independent_ict_patterns(bars_1m, i, now_dt, prof)
                    self.telemetry_counters['ict_independent_evaluated'] += 1
                    self.logger.info(f"DEBUG: Independent ICT patterns evaluated at bar {i}")
                except Exception as e:
                    self.logger.error(f"Independent ICT pattern scan error: {e}")
            elif ICT_MODULES_AVAILABLE and self.cfg:
                try:
                    self.logger.info(f"DEBUG: Starting legacy ICT scan at bar {i}, prof={prof}")
                    self._scan_ict_patterns(bars_1m, i, now_dt, prof)
                    self.logger.info(f"DEBUG: Legacy ICT patterns scanned at bar {i}")
                except Exception as e:
                    self.logger.error(f"Legacy ICT pattern scan error: {e}")
            else:
                self.logger.info(f"DEBUG: ICT scanning skipped - ict_manager={self.ict_manager}, ICT_MODULES_AVAILABLE={ICT_MODULES_AVAILABLE}, cfg={self.cfg}")

            # Update existing FVG states
            current_price = bars_1m['close'].iloc[-1]
            current_bar = bars_1m.iloc[-1]
            self._update_fvg_states(current_price, current_bar)
            
            # Clean up expired/invalid FVGs
            self._cleanup_registry()
            
            # Count by status
            status_counts = {"fresh": 0, "armed": 0, "consumed": 0, "invalid": 0}
            for fvg in self.fvg_registry.values():
                status_lower = fvg.status.lower()
                if status_lower in status_counts:
                    status_counts[status_lower] += 1

            # Log telemetry counters periodically
            # Calculate armed lifetime stats
            avg_armed_lifetime = 0
            if self.armed_lifetimes:
                avg_armed_lifetime = sum(self.armed_lifetimes) / len(self.armed_lifetimes)

            # Enhanced diagnostic telemetry logging
            diagnostic_info = (
                f"FVG_TELEMETRY bars={len(bars_1m)} "
                f"displacement_pass={self.telemetry_counters['displacement_pass']} "
                f"gap_pass={self.telemetry_counters['gap_pass']} "
                f"score_pass={self.telemetry_counters['score_pass']} "
                f"trend_candidates={self.telemetry_counters['trend_candidates_considered']} "
                f"sweep_candidates={self.telemetry_counters['sweep_candidates_considered']} "
                f"near_miss_gap={self.telemetry_counters['near_miss_gap_min']} "
                f"near_miss_body={self.telemetry_counters['near_miss_body_frac']} "
                f"near_miss_score={self.telemetry_counters['near_miss_score']} "
                f"armed_on_wick={self.telemetry_counters['armed_on_wick']} "
                f"armed_on_close={self.telemetry_counters['armed_on_close']} "
                f"entry_micro_edge={self.telemetry_counters['entry_micro_edge']} "
                f"fast_placed={self.telemetry_counters['fast_order_placed']} "
                f"fast_reject={self.telemetry_counters['fast_order_reject']} "
                f"ict_scored={self.telemetry_counters['ict_score_calculated']} "
                f"avg_armed_life_ms={avg_armed_lifetime:.0f} "
                f"arming_tolerance={self.arming_touch_frac:.3f} "
                f"structure_grace_bars={self.structure_break_grace_bars} "
                f"structure_breaks_active={len(self.structure_break_tracker)} "
                f"disabled_tags={len(self.disabled_tags)} "
                f"session_tag_counts={len(self.session_tag_counts)}"
            )
            self.logger.info(diagnostic_info)

            # Log pattern detection bottlenecks if we have near-misses
            if (self.telemetry_counters['near_miss_gap_min'] > 0 or
                self.telemetry_counters['near_miss_body_frac'] > 0 or
                self.telemetry_counters['near_miss_score'] > 0):

                bottleneck_info = (
                    f"FVG_BOTTLENECK_ANALYSIS "
                    f"gap_near_miss={self.telemetry_counters['near_miss_gap_min']} "
                    f"body_frac_near_miss={self.telemetry_counters['near_miss_body_frac']} "
                    f"score_near_miss={self.telemetry_counters['near_miss_score']} "
                    f"suggestion='Consider relaxing thresholds for London session'"
                )
                self.logger.info(bottleneck_info)

            return status_counts
            
        except Exception as e:
            import traceback
            self.logger.error(f"FVG scan error: {e}")
            self.logger.error(f"FVG scan traceback: {traceback.format_exc()}")
            return {"fresh": 0, "armed": 0, "consumed": 0, "invalid": 0}

    def _scan_ict_patterns(self, bars_1m, i: int, now_dt: datetime, prof):
        """Scan ICT patterns and register any candidates as FVG zones"""
        try:
            tick_size = self.data_cache.tick_size if hasattr(self.data_cache, 'tick_size') else 0.25
            # Convert UTC to Chicago time for ICT pattern timing
            from zoneinfo import ZoneInfo
            ct_now = now_dt.astimezone(ZoneInfo("America/Chicago")) if now_dt.tzinfo else now_dt

            # Convert bars to list format expected by ICT modules
            bars_list = []
            for idx, row in bars_1m.iterrows():
                bar_obj = type('Bar', (), {
                    'high': row['high'],
                    'low': row['low'],
                    'open': row['open'],
                    'close': row['close'],
                    'volume': row['volume']
                })()
                bars_list.append(bar_obj)

            # Scan Silver Bullet patterns
            if self.cfg.ict_modules.enable_silver_bullet:
                try:
                    candidates = ict_silver_bullet.generate(bars_list, i, self.cfg, tick_size, ct_now)
                    for candidate in candidates:
                        self._register_ict_zone(candidate, 'ict_silver_bullet', prof, now_dt)
                        self.telemetry_counters['ict_silver_bullet_detected'] += 1
                except Exception as e:
                    self.logger.debug(f"Silver Bullet scan error: {e}")

            # Scan Liquidity OB patterns
            if self.cfg.ict_modules.enable_liquidity_ob:
                try:
                    candidates = ict_liquidity_ob.generate(bars_list, i, self.cfg, tick_size, ct_now)
                    for candidate in candidates:
                        self._register_ict_zone(candidate, 'ict_liquidity_ob', prof, now_dt)
                        self.telemetry_counters['ict_liquidity_ob_detected'] += 1
                except Exception as e:
                    self.logger.debug(f"Liquidity OB scan error: {e}")

            # Scan Micro Scalp patterns
            if self.cfg.ict_modules.enable_micro_scalp:
                try:
                    candidates = ict_micro_scalp.generate(bars_list, i, self.cfg, tick_size, ct_now)
                    for candidate in candidates:
                        self._register_ict_zone(candidate, 'ict_micro_scalp', prof, now_dt)
                except Exception as e:
                    self.logger.debug(f"Micro Scalp scan error: {e}")

            # Scan other ICT patterns
            if self.cfg.ict_modules.enable_breaker_unicorn:
                try:
                    candidates = ict_breaker_unicorn.generate(bars_list, i, self.cfg, tick_size, ct_now)
                    for candidate in candidates:
                        self._register_ict_zone(candidate, 'ict_breaker_unicorn', prof, now_dt)
                except Exception as e:
                    self.logger.debug(f"Breaker Unicorn scan error: {e}")

            if self.cfg.ict_modules.enable_fvg_continuation:
                try:
                    candidates = ict_fvg_continuation.generate(bars_list, i, self.cfg, tick_size, ct_now)
                    for candidate in candidates:
                        self._register_ict_zone(candidate, 'ict_fvg_continuation', prof, now_dt)
                except Exception as e:
                    self.logger.debug(f"FVG Continuation scan error: {e}")

        except Exception as e:
            self.logger.error(f"Error scanning ICT patterns: {e}")

    def _register_ict_zone(self, candidate: Dict, source_module: str, prof, timestamp: datetime):
        """Register an ICT pattern candidate as an FVG zone"""
        try:
            # Create FVGObject from ICT candidate
            zone_id = f"ICT_{source_module}_{len(self.fvg_registry)}"

            fvg = FVGObject(
                zone_id=zone_id,
                direction=candidate.get('dir', 'unknown'),
                upper_bound=candidate.get('upper', 0.0),
                lower_bound=candidate.get('lower', 0.0),
                detection_time=timestamp,
                displacement_bar_range=0.0,  # ICT patterns don't use this
                body_fraction=1.0,  # ICT patterns are always valid
                volume_multiplier=1.0,  # ICT patterns don't use this
                gap_ticks=abs(candidate.get('upper', 0.0) - candidate.get('lower', 0.0)) / 0.25,
                quality_score=1.0,  # ICT patterns start with high quality
                pattern_tag=candidate.get('tag', source_module),
                source_module=source_module,
                module_subtype=candidate.get('subtype', 'default')
            )

            # Register the zone
            self.fvg_registry[zone_id] = fvg

            self.logger.info(f"ICT_PATTERN_DETECTED module={source_module} dir={fvg.direction} "
                           f"upper={fvg.upper_bound:.2f} lower={fvg.lower_bound:.2f} tag={fvg.pattern_tag}")

        except Exception as e:
            self.logger.error(f"Error registering ICT zone: {e}")

    def _scan_independent_ict_patterns(self, bars_1m, i: int, now_dt: datetime, prof):
        """Scan ICT patterns using independent ICT manager"""
        try:
            tick_size = self.data_cache.tick_size if hasattr(self.data_cache, 'tick_size') else 0.25

            # Convert bars to list format expected by ICT manager
            bars_list = []
            for idx, row in bars_1m.iterrows():
                bar_obj = type('Bar', (), {
                    'high': row['high'],
                    'low': row['low'],
                    'open': row['open'],
                    'close': row['close'],
                    'volume': row['volume']
                })()
                bars_list.append(bar_obj)

            # Use independent ICT manager to evaluate patterns
            ict_candidates = self.ict_manager.evaluate_ict_patterns(bars_list, i, tick_size)

            self.logger.info(f"ICT_CANDIDATES_DEBUG: Found {len(ict_candidates)} candidates at bar {i}")

            # Register any detected ICT patterns as zones
            for candidate in ict_candidates:
                self._register_independent_ict_zone(candidate, candidate.get('tag', 'ict_unknown'), prof, now_dt)
                self.telemetry_counters['ict_independent_zones_created'] += 1

                # Update session stats in ICT manager if trade is taken
                pattern_name = candidate.get('tag', '').replace('ict_', '')
                if pattern_name in self.ict_manager.session_stats:
                    # Note: This would be called from execution manager when trade is actually taken
                    pass

            if ict_candidates:
                self.logger.info(f"Independent ICT: {len(ict_candidates)} patterns detected at bar {i}")

        except Exception as e:
            self.logger.error(f"Error in independent ICT pattern scan: {e}")

    def _register_independent_ict_zone(self, candidate: Dict, pattern_tag: str, prof, timestamp: datetime):
        """Register an independent ICT pattern candidate as an FVG zone"""
        try:
            # Create FVGObject from independent ICT candidate
            zone_id = f"ICT_IND_{pattern_tag}_{len(self.fvg_registry)}"

            # Calculate quality score from ICT candidate
            base_quality = 0.75  # High base quality for ICT patterns
            quality_boost = candidate.get('quality_boost', 0.0)
            final_quality = min(1.0, base_quality + quality_boost)

            fvg = FVGObject(
                zone_id=zone_id,
                direction=candidate.get('dir', 'unknown'),
                upper_bound=candidate.get('upper', 0.0),
                lower_bound=candidate.get('lower', 0.0),
                detection_time=timestamp,
                displacement_bar_range=0.0,  # ICT patterns don't use displacement
                body_fraction=1.0,  # ICT patterns are inherently valid
                volume_multiplier=1.0,  # Not applicable for ICT
                gap_ticks=abs(candidate.get('upper', 0.0) - candidate.get('lower', 0.0)) / 0.25,
                quality_score=final_quality,
                pattern_tag=pattern_tag,
                source_module='ict_independent',
                module_subtype=candidate.get('subtype', 'independent')
            )

            # Apply special properties from ICT candidate
            if candidate.get('fastpath', False):
                fvg.fast_entry = True
                fvg.ttl_override = candidate.get('ttl_override', 30)

            if candidate.get('prefer_62_entry', False):
                fvg.prefer_62_entry = True

            if candidate.get('prefer_edge_entry', False):
                fvg.prefer_edge_entry = True

            # Register the zone
            self.fvg_registry[zone_id] = fvg

            self.logger.info(f"ICT_INDEPENDENT_DETECTED pattern={pattern_tag} dir={fvg.direction} "
                           f"upper={fvg.upper_bound:.2f} lower={fvg.lower_bound:.2f} "
                           f"quality={final_quality:.3f} subtype={fvg.module_subtype}")

        except Exception as e:
            self.logger.error(f"Error registering independent ICT zone: {e}")

    def _calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = bars['high']
        low = bars['low']
        close = bars['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _is_high_vol(self, bars: pd.DataFrame, idx: int) -> bool:
        """Check if market is in high volatility state"""
        try:
            high_vol_cfg = self.config.get('high_vol', {})

            # Calculate ATR ratios
            atr_fast = self._calculate_atr(bars, high_vol_cfg.get('atr_fast', 14))
            atr_slow = self._calculate_atr(bars, high_vol_cfg.get('atr_slow', 50))

            if len(atr_fast) > idx and len(atr_slow) > idx:
                atr_fast_val = atr_fast.iloc[idx]
                atr_slow_val = atr_slow.iloc[idx]

                if atr_slow_val > 0 and not pd.isna(atr_fast_val) and not pd.isna(atr_slow_val):
                    atr_ratio = atr_fast_val / atr_slow_val
                    cond_atr = atr_ratio >= high_vol_cfg.get('atr_ratio', 1.30)
                else:
                    cond_atr = False
            else:
                cond_atr = False

            # Calculate volume ratios
            vol_fast = bars['volume'].rolling(high_vol_cfg.get('vol_fast', 20)).mean()
            vol_slow = bars['volume'].rolling(high_vol_cfg.get('vol_slow', 60)).mean()

            if len(vol_fast) > idx and len(vol_slow) > idx:
                vol_fast_val = vol_fast.iloc[idx]
                vol_slow_val = vol_slow.iloc[idx]

                if vol_slow_val > 0 and not pd.isna(vol_fast_val) and not pd.isna(vol_slow_val):
                    vol_ratio = vol_fast_val / vol_slow_val
                    cond_vol = vol_ratio >= high_vol_cfg.get('vol_ratio', 1.25)
                else:
                    cond_vol = False
            else:
                cond_vol = False

            return bool(cond_atr or cond_vol)

        except Exception as e:
            self.logger.debug(f"High vol check error: {e}")
            return False

    def _to_ticks(self, price: float) -> int:
        """Convert price to integer ticks to avoid float comparison issues"""
        return int(round(price / self.TICK_SIZE))

    def _gap_ticks_bullish(self, bar_before: pd.Series, bar_after: pd.Series) -> int:
        """Calculate bullish gap in ticks (bar_after.low - bar_before.high)"""
        return self._to_ticks(bar_after['low']) - self._to_ticks(bar_before['high'])

    def _gap_ticks_bearish(self, bar_before: pd.Series, bar_after: pd.Series) -> int:
        """Calculate bearish gap in ticks (bar_before.low - bar_after.high)"""
        return self._to_ticks(bar_before['low']) - self._to_ticks(bar_after['high'])

    def _body_gap_ticks_bullish(self, bar_before: pd.Series, bar_after: pd.Series) -> int:
        """Calculate bullish body gap in ticks"""
        body_before = min(bar_before['open'], bar_before['close'])
        body_after = max(bar_after['open'], bar_after['close'])
        return self._to_ticks(body_after) - self._to_ticks(body_before)

    def _body_gap_ticks_bearish(self, bar_before: pd.Series, bar_after: pd.Series) -> int:
        """Calculate bearish body gap in ticks"""
        body_before = max(bar_before['open'], bar_before['close'])
        body_after = min(bar_after['open'], bar_after['close'])
        return self._to_ticks(body_before) - self._to_ticks(body_after)

    def _update_swings(self, bars: pd.DataFrame):
        """Update recent swing highs and lows"""
        if len(bars) < self.swing_lookback:
            return
        
        # Clear old swings
        self.recent_swings = []
        
        # Find swing highs (local maxima)
        for i in range(2, len(bars) - 2):
            if (bars['high'].iloc[i] > bars['high'].iloc[i-1] and 
                bars['high'].iloc[i] > bars['high'].iloc[i-2] and
                bars['high'].iloc[i] > bars['high'].iloc[i+1] and
                bars['high'].iloc[i] > bars['high'].iloc[i+2]):
                self.recent_swings.append({
                    'level': bars['high'].iloc[i],
                    'type': 'high',
                    'bar_idx': i
                })
        
        # Find swing lows (local minima)
        for i in range(2, len(bars) - 2):
            if (bars['low'].iloc[i] < bars['low'].iloc[i-1] and 
                bars['low'].iloc[i] < bars['low'].iloc[i-2] and
                bars['low'].iloc[i] < bars['low'].iloc[i+1] and
                bars['low'].iloc[i] < bars['low'].iloc[i+2]):
                self.recent_swings.append({
                    'level': bars['low'].iloc[i],
                    'type': 'low',
                    'bar_idx': i
                })
    
    def _detect_liquidity_sweep(self, bars: pd.DataFrame, idx: int, direction: str) -> Optional[float]:
        """
        Check if there was a liquidity sweep before the displacement

        Returns:
            Swing level that was swept, or None
        """
        if idx < 3:
            return None

        overshoot_ticks = self.config.get('sweep_min_overshoot_ticks', 1)  # Now 1 tick instead of 2

        for i in range(max(0, idx - 3), idx):
            bar = bars.iloc[i]

            if direction == 'long':
                # Look for wick below recent swing low
                for swing in self.recent_swings:
                    if swing['type'] == 'low' and swing['bar_idx'] < i:
                        swing_ticks = self._to_ticks(swing['level'])
                        bar_low_ticks = self._to_ticks(bar['low'])
                        bar_close_ticks = self._to_ticks(bar['close'])

                        if bar_low_ticks <= swing_ticks - overshoot_ticks:
                            # Wick extended below swing
                            if bar_close_ticks > swing_ticks:
                                # Closed back above - liquidity sweep
                                return swing['level']

            else:  # short
                # Look for wick above recent swing high
                for swing in self.recent_swings:
                    if swing['type'] == 'high' and swing['bar_idx'] < i:
                        swing_ticks = self._to_ticks(swing['level'])
                        bar_high_ticks = self._to_ticks(bar['high'])
                        bar_close_ticks = self._to_ticks(bar['close'])

                        if bar_high_ticks >= swing_ticks + overshoot_ticks:
                            # Wick extended above swing
                            if bar_close_ticks < swing_ticks:
                                # Closed back below - liquidity sweep
                                return swing['level']

        return None
    
    def _detect_fvgs(self, bars: pd.DataFrame, atr: pd.Series, avg_volume: pd.Series):
        """Detect new Fair Value Gaps"""

        # Need at least 3 bars for FVG detection
        if len(bars) < 10:
            return

        # Loop with i as the MIDDLE bar: range(1, len(bars)-1)
        # Check last few bars for new FVGs
        for i in range(max(1, len(bars) - 5), len(bars) - 1):
            
            # Skip if we already detected an FVG at this location
            if self._fvg_exists_at_index(bars, i):
                continue

            # Check for displacement bar first (bar i - the middle bar)
            displacement_bar = bars.iloc[i]
            bar_range = displacement_bar['high'] - displacement_bar['low']
            bar_body = abs(displacement_bar['close'] - displacement_bar['open'])
            
            if bar_range <= 0:
                continue
            
            body_frac = bar_body / bar_range
            
            # Get active profile (responsive or normal)
            prof = self._prof()
            detection_cfg = self.config.get('detection', self.config)
            atr14_value = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 10.0

            # Use profile-based thresholds if available, otherwise fall back to legacy
            if prof is not None:
                # Use profile configuration
                min_range = max(prof.displacement_min_points_floor,
                              prof.displacement_atr_multiple * atr14_value)

                # Check displacement criteria with high vol adjustment
                is_high_vol = self._is_high_vol(bars, i)
                min_body_frac = (prof.displacement_body_frac_min_high_vol if is_high_vol
                                else prof.displacement_body_frac_min_base)
                min_vol_mult = prof.volume_min_mult_trend
            else:
                # Legacy fallback
                if detection_cfg.get('min_displacement_mode') == 'dynamic':
                    dyn_cfg = detection_cfg.get('min_displacement_dyn', {'base_pts': 3.0, 'atr_mult': 0.6})
                    min_range = max(dyn_cfg['base_pts'], dyn_cfg['atr_mult'] * atr14_value)
                else:
                    min_range = max(detection_cfg.get('min_displacement_pts', 4.0),
                                  detection_cfg.get('min_atr_mult', 0.8) * atr14_value)

                is_high_vol = self._is_high_vol(bars, i)
                min_body_frac = (detection_cfg.get('min_body_frac_high_vol', 0.52) if is_high_vol
                                else detection_cfg.get('min_body_frac', 0.60))
                min_vol_mult = detection_cfg.get('min_vol_mult', 1.2)

            if (body_frac < min_body_frac or
                bar_range < min_range or
                displacement_bar['volume'] < min_vol_mult * avg_volume.iloc[i]):
                # Log displacement rejection details with near-miss tracking
                avg_vol_at_i = avg_volume.iloc[i] if not pd.isna(avg_volume.iloc[i]) and avg_volume.iloc[i] > 0 else 1.0
                vol_mult_actual = displacement_bar['volume'] / avg_vol_at_i

                # Track near-miss for body fraction
                if body_frac >= min_body_frac * 0.9:  # Within 10% of threshold
                    self.telemetry_counters['near_miss_body_frac'] += 1

                # Ensure all values are not None before formatting
                body_frac_safe = body_frac if body_frac is not None else 0.0
                min_body_frac_safe = min_body_frac if min_body_frac is not None else 0.0
                bar_range_safe = bar_range if bar_range is not None else 0.0
                min_range_safe = min_range if min_range is not None else 0.0
                vol_mult_actual_safe = vol_mult_actual if vol_mult_actual is not None else 0.0
                min_vol_mult_safe = min_vol_mult if min_vol_mult is not None else 0.0

                self.logger.info(f"FVG_REJECT_DISPLACEMENT body_frac={body_frac_safe:.3f} min={min_body_frac_safe:.3f} "
                                f"bar_range={bar_range_safe:.2f} min={min_range_safe:.2f} vol_mult={vol_mult_actual_safe:.2f} "
                                f"min={min_vol_mult_safe:.2f}")
                continue

            self.telemetry_counters['displacement_pass'] += 1
            
            # Calculate metrics for quality scoring
            atr_mult = bar_range / atr.iloc[i] if atr.iloc[i] > 0 else 0
            vol_mult = displacement_bar['volume'] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 0

            # Check for bullish FVG (gap up) with min gap requirement
            min_gap_ticks = self.config.get('min_gap_ticks', 1)

            # Calculate wick gap using tick helpers (i-1 = before, i+1 = after)
            gap_ticks_bullish = 0  # Initialize to 0
            if i > 0 and i < len(bars) - 1:
                gap_ticks_wick = self._gap_ticks_bullish(bars.iloc[i-1], bars.iloc[i+1])
                gap_ticks_body = self._body_gap_ticks_bullish(bars.iloc[i-1], bars.iloc[i+1])

                # Use body gap as fallback when wick gap is 0
                gap_ticks_bullish = gap_ticks_wick if gap_ticks_wick > 0 else gap_ticks_body
                gap_size_bullish = gap_ticks_bullish * self.TICK_SIZE

            if gap_ticks_bullish >= min_gap_ticks:
                self.telemetry_counters['gap_pass'] += 1

                # Check for liquidity sweep (optional if trend FVGs allowed)
                origin_swing = self._detect_liquidity_sweep(bars, i, 'long')
                allow_trend_fvgs = self.config.get('allow_trend_fvgs', True)

                # Mark as trend FVG if no sweep
                is_trend_fvg = (origin_swing is None)
                pattern_type = 'trend' if is_trend_fvg else 'sweep'

                if is_trend_fvg:
                    self.telemetry_counters['trend_candidates_considered'] += 1
                else:
                    self.telemetry_counters['sweep_candidates_considered'] += 1

                # Apply bypass logic correctly - allow trend FVGs when enabled
                if origin_swing is None and not allow_trend_fvgs:
                    self.logger.info(f"FVG_REJECT_TREND dir=long gap_ticks={gap_ticks_bullish:.1f} body_frac={body_frac:.3f} reason=trend_not_allowed")
                    continue  # Skip if no liquidity sweep and trend FVGs not allowed

                # Create FVG object
                fvg_id = f"FVG_{self.next_id}"
                self.next_id += 1

                # FVG boundaries for bullish gap: top = after bar low, bottom = before bar high
                # For a bullish FVG, the gap is UPWARD, so after bar low > before bar high
                bottom = bars.iloc[i-1]['high']  # Lower boundary
                top = bars.iloc[i+1]['low']       # Upper boundary
                mid = (top + bottom) / 2
                
                # Quality score based on displacement strength (no penalties for trend FVGs)
                quality = (body_frac * 0.3 +
                          min(atr_mult / 2, 1.0) * 0.4 +
                          min(vol_mult / 3, 1.0) * 0.3)

                # Check quality gate using profile or legacy
                if prof is not None:
                    min_quality = prof.quality_score_min_trend
                else:
                    quality_cfg = self.config.get('quality', {})
                    min_quality = quality_cfg.get('min_quality', 0.55)
                if quality < min_quality:
                    # Track near-miss for quality score
                    if quality >= min_quality * 0.9:  # Within 10% of threshold
                        self.telemetry_counters['near_miss_score'] += 1

                    self.logger.info(f"FVG_QUALITY_REJECT dir=long pattern_type={pattern_type} gap_ticks={gap_ticks_bullish:.1f} body_frac={body_frac:.3f} quality={quality:.3f} min={min_quality}")
                    continue

                self.telemetry_counters['score_pass'] += 1
                
                fvg = FVGObject(
                    id=fvg_id,
                    direction='long',
                    created_at=time.time(),
                    top=top,
                    bottom=bottom,
                    mid=mid,
                    quality=quality,
                    status='FRESH',
                    origin_swing=origin_swing,
                    body_frac=body_frac,
                    range_pts=bar_range,
                    vol_mult=vol_mult,
                    atr_mult=atr_mult
                )
                
                self.fvg_registry[fvg_id] = fvg

                # Attach ICT confluence score
                self._attach_ict_score(fvg)

                fvg_type = "TREND_FVG" if is_trend_fvg else "SWEEP_FVG"

                # Get ICT context fields
                ict_bias = getattr(self.ict_context, 'bias_dir', 'neutral') if hasattr(self, 'ict_context') else 'neutral'
                ict_draw_target = getattr(self.ict_context, 'draw_target', None) if hasattr(self, 'ict_context') else None
                ict_premium_discount = getattr(self.ict_context, 'premium_discount', 'neutral') if hasattr(self, 'ict_context') else 'neutral'
                ict_ote_overlap = getattr(self.ict_context, 'ote_overlap', False) if hasattr(self, 'ict_context') else False
                ict_raid_recent = getattr(self.ict_context, 'raid_recent', False) if hasattr(self, 'ict_context') else False
                session = getattr(self.ict_context, 'session_name', 'OTHER') if hasattr(self, 'ict_context') else 'OTHER'

                # Make sure ict_draw_target is safe for formatting
                ict_draw_target_safe = ict_draw_target if ict_draw_target is not None else 'none'

                # Ensure variables are not None before formatting
                top_safe = top if top is not None else 0.0
                bottom_safe = bottom if bottom is not None else 0.0
                gap_ticks_safe = gap_ticks_bullish if gap_ticks_bullish is not None else 0.0
                atr14_safe = atr14_value if atr14_value is not None else 0.0
                min_range_safe = min_range if min_range is not None else 0.0
                bar_range_safe = bar_range if bar_range is not None else 0.0
                body_frac_safe = body_frac if body_frac is not None else 0.0
                vol_mult_safe = vol_mult if vol_mult is not None else 0.0
                quality_safe = quality if quality is not None else 0.0
                ict_score_safe = fvg.ict_score if fvg.ict_score is not None else 0.0

                self.logger.info(f"FVG_DETECTED type={fvg_type} dir=long top={top_safe:.2f} bottom={bottom_safe:.2f} "
                               f"gap_ticks={gap_ticks_safe:.1f} atr14={atr14_safe:.2f} dyn_min_disp={min_range_safe:.2f} "
                               f"bar_range_pts={bar_range_safe:.2f} body_frac={body_frac_safe:.3f} "
                               f"vol_mult={vol_mult_safe:.2f} quality={quality_safe:.3f} high_vol={is_high_vol} "
                               f"ict_score={ict_score_safe:.3f} pattern_tag={fvg.source_module or 'core_fvg'} "
                               f"ict_bias={ict_bias} ict_draw_target={ict_draw_target_safe} "
                               f"ict_premium_discount={ict_premium_discount} "
                               f"ict_ote_overlap={ict_ote_overlap} ict_raid_recent={ict_raid_recent} "
                               f"session={session}")
            else:
                # Track near-miss for gap minimum
                if gap_ticks_bullish >= min_gap_ticks * 0.5:  # Within 50% of threshold
                    self.telemetry_counters['near_miss_gap_min'] += 1

            # Check for bearish FVG (gap down) with min gap requirement
            gap_ticks_bearish = 0  # Initialize to 0
            if i > 0 and i < len(bars) - 1:
                gap_ticks_wick = self._gap_ticks_bearish(bars.iloc[i-1], bars.iloc[i+1])
                gap_ticks_body = self._body_gap_ticks_bearish(bars.iloc[i-1], bars.iloc[i+1])

                # Use body gap as fallback when wick gap is 0
                gap_ticks_bearish = gap_ticks_wick if gap_ticks_wick > 0 else gap_ticks_body
                gap_size_bearish = gap_ticks_bearish * self.TICK_SIZE

            if gap_ticks_bearish >= min_gap_ticks:
                self.telemetry_counters['gap_pass'] += 1

                # Check for liquidity sweep (optional if trend FVGs allowed)
                origin_swing = self._detect_liquidity_sweep(bars, i, 'short')
                allow_trend_fvgs = self.config.get('allow_trend_fvgs', True)

                # Mark as trend FVG if no sweep
                is_trend_fvg = (origin_swing is None)
                pattern_type = 'trend' if is_trend_fvg else 'sweep'

                if is_trend_fvg:
                    self.telemetry_counters['trend_candidates_considered'] += 1
                else:
                    self.telemetry_counters['sweep_candidates_considered'] += 1

                # Apply bypass logic correctly - allow trend FVGs when enabled
                if origin_swing is None and not allow_trend_fvgs:
                    self.logger.info(f"FVG_REJECT_TREND dir=short gap_ticks={gap_ticks_bearish:.1f} body_frac={body_frac:.3f} reason=trend_not_allowed")
                    continue  # Skip if no liquidity sweep and trend FVGs not allowed

                # Create FVG object
                fvg_id = f"FVG_{self.next_id}"
                self.next_id += 1

                # FVG boundaries for bearish gap: top = before bar low, bottom = after bar high
                # For a bearish FVG, the gap is DOWNWARD, so before bar low > after bar high
                top = bars.iloc[i-1]['low']       # Upper boundary
                bottom = bars.iloc[i+1]['high']    # Lower boundary
                mid = (top + bottom) / 2
                
                # Quality score (no penalties for trend FVGs)
                quality = (body_frac * 0.3 +
                          min(atr_mult / 2, 1.0) * 0.4 +
                          min(vol_mult / 3, 1.0) * 0.3)

                # Check quality gate using profile or legacy
                if prof is not None:
                    min_quality = prof.quality_score_min_trend
                else:
                    quality_cfg = self.config.get('quality', {})
                    min_quality = quality_cfg.get('min_quality', 0.55)
                if quality < min_quality:
                    # Track near-miss for quality score
                    if quality >= min_quality * 0.9:  # Within 10% of threshold
                        self.telemetry_counters['near_miss_score'] += 1

                    self.logger.info(f"FVG_QUALITY_REJECT dir=short pattern_type={pattern_type} gap_ticks={gap_ticks_bearish:.1f} body_frac={body_frac:.3f} quality={quality:.3f} min={min_quality}")
                    continue

                self.telemetry_counters['score_pass'] += 1
                
                fvg = FVGObject(
                    id=fvg_id,
                    direction='short',
                    created_at=time.time(),
                    top=top,
                    bottom=bottom,
                    mid=mid,
                    quality=quality,
                    status='FRESH',
                    origin_swing=origin_swing,
                    body_frac=body_frac,
                    range_pts=bar_range,
                    vol_mult=vol_mult,
                    atr_mult=atr_mult
                )
                
                self.fvg_registry[fvg_id] = fvg

                # Attach ICT confluence score
                self._attach_ict_score(fvg)

                fvg_type = "TREND_FVG" if is_trend_fvg else "SWEEP_FVG"

                # Get ICT context fields
                ict_bias = getattr(self.ict_context, 'bias_dir', 'neutral') if hasattr(self, 'ict_context') else 'neutral'
                ict_draw_target = getattr(self.ict_context, 'draw_target', None) if hasattr(self, 'ict_context') else None
                ict_premium_discount = getattr(self.ict_context, 'premium_discount', 'neutral') if hasattr(self, 'ict_context') else 'neutral'
                ict_ote_overlap = getattr(self.ict_context, 'ote_overlap', False) if hasattr(self, 'ict_context') else False
                ict_raid_recent = getattr(self.ict_context, 'raid_recent', False) if hasattr(self, 'ict_context') else False
                session = getattr(self.ict_context, 'session_name', 'OTHER') if hasattr(self, 'ict_context') else 'OTHER'

                # Make sure ict_draw_target is safe for formatting
                ict_draw_target_safe = ict_draw_target if ict_draw_target is not None else 'none'

                # Ensure variables are not None before formatting
                top_safe = top if top is not None else 0.0
                bottom_safe = bottom if bottom is not None else 0.0
                gap_ticks_safe = gap_ticks_bearish if gap_ticks_bearish is not None else 0.0
                atr14_safe = atr14_value if atr14_value is not None else 0.0
                min_range_safe = min_range if min_range is not None else 0.0
                bar_range_safe = bar_range if bar_range is not None else 0.0
                body_frac_safe = body_frac if body_frac is not None else 0.0
                vol_mult_safe = vol_mult if vol_mult is not None else 0.0
                quality_safe = quality if quality is not None else 0.0
                ict_score_safe = fvg.ict_score if fvg.ict_score is not None else 0.0

                self.logger.info(f"FVG_DETECTED type={fvg_type} dir=short top={top_safe:.2f} bottom={bottom_safe:.2f} "
                               f"gap_ticks={gap_ticks_safe:.1f} atr14={atr14_safe:.2f} dyn_min_disp={min_range_safe:.2f} "
                               f"bar_range_pts={bar_range_safe:.2f} body_frac={body_frac_safe:.3f} "
                               f"vol_mult={vol_mult_safe:.2f} quality={quality_safe:.3f} high_vol={is_high_vol} "
                               f"ict_score={ict_score_safe:.3f} pattern_tag={fvg.source_module or 'core_fvg'} "
                               f"ict_bias={ict_bias} ict_draw_target={ict_draw_target_safe} "
                               f"ict_premium_discount={ict_premium_discount} "
                               f"ict_ote_overlap={ict_ote_overlap} ict_raid_recent={ict_raid_recent} "
                               f"session={session}")
            else:
                # Track near-miss for gap minimum
                if gap_ticks_bearish >= min_gap_ticks * 0.5:  # Within 50% of threshold
                    self.telemetry_counters['near_miss_gap_min'] += 1
    
    def _fvg_exists_at_index(self, bars: pd.DataFrame, idx: int) -> bool:
        """Check if we already have an FVG at this bar index"""
        # Simple duplicate prevention - could be enhanced
        for fvg in self.fvg_registry.values():
            if abs(fvg.created_at - time.time()) < 60:  # Recent FVGs
                # Check if levels match (within tolerance)
                if abs(fvg.mid - (bars.iloc[idx]['high'] + bars.iloc[idx]['low']) / 2) < 1:
                    return True
        return False

    def _price_touched_zone(self, price: float, zone: FVGObject, tolerance_frac: float = None) -> bool:
        """Check if price touched the zone via wick with optional tolerance"""
        if tolerance_frac is None:
            tolerance_frac = self.arming_touch_frac

        zone_height = zone.top - zone.bottom
        tolerance = zone_height * tolerance_frac

        # Expand zone boundaries by tolerance
        expanded_bottom = zone.bottom - tolerance
        expanded_top = zone.top + tolerance

        return expanded_bottom <= price <= expanded_top

    def _penetration_pct(self, zone: FVGObject, revisit_low: float, revisit_high: float) -> float:
        """Calculate how much of zone got filled on revisit"""
        if zone.direction == 'long':
            # For bullish zones, calculate penetration from bottom
            depth = min(zone.top, max(zone.bottom, revisit_high)) - zone.bottom
        else:
            # For bearish zones, calculate penetration from top
            depth = zone.top - max(zone.bottom, min(zone.top, revisit_low))

        zone_height = max(self.TICK_SIZE, zone.top - zone.bottom)
        return max(0.0, min(1.0, depth / zone_height))

    def _closed_outside_inner_pct(self, close_price: float, zone: FVGObject, pct: float) -> bool:
        """Check if close is outside the inner percentage of the zone"""
        inner_low = zone.bottom + pct * (zone.top - zone.bottom)
        inner_high = zone.top - pct * (zone.top - zone.bottom)

        # Bullish: want close >= inner_high; bearish: close <= inner_low
        if zone.direction == 'long':
            return close_price >= inner_high
        else:
            return close_price <= inner_low

    def get_micro_rsi_bounds(self, fvg: FVGObject, standard_bounds: dict) -> dict:
        """Get adjusted RSI bounds for micro-FVG if relaxation is enabled"""
        ra = self.cfg.responsive_arming if self.cfg else None

        if not (self.cfg and self.cfg.profile_active == "responsive" and ra and ra.micro_rsi_relax):
            return standard_bounds

        zone_height_ticks = int(round((fvg.top - fvg.bottom) / self.TICK_SIZE))
        is_micro = zone_height_ticks <= ra.micro_fvg_max_ticks

        if not is_micro:
            return standard_bounds

        # Apply RSI relaxation for micro-FVGs
        relax_points = ra.micro_rsi_relax_points
        adjusted_bounds = {}

        for direction, (lo, hi) in standard_bounds.items():
            adjusted_lo = max(0, lo - relax_points)
            adjusted_hi = min(100, hi + relax_points)
            adjusted_bounds[direction] = (adjusted_lo, adjusted_hi)

        self.logger.debug(f"FVG_MICRO_RSI_RELAX id={fvg.id} height={zone_height_ticks}t "
                         f"standard={standard_bounds} adjusted={adjusted_bounds}")

        return adjusted_bounds

    def _update_fvg_states(self, current_price: float, current_bar: pd.Series):
        """Update states of existing FVGs based on price action"""
        
        current_time = time.time()
        
        for fvg_id, fvg in self.fvg_registry.items():
            
            # Skip if already terminal state
            if fvg.status in ['CONSUMED', 'INVALID', 'EXPIRED']:
                continue
            
            # Check timeout
            lifecycle_cfg = self.config.get('lifecycle', self.config)
            arm_timeout = lifecycle_cfg.get('arm_timeout_sec', 600)
            if current_time - fvg.created_at > arm_timeout:
                fvg.status = 'EXPIRED'
                fvg.invalidation_reason = 'timeout'
                if fvg.status == 'FRESH':
                    self.logger.info(f"FVG_NEVER_ARMED id={fvg_id} timeout_sec={arm_timeout}")
                else:
                    self.logger.info(f"FVG_INVALID id={fvg_id} reason=timeout")
                continue
            
            # Calculate zone consumption
            zone_size = fvg.top - fvg.bottom
            if fvg.direction == 'long':
                penetration = max(0, min(current_bar['low'] - fvg.bottom, zone_size))
            else:
                penetration = max(0, min(fvg.top - current_bar['high'], zone_size))
            
            consumption_frac = penetration / zone_size if zone_size > 0 else 0
            
            # Check invalidation conditions
            lifecycle_cfg = self.config.get('lifecycle', self.config)
            if consumption_frac >= lifecycle_cfg.get('invalidate_frac', 0.75):
                fvg.status = 'INVALID'
                fvg.invalidation_reason = 'consumed'
                self._track_invalidation(fvg, 'consumed', current_time)
                self.logger.info(f"FVG_INVALID id={fvg_id} reason=consumed_{consumption_frac:.1%}")
                continue
            
            # Check structure break with grace period
            structure_break_threshold = 2.0  # points
            structure_broken = False

            if fvg.direction == 'long':
                if current_bar['close'] < fvg.bottom - structure_break_threshold:
                    structure_broken = True
                    break_price = current_bar['close']
            else:
                if current_bar['close'] > fvg.top + structure_break_threshold:
                    structure_broken = True
                    break_price = current_bar['close']

            if structure_broken:
                # Initialize or update structure break tracker
                if fvg_id not in self.structure_break_tracker:
                    self.structure_break_tracker[fvg_id] = {
                        'bar_count': 1,
                        'break_price': break_price
                    }
                else:
                    self.structure_break_tracker[fvg_id]['bar_count'] += 1

                # Check if grace period exceeded
                if self.structure_break_tracker[fvg_id]['bar_count'] > self.structure_break_grace_bars:
                    fvg.status = 'INVALID'
                    fvg.invalidation_reason = 'structure_break'
                    self.logger.info(f"FVG_INVALID id={fvg_id} reason=structure_break_grace_exceeded "
                                   f"bars={self.structure_break_tracker[fvg_id]['bar_count']}")
                    # Clean up tracker
                    del self.structure_break_tracker[fvg_id]
                    continue
                else:
                    self.logger.debug(f"FVG_STRUCTURE_BREAK_GRACE id={fvg_id} "
                                    f"bars={self.structure_break_tracker[fvg_id]['bar_count']}/{self.structure_break_grace_bars}")
            else:
                # No structure break, clear tracker if it exists
                if fvg_id in self.structure_break_tracker:
                    del self.structure_break_tracker[fvg_id]
            
            # Check arming conditions (FRESH -> ARMED) with enhanced responsive features
            if fvg.status == 'FRESH':
                prof = self._prof()
                ra = self.cfg.responsive_arming if self.cfg else None

                # Calculate zone height in ticks for micro-FVG detection
                zone_height_ticks = int(round((fvg.top - fvg.bottom) / self.TICK_SIZE))
                is_micro = ra and zone_height_ticks <= ra.micro_fvg_max_ticks

                # Check for defense overfill first
                fill_pct = self._penetration_pct(fvg, current_bar['low'], current_bar['high'])
                defense_limit = self._current_defense_cap()

                if fill_pct > defense_limit:
                    fvg.status = 'INVALID'
                    fvg.invalidation_reason = 'defense_overfill'
                    self._track_invalidation(fvg, 'defense', current_time)
                    self.logger.info(f"FVG_INVALID id={fvg_id} reason=defense_overfill fill_pct={fill_pct:.2f}")
                    continue

                # 1) Fast-arm on wick touch (responsive only)
                if (self.cfg and self.cfg.profile_active == "responsive" and
                    ra and ra.arm_on_wick):

                    # Check if price touched zone via wick with tolerance
                    price_touched = False
                    if fvg.direction == 'long':
                        price_touched = self._price_touched_zone(current_bar['low'], fvg)
                    else:  # short
                        price_touched = self._price_touched_zone(current_bar['high'], fvg)

                    if price_touched:
                        fvg.status = 'ARMED'
                        fvg.armed_at = current_time
                        fvg.last_touch_at = current_time
                        self.telemetry_counters['armed_on_wick'] += 1

                        armed_reason = "wick_touch"
                        if is_micro:
                            armed_reason += f"_micro_{zone_height_ticks}t"

                        self.logger.info(f"FVG_ARMED id={fvg_id} dir={fvg.direction} mid={fvg.mid:.2f} "
                                       f"reason={armed_reason} time={datetime.now().isoformat()}")

                        # Try fast path order
                        self._maybe_fast_order(fvg, current_bar, current_time)
                        continue

                # 2) Close-outside arming (use responsive threshold if available)
                lifecycle_cfg = self.config.get('lifecycle', self.config)

                # Use responsive 10% threshold or fallback to legacy 25%
                if (self.cfg and self.cfg.profile_active == "responsive" and
                    ra and ra.arm_close_outside_pct):
                    inner_pct = ra.arm_close_outside_pct
                else:
                    inner_pct = lifecycle_cfg.get('touch_defend_inner_frac', 0.25)

                if fvg.direction == 'long':
                    # Price wicked into zone (with tolerance)
                    if self._price_touched_zone(current_bar['low'], fvg):
                        # Check close outside inner percentage
                        if self._closed_outside_inner_pct(current_bar['close'], fvg, inner_pct):
                            fvg.status = 'ARMED'
                            fvg.armed_at = current_time
                            fvg.last_touch_at = current_time
                            self.telemetry_counters['armed_on_close'] += 1

                            armed_reason = f"close_outside_{int(inner_pct*100)}pct_tolerance_{int(self.arming_touch_frac*100)}pct"
                            if is_micro:
                                armed_reason += f"_micro_{zone_height_ticks}t"

                            self.logger.info(f"FVG_ARMED id={fvg_id} dir=long mid={fvg.mid:.2f} "
                                           f"reason={armed_reason} time={datetime.now().isoformat()}")

                            # Try fast path order
                            self._maybe_fast_order(fvg, current_bar, current_time)

                else:  # short
                    # Price wicked into zone (with tolerance)
                    if self._price_touched_zone(current_bar['high'], fvg):
                        # Check close outside inner percentage
                        if self._closed_outside_inner_pct(current_bar['close'], fvg, inner_pct):
                            fvg.status = 'ARMED'
                            fvg.armed_at = current_time
                            fvg.last_touch_at = current_time
                            self.telemetry_counters['armed_on_close'] += 1

                            armed_reason = f"close_outside_{int(inner_pct*100)}pct_tolerance_{int(self.arming_touch_frac*100)}pct"
                            if is_micro:
                                armed_reason += f"_micro_{zone_height_ticks}t"

                            self.logger.info(f"FVG_ARMED id={fvg_id} dir=short mid={fvg.mid:.2f} "
                                           f"reason={armed_reason} time={datetime.now().isoformat()}")

                            # Try fast path order
                            self._maybe_fast_order(fvg, current_bar, current_time)

            # Update last touch for ARMED FVGs (with tolerance)
            elif fvg.status == 'ARMED':
                if fvg.direction == 'long':
                    if self._price_touched_zone(current_bar['low'], fvg):
                        fvg.last_touch_at = current_time
                else:
                    if self._price_touched_zone(current_bar['high'], fvg):
                        fvg.last_touch_at = current_time
    
    def _cleanup_registry(self):
        """Remove old expired/invalid FVGs to prevent memory bloat"""
        current_time = time.time()
        max_age = 3600  # Keep for 1 hour max
        
        to_remove = []
        for fvg_id, fvg in self.fvg_registry.items():
            if fvg.status in ['INVALID', 'EXPIRED', 'CONSUMED']:
                if current_time - fvg.created_at > max_age:
                    to_remove.append(fvg_id)
        
        for fvg_id in to_remove:
            del self.fvg_registry[fvg_id]
    
    def get_best_armed(self) -> Optional[FVGObject]:
        """
        Get the best ARMED FVG for trading
        
        Returns:
            Best FVG object or None if no suitable gaps
        """
        armed_fvgs = [fvg for fvg in self.fvg_registry.values() 
                     if fvg.status == 'ARMED']
        
        if not armed_fvgs:
            return None
        
        # Sort by quality (higher is better) and recency (more recent first)
        armed_fvgs.sort(key=lambda x: (x.quality, x.armed_at or 0), reverse=True)
        
        return armed_fvgs[0]
    
    def mark_consumed(self, fvg_id: str):
        """Mark an FVG as consumed after trade entry"""
        if fvg_id in self.fvg_registry:
            self.fvg_registry[fvg_id].status = 'CONSUMED'
            self.logger.info(f"FVG_CONSUMED id={fvg_id}")
    
    def current_ATR14(self) -> float:
        """Get current ATR(14) value from 1m bars"""
        try:
            bars_1m = self.data_cache.get_bars('1m')
            if bars_1m is None or len(bars_1m) < 15:
                return 10.0  # Default fallback
            
            atr = self._calculate_atr(bars_1m, 14)
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 10.0
        except:
            return 10.0
    
    def dynamic_min_displacement(self) -> float:
        """Calculate dynamic minimum displacement based on current ATR"""
        detection_cfg = self.config.get('detection', {})
        if detection_cfg.get('min_displacement_mode') == 'dynamic':
            dyn_cfg = detection_cfg.get('min_displacement_dyn', {'base_pts': 3.0, 'atr_mult': 0.6})
            atr14 = self.current_ATR14()
            # Use max(base, atr_mult * ATR) not sum
            return max(dyn_cfg['base_pts'], dyn_cfg['atr_mult'] * atr14)
        else:
            return detection_cfg.get('min_displacement_pts', 4.0)
    
    def get_entry_signals(self, fvg: FVGObject) -> Dict[str, Any]:
        """
        Generate entry orchestration signals for an FVG

        Returns dict with entry levels and TTL settings
        """
        entry_cfg = self.config.get('entry', {})
        edge_retry_cfg = self.config.get('edge_retry', {})

        # Check if high volatility for entry adjustment
        bars_1m = self.data_cache.get_bars('1m')
        is_high_vol = False
        if bars_1m is not None and len(bars_1m) > 0:
            is_high_vol = self._is_high_vol(bars_1m, len(bars_1m) - 1)

        # Check for micro-FVG front-run entry logic (responsive only)
        prof = self._prof()
        ra = self.cfg.responsive_arming if self.cfg else None
        zone_height_ticks = int(round((fvg.top - fvg.bottom) / self.TICK_SIZE))
        is_micro = ra and zone_height_ticks <= ra.micro_fvg_max_ticks

        if (self.cfg and self.cfg.profile_active == "responsive" and
            is_micro and ra and ra.micro_front_run_ticks):

            # Micro-FVG front-run: enter at edge ± offset
            offset = ra.micro_front_run_ticks * self.TICK_SIZE
            if fvg.direction == 'long':
                entry_level = fvg.bottom + offset  # Enter near bottom edge
            else:
                entry_level = fvg.top - offset     # Enter near top edge

            entry_pct = "micro_edge"  # Special marker for micro entries
            self.telemetry_counters['entry_micro_edge'] += 1

            self.logger.info(f"FVG_MICRO_ENTRY id={fvg.id} height={zone_height_ticks}t "
                           f"entry={entry_level:.2f} offset={offset:.2f}")

        else:
            # Standard entry percentage based on volatility
            entry_pct = (entry_cfg.get('entry_pct_high_vol', 0.62) if is_high_vol
                        else entry_cfg.get('entry_pct_default', 0.50))

            # Calculate standard entry level
            zone_height = fvg.top - fvg.bottom
            if fvg.direction == 'long':
                entry_level = fvg.bottom + (entry_pct * zone_height)
            else:
                entry_level = fvg.top - (entry_pct * zone_height)

        signals = {
            'fvg_id': fvg.id,
            'direction': fvg.direction,
            'quality': fvg.quality,
            'high_vol': is_high_vol,
            'entry_pct': entry_pct
        }

        if entry_cfg.get('use_mid_entry', True):
            signals['mid_entry'] = {
                'level': entry_level,
                'ttl_sec': entry_cfg.get('ttl_sec', 90),
                'cancel_if_runs_ticks': entry_cfg.get('cancel_if_runs_ticks', 8)
            }
            self.logger.info(f"FVG_ENTRY_READY id={fvg.id} level={entry_level:.2f} pct={entry_pct:.0%} "
                           f"high_vol={is_high_vol} ttl={entry_cfg.get('ttl_sec', 90)}")
        
        if entry_cfg.get('use_edge_retry', True) and edge_retry_cfg.get('enable', True):
            edge_offset = entry_cfg.get('edge_offset_ticks', 2) * 0.25  # NQ tick size
            if fvg.direction == 'long':
                edge_level = fvg.top - edge_offset
            else:
                edge_level = fvg.bottom + edge_offset
            
            signals['edge_retry'] = {
                'level': edge_level,
                'ttl_sec': edge_retry_cfg.get('ttl_sec', 45)
            }
            self.logger.info(f"FVG_ENTRY_EDGE_READY id={fvg.id} level={edge_level:.2f} ttl={edge_retry_cfg.get('ttl_sec', 45)}")
        
        return signals