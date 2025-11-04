"""
Independent ICT Pattern Manager

Provides standalone ICT pattern evaluation without FVG dependency.
Designed for 24/7 operation with session-aware optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util


class ICTPatternManager:
    """Independent ICT pattern manager for 24/7 evaluation"""

    def __init__(self, cfg, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.ict_modules = {}
        self.session_stats = {
            'silver_bullet': {'count': 0, 'session_start': None},
            'micro_scalp': {'count': 0, 'session_start': None},
            'liquidity_ob': {'count': 0, 'session_start': None},
            'breaker_unicorn': {'count': 0, 'session_start': None}
        }

        # Load ICT modules
        self._load_ict_modules()

    def _load_ict_modules(self):
        """Load all available ICT pattern modules"""
        try:
            # Import ICT modules
            from .patterns.modules import ict_silver_bullet, ict_micro_scalp

            self.ict_modules = {
                'silver_bullet': ict_silver_bullet,
                'micro_scalp': ict_micro_scalp
            }

            # Try to load additional modules if available
            try:
                from .patterns.modules import ict_liquidity_ob
                self.ict_modules['liquidity_ob'] = ict_liquidity_ob
            except ImportError:
                pass

            try:
                from .patterns.modules import ict_breaker_unicorn
                self.ict_modules['breaker_unicorn'] = ict_breaker_unicorn
            except ImportError:
                pass

            self.logger.info(f"Loaded ICT modules: {list(self.ict_modules.keys())}")

        except Exception as e:
            self.logger.error(f"Failed to load ICT modules: {e}")
            self.ict_modules = {}

    def evaluate_ict_patterns(self, bars_1m: List[Any], i: int,
                            tick: float = 0.25) -> List[Dict[str, Any]]:
        """
        Evaluate all ICT patterns independently

        Args:
            bars_1m: 1-minute bar data
            i: Current bar index
            tick: Tick size for calculations

        Returns:
            List of ICT pattern candidates
        """
        candidates = []

        if not self.ict_modules or i < 5:
            return candidates

        # Get current Chicago time
        ct_now = self._get_chicago_time()

        # Create ICT context for bias and session info
        ict_context = self._build_ict_context(bars_1m, i, ct_now)

        # Evaluate each pattern module
        for pattern_name, module in self.ict_modules.items():
            try:
                pattern_candidates = self._evaluate_pattern_module(
                    module, pattern_name, bars_1m, i, ict_context, tick, ct_now
                )

                if pattern_candidates:
                    # Apply session optimization
                    optimized_candidates = self._apply_session_optimization(
                        pattern_candidates, pattern_name, ct_now
                    )
                    candidates.extend(optimized_candidates)

            except Exception as e:
                self.logger.debug(f"Error evaluating {pattern_name}: {e}")

        return candidates

    def _evaluate_pattern_module(self, module, pattern_name: str, bars_1m: List[Any],
                               i: int, ict_context: Dict[str, Any], tick: float,
                               ct_now: datetime) -> List[Dict[str, Any]]:
        """Evaluate a specific ICT pattern module"""
        try:
            # Check session usage limits
            session_count = self._get_session_count(pattern_name, ct_now)

            if pattern_name == 'silver_bullet':
                candidates = module.generate(bars_1m, i, self.cfg, tick, ct_now)
                self.logger.info(f"ICT_PATTERN_DEBUG: {pattern_name} returned {len(candidates)} candidates at bar {i}")
                return candidates

            elif pattern_name == 'micro_scalp':
                candidates = module.generate(bars_1m, i, self.cfg, ict_context, tick, session_count)
                self.logger.info(f"ICT_PATTERN_DEBUG: {pattern_name} returned {len(candidates)} candidates at bar {i}")
                return candidates

            elif pattern_name == 'liquidity_ob':
                candidates = module.generate(bars_1m, i, self.cfg, ict_context, tick)
                self.logger.info(f"ICT_PATTERN_DEBUG: {pattern_name} returned {len(candidates)} candidates at bar {i}")
                return candidates

            elif pattern_name == 'breaker_unicorn':
                candidates = module.generate(bars_1m, i, self.cfg, ict_context, tick)
                self.logger.info(f"ICT_PATTERN_DEBUG: {pattern_name} returned {len(candidates)} candidates at bar {i}")
                return candidates

        except Exception as e:
            self.logger.debug(f"Module {pattern_name} evaluation error: {e}")

        return []

    def _build_ict_context(self, bars_1m: List[Any], i: int, ct_now: datetime) -> Dict[str, Any]:
        """Build ICT context for pattern evaluation"""
        try:
            # Simple bias detection based on recent price action
            bias_dir = self._detect_bias_direction(bars_1m, i)

            # Session killzone detection (simplified for 24/7 operation)
            session_killzone = self._is_session_killzone(ct_now)

            return {
                'bias_dir': bias_dir,
                'session_killzone': session_killzone,
                'chicago_time': ct_now
            }

        except Exception:
            return {
                'bias_dir': 'neutral',
                'session_killzone': True,  # Default to active for 24/7 operation
                'chicago_time': ct_now
            }

    def _detect_bias_direction(self, bars_1m: List[Any], i: int) -> str:
        """Enhanced bias detection using configuration parameters"""
        try:
            params = self.cfg.ict_params
            lookback = params.bias_lookback_bars
            threshold = params.bias_slope_threshold

            if i < lookback:
                return 'neutral'

            # Look at configured lookback period moving average slope
            recent_closes = [bars_1m[j].close for j in range(i-lookback+1, i+1)]
            half_period = lookback // 2
            ma_start = sum(recent_closes[:half_period]) / half_period
            ma_end = sum(recent_closes[-half_period:]) / half_period

            if ma_end > ma_start + threshold:
                return 'long'
            elif ma_end < ma_start - threshold:
                return 'short'
            else:
                return 'neutral'

        except Exception:
            return 'neutral'

    def _is_session_killzone(self, ct_now: datetime) -> bool:
        """Check if current time is within high-activity session"""
        try:
            hour = ct_now.hour

            # High-activity periods (simplified)
            killzone_hours = [
                (2, 5),   # London session
                (8, 11),  # NY morning
                (13, 16)  # NY afternoon
            ]

            for start_h, end_h in killzone_hours:
                if start_h <= hour < end_h:
                    return True

            return False

        except Exception:
            return True  # Default to active

    def _apply_session_optimization(self, candidates: List[Dict[str, Any]],
                                  pattern_name: str, ct_now: datetime) -> List[Dict[str, Any]]:
        """Apply session-specific optimization to candidates"""
        optimized = []

        for candidate in candidates:
            try:
                # Apply session-based quality adjustments
                session_boost = self._get_session_quality_boost(ct_now)

                if 'quality_boost' in candidate:
                    candidate['quality_boost'] += session_boost
                else:
                    candidate['quality_boost'] = session_boost

                # Apply pattern-specific session optimization
                if pattern_name == 'silver_bullet':
                    candidate = self._optimize_silver_bullet_session(candidate, ct_now)
                elif pattern_name == 'micro_scalp':
                    candidate = self._optimize_micro_scalp_session(candidate, ct_now)

                optimized.append(candidate)

            except Exception as e:
                self.logger.debug(f"Session optimization error: {e}")
                optimized.append(candidate)  # Keep original if optimization fails

        return optimized

    def _get_session_quality_boost(self, ct_now: datetime) -> float:
        """Get quality boost based on current session using configuration"""
        try:
            hour = ct_now.hour
            params = self.cfg.ict_params

            # London session (2-5 CT) - highest quality
            if 2 <= hour < 5:
                return params.london_quality_boost
            # NY morning (8-11 CT) - high quality
            elif 8 <= hour < 11:
                return params.ny_morning_quality_boost
            # NY afternoon (13-16 CT) - good quality
            elif 13 <= hour < 16:
                return params.ny_afternoon_quality_boost
            # Off-hours - neutral
            else:
                return 0.0

        except Exception:
            return 0.0

    def _optimize_silver_bullet_session(self, candidate: Dict[str, Any],
                                      ct_now: datetime) -> Dict[str, Any]:
        """Apply Silver Bullet specific session optimization"""
        try:
            hour = ct_now.hour

            # Prefer London session for Silver Bullet
            if 2 <= hour < 5:
                candidate['quality_boost'] = candidate.get('quality_boost', 0) + 0.10
                candidate['session_premium'] = 'LONDON'
            elif 8 <= hour < 11:
                candidate['session_premium'] = 'NY_MORNING'
            elif 13 <= hour < 16:
                candidate['session_premium'] = 'NY_AFTERNOON'

        except Exception:
            pass

        return candidate

    def _optimize_micro_scalp_session(self, candidate: Dict[str, Any],
                                    ct_now: datetime) -> Dict[str, Any]:
        """Apply Micro Scalp specific session optimization using configuration"""
        try:
            hour = ct_now.hour
            params = self.cfg.ict_params
            base_ttl = params.micro_fastpath_ttl_s

            # Apply session-specific TTL adjustments
            if 2 <= hour < 5:  # London session
                ttl_multiplier = params.london_ttl_multiplier
                candidate['quality_boost'] = candidate.get('quality_boost', 0) + 0.03
            elif 8 <= hour < 11 or 13 <= hour < 16:  # NY active sessions
                ttl_multiplier = params.ny_active_ttl_multiplier
                candidate['quality_boost'] = candidate.get('quality_boost', 0) + 0.05
            else:  # Off-hours
                ttl_multiplier = params.off_hours_ttl_multiplier

            candidate['ttl_override'] = int(base_ttl * ttl_multiplier)

            # Apply zone size optimization
            if 'zone_size_ticks' in candidate:
                if 2 <= hour < 5:  # London - allow larger zones
                    max_zone_multiplier = params.london_max_zone_multiplier
                elif 8 <= hour < 11 or 13 <= hour < 16:  # NY active - standard
                    max_zone_multiplier = params.ny_active_max_zone_multiplier
                else:  # Off-hours - prefer smaller zones
                    max_zone_multiplier = params.off_hours_max_zone_multiplier

                candidate['max_zone_multiplier'] = max_zone_multiplier

        except Exception:
            pass

        return candidate

    def _get_session_count(self, pattern_name: str, ct_now: datetime) -> int:
        """Get current session trade count for pattern"""
        try:
            session_data = self.session_stats.get(pattern_name, {})
            session_start = session_data.get('session_start')

            # Reset session if new day
            if not session_start or session_start.date() != ct_now.date():
                self.session_stats[pattern_name] = {
                    'count': 0,
                    'session_start': ct_now
                }
                return 0

            return session_data.get('count', 0)

        except Exception:
            return 0

    def increment_session_count(self, pattern_name: str):
        """Increment session trade count for pattern"""
        try:
            if pattern_name in self.session_stats:
                self.session_stats[pattern_name]['count'] += 1
        except Exception:
            pass

    def _get_chicago_time(self) -> datetime:
        """Get current Chicago time"""
        try:
            return datetime.now(ZoneInfo("America/Chicago"))
        except Exception:
            # Fallback to UTC if timezone fails
            return datetime.now()

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get current pattern statistics"""
        return {
            'loaded_modules': list(self.ict_modules.keys()),
            'session_stats': self.session_stats.copy(),
            'chicago_time': self._get_chicago_time().isoformat()
        }