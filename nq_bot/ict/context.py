"""
ICT Context Service

Provides market structure analysis including:
- Market bias and draw on liquidity
- Premium/discount analysis relative to dealing range
- Recent raid detection
- OTE (Optimal Trade Entry) windows
- Session/killzone identification
- Optional SMT (Smart Money Technique) divergence detection
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MarketStructure:
    """Current market structure state"""
    dealing_range_high: float
    dealing_range_low: float
    dealing_range_mid: float
    bias_dir: str  # "long", "short", "neutral"
    draw_target: Optional[float]
    is_premium: bool
    is_discount: bool
    last_impulse_start: Optional[float]
    last_impulse_end: Optional[float]


class ICTContext:
    """
    ICT Context Analysis Service

    Analyzes market structure, bias, and provides confluence scoring
    for trading decisions based on ICT concepts.
    """

    def __init__(self, feed, cfg, logger):
        """
        Initialize ICT context service

        Args:
            feed: Market data feed
            cfg: ICTContextConfig configuration
            logger: Logger instance
        """
        self.feed = feed
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        # Current market structure state
        self.structure = None
        self.last_refresh = None

        # Liquidity levels tracking
        self.pdh = None  # Previous day high
        self.pdl = None  # Previous day low
        self.week_high = None
        self.week_low = None
        self.midnight_open = None
        self.ny_open_830 = None

        # Recent raid tracking
        self.recent_raids = []
        self.raid_recent = False

        # Session state
        self.session_killzone = False

        # Optional SMT state
        self.smt_support = None

        self.logger.info("ICT Context service initialized")

    def refresh(self, now_dt: datetime) -> None:
        """
        Refresh market structure analysis

        Args:
            now_dt: Current datetime
        """
        try:
            # Skip if not enough time elapsed
            if (self.last_refresh and
                (now_dt - self.last_refresh).total_seconds() < self.cfg.refresh_seconds):
                return

            self.last_refresh = now_dt

            # Get current market data
            bars = self._get_market_data()
            if bars is None or len(bars) < 10:  # Reduced minimum for off-hours
                self.logger.debug(f"Limited market data for ICT analysis: {len(bars) if bars is not None else 0} bars")
                return

            # Update key liquidity levels
            self._update_liquidity_levels(bars, now_dt)

            # Analyze market structure and bias
            self._analyze_market_structure(bars)

            # Check for recent raids
            self._check_recent_raids(bars)

            # Update session state
            self._update_session_state(now_dt)

            # Optional SMT analysis
            if self.cfg.use_smt:
                self._analyze_smt()

            # Enhanced diagnostic logging
            confluence_data = self.get_confluence_data()
            diagnostic_info = (
                f"ICT_REFRESH bias={self.bias_dir} premium={self.is_premium} "
                f"discount={self.is_discount} location={confluence_data['location_context']} "
                f"price_pos={confluence_data['price_position_pct']:.1f}% "
                f"raid_recent={self.raid_recent} killzone={self.session_killzone} "
                f"session={confluence_data['session_name']} "
                f"pdh={self.pdh:.2f if self.pdh else 'N/A'} "
                f"pdl={self.pdl:.2f if self.pdl else 'N/A'} "
                f"dealing_range={self.structure.dealing_range_low:.2f if self.structure else 'N/A'}-"
                f"{self.structure.dealing_range_high:.2f if self.structure else 'N/A'}"
            )
            self.logger.info(diagnostic_info)

        except Exception as e:
            self.logger.error(f"Error in ICT context refresh: {e}")

    def _get_market_data(self) -> Optional[List]:
        """Get market data bars for analysis"""
        try:
            # Get enough bars for structure analysis
            lookback = max(200, self.cfg.raid_lookback_bars * 2)

            if hasattr(self.feed, 'get_bars'):
                return self.feed.get_bars(lookback)
            elif hasattr(self.feed, 'bars'):
                return self.feed.bars[-lookback:] if self.feed.bars else None
            else:
                self.logger.warning("Cannot access market data from feed")
                return None

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def _update_liquidity_levels(self, bars: List, now_dt: datetime) -> None:
        """Update key liquidity levels (PDH/PDL, weekly H/L, opens)"""
        try:
            if bars is None or len(bars) == 0:
                return

            # Get high/low arrays
            highs = np.array([bar.high for bar in bars[-100:]])  # Last 100 bars
            lows = np.array([bar.low for bar in bars[-100:]])

            if self.cfg.track_pdh_pdl:
                # Simple approximation - use yesterday's range
                # In production, you'd want proper session handling
                daily_bars = bars[-1440:] if len(bars) >= 1440 else bars  # ~24h
                if daily_bars:
                    self.pdh = max(bar.high for bar in daily_bars[:-60])  # Exclude last hour
                    self.pdl = min(bar.low for bar in daily_bars[:-60])

            if self.cfg.track_week_high_low:
                # Use recent 5-day range as approximation
                weekly_bars = bars[-7200:] if len(bars) >= 7200 else bars  # ~5 days
                if weekly_bars:
                    self.week_high = max(bar.high for bar in weekly_bars)
                    self.week_low = min(bar.low for bar in weekly_bars)

            # Track session opens (simplified)
            if self.cfg.track_midnight_open and bars:
                self.midnight_open = bars[-1].open  # Simplified

            if self.cfg.track_830_open and bars:
                self.ny_open_830 = bars[-1].open  # Simplified

        except Exception as e:
            self.logger.error(f"Error updating liquidity levels: {e}")

    def _analyze_market_structure(self, bars: List) -> None:
        """Analyze current market structure and bias"""
        try:
            if len(bars) < 50:
                return

            # Get recent price data
            recent_bars = bars[-50:]  # Last 50 bars for structure
            highs = [bar.high for bar in recent_bars]
            lows = [bar.low for bar in recent_bars]
            closes = [bar.close for bar in recent_bars]

            current_price = closes[-1]

            # Identify dealing range (simplified swing analysis)
            # In production, you'd want proper swing high/low detection
            recent_high = max(highs[-20:])  # Recent 20-bar high
            recent_low = min(lows[-20:])    # Recent 20-bar low

            dealing_range_mid = (recent_high + recent_low) / 2

            # Determine bias based on price action relative to structure
            bias_dir = "neutral"
            draw_target = None

            # Simple bias logic - can be enhanced
            if current_price > dealing_range_mid * 1.002:  # Above mid + small buffer
                bias_dir = "long"
                draw_target = recent_high if self.pdh else recent_high
            elif current_price < dealing_range_mid * 0.998:  # Below mid - small buffer
                bias_dir = "short"
                draw_target = recent_low if self.pdl else recent_low

            # Enhanced premium/discount detection with configurable thresholds
            range_size = recent_high - recent_low
            if range_size > 0:
                price_position = (current_price - recent_low) / range_size

                # Use configurable thresholds or defaults
                premium_threshold = getattr(self.cfg, 'premium_threshold', 0.618)  # 61.8% (Golden ratio)
                discount_threshold = getattr(self.cfg, 'discount_threshold', 0.382)  # 38.2% (Golden ratio)

                is_premium = price_position > premium_threshold
                is_discount = price_position < discount_threshold

                # Additional location context
                location_context = "equilibrium"  # Between premium and discount
                if is_premium:
                    location_context = "premium"
                elif is_discount:
                    location_context = "discount"

                self.location_context = location_context
                self.price_position_pct = price_position * 100

            else:
                is_premium = False
                is_discount = False
                self.location_context = "equilibrium"
                self.price_position_pct = 50.0

            # Find last impulse for OTE calculation (simplified)
            last_impulse_start = None
            last_impulse_end = None

            # Look for significant moves in recent bars
            for i in range(len(recent_bars) - 5, 0, -1):
                if i >= 5:
                    move_size = abs(closes[i] - closes[i-5])
                    if move_size > range_size * 0.3:  # Significant move
                        last_impulse_start = closes[i-5]
                        last_impulse_end = closes[i]
                        break

            # Update structure
            self.structure = MarketStructure(
                dealing_range_high=recent_high,
                dealing_range_low=recent_low,
                dealing_range_mid=dealing_range_mid,
                bias_dir=bias_dir,
                draw_target=draw_target,
                is_premium=is_premium,
                is_discount=is_discount,
                last_impulse_start=last_impulse_start,
                last_impulse_end=last_impulse_end
            )

        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")

    def _check_recent_raids(self, bars: List) -> None:
        """Check for recent liquidity raids with enhanced detection"""
        try:
            if not bars or len(bars) < self.cfg.raid_lookback_bars:
                self.raid_recent = False
                return

            # Look for raids in recent bars with enhanced tolerance
            lookback_bars = bars[-self.cfg.raid_lookback_bars:]
            raid_detected = False
            raid_tolerance_ticks = getattr(self.cfg, 'raid_tolerance_ticks', 2)  # 2 tick tolerance
            tick_size = 0.25

            # Check for raids on key levels with enhanced logic
            for i, bar in enumerate(lookback_bars[-10:]):  # Last 10 bars for better detection
                try:
                    bar_high = getattr(bar, 'high', bar.get('high') if hasattr(bar, 'get') else bar['high'])
                    bar_low = getattr(bar, 'low', bar.get('low') if hasattr(bar, 'get') else bar['low'])
                    bar_close = getattr(bar, 'close', bar.get('close') if hasattr(bar, 'close') else bar['close'])
                except (AttributeError, KeyError, TypeError):
                    continue

                # Check PDH/PDL raids with tolerance
                if self.pdh:
                    raid_threshold_high = self.pdh + (raid_tolerance_ticks * tick_size)
                    if bar_high > raid_threshold_high and bar_close < self.pdh:
                        raid_detected = True
                        self.logger.info(f"ICT_RAID_DETECTED type=PDH_sweep level={self.pdh:.2f} "
                                       f"wick_high={bar_high:.2f} close={bar_close:.2f}")
                        break

                if self.pdl:
                    raid_threshold_low = self.pdl - (raid_tolerance_ticks * tick_size)
                    if bar_low < raid_threshold_low and bar_close > self.pdl:
                        raid_detected = True
                        self.logger.info(f"ICT_RAID_DETECTED type=PDL_sweep level={self.pdl:.2f} "
                                       f"wick_low={bar_low:.2f} close={bar_close:.2f}")
                        break

                # Check weekly high/low raids with tolerance
                if self.week_high:
                    raid_threshold_high = self.week_high + (raid_tolerance_ticks * tick_size)
                    if bar_high > raid_threshold_high and bar_close < self.week_high:
                        raid_detected = True
                        self.logger.info(f"ICT_RAID_DETECTED type=WEEK_HIGH_sweep level={self.week_high:.2f} "
                                       f"wick_high={bar_high:.2f} close={bar_close:.2f}")
                        break

                if self.week_low:
                    raid_threshold_low = self.week_low - (raid_tolerance_ticks * tick_size)
                    if bar_low < raid_threshold_low and bar_close > self.week_low:
                        raid_detected = True
                        self.logger.info(f"ICT_RAID_DETECTED type=WEEK_LOW_sweep level={self.week_low:.2f} "
                                       f"wick_low={bar_low:.2f} close={bar_close:.2f}")
                        break

                # Check midnight open raids (if configured)
                if self.midnight_open and getattr(self.cfg, 'track_midnight_raids', True):
                    midnight_tolerance = (raid_tolerance_ticks * tick_size)
                    if (abs(bar_high - self.midnight_open) < midnight_tolerance or
                        abs(bar_low - self.midnight_open) < midnight_tolerance):
                        if ((bar_high > self.midnight_open and bar_close < self.midnight_open) or
                            (bar_low < self.midnight_open and bar_close > self.midnight_open)):
                            raid_detected = True
                            self.logger.info(f"ICT_RAID_DETECTED type=MIDNIGHT_OPEN_sweep level={self.midnight_open:.2f} "
                                           f"wick_range={bar_low:.2f}-{bar_high:.2f} close={bar_close:.2f}")
                            break

            self.raid_recent = raid_detected

        except Exception as e:
            self.logger.error(f"Error checking raids: {e}")
            self.raid_recent = False

    def _update_session_state(self, now_dt: datetime) -> None:
        """Update session/killzone state with enhanced detection"""
        try:
            # Enhanced session detection with configurable times
            hour = now_dt.hour
            minute = now_dt.minute

            # Get configurable killzone times
            london_start = getattr(self.cfg, 'london_killzone_start', 2)  # 2 AM NY
            london_end = getattr(self.cfg, 'london_killzone_end', 5)      # 5 AM NY
            ny_am_start = getattr(self.cfg, 'ny_am_killzone_start', 8.5)  # 8:30 AM NY
            ny_am_end = getattr(self.cfg, 'ny_am_killzone_end', 11)       # 11 AM NY
            ny_pm_start = getattr(self.cfg, 'ny_pm_killzone_start', 13.5) # 1:30 PM NY
            ny_pm_end = getattr(self.cfg, 'ny_pm_killzone_end', 16)       # 4 PM NY

            # Convert hour.minute to decimal
            current_time = hour + (minute / 60.0)

            # Check killzone windows
            london_killzone = london_start <= current_time <= london_end
            ny_am_killzone = ny_am_start <= current_time <= ny_am_end
            ny_pm_killzone = ny_pm_start <= current_time <= ny_pm_end

            self.session_killzone = london_killzone or ny_am_killzone or ny_pm_killzone

            # Track current session for better context
            if london_killzone:
                self.current_session = "LONDON"
            elif ny_am_killzone:
                self.current_session = "NY_AM"
            elif ny_pm_killzone:
                self.current_session = "NY_PM"
            elif 0 <= hour <= 6:
                self.current_session = "ASIAN"
            else:
                self.current_session = "OTHER"

            # Log session changes
            if not hasattr(self, '_last_session') or self._last_session != self.current_session:
                self.logger.info(f"ICT_SESSION_CHANGE from={getattr(self, '_last_session', 'UNKNOWN')} "
                               f"to={self.current_session} killzone={self.session_killzone}")
                self._last_session = self.current_session

        except Exception as e:
            self.logger.error(f"Error updating session state: {e}")
            self.session_killzone = False
            self.current_session = "OTHER"

    def _analyze_smt(self) -> None:
        """Analyze Smart Money Technique (SMT) divergence (placeholder)"""
        try:
            # Placeholder for SMT analysis
            # Would compare current symbol with cfg.smt_symbol (e.g., ES vs NQ)
            # For now, set neutral
            self.smt_support = None

        except Exception as e:
            self.logger.error(f"Error in SMT analysis: {e}")
            self.smt_support = None

    def ote_overlap(self, zone) -> bool:
        """
        Check if zone overlaps with OTE (Optimal Trade Entry) window

        Args:
            zone: FVG zone object with top/bottom attributes

        Returns:
            True if zone overlaps with OTE range
        """
        try:
            if not self.structure or not self.structure.last_impulse_start or not self.structure.last_impulse_end:
                return False

            # Calculate OTE range (62-79% of last impulse)
            impulse_start = self.structure.last_impulse_start
            impulse_end = self.structure.last_impulse_end

            if impulse_start == impulse_end:
                return False

            # Determine direction and calculate OTE levels
            if impulse_end > impulse_start:  # Bullish impulse
                impulse_size = impulse_end - impulse_start
                ote_low = impulse_end - (impulse_size * self.cfg.ote_range[1])   # 79% retrace
                ote_high = impulse_end - (impulse_size * self.cfg.ote_range[0])  # 62% retrace
            else:  # Bearish impulse
                impulse_size = impulse_start - impulse_end
                ote_low = impulse_end + (impulse_size * self.cfg.ote_range[0])   # 62% retrace
                ote_high = impulse_end + (impulse_size * self.cfg.ote_range[1])  # 79% retrace

            # Check overlap with zone
            zone_top = getattr(zone, 'top', getattr(zone, 'high', None))
            zone_bottom = getattr(zone, 'bottom', getattr(zone, 'low', None))

            if zone_top is None or zone_bottom is None:
                return False

            # Check for overlap
            return not (zone_bottom > max(ote_low, ote_high) or zone_top < min(ote_low, ote_high))

        except Exception as e:
            self.logger.error(f"Error checking OTE overlap: {e}")
            return False

    # Property accessors for easy access
    @property
    def bias_dir(self) -> str:
        """Current market bias direction"""
        return self.structure.bias_dir if self.structure else "neutral"

    @property
    def draw_target(self) -> Optional[float]:
        """Current draw on liquidity target"""
        return self.structure.draw_target if self.structure else None

    @property
    def is_premium(self) -> bool:
        """Is price in premium range"""
        return self.structure.is_premium if self.structure else False

    @property
    def is_discount(self) -> bool:
        """Is price in discount range"""
        return self.structure.is_discount if self.structure else False

    @property
    def location_context_str(self) -> str:
        """Current location context (premium/discount/equilibrium)"""
        return getattr(self, 'location_context', 'equilibrium')

    @property
    def price_position_percentage(self) -> float:
        """Price position as percentage within dealing range"""
        return getattr(self, 'price_position_pct', 50.0)

    @property
    def session_name(self) -> str:
        """Current session name"""
        return getattr(self, 'current_session', 'OTHER')

    def get_confluence_data(self) -> dict:
        """Get all confluence data for scoring"""
        return {
            'bias_dir': self.bias_dir,
            'is_premium': self.is_premium,
            'is_discount': self.is_discount,
            'location_context': self.location_context_str,
            'price_position_pct': self.price_position_percentage,
            'raid_recent': self.raid_recent,
            'session_killzone': self.session_killzone,
            'session_name': self.session_name,
            'draw_target': self.draw_target,
            'smt_support': self.smt_support
        }