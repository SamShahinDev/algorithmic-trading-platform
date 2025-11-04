"""
DataCache - Incremental market data caching with UTC timezone handling

Maintains rolling window of market data with automatic updates every 3 seconds.
All timestamps stored in UTC internally, converted to CT only at decision boundaries.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Literal, Any, Union
import logging
import asyncio
import warnings
import talib
import random
from async_tools import await_if_coro

logger = logging.getLogger(__name__)
CT = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")

# Freshness watchdog constants
FRESHNESS_LIMIT_S = 90

# Rate limit backoff
BACKOFF_DELAYS = [1500, 3000, 5000, 10000]  # milliseconds


class DataCache:
    """
    Efficient data cache with incremental updates
    Reduces API calls by ~95% through smart caching
    """
    
    def __init__(self, broker: Any, contract_id: str, is_live: bool = False, logger=None):
        """Initialize DataCache with broker reference"""
        self.broker = broker
        self.contract_id = contract_id
        self._is_live = is_live   # False for practice (sim), True for live
        self.symbol = 'NQ'
        self.logger = logger or logging.getLogger(__name__)
        
        # Async lock for poll protection
        self._poll_lock = asyncio.Lock()
        self._backoff_index = 0
        self._is_owner = True  # This instance owns polling
        self._last_stale_log = None
        
        # Data storage - use bars_ prefix for consistency
        self.bars_1m = pd.DataFrame(columns=["open","high","low","close","volume"])
        self.bars_5m = pd.DataFrame(columns=["open","high","low","close","volume"])
        self.bars_1h = pd.DataFrame(columns=["open","high","low","close","volume"])
        
        # Legacy aliases for backward compatibility
        self.data_1m = self.bars_1m
        self.data_5m = self.bars_5m
        self.data_1h = self.bars_1h
        
        # Cache state
        self.last_update: Optional[datetime] = None
        self.initial_fetch_done = False
        
        # Performance tracking
        self.performance_stats = {
            'total_updates': 0,
            'incremental_updates': 0,
            'full_refreshes': 0,
            'cache_efficiency': 0.0
        }
        
        # Pre-computed indicators
        self.indicators: Dict[str, float] = {}
        
        # Update task
        self.update_task = None
        self._running = False
        
    @staticmethod
    def _ensure_utc_index(df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
        """
        Return a copy with a strictly UTC, monotonic DatetimeIndex.
        Handles all edge cases: naive, aware non-UTC, duplicates, unsorted.
        """
        if df.empty:
            return df
            
        out = df.copy()
        
        # Get the time data
        if time_col and time_col in out.columns:
            idx = pd.to_datetime(out[time_col], utc=True, errors="coerce")
            out = out.drop(columns=[time_col])
        else:
            # Try to parse index as datetime
            try:
                idx = pd.to_datetime(out.index, utc=True, errors="coerce")
            except:
                # If index isn't datetime-like, look for a timestamp column
                for col in ['timestamp', 'time', 'datetime', 'date']:
                    if col in out.columns:
                        idx = pd.to_datetime(out[col], utc=True, errors="coerce")
                        out = out.drop(columns=[col])
                        break
        
        # Handle timezone awareness
        if hasattr(idx, 'tz'):
            if idx.tz is not None and str(idx.tz) != "UTC":
                idx = idx.tz_convert("UTC")
            elif idx.tz is None:
                idx = idx.tz_localize("UTC")
        
        # Set index and clean
        out.index = idx
        out = out[~out.index.isna()]  # Remove NaT values
        out = out[~out.index.duplicated(keep="last")].sort_index()
        
        # Verify monotonic
        if not out.index.is_monotonic_increasing:
            logger.warning("Index not monotonic after cleaning, sorting again")
            out = out.sort_index()
        
        return out
    
    @staticmethod
    def _resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample in UTC to avoid DST edge cases. Right-closed/right-labeled bars.
        """
        if df_1m.empty:
            return pd.DataFrame()
            
        agg = {
            "open": "first",
            "high": "max", 
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        
        resampled = (
            df_1m
            .resample(rule, label="right", closed="right", origin="epoch")
            .agg(agg)
            .dropna(how="any")
        )
        
        return resampled
    
    def warmup(self, lookback_1m: int = 200):
        """Warmup cache with initial data"""
        try:
            # Import ATR baseline config
            try:
                from pattern_config import ATR_BASELINE_LOOKBACK_BARS, ATR_BASELINE_MIN_FALLBACK
            except ImportError:
                from ..pattern_config import ATR_BASELINE_LOOKBACK_BARS, ATR_BASELINE_MIN_FALLBACK
            
            # Determine actual lookback to request
            requested_lookback = max(lookback_1m, ATR_BASELINE_LOOKBACK_BARS)
            
            # Use asyncio to call the async retrieve_bars method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an event loop, use create_task
                task = asyncio.create_task(self.broker.retrieve_bars(
                    contract_id=self.contract_id,
                    start=None,
                    unit=2, unit_number=1, limit=requested_lookback,
                    include_partial=True, live=self._is_live
                ))
                # This is a workaround for the sync interface
                df = pd.DataFrame()  # Will be updated in update_incremental
            else:
                df = loop.run_until_complete(self.broker.retrieve_bars(
                    contract_id=self.contract_id,
                    start=None,
                    unit=2, unit_number=1, limit=requested_lookback,
                    include_partial=True, live=self._is_live
                ))
                
                self.bars_1m = df.tail(requested_lookback) if not df.empty else df
                self._assert_utc_index()
                self._rebuild_higher_tfs()
                
                # Log ATR baseline status
                if not df.empty:
                    actual_bars = len(self.bars_1m)
                    if actual_bars >= ATR_BASELINE_LOOKBACK_BARS:
                        atr_mode = "full_day"
                    elif actual_bars >= ATR_BASELINE_MIN_FALLBACK:
                        atr_mode = f"provisional_{actual_bars}"
                    else:
                        atr_mode = f"provisional_{ATR_BASELINE_MIN_FALLBACK}"
                    
                    last_ts = self.bars_1m.index[-1].strftime("%Y-%m-%d %H:%M:%S") + "Z" if len(self.bars_1m) > 0 else "unknown"
                    self.logger.info(f"ATR_BASELINE init_bars={actual_bars} mode={atr_mode} last_1m={last_ts}")
                
            self.logger.info("Cache warmup: last=%s 1m=%d",
                           str(self.bars_1m.index[-1]) if len(self.bars_1m) else "N/A",
                           len(self.bars_1m))
        except Exception as e:
            self.logger.error(f"Warmup failed: {e}")
    
    def update_incremental(self):
        """Update cache with incremental data"""
        try:
            # Use 90-second overlap for reliability  
            last = self.bars_1m.index[-1] if len(self.bars_1m) else None
            start = last - pd.Timedelta(seconds=90) if last is not None else None
            
            # Force practice feed (live=False)
            new_df = asyncio.run(self.broker.retrieve_bars(
                contract_id=self.contract_id,
                start=start,
                unit=2, unit_number=1,
                limit=200,
                include_partial=True,
                live=False  # PRACTICE FEED ENFORCED
            ))
            
            if new_df.empty:
                return
            
            # Merge and deduplicate 
            merged = pd.concat([self.bars_1m, new_df]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            before = len(self.bars_1m)
            self.bars_1m = merged.tail(200)
            added = len(self.bars_1m) - before
            
            # Update legacy aliases
            self.data_1m = self.bars_1m
            
            self._rebuild_higher_tfs()
            
            # Update stats
            self.performance_stats['total_updates'] += 1
            if added > 0:
                self.performance_stats['incremental_updates'] += 1
            
            # Enhanced logging with freshness check
            if not self.bars_1m.empty:
                last_ts = self.bars_1m.index[-1]
                age_s = (datetime.now(UTC) - last_ts).total_seconds()
                
                from ..pattern_config import TRACE
                if TRACE.get('data', False):
                    self.logger.info(f"DATA_OK last_1m={last_ts.strftime('%Y-%m-%d %H:%M:%S')}Z age_s={age_s:.0f}")
                else:
                    self.logger.info(f"DATA_OK last_1m={last_ts.strftime('%Y-%m-%d %H:%M:%S')}Z added={max(0, added)}")
            
            # Check freshness
            self._check_freshness()
                
        except Exception as e:
            self.logger.error(f"Incremental update failed: {e}")
    
    def _check_freshness(self):
        """Check data freshness and re-warmup if stale"""
        if not len(self.bars_1m):
            return
        
        last_ts = self.bars_1m.index[-1]
        age_s = (datetime.now(UTC) - last_ts).total_seconds()
        
        if age_s > FRESHNESS_LIMIT_S:
            # Throttle stale logging to once per minute
            now = datetime.now(UTC)
            if self._last_stale_log is None or (now - self._last_stale_log).total_seconds() > 60:
                self.logger.warning(f"DATA_STALE age_s={age_s:.0f} rewarm=True")
                self._last_stale_log = now
            
            # Do a one-shot warmup to re-sync
            try:
                df = asyncio.run(self.broker.retrieve_bars(
                    contract_id=self.contract_id, 
                    start=None, 
                    unit=2, unit_number=1, 
                    limit=200, 
                    include_partial=True,
                    live=False  # PRACTICE FEED
                ))
                if not df.empty:
                    self.bars_1m = df.tail(200)
                    self.data_1m = self.bars_1m
                    self._rebuild_higher_tfs()
                    self.logger.info("Data freshness restored after re-warmup")
            except Exception as e:
                self.logger.error(f"Re-warmup failed: {e}")
    
    async def _check_freshness_async(self):
        """Async version of freshness check"""
        if not len(self.bars_1m):
            return
        
        last_ts = self.bars_1m.index[-1]
        age_s = (datetime.now(UTC) - last_ts).total_seconds()
        
        if age_s > FRESHNESS_LIMIT_S:
            # Throttle stale logging to once per minute
            now = datetime.now(UTC)
            if self._last_stale_log is None or (now - self._last_stale_log).total_seconds() > 60:
                self.logger.warning(f"DATA_STALE age_s={age_s:.0f} rewarm=True")
                self._last_stale_log = now
            
            # Do a one-shot warmup to re-sync
            try:
                df = await self.broker.retrieve_bars(
                    contract_id=self.contract_id, 
                    start=None, 
                    unit=2, unit_number=1, 
                    limit=200, 
                    include_partial=True,
                    live=False  # PRACTICE FEED
                )
                if not df.empty:
                    self.bars_1m = df.tail(200)
                    self.data_1m = self.bars_1m
                    self._rebuild_higher_tfs()
                    self.logger.info("Data freshness restored after re-warmup")
            except Exception as e:
                self.logger.error(f"Re-warmup failed: {e}")
    
    async def _async_update_incremental(self, start):
        """Async version of incremental update with 429 handling"""
        async with self._poll_lock:  # Prevent concurrent polls
            try:
                # Add includePartialBar and 90s overlap
                actual_start = (start or (datetime.now(UTC) - timedelta(minutes=4))) - timedelta(seconds=90)
                
                new_df = await self.broker.retrieve_bars(
                    contract_id=self.contract_id,
                    start=actual_start,
                    unit=2, unit_number=1, limit=200,
                    include_partial=True,  # Critical for incremental updates
                    live=False  # PRACTICE FEED
                )
                
                # Reset backoff on success
                self._backoff_index = 0
                
                if new_df.empty:
                    return
                
                # Merge and deduplicate
                merged = pd.concat([self.bars_1m, new_df]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                before = len(self.bars_1m)
                self.bars_1m = merged.tail(200)
                added = len(self.bars_1m) - before
                
                # Update legacy aliases
                self.data_1m = self.bars_1m
                
                self._rebuild_higher_tfs()
                
                # Update stats
                self.performance_stats['total_updates'] += 1
                if added > 0:
                    self.performance_stats['incremental_updates'] += 1
                    
                # Enhanced logging with freshness check
                if not self.bars_1m.empty:
                    last_ts = self.bars_1m.index[-1]
                    age_s = (datetime.now(UTC) - last_ts).total_seconds()
                    self.logger.info(f"DATA_OK last_1m={last_ts.strftime('%Y-%m-%d %H:%M:%S')}Z added={added}")
                
                # Check freshness
                await self._check_freshness_async()
                
            except Exception as e:
                if "429" in str(e):
                    # Handle rate limit with exponential backoff
                    delay_ms = BACKOFF_DELAYS[min(self._backoff_index, len(BACKOFF_DELAYS)-1)]
                    jitter = random.uniform(0.8, 1.2)  # ±20% jitter
                    delay_ms = int(delay_ms * jitter)
                    self.logger.info(f"DATA_BACKOFF code=429 delay_ms={delay_ms}")
                    await asyncio.sleep(delay_ms / 1000.0)
                    self._backoff_index = min(self._backoff_index + 1, len(BACKOFF_DELAYS) - 1)
                else:
                    self.logger.error(f"Async incremental update failed: {e}")
    
    def _rebuild_higher_tfs(self):
        """Rebuild higher timeframes from 1m data"""
        if self.bars_1m.empty:
            return
            
        agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        
        # 5m bars
        self.bars_5m = (self.bars_1m.resample("5min", label="right", closed="right", origin="epoch")
                       .agg(agg).dropna(how="any"))
        
        # 1h bars  
        self.bars_1h = (self.bars_1m.resample("1h", label="right", closed="right", origin="epoch")
                       .agg(agg).dropna(how="any"))
        
        # Update legacy aliases
        self.data_5m = self.bars_5m
        self.data_1h = self.bars_1h
    
    def _assert_utc_index(self):
        """Assert that all dataframes have proper UTC indexes"""
        for name, df in [("bars_1m", self.bars_1m), ("bars_5m", self.bars_5m), ("bars_1h", self.bars_1h)]:
            if not df.empty:
                idx = getattr(df, "index", None)
                assert idx is not None and hasattr(idx, 'tz') and idx.tz is not None and str(idx.tz) == "UTC", \
                    f"DataCache: {name} index must be UTC tz-aware"

    async def warmup_old(self, lookback_1m: int = 200) -> bool:
        """Initial data load with proper UTC handling"""
        try:
            self.logger.info(f"DataCache warming up with {lookback_1m} bars...")
            
            # Fetch from broker
            raw_data = await self.broker.get_historical_data(
                symbol=self.symbol,
                interval='1m',
                bars=lookback_1m
            )
            
            if raw_data is None or raw_data.empty:
                self.logger.error("No data received during warmup")
                return False
            
            # Ensure UTC index
            self.data_1m = self._ensure_utc_index(raw_data)
            
            # Verify UTC
            if self.data_1m.empty:
                self.logger.error("No valid data after UTC conversion")
                return False
                
            # Build higher timeframes
            self._rebuild_all_timeframes()
            
            # Calculate indicators
            self._update_indicators()
            
            # Update state
            self.last_update = datetime.now(UTC)
            self.initial_fetch_done = True
            self.performance_stats['full_refreshes'] += 1
            self.performance_stats['total_updates'] += 1
            
            # Log status
            last_utc = self.data_1m.index[-1]
            last_ct = self.to_ct(last_utc)
            
            self.logger.info(
                f"Cache ready: last={last_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
                f"last_ct={last_ct.strftime('%Y-%m-%d %H:%M:%S CT')} | "
                f"1m={len(self.data_1m)} 5m={len(self.data_5m)} 1h={len(self.data_1h)}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Warmup failed: {e}")
            return False
    
    async def warmup(self, lookback_1m: int = 200) -> bool:
        """Async warmup - fetch initial data"""
        try:
            # Import ATR baseline config
            try:
                from pattern_config import ATR_BASELINE_LOOKBACK_BARS, ATR_BASELINE_MIN_FALLBACK
            except ImportError:
                from ..pattern_config import ATR_BASELINE_LOOKBACK_BARS, ATR_BASELINE_MIN_FALLBACK
            
            # Determine actual lookback to request
            requested_lookback = max(lookback_1m, ATR_BASELINE_LOOKBACK_BARS)
            
            # Fetch data asynchronously
            df = await self.broker.retrieve_bars(
                contract_id=self.contract_id,
                start=None,
                unit=2, unit_number=1, limit=requested_lookback,
                include_partial=True, live=self._is_live
            )
            
            if not df.empty:
                self.bars_1m = df.tail(requested_lookback)
                self._assert_utc_index()
                self._rebuild_higher_tfs()
                
                # Update state
                self.initial_fetch_done = True
                
                # Log status
                actual_bars = len(self.bars_1m)
                if actual_bars >= ATR_BASELINE_LOOKBACK_BARS:
                    atr_mode = "full_day"
                elif actual_bars >= ATR_BASELINE_MIN_FALLBACK:
                    atr_mode = f"provisional_{actual_bars}"
                else:
                    atr_mode = f"provisional_{ATR_BASELINE_MIN_FALLBACK}"
                
                last_ts = self.bars_1m.index[-1].strftime("%Y-%m-%d %H:%M:%S") + "Z" if len(self.bars_1m) > 0 else "unknown"
                self.logger.info(f"ATR_BASELINE init_bars={actual_bars} mode={atr_mode} last_1m={last_ts}")
            
            self.logger.info("Cache warmup: last=%s 1m=%d",
                           str(self.bars_1m.index[-1]) if len(self.bars_1m) else "N/A",
                           len(self.bars_1m))
            return True
        except Exception as e:
            self.logger.error(f"Async warmup failed: {e}")
            return False
    
    async def initialize(self) -> bool:
        """Alias for warmup for compatibility"""
        return await self.warmup()
    
    async def update_incremental(self) -> bool:
        """Update with only new bars using UTC throughout"""
        try:
            if self.data_1m.empty:
                return await self.warmup()
            
            # Get the last timestamp we have
            last_time = self.data_1m.index[-1]
            
            # Calculate how many bars we might have missed
            now_utc = datetime.now(UTC)
            time_diff = (now_utc - last_time).total_seconds()
            bars_to_fetch = min(int(time_diff / 60) + 5, 50)  # Fetch up to 50 bars
            
            # Fetch recent bars
            raw_new = await self.broker.get_historical_data(
                symbol=self.symbol,
                interval='1m',
                bars=bars_to_fetch
            )
            
            if raw_new is None or raw_new.empty:
                return False
            
            # Ensure UTC index
            new_data = self._ensure_utc_index(raw_new)
            
            # Filter only truly new bars
            new_bars = new_data[new_data.index > last_time]
            
            if new_bars.empty:
                return True  # No new data, but that's OK
            
            # Append new bars
            self.data_1m = pd.concat([self.data_1m, new_bars])
            self.data_1m = self.data_1m[~self.data_1m.index.duplicated(keep="last")].sort_index()
            
            # Keep rolling window (last 1000 bars for performance)
            if len(self.data_1m) > 1000:
                self.data_1m = self.data_1m.iloc[-1000:]
            
            # Rebuild timeframes and indicators
            self._rebuild_all_timeframes()
            self._update_indicators()
            
            # Update stats
            self.last_update = datetime.now(UTC)
            self.performance_stats['incremental_updates'] += 1
            self.performance_stats['total_updates'] += 1
            self._update_cache_efficiency()
            
            self.logger.debug(f"Incremental update: {len(new_bars)} new bars")
            return True
            
        except Exception as e:
            self.logger.error(f"Incremental update failed: {e}")
            return False
    
    async def update(self) -> bool:
        """Main update method - calls incremental update"""
        if not self.initial_fetch_done:
            return await self.warmup()
        return await self.update_incremental()
    
    def _rebuild_all_timeframes(self) -> None:
        """Rebuild 5m and 1h from 1m data in UTC"""
        try:
            self.data_5m = self._resample_ohlcv(self.data_1m, "5min")
            self.data_1h = self._resample_ohlcv(self.data_1m, "1H")
        except Exception as e:
            self.logger.error(f"Error building timeframes: {e}")
    
    def get_bars(self, timeframe: Literal['1m', '5m', '1h'], limit: Optional[int] = None) -> pd.DataFrame:
        """Get bars for specified timeframe with optional limit"""
        mapping = {
            '1m': self.data_1m,
            '5m': self.data_5m,
            '1h': self.data_1h
        }
        bars = mapping.get(timeframe, pd.DataFrame())
        if limit and not bars.empty:
            return bars.tail(limit)
        return bars
    
    @staticmethod
    def to_ct(ts_utc: pd.Timestamp) -> pd.Timestamp:
        """Convert UTC timestamp to CT for display/decisions"""
        if hasattr(ts_utc, 'tz'):
            if ts_utc.tz is None:
                ts_utc = ts_utc.tz_localize("UTC")
            return ts_utc.tz_convert(CT)
        else:
            # Handle datetime objects
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.replace(tzinfo=UTC)
            return ts_utc.astimezone(CT)
    
    @staticmethod
    def session_id(ts_utc: pd.Timestamp) -> pd.Timestamp:
        """Get CME session ID (5pm CT boundaries) in UTC"""
        ct = DataCache.to_ct(ts_utc)
        # CME session starts at 17:00 CT
        session_start = ct.normalize() + pd.Timedelta(hours=17)
        if ct.hour < 17:
            session_start -= pd.Timedelta(days=1)
        # Return in UTC for consistent storage
        return session_start.tz_convert("UTC")
    
    def _update_indicators(self) -> None:
        """Update pre-computed indicators using UTC data"""
        if len(self.data_1m) < 30:
            return
        
        try:
            # Use float64 for TA-Lib compatibility
            close = self.data_1m['close'].values.astype(np.float64)
            high = self.data_1m['high'].values.astype(np.float64)
            low = self.data_1m['low'].values.astype(np.float64)
            volume = self.data_1m['volume'].values.astype(np.float64)
            
            # Calculate indicators
            self.indicators = {
                'atr': talib.ATR(high, low, close, timeperiod=14)[-1],
                'adx': talib.ADX(high, low, close, timeperiod=14)[-1],
                'rsi': talib.RSI(close, timeperiod=14)[-1],
                'ma_20': talib.SMA(close, timeperiod=20)[-1],
                'ma_50': talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else None,
                'ma_200': talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else None,
                'volume_ma': talib.SMA(volume, timeperiod=20)[-1],
                'last_update_utc': self.data_1m.index[-1],
                'last_update_ct': self.to_ct(self.data_1m.index[-1]),
                'last_price': close[-1]
            }
            
            # Calculate Stochastic
            k, d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            self.indicators['stoch_k'] = k[-1] if len(k) > 0 and not np.isnan(k[-1]) else None
            self.indicators['stoch_d'] = d[-1] if len(d) > 0 and not np.isnan(d[-1]) else None
            
            # Calculate VWAP for current session
            session_data = self._get_current_session_data()
            if not session_data.empty:
                typical = (session_data['high'] + session_data['low'] + session_data['close']) / 3
                cumvol = session_data['volume'].cumsum()
                cumtypvol = (typical * session_data['volume']).cumsum()
                vwap = cumtypvol / cumvol
                self.indicators['vwap'] = vwap.iloc[-1] if len(vwap) > 0 else None
                
                # Session high/low
                self.indicators['session_high'] = session_data['high'].max()
                self.indicators['session_low'] = session_data['low'].min()
                
        except Exception as e:
            self.logger.error(f"Error updating indicators: {e}")
    
    def _get_current_session_data(self) -> pd.DataFrame:
        """Get data for current CME session"""
        if self.data_1m.empty:
            return pd.DataFrame()
        
        try:
            current_session = self.session_id(self.data_1m.index[-1])
            # Filter data from session start
            mask = self.data_1m.index >= current_session
            return self.data_1m[mask]
        except Exception as e:
            self.logger.error(f"Error getting session data: {e}")
            return pd.DataFrame()
    
    def _update_cache_efficiency(self) -> None:
        """Calculate cache efficiency metric"""
        total = self.performance_stats['total_updates']
        if total > 0:
            incremental = self.performance_stats['incremental_updates']
            self.performance_stats['cache_efficiency'] = incremental / total * 100
    
    async def start_auto_update(self, interval_seconds: int = 3) -> None:
        """Start automatic cache updates"""
        self._running = True
        
        async def update_loop():
            while self._running:
                try:
                    await asyncio.sleep(interval_seconds)
                    if self._running:  # Check again after sleep
                        # Call the async update method which handles the logic
                        await self.update()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self._running:  # Only log if we're still supposed to be running
                        self.logger.error(f"Auto-update error: {e}")
        
        self.update_task = asyncio.create_task(update_loop())
        self.logger.info(f"DataCache auto-update started ({interval_seconds}s intervals)")
    
    def stop_auto_update(self) -> None:
        """Stop automatic updates"""
        self._running = False
        if self.update_task:
            self.update_task.cancel()
            self.update_task = None
            self.logger.info("DataCache auto-update stopped")
    
    def get_indicator(
        self,
        name: str,
        tf_or_period: Union[str, int, None] = None,  # backward compatibility
        *,
        tf: str = "1m",
        n: Optional[int] = None,
        last: bool = True,
    ) -> Union[float, pd.Series]:
        """
        Backward-compatible indicator accessor.
        
        Supported calls:
          get_indicator("atr", 14)            # legacy period
          get_indicator("atr", "1m")          # legacy timeframe
          get_indicator("atr")                # default (tf="1m", period from defaults)
          get_indicator("atr", tf="5m", n=14) # explicit new style
          get_indicator("atr", last=False)    # return full Series
        
        Args:
            name: Indicator name (atr, adx, rsi, etc.)
            tf_or_period: Legacy positional arg (deprecated)
            tf: Timeframe ("1m", "5m", "1h")
            n: Period for indicator calculation
            last: If True, return last value as float; if False, return Series
            
        Returns:
            float if last=True, pd.Series if last=False
        """
        # Allowed timeframes
        ALLOWED_TF = {"1m", "5m", "1h"}
        
        # Default periods for indicators
        DEFAULT_PERIODS = {
            'atr': 14,
            'adx': 14,
            'rsi': 14,
            'ma': 20,  # for ma_20, ma_50, ma_200 we'll parse the number
            'stoch': 14,
            'volume_ma': 20
        }
        
        # Handle legacy positional argument
        if tf_or_period is not None:
            if isinstance(tf_or_period, int):
                warnings.warn(
                    f"get_indicator('{name}', {tf_or_period}): Positional period is deprecated. "
                    f"Use get_indicator('{name}', n={tf_or_period}) instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                n = n or tf_or_period
            elif isinstance(tf_or_period, str):
                # Check if it's a valid timeframe
                if tf_or_period in ALLOWED_TF:
                    warnings.warn(
                        f"get_indicator('{name}', '{tf_or_period}'): Positional timeframe is deprecated. "
                        f"Use get_indicator('{name}', tf='{tf_or_period}') instead.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    tf = tf_or_period
                else:
                    # Might be something else, ignore it
                    pass
        
        # Validate timeframe
        if tf not in ALLOWED_TF:
            raise ValueError(f"Unsupported timeframe '{tf}'. Must be one of {ALLOWED_TF}")
        
        # Normalize indicator name
        indicator_lower = name.lower().strip()
        
        # Determine period if not specified
        if n is None:
            # Check for indicators with period in name (ma_20, ma_50, etc.)
            if indicator_lower.startswith('ma_'):
                try:
                    n = int(indicator_lower.split('_')[1])
                    indicator_lower = 'ma'
                except (IndexError, ValueError):
                    n = DEFAULT_PERIODS.get('ma', 20)
            else:
                n = DEFAULT_PERIODS.get(indicator_lower, 14)
        
        # For simple indicators in the pre-computed dict, return directly
        if tf == "1m" and last and indicator_lower in self.indicators:
            value = self.indicators[indicator_lower]
            if value is not None:
                return float(value)
            else:
                return np.nan
        
        # For timeframe-specific or series requests, compute on demand
        data = self.get_bars(tf)
        if data.empty:
            return np.nan if last else pd.Series(dtype=float)
        
        # Compute the indicator
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Calculate based on indicator type
        if indicator_lower == 'atr':
            result = talib.ATR(high, low, close, timeperiod=n)
        elif indicator_lower == 'adx':
            result = talib.ADX(high, low, close, timeperiod=n)
        elif indicator_lower == 'rsi':
            result = talib.RSI(close, timeperiod=n)
        elif indicator_lower in ['ma', 'sma']:
            result = talib.SMA(close, timeperiod=n)
        elif indicator_lower == 'ema':
            result = talib.EMA(close, timeperiod=n)
        elif indicator_lower == 'stoch_k':
            result, _ = talib.STOCH(high, low, close, fastk_period=n)
        elif indicator_lower == 'stoch_d':
            _, result = talib.STOCH(high, low, close, fastk_period=n)
        elif indicator_lower == 'volume_ma':
            result = talib.SMA(volume, timeperiod=n)
        elif indicator_lower == 'vwap':
            # VWAP is session-based, return from pre-computed
            if tf == "1m" and 'vwap' in self.indicators:
                if last:
                    return float(self.indicators['vwap'])
                else:
                    # Build series aligned to data index
                    return pd.Series(self.indicators['vwap'], index=data.index[-1:])
            else:
                return np.nan if last else pd.Series(dtype=float)
        else:
            # Try to get from pre-computed indicators
            if indicator_lower in self.indicators:
                if last:
                    value = self.indicators[indicator_lower]
                    return float(value) if value is not None else np.nan
                else:
                    # Return as series
                    return pd.Series(self.indicators[indicator_lower], index=data.index[-1:])
            else:
                # Check for ma_XX patterns
                if indicator_lower.startswith('ma_'):
                    return self.indicators.get(name.lower(), np.nan if last else pd.Series(dtype=float))
                raise ValueError(f"Unknown indicator: {name}")
        
        # Return result
        if last:
            # Return last non-NaN value
            if isinstance(result, pd.Series):
                valid_values = result.dropna()
                return float(valid_values.iloc[-1]) if len(valid_values) > 0 else np.nan
            else:
                # numpy array
                valid_mask = ~np.isnan(result)
                valid_values = result[valid_mask]
                return float(valid_values[-1]) if len(valid_values) > 0 else np.nan
        else:
            # Return as Series with proper index
            return pd.Series(result, index=data.index)
    
    # Compatibility methods for existing code
    def get_adx(self) -> Optional[float]:
        """Get current ADX value"""
        return self.indicators.get('adx')
    
    def get_rsi(self) -> Optional[float]:
        """Get current RSI value"""
        return self.indicators.get('rsi')
    
    def get_atr(self) -> Optional[float]:
        """Get current ATR value"""
        return self.indicators.get('atr')
    
    def get_stoch_k(self) -> Optional[float]:
        """Get current Stochastic K value"""
        return self.indicators.get('stoch_k')
    
    def get_stoch_d(self) -> Optional[float]:
        """Get current Stochastic D value"""
        return self.indicators.get('stoch_d')
    
    def get_current_price(self) -> Optional[float]:
        """Get the current/last price"""
        if not self.data_1m.empty:
            return float(self.data_1m['close'].iloc[-1])
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.performance_stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            **self.performance_stats,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'bars_cached': {
                '1m': len(self.data_1m),
                '5m': len(self.data_5m),
                '1h': len(self.data_1h)
            },
            'indicators': len(self.indicators)
        }