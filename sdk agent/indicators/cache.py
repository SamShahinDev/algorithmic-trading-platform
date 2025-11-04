"""
Indicator caching module with rolling window.

This module provides efficient bar storage and incremental indicator updates:
- Rolling window of 1-minute bars (max 500 bars)
- Thread-safe updates for WebSocket data
- Incremental indicator calculation (no full recalculation)
- Memory-efficient storage using numpy arrays
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


class Bar:
    """Represents a single OHLCV bar."""

    __slots__ = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int
    ):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self):
        return f"Bar({self.timestamp}, O:{self.open}, H:{self.high}, L:{self.low}, C:{self.close}, V:{self.volume})"


class IndicatorCache:
    """
    Thread-safe cache for bars and indicator values.

    Maintains rolling window of bars and incrementally calculated indicators.
    Optimized for real-time WebSocket updates.
    """

    def __init__(self, max_bars: int = 500):
        """
        Initialize indicator cache.

        Args:
            max_bars: Maximum number of bars to retain (default 500)
        """
        self.max_bars = max_bars

        # Bar storage (using deque for efficient append/pop)
        self.bars: deque[Bar] = deque(maxlen=max_bars)

        # Cached indicator values (parallel to bars)
        self.indicators: Dict[str, deque] = {
            'vwap': deque(maxlen=max_bars),
            'vwap_std': deque(maxlen=max_bars),
            'rsi': deque(maxlen=max_bars),
            'ema_20': deque(maxlen=max_bars),
            'ema_50': deque(maxlen=max_bars),
            'macd': deque(maxlen=max_bars),
            'macd_signal': deque(maxlen=max_bars),
            'macd_histogram': deque(maxlen=max_bars),
            'atr': deque(maxlen=max_bars),
            'adx': deque(maxlen=max_bars),
        }

        # Incremental calculation state
        self.state: Dict[str, Any] = {
            # VWAP state
            'vwap_cumulative_tp_volume': 0.0,
            'vwap_cumulative_volume': 0,
            'vwap_squared_sum': 0.0,

            # RSI state
            'rsi_avg_gain': 0.0,
            'rsi_avg_loss': 0.0,
            'rsi_prev_close': None,

            # EMA state
            'ema_20': None,
            'ema_50': None,

            # MACD state
            'macd_ema_12': None,
            'macd_ema_26': None,
            'macd_signal_ema': None,

            # ATR state
            'atr_value': None,

            # ADX state
            'adx_plus_di': None,
            'adx_minus_di': None,
            'adx_value': None,
            'adx_prev_high': None,
            'adx_prev_low': None,
        }

        # Thread lock for concurrent access
        self.lock = threading.RLock()

        logger.info(f"IndicatorCache initialized with max {max_bars} bars")

    def add_bar(
        self,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int
    ) -> None:
        """
        Add new bar and update all indicators incrementally.

        Args:
            timestamp: Bar timestamp
            open: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
        """
        with self.lock:
            # Create bar
            bar = Bar(timestamp, open, high, low, close, volume)

            # Add to bars deque
            self.bars.append(bar)

            # Update indicators incrementally
            self._update_vwap(bar)
            self._update_rsi(bar)
            self._update_ema(bar)
            self._update_macd()
            self._update_atr(bar)
            self._update_adx(bar)

    def _update_vwap(self, bar: Bar) -> None:
        """Update VWAP incrementally."""
        # Typical price
        tp = (bar.high + bar.low + bar.close) / 3.0

        # Update cumulative sums
        self.state['vwap_cumulative_tp_volume'] += tp * bar.volume
        self.state['vwap_cumulative_volume'] += bar.volume

        # Calculate VWAP
        if self.state['vwap_cumulative_volume'] > 0:
            vwap = self.state['vwap_cumulative_tp_volume'] / self.state['vwap_cumulative_volume']
        else:
            vwap = bar.close

        # Update squared sum for standard deviation
        self.state['vwap_squared_sum'] += (tp - vwap) ** 2 * bar.volume

        # Calculate standard deviation
        if self.state['vwap_cumulative_volume'] > 0:
            variance = self.state['vwap_squared_sum'] / self.state['vwap_cumulative_volume']
            vwap_std = np.sqrt(variance)
        else:
            vwap_std = 0.0

        # Store values
        self.indicators['vwap'].append(vwap)
        self.indicators['vwap_std'].append(vwap_std)

    def _update_rsi(self, bar: Bar) -> None:
        """Update RSI incrementally using Wilder's smoothing."""
        period = 14

        if self.state['rsi_prev_close'] is None:
            # First bar - no RSI yet
            self.state['rsi_prev_close'] = bar.close
            self.indicators['rsi'].append(None)
            return

        # Calculate price change
        change = bar.close - self.state['rsi_prev_close']
        gain = max(change, 0)
        loss = max(-change, 0)

        # Initialize or update average gain/loss
        if self.state['rsi_avg_gain'] == 0.0:
            # Need at least 'period' bars to initialize
            if len(self.bars) >= period:
                # Calculate initial averages
                gains = []
                losses = []
                for i in range(len(self.bars) - period + 1, len(self.bars)):
                    if i > 0:
                        prev_bar = self.bars[i - 1]
                        curr_bar = self.bars[i]
                        change = curr_bar.close - prev_bar.close
                        gains.append(max(change, 0))
                        losses.append(max(-change, 0))

                self.state['rsi_avg_gain'] = np.mean(gains)
                self.state['rsi_avg_loss'] = np.mean(losses)
        else:
            # Wilder's smoothing
            self.state['rsi_avg_gain'] = (self.state['rsi_avg_gain'] * (period - 1) + gain) / period
            self.state['rsi_avg_loss'] = (self.state['rsi_avg_loss'] * (period - 1) + loss) / period

        # Calculate RSI
        if self.state['rsi_avg_loss'] == 0:
            rsi = 100.0
        else:
            rs = self.state['rsi_avg_gain'] / self.state['rsi_avg_loss']
            rsi = 100.0 - (100.0 / (1.0 + rs))

        self.indicators['rsi'].append(rsi)
        self.state['rsi_prev_close'] = bar.close

    def _update_ema(self, bar: Bar) -> None:
        """Update EMAs incrementally."""
        # EMA 20
        multiplier_20 = 2.0 / (20 + 1)
        if self.state['ema_20'] is None:
            self.state['ema_20'] = bar.close
        else:
            self.state['ema_20'] = (bar.close - self.state['ema_20']) * multiplier_20 + self.state['ema_20']

        self.indicators['ema_20'].append(self.state['ema_20'])

        # EMA 50
        multiplier_50 = 2.0 / (50 + 1)
        if self.state['ema_50'] is None:
            self.state['ema_50'] = bar.close
        else:
            self.state['ema_50'] = (bar.close - self.state['ema_50']) * multiplier_50 + self.state['ema_50']

        self.indicators['ema_50'].append(self.state['ema_50'])

    def _update_macd(self) -> None:
        """Update MACD incrementally."""
        if len(self.bars) == 0:
            return

        price = self.bars[-1].close

        # EMA 12
        mult_12 = 2.0 / (12 + 1)
        if self.state['macd_ema_12'] is None:
            self.state['macd_ema_12'] = price
        else:
            self.state['macd_ema_12'] = (price - self.state['macd_ema_12']) * mult_12 + self.state['macd_ema_12']

        # EMA 26
        mult_26 = 2.0 / (26 + 1)
        if self.state['macd_ema_26'] is None:
            self.state['macd_ema_26'] = price
        else:
            self.state['macd_ema_26'] = (price - self.state['macd_ema_26']) * mult_26 + self.state['macd_ema_26']

        # MACD line
        macd = self.state['macd_ema_12'] - self.state['macd_ema_26']

        # Signal line (EMA 9 of MACD)
        mult_9 = 2.0 / (9 + 1)
        if self.state['macd_signal_ema'] is None:
            self.state['macd_signal_ema'] = macd
        else:
            self.state['macd_signal_ema'] = (macd - self.state['macd_signal_ema']) * mult_9 + self.state['macd_signal_ema']

        # Histogram
        histogram = macd - self.state['macd_signal_ema']

        self.indicators['macd'].append(macd)
        self.indicators['macd_signal'].append(self.state['macd_signal_ema'])
        self.indicators['macd_histogram'].append(histogram)

    def _update_atr(self, bar: Bar) -> None:
        """Update ATR incrementally."""
        period = 14

        if len(self.bars) < 2:
            self.indicators['atr'].append(None)
            return

        # Get previous bar
        prev_bar = self.bars[-2]

        # Calculate True Range
        high_low = bar.high - bar.low
        high_close = abs(bar.high - prev_bar.close)
        low_close = abs(bar.low - prev_bar.close)
        true_range = max(high_low, high_close, low_close)

        # Initialize or update ATR using Wilder's smoothing
        if self.state['atr_value'] is None:
            if len(self.bars) >= period:
                # Calculate initial ATR
                trs = []
                for i in range(len(self.bars) - period + 1, len(self.bars)):
                    if i > 0:
                        pb = self.bars[i - 1]
                        cb = self.bars[i]
                        hl = cb.high - cb.low
                        hc = abs(cb.high - pb.close)
                        lc = abs(cb.low - pb.close)
                        trs.append(max(hl, hc, lc))
                self.state['atr_value'] = np.mean(trs)
        else:
            # Wilder's smoothing
            self.state['atr_value'] = (self.state['atr_value'] * (period - 1) + true_range) / period

        self.indicators['atr'].append(self.state['atr_value'])

    def _update_adx(self, bar: Bar) -> None:
        """Update ADX incrementally."""
        period = 14

        if len(self.bars) < 2:
            self.indicators['adx'].append(None)
            return

        prev_bar = self.bars[-2]

        # Calculate directional movement
        up_move = bar.high - prev_bar.high
        down_move = prev_bar.low - bar.low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        # True Range (already calculated in ATR)
        high_low = bar.high - bar.low
        high_close = abs(bar.high - prev_bar.close)
        low_close = abs(bar.low - prev_bar.close)
        tr = max(high_low, high_close, low_close)

        # Initialize or update smoothed DM and TR
        if self.state['adx_plus_di'] is None:
            if len(self.bars) >= period:
                # Initial calculation
                self.state['adx_plus_di'] = plus_dm / tr * 100 if tr > 0 else 0
                self.state['adx_minus_di'] = minus_dm / tr * 100 if tr > 0 else 0
                self.state['adx_value'] = 0
        else:
            # Calculate DI
            plus_di = plus_dm / tr * 100 if tr > 0 else 0
            minus_di = minus_dm / tr * 100 if tr > 0 else 0

            # Smooth DI
            self.state['adx_plus_di'] = (self.state['adx_plus_di'] * (period - 1) + plus_di) / period
            self.state['adx_minus_di'] = (self.state['adx_minus_di'] * (period - 1) + minus_di) / period

            # Calculate DX
            di_sum = self.state['adx_plus_di'] + self.state['adx_minus_di']
            if di_sum > 0:
                dx = abs(self.state['adx_plus_di'] - self.state['adx_minus_di']) / di_sum * 100
            else:
                dx = 0

            # Smooth DX to get ADX
            if self.state['adx_value'] == 0:
                self.state['adx_value'] = dx
            else:
                self.state['adx_value'] = (self.state['adx_value'] * (period - 1) + dx) / period

        self.indicators['adx'].append(self.state['adx_value'])

    # Accessor Methods

    def get_latest(self, indicator: str) -> Optional[float]:
        """
        Get latest value for indicator.

        Args:
            indicator: Indicator name

        Returns:
            Latest value or None
        """
        with self.lock:
            if indicator in self.indicators and len(self.indicators[indicator]) > 0:
                return self.indicators[indicator][-1]
            return None

    def get_array(self, indicator: str, count: int = None) -> np.ndarray:
        """
        Get numpy array of indicator values.

        Args:
            indicator: Indicator name
            count: Number of recent values (None for all)

        Returns:
            Numpy array of values
        """
        with self.lock:
            if indicator not in self.indicators:
                return np.array([])

            values = list(self.indicators[indicator])

            if count:
                values = values[-count:]

            # Filter out None values
            values = [v for v in values if v is not None]

            return np.array(values)

    def get_bars_df(self, count: int = None) -> pd.DataFrame:
        """
        Get bars as pandas DataFrame.

        Args:
            count: Number of recent bars (None for all)

        Returns:
            DataFrame with OHLCV data
        """
        with self.lock:
            bars = list(self.bars)

            if count:
                bars = bars[-count:]

            if not bars:
                return pd.DataFrame()

            return pd.DataFrame([
                {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                for bar in bars
            ])

    def get_all_indicators(self) -> Dict[str, Optional[float]]:
        """
        Get latest values for all indicators.

        Returns:
            Dict of indicator name -> latest value
        """
        with self.lock:
            return {
                name: self.get_latest(name)
                for name in self.indicators.keys()
            }

    def bar_count(self) -> int:
        """Get current number of bars in cache."""
        with self.lock:
            return len(self.bars)

    def reset(self) -> None:
        """Clear all bars and indicators."""
        with self.lock:
            self.bars.clear()
            for indicator in self.indicators.values():
                indicator.clear()

            # Reset state
            self.state = {
                'vwap_cumulative_tp_volume': 0.0,
                'vwap_cumulative_volume': 0,
                'vwap_squared_sum': 0.0,
                'rsi_avg_gain': 0.0,
                'rsi_avg_loss': 0.0,
                'rsi_prev_close': None,
                'ema_20': None,
                'ema_50': None,
                'macd_ema_12': None,
                'macd_ema_26': None,
                'macd_signal_ema': None,
                'atr_value': None,
                'adx_plus_di': None,
                'adx_minus_di': None,
                'adx_value': None,
                'adx_prev_high': None,
                'adx_prev_low': None,
            }

            logger.info("IndicatorCache reset")
