"""
Data Fetcher for NQ Futures
Fetches historical and real-time data for Nasdaq futures
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from utils.logger import setup_logger

class DataFetcher:
    """
    Fetches and manages market data for NQ futures
    """
    
    def __init__(self):
        """Initialize data fetcher"""
        self.logger = setup_logger('DataFetcher')
        self.symbol = 'NQ=F'  # Nasdaq futures symbol for yfinance
        self.cache_dir = Path('data/historical')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.data = None
        self.last_update = None
        
        self.logger.info("📊 Data Fetcher initialized")
    
    async def get_latest_data(
        self,
        symbol: str = 'NQ',
        period: str = '2y',
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Get latest market data
        
        Args:
            symbol: Trading symbol (NQ for Nasdaq)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Bar interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            self.logger.info(f"Fetching {period} of {interval} data for {symbol}")
            
            # Use NQ=F for Nasdaq futures on yfinance
            ticker = yf.Ticker(self.symbol)
            
            # Fetch data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.error("No data received from yfinance")
                # Try to load from cache
                return await self.load_cached_data()
            
            # Clean and prepare data
            data = self.prepare_data(data)
            
            # Cache the data
            await self.cache_data(data)
            
            # Store in memory
            self.data = data
            self.last_update = datetime.now()
            
            self.logger.info(f"✅ Fetched {len(data)} bars of data")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            # Try to load from cache as fallback
            return await self.load_cached_data()
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean market data
        
        Args:
            data: Raw data from yfinance
        
        Returns:
            pd.DataFrame: Cleaned data with additional indicators
        """
        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"Missing column: {col}")
                data[col] = 0
        
        # Remove any NaN values
        data = data.dropna()
        
        # Add useful calculated fields
        data['Range'] = data['High'] - data['Low']
        data['Change'] = data['Close'] - data['Open']
        data['ChangePercent'] = (data['Change'] / data['Open']) * 100
        
        # Add moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Add volume moving average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # Add ATR (Average True Range) for volatility
        data['ATR'] = self.calculate_atr(data)
        
        # Add RSI
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            data: OHLC data
            period: ATR period
        
        Returns:
            pd.Series: ATR values
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series
            period: RSI period
        
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def cache_data(self, data: pd.DataFrame):
        """
        Cache data to disk for offline use
        
        Args:
            data: Data to cache
        """
        try:
            cache_file = self.cache_dir / f"nq_data_{datetime.now().strftime('%Y%m%d')}.pkl"
            data.to_pickle(cache_file)
            self.logger.debug(f"Data cached to {cache_file}")
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
    
    async def load_cached_data(self) -> pd.DataFrame:
        """
        Load most recent cached data
        
        Returns:
            pd.DataFrame: Cached data or empty DataFrame
        """
        try:
            cache_files = list(self.cache_dir.glob("nq_data_*.pkl"))
            if not cache_files:
                self.logger.warning("No cached data available")
                return pd.DataFrame()
            
            # Get most recent cache file
            latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
            
            self.logger.info(f"Loading cached data from {latest_cache}")
            data = pd.read_pickle(latest_cache)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            return pd.DataFrame()
    
    def get_recent_bars(self, n: int = 100) -> pd.DataFrame:
        """
        Get the most recent n bars of data
        
        Args:
            n: Number of bars to return
        
        Returns:
            pd.DataFrame: Recent bars
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        return self.data.tail(n)
    
    def get_support_resistance_levels(self, lookback: int = 50) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels
        
        Args:
            lookback: Number of bars to look back
        
        Returns:
            Dict: Support and resistance levels
        """
        if self.data is None or len(self.data) < lookback:
            return {'support': [], 'resistance': []}
        
        recent_data = self.data.tail(lookback)
        
        # Find local minima and maxima
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
        # Simple peak detection
        resistance_levels = []
        support_levels = []
        
        for i in range(1, len(highs) - 1):
            # Resistance (local maxima)
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistance_levels.append(highs[i])
            
            # Support (local minima)
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                support_levels.append(lows[i])
        
        # Cluster nearby levels
        resistance_levels = self.cluster_levels(resistance_levels)
        support_levels = self.cluster_levels(support_levels)
        
        return {
            'support': sorted(support_levels)[:5],  # Top 5 support levels
            'resistance': sorted(resistance_levels, reverse=True)[:5]  # Top 5 resistance levels
        }
    
    def cluster_levels(self, levels: List[float], threshold: float = 10) -> List[float]:
        """
        Cluster nearby price levels
        
        Args:
            levels: List of price levels
            threshold: Clustering threshold in points
        
        Returns:
            List[float]: Clustered levels
        """
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level - current_cluster[-1] <= threshold:
                current_cluster.append(level)
            else:
                # Add mean of current cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def calculate_trend_line(self, lookback: int = 100) -> Tuple[float, float]:
        """
        Calculate trend line using linear regression
        
        Args:
            lookback: Number of bars for trend calculation
        
        Returns:
            Tuple[float, float]: Slope and intercept of trend line
        """
        if self.data is None or len(self.data) < lookback:
            return 0, 0
        
        recent_data = self.data.tail(lookback)
        
        # Use closing prices for trend
        prices = recent_data['Close'].values
        x = np.arange(len(prices))
        
        # Linear regression
        coefficients = np.polyfit(x, prices, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        return slope, intercept
    
    def get_current_price(self) -> float:
        """
        Get the current/latest price
        
        Returns:
            float: Current price or 0 if no data
        """
        if self.data is None or self.data.empty:
            return 0
        
        return self.data['Close'].iloc[-1]
    
    def get_market_state(self) -> str:
        """
        Determine current market state (trending up/down, ranging)
        
        Returns:
            str: Market state description
        """
        if self.data is None or len(self.data) < 50:
            return "unknown"
        
        # Check trend
        slope, _ = self.calculate_trend_line(50)
        
        # Check volatility
        recent_atr = self.data['ATR'].tail(20).mean()
        price = self.get_current_price()
        atr_percent = (recent_atr / price) * 100 if price > 0 else 0
        
        # Determine state
        if abs(slope) < 0.5 and atr_percent < 0.5:
            return "ranging"
        elif slope > 1:
            return "strong_uptrend"
        elif slope > 0:
            return "uptrend"
        elif slope < -1:
            return "strong_downtrend"
        else:
            return "downtrend"

# Async wrapper for synchronous yfinance calls
async def fetch_data_async(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Async wrapper for fetching data
    
    Args:
        symbol: Trading symbol
        period: Time period
        interval: Bar interval
    
    Returns:
        pd.DataFrame: Market data
    """
    fetcher = DataFetcher()
    return await fetcher.get_latest_data(symbol, period, interval)