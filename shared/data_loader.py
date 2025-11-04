"""
Databento Data Loader
Handles loading and decompression of daily Databento OHLCV files
"""

import pandas as pd
import numpy as np
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabentoDailyLoader:
    """Handles loading and decompression of daily Databento files"""
    
    def __init__(self, base_path: Path):
        """
        Initialize loader with base path to data files
        
        Args:
            base_path: Path to directory containing .csv.zst files
        """
        self.base_path = Path(base_path)
        self.dctx = zstd.ZstdDecompressor()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.base_path}")
            
        self.logger.info(f"Initialized data loader for: {self.base_path}")
        
    def load_single_day(self, date: str, symbol_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Load a single day's data from compressed file
        
        Args:
            date: Date string in format 'YYYYMMDD'
            symbol_filter: Optional symbol to filter for (e.g., 'ES', 'CL', 'NQ')
            
        Returns:
            DataFrame with OHLCV data for the specified day
        """
        filename = f"glbx-mdp3-{date}.ohlcv-1m.csv.zst"
        file_path = self.base_path / filename
        
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
            
        try:
            # Decompress file using stream reader for proper multi-frame support
            with open(file_path, 'rb') as f:
                with self.dctx.stream_reader(f) as reader:
                    decompressed_data = reader.read()
                
            # Parse CSV - only parse ts_event as date (ts_init may not exist)
            df = pd.read_csv(
                io.StringIO(decompressed_data.decode('utf-8')),
                parse_dates=['ts_event']
            )
            
            # Add date column for reference
            df['date'] = pd.to_datetime(date, format='%Y%m%d')
            
            # Filter by symbol if specified
            if symbol_filter and 'symbol' in df.columns:
                # Handle different symbol formats
                df = df[df['symbol'].str.contains(symbol_filter, na=False, case=False)]
                
                if len(df) == 0:
                    self.logger.warning(f"No data found for symbol {symbol_filter} on {date}")
                    
            self.logger.debug(f"Loaded {len(df)} rows from {filename}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {str(e)}")
            return pd.DataFrame()
    
    def load_date_range(self, 
                       start_date: str, 
                       end_date: str,
                       symbol_filter: Optional[str] = None,
                       skip_weekends: bool = True) -> pd.DataFrame:
        """
        Load data for a date range
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            symbol_filter: Optional symbol filter (e.g., 'ES', 'CL')
            skip_weekends: Skip Saturday and Sunday files
            
        Returns:
            Combined DataFrame for the entire date range
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        self.logger.info(f"Loading data from {start_date} to {end_date} for symbol: {symbol_filter or 'ALL'}")
        
        all_data = []
        current = start
        days_loaded = 0
        days_skipped = 0
        
        while current <= end:
            # Skip weekends if requested (futures don't trade on weekends)
            if skip_weekends and current.weekday() in [5, 6]:  # Saturday=5, Sunday=6
                current += timedelta(days=1)
                days_skipped += 1
                continue
                
            date_str = current.strftime('%Y%m%d')
            df = self.load_single_day(date_str, symbol_filter)
            
            if not df.empty:
                all_data.append(df)
                days_loaded += 1
            else:
                days_skipped += 1
                
            current += timedelta(days=1)
            
        self.logger.info(f"Loaded {days_loaded} days, skipped {days_skipped} days")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp
            if 'ts_event' in combined.columns:
                combined = combined.sort_values('ts_event').reset_index(drop=True)
                
            self.logger.info(f"Combined data shape: {combined.shape}")
            return combined
        
        self.logger.warning("No data loaded for the specified range")
        return pd.DataFrame()
    
    def load_symbols_list(self, sample_date: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols from a sample day
        
        Args:
            sample_date: Date to check symbols (default: most recent file)
            
        Returns:
            List of unique symbols found
        """
        if sample_date:
            df = self.load_single_day(sample_date)
        else:
            # Find most recent file
            files = sorted(self.base_path.glob("glbx-mdp3-*.ohlcv-1m.csv.zst"))
            if files:
                # Extract date from filename
                latest_file = files[-1].name
                date_str = latest_file.split('-')[2].split('.')[0]
                df = self.load_single_day(date_str)
            else:
                self.logger.error("No data files found")
                return []
                
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique().tolist()
            self.logger.info(f"Found {len(symbols)} unique symbols")
            return sorted(symbols)
            
        return []
    
    def get_contract_months(self, base_symbol: str, year: int) -> pd.DataFrame:
        """
        Load all contract months for a given symbol and year
        
        Args:
            base_symbol: Base symbol (e.g., 'ES', 'CL')
            year: Year to load
            
        Returns:
            DataFrame with all contract months data
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        df = self.load_date_range(start_date, end_date, base_symbol)
        
        if 'symbol' in df.columns:
            # Group by unique symbols to see all contracts
            contracts = df.groupby('symbol').agg({
                'ts_event': ['min', 'max'],
                'close': 'count'
            })
            
            self.logger.info(f"Found {len(contracts)} contracts for {base_symbol} in {year}")
            
        return df
    
    def resample_to_timeframe(self, 
                            data: pd.DataFrame, 
                            timeframe: str = '5min') -> pd.DataFrame:
        """
        Resample 1-minute data to higher timeframes
        
        Args:
            data: DataFrame with 1-minute OHLCV data
            timeframe: Target timeframe (e.g., '5min', '15min', '1H')
            
        Returns:
            Resampled DataFrame
        """
        if data.empty:
            return data
            
        # Ensure datetime index
        if 'ts_event' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('ts_event')
            
        # Resample OHLCV
        resampled = data.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        self.logger.info(f"Resampled {len(data)} rows to {len(resampled)} rows at {timeframe}")
        
        return resampled


class MultiMarketLoader:
    """Convenience class for loading multiple markets"""
    
    def __init__(self, paths_config: Dict[str, Path]):
        """
        Initialize loaders for multiple markets
        
        Args:
            paths_config: Dictionary mapping symbols to data paths
                         e.g., {'ES': Path(...), 'CL': Path(...)}
        """
        self.loaders = {}
        
        for symbol, path in paths_config.items():
            self.loaders[symbol] = DatabentoDailyLoader(path)
            logger.info(f"Initialized loader for {symbol}")
            
    def load_all_markets(self, 
                        start_date: str, 
                        end_date: str,
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for all configured markets
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            symbols: Optional list of symbols to load (default: all)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        symbols_to_load = symbols or list(self.loaders.keys())
        
        for symbol in symbols_to_load:
            if symbol in self.loaders:
                logger.info(f"Loading {symbol} data...")
                results[symbol] = self.loaders[symbol].load_date_range(
                    start_date, end_date, symbol
                )
            else:
                logger.warning(f"No loader configured for {symbol}")
                
        return results
    
    def load_synchronized(self, 
                         start_date: str, 
                         end_date: str) -> pd.DataFrame:
        """
        Load all markets and synchronize timestamps
        
        Returns:
            DataFrame with all markets aligned by timestamp
        """
        all_data = self.load_all_markets(start_date, end_date)
        
        # Align all dataframes by timestamp
        aligned_frames = []
        
        for symbol, df in all_data.items():
            if not df.empty:
                if 'ts_event' in df.columns:
                    df = df.set_index('ts_event')
                    
                # Rename columns to include symbol prefix
                df = df.add_prefix(f"{symbol}_")
                aligned_frames.append(df)
                
        if aligned_frames:
            # Join all frames on timestamp
            combined = pd.concat(aligned_frames, axis=1, join='outer')
            combined = combined.sort_index()
            
            logger.info(f"Synchronized data shape: {combined.shape}")
            return combined
            
        return pd.DataFrame()


if __name__ == "__main__":
    # Test loading ES data
    es_path = Path("/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH")
    
    if es_path.exists():
        loader = DatabentoDailyLoader(es_path)
        
        # Test loading a single day
        test_date = "20250101"
        df = loader.load_single_day(test_date, "ES")
        
        if not df.empty:
            print(f"\nLoaded {len(df)} rows for {test_date}")
            print(f"Columns: {df.columns.tolist()}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Test loading a range
            print("\n" + "="*50)
            print("Testing date range loading...")
            
            range_df = loader.load_date_range("2025-01-01", "2025-01-07", "ES")
            print(f"Loaded {len(range_df)} total rows for date range")
            
            if 'symbol' in range_df.columns:
                print(f"Unique symbols: {range_df['symbol'].unique()}")
        else:
            print(f"No data found for {test_date}")
    else:
        print(f"Path not found: {es_path}")