#!/usr/bin/env python3
"""
Test Data Loading Script
Verifies that ES and CL data can be loaded from compressed Databento files
"""

import sys
import os
sys.path.append('..')

from pathlib import Path
import pandas as pd
import zstandard as zstd
import io
from datetime import datetime
from shared.data_loader import DatabentoDailyLoader, MultiMarketLoader

def test_basic_decompression():
    """Test basic file decompression"""
    print("\n" + "="*60)
    print("Testing Basic Decompression")
    print("="*60)
    
    # ES Data path
    es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
    
    if not es_path.exists():
        print(f"ERROR: ES data path not found: {es_path}")
        return False
        
    # Find a sample file
    sample_files = list(es_path.glob("glbx-mdp3-*.ohlcv-1m.csv.zst"))
    
    if not sample_files:
        print(f"ERROR: No .csv.zst files found in {es_path}")
        return False
        
    sample_file = sample_files[0]
    print(f"Testing with file: {sample_file.name}")
    
    try:
        # Decompress file
        dctx = zstd.ZstdDecompressor()
        
        with open(sample_file, 'rb') as f:
            compressed = f.read()
            print(f"Compressed size: {len(compressed):,} bytes")
            
            decompressed = dctx.decompress(compressed)
            print(f"Decompressed size: {len(decompressed):,} bytes")
            
        # Parse CSV
        df = pd.read_csv(io.StringIO(decompressed.decode('utf-8')))
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for symbols
        if 'symbol' in df.columns:
            unique_symbols = df['symbol'].unique()
            print(f"\nUnique symbols found: {len(unique_symbols)}")
            print("Sample symbols:", unique_symbols[:10].tolist() if len(unique_symbols) > 10 else unique_symbols.tolist())
            
            # Check for ES contracts
            es_symbols = [s for s in unique_symbols if 'ES' in s]
            print(f"\nES-related symbols: {es_symbols[:5]}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to decompress/parse file: {e}")
        return False

def test_es_data_loading():
    """Test loading ES data with the data loader"""
    print("\n" + "="*60)
    print("Testing ES Data Loading")
    print("="*60)
    
    es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
    
    if not es_path.exists():
        print(f"ERROR: ES data path not found: {es_path}")
        return
        
    try:
        loader = DatabentoDailyLoader(es_path)
        
        # Test loading a single day
        test_date = "20250101"  # January 1, 2025
        print(f"\nLoading data for {test_date}...")
        
        df = loader.load_single_day(test_date)
        
        if df.empty:
            print("No data found for this date (might be a holiday)")
            # Try another date
            test_date = "20250102"
            df = loader.load_single_day(test_date)
            
        if not df.empty:
            print(f"Loaded {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            
            # Filter for ES
            print("\nFiltering for ES contracts...")
            es_df = loader.load_single_day(test_date, "ES")
            
            if not es_df.empty:
                print(f"Found {len(es_df)} ES records")
                
                if 'symbol' in es_df.columns:
                    print(f"ES symbols: {es_df['symbol'].unique()}")
                    
                print("\nSample ES data:")
                print(es_df[['ts_event', 'symbol', 'open', 'high', 'low', 'close', 'volume']].head())
            else:
                print("No ES data found")
                
        # Test date range loading
        print("\n" + "-"*40)
        print("Testing date range loading...")
        
        start_date = "2025-01-01"
        end_date = "2025-01-07"
        
        print(f"Loading ES data from {start_date} to {end_date}...")
        range_df = loader.load_date_range(start_date, end_date, "ES")
        
        if not range_df.empty:
            print(f"Loaded {len(range_df)} total rows")
            
            if 'date' in range_df.columns:
                print(f"Dates covered: {range_df['date'].unique()}")
                
            if 'symbol' in range_df.columns:
                print(f"Symbols found: {range_df['symbol'].unique()}")
                
        # Get available symbols
        print("\n" + "-"*40)
        print("Getting available symbols...")
        symbols = loader.load_symbols_list(test_date)
        
        if symbols:
            print(f"Found {len(symbols)} unique symbols")
            
            # Show relevant futures
            futures_symbols = [s for s in symbols if any(x in s for x in ['ES', 'NQ', 'CL', 'GC', 'SI'])]
            print(f"Futures symbols: {futures_symbols[:20]}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_cl_data_loading():
    """Test loading CL data"""
    print("\n" + "="*60)
    print("Testing CL Data Loading")
    print("="*60)
    
    cl_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-CR4KVBURP8')
    
    if not cl_path.exists():
        print(f"ERROR: CL data path not found: {cl_path}")
        return
        
    try:
        loader = DatabentoDailyLoader(cl_path)
        
        # Test loading a single day
        test_date = "20250102"
        print(f"\nLoading CL data for {test_date}...")
        
        cl_df = loader.load_single_day(test_date, "CL")
        
        if not cl_df.empty:
            print(f"Found {len(cl_df)} CL records")
            
            if 'symbol' in cl_df.columns:
                print(f"CL symbols: {cl_df['symbol'].unique()}")
                
            print("\nSample CL data:")
            print(cl_df[['ts_event', 'symbol', 'open', 'high', 'low', 'close', 'volume']].head())
        else:
            print("No CL data found - checking all symbols...")
            
            # Load without filter to see what's available
            all_df = loader.load_single_day(test_date)
            if not all_df.empty and 'symbol' in all_df.columns:
                symbols = all_df['symbol'].unique()
                cl_related = [s for s in symbols if 'CL' in s or 'QO' in s]  # QO is also crude
                print(f"CL-related symbols found: {cl_related}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_multi_market_loading():
    """Test loading multiple markets together"""
    print("\n" + "="*60)
    print("Testing Multi-Market Loading")
    print("="*60)
    
    paths = {
        'ES': Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH'),
        'CL': Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-CR4KVBURP8')
    }
    
    # Check paths exist
    for symbol, path in paths.items():
        if not path.exists():
            print(f"WARNING: {symbol} path not found: {path}")
            paths.pop(symbol)
            
    if not paths:
        print("ERROR: No valid data paths found")
        return
        
    try:
        multi_loader = MultiMarketLoader(paths)
        
        # Load data for both markets
        start_date = "2025-01-02"
        end_date = "2025-01-03"
        
        print(f"\nLoading data from {start_date} to {end_date}...")
        all_data = multi_loader.load_all_markets(start_date, end_date)
        
        for symbol, df in all_data.items():
            if not df.empty:
                print(f"\n{symbol}: Loaded {len(df)} rows")
                print(f"  Date range: {df['ts_event'].min()} to {df['ts_event'].max()}")
                
                if 'symbol' in df.columns:
                    print(f"  Contracts: {df['symbol'].unique()[:5]}")
                    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_data_statistics():
    """Analyze data statistics"""
    print("\n" + "="*60)
    print("Data Statistics Analysis")
    print("="*60)
    
    es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
    
    if es_path.exists():
        loader = DatabentoDailyLoader(es_path)
        
        # Load a week of data
        print("\nLoading one week of ES data for analysis...")
        df = loader.load_date_range("2025-01-06", "2025-01-10", "ES")
        
        if not df.empty:
            print(f"Total records: {len(df)}")
            
            # Basic statistics
            if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                print("\nPrice Statistics:")
                print(f"  Average close: ${df['close'].mean():.2f}")
                print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                print(f"  Average volume: {df['volume'].mean():.0f}")
                
                # Calculate returns
                if 'close' in df.columns:
                    df['returns'] = df['close'].pct_change()
                    print(f"\nReturns Statistics:")
                    print(f"  Mean return: {df['returns'].mean():.4%}")
                    print(f"  Std deviation: {df['returns'].std():.4%}")
                    print(f"  Min return: {df['returns'].min():.4%}")
                    print(f"  Max return: {df['returns'].max():.4%}")
                    
            # Time analysis
            if 'ts_event' in df.columns:
                df['ts_event'] = pd.to_datetime(df['ts_event'])
                df['hour'] = df['ts_event'].dt.hour
                
                print("\nRecords by hour:")
                hourly_counts = df['hour'].value_counts().sort_index()
                for hour, count in hourly_counts.head(10).items():
                    print(f"  Hour {hour:02d}: {count} records")

def main():
    """Run all tests"""
    print("="*60)
    print("DATABENTO DATA LOADING TEST SUITE")
    print("="*60)
    
    # Check if zstandard is installed
    try:
        import zstandard
        print("✓ zstandard library installed")
    except ImportError:
        print("✗ zstandard library not installed")
        print("  Install with: pip install zstandard")
        return
        
    # Run tests
    tests = [
        ("Basic Decompression", test_basic_decompression),
        ("ES Data Loading", test_es_data_loading),
        ("CL Data Loading", test_cl_data_loading),
        ("Multi-Market Loading", test_multi_market_loading),
        ("Data Statistics", test_data_statistics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            results.append((test_name, "PASSED"))
        except Exception as e:
            results.append((test_name, f"FAILED: {e}"))
            print(f"ERROR in {test_name}: {e}")
            
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")
        
    passed = sum(1 for _, r in results if r == "PASSED")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()