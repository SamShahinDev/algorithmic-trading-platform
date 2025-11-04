import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.data_loader import HybridDataLoader
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_loader():
    """Test hybrid data loader functionality"""
    print("\n" + "="*50)
    print("TESTING DATA LOADER")
    print("="*50)
    
    try:
        # Initialize loader
        loader = HybridDataLoader()
        
        # Test 1: Load historical data only
        print("\n1. Testing historical data loading...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = await loader.load_data(
            start_time=start_date,
            end_time=end_date,
            symbol='NQ.FUT'
        )
        
        if historical_data is not None and not historical_data.empty:
            print(f"✓ Loaded {len(historical_data)} historical records")
            print(f"  Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
            print(f"  Columns: {list(historical_data.columns)}")
            print(f"  Sample data:")
            print(historical_data.head(3))
        else:
            print("✗ Failed to load historical data")
        
        # Test 2: Check contract mapping
        print("\n2. Testing contract mapping...")
        contract_map = loader.build_contract_map()
        if contract_map:
            print(f"✓ Found {len(contract_map)} contract mappings")
            for month, info in list(contract_map.items())[:3]:
                print(f"  {month}: {info['symbol']} (Roll: {info['roll_date']})")
        else:
            print("✗ No contract mappings found")
        
        # Test 3: Data validation
        print("\n3. Testing data validation...")
        if historical_data is not None and not historical_data.empty:
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in historical_data.columns]
            
            if not missing_cols:
                print("✓ All required columns present")
                
                # Check for data quality
                nulls = historical_data[required_cols].isnull().sum()
                if nulls.sum() == 0:
                    print("✓ No null values in OHLCV data")
                else:
                    print(f"⚠ Found null values: {nulls[nulls > 0].to_dict()}")
                
                # Check price consistency
                price_checks = (
                    (historical_data['high'] >= historical_data['low']).all() and
                    (historical_data['high'] >= historical_data['open']).all() and
                    (historical_data['high'] >= historical_data['close']).all() and
                    (historical_data['low'] <= historical_data['open']).all() and
                    (historical_data['low'] <= historical_data['close']).all()
                )
                
                if price_checks:
                    print("✓ Price data is consistent (H>=L, H>=O,C, L<=O,C)")
                else:
                    print("✗ Price data inconsistencies detected")
            else:
                print(f"✗ Missing columns: {missing_cols}")
        
        # Test 4: Memory efficiency
        print("\n4. Testing memory efficiency...")
        if historical_data is not None:
            memory_usage = historical_data.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"✓ Data memory usage: {memory_usage:.2f} MB")
            print(f"  Records per MB: {len(historical_data) / max(memory_usage, 0.01):.0f}")
        
        print("\n" + "="*50)
        print("DATA LOADER TEST COMPLETE")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_data_loader())
    sys.exit(0 if success else 1)