#!/usr/bin/env python3
"""
Optimized Pattern Discovery Runner
Discovers trading patterns in ES and CL data with progress tracking
"""

import sys
sys.path.append('..')

from shared.data_loader import DatabentoDailyLoader
from shared.pattern_discovery import PatternDiscovery
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def discover_patterns_for_market(market: str, data_path: Path, 
                                start_date: str, end_date: str,
                                output_file: str):
    """
    Run pattern discovery for a specific market
    
    Args:
        market: Market symbol (ES or CL)
        data_path: Path to data files
        start_date: Start date for discovery
        end_date: End date for discovery
        output_file: Output JSON file for patterns
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DISCOVERING {market} PATTERNS")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Initialize data loader
    loader = DatabentoDailyLoader(data_path)
    
    # Load data
    logger.info(f"Loading {market} data...")
    data = loader.load_date_range(start_date, end_date, market)
    
    if data.empty:
        logger.error(f"No {market} data loaded!")
        return []
    
    logger.info(f"Loaded {len(data):,} total data points")
    
    # Filter for main contracts only (not spreads)
    main_contracts = data[data['symbol'].str.match(f'^{market}[A-Z]\\d+$', na=False)]
    logger.info(f"Main contract data points: {len(main_contracts):,}")
    
    # Show date range and contracts
    logger.info(f"Date range in data: {main_contracts['ts_event'].min()} to {main_contracts['ts_event'].max()}")
    unique_contracts = main_contracts['symbol'].unique()
    logger.info(f"Contracts found: {unique_contracts[:10].tolist()}")
    
    # Initialize pattern discovery
    logger.info("\nInitializing pattern discovery...")
    discoverer = PatternDiscovery(main_contracts, market)
    
    # Discover patterns with reduced parameters for faster execution
    logger.info("Discovering momentum patterns...")
    momentum_patterns = discoverer.discover_momentum_patterns(
        lookback_periods=[5, 10, 15],  # Reduced from [5, 10, 15, 30]
        min_samples=50  # Increased from 30 for better statistical significance
    )
    logger.info(f"Found {len(momentum_patterns)} momentum patterns")
    
    logger.info("Discovering reversal patterns...")
    reversal_patterns = discoverer.discover_reversal_patterns(min_samples=50)
    logger.info(f"Found {len(reversal_patterns)} reversal patterns")
    
    # Combine all patterns
    all_patterns = momentum_patterns + reversal_patterns
    
    # Convert to JSON-serializable format
    patterns_data = []
    for pattern in all_patterns:
        patterns_data.append({
            'name': pattern.name,
            'market': pattern.market,
            'entry_conditions': pattern.entry_conditions,
            'exit_conditions': pattern.exit_conditions,
            'time_filter': pattern.time_filter,
            'statistics': pattern.statistics
        })
    
    # Save patterns
    with open(output_file, 'w') as f:
        json.dump(patterns_data, f, indent=2)
    
    logger.info(f"\nSaved {len(patterns_data)} patterns to {output_file}")
    
    # Show top patterns
    if all_patterns:
        logger.info(f"\nTop 5 {market} Patterns by Win Rate:")
        sorted_patterns = sorted(all_patterns, 
                               key=lambda p: p.statistics.get('win_rate', 0), 
                               reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:5]):
            logger.info(f"\n{i+1}. {pattern.name}")
            logger.info(f"   Win Rate: {pattern.statistics.get('win_rate', 0):.1%}")
            logger.info(f"   Edge: {pattern.statistics.get('edge', 0):.3%}")
            logger.info(f"   Sharpe: {pattern.statistics.get('sharpe', 0):.2f}")
            logger.info(f"   Samples: {pattern.statistics.get('sample_size', 0)}")
            logger.info(f"   Entry: {pattern.entry_conditions}")
    
    elapsed = time.time() - start_time
    logger.info(f"\n{market} discovery completed in {elapsed:.1f} seconds")
    
    return all_patterns

def main():
    """Run pattern discovery for ES and CL"""
    overall_start = time.time()
    
    logger.info("="*60)
    logger.info("PATTERN DISCOVERY SYSTEM")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now()}")
    
    # Data paths
    es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
    cl_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-CR4KVBURP8')
    
    # Discovery period - using 3 months for faster execution
    # You can expand this to full year once verified
    discovery_start = '2024-01-01'
    discovery_end = '2024-03-31'  # 3 months for initial discovery
    
    all_patterns = {}
    
    # Discover ES patterns
    try:
        es_patterns = discover_patterns_for_market(
            'ES', es_path, discovery_start, discovery_end,
            '../es_bot/discovered_patterns.json'
        )
        all_patterns['ES'] = es_patterns
    except Exception as e:
        logger.error(f"ES discovery failed: {e}")
        all_patterns['ES'] = []
    
    # Discover CL patterns
    try:
        cl_patterns = discover_patterns_for_market(
            'CL', cl_path, discovery_start, discovery_end,
            '../cl_bot/discovered_patterns.json'
        )
        all_patterns['CL'] = cl_patterns
    except Exception as e:
        logger.error(f"CL discovery failed: {e}")
        all_patterns['CL'] = []
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DISCOVERY COMPLETE")
    logger.info("="*60)
    
    for market, patterns in all_patterns.items():
        logger.info(f"{market}: {len(patterns)} patterns discovered")
        
        if patterns:
            # Calculate average statistics
            avg_win_rate = sum(p.statistics.get('win_rate', 0) for p in patterns) / len(patterns)
            avg_sharpe = sum(p.statistics.get('sharpe', 0) for p in patterns) / len(patterns)
            total_samples = sum(p.statistics.get('sample_size', 0) for p in patterns)
            
            logger.info(f"  Average win rate: {avg_win_rate:.1%}")
            logger.info(f"  Average Sharpe: {avg_sharpe:.2f}")
            logger.info(f"  Total samples: {total_samples:,}")
    
    elapsed = time.time() - overall_start
    logger.info(f"\nTotal execution time: {elapsed:.1f} seconds")
    logger.info(f"End time: {datetime.now()}")
    
    # Next steps
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("1. Review patterns in es_bot/discovered_patterns.json and cl_bot/discovered_patterns.json")
    logger.info("2. Run validation on out-of-sample data (Apr-Dec 2024)")
    logger.info("3. Backtest top patterns on 2025 data")
    logger.info("4. Deploy best patterns to paper trading")
    logger.info("\nTo expand discovery period, modify discovery_start and discovery_end in the script")

if __name__ == "__main__":
    main()