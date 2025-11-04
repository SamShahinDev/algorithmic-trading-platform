#!/usr/bin/env python3
"""
Pattern Discovery Script
Discovers trading patterns in ES and CL data
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def discover_es_patterns():
    """Discover patterns in ES data"""
    logger.info("="*60)
    logger.info("DISCOVERING ES PATTERNS")
    logger.info("="*60)
    
    # Load ES data
    es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
    loader = DatabentoDailyLoader(es_path)
    
    # Use 2024 data for discovery (save 2025 for validation)
    logger.info("Loading ES data for pattern discovery (Jan-Jun 2024)...")
    es_data = loader.load_date_range('2024-01-01', '2024-06-30', 'ES')
    
    if es_data.empty:
        logger.error("No ES data loaded!")
        return []
        
    logger.info(f"Loaded {len(es_data)} ES data points")
    
    # Get main contracts only
    es_main = es_data[es_data['symbol'].str.match('^ES[A-Z]\\d+$', na=False)]
    logger.info(f"Main ES contract data points: {len(es_main)}")
    
    # Run pattern discovery
    discoverer = PatternDiscovery(es_main, 'ES')
    patterns = discoverer.discover_all_patterns(min_samples=30)
    
    # Save patterns
    patterns_data = []
    for pattern in patterns:
        patterns_data.append({
            'name': pattern.name,
            'market': pattern.market,
            'entry_conditions': pattern.entry_conditions,
            'exit_conditions': pattern.exit_conditions,
            'time_filter': pattern.time_filter,
            'statistics': pattern.statistics
        })
        
    with open('../es_bot/discovered_patterns.json', 'w') as f:
        json.dump(patterns_data, f, indent=2)
        
    logger.info(f"Saved {len(patterns_data)} ES patterns to es_bot/discovered_patterns.json")
    
    # Show top patterns
    if patterns:
        logger.info("\nTop 5 ES Patterns by Edge:")
        sorted_patterns = sorted(patterns, key=lambda p: p.statistics.get('edge', 0), reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:5]):
            logger.info(f"\n{i+1}. {pattern.name}")
            logger.info(f"   Edge: {pattern.statistics.get('edge', 0):.4%}")
            logger.info(f"   Win Rate: {pattern.statistics.get('win_rate', 0):.2%}")
            logger.info(f"   Sharpe: {pattern.statistics.get('sharpe', 0):.2f}")
            logger.info(f"   Samples: {pattern.statistics.get('sample_size', 0)}")
            
    return patterns

def discover_cl_patterns():
    """Discover patterns in CL data"""
    logger.info("="*60)
    logger.info("DISCOVERING CL PATTERNS")
    logger.info("="*60)
    
    # Load CL data
    cl_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-CR4KVBURP8')
    loader = DatabentoDailyLoader(cl_path)
    
    # Use 2024 data for discovery
    logger.info("Loading CL data for pattern discovery (Jan-Jun 2024)...")
    cl_data = loader.load_date_range('2024-01-01', '2024-06-30', 'CL')
    
    if cl_data.empty:
        logger.error("No CL data loaded!")
        return []
        
    logger.info(f"Loaded {len(cl_data)} CL data points")
    
    # Get main contracts only
    cl_main = cl_data[cl_data['symbol'].str.match('^CL[A-Z]\\d+$', na=False)]
    logger.info(f"Main CL contract data points: {len(cl_main)}")
    
    # Run pattern discovery
    discoverer = PatternDiscovery(cl_main, 'CL')
    patterns = discoverer.discover_all_patterns(min_samples=30)
    
    # Save patterns
    patterns_data = []
    for pattern in patterns:
        patterns_data.append({
            'name': pattern.name,
            'market': pattern.market,
            'entry_conditions': pattern.entry_conditions,
            'exit_conditions': pattern.exit_conditions,
            'time_filter': pattern.time_filter,
            'statistics': pattern.statistics
        })
        
    with open('../cl_bot/discovered_patterns.json', 'w') as f:
        json.dump(patterns_data, f, indent=2)
        
    logger.info(f"Saved {len(patterns_data)} CL patterns to cl_bot/discovered_patterns.json")
    
    # Show top patterns
    if patterns:
        logger.info("\nTop 5 CL Patterns by Edge:")
        sorted_patterns = sorted(patterns, key=lambda p: p.statistics.get('edge', 0), reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:5]):
            logger.info(f"\n{i+1}. {pattern.name}")
            logger.info(f"   Edge: {pattern.statistics.get('edge', 0):.4%}")
            logger.info(f"   Win Rate: {pattern.statistics.get('win_rate', 0):.2%}")
            logger.info(f"   Sharpe: {pattern.statistics.get('sharpe', 0):.2f}")
            logger.info(f"   Samples: {pattern.statistics.get('sample_size', 0)}")
            
    return patterns

def validate_patterns():
    """Validate discovered patterns on out-of-sample data"""
    logger.info("="*60)
    logger.info("VALIDATING PATTERNS ON 2024 H2 DATA")
    logger.info("="*60)
    
    # This would load July-Dec 2024 data and test the patterns
    # Implementation left for next phase
    pass

def main():
    """Run pattern discovery for all markets"""
    start_time = datetime.now()
    
    logger.info("Starting pattern discovery process...")
    logger.info(f"Start time: {start_time}")
    
    # Discover ES patterns
    es_patterns = discover_es_patterns()
    
    logger.info("\n" + "="*60 + "\n")
    
    # Discover CL patterns  
    cl_patterns = discover_cl_patterns()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*60)
    logger.info("PATTERN DISCOVERY COMPLETE")
    logger.info("="*60)
    logger.info(f"Total ES patterns: {len(es_patterns)}")
    logger.info(f"Total CL patterns: {len(cl_patterns)}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"End time: {end_time}")
    
    # Next steps
    logger.info("\nNext Steps:")
    logger.info("1. Review discovered patterns in es_bot/discovered_patterns.json and cl_bot/discovered_patterns.json")
    logger.info("2. Run validation on out-of-sample data (Jul-Dec 2024)")
    logger.info("3. Backtest top patterns on 2025 data")
    logger.info("4. Deploy best patterns to paper trading")

if __name__ == "__main__":
    main()