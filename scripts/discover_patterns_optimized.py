#!/usr/bin/env python3
"""
Optimized Pattern Discovery with Relaxed Parameters
Finds more patterns with lower thresholds for initial discovery
"""

import sys
sys.path.append('..')

from shared.data_loader import DatabentoDailyLoader
from shared.pattern_discovery import PatternDiscovery, Pattern
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedPatternDiscovery(PatternDiscovery):
    """Enhanced pattern discovery with more patterns"""
    
    def discover_simple_patterns(self) -> list:
        """Discover simpler patterns with relaxed criteria"""
        patterns = []
        
        # Time-based patterns (market open, close, lunch)
        patterns.extend(self._discover_time_patterns())
        
        # Simple breakout patterns
        patterns.extend(self._discover_breakout_patterns())
        
        # Volume surge patterns
        patterns.extend(self._discover_volume_patterns())
        
        return patterns
    
    def _discover_time_patterns(self) -> list:
        """Find patterns based on time of day"""
        patterns = []
        
        # Market open pattern (first 30 minutes)
        open_mask = (self.data.index.hour == 9) & (self.data.index.minute < 30)
        if open_mask.sum() > 20:
            open_data = self.data[open_mask]
            
            # Check if market tends to continue in opening direction
            opening_moves = []
            for date in open_data.index.date:
                day_data = self.data[self.data.index.date == date]
                if len(day_data) > 60:
                    open_move = (day_data.iloc[30]['close'] - day_data.iloc[0]['open']) / day_data.iloc[0]['open']
                    continuation = (day_data.iloc[60]['close'] - day_data.iloc[30]['close']) / day_data.iloc[30]['close']
                    
                    if abs(open_move) > 0.001:  # Significant opening move
                        opening_moves.append({
                            'open_direction': 'up' if open_move > 0 else 'down',
                            'continuation': continuation,
                            'profitable': (open_move > 0 and continuation > 0) or (open_move < 0 and continuation < 0)
                        })
            
            if len(opening_moves) > 20:
                df_moves = pd.DataFrame(opening_moves)
                for direction in ['up', 'down']:
                    dir_moves = df_moves[df_moves['open_direction'] == direction]
                    if len(dir_moves) > 10:
                        win_rate = dir_moves['profitable'].mean()
                        
                        if win_rate > 0.52:  # Relaxed from 0.55
                            patterns.append(Pattern(
                                name=f"{self.symbol}_opening_continuation_{direction}",
                                market=self.symbol,
                                entry_conditions={
                                    'pattern_type': 'opening_continuation',
                                    'direction': direction,
                                    'time_window': (9, 0, 9, 30)
                                },
                                exit_conditions={
                                    'hold_period': 30,
                                    'stop_loss_pct': 0.003,
                                    'target_pct': 0.005
                                },
                                statistics={
                                    'sample_size': len(dir_moves),
                                    'win_rate': float(win_rate),
                                    'edge': float(win_rate - 0.5) * 0.005
                                }
                            ))
                            logger.info(f"Found opening pattern: {direction} with {win_rate:.1%} win rate")
        
        # Power hour pattern (last hour of trading)
        power_mask = (self.data.index.hour == 15) | ((self.data.index.hour == 14) & (self.data.index.minute >= 30))
        if power_mask.sum() > 20:
            power_data = self.data[power_mask]
            
            # Check for momentum in last hour
            hourly_returns = power_data['returns'].dropna()
            if len(hourly_returns) > 100:
                positive_momentum = (hourly_returns > 0).mean()
                
                if positive_momentum > 0.52 or positive_momentum < 0.48:
                    direction = 'bullish' if positive_momentum > 0.52 else 'bearish'
                    patterns.append(Pattern(
                        name=f"{self.symbol}_power_hour_{direction}",
                        market=self.symbol,
                        entry_conditions={
                            'pattern_type': 'power_hour',
                            'direction': direction,
                            'time_window': (14, 30, 16, 0)
                        },
                        exit_conditions={
                            'hold_period': 15,
                            'stop_loss_pct': 0.002,
                            'target_pct': 0.003
                        },
                        statistics={
                            'sample_size': len(hourly_returns),
                            'win_rate': float(positive_momentum if direction == 'bullish' else 1 - positive_momentum),
                            'edge': abs(positive_momentum - 0.5) * 0.003
                        }
                    ))
                    logger.info(f"Found power hour pattern: {direction}")
        
        return patterns
    
    def _discover_breakout_patterns(self) -> list:
        """Find simple breakout patterns"""
        patterns = []
        
        # Calculate rolling high/low
        self.data['high_20'] = self.data['high'].rolling(20).max()
        self.data['low_20'] = self.data['low'].rolling(20).min()
        
        # Breakout signals
        bullish_breakout = self.data['close'] > self.data['high_20'].shift(1)
        bearish_breakout = self.data['close'] < self.data['low_20'].shift(1)
        
        # Analyze breakouts
        for signal, direction in [(bullish_breakout, 'bullish'), (bearish_breakout, 'bearish')]:
            signal_points = self.data.index[signal]
            
            if len(signal_points) > 30:
                results = []
                
                for idx in signal_points[:-30]:
                    future = self.data.loc[idx:].iloc[1:31]
                    if len(future) >= 30:
                        entry = self.data.loc[idx, 'close']
                        
                        for hold in [5, 10, 15]:
                            exit_price = future.iloc[hold-1]['close']
                            
                            if direction == 'bullish':
                                pnl = (exit_price - entry) / entry
                            else:
                                pnl = (entry - exit_price) / entry
                            
                            results.append({
                                'hold_period': hold,
                                'pnl': pnl,
                                'hour': idx.hour
                            })
                
                if len(results) > 30:
                    df_results = pd.DataFrame(results)
                    
                    for hold in [5, 10, 15]:
                        hold_data = df_results[df_results['hold_period'] == hold]
                        
                        if len(hold_data) > 20:
                            win_rate = (hold_data['pnl'] > 0).mean()
                            
                            if win_rate > 0.5:  # Very relaxed threshold
                                patterns.append(Pattern(
                                    name=f"{self.symbol}_{direction}_breakout_20bar_hold{hold}",
                                    market=self.symbol,
                                    entry_conditions={
                                        'pattern_type': 'breakout',
                                        'direction': direction,
                                        'lookback': 20
                                    },
                                    exit_conditions={
                                        'hold_period': hold,
                                        'stop_loss_pct': 0.003,
                                        'target_pct': 0.004
                                    },
                                    statistics={
                                        'sample_size': len(hold_data),
                                        'win_rate': float(win_rate),
                                        'avg_pnl': float(hold_data['pnl'].mean()),
                                        'edge': float(hold_data['pnl'].mean())
                                    }
                                ))
                                logger.info(f"Found breakout pattern: {direction} hold {hold} with {win_rate:.1%} win rate")
        
        return patterns
    
    def _discover_volume_patterns(self) -> list:
        """Find volume-based patterns"""
        patterns = []
        
        # Volume surges
        volume_surge = self.data['volume'] > self.data['volume'].rolling(50).mean() * 2
        
        if volume_surge.sum() > 30:
            surge_points = self.data.index[volume_surge]
            
            results = []
            for idx in surge_points[:-30]:
                future = self.data.loc[idx:].iloc[1:16]
                
                if len(future) >= 15:
                    entry = self.data.loc[idx, 'close']
                    
                    # Check if volume surge leads to continuation
                    for hold in [5, 10]:
                        exit_price = future.iloc[hold-1]['close']
                        direction = 'up' if future.iloc[0]['close'] > future.iloc[0]['open'] else 'down'
                        
                        if direction == 'up':
                            pnl = (exit_price - entry) / entry
                        else:
                            pnl = (entry - exit_price) / entry
                        
                        results.append({
                            'direction': direction,
                            'hold': hold,
                            'pnl': pnl
                        })
            
            if len(results) > 30:
                df_results = pd.DataFrame(results)
                
                for direction in ['up', 'down']:
                    for hold in [5, 10]:
                        mask = (df_results['direction'] == direction) & (df_results['hold'] == hold)
                        subset = df_results[mask]
                        
                        if len(subset) > 15:
                            win_rate = (subset['pnl'] > 0).mean()
                            
                            if win_rate > 0.5:
                                patterns.append(Pattern(
                                    name=f"{self.symbol}_volume_surge_{direction}_hold{hold}",
                                    market=self.symbol,
                                    entry_conditions={
                                        'pattern_type': 'volume_surge',
                                        'direction': direction,
                                        'volume_multiplier': 2
                                    },
                                    exit_conditions={
                                        'hold_period': hold,
                                        'stop_loss_pct': 0.002,
                                        'target_pct': 0.003
                                    },
                                    statistics={
                                        'sample_size': len(subset),
                                        'win_rate': float(win_rate),
                                        'avg_pnl': float(subset['pnl'].mean())
                                    }
                                ))
                                logger.info(f"Found volume pattern: {direction} hold {hold}")
        
        return patterns

def discover_with_optimization(market: str, data_path: Path, 
                              start_date: str, end_date: str) -> list:
    """Run optimized discovery for a market"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZED DISCOVERY FOR {market}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"{'='*60}")
    
    # Load data
    loader = DatabentoDailyLoader(data_path)
    data = loader.load_date_range(start_date, end_date, market)
    
    if data.empty:
        logger.error(f"No data loaded for {market}")
        return []
    
    # Get main contracts
    main_contracts = data[data['symbol'].str.match(f'^{market}[A-Z]\\d+$', na=False)]
    logger.info(f"Loaded {len(main_contracts):,} data points")
    
    # Run optimized discovery
    discoverer = OptimizedPatternDiscovery(main_contracts, market)
    
    all_patterns = []
    
    # Original patterns with relaxed parameters
    logger.info("\nDiscovering momentum patterns...")
    momentum = discoverer.discover_momentum_patterns(
        lookback_periods=[5, 10],
        min_samples=15  # Much lower threshold
    )
    all_patterns.extend(momentum)
    logger.info(f"Found {len(momentum)} momentum patterns")
    
    # Simple patterns
    logger.info("\nDiscovering simple patterns...")
    simple = discoverer.discover_simple_patterns()
    all_patterns.extend(simple)
    logger.info(f"Found {len(simple)} simple patterns")
    
    # Reversal patterns
    logger.info("\nDiscovering reversal patterns...")
    reversal = discoverer.discover_reversal_patterns(min_samples=20)
    all_patterns.extend(reversal)
    logger.info(f"Found {len(reversal)} reversal patterns")
    
    logger.info(f"\nTotal patterns found: {len(all_patterns)}")
    
    return all_patterns

def main():
    """Run optimized pattern discovery"""
    
    logger.info("OPTIMIZED PATTERN DISCOVERY SYSTEM")
    logger.info("="*60)
    
    # Paths
    es_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-98YG33QNQH')
    cl_path = Path('/Users/royaltyvixion/Documents/XTRADING/Historical Data/New Data/GLBX-20250828-CR4KVBURP8')
    
    # Use 1 month for faster results
    start_date = '2024-02-01'
    end_date = '2024-02-29'
    
    # Discover ES patterns
    es_patterns = discover_with_optimization('ES', es_path, start_date, end_date)
    
    # Save ES patterns
    if es_patterns:
        patterns_data = []
        for p in es_patterns:
            patterns_data.append({
                'name': p.name,
                'market': p.market,
                'entry_conditions': p.entry_conditions,
                'exit_conditions': p.exit_conditions,
                'time_filter': p.time_filter,
                'statistics': p.statistics
            })
        
        with open('../es_bot/optimized_patterns.json', 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"\nSaved {len(patterns_data)} ES patterns")
    
    # Discover CL patterns
    cl_patterns = discover_with_optimization('CL', cl_path, start_date, end_date)
    
    # Save CL patterns
    if cl_patterns:
        patterns_data = []
        for p in cl_patterns:
            patterns_data.append({
                'name': p.name,
                'market': p.market,
                'entry_conditions': p.entry_conditions,
                'exit_conditions': p.exit_conditions,
                'time_filter': p.time_filter,
                'statistics': p.statistics
            })
        
        with open('../cl_bot/optimized_patterns.json', 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"\nSaved {len(patterns_data)} CL patterns")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DISCOVERY COMPLETE")
    logger.info(f"ES Patterns: {len(es_patterns)}")
    logger.info(f"CL Patterns: {len(cl_patterns)}")
    
    # Show top patterns
    all_patterns = es_patterns + cl_patterns
    if all_patterns:
        logger.info("\nTop Patterns by Win Rate:")
        sorted_patterns = sorted(all_patterns, 
                               key=lambda p: p.statistics.get('win_rate', 0), 
                               reverse=True)[:10]
        
        for i, p in enumerate(sorted_patterns):
            logger.info(f"{i+1}. {p.name}: {p.statistics.get('win_rate', 0):.1%} WR, {p.statistics.get('sample_size', 0)} samples")

if __name__ == "__main__":
    main()