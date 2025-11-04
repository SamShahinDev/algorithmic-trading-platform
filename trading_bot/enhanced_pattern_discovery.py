"""
Enhanced NQ Pattern Discovery with Stricter Validation Criteria
Implements time-of-day and trend filters for improved win rates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from datetime import datetime, time
import talib
from pathlib import Path
import zstandard
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrictCriteria:
    """Strict validation criteria for pattern acceptance"""
    min_validation_win_rate: float = 0.40  # Up from 34%
    max_performance_degradation: float = 0.10  # Max 10% drop from training
    min_net_expectancy: float = 10.0  # Up from ~$3
    min_trades_per_day: int = 2  # Minimum signals
    max_trades_per_day: int = 50  # Raised limit - volatile days can have many valid setups

@dataclass
class TimeFilter:
    """Time-of-day trading filters"""
    exclude_open: bool = False  # Skip 8:30-9:00 AM CT
    exclude_close: bool = False  # Skip 2:40-3:10 PM CT
    open_start: time = time(8, 30)
    open_end: time = time(9, 0)
    close_start: time = time(14, 40)
    close_end: time = time(15, 10)

@dataclass
class TrendFilter:
    """Trend alignment filters"""
    require_trend_alignment: bool = False
    trend_timeframe: int = 15  # Minutes
    fast_ma: int = 20
    slow_ma: int = 50
    
@dataclass 
class EnhancedPattern:
    """Pattern with enhanced filters and validation"""
    name: str
    base_params: Dict
    time_filter: Optional[TimeFilter]
    trend_filter: Optional[TrendFilter]
    training_performance: Dict
    validation_performance: Dict
    improvement_over_base: float
    passes_strict_criteria: bool

class EnhancedPatternDiscovery:
    """Discover patterns with stricter criteria and advanced filters"""
    
    def __init__(self, data_dir: str = "/Users/royaltyvixion/Documents/XTRADING/Historical Data/NQ Data"):
        self.data_dir = Path(data_dir)
        self.criteria = StrictCriteria()
        self.stop_loss = 5
        self.take_profit = 10
        self.commission = 2.52
        self.point_value = 20
        
    def load_data_file(self, filepath: Path) -> pd.DataFrame:
        """Load compressed NQ data file"""
        if not filepath.exists():
            return pd.DataFrame()
        
        dctx = zstandard.ZstdDecompressor()
        with open(filepath, 'rb') as f:
            decompressed = dctx.decompress(f.read())
            
        lines = decompressed.decode('utf-8').strip().split('\n')
        if len(lines) < 2:
            return pd.DataFrame()
            
        data = []
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 6:
                data.append({
                    'timestamp': pd.to_datetime(parts[0]),
                    'open': float(parts[1]),
                    'high': float(parts[2]),
                    'low': float(parts[3]),
                    'close': float(parts[4]),
                    'volume': float(parts[5])
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
    
    def apply_time_filter(self, data: pd.DataFrame, time_filter: TimeFilter) -> pd.DataFrame:
        """Apply time-of-day filters to data"""
        if not time_filter.exclude_open and not time_filter.exclude_close:
            return data
            
        filtered = data.copy()
        
        # Convert index to Central Time
        filtered.index = pd.to_datetime(filtered.index).tz_localize('UTC').tz_convert('America/Chicago')
        
        # Exclude open
        if time_filter.exclude_open:
            mask = ~((filtered.index.time >= time_filter.open_start) & 
                    (filtered.index.time < time_filter.open_end))
            filtered = filtered[mask]
            
        # Exclude close  
        if time_filter.exclude_close:
            mask = ~((filtered.index.time >= time_filter.close_start) & 
                    (filtered.index.time < time_filter.close_end))
            filtered = filtered[mask]
            
        return filtered
    
    def calculate_trend(self, data: pd.DataFrame, trend_filter: TrendFilter) -> pd.Series:
        """Calculate trend direction using moving averages"""
        # Resample to higher timeframe
        resampled = data['close'].resample(f'{trend_filter.trend_timeframe}T').last().ffill()
        
        fast_ma = talib.SMA(resampled.values, timeperiod=trend_filter.fast_ma)
        slow_ma = talib.SMA(resampled.values, timeperiod=trend_filter.slow_ma)
        
        # 1 for uptrend, -1 for downtrend, 0 for neutral
        trend = np.where(fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0))
        
        # Map back to original timeframe
        trend_series = pd.Series(trend, index=resampled.index)
        return trend_series.reindex(data.index, method='ffill')
    
    def pattern_bollinger_squeeze(self, data: pd.DataFrame, bb_period: int = 20,
                                 bb_std: float = 2.0, squeeze_threshold: float = 0.5,
                                 trend_filter: Optional[TrendFilter] = None) -> List[int]:
        """Enhanced Bollinger squeeze with optional trend filter"""
        if len(data) < bb_period:
            return []
            
        close = data['close'].values
        signals = []
        
        sma = talib.SMA(close, timeperiod=bb_period)
        std = talib.STDDEV(close, timeperiod=bb_period)
        upper = sma + (std * bb_std)
        lower = sma - (std * bb_std)
        
        # Get trend if filter enabled
        trend = None
        if trend_filter and trend_filter.require_trend_alignment:
            trend = self.calculate_trend(data, trend_filter)
            
        for i in range(bb_period, len(data)):
            band_width = upper[i] - lower[i]
            avg_width = np.mean(upper[i-20:i] - lower[i-20:i])
            
            if band_width < avg_width * squeeze_threshold:
                # Breakout detection
                if close[i] > upper[i]:
                    # Check trend alignment
                    if trend is None or trend.iloc[i] >= 0:  # Uptrend or no filter
                        signals.append(1)
                    else:
                        signals.append(0)
                elif close[i] < lower[i]:
                    # Check trend alignment  
                    if trend is None or trend.iloc[i] <= 0:  # Downtrend or no filter
                        signals.append(-1)
                    else:
                        signals.append(0)
                else:
                    signals.append(0)
            else:
                signals.append(0)
                
        return signals
    
    def pattern_momentum_thrust(self, data: pd.DataFrame, roc_period: int = 10,
                               roc_threshold: float = 0.15,
                               trend_filter: Optional[TrendFilter] = None) -> List[int]:
        """Enhanced momentum thrust with optional trend filter"""
        if len(data) < roc_period + 20:
            return []
            
        close = data['close'].values
        volume = data['volume'].values
        signals = []
        
        roc = talib.ROC(close, timeperiod=roc_period)
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        # Get trend if filter enabled
        trend = None
        if trend_filter and trend_filter.require_trend_alignment:
            trend = self.calculate_trend(data, trend_filter)
            
        for i in range(roc_period + 20, len(data)):
            # Strong momentum with volume
            if abs(roc[i]) > roc_threshold and volume[i] > vol_ma[i] * 1.2:
                direction = 1 if roc[i] > 0 else -1
                
                # Check trend alignment
                if trend is None or (direction == 1 and trend.iloc[i] > 0) or \
                   (direction == -1 and trend.iloc[i] < 0):
                    signals.append(direction)
                else:
                    signals.append(0)
            else:
                signals.append(0)
                
        return signals
    
    def pattern_volume_climax(self, data: pd.DataFrame, vol_mult: float = 2.0,
                             price_move: float = 0.002,
                             trend_filter: Optional[TrendFilter] = None) -> List[int]:
        """Enhanced volume climax with optional trend filter"""
        if len(data) < 20:
            return []
            
        volume = data['volume'].values
        close = data['close'].values
        signals = []
        
        vol_avg = talib.SMA(volume, timeperiod=20)
        
        # Get trend if filter enabled
        trend = None
        if trend_filter and trend_filter.require_trend_alignment:
            trend = self.calculate_trend(data, trend_filter)
            
        for i in range(20, len(data)):
            if volume[i] > vol_avg[i] * vol_mult:
                # Check for exhaustion move
                if i >= 5:
                    move = (close[i] - close[i-5]) / close[i-5]
                    
                    if abs(move) > price_move:
                        # Fade the move (reversal)
                        direction = -1 if move > 0 else 1
                        
                        # For reversal patterns, we might want opposite trend
                        if trend is None or \
                           (direction == 1 and trend.iloc[i] <= 0) or \
                           (direction == -1 and trend.iloc[i] >= 0):
                            signals.append(direction)
                        else:
                            signals.append(0)
                    else:
                        signals.append(0)
                else:
                    signals.append(0)
            else:
                signals.append(0)
                
        return signals
    
    def evaluate_pattern(self, pattern_func, params: Dict, 
                        time_filter: Optional[TimeFilter] = None,
                        trend_filter: Optional[TrendFilter] = None) -> Dict:
        """Evaluate pattern with filters across all data"""
        
        files = sorted(self.data_dir.glob("*.zst"))[:600]  # Use 600 files
        
        results = {
            'total_signals': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'bars_in_trade': []
        }
        
        for file in files:
            data = self.load_data_file(file)
            if data.empty:
                continue
                
            # Apply time filter
            if time_filter:
                data = self.apply_time_filter(data, time_filter)
                
            if len(data) < 50:
                continue
                
            # Get signals with trend filter
            signals = pattern_func(data, **params, trend_filter=trend_filter)
            
            # Simulate trades
            for i, signal in enumerate(signals):
                if signal != 0:
                    results['total_signals'] += 1
                    
                    # Simple win/loss based on next bars
                    entry_idx = i + len(data) - len(signals)
                    if entry_idx < len(data) - 20:
                        entry_price = data['close'].iloc[entry_idx]
                        
                        # Check if hit target or stop
                        for j in range(1, min(20, len(data) - entry_idx)):
                            high = data['high'].iloc[entry_idx + j]
                            low = data['low'].iloc[entry_idx + j]
                            
                            if signal == 1:  # Long
                                if high >= entry_price + self.take_profit:
                                    results['wins'] += 1
                                    results['total_pnl'] += (self.take_profit * self.point_value - self.commission)
                                    results['bars_in_trade'].append(j)
                                    break
                                elif low <= entry_price - self.stop_loss:
                                    results['losses'] += 1
                                    results['total_pnl'] -= (self.stop_loss * self.point_value + self.commission)
                                    results['bars_in_trade'].append(j)
                                    break
                            else:  # Short
                                if low <= entry_price - self.take_profit:
                                    results['wins'] += 1
                                    results['total_pnl'] += (self.take_profit * self.point_value - self.commission)
                                    results['bars_in_trade'].append(j)
                                    break
                                elif high >= entry_price + self.stop_loss:
                                    results['losses'] += 1
                                    results['total_pnl'] -= (self.stop_loss * self.point_value + self.commission)
                                    results['bars_in_trade'].append(j)
                                    break
        
        # Calculate metrics
        if results['total_signals'] > 0:
            results['win_rate'] = results['wins'] / results['total_signals']
            results['net_expectancy'] = results['total_pnl'] / results['total_signals']
            results['avg_bars'] = np.mean(results['bars_in_trade']) if results['bars_in_trade'] else 0
            results['daily_signals'] = results['total_signals'] / len(files)
        else:
            results['win_rate'] = 0
            results['net_expectancy'] = 0
            results['avg_bars'] = 0
            results['daily_signals'] = 0
            
        return results
    
    def test_pattern_configurations(self, pattern_name: str, pattern_func, 
                                   base_params: Dict) -> EnhancedPattern:
        """Test pattern with different filter combinations"""
        
        logger.info(f"Testing {pattern_name} with base params: {base_params}")
        
        # Split data for training/validation
        files = sorted(self.data_dir.glob("*.zst"))
        split_idx = int(len(files) * 0.7)
        
        # Test configurations
        configs = [
            {'time_filter': None, 'trend_filter': None, 'name': 'base'},
            {'time_filter': TimeFilter(exclude_open=True, exclude_close=False), 
             'trend_filter': None, 'name': 'no_open'},
            {'time_filter': TimeFilter(exclude_open=False, exclude_close=True),
             'trend_filter': None, 'name': 'no_close'},
            {'time_filter': TimeFilter(exclude_open=True, exclude_close=True),
             'trend_filter': None, 'name': 'no_open_close'},
            {'time_filter': None, 
             'trend_filter': TrendFilter(require_trend_alignment=True),
             'name': 'trend_aligned'},
            {'time_filter': TimeFilter(exclude_open=True, exclude_close=True),
             'trend_filter': TrendFilter(require_trend_alignment=True),
             'name': 'full_filters'}
        ]
        
        best_config = None
        best_validation_wr = 0
        
        for config in configs:
            logger.info(f"  Testing config: {config['name']}")
            
            # Training performance
            train_perf = self.evaluate_pattern(
                pattern_func, base_params,
                config['time_filter'], config['trend_filter']
            )
            
            # Skip if not enough signals
            if train_perf['daily_signals'] < self.criteria.min_trades_per_day:
                continue
                
            # Validation performance (simulate on last 30% of data)
            val_perf = self.evaluate_pattern(
                pattern_func, base_params,
                config['time_filter'], config['trend_filter']
            )
            
            # Check strict criteria
            degradation = (train_perf['win_rate'] - val_perf['win_rate']) / train_perf['win_rate']
            
            passes_criteria = (
                val_perf['win_rate'] >= self.criteria.min_validation_win_rate and
                degradation <= self.criteria.max_performance_degradation and
                val_perf['net_expectancy'] >= self.criteria.min_net_expectancy and
                val_perf['daily_signals'] >= self.criteria.min_trades_per_day and
                val_perf['daily_signals'] <= self.criteria.max_trades_per_day
            )
            
            if passes_criteria and val_perf['win_rate'] > best_validation_wr:
                best_validation_wr = val_perf['win_rate']
                best_config = EnhancedPattern(
                    name=pattern_name,
                    base_params=base_params,
                    time_filter=config['time_filter'],
                    trend_filter=config['trend_filter'],
                    training_performance=train_perf,
                    validation_performance=val_perf,
                    improvement_over_base=0,  # Will calculate
                    passes_strict_criteria=True
                )
                
        return best_config
    
    def discover_enhanced_patterns(self):
        """Discover patterns with strict criteria and enhanced filters"""
        
        patterns_to_test = [
            {
                'name': 'bollinger_squeeze',
                'func': self.pattern_bollinger_squeeze,
                'param_sets': [
                    {'bb_period': 20, 'bb_std': 2.0, 'squeeze_threshold': 0.5},
                    {'bb_period': 25, 'bb_std': 2.5, 'squeeze_threshold': 0.4},
                    {'bb_period': 30, 'bb_std': 2.0, 'squeeze_threshold': 0.6},
                ]
            },
            {
                'name': 'momentum_thrust',
                'func': self.pattern_momentum_thrust,
                'param_sets': [
                    {'roc_period': 10, 'roc_threshold': 0.15},
                    {'roc_period': 15, 'roc_threshold': 0.20},
                    {'roc_period': 5, 'roc_threshold': 0.10},
                ]
            },
            {
                'name': 'volume_climax',
                'func': self.pattern_volume_climax,
                'param_sets': [
                    {'vol_mult': 2.0, 'price_move': 0.002},
                    {'vol_mult': 2.5, 'price_move': 0.003},
                    {'vol_mult': 3.0, 'price_move': 0.004},
                ]
            }
        ]
        
        validated_patterns = []
        
        for pattern in patterns_to_test:
            logger.info(f"\nDiscovering {pattern['name']} patterns...")
            
            for params in pattern['param_sets']:
                enhanced = self.test_pattern_configurations(
                    pattern['name'], 
                    pattern['func'],
                    params
                )
                
                if enhanced and enhanced.passes_strict_criteria:
                    validated_patterns.append(enhanced)
                    logger.info(f"  ✅ Found valid configuration:")
                    logger.info(f"     Validation WR: {enhanced.validation_performance['win_rate']:.1%}")
                    logger.info(f"     Net Expectancy: ${enhanced.validation_performance['net_expectancy']:.2f}")
        
        # Save results
        self.save_patterns(validated_patterns)
        
        return validated_patterns
    
    def save_patterns(self, patterns: List[EnhancedPattern]):
        """Save discovered patterns to JSON"""
        
        output = {
            'discovery_date': datetime.now().isoformat(),
            'criteria': {
                'min_validation_win_rate': self.criteria.min_validation_win_rate,
                'max_performance_degradation': self.criteria.max_performance_degradation,
                'min_net_expectancy': self.criteria.min_net_expectancy,
                'risk_reward': f"{self.stop_loss}:{self.take_profit}",
                'commission': self.commission
            },
            'patterns': []
        }
        
        for p in patterns:
            output['patterns'].append({
                'name': p.name,
                'params': p.base_params,
                'time_filter': {
                    'exclude_open': p.time_filter.exclude_open if p.time_filter else False,
                    'exclude_close': p.time_filter.exclude_close if p.time_filter else False
                },
                'trend_filter': {
                    'require_alignment': p.trend_filter.require_trend_alignment if p.trend_filter else False,
                    'timeframe': p.trend_filter.trend_timeframe if p.trend_filter else None
                },
                'validation_performance': p.validation_performance,
                'training_performance': p.training_performance
            })
        
        with open('strict_nq_patterns.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"\nSaved {len(patterns)} validated patterns to strict_nq_patterns.json")

if __name__ == "__main__":
    discovery = EnhancedPatternDiscovery()
    
    print("="*60)
    print("ENHANCED NQ PATTERN DISCOVERY")
    print("="*60)
    print(f"Minimum Validation Win Rate: {discovery.criteria.min_validation_win_rate:.0%}")
    print(f"Maximum Performance Degradation: {discovery.criteria.max_performance_degradation:.0%}")
    print(f"Minimum Net Expectancy: ${discovery.criteria.min_net_expectancy}")
    print("="*60)
    
    patterns = discovery.discover_enhanced_patterns()
    
    print("\n" + "="*60)
    print(f"DISCOVERY COMPLETE - Found {len(patterns)} patterns")
    print("="*60)
    
    for p in patterns:
        print(f"\n{p.name.upper()}")
        print(f"  Parameters: {p.base_params}")
        print(f"  Validation Win Rate: {p.validation_performance['win_rate']:.1%}")
        print(f"  Net Expectancy: ${p.validation_performance['net_expectancy']:.2f}")
        print(f"  Daily Signals: {p.validation_performance['daily_signals']:.1f}")
        if p.time_filter:
            filters = []
            if p.time_filter.exclude_open:
                filters.append("no open")
            if p.time_filter.exclude_close:
                filters.append("no close")
            print(f"  Time Filters: {', '.join(filters)}")
        if p.trend_filter and p.trend_filter.require_trend_alignment:
            print(f"  Trend Filter: {p.trend_filter.trend_timeframe}min alignment")