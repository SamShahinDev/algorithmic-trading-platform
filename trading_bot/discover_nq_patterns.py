#!/usr/bin/env python3
"""
NQ Pattern Discovery Script
Mines 2 years of NQ historical data to discover profitable patterns
"""

import os
import sys
import pandas as pd
import numpy as np
import zstandard as zstd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from analysis.strategy_discovery import StrategyDiscovery
from analysis.pattern_scanner import PatternScanner
from data.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pattern_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NQPatternDiscovery:
    """Discover NQ-specific profitable patterns from historical data"""
    
    def __init__(self):
        self.data_dir = Path("/Users/royaltyvixion/Documents/XTRADING/Historical Data/NQ Data/")
        self.commission = 2.52  # Round-trip commission for NQ
        self.point_value = 20  # $20 per point for NQ
        
        # Date ranges for train/test split
        self.train_start = datetime(2023, 7, 26)
        self.train_end = datetime(2025, 5, 31)
        self.test_start = datetime(2025, 6, 1)
        self.test_end = datetime(2025, 8, 25)
        
        self.strategy_discovery = StrategyDiscovery(
            min_trades=30,
            min_sharpe=1.2,  # Lower for intraday
            min_win_rate=0.48,  # Realistic for futures
            min_profit_factor=1.3
        )
        
    def load_compressed_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single compressed CSV file"""
        try:
            dctx = zstd.ZstdDecompressor()
            with open(filepath, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    df = pd.read_csv(reader)
                    
            # Parse and filter for NQ
            if 'symbol' in df.columns:
                df = df[df['symbol'].str.contains('NQ', na=False)]
            
            # Ensure datetime column
            if 'ts_event' in df.columns:
                df['timestamp'] = pd.to_datetime(df['ts_event'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Standard columns
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Keep only needed columns
            columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]
            
            return df
            
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def load_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load all historical data within date range"""
        logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")
        
        all_data = []
        files = sorted(self.data_dir.glob("glbx-mdp3-*.ohlcv-1m.csv.zst"))
        
        for file in files:
            # Extract date from filename
            date_str = file.stem.split('-')[2][:8]
            file_date = datetime.strptime(date_str, '%Y%m%d')
            
            # Check if file is in our date range
            if start_date <= file_date <= end_date:
                logger.debug(f"Loading {file.name}")
                df = self.load_compressed_file(file)
                if not df.empty:
                    all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values('timestamp')
            logger.info(f"Loaded {len(combined)} total bars from {len(all_data)} files")
            return combined
        else:
            logger.warning("No data loaded")
            return pd.DataFrame()
    
    def discover_momentum_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Discover momentum-based patterns"""
        patterns = []
        
        # Test different momentum parameters
        lookback_periods = [5, 10, 15, 20, 30]
        volume_multipliers = [1.5, 2.0, 2.5, 3.0]
        roc_thresholds = [0.001, 0.002, 0.003, 0.005]
        
        for lookback in lookback_periods:
            for vol_mult in volume_multipliers:
                for roc_thresh in roc_thresholds:
                    pattern = {
                        'type': 'momentum_burst',
                        'lookback': lookback,
                        'volume_multiplier': vol_mult,
                        'roc_threshold': roc_thresh,
                        'signals': self._test_momentum_pattern(
                            data, lookback, vol_mult, roc_thresh
                        )
                    }
                    
                    # Calculate performance
                    perf = self._calculate_pattern_performance(data, pattern['signals'])
                    if self._is_profitable_pattern(perf):
                        pattern['performance'] = perf
                        patterns.append(pattern)
                        logger.info(f"Found profitable momentum pattern: {perf}")
        
        return patterns
    
    def _test_momentum_pattern(self, data: pd.DataFrame, 
                               lookback: int, vol_mult: float, 
                               roc_thresh: float) -> pd.Series:
        """Test a specific momentum pattern configuration"""
        # Calculate rate of change
        data['roc'] = data['close'].pct_change(lookback)
        
        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Generate signals
        long_signal = (
            (data['roc'] > roc_thresh) &
            (data['volume'] > data['volume_ma'] * vol_mult)
        )
        
        short_signal = (
            (data['roc'] < -roc_thresh) &
            (data['volume'] > data['volume_ma'] * vol_mult)
        )
        
        signals = pd.Series(0, index=data.index)
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        return signals
    
    def discover_mean_reversion_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Discover mean reversion patterns"""
        patterns = []
        
        # Test different parameters
        bb_periods = [10, 15, 20, 25, 30]
        bb_stds = [1.5, 2.0, 2.5, 3.0]
        rsi_periods = [7, 14, 21]
        rsi_oversold = [20, 25, 30]
        rsi_overbought = [70, 75, 80]
        
        for bb_period in bb_periods:
            for bb_std in bb_stds:
                for rsi_period in rsi_periods:
                    for rsi_os in rsi_oversold:
                        for rsi_ob in rsi_overbought:
                            pattern = {
                                'type': 'mean_reversion',
                                'bb_period': bb_period,
                                'bb_std': bb_std,
                                'rsi_period': rsi_period,
                                'rsi_oversold': rsi_os,
                                'rsi_overbought': rsi_ob,
                                'signals': self._test_mean_reversion_pattern(
                                    data, bb_period, bb_std, rsi_period, rsi_os, rsi_ob
                                )
                            }
                            
                            perf = self._calculate_pattern_performance(data, pattern['signals'])
                            if self._is_profitable_pattern(perf):
                                pattern['performance'] = perf
                                patterns.append(pattern)
                                logger.info(f"Found profitable mean reversion pattern: {perf}")
        
        return patterns
    
    def _test_mean_reversion_pattern(self, data: pd.DataFrame,
                                    bb_period: int, bb_std: float,
                                    rsi_period: int, rsi_oversold: float,
                                    rsi_overbought: float) -> pd.Series:
        """Test mean reversion pattern"""
        # Calculate Bollinger Bands
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        data['bb_std'] = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (bb_std * data['bb_std'])
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        long_signal = (
            (data['close'] < data['bb_lower']) &
            (data['rsi'] < rsi_oversold)
        )
        
        short_signal = (
            (data['close'] > data['bb_upper']) &
            (data['rsi'] > rsi_overbought)
        )
        
        signals = pd.Series(0, index=data.index)
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        return signals
    
    def discover_breakout_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Discover breakout patterns"""
        patterns = []
        
        # Test different parameters
        lookback_periods = [10, 20, 30, 50]
        volume_confirms = [1.2, 1.5, 2.0]
        breakout_factors = [1.001, 1.002, 1.003]
        
        for lookback in lookback_periods:
            for vol_conf in volume_confirms:
                for break_factor in breakout_factors:
                    pattern = {
                        'type': 'breakout',
                        'lookback': lookback,
                        'volume_confirm': vol_conf,
                        'breakout_factor': break_factor,
                        'signals': self._test_breakout_pattern(
                            data, lookback, vol_conf, break_factor
                        )
                    }
                    
                    perf = self._calculate_pattern_performance(data, pattern['signals'])
                    if self._is_profitable_pattern(perf):
                        pattern['performance'] = perf
                        patterns.append(pattern)
                        logger.info(f"Found profitable breakout pattern: {perf}")
        
        return patterns
    
    def _test_breakout_pattern(self, data: pd.DataFrame,
                               lookback: int, vol_confirm: float,
                               break_factor: float) -> pd.Series:
        """Test breakout pattern"""
        # Calculate resistance/support levels
        data['resistance'] = data['high'].rolling(lookback).max()
        data['support'] = data['low'].rolling(lookback).min()
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Generate signals
        long_signal = (
            (data['close'] > data['resistance'] * break_factor) &
            (data['volume'] > data['volume_ma'] * vol_confirm)
        )
        
        short_signal = (
            (data['close'] < data['support'] / break_factor) &
            (data['volume'] > data['volume_ma'] * vol_confirm)
        )
        
        signals = pd.Series(0, index=data.index)
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        return signals
    
    def _calculate_pattern_performance(self, data: pd.DataFrame, 
                                      signals: pd.Series) -> Dict:
        """Calculate pattern performance metrics"""
        # Create a copy to avoid modifying original
        df = data.copy()
        df['signal'] = signals
        df['returns'] = df['close'].pct_change()
        
        # Calculate positions (hold until opposite signal or 30 bars)
        df['position'] = 0
        current_pos = 0
        bars_in_trade = 0
        max_bars = 30  # Maximum bars to hold
        
        trades = []
        entry_price = 0
        entry_idx = 0
        
        for i in range(len(df)):
            if df['signal'].iloc[i] != 0 and current_pos == 0:
                # Enter position
                current_pos = df['signal'].iloc[i]
                entry_price = df['close'].iloc[i]
                entry_idx = i
                bars_in_trade = 0
                
            elif current_pos != 0:
                bars_in_trade += 1
                
                # Exit conditions
                exit_signal = (df['signal'].iloc[i] == -current_pos) or (bars_in_trade >= max_bars)
                
                if exit_signal or i == len(df) - 1:
                    # Exit position
                    exit_price = df['close'].iloc[i]
                    # Calculate points gained/lost
                    points = (exit_price - entry_price) * current_pos
                    # Convert to dollars (1 NQ point = $20) and subtract commission
                    pnl = (points * self.point_value) - self.commission
                    
                    trades.append({
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'bars': bars_in_trade,
                        'direction': current_pos
                    })
                    
                    current_pos = 0
                    bars_in_trade = 0
        
        # Calculate metrics
        if len(trades) > 0:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(trades) if trades else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else 0
            
            # Calculate Sharpe ratio (annualized)
            if len(pnls) > 1:
                returns = pd.Series(pnls) / 10000  # Normalize by account size
                sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe = 0
            
            max_dd = self._calculate_max_drawdown(pnls)
            
            return {
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_pnl': sum(pnls),
                'expectancy': np.mean(pnls) if pnls else 0
            }
        else:
            return {
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_pnl': 0,
                'expectancy': 0
            }
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumsum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max)
        return abs(min(drawdown)) if len(drawdown) > 0 else 0
    
    def _is_profitable_pattern(self, performance: Dict) -> bool:
        """Check if pattern meets profitability criteria"""
        return (
            performance['trades'] >= 30 and
            performance['win_rate'] >= 0.48 and
            performance['profit_factor'] >= 1.3 and
            performance['sharpe_ratio'] >= 1.0 and
            performance['expectancy'] > self.commission
        )
    
    def validate_patterns(self, patterns: List[Dict], test_data: pd.DataFrame) -> List[Dict]:
        """Validate patterns on out-of-sample data"""
        logger.info(f"Validating {len(patterns)} patterns on test data")
        
        validated = []
        for pattern in patterns:
            # Re-test pattern on test data
            if pattern['type'] == 'momentum_burst':
                signals = self._test_momentum_pattern(
                    test_data, 
                    pattern['lookback'],
                    pattern['volume_multiplier'],
                    pattern['roc_threshold']
                )
            elif pattern['type'] == 'mean_reversion':
                signals = self._test_mean_reversion_pattern(
                    test_data,
                    pattern['bb_period'],
                    pattern['bb_std'],
                    pattern['rsi_period'],
                    pattern['rsi_oversold'],
                    pattern['rsi_overbought']
                )
            elif pattern['type'] == 'breakout':
                signals = self._test_breakout_pattern(
                    test_data,
                    pattern['lookback'],
                    pattern['volume_confirm'],
                    pattern['breakout_factor']
                )
            
            test_perf = self._calculate_pattern_performance(test_data, signals)
            
            # Check if still profitable on test data
            if self._is_profitable_pattern(test_perf):
                pattern['test_performance'] = test_perf
                validated.append(pattern)
                logger.info(f"Pattern validated: Train PF={pattern['performance']['profit_factor']:.2f}, "
                          f"Test PF={test_perf['profit_factor']:.2f}")
        
        return validated
    
    def run_discovery(self):
        """Run complete pattern discovery process"""
        logger.info("Starting NQ pattern discovery process")
        
        # Load training data
        logger.info("Loading training data...")
        train_data = self.load_historical_data(self.train_start, self.train_end)
        
        if train_data.empty:
            logger.error("Failed to load training data")
            return
        
        # Discover patterns
        all_patterns = []
        
        logger.info("Discovering momentum patterns...")
        momentum_patterns = self.discover_momentum_patterns(train_data)
        all_patterns.extend(momentum_patterns)
        
        logger.info("Discovering mean reversion patterns...")
        mean_rev_patterns = self.discover_mean_reversion_patterns(train_data)
        all_patterns.extend(mean_rev_patterns)
        
        logger.info("Discovering breakout patterns...")
        breakout_patterns = self.discover_breakout_patterns(train_data)
        all_patterns.extend(breakout_patterns)
        
        logger.info(f"Found {len(all_patterns)} profitable patterns in training data")
        
        # Load test data for validation
        logger.info("Loading test data for validation...")
        test_data = self.load_historical_data(self.test_start, self.test_end)
        
        if test_data.empty:
            logger.error("Failed to load test data")
            return
        
        # Validate patterns
        validated_patterns = self.validate_patterns(all_patterns, test_data)
        
        logger.info(f"Validated {len(validated_patterns)} patterns on out-of-sample data")
        
        # Sort by profit factor
        validated_patterns.sort(key=lambda x: x['test_performance']['profit_factor'], reverse=True)
        
        # Save top patterns
        top_patterns = validated_patterns[:10]  # Top 10 patterns
        
        output = {
            'discovery_date': datetime.now().isoformat(),
            'train_period': f"{self.train_start.date()} to {self.train_end.date()}",
            'test_period': f"{self.test_start.date()} to {self.test_end.date()}",
            'patterns': []
        }
        
        for i, pattern in enumerate(top_patterns, 1):
            output['patterns'].append({
                'rank': i,
                'type': pattern['type'],
                'parameters': {k: v for k, v in pattern.items() 
                             if k not in ['signals', 'performance', 'test_performance', 'type']},
                'train_performance': pattern['performance'],
                'test_performance': pattern['test_performance']
            })
        
        # Save to file
        with open('discovered_nq_patterns.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved top {len(top_patterns)} patterns to discovered_nq_patterns.json")
        
        # Print summary
        print("\n" + "="*80)
        print("TOP DISCOVERED NQ PATTERNS")
        print("="*80)
        
        for pattern_info in output['patterns']:
            print(f"\nRank #{pattern_info['rank']}: {pattern_info['type'].upper()}")
            print(f"Parameters: {pattern_info['parameters']}")
            print(f"Training Performance:")
            print(f"  - Win Rate: {pattern_info['train_performance']['win_rate']:.1%}")
            print(f"  - Profit Factor: {pattern_info['train_performance']['profit_factor']:.2f}")
            print(f"  - Sharpe Ratio: {pattern_info['train_performance']['sharpe_ratio']:.2f}")
            print(f"  - Total P&L: ${pattern_info['train_performance']['total_pnl']:.2f}")
            print(f"Test Performance (Out-of-Sample):")
            print(f"  - Win Rate: {pattern_info['test_performance']['win_rate']:.1%}")
            print(f"  - Profit Factor: {pattern_info['test_performance']['profit_factor']:.2f}")
            print(f"  - Sharpe Ratio: {pattern_info['test_performance']['sharpe_ratio']:.2f}")
            print(f"  - Total P&L: ${pattern_info['test_performance']['total_pnl']:.2f}")
            print("-" * 40)
        
        return validated_patterns


if __name__ == "__main__":
    discovery = NQPatternDiscovery()
    discovery.run_discovery()