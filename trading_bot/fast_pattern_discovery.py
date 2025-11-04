#!/usr/bin/env python3
"""
Fast NQ Pattern Discovery - Optimized version with fewer parameter combinations
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
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_pattern_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastNQPatternDiscovery:
    """Fast discovery of NQ patterns with optimized parameter search"""
    
    def __init__(self):
        self.data_dir = Path("/Users/royaltyvixion/Documents/XTRADING/Historical Data/NQ Data/")
        self.commission = 2.52  # Round-trip commission for NQ
        self.point_value = 20  # $20 per point for NQ
        
        # Use recent data for faster processing
        self.train_start = datetime(2024, 6, 1)
        self.train_end = datetime(2025, 5, 31)
        self.test_start = datetime(2025, 6, 1)
        self.test_end = datetime(2025, 8, 25)
        
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
                df = self.load_compressed_file(file)
                if not df.empty:
                    all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values('timestamp')
            logger.info(f"Loaded {len(combined)} bars from {len(all_data)} files")
            return combined
        else:
            return pd.DataFrame()
    
    def test_pattern(self, data: pd.DataFrame, pattern_type: str, params: Dict) -> pd.Series:
        """Test a specific pattern configuration"""
        signals = pd.Series(0, index=data.index)
        
        if pattern_type == 'momentum':
            # Rate of change with volume confirmation
            lookback = params['lookback']
            vol_mult = params['vol_mult']
            roc_thresh = params['roc_thresh']
            
            data['roc'] = data['close'].pct_change(lookback)
            data['vol_ma'] = data['volume'].rolling(20).mean()
            
            long_sig = (data['roc'] > roc_thresh) & (data['volume'] > data['vol_ma'] * vol_mult)
            short_sig = (data['roc'] < -roc_thresh) & (data['volume'] > data['vol_ma'] * vol_mult)
            
            signals[long_sig] = 1
            signals[short_sig] = -1
            
        elif pattern_type == 'mean_reversion':
            # Bollinger Band + RSI reversal
            bb_period = params['bb_period']
            bb_std = params['bb_std']
            rsi_period = params['rsi_period']
            
            # Bollinger Bands
            sma = data['close'].rolling(bb_period).mean()
            std = data['close'].rolling(bb_period).std()
            upper = sma + (bb_std * std)
            lower = sma - (bb_std * std)
            
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            long_sig = (data['close'] < lower) & (rsi < 30)
            short_sig = (data['close'] > upper) & (rsi > 70)
            
            signals[long_sig] = 1
            signals[short_sig] = -1
            
        elif pattern_type == 'breakout':
            # Price breakout with volume
            lookback = params['lookback']
            vol_conf = params['vol_conf']
            
            resistance = data['high'].rolling(lookback).max()
            support = data['low'].rolling(lookback).min()
            vol_ma = data['volume'].rolling(20).mean()
            
            long_sig = (data['close'] > resistance * 1.001) & (data['volume'] > vol_ma * vol_conf)
            short_sig = (data['close'] < support * 0.999) & (data['volume'] > vol_ma * vol_conf)
            
            signals[long_sig] = 1
            signals[short_sig] = -1
            
        elif pattern_type == 'volume_spike':
            # Volume spike reversal
            vol_mult = params['vol_mult']
            price_move = params['price_move']
            
            vol_ma = data['volume'].rolling(20).mean()
            price_change = data['close'].pct_change(5)
            
            # Fade extreme moves with high volume
            long_sig = (data['volume'] > vol_ma * vol_mult) & (price_change < -price_move)
            short_sig = (data['volume'] > vol_ma * vol_mult) & (price_change > price_move)
            
            signals[long_sig] = 1
            signals[short_sig] = -1
            
        return signals
    
    def calculate_performance(self, data: pd.DataFrame, signals: pd.Series) -> Dict:
        """Calculate realistic performance metrics"""
        df = data.copy()
        df['signal'] = signals
        
        # Track positions and trades
        position = 0
        trades = []
        entry_price = 0
        entry_time = None
        max_hold = 30  # Maximum 30 bars
        
        for i in range(len(df)):
            # Entry logic
            if position == 0 and df['signal'].iloc[i] != 0:
                position = df['signal'].iloc[i]
                entry_price = df['close'].iloc[i]
                entry_time = i
                
            # Exit logic
            elif position != 0:
                bars_held = i - entry_time
                
                # Exit on opposite signal or max hold
                should_exit = (df['signal'].iloc[i] == -position) or (bars_held >= max_hold)
                
                if should_exit or i == len(df) - 1:
                    exit_price = df['close'].iloc[i]
                    
                    # Calculate P&L in points then dollars
                    points = (exit_price - entry_price) * position
                    pnl = (points * self.point_value) - self.commission
                    
                    trades.append({
                        'pnl': pnl,
                        'points': points,
                        'bars': bars_held
                    })
                    
                    position = 0
        
        # Calculate metrics
        if len(trades) >= 10:  # Minimum trades for statistics
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            metrics = {
                'trades': len(trades),
                'win_rate': len(wins) / len(trades),
                'avg_win': np.mean(wins) if wins else 0,
                'avg_loss': abs(np.mean(losses)) if losses else 0,
                'profit_factor': sum(wins) / abs(sum(losses)) if losses else 0,
                'total_pnl': sum(pnls),
                'expectancy': np.mean(pnls),
                'avg_bars': np.mean([t['bars'] for t in trades])
            }
            
            # Calculate Sharpe ratio
            if len(pnls) > 1:
                returns = pd.Series(pnls)
                sharpe = np.sqrt(252 * 6.5 * 60) * returns.mean() / returns.std() if returns.std() > 0 else 0
                metrics['sharpe'] = sharpe
            else:
                metrics['sharpe'] = 0
                
            return metrics
        else:
            return None
    
    def discover_patterns(self):
        """Discover profitable patterns"""
        logger.info("Starting pattern discovery...")
        
        # Load training data
        train_data = self.load_historical_data(self.train_start, self.train_end)
        if train_data.empty:
            logger.error("No training data loaded")
            return
        
        # Define parameter grids (reduced for speed)
        pattern_configs = {
            'momentum': [
                {'lookback': 10, 'vol_mult': 1.5, 'roc_thresh': 0.002},
                {'lookback': 15, 'vol_mult': 2.0, 'roc_thresh': 0.003},
                {'lookback': 20, 'vol_mult': 2.5, 'roc_thresh': 0.004},
            ],
            'mean_reversion': [
                {'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14},
                {'bb_period': 15, 'bb_std': 2.5, 'rsi_period': 10},
                {'bb_period': 25, 'bb_std': 1.5, 'rsi_period': 21},
            ],
            'breakout': [
                {'lookback': 20, 'vol_conf': 1.5},
                {'lookback': 30, 'vol_conf': 2.0},
                {'lookback': 15, 'vol_conf': 1.2},
            ],
            'volume_spike': [
                {'vol_mult': 3.0, 'price_move': 0.003},
                {'vol_mult': 4.0, 'price_move': 0.004},
                {'vol_mult': 2.5, 'price_move': 0.002},
            ]
        }
        
        discovered = []
        
        for pattern_type, configs in pattern_configs.items():
            logger.info(f"Testing {pattern_type} patterns...")
            
            for params in configs:
                # Test on training data
                signals = self.test_pattern(train_data, pattern_type, params)
                perf = self.calculate_performance(train_data, signals)
                
                if perf and perf['expectancy'] > 0 and perf['win_rate'] > 0.45:
                    pattern = {
                        'type': pattern_type,
                        'params': params,
                        'train_performance': perf
                    }
                    discovered.append(pattern)
                    logger.info(f"Found: {pattern_type} - WR: {perf['win_rate']:.1%}, "
                              f"PF: {perf['profit_factor']:.2f}, "
                              f"Expectancy: ${perf['expectancy']:.2f}")
        
        # Validate on test data
        logger.info("Validating on test data...")
        test_data = self.load_historical_data(self.test_start, self.test_end)
        
        validated = []
        for pattern in discovered:
            signals = self.test_pattern(test_data, pattern['type'], pattern['params'])
            test_perf = self.calculate_performance(test_data, signals)
            
            if test_perf and test_perf['expectancy'] > 0:
                pattern['test_performance'] = test_perf
                validated.append(pattern)
                logger.info(f"Validated: {pattern['type']} - Test expectancy: ${test_perf['expectancy']:.2f}")
        
        # Sort by test expectancy
        validated.sort(key=lambda x: x['test_performance']['expectancy'], reverse=True)
        
        # Save results
        output = {
            'discovery_date': datetime.now().isoformat(),
            'patterns': validated[:10]  # Top 10
        }
        
        with open('nq_patterns.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("TOP NQ PATTERNS DISCOVERED")
        print("="*60)
        
        for i, p in enumerate(validated[:5], 1):
            print(f"\n#{i} {p['type'].upper()}")
            print(f"Params: {p['params']}")
            print(f"Train: {p['train_performance']['trades']} trades, "
                  f"WR={p['train_performance']['win_rate']:.1%}, "
                  f"Exp=${p['train_performance']['expectancy']:.2f}")
            print(f"Test: {p['test_performance']['trades']} trades, "
                  f"WR={p['test_performance']['win_rate']:.1%}, "
                  f"Exp=${p['test_performance']['expectancy']:.2f}")
        
        return validated

if __name__ == "__main__":
    discovery = FastNQPatternDiscovery()
    patterns = discovery.discover_patterns()