#!/usr/bin/env python3
"""
Out-of-Sample Pattern Validation
Validates discovered patterns on 2024 Q3-Q4 data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import os
import json
import zstandard as zstd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class OutOfSampleValidator:
    def __init__(self, data_path: str = "Historical Data"):
        self.data_path = data_path
        self.patterns = self.load_discovered_patterns()
        
    def load_discovered_patterns(self) -> Dict:
        """Load the patterns we discovered earlier"""
        patterns = {
            'ES': [
                {
                    'name': 'momentum_surge',
                    'type': 'momentum',
                    'params': {
                        'rsi_threshold': 70,
                        'volume_multiplier': 1.5,
                        'price_change': 0.002
                    },
                    'historical_win_rate': 0.523,
                    'avg_return': 0.0028
                },
                {
                    'name': 'volume_breakout',
                    'type': 'volume',
                    'params': {
                        'volume_spike': 2.0,
                        'price_breakout': 0.003,
                        'atr_multiplier': 1.2
                    },
                    'historical_win_rate': 0.618,
                    'avg_return': 0.0035
                },
                {
                    'name': 'range_expansion',
                    'type': 'volatility',
                    'params': {
                        'range_expansion': 1.5,
                        'volume_confirm': 1.3,
                        'trend_strength': 0.6
                    },
                    'historical_win_rate': 0.721,
                    'avg_return': 0.0041
                },
                {
                    'name': 'mean_reversion',
                    'type': 'reversal',
                    'params': {
                        'bb_width': 2.0,
                        'rsi_oversold': 30,
                        'volume_dry': 0.7
                    },
                    'historical_win_rate': 0.682,
                    'avg_return': 0.0032
                }
            ],
            'CL': [
                {
                    'name': 'oil_momentum',
                    'type': 'momentum',
                    'params': {
                        'momentum_period': 20,
                        'volume_surge': 1.8,
                        'atr_filter': 1.5
                    },
                    'historical_win_rate': 0.734,
                    'avg_return': 0.0045
                },
                {
                    'name': 'supply_demand_imbalance',
                    'type': 'orderflow',
                    'params': {
                        'bid_ask_ratio': 1.5,
                        'volume_imbalance': 0.3,
                        'price_acceleration': 0.004
                    },
                    'historical_win_rate': 0.752,
                    'avg_return': 0.0052
                }
            ],
            'NQ': [
                {
                    'name': 'momentum_thrust',
                    'type': 'momentum',
                    'params': {
                        'thrust_period': 14,
                        'volume_confirm': 1.4,
                        'rsi_threshold': 65
                    },
                    'historical_win_rate': 0.580,
                    'avg_return': 0.0038
                }
            ]
        }
        return patterns
    
    def load_oos_data(self, symbol: str, start_date: str = "2024-07-01", end_date: str = "2024-12-31") -> pd.DataFrame:
        """Load out-of-sample data for validation"""
        print(f"\nLoading {symbol} data from {start_date} to {end_date}...")
        
        # Determine the correct data directory
        if symbol == 'NQ':
            data_dir = os.path.join(self.data_path, 'NQ Data')
            pattern = 'glbx-mdp3-*.ohlcv-1m.csv.zst'
        elif symbol == 'ES':
            data_dir = os.path.join(self.data_path, 'New Data', 'ES')
            pattern = 'glbx-mdp3-*.ohlcv-1m.csv.zst'
        elif symbol == 'CL':
            data_dir = os.path.join(self.data_path, 'New Data', 'CL')
            pattern = 'glbx-mdp3-*.ohlcv-1m.csv.zst'
        else:
            return pd.DataFrame()
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist")
            return pd.DataFrame()
        
        files = glob.glob(os.path.join(data_dir, pattern))
        
        # Filter files for Q3-Q4 2024
        oos_files = []
        for file in files:
            filename = os.path.basename(file)
            # Extract date from filename (format: glbx-mdp3-YYYYMMDD-...)
            try:
                date_str = filename.split('-')[2]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if datetime(2024, 7, 1) <= file_date <= datetime(2024, 12, 31):
                    oos_files.append(file)
            except:
                continue
        
        print(f"Found {len(oos_files)} out-of-sample files for {symbol}")
        
        if not oos_files:
            print(f"No out-of-sample data found for {symbol}")
            return pd.DataFrame()
        
        # Load and combine data
        all_data = []
        for file in sorted(oos_files)[:30]:  # Limit to 30 files for speed
            try:
                df = self.read_zst_file(file)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(combined)} bars of out-of-sample data for {symbol}")
            return combined
        
        return pd.DataFrame()
    
    def read_zst_file(self, filepath: str) -> pd.DataFrame:
        """Read zstandard compressed OHLCV file"""
        try:
            dctx = zstd.ZstdDecompressor()
            with open(filepath, 'rb') as f:
                decompressed = dctx.decompress(f.read())
                
            # Parse the data
            lines = decompressed.decode('utf-8').strip().split('\n')
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
            
            # Resample to 1-minute bars
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df = df.resample('1min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                df.reset_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for pattern detection"""
        if df.empty:
            return df
            
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['atr'] = self.calculate_atr(df)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df)
        
        # Range
        df['range'] = df['high'] - df['low']
        df['range_sma'] = df['range'].rolling(20).mean()
        df['range_expansion'] = df['range'] / df['range_sma']
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: int = 2):
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def detect_pattern(self, df: pd.DataFrame, pattern: Dict) -> pd.Series:
        """Detect a specific pattern in the data"""
        signals = pd.Series(False, index=df.index)
        
        if pattern['type'] == 'momentum':
            if 'rsi_threshold' in pattern['params']:
                rsi_signal = df['rsi'] > pattern['params']['rsi_threshold']
                volume_signal = df['volume_ratio'] > pattern['params'].get('volume_multiplier', 1.0)
                signals = rsi_signal & volume_signal
                
        elif pattern['type'] == 'volume':
            if 'volume_spike' in pattern['params']:
                volume_signal = df['volume_ratio'] > pattern['params']['volume_spike']
                price_signal = df['returns'].abs() > pattern['params'].get('price_breakout', 0.001)
                signals = volume_signal & price_signal
                
        elif pattern['type'] == 'volatility':
            if 'range_expansion' in pattern['params']:
                range_signal = df['range_expansion'] > pattern['params']['range_expansion']
                volume_signal = df['volume_ratio'] > pattern['params'].get('volume_confirm', 1.0)
                signals = range_signal & volume_signal
                
        elif pattern['type'] == 'reversal':
            if 'rsi_oversold' in pattern['params']:
                rsi_signal = df['rsi'] < pattern['params']['rsi_oversold']
                bb_signal = df['close'] < df['bb_lower']
                signals = rsi_signal & bb_signal
        
        return signals
    
    def validate_pattern(self, df: pd.DataFrame, pattern: Dict, holding_period: int = 5) -> Dict:
        """Validate a pattern's performance on out-of-sample data"""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Detect pattern signals
        signals = self.detect_pattern(df, pattern)
        
        # Calculate returns for each signal
        results = []
        signal_indices = df.index[signals].tolist()
        
        for idx in signal_indices:
            if idx + holding_period < len(df):
                entry_price = df.loc[idx, 'close']
                exit_price = df.loc[idx + holding_period, 'close']
                
                # Determine direction based on pattern type
                if pattern['type'] in ['momentum', 'volume', 'volatility']:
                    # Long trades
                    trade_return = (exit_price - entry_price) / entry_price
                else:  # reversal
                    # Short trades for overbought, long for oversold
                    if 'rsi_oversold' in pattern['params']:
                        trade_return = (exit_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - exit_price) / entry_price
                
                results.append({
                    'entry_time': df.loc[idx, 'timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'profitable': trade_return > 0
                })
        
        # Calculate metrics
        if results:
            returns = [r['return'] for r in results]
            profitable = [r['profitable'] for r in results]
            
            metrics = {
                'pattern_name': pattern['name'],
                'pattern_type': pattern['type'],
                'total_signals': len(results),
                'win_rate': sum(profitable) / len(profitable),
                'avg_return': np.mean(returns),
                'total_return': sum(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'max_drawdown': min(returns) if returns else 0,
                'historical_win_rate': pattern['historical_win_rate'],
                'win_rate_diff': (sum(profitable) / len(profitable)) - pattern['historical_win_rate']
            }
        else:
            metrics = {
                'pattern_name': pattern['name'],
                'pattern_type': pattern['type'],
                'total_signals': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'historical_win_rate': pattern['historical_win_rate'],
                'win_rate_diff': 0
            }
        
        return metrics
    
    def validate_all_patterns(self):
        """Validate all patterns on out-of-sample data"""
        print("=" * 80)
        print("OUT-OF-SAMPLE PATTERN VALIDATION")
        print("Period: 2024 Q3-Q4")
        print("=" * 80)
        
        validation_results = {}
        
        for symbol in ['ES', 'CL', 'NQ']:
            print(f"\n{'='*40}")
            print(f"Validating {symbol} Patterns")
            print(f"{'='*40}")
            
            # Load out-of-sample data
            df = self.load_oos_data(symbol)
            
            if df.empty:
                print(f"No data available for {symbol}")
                continue
            
            # Validate each pattern
            symbol_results = []
            for pattern in self.patterns[symbol]:
                print(f"\nValidating pattern: {pattern['name']}")
                metrics = self.validate_pattern(df, pattern)
                symbol_results.append(metrics)
                
                # Print results
                print(f"  Signals: {metrics['total_signals']}")
                print(f"  Win Rate: {metrics['win_rate']:.1%} (Historical: {metrics['historical_win_rate']:.1%})")
                print(f"  Avg Return: {metrics['avg_return']:.4f}")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                
                # Performance assessment
                if metrics['total_signals'] > 0:
                    if abs(metrics['win_rate_diff']) < 0.1:
                        print(f"  ✅ Pattern performance CONSISTENT with historical")
                    elif metrics['win_rate_diff'] > 0:
                        print(f"  📈 Pattern performing BETTER than historical")
                    else:
                        print(f"  ⚠️ Pattern performing WORSE than historical")
            
            validation_results[symbol] = symbol_results
        
        # Generate summary report
        self.generate_summary_report(validation_results)
        
        return validation_results
    
    def generate_summary_report(self, results: Dict):
        """Generate a summary report of validation results"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY REPORT")
        print("=" * 80)
        
        # Overall statistics
        all_patterns = []
        for symbol_results in results.values():
            all_patterns.extend(symbol_results)
        
        if all_patterns:
            # Filter patterns with signals
            active_patterns = [p for p in all_patterns if p['total_signals'] > 0]
            
            if active_patterns:
                avg_win_rate = np.mean([p['win_rate'] for p in active_patterns])
                avg_return = np.mean([p['avg_return'] for p in active_patterns])
                total_signals = sum([p['total_signals'] for p in active_patterns])
                
                print(f"\nOverall Performance (Out-of-Sample):")
                print(f"  Total Patterns Tested: {len(all_patterns)}")
                print(f"  Patterns with Signals: {len(active_patterns)}")
                print(f"  Total Signals Generated: {total_signals}")
                print(f"  Average Win Rate: {avg_win_rate:.1%}")
                print(f"  Average Return per Trade: {avg_return:.4f}")
                
                # Best performing patterns
                print(f"\nTop 3 Patterns by Win Rate:")
                sorted_patterns = sorted(active_patterns, key=lambda x: x['win_rate'], reverse=True)[:3]
                for i, pattern in enumerate(sorted_patterns, 1):
                    print(f"  {i}. {pattern['pattern_name']}: {pattern['win_rate']:.1%} ({pattern['total_signals']} signals)")
                
                # Most reliable patterns (consistent with historical)
                print(f"\nMost Reliable Patterns (smallest deviation from historical):")
                reliable_patterns = sorted(active_patterns, key=lambda x: abs(x['win_rate_diff']))[:3]
                for i, pattern in enumerate(reliable_patterns, 1):
                    deviation = abs(pattern['win_rate_diff']) * 100
                    print(f"  {i}. {pattern['pattern_name']}: {deviation:.1f}% deviation")
                
                # Risk assessment
                print(f"\nRisk Assessment:")
                max_dd = min([p['max_drawdown'] for p in active_patterns])
                avg_sharpe = np.mean([p['sharpe_ratio'] for p in active_patterns if p['sharpe_ratio'] != 0])
                print(f"  Worst Drawdown: {max_dd:.4f}")
                print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
                
                # Recommendations
                print(f"\n📊 RECOMMENDATIONS:")
                
                # Patterns to keep
                good_patterns = [p for p in active_patterns if p['win_rate'] > 0.5 and abs(p['win_rate_diff']) < 0.15]
                if good_patterns:
                    print(f"\n✅ Patterns to KEEP trading:")
                    for pattern in good_patterns:
                        print(f"  - {pattern['pattern_name']} (Win: {pattern['win_rate']:.1%}, Signals: {pattern['total_signals']})")
                
                # Patterns to review
                review_patterns = [p for p in active_patterns if abs(p['win_rate_diff']) > 0.15]
                if review_patterns:
                    print(f"\n⚠️ Patterns to REVIEW/ADJUST:")
                    for pattern in review_patterns:
                        print(f"  - {pattern['pattern_name']} (Deviation: {pattern['win_rate_diff']:.1%})")
                
                # Patterns to remove
                bad_patterns = [p for p in active_patterns if p['win_rate'] < 0.45]
                if bad_patterns:
                    print(f"\n❌ Patterns to REMOVE:")
                    for pattern in bad_patterns:
                        print(f"  - {pattern['pattern_name']} (Win: {pattern['win_rate']:.1%})")
        
        # Save results to file
        with open('pattern_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to pattern_validation_results.json")

if __name__ == "__main__":
    validator = OutOfSampleValidator()
    results = validator.validate_all_patterns()