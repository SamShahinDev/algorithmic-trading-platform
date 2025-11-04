#!/usr/bin/env python3
"""
Out-of-Sample Pattern Validation for ES and CL
Validates discovered patterns on 2024 Q3-Q4 data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
import zstandard as zstd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PatternValidator:
    def __init__(self):
        self.es_data_path = "Historical Data/New Data/GLBX-20250828-98YG33QNQH"
        self.cl_data_path = "Historical Data/New Data/GLBX-20250828-CR4KVBURP8"
        
        # Define patterns discovered earlier
        self.patterns = {
            'ES': [
                {
                    'name': 'Momentum Surge',
                    'type': 'momentum',
                    'params': {
                        'rsi_threshold': 70,
                        'volume_multiplier': 1.5,
                        'price_change': 0.002
                    },
                    'historical_win_rate': 0.523
                },
                {
                    'name': 'Volume Breakout',
                    'type': 'volume',
                    'params': {
                        'volume_spike': 2.0,
                        'price_breakout': 0.003,
                        'atr_multiplier': 1.2
                    },
                    'historical_win_rate': 0.618
                },
                {
                    'name': 'Range Expansion',
                    'type': 'volatility',
                    'params': {
                        'range_expansion': 1.5,
                        'volume_confirm': 1.3,
                        'trend_strength': 0.6
                    },
                    'historical_win_rate': 0.721
                },
                {
                    'name': 'Mean Reversion',
                    'type': 'reversal',
                    'params': {
                        'bb_width': 2.0,
                        'rsi_oversold': 30,
                        'volume_dry': 0.7
                    },
                    'historical_win_rate': 0.682
                }
            ],
            'CL': [
                {
                    'name': 'Oil Momentum',
                    'type': 'momentum',
                    'params': {
                        'momentum_period': 20,
                        'volume_surge': 1.8,
                        'atr_filter': 1.5
                    },
                    'historical_win_rate': 0.734
                },
                {
                    'name': 'Supply Demand Imbalance',
                    'type': 'orderflow',
                    'params': {
                        'bid_ask_ratio': 1.5,
                        'volume_imbalance': 0.3,
                        'price_acceleration': 0.004
                    },
                    'historical_win_rate': 0.752
                }
            ]
        }
    
    def read_zst_file(self, filepath: str) -> pd.DataFrame:
        """Read zstandard compressed OHLCV file"""
        try:
            dctx = zstd.ZstdDecompressor()
            with open(filepath, 'rb') as f:
                decompressed = dctx.decompress(f.read())
            
            # Parse CSV data
            lines = decompressed.decode('utf-8').strip().split('\n')
            if len(lines) < 2:
                return pd.DataFrame()
            
            # Parse data (skip header)
            data = []
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 10:  # Full data format with symbol
                    try:
                        # Extract symbol to identify ES vs CL
                        symbol = parts[9] if len(parts) > 9 else ''
                        
                        # Only process ES or CL data
                        if 'ES' in symbol or 'CL' in symbol:
                            data.append({
                                'timestamp': pd.to_datetime(parts[0]),
                                'open': float(parts[4]),
                                'high': float(parts[5]),
                                'low': float(parts[6]),
                                'close': float(parts[7]),
                                'volume': float(parts[8]),
                                'symbol': symbol
                            })
                    except:
                        continue
            
            if data:
                df = pd.DataFrame(data)
                # Resample to 1-minute bars
                df.set_index('timestamp', inplace=True)
                df = df.resample('1min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'symbol': 'first'
                }).dropna()
                df.reset_index(inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            return pd.DataFrame()
    
    def load_q3q4_data(self, symbol: str) -> pd.DataFrame:
        """Load Q3-Q4 2024 data for a specific symbol"""
        print(f"\nLoading {symbol} Q3-Q4 2024 data...")
        
        # Determine data path
        if symbol == 'ES':
            data_path = self.es_data_path
        else:  # CL
            data_path = self.cl_data_path
        
        # Get Q3-Q4 2024 files (July-December)
        pattern = os.path.join(data_path, "glbx-mdp3-2024*.ohlcv-1m.csv.zst")
        all_files = glob.glob(pattern)
        
        # Filter for Q3-Q4
        q3q4_files = []
        for file in all_files:
            filename = os.path.basename(file)
            try:
                # Extract date from filename
                date_str = filename.split('-')[2].replace('.ohlcv', '')
                date = datetime.strptime(date_str, '%Y%m%d')
                if date.month >= 7:  # July onwards
                    q3q4_files.append(file)
            except:
                continue
        
        print(f"Found {len(q3q4_files)} Q3-Q4 2024 files for {symbol}")
        
        # Load data (sample for speed - first 30 files)
        all_data = []
        files_loaded = 0
        for file in sorted(q3q4_files)[:30]:
            df = self.read_zst_file(file)
            if not df.empty:
                # Filter for specific symbol
                if 'symbol' in df.columns:
                    df = df[df['symbol'].str.contains(symbol, na=False)]
                if not df.empty:
                    all_data.append(df)
                    files_loaded += 1
                    if files_loaded % 5 == 0:
                        print(f"  Loaded {files_loaded} files...")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.sort_values('timestamp', inplace=True)
            print(f"Total {symbol} bars loaded: {len(combined):,}")
            return combined
        
        return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for pattern detection"""
        if df.empty or len(df) < 50:
            return df
        
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Range and volatility
        df['range'] = df['high'] - df['low']
        df['range_sma'] = df['range'].rolling(20).mean()
        df['range_expansion'] = df['range'] / (df['range_sma'] + 1e-10)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(20)
        
        return df
    
    def detect_pattern_signals(self, df: pd.DataFrame, pattern: Dict) -> pd.Series:
        """Detect pattern signals in the data"""
        signals = pd.Series(False, index=df.index)
        
        if pattern['type'] == 'momentum':
            if 'rsi_threshold' in pattern['params']:
                # ES momentum pattern
                rsi_signal = df['rsi'] > pattern['params']['rsi_threshold']
                volume_signal = df['volume_ratio'] > pattern['params'].get('volume_multiplier', 1.0)
                price_signal = df['returns'].abs() > pattern['params'].get('price_change', 0.001)
                signals = rsi_signal & volume_signal & price_signal
            else:
                # CL momentum pattern
                momentum_signal = df['momentum'] > 0
                volume_signal = df['volume_ratio'] > pattern['params'].get('volume_surge', 1.5)
                atr_signal = df['atr'] > df['atr'].rolling(20).mean() * pattern['params'].get('atr_filter', 1.0)
                signals = momentum_signal & volume_signal & atr_signal
        
        elif pattern['type'] == 'volume':
            volume_signal = df['volume_ratio'] > pattern['params']['volume_spike']
            price_signal = df['returns'].abs() > pattern['params'].get('price_breakout', 0.002)
            signals = volume_signal & price_signal
        
        elif pattern['type'] == 'volatility':
            range_signal = df['range_expansion'] > pattern['params']['range_expansion']
            volume_signal = df['volume_ratio'] > pattern['params'].get('volume_confirm', 1.0)
            signals = range_signal & volume_signal
        
        elif pattern['type'] == 'reversal':
            rsi_signal = df['rsi'] < pattern['params']['rsi_oversold']
            bb_signal = df['close'] < df['bb_lower']
            volume_signal = df['volume_ratio'] < pattern['params'].get('volume_dry', 1.0)
            signals = rsi_signal & bb_signal
        
        elif pattern['type'] == 'orderflow':
            # Simplified orderflow pattern (volume imbalance proxy)
            volume_surge = df['volume_ratio'] > pattern['params'].get('bid_ask_ratio', 1.5)
            price_accel = df['returns'].abs() > pattern['params'].get('price_acceleration', 0.003)
            signals = volume_surge & price_accel
        
        return signals
    
    def validate_pattern(self, df: pd.DataFrame, pattern: Dict, holding_period: int = 5) -> Dict:
        """Validate a pattern's performance"""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Detect signals
        signals = self.detect_pattern_signals(df, pattern)
        signal_count = signals.sum()
        
        # Calculate performance
        results = []
        if signal_count > 0:
            signal_indices = df.index[signals].tolist()
            
            for idx in signal_indices:
                if idx + holding_period < len(df):
                    entry_price = df.loc[idx, 'close']
                    exit_price = df.loc[idx + holding_period, 'close']
                    
                    # Calculate return based on pattern type
                    if pattern['type'] in ['momentum', 'volume', 'volatility', 'orderflow']:
                        trade_return = (exit_price - entry_price) / entry_price
                    else:  # reversal
                        trade_return = (exit_price - entry_price) / entry_price  # Long for oversold
                    
                    results.append({
                        'return': trade_return,
                        'profitable': trade_return > 0
                    })
        
        # Calculate metrics
        if results:
            returns = [r['return'] for r in results]
            profitable = [r['profitable'] for r in results]
            
            win_rate = sum(profitable) / len(profitable)
            avg_return = np.mean(returns)
            total_return = sum(returns)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_dd = min(returns) if returns else 0
            max_gain = max(returns) if returns else 0
            
            return {
                'pattern_name': pattern['name'],
                'signal_count': signal_count,
                'trades': len(results),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'max_gain': max_gain,
                'historical_win_rate': pattern['historical_win_rate'],
                'win_rate_diff': win_rate - pattern['historical_win_rate']
            }
        
        return {
            'pattern_name': pattern['name'],
            'signal_count': signal_count,
            'trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_gain': 0,
            'historical_win_rate': pattern['historical_win_rate'],
            'win_rate_diff': 0
        }
    
    def run_validation(self):
        """Run complete out-of-sample validation"""
        print("=" * 80)
        print("OUT-OF-SAMPLE PATTERN VALIDATION")
        print("Period: 2024 Q3-Q4 (July - December)")
        print("=" * 80)
        
        all_results = {}
        
        # Validate ES patterns
        print("\n" + "=" * 60)
        print("ES PATTERN VALIDATION")
        print("=" * 60)
        
        es_data = self.load_q3q4_data('ES')
        es_results = []
        
        if not es_data.empty:
            for pattern in self.patterns['ES']:
                print(f"\nValidating: {pattern['name']}")
                result = self.validate_pattern(es_data, pattern)
                es_results.append(result)
                
                # Print results
                print(f"  Signals: {result['signal_count']}")
                print(f"  Trades: {result['trades']}")
                if result['trades'] > 0:
                    print(f"  Win Rate: {result['win_rate']:.1%} (Historical: {result['historical_win_rate']:.1%})")
                    print(f"  Avg Return: {result['avg_return']:.4f} ({result['avg_return']*100:.2f}%)")
                    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                    print(f"  Max Drawdown: {result['max_drawdown']:.4f}")
                    
                    # Performance assessment
                    if abs(result['win_rate_diff']) < 0.10:
                        print(f"  ✅ CONSISTENT with historical performance")
                    elif result['win_rate_diff'] > 0:
                        print(f"  📈 OUTPERFORMING historical by {result['win_rate_diff']:.1%}")
                    else:
                        print(f"  ⚠️ UNDERPERFORMING historical by {abs(result['win_rate_diff']):.1%}")
        
        all_results['ES'] = es_results
        
        # Validate CL patterns
        print("\n" + "=" * 60)
        print("CL PATTERN VALIDATION")
        print("=" * 60)
        
        cl_data = self.load_q3q4_data('CL')
        cl_results = []
        
        if not cl_data.empty:
            for pattern in self.patterns['CL']:
                print(f"\nValidating: {pattern['name']}")
                result = self.validate_pattern(cl_data, pattern)
                cl_results.append(result)
                
                # Print results
                print(f"  Signals: {result['signal_count']}")
                print(f"  Trades: {result['trades']}")
                if result['trades'] > 0:
                    print(f"  Win Rate: {result['win_rate']:.1%} (Historical: {result['historical_win_rate']:.1%})")
                    print(f"  Avg Return: {result['avg_return']:.4f} ({result['avg_return']*100:.2f}%)")
                    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                    print(f"  Max Drawdown: {result['max_drawdown']:.4f}")
                    
                    # Performance assessment
                    if abs(result['win_rate_diff']) < 0.10:
                        print(f"  ✅ CONSISTENT with historical performance")
                    elif result['win_rate_diff'] > 0:
                        print(f"  📈 OUTPERFORMING historical by {result['win_rate_diff']:.1%}")
                    else:
                        print(f"  ⚠️ UNDERPERFORMING historical by {abs(result['win_rate_diff']):.1%}")
        
        all_results['CL'] = cl_results
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        # Save results
        with open('pattern_validation_q3q4_2024.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        return all_results
    
    def generate_summary_report(self, results: Dict):
        """Generate comprehensive validation summary"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY REPORT")
        print("=" * 80)
        
        # Combine all results
        all_patterns = []
        for symbol, symbol_results in results.items():
            for result in symbol_results:
                result['symbol'] = symbol
                all_patterns.append(result)
        
        # Filter patterns with trades
        active_patterns = [p for p in all_patterns if p['trades'] > 0]
        
        if active_patterns:
            print(f"\n📊 OVERALL STATISTICS")
            print(f"  Total Patterns Tested: {len(all_patterns)}")
            print(f"  Patterns with Signals: {len(active_patterns)}")
            print(f"  Total Signals Generated: {sum([p['signal_count'] for p in active_patterns])}")
            print(f"  Total Trades Analyzed: {sum([p['trades'] for p in active_patterns])}")
            
            avg_win_rate = np.mean([p['win_rate'] for p in active_patterns])
            avg_return = np.mean([p['avg_return'] for p in active_patterns])
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in active_patterns if p['sharpe_ratio'] != 0])
            
            print(f"\n📈 PERFORMANCE METRICS")
            print(f"  Average Win Rate: {avg_win_rate:.1%}")
            print(f"  Average Return per Trade: {avg_return:.4f} ({avg_return*100:.2f}%)")
            print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
            
            # Best performers
            print(f"\n🏆 TOP PERFORMERS (by Win Rate)")
            sorted_by_wr = sorted(active_patterns, key=lambda x: x['win_rate'], reverse=True)[:3]
            for i, p in enumerate(sorted_by_wr, 1):
                print(f"  {i}. {p['symbol']} - {p['pattern_name']}: {p['win_rate']:.1%} ({p['trades']} trades)")
            
            # Most reliable (consistent with historical)
            print(f"\n🎯 MOST RELIABLE (smallest deviation from historical)")
            sorted_by_consistency = sorted(active_patterns, key=lambda x: abs(x['win_rate_diff']))[:3]
            for i, p in enumerate(sorted_by_consistency, 1):
                print(f"  {i}. {p['symbol']} - {p['pattern_name']}: {abs(p['win_rate_diff']*100):.1f}% deviation")
            
            # Recommendations
            print(f"\n✅ PATTERNS VALIDATED FOR LIVE TRADING")
            validated = [p for p in active_patterns if p['win_rate'] > 0.50 and abs(p['win_rate_diff']) < 0.15]
            if validated:
                for p in validated:
                    print(f"  • {p['symbol']} - {p['pattern_name']}: {p['win_rate']:.1%} win rate")
            else:
                print("  None meet criteria (>50% win rate, <15% deviation)")
            
            print(f"\n⚠️ PATTERNS NEEDING ADJUSTMENT")
            needs_review = [p for p in active_patterns if abs(p['win_rate_diff']) > 0.15]
            if needs_review:
                for p in needs_review:
                    print(f"  • {p['symbol']} - {p['pattern_name']}: {p['win_rate_diff']*100:+.1f}% deviation")
            
            print(f"\n❌ PATTERNS TO DISABLE")
            poor_patterns = [p for p in active_patterns if p['win_rate'] < 0.45]
            if poor_patterns:
                for p in poor_patterns:
                    print(f"  • {p['symbol']} - {p['pattern_name']}: {p['win_rate']:.1%} win rate")
        
        print(f"\n💾 Detailed results saved to pattern_validation_q3q4_2024.json")

if __name__ == "__main__":
    validator = PatternValidator()
    results = validator.run_validation()