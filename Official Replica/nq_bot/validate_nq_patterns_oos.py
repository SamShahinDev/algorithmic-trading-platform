#!/usr/bin/env python3
"""
Out-of-Sample Pattern Validation for NQ
Validates discovered patterns on 2024 Q3-Q4 NQ data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import json
import zstandard as zstd
import warnings
warnings.filterwarnings('ignore')

def read_zst_file(filepath):
    """Read a zstandard compressed CSV file"""
    try:
        dctx = zstd.ZstdDecompressor()
        with open(filepath, 'rb') as f:
            decompressed = dctx.decompress(f.read())
        
        # Parse CSV data
        lines = decompressed.decode('utf-8').strip().split('\n')
        if len(lines) < 2:
            return pd.DataFrame()
        
        # Parse header
        header = lines[0].split(',')
        
        # Parse data
        data = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    data.append({
                        'timestamp': pd.to_datetime(parts[0]),
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5])
                    })
                except:
                    continue
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df.empty or len(df) < 50:
        return df
    
    # Price changes
    df['returns'] = df['close'].pct_change()
    
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
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(14)
    
    return df

def validate_patterns():
    """Validate NQ patterns on out-of-sample data"""
    print("=" * 80)
    print("NQ PATTERN VALIDATION - OUT OF SAMPLE")
    print("Period: 2024 Q3-Q4")
    print("=" * 80)
    
    # Load Q3-Q4 2024 NQ data
    data_dir = "Historical Data/NQ Data"
    pattern = "glbx-mdp3-2024*.ohlcv-1m.csv.zst"
    files = glob.glob(os.path.join(data_dir, pattern))
    
    # Filter for Q3-Q4 (July-December)
    q3q4_files = []
    for file in files:
        filename = os.path.basename(file)
        try:
            date_str = filename.split('-')[2].replace('.ohlcv', '')
            date = datetime.strptime(date_str, '%Y%m%d')
            if date.month >= 7:  # July onwards
                q3q4_files.append(file)
        except:
            continue
    
    print(f"\nFound {len(q3q4_files)} Q3-Q4 2024 files")
    
    if not q3q4_files:
        print("No Q3-Q4 2024 data found")
        return
    
    # Load and combine data (sample for speed)
    all_data = []
    for file in sorted(q3q4_files)[:30]:  # Load first 30 days
        df = read_zst_file(file)
        if not df.empty:
            all_data.append(df)
            print(f"Loaded {os.path.basename(file)}: {len(df)} bars")
    
    if not all_data:
        print("Failed to load any data")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.sort_values('timestamp', inplace=True)
    print(f"\nTotal bars loaded: {len(combined_df)}")
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    combined_df = calculate_indicators(combined_df)
    
    # Define NQ patterns to validate
    patterns = [
        {
            'name': 'Momentum Thrust',
            'description': 'Strong momentum with volume confirmation',
            'conditions': lambda df: (df['rsi'] > 65) & (df['volume_ratio'] > 1.4) & (df['momentum'] > 0)
        },
        {
            'name': 'Volume Surge',
            'description': 'High volume breakout',
            'conditions': lambda df: (df['volume_ratio'] > 2.0) & (df['returns'].abs() > 0.002)
        },
        {
            'name': 'Oversold Bounce',
            'description': 'Reversal from oversold conditions',
            'conditions': lambda df: (df['rsi'] < 30) & (df['volume_ratio'] > 1.2)
        },
        {
            'name': 'Trend Continuation',
            'description': 'Price above moving averages with momentum',
            'conditions': lambda df: (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']) & (df['momentum'] > 0)
        }
    ]
    
    # Validate each pattern
    print("\n" + "=" * 60)
    print("PATTERN VALIDATION RESULTS")
    print("=" * 60)
    
    results = []
    for pattern in patterns:
        print(f"\n{pattern['name']}:")
        print(f"  Description: {pattern['description']}")
        
        # Generate signals
        signals = pattern['conditions'](combined_df)
        signal_count = signals.sum()
        
        if signal_count > 0:
            # Calculate returns after signal (5-bar holding period)
            signal_indices = combined_df.index[signals].tolist()
            returns = []
            
            for idx in signal_indices:
                if idx + 5 < len(combined_df):
                    entry_price = combined_df.loc[idx, 'close']
                    exit_price = combined_df.loc[idx + 5, 'close']
                    trade_return = (exit_price - entry_price) / entry_price
                    returns.append(trade_return)
            
            if returns:
                win_rate = sum([1 for r in returns if r > 0]) / len(returns)
                avg_return = np.mean(returns)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
                
                print(f"  Signals Generated: {signal_count}")
                print(f"  Trades Analyzed: {len(returns)}")
                print(f"  Win Rate: {win_rate:.1%}")
                print(f"  Avg Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
                print(f"  Sharpe Ratio: {sharpe:.2f}")
                print(f"  Max Loss: {min(returns):.4f}")
                print(f"  Max Gain: {max(returns):.4f}")
                
                # Performance assessment
                if win_rate > 0.55:
                    print(f"  ✅ STRONG PERFORMANCE - Keep this pattern")
                elif win_rate > 0.50:
                    print(f"  ⚠️ MODERATE PERFORMANCE - Monitor closely")
                else:
                    print(f"  ❌ WEAK PERFORMANCE - Consider removing")
                
                results.append({
                    'pattern': pattern['name'],
                    'signals': signal_count,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'sharpe': sharpe
                })
            else:
                print(f"  Signals Generated: {signal_count}")
                print(f"  ⚠️ Not enough data for validation")
        else:
            print(f"  No signals generated in Q3-Q4 2024")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if results:
        # Overall statistics
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_return = np.mean([r['avg_return'] for r in results])
        total_signals = sum([r['signals'] for r in results])
        
        print(f"\nOverall Performance (Q3-Q4 2024):")
        print(f"  Patterns Tested: {len(patterns)}")
        print(f"  Patterns with Signals: {len(results)}")
        print(f"  Total Signals: {total_signals}")
        print(f"  Average Win Rate: {avg_win_rate:.1%}")
        print(f"  Average Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
        
        # Best performers
        print(f"\nBest Performing Patterns:")
        sorted_results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"  {i}. {r['pattern']}: {r['win_rate']:.1%} win rate, {r['avg_return']:.4f} avg return")
        
        # Final recommendations
        print(f"\n📊 RECOMMENDATIONS:")
        print(f"\nBased on out-of-sample validation:")
        
        good_patterns = [r for r in results if r['win_rate'] > 0.52]
        if good_patterns:
            print(f"\n✅ Patterns validated for live trading:")
            for r in good_patterns:
                print(f"  - {r['pattern']} ({r['win_rate']:.1%} win rate)")
        
        poor_patterns = [r for r in results if r['win_rate'] < 0.48]
        if poor_patterns:
            print(f"\n❌ Patterns to disable:")
            for r in poor_patterns:
                print(f"  - {r['pattern']} ({r['win_rate']:.1%} win rate)")
        
        # Save results
        with open('nq_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to nq_validation_results.json")
    else:
        print("\nNo patterns generated sufficient signals for validation")
        print("Consider adjusting pattern parameters or using more data")

if __name__ == "__main__":
    validate_patterns()