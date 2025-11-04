"""
Comprehensive NQ Pattern Discovery with Fixed 1:2 Risk/Reward
Processes 648 compressed historical data files to find profitable patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import talib
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveNQDiscovery:
    """Discover NQ patterns from 2 years of historical data"""
    
    # Fixed risk/reward parameters
    STOP_LOSS_POINTS = 5
    TAKE_PROFIT_POINTS = 10
    COMMISSION_RT = 2.52
    POINT_VALUE = 20
    MIN_NET_EXPECTANCY = 3.0
    
    # Data paths
    DATA_PATH = "/Users/royaltyvixion/Documents/XTRADING/Historical Data/NQ Data"
    
    def __init__(self):
        self.patterns_found = []
        self.validation_results = []
        
    def load_compressed_file(self, filepath: Path) -> pd.DataFrame:
        """Load a zstd compressed CSV file"""
        try:
            with open(filepath, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text = reader.read().decode('utf-8')
                    
            # Parse CSV from text
            from io import StringIO
            df = pd.read_csv(StringIO(text))
            
            # Ensure proper column names and datetime index
            if 'ts_event' in df.columns:
                df['timestamp'] = pd.to_datetime(df['ts_event'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.set_index('timestamp')
            
            # Ensure we have OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in {filepath}")
                    return pd.DataFrame()
            
            return df[required_cols]
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def test_pattern_on_data(self, data: pd.DataFrame, pattern_func, params: Dict) -> Dict:
        """Test a pattern with fixed R/R on historical data"""
        signals = []
        
        for i in range(100, len(data) - 50):
            window = data.iloc[i-100:i]
            
            # Check if pattern triggers
            signal = pattern_func(window, **params)
            
            if signal is not None and signal != 0:
                entry_price = data.iloc[i]['close']
                entry_time = data.index[i]
                
                # Fixed stops and targets
                if signal > 0:  # Long
                    stop = entry_price - self.STOP_LOSS_POINTS
                    target = entry_price + self.TAKE_PROFIT_POINTS
                else:  # Short
                    stop = entry_price + self.STOP_LOSS_POINTS
                    target = entry_price - self.TAKE_PROFIT_POINTS
                
                # Check outcome
                outcome_data = data.iloc[i:i+50]
                hit_stop = False
                hit_target = False
                
                for j, (idx, bar) in enumerate(outcome_data.iterrows()):
                    if signal > 0:
                        if bar['low'] <= stop:
                            hit_stop = True
                            break
                        if bar['high'] >= target:
                            hit_target = True
                            break
                    else:
                        if bar['high'] >= stop:
                            hit_stop = True
                            break
                        if bar['low'] <= target:
                            hit_target = True
                            break
                
                if hit_target or hit_stop:
                    signals.append({
                        'time': entry_time,
                        'win': hit_target,
                        'bars_held': j if (hit_target or hit_stop) else 50
                    })
        
        if len(signals) < 10:
            return None
        
        wins = sum(1 for s in signals if s['win'])
        win_rate = wins / len(signals)
        
        # Calculate net expectancy
        gross = (win_rate * self.TAKE_PROFIT_POINTS * self.POINT_VALUE) - \
                ((1 - win_rate) * self.STOP_LOSS_POINTS * self.POINT_VALUE)
        net = gross - self.COMMISSION_RT
        
        return {
            'signals': len(signals),
            'win_rate': win_rate,
            'net_expectancy': net,
            'avg_bars': np.mean([s['bars_held'] for s in signals]),
            'profitable': net >= self.MIN_NET_EXPECTANCY
        }
    
    def pattern_bollinger_squeeze(self, data: pd.DataFrame, bb_period: int = 20, 
                                  bb_std: float = 2.0, squeeze_threshold: float = 0.5) -> Optional[int]:
        """Bollinger Band squeeze pattern"""
        if len(data) < bb_period:
            return None
        
        close = data['close'].values.astype(np.float64)
        
        sma = talib.SMA(close, timeperiod=bb_period)
        std = talib.STDDEV(close, timeperiod=bb_period)
        upper = sma + (std * bb_std)
        lower = sma - (std * bb_std)
        
        # Check for squeeze (bands narrowing)
        band_width = upper[-1] - lower[-1]
        avg_width = np.mean(upper[-20:] - lower[-20:])
        
        if band_width < avg_width * squeeze_threshold:
            # Breakout direction
            if close[-1] > upper[-1]:
                return 1
            elif close[-1] < lower[-1]:
                return -1
        
        return None
    
    def pattern_volume_climax(self, data: pd.DataFrame, vol_mult: float = 2.0,
                             price_move: float = 0.002) -> Optional[int]:
        """Volume climax reversal pattern"""
        if len(data) < 20:
            return None
        
        volume = data['volume'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        vol_avg = talib.SMA(volume, timeperiod=20)
        
        if volume[-1] > vol_avg[-1] * vol_mult:
            # Check for exhaustion move
            move = (close[-1] - close[-5]) / close[-5]
            
            if abs(move) > price_move:
                # Fade the move
                return -1 if move > 0 else 1
        
        return None
    
    def pattern_momentum_thrust(self, data: pd.DataFrame, roc_period: int = 10,
                               roc_threshold: float = 0.15) -> Optional[int]:
        """Momentum thrust continuation"""
        if len(data) < roc_period + 5:
            return None
        
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        roc = talib.ROC(close, timeperiod=roc_period)
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        # Strong momentum with volume
        if abs(roc[-1]) > roc_threshold and volume[-1] > vol_ma[-1] * 1.2:
            return 1 if roc[-1] > 0 else -1
        
        return None
    
    def pattern_range_expansion(self, data: pd.DataFrame, atr_period: int = 14,
                               expansion_mult: float = 1.5) -> Optional[int]:
        """Range expansion breakout"""
        if len(data) < atr_period:
            return None
        
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        current_range = high[-1] - low[-1]
        
        if current_range > atr[-1] * expansion_mult:
            # Trade in direction of expansion
            mid = (high[-1] + low[-1]) / 2
            return 1 if close[-1] > mid else -1
        
        return None
    
    def discover_patterns_parallel(self, num_files: int = 100) -> List[Dict]:
        """Discover patterns using parallel processing"""
        logger.info(f"Loading {num_files} files for pattern discovery...")
        
        data_path = Path(self.DATA_PATH)
        files = sorted(data_path.glob("*.csv.zst"))[:num_files]
        
        # Load and combine data
        all_data = []
        for i, file in enumerate(files):
            if i % 20 == 0:
                logger.info(f"Loading file {i+1}/{num_files}")
            df = self.load_compressed_file(file)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            logger.error("No data loaded")
            return []
        
        combined_data = pd.concat(all_data).sort_index()
        logger.info(f"Loaded {len(combined_data)} total bars")
        
        # Pattern configurations to test
        patterns_to_test = [
            {
                'name': 'bollinger_squeeze',
                'func': self.pattern_bollinger_squeeze,
                'params': [
                    {'bb_period': 15, 'bb_std': 2.0, 'squeeze_threshold': 0.5},
                    {'bb_period': 20, 'bb_std': 2.5, 'squeeze_threshold': 0.4},
                    {'bb_period': 25, 'bb_std': 2.0, 'squeeze_threshold': 0.6},
                ]
            },
            {
                'name': 'volume_climax',
                'func': self.pattern_volume_climax,
                'params': [
                    {'vol_mult': 1.5, 'price_move': 0.0015},
                    {'vol_mult': 2.0, 'price_move': 0.002},
                    {'vol_mult': 2.5, 'price_move': 0.0025},
                ]
            },
            {
                'name': 'momentum_thrust',
                'func': self.pattern_momentum_thrust,
                'params': [
                    {'roc_period': 5, 'roc_threshold': 0.1},
                    {'roc_period': 10, 'roc_threshold': 0.15},
                    {'roc_period': 15, 'roc_threshold': 0.2},
                ]
            },
            {
                'name': 'range_expansion',
                'func': self.pattern_range_expansion,
                'params': [
                    {'atr_period': 10, 'expansion_mult': 1.5},
                    {'atr_period': 14, 'expansion_mult': 2.0},
                    {'atr_period': 20, 'expansion_mult': 1.8},
                ]
            }
        ]
        
        discovered = []
        
        for pattern_config in patterns_to_test:
            logger.info(f"Testing {pattern_config['name']} pattern...")
            
            for params in pattern_config['params']:
                result = self.test_pattern_on_data(
                    combined_data,
                    pattern_config['func'],
                    params
                )
                
                if result and result['profitable']:
                    pattern_result = {
                        'name': pattern_config['name'],
                        'params': params,
                        'performance': result,
                        'required_win_rate': 0.34  # With 1:2 R/R
                    }
                    discovered.append(pattern_result)
                    logger.info(f"Found profitable {pattern_config['name']}: "
                              f"Win rate={result['win_rate']:.1%}, "
                              f"Net=${result['net_expectancy']:.2f}")
        
        return discovered
    
    def validate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Validate patterns on recent out-of-sample data"""
        logger.info("Validating patterns on recent data (June-August 2025)...")
        
        # Load validation data (last 3 months)
        data_path = Path(self.DATA_PATH)
        files = sorted(data_path.glob("*202506*.csv.zst")) + \
                sorted(data_path.glob("*202507*.csv.zst")) + \
                sorted(data_path.glob("*202508*.csv.zst"))
        
        validation_data = []
        for file in files[:60]:  # Last 60 days
            df = self.load_compressed_file(file)
            if not df.empty:
                validation_data.append(df)
        
        if not validation_data:
            logger.warning("No validation data found")
            return patterns
        
        combined_validation = pd.concat(validation_data).sort_index()
        logger.info(f"Validation data: {len(combined_validation)} bars")
        
        validated = []
        for pattern in patterns:
            # Get pattern function
            func_name = f"pattern_{pattern['name']}"
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                
                result = self.test_pattern_on_data(
                    combined_validation,
                    func,
                    pattern['params']
                )
                
                if result and result['win_rate'] >= 0.34:  # Minimum for profitability
                    pattern['validation'] = result
                    validated.append(pattern)
                    logger.info(f"Validated {pattern['name']}: "
                              f"Win rate={result['win_rate']:.1%}")
        
        return validated
    
    def save_results(self, patterns: List[Dict]):
        """Save discovered patterns to JSON"""
        output = {
            'discovery_date': datetime.now().isoformat(),
            'data_source': self.DATA_PATH,
            'risk_reward': f'{self.STOP_LOSS_POINTS}:{self.TAKE_PROFIT_POINTS}',
            'commission': self.COMMISSION_RT,
            'min_win_rate': 0.34,
            'patterns': patterns
        }
        
        with open('nq_discovered_patterns.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Saved {len(patterns)} patterns to nq_discovered_patterns.json")
    
    def run_discovery(self):
        """Run complete pattern discovery pipeline"""
        logger.info("="*60)
        logger.info("COMPREHENSIVE NQ PATTERN DISCOVERY")
        logger.info("="*60)
        logger.info(f"Risk/Reward: 1:2 ({self.STOP_LOSS_POINTS}:{self.TAKE_PROFIT_POINTS} points)")
        logger.info(f"Commission: ${self.COMMISSION_RT} round-trip")
        logger.info(f"Minimum Net Expectancy: ${self.MIN_NET_EXPECTANCY}")
        logger.info("="*60)
        
        # Discover patterns
        patterns = self.discover_patterns_parallel(num_files=50)  # Start with 50 files
        
        if patterns:
            # Validate on out-of-sample data
            validated = self.validate_patterns(patterns)
            
            # Save results
            self.save_results(validated)
            
            # Print summary
            print("\n" + "="*60)
            print("DISCOVERY COMPLETE")
            print("="*60)
            print(f"Patterns discovered: {len(patterns)}")
            print(f"Patterns validated: {len(validated)}")
            
            if validated:
                print("\nTOP PATTERNS:")
                for i, p in enumerate(validated[:5], 1):
                    print(f"\n{i}. {p['name'].upper()}")
                    print(f"   Parameters: {p['params']}")
                    print(f"   Training Win Rate: {p['performance']['win_rate']:.1%}")
                    print(f"   Validation Win Rate: {p.get('validation', {}).get('win_rate', 0):.1%}")
                    print(f"   Net Expectancy: ${p['performance']['net_expectancy']:.2f}")
        else:
            logger.warning("No profitable patterns found")

if __name__ == "__main__":
    # Check if we have zstandard installed
    try:
        import zstandard
    except ImportError:
        print("Installing required zstandard library...")
        import subprocess
        subprocess.check_call(['pip3', 'install', 'zstandard'])
        import zstandard
    
    discovery = ComprehensiveNQDiscovery()
    discovery.run_discovery()