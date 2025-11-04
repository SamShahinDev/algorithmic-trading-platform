"""
NQ Pattern Discovery with Fixed Risk/Reward and Multi-Timeframe Analysis
Discovers patterns that can achieve 10-point profits with 5-point stops
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, time
import logging
from pathlib import Path
import talib
from concurrent.futures import ProcessPoolExecutor, as_completed
from data.data_loader import HybridDataLoader
from data.data_transformer import DataTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NQPatternDiscovery:
    """Discover profitable NQ patterns with fixed 1:2 risk/reward"""
    
    # Fixed risk/reward parameters
    STOP_LOSS_POINTS = 5      # $100 risk per contract
    TAKE_PROFIT_POINTS = 10   # $200 reward per contract
    COMMISSION_RT = 2.52      # Round-trip commission
    MIN_NET_EXPECTANCY = 3.0  # Minimum $3 net profit after commissions
    
    # NQ contract specifications
    POINT_VALUE = 20  # $20 per point for MNQ
    
    def __init__(self, data_path: str = "/Users/royaltyvixion/Documents/XTRADING/databento_captures"):
        self.data_path = Path(data_path)
        self.data_loader = HybridDataLoader()
        self.transformer = DataTransformer()
        self.discovered_patterns = []
        
    def calculate_net_expectancy(self, win_rate: float) -> float:
        """
        Calculate net expectancy with fixed R/R and commissions
        
        Net Expectancy = (Win% × Win$) - (Loss% × Loss$) - Commission
        """
        win_dollars = self.TAKE_PROFIT_POINTS * self.POINT_VALUE
        loss_dollars = self.STOP_LOSS_POINTS * self.POINT_VALUE
        
        gross_expectancy = (win_rate * win_dollars) - ((1 - win_rate) * loss_dollars)
        net_expectancy = gross_expectancy - self.COMMISSION_RT
        
        return net_expectancy
    
    def calculate_required_winrate(self) -> float:
        """Calculate minimum win rate needed for profitability"""
        # With 1:2 R/R and commission, solve for breakeven win rate
        win = self.TAKE_PROFIT_POINTS * self.POINT_VALUE
        loss = self.STOP_LOSS_POINTS * self.POINT_VALUE
        commission = self.COMMISSION_RT
        
        # Breakeven: win_rate * win - (1-win_rate) * loss - commission = MIN_NET_EXPECTANCY
        # win_rate * (win + loss) = loss + commission + MIN_NET_EXPECTANCY
        required_winrate = (loss + commission + self.MIN_NET_EXPECTANCY) / (win + loss)
        
        return required_winrate
    
    def test_pattern_fixed_rr(self, data: pd.DataFrame, pattern_func, 
                              pattern_params: Dict) -> Dict:
        """
        Test a pattern with fixed 5-point stop and 10-point target
        
        Returns:
            Dictionary with performance metrics
        """
        signals = []
        
        for i in range(100, len(data) - 50):  # Leave room for outcome
            window = data.iloc[i-100:i]
            
            # Check if pattern triggers
            signal = pattern_func(window, **pattern_params)
            
            if signal is not None:
                entry_price = data.iloc[i]['close']
                entry_time = data.index[i]
                
                # Fixed stop and target
                if signal > 0:  # Long signal
                    stop_price = entry_price - self.STOP_LOSS_POINTS
                    target_price = entry_price + self.TAKE_PROFIT_POINTS
                else:  # Short signal
                    stop_price = entry_price + self.STOP_LOSS_POINTS
                    target_price = entry_price - self.TAKE_PROFIT_POINTS
                
                # Check outcome over next 50 bars (50 minutes)
                outcome_data = data.iloc[i:i+50]
                
                hit_stop = False
                hit_target = False
                exit_bar = None
                
                for j, (idx, bar) in enumerate(outcome_data.iterrows()):
                    if signal > 0:  # Long position
                        if bar['low'] <= stop_price:
                            hit_stop = True
                            exit_bar = j
                            break
                        if bar['high'] >= target_price:
                            hit_target = True
                            exit_bar = j
                            break
                    else:  # Short position
                        if bar['high'] >= stop_price:
                            hit_stop = True
                            exit_bar = j
                            break
                        if bar['low'] <= target_price:
                            hit_target = True
                            exit_bar = j
                            break
                
                # Record result
                if hit_target or hit_stop:
                    signals.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'direction': signal,
                        'hit_target': hit_target,
                        'hit_stop': hit_stop,
                        'bars_to_exit': exit_bar,
                        'profit_points': self.TAKE_PROFIT_POINTS if hit_target else -self.STOP_LOSS_POINTS
                    })
        
        # Calculate metrics
        if len(signals) < 10:  # Need minimum samples
            return None
        
        wins = sum(1 for s in signals if s['hit_target'])
        win_rate = wins / len(signals)
        net_expectancy = self.calculate_net_expectancy(win_rate)
        
        # Average time in trade
        avg_bars = np.mean([s['bars_to_exit'] for s in signals if s['bars_to_exit']])
        
        return {
            'total_signals': len(signals),
            'win_rate': win_rate,
            'net_expectancy': net_expectancy,
            'avg_bars_to_exit': avg_bars,
            'profitable': net_expectancy >= self.MIN_NET_EXPECTANCY,
            'params': pattern_params
        }
    
    def pattern_mean_reversion_bb(self, data: pd.DataFrame, bb_period: int = 20, 
                                  bb_std: float = 2.0, rsi_period: int = 14,
                                  rsi_oversold: int = 30, rsi_overbought: int = 70) -> Optional[int]:
        """Mean reversion pattern with Bollinger Bands and RSI"""
        if len(data) < max(bb_period, rsi_period):
            return None
        
        close = data['close'].values.astype(np.float64)
        
        # Calculate indicators
        sma = talib.SMA(close, timeperiod=bb_period)
        std = talib.STDDEV(close, timeperiod=bb_period)
        upper_band = sma + (std * bb_std)
        lower_band = sma - (std * bb_std)
        rsi = talib.RSI(close, timeperiod=rsi_period)
        
        current_close = close[-1]
        current_rsi = rsi[-1]
        
        # Check for signal
        if current_close <= lower_band[-1] and current_rsi < rsi_oversold:
            return 1  # Long signal
        elif current_close >= upper_band[-1] and current_rsi > rsi_overbought:
            return -1  # Short signal
        
        return None
    
    def pattern_volume_breakout(self, data: pd.DataFrame, vol_multiplier: float = 2.0,
                               price_threshold: float = 0.002) -> Optional[int]:
        """Volume breakout pattern"""
        if len(data) < 20:
            return None
        
        volume = data['volume'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        
        vol_ma = talib.SMA(volume, timeperiod=20)
        
        # Check for volume spike with price move
        if volume[-1] > vol_ma[-1] * vol_multiplier:
            price_change = (close[-1] - close[-2]) / close[-2]
            
            if price_change > price_threshold:
                return 1  # Long signal
            elif price_change < -price_threshold:
                return -1  # Short signal
        
        return None
    
    def pattern_momentum_continuation(self, data: pd.DataFrame, lookback: int = 10,
                                     momentum_threshold: float = 0.003) -> Optional[int]:
        """Momentum continuation pattern"""
        if len(data) < lookback + 5:
            return None
        
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        
        # Calculate momentum
        momentum = (close[-1] - close[-lookback]) / close[-lookback]
        
        # Volume confirmation
        vol_recent = volume[-3:].mean()
        vol_avg = volume[-20:].mean()
        
        if vol_recent > vol_avg * 1.5:  # Volume confirmation
            if momentum > momentum_threshold:
                return 1  # Long signal
            elif momentum < -momentum_threshold:
                return -1  # Short signal
        
        return None
    
    def pattern_opening_range_breakout(self, data: pd.DataFrame, range_minutes: int = 30,
                                      buffer_points: float = 2) -> Optional[int]:
        """Opening range breakout for RTH session"""
        if len(data) < range_minutes + 10:
            return None
        
        # Check if we're in the right time window (after opening range)
        current_time = data.index[-1].time()
        if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30 + range_minutes):
            return None
        
        # Find today's opening range
        today_start = data.index[-1].replace(hour=8, minute=30, second=0)
        opening_range = data[data.index >= today_start].head(range_minutes)
        
        if len(opening_range) < range_minutes:
            return None
        
        range_high = opening_range['high'].max()
        range_low = opening_range['low'].min()
        
        current_close = data['close'].iloc[-1]
        
        # Check for breakout with buffer
        if current_close > range_high + buffer_points:
            return 1  # Long breakout
        elif current_close < range_low - buffer_points:
            return -1  # Short breakout
        
        return None
    
    def discover_patterns_for_timeframe(self, timeframe: str, 
                                       start_date: str = "2024-06-01",
                                       end_date: str = "2025-05-31") -> List[Dict]:
        """
        Discover patterns for a specific timeframe
        
        Args:
            timeframe: '1m', '5m', or '15m'
            start_date: Start date for training data
            end_date: End date for training data
        
        Returns:
            List of discovered patterns with parameters
        """
        logger.info(f"Discovering patterns for {timeframe} timeframe...")
        
        # Load and prepare data
        all_data = []
        files = sorted(self.data_path.glob("NQ_*_S-GLBX-MDP3_*"))
        
        for file in files[:100]:  # Process subset for faster iteration
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Convert to target timeframe if needed
                if timeframe != '1m':
                    df = self.transformer.convert_to_timeframe(df, timeframe)
                
                # Filter to RTH only
                df = self.transformer.filter_rth_only(df)
                
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Error processing {file}: {e}")
                continue
        
        if not all_data:
            return []
        
        # Combine all data
        combined_data = pd.concat(all_data).sort_index()
        
        # Parameter grids for optimization
        param_grids = {
            'mean_reversion_bb': {
                'bb_period': [10, 15, 20],
                'bb_std': [1.5, 2.0, 2.5],
                'rsi_period': [10, 14],
                'rsi_oversold': [25, 30],
                'rsi_overbought': [70, 75]
            },
            'volume_breakout': {
                'vol_multiplier': [1.5, 2.0, 2.5, 3.0],
                'price_threshold': [0.001, 0.002, 0.003]
            },
            'momentum_continuation': {
                'lookback': [5, 10, 15],
                'momentum_threshold': [0.002, 0.003, 0.004]
            }
        }
        
        discovered = []
        min_required_winrate = self.calculate_required_winrate()
        
        logger.info(f"Minimum required win rate for profitability: {min_required_winrate:.1%}")
        
        # Test each pattern type
        for pattern_name, param_grid in param_grids.items():
            logger.info(f"Testing {pattern_name} pattern...")
            
            # Get pattern function
            pattern_func = getattr(self, f"pattern_{pattern_name}")
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_grid)
            
            best_result = None
            best_expectancy = -float('inf')
            
            for params in param_combinations[:20]:  # Limit combinations for speed
                result = self.test_pattern_fixed_rr(combined_data, pattern_func, params)
                
                if result and result['profitable']:
                    if result['net_expectancy'] > best_expectancy:
                        best_expectancy = result['net_expectancy']
                        best_result = result
                        best_result['pattern_name'] = pattern_name
                        best_result['timeframe'] = timeframe
            
            if best_result:
                discovered.append(best_result)
                logger.info(f"Found profitable {pattern_name}: Win rate={best_result['win_rate']:.1%}, "
                          f"Net expectancy=${best_result['net_expectancy']:.2f}")
        
        return discovered
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def discover_multi_timeframe_patterns(self) -> Dict:
        """
        Discover patterns across multiple timeframes
        Following the hierarchy: 15m bias → 5m pattern → 1m entry
        """
        results = {
            'discovery_date': datetime.now().isoformat(),
            'risk_reward': f"{self.STOP_LOSS_POINTS}:{self.TAKE_PROFIT_POINTS}",
            'commission': self.COMMISSION_RT,
            'min_net_expectancy': self.MIN_NET_EXPECTANCY,
            'patterns': []
        }
        
        # Discover patterns for each timeframe
        for tf in ['1m', '5m', '15m']:
            tf_patterns = self.discover_patterns_for_timeframe(tf)
            results['patterns'].extend(tf_patterns)
        
        # Sort by net expectancy
        results['patterns'].sort(key=lambda x: x['net_expectancy'], reverse=True)
        
        # Keep only top patterns
        results['patterns'] = results['patterns'][:10]
        
        return results
    
    def validate_on_recent_data(self, patterns: List[Dict], 
                               validation_start: str = "2025-06-01") -> List[Dict]:
        """
        Validate discovered patterns on recent out-of-sample data
        
        Args:
            patterns: List of discovered patterns
            validation_start: Start date for validation period
        
        Returns:
            Patterns with validation metrics added
        """
        logger.info("Validating patterns on recent data...")
        
        # Load recent data for validation
        validation_data = []
        files = sorted(self.data_path.glob("NQ_*_S-GLBX-MDP3_*"))
        
        for file in files[-60:]:  # Last 2 months of data
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Filter to validation period
                if df.index[-1] >= pd.Timestamp(validation_start):
                    validation_data.append(df)
            except Exception as e:
                continue
        
        if not validation_data:
            logger.warning("No validation data found")
            return patterns
        
        combined_validation = pd.concat(validation_data).sort_index()
        
        # Validate each pattern
        for pattern in patterns:
            pattern_name = pattern['pattern_name']
            pattern_func = getattr(self, f"pattern_{pattern_name}")
            timeframe = pattern.get('timeframe', '1m')
            
            # Convert data to pattern's timeframe
            if timeframe != '1m':
                test_data = self.transformer.convert_to_timeframe(combined_validation, timeframe)
            else:
                test_data = combined_validation
            
            # Test on validation data
            validation_result = self.test_pattern_fixed_rr(
                test_data, 
                pattern_func, 
                pattern['params']
            )
            
            if validation_result:
                pattern['validation_win_rate'] = validation_result['win_rate']
                pattern['validation_expectancy'] = validation_result['net_expectancy']
                pattern['validation_signals'] = validation_result['total_signals']
                pattern['validation_passed'] = validation_result['profitable']
            else:
                pattern['validation_passed'] = False
        
        # Filter to only validated patterns
        validated = [p for p in patterns if p.get('validation_passed', False)]
        
        logger.info(f"Validation complete: {len(validated)}/{len(patterns)} patterns passed")
        
        return validated
    
    def save_discovered_patterns(self, patterns: Dict, filename: str = "nq_patterns_fixed_rr.json"):
        """Save discovered patterns to JSON file"""
        with open(filename, 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        logger.info(f"Saved {len(patterns['patterns'])} patterns to {filename}")
    
    def run_discovery(self):
        """Run complete pattern discovery pipeline"""
        logger.info("Starting NQ pattern discovery with fixed 1:2 risk/reward...")
        logger.info(f"Parameters: {self.STOP_LOSS_POINTS} point stop, "
                   f"{self.TAKE_PROFIT_POINTS} point target")
        
        # Discover patterns
        results = self.discover_multi_timeframe_patterns()
        
        # Validate on recent data
        if results['patterns']:
            validated = self.validate_on_recent_data(results['patterns'])
            results['patterns'] = validated
        
        # Save results
        self.save_discovered_patterns(results)
        
        # Print summary
        print("\n" + "="*60)
        print("NQ PATTERN DISCOVERY RESULTS")
        print("="*60)
        print(f"Risk/Reward: 1:2 ({self.STOP_LOSS_POINTS} points : {self.TAKE_PROFIT_POINTS} points)")
        print(f"Commission: ${self.COMMISSION_RT} round-trip")
        print(f"Minimum Net Expectancy: ${self.MIN_NET_EXPECTANCY}")
        print(f"Discovered Patterns: {len(results['patterns'])}")
        
        if results['patterns']:
            print("\nTop Patterns:")
            print("-"*60)
            for i, pattern in enumerate(results['patterns'][:5], 1):
                print(f"\n{i}. {pattern['pattern_name']} ({pattern['timeframe']})")
                print(f"   Win Rate: {pattern['win_rate']:.1%}")
                print(f"   Net Expectancy: ${pattern['net_expectancy']:.2f}")
                print(f"   Validation Win Rate: {pattern.get('validation_win_rate', 0):.1%}")
                print(f"   Parameters: {pattern['params']}")
        
        return results

if __name__ == "__main__":
    discovery = NQPatternDiscovery()
    discovery.run_discovery()