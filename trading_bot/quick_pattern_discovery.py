"""
Quick NQ Pattern Discovery with Fixed 1:2 Risk/Reward
Focus on finding patterns that work with 5-point stops and 10-point targets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime
import logging
from pathlib import Path
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickNQDiscovery:
    """Fast pattern discovery for NQ with fixed R/R"""
    
    STOP_LOSS = 5
    TAKE_PROFIT = 10
    COMMISSION = 2.52
    POINT_VALUE = 20
    
    def load_recent_data(self, num_files: int = 30) -> pd.DataFrame:
        """Load recent NQ data files"""
        data_path = Path("/Users/royaltyvixion/Documents/XTRADING/databento_captures")
        files = sorted(data_path.glob("NQ_*_S-GLBX-MDP3_*"))[-num_files:]
        
        all_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                all_data.append(df)
                logger.info(f"Loaded {file.name}")
            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")
        
        if all_data:
            return pd.concat(all_data).sort_index()
        return pd.DataFrame()
    
    def test_bollinger_mean_reversion(self, data: pd.DataFrame) -> Dict:
        """Test Bollinger Band mean reversion with various parameters"""
        best_result = {'win_rate': 0, 'signals': 0, 'params': {}}
        
        for period in [10, 15, 20]:
            for std_dev in [1.5, 2.0, 2.5]:
                signals = []
                
                close = data['close'].values.astype(np.float64)
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                
                # Calculate Bollinger Bands
                sma = talib.SMA(close, timeperiod=period)
                std = talib.STDDEV(close, timeperiod=period)
                upper = sma + (std * std_dev)
                lower = sma - (std * std_dev)
                
                # Also calculate RSI for confirmation
                rsi = talib.RSI(close, timeperiod=14)
                
                for i in range(period + 14, len(data) - 50):
                    entry_triggered = False
                    direction = 0
                    
                    # Check for oversold bounce (long)
                    if close[i] <= lower[i] and rsi[i] < 35:
                        entry_triggered = True
                        direction = 1
                    # Check for overbought fade (short)
                    elif close[i] >= upper[i] and rsi[i] > 65:
                        entry_triggered = True
                        direction = -1
                    
                    if entry_triggered:
                        entry_price = close[i]
                        
                        # Set fixed stops and targets
                        if direction == 1:
                            stop = entry_price - self.STOP_LOSS
                            target = entry_price + self.TAKE_PROFIT
                        else:
                            stop = entry_price + self.STOP_LOSS
                            target = entry_price - self.TAKE_PROFIT
                        
                        # Check outcome
                        hit_target = False
                        hit_stop = False
                        
                        for j in range(i+1, min(i+50, len(data))):
                            if direction == 1:
                                if low[j] <= stop:
                                    hit_stop = True
                                    break
                                if high[j] >= target:
                                    hit_target = True
                                    break
                            else:
                                if high[j] >= stop:
                                    hit_stop = True
                                    break
                                if low[j] <= target:
                                    hit_target = True
                                    break
                        
                        if hit_target or hit_stop:
                            signals.append(1 if hit_target else 0)
                
                if len(signals) >= 20:  # Minimum signals needed
                    win_rate = sum(signals) / len(signals)
                    
                    # Calculate net expectancy
                    gross_exp = (win_rate * self.TAKE_PROFIT * self.POINT_VALUE) - \
                               ((1 - win_rate) * self.STOP_LOSS * self.POINT_VALUE)
                    net_exp = gross_exp - self.COMMISSION
                    
                    if net_exp > 3 and win_rate > best_result['win_rate']:
                        best_result = {
                            'pattern': 'bollinger_mean_reversion',
                            'win_rate': win_rate,
                            'signals': len(signals),
                            'net_expectancy': net_exp,
                            'params': {
                                'period': period,
                                'std_dev': std_dev,
                                'rsi_oversold': 35,
                                'rsi_overbought': 65
                            }
                        }
        
        return best_result
    
    def test_volume_spike_pattern(self, data: pd.DataFrame) -> Dict:
        """Test volume spike reversal pattern"""
        best_result = {'win_rate': 0, 'signals': 0, 'params': {}}
        
        for vol_mult in [2.0, 2.5, 3.0]:
            for price_move in [0.002, 0.003, 0.004]:
                signals = []
                
                close = data['close'].values.astype(np.float64)
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                volume = data['volume'].values.astype(np.float64)
                
                # Volume moving average
                vol_ma = talib.SMA(volume, timeperiod=20)
                
                for i in range(25, len(data) - 50):
                    # Check for volume spike
                    if volume[i] > vol_ma[i] * vol_mult:
                        # Check for price spike to fade
                        price_change = (close[i] - close[i-5]) / close[i-5]
                        
                        entry_triggered = False
                        direction = 0
                        
                        if price_change > price_move:  # Fade up move
                            entry_triggered = True
                            direction = -1
                        elif price_change < -price_move:  # Fade down move
                            entry_triggered = True
                            direction = 1
                        
                        if entry_triggered:
                            entry_price = close[i]
                            
                            # Set fixed stops and targets
                            if direction == 1:
                                stop = entry_price - self.STOP_LOSS
                                target = entry_price + self.TAKE_PROFIT
                            else:
                                stop = entry_price + self.STOP_LOSS
                                target = entry_price - self.TAKE_PROFIT
                            
                            # Check outcome
                            hit_target = False
                            hit_stop = False
                            
                            for j in range(i+1, min(i+50, len(data))):
                                if direction == 1:
                                    if low[j] <= stop:
                                        hit_stop = True
                                        break
                                    if high[j] >= target:
                                        hit_target = True
                                        break
                                else:
                                    if high[j] >= stop:
                                        hit_stop = True
                                        break
                                    if low[j] <= target:
                                        hit_target = True
                                        break
                            
                            if hit_target or hit_stop:
                                signals.append(1 if hit_target else 0)
                
                if len(signals) >= 10:
                    win_rate = sum(signals) / len(signals)
                    
                    # Calculate net expectancy
                    gross_exp = (win_rate * self.TAKE_PROFIT * self.POINT_VALUE) - \
                               ((1 - win_rate) * self.STOP_LOSS * self.POINT_VALUE)
                    net_exp = gross_exp - self.COMMISSION
                    
                    if net_exp > 3 and win_rate > best_result['win_rate']:
                        best_result = {
                            'pattern': 'volume_spike_fade',
                            'win_rate': win_rate,
                            'signals': len(signals),
                            'net_expectancy': net_exp,
                            'params': {
                                'vol_multiplier': vol_mult,
                                'price_threshold': price_move
                            }
                        }
        
        return best_result
    
    def test_momentum_pattern(self, data: pd.DataFrame) -> Dict:
        """Test momentum continuation pattern"""
        best_result = {'win_rate': 0, 'signals': 0, 'params': {}}
        
        for lookback in [5, 10, 15]:
            for mom_thresh in [0.002, 0.003, 0.004]:
                signals = []
                
                close = data['close'].values.astype(np.float64)
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                volume = data['volume'].values.astype(np.float64)
                
                vol_ma = talib.SMA(volume, timeperiod=20)
                
                for i in range(max(20, lookback), len(data) - 50):
                    # Calculate momentum
                    momentum = (close[i] - close[i-lookback]) / close[i-lookback]
                    
                    # Volume confirmation
                    vol_confirm = volume[i] > vol_ma[i] * 1.3
                    
                    entry_triggered = False
                    direction = 0
                    
                    if momentum > mom_thresh and vol_confirm:
                        entry_triggered = True
                        direction = 1
                    elif momentum < -mom_thresh and vol_confirm:
                        entry_triggered = True
                        direction = -1
                    
                    if entry_triggered:
                        entry_price = close[i]
                        
                        # Set fixed stops and targets
                        if direction == 1:
                            stop = entry_price - self.STOP_LOSS
                            target = entry_price + self.TAKE_PROFIT
                        else:
                            stop = entry_price + self.STOP_LOSS
                            target = entry_price - self.TAKE_PROFIT
                        
                        # Check outcome
                        hit_target = False
                        hit_stop = False
                        
                        for j in range(i+1, min(i+50, len(data))):
                            if direction == 1:
                                if low[j] <= stop:
                                    hit_stop = True
                                    break
                                if high[j] >= target:
                                    hit_target = True
                                    break
                            else:
                                if high[j] >= stop:
                                    hit_stop = True
                                    break
                                if low[j] <= target:
                                    hit_target = True
                                    break
                        
                        if hit_target or hit_stop:
                            signals.append(1 if hit_target else 0)
                
                if len(signals) >= 10:
                    win_rate = sum(signals) / len(signals)
                    
                    # Calculate net expectancy
                    gross_exp = (win_rate * self.TAKE_PROFIT * self.POINT_VALUE) - \
                               ((1 - win_rate) * self.STOP_LOSS * self.POINT_VALUE)
                    net_exp = gross_exp - self.COMMISSION
                    
                    if net_exp > 3 and win_rate > best_result['win_rate']:
                        best_result = {
                            'pattern': 'momentum_continuation',
                            'win_rate': win_rate,
                            'signals': len(signals),
                            'net_expectancy': net_exp,
                            'params': {
                                'lookback': lookback,
                                'momentum_threshold': mom_thresh,
                                'volume_multiplier': 1.3
                            }
                        }
        
        return best_result
    
    def run(self):
        """Run quick pattern discovery"""
        logger.info("Loading recent NQ data...")
        data = self.load_recent_data(num_files=30)
        
        if data.empty:
            logger.error("No data loaded")
            return
        
        logger.info(f"Loaded {len(data)} bars of data")
        
        # Test each pattern type
        results = []
        
        logger.info("Testing Bollinger mean reversion...")
        bb_result = self.test_bollinger_mean_reversion(data)
        if bb_result.get('win_rate', 0) > 0:
            results.append(bb_result)
        
        logger.info("Testing volume spike pattern...")
        vol_result = self.test_volume_spike_pattern(data)
        if vol_result.get('win_rate', 0) > 0:
            results.append(vol_result)
        
        logger.info("Testing momentum pattern...")
        mom_result = self.test_momentum_pattern(data)
        if mom_result.get('win_rate', 0) > 0:
            results.append(mom_result)
        
        # Sort by net expectancy
        results.sort(key=lambda x: x.get('net_expectancy', 0), reverse=True)
        
        # Create output
        output = {
            'discovery_date': datetime.now().isoformat(),
            'risk_reward': '1:2 (5 points : 10 points)',
            'commission': self.COMMISSION,
            'patterns': results
        }
        
        # Save to file
        with open('nq_patterns_1to2_rr.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        # Print results
        print("\n" + "="*60)
        print("NQ PATTERN DISCOVERY RESULTS (1:2 R/R)")
        print("="*60)
        print(f"Stop Loss: {self.STOP_LOSS} points ($100)")
        print(f"Take Profit: {self.TAKE_PROFIT} points ($200)")
        print(f"Commission: ${self.COMMISSION} round-trip")
        print(f"Minimum win rate needed: 34%")
        print()
        
        if results:
            print("DISCOVERED PATTERNS:")
            print("-"*60)
            for i, pattern in enumerate(results, 1):
                print(f"\n{i}. {pattern['pattern'].upper()}")
                print(f"   Win Rate: {pattern['win_rate']*100:.1f}%")
                print(f"   Net Expectancy: ${pattern['net_expectancy']:.2f}")
                print(f"   Total Signals: {pattern['signals']}")
                print(f"   Parameters: {pattern['params']}")
        else:
            print("No profitable patterns found with 1:2 R/R")

if __name__ == "__main__":
    discovery = QuickNQDiscovery()
    discovery.run()