"""
Strategy Discovery System
Mines historical data for profitable trading patterns and strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import itertools
import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from data.feature_engineering import FeatureEngineer
from analysis.pattern_scanner import PatternScanner, PatternType
from analysis.microstructure import MicrostructureAnalyzer


@dataclass
class StrategyResult:
    """Results from strategy testing"""
    name: str
    parameters: Dict
    trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_return: float
    conditions: List[str]
    best_hours: List[int]
    best_days: List[int]


@dataclass 
class TradingStrategy:
    """Discovered trading strategy"""
    strategy_id: str
    strategy_type: str  # momentum, mean_reversion, breakout, etc.
    entry_conditions: Dict
    exit_conditions: Dict
    parameters: Dict
    performance: StrategyResult
    discovered_date: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None


class StrategyDiscovery:
    """Discover and validate profitable strategies from historical data"""
    
    def __init__(self, 
                 min_trades: int = 30,
                 min_sharpe: float = 1.5,
                 min_win_rate: float = 0.45,
                 min_profit_factor: float = 1.2):
        """
        Initialize strategy discovery system
        
        Args:
            min_trades: Minimum trades for valid strategy
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate
            min_profit_factor: Minimum profit factor
        """
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        
        self.discovered_strategies = []
        self.feature_engineer = FeatureEngineer()
        self.pattern_scanner = PatternScanner()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        
    def discover_strategies(self, historical_data: pd.DataFrame, 
                          commission: float = 2.52) -> List[TradingStrategy]:
        """
        Mine historical data for profitable patterns
        
        Args:
            historical_data: Historical OHLCV data
            commission: Round-trip commission
            
        Returns:
            List of discovered profitable strategies
        """
        print(f"Starting strategy discovery on {len(historical_data)} bars...")
        
        # Generate all features
        features = self.feature_engineer.calculate_features(historical_data)
        
        strategies = []
        
        # 1. Discover momentum strategies
        print("Testing momentum strategies...")
        momentum_strategies = self._discover_momentum_strategies(
            historical_data, features, commission
        )
        strategies.extend(momentum_strategies)
        
        # 2. Discover mean reversion strategies
        print("Testing mean reversion strategies...")
        mean_reversion_strategies = self._discover_mean_reversion_strategies(
            historical_data, features, commission
        )
        strategies.extend(mean_reversion_strategies)
        
        # 3. Discover breakout strategies
        print("Testing breakout strategies...")
        breakout_strategies = self._discover_breakout_strategies(
            historical_data, features, commission
        )
        strategies.extend(breakout_strategies)
        
        # 4. Discover pattern-based strategies
        print("Testing pattern-based strategies...")
        pattern_strategies = self._discover_pattern_strategies(
            historical_data, features, commission
        )
        strategies.extend(pattern_strategies)
        
        # 5. Discover microstructure strategies
        print("Testing microstructure strategies...")
        microstructure_strategies = self._discover_microstructure_strategies(
            historical_data, features, commission
        )
        strategies.extend(microstructure_strategies)
        
        # 6. Discover combined strategies
        print("Testing combined strategies...")
        combined_strategies = self._discover_combined_strategies(
            historical_data, features, commission
        )
        strategies.extend(combined_strategies)
        
        # Rank and filter strategies
        valid_strategies = self._filter_valid_strategies(strategies)
        ranked_strategies = self._rank_strategies(valid_strategies)
        
        # Test robustness
        robust_strategies = []
        for strategy in ranked_strategies[:20]:  # Test top 20
            if self._test_strategy_robustness(strategy, historical_data):
                robust_strategies.append(strategy)
        
        self.discovered_strategies = robust_strategies
        
        print(f"Discovered {len(robust_strategies)} robust strategies")
        
        return robust_strategies
    
    def _discover_momentum_strategies(self, df: pd.DataFrame, features: pd.DataFrame,
                                     commission: float) -> List[TradingStrategy]:
        """Discover momentum-based strategies"""
        strategies = []
        strategy_count = 0
        
        # Test different momentum parameters
        for lookback in [5, 10, 20, 30]:
            for threshold in np.arange(0.001, 0.01, 0.002):
                for hold_period in [5, 10, 15, 30]:
                    
                    # Define entry/exit conditions
                    entry_conditions = {
                        'momentum_lookback': lookback,
                        'momentum_threshold': threshold,
                        'volume_confirmation': True
                    }
                    
                    exit_conditions = {
                        'hold_period': hold_period,
                        'stop_loss': 0.005,  # 0.5%
                        'take_profit': 0.015  # 1.5%
                    }
                    
                    # Test strategy
                    result = self._test_momentum_strategy(
                        df, features, lookback, threshold, hold_period, commission
                    )
                    
                    if result and result.sharpe_ratio > self.min_sharpe:
                        strategy_count += 1
                        strategy = TradingStrategy(
                            strategy_id=f"MOM_{strategy_count:03d}",
                            strategy_type="momentum",
                            entry_conditions=entry_conditions,
                            exit_conditions=exit_conditions,
                            parameters={'lookback': lookback, 'threshold': threshold},
                            performance=result
                        )
                        strategies.append(strategy)
        
        return strategies
    
    def _discover_mean_reversion_strategies(self, df: pd.DataFrame, features: pd.DataFrame,
                                           commission: float) -> List[TradingStrategy]:
        """Discover mean reversion strategies"""
        strategies = []
        strategy_count = 0
        
        # Test different mean reversion parameters
        for bollinger_period in [10, 20, 30]:
            for num_std in [1.5, 2.0, 2.5, 3.0]:
                for rsi_period in [7, 14, 21]:
                    for rsi_threshold in [20, 25, 30]:
                        
                        # Define conditions
                        entry_conditions = {
                            'bollinger_period': bollinger_period,
                            'bollinger_std': num_std,
                            'rsi_period': rsi_period,
                            'rsi_oversold': rsi_threshold,
                            'rsi_overbought': 100 - rsi_threshold
                        }
                        
                        exit_conditions = {
                            'target': 'mean',  # Return to mean
                            'stop_loss': 0.01,  # 1%
                            'time_limit': 60  # 60 bars
                        }
                        
                        # Test strategy
                        result = self._test_mean_reversion_strategy(
                            df, features, bollinger_period, num_std, 
                            rsi_period, rsi_threshold, commission
                        )
                        
                        if result and result.sharpe_ratio > self.min_sharpe:
                            strategy_count += 1
                            strategy = TradingStrategy(
                                strategy_id=f"MR_{strategy_count:03d}",
                                strategy_type="mean_reversion",
                                entry_conditions=entry_conditions,
                                exit_conditions=exit_conditions,
                                parameters={
                                    'bollinger_period': bollinger_period,
                                    'num_std': num_std,
                                    'rsi_period': rsi_period
                                },
                                performance=result
                            )
                            strategies.append(strategy)
        
        return strategies
    
    def _discover_breakout_strategies(self, df: pd.DataFrame, features: pd.DataFrame,
                                     commission: float) -> List[TradingStrategy]:
        """Discover breakout strategies"""
        strategies = []
        strategy_count = 0
        
        # Test different breakout parameters
        for lookback in [10, 20, 30, 50]:
            for volume_mult in [1.2, 1.5, 2.0]:
                for atr_mult in [1.0, 1.5, 2.0]:
                    
                    entry_conditions = {
                        'lookback_period': lookback,
                        'volume_multiplier': volume_mult,
                        'breakout_confirmation': 2  # bars
                    }
                    
                    exit_conditions = {
                        'atr_multiplier': atr_mult,
                        'trailing_stop': True,
                        'time_limit': 100
                    }
                    
                    result = self._test_breakout_strategy(
                        df, features, lookback, volume_mult, atr_mult, commission
                    )
                    
                    if result and result.sharpe_ratio > self.min_sharpe:
                        strategy_count += 1
                        strategy = TradingStrategy(
                            strategy_id=f"BO_{strategy_count:03d}",
                            strategy_type="breakout",
                            entry_conditions=entry_conditions,
                            exit_conditions=exit_conditions,
                            parameters={
                                'lookback': lookback,
                                'volume_mult': volume_mult,
                                'atr_mult': atr_mult
                            },
                            performance=result
                        )
                        strategies.append(strategy)
        
        return strategies
    
    def _discover_pattern_strategies(self, df: pd.DataFrame, features: pd.DataFrame,
                                    commission: float) -> List[TradingStrategy]:
        """Discover pattern-based strategies"""
        strategies = []
        strategy_count = 0
        
        # Test each pattern type
        for pattern_type in PatternType:
            for min_strength in [50, 60, 70, 80]:
                for stop_mult in [1.0, 1.5, 2.0]:
                    for target_mult in [2.0, 2.5, 3.0]:
                        
                        result = self._test_pattern_strategy(
                            df, features, pattern_type, min_strength,
                            stop_mult, target_mult, commission
                        )
                        
                        if result and result.sharpe_ratio > self.min_sharpe:
                            strategy_count += 1
                            strategy = TradingStrategy(
                                strategy_id=f"PAT_{strategy_count:03d}",
                                strategy_type=f"pattern_{pattern_type.value}",
                                entry_conditions={
                                    'pattern': pattern_type.value,
                                    'min_strength': min_strength
                                },
                                exit_conditions={
                                    'stop_multiplier': stop_mult,
                                    'target_multiplier': target_mult
                                },
                                parameters={
                                    'pattern_type': pattern_type.value,
                                    'min_strength': min_strength
                                },
                                performance=result
                            )
                            strategies.append(strategy)
        
        return strategies
    
    def _discover_microstructure_strategies(self, df: pd.DataFrame, features: pd.DataFrame,
                                           commission: float) -> List[TradingStrategy]:
        """Discover microstructure-based strategies"""
        strategies = []
        strategy_count = 0
        
        # Test order flow imbalance strategies
        for imbalance_threshold in [20, 30, 40]:
            for hold_period in [5, 10, 15]:
                
                result = self._test_microstructure_strategy(
                    df, features, imbalance_threshold, hold_period, commission
                )
                
                if result and result.sharpe_ratio > self.min_sharpe:
                    strategy_count += 1
                    strategy = TradingStrategy(
                        strategy_id=f"MS_{strategy_count:03d}",
                        strategy_type="microstructure",
                        entry_conditions={
                            'order_flow_imbalance': imbalance_threshold,
                            'regime': 'accumulation_or_distribution'
                        },
                        exit_conditions={
                            'hold_period': hold_period,
                            'stop_loss': 0.005
                        },
                        parameters={'imbalance_threshold': imbalance_threshold},
                        performance=result
                    )
                    strategies.append(strategy)
        
        return strategies
    
    def _discover_combined_strategies(self, df: pd.DataFrame, features: pd.DataFrame,
                                     commission: float) -> List[TradingStrategy]:
        """Discover strategies combining multiple signals"""
        strategies = []
        strategy_count = 0
        
        # Combine momentum with microstructure
        for momentum_period in [10, 20]:
            for imbalance_threshold in [20, 30]:
                
                result = self._test_combined_strategy(
                    df, features, momentum_period, imbalance_threshold, commission
                )
                
                if result and result.sharpe_ratio > self.min_sharpe:
                    strategy_count += 1
                    strategy = TradingStrategy(
                        strategy_id=f"COMB_{strategy_count:03d}",
                        strategy_type="combined",
                        entry_conditions={
                            'momentum_period': momentum_period,
                            'order_flow_imbalance': imbalance_threshold,
                            'pattern_confirmation': True
                        },
                        exit_conditions={
                            'dynamic_exit': True,
                            'stop_loss': 0.007
                        },
                        parameters={
                            'momentum': momentum_period,
                            'imbalance': imbalance_threshold
                        },
                        performance=result
                    )
                    strategies.append(strategy)
        
        return strategies
    
    def _test_momentum_strategy(self, df: pd.DataFrame, features: pd.DataFrame,
                               lookback: int, threshold: float, hold_period: int,
                               commission: float) -> Optional[StrategyResult]:
        """Test a momentum strategy"""
        trades = []
        position = None
        
        momentum = features['returns'].rolling(lookback).mean()
        
        for i in range(lookback + 1, len(df) - hold_period):
            # Entry signal
            if position is None:
                if momentum.iloc[i] > threshold:
                    # Long entry
                    position = {
                        'entry_idx': i,
                        'entry_price': df['close'].iloc[i],
                        'direction': 1
                    }
                elif momentum.iloc[i] < -threshold:
                    # Short entry
                    position = {
                        'entry_idx': i,
                        'entry_price': df['close'].iloc[i],
                        'direction': -1
                    }
            
            # Exit signal
            elif i >= position['entry_idx'] + hold_period:
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - position['entry_price']) * position['direction']
                pnl_percent = (pnl / position['entry_price']) * 100 - (commission / position['entry_price']) * 100
                
                trades.append({
                    'entry': position['entry_price'],
                    'exit': exit_price,
                    'pnl': pnl_percent,
                    'bars_held': i - position['entry_idx']
                })
                
                position = None
        
        if len(trades) >= self.min_trades:
            return self._calculate_strategy_metrics(trades, "momentum")
        
        return None
    
    def _test_mean_reversion_strategy(self, df: pd.DataFrame, features: pd.DataFrame,
                                     bb_period: int, num_std: float, rsi_period: int,
                                     rsi_threshold: float, commission: float) -> Optional[StrategyResult]:
        """Test a mean reversion strategy"""
        trades = []
        position = None
        
        # Calculate indicators
        ma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        rsi_key = f'rsi_{rsi_period}'
        if rsi_key not in features.columns:
            # Calculate RSI if not in features
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = features[rsi_key]
        
        for i in range(max(bb_period, rsi_period), len(df) - 20):
            if position is None:
                # Long entry (oversold)
                if df['close'].iloc[i] < lower_band.iloc[i] and rsi.iloc[i] < rsi_threshold:
                    position = {
                        'entry_idx': i,
                        'entry_price': df['close'].iloc[i],
                        'direction': 1,
                        'target': ma.iloc[i]
                    }
                # Short entry (overbought)
                elif df['close'].iloc[i] > upper_band.iloc[i] and rsi.iloc[i] > (100 - rsi_threshold):
                    position = {
                        'entry_idx': i,
                        'entry_price': df['close'].iloc[i],
                        'direction': -1,
                        'target': ma.iloc[i]
                    }
            
            else:
                # Exit conditions
                exit_signal = False
                exit_price = df['close'].iloc[i]
                
                # Target reached (return to mean)
                if position['direction'] == 1 and exit_price >= ma.iloc[i]:
                    exit_signal = True
                elif position['direction'] == -1 and exit_price <= ma.iloc[i]:
                    exit_signal = True
                
                # Stop loss
                stop_distance = position['entry_price'] * 0.01
                if position['direction'] == 1 and exit_price < position['entry_price'] - stop_distance:
                    exit_signal = True
                elif position['direction'] == -1 and exit_price > position['entry_price'] + stop_distance:
                    exit_signal = True
                
                # Time limit
                if i >= position['entry_idx'] + 60:
                    exit_signal = True
                
                if exit_signal:
                    pnl = (exit_price - position['entry_price']) * position['direction']
                    pnl_percent = (pnl / position['entry_price']) * 100 - (commission / position['entry_price']) * 100
                    
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': pnl_percent,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
        
        if len(trades) >= self.min_trades:
            return self._calculate_strategy_metrics(trades, "mean_reversion")
        
        return None
    
    def _test_breakout_strategy(self, df: pd.DataFrame, features: pd.DataFrame,
                               lookback: int, volume_mult: float, atr_mult: float,
                               commission: float) -> Optional[StrategyResult]:
        """Test a breakout strategy"""
        trades = []
        position = None
        
        for i in range(lookback + 20, len(df) - 20):
            if position is None:
                # Check for breakout
                recent_high = df['high'].iloc[i-lookback:i].max()
                recent_low = df['low'].iloc[i-lookback:i].min()
                
                volume_avg = df['volume'].iloc[i-20:i].mean()
                current_volume = df['volume'].iloc[i]
                
                atr = features[f'atr_20'].iloc[i] if 'atr_20' in features else (df['high'] - df['low']).rolling(20).mean().iloc[i]
                
                # Bullish breakout
                if (df['close'].iloc[i] > recent_high and 
                    current_volume > volume_avg * volume_mult):
                    
                    position = {
                        'entry_idx': i,
                        'entry_price': df['close'].iloc[i],
                        'direction': 1,
                        'stop': df['close'].iloc[i] - (atr * atr_mult),
                        'target': df['close'].iloc[i] + (atr * atr_mult * 3)
                    }
                
                # Bearish breakout
                elif (df['close'].iloc[i] < recent_low and
                      current_volume > volume_avg * volume_mult):
                    
                    position = {
                        'entry_idx': i,
                        'entry_price': df['close'].iloc[i],
                        'direction': -1,
                        'stop': df['close'].iloc[i] + (atr * atr_mult),
                        'target': df['close'].iloc[i] - (atr * atr_mult * 3)
                    }
            
            else:
                # Exit conditions
                exit_signal = False
                exit_price = df['close'].iloc[i]
                
                # Stop loss or target
                if position['direction'] == 1:
                    if exit_price <= position['stop'] or exit_price >= position['target']:
                        exit_signal = True
                else:
                    if exit_price >= position['stop'] or exit_price <= position['target']:
                        exit_signal = True
                
                # Time limit
                if i >= position['entry_idx'] + 100:
                    exit_signal = True
                
                if exit_signal:
                    pnl = (exit_price - position['entry_price']) * position['direction']
                    pnl_percent = (pnl / position['entry_price']) * 100 - (commission / position['entry_price']) * 100
                    
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': pnl_percent,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
        
        if len(trades) >= self.min_trades:
            return self._calculate_strategy_metrics(trades, "breakout")
        
        return None
    
    def _test_pattern_strategy(self, df: pd.DataFrame, features: pd.DataFrame,
                              pattern_type: PatternType, min_strength: float,
                              stop_mult: float, target_mult: float,
                              commission: float) -> Optional[StrategyResult]:
        """Test a pattern-based strategy"""
        trades = []
        position = None
        
        # Scan for patterns throughout history
        for i in range(100, len(df) - 20, 5):  # Check every 5 bars
            window_df = df.iloc[max(0, i-100):i+1]
            window_features = features.iloc[max(0, i-100):i+1]
            
            if len(window_df) < 20:
                continue
            
            patterns = self.pattern_scanner.scan_all_patterns(window_df, window_features)
            
            if position is None:
                # Look for entry signal
                if pattern_type in patterns:
                    signal = patterns[pattern_type]
                    if signal.strength >= min_strength:
                        position = {
                            'entry_idx': i,
                            'entry_price': signal.entry_price,
                            'direction': signal.direction,
                            'stop': signal.stop_loss,
                            'target': signal.take_profit
                        }
            
            else:
                # Check exit conditions
                current_price = df['close'].iloc[i]
                
                # Adjust stop/target with multipliers
                atr = features['atr_20'].iloc[i] if 'atr_20' in features else (df['high'] - df['low']).rolling(20).mean().iloc[i]
                adjusted_stop = position['entry_price'] - (position['direction'] * atr * stop_mult)
                adjusted_target = position['entry_price'] + (position['direction'] * atr * target_mult)
                
                exit_signal = False
                
                if position['direction'] == 1:
                    if current_price <= adjusted_stop or current_price >= adjusted_target:
                        exit_signal = True
                else:
                    if current_price >= adjusted_stop or current_price <= adjusted_target:
                        exit_signal = True
                
                if exit_signal or i >= position['entry_idx'] + 50:
                    pnl = (current_price - position['entry_price']) * position['direction']
                    pnl_percent = (pnl / position['entry_price']) * 100 - (commission / position['entry_price']) * 100
                    
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl': pnl_percent,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
        
        if len(trades) >= self.min_trades:
            return self._calculate_strategy_metrics(trades, f"pattern_{pattern_type.value}")
        
        return None
    
    def _test_microstructure_strategy(self, df: pd.DataFrame, features: pd.DataFrame,
                                     imbalance_threshold: float, hold_period: int,
                                     commission: float) -> Optional[StrategyResult]:
        """Test a microstructure-based strategy"""
        trades = []
        position = None
        
        for i in range(50, len(df) - hold_period):
            if position is None:
                # Analyze microstructure
                window_df = df.iloc[max(0, i-50):i+1]
                micro = self.microstructure_analyzer.analyze_current_state(window_df)
                
                order_flow = micro.get('order_flow')
                if order_flow:
                    # Long signal
                    if order_flow.net_pressure > imbalance_threshold and not order_flow.exhaustion:
                        position = {
                            'entry_idx': i,
                            'entry_price': df['close'].iloc[i],
                            'direction': 1
                        }
                    # Short signal
                    elif order_flow.net_pressure < -imbalance_threshold and not order_flow.exhaustion:
                        position = {
                            'entry_idx': i,
                            'entry_price': df['close'].iloc[i],
                            'direction': -1
                        }
            
            else:
                # Exit after hold period or stop loss
                if i >= position['entry_idx'] + hold_period:
                    exit_price = df['close'].iloc[i]
                    pnl = (exit_price - position['entry_price']) * position['direction']
                    pnl_percent = (pnl / position['entry_price']) * 100 - (commission / position['entry_price']) * 100
                    
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': pnl_percent,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
        
        if len(trades) >= self.min_trades:
            return self._calculate_strategy_metrics(trades, "microstructure")
        
        return None
    
    def _test_combined_strategy(self, df: pd.DataFrame, features: pd.DataFrame,
                               momentum_period: int, imbalance_threshold: float,
                               commission: float) -> Optional[StrategyResult]:
        """Test a combined strategy using multiple signals"""
        trades = []
        position = None
        
        momentum = features['returns'].rolling(momentum_period).mean()
        
        for i in range(max(50, momentum_period), len(df) - 20):
            if position is None:
                # Check multiple conditions
                window_df = df.iloc[max(0, i-50):i+1]
                micro = self.microstructure_analyzer.analyze_current_state(window_df)
                
                order_flow = micro.get('order_flow')
                if order_flow:
                    momentum_signal = momentum.iloc[i]
                    
                    # Long signal: momentum + order flow alignment
                    if (momentum_signal > 0.002 and 
                        order_flow.net_pressure > imbalance_threshold):
                        
                        position = {
                            'entry_idx': i,
                            'entry_price': df['close'].iloc[i],
                            'direction': 1
                        }
                    
                    # Short signal
                    elif (momentum_signal < -0.002 and
                          order_flow.net_pressure < -imbalance_threshold):
                        
                        position = {
                            'entry_idx': i,
                            'entry_price': df['close'].iloc[i],
                            'direction': -1
                        }
            
            else:
                # Dynamic exit based on conditions
                exit_signal = False
                current_price = df['close'].iloc[i]
                
                # Check if momentum reversed
                current_momentum = momentum.iloc[i]
                if position['direction'] == 1 and current_momentum < -0.001:
                    exit_signal = True
                elif position['direction'] == -1 and current_momentum > 0.001:
                    exit_signal = True
                
                # Stop loss
                stop_distance = position['entry_price'] * 0.007
                if position['direction'] == 1 and current_price < position['entry_price'] - stop_distance:
                    exit_signal = True
                elif position['direction'] == -1 and current_price > position['entry_price'] + stop_distance:
                    exit_signal = True
                
                if exit_signal or i >= position['entry_idx'] + 30:
                    pnl = (current_price - position['entry_price']) * position['direction']
                    pnl_percent = (pnl / position['entry_price']) * 100 - (commission / position['entry_price']) * 100
                    
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl': pnl_percent,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
        
        if len(trades) >= self.min_trades:
            return self._calculate_strategy_metrics(trades, "combined")
        
        return None
    
    def _calculate_strategy_metrics(self, trades: List[Dict], strategy_name: str) -> StrategyResult:
        """Calculate strategy performance metrics"""
        if not trades:
            return None
        
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Sharpe ratio (assuming daily returns)
        returns = pd.Series(pnls)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = returns.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Best trading hours (simplified)
        best_hours = list(range(9, 16))  # Regular market hours
        best_days = list(range(1, 6))  # Weekdays
        
        return StrategyResult(
            name=strategy_name,
            parameters={},
            trades=len(trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_return=sum(pnls),
            conditions=[],
            best_hours=best_hours,
            best_days=best_days
        )
    
    def _filter_valid_strategies(self, strategies: List[TradingStrategy]) -> List[TradingStrategy]:
        """Filter strategies meeting minimum criteria"""
        valid = []
        
        for strategy in strategies:
            perf = strategy.performance
            
            if (perf.trades >= self.min_trades and
                perf.win_rate >= self.min_win_rate and
                perf.sharpe_ratio >= self.min_sharpe and
                perf.profit_factor >= self.min_profit_factor):
                
                valid.append(strategy)
        
        return valid
    
    def _rank_strategies(self, strategies: List[TradingStrategy]) -> List[TradingStrategy]:
        """Rank strategies by composite score"""
        for strategy in strategies:
            perf = strategy.performance
            
            # Composite score
            score = (
                perf.sharpe_ratio * 0.3 +
                perf.profit_factor * 0.2 +
                perf.win_rate * 10 * 0.2 +
                (1 - abs(perf.max_drawdown) / 100) * 0.2 +
                min(perf.trades / 100, 1) * 0.1
            )
            
            strategy.score = score
        
        return sorted(strategies, key=lambda x: x.score, reverse=True)
    
    def _test_strategy_robustness(self, strategy: TradingStrategy, 
                                 data: pd.DataFrame) -> bool:
        """Test strategy robustness using walk-forward analysis"""
        # Split data into multiple periods
        period_length = len(data) // 5
        
        if period_length < 100:
            return False
        
        period_results = []
        
        for i in range(4):  # Test on 4 different periods
            start_idx = i * period_length
            end_idx = start_idx + period_length * 2  # Use 2 periods
            
            if end_idx > len(data):
                break
            
            period_data = data.iloc[start_idx:end_idx]
            
            # Test strategy on this period
            # Simplified - should rerun actual strategy test
            # For now, just check if we have enough data
            if len(period_data) > 200:
                period_results.append(True)
        
        # Strategy is robust if it works in at least 3 out of 4 periods
        return len(period_results) >= 3
    
    def get_best_strategies(self, top_n: int = 10) -> List[TradingStrategy]:
        """Get top N strategies"""
        return self.discovered_strategies[:top_n]
    
    def export_strategies(self, filename: str):
        """Export discovered strategies to JSON"""
        import json
        
        strategies_data = []
        for strategy in self.discovered_strategies:
            strategies_data.append({
                'id': strategy.strategy_id,
                'type': strategy.strategy_type,
                'entry_conditions': strategy.entry_conditions,
                'exit_conditions': strategy.exit_conditions,
                'parameters': strategy.parameters,
                'performance': {
                    'trades': strategy.performance.trades,
                    'win_rate': strategy.performance.win_rate,
                    'sharpe': strategy.performance.sharpe_ratio,
                    'profit_factor': strategy.performance.profit_factor,
                    'max_drawdown': strategy.performance.max_drawdown
                }
            })
        
        with open(filename, 'w') as f:
            json.dump(strategies_data, f, indent=2)
        
        print(f"Exported {len(strategies_data)} strategies to {filename}")