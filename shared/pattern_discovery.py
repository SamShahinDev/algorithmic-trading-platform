"""
Pattern Discovery Framework
Discovers and validates trading patterns in market data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Pattern:
    """Represents a discovered trading pattern"""
    name: str
    market: str
    entry_conditions: Dict
    exit_conditions: Dict
    time_filter: Optional[Dict] = None
    statistics: Optional[Dict] = None
    
class PatternDiscovery:
    """Base class for discovering patterns in market data"""
    
    def __init__(self, data: pd.DataFrame, symbol: str):
        """
        Initialize pattern discovery
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol (ES, CL, etc.)
        """
        self.data = data.copy()
        self.symbol = symbol
        self.patterns = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Filter for main contracts only (not spreads)
        if 'symbol' in self.data.columns:
            # Keep only main contracts (e.g., ESH4, CLJ4, not spreads like ESH4-ESM4)
            self.data = self.data[~self.data['symbol'].str.contains('-', na=False)]
            
            # Focus on front month contract for consistency
            # This is simplified - in production you'd handle rollovers properly
            if len(self.data) > 0:
                symbol_counts = self.data['symbol'].value_counts()
                if len(symbol_counts) > 0:
                    self.primary_contract = symbol_counts.index[0]
                    self.logger.info(f"Primary contract: {self.primary_contract}")
                    
        # Ensure data has necessary columns
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data with technical indicators"""
        # Ensure we have a datetime index
        if 'ts_event' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['ts_event'])
            self.data = self.data.sort_values('datetime')
            self.data.set_index('datetime', inplace=True)
            
        # Remove any duplicates
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        
        # Add basic features
        self.data['returns'] = self.data['close'].pct_change()
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Volume features
        self.data['volume_ma_20'] = self.data['volume'].rolling(20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma_20']
        
        # Price features
        self.data['range'] = self.data['high'] - self.data['low']
        self.data['range_pct'] = self.data['range'] / self.data['close']
        
        # Volatility
        self.data['volatility_20'] = self.data['returns'].rolling(20).std()
        
        # Time features
        self.data['hour'] = self.data.index.hour
        self.data['minute'] = self.data.index.minute
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['time_of_day'] = self.data['hour'] * 60 + self.data['minute']
        
        # Moving averages
        self.data['sma_10'] = self.data['close'].rolling(10).mean()
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['sma_50'] = self.data['close'].rolling(50).mean()
        
        # RSI
        self.data['rsi'] = self._calculate_rsi(self.data['close'], 14)
        
        # Bollinger Bands
        self.data['bb_upper'], self.data['bb_middle'], self.data['bb_lower'] = self._calculate_bollinger_bands(
            self.data['close'], 20, 2
        )
        
        self.logger.info(f"Prepared data with shape: {self.data.shape}")
        
    def discover_momentum_patterns(self, 
                                 lookback_periods: List[int] = [5, 10, 15, 30],
                                 min_samples: int = 100) -> List[Pattern]:
        """Discover momentum-based patterns"""
        patterns = []
        
        for period in lookback_periods:
            self.logger.info(f"Analyzing momentum period: {period}")
            
            # Calculate momentum
            self.data[f'momentum_{period}'] = self.data['close'] - self.data['close'].shift(period)
            self.data[f'momentum_pct_{period}'] = self.data[f'momentum_{period}'] / self.data['close'].shift(period)
            
            # Find strong momentum moves
            threshold = self.data[f'momentum_pct_{period}'].std() * 1.5
            
            # Bullish momentum
            bullish_signal = self.data[f'momentum_pct_{period}'] > threshold
            patterns.extend(self._analyze_momentum_pattern(
                bullish_signal, 'bullish', period, min_samples
            ))
            
            # Bearish momentum  
            bearish_signal = self.data[f'momentum_pct_{period}'] < -threshold
            patterns.extend(self._analyze_momentum_pattern(
                bearish_signal, 'bearish', period, min_samples
            ))
            
        return patterns
        
    def _analyze_momentum_pattern(self, 
                                signal: pd.Series, 
                                direction: str,
                                period: int,
                                min_samples: int) -> List[Pattern]:
        """Analyze a specific momentum pattern"""
        patterns = []
        
        # Get indices where signal is True
        signal_indices = self.data.index[signal]
        
        if len(signal_indices) < min_samples:
            return patterns
            
        self.logger.info(f"Found {len(signal_indices)} {direction} momentum signals for period {period}")
        
        # Analyze performance after signal
        results = []
        for idx in signal_indices[:-60]:  # Leave last hour for forward looking
            try:
                # Get next 60 minutes of data
                future_data = self.data.loc[idx:].iloc[:61]  # Include current bar
                
                if len(future_data) < 60:
                    continue
                    
                entry_price = future_data.iloc[0]['close']
                
                # Calculate various exit points
                if direction == 'bullish':
                    max_favorable = future_data['high'][1:].max()
                    max_adverse = future_data['low'][1:].min()
                else:
                    max_favorable = future_data['low'][1:].min()
                    max_adverse = future_data['high'][1:].max()
                
                # Calculate excursions
                mfe = abs(max_favorable - entry_price) / entry_price if entry_price != 0 else 0
                mae = abs(max_adverse - entry_price) / entry_price if entry_price != 0 else 0
                
                # Various holding periods
                for hold_period in [5, 10, 15, 30, 60]:
                    if len(future_data) > hold_period:
                        exit_price = future_data.iloc[hold_period]['close']
                        
                        if direction == 'bullish':
                            pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
                        else:
                            pnl_pct = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
                            
                        results.append({
                            'timestamp': idx,
                            'hour': idx.hour,
                            'minute': idx.minute,
                            'day_of_week': idx.dayofweek,
                            'hold_period': hold_period,
                            'pnl_pct': pnl_pct,
                            'mfe': mfe,
                            'mae': mae,
                            'volume_ratio': future_data.iloc[0].get('volume_ratio', 1)
                        })
                        
            except Exception as e:
                self.logger.debug(f"Error analyzing pattern at {idx}: {str(e)}")
                continue
                
        if not results:
            return patterns
            
        # Analyze results
        results_df = pd.DataFrame(results)
        
        if len(results_df) < min_samples:
            return patterns
            
        # Group by hour and holding period for time-based patterns
        for hour in range(24):
            hour_data = results_df[results_df['hour'] == hour]
            
            if len(hour_data) < 20:  # Min samples per hour
                continue
                
            for hold_period in [5, 10, 15, 30]:
                period_data = hour_data[hour_data['hold_period'] == hold_period]
                
                if len(period_data) < 20:
                    continue
                    
                # Calculate statistics
                win_rate = (period_data['pnl_pct'] > 0).mean()
                
                if win_rate < 0.55:  # Minimum win rate
                    continue
                    
                wins = period_data[period_data['pnl_pct'] > 0]
                losses = period_data[period_data['pnl_pct'] <= 0]
                
                avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
                avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
                
                # Calculate edge
                edge = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
                
                if edge > 0.0002:  # 0.02% edge minimum
                    pattern = Pattern(
                        name=f"{self.symbol}_{direction}_momentum_{period}min_hour{hour:02d}_hold{hold_period}",
                        market=self.symbol,
                        entry_conditions={
                            'momentum_period': period,
                            'direction': direction,
                            'threshold_multiplier': 1.5,
                            'hour': hour
                        },
                        exit_conditions={
                            'hold_period': hold_period,
                            'stop_loss_pct': float(period_data['mae'].quantile(0.75)),
                            'target_pct': float(period_data['mfe'].quantile(0.25))
                        },
                        time_filter={
                            'hour': hour,
                            'minute_range': (0, 59)
                        },
                        statistics={
                            'sample_size': len(period_data),
                            'win_rate': float(win_rate),
                            'avg_win': float(avg_win),
                            'avg_loss': float(avg_loss),
                            'edge': float(edge),
                            'sharpe': float(self._calculate_sharpe(period_data['pnl_pct'])),
                            'max_mae': float(period_data['mae'].max())
                        }
                    )
                    patterns.append(pattern)
                    self.logger.info(f"Found pattern: {pattern.name} with edge {edge:.4%}")
                    
        return patterns
        
    def discover_reversal_patterns(self, 
                                 min_samples: int = 50) -> List[Pattern]:
        """Discover reversal patterns at extremes"""
        patterns = []
        
        # Oversold bounce pattern
        oversold = (
            (self.data['rsi'] < 30) & 
            (self.data['close'] < self.data['bb_lower']) &
            (self.data['volume_ratio'] > 1.2)
        )
        
        # Overbought reversal pattern
        overbought = (
            (self.data['rsi'] > 70) & 
            (self.data['close'] > self.data['bb_upper']) &
            (self.data['volume_ratio'] > 1.2)
        )
        
        # Analyze patterns
        patterns.extend(self._analyze_reversal_pattern(oversold, 'oversold_bounce', min_samples))
        patterns.extend(self._analyze_reversal_pattern(overbought, 'overbought_reversal', min_samples))
        
        return patterns
        
    def _analyze_reversal_pattern(self, 
                                signal: pd.Series, 
                                pattern_name: str,
                                min_samples: int) -> List[Pattern]:
        """Analyze reversal pattern performance"""
        patterns = []
        
        signal_indices = self.data.index[signal]
        
        if len(signal_indices) < min_samples:
            return patterns
            
        self.logger.info(f"Found {len(signal_indices)} {pattern_name} signals")
        
        results = []
        
        for idx in signal_indices[:-60]:
            try:
                future_data = self.data.loc[idx:].iloc[:61]
                
                if len(future_data) < 30:
                    continue
                    
                entry_price = future_data.iloc[0]['close']
                
                # For reversals, we expect price to reverse
                if 'oversold' in pattern_name:
                    # Expect bounce up
                    for hold_period in [5, 10, 15, 30]:
                        if len(future_data) > hold_period:
                            exit_price = future_data.iloc[hold_period]['close']
                            pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
                            
                            results.append({
                                'timestamp': idx,
                                'hour': idx.hour,
                                'hold_period': hold_period,
                                'pnl_pct': pnl_pct,
                                'rsi': future_data.iloc[0]['rsi'],
                                'bb_position': (entry_price - future_data.iloc[0]['bb_lower']) / 
                                             (future_data.iloc[0]['bb_upper'] - future_data.iloc[0]['bb_lower'])
                            })
                else:
                    # Expect reversal down
                    for hold_period in [5, 10, 15, 30]:
                        if len(future_data) > hold_period:
                            exit_price = future_data.iloc[hold_period]['close']
                            pnl_pct = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
                            
                            results.append({
                                'timestamp': idx,
                                'hour': idx.hour,
                                'hold_period': hold_period,
                                'pnl_pct': pnl_pct,
                                'rsi': future_data.iloc[0]['rsi'],
                                'bb_position': (entry_price - future_data.iloc[0]['bb_lower']) / 
                                             (future_data.iloc[0]['bb_upper'] - future_data.iloc[0]['bb_lower'])
                            })
                            
            except Exception as e:
                self.logger.debug(f"Error in reversal analysis: {e}")
                continue
                
        if not results:
            return patterns
            
        results_df = pd.DataFrame(results)
        
        # Find optimal holding period
        for hold_period in [5, 10, 15, 30]:
            period_data = results_df[results_df['hold_period'] == hold_period]
            
            if len(period_data) < min_samples:
                continue
                
            win_rate = (period_data['pnl_pct'] > 0).mean()
            
            if win_rate > 0.55:
                avg_win = period_data[period_data['pnl_pct'] > 0]['pnl_pct'].mean()
                avg_loss = abs(period_data[period_data['pnl_pct'] <= 0]['pnl_pct'].mean())
                
                pattern = Pattern(
                    name=f"{self.symbol}_{pattern_name}_hold{hold_period}",
                    market=self.symbol,
                    entry_conditions={
                        'pattern_type': pattern_name,
                        'rsi_threshold': 30 if 'oversold' in pattern_name else 70,
                        'bb_position': 'below_lower' if 'oversold' in pattern_name else 'above_upper',
                        'volume_ratio_min': 1.2
                    },
                    exit_conditions={
                        'hold_period': hold_period,
                        'stop_loss_pct': avg_loss * 1.5 if avg_loss > 0 else 0.01,
                        'target_pct': avg_win * 0.8 if avg_win > 0 else 0.01
                    },
                    statistics={
                        'sample_size': len(period_data),
                        'win_rate': float(win_rate),
                        'avg_win': float(avg_win),
                        'avg_loss': float(avg_loss),
                        'sharpe': float(self._calculate_sharpe(period_data['pnl_pct']))
                    }
                )
                patterns.append(pattern)
                self.logger.info(f"Found reversal pattern: {pattern.name}")
                
        return patterns
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, 
                                  std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
        
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
            
        excess_returns = returns - risk_free_rate / 252 / 1440  # Per minute risk-free rate
        
        if excess_returns.std() == 0:
            return 0
            
        # Annualized Sharpe (assuming 1440 minutes per day, 252 trading days)
        return np.sqrt(252 * 1440) * excess_returns.mean() / excess_returns.std()
        
    def discover_all_patterns(self, min_samples: int = 50) -> List[Pattern]:
        """Discover all pattern types"""
        all_patterns = []
        
        self.logger.info("Starting pattern discovery...")
        
        # Momentum patterns
        self.logger.info("Discovering momentum patterns...")
        momentum_patterns = self.discover_momentum_patterns(
            lookback_periods=[5, 10, 15],
            min_samples=min_samples
        )
        all_patterns.extend(momentum_patterns)
        self.logger.info(f"Found {len(momentum_patterns)} momentum patterns")
        
        # Reversal patterns
        self.logger.info("Discovering reversal patterns...")
        reversal_patterns = self.discover_reversal_patterns(min_samples=min_samples)
        all_patterns.extend(reversal_patterns)
        self.logger.info(f"Found {len(reversal_patterns)} reversal patterns")
        
        self.logger.info(f"Total patterns discovered: {len(all_patterns)}")
        
        return all_patterns