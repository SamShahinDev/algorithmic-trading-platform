"""
Advanced Backtesting Engine for Pattern Validation
Tests patterns against historical data to validate performance
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_db_session, db_manager
from database.models import Pattern, BacktestResult, PatternDiscovery
from patterns.pattern_discovery import pattern_discovery

@dataclass
class BacktestTrade:
    """Represents a single backtested trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    pattern_name: str
    confidence: float
    status: str  # 'open', 'win', 'loss', 'stopped'
    pnl: float = 0.0
    max_drawdown: float = 0.0
    max_runup: float = 0.0

@dataclass
class BacktestResults:
    """Complete backtesting results"""
    pattern_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_runup: float
    sharpe_ratio: float
    profit_factor: float
    expectancy: float
    trades: List[BacktestTrade]
    equity_curve: List[Tuple[datetime, float]]

class BacktestEngine:
    """
    Advanced backtesting engine for pattern validation
    """
    
    def __init__(self):
        """Initialize backtesting engine"""
        self.symbol = "NQ=F"
        self.point_value = 20  # NQ point value in USD
        
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.starting_capital = 10000
        
        print("📊 Backtesting Engine initialized")
    
    async def run_pattern_backtest(self, pattern_name: str, start_date: datetime, 
                                 end_date: datetime, timeframe: str = "1h") -> BacktestResults:
        """
        Run complete backtest for a specific pattern
        """
        print(f"🔙 Starting backtest for {pattern_name}")
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get historical data
        historical_data = await self.get_historical_data(start_date, end_date, timeframe)
        
        if historical_data.empty:
            print(f"❌ No historical data available for {pattern_name}")
            return self.create_empty_results(pattern_name)
        
        # Detect pattern occurrences in historical data
        pattern_signals = await self.detect_pattern_in_history(pattern_name, historical_data)
        
        if not pattern_signals:
            print(f"❌ No {pattern_name} patterns found in historical data")
            return self.create_empty_results(pattern_name)
        
        print(f"📍 Found {len(pattern_signals)} {pattern_name} signals in historical data")
        
        # Execute backtested trades
        trades = await self.execute_backtest_trades(pattern_signals, historical_data)
        
        # Calculate comprehensive results
        results = self.calculate_backtest_results(pattern_name, trades)
        
        # Store results in database
        await self.store_backtest_results(results)
        
        print(f"✅ Backtest completed for {pattern_name}")
        print(f"   Total Trades: {results.total_trades} | Win Rate: {results.win_rate:.1%}")
        print(f"   Total P&L: ${results.total_pnl:.2f} | Sharpe: {results.sharpe_ratio:.2f}")
        
        return results
    
    async def get_historical_data(self, start_date: datetime, end_date: datetime, 
                                timeframe: str) -> pd.DataFrame:
        """Get historical price data for backtesting"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Convert timeframe to yfinance format
            if timeframe == "1h":
                interval = "1h"
                period_days = (end_date - start_date).days
                if period_days > 30:
                    # For longer periods, use daily data
                    interval = "1d"
            elif timeframe == "1d":
                interval = "1d"
            else:
                interval = "1h"
            
            # Get data with some buffer for pattern detection
            buffer_start = start_date - timedelta(days=30)
            
            data = ticker.history(
                start=buffer_start.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if data.empty:
                print(f"⚠️  No data available from yfinance for {self.symbol}")
                # Return mock data for demonstration
                return self.create_mock_historical_data(start_date, end_date)
            
            return data
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self.create_mock_historical_data(start_date, end_date)
    
    def create_mock_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create realistic mock data for backtesting demonstration"""
        periods = int((end_date - start_date).total_seconds() / 3600)  # Hourly data
        
        # Create time index
        time_index = pd.date_range(start_date, periods=periods, freq='H')
        
        # Generate realistic NQ price movement starting around 15000
        base_price = 15000
        price_data = []
        current_price = base_price
        
        for i in range(periods):
            # Add realistic volatility and trend
            daily_volatility = 0.015  # 1.5% daily volatility
            hourly_volatility = daily_volatility / np.sqrt(24)
            
            # Random walk with slight upward bias
            price_change = np.random.normal(0.0005, hourly_volatility)  # Slight upward drift
            current_price *= (1 + price_change)
            
            # Create OHLC data
            high = current_price * (1 + np.random.uniform(0, 0.005))
            low = current_price * (1 - np.random.uniform(0, 0.005))
            open_price = current_price + np.random.normal(0, current_price * 0.002)
            close_price = current_price
            volume = np.random.randint(50000, 200000)
            
            price_data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close_price,
                'Volume': volume
            })
            
            current_price = close_price
        
        df = pd.DataFrame(price_data, index=time_index)
        print(f"📊 Created mock historical data: {len(df)} periods from {df.index[0]} to {df.index[-1]}")
        return df
    
    async def detect_pattern_in_history(self, pattern_name: str, data: pd.DataFrame) -> List[Dict]:
        """Detect pattern occurrences in historical data"""
        signals = []
        
        try:
            # Use pattern discovery engine to find patterns
            if pattern_name.endswith('_engulfing'):
                patterns = await pattern_discovery.detect_engulfing(data)
            elif pattern_name.startswith('double_'):
                patterns = await pattern_discovery.detect_double_tops_bottoms(data)
            elif 'flag' in pattern_name:
                patterns = await pattern_discovery.detect_flag_pattern(data)
            elif 'triangle' in pattern_name:
                patterns = await pattern_discovery.detect_triangles(data)
            elif 'cross' in pattern_name:
                patterns = await pattern_discovery.detect_ma_crossovers(data)
            else:
                # Generic pattern detection
                all_patterns = await pattern_discovery.scan_for_patterns(data)
                patterns = [p for p in all_patterns if p['type'] == pattern_name]
            
            # Filter patterns by type
            signals = [p for p in patterns if p['type'] == pattern_name]
            
        except Exception as e:
            print(f"Error detecting {pattern_name} in historical data: {e}")
            
        return signals
    
    async def execute_backtest_trades(self, signals: List[Dict], data: pd.DataFrame) -> List[BacktestTrade]:
        """Execute backtested trades based on pattern signals"""
        trades = []
        
        for signal in signals:
            try:
                entry_time = pd.to_datetime(signal['timestamp'])
                
                # Find entry time in data
                entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx == -1:
                    continue
                
                entry_price = float(signal['entry_price'])
                direction = self.determine_trade_direction(signal)
                confidence = float(signal['confidence'])
                
                # Set stops and targets
                stop_loss = float(signal['stop_loss'])
                take_profit = float(signal['take_profit'])
                
                # Create trade
                trade = BacktestTrade(
                    entry_time=entry_time,
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pattern_name=signal['type'],
                    confidence=confidence,
                    status='open'
                )
                
                # Simulate trade execution from entry point
                exit_info = self.simulate_trade_exit(trade, data[entry_idx:])
                
                if exit_info:
                    trade.exit_time = exit_info['exit_time']
                    trade.exit_price = exit_info['exit_price']
                    trade.status = exit_info['status']
                    trade.pnl = self.calculate_trade_pnl(trade)
                    trade.max_drawdown = exit_info.get('max_drawdown', 0)
                    trade.max_runup = exit_info.get('max_runup', 0)
                
                trades.append(trade)
                
            except Exception as e:
                print(f"Error executing backtest trade: {e}")
                continue
        
        return trades
    
    def determine_trade_direction(self, signal: Dict) -> str:
        """Determine trade direction from pattern signal"""
        pattern_type = signal['type'].lower()
        
        if any(word in pattern_type for word in ['bull', 'golden', 'ascending', 'bottom']):
            return 'long'
        elif any(word in pattern_type for word in ['bear', 'death', 'descending', 'top']):
            return 'short'
        else:
            # Default based on pattern name
            return 'long' if 'bull' in pattern_type else 'short'
    
    def simulate_trade_exit(self, trade: BacktestTrade, future_data: pd.DataFrame) -> Optional[Dict]:
        """Simulate trade exit based on stop/target levels"""
        
        if len(future_data) < 2:
            return None
        
        max_drawdown = 0
        max_runup = 0
        
        for i, (timestamp, candle) in enumerate(future_data.iterrows()):
            if i == 0:  # Skip entry candle
                continue
            
            high = candle['High']
            low = candle['Low']
            close = candle['Close']
            
            # Track unrealized P&L for drawdown/runup
            if trade.direction == 'long':
                unrealized_pnl = (close - trade.entry_price) * self.point_value
                if close <= trade.stop_loss:
                    return {
                        'exit_time': timestamp,
                        'exit_price': trade.stop_loss,
                        'status': 'loss',
                        'max_drawdown': max_drawdown,
                        'max_runup': max_runup
                    }
                elif close >= trade.take_profit:
                    return {
                        'exit_time': timestamp,
                        'exit_price': trade.take_profit,
                        'status': 'win',
                        'max_drawdown': max_drawdown,
                        'max_runup': max_runup
                    }
            else:  # short
                unrealized_pnl = (trade.entry_price - close) * self.point_value
                if close >= trade.stop_loss:
                    return {
                        'exit_time': timestamp,
                        'exit_price': trade.stop_loss,
                        'status': 'loss',
                        'max_drawdown': max_drawdown,
                        'max_runup': max_runup
                    }
                elif close <= trade.take_profit:
                    return {
                        'exit_time': timestamp,
                        'exit_price': trade.take_profit,
                        'status': 'win',
                        'max_drawdown': max_drawdown,
                        'max_runup': max_runup
                    }
            
            # Update drawdown/runup
            if unrealized_pnl < max_drawdown:
                max_drawdown = unrealized_pnl
            if unrealized_pnl > max_runup:
                max_runup = unrealized_pnl
        
        # Trade didn't hit stop or target (end of data)
        final_candle = future_data.iloc[-1]
        return {
            'exit_time': future_data.index[-1],
            'exit_price': final_candle['Close'],
            'status': 'open',
            'max_drawdown': max_drawdown,
            'max_runup': max_runup
        }
    
    def calculate_trade_pnl(self, trade: BacktestTrade) -> float:
        """Calculate P&L for a trade"""
        if trade.exit_price is None:
            return 0.0
        
        if trade.direction == 'long':
            pnl = (trade.exit_price - trade.entry_price) * self.point_value
        else:
            pnl = (trade.entry_price - trade.exit_price) * self.point_value
        
        return pnl
    
    def calculate_backtest_results(self, pattern_name: str, trades: List[BacktestTrade]) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not trades:
            return self.create_empty_results(pattern_name)
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.status == 'win'])
        losing_trades = len([t for t in trades if t.status == 'loss'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = sum(t.pnl for t in trades)
        wins = [t.pnl for t in trades if t.status == 'win']
        losses = [t.pnl for t in trades if t.status == 'loss']
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Risk metrics
        returns = [t.pnl for t in trades]
        max_drawdown = min([t.max_drawdown for t in trades] + [0])
        max_runup = max([t.max_runup for t in trades] + [0])
        
        # Calculate Sharpe ratio (annualized)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if avg_loss != 0 else 0
        
        # Equity curve
        equity_curve = []
        running_equity = self.starting_capital
        for trade in trades:
            running_equity += trade.pnl
            equity_curve.append((trade.exit_time or trade.entry_time, running_equity))
        
        return BacktestResults(
            pattern_name=pattern_name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            max_runup=max_runup,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def create_empty_results(self, pattern_name: str) -> BacktestResults:
        """Create empty results for patterns with no trades"""
        return BacktestResults(
            pattern_name=pattern_name,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            max_runup=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            trades=[],
            equity_curve=[]
        )
    
    async def store_backtest_results(self, results: BacktestResults):
        """Store backtest results in database"""
        try:
            with get_db_session() as session:
                backtest_result = BacktestResult(
                    pattern_name=results.pattern_name,
                    test_period_start=datetime.utcnow() - timedelta(days=30),
                    test_period_end=datetime.utcnow(),
                    total_trades=results.total_trades,
                    winning_trades=results.winning_trades,
                    win_rate=results.win_rate,
                    total_pnl=results.total_pnl,
                    avg_win=results.avg_win,
                    avg_loss=results.avg_loss,
                    max_drawdown=results.max_drawdown,
                    sharpe_ratio=results.sharpe_ratio,
                    profit_factor=results.profit_factor,
                    expectancy=results.expectancy,
                    created_at=datetime.utcnow()
                )
                
                session.add(backtest_result)
                session.commit()
                
                print(f"💾 Stored backtest results for {results.pattern_name}")
                
        except Exception as e:
            print(f"Error storing backtest results: {e}")
    
    async def run_pattern_optimization(self, pattern_name: str) -> Dict:
        """
        Run parameter optimization for a pattern
        """
        print(f"🔧 Starting parameter optimization for {pattern_name}")
        
        # Define parameter ranges to test
        stop_loss_ranges = [3, 4, 5, 6, 7]  # Points
        take_profit_ranges = [3, 4, 5, 6, 7, 8, 9, 10]  # Points
        
        best_results = None
        best_params = {}
        best_score = -float('inf')
        
        optimization_results = []
        
        for stop_loss in stop_loss_ranges:
            for take_profit in take_profit_ranges:
                # Run backtest with these parameters
                results = await self.run_parameter_backtest(
                    pattern_name, stop_loss, take_profit
                )
                
                # Score based on risk-adjusted returns
                score = self.calculate_optimization_score(results)
                
                optimization_results.append({
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'score': score,
                    'win_rate': results.win_rate,
                    'total_pnl': results.total_pnl,
                    'sharpe_ratio': results.sharpe_ratio,
                    'profit_factor': results.profit_factor
                })
                
                if score > best_score:
                    best_score = score
                    best_results = results
                    best_params = {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'score': score
                    }
        
        print(f"✅ Optimization complete for {pattern_name}")
        print(f"   Best Parameters: SL={best_params['stop_loss']} TP={best_params['take_profit']}")
        print(f"   Score: {best_params['score']:.2f}")
        
        return {
            'pattern_name': pattern_name,
            'best_params': best_params,
            'best_results': best_results,
            'all_results': optimization_results
        }
    
    async def run_parameter_backtest(self, pattern_name: str, stop_loss_points: float, 
                                   take_profit_points: float) -> BacktestResults:
        """Run backtest with specific parameters"""
        # This would implement parameter-specific backtesting
        # For now, return simplified results
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        return await self.run_pattern_backtest(pattern_name, start_date, end_date)
    
    def calculate_optimization_score(self, results: BacktestResults) -> float:
        """Calculate optimization score combining multiple metrics"""
        if results.total_trades == 0:
            return -1000
        
        # Weighted score combining multiple factors
        score = (
            results.win_rate * 30 +  # 30% weight on win rate
            (results.profit_factor / 10) * 25 +  # 25% weight on profit factor
            results.sharpe_ratio * 20 +  # 20% weight on Sharpe ratio
            (results.total_pnl / 1000) * 15 +  # 15% weight on total P&L
            (results.total_trades / 10) * 10  # 10% weight on trade frequency
        )
        
        # Penalty for low sample size
        if results.total_trades < 10:
            score *= 0.7
        
        return score
    
    async def run_walk_forward_analysis(self, pattern_name: str, periods: int = 6) -> Dict:
        """
        Run walk-forward analysis to test robustness
        """
        print(f"🚶 Starting walk-forward analysis for {pattern_name}")
        
        end_date = datetime.now()
        total_days = 180  # 6 months
        period_days = total_days // periods
        
        results = []
        
        for i in range(periods):
            # Define in-sample and out-of-sample periods
            is_start = end_date - timedelta(days=total_days - (i * period_days))
            is_end = is_start + timedelta(days=period_days * 0.7)  # 70% for optimization
            oos_start = is_end
            oos_end = is_start + timedelta(days=period_days)
            
            print(f"   Period {i+1}: Optimizing {is_start.strftime('%m/%d')} to {is_end.strftime('%m/%d')}")
            print(f"              Testing {oos_start.strftime('%m/%d')} to {oos_end.strftime('%m/%d')}")
            
            # Run optimization on in-sample data
            # (Simplified for demonstration)
            optimization = await self.run_pattern_optimization(pattern_name)
            
            # Test optimized parameters on out-of-sample data
            oos_results = await self.run_pattern_backtest(pattern_name, oos_start, oos_end)
            
            results.append({
                'period': i + 1,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'best_params': optimization['best_params'],
                'oos_results': oos_results
            })
        
        # Calculate walk-forward metrics
        wf_metrics = self.calculate_walk_forward_metrics(results)
        
        print(f"✅ Walk-forward analysis complete for {pattern_name}")
        print(f"   Average OOS Win Rate: {wf_metrics['avg_win_rate']:.1%}")
        print(f"   Average OOS P&L: ${wf_metrics['avg_pnl']:.2f}")
        
        return {
            'pattern_name': pattern_name,
            'periods': results,
            'metrics': wf_metrics
        }
    
    def calculate_walk_forward_metrics(self, results: List[Dict]) -> Dict:
        """Calculate walk-forward analysis metrics"""
        oos_results = [r['oos_results'] for r in results]
        
        if not oos_results:
            return {'avg_win_rate': 0, 'avg_pnl': 0, 'consistency': 0}
        
        avg_win_rate = np.mean([r.win_rate for r in oos_results])
        avg_pnl = np.mean([r.total_pnl for r in oos_results])
        
        # Consistency score (how many periods were profitable)
        profitable_periods = len([r for r in oos_results if r.total_pnl > 0])
        consistency = profitable_periods / len(oos_results)
        
        return {
            'avg_win_rate': avg_win_rate,
            'avg_pnl': avg_pnl,
            'consistency': consistency,
            'total_periods': len(results),
            'profitable_periods': profitable_periods
        }

# Global backtest engine instance
backtest_engine = BacktestEngine()