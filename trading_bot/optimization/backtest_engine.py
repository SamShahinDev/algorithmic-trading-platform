"""
Sophisticated Backtesting Engine with Realistic Execution
Simulates real trading conditions including slippage, commissions, and market impact
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from execution.confidence_engine import AdvancedConfidenceEngine, TradeAction
from data.feature_engineering import FeatureEngineer
from analysis.pattern_scanner import PatternScanner


class ExecutionModel(Enum):
    """Execution model types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    ADAPTIVE = "adaptive"


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 50000
    position_sizing: str = "fixed"  # fixed, kelly, risk_parity
    max_position_size: int = 2
    commission: float = 2.52  # TopStep MNQ round-trip
    slippage_model: str = "realistic"  # zero, fixed, realistic, adverse
    execution_model: ExecutionModel = ExecutionModel.MARKET
    risk_limits: Dict = field(default_factory=lambda: {
        'max_daily_loss': 1500,
        'trailing_drawdown': 2000,
        'max_positions': 1
    })
    

@dataclass
class Trade:
    """Individual trade record"""
    trade_id: int
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: int  # 1 long, -1 short
    size: int
    entry_price: float
    exit_price: Optional[float]
    commission: float
    slippage: float
    pnl: float = 0
    pnl_percent: float = 0
    pattern: Optional[str] = None
    confidence: float = 0
    mae: float = 0  # Maximum adverse excursion
    mfe: float = 0  # Maximum favorable excursion
    bars_held: int = 0
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """Complete backtest results"""
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    kelly_criterion: float
    
    # Risk metrics
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    downside_deviation: float
    ulcer_index: float
    
    # Efficiency metrics
    avg_bars_held: float
    avg_mae: float
    avg_mfe: float
    efficiency_ratio: float
    
    # Pattern performance
    pattern_performance: Dict
    confidence_distribution: Dict
    hourly_performance: Dict
    daily_performance: Dict


class BacktestEngine:
    """Sophisticated backtesting engine with realistic execution simulation"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.confidence_engine = AdvancedConfidenceEngine()
        self.feature_engineer = FeatureEngineer()
        self.pattern_scanner = PatternScanner()
        
        # Tracking variables
        self.trades = []
        self.open_trades = {}
        self.equity_curve = [self.config.initial_capital]
        self.daily_pnl = 0
        self.trade_counter = 0
        
        # Slippage models
        self.slippage_functions = {
            'zero': self._zero_slippage,
            'fixed': self._fixed_slippage,
            'realistic': self._realistic_slippage,
            'adverse': self._adverse_slippage
        }
    
    def backtest_strategy(self, 
                         data: pd.DataFrame,
                         strategy_func: Optional[Callable] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            data: Historical OHLCV data
            strategy_func: Custom strategy function (optional)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Complete backtest results
        """
        print(f"Starting backtest on {len(data)} bars...")
        
        # Filter data by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Calculate features once
        features = self.feature_engineer.calculate_features(data)
        
        # Reset state
        self._reset_state()
        
        # Main backtest loop
        for i in range(100, len(data)):
            current_bar = data.iloc[i]
            current_time = data.index[i]
            
            # Check daily reset
            self._check_daily_reset(current_time)
            
            # Update open positions
            self._update_open_positions(current_bar, i)
            
            # Check risk limits
            if not self._check_risk_limits():
                continue
            
            # Generate signals
            if strategy_func:
                # Use custom strategy
                signal = strategy_func(data.iloc[:i+1], features.iloc[:i+1])
            else:
                # Use confidence engine
                signal = self._generate_signal(data.iloc[:i+1], features.iloc[:i+1])
            
            # Execute trades
            if signal:
                self._execute_signal(signal, current_bar, current_time, i)
            
            # Update equity
            self._update_equity(current_bar)
        
        # Close any remaining positions
        self._close_all_positions(data.iloc[-1], len(data)-1)
        
        # Calculate results
        results = self._calculate_results(data)
        
        print(f"Backtest complete: {results.total_trades} trades, "
              f"{results.win_rate:.1%} win rate, Sharpe: {results.sharpe_ratio:.2f}")
        
        return results
    
    def _generate_signal(self, data: pd.DataFrame, features: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal using confidence engine"""
        if len(data) < 100:
            return None
        
        # Use last 100 bars for analysis
        analysis_data = data.iloc[-100:]
        
        # Get confidence result
        confidence_result = self.confidence_engine.calculate_confidence(analysis_data)
        
        decision = confidence_result['trade_decision']
        confidence = confidence_result['confidence']
        
        # Check if we should trade
        if decision.action in [TradeAction.BUY, TradeAction.SELL]:
            if confidence >= self.confidence_engine.min_confidence:
                # Check if we already have a position
                if len(self.open_trades) >= self.config.risk_limits['max_positions']:
                    return None
                
                return {
                    'action': decision.action,
                    'confidence': confidence,
                    'size': self._calculate_position_size(confidence),
                    'entry_price': decision.entry_price,
                    'stop_loss': decision.stop_loss,
                    'take_profit': decision.take_profit,
                    'pattern': list(confidence_result['patterns'].keys())[0] if confidence_result['patterns'] else None
                }
        
        return None
    
    def _execute_signal(self, signal: Dict, current_bar: pd.Series, 
                       current_time: datetime, bar_idx: int):
        """Execute trading signal with realistic fills"""
        # Calculate entry price with slippage
        entry_price = current_bar['close']
        
        direction = 1 if signal['action'] == TradeAction.BUY else -1
        slippage = self._calculate_slippage(entry_price, signal['size'], direction, 'entry')
        
        # Adjust entry price for slippage
        entry_price += slippage * direction
        
        # Create trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            entry_time=current_time,
            exit_time=None,
            symbol='NQ.FUT',
            direction=direction,
            size=signal['size'],
            entry_price=entry_price,
            exit_price=None,
            commission=self.config.commission / 2,  # Half on entry
            slippage=abs(slippage),
            pattern=signal['pattern'].value if signal['pattern'] else None,
            confidence=signal['confidence']
        )
        
        # Add to open trades
        self.open_trades[trade.trade_id] = {
            'trade': trade,
            'bar_idx': bar_idx,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'highest': entry_price,
            'lowest': entry_price
        }
        
        # Update capital
        cost = (entry_price * trade.size * 20) + trade.commission
        self.equity_curve.append(self.equity_curve[-1] - trade.commission)
    
    def _update_open_positions(self, current_bar: pd.Series, bar_idx: int):
        """Update open positions and check exits"""
        closed_trades = []
        
        for trade_id, position_data in self.open_trades.items():
            trade = position_data['trade']
            entry_bar_idx = position_data['bar_idx']
            
            # Update MAE/MFE
            if trade.direction == 1:  # Long
                position_data['highest'] = max(position_data['highest'], current_bar['high'])
                position_data['lowest'] = min(position_data['lowest'], current_bar['low'])
                
                trade.mfe = (position_data['highest'] - trade.entry_price) / trade.entry_price * 100
                trade.mae = (trade.entry_price - position_data['lowest']) / trade.entry_price * 100
            else:  # Short
                position_data['highest'] = max(position_data['highest'], current_bar['high'])
                position_data['lowest'] = min(position_data['lowest'], current_bar['low'])
                
                trade.mfe = (trade.entry_price - position_data['lowest']) / trade.entry_price * 100
                trade.mae = (position_data['highest'] - trade.entry_price) / trade.entry_price * 100
            
            # Check exit conditions
            exit_signal = False
            exit_reason = ""
            exit_price = current_bar['close']
            
            # Stop loss
            if trade.direction == 1:
                if current_bar['low'] <= position_data['stop_loss']:
                    exit_signal = True
                    exit_reason = "stop_loss"
                    exit_price = min(current_bar['open'], position_data['stop_loss'])
            else:
                if current_bar['high'] >= position_data['stop_loss']:
                    exit_signal = True
                    exit_reason = "stop_loss"
                    exit_price = max(current_bar['open'], position_data['stop_loss'])
            
            # Take profit
            if not exit_signal:
                if trade.direction == 1:
                    if current_bar['high'] >= position_data['take_profit']:
                        exit_signal = True
                        exit_reason = "take_profit"
                        exit_price = min(current_bar['open'], position_data['take_profit'])
                else:
                    if current_bar['low'] <= position_data['take_profit']:
                        exit_signal = True
                        exit_reason = "take_profit"
                        exit_price = max(current_bar['open'], position_data['take_profit'])
            
            # Time exit (optional)
            if not exit_signal and (bar_idx - entry_bar_idx) > 100:
                exit_signal = True
                exit_reason = "time_limit"
                exit_price = current_bar['close']
            
            if exit_signal:
                # Calculate exit with slippage
                slippage = self._calculate_slippage(exit_price, trade.size, -trade.direction, 'exit')
                exit_price += slippage * (-trade.direction)
                
                # Update trade
                trade.exit_time = current_bar.name
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.bars_held = bar_idx - entry_bar_idx
                
                # Calculate PnL
                gross_pnl = (exit_price - trade.entry_price) * trade.direction * trade.size * 20
                trade.commission += self.config.commission / 2  # Other half on exit
                trade.pnl = gross_pnl - trade.commission
                trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.size * 20)) * 100
                
                # Update daily PnL
                self.daily_pnl += trade.pnl
                
                # Add to completed trades
                self.trades.append(trade)
                closed_trades.append(trade_id)
                
                # Update confidence engine with result
                if trade.pattern:
                    from analysis.pattern_scanner import PatternType
                    for pattern_type in PatternType:
                        if pattern_type.value == trade.pattern:
                            self.confidence_engine.record_trade(
                                trade.entry_price,
                                trade.exit_price,
                                trade.direction,
                                trade.confidence,
                                pattern_type
                            )
                            break
        
        # Remove closed trades
        for trade_id in closed_trades:
            del self.open_trades[trade_id]
    
    def _calculate_position_size(self, confidence: float) -> int:
        """Calculate position size based on confidence and risk management"""
        if self.config.position_sizing == "fixed":
            return 1
        
        elif self.config.position_sizing == "kelly":
            # Kelly criterion sizing
            if len(self.trades) > 30:
                wins = [t for t in self.trades if t.pnl > 0]
                losses = [t for t in self.trades if t.pnl <= 0]
                
                if wins and losses:
                    win_rate = len(wins) / len(self.trades)
                    avg_win = np.mean([t.pnl_percent for t in wins])
                    avg_loss = abs(np.mean([t.pnl_percent for t in losses]))
                    
                    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly = max(0, min(0.25, kelly))  # Cap at 25%
                    
                    size = int(kelly * 4)  # Convert to position size
                    return max(1, min(size, self.config.max_position_size))
            
            return 1
        
        elif self.config.position_sizing == "risk_parity":
            # Size based on volatility
            return 1
        
        else:
            return 1
    
    def _calculate_slippage(self, price: float, size: int, direction: int, 
                           side: str) -> float:
        """Calculate slippage based on model"""
        slippage_func = self.slippage_functions.get(
            self.config.slippage_model, 
            self._realistic_slippage
        )
        return slippage_func(price, size, direction, side)
    
    def _zero_slippage(self, price: float, size: int, direction: int, side: str) -> float:
        """No slippage"""
        return 0
    
    def _fixed_slippage(self, price: float, size: int, direction: int, side: str) -> float:
        """Fixed slippage in ticks"""
        tick_size = 0.25  # NQ tick size
        return tick_size * 1  # 1 tick slippage
    
    def _realistic_slippage(self, price: float, size: int, direction: int, side: str) -> float:
        """Realistic slippage based on size and market conditions"""
        tick_size = 0.25
        base_slippage = tick_size
        
        # Size impact
        size_factor = 1 + (size - 1) * 0.5
        
        # Entry vs exit
        side_factor = 1.2 if side == 'exit' else 1.0
        
        # Random component
        random_factor = np.random.uniform(0.8, 1.2)
        
        return base_slippage * size_factor * side_factor * random_factor
    
    def _adverse_slippage(self, price: float, size: int, direction: int, side: str) -> float:
        """Adverse slippage for stress testing"""
        tick_size = 0.25
        return tick_size * 2 * size  # 2 ticks per contract
    
    def _check_risk_limits(self) -> bool:
        """Check if trading is within risk limits"""
        # Daily loss limit
        if abs(self.daily_pnl) >= self.config.risk_limits['max_daily_loss']:
            return False
        
        # Position limit
        if len(self.open_trades) >= self.config.risk_limits['max_positions']:
            return False
        
        # Trailing drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = peak - current
            
            if drawdown >= self.config.risk_limits['trailing_drawdown']:
                return False
        
        return True
    
    def _check_daily_reset(self, current_time: datetime):
        """Reset daily metrics"""
        # Simple daily reset check (should be improved for actual trading hours)
        if hasattr(self, 'last_date'):
            if current_time.date() != self.last_date:
                self.daily_pnl = 0
                self.last_date = current_time.date()
        else:
            self.last_date = current_time.date()
    
    def _update_equity(self, current_bar: pd.Series):
        """Update equity curve with current positions"""
        current_equity = self.equity_curve[-1]
        
        # Add unrealized PnL
        for position_data in self.open_trades.values():
            trade = position_data['trade']
            current_price = current_bar['close']
            unrealized_pnl = (current_price - trade.entry_price) * trade.direction * trade.size * 20
            current_equity += unrealized_pnl
        
        # Don't append duplicate, update last value
        if len(self.equity_curve) > 1 and self.equity_curve[-1] != current_equity:
            self.equity_curve[-1] = current_equity
    
    def _close_all_positions(self, last_bar: pd.Series, bar_idx: int):
        """Close all remaining positions at end of backtest"""
        for position_data in list(self.open_trades.values()):
            trade = position_data['trade']
            
            # Force exit at last bar close
            exit_price = last_bar['close']
            trade.exit_time = last_bar.name
            trade.exit_price = exit_price
            trade.exit_reason = "backtest_end"
            trade.bars_held = bar_idx - position_data['bar_idx']
            
            # Calculate PnL
            gross_pnl = (exit_price - trade.entry_price) * trade.direction * trade.size * 20
            trade.commission += self.config.commission / 2
            trade.pnl = gross_pnl - trade.commission
            trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.size * 20)) * 100
            
            self.trades.append(trade)
        
        self.open_trades.clear()
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            # No trades executed
            return BacktestResults(
                trades=[],
                equity_curve=pd.Series(self.equity_curve),
                drawdown_curve=pd.Series([0]),
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                max_drawdown_duration=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                expectancy=0,
                kelly_criterion=0,
                var_95=0,
                cvar_95=0,
                downside_deviation=0,
                ulcer_index=0,
                avg_bars_held=0,
                avg_mae=0,
                avg_mfe=0,
                efficiency_ratio=0,
                pattern_performance={},
                confidence_distribution={},
                hourly_performance={},
                daily_performance={}
            )
        
        # Convert equity curve to series
        equity_series = pd.Series(self.equity_curve, index=data.index[:len(self.equity_curve)])
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] - self.config.initial_capital) / self.config.initial_capital * 100
        
        # Annualized return (assuming minute bars, 252 trading days)
        days = (data.index[-1] - data.index[0]).days
        annual_return = (((equity_series.iloc[-1] / self.config.initial_capital) ** (365 / days)) - 1) * 100 if days > 0 else 0
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 390)  # Annualized for minute bars
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std()
            sortino_ratio = returns.mean() / downside_deviation * np.sqrt(252 * 390) if downside_deviation > 0 else 0
        else:
            downside_deviation = 0
            sortino_ratio = 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Kelly criterion
        if avg_win > 0 and avg_loss < 0:
            kelly_criterion = (win_rate * avg_win + (1 - win_rate) * avg_loss) / avg_win
        else:
            kelly_criterion = 0
        
        # Risk metrics
        var_95 = returns.quantile(0.05) * 100 if len(returns) > 20 else 0
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100 if len(returns) > 20 else 0
        
        # Ulcer index
        if len(drawdown) > 0:
            ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        else:
            ulcer_index = 0
        
        # Efficiency metrics
        avg_bars_held = np.mean([t.bars_held for t in self.trades]) if self.trades else 0
        avg_mae = np.mean([t.mae for t in self.trades]) if self.trades else 0
        avg_mfe = np.mean([t.mfe for t in self.trades]) if self.trades else 0
        
        # Efficiency ratio (average profit vs maximum possible)
        if avg_mfe > 0:
            avg_profit = np.mean([t.pnl_percent for t in self.trades])
            efficiency_ratio = avg_profit / avg_mfe if self.trades else 0
        else:
            efficiency_ratio = 0
        
        # Pattern performance
        pattern_performance = self._analyze_pattern_performance()
        
        # Confidence distribution
        confidence_distribution = self._analyze_confidence_distribution()
        
        # Time-based performance
        hourly_performance = self._analyze_hourly_performance()
        daily_performance = self._analyze_daily_performance()
        
        # Maximum drawdown duration
        if len(drawdown) > 0:
            is_drawdown = drawdown < 0
            drawdown_periods = []
            current_period = 0
            
            for dd in is_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            max_drawdown_duration = 0
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion,
            var_95=var_95,
            cvar_95=cvar_95,
            downside_deviation=downside_deviation,
            ulcer_index=ulcer_index,
            avg_bars_held=avg_bars_held,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            efficiency_ratio=efficiency_ratio,
            pattern_performance=pattern_performance,
            confidence_distribution=confidence_distribution,
            hourly_performance=hourly_performance,
            daily_performance=daily_performance
        )
    
    def _analyze_pattern_performance(self) -> Dict:
        """Analyze performance by pattern type"""
        pattern_stats = {}
        
        for trade in self.trades:
            if trade.pattern:
                if trade.pattern not in pattern_stats:
                    pattern_stats[trade.pattern] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0,
                        'avg_confidence': 0
                    }
                
                pattern_stats[trade.pattern]['trades'] += 1
                if trade.pnl > 0:
                    pattern_stats[trade.pattern]['wins'] += 1
                pattern_stats[trade.pattern]['total_pnl'] += trade.pnl
                pattern_stats[trade.pattern]['avg_confidence'] += trade.confidence
        
        # Calculate averages
        for pattern in pattern_stats:
            stats = pattern_stats[pattern]
            stats['win_rate'] = stats['wins'] / stats['trades']
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
            stats['avg_confidence'] = stats['avg_confidence'] / stats['trades']
        
        return pattern_stats
    
    def _analyze_confidence_distribution(self) -> Dict:
        """Analyze performance by confidence level"""
        confidence_bins = {
            '60-65': {'trades': 0, 'wins': 0, 'pnl': 0},
            '65-70': {'trades': 0, 'wins': 0, 'pnl': 0},
            '70-75': {'trades': 0, 'wins': 0, 'pnl': 0},
            '75-80': {'trades': 0, 'wins': 0, 'pnl': 0},
            '80-85': {'trades': 0, 'wins': 0, 'pnl': 0},
            '85-90': {'trades': 0, 'wins': 0, 'pnl': 0},
            '90-95': {'trades': 0, 'wins': 0, 'pnl': 0},
            '95-100': {'trades': 0, 'wins': 0, 'pnl': 0}
        }
        
        for trade in self.trades:
            conf = trade.confidence
            
            if 60 <= conf < 65:
                bin_key = '60-65'
            elif 65 <= conf < 70:
                bin_key = '65-70'
            elif 70 <= conf < 75:
                bin_key = '70-75'
            elif 75 <= conf < 80:
                bin_key = '75-80'
            elif 80 <= conf < 85:
                bin_key = '80-85'
            elif 85 <= conf < 90:
                bin_key = '85-90'
            elif 90 <= conf < 95:
                bin_key = '90-95'
            else:
                bin_key = '95-100'
            
            confidence_bins[bin_key]['trades'] += 1
            if trade.pnl > 0:
                confidence_bins[bin_key]['wins'] += 1
            confidence_bins[bin_key]['pnl'] += trade.pnl
        
        # Calculate win rates
        for bin_key in confidence_bins:
            if confidence_bins[bin_key]['trades'] > 0:
                confidence_bins[bin_key]['win_rate'] = (
                    confidence_bins[bin_key]['wins'] / confidence_bins[bin_key]['trades']
                )
            else:
                confidence_bins[bin_key]['win_rate'] = 0
        
        return confidence_bins
    
    def _analyze_hourly_performance(self) -> Dict:
        """Analyze performance by hour of day"""
        hourly_stats = {}
        
        for trade in self.trades:
            if trade.entry_time:
                hour = trade.entry_time.hour
                
                if hour not in hourly_stats:
                    hourly_stats[hour] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0
                    }
                
                hourly_stats[hour]['trades'] += 1
                if trade.pnl > 0:
                    hourly_stats[hour]['wins'] += 1
                hourly_stats[hour]['total_pnl'] += trade.pnl
        
        # Calculate win rates
        for hour in hourly_stats:
            stats = hourly_stats[hour]
            stats['win_rate'] = stats['wins'] / stats['trades']
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
        
        return hourly_stats
    
    def _analyze_daily_performance(self) -> Dict:
        """Analyze performance by day of week"""
        daily_stats = {}
        
        for trade in self.trades:
            if trade.entry_time:
                day = trade.entry_time.weekday()  # 0=Monday, 6=Sunday
                
                if day not in daily_stats:
                    daily_stats[day] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0
                    }
                
                daily_stats[day]['trades'] += 1
                if trade.pnl > 0:
                    daily_stats[day]['wins'] += 1
                daily_stats[day]['total_pnl'] += trade.pnl
        
        # Calculate win rates
        for day in daily_stats:
            stats = daily_stats[day]
            stats['win_rate'] = stats['wins'] / stats['trades']
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
        
        return daily_stats
    
    def _reset_state(self):
        """Reset engine state for new backtest"""
        self.trades = []
        self.open_trades = {}
        self.equity_curve = [self.config.initial_capital]
        self.daily_pnl = 0
        self.trade_counter = 0
    
    def export_results(self, results: BacktestResults, filename: str):
        """Export backtest results to file"""
        # Prepare data for export
        export_data = {
            'summary': {
                'total_return': results.total_return,
                'annual_return': results.annual_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'total_trades': results.total_trades,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor
            },
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'pattern': t.pattern,
                    'confidence': t.confidence
                }
                for t in results.trades
            ],
            'pattern_performance': results.pattern_performance,
            'confidence_distribution': results.confidence_distribution
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filename}")