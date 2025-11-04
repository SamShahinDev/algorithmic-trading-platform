"""
Performance Tracker for Comprehensive Analytics
Real-time tracking and analysis of trading performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Complete performance metrics"""
    # Returns
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    
    # Efficiency metrics
    avg_trade_duration: float
    avg_mae: float  # Maximum adverse excursion
    avg_mfe: float  # Maximum favorable excursion
    efficiency_ratio: float
    
    # Consistency metrics
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    ulcer_index: float
    
    # Pattern analysis
    best_pattern: str
    worst_pattern: str
    best_hour: int
    worst_hour: int
    best_day: int
    worst_day: int


class PerformanceTracker:
    """Advanced performance tracking and analytics system"""
    
    def __init__(self, 
                 initial_capital: float = 50000,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize performance tracker
        
        Args:
            initial_capital: Starting capital
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Trade history
        self.trades = []
        self.daily_pnl = defaultdict(float)
        self.monthly_pnl = defaultdict(float)
        
        # Equity curve
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        
        # Pattern performance
        self.pattern_stats = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_duration': 0
        })
        
        # Time-based performance
        self.hourly_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0})
        self.daily_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0})
        self.monthly_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0})
        
        # Confidence analysis
        self.confidence_buckets = {
            range(60, 65): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(65, 70): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(70, 75): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(75, 80): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(80, 85): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(85, 90): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(90, 95): {'trades': 0, 'wins': 0, 'pnl': 0},
            range(95, 101): {'trades': 0, 'wins': 0, 'pnl': 0}
        }
        
        # Rolling windows for adaptive metrics
        self.rolling_sharpe_window = []
        self.rolling_win_rate_window = []
        
        # Current streak tracking
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
    def record_trade(self,
                    entry_time: datetime,
                    exit_time: datetime,
                    symbol: str,
                    side: int,  # 1 for long, -1 for short
                    entry_price: float,
                    exit_price: float,
                    quantity: int,
                    commission: float = 2.52,
                    pattern: Optional[str] = None,
                    confidence: float = 0,
                    mae: float = 0,
                    mfe: float = 0):
        """
        Record a completed trade
        
        Args:
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            symbol: Trading symbol
            side: Trade direction
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            commission: Trading commission
            pattern: Pattern that triggered trade
            confidence: Confidence score
            mae: Maximum adverse excursion
            mfe: Maximum favorable excursion
        """
        # Calculate PnL
        gross_pnl = (exit_price - entry_price) * side * quantity * 20  # NQ point value
        net_pnl = gross_pnl - commission
        pnl_percent = net_pnl / (entry_price * quantity * 20) * 100
        
        # Calculate duration
        duration = (exit_time - entry_time).total_seconds() / 60  # minutes
        
        # Create trade record
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'pnl_percent': pnl_percent,
            'duration': duration,
            'pattern': pattern,
            'confidence': confidence,
            'mae': mae,
            'mfe': mfe,
            'won': net_pnl > 0
        }
        
        self.trades.append(trade)
        
        # Update capital
        self.current_capital += net_pnl
        self.equity_curve.append(self.current_capital)
        self.timestamps.append(exit_time)
        
        # Update daily PnL
        date = exit_time.date()
        self.daily_pnl[date] += net_pnl
        
        # Update monthly PnL
        month = f"{exit_time.year}-{exit_time.month:02d}"
        self.monthly_pnl[month] += net_pnl
        
        # Update pattern stats
        if pattern:
            self.pattern_stats[pattern]['trades'] += 1
            self.pattern_stats[pattern]['total_pnl'] += net_pnl
            if net_pnl > 0:
                self.pattern_stats[pattern]['wins'] += 1
            
            # Update average duration
            current_avg = self.pattern_stats[pattern]['avg_duration']
            current_count = self.pattern_stats[pattern]['trades'] - 1
            self.pattern_stats[pattern]['avg_duration'] = (
                (current_avg * current_count + duration) / self.pattern_stats[pattern]['trades']
            )
        
        # Update time-based stats
        hour = entry_time.hour
        self.hourly_stats[hour]['trades'] += 1
        self.hourly_stats[hour]['pnl'] += net_pnl
        
        weekday = entry_time.weekday()
        self.daily_stats[weekday]['trades'] += 1
        self.daily_stats[weekday]['pnl'] += net_pnl
        
        self.monthly_stats[month]['trades'] += 1
        self.monthly_stats[month]['pnl'] += net_pnl
        
        # Update confidence analysis
        confidence_int = int(confidence)
        for bucket_range, stats in self.confidence_buckets.items():
            if confidence_int in bucket_range:
                stats['trades'] += 1
                stats['pnl'] += net_pnl
                if net_pnl > 0:
                    stats['wins'] += 1
                break
        
        # Update streaks
        if net_pnl > 0:
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))
        
        # Update rolling windows
        self._update_rolling_metrics(net_pnl, pnl_percent, trade['won'])
    
    def _update_rolling_metrics(self, pnl: float, pnl_percent: float, won: bool):
        """Update rolling performance metrics"""
        # Keep last 100 trades for rolling metrics
        window_size = 100
        
        # Rolling Sharpe
        self.rolling_sharpe_window.append(pnl_percent)
        if len(self.rolling_sharpe_window) > window_size:
            self.rolling_sharpe_window.pop(0)
        
        # Rolling win rate
        self.rolling_win_rate_window.append(1 if won else 0)
        if len(self.rolling_win_rate_window) > window_size:
            self.rolling_win_rate_window.pop(0)
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            PerformanceMetrics object with all metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['won']]
        losing_trades = [t for t in self.trades if not t['won']]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades
        
        # PnL statistics
        all_pnl = [t['net_pnl'] for t in self.trades]
        win_pnl = [t['net_pnl'] for t in winning_trades] if winning_trades else [0]
        loss_pnl = [t['net_pnl'] for t in losing_trades] if losing_trades else [0]
        
        avg_win = np.mean(win_pnl) if win_pnl else 0
        avg_loss = np.mean(loss_pnl) if loss_pnl else 0
        largest_win = max(win_pnl) if win_pnl else 0
        largest_loss = min(loss_pnl) if loss_pnl else 0
        
        # Returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Annual return (assuming minute data)
        if len(self.timestamps) > 1:
            days = (self.timestamps[-1] - self.timestamps[0]).days
            annual_return = ((self.current_capital / self.initial_capital) ** (365 / max(days, 1)) - 1) * 100 if days > 0 else 0
        else:
            annual_return = 0
        
        # Calculate returns series
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 390) * 100  # Annualized for minute bars
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252 * 390) * 100 if len(downside_returns) > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Max drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_duration = 0
        for in_dd in is_drawdown:
            if in_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # VaR and CVaR
        if len(returns) > 20:
            var_95 = np.percentile(returns, 5) * 100
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        else:
            var_95 = 0
            cvar_95 = 0
        
        # Risk-adjusted returns
        risk_free_daily = self.risk_free_rate / 252
        excess_returns = returns - risk_free_daily
        
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252 * 390) if returns.std() > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252 * 390) if downside_deviation > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Information ratio (if benchmark provided)
        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(returns):
            active_returns = returns - self.benchmark_returns
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
        else:
            information_ratio = 0
        
        # Treynor ratio (simplified, using volatility as proxy for beta)
        treynor_ratio = excess_returns.mean() / volatility * 252 if volatility > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['net_pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['net_pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Payoff ratio
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Efficiency metrics
        avg_duration = np.mean([t['duration'] for t in self.trades])
        avg_mae = np.mean([t['mae'] for t in self.trades])
        avg_mfe = np.mean([t['mfe'] for t in self.trades])
        
        # Efficiency ratio (how much of MFE was captured)
        avg_pnl_percent = np.mean([t['pnl_percent'] for t in self.trades])
        efficiency_ratio = avg_pnl_percent / avg_mfe if avg_mfe > 0 else 0
        
        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Ulcer index
        if len(drawdown) > 0:
            ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        else:
            ulcer_index = 0
        
        # Best/worst patterns
        pattern_performance = self._analyze_pattern_performance()
        best_pattern = max(pattern_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if pattern_performance else "None"
        worst_pattern = min(pattern_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if pattern_performance else "None"
        
        # Best/worst times
        hourly_performance = self._analyze_hourly_performance()
        best_hour = max(hourly_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if hourly_performance else 0
        worst_hour = min(hourly_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if hourly_performance else 0
        
        daily_performance = self._analyze_daily_performance()
        best_day = max(daily_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if daily_performance else 0
        worst_day = min(daily_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if daily_performance else 0
        
        # Monthly return
        monthly_returns = []
        for month, pnl in self.monthly_pnl.items():
            monthly_returns.append(pnl / self.initial_capital * 100)
        monthly_return = np.mean(monthly_returns) if monthly_returns else 0
        
        # Daily return
        daily_returns = []
        for date, pnl in self.daily_pnl.items():
            daily_returns.append(pnl / self.initial_capital * 100)
        daily_return = np.mean(daily_returns) if daily_returns else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_return,
            daily_return=daily_return,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            payoff_ratio=payoff_ratio,
            avg_trade_duration=avg_duration,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            efficiency_ratio=efficiency_ratio,
            consecutive_wins=self.max_win_streak,
            consecutive_losses=self.max_loss_streak,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            best_pattern=best_pattern,
            worst_pattern=worst_pattern,
            best_hour=best_hour,
            worst_hour=worst_hour,
            best_day=best_day,
            worst_day=worst_day
        )
    
    def _analyze_pattern_performance(self) -> Dict:
        """Analyze performance by pattern"""
        pattern_analysis = {}
        
        for pattern, stats in self.pattern_stats.items():
            if stats['trades'] > 0:
                pattern_analysis[pattern] = {
                    'trades': stats['trades'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'total_pnl': stats['total_pnl'],
                    'avg_pnl': stats['total_pnl'] / stats['trades'],
                    'avg_duration': stats['avg_duration']
                }
        
        return pattern_analysis
    
    def _analyze_hourly_performance(self) -> Dict:
        """Analyze performance by hour"""
        hourly_analysis = {}
        
        for hour, stats in self.hourly_stats.items():
            if stats['trades'] > 0:
                hourly_analysis[hour] = {
                    'trades': stats['trades'],
                    'total_pnl': stats['pnl'],
                    'avg_pnl': stats['pnl'] / stats['trades']
                }
        
        return hourly_analysis
    
    def _analyze_daily_performance(self) -> Dict:
        """Analyze performance by day of week"""
        daily_analysis = {}
        
        for day, stats in self.daily_stats.items():
            if stats['trades'] > 0:
                daily_analysis[day] = {
                    'trades': stats['trades'],
                    'total_pnl': stats['pnl'],
                    'avg_pnl': stats['pnl'] / stats['trades']
                }
        
        return daily_analysis
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, monthly_return=0, daily_return=0,
            volatility=0, downside_deviation=0, max_drawdown=0, max_drawdown_duration=0,
            var_95=0, cvar_95=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            information_ratio=0, treynor_ratio=0, total_trades=0, winning_trades=0,
            losing_trades=0, win_rate=0, avg_win=0, avg_loss=0, largest_win=0,
            largest_loss=0, profit_factor=0, expectancy=0, payoff_ratio=0,
            avg_trade_duration=0, avg_mae=0, avg_mfe=0, efficiency_ratio=0,
            consecutive_wins=0, consecutive_losses=0, recovery_factor=0, ulcer_index=0,
            best_pattern="None", worst_pattern="None", best_hour=0, worst_hour=0,
            best_day=0, worst_day=0
        )
    
    def get_rolling_metrics(self, window: int = 100) -> Dict:
        """Get rolling performance metrics"""
        if len(self.rolling_sharpe_window) < 2:
            return {
                'rolling_sharpe': 0,
                'rolling_win_rate': 0,
                'rolling_expectancy': 0
            }
        
        # Rolling Sharpe
        returns = pd.Series(self.rolling_sharpe_window)
        rolling_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Rolling win rate
        rolling_win_rate = np.mean(self.rolling_win_rate_window) if self.rolling_win_rate_window else 0
        
        # Rolling expectancy
        recent_trades = self.trades[-window:] if len(self.trades) > window else self.trades
        if recent_trades:
            recent_wins = [t['net_pnl'] for t in recent_trades if t['won']]
            recent_losses = [t['net_pnl'] for t in recent_trades if not t['won']]
            
            if recent_wins and recent_losses:
                win_rate = len(recent_wins) / len(recent_trades)
                avg_win = np.mean(recent_wins)
                avg_loss = np.mean(recent_losses)
                rolling_expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            else:
                rolling_expectancy = 0
        else:
            rolling_expectancy = 0
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_win_rate': rolling_win_rate,
            'rolling_expectancy': rolling_expectancy
        }
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Generate performance plots"""
        if not self.trades:
            print("No trades to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Equity curve
        ax = axes[0, 0]
        ax.plot(self.equity_curve)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Capital ($)')
        ax.grid(True)
        
        # 2. Drawdown
        ax = axes[0, 1]
        equity_series = pd.Series(self.equity_curve)
        cumulative = equity_series / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax.set_title('Drawdown')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        # 3. Win rate by confidence
        ax = axes[0, 2]
        confidence_data = []
        confidence_labels = []
        for bucket_range, stats in self.confidence_buckets.items():
            if stats['trades'] > 0:
                confidence_data.append(stats['wins'] / stats['trades'] * 100)
                confidence_labels.append(f"{min(bucket_range)}-{max(bucket_range)}")
        
        if confidence_data:
            ax.bar(confidence_labels, confidence_data)
            ax.set_title('Win Rate by Confidence')
            ax.set_xlabel('Confidence Range')
            ax.set_ylabel('Win Rate (%)')
            ax.tick_params(axis='x', rotation=45)
        
        # 4. PnL distribution
        ax = axes[1, 0]
        pnl_data = [t['net_pnl'] for t in self.trades]
        ax.hist(pnl_data, bins=30, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title('PnL Distribution')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        
        # 5. Pattern performance
        ax = axes[1, 1]
        pattern_pnl = []
        pattern_labels = []
        for pattern, stats in self.pattern_stats.items():
            if stats['trades'] > 0:
                pattern_pnl.append(stats['total_pnl'])
                pattern_labels.append(pattern[:10])  # Truncate long names
        
        if pattern_pnl:
            colors = ['green' if p > 0 else 'red' for p in pattern_pnl]
            ax.bar(pattern_labels, pattern_pnl, color=colors)
            ax.set_title('Pattern Performance')
            ax.set_xlabel('Pattern')
            ax.set_ylabel('Total PnL ($)')
            ax.tick_params(axis='x', rotation=45)
        
        # 6. Hourly performance heatmap
        ax = axes[1, 2]
        hourly_pnl = np.zeros(24)
        for hour, stats in self.hourly_stats.items():
            hourly_pnl[hour] = stats['pnl']
        
        im = ax.imshow(hourly_pnl.reshape(6, 4), cmap='RdYlGn', aspect='auto')
        ax.set_title('Hourly PnL Heatmap')
        ax.set_xlabel('Hour (grouped)')
        ax.set_ylabel('Hour Block')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Performance plots saved to {save_path}")
        else:
            plt.show()
    
    def export_report(self, filename: str):
        """Export comprehensive performance report"""
        metrics = self.calculate_metrics()
        rolling = self.get_rolling_metrics()
        
        report = {
            'summary': {
                'total_return': f"{metrics.total_return:.2f}%",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.2f}%",
                'win_rate': f"{metrics.win_rate:.2%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'total_trades': metrics.total_trades
            },
            'returns': {
                'total': f"{metrics.total_return:.2f}%",
                'annual': f"{metrics.annual_return:.2f}%",
                'monthly': f"{metrics.monthly_return:.2f}%",
                'daily': f"{metrics.daily_return:.2f}%"
            },
            'risk_metrics': {
                'volatility': f"{metrics.volatility:.2f}%",
                'downside_deviation': f"{metrics.downside_deviation:.2f}%",
                'max_drawdown': f"{metrics.max_drawdown:.2f}%",
                'max_dd_duration': f"{metrics.max_drawdown_duration} bars",
                'var_95': f"{metrics.var_95:.2f}%",
                'cvar_95': f"{metrics.cvar_95:.2f}%"
            },
            'risk_adjusted': {
                'sharpe': f"{metrics.sharpe_ratio:.2f}",
                'sortino': f"{metrics.sortino_ratio:.2f}",
                'calmar': f"{metrics.calmar_ratio:.2f}",
                'information': f"{metrics.information_ratio:.2f}",
                'treynor': f"{metrics.treynor_ratio:.2f}"
            },
            'trade_stats': {
                'total': metrics.total_trades,
                'wins': metrics.winning_trades,
                'losses': metrics.losing_trades,
                'win_rate': f"{metrics.win_rate:.2%}",
                'avg_win': f"${metrics.avg_win:.2f}",
                'avg_loss': f"${metrics.avg_loss:.2f}",
                'largest_win': f"${metrics.largest_win:.2f}",
                'largest_loss': f"${metrics.largest_loss:.2f}",
                'expectancy': f"${metrics.expectancy:.2f}",
                'payoff_ratio': f"{metrics.payoff_ratio:.2f}"
            },
            'efficiency': {
                'avg_duration': f"{metrics.avg_trade_duration:.1f} min",
                'avg_mae': f"{metrics.avg_mae:.2f}%",
                'avg_mfe': f"{metrics.avg_mfe:.2f}%",
                'efficiency_ratio': f"{metrics.efficiency_ratio:.2%}"
            },
            'consistency': {
                'max_win_streak': metrics.consecutive_wins,
                'max_loss_streak': metrics.consecutive_losses,
                'recovery_factor': f"{metrics.recovery_factor:.2f}",
                'ulcer_index': f"{metrics.ulcer_index:.2f}"
            },
            'best_worst': {
                'best_pattern': metrics.best_pattern,
                'worst_pattern': metrics.worst_pattern,
                'best_hour': metrics.best_hour,
                'worst_hour': metrics.worst_hour,
                'best_day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][metrics.best_day],
                'worst_day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][metrics.worst_day]
            },
            'rolling_metrics': rolling,
            'pattern_performance': self._analyze_pattern_performance(),
            'hourly_performance': self._analyze_hourly_performance(),
            'confidence_analysis': {
                f"{min(r)}-{max(r)}": {
                    'trades': s['trades'],
                    'win_rate': f"{s['wins']/s['trades']:.2%}" if s['trades'] > 0 else "0%",
                    'pnl': f"${s['pnl']:.2f}"
                }
                for r, s in self.confidence_buckets.items() if s['trades'] > 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Performance report exported to {filename}")
    
    def get_summary(self) -> str:
        """Get performance summary string"""
        metrics = self.calculate_metrics()
        
        summary = f"""
========================================
         PERFORMANCE SUMMARY
========================================
Total Return:     {metrics.total_return:>8.2f}%
Annual Return:    {metrics.annual_return:>8.2f}%
Sharpe Ratio:     {metrics.sharpe_ratio:>8.2f}
Max Drawdown:     {metrics.max_drawdown:>8.2f}%

Total Trades:     {metrics.total_trades:>8}
Win Rate:         {metrics.win_rate:>8.2%}
Profit Factor:    {metrics.profit_factor:>8.2f}
Expectancy:       ${metrics.expectancy:>7.2f}

Avg Win:          ${metrics.avg_win:>7.2f}
Avg Loss:         ${metrics.avg_loss:>7.2f}
Largest Win:      ${metrics.largest_win:>7.2f}
Largest Loss:     ${metrics.largest_loss:>7.2f}

Best Pattern:     {metrics.best_pattern}
Best Hour:        {metrics.best_hour:02d}:00
========================================
"""
        return summary