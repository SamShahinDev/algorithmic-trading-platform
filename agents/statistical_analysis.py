"""
Statistical Analysis Agent
Analyzes trading patterns and calculates performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict

from agents.base_agent import BaseAgent
from utils.logger import setup_logger

class StatisticalAnalysisAgent(BaseAgent):
    """
    Performs deep statistical analysis on trading patterns and results
    This is your performance analyst
    """
    
    def __init__(self):
        """Initialize statistical analysis agent"""
        super().__init__('StatisticalAnalysis')
        self.logger = setup_logger('StatisticalAnalysis')
        
        # Analysis parameters
        self.min_trades_for_significance = 30
        self.confidence_level = 0.95
        
        self.logger.info("📈 Statistical Analysis Agent initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.logger.info("Initializing statistical analysis systems...")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, results: Dict) -> Dict:
        """
        Main execution - analyze results
        
        Args:
            results: Backtest or trading results
        
        Returns:
            Dict: Statistical analysis
        """
        return await self.analyze_results(results)
    
    async def analyze_results(self, results: Dict) -> Dict:
        """
        Perform comprehensive statistical analysis on trading results
        
        Args:
            results: Trading/backtest results containing trades
        
        Returns:
            Dict: Detailed statistical metrics
        """
        self.logger.info("🔬 Performing statistical analysis...")
        
        try:
            trades = results.get('trades', [])
            
            if not trades:
                self.logger.warning("No trades to analyze")
                return self.empty_metrics()
            
            # Basic metrics
            basic_metrics = self.calculate_basic_metrics(trades)
            
            # Risk metrics
            risk_metrics = self.calculate_risk_metrics(trades)
            
            # Distribution analysis
            distribution_metrics = self.analyze_distribution(trades)
            
            # Time-based analysis
            time_metrics = self.analyze_time_patterns(trades)
            
            # Streak analysis
            streak_metrics = self.analyze_streaks(trades)
            
            # Monte Carlo simulation
            monte_carlo = await self.run_monte_carlo(trades)
            
            # Statistical significance tests
            significance = self.test_statistical_significance(trades)
            
            # Edge analysis
            edge_metrics = self.analyze_edge(trades)
            
            # Combine all metrics
            comprehensive_stats = {
                **basic_metrics,
                **risk_metrics,
                **distribution_metrics,
                **time_metrics,
                **streak_metrics,
                **monte_carlo,
                **significance,
                **edge_metrics,
                'confidence': self.calculate_confidence_score(basic_metrics, risk_metrics, len(trades))
            }
            
            # Log summary
            self.log_analysis_summary(comprehensive_stats)
            
            # Record success
            self.record_success()
            
            return comprehensive_stats
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            self.record_error(e)
            return self.empty_metrics()
    
    def calculate_basic_metrics(self, trades: List) -> Dict:
        """
        Calculate basic trading metrics
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Basic metrics
        """
        total_trades = len(trades)
        
        # Separate wins and losses
        wins = [t for t in trades if hasattr(t, 'pnl') and t.pnl > 0]
        losses = [t for t in trades if hasattr(t, 'pnl') and t.pnl < 0]
        breakeven = [t for t in trades if hasattr(t, 'pnl') and t.pnl == 0]
        
        # Win rate
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl is not None)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if total_trades > 0 else 0
        
        # Average trade
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'breakeven_trades': len(breakeven),
            'win_rate': win_rate,
            'loss_rate': 1 - win_rate,
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'largest_win': max([t.pnl for t in wins]) if wins else 0,
            'largest_loss': min([t.pnl for t in losses]) if losses else 0,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'average_trade': avg_trade,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
    
    def calculate_risk_metrics(self, trades: List) -> Dict:
        """
        Calculate risk-adjusted metrics
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Risk metrics
        """
        if not trades:
            return {}
        
        # Get returns
        returns = [t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
        
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Sharpe Ratio (annualized)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino Ratio (only downside volatility)
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.001
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio
        max_dd = self.calculate_max_drawdown(trades)
        calmar_ratio = (mean_return * 252) / abs(max_dd) if max_dd != 0 else 0
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns_array[returns_array <= var_95]) if len(returns_array[returns_array <= var_95]) > 0 else 0
        
        # Risk of Ruin (simplified)
        risk_of_ruin = self.calculate_risk_of_ruin(trades)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'max_drawdown_duration': self.calculate_max_drawdown_duration(trades),
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95,
            'risk_of_ruin': risk_of_ruin,
            'return_std': std_return,
            'downside_deviation': downside_std,
            'ulcer_index': self.calculate_ulcer_index(trades)
        }
    
    def analyze_distribution(self, trades: List) -> Dict:
        """
        Analyze the distribution of returns
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Distribution metrics
        """
        returns = [t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
        
        if len(returns) < 3:
            return {}
        
        returns_array = np.array(returns)
        
        # Distribution statistics
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        
        # Normality test
        if len(returns) >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(returns_array)
            is_normal = shapiro_p > 0.05
        else:
            shapiro_stat, shapiro_p = 0, 0
            is_normal = False
        
        # Percentiles
        percentiles = {
            'p5': np.percentile(returns_array, 5),
            'p25': np.percentile(returns_array, 25),
            'p50': np.percentile(returns_array, 50),  # Median
            'p75': np.percentile(returns_array, 75),
            'p95': np.percentile(returns_array, 95)
        }
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal_distribution': is_normal,
            'shapiro_p_value': shapiro_p,
            'median_return': percentiles['p50'],
            'percentiles': percentiles,
            'positive_skew': skewness > 0,
            'fat_tails': kurtosis > 3
        }
    
    def analyze_time_patterns(self, trades: List) -> Dict:
        """
        Analyze time-based patterns in trading
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Time-based metrics
        """
        if not trades:
            return {}
        
        # Group by hour of day
        hourly_performance = defaultdict(list)
        daily_performance = defaultdict(list)
        
        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'pnl') and trade.pnl is not None:
                hour = trade.entry_time.hour if hasattr(trade.entry_time, 'hour') else 0
                day = trade.entry_time.weekday() if hasattr(trade.entry_time, 'weekday') else 0
                
                hourly_performance[hour].append(trade.pnl)
                daily_performance[day].append(trade.pnl)
        
        # Best/worst hours
        best_hour = max(hourly_performance.items(), key=lambda x: np.mean(x[1])) if hourly_performance else (0, [0])
        worst_hour = min(hourly_performance.items(), key=lambda x: np.mean(x[1])) if hourly_performance else (0, [0])
        
        # Best/worst days
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        best_day = max(daily_performance.items(), key=lambda x: np.mean(x[1])) if daily_performance else (0, [0])
        worst_day = min(daily_performance.items(), key=lambda x: np.mean(x[1])) if daily_performance else (0, [0])
        
        # Average hold time
        hold_times = []
        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time') and trade.exit_time:
                hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # Hours
                hold_times.append(hold_time)
        
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        return {
            'best_hour': best_hour[0],
            'best_hour_avg_pnl': np.mean(best_hour[1]) if best_hour[1] else 0,
            'worst_hour': worst_hour[0],
            'worst_hour_avg_pnl': np.mean(worst_hour[1]) if worst_hour[1] else 0,
            'best_day': day_names[best_day[0]] if best_day[0] < len(day_names) else 'Unknown',
            'best_day_avg_pnl': np.mean(best_day[1]) if best_day[1] else 0,
            'worst_day': day_names[worst_day[0]] if worst_day[0] < len(day_names) else 'Unknown',
            'worst_day_avg_pnl': np.mean(worst_day[1]) if worst_day[1] else 0,
            'average_hold_time_hours': avg_hold_time,
            'max_hold_time_hours': max(hold_times) if hold_times else 0,
            'min_hold_time_hours': min(hold_times) if hold_times else 0
        }
    
    def analyze_streaks(self, trades: List) -> Dict:
        """
        Analyze winning and losing streaks
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Streak metrics
        """
        if not trades:
            return {}
        
        # Track streaks
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        win_streaks = []
        loss_streaks = []
        
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                if trade.pnl > 0:
                    current_win_streak += 1
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                        current_loss_streak = 0
                    max_win_streak = max(max_win_streak, current_win_streak)
                elif trade.pnl < 0:
                    current_loss_streak += 1
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                        current_win_streak = 0
                    max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        # Add final streaks
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'average_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'average_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'current_streak': current_win_streak if current_win_streak > 0 else -current_loss_streak,
            'total_win_streaks': len(win_streaks),
            'total_loss_streaks': len(loss_streaks)
        }
    
    async def run_monte_carlo(self, trades: List, simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation on trade results
        
        Args:
            trades: List of trades
            simulations: Number of simulations to run
        
        Returns:
            Dict: Monte Carlo results
        """
        if not trades:
            return {}
        
        returns = [t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
        
        if len(returns) < 10:
            return {}
        
        self.logger.debug(f"Running {simulations} Monte Carlo simulations...")
        
        # Run simulations
        final_equities = []
        max_drawdowns = []
        
        for _ in range(simulations):
            # Randomly sample returns with replacement
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate equity curve
            equity = 50000  # Starting capital
            peak = equity
            max_dd = 0
            
            for ret in simulated_returns:
                equity += ret
                if equity > peak:
                    peak = equity
                else:
                    dd = (peak - equity) / peak
                    max_dd = max(max_dd, dd)
            
            final_equities.append(equity)
            max_drawdowns.append(max_dd)
        
        # Calculate statistics
        return {
            'monte_carlo_median_equity': np.median(final_equities),
            'monte_carlo_95_percentile': np.percentile(final_equities, 95),
            'monte_carlo_5_percentile': np.percentile(final_equities, 5),
            'monte_carlo_probability_profit': sum(1 for e in final_equities if e > 50000) / simulations,
            'monte_carlo_avg_max_drawdown': np.mean(max_drawdowns),
            'monte_carlo_worst_drawdown': np.max(max_drawdowns)
        }
    
    def test_statistical_significance(self, trades: List) -> Dict:
        """
        Test statistical significance of results
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Statistical significance metrics
        """
        if not trades:
            return {}
        
        returns = [t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
        
        if len(returns) < 2:
            return {}
        
        # T-test: Is average return significantly different from 0?
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        is_significant = p_value < 0.05
        
        # Calculate confidence interval for mean return
        mean_return = np.mean(returns)
        std_error = stats.sem(returns)
        confidence_interval = stats.t.interval(
            0.95, 
            len(returns) - 1, 
            loc=mean_return, 
            scale=std_error
        )
        
        # Calculate required sample size for significance
        effect_size = mean_return / np.std(returns) if np.std(returns) > 0 else 0
        
        return {
            'is_statistically_significant': is_significant,
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_return_ci_lower': confidence_interval[0],
            'mean_return_ci_upper': confidence_interval[1],
            'effect_size': effect_size,
            'sample_size': len(returns),
            'min_sample_size_needed': self.min_trades_for_significance
        }
    
    def analyze_edge(self, trades: List) -> Dict:
        """
        Analyze the trading edge
        
        Args:
            trades: List of trades
        
        Returns:
            Dict: Edge analysis metrics
        """
        if not trades:
            return {}
        
        wins = [t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if hasattr(t, 'pnl') and t.pnl < 0]
        
        if not wins or not losses:
            return {}
        
        # Kelly Criterion
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0
        
        # Edge calculation
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        edge_ratio = edge / avg_loss if avg_loss > 0 else 0
        
        return {
            'trading_edge': edge,
            'edge_ratio': edge_ratio,
            'kelly_fraction': kelly_fraction,
            'optimal_position_size_pct': kelly_fraction * 100,
            'breakeven_win_rate': avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 0
        }
    
    def calculate_max_drawdown(self, trades: List) -> float:
        """
        Calculate maximum drawdown from trades
        
        Args:
            trades: List of trades
        
        Returns:
            float: Maximum drawdown percentage
        """
        if not trades:
            return 0
        
        equity = 50000  # Starting capital
        peak = equity
        max_dd = 0
        
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                equity += trade.pnl
                if equity > peak:
                    peak = equity
                else:
                    dd = (peak - equity) / peak
                    max_dd = max(max_dd, dd)
        
        return max_dd
    
    def calculate_max_drawdown_duration(self, trades: List) -> int:
        """
        Calculate maximum drawdown duration in number of trades
        
        Args:
            trades: List of trades
        
        Returns:
            int: Maximum drawdown duration
        """
        if not trades:
            return 0
        
        equity = 50000
        peak = equity
        drawdown_start = 0
        max_duration = 0
        current_duration = 0
        
        for i, trade in enumerate(trades):
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                equity += trade.pnl
                
                if equity >= peak:
                    peak = equity
                    if current_duration > 0:
                        max_duration = max(max_duration, current_duration)
                        current_duration = 0
                else:
                    current_duration += 1
        
        return max(max_duration, current_duration)
    
    def calculate_risk_of_ruin(self, trades: List, ruin_level: float = 0.5) -> float:
        """
        Calculate risk of ruin (simplified)
        
        Args:
            trades: List of trades
            ruin_level: Fraction of capital considered ruin
        
        Returns:
            float: Risk of ruin probability
        """
        if not trades:
            return 0
        
        wins = [t for t in trades if hasattr(t, 'pnl') and t.pnl > 0]
        losses = [t for t in trades if hasattr(t, 'pnl') and t.pnl < 0]
        
        if not wins or not losses:
            return 0
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl for t in wins])
        avg_loss = abs(np.mean([t.pnl for t in losses]))
        
        if avg_win == 0:
            return 1.0
        
        # Simplified risk of ruin formula
        if win_rate >= 0.5:
            risk_of_ruin = ((1 - win_rate) / win_rate) ** (50000 * ruin_level / avg_loss)
        else:
            risk_of_ruin = 1.0
        
        return min(1.0, risk_of_ruin)
    
    def calculate_ulcer_index(self, trades: List) -> float:
        """
        Calculate Ulcer Index (measures downside volatility)
        
        Args:
            trades: List of trades
        
        Returns:
            float: Ulcer Index
        """
        if not trades:
            return 0
        
        equity_curve = [50000]
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                equity_curve.append(equity_curve[-1] + trade.pnl)
        
        if len(equity_curve) < 2:
            return 0
        
        # Calculate percentage drawdowns from rolling peak
        drawdowns = []
        peak = equity_curve[0]
        
        for equity in equity_curve[1:]:
            if equity > peak:
                peak = equity
                drawdowns.append(0)
            else:
                dd_pct = ((peak - equity) / peak) * 100
                drawdowns.append(dd_pct)
        
        # Ulcer Index = sqrt(mean(drawdowns^2))
        if drawdowns:
            ulcer_index = np.sqrt(np.mean([d**2 for d in drawdowns]))
        else:
            ulcer_index = 0
        
        return ulcer_index
    
    def calculate_confidence_score(self, basic_metrics: Dict, risk_metrics: Dict, sample_size: int) -> float:
        """
        Calculate overall confidence score for the pattern
        
        Args:
            basic_metrics: Basic performance metrics
            risk_metrics: Risk-adjusted metrics
            sample_size: Number of trades
        
        Returns:
            float: Confidence score (0-1)
        """
        score = 0
        
        # Win rate component (25%)
        win_rate = basic_metrics.get('win_rate', 0)
        if win_rate > 0.6:
            score += 0.25
        elif win_rate > 0.55:
            score += 0.15
        elif win_rate > 0.5:
            score += 0.10
        
        # Profit factor component (25%)
        profit_factor = basic_metrics.get('profit_factor', 0)
        if profit_factor > 2:
            score += 0.25
        elif profit_factor > 1.5:
            score += 0.15
        elif profit_factor > 1.2:
            score += 0.10
        
        # Sharpe ratio component (20%)
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            score += 0.20
        elif sharpe > 1.5:
            score += 0.12
        elif sharpe > 1:
            score += 0.08
        
        # Sample size component (20%)
        if sample_size > 100:
            score += 0.20
        elif sample_size > 50:
            score += 0.12
        elif sample_size > 30:
            score += 0.08
        
        # Max drawdown component (10%)
        max_dd = risk_metrics.get('max_drawdown', 1)
        if max_dd < 0.10:
            score += 0.10
        elif max_dd < 0.15:
            score += 0.06
        elif max_dd < 0.20:
            score += 0.03
        
        return min(1.0, score)
    
    def empty_metrics(self) -> Dict:
        """
        Return empty metrics structure
        
        Returns:
            Dict: Empty metrics
        """
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'confidence': 0,
            'sample_size': 0
        }
    
    def log_analysis_summary(self, stats: Dict):
        """
        Log a summary of the analysis
        
        Args:
            stats: Statistical metrics
        """
        self.logger.info("📊 Analysis Summary:")
        self.logger.info(f"  Total Trades: {stats.get('total_trades', 0)}")
        self.logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
        self.logger.info(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
        self.logger.info(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"  Max Drawdown: {stats.get('max_drawdown', 0):.1%}")
        self.logger.info(f"  Confidence Score: {stats.get('confidence', 0):.1%}")
        
        if stats.get('is_statistically_significant'):
            self.logger.info("  ✅ Results are statistically significant")
        else:
            self.logger.info("  ⚠️ Results not yet statistically significant")