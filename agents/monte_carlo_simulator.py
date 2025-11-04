"""
Monte Carlo Simulation Module
Provides robust pattern testing through parameter variations and market simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import concurrent.futures
from scipy import stats
from dataclasses import dataclass

from agents.base_agent import BaseAgent
from utils.logger import setup_logger
from utils.slack_notifier import slack_notifier

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    iterations: int
    median_win_rate: float
    win_rate_std: float
    median_profit_factor: float
    profit_factor_std: float
    median_sharpe: float
    sharpe_std: float
    median_max_drawdown: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    distribution_data: Dict
    parameter_sensitivity: Dict
    robustness_score: float

class MonteCarloSimulator(BaseAgent):
    """
    Monte Carlo simulation for robust pattern validation
    Tests patterns with parameter variations and different market conditions
    """
    
    def __init__(self):
        """Initialize Monte Carlo simulator"""
        super().__init__('MonteCarloSimulator')
        self.logger = setup_logger('MonteCarloSimulator')
        
        # Simulation parameters
        self.default_iterations = 1000
        self.parameter_noise_levels = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10% variations
        self.slippage_range = (0, 3)  # 0-3 ticks of slippage
        self.commission_variations = [1.5, 2.25, 3.0]  # Different commission levels
        
        # Market regime variations
        self.volatility_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]  # Volatility scenarios
        self.trend_strength_variations = [-0.5, -0.25, 0, 0.25, 0.5]  # Trend adjustments
        
        self.logger.info("🎲 Monte Carlo Simulator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the simulator"""
        try:
            self.logger.info("Monte Carlo simulation engine ready")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, pattern: Dict, data: pd.DataFrame, iterations: int = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on a pattern
        
        Args:
            pattern: Pattern to test
            data: Historical data
            iterations: Number of simulations
        
        Returns:
            MonteCarloResult: Simulation results
        """
        return await self.simulate_pattern(pattern, data, iterations or self.default_iterations)
    
    async def simulate_pattern(self, pattern: Dict, data: pd.DataFrame, 
                              iterations: int = 1000) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on a trading pattern
        
        Args:
            pattern: Pattern definition
            data: Historical market data
            iterations: Number of simulation runs
        
        Returns:
            MonteCarloResult: Comprehensive simulation results
        """
        self.logger.info(f"🎲 Running {iterations} Monte Carlo simulations for {pattern.get('name', 'Unknown')}")
        
        try:
            simulation_results = []
            
            # Run simulations with different variations
            for i in range(iterations):
                if i % 100 == 0:
                    self.logger.debug(f"  Simulation {i}/{iterations}...")
                
                # Create variation for this iteration
                variation = self.create_pattern_variation(pattern, i)
                
                # Run backtest with variation
                backtest_result = await self.run_single_simulation(variation, data, i)
                simulation_results.append(backtest_result)
            
            # Analyze results
            result = self.analyze_simulation_results(simulation_results, pattern)
            
            # Calculate parameter sensitivity
            result.parameter_sensitivity = await self.analyze_parameter_sensitivity(
                pattern, data, iterations=100
            )
            
            self.logger.info(f"✅ Monte Carlo simulation complete")
            self.logger.info(f"   Median Win Rate: {result.median_win_rate:.1%} ± {result.win_rate_std:.1%}")
            self.logger.info(f"   Robustness Score: {result.robustness_score:.1%}")
            
            # Send Slack notification for Monte Carlo results
            import asyncio
            asyncio.create_task(slack_notifier.monte_carlo_complete(
                pattern.get('name', 'Unknown'),
                {
                    'iterations': result.iterations,
                    'median_win_rate': result.median_win_rate,
                    'win_rate_std': result.win_rate_std,
                    'robustness_score': result.robustness_score
                }
            ))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            raise
    
    def create_pattern_variation(self, pattern: Dict, seed: int) -> Dict:
        """
        Create a variation of the pattern with random parameter adjustments
        
        Args:
            pattern: Original pattern
            seed: Random seed for reproducibility
        
        Returns:
            Dict: Modified pattern
        """
        np.random.seed(seed)
        
        # Deep copy pattern
        import copy
        varied_pattern = copy.deepcopy(pattern)
        
        # Select noise level
        noise_level = np.random.choice(self.parameter_noise_levels)
        
        # Vary entry conditions
        if 'entry_conditions' in varied_pattern:
            varied_pattern['entry_conditions'] = self.vary_parameters(
                varied_pattern['entry_conditions'], noise_level
            )
        
        # Vary exit conditions
        if 'exit_conditions' in varied_pattern:
            varied_pattern['exit_conditions'] = self.vary_parameters(
                varied_pattern['exit_conditions'], noise_level
            )
        
        # Vary filters
        if 'filters' in varied_pattern:
            varied_pattern['filters'] = self.vary_parameters(
                varied_pattern['filters'], noise_level
            )
        
        return varied_pattern
    
    def vary_parameters(self, params: Dict, noise_level: float) -> Dict:
        """
        Add random variations to parameters
        
        Args:
            params: Parameters to vary
            noise_level: Amount of variation (0-1)
        
        Returns:
            Dict: Varied parameters
        """
        varied = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Add gaussian noise to numeric parameters
                noise = np.random.normal(0, noise_level)
                varied[key] = value * (1 + noise)
            elif isinstance(value, bool):
                # Occasionally flip boolean parameters
                varied[key] = value if np.random.random() > noise_level else not value
            else:
                # Keep other types unchanged
                varied[key] = value
        
        return varied
    
    async def run_single_simulation(self, pattern: Dict, data: pd.DataFrame, 
                                   iteration: int) -> Dict:
        """
        Run a single simulation iteration
        
        Args:
            pattern: Pattern variation to test
            data: Market data
            iteration: Iteration number
        
        Returns:
            Dict: Simulation results
        """
        # Apply market variations
        modified_data = self.apply_market_variations(data, iteration)
        
        # Apply trading cost variations
        slippage = np.random.uniform(*self.slippage_range)
        commission = np.random.choice(self.commission_variations)
        
        # Run simplified backtest
        trades = self.backtest_pattern(pattern, modified_data, slippage, commission)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        return {
            'iteration': iteration,
            'trades': len(trades),
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'total_return': metrics['total_return'],
            'slippage': slippage,
            'commission': commission
        }
    
    def apply_market_variations(self, data: pd.DataFrame, seed: int) -> pd.DataFrame:
        """
        Apply market condition variations to data
        
        Args:
            data: Original market data
            seed: Random seed
        
        Returns:
            pd.DataFrame: Modified market data
        """
        np.random.seed(seed)
        
        # Create copy
        modified_data = data.copy()
        
        # Apply volatility change
        volatility_mult = np.random.choice(self.volatility_multipliers)
        if volatility_mult != 1.0:
            # Adjust high/low ranges
            midpoint = (modified_data['High'] + modified_data['Low']) / 2
            range_size = modified_data['High'] - modified_data['Low']
            modified_data['High'] = midpoint + (range_size / 2) * volatility_mult
            modified_data['Low'] = midpoint - (range_size / 2) * volatility_mult
        
        # Apply trend adjustment
        trend_adj = np.random.choice(self.trend_strength_variations)
        if trend_adj != 0:
            # Add linear trend to prices
            trend = np.linspace(0, trend_adj, len(modified_data))
            modified_data['Open'] *= (1 + trend)
            modified_data['High'] *= (1 + trend)
            modified_data['Low'] *= (1 + trend)
            modified_data['Close'] *= (1 + trend)
        
        # Add random gaps (5% chance)
        if np.random.random() < 0.05:
            gap_indices = np.random.choice(len(modified_data), size=int(len(modified_data) * 0.01))
            for idx in gap_indices:
                if idx > 0:
                    gap_size = np.random.uniform(-0.02, 0.02)  # ±2% gaps
                    modified_data.loc[idx:, ['Open', 'High', 'Low', 'Close']] *= (1 + gap_size)
        
        return modified_data
    
    def backtest_pattern(self, pattern: Dict, data: pd.DataFrame, 
                        slippage: float, commission: float) -> List[Dict]:
        """
        Simplified backtest for Monte Carlo simulation
        
        Args:
            pattern: Pattern to test
            data: Market data
            slippage: Slippage in ticks
            commission: Commission per trade
        
        Returns:
            List[Dict]: Trade results
        """
        trades = []
        
        # Simplified pattern matching based on type
        pattern_type = pattern.get('type', 'unknown')
        
        if pattern_type == 'trend_bounce':
            trades = self.backtest_trend_bounce_simple(pattern, data, slippage, commission)
        elif pattern_type == 'sr_bounce':
            trades = self.backtest_sr_bounce_simple(pattern, data, slippage, commission)
        else:
            # Generic pattern backtest
            trades = self.backtest_generic_pattern(pattern, data, slippage, commission)
        
        return trades
    
    def backtest_trend_bounce_simple(self, pattern: Dict, data: pd.DataFrame,
                                    slippage: float, commission: float) -> List[Dict]:
        """Simplified trend bounce backtest"""
        trades = []
        conditions = pattern.get('entry_conditions', {})
        
        # Simple trend line calculation
        lookback = int(conditions.get('lookback_period', 100))
        
        for i in range(lookback, len(data) - 10):
            # Calculate simple trend line
            recent_lows = data['Low'].iloc[i-lookback:i]
            x = np.arange(len(recent_lows))
            slope, intercept = np.polyfit(x, recent_lows.values, 1)
            
            # Current trend value
            trend_value = slope * lookback + intercept
            current_low = data['Low'].iloc[i]
            
            # Check for bounce
            if abs(current_low - trend_value) / trend_value < 0.003:  # Within 0.3%
                entry_price = data['Close'].iloc[i] + slippage
                
                # Simple exit after 5 bars
                exit_idx = min(i + 5, len(data) - 1)
                exit_price = data['Close'].iloc[exit_idx] - slippage
                
                pnl = (exit_price - entry_price) - (commission * 2)
                
                trades.append({
                    'entry_idx': i,
                    'exit_idx': exit_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': pnl / entry_price
                })
        
        return trades
    
    def backtest_sr_bounce_simple(self, pattern: Dict, data: pd.DataFrame,
                                 slippage: float, commission: float) -> List[Dict]:
        """Simplified S/R bounce backtest"""
        trades = []
        
        # Find simple S/R levels using percentiles
        levels = [
            data['Low'].quantile(0.1),
            data['Low'].quantile(0.25),
            data['High'].quantile(0.75),
            data['High'].quantile(0.9)
        ]
        
        for i in range(50, len(data) - 10):
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]
            
            for level in levels:
                # Check for bounce off level
                if abs(current_low - level) < 5:  # Within 5 points
                    entry_price = data['Close'].iloc[i] + slippage
                    
                    # Exit after 3 bars
                    exit_idx = min(i + 3, len(data) - 1)
                    exit_price = data['Close'].iloc[exit_idx] - slippage
                    
                    pnl = (exit_price - entry_price) - (commission * 2)
                    
                    trades.append({
                        'entry_idx': i,
                        'exit_idx': exit_idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return': pnl / entry_price
                    })
                    break
        
        return trades
    
    def backtest_generic_pattern(self, pattern: Dict, data: pd.DataFrame,
                                slippage: float, commission: float) -> List[Dict]:
        """Generic pattern backtest"""
        trades = []
        
        # Random entry points for generic testing
        num_trades = int(len(data) * 0.05)  # 5% of bars
        entry_indices = np.random.choice(range(50, len(data) - 20), num_trades, replace=False)
        
        for i in sorted(entry_indices):
            entry_price = data['Close'].iloc[i] + slippage
            
            # Random hold period 1-10 bars
            hold_period = np.random.randint(1, 11)
            exit_idx = min(i + hold_period, len(data) - 1)
            exit_price = data['Close'].iloc[exit_idx] - slippage
            
            pnl = (exit_price - entry_price) - (commission * 2)
            
            trades.append({
                'entry_idx': i,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'return': pnl / entry_price
            })
        
        return trades
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        returns = [t['return'] for t in trades]
        pnls = [t['pnl'] for t in trades]
        
        # Win rate
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(trades)
        
        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Total return
        total_return = sum(returns)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }
    
    def analyze_simulation_results(self, results: List[Dict], pattern: Dict) -> MonteCarloResult:
        """
        Analyze Monte Carlo simulation results
        
        Args:
            results: List of simulation results
            pattern: Original pattern
        
        Returns:
            MonteCarloResult: Analysis results
        """
        # Extract metrics
        win_rates = [r['win_rate'] for r in results]
        profit_factors = [r['profit_factor'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        
        # Calculate statistics
        median_win_rate = np.median(win_rates)
        win_rate_std = np.std(win_rates)
        
        median_profit_factor = np.median(profit_factors)
        profit_factor_std = np.std(profit_factors)
        
        median_sharpe = np.median(sharpe_ratios)
        sharpe_std = np.std(sharpe_ratios)
        
        median_max_drawdown = np.median(max_drawdowns)
        
        # Calculate confidence intervals (95%)
        confidence_interval_lower = np.percentile(win_rates, 2.5)
        confidence_interval_upper = np.percentile(win_rates, 97.5)
        
        # Distribution data for visualization
        distribution_data = {
            'win_rates': win_rates,
            'profit_factors': profit_factors,
            'sharpe_ratios': sharpe_ratios,
            'max_drawdowns': max_drawdowns
        }
        
        # Calculate robustness score
        robustness_score = self.calculate_robustness_score(results)
        
        return MonteCarloResult(
            iterations=len(results),
            median_win_rate=median_win_rate,
            win_rate_std=win_rate_std,
            median_profit_factor=median_profit_factor,
            profit_factor_std=profit_factor_std,
            median_sharpe=median_sharpe,
            sharpe_std=sharpe_std,
            median_max_drawdown=median_max_drawdown,
            confidence_interval_lower=confidence_interval_lower,
            confidence_interval_upper=confidence_interval_upper,
            distribution_data=distribution_data,
            parameter_sensitivity={},  # Will be filled later
            robustness_score=robustness_score
        )
    
    def calculate_robustness_score(self, results: List[Dict]) -> float:
        """
        Calculate pattern robustness score
        
        Args:
            results: Simulation results
        
        Returns:
            float: Robustness score (0-1)
        """
        if not results:
            return 0
        
        win_rates = [r['win_rate'] for r in results]
        profit_factors = [r['profit_factor'] for r in results]
        
        # Components of robustness
        
        # 1. Consistency (low variance is good)
        wr_consistency = 1 - min(np.std(win_rates) / (np.mean(win_rates) + 0.001), 1)
        pf_consistency = 1 - min(np.std(profit_factors) / (np.mean(profit_factors) + 0.001), 1)
        
        # 2. Profitability (high median is good)
        wr_profitability = min(np.median(win_rates), 1)
        pf_profitability = min(np.median(profit_factors) / 2, 1)  # PF of 2 = max score
        
        # 3. Tail risk (few extreme losses)
        worst_10pct = np.percentile(win_rates, 10)
        tail_risk_score = min(worst_10pct / 0.4, 1)  # 40% win rate in worst 10% = max score
        
        # Weighted average
        robustness = (
            wr_consistency * 0.2 +
            pf_consistency * 0.2 +
            wr_profitability * 0.25 +
            pf_profitability * 0.25 +
            tail_risk_score * 0.1
        )
        
        return min(max(robustness, 0), 1)
    
    async def analyze_parameter_sensitivity(self, pattern: Dict, data: pd.DataFrame,
                                           iterations: int = 100) -> Dict:
        """
        Analyze sensitivity to parameter changes
        
        Args:
            pattern: Pattern to analyze
            data: Market data
            iterations: Number of iterations per parameter
        
        Returns:
            Dict: Parameter sensitivity analysis
        """
        sensitivity = {}
        
        # Test each numeric parameter
        for param_type in ['entry_conditions', 'exit_conditions', 'filters']:
            if param_type not in pattern:
                continue
            
            for param_name, param_value in pattern[param_type].items():
                if isinstance(param_value, (int, float)):
                    # Test different values
                    test_values = [
                        param_value * 0.8,
                        param_value * 0.9,
                        param_value,
                        param_value * 1.1,
                        param_value * 1.2
                    ]
                    
                    win_rates = []
                    for test_val in test_values:
                        # Create modified pattern
                        import copy
                        test_pattern = copy.deepcopy(pattern)
                        test_pattern[param_type][param_name] = test_val
                        
                        # Run quick test
                        trades = self.backtest_pattern(test_pattern, data, 0, 2.25)
                        metrics = self.calculate_metrics(trades)
                        win_rates.append(metrics['win_rate'])
                    
                    # Calculate sensitivity (variance in results)
                    sensitivity[f"{param_type}.{param_name}"] = {
                        'values': test_values,
                        'win_rates': win_rates,
                        'sensitivity': np.std(win_rates),
                        'optimal_value': test_values[np.argmax(win_rates)]
                    }
        
        return sensitivity
    
    async def walk_forward_analysis(self, pattern: Dict, data: pd.DataFrame,
                                   window_size: int = 252,
                                   step_size: int = 63) -> Dict:
        """
        Perform walk-forward analysis
        
        Args:
            pattern: Pattern to test
            data: Market data
            window_size: Training window size in bars
            step_size: Step size for moving window
        
        Returns:
            Dict: Walk-forward analysis results
        """
        self.logger.info(f"🚶 Running walk-forward analysis")
        
        results = []
        
        for i in range(0, len(data) - window_size - step_size, step_size):
            # Training data
            train_data = data.iloc[i:i+window_size]
            
            # Test data
            test_data = data.iloc[i+window_size:i+window_size+step_size]
            
            # Run backtest on test data
            trades = self.backtest_pattern(pattern, test_data, 1, 2.25)
            metrics = self.calculate_metrics(trades)
            
            results.append({
                'period': i // step_size,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'trades': len(trades)
            })
        
        # Analyze consistency
        win_rates = [r['win_rate'] for r in results]
        consistency = 1 - (np.std(win_rates) / (np.mean(win_rates) + 0.001)) if win_rates else 0
        
        return {
            'periods': results,
            'avg_win_rate': np.mean(win_rates),
            'consistency': consistency,
            'worst_period': min(win_rates) if win_rates else 0,
            'best_period': max(win_rates) if win_rates else 0
        }
    
    def generate_report(self, result: MonteCarloResult, pattern: Dict) -> str:
        """
        Generate human-readable report of Monte Carlo results
        
        Args:
            result: Monte Carlo results
            pattern: Pattern tested
        
        Returns:
            str: Formatted report
        """
        report = f"""
📊 Monte Carlo Simulation Report
================================
Pattern: {pattern.get('name', 'Unknown')}
Type: {pattern.get('type', 'unknown')}
Iterations: {result.iterations}

Performance Summary:
-------------------
Win Rate: {result.median_win_rate:.1%} ± {result.win_rate_std:.1%}
Profit Factor: {result.median_profit_factor:.2f} ± {result.profit_factor_std:.2f}
Sharpe Ratio: {result.median_sharpe:.2f} ± {result.sharpe_std:.2f}
Max Drawdown: {result.median_max_drawdown:.1%}

Confidence Intervals (95%):
--------------------------
Win Rate: [{result.confidence_interval_lower:.1%}, {result.confidence_interval_upper:.1%}]

Robustness Score: {result.robustness_score:.1%}
{'✅ ROBUST' if result.robustness_score > 0.7 else '⚠️ MODERATE' if result.robustness_score > 0.5 else '❌ FRAGILE'}

Parameter Sensitivity:
---------------------
"""
        
        if result.parameter_sensitivity:
            for param, data in result.parameter_sensitivity.items():
                report += f"  {param}: Sensitivity = {data['sensitivity']:.3f}\n"
                report += f"    Optimal: {data['optimal_value']:.2f}\n"
        
        return report