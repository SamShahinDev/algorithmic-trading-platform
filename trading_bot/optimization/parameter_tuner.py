"""
Parameter Tuner for Strategy Optimization
Uses various optimization techniques to find optimal trading parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import itertools
from scipy import optimize
from sklearn.model_selection import TimeSeriesSplit
import optuna
import json
import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from optimization.backtest_engine import BacktestEngine, BacktestConfig


class OptimizationMethod(Enum):
    """Optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    WALK_FORWARD = "walk_forward"


@dataclass
class ParameterRange:
    """Parameter range for optimization"""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    param_type: str = "float"  # float, int, categorical
    categories: Optional[List] = None


@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_method: OptimizationMethod
    metric_optimized: str
    
    # Performance metrics
    in_sample_performance: Dict
    out_sample_performance: Optional[Dict]
    
    # Analysis
    parameter_importance: Dict[str, float]
    parameter_stability: Dict[str, float]
    overfitting_score: float
    
    # Convergence
    convergence_history: List[float]
    total_iterations: int
    time_elapsed: float


class ParameterTuner:
    """Advanced parameter optimization for trading strategies"""
    
    def __init__(self,
                 strategy_func: Callable,
                 parameter_ranges: List[ParameterRange],
                 optimization_metric: str = "sharpe_ratio",
                 minimize: bool = False):
        """
        Initialize parameter tuner
        
        Args:
            strategy_func: Strategy function to optimize
            parameter_ranges: List of parameter ranges
            optimization_metric: Metric to optimize
            minimize: Whether to minimize (False = maximize)
        """
        self.strategy_func = strategy_func
        self.parameter_ranges = parameter_ranges
        self.optimization_metric = optimization_metric
        self.minimize = minimize
        
        # Backtesting engine
        self.backtest_engine = BacktestEngine()
        
        # Results tracking
        self.all_results = []
        self.best_params = None
        self.best_score = float('inf') if minimize else float('-inf')
        
        # Convergence tracking
        self.convergence_history = []
        self.iteration_count = 0
        
    def optimize(self,
                data: pd.DataFrame,
                method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                n_iterations: int = 100,
                validation_split: float = 0.2,
                n_jobs: int = 1) -> OptimizationResult:
        """
        Run parameter optimization
        
        Args:
            data: Historical OHLCV data
            method: Optimization method
            n_iterations: Number of iterations
            validation_split: Validation data percentage
            n_jobs: Number of parallel jobs
            
        Returns:
            OptimizationResult object
        """
        print(f"Starting {method.value} optimization for {n_iterations} iterations...")
        
        # Split data
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        # Run optimization based on method
        if method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search(train_data, n_iterations)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search(train_data, n_iterations)
        elif method == OptimizationMethod.BAYESIAN:
            result = self._bayesian_optimization(train_data, n_iterations)
        elif method == OptimizationMethod.GENETIC:
            result = self._genetic_algorithm(train_data, n_iterations)
        elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._differential_evolution(train_data, n_iterations)
        elif method == OptimizationMethod.WALK_FORWARD:
            result = self._walk_forward_optimization(data, n_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Validate on out-of-sample data
        if validation_split > 0:
            out_sample_perf = self._evaluate_parameters(val_data, self.best_params)
        else:
            out_sample_perf = None
        
        # Calculate parameter importance and stability
        param_importance = self._calculate_parameter_importance()
        param_stability = self._calculate_parameter_stability()
        
        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(
            result['in_sample_performance'],
            out_sample_perf
        )
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=self.all_results,
            optimization_method=method,
            metric_optimized=self.optimization_metric,
            in_sample_performance=result['in_sample_performance'],
            out_sample_performance=out_sample_perf,
            parameter_importance=param_importance,
            parameter_stability=param_stability,
            overfitting_score=overfitting_score,
            convergence_history=self.convergence_history,
            total_iterations=self.iteration_count,
            time_elapsed=result.get('time_elapsed', 0)
        )
    
    def _grid_search(self, data: pd.DataFrame, max_iterations: int) -> Dict:
        """Grid search optimization"""
        import time
        start_time = time.time()
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid()
        
        # Limit iterations
        if len(param_grid) > max_iterations:
            param_grid = param_grid[:max_iterations]
        
        print(f"Testing {len(param_grid)} parameter combinations...")
        
        for params in param_grid:
            score = self._evaluate_parameters(data, params)
            self._update_best(params, score)
            self.iteration_count += 1
        
        return {
            'in_sample_performance': self._evaluate_parameters(data, self.best_params),
            'time_elapsed': time.time() - start_time
        }
    
    def _random_search(self, data: pd.DataFrame, n_iterations: int) -> Dict:
        """Random search optimization"""
        import time
        import random
        start_time = time.time()
        
        for _ in range(n_iterations):
            # Generate random parameters
            params = {}
            for param_range in self.parameter_ranges:
                if param_range.param_type == "float":
                    value = random.uniform(param_range.min_value, param_range.max_value)
                elif param_range.param_type == "int":
                    value = random.randint(int(param_range.min_value), int(param_range.max_value))
                elif param_range.param_type == "categorical":
                    value = random.choice(param_range.categories)
                else:
                    value = param_range.min_value
                
                params[param_range.name] = value
            
            score = self._evaluate_parameters(data, params)
            self._update_best(params, score)
            self.iteration_count += 1
        
        return {
            'in_sample_performance': self._evaluate_parameters(data, self.best_params),
            'time_elapsed': time.time() - start_time
        }
    
    def _bayesian_optimization(self, data: pd.DataFrame, n_iterations: int) -> Dict:
        """Bayesian optimization using Optuna"""
        import time
        start_time = time.time()
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_range in self.parameter_ranges:
                if param_range.param_type == "float":
                    value = trial.suggest_float(
                        param_range.name,
                        param_range.min_value,
                        param_range.max_value
                    )
                elif param_range.param_type == "int":
                    value = trial.suggest_int(
                        param_range.name,
                        int(param_range.min_value),
                        int(param_range.max_value)
                    )
                elif param_range.param_type == "categorical":
                    value = trial.suggest_categorical(
                        param_range.name,
                        param_range.categories
                    )
                else:
                    value = param_range.min_value
                
                params[param_range.name] = value
            
            # Evaluate
            score = self._evaluate_parameters(data, params)
            self._update_best(params, score)
            self.iteration_count += 1
            
            # Return score (Optuna minimizes by default)
            return -score if not self.minimize else score
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_iterations, show_progress_bar=True)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = -study.best_value if not self.minimize else study.best_value
        
        return {
            'in_sample_performance': self._evaluate_parameters(data, self.best_params),
            'time_elapsed': time.time() - start_time
        }
    
    def _genetic_algorithm(self, data: pd.DataFrame, n_iterations: int) -> Dict:
        """Genetic algorithm optimization"""
        import time
        import random
        start_time = time.time()
        
        # GA parameters
        population_size = 50
        mutation_rate = 0.1
        crossover_rate = 0.7
        elite_size = 5
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_range in self.parameter_ranges:
                if param_range.param_type == "float":
                    value = random.uniform(param_range.min_value, param_range.max_value)
                elif param_range.param_type == "int":
                    value = random.randint(int(param_range.min_value), int(param_range.max_value))
                else:
                    value = param_range.min_value
                individual[param_range.name] = value
            population.append(individual)
        
        # Evolution loop
        generations = n_iterations // population_size
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = self._evaluate_parameters(data, individual)
                fitness_scores.append(score)
                self._update_best(individual, score)
                self.iteration_count += 1
            
            # Selection
            sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), 
                                              key=lambda pair: pair[0], 
                                              reverse=(not self.minimize))]
            
            # Keep elite
            new_population = sorted_pop[:elite_size]
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Select parents
                parent1 = self._tournament_selection(sorted_pop[:population_size//2], fitness_scores)
                parent2 = self._tournament_selection(sorted_pop[:population_size//2], fitness_scores)
                
                # Crossover
                if random.random() < crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        return {
            'in_sample_performance': self._evaluate_parameters(data, self.best_params),
            'time_elapsed': time.time() - start_time
        }
    
    def _differential_evolution(self, data: pd.DataFrame, n_iterations: int) -> Dict:
        """Differential evolution optimization"""
        import time
        start_time = time.time()
        
        # Convert parameter ranges to bounds
        bounds = []
        param_names = []
        for param_range in self.parameter_ranges:
            if param_range.param_type in ["float", "int"]:
                bounds.append((param_range.min_value, param_range.max_value))
                param_names.append(param_range.name)
        
        def objective_func(x):
            # Convert array to parameter dictionary
            params = dict(zip(param_names, x))
            
            # Convert integers
            for param_range in self.parameter_ranges:
                if param_range.name in params and param_range.param_type == "int":
                    params[param_range.name] = int(params[param_range.name])
            
            score = self._evaluate_parameters(data, params)
            self.iteration_count += 1
            
            # DE minimizes
            return -score if not self.minimize else score
        
        # Run differential evolution
        result = optimize.differential_evolution(
            objective_func,
            bounds,
            maxiter=n_iterations // 15,  # Each iteration tests ~15 candidates
            seed=42
        )
        
        # Get best parameters
        self.best_params = dict(zip(param_names, result.x))
        for param_range in self.parameter_ranges:
            if param_range.name in self.best_params and param_range.param_type == "int":
                self.best_params[param_range.name] = int(self.best_params[param_range.name])
        
        self.best_score = -result.fun if not self.minimize else result.fun
        
        return {
            'in_sample_performance': self._evaluate_parameters(data, self.best_params),
            'time_elapsed': time.time() - start_time
        }
    
    def _walk_forward_optimization(self, data: pd.DataFrame, n_iterations: int) -> Dict:
        """Walk-forward optimization for robustness"""
        import time
        start_time = time.time()
        
        # Setup walk-forward windows
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        walk_forward_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
            print(f"Walk-forward fold {fold + 1}/{n_splits}")
            
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Optimize on training data
            fold_best_params = None
            fold_best_score = float('-inf')
            
            # Use random search for each fold
            for _ in range(n_iterations // n_splits):
                params = {}
                for param_range in self.parameter_ranges:
                    if param_range.param_type == "float":
                        import random
                        value = random.uniform(param_range.min_value, param_range.max_value)
                    elif param_range.param_type == "int":
                        import random
                        value = random.randint(int(param_range.min_value), int(param_range.max_value))
                    else:
                        value = param_range.min_value
                    params[param_range.name] = value
                
                score = self._evaluate_parameters(train_data, params)
                
                if score > fold_best_score:
                    fold_best_score = score
                    fold_best_params = params
                
                self.iteration_count += 1
            
            # Validate on out-of-sample
            val_score = self._evaluate_parameters(val_data, fold_best_params)
            
            walk_forward_results.append({
                'fold': fold,
                'params': fold_best_params,
                'train_score': fold_best_score,
                'val_score': val_score
            })
        
        # Select best parameters based on average validation score
        avg_val_scores = {}
        for result in walk_forward_results:
            params_str = str(result['params'])
            if params_str not in avg_val_scores:
                avg_val_scores[params_str] = []
            avg_val_scores[params_str].append(result['val_score'])
        
        # Find best average
        best_params_str = max(avg_val_scores.keys(), 
                             key=lambda k: np.mean(avg_val_scores[k]))
        
        # Get actual parameters
        for result in walk_forward_results:
            if str(result['params']) == best_params_str:
                self.best_params = result['params']
                self.best_score = np.mean(avg_val_scores[best_params_str])
                break
        
        return {
            'in_sample_performance': self._evaluate_parameters(data, self.best_params),
            'walk_forward_results': walk_forward_results,
            'time_elapsed': time.time() - start_time
        }
    
    def _evaluate_parameters(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Evaluate parameters using backtesting"""
        # Create strategy function with parameters
        def strategy_with_params(df, features):
            return self.strategy_func(df, features, **params)
        
        # Run backtest
        results = self.backtest_engine.backtest_strategy(data, strategy_with_params)
        
        # Get metric
        if self.optimization_metric == "sharpe_ratio":
            score = results.sharpe_ratio
        elif self.optimization_metric == "total_return":
            score = results.total_return
        elif self.optimization_metric == "win_rate":
            score = results.win_rate
        elif self.optimization_metric == "profit_factor":
            score = results.profit_factor
        elif self.optimization_metric == "calmar_ratio":
            score = results.calmar_ratio
        elif self.optimization_metric == "sortino_ratio":
            score = results.sortino_ratio
        else:
            score = results.total_return
        
        # Store result
        self.all_results.append({
            'params': params.copy(),
            'score': score,
            'metrics': {
                'sharpe': results.sharpe_ratio,
                'return': results.total_return,
                'max_dd': results.max_drawdown,
                'trades': results.total_trades
            }
        })
        
        return score
    
    def _update_best(self, params: Dict, score: float):
        """Update best parameters if improved"""
        if self.minimize:
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        else:
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        self.convergence_history.append(self.best_score)
    
    def _generate_parameter_grid(self) -> List[Dict]:
        """Generate parameter grid for grid search"""
        param_values = []
        param_names = []
        
        for param_range in self.parameter_ranges:
            param_names.append(param_range.name)
            
            if param_range.param_type == "categorical":
                values = param_range.categories
            elif param_range.step:
                values = np.arange(param_range.min_value, 
                                 param_range.max_value + param_range.step,
                                 param_range.step)
            else:
                # Default to 10 values
                values = np.linspace(param_range.min_value, 
                                   param_range.max_value, 10)
            
            if param_range.param_type == "int":
                values = [int(v) for v in values]
            
            param_values.append(values)
        
        # Generate all combinations
        grid = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            grid.append(params)
        
        return grid
    
    def _tournament_selection(self, population: List[Dict], fitness: List[float], 
                             tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm"""
        import random
        
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        winner = max(tournament, key=lambda x: x[1] if not self.minimize else -x[1])
        
        return winner[0]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for genetic algorithm"""
        import random
        
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutation for genetic algorithm"""
        import random
        
        mutated = individual.copy()
        
        # Mutate one random parameter
        param_to_mutate = random.choice(self.parameter_ranges)
        
        if param_to_mutate.param_type == "float":
            # Add Gaussian noise
            current = mutated[param_to_mutate.name]
            noise = random.gauss(0, (param_to_mutate.max_value - param_to_mutate.min_value) * 0.1)
            mutated[param_to_mutate.name] = np.clip(
                current + noise,
                param_to_mutate.min_value,
                param_to_mutate.max_value
            )
        elif param_to_mutate.param_type == "int":
            # Random new value
            mutated[param_to_mutate.name] = random.randint(
                int(param_to_mutate.min_value),
                int(param_to_mutate.max_value)
            )
        
        return mutated
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate importance of each parameter"""
        if len(self.all_results) < 10:
            return {}
        
        importance = {}
        
        for param_range in self.parameter_ranges:
            param_name = param_range.name
            
            # Group results by parameter value
            param_scores = {}
            for result in self.all_results:
                param_value = result['params'].get(param_name)
                if param_value is not None:
                    if param_value not in param_scores:
                        param_scores[param_value] = []
                    param_scores[param_value].append(result['score'])
            
            # Calculate variance in scores
            if param_scores:
                mean_scores = [np.mean(scores) for scores in param_scores.values()]
                importance[param_name] = np.std(mean_scores)
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _calculate_parameter_stability(self) -> Dict[str, float]:
        """Calculate stability of parameter values"""
        if len(self.all_results) < 20:
            return {}
        
        stability = {}
        
        # Get top 20% of results
        sorted_results = sorted(self.all_results, 
                              key=lambda x: x['score'],
                              reverse=(not self.minimize))
        top_results = sorted_results[:max(1, len(sorted_results) // 5)]
        
        for param_range in self.parameter_ranges:
            param_name = param_range.name
            
            # Get parameter values from top results
            values = [r['params'].get(param_name) for r in top_results 
                     if param_name in r['params']]
            
            if values and param_range.param_type in ["float", "int"]:
                # Calculate coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    stability[param_name] = 1 / (1 + cv)  # Higher is more stable
                else:
                    stability[param_name] = 1.0
            else:
                stability[param_name] = 0.5  # Default for categorical
        
        return stability
    
    def _calculate_overfitting_score(self, in_sample: Dict, out_sample: Optional[Dict]) -> float:
        """Calculate overfitting score"""
        if not out_sample or not isinstance(in_sample, dict):
            return 0
        
        # Simple version - compare key metric
        in_score = in_sample if isinstance(in_sample, (int, float)) else self.best_score
        out_score = out_sample if isinstance(out_sample, (int, float)) else 0
        
        if in_score != 0:
            degradation = (in_score - out_score) / abs(in_score)
            return min(1.0, max(0, degradation))
        
        return 0
    
    def plot_convergence(self):
        """Plot optimization convergence"""
        import matplotlib.pyplot as plt
        
        if not self.convergence_history:
            print("No convergence history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history)
        plt.xlabel('Iteration')
        plt.ylabel(f'{self.optimization_metric}')
        plt.title('Optimization Convergence')
        plt.grid(True)
        plt.show()
    
    def export_results(self, filename: str):
        """Export optimization results"""
        if not self.best_params:
            print("No optimization results to export")
            return
        
        export_data = {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'metric_optimized': self.optimization_metric,
            'total_iterations': self.iteration_count,
            'convergence_history': self.convergence_history,
            'all_results': self.all_results[:100]  # Limit to top 100
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Results exported to {filename}")