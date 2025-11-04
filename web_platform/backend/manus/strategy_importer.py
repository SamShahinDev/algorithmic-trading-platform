"""
Manus AI Strategy Importer
Imports and validates strategies from Manus AI research system
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio

@dataclass
class ManusStrategy:
    """Strategy imported from Manus AI"""
    # Metadata
    id: str
    name: str
    version: str
    created_date: datetime
    
    # Classification
    tier: int
    category: str
    timeframe: str
    instruments: List[str]
    topstepx_compatible: bool
    
    # Performance
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    # Risk parameters
    max_contracts: int
    stop_loss_points: float
    take_profit_points: float
    max_daily_trades: int
    
    # Market conditions
    best_regimes: List[str]
    worst_regimes: List[str]
    
    # Validation
    confidence_score: float
    deployment_priority: int
    
    # Logic
    entry_rules: Dict
    exit_rules: Dict

class ManusStrategyImporter:
    """
    Imports and validates strategies from Manus AI
    """
    
    def __init__(self):
        """Initialize the importer"""
        self.imported_strategies = {}
        self.validation_results = {}
        self.paper_trade_results = {}
        
        # Minimum requirements
        self.min_win_rate = 0.50
        self.min_sharpe = 1.0
        self.max_allowed_drawdown = 1000
        self.min_profit_factor = 1.3
        self.min_backtest_trades = 500
        
        print("🤖 Manus Strategy Importer initialized")
    
    async def import_strategy(self, strategy_data: Dict) -> Dict:
        """
        Import a strategy from Manus AI
        
        Args:
            strategy_data: Strategy data in Manus format
            
        Returns:
            Import result with validation status
        """
        try:
            # Parse strategy metadata
            strategy = self._parse_strategy(strategy_data)
            
            # Validate strategy
            validation = await self._validate_strategy(strategy, strategy_data)
            
            if not validation['passed']:
                return {
                    'success': False,
                    'strategy_id': strategy.id,
                    'errors': validation['errors'],
                    'warnings': validation['warnings']
                }
            
            # Check correlation with existing strategies
            correlation_check = await self._check_correlation(strategy)
            
            if not correlation_check['passed']:
                return {
                    'success': False,
                    'strategy_id': strategy.id,
                    'errors': [f"High correlation with {correlation_check['conflicting_strategy']}"],
                    'warnings': []
                }
            
            # Store strategy
            self.imported_strategies[strategy.id] = strategy
            self.validation_results[strategy.id] = validation
            
            # Schedule paper trading
            asyncio.create_task(self._paper_trade_test(strategy))
            
            return {
                'success': True,
                'strategy_id': strategy.id,
                'name': strategy.name,
                'tier': strategy.tier,
                'confidence': strategy.confidence_score,
                'validation': validation,
                'message': f"Strategy {strategy.name} imported successfully"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to import strategy"
            }
    
    def _parse_strategy(self, data: Dict) -> ManusStrategy:
        """Parse strategy data into ManusStrategy object"""
        metadata = data['strategy_metadata']
        
        return ManusStrategy(
            # Metadata
            id=metadata['id'],
            name=metadata['name'],
            version=metadata['version'],
            created_date=datetime.fromisoformat(metadata['created_date'].replace('Z', '+00:00')),
            
            # Classification
            tier=metadata['classification']['tier'],
            category=metadata['classification']['category'],
            timeframe=metadata['classification']['timeframe'],
            instruments=metadata['classification']['instruments'],
            topstepx_compatible=metadata['classification']['topstepx_compatible'],
            
            # Performance
            win_rate=metadata['performance_metrics']['win_rate'],
            sharpe_ratio=metadata['performance_metrics']['sharpe_ratio'],
            max_drawdown=metadata['performance_metrics']['max_drawdown'],
            profit_factor=metadata['performance_metrics']['profit_factor'],
            
            # Risk
            max_contracts=metadata['risk_parameters']['max_contracts'],
            stop_loss_points=metadata['risk_parameters']['stop_loss_points'],
            take_profit_points=metadata['risk_parameters']['take_profit_points'],
            max_daily_trades=metadata['risk_parameters']['max_daily_trades'],
            
            # Market conditions
            best_regimes=metadata['market_conditions']['best_regimes'],
            worst_regimes=metadata['market_conditions']['worst_regimes'],
            
            # Validation
            confidence_score=metadata['confidence_score'],
            deployment_priority=metadata['deployment_priority'],
            
            # Logic
            entry_rules=data.get('entry_rules', {}),
            exit_rules=data.get('exit_rules', {})
        )
    
    async def _validate_strategy(self, strategy: ManusStrategy, full_data: Dict) -> Dict:
        """
        Validate strategy meets requirements
        """
        errors = []
        warnings = []
        
        # Performance validation
        if strategy.win_rate < self.min_win_rate:
            errors.append(f"Win rate {strategy.win_rate:.2%} below minimum {self.min_win_rate:.2%}")
        
        if strategy.sharpe_ratio < self.min_sharpe:
            errors.append(f"Sharpe ratio {strategy.sharpe_ratio:.2f} below minimum {self.min_sharpe:.2f}")
        
        if strategy.max_drawdown > self.max_allowed_drawdown:
            errors.append(f"Max drawdown ${strategy.max_drawdown:.2f} exceeds limit ${self.max_allowed_drawdown:.2f}")
        
        if strategy.profit_factor < self.min_profit_factor:
            errors.append(f"Profit factor {strategy.profit_factor:.2f} below minimum {self.min_profit_factor:.2f}")
        
        # TopStepX compliance
        if not strategy.topstepx_compatible:
            errors.append("Strategy not marked as TopStepX compatible")
        
        if strategy.max_contracts > 1:
            warnings.append(f"Strategy uses {strategy.max_contracts} contracts (will be limited to 1)")
        
        # Risk validation
        if not strategy.stop_loss_points:
            errors.append("No stop loss defined")
        
        if strategy.max_daily_trades > 10:
            warnings.append(f"High daily trade count: {strategy.max_daily_trades}")
        
        # Backtest validation
        metadata = full_data.get('strategy_metadata', {})
        perf_metrics = metadata.get('performance_metrics', {})
        
        if perf_metrics.get('total_trades', 0) < self.min_backtest_trades:
            errors.append(f"Insufficient backtest trades: {perf_metrics.get('total_trades', 0)}")
        
        # Validation results
        validation_data = metadata.get('validation', {})
        
        if not validation_data.get('stress_test_passed', False):
            warnings.append("Strategy failed stress test")
        
        if validation_data.get('monte_carlo_confidence', 0) < 0.90:
            warnings.append(f"Low Monte Carlo confidence: {validation_data.get('monte_carlo_confidence', 0):.2%}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'validation_time': datetime.now().isoformat()
        }
    
    async def _check_correlation(self, strategy: ManusStrategy) -> Dict:
        """
        Check correlation with existing strategies
        """
        # Import strategy manager to check existing strategies
        from strategies.strategy_manager import strategy_manager
        
        for existing_id, existing in strategy_manager.strategies.items():
            # Check if strategies are in same category and tier
            if (existing.tier.value == strategy.tier and 
                strategy.category in existing.name.lower()):
                
                # Check if they trade in same market conditions
                regime_overlap = set(strategy.best_regimes) & set(existing.best_regimes)
                
                if len(regime_overlap) > 1:
                    return {
                        'passed': False,
                        'correlation': 0.8,
                        'conflicting_strategy': existing.name
                    }
        
        return {
            'passed': True,
            'correlation': 0.0,
            'conflicting_strategy': None
        }
    
    async def _paper_trade_test(self, strategy: ManusStrategy):
        """
        Run paper trading test for strategy
        """
        print(f"📝 Starting paper trade test for {strategy.name}")
        
        # Simulate 100 trades
        paper_trades = []
        wins = 0
        total_pnl = 0
        
        for i in range(100):
            # Simulate trade based on strategy parameters
            import random
            
            # Use strategy win rate for simulation
            is_win = random.random() < strategy.win_rate
            
            if is_win:
                pnl = strategy.take_profit_points * 20  # NQ point value
                wins += 1
            else:
                pnl = -strategy.stop_loss_points * 20
            
            total_pnl += pnl
            paper_trades.append({
                'trade_num': i + 1,
                'result': 'win' if is_win else 'loss',
                'pnl': pnl
            })
            
            # Small delay to simulate real trading
            await asyncio.sleep(0.1)
        
        # Calculate results
        actual_win_rate = wins / 100
        avg_pnl = total_pnl / 100
        
        self.paper_trade_results[strategy.id] = {
            'total_trades': 100,
            'wins': wins,
            'losses': 100 - wins,
            'win_rate': actual_win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'passed': actual_win_rate >= 0.45 and total_pnl > 0
        }
        
        print(f"✅ Paper trade complete for {strategy.name}: {wins}/100 wins, ${total_pnl:.2f} P&L")
    
    async def deploy_strategy(self, strategy_id: str) -> Dict:
        """
        Deploy a validated strategy to production
        """
        if strategy_id not in self.imported_strategies:
            return {
                'success': False,
                'error': 'Strategy not found'
            }
        
        strategy = self.imported_strategies[strategy_id]
        
        # Check paper trade results
        if strategy_id not in self.paper_trade_results:
            return {
                'success': False,
                'error': 'Paper trading not complete'
            }
        
        if not self.paper_trade_results[strategy_id]['passed']:
            return {
                'success': False,
                'error': 'Failed paper trading validation'
            }
        
        # Add to strategy manager
        from strategies.strategy_manager import strategy_manager, Strategy, StrategyTier, MarketRegime
        
        # Convert to internal strategy format
        tier_map = {1: StrategyTier.TIER_1, 2: StrategyTier.TIER_2, 3: StrategyTier.TIER_3}
        
        regime_map = {
            'ranging': MarketRegime.RANGING,
            'trending_up': MarketRegime.TRENDING_UP,
            'trending_down': MarketRegime.TRENDING_DOWN,
            'high_volatility': MarketRegime.HIGH_VOLATILITY,
            'low_volatility': MarketRegime.LOW_VOLATILITY
        }
        
        internal_strategy = Strategy(
            id=strategy.id,
            name=strategy.name,
            tier=tier_map.get(strategy.tier, StrategyTier.TIER_3),
            description=f"Imported from Manus AI v{strategy.version}",
            win_rate=strategy.win_rate,
            sharpe_ratio=strategy.sharpe_ratio,
            max_drawdown=strategy.max_drawdown,
            avg_trade_duration=45,  # Default
            best_regimes=[regime_map.get(r, MarketRegime.RANGING) for r in strategy.best_regimes],
            worst_regimes=[regime_map.get(r, MarketRegime.RANGING) for r in strategy.worst_regimes],
            max_positions=1,  # Force to 1 for TopStepX
            stop_loss_points=strategy.stop_loss_points,
            take_profit_points=strategy.take_profit_points,
            is_active=False,
            confidence_score=strategy.confidence_score,
            last_trade_time=None,
            daily_trades=0,
            daily_pnl=0
        )
        
        # Add to strategy manager
        strategy_manager.strategies[strategy.id] = internal_strategy
        
        return {
            'success': True,
            'strategy_id': strategy.id,
            'name': strategy.name,
            'message': f"Strategy {strategy.name} deployed successfully"
        }
    
    async def get_import_status(self) -> Dict:
        """Get status of all imported strategies"""
        return {
            'total_imported': len(self.imported_strategies),
            'strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'tier': s.tier,
                    'confidence': s.confidence_score,
                    'validation': self.validation_results.get(s.id, {}),
                    'paper_trade': self.paper_trade_results.get(s.id, {})
                }
                for s in self.imported_strategies.values()
            ]
        }

# Global importer instance
manus_importer = ManusStrategyImporter()