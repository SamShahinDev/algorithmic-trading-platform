"""
Strategy Priority Matrix and Management System
Implements tiered strategy selection based on market conditions
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import asyncio

class StrategyTier(Enum):
    """Strategy priority tiers"""
    TIER_1 = 1  # Highest priority - Mean reversion, momentum breakouts
    TIER_2 = 2  # Medium priority - Microstructure, order flow
    TIER_3 = 3  # Lowest priority - Pairs trading, stat arb

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class Strategy:
    """Individual strategy configuration"""
    id: str
    name: str
    tier: StrategyTier
    description: str
    
    # Performance metrics
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: int  # minutes
    
    # Market conditions
    best_regimes: List[MarketRegime]
    worst_regimes: List[MarketRegime]
    
    # Risk parameters
    max_positions: int
    stop_loss_points: float
    take_profit_points: float
    
    # Status
    is_active: bool
    confidence_score: float
    last_trade_time: Optional[datetime]
    daily_trades: int
    daily_pnl: float

class StrategyManager:
    """
    Manages strategy prioritization and selection
    Based on market conditions and performance
    """
    
    def __init__(self):
        """Initialize strategy manager"""
        self.strategies = self._initialize_strategies()
        self.current_regime = MarketRegime.RANGING
        self.active_strategies = []
        self.performance_history = {}
        
        print("📊 Strategy Manager initialized with priority matrix")
    
    def _initialize_strategies(self) -> Dict[str, Strategy]:
        """Initialize available strategies with their configurations"""
        strategies = {
            # Tier 1 Strategies - Highest Priority
            "mean_reversion_sr": Strategy(
                id="mean_reversion_sr",
                name="Mean Reversion (Support/Resistance)",
                tier=StrategyTier.TIER_1,
                description="Trade reversals at key S/R levels",
                win_rate=0.65,
                sharpe_ratio=1.8,
                max_drawdown=300,
                avg_trade_duration=30,
                best_regimes=[MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY],
                worst_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
                max_positions=1,
                stop_loss_points=5,
                take_profit_points=5,
                is_active=False,
                confidence_score=0.85,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            ),
            
            "momentum_breakout": Strategy(
                id="momentum_breakout",
                name="Momentum Breakout",
                tier=StrategyTier.TIER_1,
                description="Trade breakouts with strong momentum",
                win_rate=0.58,
                sharpe_ratio=1.6,
                max_drawdown=400,
                avg_trade_duration=45,
                best_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
                worst_regimes=[MarketRegime.RANGING],
                max_positions=1,
                stop_loss_points=6,
                take_profit_points=8,
                is_active=False,
                confidence_score=0.80,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            ),
            
            "engulfing_reversal": Strategy(
                id="engulfing_reversal",
                name="Engulfing Pattern Reversal",
                tier=StrategyTier.TIER_1,
                description="Trade reversals on engulfing patterns",
                win_rate=0.62,
                sharpe_ratio=1.7,
                max_drawdown=350,
                avg_trade_duration=35,
                best_regimes=[MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY],
                worst_regimes=[MarketRegime.LOW_VOLATILITY],
                max_positions=1,
                stop_loss_points=5,
                take_profit_points=6,
                is_active=False,
                confidence_score=0.75,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            ),
            
            # Tier 2 Strategies - Medium Priority
            "microstructure_flow": Strategy(
                id="microstructure_flow",
                name="Microstructure Order Flow",
                tier=StrategyTier.TIER_2,
                description="Trade based on order flow imbalances",
                win_rate=0.55,
                sharpe_ratio=1.4,
                max_drawdown=450,
                avg_trade_duration=15,
                best_regimes=[MarketRegime.HIGH_VOLATILITY],
                worst_regimes=[MarketRegime.LOW_VOLATILITY],
                max_positions=1,
                stop_loss_points=4,
                take_profit_points=4,
                is_active=False,
                confidence_score=0.70,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            ),
            
            "volume_profile": Strategy(
                id="volume_profile",
                name="Volume Profile Trading",
                tier=StrategyTier.TIER_2,
                description="Trade around high volume nodes",
                win_rate=0.57,
                sharpe_ratio=1.5,
                max_drawdown=400,
                avg_trade_duration=40,
                best_regimes=[MarketRegime.RANGING],
                worst_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
                max_positions=1,
                stop_loss_points=5,
                take_profit_points=5,
                is_active=False,
                confidence_score=0.72,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            ),
            
            # Tier 3 Strategies - Lowest Priority
            "pairs_spread": Strategy(
                id="pairs_spread",
                name="Pairs Spread Trading",
                tier=StrategyTier.TIER_3,
                description="Trade NQ vs ES spread divergences",
                win_rate=0.52,
                sharpe_ratio=1.2,
                max_drawdown=500,
                avg_trade_duration=60,
                best_regimes=[MarketRegime.LOW_VOLATILITY],
                worst_regimes=[MarketRegime.HIGH_VOLATILITY],
                max_positions=2,
                stop_loss_points=7,
                take_profit_points=7,
                is_active=False,
                confidence_score=0.65,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            ),
            
            "statistical_arbitrage": Strategy(
                id="statistical_arbitrage",
                name="Statistical Arbitrage",
                tier=StrategyTier.TIER_3,
                description="Trade statistical mispricings",
                win_rate=0.51,
                sharpe_ratio=1.1,
                max_drawdown=600,
                avg_trade_duration=90,
                best_regimes=[MarketRegime.RANGING],
                worst_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
                max_positions=1,
                stop_loss_points=8,
                take_profit_points=8,
                is_active=False,
                confidence_score=0.60,
                last_trade_time=None,
                daily_trades=0,
                daily_pnl=0
            )
        }
        
        return strategies
    
    async def detect_market_regime(self, price_data: Optional[List[float]] = None) -> MarketRegime:
        """
        Detect current market regime based on price action
        """
        if not price_data:
            # Simplified detection for now
            # In production, would use actual price data
            import random
            self.current_regime = random.choice(list(MarketRegime))
            return self.current_regime
        
        # Calculate regime indicators
        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.std(returns)
        trend = np.polyfit(range(len(price_data)), price_data, 1)[0]
        
        # Classify regime
        if volatility > 0.02:
            self.current_regime = MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.005:
            self.current_regime = MarketRegime.LOW_VOLATILITY
        elif abs(trend) < 0.1:
            self.current_regime = MarketRegime.RANGING
        elif trend > 0.1:
            self.current_regime = MarketRegime.TRENDING_UP
        else:
            self.current_regime = MarketRegime.TRENDING_DOWN
        
        return self.current_regime
    
    async def select_strategies(self, max_active: int = 3) -> List[Strategy]:
        """
        Select best strategies based on current market regime and priority
        """
        # Detect current market regime
        regime = await self.detect_market_regime()
        
        # Score all strategies
        strategy_scores = []
        
        for strategy_id, strategy in self.strategies.items():
            score = self._calculate_strategy_score(strategy, regime)
            strategy_scores.append((strategy, score))
        
        # Sort by score (descending) and tier (ascending)
        strategy_scores.sort(key=lambda x: (-x[1], x[0].tier.value))
        
        # Select top strategies
        selected = []
        for strategy, score in strategy_scores[:max_active]:
            if score > 0.5:  # Minimum score threshold
                strategy.is_active = True
                selected.append(strategy)
                print(f"✅ Activated: {strategy.name} (Tier {strategy.tier.value}, Score: {score:.2f})")
        
        self.active_strategies = selected
        return selected
    
    def _calculate_strategy_score(self, strategy: Strategy, regime: MarketRegime) -> float:
        """
        Calculate strategy score based on multiple factors
        """
        score = 0.0
        
        # Tier weighting (40%)
        tier_weights = {
            StrategyTier.TIER_1: 1.0,
            StrategyTier.TIER_2: 0.7,
            StrategyTier.TIER_3: 0.4
        }
        score += tier_weights[strategy.tier] * 0.4
        
        # Regime compatibility (30%)
        if regime in strategy.best_regimes:
            score += 0.3
        elif regime in strategy.worst_regimes:
            score -= 0.2
        else:
            score += 0.1
        
        # Performance metrics (20%)
        performance_score = (
            strategy.win_rate * 0.3 +
            min(strategy.sharpe_ratio / 2, 1.0) * 0.4 +
            (1 - min(strategy.max_drawdown / 1000, 1.0)) * 0.3
        )
        score += performance_score * 0.2
        
        # Confidence score (10%)
        score += strategy.confidence_score * 0.1
        
        # Recent performance adjustment
        if strategy.daily_pnl < -200:
            score *= 0.7  # Reduce score if losing today
        elif strategy.daily_pnl > 200:
            score *= 1.1  # Boost score if winning today
        
        return min(max(score, 0), 1)  # Clamp between 0 and 1
    
    async def update_strategy_performance(self, strategy_id: str, pnl: float, 
                                         trade_duration: int):
        """
        Update strategy performance metrics
        """
        if strategy_id not in self.strategies:
            return
        
        strategy = self.strategies[strategy_id]
        strategy.daily_pnl += pnl
        strategy.daily_trades += 1
        strategy.last_trade_time = datetime.now()
        
        # Update confidence based on performance
        if pnl > 0:
            strategy.confidence_score = min(strategy.confidence_score * 1.05, 1.0)
        else:
            strategy.confidence_score = max(strategy.confidence_score * 0.95, 0.3)
        
        # Store in history
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []
        
        self.performance_history[strategy_id].append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'duration': trade_duration,
            'regime': self.current_regime
        })
    
    async def should_rotate_strategies(self) -> bool:
        """
        Check if strategies should be rotated based on performance or regime change
        """
        # Check if regime has changed significantly
        previous_regime = self.current_regime
        current_regime = await self.detect_market_regime()
        
        if previous_regime != current_regime:
            print(f"🔄 Market regime changed: {previous_regime} → {current_regime}")
            return True
        
        # Check if active strategies are underperforming
        if self.active_strategies:
            avg_confidence = np.mean([s.confidence_score for s in self.active_strategies])
            if avg_confidence < 0.5:
                print("📉 Low confidence in active strategies")
                return True
        
        # Rotate every hour regardless
        if self.active_strategies:
            oldest_activation = min([s.last_trade_time or datetime.now() 
                                   for s in self.active_strategies])
            if datetime.now() - oldest_activation > timedelta(hours=1):
                print("⏰ Hourly strategy rotation")
                return True
        
        return False
    
    async def deactivate_all_strategies(self):
        """Deactivate all strategies"""
        for strategy in self.strategies.values():
            strategy.is_active = False
        self.active_strategies = []
        print("🛑 All strategies deactivated")
    
    async def reset_daily_metrics(self):
        """Reset daily metrics for all strategies"""
        for strategy in self.strategies.values():
            strategy.daily_trades = 0
            strategy.daily_pnl = 0
        print("📅 Strategy daily metrics reset")
    
    async def get_strategy_report(self) -> Dict:
        """Generate comprehensive strategy report"""
        return {
            'current_regime': self.current_regime.value,
            'active_strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'tier': s.tier.value,
                    'confidence': s.confidence_score,
                    'daily_pnl': s.daily_pnl,
                    'daily_trades': s.daily_trades
                }
                for s in self.active_strategies
            ],
            'all_strategies': {
                s_id: {
                    'name': s.name,
                    'tier': s.tier.value,
                    'win_rate': s.win_rate,
                    'sharpe_ratio': s.sharpe_ratio,
                    'confidence': s.confidence_score,
                    'is_active': s.is_active,
                    'daily_pnl': s.daily_pnl
                }
                for s_id, s in self.strategies.items()
            },
            'performance_summary': {
                'total_daily_pnl': sum(s.daily_pnl for s in self.strategies.values()),
                'total_daily_trades': sum(s.daily_trades for s in self.strategies.values()),
                'best_strategy': max(self.strategies.values(), 
                                    key=lambda s: s.daily_pnl).name if self.strategies else None,
                'worst_strategy': min(self.strategies.values(), 
                                     key=lambda s: s.daily_pnl).name if self.strategies else None
            }
        }
    
    def get_strategy_by_pattern(self, pattern_name: str) -> Optional[Strategy]:
        """
        Map pattern name to appropriate strategy
        """
        pattern_strategy_map = {
            'engulfing': 'engulfing_reversal',
            'double_top': 'mean_reversion_sr',
            'double_bottom': 'mean_reversion_sr',
            'breakout': 'momentum_breakout',
            'support_bounce': 'mean_reversion_sr',
            'resistance_rejection': 'mean_reversion_sr',
            'momentum': 'momentum_breakout',
            'volume_spike': 'volume_profile',
            'order_flow': 'microstructure_flow'
        }
        
        # Find matching strategy
        for pattern_key, strategy_id in pattern_strategy_map.items():
            if pattern_key in pattern_name.lower():
                return self.strategies.get(strategy_id)
        
        # Default to mean reversion if no match
        return self.strategies.get('mean_reversion_sr')

# Global strategy manager instance
strategy_manager = StrategyManager()