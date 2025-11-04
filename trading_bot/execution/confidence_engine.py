"""
Advanced Confidence Engine with Pattern Recognition
Multi-factor confidence scoring system with adaptive learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from data.feature_engineering import FeatureEngineer
from analysis.optimized_pattern_scanner import OptimizedPatternScanner as PatternScanner, PatternType
from analysis.microstructure import MicrostructureAnalyzer, MicrostructureRegime


class TradeAction(Enum):
    """Trade action types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class TradeDecision:
    """Trade decision with confidence"""
    action: TradeAction
    confidence: float  # 0-100
    size_multiplier: float  # Position size multiplier based on confidence
    entry_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str]
    pattern_signals: Dict
    risk_reward_ratio: float


@dataclass
class PatternPerformance:
    """Track pattern performance metrics"""
    pattern_type: PatternType
    occurrences: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0
    avg_pnl: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    avg_confidence: float = 0
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedConfidenceEngine:
    """Multi-factor confidence scoring with pattern recognition and adaptive learning"""
    
    def __init__(self, min_confidence: float = 60, adaptive: bool = True):
        """
        Initialize confidence engine
        
        Args:
            min_confidence: Minimum confidence threshold for trading (adaptive)
            adaptive: Whether to adapt weights based on performance
        """
        self.min_confidence = min_confidence
        self.adaptive = adaptive
        
        # Base weights (sum to 100)
        self.base_weights = {
            'pattern_quality': 30,
            'microstructure': 25,
            'technical_alignment': 20,
            'regime_alignment': 15,
            'risk_reward': 10
        }
        
        # Adaptive weights (will be adjusted based on performance)
        self.current_weights = self.base_weights.copy()
        
        # Pattern performance tracking
        self.pattern_performance = {}
        self.load_pattern_performance()
        
        # Feature engineer and analyzers
        self.feature_engineer = FeatureEngineer()
        self.pattern_scanner = PatternScanner()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        
        # Performance history for adaptive learning
        self.trade_history = []
        self.confidence_history = []
        
        # Market regime detection
        self.current_regime = None
        self.regime_history = []
        
    def calculate_confidence(self, current_data: pd.DataFrame, 
                           discovered_strategies: Optional[List] = None) -> Dict:
        """
        Sophisticated confidence calculation with multi-factor scoring
        
        Args:
            current_data: Recent OHLCV data
            discovered_strategies: Optional list of discovered profitable strategies
            
        Returns:
            Complete confidence analysis with trade decision
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Feature extraction
        features = self.feature_engineer.calculate_features(current_data)
        
        # Pattern detection
        patterns = self.pattern_scanner.scan_all_patterns(current_data, features)
        logger.debug(f"Patterns received from scanner: {list(patterns.keys()) if patterns else 'None'}")
        
        # Microstructure analysis
        micro = self.microstructure_analyzer.analyze_current_state(current_data)
        
        # Market regime identification
        regime = self._identify_market_regime(current_data, features)
        self.current_regime = regime
        
        # Calculate component scores
        scores = {}
        
        # 1. Pattern Quality Score (30%)
        pattern_score = self._score_pattern_quality(patterns)
        scores['pattern_quality'] = min(pattern_score, self.current_weights['pattern_quality'])
        logger.debug(f"Pattern quality raw score: {pattern_score:.2f}, weighted: {scores['pattern_quality']:.2f}")
        
        # 2. Microstructure Score (25%)
        micro_score = self._score_microstructure(micro)
        scores['microstructure'] = min(micro_score, self.current_weights['microstructure'])
        logger.debug(f"Microstructure raw score: {micro_score:.2f}, weighted: {scores['microstructure']:.2f}")
        
        # 3. Technical Alignment Score (20%)
        tech_score = self._score_technical_alignment(features)
        scores['technical_alignment'] = min(tech_score, self.current_weights['technical_alignment'])
        logger.debug(f"Technical raw score: {tech_score:.2f}, weighted: {scores['technical_alignment']:.2f}")
        
        # 4. Regime Alignment Score (15%)
        regime_score = self._score_regime_alignment(patterns, regime, features)
        scores['regime_alignment'] = min(regime_score, self.current_weights['regime_alignment'])
        logger.debug(f"Regime raw score: {regime_score:.2f}, weighted: {scores['regime_alignment']:.2f}")
        
        # 5. Risk/Reward Score (10%)
        rr_score = self._calculate_risk_reward_score(current_data, patterns)
        scores['risk_reward'] = min(rr_score, self.current_weights['risk_reward'])
        logger.debug(f"Risk/Reward raw score: {rr_score:.2f}, weighted: {scores['risk_reward']:.2f}")
        
        logger.debug(f"All component scores: {scores}")
        logger.debug(f"Current weights being used: {self.current_weights}")
        
        # Calculate total confidence
        total_confidence = sum(scores.values())
        
        # Apply discovered strategy boost if available
        if discovered_strategies:
            strategy_boost = self._apply_strategy_boost(features, discovered_strategies)
            total_confidence = min(100, total_confidence + strategy_boost)
        
        # Make trade decision
        logger.debug(f"About to call _make_trade_decision with confidence={total_confidence:.1f}%, patterns={patterns}")
        trade_decision = self._make_trade_decision(total_confidence, patterns, current_data)
        logger.debug(f"_make_trade_decision returned action: {trade_decision.action.value}")
        
        # Store confidence for adaptive learning
        self.confidence_history.append({
            'timestamp': datetime.now(),
            'confidence': total_confidence,
            'components': scores.copy(),
            'decision': trade_decision.action,
            'patterns': list(patterns.keys()) if patterns else []
        })
        
        # REMOVED: Auto-adjustment of confidence thresholds
        # Thresholds should only be changed by explicit user configuration
        # if self.adaptive:
        #     self._adjust_confidence_threshold()
        
        return {
            'confidence': total_confidence,
            'components': scores,
            'patterns': patterns,
            'microstructure': micro,
            'regime': regime,
            'trade_decision': trade_decision,
            'threshold': self.min_confidence,
            'pattern_performance': self._get_relevant_pattern_performance(patterns)
        }
    
    def _score_pattern_quality(self, patterns: Dict) -> float:
        """
        Score pattern quality based on historical performance
        
        Args:
            patterns: Detected patterns
            
        Returns:
            Pattern quality score (0-30)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not patterns:
            logger.debug("No patterns detected, pattern score = 0")
            return 0
        
        total_score = 0
        pattern_count = 0
        
        for pattern_type, pattern_signal in patterns.items():
            logger.debug(f"Processing pattern: {pattern_type}, signal: {pattern_signal}")
            
            # Get historical performance for this pattern
            perf = self.pattern_performance.get(pattern_type.value)
            
            if perf:
                # Score based on win rate and profit factor
                win_rate_score = perf.win_rate * 10  # Max 10 points
                profit_factor_score = min(10, perf.profit_factor * 5)  # Max 10 points
                
                # Weight by pattern strength
                pattern_weight = pattern_signal.strength / 100
                
                pattern_score = (win_rate_score + profit_factor_score) * pattern_weight
                logger.debug(f"Pattern {pattern_type} has history: win_rate_score={win_rate_score:.2f}, profit_factor_score={profit_factor_score:.2f}")
            else:
                # No history, use pattern strength only
                pattern_score = pattern_signal.strength * 0.2
                logger.debug(f"Pattern {pattern_type} no history, using strength: {pattern_signal.strength:.2f}")
            
            # Weight by pattern confidence
            pattern_score *= (pattern_signal.confidence / 100)
            logger.debug(f"Pattern {pattern_type} weighted score: {pattern_score:.2f}")
            
            total_score += pattern_score
            pattern_count += 1
        
        # Average score across all patterns, then scale to max 30
        if pattern_count > 0:
            avg_score = total_score / pattern_count
            final_score = min(30, avg_score * 1.5)  # Scale up slightly
            logger.debug(f"Pattern quality final: total={total_score:.2f}, avg={avg_score:.2f}, final={final_score:.2f}")
            return final_score
        
        return 0
    
    def _score_microstructure(self, micro: Dict) -> float:
        """
        Score microstructure conditions
        
        Args:
            micro: Microstructure analysis
            
        Returns:
            Microstructure score (0-25)
        """
        score = 0
        
        # Order flow score (max 10 points)
        order_flow = micro.get('order_flow')
        if order_flow:
            # Strong directional pressure is good
            net_pressure = abs(order_flow.net_pressure)
            pressure_score = min(10, net_pressure / 10)
            
            # Absorption and exhaustion are trading opportunities
            if order_flow.absorption:
                pressure_score += 2
            if order_flow.exhaustion:
                pressure_score += 2
            
            score += min(10, pressure_score)
        
        # Volume profile score (max 8 points)
        volume_profile = micro.get('volume_profile')
        if volume_profile:
            # Trading near POC is good for mean reversion
            current_price = micro.get('current_price', 0)
            poc = volume_profile.point_of_control
            
            if poc > 0:
                distance_from_poc = abs(current_price - poc) / poc
                if distance_from_poc < 0.005:  # Within 0.5% of POC
                    score += 4
                elif distance_from_poc > 0.02:  # Far from POC (opportunity)
                    score += 6
            
            # Volume delta shows direction
            if abs(volume_profile.delta) > volume_profile.total_volume * 0.2:
                score += 2
        
        # Liquidity score (max 7 points)
        liquidity = micro.get('liquidity', {})
        liquidity_score = liquidity.get('liquidity_score', 50) / 100 * 7
        score += liquidity_score
        
        return min(25, score)
    
    def _score_technical_alignment(self, features: pd.DataFrame) -> float:
        """
        Score technical indicator alignment
        
        Args:
            features: Technical features
            
        Returns:
            Technical alignment score (0-20)
        """
        score = 0
        
        # Get latest values
        if len(features) == 0:
            return 0
        
        latest = features.iloc[-1]
        
        # Trend alignment (max 8 points)
        trend_signals = 0
        trend_checks = 0
        
        # Moving average alignment
        for period in [20, 50]:
            ma_key = f'ma_{period}'
            if ma_key in latest:
                trend_checks += 1
                close_price = latest.get('close', 0)
                if close_price > latest[ma_key]:
                    trend_signals += 1
        
        if trend_checks > 0:
            trend_score = (trend_signals / trend_checks) * 8
            score += trend_score
        
        # Momentum alignment (max 6 points)
        momentum_signals = 0
        momentum_checks = 0
        
        # RSI signals
        for period in [14, 20]:
            rsi_key = f'rsi_{period}'
            if rsi_key in latest:
                momentum_checks += 1
                rsi_value = latest[rsi_key]
                if 40 < rsi_value < 60:  # Neutral zone
                    momentum_signals += 0.5
                elif rsi_value > 60 or rsi_value < 40:  # Trending
                    momentum_signals += 1
        
        if momentum_checks > 0:
            momentum_score = (momentum_signals / momentum_checks) * 6
            score += momentum_score
        
        # Volatility conditions (max 6 points)
        if 'realized_volatility' in latest:
            vol = latest['realized_volatility']
            # Moderate volatility is best for trading
            if 0.15 < vol < 0.35:
                score += 6
            elif 0.10 < vol < 0.40:
                score += 3
        
        return min(20, score)
    
    def _score_regime_alignment(self, patterns: Dict, regime: str, features: pd.DataFrame) -> float:
        """
        Score pattern-regime alignment
        
        Args:
            patterns: Detected patterns
            regime: Current market regime
            features: Technical features
            
        Returns:
            Regime alignment score (0-15)
        """
        score = 0
        
        if not patterns:
            return 0
        
        # Check if patterns align with regime
        for pattern_type, pattern_signal in patterns.items():
            aligned = False
            
            # Trending regime alignments
            if regime == 'trending_up':
                if pattern_type in [PatternType.MOMENTUM_BURST, PatternType.BREAKOUT, 
                                   PatternType.TREND_CONTINUATION] and pattern_signal.direction == 1:
                    aligned = True
            
            elif regime == 'trending_down':
                if pattern_type in [PatternType.MOMENTUM_BURST, PatternType.BREAKOUT,
                                   PatternType.TREND_CONTINUATION] and pattern_signal.direction == -1:
                    aligned = True
            
            # Ranging regime alignments
            elif regime == 'ranging':
                if pattern_type in [PatternType.MEAN_REVERSION, PatternType.FADE_EXTREME]:
                    aligned = True
            
            # Volatile regime alignments
            elif regime == 'volatile':
                if pattern_type in [PatternType.VOLUME_CLIMAX, PatternType.REVERSAL]:
                    aligned = True
            
            if aligned:
                score += 7.5  # Half of max score per aligned pattern
        
        # Cap at maximum
        return min(15, score)
    
    def _calculate_risk_reward_score(self, df: pd.DataFrame, patterns: Dict) -> float:
        """
        Calculate risk/reward score
        
        Args:
            df: OHLCV data
            patterns: Detected patterns
            
        Returns:
            Risk/reward score (0-10)
        """
        if not patterns:
            return 0
        
        best_rr_ratio = 0
        
        for pattern_type, pattern_signal in patterns.items():
            # Calculate risk/reward ratio
            entry = pattern_signal.entry_price
            stop = pattern_signal.stop_loss
            target = pattern_signal.take_profit
            
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            if risk > 0:
                rr_ratio = reward / risk
                best_rr_ratio = max(best_rr_ratio, rr_ratio)
        
        # Score based on R:R ratio
        if best_rr_ratio >= 3:
            return 10
        elif best_rr_ratio >= 2.5:
            return 8
        elif best_rr_ratio >= 2:
            return 6
        elif best_rr_ratio >= 1.5:
            return 4
        elif best_rr_ratio >= 1:
            return 2
        else:
            return 0
    
    def _identify_market_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> str:
        """
        Identify current market regime
        
        Args:
            df: OHLCV data
            features: Technical features
            
        Returns:
            Market regime string
        """
        if len(df) < 50:
            return 'unknown'
        
        # Get regime from features if available
        if 'trend_regime' in features.columns:
            trend_regime = features['trend_regime'].iloc[-1]
            
            if trend_regime == 1:
                base_regime = 'trending_up'
            elif trend_regime == -1:
                base_regime = 'trending_down'
            else:
                base_regime = 'ranging'
        else:
            # Manual calculation
            ma_20 = df['close'].rolling(20).mean().iloc[-1]
            ma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            if ma_20 > ma_50 * 1.02:
                base_regime = 'trending_up'
            elif ma_20 < ma_50 * 0.98:
                base_regime = 'trending_down'
            else:
                base_regime = 'ranging'
        
        # Check volatility overlay
        if 'volatility_regime' in features.columns:
            vol_regime = features['volatility_regime'].iloc[-1]
            if vol_regime == 2:  # High volatility
                if base_regime == 'ranging':
                    return 'volatile'
                else:
                    return f'{base_regime}_volatile'
        
        return base_regime
    
    def _make_trade_decision(self, confidence: float, patterns: Dict, df: pd.DataFrame) -> TradeDecision:
        """
        Make trade decision based on confidence and patterns
        
        Args:
            confidence: Total confidence score
            patterns: Detected patterns
            df: OHLCV data
            
        Returns:
            Trade decision object
        """
        import logging
        import numpy as np
        logger = logging.getLogger(__name__)
        
        logger.debug(f"=== _make_trade_decision called ===")
        logger.debug(f"Confidence: {confidence:.1f}%, Min threshold: {self.min_confidence}%")
        logger.debug(f"Patterns received: {patterns}")
        
        # Default decision is hold
        if confidence < self.min_confidence:
            logger.debug(f"Confidence {confidence:.1f}% below threshold {self.min_confidence}%, returning HOLD")
            return TradeDecision(
                action=TradeAction.HOLD,
                confidence=confidence,
                size_multiplier=0,
                entry_price=df['close'].iloc[-1],
                stop_loss=0,
                take_profit=0,
                reasons=['Confidence below threshold'],
                pattern_signals=patterns,
                risk_reward_ratio=0
            )
        
        # Determine direction from patterns or technical indicators
        if not patterns:
            logger.debug("No patterns detected, checking technical indicators for direction")
            
            # If confidence is high enough without patterns, use technical indicators
            if confidence >= self.min_confidence:
                # Calculate technical direction from price action
                close_prices = df['close'].values[-20:]
                sma_fast = np.mean(close_prices[-5:])
                sma_slow = np.mean(close_prices[-20:])
                
                # Simple momentum check
                recent_momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5] * 100
                
                logger.debug(f"Technical analysis: Fast SMA={sma_fast:.2f}, Slow SMA={sma_slow:.2f}, Momentum={recent_momentum:.2f}%")
                
                # Determine direction based on technical indicators
                if sma_fast > sma_slow and recent_momentum > 0.1:
                    action = TradeAction.BUY
                    direction = 1
                    logger.debug("Technical indicators suggest BUY")
                elif sma_fast < sma_slow and recent_momentum < -0.1:
                    action = TradeAction.SELL  
                    direction = -1
                    logger.debug("Technical indicators suggest SELL")
                else:
                    logger.debug("Technical indicators are neutral, returning HOLD")
                    return TradeDecision(
                        action=TradeAction.HOLD,
                        confidence=confidence,
                        size_multiplier=0,
                        entry_price=df['close'].iloc[-1],
                        stop_loss=0,
                        take_profit=0,
                        reasons=['Technical indicators neutral'],
                        pattern_signals={},
                        risk_reward_ratio=0
                    )
                
                # Create synthetic pattern signal for technical-based trades
                current_price = df['close'].iloc[-1]
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.002
                
                # Calculate stops and targets
                stop_distance = atr * 1.5
                target_distance = atr * 2.5
                
                if action == TradeAction.BUY:
                    stop_loss = current_price - stop_distance
                    take_profit = current_price + target_distance
                else:
                    stop_loss = current_price + stop_distance
                    take_profit = current_price - target_distance
                
                # Calculate position size based on confidence
                if confidence >= 40:
                    size_multiplier = 1.0
                elif confidence >= 30:
                    size_multiplier = 0.8
                else:
                    size_multiplier = 0.6
                
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
                
                reasons = [
                    f"Technical-based trade (no patterns)",
                    f"Confidence: {confidence:.1f}%",
                    f"Direction: {'Bullish' if action == TradeAction.BUY else 'Bearish'}",
                    f"Risk/Reward: {rr_ratio:.2f}"
                ]
                
                return TradeDecision(
                    action=action,
                    confidence=confidence,
                    size_multiplier=size_multiplier,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasons=reasons,
                    pattern_signals={},
                    risk_reward_ratio=rr_ratio
                )
            else:
                logger.debug("No patterns and confidence below threshold, returning HOLD")
                return TradeDecision(
                    action=TradeAction.HOLD,
                    confidence=confidence,
                    size_multiplier=0,
                    entry_price=df['close'].iloc[-1],
                    stop_loss=0,
                    take_profit=0,
                    reasons=['No patterns detected, confidence below threshold'],
                    pattern_signals={},
                    risk_reward_ratio=0
                )
        
        # Get strongest pattern
        strongest_pattern = max(patterns.items(), key=lambda x: x[1].strength)
        pattern_type, pattern_signal = strongest_pattern
        
        logger.debug(f"Strongest pattern: {pattern_type}")
        logger.debug(f"Pattern signal details: {pattern_signal}")
        logger.debug(f"Pattern direction: {pattern_signal.direction}")
        logger.debug(f"Pattern strength: {pattern_signal.strength}")
        
        # Determine action
        if pattern_signal.direction == 1:
            action = TradeAction.BUY
            logger.debug("Pattern direction is 1, action = BUY")
        elif pattern_signal.direction == -1:
            action = TradeAction.SELL
            logger.debug("Pattern direction is -1, action = SELL")
        else:
            action = TradeAction.HOLD
            logger.debug(f"Pattern direction is {pattern_signal.direction}, action = HOLD")
        
        # Calculate position size multiplier based on confidence
        if confidence >= 90:
            size_multiplier = 1.5
        elif confidence >= 80:
            size_multiplier = 1.2
        elif confidence >= 70:
            size_multiplier = 1.0
        else:
            size_multiplier = 0.8
        
        # Calculate risk/reward
        risk = abs(pattern_signal.entry_price - pattern_signal.stop_loss)
        reward = abs(pattern_signal.take_profit - pattern_signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Build reasons
        reasons = [
            f"Primary pattern: {pattern_type.value}",
            f"Pattern strength: {pattern_signal.strength:.1f}",
            f"Confidence: {confidence:.1f}",
            f"Risk/Reward: {rr_ratio:.2f}"
        ]
        
        return TradeDecision(
            action=action,
            confidence=confidence,
            size_multiplier=size_multiplier,
            entry_price=pattern_signal.entry_price,
            stop_loss=pattern_signal.stop_loss,
            take_profit=pattern_signal.take_profit,
            reasons=reasons,
            pattern_signals={k.value: v for k, v in patterns.items()},
            risk_reward_ratio=rr_ratio
        )
    
    def _apply_strategy_boost(self, features: pd.DataFrame, strategies: List) -> float:
        """
        Apply confidence boost from discovered strategies
        
        Args:
            features: Technical features
            strategies: List of discovered strategies
            
        Returns:
            Confidence boost (0-10)
        """
        boost = 0
        
        for strategy in strategies[:3]:  # Consider top 3 strategies
            if strategy.get('triggers_met'):
                strategy_confidence = strategy.get('confidence', 0)
                strategy_performance = strategy.get('sharpe', 1.0)
                
                # Boost based on strategy quality
                boost += min(5, strategy_confidence * strategy_performance * 0.05)
        
        return min(10, boost)
    
    def _adjust_confidence_threshold(self):
        """
        DISABLED: Confidence thresholds should NEVER be auto-adjusted.
        
        This method has been intentionally disabled to prevent autonomous
        threshold adjustments. All confidence thresholds should only be
        modified through explicit user configuration.
        
        Auto-adjustment can lead to unexpected behavior where the bot
        changes its own trading parameters without user knowledge.
        """
        pass  # Method disabled - no auto-adjustment allowed
    
    def update_pattern_performance(self, pattern_type: PatternType, pnl: float, won: bool):
        """
        Update pattern performance metrics
        
        Args:
            pattern_type: Type of pattern
            pnl: Profit/loss from trade
            won: Whether trade was profitable
        """
        key = pattern_type.value
        
        if key not in self.pattern_performance:
            self.pattern_performance[key] = PatternPerformance(pattern_type)
        
        perf = self.pattern_performance[key]
        perf.occurrences += 1
        
        if won:
            perf.wins += 1
        else:
            perf.losses += 1
        
        perf.total_pnl += pnl
        perf.avg_pnl = perf.total_pnl / perf.occurrences
        perf.win_rate = perf.wins / perf.occurrences if perf.occurrences > 0 else 0
        
        # Calculate profit factor
        if perf.losses > 0:
            total_wins = sum(1 for t in self.trade_history 
                           if t.get('pattern') == key and t['pnl'] > 0)
            total_losses = sum(1 for t in self.trade_history 
                             if t.get('pattern') == key and t['pnl'] <= 0)
            
            if total_losses > 0:
                avg_win_pnl = sum(t['pnl'] for t in self.trade_history 
                                if t.get('pattern') == key and t['pnl'] > 0) / (total_wins or 1)
                avg_loss_pnl = abs(sum(t['pnl'] for t in self.trade_history 
                                     if t.get('pattern') == key and t['pnl'] <= 0) / total_losses)
                
                perf.profit_factor = avg_win_pnl / avg_loss_pnl if avg_loss_pnl > 0 else 0
        
        perf.last_updated = datetime.now()
        
        # Save to file
        self.save_pattern_performance()
    
    def record_trade(self, entry_price: float, exit_price: float, direction: int, 
                    confidence: float, pattern: Optional[PatternType] = None):
        """
        Record trade result for learning
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: 1 for long, -1 for short
            confidence: Confidence at entry
            pattern: Pattern that triggered trade
        """
        pnl = (exit_price - entry_price) * direction
        pnl_percent = (pnl / entry_price) * 100
        won = pnl > 0
        
        trade_record = {
            'timestamp': datetime.now(),
            'entry': entry_price,
            'exit': exit_price,
            'direction': direction,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'won': won,
            'confidence': confidence,
            'pattern': pattern.value if pattern else None,
            'regime': self.current_regime
        }
        
        self.trade_history.append(trade_record)
        
        # Update pattern performance if applicable
        if pattern:
            self.update_pattern_performance(pattern, pnl_percent, won)
        
        # Update adaptive weights if enabled
        if self.adaptive:
            self._update_weights_based_on_performance()
    
    def _update_weights_based_on_performance(self):
        """
        Update component weights based on recent performance
        """
        if len(self.confidence_history) < 50:
            return  # Not enough data
        
        # Analyze which components correlate with winning trades
        recent_confidence = self.confidence_history[-50:]
        recent_trades = self.trade_history[-50:]
        
        if len(recent_trades) < 20:
            return
        
        # Calculate correlation between each component and trade success
        component_performance = {}
        
        for component in self.base_weights.keys():
            component_values = [c['components'].get(component, 0) for c in recent_confidence]
            trade_results = [1 if t['won'] else 0 for t in recent_trades[:len(component_values)]]
            
            if len(component_values) == len(trade_results):
                correlation = np.corrcoef(component_values, trade_results)[0, 1]
                component_performance[component] = correlation
        
        # Adjust weights based on correlation
        total_positive_corr = sum(max(0, corr) for corr in component_performance.values())
        
        if total_positive_corr > 0:
            for component, correlation in component_performance.items():
                if correlation > 0:
                    # Increase weight for positive correlation
                    weight_adjustment = (correlation / total_positive_corr) * 10
                    self.current_weights[component] = min(40, 
                        self.base_weights[component] + weight_adjustment)
                else:
                    # Decrease weight for negative correlation
                    self.current_weights[component] = max(5,
                        self.base_weights[component] * 0.9)
        
        # Normalize weights to sum to 100
        total_weight = sum(self.current_weights.values())
        for component in self.current_weights:
            self.current_weights[component] = (self.current_weights[component] / total_weight) * 100
    
    def save_pattern_performance(self):
        """Save pattern performance to file"""
        performance_file = '/Users/royaltyvixion/Documents/XTRADING/trading_bot/pattern_performance.json'
        
        data = {}
        for key, perf in self.pattern_performance.items():
            data[key] = {
                'occurrences': perf.occurrences,
                'wins': perf.wins,
                'losses': perf.losses,
                'total_pnl': perf.total_pnl,
                'avg_pnl': perf.avg_pnl,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'last_updated': perf.last_updated.isoformat()
            }
        
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)
        with open(performance_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_pattern_performance(self):
        """Load pattern performance from file"""
        performance_file = '/Users/royaltyvixion/Documents/XTRADING/trading_bot/pattern_performance.json'
        
        if os.path.exists(performance_file):
            with open(performance_file, 'r') as f:
                data = json.load(f)
                
            for key, values in data.items():
                # Find matching PatternType
                for pattern_type in PatternType:
                    if pattern_type.value == key:
                        perf = PatternPerformance(pattern_type)
                        perf.occurrences = values['occurrences']
                        perf.wins = values['wins']
                        perf.losses = values['losses']
                        perf.total_pnl = values['total_pnl']
                        perf.avg_pnl = values['avg_pnl']
                        perf.win_rate = values['win_rate']
                        perf.profit_factor = values['profit_factor']
                        perf.last_updated = datetime.fromisoformat(values['last_updated'])
                        
                        self.pattern_performance[key] = perf
                        break
    
    def _get_relevant_pattern_performance(self, patterns: Dict) -> Dict:
        """
        Get performance metrics for detected patterns
        
        Args:
            patterns: Currently detected patterns
            
        Returns:
            Performance metrics for relevant patterns
        """
        relevant_performance = {}
        
        for pattern_type in patterns.keys():
            key = pattern_type.value
            if key in self.pattern_performance:
                perf = self.pattern_performance[key]
                relevant_performance[key] = {
                    'win_rate': perf.win_rate,
                    'avg_pnl': perf.avg_pnl,
                    'profit_factor': perf.profit_factor,
                    'occurrences': perf.occurrences
                }
        
        return relevant_performance
    
    def get_performance_summary(self) -> Dict:
        """
        Get overall performance summary
        
        Returns:
            Performance metrics dictionary
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'expectancy': 0,
                'profit_factor': 0,
                'avg_confidence': 0,
                'current_threshold': self.min_confidence
            }
        
        total_trades = len(self.trade_history)
        wins = [t for t in self.trade_history if t['won']]
        losses = [t for t in self.trade_history if not t['won']]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t['pnl_percent'] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t['pnl_percent'] for t in losses) / len(losses)) if losses else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        profit_factor = (sum(t['pnl'] for t in wins) / 
                        abs(sum(t['pnl'] for t in losses))) if losses and sum(t['pnl'] for t in losses) != 0 else 0
        
        avg_confidence = sum(t['confidence'] for t in self.trade_history) / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'avg_confidence': avg_confidence,
            'current_threshold': self.min_confidence,
            'current_weights': self.current_weights,
            'pattern_performance': self.pattern_performance
        }