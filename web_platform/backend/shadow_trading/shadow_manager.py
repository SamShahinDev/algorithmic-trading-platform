"""
Shadow Trading Manager
Manages all shadow/paper trades for pattern validation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from database.connection import get_db_session, db_manager
from database.models import ShadowTrade, Pattern, PatternStatus, TradeStatus, PatternDiscovery

class ShadowTradeManager:
    """
    Manages shadow trades for pattern validation
    Tracks ALL pattern occurrences, not just high-confidence ones
    """
    
    def __init__(self):
        self.active_shadows = []  # Currently monitoring shadow trades
        self.pattern_validators = {}  # Pattern validation thresholds
        self.monitoring = True
        self.shadow_stats = {}
        
        # Default validation thresholds
        self.validation_thresholds = {
            'min_shadow_trades': 50,
            'min_win_rate_for_testing': 65,
            'min_win_rate_for_active': 60,
            'auto_disable_threshold': 55
        }
    
    async def create_shadow_trade(self, pattern_data: Dict) -> ShadowTrade:
        """
        Create a shadow trade for any detected pattern
        This runs for EVERY pattern detection, regardless of confidence
        """
        shadow_trade = {
            'pattern_id': pattern_data['pattern_id'],
            'pattern_name': pattern_data['pattern_name'],
            'entry_time': datetime.utcnow(),
            'entry_price': pattern_data['current_price'],
            'direction': pattern_data['direction'],
            'stop_loss': pattern_data['stop_loss'],
            'take_profit': pattern_data['take_profit'],
            'confidence': pattern_data.get('confidence', 50),
            'would_have_traded': pattern_data.get('confidence', 50) >= 60,  # Lowered to 60% for more opportunities
            'pattern_quality': pattern_data.get('pattern_quality', 0),
            'market_conditions': pattern_data.get('market_conditions', {}),
            'status': TradeStatus.OPEN
        }
        
        # Add to database
        with get_db_session() as session:
            shadow = db_manager.add_shadow_trade(session, shadow_trade)
            shadow_id = shadow.id
        
        # Add to active monitoring
        shadow_trade['id'] = shadow_id
        self.active_shadows.append(shadow_trade)
        
        print(f"👁️ Shadow Trade Created: {pattern_data['pattern_name']} at ${pattern_data['current_price']:.2f}")
        print(f"   Confidence: {pattern_data.get('confidence', 50):.1f}% | Would Trade Live: {shadow_trade['would_have_traded']}")
        
        return shadow_trade
    
    async def monitor_shadow_trades(self, current_price: float):
        """
        Monitor all active shadow trades for completion
        Called regularly with current market price
        """
        completed_shadows = []
        
        for shadow in self.active_shadows:
            if shadow['status'] != TradeStatus.OPEN:
                continue
            
            # Check if hit stop loss
            if shadow['direction'] == 'long':
                if current_price <= shadow['stop_loss']:
                    shadow['exit_price'] = shadow['stop_loss']
                    shadow['exit_time'] = datetime.utcnow()
                    shadow['pnl'] = shadow['stop_loss'] - shadow['entry_price']
                    shadow['hit_stop'] = True
                    shadow['status'] = TradeStatus.CLOSED
                    completed_shadows.append(shadow)
                    
                elif current_price >= shadow['take_profit']:
                    shadow['exit_price'] = shadow['take_profit']
                    shadow['exit_time'] = datetime.utcnow()
                    shadow['pnl'] = shadow['take_profit'] - shadow['entry_price']
                    shadow['hit_target'] = True
                    shadow['status'] = TradeStatus.CLOSED
                    completed_shadows.append(shadow)
            
            else:  # short
                if current_price >= shadow['stop_loss']:
                    shadow['exit_price'] = shadow['stop_loss']
                    shadow['exit_time'] = datetime.utcnow()
                    shadow['pnl'] = shadow['entry_price'] - shadow['stop_loss']
                    shadow['hit_stop'] = True
                    shadow['status'] = TradeStatus.CLOSED
                    completed_shadows.append(shadow)
                    
                elif current_price <= shadow['take_profit']:
                    shadow['exit_price'] = shadow['take_profit']
                    shadow['exit_time'] = datetime.utcnow()
                    shadow['pnl'] = shadow['entry_price'] - shadow['take_profit']
                    shadow['hit_target'] = True
                    shadow['status'] = TradeStatus.CLOSED
                    completed_shadows.append(shadow)
        
        # Process completed shadow trades
        for shadow in completed_shadows:
            await self.process_completed_shadow(shadow)
            self.active_shadows.remove(shadow)
    
    async def process_completed_shadow(self, shadow: Dict):
        """Process a completed shadow trade and update statistics"""
        
        # Update database
        with get_db_session() as session:
            # Update shadow trade record
            shadow_record = session.query(ShadowTrade).filter_by(id=shadow['id']).first()
            if shadow_record:
                shadow_record.exit_price = shadow['exit_price']
                shadow_record.exit_time = shadow['exit_time']
                shadow_record.pnl = shadow['pnl']
                shadow_record.hit_stop = shadow.get('hit_stop', False)
                shadow_record.hit_target = shadow.get('hit_target', False)
                shadow_record.status = TradeStatus.CLOSED
            
            # Update pattern shadow statistics
            pattern = db_manager.update_shadow_stats(session, shadow['pattern_id'], shadow)
            
            # Check for pattern promotion
            await self.check_pattern_promotion(pattern)
            
            # Log result
            result = "WIN 🎯" if shadow['pnl'] > 0 else "LOSS ❌"
            print(f"👁️ Shadow Trade Closed: {shadow['pattern_name']} - {result}")
            print(f"   P&L: ${shadow['pnl']:.2f} | Shadow Win Rate: {pattern.shadow_win_rate:.1f}%")
    
    async def check_pattern_promotion(self, pattern: Pattern):
        """Check if pattern should be promoted based on shadow performance"""
        
        with get_db_session() as session:
            # Check for promotion from shadow_testing to testing
            if pattern.status == PatternStatus.SHADOW_TESTING:
                if pattern.shadow_trades >= 20 and pattern.shadow_win_rate > 65:
                    pattern.status = PatternStatus.TESTING
                    session.commit()
                    await self.send_promotion_alert(pattern, "TESTING")
                    print(f"🎉 Pattern {pattern.name} promoted to TESTING! Shadow Win Rate: {pattern.shadow_win_rate:.1f}%")
            
            # Check for promotion from testing to active
            elif pattern.status == PatternStatus.TESTING:
                if pattern.shadow_trades >= 50 and pattern.shadow_win_rate > 60:
                    pattern.status = PatternStatus.ACTIVE
                    session.commit()
                    await self.send_promotion_alert(pattern, "ACTIVE")
                    print(f"🚀 Pattern {pattern.name} promoted to ACTIVE! Ready for live trading!")
            
            # Check for demotion if performance degrades
            elif pattern.status in [PatternStatus.ACTIVE, PatternStatus.TESTING]:
                if pattern.shadow_win_rate < 55 and pattern.shadow_trades > 30:
                    pattern.status = PatternStatus.DISABLED
                    pattern.is_deployed = False
                    session.commit()
                    await self.send_demotion_alert(pattern)
                    print(f"⚠️ Pattern {pattern.name} disabled due to poor performance. Shadow Win Rate: {pattern.shadow_win_rate:.1f}%")
    
    async def run_ab_test(self, pattern_id: str, variations: List[Dict]):
        """
        Run A/B test with different parameters for the same pattern
        Test different stop/target levels to find optimal configuration
        """
        results = []
        
        for variation in variations:
            # Create shadow trade with variation parameters
            shadow_data = {
                'pattern_id': f"{pattern_id}_v{variation['id']}",
                'pattern_name': f"{pattern_id} (Variation {variation['id']})",
                'stop_loss': variation['stop'],
                'take_profit': variation['target'],
                # ... other parameters
            }
            
            shadow = await self.create_shadow_trade(shadow_data)
            results.append({
                'variation_id': variation['id'],
                'parameters': variation,
                'shadow_trade_id': shadow['id']
            })
        
        return results
    
    async def get_shadow_performance(self, pattern_id: str) -> Dict:
        """Get detailed shadow performance for a pattern"""
        
        with get_db_session() as session:
            pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
            if not pattern:
                return {}
            
            # Get recent shadow trades
            recent_shadows = db_manager.get_recent_shadow_trades(session, pattern_id, 100)
            
            # Calculate performance metrics
            total_pnl = sum(s.pnl for s in recent_shadows if s.pnl)
            wins = [s for s in recent_shadows if s.pnl and s.pnl > 0]
            losses = [s for s in recent_shadows if s.pnl and s.pnl <= 0]
            
            # Market condition analysis
            market_performance = self.analyze_market_conditions(recent_shadows)
            
            return {
                'pattern_id': pattern_id,
                'pattern_name': pattern.name,
                'shadow_trades': pattern.shadow_trades,
                'shadow_win_rate': pattern.shadow_win_rate,
                'total_shadow_pnl': total_pnl,
                'avg_win': sum(s.pnl for s in wins) / len(wins) if wins else 0,
                'avg_loss': sum(s.pnl for s in losses) / len(losses) if losses else 0,
                'best_market_conditions': market_performance,
                'would_have_traded_count': sum(1 for s in recent_shadows if s.would_have_traded),
                'pattern_quality_avg': sum(s.pattern_quality for s in recent_shadows) / len(recent_shadows) if recent_shadows else 0
            }
    
    def analyze_market_conditions(self, shadow_trades: List[ShadowTrade]) -> Dict:
        """Analyze which market conditions work best for pattern"""
        
        conditions_performance = {
            'trending': {'trades': 0, 'wins': 0},
            'ranging': {'trades': 0, 'wins': 0},
            'volatile': {'trades': 0, 'wins': 0}
        }
        
        for trade in shadow_trades:
            if trade.market_conditions:
                regime = trade.market_conditions.get('market_regime', 'unknown')
                if regime in conditions_performance:
                    conditions_performance[regime]['trades'] += 1
                    if trade.pnl and trade.pnl > 0:
                        conditions_performance[regime]['wins'] += 1
        
        # Calculate win rates per condition
        for condition in conditions_performance:
            trades = conditions_performance[condition]['trades']
            wins = conditions_performance[condition]['wins']
            conditions_performance[condition]['win_rate'] = (wins / trades * 100) if trades > 0 else 0
        
        return conditions_performance
    
    async def send_promotion_alert(self, pattern: Pattern, new_status: str):
        """Send alert when pattern is promoted"""
        # This will integrate with WebSocket to notify frontend
        alert = {
            'type': 'pattern_promotion',
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.name,
            'new_status': new_status,
            'shadow_win_rate': pattern.shadow_win_rate,
            'shadow_trades': pattern.shadow_trades,
            'message': f"Pattern {pattern.name} promoted to {new_status}!"
        }
        # TODO: Send via WebSocket
        return alert
    
    async def send_demotion_alert(self, pattern: Pattern):
        """Send alert when pattern is demoted/disabled"""
        alert = {
            'type': 'pattern_demotion',
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.name,
            'shadow_win_rate': pattern.shadow_win_rate,
            'message': f"Pattern {pattern.name} disabled due to poor performance"
        }
        # TODO: Send via WebSocket
        return alert
    
    async def get_shadow_vs_live_comparison(self, pattern_id: str) -> Dict:
        """Compare shadow performance vs live performance"""
        
        with get_db_session() as session:
            pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
            if not pattern:
                return {}
            
            return {
                'pattern_id': pattern_id,
                'pattern_name': pattern.name,
                'live_performance': {
                    'trades': pattern.total_trades,
                    'win_rate': pattern.win_rate,
                    'avg_profit': pattern.avg_profit,
                    'avg_loss': pattern.avg_loss,
                    'expected_value': pattern.expected_value
                },
                'shadow_performance': {
                    'trades': pattern.shadow_trades,
                    'win_rate': pattern.shadow_win_rate,
                    'theoretical_pnl': pattern.shadow_wins * pattern.avg_profit + pattern.shadow_losses * pattern.avg_loss
                },
                'divergence': abs(pattern.win_rate - pattern.shadow_win_rate),
                'recommendation': self.get_recommendation(pattern)
            }
    
    def get_recommendation(self, pattern: Pattern) -> str:
        """Get recommendation based on shadow vs live performance"""
        
        if pattern.shadow_win_rate > pattern.win_rate + 10:
            return "Consider increasing position size - shadow shows better performance"
        elif pattern.win_rate > pattern.shadow_win_rate + 10:
            return "Live trading outperforming - current parameters optimal"
        elif pattern.shadow_win_rate < 55:
            return "Consider disabling - shadow performance declining"
        else:
            return "Performance stable - continue monitoring"

# Global instance
shadow_manager = ShadowTradeManager()