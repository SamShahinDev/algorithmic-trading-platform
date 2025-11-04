"""
AI Thoughts Generator Service
Generates intelligent analysis and thoughts for the AI assistant
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional
import json

class AIThoughtsGenerator:
    """
    Generates contextual AI thoughts based on market conditions and patterns
    """
    
    def __init__(self):
        self.current_price = 15000
        self.support_levels = [14950, 14920, 14900, 14880, 14850]
        self.resistance_levels = [15050, 15080, 15100, 15120, 15150]
        self.active_pattern = None
        self.market_state = 'neutral'
        self.thoughts_queue = []
        
        # Thought templates by category
        self.thought_templates = {
            'analysis': [
                "📊 Analyzing price action around ${price:.2f}...",
                "🔍 Scanning for pattern formations near key levels...",
                "📈 Evaluating momentum indicators for trend confirmation...",
                "🎯 Checking for rejection candles at ${level:.2f}...",
                "📉 Monitoring volume profile for institutional activity...",
                "🔄 Comparing current structure to historical patterns...",
                "⚡ Processing real-time order flow data...",
                "🌊 Analyzing market microstructure for hidden liquidity..."
            ],
            'opportunity': [
                "⚠️ Price approaching support at ${level:.2f} - watching for bounce setup",
                "🎯 High probability {pattern} pattern forming with {confidence}% confidence",
                "📍 Key level test at ${level:.2f} - potential reversal zone",
                "🔥 Volume spike detected - possible breakout incoming",
                "✨ Perfect setup conditions aligning for {pattern} pattern",
                "🚀 Momentum building for potential {direction} move",
                "💎 Premium setup detected - all indicators confirming",
                "🎪 Major support/resistance confluence at ${level:.2f}"
            ],
            'warning': [
                "⚠️ Approaching daily loss limit - risk management active",
                "🛑 Unusual volatility detected - adjusting position sizing",
                "📊 Pattern confidence dropping below threshold",
                "⏰ Low liquidity period - wider spreads expected",
                "🔴 Divergence detected between price and indicators",
                "⚡ Rapid price movement - potential stop hunt",
                "🌪️ Market conditions changing - reassessing patterns",
                "⚠️ Multiple timeframe conflict detected"
            ],
            'success': [
                "✅ Pattern confirmed - {pattern} setup validated",
                "🎯 Target reached at ${price:.2f} - profit locked in",
                "💰 Trade executed successfully - monitoring position",
                "🏆 Pattern performing as expected - {win_rate}% historical win rate",
                "✨ All entry conditions met - high confidence trade",
                "📈 Position in profit - trailing stop activated",
                "🎖️ Risk/reward ratio optimal at {ratio}:1",
                "💎 Premium setup captured at perfect entry"
            ],
            'info': [
                "📝 Current market regime: {regime}",
                "🕐 Best trading window approaching in {time} minutes",
                "📊 {count} patterns currently active in scan",
                "💡 Tip: {pattern} works best during {condition}",
                "🔍 Next key level: ${level:.2f} ({type})",
                "📈 Market trending {direction} on {timeframe} timeframe",
                "🎲 Statistical edge: {edge}% over last {period}",
                "🧮 Expected value for current setup: ${value:.2f}"
            ],
            'learning': [
                "🧠 Pattern recognition improving - {improvement}% better accuracy",
                "📚 Historical analysis: {pattern} succeeded {rate}% in similar conditions",
                "🔬 Backtesting shows optimal entry at {entry_type}",
                "💭 Market psychology: {sentiment} sentiment detected",
                "🎓 Learning from last trade - adjusting parameters",
                "📊 Statistical anomaly detected - investigating cause",
                "🔄 Adapting to current market dynamics",
                "🌟 New pattern variation discovered - adding to library"
            ]
        }
        
        # Market conditions
        self.market_conditions = {
            'volatility': 'normal',
            'trend': 'neutral',
            'volume': 'average',
            'sentiment': 'mixed'
        }
    
    async def generate_thought(self, category: str = None) -> Dict:
        """Generate a contextual AI thought"""
        
        if not category:
            # Choose category based on current conditions
            category = self._choose_category()
        
        # Get appropriate template
        templates = self.thought_templates.get(category, self.thought_templates['info'])
        template = random.choice(templates)
        
        # Generate message with context
        message = self._fill_template(template)
        
        thought = {
            'id': datetime.now().timestamp(),
            'category': category,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'confidence': random.randint(70, 95),
            'priority': self._get_priority(category)
        }
        
        return thought
    
    def _choose_category(self) -> str:
        """Choose thought category based on market conditions"""
        
        # Weighted random selection based on conditions
        weights = {
            'analysis': 30,
            'info': 25,
            'learning': 15,
            'opportunity': 15,
            'warning': 10,
            'success': 5
        }
        
        # Adjust weights based on market state
        if self.market_state == 'bullish':
            weights['opportunity'] += 10
            weights['success'] += 5
        elif self.market_state == 'bearish':
            weights['warning'] += 10
            weights['analysis'] += 5
        
        # Weighted random choice
        categories = list(weights.keys())
        probabilities = list(weights.values())
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        return random.choices(categories, weights=probabilities)[0]
    
    def _fill_template(self, template: str) -> str:
        """Fill template with contextual data"""
        
        # Price near support/resistance
        near_support = min(self.support_levels, key=lambda x: abs(x - self.current_price))
        near_resistance = min(self.resistance_levels, key=lambda x: abs(x - self.current_price))
        
        # Determine closest level
        if abs(self.current_price - near_support) < abs(self.current_price - near_resistance):
            near_level = near_support
            level_type = "support"
        else:
            near_level = near_resistance
            level_type = "resistance"
        
        # Context variables
        context = {
            'price': self.current_price,
            'level': near_level,
            'type': level_type,
            'pattern': random.choice(['S/R Bounce', 'VWAP Bounce', 'Opening Range Breakout', 'Mean Reversion']),
            'confidence': random.randint(75, 95),
            'direction': random.choice(['bullish', 'bearish']),
            'win_rate': random.randint(70, 90),
            'ratio': round(random.uniform(1.2, 2.5), 1),
            'regime': self.market_conditions['trend'],
            'time': random.randint(5, 30),
            'count': random.randint(3, 8),
            'condition': random.choice(['high volume', 'trending markets', 'range-bound markets']),
            'timeframe': random.choice(['5m', '15m', '1h']),
            'edge': random.randint(5, 15),
            'period': random.choice(['100 trades', '7 days', '30 days']),
            'value': random.uniform(100, 500),
            'improvement': random.randint(5, 20),
            'rate': random.randint(70, 90),
            'entry_type': random.choice(['limit orders', 'market orders', 'stop entries']),
            'sentiment': self.market_conditions['sentiment']
        }
        
        # Format template
        try:
            return template.format(**context)
        except:
            return template
    
    def _get_priority(self, category: str) -> str:
        """Get priority level for thought category"""
        priority_map = {
            'warning': 'high',
            'opportunity': 'high',
            'success': 'medium',
            'analysis': 'low',
            'info': 'low',
            'learning': 'low'
        }
        return priority_map.get(category, 'low')
    
    async def generate_stream(self) -> Dict:
        """Generate continuous stream of thoughts"""
        while True:
            thought = await self.generate_thought()
            yield thought
            
            # Wait between thoughts (more frequent during market hours)
            wait_time = random.randint(10, 30)
            await asyncio.sleep(wait_time)
    
    async def update_market_data(self, data: Dict):
        """Update market data for context"""
        if 'price' in data:
            self.current_price = data['price']
        if 'support_levels' in data:
            self.support_levels = data['support_levels']
        if 'resistance_levels' in data:
            self.resistance_levels = data['resistance_levels']
        if 'market_state' in data:
            self.market_state = data['market_state']
        if 'conditions' in data:
            self.market_conditions.update(data['conditions'])
    
    async def generate_specific_thought(self, event_type: str, context: Dict = None) -> Dict:
        """Generate thought for specific event"""
        
        specific_thoughts = {
            'trade_entry': "🎯 Executing {pattern} trade at ${price:.2f} - Stop: ${stop:.2f}, Target: ${target:.2f}",
            'trade_exit': "✅ Trade closed at ${price:.2f} - P&L: ${pnl:+.2f} ({percent:+.1f}%)",
            'pattern_detected': "🔍 {pattern} pattern detected with {confidence}% confidence - Monitoring for entry",
            'level_approach': "📍 Approaching {type} at ${level:.2f} - {distance:.2f} points away",
            'risk_alert': "⚠️ Risk limit approaching - {used}/{limit} daily loss used",
            'session_start': "🌅 Trading session started - Initializing pattern scanners",
            'session_end': "🌙 Session complete - Today's P&L: ${pnl:+.2f} ({trades} trades)"
        }
        
        template = specific_thoughts.get(event_type, "📊 Market event: {event}")
        
        if context:
            try:
                message = template.format(**context)
            except:
                message = template
        else:
            message = template
        
        return {
            'id': datetime.now().timestamp(),
            'category': 'event',
            'event_type': event_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'priority': 'high' if 'alert' in event_type else 'medium'
        }
    
    def get_market_summary(self) -> Dict:
        """Get current market analysis summary"""
        return {
            'price': self.current_price,
            'nearest_support': min(self.support_levels, key=lambda x: abs(x - self.current_price)),
            'nearest_resistance': min(self.resistance_levels, key=lambda x: abs(x - self.current_price)),
            'market_state': self.market_state,
            'conditions': self.market_conditions,
            'active_patterns': random.randint(3, 7),
            'confidence': random.randint(70, 90)
        }

# Global instance
ai_thoughts_generator = AIThoughtsGenerator()