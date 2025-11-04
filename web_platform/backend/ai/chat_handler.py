"""
AI Chat Handler - Main interface for AI assistant
Processes user queries and provides strategic recommendations
"""

import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

class AIAssistant:
    """Main AI chat interface for strategic trading recommendations"""
    
    def __init__(self):
        """Initialize AI Assistant with DeepSeek API"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY', 'sk-bb508eca10bc42958a0aafe83fb422c5')
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Initialize analyzers
        from .strategy_analyzer import StrategyAnalyzer
        from .market_tracker import MarketTracker
        
        self.analyzer = StrategyAnalyzer()
        self.market_tracker = MarketTracker()
        
        # Conversation history
        self.conversation_history = []
        
    async def process_query(self, user_input: str) -> Dict:
        """
        Process user query and return strategic recommendations
        
        Args:
            user_input: User's question or request
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Check if trading is active via API endpoint
            try:
                import requests
                response = requests.get('http://localhost:8000/api/status')
                if response.status_code == 200:
                    trading_active = response.json().get('monitoring', False)
                else:
                    trading_active = False
            except:
                trading_active = False
            
            # Detect query intent
            intent = self._detect_intent(user_input)
            
            # Gather relevant context based on intent
            context = await self._gather_context(intent)
            context['trading_active'] = trading_active
            
            # Build prompt for DeepSeek
            prompt = self._build_prompt(user_input, context, intent)
            
            # Call DeepSeek API (passes intent for fallback)
            ai_response = await self._call_deepseek(prompt, intent)
            
            # Format and store response
            formatted_response = self._format_response(ai_response, intent)
            
            # Add trading status notice if not active
            if not trading_active:
                formatted_response = f"⏸️ **Trading Status: Inactive** (Click 'Start Trading' to begin)\n\n{formatted_response}"
            
            # Store conversation
            self._store_conversation(user_input, formatted_response)
            
            return {
                'success': True,
                'response': formatted_response,
                'intent': intent,
                'timestamp': datetime.now().isoformat(),
                'trading_active': trading_active
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': "I apologize, but I encountered an error processing your request. Please try again."
            }
    
    def _detect_intent(self, user_input: str) -> str:
        """Detect the intent of user's query"""
        user_input_lower = user_input.lower()
        
        # Check for money/profit questions first (highest priority)
        if '$' in user_input or any(word in user_input_lower for word in ['dollar', 'money', 'profit', 'earn', 'make']):
            return 'profit_target'
        elif any(word in user_input_lower for word in ['strategy', 'recommend', 'trade', 'tomorrow', 'today', 'think']):
            return 'strategy_recommendation'
        elif any(word in user_input_lower for word in ['performance', 'analyze', 'how did', 'results']):
            return 'performance_analysis'
        elif any(word in user_input_lower for word in ['risk', 'position', 'size', 'kelly']):
            return 'risk_assessment'
        elif any(word in user_input_lower for word in ['market', 'condition', 'trend', 'ranging']):
            return 'market_analysis'
        elif any(word in user_input_lower for word in ['pattern', 'which', 'best', 'profitable']):
            return 'pattern_analysis'
        else:
            return 'general'
    
    async def _gather_context(self, intent: str) -> Dict:
        """Gather relevant context based on query intent"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'intent': intent
        }
        
        # Get performance data
        if intent in ['strategy_recommendation', 'performance_analysis', 'pattern_analysis']:
            context['performance'] = await self.analyzer.get_recent_performance()
            context['pattern_stats'] = await self.analyzer.get_pattern_statistics()
        
        # Get market conditions
        if intent in ['strategy_recommendation', 'market_analysis']:
            context['market'] = await self.market_tracker.get_current_conditions()
            context['support_resistance'] = await self.market_tracker.get_sr_levels()
        
        # Get risk metrics
        if intent in ['risk_assessment', 'strategy_recommendation']:
            context['risk'] = await self.analyzer.get_risk_metrics()
            context['position_sizing'] = await self.analyzer.calculate_position_sizes()
        
        return context
    
    def _build_prompt(self, user_input: str, context: Dict, intent: str) -> str:
        """Build comprehensive prompt for DeepSeek"""
        
        # System prompt defining the AI's role
        system_prompt = """You are an expert AI trading assistant for an NQ futures trading platform. 
        You provide data-driven strategy recommendations based on actual trading performance.
        You have access to real-time market data, pattern performance metrics, and risk analytics.
        Always be specific with numbers and actionable recommendations.
        Focus on practical, implementable strategies."""
        
        # Context prompt with current data
        context_prompt = f"""
        Current Trading Context:
        - Timestamp: {context.get('timestamp')}
        - Intent: {intent}
        """
        
        if 'performance' in context:
            context_prompt += f"\n- Recent Performance: {json.dumps(context['performance'], indent=2)}"
        
        if 'market' in context:
            context_prompt += f"\n- Market Conditions: {json.dumps(context['market'], indent=2)}"
        
        if 'pattern_stats' in context:
            context_prompt += f"\n- Pattern Statistics: {json.dumps(context['pattern_stats'], indent=2)}"
        
        if 'risk' in context:
            context_prompt += f"\n- Risk Metrics: {json.dumps(context['risk'], indent=2)}"
        
        # User query
        user_prompt = f"User Question: {user_input}"
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{context_prompt}\n\n{user_prompt}"
        
        return full_prompt
    
    async def _call_deepseek(self, prompt: str, intent: str = None) -> str:
        """Call DeepSeek API for response"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert trading assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                # Fallback to rule-based response if API fails
                return await self._generate_fallback_response(prompt, intent)
                
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return await self._generate_fallback_response(prompt, intent)
    
    async def _generate_fallback_response(self, prompt: str, intent: str = None) -> str:
        """Generate fallback response if API fails - use real data"""
        
        # Use intent if provided, otherwise detect from prompt
        if intent == 'profit_target' or ('$' in prompt or any(word in prompt.lower() for word in ['dollar', 'money', 'profit', 'earn', 'make'])):
            # Handle money/profit specific questions
            performance = await self.analyzer.get_recent_performance()
            pattern_stats = await self.analyzer.get_pattern_statistics()
            
            # Extract target amount if mentioned
            import re
            money_match = re.search(r'\$(\d+)', prompt)
            target_amount = int(money_match.group(1)) if money_match else 100
            
            # Calculate how many trades needed for target
            avg_win = 125  # Default NQ win is about $125 per contract with 5 point target
            trades_needed = target_amount / avg_win if avg_win > 0 else 1
            
            response = f"""💰 Profit Target Analysis: ${target_amount}/hour

📊 Based on Historical Performance:
• Average Win per Trade: ${avg_win:.2f}
• Trades Needed: {trades_needed:.1f} winning trades
• Your Win Rate: {performance.get('overall_win_rate', 0):.1f}%
• Total Attempts Needed: {trades_needed / max(0.01, performance.get('overall_win_rate', 50) / 100):.0f} trades

🎯 Most Profitable Patterns Last Week:
"""
            # Add top patterns by profit
            if pattern_stats:
                sorted_patterns = sorted(pattern_stats.items(), 
                                       key=lambda x: x[1].get('total_pnl', 0), 
                                       reverse=True)[:3]
                for i, (pattern_name, stats) in enumerate(sorted_patterns, 1):
                    response += f"\n{i}. {pattern_name}: ${stats.get('total_pnl', 0):.2f} ({stats.get('total_trades', 0)} trades)"
            else:
                response += "\nNo pattern data available yet - start shadow trading to build history"
            
            response += f"""

⚡ Strategy to Achieve ${target_amount}/hour:
• Trade 2-3 high-confidence patterns simultaneously
• Focus on 9:30-11:30 AM (highest volume period)
• Use 2 contracts to double profit potential
• Set alerts at key support/resistance levels
• Exit quickly at target (don't get greedy)

⚠️ Reality Check:
Making ${target_amount}/hour consistently requires discipline and proper risk management. Start with paper trading to validate the strategy."""
            
            return response
            
        elif intent == 'strategy_recommendation' or 'strategy' in prompt.lower() or 'recommend' in prompt.lower() or 'think' in prompt.lower():
            # Get real market conditions and user performance
            market = await self.market_tracker.get_current_conditions()
            trend = market.get('trend', {}).get('direction', 'unknown')
            volatility = market.get('volatility', {}).get('volatility_level', 'unknown')
            current_price = market.get('support_resistance', {}).get('current_price', 0)
            
            # Get support and resistance levels
            support_levels = market.get('support_resistance', {}).get('support', [])
            resistance_levels = market.get('support_resistance', {}).get('resistance', [])
            
            # Get user's performance data
            performance = await self.analyzer.get_recent_performance()
            pattern_stats = await self.analyzer.get_pattern_statistics()
            
            # Get current active patterns
            from database.connection import get_db_session
            from database.models import Pattern, PatternStatus
            with get_db_session() as session:
                active_patterns = session.query(Pattern).filter(
                    Pattern.status == PatternStatus.ACTIVE
                ).count()
            
            # Format support and resistance info
            support_text = f"${support_levels[0]:,.2f} ({abs(current_price - support_levels[0]):.0f} points away)" if support_levels else "No support identified"
            resistance_text = f"${resistance_levels[0]:,.2f} ({abs(resistance_levels[0] - current_price):.0f} points away)" if resistance_levels else "No resistance identified"
            
            response = f"""📊 Strategy Recommendation based on current market:
            
📈 Market Analysis:
• Trend: {trend.replace('_', ' ').title()}
• Volatility: {volatility.replace('_', ' ').title()}
• Current Price: ${current_price:,.2f}
• Nearest Support: {support_text}
• Nearest Resistance: {resistance_text}

📊 Your Performance:
• Win Rate: {performance.get('overall_win_rate', 0):.1f}%
• Active Patterns: {active_patterns}
• Best Pattern: {performance.get('best_pattern', 'None')}
• Total P&L (7d): ${performance.get('total_pnl', 0):,.2f}

🎯 Recommended Strategy:
"""
            
            if 'ranging' in trend or volatility in ['low', 'very_low']:
                # Get best patterns for this market condition
                best_patterns = []
                if pattern_stats:
                    for pattern_name, stats in pattern_stats.items():
                        if 'bounce' in pattern_name.lower() or 'double' in pattern_name.lower():
                            if stats.get('win_rate', 0) > 70:
                                best_patterns.append(f"{pattern_name} ({stats.get('win_rate', 0):.0f}% win rate)")
                
                pattern_text = f"\n• Deploy these patterns: {', '.join(best_patterns[:3])}" if best_patterns else ""
                
                response += f"""1. Focus on mean reversion patterns (S/R Bounce, Double Top/Bottom)
2. Use tighter stops (4-5 points) due to low volatility
3. Target the support at {support_text} for entries
4. Best times: 9:30-11:30 AM and 2:00-3:30 PM EST{pattern_text}"""
            elif 'strong' in trend:
                response += """1. Follow the trend with breakout patterns
2. Use wider stops (6-8 points) to avoid premature exits
3. Let winners run with trailing stops
4. Avoid counter-trend trades"""
            else:
                response += """1. Mixed conditions - be selective with trades
2. Standard 5-point stops and targets
3. Wait for high-confidence setups (>80%)
4. Reduce position size for safety"""
                
            return response
            
        elif 'performance' in prompt.lower() or 'analyze' in prompt.lower():
            # Get real performance data
            performance = await self.analyzer.get_recent_performance()
            
            response = f"""📈 Performance Analysis (Last 7 Days):

Summary:
• Total Trades: {performance.get('total_trades', 0)}
• Win Rate: {performance.get('overall_win_rate', 0):.1f}%
• Total P&L: ${performance.get('total_pnl', 0):,.2f}
• Best Pattern: {performance.get('best_pattern', 'None')}
• Worst Pattern: {performance.get('worst_pattern', 'None')}

Recommendations:
"""
            if performance.get('overall_win_rate', 0) > 60:
                response += """✅ Strong performance! Consider:
- Increasing position size gradually
- Deploying more patterns for live trading
- Maintaining current risk parameters"""
            else:
                response += """⚠️ Room for improvement:
- Focus on your best performing patterns
- Review losing trades for common mistakes
- Consider tighter risk management"""
            
            return response
                
        elif 'risk' in prompt.lower() or 'position' in prompt.lower():
            # Get risk metrics
            try:
                from risk_management.risk_manager import risk_manager
                risk_metrics = await risk_manager.get_risk_metrics()
                
                return f"""⚠️ Risk Management Status:

Current Metrics:
• Daily P&L: ${risk_metrics.get('daily_pnl', 0):,.2f}
• Risk Score: {risk_metrics.get('risk_score', 0)}/100
• Open Positions: {risk_metrics.get('open_positions', 0)}
• Daily Loss Limit: ${risk_metrics.get('daily_limit', 1500):,.2f}

Recommendations:
• Max positions: {risk_metrics.get('max_positions', 2)}
• Position size: {risk_metrics.get('recommended_size', 1)} contract(s)
• Stop loss: {risk_metrics.get('recommended_stop', 5)} points
"""
            except:
                return """Risk management system initializing. Please try again in a moment."""
                
        else:
            return """I'm here to help with:
📊 Strategy recommendations based on market conditions
📈 Performance analysis of your trading patterns
⚠️ Risk management and position sizing
🎯 Pattern selection for current market regime

What would you like to explore?"""
    
    def _format_response(self, ai_response: str, intent: str) -> str:
        """Format AI response for display"""
        # Don't add extra formatting if response already has emojis
        if ai_response.startswith("📊") or ai_response.startswith("📈") or ai_response.startswith("⚠️"):
            return ai_response
            
        # Add intent-specific formatting if needed
        if intent == 'strategy_recommendation':
            if 'Market Conditions:' not in ai_response:
                ai_response = "📊 Strategy Recommendation\n\n" + ai_response
        elif intent == 'performance_analysis':
            if 'Performance Analysis' not in ai_response:
                ai_response = "📈 Performance Analysis\n\n" + ai_response
        
        return ai_response
    
    def _store_conversation(self, user_input: str, ai_response: str):
        """Store conversation in history and database"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_input,
            'ai_response': ai_response
        }
        
        # Add to memory
        self.conversation_history.append(conversation)
        
        # Keep only last 50 conversations in memory
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        # TODO: Store in database table ai_conversations
    
    async def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on current context"""
        market_conditions = await self.market_tracker.get_current_conditions()
        
        suggestions = [
            "What's the best strategy for today?",
            "Analyze my last 7 days performance",
            "Which patterns are most profitable?",
            "Should I adjust my risk parameters?"
        ]
        
        # Add context-specific suggestions
        if market_conditions.get('trend') == 'ranging':
            suggestions.append("What patterns work best in ranging markets?")
        elif market_conditions.get('volatility') == 'high':
            suggestions.append("How should I adjust for high volatility?")
        
        return suggestions

# Global instance
ai_assistant = AIAssistant()