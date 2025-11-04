"""
Slack Integration Module for Trading Agents
Sends real-time notifications to dedicated Slack channels
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    print("⚠️ Slack SDK not installed. Run: pip install slack-sdk")

from dotenv import load_dotenv

# Load Slack configuration
load_dotenv('.env.slack')

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = "critical"  # Red - Immediate attention
    HIGH = "high"          # Orange - Important
    NORMAL = "normal"      # Blue - Regular updates
    LOW = "low"            # Gray - Informational

class ChannelType(Enum):
    """Slack channel types"""
    ORCHESTRATOR = "orchestrator"
    PATTERNS = "patterns"
    BACKTEST = "backtest"
    TRADES = "trades"
    RISK = "risk"
    PERFORMANCE = "performance"
    REGIME = "regime"
    ML = "ml"

@dataclass
class SlackMessage:
    """Slack message structure"""
    text: str
    channel: ChannelType
    priority: MessagePriority = MessagePriority.NORMAL
    thread_ts: Optional[str] = None
    attachments: Optional[List[Dict]] = None
    blocks: Optional[List[Dict]] = None

class SlackNotifier:
    """
    Manages Slack notifications for trading agents
    """
    
    def __init__(self):
        """Initialize Slack notifier"""
        self.enabled = os.getenv('SLACK_ENABLED', 'false').lower() == 'true'
        
        if not self.enabled:
            print("📵 Slack notifications disabled")
            return
        
        if not SLACK_AVAILABLE:
            print("⚠️ Slack SDK not available - notifications disabled")
            self.enabled = False
            return
        
        # Initialize Slack client
        self.token = os.getenv('SLACK_BOT_TOKEN', '')
        if not self.token or self.token == 'your-bot-token-here':
            print("⚠️ Slack bot token not configured - notifications disabled")
            self.enabled = False
            return
        
        self.client = WebClient(token=self.token)
        
        # Channel mapping
        self.channels = {
            ChannelType.ORCHESTRATOR: os.getenv('SLACK_ORCHESTRATOR_CHANNEL'),
            ChannelType.PATTERNS: os.getenv('SLACK_PATTERNS_CHANNEL'),
            ChannelType.BACKTEST: os.getenv('SLACK_BACKTEST_CHANNEL'),
            ChannelType.TRADES: os.getenv('SLACK_TRADES_CHANNEL'),
            ChannelType.RISK: os.getenv('SLACK_RISK_CHANNEL'),
            ChannelType.PERFORMANCE: os.getenv('SLACK_PERFORMANCE_CHANNEL'),
            ChannelType.REGIME: os.getenv('SLACK_REGIME_CHANNEL'),
            ChannelType.ML: os.getenv('SLACK_ML_CHANNEL'),
        }
        
        # Settings
        self.threading_enabled = os.getenv('SLACK_THREADING_ENABLED', 'true').lower() == 'true'
        self.rich_formatting = os.getenv('SLACK_RICH_FORMATTING', 'true').lower() == 'true'
        self.update_frequency = os.getenv('SLACK_UPDATE_FREQUENCY', 'all')
        
        # Alert thresholds
        self.risk_threshold = float(os.getenv('SLACK_RISK_ALERT_THRESHOLD', '500'))
        self.drawdown_threshold = float(os.getenv('SLACK_DRAWDOWN_ALERT', '0.05'))
        self.win_streak_threshold = int(os.getenv('SLACK_WIN_STREAK_NOTIFY', '3'))
        
        # Thread tracking
        self.active_threads = {}
        
        # Message queue for rate limiting
        self.message_queue = asyncio.Queue()
        self.is_processing = False
        
        # Logger
        self.logger = logging.getLogger('SlackNotifier')
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test Slack connection"""
        if not self.enabled:
            return
        
        try:
            response = self.client.auth_test()
            print(f"✅ Connected to Slack as @{response['user']}")
            print(f"📊 Channels configured: {sum(1 for c in self.channels.values() if c)}/8")
        except Exception as e:
            print(f"❌ Slack connection failed: {e}")
            self.enabled = False
    
    async def send_message(self, message: SlackMessage):
        """
        Send a message to Slack
        
        Args:
            message: SlackMessage object
        """
        if not self.enabled:
            return
        
        # Check update frequency
        if not self._should_send(message):
            return
        
        # Queue message
        await self.message_queue.put(message)
        
        # Start processor if not running
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process message queue"""
        self.is_processing = True
        
        while not self.message_queue.empty():
            try:
                message = await self.message_queue.get()
                await self._send_to_slack(message)
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
        
        self.is_processing = False
    
    async def _send_to_slack(self, message: SlackMessage):
        """Actually send message to Slack"""
        try:
            channel = self.channels.get(message.channel)
            if not channel:
                return
            
            # Build message
            slack_message = {
                'channel': channel,
                'text': message.text,
                'mrkdwn': self.rich_formatting
            }
            
            # Add threading
            if self.threading_enabled and message.thread_ts:
                slack_message['thread_ts'] = message.thread_ts
            
            # Add attachments or blocks
            if message.attachments:
                slack_message['attachments'] = message.attachments
            elif message.blocks:
                slack_message['blocks'] = message.blocks
            elif self.rich_formatting:
                # Auto-format based on priority
                slack_message['attachments'] = [self._create_attachment(message)]
            
            # Send message
            response = self.client.chat_postMessage(**slack_message)
            
            # Track thread
            if response['ts'] and message.channel in [ChannelType.TRADES, ChannelType.PATTERNS]:
                self.active_threads[f"{message.channel}_latest"] = response['ts']
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e.response['error']}")
        except Exception as e:
            self.logger.error(f"Error sending to Slack: {e}")
    
    def _should_send(self, message: SlackMessage) -> bool:
        """Check if message should be sent based on frequency settings"""
        if self.update_frequency == 'all':
            return True
        elif self.update_frequency == 'milestones':
            return message.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]
        elif self.update_frequency == 'critical':
            return message.priority == MessagePriority.CRITICAL
        return True
    
    def _create_attachment(self, message: SlackMessage) -> Dict:
        """Create formatted attachment based on priority"""
        colors = {
            MessagePriority.CRITICAL: 'danger',
            MessagePriority.HIGH: 'warning',
            MessagePriority.NORMAL: 'good',
            MessagePriority.LOW: '#808080'
        }
        
        return {
            'color': colors[message.priority],
            'text': message.text,
            'footer': f"Trading Bot - {message.channel.value.title()}",
            'ts': int(datetime.now().timestamp())
        }
    
    # Convenience methods for different notification types
    
    async def system_status(self, status: str, details: str = ""):
        """Send system status update"""
        emoji = "🚀" if "start" in status.lower() else "🛑" if "stop" in status.lower() else "ℹ️"
        
        message = SlackMessage(
            text=f"{emoji} *System Status*: {status}\n{details}",
            channel=ChannelType.ORCHESTRATOR,
            priority=MessagePriority.HIGH if "start" in status.lower() or "stop" in status.lower() else MessagePriority.NORMAL
        )
        await self.send_message(message)
    
    async def pattern_discovered(self, pattern_name: str, pattern_type: str, 
                                confidence: float, stats: Dict):
        """Notify about new pattern discovery"""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "✨ New Pattern Discovered"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Name:*\n{pattern_name}"},
                    {"type": "mrkdwn", "text": f"*Type:*\n{pattern_type}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.1%}"},
                    {"type": "mrkdwn", "text": f"*Win Rate:*\n{stats.get('win_rate', 0):.1%}"},
                    {"type": "mrkdwn", "text": f"*Occurrences:*\n{stats.get('occurrences', 0)}"},
                    {"type": "mrkdwn", "text": f"*Profit Factor:*\n{stats.get('profit_factor', 0):.2f}"}
                ]
            }
        ]
        
        message = SlackMessage(
            text=f"New pattern discovered: {pattern_name}",
            channel=ChannelType.PATTERNS,
            priority=MessagePriority.HIGH if confidence > 0.7 else MessagePriority.NORMAL,
            blocks=blocks
        )
        await self.send_message(message)
    
    async def backtest_complete(self, pattern_name: str, results: Dict):
        """Notify about backtest completion"""
        win_rate = results.get('win_rate', 0)
        emoji = "✅" if win_rate > 0.6 else "⚠️" if win_rate > 0.5 else "❌"
        
        attachments = [{
            'color': 'good' if win_rate > 0.6 else 'warning' if win_rate > 0.5 else 'danger',
            'title': f'{emoji} Backtest Complete: {pattern_name}',
            'fields': [
                {'title': 'Win Rate', 'value': f"{win_rate:.1%}", 'short': True},
                {'title': 'Profit Factor', 'value': f"{results.get('profit_factor', 0):.2f}", 'short': True},
                {'title': 'Sharpe Ratio', 'value': f"{results.get('sharpe_ratio', 0):.2f}", 'short': True},
                {'title': 'Max Drawdown', 'value': f"{results.get('max_drawdown', 0):.1%}", 'short': True},
                {'title': 'Total Trades', 'value': str(results.get('total_trades', 0)), 'short': True},
                {'title': 'Sample Size', 'value': str(results.get('sample_size', 0)), 'short': True}
            ],
            'footer': 'Backtest Validator',
            'ts': int(datetime.now().timestamp())
        }]
        
        message = SlackMessage(
            text=f"Backtest complete for {pattern_name}",
            channel=ChannelType.BACKTEST,
            priority=MessagePriority.NORMAL,
            attachments=attachments
        )
        await self.send_message(message)
    
    async def trade_executed(self, trade: Dict):
        """Notify about trade execution"""
        direction = trade.get('direction', 'unknown')
        emoji = "🟢" if direction == 'long' else "🔴" if direction == 'short' else "⚪"
        
        text = (f"{emoji} *Trade {trade.get('action', 'Executed')}*\n"
                f"Pattern: {trade.get('pattern_name', 'Unknown')}\n"
                f"Direction: {direction.upper()}\n"
                f"Price: ${trade.get('price', 0):,.2f}\n"
                f"Contracts: {trade.get('quantity', 0)}")
        
        if trade.get('action') == 'Exit':
            pnl = trade.get('pnl', 0)
            pnl_emoji = "💰" if pnl > 0 else "💸"
            text += f"\nP&L: {pnl_emoji} ${pnl:,.2f}"
        
        # Get thread for related trades
        thread_ts = None
        if self.threading_enabled and trade.get('trade_id'):
            thread_key = f"{ChannelType.TRADES}_trade_{trade['trade_id']}"
            thread_ts = self.active_threads.get(thread_key)
        
        message = SlackMessage(
            text=text,
            channel=ChannelType.TRADES,
            priority=MessagePriority.HIGH if abs(trade.get('pnl', 0)) > 500 else MessagePriority.NORMAL,
            thread_ts=thread_ts
        )
        
        response = await self.send_message(message)
        
        # Track thread for this trade
        if trade.get('trade_id') and trade.get('action') == 'Entry':
            self.active_threads[f"{ChannelType.TRADES}_trade_{trade['trade_id']}"] = response
    
    async def risk_alert(self, alert_type: str, details: Dict):
        """Send risk management alert"""
        emoji_map = {
            'drawdown': '📉',
            'position_limit': '🚫',
            'daily_loss': '🔴',
            'stop_loss': '🛑',
            'margin': '⚠️'
        }
        
        emoji = emoji_map.get(alert_type, '⚠️')
        priority = MessagePriority.CRITICAL if 'daily_loss' in alert_type else MessagePriority.HIGH
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} Risk Alert: {alert_type.replace('_', ' ').title()}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": details.get('message', 'Risk threshold triggered')}
            }
        ]
        
        if details.get('metrics'):
            fields = []
            for key, value in details['metrics'].items():
                fields.append({"type": "mrkdwn", "text": f"*{key}:*\n{value}"})
            
            blocks.append({
                "type": "section",
                "fields": fields[:10]  # Slack limit
            })
        
        message = SlackMessage(
            text=f"Risk Alert: {alert_type}",
            channel=ChannelType.RISK,
            priority=priority,
            blocks=blocks
        )
        await self.send_message(message)
    
    async def performance_update(self, metrics: Dict):
        """Send performance metrics update"""
        daily_pnl = metrics.get('daily_pnl', 0)
        emoji = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "➖"
        
        attachments = [{
            'color': 'good' if daily_pnl > 0 else 'danger' if daily_pnl < 0 else '#808080',
            'title': f'{emoji} Performance Update',
            'fields': [
                {'title': 'Daily P&L', 'value': f"${daily_pnl:,.2f}", 'short': True},
                {'title': 'Total Trades', 'value': str(metrics.get('total_trades', 0)), 'short': True},
                {'title': 'Win Rate', 'value': f"{metrics.get('win_rate', 0):.1%}", 'short': True},
                {'title': 'Active Patterns', 'value': str(metrics.get('active_patterns', 0)), 'short': True},
                {'title': 'Best Pattern', 'value': metrics.get('best_pattern', 'N/A'), 'short': True},
                {'title': 'Worst Pattern', 'value': metrics.get('worst_pattern', 'N/A'), 'short': True}
            ],
            'footer': 'Performance Monitor',
            'ts': int(datetime.now().timestamp())
        }]
        
        message = SlackMessage(
            text="Daily performance update",
            channel=ChannelType.PERFORMANCE,
            priority=MessagePriority.HIGH if abs(daily_pnl) > 1000 else MessagePriority.NORMAL,
            attachments=attachments
        )
        await self.send_message(message)
    
    async def regime_change(self, old_regime: str, new_regime: str, confidence: float):
        """Notify about market regime change"""
        regime_emojis = {
            'trending_up': '🟢📈',
            'trending_down': '🔴📉',
            'ranging': '🟡↔️',
            'volatile': '🟣💥',
            'quiet': '⚪😴'
        }
        
        old_emoji = regime_emojis.get(old_regime, '❓')
        new_emoji = regime_emojis.get(new_regime, '❓')
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔄 Market Regime Change Detected"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*From:*\n{old_emoji} {old_regime.replace('_', ' ').title()}"},
                    {"type": "mrkdwn", "text": f"*To:*\n{new_emoji} {new_regime.replace('_', ' ').title()}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.1%}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": self._get_regime_recommendations(new_regime)
                }
            }
        ]
        
        message = SlackMessage(
            text=f"Market regime changed: {old_regime} → {new_regime}",
            channel=ChannelType.REGIME,
            priority=MessagePriority.HIGH,
            blocks=blocks
        )
        await self.send_message(message)
    
    async def ml_pattern_found(self, pattern_type: str, cluster_id: int, stats: Dict):
        """Notify about ML-discovered pattern"""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🤖 ML Pattern Discovered"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Type:*\n{pattern_type}"},
                    {"type": "mrkdwn", "text": f"*Cluster ID:*\n{cluster_id}"},
                    {"type": "mrkdwn", "text": f"*Occurrences:*\n{stats.get('occurrences', 0)}"},
                    {"type": "mrkdwn", "text": f"*Win Rate:*\n{stats.get('win_rate', 0):.1%}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{stats.get('confidence', 0):.1%}"}
                ]
            }
        ]
        
        message = SlackMessage(
            text=f"ML pattern discovered: {pattern_type}",
            channel=ChannelType.ML,
            priority=MessagePriority.NORMAL,
            blocks=blocks
        )
        await self.send_message(message)
    
    async def monte_carlo_complete(self, pattern_name: str, results: Dict):
        """Notify about Monte Carlo simulation results"""
        robustness = results.get('robustness_score', 0)
        emoji = "💎" if robustness > 0.7 else "⚠️" if robustness > 0.5 else "❌"
        
        text = (f"{emoji} *Monte Carlo Analysis: {pattern_name}*\n"
                f"Iterations: {results.get('iterations', 0)}\n"
                f"Median Win Rate: {results.get('median_win_rate', 0):.1%} ± {results.get('win_rate_std', 0):.1%}\n"
                f"Robustness Score: {robustness:.1%}\n"
                f"Status: {'ROBUST' if robustness > 0.7 else 'MODERATE' if robustness > 0.5 else 'FRAGILE'}")
        
        message = SlackMessage(
            text=text,
            channel=ChannelType.BACKTEST,
            priority=MessagePriority.HIGH if robustness > 0.7 else MessagePriority.NORMAL
        )
        await self.send_message(message)
    
    def _get_regime_recommendations(self, regime: str) -> str:
        """Get trading recommendations for regime"""
        recommendations = {
            'trending_up': "*Recommendations:*\n• Favor trend-following patterns\n• Use wider stops\n• Trail profits",
            'trending_down': "*Recommendations:*\n• Be cautious with longs\n• Tighten risk management\n• Consider shorts",
            'ranging': "*Recommendations:*\n• Trade support/resistance\n• Avoid breakout patterns\n• Use range boundaries",
            'volatile': "*Recommendations:*\n• Reduce position sizes\n• Widen stops\n• Take quick profits",
            'quiet': "*Recommendations:*\n• Wait for volatility expansion\n• Prepare for breakouts\n• Reduce activity"
        }
        return recommendations.get(regime, "*Recommendations:*\n• Monitor closely")

# Global instance
slack_notifier = SlackNotifier()

# Convenience function for quick sends
async def send_slack(text: str, channel: ChannelType = ChannelType.ORCHESTRATOR, 
                    priority: MessagePriority = MessagePriority.NORMAL):
    """Quick send function"""
    message = SlackMessage(text=text, channel=channel, priority=priority)
    await slack_notifier.send_message(message)