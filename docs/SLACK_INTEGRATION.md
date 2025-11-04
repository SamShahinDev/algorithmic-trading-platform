# 📱 Slack Integration Guide

## Overview

The Trading Bot includes comprehensive Slack integration to keep you informed about all trading activities in real-time. Messages are automatically routed to dedicated channels based on content type.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install slack-sdk python-dotenv
```

### 2. Run Setup Script
```bash
python setup_slack.py
```

This will:
- Guide you through creating a Slack app
- Create all necessary channels
- Configure your bot token and credentials
- Send a test message

### 3. Manual Setup (Alternative)

#### Create Slack App
1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" > "From scratch"
3. Name: "Trading Bot"
4. Select your workspace

#### Configure Permissions
Add these Bot Token Scopes:
- `chat:write` - Send messages
- `channels:read` - Read channel info
- `channels:manage` - Create channels
- `groups:write` - Post to private channels

#### Install to Workspace
1. Click "Install to Workspace"
2. Copy the Bot User OAuth Token (starts with `xoxb-`)

#### Create Channels
Create these channels in Slack:
- `#trading-orchestrator`
- `#pattern-discovery`
- `#backtest-results`
- `#live-trades`
- `#risk-alerts`
- `#performance-metrics`
- `#market-regime`
- `#ml-patterns`

#### Configure .env.slack
Copy `.env.slack.example` and fill in your credentials.

## 📊 Channel Organization

### #trading-orchestrator
Main system notifications:
- System startup/shutdown
- Configuration changes
- Major milestones

### #pattern-discovery
Pattern discovery notifications:
- New patterns found
- Pattern statistics
- Confidence scores

### #backtest-results
Backtesting results:
- Validation complete
- Win rates and metrics
- Monte Carlo results

### #live-trades
Trade execution:
- Entry notifications (🟢)
- Exit notifications (🔴)
- P&L updates

### #risk-alerts
Risk management:
- Daily loss limits (🚨)
- Drawdown warnings (📉)
- Position limit alerts (🚫)

### #performance-metrics
Performance tracking:
- Daily P&L summaries
- Win rate updates
- Best/worst patterns

### #market-regime
Market conditions:
- Regime changes
- Trading recommendations
- Volatility alerts

### #ml-patterns
Machine learning discoveries:
- Cluster patterns
- Anomaly detections
- Sequence patterns

## 🎨 Message Formatting

### Priority Levels
- 🔴 **Critical**: Immediate attention required
- 🟠 **High**: Important updates
- 🔵 **Normal**: Regular notifications
- ⚫ **Low**: Informational only

### Trade Notifications
```
🟢 Trade Entry
Pattern: Trend Line Bounce
Direction: LONG
Price: $23,456.50
Contracts: 2

🔴 Trade Exit
P&L: 💰 $240.00
```

### Pattern Discovery
```
✨ New Pattern Discovered
Name: MA50 Bounce
Type: ma_bounce
Confidence: 75.3%
Win Rate: 68.2%
Occurrences: 42
```

### Risk Alerts
```
🚨 Risk Alert: Daily Loss Limit
Daily P&L: -$1,000.00
Action: Trading halted
```

## ⚙️ Configuration Options

### Update Frequency
In `.env.slack`:
- `all` - All updates
- `milestones` - Important events only
- `critical` - Critical alerts only

### Alert Thresholds
```env
SLACK_RISK_ALERT_THRESHOLD=500  # Alert if loss > $500
SLACK_DRAWDOWN_ALERT=0.05       # Alert if drawdown > 5%
SLACK_WIN_STREAK_NOTIFY=3       # Notify after 3 wins
```

### Threading
- Related messages stay in threads
- Trades tracked from entry to exit
- Clean channel organization

## 🔧 Troubleshooting

### Bot Not Posting
1. Check token in `.env.slack`
2. Verify bot is in channels
3. Check `SLACK_ENABLED=true`

### Missing Channels
Run: `python setup_slack.py`

### Rate Limiting
The bot automatically queues messages to avoid rate limits.

### Connection Issues
Check network and Slack API status.

## 📝 Custom Notifications

Add custom notifications in your code:

```python
from utils.slack_notifier import slack_notifier, ChannelType, MessagePriority

# Simple message
await slack_notifier.send_slack(
    "Custom message",
    ChannelType.ORCHESTRATOR,
    MessagePriority.NORMAL
)

# Trade notification
await slack_notifier.trade_executed({
    'action': 'Entry',
    'pattern_name': 'My Pattern',
    'direction': 'long',
    'price': 23456.50,
    'quantity': 2
})

# Risk alert
await slack_notifier.risk_alert('custom_risk', {
    'message': 'Custom risk detected',
    'metrics': {'Risk Level': 'High'}
})
```

## 🚫 Security

- Never commit `.env.slack` to git
- Keep bot tokens secret
- Use environment variables
- Rotate tokens periodically

## 📊 Message Flow

```
Trading Bot
    ↓
Slack Notifier
    ↓
Smart Router (keywords)
    ↓
Appropriate Channel
    ↓
Your Slack Workspace
```

## 🎯 Best Practices

1. **Channel Hygiene**: Keep channels focused
2. **Threading**: Related messages in threads
3. **Priorities**: Use appropriate priority levels
4. **Filtering**: Configure update frequency
5. **Monitoring**: Check channels regularly

## 📈 Benefits

- **Real-time Updates**: Never miss a trade
- **Risk Awareness**: Instant risk alerts
- **Performance Tracking**: Daily summaries
- **Pattern Discovery**: New opportunities
- **Team Collaboration**: Share with team
- **Mobile Access**: Trade from anywhere
- **Historical Record**: Searchable history

## 🆘 Support

For issues or questions:
1. Check this documentation
2. Review `.env.slack` configuration
3. Run `python setup_slack.py` to reconfigure
4. Check Slack API status

---

*Happy Trading! 🚀*