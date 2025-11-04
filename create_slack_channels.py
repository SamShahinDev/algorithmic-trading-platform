import os
#!/usr/bin/env python3
"""
Create Slack channels for Trading Bot using existing NovaGent token
"""

import os
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Use the token from NovaGent's .env.slack
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "your-slack-bot-token-here")

def create_trading_channels():
    """Create trading bot channels"""
    
    client = WebClient(token=SLACK_BOT_TOKEN)
    
    try:
        # Test connection
        auth = client.auth_test()
        print(f"✅ Connected as @{auth['user']}")
        
        # Channels to create with descriptions
        channels = [
            ('trading-orchestrator', '🎯 Main system status and coordination'),
            ('trading-patterns', '✨ New trading patterns discovered'),
            ('trading-backtest', '📊 Pattern validation and backtesting'),
            ('trading-live', '💹 Live trade execution notifications'),
            ('trading-risk', '🚨 Risk management warnings'),
            ('trading-performance', '📈 Performance tracking and P&L'),
            ('trading-regime', '🔄 Market condition changes'),
            ('trading-ml', '🤖 Machine learning discoveries')
        ]
        
        print("\n📊 Creating Trading Bot Channels:")
        print("-" * 50)
        
        channel_ids = {}
        created_count = 0
        existing_count = 0
        
        for channel_name, description in channels:
            try:
                # Try to create channel
                result = client.conversations_create(
                    name=channel_name,
                    is_private=False
                )
                channel_id = result['channel']['id']
                print(f"✅ Created #{channel_name} ({channel_id})")
                created_count += 1
                
                # Set topic/description
                client.conversations_setTopic(
                    channel=channel_id,
                    topic=description
                )
                
            except SlackApiError as e:
                if e.response['error'] == 'name_taken':
                    # Channel exists, find it
                    result = client.conversations_list(limit=1000)
                    for channel in result['channels']:
                        if channel['name'] == channel_name:
                            channel_id = channel['id']
                            print(f"ℹ️ Found existing #{channel_name} ({channel_id})")
                            existing_count += 1
                            
                            # Update topic anyway
                            try:
                                client.conversations_setTopic(
                                    channel=channel_id,
                                    topic=description
                                )
                            except:
                                pass
                            break
                    else:
                        print(f"❌ Could not find #{channel_name}")
                        continue
                else:
                    print(f"❌ Error with #{channel_name}: {e.response['error']}")
                    continue
            
            channel_ids[channel_name] = channel_id
            
            # Join channel as bot
            try:
                client.conversations_join(channel=channel_id)
            except:
                pass  # Already in channel
        
        # Update .env.slack with channel IDs
        print("\n📝 Updating .env.slack with channel IDs...")
        
        env_content = f"""# Slack Integration Configuration for Trading Agents
# DO NOT COMMIT THIS FILE TO GIT

# Bot User OAuth Token (xoxb-...)
SLACK_BOT_TOKEN={SLACK_BOT_TOKEN}

# Signing Secret for verifying requests
SLACK_SIGNING_SECRET=a5bd2ebb0149d13b6ed54e8599588973

# Channel IDs for different agent notifications
SLACK_ORCHESTRATOR_CHANNEL={channel_ids.get('trading-orchestrator', '')}
SLACK_PATTERNS_CHANNEL={channel_ids.get('trading-patterns', '')}
SLACK_BACKTEST_CHANNEL={channel_ids.get('trading-backtest', '')}
SLACK_TRADES_CHANNEL={channel_ids.get('trading-live', '')}
SLACK_RISK_CHANNEL={channel_ids.get('trading-risk', '')}
SLACK_PERFORMANCE_CHANNEL={channel_ids.get('trading-performance', '')}
SLACK_REGIME_CHANNEL={channel_ids.get('trading-regime', '')}
SLACK_ML_CHANNEL={channel_ids.get('trading-ml', '')}

# Notification Settings
SLACK_ENABLED=true
SLACK_UPDATE_FREQUENCY=all  # all, milestones, critical
SLACK_THREADING_ENABLED=true
SLACK_RICH_FORMATTING=true

# Alert Thresholds
SLACK_RISK_ALERT_THRESHOLD=500  # Alert if loss > $500
SLACK_DRAWDOWN_ALERT=0.05  # Alert if drawdown > 5%
SLACK_WIN_STREAK_NOTIFY=3  # Notify after 3 wins in a row
"""
        
        with open('.env.slack', 'w') as f:
            f.write(env_content)
        
        print("✅ Configuration saved to .env.slack")
        
        # Send welcome message to main channel
        if 'trading-orchestrator' in channel_ids:
            print("\n📬 Sending welcome message...")
            
            client.chat_postMessage(
                channel=channel_ids['trading-orchestrator'],
                text="🎉 Trading Bot Slack Integration Ready!",
                blocks=[
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🚀 Trading Bot Connected"}
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*System Overview:*\n• NQ Futures Trading\n• Pattern Discovery & Validation\n• Monte Carlo Simulations\n• Market Regime Detection\n• Machine Learning Discovery"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Channels Created:*\n{created_count}"},
                            {"type": "mrkdwn", "text": f"*Channels Found:*\n{existing_count}"},
                            {"type": "mrkdwn", "text": f"*Total Active:*\n{len(channel_ids)}"},
                            {"type": "mrkdwn", "text": "*Status:*\n🟢 Ready"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Channel Organization:*\n" + "\n".join([
                                f"• #{name} - {desc.split(' ', 1)[1] if ' ' in desc else desc}"
                                for name, desc in channels
                            ])
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "Trading bot will send automated notifications to appropriate channels"
                            }
                        ]
                    }
                ]
            )
            print("✅ Welcome message sent!")
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS!")
        print(f"Created: {created_count} new channels")
        print(f"Found: {existing_count} existing channels")
        print(f"Total: {len(channel_ids)} channels configured")
        print("\nYour trading bot is now connected to Slack!")
        print("Notifications will be automatically routed to the appropriate channels.")
        
        return True
        
    except SlackApiError as e:
        print(f"❌ Slack API error: {e.response['error']}")
        if e.response['error'] == 'invalid_auth':
            print("\nThe token appears to be invalid or expired.")
            print("Please update the SLACK_BOT_TOKEN in this script.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Trading Bot Slack Channel Creator")
    print("=" * 50)
    success = create_trading_channels()
    sys.exit(0 if success else 1)