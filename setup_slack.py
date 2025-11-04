#!/usr/bin/env python3
"""
Slack Setup Script for Trading Bot
Helps configure Slack integration with channel creation
"""

import os
import sys
from pathlib import Path

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    from dotenv import load_dotenv
except ImportError:
    print("❌ Slack SDK not installed")
    print("Please run: pip install slack-sdk python-dotenv")
    sys.exit(1)

def create_slack_channels():
    """Create required Slack channels"""
    
    # Load existing .env.slack if it exists
    env_path = Path('.env.slack')
    if env_path.exists():
        load_dotenv(env_path)
    
    print("🚀 Trading Bot Slack Setup")
    print("=" * 50)
    
    # Get bot token
    token = os.getenv('SLACK_BOT_TOKEN', '')
    if not token or token == 'your-bot-token-here':
        print("\n📝 First, create a Slack app:")
        print("1. Go to https://api.slack.com/apps")
        print("2. Click 'Create New App' > 'From scratch'")
        print("3. Name it 'Trading Bot' and select your workspace")
        print("4. Go to 'OAuth & Permissions'")
        print("5. Add these Bot Token Scopes:")
        print("   - chat:write")
        print("   - channels:read")
        print("   - channels:manage")
        print("   - groups:write")
        print("6. Click 'Install to Workspace'")
        print("7. Copy the Bot User OAuth Token\n")
        
        token = input("Enter your Bot User OAuth Token (xoxb-...): ").strip()
        if not token.startswith('xoxb-'):
            print("❌ Invalid token format")
            return False
    
    # Initialize client
    client = WebClient(token=token)
    
    try:
        # Test connection
        auth = client.auth_test()
        print(f"✅ Connected as @{auth['user']}")
        
        # Channels to create
        channels = [
            ('trading-orchestrator', 'Main system status and coordination'),
            ('pattern-discovery', 'New trading patterns discovered'),
            ('backtest-results', 'Pattern validation and backtesting'),
            ('live-trades', 'Trade execution notifications'),
            ('risk-alerts', 'Risk management warnings'),
            ('performance-metrics', 'Performance tracking and P&L'),
            ('market-regime', 'Market condition changes'),
            ('ml-patterns', 'Machine learning discoveries')
        ]
        
        print("\n📊 Creating/Finding Channels:")
        print("-" * 50)
        
        channel_ids = {}
        
        for channel_name, description in channels:
            try:
                # Try to create channel
                result = client.conversations_create(
                    name=channel_name,
                    is_private=False
                )
                channel_id = result['channel']['id']
                print(f"✅ Created #{channel_name} ({channel_id})")
                
                # Set topic
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
                            break
                    else:
                        print(f"❌ Could not find #{channel_name}")
                        continue
                else:
                    print(f"❌ Error with #{channel_name}: {e.response['error']}")
                    continue
            
            channel_ids[channel_name] = channel_id
            
            # Invite bot to channel
            try:
                client.conversations_join(channel=channel_id)
            except:
                pass  # Already in channel
        
        # Get signing secret
        print("\n🔐 Security Configuration:")
        print("1. Go to your app's 'Basic Information' page")
        print("2. Find 'Signing Secret' under 'App Credentials'")
        signing_secret = input("Enter your Signing Secret: ").strip()
        
        # Write .env.slack file
        print("\n📝 Writing configuration to .env.slack...")
        
        env_content = f"""# Slack Integration Configuration for Trading Agents
# DO NOT COMMIT THIS FILE TO GIT

# Bot User OAuth Token (xoxb-...)
SLACK_BOT_TOKEN={token}

# Signing Secret for verifying requests
SLACK_SIGNING_SECRET={signing_secret}

# Channel IDs for different agent notifications
SLACK_ORCHESTRATOR_CHANNEL={channel_ids.get('trading-orchestrator', '')}
SLACK_PATTERNS_CHANNEL={channel_ids.get('pattern-discovery', '')}
SLACK_BACKTEST_CHANNEL={channel_ids.get('backtest-results', '')}
SLACK_TRADES_CHANNEL={channel_ids.get('live-trades', '')}
SLACK_RISK_CHANNEL={channel_ids.get('risk-alerts', '')}
SLACK_PERFORMANCE_CHANNEL={channel_ids.get('performance-metrics', '')}
SLACK_REGIME_CHANNEL={channel_ids.get('market-regime', '')}
SLACK_ML_CHANNEL={channel_ids.get('ml-patterns', '')}

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
        
        # Add to .gitignore
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            if '.env.slack' not in content:
                with open(gitignore_path, 'a') as f:
                    f.write('\n# Slack configuration\n.env.slack\n')
                print("✅ Added .env.slack to .gitignore")
        
        # Test message
        print("\n📬 Sending test message...")
        
        test_channel = channel_ids.get('trading-orchestrator', '')
        if test_channel:
            client.chat_postMessage(
                channel=test_channel,
                text="🎉 Trading Bot Slack integration configured successfully!",
                blocks=[
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "✅ Slack Integration Ready"}
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Channels configured:*\n" + "\n".join([f"• #{ch}" for ch in channel_ids.keys()])
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "The trading bot will now send notifications to these channels automatically!"
                        }
                    }
                ]
            )
            print("✅ Test message sent to #trading-orchestrator")
        
        print("\n🎉 Setup Complete!")
        print("Your trading bot is now connected to Slack.")
        print("\nNext steps:")
        print("1. Install required packages: pip install slack-sdk python-dotenv")
        print("2. Start your trading bot normally")
        print("3. Notifications will appear in the appropriate channels")
        
        return True
        
    except SlackApiError as e:
        print(f"❌ Slack API error: {e.response['error']}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_slack_channels()
    sys.exit(0 if success else 1)