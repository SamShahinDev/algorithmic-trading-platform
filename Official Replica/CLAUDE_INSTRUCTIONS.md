# Instructions for Claude Code - NQ Trading Bot Setup

## Your Mission
Set up and run the NQ Trading Bot on this new system with a different TopStepX account.

## Priority Tasks (In Order)

### 1. Check System Requirements
```bash
python3 --version  # Must be 3.9+
pwd               # Note current directory
ls -la            # Verify all files present
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get TopStepX Credentials from User
Ask the user for:
- TopStepX Username
- TopStepX Password  
- TopStepX Account ID

### 4. Configure Credentials
```bash
# Edit .env.topstepx file with provided credentials
# Use the Edit tool to update the file
```

### 5. Verify Setup
```bash
# Test credentials are loaded
python3 -c "from dotenv import load_dotenv; import os; load_dotenv('.env.topstepx'); print('Credentials loaded:', bool(os.getenv('TOPSTEPX_USERNAME')))"

# Create required directories
mkdir -p logs nq_bot/state
```

### 6. Start the Bot
```bash
PYTHONPATH=. python3 -m nq_bot.nq_bot
```

## Key Files to Know

### Configuration Files
- `.env.topstepx` - **MUST EDIT**: Contains account credentials
- `nq_bot/pattern_config.py` - Trading parameters and thresholds

### Main Entry Point
- `nq_bot/nq_bot.py` - Start the bot from here

### Monitoring Files
- `logs/nq_bot.log` - Main activity log
- `logs/nq_discovery_telemetry.csv` - Trading signals data
- `logs/nq_bot.heartbeat.json` - Real-time status

## Common Commands You'll Need

### Start Bot
```bash
PYTHONPATH=. python3 -m nq_bot.nq_bot
```

### Stop Bot
```bash
pkill -f nq_bot
```

### Check Status
```bash
ps aux | grep nq_bot
tail -f logs/nq_bot.log
```

### View Recent Trades
```bash
tail -100 logs/nq_bot.log | grep -E "Trade executed|Position|P&L"
```

## Important Settings

### Discovery Mode (Default)
- Located in `nq_bot/pattern_config.py`
- `DISCOVERY_MODE = True` - 24/7 practice trading
- `GLOBAL_MIN_CONFIDENCE = 0.35` - Lower threshold for more trades

### Risk Limits
- Position size: 1 contract (hardcoded safety)
- Stop loss: Always set on every trade
- Maximum slippage: 2 ticks

## What to Tell the User

When setup is complete, inform them:
1. Bot is running in DISCOVERY MODE (practice/learning mode)
2. Using practice account: [show account ID]
3. Minimum score threshold: 0.35 (will take more trades for learning)
4. Monitoring files are in the `logs/` directory
5. To stop: Press Ctrl+C or run `pkill -f nq_bot`

## Troubleshooting Guide

### If "ModuleNotFoundError"
Always run with: `PYTHONPATH=. python3 -m nq_bot.nq_bot`

### If "Authentication Failed"
1. Check credentials in `.env.topstepx`
2. Verify account is active
3. Check internet connection

### If "No trades executing"
This is normal - bot waits for specific market conditions:
- Momentum thrust patterns
- Trend line bounces
- Minimum ADX > 20 for trends

### If multiple instances running
```bash
pkill -9 -f nq_bot  # Kill all
# Then restart
```

## Safety Features to Highlight

1. **Practice Mode**: Bot runs on practice account by default
2. **Position Limit**: Maximum 1 contract at a time
3. **Stop Loss**: Automatic on every trade
4. **OCO Orders**: One-Cancels-Other bracket orders
5. **Slippage Protection**: Max 2 tick slippage

## Expected Behavior

The bot will:
1. Connect to TopStepX
2. Load historical data
3. Start scanning for patterns
4. Execute trades when patterns match (score > 0.35)
5. Log all activity to files
6. Update heartbeat every few seconds

## Success Indicators

✅ "Connected to TopStepX ProjectX Gateway"
✅ "NQ BOT WITH PATTERNS STARTED"
✅ "DISCOVERY MODE ACTIVE"
✅ "Main trading loop started"
✅ Regular "DATA_OK" messages

## Red Flags

❌ "Failed to authenticate"
❌ "Connection refused"
❌ Multiple "Error" messages
❌ No activity for >1 minute

## Your Responses Should Be

- **Concise**: Give status updates briefly
- **Factual**: Report actual log output
- **Helpful**: Suggest fixes for any errors
- **Safe**: Always verify practice mode is active

## Remember

- Never modify trading logic without user request
- Always preserve safety features
- Keep credentials secure (never log them)
- Monitor for duplicate instances
- Report actual trades and P&L from logs

---
This file is specifically for Claude Code to understand the NQ bot setup process.