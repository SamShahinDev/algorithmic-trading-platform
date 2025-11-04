# NQ Trading Bot - Setup Instructions

## Prerequisites
- Python 3.9 or higher
- TopStepX account (practice or live)
- Internet connection for real-time market data

## Installation Steps

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure TopStepX Credentials
Edit the `.env.topstepx` file and replace with your actual credentials:
- `TOPSTEPX_USERNAME`: Your TopStepX username
- `TOPSTEPX_PASSWORD`: Your TopStepX password  
- `TOPSTEPX_ACCOUNT_ID`: Your TopStepX account ID

### 3. Running the Bot

#### Start the bot:
```bash
python -m nq_bot.nq_bot
```

Or with explicit path:
```bash
PYTHONPATH=. python3 -m nq_bot.nq_bot
```

#### Stop the bot:
Press `Ctrl+C` in the terminal

## Features

### Discovery Mode (Default)
- 24/7 practice trading
- Relaxed thresholds (0.35 minimum score)
- No time restrictions
- Full telemetry logging

### Pattern Trading
- **Momentum Thrust**: Detects strong directional momentum
- **Trend Line Bounce**: Trades bounces off support/resistance
- **Technical Analysis Fallback**: Uses traditional TA when patterns fail

### Risk Management
- Position size: 1 contract (configurable)
- Stop-loss on every trade
- OCO (One-Cancels-Other) bracket orders
- Maximum slippage protection

## File Structure
```
Official Replica/
├── nq_bot/               # Main bot package
│   ├── nq_bot.py        # Main entry point
│   ├── pattern_config.py # Configuration settings
│   ├── patterns/        # Trading patterns
│   └── utils/           # Utility modules
├── web_platform/        # Broker integration
├── tests/               # Test suite
├── logs/                # Log files (created automatically)
├── .env.topstepx        # Credentials (edit this!)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

### Adjust Trading Parameters
Edit `nq_bot/pattern_config.py`:
- `min_confidence`: Minimum score to take trades (0.35 in discovery, 0.60 normal)
- `max_daily_trades`: Maximum trades per pattern per day
- `stop_ticks`: Stop loss distance in ticks

### Market Hours
Bot runs 24/7 in discovery mode. For specific hours, modify time restrictions in pattern_config.py

## Monitoring

### Log Files
- `logs/nq_bot.log`: Main activity log
- `logs/nq_discovery_telemetry.csv`: Detailed pattern evaluation data
- `logs/nq_bot.heartbeat.json`: Real-time status

### Console Output
The bot displays:
- Connection status
- Pattern evaluations
- Trade executions
- Position updates
- P&L tracking

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: 
   - Ensure you're in the correct directory
   - Use `PYTHONPATH=.` before the command

2. **Authentication Failed**:
   - Check credentials in `.env.topstepx`
   - Verify account is active on TopStepX

3. **No Trades Executing**:
   - Market conditions may not meet pattern criteria
   - Check ADX levels (needs >20 for good trends)
   - Review minimum confidence thresholds

4. **Multiple Bot Instances**:
   ```bash
   pkill -9 -f "nq_bot"  # Kill all instances
   ```

## Safety Features
- Practice mode by default
- 1-contract position limit
- Automatic stop-loss orders
- Slippage protection
- Risk state persistence

## Support
For issues or questions about the bot structure, refer to the test files in the `tests/` directory for examples of how components work together.

## Disclaimer
This bot is for educational and practice purposes. Always test thoroughly in practice mode before considering live trading. Trading futures involves substantial risk of loss.