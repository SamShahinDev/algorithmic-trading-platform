# TopStepX ProjectX Gateway Setup

## Current Status
✅ API client updated with correct endpoints  
✅ Environment configuration file created  
⚠️ Username required for authentication  

## Quick Setup

1. **Add your TopStepX username to `.env.topstepx`:**
   ```bash
   TOPSTEPX_USERNAME=your_username_here
   ```

2. **Verify your API key is correct:**
   - Current key in `.env.topstepx`: `86fzI3xGFVj06PWyiFYhDQGZ50QzxiJzhXW04F347h8=`
   - If this is not your actual API key, update it

3. **Restart the application:**
   ```bash
   python3 app.py
   ```

## API Endpoints Configured

Based on official TopStepX ProjectX Gateway documentation:

- **Base API**: `https://api.topstepx.com/api`
- **User Hub**: `https://rtc.topstepx.com/hubs/user` (WebSocket/SignalR)
- **Market Hub**: `https://rtc.topstepx.com/hubs/market` (WebSocket/SignalR)

### Key Endpoints Used:
- `/Auth/loginKey` - Authentication with username and API key
- `/Auth/validate` - Session validation (24-hour tokens)
- `/Account/search` - Get account information
- `/Contract/available` - Get tradeable contracts
- `/Order/place` - Place orders

## Configuration Parameters

All settings in `.env.topstepx`:

```env
# Authentication
TOPSTEPX_API_KEY=your_api_key
TOPSTEPX_USERNAME=your_username  # ← ADD THIS

# Trading Settings
DEFAULT_CONTRACT_SIZE=1           # Conservative size
USE_AUTO_BRACKETS=true           # TopStepX automatic brackets
BRACKET_PROFIT_TARGET=100        # $100 profit per trade
BRACKET_STOP_LOSS=100           # $100 stop loss per trade
```

## Trading Rules Enforced

The system enforces TopStepX evaluation account rules:

1. **Daily Loss Limit**: $1,500 max loss per day
2. **Trailing Drawdown**: $2,000 max from highest point
3. **Profit Target**: $3,000 to pass evaluation
4. **Position Limits**: 
   - Max 1 contract at a time
   - Max 10 trades per day
5. **No Overnight Holding** (initially)
6. **Commission**: $5 per round turn tracked

## System Components

### 1. Compliance Module (`topstepx/compliance.py`)
- Tracks daily P&L against limits
- Monitors trailing drawdown
- Activates recovery mode when approaching limits
- Blocks trades that would violate rules

### 2. Strategy Orchestrator (`strategies/orchestrator.py`)
- Manages 7 trading strategies across 3 tiers
- Tier 1: Mean Reversion, Momentum Breakout
- Tier 2: Microstructure, Order Flow
- Tier 3: Pairs Trading, Statistical Arbitrage
- Rotates strategies based on market conditions

### 3. Risk Manager (`risk_management/risk_manager.py`)
- Position sizing with Kelly Criterion (25% fraction)
- Correlation limits (max 0.6)
- Recovery mode activation
- Emergency stop functionality

### 4. TopStepX Client (`brokers/topstepx_client.py`)
- Handles API authentication
- Places orders with automatic brackets
- Monitors positions and P&L
- Manages WebSocket connections for real-time data

## Testing Connection

Once username is added, the system will:
1. Authenticate with TopStepX ProjectX Gateway
2. Retrieve account information
3. Find the correct NQ futures contract
4. Begin streaming market data
5. Enable order placement

## Market Hours

The system respects CME Globex futures hours:
- **Sunday**: Opens at 5:00 PM CT
- **Monday-Thursday**: 5:00 PM - 4:00 PM CT (next day)
- **Friday**: Closes at 4:00 PM CT
- **Saturday**: Closed

Daily maintenance break: 4:00 PM - 5:00 PM CT

## Troubleshooting

If connection fails after adding username:
1. Verify API key is correct
2. Check if account is active on TopStepX
3. Ensure you're connecting during market hours
4. Check logs for specific error messages

## Current Issues Resolved

✅ Fixed incorrect API endpoints (was using Tradovate)  
✅ Added proper session token management  
✅ Implemented contract ID lookup for NQ  
✅ Added environment variable configuration  
✅ Fixed market data fallback for weekends  

## Next Steps

1. Add your username to `.env.topstepx`
2. Verify connection when market opens (Sunday 5PM CT)
3. Monitor compliance dashboard in web interface
4. System will auto-trade based on strategy signals

---

**IMPORTANT**: The system is in SAFE MODE by default. It will:
- Start with minimum position size (1 contract)
- Use conservative Kelly fraction (25%)
- Activate recovery mode after any loss > $500
- Stop trading if daily loss approaches $1,000
- Emergency close all positions at $1,400 loss