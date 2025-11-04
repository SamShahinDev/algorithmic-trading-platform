# XTRADING Bot Setup Guide for Claude Code

## IMPORTANT: Read This First
This guide is specifically written for Claude Code to help set up and run the XTRADING bot system on a new machine. Follow these instructions carefully to ensure proper setup and operation.

## System Overview
XTRADING is a sophisticated automated trading system for futures trading on TopStepX. It includes multiple trading bots (ES, CL, NQ) with pattern recognition, risk management, and real-time trading capabilities.

## Prerequisites Check
First, verify these are installed:
```bash
python --version  # Should be 3.8 or higher
pip --version
git --version
```

## Step 1: Initial Setup

### 1.1 Install Python Dependencies
```bash
cd /path/to/XTRADING
pip install -r requirements.txt
```

### 1.2 Critical Environment Variables
The system uses TopStepX API. You MUST configure these files:
- `.env.topstepx` - Main environment file
- `web_platform/backend/.env.topstepx` - Backend environment file

**⚠️ IMPORTANT: DIFFERENT ACCOUNT SETUP ⚠️**
This system was copied from another machine with a DIFFERENT TopStepX account.
You MUST update ALL credentials to your NEW account:
1. Get your NEW API credentials from TopStepX
2. Replace ALL account-specific values below
3. Do NOT use the old account credentials

Example `.env.topstepx` structure:
```env
# TopStepX API Credentials - MUST BE UPDATED WITH NEW ACCOUNT!
TOPSTEPX_API_KEY=your_NEW_api_key_here  # ← CHANGE THIS
TOPSTEPX_BASE_URL=https://api.topstepx.com
TOPSTEPX_ACCOUNT_ID=your_NEW_account_id  # ← CHANGE THIS
TOPSTEPX_SECRET_KEY=your_NEW_secret_key  # ← CHANGE THIS (if applicable)

# Trading Configuration
DEFAULT_SYMBOL=NQ
DEFAULT_QUANTITY=1
MAX_DAILY_LOSS=1500
MAX_POSITION_SIZE=5

# Risk Management
USE_STOP_LOSS=true
STOP_LOSS_TICKS=10
TAKE_PROFIT_TICKS=20
```

**Files that MUST be updated with new account credentials:**
1. `/XTRADING/.env.topstepx` - Main config
2. `/XTRADING/web_platform/backend/.env.topstepx` - Backend config
3. Any other `.env` files in the project

**To get your new credentials:**
1. Log into your TopStepX account
2. Navigate to API settings
3. Generate new API keys if needed
4. Copy Account ID from your dashboard

## Step 2: Database Setup
```bash
# Initialize the trading database
python -c "from web_platform.backend.database.connection import init_db; init_db()"
```

## Step 3: Verify Installation
```bash
# Run the prerequisite check
python verify_prerequisites.py

# Test API connection
python test_auth.py
```

## Step 4: Running the Trading Bots

### Option 1: Run Individual Bot (NQ Bot - Most Stable)
```bash
# The NQ bot with patterns is the most tested
python run_nq_bot_with_patterns.py
```

### Option 2: Run All Bots
```bash
python run_all_bots.py
```

### Option 3: Run Web Platform
```bash
cd web_platform/backend
python app.py
# Then open browser to http://localhost:5000
```

## Important Files and Their Purposes

### Core Bot Files
- `run_nq_bot_with_patterns.py` - Main NQ bot runner with pattern detection
- `es_bot/` - ES futures bot implementation
- `cl_bot/` - CL futures bot implementation  
- `production_trading_bot.py` - Production-ready trading bot with all safety features

### Configuration Files
- `configs/` - Bot-specific configurations
- `pattern_config.py` - Pattern detection settings
- `utils/market_hours.py` - Market hours and session management

### Risk Management
- `utils/risk_manager.py` - Core risk management
- `utils/position_manager.py` - Position tracking
- `utils/dual_stop_manager.py` - Advanced stop-loss management
- `utils/exit_manager.py` - Exit strategy management

### Pattern Recognition
- `patterns/` - Pattern detection algorithms
- `utils/trend_line_detector.py` - Trend line detection
- `utils/pattern_memory.py` - Pattern memory and learning

### Critical Safety Features
- `utils/instance_lock.py` - Prevents multiple bot instances
- `utils/outage_manager.py` - Handles connection outages
- `utils/restart_recovery.py` - Graceful restart handling
- `utils/enhanced_rate_limiter.py` - API rate limiting

## Monitor and Logs

### Check Bot Status
```bash
# View running bots
ps aux | grep -E "python.*bot"

# Check bot heartbeat
cat logs/nq_bot.heartbeat.json

# Monitor logs
tail -f logs/nq_bot.log
```

### Log Files Location
- `logs/nq_bot.log` - Main NQ bot log
- `logs/es_bot.log` - ES bot log
- `logs/trades/` - Individual trade logs
- `web_platform/backend/database/trades.db` - Trade database

## Common Issues and Solutions

### Issue: "Another instance is already running"
```bash
# Force unlock if needed (use carefully)
rm -f locks/*.lock
```

### Issue: API Connection Failed
1. Check `.env.topstepx` has correct credentials
2. Verify account is active on TopStepX
3. Check internet connection

### Issue: Pattern detection not working
```bash
# Validate patterns
python validate_patterns_q3q4_2024.py
```

### Issue: Bot not placing trades
1. Check market hours: `python utils/market_hours.py`
2. Verify account has sufficient balance
3. Check risk limits in configuration

## Production Deployment Checklist

1. [ ] Environment variables configured
2. [ ] Database initialized
3. [ ] API connection tested
4. [ ] Risk limits set appropriately
5. [ ] Stop-loss configured
6. [ ] Rate limiting enabled
7. [ ] Instance lock working
8. [ ] Logs directory writable
9. [ ] Market hours configured correctly
10. [ ] Emergency shutdown tested

## Emergency Commands

### Stop All Bots Immediately
```bash
pkill -f "python.*bot"
```

### Flatten All Positions
```bash
python emergency_flatten.py
```

### Check Current Positions
```bash
python check_es_position.py
```

## Performance Optimization

### For Better Performance:
1. Run on a machine with stable internet
2. Use SSD for logs directory
3. Keep log files under 1GB (rotate regularly)
4. Monitor CPU usage (should stay under 50%)
5. Ensure at least 4GB RAM available

## Testing Mode

### Run in Test Mode First
```bash
# Set in .env.topstepx
TRADING_MODE=PAPER
USE_REAL_MONEY=false
```

### Validate Historical Patterns
```bash
python validate_nq_patterns_oos.py
```

## Advanced Configuration

### Pattern Sensitivity
Edit `pattern_config.py`:
```python
PATTERN_MIN_CONFIDENCE = 0.7  # Increase for fewer, higher quality signals
PATTERN_LOOKBACK_PERIODS = 20  # Adjust for pattern detection window
```

### Risk Parameters
Edit `utils/risk_manager.py` or set in `.env.topstepx`:
```python
MAX_DAILY_LOSS = 1500
MAX_DRAWDOWN = 2000
POSITION_SIZE_LIMITS = {"NQ": 5, "ES": 3, "CL": 2}
```

## Support and Troubleshooting

### Debug Mode
```bash
# Run with debug logging
export DEBUG=true
python run_nq_bot_with_patterns.py
```

### Performance Analysis
```bash
python analyze_trades.py
python utils/performance_analytics.py
```

### Pattern Analysis
```bash
python pattern_analysis_comparison.md
```

## CRITICAL WARNINGS

⚠️ **NEVER** run multiple instances of the same bot on the same account
⚠️ **ALWAYS** test in paper trading mode first
⚠️ **MONITOR** the first 24 hours closely after deployment
⚠️ **SET** appropriate risk limits for your account size
⚠️ **CHECK** market hours configuration matches your timezone

## Quick Start Commands

```bash
# 1. Setup
cd /path/to/XTRADING
pip install -r requirements.txt

# 2. Configure
# Edit .env.topstepx with your credentials

# 3. Test
python verify_prerequisites.py
python test_auth.py

# 4. Run
python run_nq_bot_with_patterns.py

# 5. Monitor
tail -f logs/nq_bot.log
```

## Notes for Claude Code

When helping with this system:
1. The NQ bot is the most stable and tested
2. Pattern detection is CPU-intensive - monitor performance
3. TopStepX has rate limits - respect them
4. Always check market hours before troubleshooting "no trades"
5. The web platform provides a good UI for monitoring
6. Database queries should be optimized for large trade volumes
7. Emergency flatten is available but use cautiously
8. Logs are your friend - check them first when debugging

## Architecture Overview

```
XTRADING/
├── Bot Implementations
│   ├── es_bot/          # ES futures bot
│   ├── cl_bot/          # CL futures bot
│   └── nq_bot/          # NQ futures bot (primary)
├── Core Systems
│   ├── utils/           # Risk, position, pattern management
│   ├── patterns/        # Pattern detection algorithms
│   └── shared/          # Shared components
├── Web Platform
│   ├── backend/         # Flask API, TopStepX integration
│   └── frontend/        # Dashboard, monitoring UI
├── Configuration
│   ├── configs/         # Bot configurations
│   └── .env.topstepx    # Environment variables
└── Operations
    ├── logs/            # System logs
    ├── locks/           # Instance locks
    └── scripts/         # Utility scripts
```

## Final Setup Verification

Run this checklist:
```bash
# 1. Check Python
python --version

# 2. Check dependencies
pip list | grep -E "(pandas|numpy|websocket|requests)"

# 3. Check environment
python -c "import os; print('API Key Set:', 'TOPSTEPX_API_KEY' in os.environ or os.path.exists('.env.topstepx'))"

# 4. Check database
python -c "from web_platform.backend.database.connection import get_db_connection; conn = get_db_connection(); print('DB OK')"

# 5. Test connection
python test_auth.py

# If all pass, you're ready to trade!
```

---
Generated: $(date)
Version: XTRADING v2.0
Purpose: Claude Code Setup Guide for New Installation