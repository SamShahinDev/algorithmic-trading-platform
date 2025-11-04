# NQ Trading Bot - Complete Setup Guide for New Installation

## Quick Start Checklist
- [ ] Python 3.9+ installed
- [ ] All files from this folder copied to new PC
- [ ] TopStepX account credentials ready
- [ ] Internet connection available

## Step-by-Step Setup Instructions

### 1. System Requirements Verification
```bash
# Check Python version (must be 3.9 or higher)
python3 --version

# If Python not installed, download from https://python.org
```

### 2. Navigate to Project Directory
```bash
# Change to the directory where you copied the Official Replica folder
cd /path/to/Official\ Replica
```

### 3. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installations
pip list
```

### 5. Configure TopStepX Credentials
```bash
# Edit the .env.topstepx file
# On Windows: notepad .env.topstepx
# On Mac: nano .env.topstepx
# On Linux: nano .env.topstepx
```

Replace these values with your actual credentials:
- `TOPSTEPX_USERNAME=your_actual_username`
- `TOPSTEPX_PASSWORD=your_actual_password`
- `TOPSTEPX_ACCOUNT_ID=your_actual_account_id`

### 6. Create Required Directories
```bash
# Create logs directory if it doesn't exist
mkdir -p logs

# Create state directory for bot persistence
mkdir -p nq_bot/state
```

### 7. Test Connection to TopStepX
```python
# Test script to verify credentials (run this first)
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv('.env.topstepx')
print('Username:', os.getenv('TOPSTEPX_USERNAME'))
print('Account ID:', os.getenv('TOPSTEPX_ACCOUNT_ID'))
print('Password:', '***' if os.getenv('TOPSTEPX_PASSWORD') else 'NOT SET')
"
```

### 8. Run the Bot

#### Option A: Standard Run
```bash
PYTHONPATH=. python3 -m nq_bot.nq_bot
```

#### Option B: Background Run (Linux/Mac)
```bash
nohup python3 -m nq_bot.nq_bot > logs/nq_bot_output.log 2>&1 &
```

#### Option C: Windows Run
```cmd
set PYTHONPATH=.
python -m nq_bot.nq_bot
```

### 9. Verify Bot is Running
```bash
# Check if bot process is running
# On Mac/Linux:
ps aux | grep nq_bot

# On Windows:
tasklist | findstr python

# Check log file for activity
tail -f logs/nq_bot.log
```

### 10. Monitor Bot Activity
```bash
# Watch real-time logs
tail -f logs/nq_bot.log

# Check telemetry data
tail -f logs/nq_discovery_telemetry.csv

# View heartbeat status
cat logs/nq_bot.heartbeat.json
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or run with: PYTHONPATH=. python3 -m nq_bot.nq_bot
```

### Issue 2: Permission Denied
```bash
# Solution: Make script executable
chmod +x nq_bot/nq_bot.py
```

### Issue 3: Connection Failed
```bash
# Check credentials in .env.topstepx
# Verify internet connection
# Check if TopStepX API is accessible
ping api.topstepx.com
```

### Issue 4: Multiple Bot Instances
```bash
# Kill all existing instances first
pkill -f nq_bot
# Then start fresh
```

## File Structure Overview
```
Official Replica/
├── .env.topstepx          # EDIT THIS: Your credentials
├── requirements.txt       # Python dependencies
├── README.md             # General documentation
├── SETUP_GUIDE.md        # This file
├── nq_bot/               # Main bot code
│   ├── nq_bot.py        # Entry point
│   ├── pattern_config.py # Trading configuration
│   ├── patterns/        # Trading patterns
│   ├── utils/           # Utility modules
│   └── state/           # Persistent state (created)
├── web_platform/         # Broker integration
├── tests/                # Test suite
└── logs/                 # Log files (created)
```

## Configuration Adjustments

### Change Trading Mode (Discovery vs Production)
Edit `nq_bot/pattern_config.py`:
```python
# Line ~15-20
DISCOVERY_MODE = True  # Set to False for production
```

### Adjust Risk Parameters
Edit `nq_bot/pattern_config.py`:
```python
# Minimum confidence scores
'min_confidence': 0.35,  # Discovery mode
# Change to 0.60 for production mode
```

### Set Account Type
Edit `.env.topstepx`:
```bash
TOPSTEPX_ACCOUNT_TYPE=practice  # or 'live' for real trading
```

## Automated Startup Script

### Create start_bot.sh (Mac/Linux)
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
export PYTHONPATH=.
python3 -m nq_bot.nq_bot
```

### Create start_bot.bat (Windows)
```batch
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat 2>nul
set PYTHONPATH=.
python -m nq_bot.nq_bot
pause
```

## Safety Checks Before Running

1. **Verify Practice Mode**: Check that you're using a practice account first
2. **Check Balance**: Ensure account has sufficient funds
3. **Test Hours**: Run during market hours for live data
4. **Monitor Initially**: Watch the first few trades closely

## Stop the Bot Safely

### Graceful Shutdown
1. Press `Ctrl+C` in the terminal
2. Wait for "Bot stopped" message
3. Check no positions are left open

### Force Stop (Emergency)
```bash
# Mac/Linux
pkill -9 -f nq_bot

# Windows
taskkill /F /IM python.exe
```

## Support Resources

### Log Files to Check
- `logs/nq_bot.log` - Main activity
- `logs/nq_discovery_telemetry.csv` - Pattern evaluations  
- `logs/rate_limits.log` - API rate limiting
- `logs/nq_bot.heartbeat.json` - Current status

### Test the Setup
```bash
# Run tests to verify everything works
python -m pytest tests/ -v
```

## Important Notes

⚠️ **ALWAYS TEST IN PRACTICE MODE FIRST**
⚠️ **Never share your .env.topstepx file**
⚠️ **Keep your credentials secure**
⚠️ **Monitor the bot regularly**

## Contact for Issues
If you encounter issues not covered here:
1. Check the README.md for additional details
2. Review test files for component examples
3. Check logs for error messages
4. Verify all dependencies are installed

---
Last Updated: September 2025
Bot Version: Discovery Mode v1.0