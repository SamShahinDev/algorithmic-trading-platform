# XTRADING Platform - Multi-Bot Trading System Documentation
## 🚨 CRITICAL: Read this file first when starting any Claude session for this project

### Quick Recovery Command
```
Read /Users/royaltyvixion/Documents/XTRADING/web_platform/XTRADING_CONTEXT.md
```

---

## 📊 Project Overview
- **Purpose**: Multi-instrument automated futures trading system (NQ, ES, CL)
- **Current Status**: PRODUCTION - 3 bots actively trading on TopStepX
- **Last Updated**: August 28, 2025 (Thursday) - 10:45 PM
- **Architecture**: Multiple specialized trading bots with centralized risk management
- **Account**: PRAC-V2-XXXXX-XXXXXXXX (Practice Account)
- **Balance**: $149,384.16
- **Broker**: TopStepX via ProjectX Gateway

---

## 🤖 Active Trading Bots

### 1. NQ Bot (Nasdaq E-mini Futures)
- **File**: `trading_bot/intelligent_trading_bot_fixed_v2.py`
- **Status**: ✅ Running
- **Contract**: ENQ (mapped from NQ)
- **Patterns**: Momentum Thrust
- **Confidence Threshold**: 50%
- **Update Cycle**: 5 seconds
- **Position Tracking**: Fixed (was inverted, now correct)
- **Historical Win Rate**: 58%

### 2. ES Bot (S&P 500 E-mini Futures)
- **File**: `es_bot/es_bot_enhanced.py`
- **Status**: ✅ Running
- **Contract**: EP (mapped from ES)
- **Trade Execution**: Verified working
- **Discovered Patterns**: 4 patterns
  - Momentum Surge: 52.3% win rate
  - Volume Breakout: 61.8% win rate
  - Range Expansion: 72.1% win rate
  - Mean Reversion: 68.2% win rate

### 3. CL Bot (Crude Oil Futures)
- **File**: `cl_bot/cl_bot_enhanced.py`
- **Status**: ⚠️ Running with limitations
- **Contract**: CLU25/CLV25
- **API Issue**: Quote timeouts (TopStepX limitation)
- **Discovered Patterns**: 2 patterns
  - Oil Momentum: 73.4% win rate
  - Supply/Demand Imbalance: 75.2% win rate

### Bot Coordinator
- **File**: `run_es_cl_bots.py`
- **Function**: Manages ES and CL bots concurrently
- **Portfolio Monitoring**: Every 10 seconds
- **Logging**: Centralized to `logs/es_cl_bot.log`

---

## 🛡️ Rate Limiting & Safety Systems

### Rate Limiter Module
- **Location**: `shared/rate_limiter.py`
- **Limits**:
  - General API: 180 requests per 60 seconds
  - Historical Data: 50 requests per 30 seconds
  - Order Placement: 20 requests per 60 seconds
- **Features**:
  - Smart throttling (slows down at 80% usage)
  - Emergency stop at 95% usage
  - Per-limiter tracking and statistics
  - Deque-based sliding window implementation

### Monitoring Dashboard
- **File**: `monitor_rate_limits.py`
- **Features**: Real-time visualization of API usage
- **Color Coding**: Green (<50%), Yellow (50-70%), Red (>85%)

---

## 🔧 Operational Commands

### Starting the Bots
```bash
# Start NQ bot
nohup python3 trading_bot/intelligent_trading_bot_fixed_v2.py >> logs/nq_bot.log 2>&1 &

# Start ES/CL bots together
nohup python3 run_es_cl_bots.py >> logs/es_cl_bot.log 2>&1 &

# Monitor rate limits
python3 monitor_rate_limits.py
```

### Checking Bot Status
```bash
# Check if bots are running
ps aux | grep python3 | grep -E "(intelligent_trading|run_es_cl)" | grep -v grep

# Count running bots
pgrep -f "intelligent_trading_bot_fixed_v2.py\|run_es_cl_bots.py" | wc -l

# View NQ bot logs
tail -f logs/nq_bot.log

# View ES/CL bot logs
tail -f logs/es_cl_bot.log

# Check last activity
tail -n 20 logs/nq_bot.log | grep "Confidence"
```

### Stopping the Bots
```bash
# Stop NQ bot
pkill -f "intelligent_trading_bot_fixed_v2.py"

# Stop ES/CL bots
pkill -f "run_es_cl_bots.py"

# Stop all Python processes (careful!)
pkill -f python3
```

### Testing & Validation
```bash
# Test ES trade execution
python3 test_es_trade_execution.py

# Check positions
python3 check_es_position.py

# Validate patterns on historical data
python3 validate_patterns_q3q4_2024.py
```

---

## 🏗️ System Architecture

### Directory Structure
```
XTRADING/
├── trading_bot/
│   ├── intelligent_trading_bot_fixed_v2.py  # NQ bot
│   ├── data/                                # Data loaders
│   ├── analysis/                            # Pattern scanners
│   └── execution/                           # Trade execution
├── es_bot/
│   ├── es_bot_enhanced.py                   # ES trading bot
│   └── optimized_patterns.json              # ES patterns
├── cl_bot/
│   ├── cl_bot_enhanced.py                   # CL trading bot
│   └── optimized_patterns.json              # CL patterns
├── shared/
│   └── rate_limiter.py                      # Rate limiting
├── web_platform/
│   ├── backend/
│   │   ├── brokers/
│   │   │   └── topstepx_client.py          # TopStepX API
│   │   └── database/
│   │       └── trades.db                    # Trade database
│   └── frontend/
│       └── dashboard_spa.html               # Web dashboard
├── logs/
│   ├── nq_bot.log                          # NQ bot logs
│   └── es_cl_bot.log                       # ES/CL bot logs
└── Historical Data/
    └── New Data/                            # Compressed market data
```

### Key Components

#### TopStepX Client (`web_platform/backend/brokers/topstepx_client.py`)
- Handles all API communication
- Contract mapping (NQ→ENQ, ES→EP, CL→CLU25)
- Order placement and position management
- Integrated rate limiting
- Timeout protection for CL quotes

#### Pattern Discovery System
- **Files**: Various `*_pattern_discovery.py` files
- **Function**: Identifies profitable trading patterns
- **Data**: 2+ years of historical data (2023-2025)
- **Validation**: Out-of-sample testing attempted (data limitations)

#### Risk Management
- Position sizing: 1 contract per trade
- Stop loss: Dynamic based on pattern
- Daily limits: Configured per account rules
- Emergency stops: Automatic at rate limit threshold

---

## 📈 Pattern Discovery Results

### Summary Statistics
- **Total Patterns Discovered**: 7 (across all instruments)
- **Average Historical Win Rate**: 65.4%
- **Best Pattern**: CL Supply/Demand Imbalance (75.2%)
- **Most Signals**: ES Volume Breakout

### Validation Status
- **In-Sample**: ✅ Complete (2023-2025 data)
- **Out-of-Sample**: ❌ Data files empty for Q3-Q4 2024
- **Live Validation**: ✅ Ongoing in production

---

## 🐛 Troubleshooting Guide

### Common Issues and Solutions

#### Bot Not Starting
```bash
# Check for errors in log
tail -n 50 logs/nq_bot.log | grep ERROR

# Common fixes:
1. Check API credentials in topstepx_client.py
2. Ensure market is open (Sunday 5PM - Friday 4PM CT)
3. Verify rate limits not exceeded
```

#### Position Tracking Issues
- **Problem**: Bot shows wrong position (was showing SHORT when LONG)
- **Solution**: Fixed in `intelligent_trading_bot_fixed_v2.py`
- **Key**: LONG = BUY to open, SELL to close; SHORT = SELL to open, BUY to close

#### CL Bot Quote Timeouts
- **Problem**: CL quotes timing out after 5 seconds
- **Cause**: TopStepX API limitation for CL
- **Workaround**: Using historical bars fallback
- **Status**: Awaiting TopStepX fix

#### Rate Limit Warnings
```bash
# Monitor current usage
python3 monitor_rate_limits.py

# If critical (>90%), bots will auto-stop
# Wait 60 seconds for limit reset
```

#### Data Column Errors
- **Problem**: KeyError: 'close' not found
- **Solution**: `standardize_dataframe()` method handles variations
- **Columns**: Maps various formats to standard OHLCV

---

## 🔐 Critical Production Information

### TopStepX Credentials
- **API Key**: `86fzI3xGFVj06PWyiFYhDQGZ50QzxiJzhXW04F347h8=`
- **Username**: exotictrades
- **Account ID**: 10983875
- **Account Name**: PRAC-V2-XXXXX-XXXXXXXX

### Contract Mappings
```python
CONTRACT_MAPPING = {
    "NQ": "CON.F.US.ENQ.U25",  # Nasdaq E-mini
    "ES": "CON.F.US.EP.U25",   # S&P 500 E-mini
    "CL": "CLU25"               # Crude Oil (simplified)
}
```

### Trading Parameters
- **Position Size**: 1 contract (all bots)
- **Confidence Threshold**: 50% (adjustable)
- **Update Frequency**: 5 seconds
- **Max Daily Loss**: Account dependent
- **Market Hours**: Sunday 5:00 PM - Friday 4:00 PM CT

---

## 📊 Performance Tracking

### Current Session (August 28, 2025)
- **NQ Bot**: No trades (confidence 13-19%)
- **ES Bot**: Test trade executed successfully
- **CL Bot**: No trades (API limitations)
- **Account P&L**: -$1.40 (from ES test trade)

### Monitoring Metrics
- Win rate per pattern
- Average return per trade
- Sharpe ratio
- Maximum drawdown
- API usage statistics

---

## 🚀 Quick Start Guide

### After System Restart
```bash
# 1. Navigate to project
cd /Users/royaltyvixion/Documents/XTRADING

# 2. Start NQ bot
nohup python3 trading_bot/intelligent_trading_bot_fixed_v2.py >> logs/nq_bot.log 2>&1 &

# 3. Start ES/CL bots
nohup python3 run_es_cl_bots.py >> logs/es_cl_bot.log 2>&1 &

# 4. Monitor (optional)
python3 monitor_rate_limits.py

# 5. Check status
tail -f logs/nq_bot.log
```

### Daily Checks
1. Verify all bots running: `ps aux | grep python3`
2. Check rate limit health: `python3 monitor_rate_limits.py`
3. Review logs for errors: `grep ERROR logs/*.log`
4. Monitor account balance in TopStepX

---

## 🔄 Session Recovery

### For New Claude Sessions
```
We're working on the XTRADING multi-bot trading platform.
Read /Users/royaltyvixion/Documents/XTRADING/web_platform/XTRADING_CONTEXT.md for full context.

Current status:
- 3 bots running (NQ, ES, CL) on TopStepX practice account
- Using intelligent pattern recognition with 50% confidence threshold
- Rate limiting implemented to prevent API violations
- ES bot trade execution verified, CL has API limitations
```

### Key File Locations
- **Main Documentation**: This file
- **NQ Bot**: `trading_bot/intelligent_trading_bot_fixed_v2.py`
- **ES Bot**: `es_bot/es_bot_enhanced.py`
- **CL Bot**: `cl_bot/cl_bot_enhanced.py`
- **Coordinator**: `run_es_cl_bots.py`
- **Rate Limiter**: `shared/rate_limiter.py`
- **TopStepX Client**: `web_platform/backend/brokers/topstepx_client.py`
- **Logs**: `logs/nq_bot.log`, `logs/es_cl_bot.log`
- **Database**: `web_platform/backend/database/trades.db`

---

## 📝 Development Notes

### Recent Fixes (August 28, 2025)
1. **Position Tracking**: Fixed inverted LONG/SHORT logic in NQ bot
2. **Data Format**: Added `standardize_dataframe()` for column variations
3. **Rate Limiting**: Implemented comprehensive API throttling
4. **CL Timeout**: Added 5-second timeout protection
5. **ES Execution**: Verified order placement working

### Known Limitations
1. **CL API**: Quotes timing out (TopStepX issue)
2. **Historical Data**: Q3-Q4 2024 files empty
3. **Market Hours**: Bots idle during closed markets
4. **Confidence**: Low volatility = low confidence = no trades

### Future Enhancements
1. Lower confidence threshold to 40% for more signals
2. Add more sophisticated patterns
3. Implement cross-instrument correlation
4. Add voice/SMS alerts for trades
5. Build performance analytics dashboard

---

## 🆘 Emergency Procedures

### Rate Limit Emergency
```bash
# All bots will auto-stop at 95% usage
# To manually stop:
pkill -f python3

# Wait 60 seconds, then restart
sleep 60
# Then start bots again
```

### Position Mismatch
```bash
# Check broker positions
python3 check_es_position.py

# Force position sync on restart
# (Happens automatically on bot initialization)
```

### Complete System Reset
```bash
# Stop everything
pkill -f python3

# Clear logs (optional)
> logs/nq_bot.log
> logs/es_cl_bot.log

# Restart bots
./start_all_bots.sh  # (if created)
```

---

## 📞 Support & Resources

### TopStepX Support
- API Issues: Contact TopStepX support
- Contract specifications: CME Group website
- Trading hours: Check exchange websites

### System Requirements
- Python 3.9+
- Required packages: See `requirements.txt`
- Disk space: 10GB+ for historical data
- RAM: 4GB minimum
- Network: Stable internet connection

---

*This documentation represents the complete state of the XTRADING multi-bot trading system as of August 28, 2025.*
*For questions or issues, refer to the troubleshooting guide or check the logs.*