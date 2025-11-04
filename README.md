# XTrading - Algorithmic Trading Platform

Advanced algorithmic trading system for futures markets with pattern recognition, risk management, and real-time execution capabilities.

## 🎯 Overview

Professional-grade automated trading platform built for NQ, ES, and CL futures contracts. Features sophisticated pattern detection algorithms, comprehensive risk management, and real-time market analysis.

**Performance:** Backtested pattern recognition achieving 89.5% win rate on S/R bounce patterns across 247+ historical trades.

## 🛠️ Tech Stack

**Backend:** Python 3.10+, asyncio for concurrent operations  
**Trading API:** TopStepX Direct API integration  
**Data Processing:** Pandas, NumPy for market analysis  
**Real-time:** WebSocket connections for live market data  
**Communication:** Slack integration for monitoring & alerts  
**Web Dashboard:** React, Node.js, real-time updates  
**Database:** SQLite for pattern storage & trade logging  

## ✨ Key Features

### Pattern Recognition
- **S/R Bounce Detection** - Support/resistance level identification
- **Fair Value Gap (FVG) Analysis** - ICT-based pattern detection
- **Market Regime Classification** - Trend, range, breakout identification
- **Volume Confirmation** - 1.2x average volume threshold validation

### Risk Management
- **Max Daily Loss Limits** - Configurable stop-loss per day
- **Position Sizing** - Dynamic risk calculation per trade
- **Max Concurrent Positions** - Prevent overexposure
- **Stop After Losses** - Auto-halt after consecutive losses
- **Real-time P&L Tracking** - Live profit/loss monitoring

### Execution Engine
- **Multi-Bot Architecture** - ES, NQ, CL specialized bots
- **Order Management** - Market orders with slippage control
- **Position Monitoring** - Real-time position tracking
- **Automated Entry/Exit** - Rule-based execution
- **Conflict Detection** - Prevents duplicate orders

### Monitoring & Analytics
- **Web Dashboard** - Real-time trading dashboard
- **Slack Notifications** - Trade alerts and system status
- **Performance Metrics** - Win rate, P&L, drawdown analysis
- **Pattern Validation** - Out-of-sample testing
- **Trade Logging** - Complete audit trail

## 📁 Project Structure
```
xtrading/
├── trading_bot/          # Core trading engine
│   ├── strategy/        # Pattern detection algorithms
│   ├── execution/       # Order management
│   └── risk/            # Risk management
├── es_bot/              # E-mini S&P 500 bot
├── nq_bot/              # E-mini NASDAQ bot
├── cl_bot/              # Crude Oil bot
├── patterns/            # Pattern recognition modules
├── shared/              # Common utilities
├── web_platform/        # React dashboard
└── tests/               # Unit & integration tests
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- TopStepX API credentials
- Node.js 16+ (for web dashboard)

### Installation
```bash
# Clone repository
git clone https://github.com/SamShahinDev/algorithmic-trading-platform.git
cd algorithmic-trading-platform

# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API credentials

# Run tests
python -m pytest tests/
```

### Running the Platform
```bash
# Start single bot (paper trading)
python run_nq_bot.py

# Start all bots
python run_all_bots.py

# Start with web dashboard
cd web_platform && npm install && npm start
python run_all_bots.py
```

## 📊 Trading Strategy

### Pattern Detection
The system identifies high-probability setups using:
- **Support/Resistance Bounces** - Price rejection at key levels
- **Fair Value Gaps** - Imbalances in price action
- **Volume Confirmation** - Above-average volume validation
- **Market Structure** - Trend alignment

### Entry Criteria
- Pattern confirmation with volume
- Risk/reward ratio minimum 2:1
- Market conditions favorable (regime check)
- No conflicting signals from other patterns

### Exit Strategy
- Profit target: 5 points (NQ), 4 points (ES)
- Stop loss: 3 points (NQ), 2.5 points (ES)
- Trailing stops for extended moves
- Time-based exits for stale positions

### Risk Controls
- Maximum 5 trades per day
- Stop trading after 2 consecutive losses
- Max daily loss: $1,000 (configurable)
- Position size: 1 contract per trade

## 🔐 Security & Safety

- **Paper Trading Ready** - Test strategies risk-free
- **API Key Protection** - Environment variable storage
- **Rate Limiting** - Prevents API throttling
- **Position Verification** - Double-check before orders
- **Emergency Flatten** - Quick position liquidation
- **Lock Files** - Prevents duplicate bot instances

## 📈 Performance Metrics

**Backtested Results (2024 Q3-Q4):**
- Win Rate: 89.5% on S/R bounce pattern
- Total Trades: 247
- Average Win: +$120
- Average Loss: -$85
- Max Drawdown: $680

**Note:** Past performance does not guarantee future results. All statistics are from backtested data.

## 🧪 Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific bot
python test_nq_bot.py

# Backtest patterns
python validate_patterns_q3q4_2024.py

# Test API connection
python test_topstepx_connection.py
```

## 📝 Configuration

Edit `.env` file for customization:
```bash
# Trading parameters
MAX_DAILY_LOSS=1000
MAX_POSITIONS=2
RISK_PER_TRADE=0.01

# Bot behavior
TRADING_MODE=demo  # or 'live'
LOG_LEVEL=INFO
```

## ⚠️ Disclaimer

**FOR EDUCATIONAL AND PORTFOLIO DEMONSTRATION PURPOSES ONLY**

This software is provided for educational purposes to demonstrate algorithmic trading concepts, system architecture, and software engineering practices. 

- Not financial advice
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test thoroughly in paper trading before live deployment
- Consult licensed professionals before trading

## 🛠️ Built With

- **Python** - Core language
- **TopStepX API** - Futures trading platform
- **Pandas/NumPy** - Data analysis
- **React** - Web dashboard
- **Slack API** - Notifications
- **Pytest** - Testing framework

## 👤 Author

**Hussam Shahin**  
[LinkedIn](https://www.linkedin.com/in/hussamshahin) | [GitHub](https://github.com/SamShahinDev)

---

**Status:** Portfolio demonstration project showcasing quantitative trading system development, pattern recognition algorithms, and real-time market data processing.
