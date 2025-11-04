# SDK Trading Bot - Management Guide

## 🤖 Bot is Running in Background

Your SDK Trading Agent is running persistently using `nohup`, which means:
- ✅ It will continue running even after you close your laptop
- ✅ It will survive terminal disconnections
- ✅ All output is logged to `logs/sdk_persistent.log`

---

## 📋 Quick Commands

### Check if Bot is Running
```bash
ps aux | grep "python main.py" | grep -v grep
```

### View Live Logs
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
tail -f logs/sdk_persistent.log
```

### Stop the Bot
```bash
pkill -f "python main.py"
```

### Start the Bot
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
source venv/bin/activate
nohup python main.py > logs/sdk_persistent.log 2>&1 &
```

### Restart the Bot
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
pkill -f "python main.py"
sleep 2
source venv/bin/activate
nohup python main.py > logs/sdk_persistent.log 2>&1 &
```

---

## 📊 Check Bot Status

### Quick Status Check
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
tail -n 20 logs/sdk_persistent.log | grep -E "Market Data|strategy_selector"
```

### Last 10 Market Updates
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
grep "📊 Market Data" logs/sdk_persistent.log | tail -n 10
```

### Check for Trades
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
grep -i "trade signal\|order placed" logs/sdk_persistent.log
```

### Check for Errors
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
grep -i "error\|exception\|failed" logs/sdk_persistent.log | tail -n 20
```

---

## 🔍 Current Configuration

- **Account**: PRAC-V2-XXXXX-XXXXXXXX
- **Environment**: DEMO
- **Contract**: CON.F.US.ENQ.Z25 (E-mini NASDAQ-100)
- **Strategies**: VWAP, Breakout, Momentum
- **Update Interval**: 60 seconds
- **Risk Limits**: $250 target, -$150 max loss, 8 max trades

---

## 🛠️ Troubleshooting

### Bot Not Running?
```bash
cd "/Users/royaltyvixion/Documents/XTRADING/sdk agent"
source venv/bin/activate
python main.py
# Check for any error messages
```

### Authentication Issues?
Check `.env` file has correct credentials:
```bash
cat /Users/royaltyvixion/Documents/XTRADING/.env | grep TOPSTEPX
```

### Market Data Issues?
```bash
tail -n 50 logs/sdk_persistent.log | grep -E "Retrieved|Failed"
```

---

## 📝 Notes

- The bot is designed to be selective - no trades is often correct behavior
- Strategies require specific market conditions (VWAP deviation, breakouts, momentum)
- All activity is logged to `logs/sdk_persistent.log`
- Bot runs 24/7 monitoring NQ futures market

---

## ⚠️ Before Closing Laptop

The bot is already running with `nohup`, so you can safely:
1. ✅ Close your terminal
2. ✅ Close your laptop
3. ✅ Disconnect from network (bot needs internet to trade)

The bot will continue running until you explicitly stop it or the system restarts.

**To verify it's still running after reopening laptop:**
```bash
ps aux | grep "python main.py" | grep -v grep
```
