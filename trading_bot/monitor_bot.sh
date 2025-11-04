#!/bin/bash

# Bot Monitoring Script
echo "="
echo "🤖 INTELLIGENT TRADING BOT MONITOR"
echo "="
echo "Practice Account: 10983875"
echo "Balance: \$149,882.20"
echo "="
echo

# Check if bot is running
if pgrep -f intelligent_trading_bot.py > /dev/null; then
    echo "✅ Bot Status: RUNNING"
    echo "Process ID: $(pgrep -f intelligent_trading_bot.py)"
else
    echo "❌ Bot Status: NOT RUNNING"
    echo "To start: python3 intelligent_trading_bot.py"
    exit 1
fi

echo
echo "📊 Recent Activity (Last 10 entries):"
echo "-"
tail -n 10 bot_output.log | grep -E "Confidence|Trade|Position|Pattern" || tail -n 10 bot_output.log

echo
echo "📈 Trading Statistics:"
echo "-"

# Count trades from log
TRADES=$(grep -c "Entering position" bot_output.log 2>/dev/null || echo "0")
echo "Total Trades: $TRADES"

# Get latest confidence
LATEST_CONFIDENCE=$(tail -n 100 bot_output.log | grep "Confidence:" | tail -n 1 | cut -d':' -f4 | cut -d'%' -f1 || echo "N/A")
echo "Latest Confidence: ${LATEST_CONFIDENCE}%"

# Check for any errors
ERROR_COUNT=$(tail -n 100 bot_output.log | grep -c ERROR || echo "0")
if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠️  Errors detected: $ERROR_COUNT"
    echo "Latest error:"
    tail -n 100 bot_output.log | grep ERROR | tail -n 1
fi

echo
echo "💡 Live Monitoring (Press Ctrl+C to stop):"
echo "-"
echo "Watching for trades and confidence updates..."
echo

# Live monitoring
tail -f bot_output.log | grep --line-buffered -E "Confidence|TRADE|Position|Pattern|ERROR" | while read line
do
    echo "[$(date '+%H:%M:%S')] $line"
done