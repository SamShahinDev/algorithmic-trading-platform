#!/bin/bash
"""
Bot Status Check Script for XTRADING Multi-Bot System
Shows the status of all trading bots
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=================================="
echo -e "${BLUE}Trading Bot Status${NC}"
echo "=================================="
echo ""

# Check for kill switch
if [ -f logs/GLOBAL_KILL_SWITCH.json ]; then
    echo -e "${RED}⚠ KILL SWITCH ACTIVE${NC}"
    echo "Kill switch details:"
    cat logs/GLOBAL_KILL_SWITCH.json | python3 -m json.tool 2>/dev/null || cat logs/GLOBAL_KILL_SWITCH.json
    echo ""
fi

# Check NQ bot
echo -e "${BLUE}NQ Bot Status:${NC}"
if [ -f logs/nq_bot.pid ]; then
    NQ_PID=$(cat logs/nq_bot.pid)
    if kill -0 $NQ_PID 2>/dev/null; then
        echo -e "  ${GREEN}✓ Running${NC} (PID: $NQ_PID)"
        # Get memory usage
        MEM=$(ps -o rss= -p $NQ_PID | awk '{printf "%.1f MB", $1/1024}')
        echo "    Memory: $MEM"
    else
        echo -e "  ${RED}✗ Not running${NC} (stale PID file)"
        rm logs/nq_bot.pid
    fi
else
    # Check if running without PID file
    if pgrep -f "intelligent_trading_bot_fixed_v2.py" > /dev/null; then
        echo -e "  ${YELLOW}⚠ Running without PID file${NC}"
    else
        echo -e "  ${RED}✗ Not running${NC}"
    fi
fi

# Check ES bot
echo -e "${BLUE}ES Bot Status:${NC}"
if [ -f logs/es_bot.pid ]; then
    ES_PID=$(cat logs/es_bot.pid)
    if kill -0 $ES_PID 2>/dev/null; then
        echo -e "  ${GREEN}✓ Running${NC} (PID: $ES_PID)"
        MEM=$(ps -o rss= -p $ES_PID | awk '{printf "%.1f MB", $1/1024}')
        echo "    Memory: $MEM"
    else
        echo -e "  ${RED}✗ Not running${NC} (stale PID file)"
        rm logs/es_bot.pid
    fi
else
    if pgrep -f "es_bot/main.py" > /dev/null; then
        echo -e "  ${YELLOW}⚠ Running without PID file${NC}"
    else
        echo -e "  ${RED}✗ Not running${NC}"
    fi
fi

# Check CL bot
echo -e "${BLUE}CL Bot Status:${NC}"
if [ -f logs/cl_bot.pid ]; then
    CL_PID=$(cat logs/cl_bot.pid)
    if kill -0 $CL_PID 2>/dev/null; then
        echo -e "  ${GREEN}✓ Running${NC} (PID: $CL_PID)"
        MEM=$(ps -o rss= -p $CL_PID | awk '{printf "%.1f MB", $1/1024}')
        echo "    Memory: $MEM"
    else
        echo -e "  ${RED}✗ Not running${NC} (stale PID file)"
        rm logs/cl_bot.pid
    fi
else
    if pgrep -f "cl_bot/main.py" > /dev/null; then
        echo -e "  ${YELLOW}⚠ Running without PID file${NC}"
    else
        echo -e "  ${RED}✗ Not running${NC}"
    fi
fi

# Check for combined ES/CL bot (legacy)
if pgrep -f "run_es_cl_bots.py" > /dev/null; then
    echo -e "${YELLOW}⚠ Legacy combined ES/CL bot is running${NC}"
    echo "  Consider stopping it and using separate bots"
fi

# Check Production Monitor
echo -e "${BLUE}Production Monitor:${NC}"
if pgrep -f "production_monitor.py" > /dev/null; then
    echo -e "  ${GREEN}✓ Running${NC}"
else
    echo -e "  ${RED}✗ Not running${NC}"
fi

echo ""
echo "=================================="
echo -e "${BLUE}Recent Activity${NC}"
echo "=================================="

# Show recent NQ activity
if [ -f logs/nq_bot.log ]; then
    echo -e "${BLUE}NQ Bot:${NC}"
    LAST_LINE=$(tail -1 logs/nq_bot.log 2>/dev/null)
    if [ ! -z "$LAST_LINE" ]; then
        TIMESTAMP=$(echo "$LAST_LINE" | cut -d' ' -f1-2)
        echo "  Last activity: $TIMESTAMP"
    fi
fi

# Show recent ES activity
if [ -f logs/es_bot.log ]; then
    echo -e "${BLUE}ES Bot:${NC}"
    LAST_LINE=$(tail -1 logs/es_bot.log 2>/dev/null)
    if [ ! -z "$LAST_LINE" ]; then
        TIMESTAMP=$(echo "$LAST_LINE" | cut -d' ' -f1-2)
        echo "  Last activity: $TIMESTAMP"
    fi
fi

# Show recent CL activity
if [ -f logs/cl_bot.log ]; then
    echo -e "${BLUE}CL Bot:${NC}"
    LAST_LINE=$(tail -1 logs/cl_bot.log 2>/dev/null)
    if [ ! -z "$LAST_LINE" ]; then
        TIMESTAMP=$(echo "$LAST_LINE" | cut -d' ' -f1-2)
        echo "  Last activity: $TIMESTAMP"
    fi
fi

echo ""
echo "=================================="
echo -e "${BLUE}Monitor Status${NC}"
echo "=================================="

# Check monitor status file
if [ -f logs/monitor_status.json ]; then
    echo "Last monitor update:"
    cat logs/monitor_status.json | python3 -c "
import json, sys
from datetime import datetime

try:
    data = json.load(sys.stdin)
    
    # Parse timestamp
    if 'timestamp' in data:
        ts = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        print(f'  Time: {ts.strftime(\"%Y-%m-%d %H:%M:%S\")}')
    
    # Show bot status
    if 'bots' in data:
        for bot, status in data['bots'].items():
            if status.get('ready'):
                print(f'  {bot}: ✓ Ready')
            else:
                print(f'  {bot}: ✗ Not ready')
            
            if 'position' in status and status['position'] != 0:
                print(f'    Position: {status[\"position\"]} contracts')
            
            if 'daily_pnl' in status and status['daily_pnl'] != 0:
                print(f'    Daily P&L: \${status[\"daily_pnl\"]:.2f}')
    
except Exception as e:
    print(f'  Error parsing monitor status: {e}')
" 2>/dev/null || echo "  Unable to parse monitor status"
else
    echo "No monitor status file found"
fi

echo ""
echo "=================================="
echo -e "${BLUE}Commands${NC}"
echo "=================================="
echo "Start all bots:  ./start_production.sh"
echo "Stop all bots:   ./stop_production.sh"
echo "View NQ logs:    tail -f logs/nq_bot.log"
echo "View ES logs:    tail -f logs/es_bot.log"
echo "View CL logs:    tail -f logs/cl_bot.log"
echo "==================================