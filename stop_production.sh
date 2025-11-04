#!/bin/bash
"""
Graceful Shutdown Script for XTRADING Multi-Bot System
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "XTRADING Graceful Shutdown"
echo "=================================="

# Write control files to stop trading first
echo "Sending stop trading signals..."
echo '{"action": "stop_trading"}' > logs/nq_bot.control.json
echo '{"action": "stop_trading"}' > logs/es_bot.control.json
echo '{"action": "stop_trading"}' > logs/cl_bot.control.json

echo "Waiting for orders to cancel..."
sleep 5

# Check if Docker is running
if docker-compose ps 2>/dev/null | grep -q "xtrading"; then
    echo -e "${YELLOW}Stopping Docker containers...${NC}"
    
    # Stop containers gracefully
    docker-compose stop
    
    echo -e "${GREEN}Docker containers stopped${NC}"
    
else
    echo -e "${YELLOW}Stopping processes...${NC}"
    
    # Stop NQ bot
    if [ -f logs/nq_bot.pid ]; then
        NQ_PID=$(cat logs/nq_bot.pid)
        echo "Stopping NQ bot (PID: $NQ_PID)..."
        kill -TERM $NQ_PID 2>/dev/null
        sleep 2
        kill -9 $NQ_PID 2>/dev/null
        rm logs/nq_bot.pid
        echo -e "${GREEN}✓ NQ bot stopped${NC}"
    else
        echo "NQ bot PID file not found, trying pkill..."
        pkill -TERM -f "intelligent_trading_bot_fixed_v2.py"
        sleep 2
        pkill -9 -f "intelligent_trading_bot_fixed_v2.py" 2>/dev/null
    fi
    
    # Stop ES bot
    if [ -f logs/es_bot.pid ]; then
        ES_PID=$(cat logs/es_bot.pid)
        echo "Stopping ES bot (PID: $ES_PID)..."
        kill -TERM $ES_PID 2>/dev/null
        sleep 2
        kill -9 $ES_PID 2>/dev/null
        rm logs/es_bot.pid
        echo -e "${GREEN}✓ ES bot stopped${NC}"
    else
        echo "ES bot PID file not found, trying pkill..."
        pkill -TERM -f "es_bot/main.py"
        sleep 2
        pkill -9 -f "es_bot/main.py" 2>/dev/null
    fi
    
    # Stop CL bot
    if [ -f logs/cl_bot.pid ]; then
        CL_PID=$(cat logs/cl_bot.pid)
        echo "Stopping CL bot (PID: $CL_PID)..."
        kill -TERM $CL_PID 2>/dev/null
        sleep 2
        kill -9 $CL_PID 2>/dev/null
        rm logs/cl_bot.pid
        echo -e "${GREEN}✓ CL bot stopped${NC}"
    else
        echo "CL bot PID file not found, trying pkill..."
        pkill -TERM -f "cl_bot/main.py"
        sleep 2
        pkill -9 -f "cl_bot/main.py" 2>/dev/null
    fi
    
    # Stop combined ES/CL bot if still running
    echo "Stopping any combined ES/CL bot..."
    pkill -TERM -f "run_es_cl_bots.py"
    sleep 2
    pkill -9 -f "run_es_cl_bots.py" 2>/dev/null
    
    # Stop monitor last
    echo "Stopping monitor..."
    pkill -TERM -f "production_monitor.py"
    sleep 2
    pkill -9 -f "production_monitor.py" 2>/dev/null
    
    echo -e "${GREEN}All processes stopped${NC}"
fi

# Clean up lock files
echo "Cleaning up lock files..."
rm -f logs/*.lock
rm -f logs/*.control.json
rm -f logs/*.pid

# Create kill switch
echo "Creating kill switch..."
echo '{"kill_switch": true, "stopped_at": "'$(date)'", "reason": "Manual shutdown", "restart_allowed": false}' > logs/GLOBAL_KILL_SWITCH.json
echo -e "${GREEN}✓ Kill switch activated${NC}"

# Check if any positions are still open
if [ -f "logs/monitor_status.json" ]; then
    echo ""
    echo "Last known status:"
    cat logs/monitor_status.json | python3 -m json.tool | grep -E '"positions"|"ready"' || true
fi

# Verify all stopped
echo ""
echo "Verifying shutdown..."

# Check for any remaining processes
REMAINING=$(ps aux | grep -E "trading_bot|nq_bot|es_bot|cl_bot|production_monitor|run_es_cl" | grep -v grep | wc -l)

if [ "$REMAINING" -eq "0" ]; then
    echo -e "${GREEN}✓ All bots successfully stopped${NC}"
else
    echo -e "${YELLOW}⚠ Warning: $REMAINING process(es) may still be running${NC}"
    echo "Running processes:"
    ps aux | grep -E "trading_bot|nq_bot|es_bot|cl_bot|production_monitor|run_es_cl" | grep -v grep
fi

echo ""
echo -e "${GREEN}Shutdown complete${NC}"
echo "=================================="
echo "To restart, remove kill switch and run:"
echo "  rm logs/GLOBAL_KILL_SWITCH.json"
echo "  ./start_production.sh"
echo "=================================="