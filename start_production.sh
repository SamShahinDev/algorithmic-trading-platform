#!/bin/bash
"""
Production Startup Script for XTRADING Multi-Bot System
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "XTRADING Production Startup"
echo "=================================="

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then 
   echo -e "${YELLOW}Warning: Running as root is not recommended${NC}"
fi

# CRITICAL: Pre-start safety checks
echo "Running pre-start safety checks..."
echo "----------------------------------"

# Check for existing bot processes
EXISTING_BOTS=0
if pgrep -f "intelligent_trading_bot" > /dev/null; then
    echo -e "${RED}✗ NQ bot is already running!${NC}"
    EXISTING_BOTS=$((EXISTING_BOTS + 1))
fi

if pgrep -f "es_bot/main.py" > /dev/null; then
    echo -e "${RED}✗ ES bot is already running!${NC}"
    EXISTING_BOTS=$((EXISTING_BOTS + 1))
fi

if pgrep -f "cl_bot/main.py" > /dev/null; then
    echo -e "${RED}✗ CL bot is already running!${NC}"
    EXISTING_BOTS=$((EXISTING_BOTS + 1))
fi

if pgrep -f "run_es_cl_bots" > /dev/null; then
    echo -e "${RED}✗ Combined ES/CL bot is already running!${NC}"
    EXISTING_BOTS=$((EXISTING_BOTS + 1))
fi

if [ $EXISTING_BOTS -gt 0 ]; then
    echo ""
    echo -e "${RED}ERROR: $EXISTING_BOTS bot(s) already running!${NC}"
    echo ""
    echo "Options:"
    echo "1. Run './stop_production.sh' to stop existing bots"
    echo "2. Run './check_bots_status.sh' to see what's running"
    echo "3. Run './force_unlock.sh' if bots crashed but locks remain"
    echo ""
    echo "Aborting startup for safety."
    exit 1
fi

# Check for stale lock files
if [ -d "locks" ]; then
    LOCK_COUNT=$(ls -1 locks/*.lock locks/*.info 2>/dev/null | wc -l)
    if [ $LOCK_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $LOCK_COUNT lock file(s)${NC}"
        echo "Checking if locks are stale..."
        
        # Run verification script
        if [ -f "verify_single_instance.py" ]; then
            python3 verify_single_instance.py
            if [ $? -ne 0 ]; then
                echo -e "${RED}Lock verification failed!${NC}"
                echo "Run './force_unlock.sh' to clean up stale locks"
                exit 1
            fi
        fi
    fi
fi

echo -e "${GREEN}✓ Pre-start safety checks passed${NC}"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs state locks

# Clean old heartbeat files
echo "Cleaning old heartbeat files..."
rm -f logs/*.heartbeat.json
rm -f logs/GLOBAL_KILL_SWITCH.json

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker detected - using container deployment${NC}"
    
    # Build Docker images
    echo "Building Docker images..."
    docker-compose build
    
    # Start services
    echo "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check service status
    docker-compose ps
    
    echo -e "${GREEN}Services started successfully!${NC}"
    echo ""
    echo "Monitor logs with:"
    echo "  docker-compose logs -f"
    echo ""
    echo "Stop services with:"
    echo "  docker-compose down"
    
else
    echo -e "${YELLOW}Docker not found - using process deployment${NC}"
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    # Stop any existing bots
    echo "Stopping existing bots..."
    pkill -f "intelligent_trading_bot_fixed_v2.py" 2>/dev/null
    pkill -f "run_es_cl_bots.py" 2>/dev/null
    pkill -f "es_bot/main.py" 2>/dev/null
    pkill -f "cl_bot/main.py" 2>/dev/null
    pkill -f "production_monitor.py" 2>/dev/null
    
    sleep 2
    
    # Start monitor first
    echo "Starting production monitor..."
    nohup python3 production_monitor.py >> logs/production_monitor.log 2>&1 &
    MONITOR_PID=$!
    echo "Monitor started with PID: $MONITOR_PID"
    
    # Start NQ bot
    echo "Starting NQ bot..."
    nohup python3 trading_bot/intelligent_trading_bot_fixed_v2.py >> logs/nq_bot.log 2>&1 &
    NQ_PID=$!
    echo "$NQ_PID" > logs/nq_bot.pid
    echo "NQ bot started with PID: $NQ_PID"
    
    # Start ES bot
    echo "Starting ES bot..."
    nohup python3 es_bot/main.py >> logs/es_bot.log 2>&1 &
    ES_PID=$!
    echo "$ES_PID" > logs/es_bot.pid
    echo "ES bot started with PID: $ES_PID"
    
    # Start CL bot
    echo "Starting CL bot..."
    nohup python3 cl_bot/main.py >> logs/cl_bot.log 2>&1 &
    CL_PID=$!
    echo "$CL_PID" > logs/cl_bot.pid
    echo "CL bot started with PID: $CL_PID"
    
    # Wait a bit for processes to stabilize
    sleep 5
    
    # Verify processes are running
    echo ""
    echo "Verifying processes..."
    
    if ps -p $MONITOR_PID > /dev/null; then
        echo -e "${GREEN}✓ Monitor running${NC}"
    else
        echo -e "${RED}✗ Monitor failed to start${NC}"
    fi
    
    if ps -p $NQ_PID > /dev/null; then
        echo -e "${GREEN}✓ NQ bot running${NC}"
    else
        echo -e "${RED}✗ NQ bot failed to start${NC}"
    fi
    
    if ps -p $ES_PID > /dev/null; then
        echo -e "${GREEN}✓ ES bot running${NC}"
    else
        echo -e "${RED}✗ ES bot failed to start${NC}"
    fi
    
    if ps -p $CL_PID > /dev/null; then
        echo -e "${GREEN}✓ CL bot running${NC}"
    else
        echo -e "${RED}✗ CL bot failed to start${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}Bots started successfully!${NC}"
    echo ""
    echo "Monitor logs with:"
    echo "  tail -f logs/production_monitor.log"
    echo "  tail -f logs/nq_bot.log"
    echo "  tail -f logs/es_bot.log"
    echo "  tail -f logs/cl_bot.log"
    echo ""
    echo "Stop all bots with:"
    echo "  ./stop_production.sh"
fi

echo ""
echo "Monitor status at: logs/monitor_status.json"
echo "=================================="