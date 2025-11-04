#!/bin/bash

# Multi-Market Trading System Launcher

echo "========================================"
echo "Multi-Market Trading System"
echo "========================================"

# Check Python version
python3 --version

# Install required packages if needed
pip3 install pandas numpy zstandard asyncio pathlib

# Display menu
echo ""
echo "Select trading mode:"
echo "1. Paper Trading (Simulated)"
echo "2. Backtesting (Historical Data)"
echo "3. Live Trading (Real Money - USE WITH CAUTION)"
echo "4. Quick Backtest (Last 10 days)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Starting Paper Trading Mode..."
        python3 main.py --mode paper
        ;;
    2)
        read -p "Enter start date (YYYY-MM-DD) or press Enter for all data: " start_date
        read -p "Enter end date (YYYY-MM-DD) or press Enter for all data: " end_date
        
        if [ -z "$start_date" ]; then
            python3 main.py --mode backtest
        else
            python3 main.py --mode backtest --start-date "$start_date" --end-date "$end_date"
        fi
        ;;
    3)
        echo "WARNING: Live trading mode will use real money!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python3 main.py --mode live
        else
            echo "Live trading cancelled"
        fi
        ;;
    4)
        echo "Running quick backtest on last 10 days..."
        python3 main.py --mode backtest --start-date "2025-08-18"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac