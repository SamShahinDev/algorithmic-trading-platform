#!/bin/bash

echo "===================================="
echo "FORCE UNLOCK - EMERGENCY RECOVERY"
echo "===================================="
echo ""
echo "WARNING: This will forcefully remove all lock files!"
echo "Only use this if you're sure no bots are running."
echo ""

# Check for running bots
echo "Checking for running bot processes..."
echo "------------------------------------"

NQ_RUNNING=$(pgrep -f "intelligent_trading_bot" | wc -l)
ES_RUNNING=$(pgrep -f "es_bot/main.py" | wc -l)
CL_RUNNING=$(pgrep -f "cl_bot/main.py" | wc -l)
COMBINED_RUNNING=$(pgrep -f "run_es_cl_bots" | wc -l)

if [ $NQ_RUNNING -gt 0 ]; then
    echo "⚠️  WARNING: Found $NQ_RUNNING NQ bot process(es) running!"
fi

if [ $ES_RUNNING -gt 0 ]; then
    echo "⚠️  WARNING: Found $ES_RUNNING ES bot process(es) running!"
fi

if [ $CL_RUNNING -gt 0 ]; then
    echo "⚠️  WARNING: Found $CL_RUNNING CL bot process(es) running!"
fi

if [ $COMBINED_RUNNING -gt 0 ]; then
    echo "⚠️  WARNING: Found $COMBINED_RUNNING combined ES/CL process(es) running!"
fi

TOTAL_RUNNING=$((NQ_RUNNING + ES_RUNNING + CL_RUNNING + COMBINED_RUNNING))

if [ $TOTAL_RUNNING -gt 0 ]; then
    echo ""
    echo "❌ SAFETY CHECK FAILED: $TOTAL_RUNNING bot process(es) still running!"
    echo ""
    echo "Options:"
    echo "1. Run './stop_production.sh' to stop all bots properly"
    echo "2. Run 'pkill -f trading_bot && pkill -f main.py' to force kill"
    echo "3. Use '--force' flag to override safety (DANGEROUS)"
    echo ""
    
    if [ "$1" != "--force" ]; then
        echo "Aborting force unlock for safety."
        exit 1
    else
        echo "⚠️  --force flag detected, proceeding despite running processes..."
    fi
else
    echo "✓ No bot processes detected"
fi

echo ""
echo "Lock Files Status:"
echo "------------------------------------"

# Check and display lock files
if [ -d "locks" ]; then
    LOCK_COUNT=$(ls -1 locks/*.lock 2>/dev/null | wc -l)
    INFO_COUNT=$(ls -1 locks/*.info 2>/dev/null | wc -l)
    
    if [ $LOCK_COUNT -gt 0 ] || [ $INFO_COUNT -gt 0 ]; then
        echo "Found lock files:"
        ls -la locks/ 2>/dev/null
        
        # Show lock info content
        for info_file in locks/*.info; do
            if [ -f "$info_file" ]; then
                echo ""
                echo "Content of $info_file:"
                cat "$info_file" | python3 -m json.tool 2>/dev/null || cat "$info_file"
            fi
        done
    else
        echo "No lock files found"
    fi
else
    echo "No locks directory found"
fi

echo ""
echo "Proceeding with force unlock..."
echo "------------------------------------"

# Remove all lock files
if [ -d "locks" ]; then
    echo "Removing lock files..."
    rm -f locks/*.lock 2>/dev/null
    rm -f locks/*.info 2>/dev/null
    
    # Check if removal was successful
    REMAINING=$(ls -1 locks/*.lock locks/*.info 2>/dev/null | wc -l)
    if [ $REMAINING -eq 0 ]; then
        echo "✓ All lock files removed successfully"
    else
        echo "⚠️  Warning: $REMAINING lock file(s) could not be removed"
        ls -la locks/*.lock locks/*.info 2>/dev/null
    fi
else
    echo "No locks directory to clean"
fi

# Clean up stale PID files
echo ""
echo "Cleaning PID files..."
echo "------------------------------------"

for pid_file in logs/*.pid; do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        BOT_NAME=$(basename "$pid_file" .pid)
        
        if kill -0 $PID 2>/dev/null; then
            echo "⚠️  Process $PID for $BOT_NAME is still running!"
        else
            echo "Removing stale PID file for $BOT_NAME (PID $PID not running)"
            rm -f "$pid_file"
        fi
    fi
done

# Verify cleanup
echo ""
echo "Verification:"
echo "------------------------------------"

# Re-run the verify script if available
if [ -f "verify_single_instance.py" ]; then
    echo "Running instance verification..."
    python3 verify_single_instance.py
else
    echo "verify_single_instance.py not found, skipping verification"
fi

echo ""
echo "===================================="
echo "Force unlock complete!"
echo ""
echo "Next steps:"
echo "1. Check that no bots are actually running: ps aux | grep -E 'trading_bot|main.py'"
echo "2. If safe, restart bots: ./start_production.sh"
echo "3. Monitor logs for any issues: tail -f logs/*.log"
echo "===================================="