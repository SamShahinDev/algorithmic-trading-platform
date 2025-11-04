#!/usr/bin/env python3
"""
Verify no duplicate bot instances are running
"""
import psutil
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def check_duplicate_bots():
    """Check for duplicate trading bot processes"""
    bot_processes = defaultdict(list)
    
    print("=" * 60)
    print("CHECKING FOR DUPLICATE BOT INSTANCES")
    print("=" * 60)
    
    # Scan all processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            # Skip grep and this script
            if 'grep' in cmdline or 'verify_single_instance' in cmdline:
                continue
            
            # Identify trading bots
            if 'intelligent_trading_bot' in cmdline or 'fixed_v2' in cmdline:
                bot_processes['NQ'].append(proc)
            elif 'es_bot/main.py' in cmdline or 'es_bot_enhanced' in cmdline:
                bot_processes['ES'].append(proc)
            elif 'cl_bot/main.py' in cmdline or 'cl_bot_enhanced' in cmdline:
                bot_processes['CL'].append(proc)
            elif 'run_es_cl_bots' in cmdline:
                bot_processes['ES_CL_Combined'].append(proc)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Check lock files
    print("\nLock File Status:")
    print("-" * 40)
    locks_dir = Path("locks")
    if locks_dir.exists():
        for lock_file in locks_dir.glob("*.info"):
            try:
                with open(lock_file) as f:
                    info = json.load(f)
                bot_name = info['bot_name']
                pid = info['pid']
                timestamp = info['timestamp']
                
                # Check if process is still running
                if psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        print(f"✓ {bot_name}: PID {pid} (Started: {timestamp})")
                    except psutil.NoSuchProcess:
                        print(f"✗ {bot_name}: Stale lock (PID {pid} not running)")
                else:
                    print(f"✗ {bot_name}: Stale lock (PID {pid} not found)")
                    
            except Exception as e:
                print(f"✗ Error reading {lock_file}: {e}")
    else:
        print("No lock directory found")
    
    # Report findings
    print("\nProcess Analysis:")
    print("-" * 40)
    
    issues_found = False
    for bot_type, processes in bot_processes.items():
        if len(processes) == 0:
            continue
        elif len(processes) == 1:
            proc = processes[0]
            create_time = datetime.fromtimestamp(proc.info['create_time'])
            print(f"✓ {bot_type}: Single instance (PID: {proc.pid}, Started: {create_time})")
        else:
            issues_found = True
            print(f"❌ DUPLICATE {bot_type} BOTS DETECTED! ({len(processes)} instances)")
            for proc in processes:
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                print(f"   PID: {proc.pid}, Started: {create_time}")
                print(f"   Command: {' '.join(proc.cmdline()[:3])}...")
    
    if not any(bot_processes.values()):
        print("No bot processes found running")
    
    # Check for orphaned processes
    print("\nOrphaned Process Check:")
    print("-" * 40)
    
    orphans = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if ('trading_bot' in cmdline or 'main.py' in cmdline) and \
               proc.pid not in [p.pid for procs in bot_processes.values() for p in procs]:
                orphans.append(proc)
        except:
            continue
    
    if orphans:
        print(f"⚠️  Found {len(orphans)} potential orphaned processes:")
        for proc in orphans:
            print(f"   PID: {proc.pid}, Command: {' '.join(proc.cmdline()[:3])}...")
    else:
        print("✓ No orphaned processes found")
    
    # Summary
    print("\n" + "=" * 60)
    if issues_found:
        print("❌ ISSUES FOUND - DUPLICATE INSTANCES DETECTED")
        print("\nTo fix:")
        print("1. Run: ./stop_production.sh")
        print("2. Run: pkill -f trading_bot")
        print("3. Run: rm -rf locks/*")
        print("4. Run: ./start_production.sh")
    else:
        print("✅ No duplicate bot instances found")
    print("=" * 60)
    
    return issues_found

def kill_duplicates(bot_type=None):
    """Kill duplicate bot instances, keeping the oldest"""
    bot_processes = defaultdict(list)
    
    # Find all bot processes
    for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            if 'intelligent_trading_bot' in cmdline or 'fixed_v2' in cmdline:
                if not bot_type or bot_type == 'NQ':
                    bot_processes['NQ'].append(proc)
            elif 'es_bot/main.py' in cmdline:
                if not bot_type or bot_type == 'ES':
                    bot_processes['ES'].append(proc)
            elif 'cl_bot/main.py' in cmdline:
                if not bot_type or bot_type == 'CL':
                    bot_processes['CL'].append(proc)
        except:
            continue
    
    killed = []
    for bot, processes in bot_processes.items():
        if len(processes) > 1:
            # Sort by create time, keep oldest
            processes.sort(key=lambda p: p.info['create_time'])
            keeper = processes[0]
            
            print(f"Keeping {bot} PID {keeper.pid}, killing {len(processes)-1} duplicates")
            
            for proc in processes[1:]:
                try:
                    print(f"  Killing PID {proc.pid}")
                    proc.terminate()
                    killed.append(proc.pid)
                except:
                    pass
    
    return killed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify single bot instances')
    parser.add_argument('--kill', action='store_true', help='Kill duplicate instances')
    parser.add_argument('--bot', help='Specific bot to check (NQ, ES, CL)')
    
    args = parser.parse_args()
    
    if args.kill:
        killed = kill_duplicates(args.bot)
        if killed:
            print(f"\nKilled {len(killed)} duplicate processes")
            sys.exit(0)
        else:
            print("\nNo duplicates to kill")
            sys.exit(0)
    else:
        if check_duplicate_bots():
            sys.exit(1)
        sys.exit(0)