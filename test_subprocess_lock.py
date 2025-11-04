#!/usr/bin/env python3
"""Test subprocess for lock mechanism"""
import sys
import time
from utils.instance_lock import InstanceLock

lock = InstanceLock("subprocess_test")
if lock.acquire():
    print("SUBPROCESS: Lock acquired", flush=True)
    time.sleep(2)
    lock.release()
    print("SUBPROCESS: Lock released", flush=True)
    sys.exit(0)
else:
    print("SUBPROCESS: Could not acquire lock (blocked)", flush=True)
    sys.exit(1)