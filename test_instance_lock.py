#!/usr/bin/env python3
"""
Test script to verify single instance lock mechanism
"""

import sys
import time
import subprocess
from pathlib import Path

# Add path for imports
sys.path.append(str(Path(__file__).parent))

from utils.instance_lock import InstanceLock

def test_single_instance():
    """Test that only one instance can run"""
    print("=" * 60)
    print("TESTING SINGLE INSTANCE LOCK MECHANISM")
    print("=" * 60)
    
    # Test 1: First lock should succeed
    print("\n1. Testing first lock acquisition...")
    lock1 = InstanceLock("test_bot")
    if lock1.acquire():
        print("✅ First lock acquired successfully")
    else:
        print("❌ First lock failed (unexpected)")
        return False
    
    # Test 2: Second lock should fail
    print("\n2. Testing duplicate prevention...")
    lock2 = InstanceLock("test_bot")
    if lock2.acquire():
        print("❌ Second lock acquired (should have failed!)")
        lock1.release()
        return False
    else:
        print("✅ Second lock prevented (as expected)")
    
    # Test 3: Release and re-acquire
    print("\n3. Testing lock release and re-acquisition...")
    lock1.release()
    
    lock3 = InstanceLock("test_bot")
    if lock3.acquire():
        print("✅ Lock re-acquired after release")
        lock3.release()
    else:
        print("❌ Could not re-acquire after release")
        return False
    
    # Test 4: Multiple bot names
    print("\n4. Testing multiple bot isolation...")
    lock_nq = InstanceLock("nq_bot")
    lock_es = InstanceLock("es_bot")
    
    if lock_nq.acquire() and lock_es.acquire():
        print("✅ Different bots can run simultaneously")
        lock_nq.release()
        lock_es.release()
    else:
        print("❌ Different bots blocked each other")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    return True

def test_with_subprocess():
    """Test with actual subprocess to simulate real scenario"""
    print("\n" + "=" * 60)
    print("TESTING WITH SUBPROCESS")
    print("=" * 60)
    
    # Start first process
    print("\n1. Starting first subprocess...")
    proc1 = subprocess.Popen(
        [sys.executable, 'test_subprocess_lock.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it time to acquire lock
    time.sleep(0.5)
    
    # Try second process (should fail)
    print("2. Starting second subprocess (should be blocked)...")
    proc2 = subprocess.Popen(
        [sys.executable, 'test_subprocess_lock.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for both to complete
    out1, err1 = proc1.communicate()
    out2, err2 = proc2.communicate()
    
    print("\nFirst process output:")
    print(out1.strip())
    if err1:
        print("First process errors:", err1)
    
    print("\nSecond process output:")
    print(out2.strip())
    if err2:
        print("Second process errors:", err2)
    
    if proc1.returncode == 0 and proc2.returncode != 0:
        print("\n✅ Subprocess test passed - second instance blocked")
        return True
    else:
        print(f"\n❌ Subprocess test failed (proc1: {proc1.returncode}, proc2: {proc2.returncode})")
        return False

if __name__ == "__main__":
    # Ensure locks directory exists
    Path("locks").mkdir(exist_ok=True)
    
    # Run tests
    if test_single_instance():
        if test_with_subprocess():
            print("\n✅ ALL SINGLE INSTANCE TESTS PASSED!")
            print("\nThe dual bot instance problem that caused $490 in losses")
            print("should now be prevented by this lock mechanism.")
            sys.exit(0)
    
    print("\n❌ Some tests failed")
    sys.exit(1)