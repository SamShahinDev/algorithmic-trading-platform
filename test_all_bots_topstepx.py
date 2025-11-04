#!/usr/bin/env python3
"""
Test script to verify all three bots (NQ, ES, CL) connect to TopStepX
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load TopStepX credentials
load_dotenv('web_platform/backend/.env.topstepx')

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_platform', 'backend'))

async def test_nq_bot():
    """Test NQ bot connection to TopStepX"""
    print("\n" + "="*60)
    print("Testing NQ Bot (EnhancedSmartScalper)")
    print("="*60)
    
    try:
        from web_platform.backend.agents.smart_scalper_enhanced import EnhancedSmartScalper
        
        nq_bot = EnhancedSmartScalper()
        success = await nq_bot.initialize()
        
        if success:
            print("✅ NQ Bot initialized successfully")
            print(f"   Account ID: {nq_bot.account_id}")
            print(f"   Position: {nq_bot.get_position_status()}")
        else:
            print("❌ NQ Bot initialization failed")
            
        return success
        
    except Exception as e:
        print(f"❌ NQ Bot error: {e}")
        return False

async def test_es_bot():
    """Test ES bot connection to TopStepX"""
    print("\n" + "="*60)
    print("Testing ES Bot")
    print("="*60)
    
    try:
        from es_bot.es_bot import ESBot
        
        es_bot = ESBot()
        
        # Connect to TopStepX
        if hasattr(es_bot, 'connect_to_topstepx'):
            connected = await es_bot.connect_to_topstepx()
            
            if connected:
                print("✅ ES Bot connected to TopStepX")
                print(f"   Account ID: {es_bot.config.get('topstepx_account_id')}")
                print(f"   Patterns loaded: {len(es_bot.patterns)}")
                print(f"   Pattern names: {[p['name'] for p in es_bot.patterns]}")
            else:
                print("❌ ES Bot failed to connect to TopStepX")
                
            return connected
        else:
            print("⚠️  ES Bot doesn't have TopStepX connection method")
            return False
            
    except Exception as e:
        print(f"❌ ES Bot error: {e}")
        return False

async def test_cl_bot():
    """Test CL bot connection to TopStepX"""
    print("\n" + "="*60)
    print("Testing CL Bot")
    print("="*60)
    
    try:
        from cl_bot.cl_bot import CLBot
        
        cl_bot = CLBot()
        
        # Connect to TopStepX
        if hasattr(cl_bot, 'connect_to_topstepx'):
            connected = await cl_bot.connect_to_topstepx()
            
            if connected:
                print("✅ CL Bot connected to TopStepX")
                print(f"   Account ID: {cl_bot.config.get('topstepx_account_id')}")
                print(f"   Patterns loaded: {len(cl_bot.patterns)}")
                print(f"   Pattern names: {[p['name'] for p in cl_bot.patterns]}")
            else:
                print("❌ CL Bot failed to connect to TopStepX")
                
            return connected
        else:
            print("⚠️  CL Bot doesn't have TopStepX connection method")
            return False
            
    except Exception as e:
        print(f"❌ CL Bot error: {e}")
        return False

async def main():
    """Test all bots"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TopStepX Multi-Bot Connection Test                   ║
    ║     Testing NQ, ES, and CL Bots                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check credentials
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    username = os.getenv('TOPSTEPX_USERNAME')
    
    print(f"TopStepX Configuration:")
    print(f"  Username: {username}")
    print(f"  Account ID: {account_id}")
    
    # Test all bots
    results = {}
    
    results['NQ'] = await test_nq_bot()
    results['ES'] = await test_es_bot()
    results['CL'] = await test_cl_bot()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for bot, success in results.items():
        status = "✅ Connected" if success else "❌ Failed"
        print(f"{bot} Bot: {status}")
    
    all_connected = all(results.values())
    if all_connected:
        print("\n🎉 All bots successfully connected to TopStepX!")
    else:
        print("\n⚠️  Some bots failed to connect. Check the errors above.")
    
    return all_connected

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)