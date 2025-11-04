import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from parent directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

print("=" * 60)
print("SDK AGENT CONFIGURATION CHECK")
print("=" * 60)

# Check Anthropic API
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
print(f"\n✓ Anthropic API Key: {'*' * 40}{anthropic_key[-10:] if anthropic_key else 'NOT FOUND'}")

# Check TopStepX API
topstepx_key = os.getenv('TOPSTEPX_API_KEY')
print(f"✓ TopStepX API Key: {'*' * 20}{topstepx_key[-10:] if topstepx_key else 'NOT FOUND'}")

# Check TopStepX Account ID
account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
print(f"✓ TopStepX Account ID: {account_id if account_id else 'NOT FOUND'}")

# Check Environment
environment = os.getenv('TOPSTEPX_ENVIRONMENT')
print(f"✓ TopStepX Environment: {environment if environment else 'NOT FOUND'}")

print("\n" + "=" * 60)

if anthropic_key and topstepx_key and account_id:
    print("✅ ALL REQUIRED CREDENTIALS CONFIGURED!")
else:
    print("❌ MISSING CREDENTIALS - CHECK .env FILE")
    
print("=" * 60)
