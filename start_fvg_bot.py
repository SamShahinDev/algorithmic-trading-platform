#!/usr/bin/env python3
"""
Simple FVG bot launcher that bypasses import issues
"""
import os
import sys
import asyncio

# Set up proper paths
xtrading_dir = "/Users/royaltyvixion/Documents/XTRADING"
nq_bot_dir = os.path.join(xtrading_dir, "nq_bot")

# Add to Python path
sys.path.insert(0, xtrading_dir)
sys.path.insert(0, nq_bot_dir)

# Set environment
os.environ["PYTHONPATH"] = f"{xtrading_dir}:{nq_bot_dir}"

# Change to correct directory
os.chdir(xtrading_dir)


async def main():
    """Main entry point for FVG bot"""
    from nq_bot.fvg_runner import FVGRunner

    runner = FVGRunner()
    await runner.run()


# Import and run the bot
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error starting FVG bot: {e}")
        import traceback
        traceback.print_exc()