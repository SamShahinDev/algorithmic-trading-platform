"""
Test the fixed dashboard
"""

import asyncio
from playwright.async_api import async_playwright
import time

async def test_dashboard():
    """Test the fixed dashboard"""
    
    async with async_playwright() as p:
        # Launch browser in visible mode
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("🚀 Opening dashboard...")
        await page.goto('http://localhost:8000/frontend/')
        
        # Force reload to get latest changes
        await page.reload()
        print("🔄 Page reloaded with fixes")
        
        # Wait for chart to load
        await page.wait_for_timeout(5000)
        
        # Take screenshot
        await page.screenshot(path='dashboard_fixed.png')
        print("📸 Screenshot saved as dashboard_fixed.png")
        
        # Monitor for 15 seconds
        print("\n👀 Monitoring for 15 seconds...")
        print("   Watch the browser window to see if chart is stable")
        
        for i in range(3):
            await page.wait_for_timeout(5000)
            
            # Get current price
            price_element = await page.query_selector('#current-price')
            if price_element:
                price = await price_element.inner_text()
                print(f"   [{(i+1)*5}s] Price: {price}")
            
            # Check for chart iframe
            frames = page.frames
            print(f"   [{(i+1)*5}s] Frames on page: {len(frames)}")
        
        print("\n✅ Test complete - check the browser window")
        print("   The chart should be stable now (not reloading)")
        print("   Using QQQ as proxy for NQ futures")
        
        # Keep browser open for manual inspection
        print("\n📌 Browser will stay open for 10 more seconds...")
        await page.wait_for_timeout(10000)
        
        await browser.close()
        print("🏁 Done!")

if __name__ == "__main__":
    asyncio.run(test_dashboard())