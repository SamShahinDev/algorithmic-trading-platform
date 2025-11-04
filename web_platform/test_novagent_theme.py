"""
Test the NovaGent themed dashboard
"""

import asyncio
from playwright.async_api import async_playwright

async def test_themed_dashboard():
    """Test the NovaGent themed dashboard"""
    
    async with async_playwright() as p:
        # Launch browser in visible mode
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("🎨 Opening NovaGent themed dashboard...")
        await page.goto('http://localhost:8000/frontend/')
        
        # Force reload to get latest changes
        await page.reload()
        print("🔄 Page reloaded with NovaGent theme")
        
        # Wait for everything to load
        await page.wait_for_timeout(3000)
        
        # Take screenshot
        await page.screenshot(path='dashboard_novagent_theme.png', full_page=True)
        print("📸 Screenshot saved as dashboard_novagent_theme.png")
        
        print("\n✨ NovaGent Theme Applied:")
        print("   • Purple gradient background")
        print("   • Glass morphism cards")
        print("   • Subtle purple accents")
        print("   • Backdrop blur effects")
        
        print("\n📌 Browser will stay open for 15 seconds to view...")
        await page.wait_for_timeout(15000)
        
        await browser.close()
        print("🏁 Done!")

if __name__ == "__main__":
    asyncio.run(test_themed_dashboard())