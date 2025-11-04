"""
Capture and analyze the dashboard to diagnose chart issues
"""

import asyncio
from playwright.async_api import async_playwright
import time
import json

async def capture_dashboard():
    """Capture dashboard screenshots and analyze behavior"""
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)  # Set to False to see it live
        page = await browser.new_page()
        
        # Go to dashboard
        await page.goto('http://localhost:8000/frontend/')
        print("✅ Dashboard loaded")
        
        # Wait for initial load
        await page.wait_for_timeout(3000)
        
        # Capture initial screenshot
        await page.screenshot(path='dashboard_initial.png')
        print("📸 Initial screenshot captured")
        
        # Monitor for 20 seconds and capture behavior
        print("\n🔍 Monitoring dashboard for 20 seconds...")
        
        observations = []
        
        for i in range(4):  # 4 observations, 5 seconds apart
            await page.wait_for_timeout(5000)
            
            # Check if chart is visible
            chart_visible = await page.is_visible('#tradingview-widget')
            
            # Get current price
            price_element = await page.query_selector('#current-price')
            current_price = await price_element.inner_text() if price_element else 'N/A'
            
            # Check for any console errors
            console_errors = []
            page.on('console', lambda msg: console_errors.append(msg.text()) if msg.type == 'error' else None)
            
            observation = {
                'time': i * 5,
                'chart_visible': chart_visible,
                'current_price': current_price,
                'console_errors': console_errors
            }
            
            observations.append(observation)
            print(f"  [{i*5}s] Price: {current_price}, Chart: {'✅' if chart_visible else '❌'}")
            
            # Capture screenshot
            await page.screenshot(path=f'dashboard_{i*5}s.png')
        
        # Check the TradingView widget specifically
        print("\n🔍 Analyzing TradingView widget...")
        
        # Get the iframe if it exists
        frames = page.frames
        print(f"  Found {len(frames)} frames on page")
        
        # Try to get chart container dimensions
        chart_container = await page.query_selector('#chart')
        if chart_container:
            box = await chart_container.bounding_box()
            print(f"  Chart container size: {box['width']}x{box['height']}")
        
        # Final screenshot
        await page.screenshot(path='dashboard_final.png', full_page=True)
        print("\n📸 Final screenshot captured")
        
        # Close browser
        await browser.close()
        
        return observations

async def analyze_and_fix():
    """Analyze the dashboard and suggest fixes"""
    
    print("🚀 Starting dashboard analysis...\n")
    
    observations = await capture_dashboard()
    
    print("\n📊 Analysis Results:")
    print("-" * 40)
    
    # Check for issues
    issues_found = []
    
    # Check if prices are updating
    prices = [obs['current_price'] for obs in observations]
    if len(set(prices)) == 1:
        issues_found.append("Price not updating")
    
    # Check for console errors
    for obs in observations:
        if obs['console_errors']:
            issues_found.append(f"Console errors at {obs['time']}s: {obs['console_errors']}")
    
    if issues_found:
        print("⚠️ Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("✅ No major issues detected")
    
    print("\n💡 Recommendations:")
    print("  1. Check if TradingView widget is loading multiple times")
    print("  2. Verify WebSocket connection stability")
    print("  3. Check for any rate limiting on API calls")
    
    return observations

if __name__ == "__main__":
    asyncio.run(analyze_and_fix())