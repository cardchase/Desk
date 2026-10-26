import asyncio
import os
import sys
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def debug_match():
    url = "https://www.oddsportal.com/basketball/h2h/new-york-knicks-WCNO4nbt/san-antonio-spurs-IwmkErSH/#pnOhtTOH"
    print(f"Loading: {url}")
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        print(f"HTTP Response Status: {resp.status if resp else 'None'}")
        await page.wait_for_timeout(3000)
        
        print("Page URL:", page.url)
        print("Page Title:", await page.title())
        
        content = await page.content()
        print("Content length:", len(content))
        if "cloudflare" in content.lower() or "challenge" in content.lower() or "just a moment" in content.lower():
            print("🚨 CLOUDFLARE BOT CHALLENGE TRIGGERED!")
            
        h1 = await page.locator('h1').all_inner_texts()
        print("H1 elements:", h1)
        
        await b.close()

if __name__ == "__main__":
    asyncio.run(debug_match())
