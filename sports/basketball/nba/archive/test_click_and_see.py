import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def test_click_and_see():
    url = "https://www.oddsportal.com/basketball/usa/nba/"
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3000)
        
        # Accept cookie
        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
            if await accept_btn.count() > 0: await accept_btn.click(timeout=1500)
        except Exception: pass

        # Scroll to load the upcoming rows
        for _ in range(6):
            await page.evaluate("window.scrollBy(0, 600)")
            await page.wait_for_timeout(300)

        # Locate Detroit Pistons vs Boston Celtics link
        link = page.locator('a[href*="boston-celtics"][href*="detroit-pistons"]').first
        print("Link count:", await link.count())
        if await link.count() > 0:
            href = await link.get_attribute('href')
            print("Href on element:", href)
            
            # Click it!
            await link.click()
            await page.wait_for_timeout(4000)
            
            print("Current URL after click:", page.url)
            print("Current Title:", await page.title())
            
            # Check what's on the page now!
            text_sample = await page.evaluate(r"""() => {
                let lines = (document.body.innerText || '').split('\n').map(s => s.trim()).filter(Boolean);
                return lines.slice(0, 50);
            }""")
            print("\nPage text after click (first 50 lines):")
            for l in text_sample:
                print("  ", l)
        await b.close()

if __name__ == "__main__":
    asyncio.run(test_click_and_see())
