import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def test():
    url = 'https://www.oddsportal.com/basketball/h2h/boston-celtics-KYD9hVEm/detroit-pistons-UcjMRj6J/#Ea72eQdM'
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        
        # Accept cookie
        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
            if await accept_btn.count() > 0: await accept_btn.click(timeout=1500)
        except Exception: pass

        for i in range(25):
            count = await page.evaluate(r"""() => {
                return document.querySelectorAll('tr, div[data-testid="bookmaker-table-row"], div.border-b, div[data-testid="odd-container"]').length;
            }""")
            print(f"Poll {i}: {count} elements")
            if count > 5:
                break
            await page.wait_for_timeout(400)
            
        tabs = await page.evaluate(r"""() => {
            return Array.from(document.querySelectorAll('li, button, a, div[role="tab"]')).map(el => (el.innerText || '').trim()).filter(t => t.length < 30);
        }""")
        print("Market tabs found:", [t for t in tabs if any(k in t.lower() for k in ['over', 'under', 'home', 'away', 'handicap', '1x2', 'quarter', 'half'])])
        
        h1 = await page.evaluate("() => document.querySelector('h1') ? document.querySelector('h1').innerText : ''")
        print("H1:", h1)
        
        # Check text in body
        body_text = await page.evaluate("() => (document.body.innerText || '').slice(0, 400).replace(/\s+/g, ' ')")
        print("Body sample:", body_text)
        
        await b.close()

if __name__ == "__main__":
    asyncio.run(test())
