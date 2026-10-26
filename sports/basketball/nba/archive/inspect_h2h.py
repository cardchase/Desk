import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def inspect():
    url = "https://www.oddsportal.com/basketball/h2h/boston-celtics-KYD9hVEm/detroit-pistons-UcjMRj6J/#Ea72eQdM"
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        resp = await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(3000)
        
        current_url = page.url
        print("Final URL after navigation:", current_url)
        title = await page.title()
        print("Page Title:", title)
        
        # Check text in body
        body_text = await page.evaluate("() => document.body.innerText")
        lines = [l.strip() for l in body_text.split('\n') if l.strip()]
        print(f"Total text lines: {len(lines)}")
        print("First 30 text lines:")
        for l in lines[:30]:
            print("  ", l)
            
        # Check all links on this page
        links = await page.evaluate(r"""() => {
            return Array.from(document.querySelectorAll('a')).map(a => ({ text: a.innerText.trim(), href: a.href })).filter(x => x.text && x.href.includes('/basketball/'));
        }""")
        print(f"\nBasketball links on this page ({len(links)}):")
        for l in links[:20]:
            print("  ", l)
            
        await b.close()

if __name__ == "__main__":
    asyncio.run(inspect())
