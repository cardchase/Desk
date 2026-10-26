import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def inspect_rows():
    url = "https://www.oddsportal.com/basketball/h2h/new-york-knicks-WCNO4nbt/san-antonio-spurs-IwmkErSH/#pnOhtTOH"
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3500)
        
        row_texts = await page.evaluate(r"""() => {
            let rows = Array.from(document.querySelectorAll('tr, div[data-testid="bookmaker-table-row"], div.border-b'));
            return rows.map(r => (r.innerText || '').replace(/\s+/g, ' ').trim()).filter(Boolean);
        }""")
        
        print(f"Total rows: {len(row_texts)}")
        for r in row_texts[:20]:
            print("  ", r)
            
        await b.close()

if __name__ == "__main__":
    asyncio.run(inspect_rows())
