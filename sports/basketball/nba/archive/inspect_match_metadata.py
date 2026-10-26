import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def inspect_match_metadata():
    url = "https://www.oddsportal.com/basketball/h2h/new-york-knicks-WCNO4nbt/san-antonio-spurs-IwmkErSH/#pnOhtTOH"
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3500)
        
        # Check text in body
        body_text = await page.evaluate("() => document.body.innerText")
        lines = [l.strip() for l in body_text.split('\n') if l.strip()]
        
        # Check for Over/Under and odds elements
        data = await page.evaluate(r"""() => {
            let clean = (s) => s ? s.replace(/[\u00a0\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim() : '';
            let h1 = document.querySelector('h1');
            let h1Text = h1 ? clean(h1.innerText) : '';
            
            // Check final scores
            let scoreMatch = clean(document.body.innerText).match(/(?:Final\s*result|After\s*OT)\s*[:\s]*(\d+)\s*[:-]\s*(\d+)/i);
            
            // Check all buttons/tabs for Over/Under
            let tabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]')).map(el => clean(el.innerText)).filter(Boolean);
            
            // Check for bookmaker odds rows
            let rows = Array.from(document.querySelectorAll('tr, div[data-testid="bookmaker-table-row"], div.border-b'));
            
            return {
                h1: h1Text,
                scoreMatch: scoreMatch ? scoreMatch[0] : 'No score match',
                tabs: tabs.slice(0, 30),
                rowCount: rows.length
            };
        }""")
        
        print("Data:", data)
        await b.close()

if __name__ == "__main__":
    asyncio.run(inspect_match_metadata())
