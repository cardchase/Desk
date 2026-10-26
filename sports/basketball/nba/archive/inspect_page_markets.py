import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def inspect_page_markets():
    url = "https://www.oddsportal.com/basketball/usa/nba/"
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        page = await (await b.new_context(viewport={'width': 1920, 'height': 1080})).new_page()
        await Stealth().apply_stealth_async(page)
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3000)
        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
            if await accept_btn.count() > 0: await accept_btn.click(timeout=1500)
        except Exception: pass
        await page.wait_for_timeout(1000)

        # Check for market selectors, buttons, tabs near the top of the table
        markets_info = await page.evaluate(r"""() => {
            // Find all buttons, tabs, dropdowns on page
            let buttons = Array.from(document.querySelectorAll('button, div[role="tab"], a, select')).map(el => {
                let t = (el.innerText || '').trim();
                return { tag: el.tagName, text: t, class: el.className };
            }).filter(x => x.text && x.text.length < 40);
            
            // Check specific text like 'Over/Under', 'Handicap', '1X2', 'Home/Away'
            let marketKeywords = ['over/under', 'totals', 'handicap', 'spread', 'home/away', '1x2', 'quarter', 'half'];
            let foundKeywords = buttons.filter(b => marketKeywords.some(k => b.text.toLowerCase().includes(k)));
            
            return {
                foundKeywords: foundKeywords,
                allButtonsSample: buttons.slice(0, 40)
            };
        }""")
        
        print("Market keywords found on main page:")
        for k in markets_info['foundKeywords']:
            print("  ", k)
            
        print("\nAll buttons/tabs sample:")
        for b in markets_info['allButtonsSample']:
            print("  ", b)
            
        await b.close()

if __name__ == "__main__":
    asyncio.run(inspect_page_markets())
