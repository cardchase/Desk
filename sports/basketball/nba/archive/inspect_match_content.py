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
        await page.wait_for_timeout(3500)
        
        # Check elements near H1
        details = await page.evaluate(r"""() => {
            let h1 = document.querySelector('h1');
            if (!h1) return 'No H1';
            
            // Look at parent container of H1
            let container = h1.closest('main') || h1.parentElement.parentElement || document.body;
            
            // Find all buttons, tabs, links in container
            let items = Array.from(container.querySelectorAll('li, button, div[role="tab"], a, [data-testid]')).map(el => {
                let t = (el.innerText || '').trim();
                let role = el.getAttribute('role') || el.tagName;
                return { text: t, role: role, tag: el.tagName, id: el.getAttribute('data-testid') || '' };
            }).filter(x => x.text && x.text.length < 50 && !x.text.includes('\n'));
            
            // Find all text in container
            let textLines = container.innerText.split('\n').map(s => s.trim()).filter(Boolean);
            
            return {
                items: items.slice(0, 40),
                textSample: textLines.slice(0, 30)
            };
        }""")
        
        print("Text lines near H1:")
        for l in details['textSample']:
            print("  ", l)
            
        print("\nInteractive items near H1:")
        for item in details['items'][:25]:
            print("  ", item)
            
        await b.close()

if __name__ == "__main__":
    asyncio.run(test())
