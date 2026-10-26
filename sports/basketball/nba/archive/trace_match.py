import asyncio
import re
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def trace_match(url: str):
    print("=" * 70)
    print("TRACING EXECUTE_MATCH_ENGINE ON:", url)
    print("=" * 70)

    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        context = await b.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        # Step 1: goto
        print("Step 1: goto...")
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        print("HTTP Status:", resp.status if resp else "None")
        print("URL in page:", page.url)
        await page.wait_for_timeout(2500)
        
        # Dismiss cookie
        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
            if await accept_btn.count() > 0:
                print("Cookie popup found, clicking accept...")
                await accept_btn.click(timeout=1500)
        except Exception as e:
            print("Cookie check:", e)
            
        await page.wait_for_timeout(1000)

        # Check wait_for_selector
        try:
            print("Step 2: wait_for_selector('h1, tr, [data-testid=\"game-row\"]')...")
            await page.wait_for_selector('h1, tr, [data-testid="game-row"]', timeout=8000)
            print("Selector found!")
        except Exception as e:
            print("wait_for_selector timed out:", e)

        # Check is_live
        is_live = await page.evaluate("""() => {
            let t = document.body.innerText || '';
            return t.includes('In-Play Odds') || t.includes('4th Quarter') || t.includes('3rd Quarter') || t.includes('2nd Quarter') || t.includes('1st Quarter');
        }""")
        print("is_live check:", is_live)

        # Wait for table elements
        has_table = False
        for i in range(15):
            count = await page.evaluate("() => document.querySelectorAll('a[href*=\"/proxy/bookmakers/\"], [data-testid=\"odd-container\"], tr').length")
            print(f"Table elements poll {i}: {count}")
            if count > 5:
                has_table = True
                break
            await page.wait_for_timeout(400)
        print("has_table:", has_table)

        # Extract metadata
        h1 = await page.evaluate("() => document.querySelector('h1') ? document.querySelector('h1').innerText : ''")
        print("H1 text:", h1)
        
        body_sample = await page.evaluate("() => (document.body.innerText || '').slice(0, 500).replace(/\\s+/g, ' ')")
        print("Body text sample:", body_sample)
        
        scoreMatch = await page.evaluate(r"""() => {
            let clean = (s) => s ? s.replace(/[\u00a0\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim() : '';
            let bodyText = clean(document.body.innerText);
            let m = bodyText.match(/(?:Final\s*result|After\s*OT)\s*[:\s]*(\d+)\s*[:-]\s*(\d+)\s*\(([^)]+)\)/i);
            if (!m) m = bodyText.match(/(\d{2,3})\s*[:-]\s*(\d{2,3})\s*\(([^)]+)\)/);
            return m ? m[0] : 'None';
        }""")
        print("scoreMatch:", scoreMatch)

        await b.close()

if __name__ == "__main__":
    url = "https://www.oddsportal.com/basketball/h2h/minnesota-timberwolves-KjBIVQcI/san-antonio-spurs-IwmkErSH/#Gz6nfXBd"
    asyncio.run(trace_match(url))
