import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def test_click_row():
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

        # Find the link or row for Detroit Pistons vs Boston Celtics
        link_target = await page.evaluate(r"""() => {
            let rows = Array.from(document.querySelectorAll('div[data-testid="game-row"], tr, div.eventRow, div[class*="border-b"]'));
            for (let r of rows) {
                let t = r.innerText || '';
                if (t.includes('Detroit Pistons') && t.includes('Boston Celtics')) {
                    // Check all links inside this row
                    let links = Array.from(r.querySelectorAll('a')).map(a => ({ text: a.innerText, href: a.href, html: a.outerHTML }));
                    return { text: t, links: links, rowHtml: r.outerHTML.slice(0, 500) };
                }
            }
            return null;
        }""")
        print("Link target:", link_target)

        # Click the link
        print("\nClicking link directly on the page...")
        target_a = page.locator('a:has-text("Detroit Pistons"), div:has-text("Detroit Pistons") a[href*="/basketball/"]').first
        if await target_a.count() > 0:
            async with page.expect_navigation(timeout=30000) as nav:
                await target_a.click()
            print("Navigated to URL:", page.url)
            print("Title:", await page.title())
            await page.wait_for_timeout(3000)
            
            # Print body text
            body_text = await page.evaluate("() => document.body.innerText")
            lines = [l.strip() for l in body_text.split('\n') if l.strip()]
            print("Lines count:", len(lines))
            print("Sample lines:")
            for l in lines[:30]:
                print("  ", l)
        else:
            print("Target link not found!")
            
        await b.close()

if __name__ == "__main__":
    asyncio.run(test_click_row())
