import asyncio
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

# Add backend/scripts
sys.path.insert(0, r"d:\Project\LoomFlow - Sports\backend\scripts")
from basketball import OddsPortalBasketballScraper, HEADERS

async def test_upcoming_match_odds(url: str):
    print("=" * 70)
    print(f"Testing deep odds extraction on upcoming match: {url}")
    print("=" * 70)

    scraper = OddsPortalBasketballScraper(headless=True, max_workers=1)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        # Load page
        await page.goto(url, wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3000)
        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept"), .accept-choices').first
            if await accept_btn.count() > 0: await accept_btn.click(timeout=1500)
        except Exception: pass
        await page.wait_for_timeout(1000)

        # Let's inspect what tabs exist on this match page
        tabs = await page.evaluate(r"""() => {
            let allTabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]')).map(el => (el.innerText || '').trim()).filter(Boolean);
            return allTabs.filter(t => t.length < 30);
        }""")
        print("Visible tabs:", tabs[:25])

        # Test Over/Under tab clicking and line detection
        ou_status = await page.evaluate(r"""() => {
            let tabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]'));
            let ou = tabs.find(t => (t.innerText || '').trim().toLowerCase() === 'over/under');
            if (ou) {
                let btn = ou.querySelector('button, a') || ou;
                btn.click();
                return 'clicked_ou';
            }
            return 'ou_not_found';
        }""")
        print(f"Over/Under Tab Click: {ou_status}")
        await page.wait_for_timeout(2500)

        # Inspect headers under Over/Under
        ou_lines = await page.evaluate(r"""() => {
            let rows = Array.from(document.querySelectorAll('div, tr')).filter(el => {
                let t = (el.innerText || '').trim();
                return /^Over\/Under\s*[\+\-]?\d+(\.\d+)?/i.test(t) && t.length < 150;
            });
            return rows.map(r => r.innerText.replace(/\s+/g, ' ').trim());
        }""")
        print(f"Over/Under lines found: {len(ou_lines)}")
        for l in ou_lines[:10]:
            print("  ", l)

        # Also inspect Home/Away tab
        await page.evaluate(r"""() => {
            let tabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]'));
            let ha = tabs.find(t => (t.innerText || '').trim().toLowerCase() === 'home/away');
            if (ha) {
                let btn = ha.querySelector('button, a') || ha;
                btn.click();
            }
        }""")
        await page.wait_for_timeout(2000)

        ha_rows = await page.evaluate(r"""() => {
            let rows = Array.from(document.querySelectorAll('tr, div[data-testid="bookmaker-table-row"], div.border-b'));
            let odds = [];
            for (let r of rows) {
                let t = (r.innerText || '').replace(/\s+/g, ' ').trim();
                let m = t.match(/\b\d+\.\d{2}\b/g);
                if (m && m.length >= 2) {
                    odds.push({ text: t.slice(0, 80), odds: m.slice(0, 2) });
                }
            }
            return odds.slice(0, 10);
        }""")
        print(f"\nHome/Away odds found: {len(ha_rows)}")
        for h in ha_rows:
            print("  ", h)

        await browser.close()

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.oddsportal.com/basketball/h2h/boston-celtics-KYD9hVEm/detroit-pistons-UcjMRj6J/#Ea72eQdM"
    asyncio.run(test_upcoming_match_odds(url))
