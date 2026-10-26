import asyncio
import os
import sys
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

sys.path.insert(0, r"d:\Project\LoomFlow - Sports\backend\scripts")
from basketball import OddsPortalBasketballScraper, HEADERS

async def trace_engine():
    url = "https://www.oddsportal.com/basketball/h2h/minnesota-timberwolves-KjBIVQcI/san-antonio-spurs-IwmkErSH/#Gz6nfXBd"
    scraper = OddsPortalBasketballScraper(headless=True, max_workers=1)
    
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])
        context = await b.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)
        
        row = await scraper.execute_match_engine(page, url)
        print("execute_match_engine result:")
        print("  _skip:", row.get('_skip'))
        print("  HomeTeam:", row.get('HomeTeam'))
        print("  AwayTeam:", row.get('AwayTeam'))
        print("  FT_HomeScore:", row.get('FT_HomeScore'))
        print("  FT_AwayScore:", row.get('FT_AwayScore'))
        print("  HA_FT_Home:", row.get('HA_FT_Home'))
        print("  HA_FT_Away:", row.get('HA_FT_Away'))
        print("  OU_FT_Line:", row.get('OU_FT_Line'))
        print("  OU_FT_Over:", row.get('OU_FT_Over'))
        print("  OU_FT_Under:", row.get('OU_FT_Under'))
        print("  OU_1H_Line:", row.get('OU_1H_Line'))
        print("  OU_1H_Over:", row.get('OU_1H_Over'))
        print("  OU_1H_Under:", row.get('OU_1H_Under'))
        print("  OU_Q1_Line:", row.get('OU_Q1_Line'))
        print("  OU_Q1_Over:", row.get('OU_Q1_Over'))
        print("  OU_Q1_Under:", row.get('OU_Q1_Under'))
        
        await b.close()

if __name__ == "__main__":
    asyncio.run(trace_engine())
