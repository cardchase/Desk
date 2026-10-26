"""
Basketball Single Match Test Script
===================================
Tests OddsPortalBasketballScraper on an individual match or H2H URL:
e.g. Knicks vs Spurs:
https://www.oddsportal.com/basketball/h2h/new-york-knicks-WCNO4nbt/san-antonio-spurs-IwmkErSH/#pnOhtTOH
"""

import asyncio
import sys
import os
import polars as pl

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add current scripts directory to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basketball import OddsPortalBasketballScraper, HEADERS

async def test_single_match(url: str):
    print("=" * 75)
    print("🏀 ODDS_PORTAL BASKETBALL SINGLE MATCH EXTRACTION TEST")
    print(f"Target URL: {url}")
    print("=" * 75)

    scraper = OddsPortalBasketballScraper(headless=True, max_workers=1)
    df = await scraper.run(url)

    print("\n" + "=" * 75)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 75)

    if df.is_empty():
        print("❌ No rows returned. Extraction failed or page blocked.")
        return df

    row = df.to_dicts()[-1]

    # Categorized output display
    categories = {
        "Match Metadata": ["Date", "Time", "Country", "Competition", "Season", "HomeTeam", "AwayTeam", "Match_Status", "Went_To_OT", "Match_Winner_Final"],
        "Score Progression": ["FT_HomeScore", "FT_AwayScore", "HT_HomeScore", "HT_AwayScore", "SH_HomeScore", "SH_AwayScore", "Q1_HomeScore", "Q1_AwayScore", "Q2_HomeScore", "Q2_AwayScore", "Q3_HomeScore", "Q3_AwayScore", "Q4_HomeScore", "Q4_AwayScore", "OT_HomeScore", "OT_AwayScore"],
        "Home/Away 2-Way Odds": ["HA_FT_Home", "HA_FT_Away", "HA_1H_Home", "HA_1H_Away", "HA_2H_Home", "HA_2H_Away", "HA_Q1_Home", "HA_Q1_Away", "HA_Q2_Home", "HA_Q2_Away", "HA_Q3_Home", "HA_Q3_Away", "HA_Q4_Home", "HA_Q4_Away"],
        "Over/Under Totals": ["OU_FT_Line", "OU_FT_Over", "OU_FT_Under", "OU_1H_Line", "OU_1H_Over", "OU_1H_Under", "OU_2H_Line", "OU_2H_Over", "OU_2H_Under", "OU_Q1_Line", "OU_Q1_Over", "OU_Q1_Under", "OU_Q2_Line", "OU_Q2_Over", "OU_Q2_Under", "OU_Q3_Line", "OU_Q3_Over", "OU_Q3_Under", "OU_Q4_Line", "OU_Q4_Over", "OU_Q4_Under"]
    }

    for cat_name, fields in categories.items():
        print(f"\n[{cat_name}]")
        for f in fields:
            val = row.get(f)
            formatted_val = str(val) if val is not None else "-"
            print(f"  {f:<22}: {formatted_val}")

    populated_count = sum(1 for v in row.values() if v is not None and v != "")
    print("\n" + "-" * 75)
    print(f"Total Schema Columns: {len(HEADERS)} | Populated Columns: {populated_count} ({populated_count/len(HEADERS)*100:.1f}%)")
    print("-" * 75)

    return df

if __name__ == "__main__":
    target_match_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://www.oddsportal.com/basketball/h2h/new-york-knicks-WCNO4nbt/san-antonio-spurs-IwmkErSH/#pnOhtTOH"
    )
    asyncio.run(test_single_match(target_match_url))
