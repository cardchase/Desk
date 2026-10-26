"""
Basketball Upcoming Odds & Market Splits Scraper
=================================================
Scrapes upcoming fixtures and comprehensive market odds from OddsPortal:
- Navigates into each upcoming match URL to capture detailed market splits:
  * Home/Away: FT including OT, 1H, 2H, Q1, Q2, Q3, Q4
  * Over/Under: FT including OT, 1H, 2H, Q1, Q2, Q3, Q4 (Line, Over, Under)
- Prioritizes Bet365 balanced lines (~1.91/1.91) and filters out struck-through/disabled lines.
- Saves upcoming games to test_NBA/upcoming_games.csv and backend/outputs/intermediates/upcoming_games.csv
"""

import asyncio
import os
import sys
import re
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import polars as pl
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
NBA_DIR = BASE_DIR.parent
DATA_DIR = NBA_DIR / "data"

UPCOMING_CSV_PATH = DATA_DIR / "upcoming_games.csv"

HEADERS = [
    # Core Identifiers & Tournament State
    "Date", "Time", "Country", "Competition", "Season", "HomeTeam", "AwayTeam", "Match_Status", "URL",
    "Went_To_OT", "Match_Winner_Final",

    # Score Progression Timeline (Blank for upcoming)
    "FT_HomeScore", "FT_AwayScore",
    "Q1_HomeScore", "Q1_AwayScore",
    "Q2_HomeScore", "Q2_AwayScore",
    "Q3_HomeScore", "Q3_AwayScore",
    "Q4_HomeScore", "Q4_AwayScore",
    "HT_HomeScore", "HT_AwayScore",
    "SH_HomeScore", "SH_AwayScore",
    "OT_HomeScore", "OT_AwayScore",

    # Home/Away 2-Way Outcome Lines
    "HA_FT_Home", "HA_FT_Away",
    "HA_1H_Home", "HA_1H_Away",
    "HA_2H_Home", "HA_2H_Away",
    "HA_Q1_Home", "HA_Q1_Away",
    "HA_Q2_Home", "HA_Q2_Away",
    "HA_Q3_Home", "HA_Q3_Away",
    "HA_Q4_Home", "HA_Q4_Away",

    # Primary Over/Under Splits
    "OU_FT_Line", "OU_FT_Over", "OU_FT_Under",
    "OU_1H_Line", "OU_1H_Over", "OU_1H_Under",
    "OU_2H_Line", "OU_2H_Over", "OU_2H_Under",
    "OU_Q1_Line", "OU_Q1_Over", "OU_Q1_Under",
    "OU_Q2_Line", "OU_Q2_Over", "OU_Q2_Under",
    "OU_Q3_Line", "OU_Q3_Over", "OU_Q3_Under",
    "OU_Q4_Line", "OU_Q4_Over", "OU_Q4_Under"
]


def log(msg: str):
    print(f"[UpcomingScraper] {msg}", flush=True)


def parse_clean_date(date_raw: str, current_year: int = 2026) -> str:
    d = date_raw.strip()
    m_day_month = re.search(r'(\d{1,2})\s+([A-Za-z]{3})', d)
    m_year = re.search(r'\b(20\d{2})\b', d)
    
    if m_day_month:
        day = int(m_day_month.group(1))
        month = m_day_month.group(2).capitalize()
        year = int(m_year.group(1)) if m_year else current_year
        return f"{day:02d} {month} {year}"
    elif "today" in d.lower():
        now = datetime.datetime.now()
        return now.strftime("%d %b %Y")
    elif "tomorrow" in d.lower():
        tom = datetime.datetime.now() + datetime.timedelta(days=1)
        return tom.strftime("%d %b %Y")
    return d


async def click_sub_tab(page, name: str) -> bool:
    return await page.evaluate(r"""(targetName) => {
        let norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
        let goal = norm(targetName);

        let aliases = [goal];
        if (goal.includes('ftincludingot') || goal === 'fulltime') {
            aliases = ['ftincludingot', 'fulltime', 'ftincot', 'incot'];
        }
        if (goal === '1sthalf') aliases = ['1sthalf', '1h', 'half1'];
        if (goal === '2ndhalf') aliases = ['2ndhalf', '2h', 'half2'];
        if (goal === '1stquarter') aliases = ['1stquarter', '1q', 'quarter1'];
        if (goal === '2ndquarter') aliases = ['2ndquarter', '2q', 'quarter2'];
        if (goal === '3rdquarter') aliases = ['3rdquarter', '3q', 'quarter3'];
        if (goal === '4thquarter') aliases = ['4thquarter', '4q', 'quarter4'];

        let candidates = Array.from(document.querySelectorAll('div, button, a, span, p, li')).filter(el => {
            let text = norm(el.innerText || '');
            return aliases.some(a => text === a);
        });

        if (candidates.length === 0) {
            candidates = Array.from(document.querySelectorAll('div, button, a, span, p, li')).filter(el => {
                let text = norm(el.innerText || '');
                return aliases.some(a => text.startsWith(a) && text.length <= a.length + 8);
            });
        }

        if (candidates.length > 0) {
            let target = candidates[candidates.length - 1];
            let btn = target.closest('button, a, div[role="tab"]') || target;
            btn.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'center' });
            btn.click();
            return true;
        }
        return false;
    }""", name)


async def extract_visible_ha_odds(page) -> Optional[List[float]]:
    return await page.evaluate(r"""
    () => {
        let rows = Array.from(document.querySelectorAll('tr, div[data-testid="bookmaker-table-row"], div.border-b'));
        let firstAvailable = null;
        let preferredFallback = null;

        for (let r of rows) {
            let text = (r.innerText || '').toLowerCase();
            let bookmakerLink = r.querySelector('a[href*="/proxy/bookmakers/"], a[href*="/bookmakers/"]');
            let bookieHref = bookmakerLink ? bookmakerLink.getAttribute('href').toLowerCase() : '';
            let isBet365 = bookieHref.includes('bet365') || text.includes('bet365');

            // Exclude struck-through or disabled bookmaker lines
            let isStruck = !!r.querySelector('.line-through, [class*="line-through"], del, s') ||
                           !!r.querySelector('.text-gray-400, .opacity-40, [class*="disabled"]');
            if (isStruck) continue;

            let cells = Array.from(r.querySelectorAll('td'));
            let vals = [];

            if (cells.length >= 3) {
                for (let colIdx of [1, 2]) {
                    let cell = cells[colIdx];
                    let p = cell.querySelector('p');
                    let raw = p ? p.innerText.trim() : cell.innerText.trim();
                    let numMatch = raw.match(/^\d+\.\d{2}$/);
                    if (numMatch) {
                        let v = parseFloat(numMatch[0]);
                        if (v >= 1.01 && v <= 50.0) vals.push(v);
                    }
                }
            }

            if (vals.length < 2) {
                let potentialOdds = Array.from(r.querySelectorAll('td p.font-main.text-xs, td p, [data-testid="odd-container"] p, [data-testid="odd-container"]'));
                for (let el of potentialOdds) {
                    let cleanText = (el.innerText || '').trim();
                    if (/^\d+\.\d{2}$/.test(cleanText) && !cleanText.includes('%')) {
                        let val = parseFloat(cleanText);
                        if (val >= 1.01 && val <= 50.0 && !vals.includes(val)) {
                            vals.push(val);
                        }
                    }
                }
            }

            if (vals.length >= 2) {
                let pair = [vals[0], vals[1]];
                if (isBet365) return pair;

                let isSharp = bookieHref.includes('pinnacle') || bookieHref.includes('888sport') || 
                              bookieHref.includes('bet-at-home') || text.includes('pinnacle') || 
                              text.includes('888sport') || text.includes('bet-at-home');
                if (isSharp && !preferredFallback) preferredFallback = pair;
                if (!firstAvailable) firstAvailable = pair;
            }
        }
        return preferredFallback || firstAvailable;
    }
    """)


async def scrape_home_away(page, sub_tab_name: str) -> Optional[List[float]]:
    try:
        await page.evaluate("""() => {
            let tabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]'));
            let ha = tabs.find(t => (t.innerText || '').trim().toLowerCase() === 'home/away');
            if (ha && !ha.className.includes('active-item-feed') && !ha.className.includes('active')) {
                let btn = ha.querySelector('button, a') || ha;
                btn.click();
            }
        }""")
        await page.wait_for_timeout(600)

        clicked = await click_sub_tab(page, sub_tab_name)
        if clicked or sub_tab_name in ["FT including OT", "Full Time"]:
            await page.wait_for_timeout(1500)
            odds = await extract_visible_ha_odds(page)
            if not odds:
                await page.wait_for_timeout(1000)
                odds = await extract_visible_ha_odds(page)
            return odds
        return None
    except Exception:
        return None


async def scrape_over_under_primary(page, sub_tab_name: str) -> Optional[Dict[str, float]]:
    try:
        # 1. Switch to Over/Under Market Tab
        await page.evaluate("""() => {
            let tabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]'));
            let ou = tabs.find(t => (t.innerText || '').trim().toLowerCase() === 'over/under');
            if (ou && !ou.className.includes('active-item-feed') && !ou.className.includes('active')) {
                let btn = ou.querySelector('button, a') || ou;
                btn.click();
            }
        }""")
        await page.wait_for_timeout(800)

        # 2. Click Period Sub-Tab
        await click_sub_tab(page, sub_tab_name)
        await page.wait_for_timeout(1600)

        # 3. Locate handicap accordion lines with retry for DOM re-render
        headers_data = []
        for _ in range(3):
            headers_data = await page.evaluate(r"""
            (tabName) => {
                let allRows = Array.from(document.querySelectorAll('div, tr')).filter(el => {
                    let t = (el.innerText || '').trim();
                    return /^Over\/Under\s*[\+\-]?\d+(\.\d+)?/i.test(t) && t.length < 150;
                });

                let distinctHeaders = [];
                for (let row of allRows) {
                    let t = row.innerText.trim();
                    let m = t.match(/Over\/Under\s*([\+\-]?\d+(?:\.\d+)?)/i);
                    if (!m) continue;
                    let line = parseFloat(m[1]);

                    if (tabName.includes('Quarter') && (line < 25 || line > 90)) continue;
                    if (tabName.includes('Half') && (line < 65 || line > 175)) continue;
                    if ((tabName.includes('FT') || tabName.includes('Full Time')) && line < 130) continue;

                    if (!distinctHeaders.some(h => Math.abs(h.line - line) < 0.01)) {
                        let nums = [];
                        let textParts = t.split(/\s+/);
                        for (let p of textParts) {
                            if (/^\d+\.\d{2}$/.test(p) && !p.includes('%')) {
                                let v = parseFloat(p);
                                if (v >= 1.01 && v <= 50.0 && Math.abs(v - Math.abs(line)) > 0.1) {
                                    nums.push(v);
                                }
                            }
                        }

                        let isExpanded = t.includes('▲') || (row.className || '').includes('expanded') || 
                                         row.getAttribute('aria-expanded') === 'true';

                        distinctHeaders.push({
                            line: line,
                            consensusOver: nums.length >= 2 ? nums[0] : null,
                            consensusUnder: nums.length >= 2 ? nums[1] : null,
                            diff: nums.length >= 2 ? Math.abs(nums[0] - nums[1]) : 999.0,
                            isExpanded: isExpanded
                        });
                    }
                }

                distinctHeaders.sort((a, b) => a.diff - b.diff);
                return distinctHeaders;
            }
            """, sub_tab_name)
            if headers_data:
                break
            await asyncio.sleep(0.5)

        if not headers_data:
            return None

        async def inspect_mounted_line(line_val: float) -> Optional[Dict[str, Any]]:
            # Polling retry loop to handle async DOM mounting of Bet365 table rows
            last_res = None
            for _ in range(3):
                res = await page.evaluate(r"""
                (targetLine) => {
                    let allRows = Array.from(document.querySelectorAll('div, tr')).filter(el => {
                        let t = (el.innerText || '').trim();
                        return /^Over\/Under\s*[\+\-]?\d+(\.\d+)?/i.test(t) && t.length < 150;
                    });

                    let targetRow = allRows.find(el => {
                        let m = el.innerText.trim().match(/Over\/Under\s*([\+\-]?\d+(?:\.\d+)?)/i);
                        return m && Math.abs(parseFloat(m[1]) - targetLine) < 0.01;
                    });

                    if (!targetRow) return null;

                    let bet365Match = null;
                    let firstCleanMatch = null;
                    let sharpMatch = null;

                    let curr = targetRow.nextElementSibling;
                    let steps = 0;
                    while (curr && steps < 40) {
                        steps++;
                        let text = (curr.innerText || '').trim();
                        if (/^Over\/Under\s*[\+\-]?\d+/i.test(text)) break;

                        let childRows = Array.from(curr.querySelectorAll('tr.h-9, tr[class*="h-9"], div[data-testid="bookmaker-table-row"], tr, div.border-b'));
                        if (curr.tagName === 'TR' || (curr.className && curr.className.includes('h-9'))) {
                            childRows.push(curr);
                        }

                        for (let r of childRows) {
                            let rText = (r.innerText || '').toLowerCase();
                            let isStruck = !!r.querySelector('.line-through, [class*="line-through"], del, s') ||
                                           !!r.querySelector('.text-gray-400, .opacity-40, [class*="disabled"]');
                            if (isStruck) continue;

                            let bookieLink = r.querySelector('a[href*="/proxy/bookmakers/"], a[href*="/bookmakers/"]');
                            let href = bookieLink ? bookieLink.getAttribute('href').toLowerCase() : '';
                            let isBet365 = href.includes('bet365') || rText.includes('bet365');
                            let isSharp = href.includes('pinnacle') || href.includes('888sport') || 
                                          href.includes('bet-at-home') || rText.includes('pinnacle') || 
                                          rText.includes('888sport') || rText.includes('bet-at-home');

                            let cells = Array.from(r.querySelectorAll('td'));
                            let vals = [];
                            if (cells.length >= 3) {
                                for (let c of cells.slice(1, 4)) {
                                    let p = c.querySelector('p');
                                    let raw = p ? p.innerText.trim() : c.innerText.trim();
                                    let numMatch = raw.match(/^\d+\.\d{2}$/);
                                    if (numMatch) {
                                        let v = parseFloat(numMatch[0]);
                                        if (v >= 1.01 && v <= 50.0 && Math.abs(v - Math.abs(targetLine)) > 0.1) {
                                            vals.push(v);
                                        }
                                    }
                                }
                            }

                            if (vals.length < 2) {
                                let pTags = Array.from(r.querySelectorAll('p.font-main.text-xs, p, [data-testid="odd-container"] p, [data-testid="odd-container"]'));
                                for (let pt of pTags) {
                                    let ct = pt.innerText.trim();
                                    if (/^\d+\.\d{2}$/.test(ct) && !ct.includes('%')) {
                                        let v = parseFloat(ct);
                                        if (v >= 1.01 && v <= 50.0 && Math.abs(v - Math.abs(targetLine)) > 0.1 && !vals.includes(v)) {
                                            vals.push(v);
                                        }
                                    }
                                }
                            }

                            if (vals.length >= 2) {
                                let item = { line: targetLine, over: vals[0], under: vals[1], diff: Math.abs(vals[0] - vals[1]) };
                                if (isBet365) return { bet365: item, fallback: null };
                                if (isSharp && !sharpMatch) sharpMatch = item;
                                if (!firstCleanMatch) firstCleanMatch = item;
                            }
                        }
                        curr = curr.nextElementSibling;
                    }

                    return { bet365: null, fallback: sharpMatch || firstCleanMatch };
                }
                """, line_val)

                if res:
                    last_res = res
                    if res.get("bet365"):
                        return res
                await asyncio.sleep(0.4)
            return last_res

        async def open_accordion_header(line_val: float):
            await page.evaluate(r"""
            (targetLine) => {
                let allRows = Array.from(document.querySelectorAll('div, tr')).filter(el => {
                    let t = (el.innerText || '').trim();
                    return /^Over\/Under\s*[\+\-]?\d+(\.\d+)?/i.test(t) && t.length < 150;
                });

                let targetRow = allRows.find(el => {
                    let m = el.innerText.trim().match(/Over\/Under\s*([\+\-]?\d+(?:\.\d+)?)/i);
                    return m && Math.abs(parseFloat(m[1]) - targetLine) < 0.01;
                });

                if (targetRow) {
                    let text = targetRow.innerText;
                    let isAlreadyOpen = text.includes('▲') || (targetRow.className || '').includes('expanded') || 
                                        targetRow.getAttribute('aria-expanded') === 'true';
                    if (!isAlreadyOpen) {
                        targetRow.scrollIntoView({ behavior: 'instant', block: 'center' });
                        targetRow.click();
                    }
                }
            }
            """, line_val)
            await page.wait_for_timeout(1200)

        # 1. Check already opened accordions
        for h in headers_data:
            if h.get("isExpanded"):
                info = await inspect_mounted_line(h["line"])
                if info and info.get("bet365"):
                    return info["bet365"]
                elif info and info.get("fallback"):
                    return info["fallback"]

        # 2. Check top 4 most balanced lines
        best_bet365 = None
        best_fallback = None

        for h in headers_data[:4]:
            await open_accordion_header(h["line"])
            info = await inspect_mounted_line(h["line"])
            if info:
                if info.get("bet365"):
                    b365 = info["bet365"]
                    if not best_bet365 or b365["diff"] < best_bet365["diff"]:
                        best_bet365 = b365
                        if b365["diff"] <= 0.04:
                            break
                elif not best_fallback and info.get("fallback"):
                    best_fallback = info["fallback"]

        if best_bet365:
            return best_bet365
        if best_fallback:
            return best_fallback

        primary_h = headers_data[0]
        if primary_h.get("consensusOver") and primary_h.get("consensusUnder"):
            return {
                "line": primary_h["line"],
                "over": primary_h["consensusOver"],
                "under": primary_h["consensusUnder"]
            }

        return None
    except Exception:
        return None


async def scrape_single_upcoming_match(context, fixture: Dict[str, Any]) -> Dict[str, Any]:
    url = fixture["url"]
    row = {h: None for h in HEADERS}
    row.update({
        "Date": fixture["date"],
        "Time": fixture["time"],
        "Country": "USA",
        "Competition": "NBA",
        "Season": "2025/2026",
        "HomeTeam": fixture["home_team"],
        "AwayTeam": fixture["away_team"],
        "Match_Status": "Upcoming",
        "URL": url,
        "Went_To_OT": "No",
        "HA_FT_Home": fixture.get("h_odds"),
        "HA_FT_Away": fixture.get("a_odds")
    })

    page = await context.new_page()
    await Stealth().apply_stealth_async(page)

    try:
        log(f"Scraping match detail: {fixture['home_team']} vs {fixture['away_team']}")
        await page.goto(url, wait_until="domcontentloaded", timeout=35000)
        await page.wait_for_timeout(2500)

        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
            if await accept_btn.count() > 0:
                await accept_btn.click(timeout=1500)
        except Exception:
            pass

        # Check breadcrumbs/meta if missing
        details = await page.evaluate(r"""
        () => {
            let h1 = document.querySelector('h1');
            let text = h1 ? h1.innerText : '';
            let dItem = document.querySelector('[data-testid="game-time-item"]');
            let dText = dItem ? dItem.innerText : '';
            return { header: text, date_text: dText };
        }
        """)

        # Refine team names if top-level parse was uncertain
        if details.get("header") and (" - " in details["header"] or " vs " in details["header"]):
            spl = " vs " if " vs " in details["header"] else " - "
            pts = details["header"].split(spl)
            if len(pts) >= 2:
                row["HomeTeam"] = pts[0].strip()
                row["AwayTeam"] = re.sub(r'\s*(Odds|Scores|H2H|-).*$', '', pts[1], flags=re.I).strip()

        # 1. Scrape Home/Away splits across quarters and halves
        for period, col_h, col_a in [
            ("FT including OT", "HA_FT_Home", "HA_FT_Away"),
            ("1st Half", "HA_1H_Home", "HA_1H_Away"),
            ("2nd Half", "HA_2H_Home", "HA_2H_Away"),
            ("1st Quarter", "HA_Q1_Home", "HA_Q1_Away"),
            ("2nd Quarter", "HA_Q2_Home", "HA_Q2_Away"),
            ("3rd Quarter", "HA_Q3_Home", "HA_Q3_Away"),
            ("4th Quarter", "HA_Q4_Home", "HA_Q4_Away")
        ]:
            ha_pair = await scrape_home_away(page, period)
            if ha_pair:
                row[col_h], row[col_a] = ha_pair[0], ha_pair[1]

        # 2. Scrape Over/Under splits across quarters and halves
        for period, col_l, col_o, col_u in [
            ("FT including OT", "OU_FT_Line", "OU_FT_Over", "OU_FT_Under"),
            ("1st Half", "OU_1H_Line", "OU_1H_Over", "OU_1H_Under"),
            ("2nd Half", "OU_2H_Line", "OU_2H_Over", "OU_2H_Under"),
            ("1st Quarter", "OU_Q1_Line", "OU_Q1_Over", "OU_Q1_Under"),
            ("2nd Quarter", "OU_Q2_Line", "OU_Q2_Over", "OU_Q2_Under"),
            ("3rd Quarter", "OU_Q3_Line", "OU_Q3_Over", "OU_Q3_Under"),
            ("4th Quarter", "OU_Q4_Line", "OU_Q4_Over", "OU_Q4_Under")
        ]:
            ou_data = await scrape_over_under_primary(page, period)
            if ou_data:
                row[col_l] = ou_data["line"]
                row[col_o] = ou_data["over"]
                row[col_u] = ou_data["under"]

        log(f"  -> Extracted {row['HomeTeam']} vs {row['AwayTeam']} | FT OU: {row['OU_FT_Line']} (O:{row['OU_FT_Over']}/U:{row['OU_FT_Under']}) | Q1 OU: {row['OU_Q1_Line']}")
        return row
    except Exception as e:
        log(f"Error scraping match page {url}: {e}")
        return row
    finally:
        await page.close()


async def scrape_nba_upcoming(
    target_url: str = "https://www.oddsportal.com/basketball/usa/nba/",
    filter_date: Optional[str] = None,
    concurrency: int = 10,
    custom_output: Optional[Path] = None
):
    log("=" * 75)
    log("🏀 ODDS_PORTAL NBA UPCOMING GAMES & MARKET SPLITS SCRAPER")
    log(f"Target URL: {target_url} | Date Filter: {filter_date} | Concurrency: {concurrency}")
    log("=" * 75)

    async with async_playwright() as p:
        browser_args = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--headless=new']
        browser = await p.chromium.launch(headless=True, args=browser_args)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        log("Loading OddsPortal NBA overview...")
        for attempt in range(3):
            try:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(3000)
                try:
                    accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept"), .accept-choices').first
                    if await accept_btn.count() > 0:
                        await accept_btn.click(timeout=2000)
                except Exception:
                    pass
                break
            except Exception as e:
                log(f"Navigation attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    await browser.close()
                    return []
                await asyncio.sleep(2)

        for _ in range(8):
            await page.evaluate("window.scrollBy(0, 800)")
            await page.wait_for_timeout(300)

        # Harvest fixture links and basic upcoming matchups
        upcoming_fixtures = await page.evaluate(r"""() => {
            let elements = Array.from(document.querySelectorAll(
                'div[class*="bg-gray-light"], div[class*="bg-gray-med_light"], div[data-testid="game-row"], tr, div.eventRow, div[class*="border-black-borders"], div[class*="border-b"]'
            ));
            let curDate = '';
            let fixtures = [];
            let seen = new Set();

            for (let el of elements) {
                let text = (el.innerText || '').trim();
                let isDate = (el.className || '').includes('bg-gray-light') || (el.className || '').includes('bg-gray-med_light');
                let isMatchup = text.includes('Finished') || /\b\d{2}:\d{2}\b/.test(text) || text.includes('After OT');
                
                let dateMatch = text.match(/\b(Today[^\n]*|Tomorrow[^\n]*|\d{1,2}\s+[A-Za-z]{3}(?:\s+\d{4})?)/i);
                if (!isMatchup && (isDate || dateMatch)) {
                    if (dateMatch) {
                        curDate = dateMatch[1].trim();
                        continue;
                    }
                }
                
                let a = el.querySelector('a[href*="/basketball/"]');
                if (a) {
                    let href = a.href.split('?')[0];
                    if (!href.includes('/h2h/') && !/-[a-zA-Z0-9]{8}\/?(?:#.*)?$/.test(href)) continue;
                    if (href.includes('/results') || href.includes('/standings') || href.includes('/outrights')) continue;
                    if (seen.has(href)) continue;

                    let cleanText = text.replace(/\s+/g, ' ').trim();
                    let isFinished = cleanText.includes('Finished') || cleanText.includes('After OT');
                    let isLive = cleanText.includes('Live') || cleanText.includes('In-Play');
                    if (isFinished || isLive) continue; // upcoming games only

                    seen.add(href);

                    let mTime = cleanText.match(/\b(\d{2}:\d{2})\b/);
                    let matchTime = mTime ? mTime[1] : "Upcoming";

                    let sClean = cleanText.replace(/\b\d{2}:\d{2}\b/, '').trim();
                    let teams = sClean.split(' - ');
                    let hTeam = teams.length >= 2 ? teams[0].trim() : '';
                    let aTeam = teams.length >= 2 ? teams[1].split(/\s+\d+\.\d{2}/)[0].trim() : '';

                    let nums = cleanText.match(/\b\d+\.\d{2}\b/g) || [];
                    let hOdds = nums.length >= 2 ? parseFloat(nums[0]) : null;
                    let aOdds = nums.length >= 2 ? parseFloat(nums[1]) : null;

                    fixtures.push({
                        date_raw: curDate,
                        time: matchTime,
                        home_team: hTeam,
                        away_team: aTeam,
                        url: href,
                        h_odds: hOdds,
                        a_odds: aOdds
                    });
                }
            }
            return fixtures;
        }""")

        await page.close()
        log(f"Identified {len(upcoming_fixtures)} raw upcoming fixtures. Formatting and deduplicating...")

        formatted_fixtures = []
        current_year = datetime.datetime.now().year
        seen_matchups = set()

        for f in upcoming_fixtures:
            f["date"] = parse_clean_date(f["date_raw"], current_year=current_year)
            m_key = (f["home_team"].strip().lower(), f["away_team"].strip().lower())
            if m_key not in seen_matchups and f["home_team"] and f["away_team"]:
                seen_matchups.add(m_key)
                formatted_fixtures.append(f)

        # Optional date filtering
        if filter_date:
            norm_target = filter_date.strip().lower()
            if norm_target == "tomorrow":
                target_str = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%d %b %Y").lower()
            elif norm_target == "today":
                target_str = datetime.datetime.now().strftime("%d %b %Y").lower()
            else:
                target_str = norm_target

            target_fixtures = []
            for f in formatted_fixtures:
                d_str = f["date"].lower()
                d_raw = f["date_raw"].lower()
                if (target_str in d_str or target_str in d_raw or
                    ("24" in target_str and "24" in d_str and "oct" in d_str)):
                    target_fixtures.append(f)
            log(f"Filtered {len(formatted_fixtures)} fixtures down to {len(target_fixtures)} for target date '{filter_date}'.")
            fixtures_to_scrape = target_fixtures
        else:
            fixtures_to_scrape = formatted_fixtures

        total_fixtures = len(fixtures_to_scrape)
        log(f"Proceeding to scrape detailed odds for {total_fixtures} matches with concurrency={concurrency}...")

        # Concurrent scraping pool with Semaphore and real-time incremental disk persistence
        sem = asyncio.Semaphore(concurrency)
        save_lock = asyncio.Lock()
        completed_count = 0
        upcoming_rows: List[Dict[str, Any]] = []

        async def worker(idx: int, fix: Dict[str, Any]):
            nonlocal completed_count
            async with sem:
                log(f"[{completed_count + 1}/{total_fixtures}] Starting: {fix['home_team']} vs {fix['away_team']} ({fix['date']} {fix['time']})")
                row = await scrape_single_upcoming_match(context, fix)
                
                async with save_lock:
                    completed_count += 1
                    upcoming_rows.append(row)
                    # Real-time incremental save to disk so user sees continuous progress
                    save_upcoming_data(upcoming_rows, custom_output=custom_output, preview=False)
                    log(f"  ✅ [Done {completed_count}/{total_fixtures}] Saved {row['HomeTeam']} vs {row['AwayTeam']} | FT OU: {row['OU_FT_Line']} | 1H OU: {row['OU_1H_Line']} to upcoming_games.csv")

        # Stagger worker launches slightly (1.0s) so concurrent browser pages do not hit OddsPortal at the exact same millisecond
        tasks = []
        for idx, fix in enumerate(fixtures_to_scrape, 1):
            tasks.append(asyncio.create_task(worker(idx, fix)))
            await asyncio.sleep(1.0)

        await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()
        return upcoming_rows


def save_upcoming_data(upcoming_rows: List[Dict[str, Any]], custom_output: Optional[Path] = None, preview: bool = True):
    string_cols = ["Date", "Time", "Country", "Competition", "Season", "HomeTeam", "AwayTeam", "Match_Status", "URL", "Match_Winner_Final", "Went_To_OT"]
    schema = {h: pl.Utf8 if h in string_cols else pl.Float64 for h in HEADERS}

    if upcoming_rows:
        upcoming_df = pl.DataFrame(upcoming_rows, schema=schema)
        
        target_path = custom_output if custom_output else UPCOMING_CSV_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        upcoming_df.write_csv(target_path)

        if target_path != UPCOMING_CSV_PATH:
            upcoming_df.write_csv(UPCOMING_CSV_PATH)

        if preview:
            log(f"✅ Final save complete: {upcoming_df.height} upcoming games saved to: {target_path}")
            log("\n--- PREVIEW OF EXTRACTED UPCOMING SPLITS ---")
            prv = upcoming_df.select([
                "Date", "Time", "HomeTeam", "AwayTeam", 
                "HA_FT_Home", "HA_FT_Away", 
                "OU_FT_Line", "OU_FT_Over", "OU_FT_Under",
                "OU_1H_Line", "OU_1H_Over", "OU_1H_Under"
            ]).to_pandas()
            print(prv.to_string(index=False))
    else:
        if preview:
            log("No upcoming matches extracted.")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="OddsPortal NBA Upcoming Odds & Splits Scraper")
    parser.add_argument("--date", "-d", type=str, default=None, help="Target date to filter fixtures (e.g. '24 Oct 2026', 'tomorrow')")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path (e.g. 'oct24_live_odds.csv')")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Concurrent browser workers (default: 10)")
    args = parser.parse_args()

    out_path = Path(args.output).resolve() if args.output else None
    rows = await scrape_nba_upcoming(filter_date=args.date, concurrency=args.concurrency, custom_output=out_path)
    save_upcoming_data(rows, custom_output=out_path, preview=True)


if __name__ == "__main__":
    asyncio.run(main())
