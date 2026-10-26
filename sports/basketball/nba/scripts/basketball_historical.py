"""
Basketball Historical Odds Scraper (Standalone)
==============================================
Extracts historical NBA / Basketball match odds, quarter splits, and Over/Under totals
from OddsPortal using Playwright and Polars.

Prioritizes active Bet365 lines (closest to balanced odds ~1.91/1.91); 
falls back gracefully to the most balanced active line with clean non-strikethrough bookmakers.
"""

import asyncio
import random
import re
import os
import sys
import argparse
from typing import Dict, Any, List, Optional
import polars as pl
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


HEADERS = [
    # Core Identifiers & Tournament State
    "Date", "Time", "Country", "Competition", "Season", "HomeTeam", "AwayTeam", "Match_Status", "URL",
    "Went_To_OT", "Match_Winner_Final",

    # Score Progression Timeline
    "FT_HomeScore", "FT_AwayScore",
    "Q1_HomeScore", "Q1_AwayScore",
    "Q2_HomeScore", "Q2_AwayScore",
    "Q3_HomeScore", "Q3_AwayScore",
    "Q4_HomeScore", "Q4_AwayScore",
    "HT_HomeScore", "HT_AwayScore",
    "SH_HomeScore", "SH_AwayScore",
    "OT_HomeScore", "OT_AwayScore",

    # Home/Away 2-Way Outcome Lines (bet365 with First Available Fallback)
    "HA_FT_Home", "HA_FT_Away",
    "HA_1H_Home", "HA_1H_Away",
    "HA_2H_Home", "HA_2H_Away",
    "HA_Q1_Home", "HA_Q1_Away",
    "HA_Q2_Home", "HA_Q2_Away",
    "HA_Q3_Home", "HA_Q3_Away",
    "HA_Q4_Home", "HA_Q4_Away",

    # Primary Over/Under Splits (Line, Over Odds, Under Odds)
    "OU_FT_Line", "OU_FT_Over", "OU_FT_Under",     # FT including OT
    "OU_1H_Line", "OU_1H_Over", "OU_1H_Under",
    "OU_2H_Line", "OU_2H_Over", "OU_2H_Under",
    "OU_Q1_Line", "OU_Q1_Over", "OU_Q1_Under",
    "OU_Q2_Line", "OU_Q2_Over", "OU_Q2_Under",
    "OU_Q3_Line", "OU_Q3_Over", "OU_Q3_Under",
    "OU_Q4_Line", "OU_Q4_Over", "OU_Q4_Under"
]


class OddsPortalBasketballScraper:
    def __init__(self, headless: bool = True, max_workers: int = 10, auto_save_csv: Optional[str] = None):
        if auto_save_csv is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            auto_save_csv = os.path.join(data_dir, "nba_historical.csv")
        self.headless = headless
        self.max_workers = max_workers
        self.auto_save_csv = auto_save_csv
        self.is_cancelled = False
        self.global_pause_event = asyncio.Event()
        self.global_pause_event.set()

    def log(self, msg: str):
        print(f"[BasketballScraper] {msg}", flush=True)

    async def wait_for_clearance(self):
        if not self.global_pause_event.is_set():
            await self.global_pause_event.wait()

    def trigger_rate_limit_pause(self, seconds=60):
        if self.global_pause_event.is_set():
            self.log(f"🛑 [RATE LIMIT DETECTED] Pausing worker pool for {seconds}s...")
            self.global_pause_event.clear()
            async def release():
                await asyncio.sleep(seconds)
                self.log("🟢 [RATE LIMIT CLEARED] Resuming scraping pool.")
                self.global_pause_event.set()
            asyncio.create_task(release())

    async def run(self, target_url: str, scrape_all_seasons: bool = False) -> pl.DataFrame:
        clean_url = target_url.strip().replace(" ", "-")
        if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
            clean_url = "https://" + clean_url
        if "oddsportal.com" in clean_url and "www.oddsportal.com" not in clean_url:
            clean_url = clean_url.replace("oddsportal.com", "www.oddsportal.com")
        is_match = "/h2h/" in clean_url or re.search(r'-[a-zA-Z0-9]{8}/?(?:#.*)?$', clean_url)
        
        string_cols = ["Date", "Time", "Country", "Competition", "Season", "HomeTeam", "AwayTeam", "Match_Status", "URL", "Match_Winner_Final", "Went_To_OT"]
        schema = {h: pl.Utf8 if h in string_cols else pl.Float64 for h in HEADERS}

        async with async_playwright() as p:
            browser_args = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
            if self.headless:
                browser_args.extend(['--headless=new'])
            else:
                browser_args.extend(['--start-maximized'])

            browser = await p.chromium.launch(headless=self.headless, args=browser_args)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            try:
                if is_match:
                    page = await context.new_page()
                    await Stealth().apply_stealth_async(page)
                    
                    target_match_url = clean_url
                    if "/h2h/" in clean_url and "#" not in clean_url:
                        try:
                            await page.goto(clean_url, wait_until="domcontentloaded", timeout=25000)
                            await page.wait_for_timeout(2500)
                            try:
                                btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
                                if await btn.count() > 0: await btn.click(timeout=1500)
                            except Exception:
                                pass
                            await page.wait_for_timeout(1000)

                            resolved_link = await page.evaluate(r"""() => {
                                let path = window.location.pathname;
                                let parts = path.split('/h2h/')[1].split('/').filter(Boolean);
                                let t1 = parts[0] || '';
                                let t2 = parts[1] || '';
                                
                                let links = Array.from(document.querySelectorAll('a[href*="/h2h/"]')).map(a => a.href);
                                let exactMatches = links.filter(l => l.includes('#') && (
                                    (l.includes(t1) && l.includes(t2)) || 
                                    (l.includes(parts[0].split('-')[0]) && l.includes(parts[1].split('-')[0]))
                                ));
                                return exactMatches.length > 0 ? exactMatches[0] : (links.find(l => l.includes('#')) || null);
                            }""")
                            if resolved_link:
                                target_match_url = resolved_link
                                self.log(f"Resolved H2H overview to latest completed encounter: {target_match_url}")
                        except Exception as e:
                            self.log(f"Notice resolving H2H link: {e}")

                    row = await self.execute_match_engine(page, target_match_url)
                    await page.close()
                    clean_row = [{k: v for k, v in row.items() if k in HEADERS}] if (row and not row.get("_skip")) else []
                    df = pl.DataFrame(clean_row, schema=schema) if clean_row else pl.DataFrame([], schema=schema)
                    if self.auto_save_csv and not df.is_empty():
                        out_dir = os.path.dirname(os.path.abspath(self.auto_save_csv))
                        if out_dir:
                            os.makedirs(out_dir, exist_ok=True)
                        if os.path.exists(self.auto_save_csv):
                            try:
                                existing_df = pl.read_csv(self.auto_save_csv)
                                if "URL" in existing_df.columns:
                                    existing_df = existing_df.filter(pl.col("URL") != clean_url)
                                df = pl.concat([existing_df, df], how="diagonal_relaxed")
                            except Exception as e:
                                self.log(f"Warning reading existing CSV: {e}")
                        df.write_csv(self.auto_save_csv)
                        self.log(f"Saved dataset ({df.height} row) to {self.auto_save_csv}")
                    return df

                # Competition Crawl
                competition_page = await context.new_page()
                await Stealth().apply_stealth_async(competition_page)

                if scrape_all_seasons:
                    extracted_seasons = await self.extract_season_links(competition_page, clean_url)
                    
                    # Detect if clean_url targeted a specific starting season (e.g. nba-2019-2020)
                    start_season_match = re.search(r'-(\d{4})(?:-(\d{4}))?/results/?$', clean_url)
                    if start_season_match:
                        start_year = int(start_season_match.group(1))
                        self.log(f"Initiating historical crawl backwards from season start year <= {start_year}...")

                        def get_year(u: str) -> int:
                            m = re.search(r'-(\d{4})(?:-(\d{4}))?/results/?$', u)
                            return int(m.group(1)) if m else 9999

                        season_urls = [s for s in extracted_seasons if get_year(s) <= start_year]
                        if not any(clean_url.rstrip('/') == s.rstrip('/') for s in season_urls):
                            season_urls.insert(0, clean_url)
                        self.log(f"Queued {len(season_urls)} historical seasons starting with {season_urls[0]}")
                    else:
                        season_urls = extracted_seasons
                else:
                    season_urls = [clean_url]

                all_valid_rows = []
                scraped_urls = set()

                if self.auto_save_csv and os.path.exists(self.auto_save_csv):
                    try:
                        existing_df = pl.read_csv(self.auto_save_csv, schema_overrides=schema)
                        if "URL" in existing_df.columns:
                            scraped_urls = set(existing_df["URL"].drop_nulls().to_list())
                            all_valid_rows = existing_df.to_dicts()
                            self.log(f"Resuming from {self.auto_save_csv}. Preserved {len(scraped_urls)} existing matches.")
                    except Exception as e:
                        self.log(f"Warning reading CSV resume file: {e}")

                semaphore = asyncio.Semaphore(self.max_workers)

                for s_idx, season_url in enumerate(season_urls):
                    season_slug = season_url.split('basketball/')[-1] if 'basketball/' in season_url else season_url
                    self.log(f"[Season {s_idx+1}/{len(season_urls)}] Scanning: {season_slug}")
                    
                    season_match = re.search(r'-(\d{4})-(\d{4})', season_url) or re.search(r'-(\d{4})/', season_url)
                    if season_match:
                        if season_match.lastindex == 2 and season_match.group(2):
                            current_season = f"{season_match.group(1)}/{season_match.group(2)}"
                        else:
                            current_season = season_match.group(1)
                    else:
                        current_season = "2025/2026"

                    self.log(f"[Season {s_idx+1}/{len(season_urls)}] Active Season: {current_season} ({season_slug})")
                    match_links = await self.extract_match_links(competition_page, season_url, scraped_urls)
                    
                    unscraped_links = [m for m in match_links if m not in scraped_urls]
                    self.log(f"Found {len(unscraped_links)} new matches to scrape in this season.")

                    async def process_match(match_url, order_idx):
                        await self.wait_for_clearance()
                        await asyncio.sleep(random.uniform(0.1, 0.8))
                        async with semaphore:
                            m_page = None
                            try:
                                m_page = await context.new_page()
                                await Stealth().apply_stealth_async(m_page)
                                res = await self.execute_match_engine(m_page, match_url, season=current_season)
                                return res
                            except Exception as e:
                                self.log(f"Error scraping {match_url}: {e}")
                                return None
                            finally:
                                if m_page:
                                    try:
                                        await m_page.close()
                                    except:
                                        pass

                    tasks = [asyncio.create_task(process_match(link, i)) for i, link in enumerate(unscraped_links)]
                    for completed_task in asyncio.as_completed(tasks):
                        r = await completed_task
                        if r and not r.get("_skip") and r.get("HomeTeam") and r.get("AwayTeam"):
                            clean_r = {k: v for k, v in r.items() if k in HEADERS}
                            all_valid_rows.append(clean_r)
                            scraped_urls.add(r["URL"])
                            
                            self.log(f"✅ [Saved] {r.get('HomeTeam')} vs {r.get('AwayTeam')} | FT: {r.get('FT_HomeScore')}-{r.get('FT_AwayScore')}")
                            
                            if self.auto_save_csv:
                                try:
                                    partial_df = pl.DataFrame(all_valid_rows, schema=schema)
                                    tmp_csv = self.auto_save_csv + ".tmp"
                                    partial_df.write_csv(tmp_csv)
                                    for _ in range(5):
                                        try:
                                            os.replace(tmp_csv, self.auto_save_csv)
                                            break
                                        except Exception:
                                            await asyncio.sleep(0.1)
                                except Exception as e:
                                    self.log(f"CSV autosave error: {e}")

                await competition_page.close()
                return pl.DataFrame(all_valid_rows, schema=schema) if all_valid_rows else pl.DataFrame([], schema=schema)

            finally:
                await browser.close()

    async def extract_season_links(self, page, base_url: str) -> List[str]:
        try:
            clean_url = base_url.strip().replace(" ", "-")
            if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
                clean_url = "https://" + clean_url
            if "oddsportal.com" in clean_url and "www.oddsportal.com" not in clean_url:
                clean_url = clean_url.replace("oddsportal.com", "www.oddsportal.com")

            clean_root = re.sub(r'-\d{4}(?:-\d{4})?.*$', '', clean_url).split('/results')[0].split('/standings')[0].rstrip('/')

            await page.goto(clean_url, wait_until="domcontentloaded", timeout=40000)
            try:
                accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
                if await accept_btn.count() > 0:
                    await accept_btn.click(timeout=3000)
            except Exception:
                pass
            await page.wait_for_timeout(4000)
            links = await page.evaluate("() => Array.from(document.body.querySelectorAll('a')).map(a => a.href)")
            
            season_links = []
            for l in links:
                if not l:
                    continue
                norm = l.split('#')[0].split('?')[0].rstrip('/') + '/'
                if any(f"/{lang}/" in norm for lang in ['pl', 'de', 'es', 'ru', 'it', 'fr', 'gr']):
                    continue
                if norm.startswith(clean_root) and '/results/' in norm:
                    if re.search(r'-\d{4}(?:-\d{4})?/results/?$', norm) or norm == clean_root + '/results/':
                        season_links.append(norm)

            def get_season_year(url: str) -> int:
                m = re.search(r'-(\d{4})(?:-(\d{4}))?/results/?$', url)
                if m:
                    return int(m.group(1))
                return 9999

            unique = list(dict.fromkeys(season_links))
            if unique:
                unique.sort(key=get_season_year, reverse=True)
                return unique
            return [clean_url]
        except Exception as e:
            self.log(f"Notice extracting season links: {e}")
            return [base_url]

    async def extract_match_links(self, page, competition_url: str, scraped_urls: set) -> List[str]:
        match_links = []
        seen = set()
        loaded = False
        for attempt in range(2):
            try:
                await page.goto(competition_url, wait_until="domcontentloaded", timeout=40000)
                await page.wait_for_timeout(3000)
                loaded = True
                break
            except Exception as e:
                self.log(f"Attempt {attempt + 1} to load competition base failed: {e}")
                if attempt == 0:
                    await asyncio.sleep(2)
        if not loaded:
            self.log(f"Failed to load competition base {competition_url} after retries.")
            return []

        try:
            accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept"), .accept-choices').first
            if await accept_btn.count() > 0:
                await accept_btn.click(timeout=2500)
        except Exception:
            pass

        for page_num in range(1, 55):
            self.log(f"Scanning Page {page_num}...")
            page_seen_before = len(match_links)
            for _ in range(15):
                await page.evaluate("window.scrollBy(0, 700)")
                await page.wait_for_timeout(350)
                
                batch = await page.evaluate(r"""
                () => {
                    let rows = Array.from(document.querySelectorAll('div[data-testid="game-row"], tr, div.eventRow, div[class*="border-black-borders"], div[class*="border-b"]'));
                    let urls = [];
                    for (let r of rows) {
                        let text = r.innerText || "";
                        let isFinished = text.includes('Finished') || text.includes('After OT') || text.includes('AOT');
                        let isLive = text.includes('Live') || text.includes('In-Play') || text.includes('4th Quarter') || text.includes('3rd Quarter') || text.includes('2nd Quarter') || text.includes('1st Quarter');
                        if (!isFinished || isLive) continue;

                        let a = r.querySelector('a[href*="/basketball/"]');
                        if (a) {
                            let cleanUrl = a.href.split('?')[0];
                            let isExcluded = cleanUrl.includes('/results') || cleanUrl.includes('/standings') || cleanUrl.includes('/outrights') || cleanUrl.includes('/user/');
                            let isValid = (cleanUrl.includes('/h2h/') || /-[a-zA-Z0-9]{8}\/?(?:#.*)?$/.test(cleanUrl)) && !isExcluded;
                            if (isValid) {
                                urls.push(cleanUrl);
                            }
                        }
                    }
                    return urls;
                }
                """)
                for u in batch:
                    if u not in seen:
                        seen.add(u)
                        match_links.append(u)

            page_found = len(match_links) - page_seen_before
            self.log(f"  -> Page {page_num}: Extracted {page_found} matches (Total: {len(match_links)})")

            if page_found == 0 and page_num > 1:
                self.log(f"No additional matches found on page {page_num}. Completed season results crawl.")
                break

            # Navigate to the Next page strictly within nav[aria-label="Pagination"] / nav.pagination
            click_res = await page.evaluate(r"""(cur) => {
                let nav = document.querySelector('nav[aria-label="Pagination"], nav.pagination');
                if (!nav) return { status: 'no_nav' };

                let nextBtn = Array.from(nav.querySelectorAll('button, a')).find(b => {
                    let t = (b.innerText || '').trim().toLowerCase();
                    let aria = (b.getAttribute('aria-label') || '').toLowerCase();
                    return t === 'next' || aria.includes('next');
                });

                if (!nextBtn) {
                    let targetText = String(cur + 1);
                    nextBtn = Array.from(nav.querySelectorAll('button, a')).find(b => (b.innerText || '').trim() === targetText);
                }

                if (!nextBtn) {
                    let nextSpan = Array.from(nav.querySelectorAll('span')).find(s => (s.innerText || '').trim().toLowerCase() === 'next');
                    if (nextSpan) return { status: 'disabled' };
                    return { status: 'not_found' };
                }

                let isDis = nextBtn.disabled || 
                            nextBtn.getAttribute('aria-disabled') === 'true' || 
                            (nextBtn.className || '').includes('cursor-not-allowed') || 
                            (nextBtn.className || '').includes('text-gray-400') ||
                            (nextBtn.className || '').includes('pointer-events-none');
                if (isDis) return { status: 'disabled' };

                nextBtn.scrollIntoView({ behavior: 'instant', block: 'center' });
                nextBtn.click();
                return { status: 'clicked' };
            }""", page_num)

            if click_res.get('status') == 'disabled':
                self.log(f"Next button disabled. Reached final page {page_num}.")
                break
            elif click_res.get('status') != 'clicked':
                self.log(f"No pagination Next button found on page {page_num} ({click_res.get('status')}). Completed season crawl.")
                break

            try:
                await page.wait_for_function(
                    """(targetPage) => {
                        let nav = document.querySelector('nav[aria-label="Pagination"], nav.pagination');
                        if (!nav) return false;
                        let active = nav.querySelector('[aria-current="page"]');
                        return active && active.innerText.trim() === String(targetPage);
                    }""",
                    arg=page_num + 1,
                    timeout=8000
                )
            except Exception:
                await page.wait_for_timeout(2500)

            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1000)

        return match_links

    async def execute_match_engine(self, page, url: str, season: Optional[str] = None) -> Dict[str, Any]:
        is_valid_match = ("/h2h/" in url) or bool(re.search(r'-[a-zA-Z0-9]{8}/?(?:#.*)?$', url))
        if not is_valid_match:
            return {"_skip": True, "URL": url}

        extracted_row = {h: None for h in HEADERS}
        extracted_row["URL"] = url

        # 1. Load match page
        for attempt in range(3):
            try:
                await self.wait_for_clearance()
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2500)

                # Dismiss cookie consent if shown
                try:
                    accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept")').first
                    if await accept_btn.count() > 0:
                        await accept_btn.click(timeout=1500)
                except Exception:
                    pass

                try:
                    await page.wait_for_selector('h1, tr, [data-testid="game-row"]', timeout=8000)
                except Exception:
                    pass

                if resp and resp.status in [403, 429, 502, 503]:
                    self.trigger_rate_limit_pause(60)
                    if attempt == 2: return {"_skip": True, "URL": url}
                    continue

                if '/basketball/' not in page.url.lower():
                    return {"_skip": True, "URL": url}

                break
            except Exception:
                if attempt == 2: return {"_skip": True, "URL": url}
                await asyncio.sleep(4.0)

        is_live = await page.evaluate("""() => {
            let t = document.body.innerText || '';
            return t.includes('In-Play Odds') || t.includes('4th Quarter') || t.includes('3rd Quarter') || t.includes('2nd Quarter') || t.includes('1st Quarter');
        }""")
        if is_live and not ('Final result' in await page.content() or 'After OT' in await page.content()):
            return {"_skip": True, "URL": url}

        # Wait for table elements to mount
        for _ in range(15):
            has_table = await page.evaluate("() => document.querySelectorAll('a[href*=\"/proxy/bookmakers/\"], [data-testid=\"odd-container\"], tr').length > 5")
            if has_table: break
            await page.wait_for_timeout(400)

        # 2. Extract scores, metadata, and quarter progression
        metadata = await page.evaluate(r"""
        () => {
            let res = {
                HomeTeam: '', AwayTeam: '', Country: '', Competition: '', Season: '', Date: '', Time: '', Match_Status: 'Finished',
                FT_HomeScore: null, FT_AwayScore: null,
                Q1_HomeScore: null, Q1_AwayScore: null, Q2_HomeScore: null, Q2_AwayScore: null,
                Q3_HomeScore: null, Q3_AwayScore: null, Q4_HomeScore: null, Q4_AwayScore: null,
                HT_HomeScore: null, HT_AwayScore: null, SH_HomeScore: null, SH_AwayScore: null,
                OT_HomeScore: null, OT_AwayScore: null, Went_To_OT: "No", Match_Winner_Final: null
            };

            let clean = (s) => s ? s.replace(/[\u00a0\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim() : '';

            // Date & Time
            let timeItem = document.querySelector('[data-testid="game-time-item"]');
            if (timeItem) {
                let pTags = Array.from(timeItem.querySelectorAll('p')).map(p => clean(p.innerText));
                for (let p of pTags) {
                    let d = p.match(/\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b/);
                    if (d) res.Date = d[1];
                    let t = p.match(/\b(\d{2}:\d{2})\b/);
                    if (t) res.Time = t[1];
                }
            }
            if (!res.Date || !res.Time) {
                let pList = Array.from(document.querySelectorAll('p, div.text-xs, span')).map(p => clean(p.innerText));
                for (let text of pList) {
                    if (!res.Date) {
                        let d = text.match(/\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b/);
                        if (d) res.Date = d[1];
                    }
                    if (!res.Time) {
                        let t = text.match(/\b(\d{2}:\d{2})\b/);
                        if (t) res.Time = t[1];
                    }
                }
            }

            // Teams from H1
            let h1 = document.querySelector('h1');
            if (h1) {
                let h1Text = clean(h1.innerText);
                let parts = h1Text.includes(' vs ') ? h1Text.split(' vs ') : h1Text.split(' - ');
                if (parts.length >= 2) {
                    res.HomeTeam = parts[0].trim();
                    res.AwayTeam = parts[1].replace(/\s*(Odds|Scores|H2H|-).*$/i, '').trim();
                }
            }
            if (!res.HomeTeam || !res.AwayTeam) {
                let title = document.title || '';
                let parts = title.includes(' vs ') ? title.split(' vs ') : (title.includes(' - ') ? title.split(' - ') : []);
                if (parts.length >= 2) {
                    if (!res.HomeTeam) res.HomeTeam = clean(parts[0]);
                    if (!res.AwayTeam) res.AwayTeam = clean(parts[1].replace(/\s*(Odds|Scores|H2H|Basketball).*$/i, ''));
                }
            }

            // Breadcrumbs
            let a2 = document.querySelector('a[data-testid="2"]');
            if (a2) res.Country = clean(a2.innerText);
            let a3 = document.querySelector('a[data-testid="3"]');
            if (a3) {
                let compText = clean(a3.innerText);
                let sm = compText.match(/\d{4}\/\d{4}|\d{4}/);
                if (sm) {
                    res.Season = sm[0];
                    res.Competition = compText.replace(sm[0], '').trim();
                } else {
                    res.Competition = compText;
                }
            }
            if (!res.Competition) res.Competition = "NBA";
            if (!res.Country) res.Country = "USA";

            // Live / Final Score & Period Tokens
            let bodyText = clean(document.body.innerText);
            let scoreMatch = bodyText.match(/(?:Final\s*result|After\s*OT)\s*[:\s]*(\d+)\s*[:-]\s*(\d+)\s*\(([^)]+)\)/i);
            if (!scoreMatch) {
                scoreMatch = bodyText.match(/(\d{2,3})\s*[:-]\s*(\d{2,3})\s*\(([^)]+)\)/);
            }
            if (scoreMatch) {
                res.FT_HomeScore = parseFloat(scoreMatch[1]);
                res.FT_AwayScore = parseFloat(scoreMatch[2]);
                let splits = scoreMatch[3].split(',').map(s => clean(s));

                let parsePair = (seg) => {
                    if (!seg) return [null, null];
                    let parts = seg.split(/[:-]/);
                    if (parts.length >= 2) {
                        return [parseFloat(parts[0].replace(/[^\d.-]/g, '')), parseFloat(parts[1].replace(/[^\d.-]/g, ''))];
                    }
                    return [null, null];
                };

                if (splits.length >= 4) {
                    let q1 = parsePair(splits[0]);
                    let q2 = parsePair(splits[1]);
                    let q3 = parsePair(splits[2]);
                    let q4 = parsePair(splits[3]);

                    res.Q1_HomeScore = q1[0]; res.Q1_AwayScore = q1[1];
                    res.Q2_HomeScore = q2[0]; res.Q2_AwayScore = q2[1];
                    res.Q3_HomeScore = q3[0]; res.Q3_AwayScore = q3[1];
                    res.Q4_HomeScore = q4[0]; res.Q4_AwayScore = q4[1];

                    if (q1[0] !== null && q2[0] !== null) {
                        res.HT_HomeScore = q1[0] + q2[0];
                        res.HT_AwayScore = q1[1] + q2[1];
                    }
                    if (q3[0] !== null && q4[0] !== null) {
                        res.SH_HomeScore = q3[0] + q4[0];
                        res.SH_AwayScore = q3[1] + q4[1];
                    }
                }

                if (splits.length >= 5 || bodyText.toLowerCase().includes('after ot')) {
                    res.Went_To_OT = "Yes";
                    if (splits.length >= 5) {
                        let ot = parsePair(splits[4]);
                        res.OT_HomeScore = ot[0]; res.OT_AwayScore = ot[1];
                    }
                }
            }

            let finalScoreNode = document.querySelector('[data-testid="game-result"], div.result, div[class*="text-black-main font-bold"]');
            if (res.FT_HomeScore === null || res.FT_AwayScore === null) {
                let cleanHeader = document.querySelector('h1') ? document.querySelector('h1').parentElement.innerText : '';
                let strictScore = cleanHeader.match(/(?:Final\s*result|After\s*OT)\s*[:\s]*(\d{2,3})\s*[:-]\s*(\d{2,3})/i);
                if (strictScore) {
                    res.FT_HomeScore = parseFloat(strictScore[1]);
                    res.FT_AwayScore = parseFloat(strictScore[2]);
                } else if (finalScoreNode) {
                    let nodeScore = clean(finalScoreNode.innerText).match(/^(\d{2,3})\s*[:-]\s*(\d{2,3})$/);
                    if (nodeScore) {
                        res.FT_HomeScore = parseFloat(nodeScore[1]);
                        res.FT_AwayScore = parseFloat(nodeScore[2]);
                    }
                }
            }

            if (res.FT_HomeScore !== null && res.FT_AwayScore !== null) {
                res.Match_Winner_Final = res.FT_HomeScore > res.FT_AwayScore ? "Home" : "Away";
            }

            return res;
        }
        """)
        extracted_row.update(metadata)

        if season:
            extracted_row["Season"] = season
        elif not extracted_row.get("Season") or extracted_row["Season"] in ["", "None", None]:
            s_url_match = re.search(r'nba-(\d{4})-(\d{4})', url) or re.search(r'-(\d{4})-(\d{4})', url)
            if s_url_match:
                extracted_row["Season"] = f"{s_url_match.group(1)}/{s_url_match.group(2)}"
            else:
                date_str = extracted_row.get("Date", "")
                if date_str:
                    d_parts = date_str.split()
                    if len(d_parts) == 3 and d_parts[2].isdigit():
                        month = d_parts[1].lower()
                        year = int(d_parts[2])
                        if month in ["oct", "nov", "dec"]:
                            extracted_row["Season"] = f"{year}/{year + 1}"
                        else:
                            extracted_row["Season"] = f"{year - 1}/{year}"

        if not extracted_row.get("HomeTeam") or not extracted_row.get("AwayTeam"):
            return {"_skip": True, "URL": url}

        if extracted_row.get("FT_HomeScore") is None or extracted_row.get("FT_AwayScore") is None:
            return {"_skip": True, "URL": url}

        # Sub-tab click helper with resilient normalization and aliases
        async def click_sub_tab(name: str) -> bool:
            return await page.evaluate(r"""(targetName) => {
                let norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
                let goal = norm(targetName);

                let aliases = [goal];
                if (goal.includes('ftincludingot') || goal === 'fulltime') {
                    aliases = ['ftincludingot', 'fulltime', 'ftincot', 'incot'];
                }
                if (goal === '1sthalf') {
                    aliases = ['1sthalf', '1h', 'half1'];
                }
                if (goal === '2ndhalf') {
                    aliases = ['2ndhalf', '2h', 'half2'];
                }

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

        # 3. Robust Home/Away Extractor with Strikethrough Rejection & Bet365 Priority
        async def extract_visible_ha_odds() -> Optional[List[float]]:
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

                    // Skip struck-through, grayed-out, or disabled rows
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

        async def scrape_home_away(sub_tab_name: str) -> Optional[List[float]]:
            try:
                # Ensure Home/Away market tab is selected
                await page.evaluate("""() => {
                    let tabs = Array.from(document.querySelectorAll('li, button, a, div[role="tab"]'));
                    let ha = tabs.find(t => (t.innerText || '').trim().toLowerCase() === 'home/away');
                    if (ha && !ha.className.includes('active-item-feed') && !ha.className.includes('active')) {
                        let btn = ha.querySelector('button, a') || ha;
                        btn.click();
                    }
                }""")
                await page.wait_for_timeout(600)

                clicked = await click_sub_tab(sub_tab_name)
                if clicked or sub_tab_name in ["FT including OT", "Full Time"]:
                    await page.wait_for_timeout(1800)
                    odds = await extract_visible_ha_odds()
                    if not odds:
                        await page.wait_for_timeout(1000)
                        odds = await extract_visible_ha_odds()
                    return odds
                return None
            except Exception:
                return None

        # Execute Home/Away passes
        for period, col_h, col_a in [
            ("FT including OT", "HA_FT_Home", "HA_FT_Away"),
            ("1st Half", "HA_1H_Home", "HA_1H_Away"),
            ("2nd Half", "HA_2H_Home", "HA_2H_Away"),
            ("1st Quarter", "HA_Q1_Home", "HA_Q1_Away"),
            ("2nd Quarter", "HA_Q2_Home", "HA_Q2_Away"),
            ("3rd Quarter", "HA_Q3_Home", "HA_Q3_Away"),
            ("4th Quarter", "HA_Q4_Home", "HA_Q4_Away")
        ]:
            odds = await scrape_home_away(period)
            if odds:
                extracted_row[col_h], extracted_row[col_a] = odds[0], odds[1]

        # 4. Over/Under Engine: Evaluates Bet365 Balanced Lines & Fallbacks Cleanly
        async def scrape_over_under_primary(sub_tab_name: str) -> Optional[Dict[str, float]]:
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
                await click_sub_tab(sub_tab_name)
                await page.wait_for_timeout(1800)

                # 3. Locate and inspect all handicap accordion headers
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

                        // Range validation per market period
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

                    // Sort headers primarily by most balanced consensus diff
                    distinctHeaders.sort((a, b) => a.diff - b.diff);
                    return distinctHeaders;
                }
                """, sub_tab_name)

                if not headers_data:
                    return None

                # 4. Helper to inspect the child bookmaker rows currently mounted under a specific line
                async def inspect_mounted_line(line_val: float) -> Optional[Dict[str, Any]]:
                    return await page.evaluate(r"""
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

                        // Traverse sibling elements until next accordion header
                        let curr = targetRow.nextElementSibling;
                        let steps = 0;
                        while (curr && steps < 25) {
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
                                    if (isBet365) return { bet365: item };
                                    if (isSharp && !sharpMatch) sharpMatch = item;
                                    if (!firstCleanMatch) firstCleanMatch = item;
                                }
                            }
                            curr = curr.nextElementSibling;
                        }

                        return { bet365: null, fallback: sharpMatch || firstCleanMatch };
                    }
                    """, line_val)

                # Helper to safely expand an accordion header WITHOUT closing already-opened ones
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
                            // ONLY click if currently collapsed (▼)
                            if (!isAlreadyOpen) {
                                targetRow.scrollIntoView({ behavior: 'instant', block: 'center' });
                                targetRow.click();
                            }
                        }
                    }
                    """, line_val)
                    await page.wait_for_timeout(1000)

                # Phase 1: Check lines that are ALREADY open for bet365
                for h in headers_data:
                    if h.get("isExpanded"):
                        line_info = await inspect_mounted_line(h["line"])
                        if line_info and line_info.get("bet365"):
                            return line_info["bet365"]

                # Phase 2: Check top balanced candidates (expand each if needed) to find bet365
                best_bet365_candidate = None
                best_fallback_candidate = None

                for h in headers_data[:4]:  # Inspect top 4 most balanced lines
                    await open_accordion_header(h["line"])
                    line_info = await inspect_mounted_line(h["line"])
                    if line_info:
                        if line_info.get("bet365"):
                            b365 = line_info["bet365"]
                            if not best_bet365_candidate or b365["diff"] < best_bet365_candidate["diff"]:
                                best_bet365_candidate = b365
                                if b365["diff"] <= 0.04:  # Balanced line found (e.g., 1.91 / 1.91)
                                    break
                        elif line_info.get("fallback") and not best_fallback_candidate:
                            best_fallback_candidate = line_info["fallback"]

                if best_bet365_candidate:
                    return best_bet365_candidate

                if best_fallback_candidate:
                    return best_fallback_candidate

                # Phase 3: Final safety fallback to balanced header consensus
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

        # Execute Over/Under passes across all periods
        for period, col_l, col_o, col_u in [
            ("FT including OT", "OU_FT_Line", "OU_FT_Over", "OU_FT_Under"),
            ("1st Half", "OU_1H_Line", "OU_1H_Over", "OU_1H_Under"),
            ("2nd Half", "OU_2H_Line", "OU_2H_Over", "OU_2H_Under"),
            ("1st Quarter", "OU_Q1_Line", "OU_Q1_Over", "OU_Q1_Under"),
            ("2nd Quarter", "OU_Q2_Line", "OU_Q2_Over", "OU_Q2_Under"),
            ("3rd Quarter", "OU_Q3_Line", "OU_Q3_Over", "OU_Q3_Under"),
            ("4th Quarter", "OU_Q4_Line", "OU_Q4_Over", "OU_Q4_Under")
        ]:
            res = await scrape_over_under_primary(period)
            if res:
                extracted_row[col_l] = res["line"]
                extracted_row[col_o] = res["over"]
                extracted_row[col_u] = res["under"]

        return extracted_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OddsPortal Historical Basketball Scraper")
    parser.add_argument("positional_url", nargs="?", default=None, help="Target URL (positional)")
    parser.add_argument("--url", type=str, default=None, help="Target competition or match URL")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent browser workers (default: 10)")
    parser.add_argument("--single-season", action="store_true", help="Scrape only the target season, without historical backward crawl")
    parser.add_argument("--output", type=str, default=None, help="Output CSV filepath")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible (non-headless) mode")

    args = parser.parse_args()

    test_url = args.url or args.positional_url or "https://www.oddsportal.com/basketball/usa/nba-2019-2020/results/"
    
    if args.output:
        target_csv = args.output
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        target_csv = os.path.join(data_dir, "nba_historical.csv")
    
    print("=" * 75)
    print("🏀 BASKETBALL SCRAPER EXECUTION")
    print(f"Target URL: {test_url}")
    print(f"Mode: {'Single Season' if args.single_season else 'Historical Backward Crawl'}")
    print(f"Workers: {args.max_workers} | Headless: {not args.headed}")
    print(f"Output CSV: {target_csv}")
    print("=" * 75)

    scraper = OddsPortalBasketballScraper(headless=not args.headed, max_workers=args.max_workers, auto_save_csv=target_csv)
    df = asyncio.run(scraper.run(test_url, scrape_all_seasons=not args.single_season))
    
    print("\nExtraction Complete! Previewing record:\n")
    if not df.is_empty():
        print(f"Total Matches in Dataset: {df.height}")
        row_dict = df.to_dicts()[-1]
        print(f"\n--- Latest Record: {row_dict.get('HomeTeam')} vs {row_dict.get('AwayTeam')} ---")
        for col, val in row_dict.items():
            if val is not None:
                print(f"  {col:<20}: {val}")
        print(f"\n✅ Successfully saved to: {target_csv}")
    else:
        print("No record extracted.")