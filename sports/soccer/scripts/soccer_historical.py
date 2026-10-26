"""
Soccer / Football Historical Odds Scraper (Standalone Engine)
=============================================================
High-concurrency, client-side V8 execution engine extracting historical soccer/football 
match odds, score progression timelines, and comprehensive market lines from OddsPortal.

Features:
- Full 10x-15x performance architecture ported from basketball_historical.py.
- Client-side V8 in-browser DOM execution (page.evaluate) bypassing Playwright RPC overhead.
- High-concurrency worker pool with ephemeral page isolation and clean teardown.
- Strikethrough and suspended bookmaker filtering (isStruck rejection).
- Bet365 prioritized extraction with first-available clean fallback.
- Granular score timeline: FT, HT (1st Half), SH (2nd Half), Extra Time, and Penalties.
- Full market matrix: 1X2 (FT, 1H, 2H), Over/Under (0.5 to 5.5), BTTS (FT, 1H, 2H),
  Double Chance (FT, 1H, 2H), and Draw No Bet (FT).
- Robust pagination with synthetic React page clicking and sidebar leakage isolation.
- Automatic CSV checkpointing and crash-safe resume.
"""

import asyncio
import random
import re
import os
import sys
import json
import argparse
import time
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Set
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
    "Is_Knockout", "Went_To_ET", "Match_Winner_Final",

    # Granular Score Progression Timeline
    "FT_HomeScore", "FT_AwayScore",           # Official score at 90' + stoppage
    "HT_HomeScore", "HT_AwayScore",           # Score at 1st Half whistle
    "SH_HomeScore", "SH_AwayScore",           # Isolated 2nd Half goals (FT minus HT)
    "ET_HomeScore", "ET_AwayScore",           # Goals scored strictly during Extra Time
    "Penalties_HomeScore", "Penalties_AwayScore", # Shootout goals converted
    
    # 1X2 Outcome Lines (Bet365 / Clean Fallback)
    "FT_HomeOdds", "FT_DrawOdds", "FT_AwayOdds",
    "1H_HomeOdds", "1H_DrawOdds", "1H_AwayOdds",
    "SH_HomeOdds", "SH_DrawOdds", "SH_AwayOdds",  
    
    # Total Goals Over/Under Split Matrix
    "OU05_Over", "OU05_Under", "OU15_Over", "OU15_Under", 
    "OU25_Over", "OU25_Under", "OU35_Over", "OU35_Under", 
    "OU45_Over", "OU45_Under", "OU55_Over", "OU55_Under",
    
    # Both Teams to Score (BTTS Splits)
    "BTTS_Yes", "BTTS_No", 
    "BTTS_1H_Yes", "BTTS_1H_No", 
    "BTTS_2H_Yes", "BTTS_2H_No",
    
    # Draw No Bet Splits
    "DNB_Home", "DNB_Away", 
    
    # Double Chance Splits
    "DC_FT_1X", "DC_FT_12", "DC_FT_X2",   
    "DC_1H_1X", "DC_1H_12", "DC_1H_X2",   
    "DC_2H_1X", "DC_2H_12", "DC_2H_X2"    
]


class OddsPortalSoccerScraper:
    def __init__(self, headless: bool = True, max_workers: int = 10, auto_save_csv: Optional[str] = None):
        if auto_save_csv is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            auto_save_csv = os.path.join(data_dir, "soccer_historical.csv")
        self.headless = headless
        self.max_workers = max_workers
        self.auto_save_csv = auto_save_csv
        self.state_file = os.path.splitext(self.auto_save_csv)[0] + "_state.json"
        self.completed_competitions: Set[str] = set()
        self.completed_seasons: Set[str] = set()
        self.is_cancelled = False
        self.global_pause_event = asyncio.Event()
        self.global_pause_event.set()

    def log(self, msg: str):
        print(f"[SoccerScraper] {msg}", flush=True)

    def load_state(self):
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.completed_competitions = set(data.get("completed_competitions", []))
                    self.completed_seasons = set(data.get("completed_seasons", []))
                    last_comp = data.get("current_competition")
                    last_season = data.get("current_season")
                    total_scraped = data.get("total_matches_scraped", 0)
                    self.log(f"Loaded progress checkpoint: {len(self.completed_competitions)} completed competitions, {len(self.completed_seasons)} completed seasons.")
                    if last_comp:
                        self.log(f"  Checkpoint resume pointer: {last_comp} (Season: {last_season})")
                    if total_scraped:
                        self.log(f"  Total historical matches in checkpoint: {total_scraped}")
            except Exception as e:
                self.log(f"Warning loading state checkpoint: {e}")

    def save_state(self, current_comp: Optional[str] = None, current_season: Optional[str] = None, total_matches: int = 0):
        if not self.state_file:
            return
        try:
            out_dir = os.path.dirname(os.path.abspath(self.state_file))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            data = {
                "completed_competitions": sorted(list(self.completed_competitions)),
                "completed_seasons": sorted(list(self.completed_seasons)),
                "current_competition": current_comp,
                "current_season": current_season,
                "total_matches_scraped": total_matches,
                "last_updated": datetime.now().isoformat()
            }
            # Write in-place so editor file watchers (VS Code) trigger live tab reload
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log(f"Warning saving state checkpoint: {e}")

    async def wait_for_clearance(self):
        if not self.global_pause_event.is_set():
            await self.global_pause_event.wait()

    def trigger_rate_limit_pause(self, seconds: int = 60):
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
        
        string_cols = ["Date", "Time", "Country", "Competition", "Season", "HomeTeam", "AwayTeam", "Match_Status", "URL", "Is_Knockout", "Went_To_ET", "Match_Winner_Final"]
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
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="en-US",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
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
                                existing_df = pl.read_csv(self.auto_save_csv, schema_overrides=schema)
                                if "URL" in existing_df.columns:
                                    existing_df = existing_df.filter(pl.col("URL") != clean_url)
                                df = pl.concat([existing_df, df], how="diagonal_relaxed")
                            except Exception as e:
                                self.log(f"Warning reading existing CSV: {e}")
                        df.write_csv(self.auto_save_csv)
                        self.log(f"Saved dataset ({df.height} row) to {self.auto_save_csv}")
                    return df

                is_catalog = ("/results/football" in clean_url) or clean_url.rstrip('/').endswith("/football")

                # Competition Crawl
                competition_page = await context.new_page()
                await Stealth().apply_stealth_async(competition_page)

                self.load_state()
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

                if is_catalog:
                    raw_comps = await self.extract_competition_links(competition_page, clean_url)
                    competition_urls = raw_comps if raw_comps else [clean_url]
                else:
                    competition_urls = [clean_url]

                uncompleted_comps = [c for c in competition_urls if c not in self.completed_competitions]
                self.log(f"Competitions: {len(competition_urls)} total | {len(self.completed_competitions)} already completed | {len(uncompleted_comps)} remaining to process.")

                semaphore = asyncio.Semaphore(self.max_workers)

                for c_idx, comp_url in enumerate(competition_urls):
                    if comp_url in self.completed_competitions:
                        continue

                    comp_slug = comp_url.split('football/')[-1].strip('/')
                    remaining_count = len([c for c in competition_urls[c_idx:] if c not in self.completed_competitions])
                    self.log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    self.log(f"[Competition {c_idx+1}/{len(competition_urls)}] Starting: {comp_slug} ({len(self.completed_competitions)} completed, {remaining_count} remaining)")
                    self.log(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    self.save_state(current_comp=comp_url, current_season=None, total_matches=len(all_valid_rows))

                    if scrape_all_seasons:
                        extracted_seasons = await self.extract_season_links(competition_page, comp_url)
                        start_season_match = re.search(r'-(\d{4})(?:-(\d{4}))?/results/?$', comp_url)
                        if start_season_match:
                            start_year = int(start_season_match.group(1))
                            def get_year(u: str) -> int:
                                m = re.search(r'-(\d{4})(?:-(\d{4}))?/results/?$', u)
                                return int(m.group(1)) if m else 9999
                            season_urls = [s for s in extracted_seasons if get_year(s) <= start_year]
                            if not any(comp_url.rstrip('/') == s.rstrip('/') for s in season_urls):
                                season_urls.insert(0, comp_url)
                        else:
                            season_urls = extracted_seasons
                    else:
                        season_urls = [comp_url]

                    for s_idx, season_url in enumerate(season_urls):
                        if season_url in self.completed_seasons:
                            self.log(f"  [Season {s_idx+1}/{len(season_urls)}] Already marked completed: {season_url.split('football/')[-1]}. Skipping.")
                            continue

                        season_slug = season_url.split('football/')[-1] if 'football/' in season_url else season_url
                        self.log(f"  [Season {s_idx+1}/{len(season_urls)}] Scanning: {season_slug}")
                        
                        season_match = re.search(r'-(\d{4}-\d{4})', season_url) or re.search(r'-(\d{4})/', season_url)
                        if season_match:
                            raw_season = season_match.group(1)
                            current_season = raw_season.replace('-', '/') if '-' in raw_season else raw_season
                        else:
                            current_season = "2024/2025"

                        self.log(f"  [Season {s_idx+1}/{len(season_urls)}] Active Season: {current_season} ({season_slug})")
                        self.save_state(current_comp=comp_url, current_season=season_url, total_matches=len(all_valid_rows))
                        match_links = await self.extract_match_links(competition_page, season_url, scraped_urls)
                        
                        unscraped_links = [m for m in match_links if m not in scraped_urls]
                        if not unscraped_links:
                            self.log(f"  -> All {len(match_links)} matches already scraped for this season. Marking completed.")
                            self.completed_seasons.add(season_url)
                            self.save_state(current_comp=comp_url, current_season=season_url, total_matches=len(all_valid_rows))
                            continue

                        self.log(f"  -> Found {len(unscraped_links)} new matches to scrape in this season (Total in season: {len(match_links)}).")

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
                                
                                self.log(f"✅ [Saved] {r.get('HomeTeam')} vs {r.get('AwayTeam')} | FT: {r.get('FT_HomeScore')}-{r.get('FT_AwayScore')} | 1X2: {r.get('FT_HomeOdds')}/{r.get('FT_DrawOdds')}/{r.get('FT_AwayOdds')}")
                                
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
                                
                                # Real-time state synchronization
                                self.save_state(current_comp=comp_url, current_season=season_url, total_matches=len(all_valid_rows))

                        # Season finalized
                        self.completed_seasons.add(season_url)
                        self.save_state(current_comp=comp_url, current_season=season_url, total_matches=len(all_valid_rows))
                        self.log(f"  ✨ [Season Complete] {season_slug} finished. Total matches in dataset: {len(all_valid_rows)}")

                    # Competition finalized
                    self.completed_competitions.add(comp_url)
                    self.save_state(current_comp=comp_url, current_season=None, total_matches=len(all_valid_rows))
                    self.log(f"🏆 [Competition Complete] {comp_slug} successfully harvested!")

                await competition_page.close()
                return pl.DataFrame(all_valid_rows, schema=schema) if all_valid_rows else pl.DataFrame([], schema=schema)

            finally:
                await browser.close()

    async def extract_competition_links(self, page, catalog_url: str) -> List[str]:
        try:
            self.log(f"Fetching master competition catalog from {catalog_url}...")
            await page.goto(catalog_url, wait_until="domcontentloaded", timeout=45000)
            try:
                accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept"), .accept-choices').first
                if await accept_btn.count() > 0:
                    await accept_btn.click(timeout=3000)
            except Exception:
                pass
            await page.wait_for_timeout(3500)
            
            links = await page.evaluate("() => Array.from(document.body.querySelectorAll('a')).map(a => a.href)")
            
            comp_links = []
            seen = set()
            for l in links:
                if not l: continue
                norm = l.split('#')[0].split('?')[0].rstrip('/') + '/'
                if any(f"/{lang}/" in norm for lang in ['pl', 'de', 'es', 'ru', 'it', 'fr', 'gr']):
                    continue
                path_parts = [x for x in urlparse(norm).path.split('/') if x]
                if len(path_parts) == 4 and path_parts[0] == 'football' and path_parts[3] == 'results':
                    country = path_parts[1]
                    raw_league = path_parts[2]
                    if country in ['user', 'standings', 'results', 'outrights'] or raw_league in ['user', 'standings', 'results', 'outrights']:
                        continue
                    clean_league = re.sub(r'-\d{4}(?:-\d{4})?$', '', raw_league)
                    clean_comp = f"https://www.oddsportal.com/football/{country}/{clean_league}/results/"
                    if clean_comp not in seen:
                        seen.add(clean_comp)
                        comp_links.append(clean_comp)

            self.log(f"Discovered {len(comp_links)} football competitions across all regions.")
            return comp_links
        except Exception as e:
            self.log(f"Error extracting competition links from catalog: {e}")
            return []

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
                accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept"), .accept-choices').first
                if await accept_btn.count() > 0:
                    await accept_btn.click(timeout=3000)
            except Exception:
                pass
            await page.wait_for_timeout(3500)
            links = await page.evaluate("() => Array.from(document.body.querySelectorAll('a')).map(a => a.href)")
            
            season_links = []
            for l in links:
                if not l: continue
                norm = l.split('#')[0].split('?')[0].rstrip('/') + '/'
                if any(f"/{lang}/" in norm for lang in ['pl', 'de', 'es', 'ru', 'it', 'fr', 'gr']):
                    continue
                if norm.startswith(clean_root) and '/results/' in norm:
                    if re.search(r'-\d{4}(?:-\d{4})?/results/?$', norm) or norm == clean_root + '/results/':
                        season_links.append(norm)

            def get_season_year(u: str) -> int:
                m = re.search(r'-(\d{4})(?:-(\d{4}))?/results/?$', u)
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
                    let nav = document.querySelector('nav[aria-label="Pagination"], nav.pagination');
                    let parent = nav ? nav.parentElement : null;
                    let container = parent && parent.children.length > 0 ? parent.children[0] : document.body;

                    let links = Array.from(container.querySelectorAll('a[href*="/football/"]')).map(a => a.href);
                    let valid = [];
                    for (let l of links) {
                        let cleanUrl = l.split('?')[0];
                        let isExcluded = cleanUrl.includes('/results') || cleanUrl.includes('/standings') || cleanUrl.includes('/outrights') || cleanUrl.includes('/user/');
                        let isValid = (cleanUrl.includes('/h2h/') || /-[a-zA-Z0-9]{8}\/?(?:#.*)?$/.test(cleanUrl)) && !isExcluded;
                        if (isValid) {
                            valid.push(cleanUrl);
                        }
                    }
                    return valid;
                }
                """)
                for u in batch:
                    if u not in seen:
                        seen.add(u)
                        match_links.append(u)

            page_found = len(match_links) - page_seen_before
            self.log(f"  -> Page {page_num}: Extracted {page_found} matches (Total: {len(match_links)})")

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
        extracted_row["Is_Knockout"] = "No"
        extracted_row["Went_To_ET"] = "No"

        # 1. Load match page
        for attempt in range(3):
            try:
                await self.wait_for_clearance()
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)

                try:
                    accept_btn = page.locator('#onetrust-accept-btn-handler, button:has-text("I Accept"), .accept-choices').first
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

                if '/football/' not in page.url.lower():
                    return {"_skip": True, "URL": url}

                break
            except Exception:
                if attempt == 2: return {"_skip": True, "URL": url}
                await asyncio.sleep(4.0)

        # Wait for table elements to mount
        for _ in range(15):
            has_table = await page.evaluate("() => document.querySelectorAll('a[href*=\"/proxy/bookmakers/\"], [data-testid=\"odd-container\"], tr').length > 5")
            if has_table: break
            await page.wait_for_timeout(350)

        # 2. Extract scores, metadata, and score progression
        metadata = await page.evaluate(r"""
        () => {
            let res = {
                HomeTeam: '', AwayTeam: '', Country: '', Competition: '', Season: '', Date: '', Time: '', Match_Status: 'Finished',
                FT_HomeScore: null, FT_AwayScore: null, HT_HomeScore: null, HT_AwayScore: null,
                SH_HomeScore: null, SH_AwayScore: null, ET_HomeScore: null, ET_AwayScore: null,
                Penalties_HomeScore: null, Penalties_AwayScore: null, Went_To_ET: "No", Is_Knockout: "No", Match_Winner_Final: null
            };

            let clean = (s) => s ? s.replace(/[\u00a0\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim() : '';

            // Date & Time from data-testid="game-time-item"
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
                    if (!res.AwayTeam) res.AwayTeam = clean(parts[1].replace(/\s*(Odds|Scores|H2H|Football|Soccer).*$/i, ''));
                }
            }

            // Visual Breadcrumbs
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

            // Fallback from JSON-LD
            document.querySelectorAll('script[type="application/ld+json"]').forEach(script => {
                try {
                    let d = JSON.parse(script.textContent);
                    if (d["@type"] === "BreadcrumbList" && d.itemListElement) {
                        let items = d.itemListElement;
                        let cItem = items.find(i => i.position === "3" || i.position === 3);
                        let cmpItem = items.find(i => i.position === "4" || i.position === 4);
                        if (cItem && !res.Country) res.Country = cItem.name;
                        if (cmpItem && !res.Competition) res.Competition = cmpItem.name;
                    }
                    if (d["@type"]?.includes("Event")) {
                        let ldHome = d.homeTeam?.name;
                        let ldAway = d.awayTeam?.name;
                        if (ldHome && ldAway && (!res.HomeTeam || !res.AwayTeam)) {
                            res.HomeTeam = ldHome;
                            res.AwayTeam = ldAway;
                        }
                    }
                } catch(e) {}
            });

            // Score Timeline Processing
            let bodyText = clean(document.body.innerText);
            let scoreMatch = bodyText.match(/(?:Final\s*result|After\s*ET|Finished)\s*[:\s]*(\d+)\s*[:-]\s*(\d+)(?:\s*\(([^)]+)\))?/i);
            if (!scoreMatch) {
                scoreMatch = bodyText.match(/(\d+)\s*[:-]\s*(\d+)\s*\(([^)]+)\)/);
            }

            if (scoreMatch) {
                let pHome = parseFloat(scoreMatch[1]);
                let pAway = parseFloat(scoreMatch[2]);
                let splitText = scoreMatch[3] || "";

                let parsePair = (seg) => {
                    if (!seg) return [null, null];
                    let parts = seg.split(/[:-]/);
                    if (parts.length >= 2) {
                        return [parseFloat(parts[0].replace(/[^\d.-]/g, '')), parseFloat(parts[1].replace(/[^\d.-]/g, ''))];
                    }
                    return [null, null];
                };

                let isPen = bodyText.toLowerCase().includes('penalties') || splitText.toLowerCase().includes('pen');
                if (isPen) {
                    res.Penalties_HomeScore = pHome;
                    res.Penalties_AwayScore = pAway;
                    res.Went_To_ET = "Yes";
                    res.Is_Knockout = "Yes";

                    if (splitText) {
                        let segments = splitText.split(',').map(s => clean(s));
                        let s0 = parsePair(segments[0]);
                        if (s0[0] !== null) { res.FT_HomeScore = s0[0]; res.FT_AwayScore = s0[1]; }
                        let s1 = parsePair(segments[1]);
                        if (s1[0] !== null) { res.HT_HomeScore = s1[0]; res.HT_AwayScore = s1[1]; }
                        let s2 = parsePair(segments[2]);
                        if (s2[0] !== null) { res.SH_HomeScore = s2[0]; res.SH_AwayScore = s2[1]; }
                        let s3 = parsePair(segments[3]);
                        if (s3[0] !== null) { res.ET_HomeScore = s3[0]; res.ET_AwayScore = s3[1]; }
                    }
                } else {
                    res.FT_HomeScore = pHome;
                    res.FT_AwayScore = pAway;
                    if (splitText) {
                        let segments = splitText.split(',').map(s => clean(s));
                        let s0 = parsePair(segments[0]);
                        if (s0[0] !== null) { res.HT_HomeScore = s0[0]; res.HT_AwayScore = s0[1]; }
                        let s1 = parsePair(segments[1]);
                        if (s1[0] !== null) { res.SH_HomeScore = s1[0]; res.SH_AwayScore = s1[1]; }
                        if (segments.length > 2 && segments[2]) {
                            let s2 = parsePair(segments[2]);
                            if (s2[0] !== null) {
                                res.ET_HomeScore = s2[0];
                                res.ET_AwayScore = s2[1];
                                res.Went_To_ET = "Yes";
                                res.Is_Knockout = "Yes";
                            }
                        }
                    }
                }

                // If SH score is missing but FT and HT exist, calculate SH = FT - HT
                if (res.FT_HomeScore !== null && res.HT_HomeScore !== null && res.SH_HomeScore === null) {
                    res.SH_HomeScore = res.FT_HomeScore - res.HT_HomeScore;
                }
                if (res.FT_AwayScore !== null && res.HT_AwayScore !== null && res.SH_AwayScore === null) {
                    res.SH_AwayScore = res.FT_AwayScore - res.HT_AwayScore;
                }

                if (res.FT_HomeScore !== null && res.FT_AwayScore !== null) {
                    if (res.FT_HomeScore > res.FT_AwayScore) res.Match_Winner_Final = "Home";
                    else if (res.FT_HomeScore < res.FT_AwayScore) res.Match_Winner_Final = "Away";
                    else res.Match_Winner_Final = "Draw";
                }
            }

            return res;
        }
        """)
        extracted_row.update(metadata)

        if season:
            extracted_row["Season"] = season
        elif not extracted_row.get("Season") or extracted_row["Season"] in ["", "None", None]:
            s_url_match = re.search(r'-(\d{4}-\d{4})', url) or re.search(r'-(\d{4})/', url)
            if s_url_match:
                extracted_row["Season"] = s_url_match.group(1).replace('-', '/')
            else:
                date_str = extracted_row.get("Date", "")
                if date_str:
                    d_parts = date_str.split()
                    if len(d_parts) == 3 and d_parts[2].isdigit():
                        month = d_parts[1].lower()
                        year = int(d_parts[2])
                        if month in ["aug", "sep", "oct", "nov", "dec"]:
                            extracted_row["Season"] = f"{year}/{year + 1}"
                        else:
                            extracted_row["Season"] = f"{year - 1}/{year}"

        if not extracted_row.get("HomeTeam") or not extracted_row.get("AwayTeam"):
            return {"_skip": True, "URL": url}

        if extracted_row.get("FT_HomeScore") is None or extracted_row.get("FT_AwayScore") is None:
            return {"_skip": True, "URL": url}

        # 3. Client-side V8 Market Switcher
        async def switch_market(main_target: str) -> bool:
            return await page.evaluate(r"""(target) => {
                let norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
                let goal = norm(target);

                // Check direct tabs
                let allBtns = Array.from(document.querySelectorAll('button, a, li.tab-item, div[role="tab"]'));
                let direct = allBtns.find(b => {
                    let t = norm(b.innerText || '');
                    return t === goal || (goal === 'dnb' && (t === 'dnb' || t === 'drawnobet'));
                });

                if (direct && !direct.className.includes('hidden') && direct.offsetParent !== null) {
                    let clickTarget = direct.querySelector('button, a') || direct;
                    clickTarget.click();
                    return true;
                }

                // Check if dropdown menu is already open
                let menuOpen = Array.from(document.querySelectorAll('div, ul')).some(el => {
                    return (el.className || '').includes('dropdown-menu') || el.getAttribute('role') === 'menu';
                });

                if (!menuOpen) {
                    let moreBtn = Array.from(document.querySelectorAll('div, button, span')).find(el => {
                        let t = (el.innerText || '').trim().toLowerCase();
                        return t === 'more' && el.children.length <= 2;
                    });
                    if (moreBtn) {
                        let clk = moreBtn.closest('div.cursor-pointer, button') || moreBtn;
                        clk.click();
                    }
                }

                let dropItems = Array.from(document.querySelectorAll('button, a, li, div[role="menuitem"], div[class*="dropdown"] *'));
                let dropItem = dropItems.find(b => {
                    let t = norm(b.innerText || '');
                    return t === goal || (goal === 'dnb' && (t === 'dnb' || t === 'drawnobet'));
                });
                if (dropItem) {
                    let clickTarget = dropItem.querySelector('button, a') || dropItem;
                    clickTarget.click();
                    return true;
                }
                return false;
            }""", main_target)

        # 4. Client-side V8 Sub-tab Switcher
        async def switch_sub(sub_target: str) -> bool:
            return await page.evaluate(r"""(subName) => {
                let norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
                let goalSub = norm(subName);
                let subAliases = [goalSub];
                if (goalSub === 'fulltime') subAliases = ['fulltime', 'ft'];
                if (goalSub === '1sthalf') subAliases = ['1sthalf', '1h', 'half1'];
                if (goalSub === '2ndhalf') subAliases = ['2ndhalf', '2h', 'half2'];

                let subTabs = Array.from(document.querySelectorAll('ul li, button, a, div[role="tab"], div.cursor-pointer')).filter(el => {
                    let text = norm(el.innerText || '');
                    return subAliases.some(a => text === a) && el.offsetParent !== null;
                });
                if (subTabs.length > 0) {
                    let target = subTabs[subTabs.length - 1];
                    let subBtn = target.querySelector('button, a') || target;
                    subBtn.click();
                    return true;
                }
                return false;
            }""", sub_target)

        # 5. Client-side V8 Odds Extractor with Strikethrough Filtering & Bet365 Priority
        async def extract_active_odds(exp_count: int = 3) -> Optional[List[float]]:
            for _ in range(4):
                odds = await page.evaluate(r"""(expCount) => {
                    let clean = s => s ? s.replace(/[\u00a0\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim() : '';
                    let rows = Array.from(document.querySelectorAll('tr, div[data-testid="bookmaker-table-row"], div.border-b'));
                    let bet365Odds = null;
                    let preferredOdds = null;
                    let fallbackOdds = null;

                    for (let r of rows) {
                        let text = clean(r.innerText).toLowerCase();
                        let bookieLink = r.querySelector('a[href*="/proxy/bookmakers/"], a[href*="/bookmakers/"]');
                        let href = bookieLink ? bookieLink.getAttribute('href').toLowerCase() : '';
                        let isBet365 = href.includes('bet365') || text.includes('bet365');

                        // Skip struck-through or disabled bookmaker rows
                        let isStruck = !!r.querySelector('.line-through, [class*="line-through"], del, s') ||
                                       !!r.querySelector('.text-gray-400, .opacity-40, [class*="disabled"]');
                        if (isStruck) continue;

                        let oddsNodes = r.querySelectorAll('p.font-main.text-xs, [data-testid="odd-container"] p, .odds-text');
                        let odds = [];
                        for (let n of oddsNodes) {
                            let valStr = clean(n.innerText);
                            if (/^\d+\.\d{2}$/.test(valStr) && !valStr.includes('%')) {
                                let f = parseFloat(valStr);
                                if (f >= 1.01 && f <= 100.0) odds.push(f);
                            }
                        }

                        if (odds.length >= expCount) {
                            let slice = odds.slice(0, expCount);
                            if (isBet365) return slice;
                            let isSharp = href.includes('pinnacle') || href.includes('888sport') || href.includes('bwin') || href.includes('unibet');
                            if (isSharp && !preferredOdds) preferredOdds = slice;
                            if (!fallbackOdds) fallbackOdds = slice;
                        }
                    }
                    return preferredOdds || fallbackOdds;
                }""", exp_count)
                if odds:
                    return odds
                await page.wait_for_timeout(400)
            return None

        # ----------------- PHASE 1: 1X2 MARKETS -----------------
        # Full Time
        await switch_market("1X2")
        await page.wait_for_timeout(800)
        await switch_sub("Full Time")
        await page.wait_for_timeout(600)
        ft_odds = await extract_active_odds(3)
        if ft_odds and len(ft_odds) >= 3:
            extracted_row["FT_HomeOdds"], extracted_row["FT_DrawOdds"], extracted_row["FT_AwayOdds"] = ft_odds[:3]

        # 1st Half
        await switch_sub("1st Half")
        await page.wait_for_timeout(600)
        h1_odds = await extract_active_odds(3)
        if h1_odds and len(h1_odds) >= 3:
            extracted_row["1H_HomeOdds"], extracted_row["1H_DrawOdds"], extracted_row["1H_AwayOdds"] = h1_odds[:3]

        # 2nd Half
        await switch_sub("2nd Half")
        await page.wait_for_timeout(600)
        h2_odds = await extract_active_odds(3)
        if h2_odds and len(h2_odds) >= 3:
            extracted_row["SH_HomeOdds"], extracted_row["SH_DrawOdds"], extracted_row["SH_AwayOdds"] = h2_odds[:3]

        # ----------------- PHASE 2: OVER/UNDER ACCORDION EXTRACTION -----------------
        try:
            await switch_market("Over/Under")
            await page.wait_for_timeout(1000)

            ou_extracted = await page.evaluate(r"""() => {
                let clean = s => s ? s.replace(/[\u00a0\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim() : '';
                let targetLines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
                let allRows = Array.from(document.querySelectorAll('div, tr')).filter(el => {
                    let t = (el.innerText || '').trim();
                    return /^Over\/Under\s*[\+\-]?\d+(\.\d+)?/i.test(t) && t.length < 150;
                });

                let results = {};
                for (let line of targetLines) {
                    let header = allRows.find(el => {
                        let m = el.innerText.trim().match(/Over\/Under\s*([\+\-]?\d+(?:\.\d+)?)/i);
                        return m && Math.abs(parseFloat(m[1]) - line) < 0.01;
                    });
                    if (header) {
                        let text = clean(header.innerText);
                        let nums = [];
                        for (let p of text.split(/\s+/)) {
                            if (/^\d+\.\d{2}$/.test(p) && !p.includes('%')) {
                                let f = parseFloat(p);
                                if (f >= 1.01 && f <= 100.0 && Math.abs(f - line) > 0.01) nums.push(f);
                            }
                        }
                        if (nums.length >= 2) {
                            results[line] = [nums[0], nums[1]];
                        }
                    }
                }
                return results;
            }""")

            if ou_extracted:
                for line, odds_pair in ou_extracted.items():
                    line_val = float(line)
                    key_prefix = f"OU{int(line_val * 10):02d}"
                    if odds_pair and len(odds_pair) >= 2:
                        extracted_row[f"{key_prefix}_Over"] = odds_pair[0]
                        extracted_row[f"{key_prefix}_Under"] = odds_pair[1]
        except Exception as e:
            self.log(f"Notice during Over/Under pass: {e}")

        # ----------------- PHASE 3: BOTH TEAMS TO SCORE (BTTS) -----------------
        # Full Time
        await switch_market("Both Teams to Score")
        await page.wait_for_timeout(800)
        await switch_sub("Full Time")
        await page.wait_for_timeout(600)
        btts_ft = await extract_active_odds(2)
        if btts_ft and len(btts_ft) >= 2:
            extracted_row["BTTS_Yes"], extracted_row["BTTS_No"] = btts_ft[:2]

        # 1st Half
        await switch_sub("1st Half")
        await page.wait_for_timeout(600)
        btts_1h = await extract_active_odds(2)
        if btts_1h and len(btts_1h) >= 2:
            extracted_row["BTTS_1H_Yes"], extracted_row["BTTS_1H_No"] = btts_1h[:2]

        # 2nd Half
        await switch_sub("2nd Half")
        await page.wait_for_timeout(600)
        btts_2h = await extract_active_odds(2)
        if btts_2h and len(btts_2h) >= 2:
            extracted_row["BTTS_2H_Yes"], extracted_row["BTTS_2H_No"] = btts_2h[:2]

        # ----------------- PHASE 4: DOUBLE CHANCE -----------------
        # Full Time
        await switch_market("Double Chance")
        await page.wait_for_timeout(1000)
        await switch_sub("Full Time")
        await page.wait_for_timeout(600)
        dc_ft = await extract_active_odds(3)
        if dc_ft and len(dc_ft) >= 3:
            extracted_row["DC_FT_1X"], extracted_row["DC_FT_12"], extracted_row["DC_FT_X2"] = dc_ft[:3]

        # 1st Half
        await switch_sub("1st Half")
        await page.wait_for_timeout(600)
        dc_1h = await extract_active_odds(3)
        if dc_1h and len(dc_1h) >= 3:
            extracted_row["DC_1H_1X"], extracted_row["DC_1H_12"], extracted_row["DC_1H_X2"] = dc_1h[:3]

        # 2nd Half
        await switch_sub("2nd Half")
        await page.wait_for_timeout(600)
        dc_2h = await extract_active_odds(3)
        if dc_2h and len(dc_2h) >= 3:
            extracted_row["DC_2H_1X"], extracted_row["DC_2H_12"], extracted_row["DC_2H_X2"] = dc_2h[:3]

        # ----------------- PHASE 5: DRAW NO BET (DNB) -----------------
        await switch_market("Draw No Bet")
        await page.wait_for_timeout(1000)
        await switch_sub("Full Time")
        await page.wait_for_timeout(600)
        dnb_ft = await extract_active_odds(2)
        if dnb_ft and len(dnb_ft) >= 2:
            extracted_row["DNB_Home"], extracted_row["DNB_Away"] = dnb_ft[:2]

        return extracted_row


def main():
    parser = argparse.ArgumentParser(description="OddsPortal Historical Soccer Scraper")
    parser.add_argument("positional_url", nargs="?", default=None, help="Target URL (positional)")
    parser.add_argument("--url", type=str, default=None, help="Target competition, catalog, or match URL")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent browser workers (default: 10)")
    parser.add_argument("--single-season", action="store_true", help="Scrape only a single season without historical crawl")
    parser.add_argument("--output", type=str, default=None, help="Output CSV filepath")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible (non-headless) mode")

    args = parser.parse_args()

    test_url = args.url or args.positional_url or "https://www.oddsportal.com/results/football/"

    scraper = OddsPortalSoccerScraper(
        headless=not args.headed,
        max_workers=args.max_workers,
        auto_save_csv=args.output
    )

    print("=" * 75)
    print("⚽ SOCCER HISTORICAL ODDS SCRAPER")
    print(f"Target URL: {test_url}")
    print(f"Mode: {'Single Season' if args.single_season else 'Comprehensive Historical Crawl'}")
    print(f"Workers: {args.max_workers} | Headless: {not args.headed}")
    print(f"Output CSV: {scraper.auto_save_csv}")
    print(f"State File: {scraper.state_file}")
    print("=" * 75)

    df = asyncio.run(scraper.run(test_url, scrape_all_seasons=not args.single_season))
    print(f"\nScraping complete. Harvested {df.height} matches with {len(df.columns)} fields.")


if __name__ == "__main__":
    main()
