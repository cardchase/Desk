# OddsPortal Scraper Engineering Rules

When working on the OddsPortal scraper (`odds_portal_historical.py`, `BaseNode`, or any batch runners), you MUST adhere to these hard-learned rules to prevent breaking the ETL pipeline:

### 1. React DOM Hydration (The "Jiggle")
OddsPortal is a heavily obfuscated React SPA. Simply waiting for network requests is not enough. If the DOM stalls while polling for odds, you MUST use "Jiggle Physics" — programmatically clicking an adjacent sub-tab (like "1st Half") and then immediately clicking back to the target tab (like "Full Time") to force React to re-render the market table.

### 2. Accordion Expansion Safety
When expanding markets like Over/Under, strictly target `[data-testid="over-under-collapsed-row"]`. NEVER use broad selectors like `.cursor-pointer`, as this will accidentally click and close accordions that OddsPortal has already opened, resulting in zero extracted odds.

### 3. Micro-Timeouts for Historical Data
Historical matches frequently lack granular markets (e.g., 2010 matches may not have "2nd Half" odds). You must keep Playwright `wait_for()` timeouts for sub-tabs extremely short (e.g., `2500ms`). If you use long timeouts (like 25 seconds), the headless browsers will sleep indefinitely on missing tabs, causing CPU utilization to plummet and the scrape to freeze.

### 4. Hardcore Ad-Blocking
OddsPortal aggressively spawns hidden pop-under tabs for casinos (e.g., n1-bet). You must maintain a `context.on("page")` event listener that immediately inspects all new tabs. If the URL does not contain `oddsportal.com/football`, the tab must be forcefully and instantly closed before it loads.

### 5. Cross-Season State Isolation
When iterating backward through historical seasons, pagination states (like counting how many pages are fully scraped) MUST be completely reset to `0` at the start of every new season. Failing to isolate this state will cause the scraper to skip older seasons if a newer season was already completed.

### 6. Terminal Telemetry & Timestamps
Because the nodes run asynchronously in background ThreadPools or Pools, `sys.stdout` logging is easily lost. The root runner (e.g., `run_batch_scraping.py`) must explicitly configure `logging.basicConfig` pointing to `sys.stdout`. Furthermore, the base logger (`BaseNode.log`) must explicitly generate and inject its own `[{timestamp}]` tag into the string, as the background multiprocessing can strip standard log formats.
