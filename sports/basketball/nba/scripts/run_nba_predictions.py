"""
NBA PREDICTION & BETTING COMMAND CENTER (INTERACTIVE RUNNER)
============================================================
Institutionalized, single-entrypoint command for NBA sports predictions and bet placements.

Workflow:
1. Interactive prompt or CLI flags:
   - Option to scrape fresh odds from OddsPortal (or use cached upcoming_games.csv)
   - Slate selection (Today, Tomorrow, All, or Custom Date)
2. Executes Master ML Prediction Engine (nba_master_predictor.py)
3. Generates two deliverables:
   - Detailed Multi-Market Card: test_NBA/weekly_betting_card.csv
   - Executive Actionable Checklist: test_NBA/actionable_betting_checklist.csv
4. Displays formatted Actionable Checklist with plain-English game notes directly on the console.

Usage:
  python test_NBA/run_nba_predictions.py                 # Fully interactive prompt menu
  python test_NBA/run_nba_predictions.py --today         # Predict today's games (skips prompts)
  python test_NBA/run_nba_predictions.py --tomorrow      # Predict tomorrow's games
  python test_NBA/run_nba_predictions.py --all           # Predict all upcoming games
  python test_NBA/run_nba_predictions.py --scrape        # Force scrape live OddsPortal odds first
  python test_NBA/run_nba_predictions.py --date "24 Oct" # Predict specific date
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import datetime

# Configure Windows terminal encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
SCRAPER_SCRIPT = SCRIPT_DIR / "basketball_upcoming.py"
PREDICTOR_SCRIPT = SCRIPT_DIR / "nba_master_predictor.py"
UPCOMING_CSV = DATA_DIR / "upcoming_games.csv"
ACTIONABLE_CSV = DATA_DIR / "actionable_betting_checklist.csv"


def prompt_user_menu() -> dict:
    """Interactively prompts the user for run preferences."""
    now = datetime.datetime.now()
    today_str = now.strftime('%d %b %Y')
    tomorrow_str = (now + datetime.timedelta(days=1)).strftime('%d %b %Y')

    print("\n" + "=" * 80)
    print("🏀 NBA PREDICTION & BETTING COMMAND CENTER")
    print(f"   Current System Date/Time: {now.strftime('%A, %d %B %Y - %I:%M %p')}")
    print("=" * 80)

    # Question 1: Scrape live odds?
    file_status = "Not Found"
    if UPCOMING_CSV.exists():
        mtime = datetime.datetime.fromtimestamp(UPCOMING_CSV.stat().st_mtime)
        file_status = f"Found (Last scraped: {mtime.strftime('%d %b %I:%M %p')})"

    print(f"\n[STEP 1: ODDS & FIXTURE DATA] (Current upcoming_games.csv: {file_status})")
    print("Do you want to scrape fresh upcoming games & odds from OddsPortal?")
    print("  [1] Yes - Scrape live odds now (~60-90 seconds)")
    print("  [2] No  - Use existing upcoming_games.csv (Instant, Recommended if already scraped)")
    
    choice_scrape = input("Enter choice [1 or 2] (default=2): ").strip()
    should_scrape = (choice_scrape == "1")

    # Question 2: Which games to predict?
    print("\n[STEP 2: PREDICTION SLATE SELECTION]")
    print("Which games do you want to run predictions for?")
    print(f"  [1] Today's Games ({today_str})")
    print(f"  [2] Tomorrow's Games ({tomorrow_str})")
    print("  [3] All Upcoming Games in Slate")
    print("  [4] Custom Date (Enter any date or team keyword)")

    choice_date = input("Enter choice [1-4] (default=1): ").strip()
    if choice_date == "2":
        date_arg = "tomorrow"
    elif choice_date == "3":
        date_arg = "all"
    elif choice_date == "4":
        custom = input("Enter target date (e.g., '24 Oct', '2026-10-24'): ").strip()
        date_arg = custom if custom else "today"
    else:
        date_arg = "today"

    return {
        "scrape": should_scrape,
        "date": date_arg
    }


def run_scraper(date_filter: str = None, concurrency: int = 10) -> bool:
    """Executes the Playwright scraper to refresh upcoming games and odds."""
    print("\n" + "=" * 80)
    print(f"📡 SCRAPING LIVE ODDS FROM ODDSPORTAL (Concurrency: {concurrency})...")
    print("=" * 80)
    cmd = [sys.executable, str(SCRAPER_SCRIPT), "--concurrency", str(concurrency)]
    if date_filter and date_filter not in ["all", "full"]:
        cmd.extend(["--date", date_filter])

    try:
        proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)
        return proc.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error during scraping: {e}")
        return False


def run_predictor(date_arg: str, stress_test: bool = True) -> bool:
    """Executes the master prediction pipeline."""
    print("\n" + "=" * 80)
    print(f"🧠 RUNNING MASTER PREDICTOR ENGINE FOR SLATE: [{date_arg.upper()}]...")
    print("=" * 80)
    cmd = [sys.executable, str(PREDICTOR_SCRIPT), "--date", date_arg]
    if stress_test:
        cmd.append("--stress")

    try:
        proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)
        return proc.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction execution: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Command Center")
    parser.add_argument("--date", type=str, default=None, help="Target date (e.g. '24 Oct', '2026-10-24')")
    parser.add_argument("--today", action="store_true", help="Predict today's games immediately")
    parser.add_argument("--tomorrow", action="store_true", help="Predict tomorrow's games immediately")
    parser.add_argument("--all", action="store_true", help="Predict all upcoming games immediately")
    parser.add_argument("--scrape", action="store_true", help="Force live scrape before predicting")
    parser.add_argument("--no-scrape", action="store_true", help="Skip live scrape")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Scraper concurrency (default: 10)")
    parser.add_argument("--no-stress", action="store_true", help="Disable line sensitivity stress test")
    args = parser.parse_args()

    # Determine execution mode (interactive vs command-line flags)
    has_cli_flags = any([args.date, args.today, args.tomorrow, args.all, args.scrape, args.no_scrape])

    if has_cli_flags:
        if args.today:
            target_date = "today"
        elif args.tomorrow:
            target_date = "tomorrow"
        elif args.all:
            target_date = "all"
        elif args.date:
            target_date = args.date
        else:
            target_date = "today"

        should_scrape = args.scrape and not args.no_scrape
    else:
        # Interactive Prompt Menu
        user_pref = prompt_user_menu()
        should_scrape = user_pref["scrape"]
        target_date = user_pref["date"]

    # Step 1: Optional Scraping
    if should_scrape:
        success = run_scraper(date_filter=target_date if target_date not in ["all", "full"] else None, concurrency=args.concurrency)
        if not success:
            print("Scraping encountered an issue. Falling back to existing upcoming_games.csv...")

    # Step 2: Prediction & Checklist Generation
    run_predictor(date_arg=target_date, stress_test=not args.no_stress)

    # Step 3: Final confirmation
    if ACTIONABLE_CSV.exists():
        print("=" * 80)
        print("✅ SUCCESS! Your Actionable Betting Checklist is updated and ready at:")
        print(f"   {ACTIONABLE_CSV}")
        print("   Open this CSV anytime in Excel or Google Sheets for your exact bet placements.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
