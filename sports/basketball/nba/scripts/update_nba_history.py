"""
NBA HISTORICAL RESULTS UPDATER
==============================
Scrapes newly finished NBA matches from OddsPortal and automatically appends them 
to the master historical training dataset (backend/outputs/intermediates/nba_historical.csv).

Features:
- Incremental / Resume logic: Checks which matches already exist in nba_historical.csv.
- Only scrapes newly finished games (final scores, quarter progression, closing odds).
- Safely appends and updates rolling features for subsequent predictions.

Usage:
  python test_NBA/update_nba_history.py
"""

import sys
import os
import subprocess
from pathlib import Path
import polars as pl

# Configure Windows terminal encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DATA_DIR = SCRIPT_DIR.parent / "data"
HISTORICAL_SCRAPER = SCRIPT_DIR / "basketball_historical.py"
HISTORICAL_CSV = DATA_DIR / "nba_historical.csv"
RESULTS_URL = "https://www.oddsportal.com/basketball/usa/nba/results/"


def update_history():
    print("=" * 80)
    print("🔄 NBA HISTORICAL DATABASE INCREMENTAL UPDATER")
    print("=" * 80)

    initial_count = 0
    if HISTORICAL_CSV.exists():
        try:
            df = pl.read_csv(HISTORICAL_CSV)
            initial_count = df.height
            last_date = df["Date"][-1] if "Date" in df.columns and df.height > 0 else "Unknown"
            print(f"📊 Current Historical Records : {initial_count:,} matches")
            print(f"📅 Latest Match in Database   : {last_date}")
        except Exception as e:
            print(f"Notice reading existing CSV: {e}")

    print(f"\n📡 Scanning OddsPortal Results page for newly finished games...")
    print(f"   Target: {RESULTS_URL}\n")

    cmd = [
        sys.executable,
        str(HISTORICAL_SCRAPER),
        RESULTS_URL,
        "--single-season"
    ]

    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        if proc.returncode == 0 and HISTORICAL_CSV.exists():
            df_after = pl.read_csv(HISTORICAL_CSV)
            new_count = df_after.height
            diff = new_count - initial_count
            print("\n" + "=" * 80)
            if diff > 0:
                print(f"✅ SUCCESS! Added {diff} newly finished matches to historical dataset.")
                print(f"📊 Total Historical Records Now: {new_count:,} matches.")
            else:
                print("✅ UP TO DATE! No new finished matches found on OddsPortal.")
                print(f"📊 Historical database remains at {initial_count:,} matches.")
            print("=" * 80 + "\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error updating historical results: {e}")


if __name__ == "__main__":
    update_history()
