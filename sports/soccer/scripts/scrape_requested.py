import concurrent.futures
import sys
import os

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
from run_batch_scraping import process_competition

urls = [
    "https://www.oddsportal.com/football/england/fa-cup/results/",
    "https://www.oddsportal.com/football/europe/champions-league/results/",
    "https://www.oddsportal.com/football/europe/europa-league/results/",
    "https://www.oddsportal.com/football/europe/conference-league/results/"
]

if __name__ == "__main__":
    print("Starting custom batch scraper for requested leagues...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_competition, url, 3, True) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")
