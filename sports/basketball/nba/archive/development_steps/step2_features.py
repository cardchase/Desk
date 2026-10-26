import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV_PATH = BASE_DIR.parent / "backend" / "outputs" / "intermediates" / "nba_historical.csv"
OUTPUT_CSV_PATH = BASE_DIR / "nba_features.csv"

# Fallback to absolute path if running from strange context
if not RAW_CSV_PATH.exists():
    RAW_CSV_PATH = Path(r"d:\Project\LoomFlow - Sports\backend\outputs\intermediates\nba_historical.csv")

print("=" * 65)
print("PHASE 2: LEAK-FREE FEATURE ENGINEERING PIPELINE")
print("=" * 65)
print(f"Reading from: {RAW_CSV_PATH}")

# 1. Load & Clean Dates
df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
df['Date_Clean'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')

# 2. Filter out the 11 Exhibition / All-Star Teams & Invalid Scores
non_standard_teams = [
    'East', 'West', 'Team Chuck', 'Team Durant', 'Team Giannis',
    'Team LeBron', 'Team Shaq', 'Team Stars', 'Team Stripes',
    'Team USA', 'Team World'
]
mask_exhibition = df['HomeTeam'].isin(non_standard_teams) | df['AwayTeam'].isin(non_standard_teams)
print(f"Removed {mask_exhibition.sum()} exhibition/All-Star games.")
df = df[~mask_exhibition].copy()

# Ensure core score columns are numeric and drop rows missing actual scores
score_cols = [
    'FT_HomeScore', 'FT_AwayScore',
    'Q1_HomeScore', 'Q1_AwayScore', 'Q2_HomeScore', 'Q2_AwayScore',
    'Q3_HomeScore', 'Q3_AwayScore', 'Q4_HomeScore', 'Q4_AwayScore'
]
for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

before_drop = len(df)
df = df.dropna(subset=score_cols).copy()
print(f"Removed {before_drop - len(df)} rows with missing quarter/FT scores.")

# Recompute 1st Half and 2nd Half cleanly from Quarters
df['1H_HomeScore'] = df['Q1_HomeScore'] + df['Q2_HomeScore']
df['1H_AwayScore'] = df['Q1_AwayScore'] + df['Q2_AwayScore']
df['2H_HomeScore'] = df['Q3_HomeScore'] + df['Q4_HomeScore']
df['2H_AwayScore'] = df['Q3_AwayScore'] + df['Q4_AwayScore']

# Sort chronologically Oldest -> Newest and assign a unique Game_ID
df = df.sort_values(by=['Date_Clean', 'Time', 'HomeTeam']).reset_index(drop=True)
df['Game_ID'] = np.arange(len(df))
print(f"Clean NBA games ready for processing: {len(df):,}")

# -----------------------------------------------------------------
# 3. DYNAMIC ELO RATING SYSTEM (With Margin of Victory & Season Reset)
# -----------------------------------------------------------------
print("Calculating Dynamic Elo Ratings (K=20, Home Advantage=70)...")
elo_dict = {team: 1500.0 for team in df['HomeTeam'].unique()}
K_FACTOR = 20.0
HOME_ADV = 70.0

home_elo_pre = []
away_elo_pre = []
elo_win_prob_home = []

current_season = None

for idx, row in df.iterrows():
    season = row['Season']
    # Regress Elo 25% toward 1500 at the start of every new season
    if current_season is not None and season != current_season:
        for t in elo_dict:
            elo_dict[t] = 0.75 * elo_dict[t] + 0.25 * 1500.0
    current_season = season

    h_team = row['HomeTeam']
    a_team = row['AwayTeam']
    h_elo = elo_dict.get(h_team, 1500.0)
    a_elo = elo_dict.get(a_team, 1500.0)

    # Record PRE-GAME Elo (Zero leakage!)
    home_elo_pre.append(h_elo)
    away_elo_pre.append(a_elo)

    # Expected Home Win Probability
    exp_home = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + HOME_ADV)) / 400.0))
    elo_win_prob_home.append(exp_home)

    # Post-game update using Margin of Victory multiplier (FiveThirtyEight formula)
    margin = row['FT_HomeScore'] - row['FT_AwayScore']
    actual_home = 1.0 if margin > 0 else 0.0
    elo_diff_winner = (h_elo + HOME_ADV - a_elo) if margin > 0 else (a_elo - (h_elo + HOME_ADV))
    mov_mult = ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)

    shift = K_FACTOR * mov_mult * (actual_home - exp_home)
    elo_dict[h_team] = h_elo + shift
    elo_dict[a_team] = a_elo - shift

df['Home_Elo_Pre'] = home_elo_pre
df['Away_Elo_Pre'] = away_elo_pre
df['Elo_Diff'] = df['Home_Elo_Pre'] - df['Away_Elo_Pre']
df['Elo_Home_Win_Prob'] = elo_win_prob_home

# -----------------------------------------------------------------
# 4. TEAM-LEVEL ROLLING STATS (Last 5 & Last 15 Games)
# -----------------------------------------------------------------
print("Computing leak-free rolling team statistics (L5 & L15 windows)...")

# Ensure key market lines are numeric
odds_num_cols = [
    'HA_FT_Home', 'HA_FT_Away', 'HA_1H_Home', 'HA_1H_Away', 'HA_Q1_Home', 'HA_Q1_Away',
    'OU_FT_Line', 'OU_FT_Over', 'OU_FT_Under',
    'OU_1H_Line', 'OU_1H_Over', 'OU_1H_Under',
    'OU_Q1_Line', 'OU_Q1_Over', 'OU_Q1_Under'
]
for col in odds_num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Build Long Table (1 row per team per game)
home_log = pd.DataFrame({
    'Game_ID': df['Game_ID'],
    'Date_Clean': df['Date_Clean'],
    'Season': df['Season'],
    'Team': df['HomeTeam'],
    'Is_Home': 1,
    'Win': (df['FT_HomeScore'] > df['FT_AwayScore']).astype(float),
    'FT_Pts_For': df['FT_HomeScore'],
    'FT_Pts_Against': df['FT_AwayScore'],
    'FT_Total': df['FT_HomeScore'] + df['FT_AwayScore'],
    'FT_Margin': df['FT_HomeScore'] - df['FT_AwayScore'],
    '1H_Pts_For': df['1H_HomeScore'],
    '1H_Pts_Against': df['1H_AwayScore'],
    '1H_Total': df['1H_HomeScore'] + df['1H_AwayScore'],
    '1H_Margin': df['1H_HomeScore'] - df['1H_AwayScore'],
    'Q1_Pts_For': df['Q1_HomeScore'],
    'Q1_Pts_Against': df['Q1_AwayScore'],
    'Q1_Total': df['Q1_HomeScore'] + df['Q1_AwayScore'],
    'Q1_Margin': df['Q1_HomeScore'] - df['Q1_AwayScore'],
    'Beat_OU_FT': (df['FT_HomeScore'] + df['FT_AwayScore']) - df['OU_FT_Line'],
    'Beat_OU_Q1': (df['Q1_HomeScore'] + df['Q1_AwayScore']) - df['OU_Q1_Line'],
})

away_log = pd.DataFrame({
    'Game_ID': df['Game_ID'],
    'Date_Clean': df['Date_Clean'],
    'Season': df['Season'],
    'Team': df['AwayTeam'],
    'Is_Home': 0,
    'Win': (df['FT_AwayScore'] > df['FT_HomeScore']).astype(float),
    'FT_Pts_For': df['FT_AwayScore'],
    'FT_Pts_Against': df['FT_HomeScore'],
    'FT_Total': df['FT_HomeScore'] + df['FT_AwayScore'],
    'FT_Margin': df['FT_AwayScore'] - df['FT_HomeScore'],
    '1H_Pts_For': df['1H_AwayScore'],
    '1H_Pts_Against': df['1H_HomeScore'],
    '1H_Total': df['1H_HomeScore'] + df['1H_AwayScore'],
    '1H_Margin': df['1H_AwayScore'] - df['1H_HomeScore'],
    'Q1_Pts_For': df['Q1_AwayScore'],
    'Q1_Pts_Against': df['Q1_HomeScore'],
    'Q1_Total': df['Q1_HomeScore'] + df['Q1_AwayScore'],
    'Q1_Margin': df['Q1_AwayScore'] - df['Q1_HomeScore'],
    'Beat_OU_FT': (df['FT_HomeScore'] + df['FT_AwayScore']) - df['OU_FT_Line'],
    'Beat_OU_Q1': (df['Q1_HomeScore'] + df['Q1_AwayScore']) - df['OU_Q1_Line'],
})

team_log = pd.concat([home_log, away_log], ignore_index=True)
team_log = team_log.sort_values(by=['Team', 'Date_Clean', 'Game_ID']).reset_index(drop=True)

# Calculate Rest Days & Back-to-Back (Cap at 7 days to handle missing scraped stretches)
team_log['Prev_Game_Date'] = team_log.groupby('Team')['Date_Clean'].shift(1)
team_log['Rest_Days'] = (team_log['Date_Clean'] - team_log['Prev_Game_Date']).dt.days.clip(lower=1, upper=7).fillna(7)
team_log['Is_B2B'] = (team_log['Rest_Days'] == 1).astype(int)

# Calculate Rolling Metrics strictly WITHIN each team group using .shift(1)
# Using groupby('Team')[col].transform(...) guarantees ZERO bleed across team boundaries!
metrics_to_roll = [
    'Win', 'FT_Pts_For', 'FT_Pts_Against', 'FT_Total', 'FT_Margin',
    '1H_Pts_For', '1H_Pts_Against', '1H_Total', '1H_Margin',
    'Q1_Pts_For', 'Q1_Pts_Against', 'Q1_Total', 'Q1_Margin',
    'Beat_OU_FT', 'Beat_OU_Q1'
]

for col in metrics_to_roll:
    # Strictly group-aware rolling with shift(1)
    team_log[f'{col}_L5'] = (
        team_log.groupby('Team')[col]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    )
    team_log[f'{col}_L15'] = (
        team_log.groupby('Team')[col]
        .transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())
    )

# Extract only the pre-game rolling columns to merge back
roll_cols = ['Rest_Days', 'Is_B2B'] + [f'{c}_{w}' for c in metrics_to_roll for w in ['L5', 'L15']]

home_features = team_log[team_log['Is_Home'] == 1][['Game_ID'] + roll_cols].copy()
home_features.columns = ['Game_ID'] + [f'H_{c}' for c in roll_cols]

away_features = team_log[team_log['Is_Home'] == 0][['Game_ID'] + roll_cols].copy()
away_features.columns = ['Game_ID'] + [f'A_{c}' for c in roll_cols]

df = df.merge(home_features, on='Game_ID', how='left')
df = df.merge(away_features, on='Game_ID', how='left')

# -----------------------------------------------------------------
# 5. DIFFERENTIAL & MARKET IMPLIED PROBABILITY FEATURES
# -----------------------------------------------------------------
# Matchup Projected Totals & Margins (Combining Home Offense + Away Defense)
df['Proj_FT_Total_L5'] = (df['H_FT_Pts_For_L5'] + df['A_FT_Pts_Against_L5'] + df['A_FT_Pts_For_L5'] + df['H_FT_Pts_Against_L5']) / 2.0
df['Proj_FT_Total_L15'] = (df['H_FT_Pts_For_L15'] + df['A_FT_Pts_Against_L15'] + df['A_FT_Pts_For_L15'] + df['H_FT_Pts_Against_L15']) / 2.0

df['Proj_1H_Total_L15'] = (df['H_1H_Pts_For_L15'] + df['A_1H_Pts_Against_L15'] + df['A_1H_Pts_For_L15'] + df['H_1H_Pts_Against_L15']) / 2.0
df['Proj_Q1_Total_L15'] = (df['H_Q1_Pts_For_L15'] + df['A_Q1_Pts_Against_L15'] + df['A_Q1_Pts_For_L15'] + df['H_Q1_Pts_Against_L15']) / 2.0

# Difference vs Bookmaker Line (Positive = Our Rolling Stats Project HIGHER than Bookie Line)
df['Edge_FT_Total_vs_Line_L15'] = df['Proj_FT_Total_L15'] - df['OU_FT_Line']
df['Edge_1H_Total_vs_Line_L15'] = df['Proj_1H_Total_L15'] - df['OU_1H_Line']
df['Edge_Q1_Total_vs_Line_L15'] = df['Proj_Q1_Total_L15'] - df['OU_Q1_Line']

df['Rest_Diff'] = df['H_Rest_Days'] - df['A_Rest_Days']
df['Net_Margin_Diff_L15'] = df['H_FT_Margin_L15'] - df['A_FT_Margin_L15']
df['Q1_Margin_Diff_L15'] = df['H_Q1_Margin_L15'] - df['A_Q1_Margin_L15']

# Vig-Free Bookmaker Implied Probabilities (Moneyline)
raw_p_home = 1.0 / df['HA_FT_Home']
raw_p_away = 1.0 / df['HA_FT_Away']
df['Market_Implied_Home_Prob'] = raw_p_home / (raw_p_home + raw_p_away)
df['Elo_vs_Market_Prob_Diff'] = df['Elo_Home_Win_Prob'] - df['Market_Implied_Home_Prob']

# -----------------------------------------------------------------
# 6. TARGET VARIABLES FOR MODEL TRAINING
# -----------------------------------------------------------------
df['Target_Home_Win'] = (df['FT_HomeScore'] > df['FT_AwayScore']).astype(int)
df['Actual_FT_Total'] = df['FT_HomeScore'] + df['FT_AwayScore']
df['Actual_1H_Total'] = df['1H_HomeScore'] + df['1H_AwayScore']
df['Actual_Q1_Total'] = df['Q1_HomeScore'] + df['Q1_AwayScore']

df['Target_FT_Over'] = np.where(df['OU_FT_Line'].notnull(), (df['Actual_FT_Total'] > df['OU_FT_Line']).astype(float), np.nan)
df['Target_1H_Over'] = np.where(df['OU_1H_Line'].notnull(), (df['Actual_1H_Total'] > df['OU_1H_Line']).astype(float), np.nan)
df['Target_Q1_Over'] = np.where(df['OU_Q1_Line'].notnull(), (df['Actual_Q1_Total'] > df['OU_Q1_Line']).astype(float), np.nan)

# Save Clean Feature Dataset
OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n" + "=" * 65)
print("FEATURE ENGINEERING SUMMARY & SANITY CHECKS")
print("=" * 65)
print(f"Saved {len(df):,} rows and {len(df.columns)} columns to: {OUTPUT_CSV_PATH}")

# Check Top 5 Highest Elo Teams at the end of the dataset (2025/2026 season)
top_elo = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 Teams by Final Elo Rating (June 2026):")
for rank, (t, e) in enumerate(top_elo, 1):
    print(f"  {rank}. {t:<25} {e:.1f}")

# Check Elo standalone accuracy vs Bookie accuracy on games with valid ML odds
valid_eval = df.dropna(subset=['Market_Implied_Home_Prob', 'H_FT_Margin_L15', 'A_FT_Margin_L15']).copy()
elo_acc = ((valid_eval['Elo_Home_Win_Prob'] > 0.5) == valid_eval['Target_Home_Win']).mean() * 100
mkt_acc = ((valid_eval['Market_Implied_Home_Prob'] > 0.5) == valid_eval['Target_Home_Win']).mean() * 100
print(f"\nStandalone Pre-Game Elo Accuracy:     {elo_acc:.2f}% (across {len(valid_eval):,} games)")
print(f"Bookmaker Implied Favorite Accuracy: {mkt_acc:.2f}%")
print("=" * 65)
