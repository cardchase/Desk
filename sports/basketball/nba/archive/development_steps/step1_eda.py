import os
import sys
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# Path to the actual NBA historical dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'backend', 'outputs', 'intermediates', 'nba_historical.csv'))

print(f"Loading data from: {FILE_PATH}")
if not os.path.exists(FILE_PATH):
    print(f"ERROR: File not found at {FILE_PATH}")
    sys.exit(1)

# 1. Load dataset
try:
    df = pd.read_csv(FILE_PATH, low_memory=False)
except Exception as e:
    print(f"Error loading CSV with standard engine, falling back to python engine: {e}")
    df = pd.read_csv(FILE_PATH, sep=None, engine='python')

print("=" * 65)
print("1. DATASET OVERVIEW & HEALTH CHECK")
print("=" * 65)
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")

# Check Match_Status and Competition varieties
print("\n--- Match Status Breakdown ---")
print(df['Match_Status'].value_counts(dropna=False).to_string())

print("\n--- Competition Varieties ---")
if 'Competition' in df.columns:
    print(df['Competition'].value_counts(dropna=False).to_string())

print("\n--- Seasons Present ---")
if 'Season' in df.columns:
    print(df['Season'].value_counts().sort_index().to_string())

# 2. Clean Dates & Sort Chronologically (Oldest -> Newest)
df['Date_Clean'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
bad_dates = df['Date_Clean'].isnull().sum()
print(f"\nUnparseable Dates: {bad_dates}")

# Sort chronologically
if 'Time' in df.columns:
    df = df.sort_values(by=['Date_Clean', 'Time'], ascending=[True, True]).reset_index(drop=True)
else:
    df = df.sort_values(by=['Date_Clean'], ascending=[True]).reset_index(drop=True)

if bad_dates < len(df):
    print(f"Chronological Span: {df['Date_Clean'].min().strftime('%d %b %Y')} -> {df['Date_Clean'].max().strftime('%d %b %Y')}")

# Unique Teams check
home_teams = sorted([str(t) for t in df['HomeTeam'].dropna().unique()])
away_teams = sorted([str(t) for t in df['AwayTeam'].dropna().unique()])
all_teams = sorted(list(set(home_teams + away_teams)))
print(f"\nUnique Teams Found ({len(all_teams)}):")
for i in range(0, len(all_teams), 4):
    print("  " + ", ".join(all_teams[i:i+4]))

# 3. Missing Data & Market Coverage Audit
print("\n" + "=" * 65)
print("2. BETTING MARKET COMPLETENESS & BOOKMAKER VIG")
print("=" * 65)
markets = [
    ('Full Time Moneyline', 'HA_FT_Home', 'HA_FT_Away'),
    ('1st Half Moneyline', 'HA_1H_Home', 'HA_1H_Away'),
    ('2nd Half Moneyline', 'HA_2H_Home', 'HA_2H_Away'),
    ('Q1 Moneyline', 'HA_Q1_Home', 'HA_Q1_Away'),
    ('Q2 Moneyline', 'HA_Q2_Home', 'HA_Q2_Away'),
    ('Q3 Moneyline', 'HA_Q3_Home', 'HA_Q3_Away'),
    ('Q4 Moneyline', 'HA_Q4_Home', 'HA_Q4_Away'),
    ('Full Time Over/Under', 'OU_FT_Over', 'OU_FT_Under'),
    ('1st Half Over/Under', 'OU_1H_Over', 'OU_1H_Under'),
    ('2nd Half Over/Under', 'OU_2H_Over', 'OU_2H_Under'),
    ('Q1 Over/Under', 'OU_Q1_Over', 'OU_Q1_Under'),
    ('Q2 Over/Under', 'OU_Q2_Over', 'OU_Q2_Under'),
    ('Q3 Over/Under', 'OU_Q3_Over', 'OU_Q3_Under'),
    ('Q4 Over/Under', 'OU_Q4_Over', 'OU_Q4_Under'),
]

market_stats = []
for name, col1, col2 in markets:
    if col1 in df.columns and col2 in df.columns:
        s1 = pd.to_numeric(df[col1], errors='coerce')
        s2 = pd.to_numeric(df[col2], errors='coerce')
        valid = s1.notnull() & s2.notnull() & (s1 > 1.0) & (s2 > 1.0)
        coverage = valid.mean() * 100
        # Calculate bookmaker margin (overround / vig)
        if valid.sum() > 0:
            overround = ((1.0 / s1[valid]) + (1.0 / s2[valid]) - 1.0) * 100
            avg_vig = overround.mean()
        else:
            avg_vig = np.nan
        market_stats.append({
            'Market': name,
            'Valid Rows': int(valid.sum()),
            'Coverage (%)': round(coverage, 1),
            'Avg Bookie Vig (%)': round(avg_vig, 2)
        })

market_df = pd.DataFrame(market_stats)
print(market_df.to_string(index=False))

# 4. Score Consistencies Audit
print("\n" + "=" * 65)
print("3. SCORE CONSISTENCY & DATA INTEGRITY CHECK")
print("=" * 65)
for score_col in ['FT_HomeScore', 'FT_AwayScore', 'Q1_HomeScore', 'Q1_AwayScore', 
                  'Q2_HomeScore', 'Q2_AwayScore', 'Q3_HomeScore', 'Q3_AwayScore', 
                  'Q4_HomeScore', 'Q4_AwayScore', 'OT_HomeScore', 'OT_AwayScore']:
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')

# Check if Q1 + Q2 + Q3 + Q4 (+ OT) == FT
df_scores = df.dropna(subset=['FT_HomeScore', 'FT_AwayScore', 'Q1_HomeScore', 'Q1_AwayScore', 
                             'Q2_HomeScore', 'Q2_AwayScore', 'Q3_HomeScore', 'Q3_AwayScore', 
                             'Q4_HomeScore', 'Q4_AwayScore']).copy()
df_scores['Calc_Home_Reg'] = df_scores['Q1_HomeScore'] + df_scores['Q2_HomeScore'] + df_scores['Q3_HomeScore'] + df_scores['Q4_HomeScore']
df_scores['Calc_Away_Reg'] = df_scores['Q1_AwayScore'] + df_scores['Q2_AwayScore'] + df_scores['Q3_AwayScore'] + df_scores['Q4_AwayScore']

ot_home = df_scores['OT_HomeScore'].fillna(0)
ot_away = df_scores['OT_AwayScore'].fillna(0)
df_scores['Calc_Home_Total'] = df_scores['Calc_Home_Reg'] + ot_home
df_scores['Calc_Away_Total'] = df_scores['Calc_Away_Reg'] + ot_away

home_mismatch = (df_scores['Calc_Home_Total'] != df_scores['FT_HomeScore']).sum()
away_mismatch = (df_scores['Calc_Away_Total'] != df_scores['FT_AwayScore']).sum()
print(f"Games with complete Q1-Q4 scores: {len(df_scores):,}")
print(f"Home score mismatches (Q1..Q4 + OT != FT): {home_mismatch}")
print(f"Away score mismatches (Q1..Q4 + OT != FT): {away_mismatch}")

ot_games = (df['Went_To_OT'] == 'Yes').sum() if 'Went_To_OT' in df.columns else 0
print(f"Overtime games flag count: {ot_games:,} ({ot_games/len(df)*100:.1f}%)")

# 5. Baseline Market Benchmarks
print("\n" + "=" * 65)
print("4. BASELINE BENCHMARKS (WHAT OUR MODEL MUST BEAT)")
print("=" * 65)

for col in ['HA_FT_Home', 'HA_FT_Away', 'OU_FT_Line', 'OU_Q1_Line']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Benchmark A: Home Court Advantage & Moneyline Favorite Win Rate
valid_ml = df.dropna(subset=['FT_HomeScore', 'FT_AwayScore', 'HA_FT_Home', 'HA_FT_Away']).copy()
valid_ml['Actual_Home_Win'] = valid_ml['FT_HomeScore'] > valid_ml['FT_AwayScore']
valid_ml['Bookie_Fav_Is_Home'] = valid_ml['HA_FT_Home'] < valid_ml['HA_FT_Away']
valid_ml['Bookie_Correct'] = (
    ((valid_ml['HA_FT_Home'] < valid_ml['HA_FT_Away']) & valid_ml['Actual_Home_Win']) |
    ((valid_ml['HA_FT_Away'] < valid_ml['HA_FT_Home']) & (~valid_ml['Actual_Home_Win']))
)
fav_win_rate = valid_ml['Bookie_Correct'].mean() * 100
home_win_rate = valid_ml['Actual_Home_Win'].mean() * 100

print(f"Overall Home Team Win Rate: {home_win_rate:.2f}%")
print(f"Bookie Moneyline Favorite Win Rate: {fav_win_rate:.2f}% (across {len(valid_ml):,} games)")

# Benchmark B: Full-Time Over/Under Baseline
valid_ou = df.dropna(subset=['FT_HomeScore', 'FT_AwayScore', 'OU_FT_Line']).copy()
valid_ou['Actual_Total'] = valid_ou['FT_HomeScore'] + valid_ou['FT_AwayScore']
valid_ou['Over_Hit'] = valid_ou['Actual_Total'] > valid_ou['OU_FT_Line']
valid_ou['Under_Hit'] = valid_ou['Actual_Total'] < valid_ou['OU_FT_Line']
valid_ou['Push'] = valid_ou['Actual_Total'] == valid_ou['OU_FT_Line']
mae_ft = (valid_ou['Actual_Total'] - valid_ou['OU_FT_Line']).abs().mean()
print(f"\nFull-Time Total Points ({len(valid_ou):,} games with OU_FT_Line):")
print(f"  Over Hits: {valid_ou['Over_Hit'].mean()*100:.2f}% | Under Hits: {valid_ou['Under_Hit'].mean()*100:.2f}% | Pushes: {valid_ou['Push'].mean()*100:.2f}%")
print(f"  Bookie FT Line Mean Absolute Error (MAE): {mae_ft:.2f} points")

# Benchmark C: Q1 Over/Under Baseline
valid_q1 = df.dropna(subset=['Q1_HomeScore', 'Q1_AwayScore', 'OU_Q1_Line']).copy()
valid_q1['Actual_Q1_Total'] = valid_q1['Q1_HomeScore'] + valid_q1['Q1_AwayScore']
valid_q1['Q1_Over_Hit'] = valid_q1['Actual_Q1_Total'] > valid_q1['OU_Q1_Line']
valid_q1['Q1_Under_Hit'] = valid_q1['Actual_Q1_Total'] < valid_q1['OU_Q1_Line']
valid_q1['Q1_Push'] = valid_q1['Actual_Q1_Total'] == valid_q1['OU_Q1_Line']
mae_q1 = (valid_q1['Actual_Q1_Total'] - valid_q1['OU_Q1_Line']).abs().mean()
print(f"\nQ1 Total Points ({len(valid_q1):,} games with OU_Q1_Line):")
print(f"  Q1 Over Hits: {valid_q1['Q1_Over_Hit'].mean()*100:.2f}% | Under Hits: {valid_q1['Q1_Under_Hit'].mean()*100:.2f}% | Pushes: {valid_q1['Q1_Push'].mean()*100:.2f}%")
print(f"  Bookie Q1 Line Mean Absolute Error (MAE): {mae_q1:.2f} points")
print("=" * 65)
