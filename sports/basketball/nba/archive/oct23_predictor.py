import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
from math import erf, sqrt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def norm_cdf(x):
    """Standard normal cumulative distribution function (no scipy required)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HISTORICAL_CSV   = r"d:\Project\LoomFlow - Sports\backend\outputs\intermediates\nba_historical.csv"
LIVE_OCT23_CSV   = r"d:\Project\LoomFlow - Sports\test_NBA\oct23_live_odds.csv"
OUTPUT_OCT23_CSV = r"d:\Project\LoomFlow - Sports\test_NBA\oct23_predictions.csv"

print("=" * 95)
print("NBA PREDICTION ENGINE (v3): 23RD OCTOBER 2026 SLATE")
print("UPGRADES: Tie Filtering | Dual Pure+Residual Regressors | Play-To Guardrails | Steam Warning")
print("=" * 95)

# 1. Load Data
raw_df = pd.read_csv(HISTORICAL_CSV, low_memory=False)
raw_df['Date_Clean'] = pd.to_datetime(raw_df['Date'], format='mixed', dayfirst=True, errors='coerce')

non_standard_teams = [
    'East', 'West', 'Team Chuck', 'Team Durant', 'Team Giannis',
    'Team LeBron', 'Team Shaq', 'Team Stars', 'Team Stripes',
    'Team USA', 'Team World'
]
raw_df = raw_df[~(raw_df['HomeTeam'].isin(non_standard_teams) | raw_df['AwayTeam'].isin(non_standard_teams))].copy()

score_cols = [
    'FT_HomeScore', 'FT_AwayScore',
    'Q1_HomeScore', 'Q1_AwayScore', 'Q2_HomeScore', 'Q2_AwayScore',
    'Q3_HomeScore', 'Q3_AwayScore', 'Q4_HomeScore', 'Q4_AwayScore'
]
for col in score_cols:
    if col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

hist_df = raw_df.dropna(subset=score_cols).sort_values(by=['Date_Clean', 'Time', 'HomeTeam']).reset_index(drop=True)
hist_df['Game_ID'] = np.arange(len(hist_df))
last_hist_season = hist_df['Season'].iloc[-1]

upcoming_df = pd.read_csv(LIVE_OCT23_CSV, low_memory=False)
upcoming_df['Date_Clean'] = pd.to_datetime(upcoming_df['Date'], format='mixed', dayfirst=True, errors='coerce')

print(f"Historical Training Base: {len(hist_df):,} matches (through {hist_df['Date_Clean'].max().strftime('%d %b %Y')})")
print(f"Matches to Analyze for 23 Oct: {len(upcoming_df)} matches\n")

# 2. Features on History
hist_df['1H_HomeScore'] = hist_df['Q1_HomeScore'] + hist_df['Q2_HomeScore']
hist_df['1H_AwayScore'] = hist_df['Q1_AwayScore'] + hist_df['Q2_AwayScore']

odds_cols = [
    'HA_FT_Home', 'HA_FT_Away', 'HA_1H_Home', 'HA_1H_Away', 'HA_Q1_Home', 'HA_Q1_Away',
    'OU_FT_Line', 'OU_FT_Over', 'OU_FT_Under',
    'OU_1H_Line', 'OU_1H_Over', 'OU_1H_Under',
    'OU_Q1_Line', 'OU_Q1_Over', 'OU_Q1_Under'
]
for col in odds_cols:
    if col in hist_df.columns:
        hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
    if col in upcoming_df.columns:
        upcoming_df[col] = pd.to_numeric(upcoming_df[col], errors='coerce')

# Elo with 25% inter-season regression
elo_dict = {team: 1500.0 for team in hist_df['HomeTeam'].unique()}
K_FACTOR, HOME_ADV = 20.0, 70.0
h_elo_list, a_elo_list, elo_prob_list = [], [], []
curr_season = None

for _, row in hist_df.iterrows():
    if curr_season is not None and row['Season'] != curr_season:
        for t in elo_dict:
            elo_dict[t] = 0.75 * elo_dict[t] + 0.25 * 1500.0
    curr_season = row['Season']

    h_t, a_t = row['HomeTeam'], row['AwayTeam']
    h_e, a_e = elo_dict.get(h_t, 1500.0), elo_dict.get(a_t, 1500.0)
    h_elo_list.append(h_e)
    a_elo_list.append(a_e)
    exp_h = 1.0 / (1.0 + 10.0 ** ((a_e - (h_e + HOME_ADV)) / 400.0))
    elo_prob_list.append(exp_h)

    margin = row['FT_HomeScore'] - row['FT_AwayScore']
    act_h = 1.0 if margin > 0 else 0.0
    elo_diff_w = (h_e + HOME_ADV - a_e) if margin > 0 else (a_e - (h_e + HOME_ADV))
    mov_mult = ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_w)
    shift = K_FACTOR * mov_mult * (act_h - exp_h)
    elo_dict[h_t] = h_e + shift
    elo_dict[a_t] = a_e - shift

hist_df['Home_Elo_Pre'] = h_elo_list
hist_df['Away_Elo_Pre'] = a_elo_list
hist_df['Elo_Diff'] = hist_df['Home_Elo_Pre'] - hist_df['Away_Elo_Pre']
hist_df['Elo_Home_Win_Prob'] = elo_prob_list

# Team Log
def make_team_log(df_in):
    h_log = pd.DataFrame({
        'Game_ID': df_in['Game_ID'], 'Date_Clean': df_in['Date_Clean'], 'Team': df_in['HomeTeam'], 'Is_Home': 1,
        'Win': (df_in['FT_HomeScore'] > df_in['FT_AwayScore']).astype(float),
        'FT_Pts_For': df_in['FT_HomeScore'], 'FT_Pts_Against': df_in['FT_AwayScore'],
        'FT_Total': df_in['FT_HomeScore'] + df_in['FT_AwayScore'], 'FT_Margin': df_in['FT_HomeScore'] - df_in['FT_AwayScore'],
        '1H_Pts_For': df_in['1H_HomeScore'], '1H_Pts_Against': df_in['1H_AwayScore'],
        '1H_Total': df_in['1H_HomeScore'] + df_in['1H_AwayScore'], '1H_Margin': df_in['1H_HomeScore'] - df_in['1H_AwayScore'],
        'Q1_Pts_For': df_in['Q1_HomeScore'], 'Q1_Pts_Against': df_in['Q1_AwayScore'],
        'Q1_Total': df_in['Q1_HomeScore'] + df_in['Q1_AwayScore'], 'Q1_Margin': df_in['Q1_HomeScore'] - df_in['Q1_AwayScore'],
        'Beat_OU_FT': (df_in['FT_HomeScore'] + df_in['FT_AwayScore']) - df_in['OU_FT_Line'],
        'Beat_OU_Q1': (df_in['Q1_HomeScore'] + df_in['Q1_AwayScore']) - df_in['OU_Q1_Line'],
    })
    a_log = pd.DataFrame({
        'Game_ID': df_in['Game_ID'], 'Date_Clean': df_in['Date_Clean'], 'Team': df_in['AwayTeam'], 'Is_Home': 0,
        'Win': (df_in['FT_AwayScore'] > df_in['FT_HomeScore']).astype(float),
        'FT_Pts_For': df_in['FT_AwayScore'], 'FT_Pts_Against': df_in['FT_HomeScore'],
        'FT_Total': df_in['FT_HomeScore'] + df_in['FT_AwayScore'], 'FT_Margin': df_in['FT_AwayScore'] - df_in['FT_HomeScore'],
        '1H_Pts_For': df_in['1H_AwayScore'], '1H_Pts_Against': df_in['1H_HomeScore'],
        '1H_Total': df_in['1H_HomeScore'] + df_in['1H_AwayScore'], '1H_Margin': df_in['1H_AwayScore'] - df_in['1H_HomeScore'],
        'Q1_Pts_For': df_in['Q1_AwayScore'], 'Q1_Pts_Against': df_in['Q1_HomeScore'],
        'Q1_Total': df_in['Q1_HomeScore'] + df_in['Q1_AwayScore'], 'Q1_Margin': df_in['Q1_AwayScore'] - df_in['Q1_HomeScore'],
        'Beat_OU_FT': (df_in['FT_HomeScore'] + df_in['FT_AwayScore']) - df_in['OU_FT_Line'],
        'Beat_OU_Q1': (df_in['Q1_HomeScore'] + df_in['Q1_AwayScore']) - df_in['OU_Q1_Line'],
    })
    return pd.concat([h_log, a_log], ignore_index=True).sort_values(by=['Team', 'Date_Clean', 'Game_ID']).reset_index(drop=True)

team_log = make_team_log(hist_df)
team_log['Prev_Game_Date'] = team_log.groupby('Team')['Date_Clean'].shift(1)
team_log['Rest_Days'] = (team_log['Date_Clean'] - team_log['Prev_Game_Date']).dt.days.clip(lower=1, upper=7).fillna(7)
team_log['Is_B2B'] = (team_log['Rest_Days'] == 1).astype(int)

metrics_to_roll = [
    'Win', 'FT_Pts_For', 'FT_Pts_Against', 'FT_Total', 'FT_Margin',
    '1H_Pts_For', '1H_Pts_Against', '1H_Total', '1H_Margin',
    'Q1_Pts_For', 'Q1_Pts_Against', 'Q1_Total', 'Q1_Margin',
    'Beat_OU_FT', 'Beat_OU_Q1'
]

for col in metrics_to_roll:
    team_log[f'{col}_L5'] = team_log.groupby('Team')[col].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    team_log[f'{col}_L15'] = team_log.groupby('Team')[col].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())

roll_cols = ['Rest_Days', 'Is_B2B'] + [f'{c}_{w}' for c in metrics_to_roll for w in ['L5', 'L15']]
h_feats = team_log[team_log['Is_Home'] == 1][['Game_ID'] + roll_cols].rename(columns={c: f'H_{c}' for c in roll_cols})
a_feats = team_log[team_log['Is_Home'] == 0][['Game_ID'] + roll_cols].rename(columns={c: f'A_{c}' for c in roll_cols})
hist_df = hist_df.merge(h_feats, on='Game_ID', how='left').merge(a_feats, on='Game_ID', how='left')

# Latest state
latest_team_state = {}
for team, grp in team_log.groupby('Team'):
    last_date = grp['Date_Clean'].max()
    state = {'Last_Game_Date': last_date, 'Elo': elo_dict.get(team, 1500.0)}
    for col in metrics_to_roll:
        state[f'{col}_L5'] = grp[col].dropna().tail(5).mean()
        state[f'{col}_L15'] = grp[col].dropna().tail(15).mean()
    latest_team_state[team] = state

# Build features for Oct 23 upcoming games
up_rows = []
for idx, r in upcoming_df.iterrows():
    h_t, a_t = r['HomeTeam'], r['AwayTeam']
    h_st = latest_team_state.get(h_t, {})
    a_st = latest_team_state.get(a_t, {})
    g_date = r['Date_Clean']
    up_season = str(r.get('Season', last_hist_season))

    row_dict = r.to_dict()
    h_e, a_e = h_st.get('Elo', 1500.0), a_st.get('Elo', 1500.0)
    
    # 25% Elo Regression for the new season
    days_since_hist = (g_date - hist_df['Date_Clean'].max()).days if pd.notnull(g_date) else 0
    if (up_season != str(last_hist_season)) or (days_since_hist > 60):
        h_e = 0.75 * h_e + 0.25 * 1500.0
        a_e = 0.75 * a_e + 0.25 * 1500.0

    row_dict['Home_Elo_Pre'] = h_e
    row_dict['Away_Elo_Pre'] = a_e
    row_dict['Elo_Diff'] = h_e - a_e
    row_dict['Elo_Home_Win_Prob'] = 1.0 / (1.0 + 10.0 ** ((a_e - (h_e + HOME_ADV)) / 400.0))

    h_last = h_st.get('Last_Game_Date', g_date)
    a_last = a_st.get('Last_Game_Date', g_date)
    h_rest = np.clip((g_date - h_last).days, 1, 7) if pd.notnull(g_date) and pd.notnull(h_last) else 3
    a_rest = np.clip((g_date - a_last).days, 1, 7) if pd.notnull(g_date) and pd.notnull(a_last) else 3
    row_dict['H_Rest_Days'], row_dict['A_Rest_Days'] = h_rest, a_rest
    row_dict['H_Is_B2B'], row_dict['A_Is_B2B'] = int(h_rest == 1), int(a_rest == 1)

    for col in metrics_to_roll:
        for w in ['L5', 'L15']:
            row_dict[f'H_{col}_{w}'] = h_st.get(f'{col}_{w}', np.nan)
            row_dict[f'A_{col}_{w}'] = a_st.get(f'{col}_{w}', np.nan)
    up_rows.append(row_dict)

upcoming_feat_df = pd.DataFrame(up_rows)

def add_derived_features(d):
    d['Proj_FT_Total_L5'] = (d['H_FT_Pts_For_L5'] + d['A_FT_Pts_Against_L5'] + d['A_FT_Pts_For_L5'] + d['H_FT_Pts_Against_L5']) / 2.0
    d['Proj_FT_Total_L15'] = (d['H_FT_Pts_For_L15'] + d['A_FT_Pts_Against_L15'] + d['A_FT_Pts_For_L15'] + d['H_FT_Pts_Against_L15']) / 2.0
    d['Proj_1H_Total_L5'] = (d['H_1H_Pts_For_L5'] + d['A_1H_Pts_Against_L5'] + d['A_1H_Pts_For_L5'] + d['H_1H_Pts_Against_L5']) / 2.0
    d['Proj_1H_Total_L15'] = (d['H_1H_Pts_For_L15'] + d['A_1H_Pts_Against_L15'] + d['A_1H_Pts_For_L15'] + d['H_1H_Pts_Against_L15']) / 2.0
    d['Proj_Q1_Total_L5'] = (d['H_Q1_Pts_For_L5'] + d['A_Q1_Pts_Against_L5'] + d['A_Q1_Pts_For_L5'] + d['H_Q1_Pts_Against_L5']) / 2.0
    d['Proj_Q1_Total_L15'] = (d['H_Q1_Pts_For_L15'] + d['A_Q1_Pts_Against_L15'] + d['A_Q1_Pts_For_L15'] + d['H_Q1_Pts_Against_L15']) / 2.0

    d['Edge_FT_Total_vs_Line_L5'] = d['Proj_FT_Total_L5'] - d['OU_FT_Line']
    d['Edge_FT_Total_vs_Line_L15'] = d['Proj_FT_Total_L15'] - d['OU_FT_Line']
    d['Edge_1H_Total_vs_Line_L5'] = d['Proj_1H_Total_L5'] - d['OU_1H_Line']
    d['Edge_1H_Total_vs_Line_L15'] = d['Proj_1H_Total_L15'] - d['OU_1H_Line']
    d['Edge_Q1_Total_vs_Line_L5'] = d['Proj_Q1_Total_L5'] - d['OU_Q1_Line']
    d['Edge_Q1_Total_vs_Line_L15'] = d['Proj_Q1_Total_L15'] - d['OU_Q1_Line']

    d['Rest_Diff'] = d['H_Rest_Days'] - d['A_Rest_Days']
    d['Net_Margin_Diff_L15'] = d['H_FT_Margin_L15'] - d['A_FT_Margin_L15']
    d['Q1_Margin_Diff_L15'] = d['H_Q1_Margin_L15'] - d['A_Q1_Margin_L15']
    d['H_Momentum_Margin'] = d['H_FT_Margin_L5'] - d['H_FT_Margin_L15']
    d['A_Momentum_Margin'] = d['A_FT_Margin_L5'] - d['A_FT_Margin_L15']
    d['H_Pace_Shift'] = d['H_FT_Total_L5'] - d['H_FT_Total_L15']
    d['A_Pace_Shift'] = d['A_FT_Total_L5'] - d['A_FT_Total_L15']
    d['Combined_Beat_OU_FT_L15'] = d['H_Beat_OU_FT_L15'] + d['A_Beat_OU_FT_L15']
    d['Combined_Beat_OU_Q1_L15'] = d['H_Beat_OU_Q1_L15'] + d['A_Beat_OU_Q1_L15']

    for pref, c1, c2 in [('FT_ML', 'HA_FT_Home', 'HA_FT_Away'), ('1H_ML', 'HA_1H_Home', 'HA_1H_Away'), ('Q1_ML', 'HA_Q1_Home', 'HA_Q1_Away'),
                         ('FT_OU', 'OU_FT_Over', 'OU_FT_Under'), ('1H_OU', 'OU_1H_Over', 'OU_1H_Under'), ('Q1_OU', 'OU_Q1_Over', 'OU_Q1_Under')]:
        if c1 in d.columns and c2 in d.columns:
            p1 = np.where(d[c1] > 1.0, 1.0 / d[c1], np.nan)
            p2 = np.where(d[c2] > 1.0, 1.0 / d[c2], np.nan)
            denom = p1 + p2
            d[f'Mkt_Prob_{pref}_1'] = np.where(denom > 0, p1 / denom, np.nan)
        else:
            d[f'Mkt_Prob_{pref}_1'] = np.nan

    d['Market_Implied_Home_Prob'] = d['Mkt_Prob_FT_ML_1']
    d['Elo_vs_Market_Prob_Diff'] = d['Elo_Home_Win_Prob'] - d['Market_Implied_Home_Prob']
    return d

hist_df = add_derived_features(hist_df)
upcoming_feat_df = add_derived_features(upcoming_feat_df)

hist_df['Target_Home_Win'] = (hist_df['FT_HomeScore'] > hist_df['FT_AwayScore']).astype(int)
hist_df['Target_1H_Home_Win'] = (hist_df['1H_HomeScore'] > hist_df['1H_AwayScore']).astype(int)
hist_df['Target_Q1_Home_Win'] = (hist_df['Q1_HomeScore'] > hist_df['Q1_AwayScore']).astype(int)

hist_df['Actual_FT_Total'] = hist_df['FT_HomeScore'] + hist_df['FT_AwayScore']
hist_df['Actual_1H_Total'] = hist_df['1H_HomeScore'] + hist_df['1H_AwayScore']
hist_df['Actual_Q1_Total'] = hist_df['Q1_HomeScore'] + hist_df['Q1_AwayScore']

hist_df['Target_FT_Over'] = (hist_df['Actual_FT_Total'] > hist_df['OU_FT_Line']).astype(int)
hist_df['Target_1H_Over'] = (hist_df['Actual_1H_Total'] > hist_df['OU_1H_Line']).astype(int)
hist_df['Target_Q1_Over'] = (hist_df['Actual_Q1_Total'] > hist_df['OU_Q1_Line']).astype(int)

# 3. Models Setup
base_team_features = [
    'Home_Elo_Pre', 'Away_Elo_Pre', 'Elo_Diff', 'Elo_Home_Win_Prob',
    'H_Rest_Days', 'A_Rest_Days', 'Rest_Diff', 'H_Is_B2B', 'A_Is_B2B',
    'H_Win_L5', 'H_Win_L15', 'A_Win_L5', 'A_Win_L15',
    'H_FT_Margin_L5', 'H_FT_Margin_L15', 'A_FT_Margin_L5', 'A_FT_Margin_L15',
    'Net_Margin_Diff_L15', 'H_Momentum_Margin', 'A_Momentum_Margin',
    'H_1H_Margin_L5', 'H_1H_Margin_L15', 'A_1H_Margin_L5', 'A_1H_Margin_L15',
    'H_Q1_Margin_L5', 'H_Q1_Margin_L15', 'A_Q1_Margin_L5', 'A_Q1_Margin_L15',
    'Q1_Margin_Diff_L15'
]

# Pure Pace Features (STRICTLY EXCLUDING ANY BOOKMAKER LINE, EDGE, OR BEAT_OU FEATURE)
pure_pace_features = [
    'H_FT_Pts_For_L5', 'H_FT_Pts_Against_L5', 'H_FT_Pts_For_L15', 'H_FT_Pts_Against_L15',
    'A_FT_Pts_For_L5', 'A_FT_Pts_Against_L5', 'A_FT_Pts_For_L15', 'A_FT_Pts_Against_L15',
    'H_1H_Pts_For_L5', 'H_1H_Pts_Against_L5', 'H_1H_Pts_For_L15', 'H_1H_Pts_Against_L15',
    'A_1H_Pts_For_L5', 'A_1H_Pts_Against_L5', 'A_1H_Pts_For_L15', 'A_1H_Pts_Against_L15',
    'H_Q1_Pts_For_L5', 'H_Q1_Pts_Against_L5', 'H_Q1_Pts_For_L15', 'H_Q1_Pts_Against_L15',
    'A_Q1_Pts_For_L5', 'A_Q1_Pts_Against_L5', 'A_Q1_Pts_For_L15', 'A_Q1_Pts_Against_L15',
    'Proj_FT_Total_L5', 'Proj_FT_Total_L15', 'Proj_1H_Total_L5', 'Proj_1H_Total_L15',
    'Proj_Q1_Total_L5', 'Proj_Q1_Total_L15', 'H_Pace_Shift', 'A_Pace_Shift'
]
pure_total_features = base_team_features + pure_pace_features

# Market Augmented Features
all_pace_total_features = pure_pace_features + [
    'H_Beat_OU_FT_L5', 'H_Beat_OU_FT_L15', 'A_Beat_OU_FT_L5', 'A_Beat_OU_FT_L15',
    'H_Beat_OU_Q1_L5', 'H_Beat_OU_Q1_L15', 'A_Beat_OU_Q1_L5', 'A_Beat_OU_Q1_L15',
    'Combined_Beat_OU_FT_L15', 'Combined_Beat_OU_Q1_L15'
]

market_configs = {
    'FT_ML': {
        'target': 'Target_Home_Win', 'reg_target': None, 'line_col': None, 'std_dev': None,
        'odds_1': 'HA_FT_Home', 'odds_2': 'HA_FT_Away',
        'side_1': 'Home ML', 'side_2': 'Away ML', 'blend': 0.65,
        'features': base_team_features + ['HA_FT_Home', 'HA_FT_Away', 'Market_Implied_Home_Prob', 'Elo_vs_Market_Prob_Diff', 'OU_FT_Line'],
        'tie_check': None
    },
    '1H_ML': {
        'target': 'Target_1H_Home_Win', 'reg_target': None, 'line_col': None, 'std_dev': None,
        'odds_1': 'HA_1H_Home', 'odds_2': 'HA_1H_Away',
        'side_1': '1H Home ML', 'side_2': '1H Away ML', 'blend': 0.65,
        'features': base_team_features + ['HA_1H_Home', 'HA_1H_Away', 'OU_1H_Line'],
        'tie_check': ('1H_HomeScore', '1H_AwayScore')  # UPGRADE 1: Strictly filter ties!
    },
    'Q1_ML': {
        'target': 'Target_Q1_Home_Win', 'reg_target': None, 'line_col': None, 'std_dev': None,
        'odds_1': 'HA_Q1_Home', 'odds_2': 'HA_Q1_Away',
        'side_1': 'Q1 Home ML', 'side_2': 'Q1 Away ML', 'blend': 0.65,
        'features': base_team_features + ['HA_Q1_Home', 'HA_Q1_Away', 'OU_Q1_Line'],
        'tie_check': ('Q1_HomeScore', 'Q1_AwayScore')  # UPGRADE 1: Strictly filter ties!
    },
    'FT_OU': {
        'target': 'Target_FT_Over', 'reg_target': 'Actual_FT_Total', 'line_col': 'OU_FT_Line', 'std_dev': 17.5,
        'odds_1': 'OU_FT_Over', 'odds_2': 'OU_FT_Under',
        'side_1': 'FT Over', 'side_2': 'FT Under', 'blend': 0.70,
        'features': base_team_features + all_pace_total_features + ['OU_FT_Line', 'Mkt_Prob_FT_OU_1', 'Edge_FT_Total_vs_Line_L5', 'Edge_FT_Total_vs_Line_L15', 'Market_Implied_Home_Prob'],
        'tie_check': None
    },
    '1H_OU': {
        'target': 'Target_1H_Over', 'reg_target': 'Actual_1H_Total', 'line_col': 'OU_1H_Line', 'std_dev': 11.2,
        'odds_1': 'OU_1H_Over', 'odds_2': 'OU_1H_Under',
        'side_1': '1H Over', 'side_2': '1H Under', 'blend': 0.70,
        'features': base_team_features + all_pace_total_features + ['OU_1H_Line', 'OU_FT_Line', 'Mkt_Prob_1H_OU_1', 'Edge_1H_Total_vs_Line_L5', 'Edge_1H_Total_vs_Line_L15', 'Market_Implied_Home_Prob'],
        'tie_check': None
    },
    'Q1_OU': {
        'target': 'Target_Q1_Over', 'reg_target': 'Actual_Q1_Total', 'line_col': 'OU_Q1_Line', 'std_dev': 7.8,
        'odds_1': 'OU_Q1_Over', 'odds_2': 'OU_Q1_Under',
        'side_1': 'Q1 Over', 'side_2': 'Q1 Under', 'blend': 0.70,
        'features': base_team_features + all_pace_total_features + ['OU_Q1_Line', 'OU_FT_Line', 'Mkt_Prob_Q1_OU_1', 'Edge_Q1_Total_vs_Line_L5', 'Edge_Q1_Total_vs_Line_L15', 'Market_Implied_Home_Prob'],
        'tie_check': None
    }
}

def make_clf():
    if HAS_LGBM:
        base = LGBMClassifier(n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=7,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                              reg_alpha=2.0, reg_lambda=5.0, random_state=42, verbose=-1)
    else:
        base = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.03, max_depth=3, max_leaf_nodes=7,
                                              min_samples_leaf=50, l2_regularization=5.0, random_state=42)
    return CalibratedClassifierCV(base, method='sigmoid', cv=3)

def make_reg():
    if HAS_LGBM:
        return LGBMRegressor(n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=7,
                             subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                             reg_alpha=2.0, reg_lambda=5.0, random_state=42, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=140, learning_rate=0.03, max_depth=3, max_leaf_nodes=7,
                                         min_samples_leaf=50, l2_regularization=5.0, random_state=42)

all_predictions = []
market_models = {}

for m_key, cfg in market_configs.items():
    feats = [f for f in cfg['features'] if f in hist_df.columns]
    valid_train = hist_df[
        hist_df[cfg['target']].notnull() &
        hist_df[cfg['odds_1']].notnull() & (hist_df[cfg['odds_1']] > 1.01) &
        hist_df[cfg['odds_2']].notnull() & (hist_df[cfg['odds_2']] > 1.01) &
        hist_df['H_FT_Margin_L15'].notnull() & hist_df['A_FT_Margin_L15'].notnull()
    ].copy()

    # UPGRADE 1: Strictly filter out TIES so road underdogs are not artificially inflated!
    if cfg.get('tie_check'):
        c_h, c_a = cfg['tie_check']
        valid_train = valid_train[valid_train[c_h] != valid_train[c_a]]

    if cfg['line_col']:
        valid_train = valid_train[
            valid_train[cfg['line_col']].notnull() &
            (valid_train[cfg['reg_target']] != valid_train[cfg['line_col']])
        ]

    medians = valid_train[feats].median()
    X_tr = valid_train[feats].fillna(medians)
    y_tr = valid_train[cfg['target']].astype(int)

    clf = make_clf()
    clf.fit(X_tr, y_tr)

    # UPGRADE 2: Train TWO Regressors for Totals (Pure Pace + Market Residual)
    reg_pure = None
    reg_resid = None
    pure_f = [f for f in pure_total_features if f in hist_df.columns]
    pure_meds = valid_train[pure_f].median()

    if cfg['reg_target']:
        # 1. Pure Pace Regressor (100% line-agnostic)
        reg_pure = make_reg()
        X_tr_pure = valid_train[pure_f].fillna(pure_meds)
        reg_pure.fit(X_tr_pure, valid_train[cfg['reg_target']].astype(float))

        # 2. Market Residual Regressor
        reg_resid = make_reg()
        y_residual = (valid_train[cfg['reg_target']] - valid_train[cfg['line_col']]).astype(float)
        reg_resid.fit(X_tr, y_residual)

    market_models[m_key] = {
        'clf': clf, 'reg_pure': reg_pure, 'reg_resid': reg_resid,
        'medians': medians, 'feats': feats,
        'pure_meds': pure_meds, 'pure_f': pure_f,
        'cfg': cfg
    }

    up_valid = upcoming_feat_df[
        upcoming_feat_df[cfg['odds_1']].notnull() & (upcoming_feat_df[cfg['odds_1']] > 1.01) &
        upcoming_feat_df[cfg['odds_2']].notnull() & (upcoming_feat_df[cfg['odds_2']] > 1.01)
    ].copy().reset_index(drop=True)

    if len(up_valid) == 0:
        continue

    X_up = up_valid[feats].fillna(medians)
    clf_p1 = clf.predict_proba(X_up)[:, 1]

    p1_imp = 1.0 / up_valid[cfg['odds_1']]
    p2_imp = 1.0 / up_valid[cfg['odds_2']]
    mkt_p1 = (p1_imp / (p1_imp + p2_imp)).values

    if cfg['reg_target']:
        # UPGRADE 2: Combine Pure Pace and Market Residual 50/50
        X_up_pure = up_valid[pure_f].fillna(pure_meds)
        pure_pts = reg_pure.predict(X_up_pure)
        resid_pts = up_valid[cfg['line_col']].values + reg_resid.predict(X_up)
        proj_pts = 0.50 * pure_pts + 0.50 * resid_pts

        reg_p1 = np.array([norm_cdf((p - line_v) / cfg['std_dev']) for p, line_v in zip(proj_pts, up_valid[cfg['line_col']].values)])
        prob_1 = 0.55 * reg_p1 + 0.25 * clf_p1 + 0.20 * mkt_p1
    else:
        proj_pts = np.full(len(up_valid), np.nan)
        pure_pts = np.full(len(up_valid), np.nan)
        prob_1 = cfg['blend'] * clf_p1 + (1.0 - cfg['blend']) * mkt_p1

    prob_2 = 1.0 - prob_1

    for idx_i, (_, r) in enumerate(up_valid.iterrows()):
        p1, p2 = prob_1[idx_i], prob_2[idx_i]
        m1, m2 = mkt_p1[idx_i], 1.0 - mkt_p1[idx_i]
        o1, o2 = r[cfg['odds_1']], r[cfg['odds_2']]
        line_val = r[cfg['line_col']] if cfg['line_col'] else np.nan
        p_pts = proj_pts[idx_i]

        for side_lbl, p_mod, p_mkt, odds_val, is_side1 in [
            (f"{r['HomeTeam']} ML" if 'ML' in m_key else f"{cfg['side_1']} {line_val}", p1, m1, o1, True),
            (f"{r['AwayTeam']} ML" if 'ML' in m_key else f"{cfg['side_2']} {line_val}", p2, m2, o2, False)
        ]:
            ev = (p_mod * odds_val) - 1.0
            kelly_full = max(0.0, (p_mod * odds_val - 1.0) / max(0.01, odds_val - 1.0))
            units = round(min(5.0, kelly_full * 0.25 * 100.0), 2)
            pts_diff = (p_pts - line_val) if (pd.notnull(p_pts) and is_side1) else ((line_val - p_pts) if pd.notnull(p_pts) else np.nan)

            # UPGRADE 3: Add "Play-To" Guardrail Columns
            # Min_Playable_Odds maintains at least +2.0% EV: p_mod * odds - 1 >= 0.02 -> odds >= 1.02 / p_mod
            min_odds = round(1.02 / max(0.01, p_mod), 2)

            # Play_To_Line: Cushion threshold to maintain >= 53.0% win prob (0.075 * std_dev)
            if pd.notnull(line_val):
                cushion_buffer = 0.075 * cfg['std_dev']
                if is_side1:  # Over pick
                    play_to_line = round(p_pts - cushion_buffer, 1)
                else:          # Under pick
                    play_to_line = round(p_pts + cushion_buffer, 1)
            else:
                play_to_line = np.nan

            all_predictions.append({
                'Date': r['Date_Clean'].strftime('%Y-%m-%d') if pd.notnull(r['Date_Clean']) else str(r.get('Date', '')),
                'Matchup': f"{r['AwayTeam']} @ {r['HomeTeam']}",
                'Market': m_key,
                'Pick': side_lbl,
                'Odds': round(odds_val, 2),
                'Model_Prob_%': round(p_mod * 100, 1),
                'Bookie_Prob_%': round(p_mkt * 100, 1),
                'Edge_%': round((p_mod - p_mkt) * 100, 2),
                'EV_%': round(ev * 100, 2),
                'Min_Playable_Odds': min_odds,
                'Model_Proj_Pts': round(p_pts, 1) if pd.notnull(p_pts) else np.nan,
                'Line_Cushion_Pts': round(pts_diff, 1) if pd.notnull(pts_diff) else np.nan,
                'Play_To_Line': play_to_line,
                'Rec_Units': units
            })

pred_df = pd.DataFrame(all_predictions)
pred_df.to_csv(OUTPUT_OCT23_CSV, index=False)

# ==============================================================================
# 4. PRINT PRESENTATION OUTPUTS
# ==============================================================================

print("\n" + "=" * 95)
print("MATCH 1: CLEVELAND CAVALIERS @ PHILADELPHIA 76ERS (01:00 AM)")
print("=" * 95)
m1_df = pred_df[pred_df['Matchup'].str.contains('Philadelphia 76ers')]
cols_show = ['Market', 'Pick', 'Odds', 'Model_Prob_%', 'Edge_%', 'EV_%', 'Min_Playable_Odds', 'Model_Proj_Pts', 'Line_Cushion_Pts', 'Play_To_Line', 'Rec_Units']
print(m1_df[cols_show].to_string(index=False))

print("\n" + "=" * 95)
print("MATCH 2: DENVER NUGGETS @ OKLAHOMA CITY THUNDER (03:30 AM)")
print("=" * 95)
m2_df = pred_df[pred_df['Matchup'].str.contains('Oklahoma City Thunder')]
print(m2_df[cols_show].to_string(index=False))

# ==============================================================================
# 5. LINE-DRIFT & SPREAD SENSITIVITY STRESS TEST WITH BLENDED REGRESSOR
# ==============================================================================
print("\n" + "=" * 95)
print("[STRESS TEST] SPREAD & TOTAL LINE SENSITIVITY (50/50 BLENDED PURE+RESIDUAL REGRESSOR)")
print("              Anticipating line movements so you know exact safe entry thresholds!")
print("=" * 95)

for idx, r in upcoming_feat_df.iterrows():
    match_str = f"{r['AwayTeam']} @ {r['HomeTeam']}"
    ou_cfg = market_configs['FT_OU']
    reg_pure = market_models['FT_OU']['reg_pure']
    reg_resid = market_models['FT_OU']['reg_resid']
    feats = market_models['FT_OU']['feats']
    meds = market_models['FT_OU']['medians']
    pure_f = market_models['FT_OU']['pure_f']
    pure_meds = market_models['FT_OU']['pure_meds']

    base_line = r['OU_FT_Line']
    base_over_odds = r['OU_FT_Over']
    base_under_odds = r['OU_FT_Under']

    x_row = pd.DataFrame([r[feats]]).fillna(meds)
    x_row_pure = pd.DataFrame([r[pure_f]]).fillna(pure_meds)

    p_pure = reg_pure.predict(x_row_pure)[0]
    p_resid = base_line + reg_resid.predict(x_row)[0]
    proj_total = 0.50 * p_pure + 0.50 * p_resid

    print(f"\n[{match_str}]")
    print(f"  * Pure Pace Score: {p_pure:.1f} pts | Market Residual Score: {p_resid:.1f} pts")
    print(f"  * Blended 50/50 Projected Total: {proj_total:.1f} pts (Current Market Line: {base_line})")

    # Simulate line shifts: -2.0, -1.0, 0, +1.0, +2.0
    print("  * LINE SHIFT SIMULATION (FT Over/Under):")
    print("    Shift | Line  | Pick       | Model Prob | Odds  | EV%    | Cushion | Safe Action")
    print("    ----------------------------------------------------------------------------------------")
    for shift in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        test_line = base_line + shift
        u_cushion = test_line - proj_total
        o_cushion = proj_total - test_line

        if u_cushion >= o_cushion:
            pick_side = f"Under {test_line:.1f}"
            cush = u_cushion
            p_gauss = norm_cdf(cush / ou_cfg['std_dev'])
            odds_used = base_under_odds
        else:
            pick_side = f"Over {test_line:.1f}"
            cush = o_cushion
            p_gauss = norm_cdf(cush / ou_cfg['std_dev'])
            odds_used = base_over_odds

        p_final = 0.60 * p_gauss + 0.40 * 0.50
        ev = (p_final * odds_used) - 1.0

        status = "[STRONG BUY]" if ev >= 0.04 else ("[VALUE PLAY]" if ev > 0.0 else "[PASS / NO EDGE]")
        print(f"    {shift:+4.1f} | {test_line:5.1f} | {pick_side:10s} | {p_final*100:9.1f}% | {odds_used:5.2f} | {ev*100:+5.2f}% | {cush:+6.1f}  | {status}")

# UPGRADE 4: Reverse Line Movement / Steam Warning Rule
print("\n" + "=" * 95)
print("REVERSE LINE MOVEMENT (STEAM) & INJURY PROTOCOL")
print("=" * 95)
print("RULE: If a Moneyline odds on your betting app has drifted by MORE than +0.15 higher")
print("      than the scraped OddsPortal odds (e.g. Utah 1.80 scraped -> 1.98 on your app),")
print("      TREAT AS STEAM / INJURY ALERT: DO NOT BET until starting lineups are confirmed!")
print("=" * 95)
print(f"Predictions saved to: {OUTPUT_OCT23_CSV}")
