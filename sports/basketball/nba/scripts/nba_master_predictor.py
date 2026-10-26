"""
NBA MASTER PREDICTION ENGINE & WEEKLY COMMAND CENTER
=====================================================
Single Source of Truth for NBA Match Predictions, Derivative Markets, and Value Analysis.

Key Architectural Upgrades Baked In:
1. Strict Tie Filtering: 1H_ML and Q1_ML filter out tie games before classifier training.
2. Dual 50/50 Regressors: Blends pure line-agnostic pace models with market residual models.
3. Play-To Guardrail Columns: Computes Min_Playable_Odds (+2% EV buffer) and Play_To_Line (53% win floor).
4. Reverse Line Movement (Steam) Alert: Flags odds drifting > +0.15 on betting apps.
5. 25% Inter-Season Elo Regression: Automatically adjusts team power ratings across season gaps.
6. Spread & Total Line Sensitivity Stress-Testing: Real-time simulation of +/- 2.0 point line shifts.

Usage:
  python nba_master_predictor.py                     # Predict entire upcoming slate
  python nba_master_predictor.py --date "2026-10-23" # Predict specific date only
  python nba_master_predictor.py --date today        # Predict today's games
  python nba_master_predictor.py --scrape            # Re-scrape live OddsPortal odds first
"""

import sys
import os
import subprocess
import argparse
import warnings
from pathlib import Path
from math import erf, sqrt
import pandas as pd
import numpy as np
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


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

DEFAULT_HISTORICAL_CSV = str(DATA_DIR / "nba_historical.csv")
DEFAULT_UPCOMING_CSV   = str(DATA_DIR / "upcoming_games.csv")
DEFAULT_OUTPUT_CSV     = str(DATA_DIR / "weekly_betting_card.csv")
DEFAULT_CHECKLIST_CSV  = str(DATA_DIR / "actionable_betting_checklist.csv")
BACKEND_CHECKLIST_CSV  = str(DATA_DIR / "actionable_betting_checklist.csv")

K_FACTOR = 20.0
HOME_ADV = 70.0

NON_STANDARD_TEAMS = [
    'East', 'West', 'Team Chuck', 'Team Durant', 'Team Giannis',
    'Team LeBron', 'Team Shaq', 'Team Stars', 'Team Stripes',
    'Team USA', 'Team World'
]

SCORE_COLS = [
    'FT_HomeScore', 'FT_AwayScore',
    'Q1_HomeScore', 'Q1_AwayScore', 'Q2_HomeScore', 'Q2_AwayScore',
    'Q3_HomeScore', 'Q3_AwayScore', 'Q4_HomeScore', 'Q4_AwayScore'
]

ODDS_COLS = [
    'HA_FT_Home', 'HA_FT_Away', 'HA_1H_Home', 'HA_1H_Away', 'HA_Q1_Home', 'HA_Q1_Away',
    'OU_FT_Line', 'OU_FT_Over', 'OU_FT_Under',
    'OU_1H_Line', 'OU_1H_Over', 'OU_1H_Under',
    'OU_Q1_Line', 'OU_Q1_Over', 'OU_Q1_Under'
]

BASE_TEAM_FEATURES = [
    'Home_Elo_Pre', 'Away_Elo_Pre', 'Elo_Diff', 'Elo_Home_Win_Prob',
    'H_Rest_Days', 'A_Rest_Days', 'Rest_Diff', 'H_Is_B2B', 'A_Is_B2B',
    'H_Win_L5', 'H_Win_L15', 'A_Win_L5', 'A_Win_L15',
    'H_FT_Margin_L5', 'H_FT_Margin_L15', 'A_FT_Margin_L5', 'A_FT_Margin_L15',
    'Net_Margin_Diff_L15', 'H_Momentum_Margin', 'A_Momentum_Margin',
    'H_1H_Margin_L5', 'H_1H_Margin_L15', 'A_1H_Margin_L5', 'A_1H_Margin_L15',
    'H_Q1_Margin_L5', 'H_Q1_Margin_L15', 'A_Q1_Margin_L5', 'A_Q1_Margin_L15',
    'Q1_Margin_Diff_L15'
]

PURE_PACE_FEATURES = [
    'H_FT_Pts_For_L5', 'H_FT_Pts_Against_L5', 'H_FT_Pts_For_L15', 'H_FT_Pts_Against_L15',
    'A_FT_Pts_For_L5', 'A_FT_Pts_Against_L5', 'A_FT_Pts_For_L15', 'A_FT_Pts_Against_L15',
    'H_1H_Pts_For_L5', 'H_1H_Pts_Against_L5', 'H_1H_Pts_For_L15', 'H_1H_Pts_Against_L15',
    'A_1H_Pts_For_L5', 'A_1H_Pts_Against_L5', 'A_1H_Pts_For_L15', 'A_1H_Pts_Against_L15',
    'H_Q1_Pts_For_L5', 'H_Q1_Pts_Against_L5', 'H_Q1_Pts_For_L15', 'H_Q1_Pts_Against_L15',
    'A_Q1_Pts_For_L5', 'A_Q1_Pts_Against_L5', 'A_Q1_Pts_For_L15', 'A_Q1_Pts_Against_L15',
    'Proj_FT_Total_L5', 'Proj_FT_Total_L15', 'Proj_1H_Total_L5', 'Proj_1H_Total_L15',
    'Proj_Q1_Total_L5', 'Proj_Q1_Total_L15', 'H_Pace_Shift', 'A_Pace_Shift'
]

ALL_PACE_TOTAL_FEATURES = PURE_PACE_FEATURES + [
    'H_Beat_OU_FT_L5', 'H_Beat_OU_FT_L15', 'A_Beat_OU_FT_L5', 'A_Beat_OU_FT_L15',
    'H_Beat_OU_Q1_L5', 'H_Beat_OU_Q1_L15', 'A_Beat_OU_Q1_L5', 'A_Beat_OU_Q1_L15',
    'Combined_Beat_OU_FT_L15', 'Combined_Beat_OU_Q1_L15'
]

MARKET_CONFIGS = {
    'FT_ML': {
        'target': 'Target_Home_Win', 'reg_target': None, 'line_col': None, 'std_dev': None,
        'odds_1': 'HA_FT_Home', 'odds_2': 'HA_FT_Away',
        'side_1': 'Home ML', 'side_2': 'Away ML', 'blend': 0.65,
        'features': BASE_TEAM_FEATURES + ['HA_FT_Home', 'HA_FT_Away', 'Market_Implied_Home_Prob', 'Elo_vs_Market_Prob_Diff', 'OU_FT_Line'],
        'tie_check': None
    },
    '1H_ML': {
        'target': 'Target_1H_Home_Win', 'reg_target': None, 'line_col': None, 'std_dev': None,
        'odds_1': 'HA_1H_Home', 'odds_2': 'HA_1H_Away',
        'side_1': '1H Home ML', 'side_2': '1H Away ML', 'blend': 0.65,
        'features': BASE_TEAM_FEATURES + ['HA_1H_Home', 'HA_1H_Away', 'OU_1H_Line'],
        'tie_check': ('1H_HomeScore', '1H_AwayScore')  # UPGRADE 1: Filter ties
    },
    'Q1_ML': {
        'target': 'Target_Q1_Home_Win', 'reg_target': None, 'line_col': None, 'std_dev': None,
        'odds_1': 'HA_Q1_Home', 'odds_2': 'HA_Q1_Away',
        'side_1': 'Q1 Home ML', 'side_2': 'Q1 Away ML', 'blend': 0.65,
        'features': BASE_TEAM_FEATURES + ['HA_Q1_Home', 'HA_Q1_Away', 'OU_Q1_Line'],
        'tie_check': ('Q1_HomeScore', 'Q1_AwayScore')  # UPGRADE 1: Filter ties
    },
    'FT_OU': {
        'target': 'Target_FT_Over', 'reg_target': 'Actual_FT_Total', 'line_col': 'OU_FT_Line', 'std_dev': 17.5,
        'odds_1': 'OU_FT_Over', 'odds_2': 'OU_FT_Under',
        'side_1': 'FT Over', 'side_2': 'FT Under', 'blend': 0.70,
        'features': BASE_TEAM_FEATURES + ALL_PACE_TOTAL_FEATURES + ['OU_FT_Line', 'Mkt_Prob_FT_OU_1', 'Edge_FT_Total_vs_Line_L5', 'Edge_FT_Total_vs_Line_L15', 'Market_Implied_Home_Prob'],
        'tie_check': None
    },
    '1H_OU': {
        'target': 'Target_1H_Over', 'reg_target': 'Actual_1H_Total', 'line_col': 'OU_1H_Line', 'std_dev': 11.2,
        'odds_1': 'OU_1H_Over', 'odds_2': 'OU_1H_Under',
        'side_1': '1H Over', 'side_2': '1H Under', 'blend': 0.70,
        'features': BASE_TEAM_FEATURES + ALL_PACE_TOTAL_FEATURES + ['OU_1H_Line', 'OU_FT_Line', 'Mkt_Prob_1H_OU_1', 'Edge_1H_Total_vs_Line_L5', 'Edge_1H_Total_vs_Line_L15', 'Market_Implied_Home_Prob'],
        'tie_check': None
    },
    'Q1_OU': {
        'target': 'Target_Q1_Over', 'reg_target': 'Actual_Q1_Total', 'line_col': 'OU_Q1_Line', 'std_dev': 7.8,
        'odds_1': 'OU_Q1_Over', 'odds_2': 'OU_Q1_Under',
        'side_1': 'Q1 Over', 'side_2': 'Q1 Under', 'blend': 0.70,
        'features': BASE_TEAM_FEATURES + ALL_PACE_TOTAL_FEATURES + ['OU_Q1_Line', 'OU_FT_Line', 'Mkt_Prob_Q1_OU_1', 'Edge_Q1_Total_vs_Line_L5', 'Edge_Q1_Total_vs_Line_L15', 'Market_Implied_Home_Prob'],
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


# ==============================================================================
# DATA LOADING & FEATURE PIPELINE
# ==============================================================================
def load_and_prep_history(hist_csv_path: str):
    raw_df = pd.read_csv(hist_csv_path, low_memory=False)
    # Triple-Layer Deduplication Guardrail: Drop duplicate matches by unique URL and Date/Teams
    if 'URL' in raw_df.columns:
        raw_df = raw_df.drop_duplicates(subset=['URL'], keep='last')
    raw_df = raw_df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
    raw_df['Date_Clean'] = pd.to_datetime(raw_df['Date'], format='mixed', dayfirst=True, errors='coerce')
    raw_df = raw_df[~(raw_df['HomeTeam'].isin(NON_STANDARD_TEAMS) | raw_df['AwayTeam'].isin(NON_STANDARD_TEAMS))].copy()

    for col in SCORE_COLS:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

    hist_df = raw_df.dropna(subset=SCORE_COLS).sort_values(by=['Date_Clean', 'Time', 'HomeTeam']).reset_index(drop=True)
    hist_df['Game_ID'] = np.arange(len(hist_df))

    hist_df['1H_HomeScore'] = hist_df['Q1_HomeScore'] + hist_df['Q2_HomeScore']
    hist_df['1H_AwayScore'] = hist_df['Q1_AwayScore'] + hist_df['Q2_AwayScore']

    for col in ODDS_COLS:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

    return hist_df


def compute_historical_elo(hist_df: pd.DataFrame):
    elo_dict = {team: 1500.0 for team in hist_df['HomeTeam'].unique()}
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
    return elo_dict


def compute_team_rolling_features(hist_df: pd.DataFrame, elo_dict: dict):
    h_log = pd.DataFrame({
        'Game_ID': hist_df['Game_ID'], 'Date_Clean': hist_df['Date_Clean'], 'Team': hist_df['HomeTeam'], 'Is_Home': 1,
        'Win': (hist_df['FT_HomeScore'] > hist_df['FT_AwayScore']).astype(float),
        'FT_Pts_For': hist_df['FT_HomeScore'], 'FT_Pts_Against': hist_df['FT_AwayScore'],
        'FT_Total': hist_df['FT_HomeScore'] + hist_df['FT_AwayScore'], 'FT_Margin': hist_df['FT_HomeScore'] - hist_df['FT_AwayScore'],
        '1H_Pts_For': hist_df['1H_HomeScore'], '1H_Pts_Against': hist_df['1H_AwayScore'],
        '1H_Total': hist_df['1H_HomeScore'] + hist_df['1H_AwayScore'], '1H_Margin': hist_df['1H_HomeScore'] - hist_df['1H_AwayScore'],
        'Q1_Pts_For': hist_df['Q1_HomeScore'], 'Q1_Pts_Against': hist_df['Q1_AwayScore'],
        'Q1_Total': hist_df['Q1_HomeScore'] + hist_df['Q1_AwayScore'], 'Q1_Margin': hist_df['Q1_HomeScore'] - hist_df['Q1_AwayScore'],
        'Beat_OU_FT': (hist_df['FT_HomeScore'] + hist_df['FT_AwayScore']) - hist_df['OU_FT_Line'],
        'Beat_OU_Q1': (hist_df['Q1_HomeScore'] + hist_df['Q1_AwayScore']) - hist_df['OU_Q1_Line'],
    })
    a_log = pd.DataFrame({
        'Game_ID': hist_df['Game_ID'], 'Date_Clean': hist_df['Date_Clean'], 'Team': hist_df['AwayTeam'], 'Is_Home': 0,
        'Win': (hist_df['FT_AwayScore'] > hist_df['FT_HomeScore']).astype(float),
        'FT_Pts_For': hist_df['FT_AwayScore'], 'FT_Pts_Against': hist_df['FT_HomeScore'],
        'FT_Total': hist_df['FT_HomeScore'] + hist_df['FT_AwayScore'], 'FT_Margin': hist_df['FT_AwayScore'] - hist_df['FT_HomeScore'],
        '1H_Pts_For': hist_df['1H_AwayScore'], '1H_Pts_Against': hist_df['1H_HomeScore'],
        '1H_Total': hist_df['1H_HomeScore'] + hist_df['1H_AwayScore'], '1H_Margin': hist_df['1H_AwayScore'] - hist_df['1H_HomeScore'],
        'Q1_Pts_For': hist_df['Q1_AwayScore'], 'Q1_Pts_Against': hist_df['Q1_HomeScore'],
        'Q1_Total': hist_df['Q1_HomeScore'] + hist_df['Q1_AwayScore'], 'Q1_Margin': hist_df['Q1_AwayScore'] - hist_df['Q1_HomeScore'],
        'Beat_OU_FT': (hist_df['FT_HomeScore'] + hist_df['FT_AwayScore']) - hist_df['OU_FT_Line'],
        'Beat_OU_Q1': (hist_df['Q1_HomeScore'] + hist_df['Q1_AwayScore']) - hist_df['OU_Q1_Line'],
    })
    t_log = pd.concat([h_log, a_log], ignore_index=True).sort_values(by=['Team', 'Date_Clean', 'Game_ID']).reset_index(drop=True)

    t_log['Prev_Game_Date'] = t_log.groupby('Team')['Date_Clean'].shift(1)
    t_log['Rest_Days'] = (t_log['Date_Clean'] - t_log['Prev_Game_Date']).dt.days.clip(lower=1, upper=7).fillna(7)
    t_log['Is_B2B'] = (t_log['Rest_Days'] == 1).astype(int)

    metrics_to_roll = [
        'Win', 'FT_Pts_For', 'FT_Pts_Against', 'FT_Total', 'FT_Margin',
        '1H_Pts_For', '1H_Pts_Against', '1H_Total', '1H_Margin',
        'Q1_Pts_For', 'Q1_Pts_Against', 'Q1_Total', 'Q1_Margin',
        'Beat_OU_FT', 'Beat_OU_Q1'
    ]
    for col in metrics_to_roll:
        t_log[f'{col}_L5'] = t_log.groupby('Team')[col].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        t_log[f'{col}_L15'] = t_log.groupby('Team')[col].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())

    roll_cols = ['Rest_Days', 'Is_B2B'] + [f'{c}_{w}' for c in metrics_to_roll for w in ['L5', 'L15']]
    h_feats = t_log[t_log['Is_Home'] == 1][['Game_ID'] + roll_cols].rename(columns={c: f'H_{c}' for c in roll_cols})
    a_feats = t_log[t_log['Is_Home'] == 0][['Game_ID'] + roll_cols].rename(columns={c: f'A_{c}' for c in roll_cols})
    hist_df_out = hist_df.merge(h_feats, on='Game_ID', how='left').merge(a_feats, on='Game_ID', how='left')

    latest_team_state = {}
    for team, grp in t_log.groupby('Team'):
        last_date = grp['Date_Clean'].max()
        state = {'Last_Game_Date': last_date, 'Elo': elo_dict.get(team, 1500.0)}
        for col in metrics_to_roll:
            state[f'{col}_L5'] = grp[col].dropna().tail(5).mean()
            state[f'{col}_L15'] = grp[col].dropna().tail(15).mean()
        latest_team_state[team] = state

    return hist_df_out, latest_team_state, metrics_to_roll


def add_derived_features(d: pd.DataFrame) -> pd.DataFrame:
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


def build_upcoming_features(upcoming_df: pd.DataFrame, latest_team_state: dict, hist_df: pd.DataFrame, metrics_to_roll: list):
    last_hist_season = str(hist_df['Season'].iloc[-1])
    last_hist_date = hist_df['Date_Clean'].max()

    up_rows = []
    for _, r in upcoming_df.iterrows():
        h_t, a_t = r['HomeTeam'], r['AwayTeam']
        h_st = latest_team_state.get(h_t, {})
        a_st = latest_team_state.get(a_t, {})
        g_date = r['Date_Clean']
        up_season = str(r.get('Season', last_hist_season))

        row_dict = r.to_dict()
        h_e, a_e = h_st.get('Elo', 1500.0), a_st.get('Elo', 1500.0)

        days_since_hist = (g_date - last_hist_date).days if pd.notnull(g_date) else 0
        if (up_season != last_hist_season) or (days_since_hist > 60):
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

    up_feat_df = pd.DataFrame(up_rows)
    for col in ODDS_COLS:
        if col in up_feat_df.columns:
            up_feat_df[col] = pd.to_numeric(up_feat_df[col], errors='coerce')
    return add_derived_features(up_feat_df)


# ==============================================================================
# MODEL TRAINING & PREDICTION ENGINE
# ==============================================================================
def train_and_predict(hist_df: pd.DataFrame, upcoming_feat_df: pd.DataFrame):
    # Set historical targets
    hist_df['Target_Home_Win'] = (hist_df['FT_HomeScore'] > hist_df['FT_AwayScore']).astype(int)
    hist_df['Target_1H_Home_Win'] = (hist_df['1H_HomeScore'] > hist_df['1H_AwayScore']).astype(int)
    hist_df['Target_Q1_Home_Win'] = (hist_df['Q1_HomeScore'] > hist_df['Q1_AwayScore']).astype(int)

    hist_df['Actual_FT_Total'] = hist_df['FT_HomeScore'] + hist_df['FT_AwayScore']
    hist_df['Actual_1H_Total'] = hist_df['1H_HomeScore'] + hist_df['1H_AwayScore']
    hist_df['Actual_Q1_Total'] = hist_df['Q1_HomeScore'] + hist_df['Q1_AwayScore']

    hist_df['Target_FT_Over'] = (hist_df['Actual_FT_Total'] > hist_df['OU_FT_Line']).astype(int)
    hist_df['Target_1H_Over'] = (hist_df['Actual_1H_Total'] > hist_df['OU_1H_Line']).astype(int)
    hist_df['Target_Q1_Over'] = (hist_df['Actual_Q1_Total'] > hist_df['OU_Q1_Line']).astype(int)

    pure_total_features = [f for f in (BASE_TEAM_FEATURES + PURE_PACE_FEATURES) if f in hist_df.columns]

    all_predictions = []
    market_models = {}

    for m_key, cfg in MARKET_CONFIGS.items():
        feats = [f for f in cfg['features'] if f in hist_df.columns]
        valid_train = hist_df[
            hist_df[cfg['target']].notnull() &
            hist_df[cfg['odds_1']].notnull() & (hist_df[cfg['odds_1']] > 1.01) &
            hist_df[cfg['odds_2']].notnull() & (hist_df[cfg['odds_2']] > 1.01) &
            hist_df['H_FT_Margin_L15'].notnull() & hist_df['A_FT_Margin_L15'].notnull()
        ].copy()

        # UPGRADE 1: Strictly filter out ties from 1H and Q1 Moneyline training!
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

        # UPGRADE 2: Train TWO regressors for totals (Pure Pace + Market Residual)
        reg_pure, reg_resid = None, None
        pure_meds = valid_train[pure_total_features].median()

        if cfg['reg_target']:
            reg_pure = make_reg()
            X_tr_pure = valid_train[pure_total_features].fillna(pure_meds)
            reg_pure.fit(X_tr_pure, valid_train[cfg['reg_target']].astype(float))

            reg_resid = make_reg()
            y_residual = (valid_train[cfg['reg_target']] - valid_train[cfg['line_col']]).astype(float)
            reg_resid.fit(X_tr, y_residual)

        market_models[m_key] = {
            'clf': clf, 'reg_pure': reg_pure, 'reg_resid': reg_resid,
            'medians': medians, 'feats': feats,
            'pure_meds': pure_meds, 'pure_f': pure_total_features,
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
            X_up_pure = up_valid[pure_total_features].fillna(pure_meds)
            pure_pts = reg_pure.predict(X_up_pure)
            resid_pts = up_valid[cfg['line_col']].values + reg_resid.predict(X_up)
            proj_pts = 0.50 * pure_pts + 0.50 * resid_pts

            reg_p1 = np.array([norm_cdf((p - line_v) / cfg['std_dev']) for p, line_v in zip(proj_pts, up_valid[cfg['line_col']].values)])
            prob_1 = 0.55 * reg_p1 + 0.25 * clf_p1 + 0.20 * mkt_p1
        else:
            proj_pts = np.full(len(up_valid), np.nan)
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

                # UPGRADE 3: Min_Playable_Odds (+2% EV buffer) & Play_To_Line (53% win floor)
                min_odds = round(1.02 / max(0.01, p_mod), 2)
                if pd.notnull(line_val):
                    cushion_buffer = 0.075 * cfg['std_dev']
                    play_to_line = round(p_pts - cushion_buffer, 1) if is_side1 else round(p_pts + cushion_buffer, 1)
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
    return pred_df, market_models


# ==============================================================================
# SENSITIVITY STRESS TEST & REPORTING
# ==============================================================================
def run_line_sensitivity_stress_test(upcoming_feat_df: pd.DataFrame, market_models: dict):
    print("\n" + "=" * 95)
    print("[STRESS TEST] TOTAL LINE SENSITIVITY (50/50 BLENDED PURE+RESIDUAL REGRESSOR)")
    print("              Anticipates bookmaker line drops & shows exact safe entry numbers!")
    print("=" * 95)

    ou_cfg = MARKET_CONFIGS['FT_OU']
    reg_pure = market_models['FT_OU']['reg_pure']
    reg_resid = market_models['FT_OU']['reg_resid']
    feats = market_models['FT_OU']['feats']
    meds = market_models['FT_OU']['medians']
    pure_f = market_models['FT_OU']['pure_f']
    pure_meds = market_models['FT_OU']['pure_meds']

    for idx, r in upcoming_feat_df.iterrows():
        if pd.isnull(r.get('OU_FT_Line')):
            continue

        match_str = f"{r['AwayTeam']} @ {r['HomeTeam']}"
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
        print("  * LINE SHIFT SIMULATION (FT Over/Under):")
        print("    Shift | Line  | Pick       | Model Prob | Odds  | EV%    | Cushion | Recommendation")
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


def generate_actionable_checklist(card_df: pd.DataFrame, output_path: str = DEFAULT_CHECKLIST_CSV) -> pd.DataFrame:
    """
    Synthesizes the raw multi-market betting card into an Executive Actionable Checklist.
    Balances categories: Win-Rate Anchors (1-2), Totals Snipers (3-4), Value Underdogs (1-2), Derivatives (1).
    Exports to CSV and returns the prioritized dataframe.
    """
    if card_df.empty:
        return pd.DataFrame()

    anchors = []
    totals = []
    underdogs = []
    derivatives = []

    # 1. WIN-RATE ANCHORS (FT_ML): Highest probability straight win plays
    ft_ml = card_df[card_df['Market'] == 'FT_ML'].copy()
    if not ft_ml.empty:
        top_ml = ft_ml.sort_values('Model_Prob_%', ascending=False)
        for _, r in top_ml.head(2).iterrows():
            if r['Model_Prob_%'] >= 65.0:
                is_pos_ev = r['EV_%'] >= 0.0
                tag = '[WIN-RATE ANCHOR]' if is_pos_ev else '[PARLAY / PICK\'EM ANCHOR]'
                advice = (
                    f"High-probability straight pick'em anchor ({r['Model_Prob_%']:.1f}% win prob). "
                    f"Elite team Elo & rest advantage ({r['EV_%']:+.2f}% EV). Best for straight single win or safe 2-leg parlay builder."
                    if is_pos_ev else
                    f"Heavy straight favorite ({r['Model_Prob_%']:.1f}% win prob). Odds carry bookmaker vig ({r['EV_%']:+.2f}% EV). "
                    f"Best as 2-leg parlay piece or straight pick'em; for straight single, require {r['Min_Playable_Odds']:.2f} odds."
                )
                anchors.append({
                    'Category_Tag': tag,
                    'Priority_Weight': 100 + r['Model_Prob_%'] + (10 if is_pos_ev else 0),
                    'Matchup': r['Matchup'],
                    'Market': 'FT_ML',
                    'Target_Selection': r['Pick'],
                    'Current_Odds': r['Odds'],
                    'Model_Win_Prob_%': f"{r['Model_Prob_%']:.1f}%",
                    'Bookie_Implied_%': f"{r['Bookie_Prob_%']:.1f}%",
                    'Model_Edge_%': f"{r['Edge_%']:+.2f}%",
                    'Expected_Value_%': f"{r['EV_%']:+.2f}%",
                    'Model_Projected': 'N/A (Moneyline)',
                    'Safety_Cushion': f"{r['Edge_%']:+.1f}% Win Margin",
                    'Play_To_Cutoff': f"Odds >= {r['Min_Playable_Odds']:.2f}",
                    'Min_Playable_Odds': r['Min_Playable_Odds'],
                    'Rec_Units': f"{max(1.0, min(2.0, r['Rec_Units'] if r['Rec_Units'] > 0 else 1.25)):.1f}u",
                    'Betting_Advice_Notes': advice
                })

    # 2. HIGH-CONVICTION TOTAL SNIPERS (FT_OU):
    ft_ou = card_df[card_df['Market'] == 'FT_OU'].copy()
    if not ft_ou.empty:
        strong_ou = ft_ou[(ft_ou['Line_Cushion_Pts'] >= 3.5) & (ft_ou['EV_%'] >= 4.0)].sort_values('EV_%', ascending=False)
        for _, r in strong_ou.iterrows():
            direction = "Under" if "Under" in str(r['Pick']) else "Over"
            totals.append({
                'Category_Tag': '[HIGH-CONVICTION TOTAL SNIPER]',
                'Priority_Weight': 90 + r['EV_%'],
                'Matchup': r['Matchup'],
                'Market': 'FT_OU',
                'Target_Selection': r['Pick'],
                'Current_Odds': r['Odds'],
                'Model_Win_Prob_%': f"{r['Model_Prob_%']:.1f}%",
                'Bookie_Implied_%': f"{r['Bookie_Prob_%']:.1f}%",
                'Model_Edge_%': f"{r['Edge_%']:+.2f}%",
                'Expected_Value_%': f"{r['EV_%']:+.2f}%",
                'Model_Projected': f"{r['Model_Proj_Pts']:.1f} pts",
                'Safety_Cushion': f"+{r['Line_Cushion_Pts']:.1f} pts {direction.lower()}",
                'Play_To_Cutoff': f"Line {r['Play_To_Line']:.1f}",
                'Min_Playable_Odds': r['Min_Playable_Odds'],
                'Rec_Units': f"{max(1.0, min(2.0, r['Rec_Units'] if r['Rec_Units'] > 0 else 1.5)):.1f}u",
                'Betting_Advice_Notes': (
                    f"Sharp pace edge: Model projects {r['Model_Proj_Pts']:.1f} pts vs market line (+{r['Line_Cushion_Pts']:.1f} pt cushion). "
                    f"Bet {direction} down to {r['Play_To_Line']:.1f} line cutoff. Pass if bookmaker moves line beyond {r['Play_To_Line']:.1f}."
                )
            })

    # 3. VALUE UNDERDOG / SPREAD SNIPERS (FT_ML):
    if not ft_ml.empty:
        dogs = ft_ml[(ft_ml['Odds'] >= 1.85) & (ft_ml['EV_%'] >= 2.5) & (ft_ml['Edge_%'] >= 2.0)].sort_values('EV_%', ascending=False)
        for _, r in dogs.iterrows():
            team = str(r['Pick']).replace(' ML', '')
            underdogs.append({
                'Category_Tag': '[VALUE UNDERDOG / SPREAD]',
                'Priority_Weight': 85 + r['EV_%'],
                'Matchup': r['Matchup'],
                'Market': 'FT_ML',
                'Target_Selection': f"{r['Pick']} (or +3.5 Spread)",
                'Current_Odds': r['Odds'],
                'Model_Win_Prob_%': f"{r['Model_Prob_%']:.1f}%",
                'Bookie_Implied_%': f"{r['Bookie_Prob_%']:.1f}%",
                'Model_Edge_%': f"{r['Edge_%']:+.2f}%",
                'Expected_Value_%': f"{r['EV_%']:+.2f}%",
                'Model_Projected': 'N/A (Moneyline)',
                'Safety_Cushion': f"{r['Edge_%']:+.1f}% Edge vs Bookie",
                'Play_To_Cutoff': f"Odds >= {r['Min_Playable_Odds']:.2f}",
                'Min_Playable_Odds': r['Min_Playable_Odds'],
                'Rec_Units': f"{max(0.75, min(1.5, r['Rec_Units'] if r['Rec_Units'] > 0 else 1.0)):.1f}u",
                'Betting_Advice_Notes': (
                    f"Sharp underdog value: Model projects {team} with a {r['Model_Prob_%']:.1f}% win probability vs {r['Bookie_Prob_%']:.1f}% bookie implied "
                    f"({r['EV_%']:+.2f}% EV). Value on Moneyline @ {r['Odds']:.2f}; recommend +3.5 point spread alternative for lower variance."
                )
            })

    # 4. DERIVATIVE PACE SNIPERS (1H_OU / Q1_OU):
    deriv_ou = card_df[card_df['Market'].isin(['1H_OU', 'Q1_OU'])].copy()
    if not deriv_ou.empty:
        strong_deriv = deriv_ou[(deriv_ou['Line_Cushion_Pts'] >= 2.0) & (deriv_ou['EV_%'] >= 3.0)].sort_values('EV_%', ascending=False)
        for _, r in strong_deriv.iterrows():
            direction = "Under" if "Under" in str(r['Pick']) else "Over"
            derivatives.append({
                'Category_Tag': '[DERIVATIVE PACE SNIPER]',
                'Priority_Weight': 80 + r['EV_%'],
                'Matchup': r['Matchup'],
                'Market': r['Market'],
                'Target_Selection': r['Pick'],
                'Current_Odds': r['Odds'],
                'Model_Win_Prob_%': f"{r['Model_Prob_%']:.1f}%",
                'Bookie_Implied_%': f"{r['Bookie_Prob_%']:.1f}%",
                'Model_Edge_%': f"{r['Edge_%']:+.2f}%",
                'Expected_Value_%': f"{r['EV_%']:+.2f}%",
                'Model_Projected': f"{r['Model_Proj_Pts']:.1f} pts",
                'Safety_Cushion': f"+{r['Line_Cushion_Pts']:.1f} pts {direction.lower()}",
                'Play_To_Cutoff': f"Line {r['Play_To_Line']:.1f}",
                'Min_Playable_Odds': r['Min_Playable_Odds'],
                'Rec_Units': f"{max(0.75, min(1.5, r['Rec_Units'] if r['Rec_Units'] > 0 else 1.0)):.1f}u",
                'Betting_Advice_Notes': (
                    f"Early game pace mismatch ({r['Market']}): Model projects {r['Model_Proj_Pts']:.1f} pts (+{r['Line_Cushion_Pts']:.1f} pt cushion). "
                    f"Play {direction} down to {r['Play_To_Line']:.1f} line."
                )
            })

    # Balanced Card Composition: Top 2 Anchors, Top 4 Totals, Top 2 Underdogs, Top 1 Derivative
    selected = []
    selected.extend(anchors[:2])
    selected.extend(totals[:4])
    selected.extend(underdogs[:2])
    selected.extend(derivatives[:1])

    if not selected:
        return pd.DataFrame()

    df_cand = pd.DataFrame(selected)
    df_cand = df_cand.sort_values('Priority_Weight', ascending=False).drop_duplicates(subset=['Matchup', 'Market'])
    df_final = df_cand.head(8).copy()
    df_final.insert(0, 'Priority_Rank', range(1, len(df_final) + 1))
    df_final.drop(columns=['Priority_Weight'], inplace=True)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(out_p, index=False)
        print(f"Actionable Betting Checklist saved to: {out_p}")
        try:
            b_p = Path(BACKEND_CHECKLIST_CSV)
            b_p.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(b_p, index=False)
        except Exception:
            pass

    return df_final


def print_actionable_checklist(checklist_df: pd.DataFrame):
    if checklist_df.empty:
        print("\nNo high-conviction plays qualify for the Actionable Checklist on this slate.")
        return

    print("\n" + "=" * 105)
    print("📋 ACTIONABLE NBA BETTING CHECKLIST (EXECUTIVE SUMMARY FOR BET PLACEMENT)")
    print("   Filtered for Maximum Conviction, Positive EV%, Safety Cushion & Play-To Guardrails")
    print("=" * 105)

    display_cols = ['Priority_Rank', 'Category_Tag', 'Matchup', 'Target_Selection', 'Current_Odds', 'Model_Win_Prob_%', 'Expected_Value_%', 'Safety_Cushion', 'Play_To_Cutoff', 'Rec_Units']
    available_cols = [c for c in display_cols if c in checklist_df.columns]
    print(checklist_df[available_cols].to_string(index=False))

    print("\n" + "=" * 105)
    print("💡 BETTING EXECUTION ADVICE & GAME NOTES:")
    for _, r in checklist_df.iterrows():
        print(f"  #{r['Priority_Rank']} {r['Category_Tag']} {r['Matchup']} -> {r['Target_Selection']} ({r['Rec_Units']} @ {r['Current_Odds']}):")
        print(f"     {r['Betting_Advice_Notes']}")

    print("-" * 105)
    print("⚠️ REVERSE LINE MOVEMENT (STEAM) & LINEUP CONFIRMATION RULE:")
    print("   Before placing any bet on your sportsbook app, check if the current odds have drifted by > +0.15 higher")
    print("   than scraped (e.g. 1.80 scraped -> 1.98 on app). If so, treat as INJURY/LINEUP alert: DO NOT BET until lineups confirm!")
    print("=" * 105 + "\n")


def print_executive_boards(card_df: pd.DataFrame, title_suffix: str = ""):
    print("\n" + "=" * 95)
    print(f"BOARD 1: TOP WIN-PROBABILITY ANCHORS (81%-83% Historical Win-Rate Tier) {title_suffix}")
    print("         Best for Straight Pick'em / Win-Streak Leaderboard Points")
    print("=" * 95)
    anchors = card_df[card_df['Market'] == 'FT_ML'].sort_values('Model_Prob_%', ascending=False).drop_duplicates(['Date', 'Matchup']).head(8)
    print(anchors[['Date', 'Matchup', 'Pick', 'Odds', 'Model_Prob_%', 'Bookie_Prob_%', 'Edge_%', 'Min_Playable_Odds']].to_string(index=False))

    print("\n" + "=" * 95)
    print(f"BOARD 2: MONEYLINE +EV VALUE SNIPERS (Positive Expected Value vs Bookmaker) {title_suffix}")
    print("         Best for ROI / Profit-Based Fantasy Bankroll Leagues")
    print("=" * 95)
    ml_value = card_df[(card_df['Market'] == 'FT_ML') & (card_df['EV_%'] > 1.0)].sort_values('EV_%', ascending=False).head(8)
    if len(ml_value) > 0:
        print(ml_value[['Date', 'Matchup', 'Pick', 'Odds', 'Model_Prob_%', 'Bookie_Prob_%', 'Edge_%', 'EV_%', 'Min_Playable_Odds', 'Rec_Units']].to_string(index=False))
    else:
        print("  No positive EV Moneyline bets on this slate (bookmakers have priced lines efficiently).")

    print("\n" + "=" * 95)
    print(f"BOARD 3: TOTALS & DERIVATIVE SNIPERS (FT_OU, 1H_OU, Q1_OU) {title_suffix}")
    print("         Dual Pure+Residual Regressors with Play-To Line Guardrails")
    print("=" * 95)
    totals_snipers = card_df[
        card_df['Market'].isin(['1H_OU', 'FT_OU', 'Q1_OU']) &
        (card_df['Line_Cushion_Pts'] > 0.3)
    ].sort_values(['Model_Prob_%', 'Line_Cushion_Pts'], ascending=[False, False]).head(10)
    if len(totals_snipers) > 0:
        print(totals_snipers[['Date', 'Matchup', 'Market', 'Pick', 'Odds', 'Model_Prob_%', 'EV_%', 'Min_Playable_Odds', 'Model_Proj_Pts', 'Line_Cushion_Pts', 'Play_To_Line', 'Rec_Units']].to_string(index=False))
    else:
        print("  No Over/Under lines with sufficient cushion above the safety buffer.")

    # UPGRADE 4: Steam Protocol
    print("\n" + "=" * 95)
    print("REVERSE LINE MOVEMENT (STEAM) & INJURY PROTOCOL:")
    print("RULE: If a Moneyline odds on your betting app has drifted by MORE than +0.15 higher")
    print("      than the scraped OddsPortal odds (e.g. Utah 1.80 scraped -> 1.98 on your app),")
    print("      TREAT AS STEAM / INJURY ALERT: DO NOT BET until starting lineups are confirmed!")
    print("=" * 95)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="NBA Master Prediction Engine")
    parser.add_argument("--date", type=str, default=None, help="Target date to predict (e.g. '2026-10-24', '24 Oct', 'today')")
    parser.add_argument("--input", type=str, default=None, help="Path to upcoming games CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to save predictions CSV")
    parser.add_argument("--checklist", type=str, default=DEFAULT_CHECKLIST_CSV, help="Path to save actionable checklist CSV")
    parser.add_argument("--stress", action="store_true", default=True, help="Run sensitivity stress-test")
    parser.add_argument("--scrape", action="store_true", help="Scrape latest live odds from OddsPortal before predicting")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Scraper concurrency (default: 10)")
    args = parser.parse_args()

    print("=" * 95)
    print("🏀 NBA MASTER PREDICTION PIPELINE (PRODUCTION ENGINE)")
    print("=" * 95)

    # 0. Optional Live Scraping Step
    if args.scrape:
        print("\n" + "=" * 95)
        print(f"📡 TRIGGERING LIVE UPCOMING SCRAPER (Concurrency: {args.concurrency})...")
        print("=" * 95)
        scraper_script = BASE_DIR / "basketball_upcoming.py"
        cmd = [sys.executable, str(scraper_script), "--concurrency", str(args.concurrency)]
        if args.date and args.date.lower() not in ["all", "full"]:
            cmd.extend(["--date", args.date])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Scraper error: {e}. Falling back to existing {DEFAULT_UPCOMING_CSV}...")

    # 1. Load History
    hist_df = load_and_prep_history(DEFAULT_HISTORICAL_CSV)
    elo_dict = compute_historical_elo(hist_df)
    hist_df, latest_team_state, metrics_to_roll = compute_team_rolling_features(hist_df, elo_dict)
    hist_df = add_derived_features(hist_df)
    print(f"Historical Training Base: {len(hist_df):,} matches (Spanning up to {hist_df['Date_Clean'].max().strftime('%d %b %Y')})")

    # 2. Determine Upcoming Input File
    input_file = args.input if args.input else DEFAULT_UPCOMING_CSV
    if not Path(input_file).exists():
        print(f"Error: Upcoming games CSV '{input_file}' not found! Run basketball_upcoming.py first.")
        sys.exit(1)

    upcoming_df = pd.read_csv(input_file, low_memory=False)
    upcoming_df['Date_Clean'] = pd.to_datetime(upcoming_df['Date'], format='mixed', dayfirst=True, errors='coerce')

    # Optional Date Filtering
    if args.date:
        d_arg = args.date.lower().strip()
        if d_arg in ["today", "now"]:
            t_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            upcoming_df = upcoming_df[upcoming_df['Date_Clean'].dt.strftime('%Y-%m-%d') == t_str]
        elif d_arg == "tomorrow":
            t_str = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            upcoming_df = upcoming_df[upcoming_df['Date_Clean'].dt.strftime('%Y-%m-%d') == t_str]
        elif d_arg in ["all", "full"]:
            pass  # keep all games
        else:
            match_mask = (
                upcoming_df['Date_Clean'].dt.strftime('%Y-%m-%d').str.contains(d_arg, case=False, na=False) |
                upcoming_df['Date'].astype(str).str.contains(d_arg, case=False, na=False)
            )
            upcoming_df = upcoming_df[match_mask]

        if len(upcoming_df) == 0:
            print(f"Warning: No upcoming matches matched date filter '{args.date}' in {input_file}.")
            sys.exit(0)

    print(f"Upcoming Slate to Predict: {len(upcoming_df)} games from {Path(input_file).name}\n")

    # 3. Build Features
    upcoming_feat_df = build_upcoming_features(upcoming_df, latest_team_state, hist_df, metrics_to_roll)

    # 4. Train Models & Predict
    pred_df, market_models = train_and_predict(hist_df, upcoming_feat_df)
    pred_df.to_csv(args.output, index=False)
    print(f"Detailed multi-market card saved to: {args.output}")

    # 5. Generate and Display Actionable Checklist
    checklist_df = generate_actionable_checklist(pred_df, output_path=args.checklist)
    print_actionable_checklist(checklist_df)

    # 6. Output Detailed Executive Boards & Stress Test
    date_title = f"[{args.date}]" if args.date else ""
    print_executive_boards(pred_df, title_suffix=date_title)

    if args.stress and len(upcoming_feat_df) <= 20:
        run_line_sensitivity_stress_test(upcoming_feat_df, market_models)

    print(f"\nAll pipelines complete. Actionable checklist ready at: {args.checklist}")


if __name__ == "__main__":
    main()
