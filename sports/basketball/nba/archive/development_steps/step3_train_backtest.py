import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# Try importing LightGBM/XGBoost if available, fallback cleanly to sklearn's native HistGradientBoosting (LightGBM clone)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
FEATURES_CSV_PATH = BASE_DIR / "nba_features.csv"
PREDICTIONS_OUT_PATH = BASE_DIR / "nba_backtest_predictions.csv"

if not FEATURES_CSV_PATH.exists():
    FEATURES_CSV_PATH = Path(r"d:\Project\LoomFlow - Sports\test_NBA\nba_features.csv")

print("=" * 75)
print("PHASE 3: WALK-FORWARD MULTI-MARKET BACKTESTER (2022/23 -> 2025/26)")
print("=" * 75)
print(f"Loading features from: {FEATURES_CSV_PATH}")

df = pd.read_csv(FEATURES_CSV_PATH, low_memory=False)
df['Date_Clean'] = pd.to_datetime(df['Date_Clean'])
df = df.sort_values(by=['Date_Clean', 'Game_ID']).reset_index(drop=True)

# -----------------------------------------------------------------
# 1. ENGINEER EXTRA PRE-GAME INTERACTION & MARKET FEATURES
# -----------------------------------------------------------------
# Vig-free market implied probabilities for all 4 core markets
def add_market_probs(df, prefix, over_col, under_col):
    p1 = 1.0 / pd.to_numeric(df[over_col], errors='coerce')
    p2 = 1.0 / pd.to_numeric(df[under_col], errors='coerce')
    df[f'Mkt_Prob_{prefix}_1'] = p1 / (p1 + p2)
    df[f'Mkt_Vig_{prefix}'] = (p1 + p2) - 1.0

add_market_probs(df, 'FT_ML', 'HA_FT_Home', 'HA_FT_Away')
add_market_probs(df, 'FT_OU', 'OU_FT_Over', 'OU_FT_Under')
add_market_probs(df, '1H_OU', 'OU_1H_Over', 'OU_1H_Under')
add_market_probs(df, 'Q1_OU', 'OU_Q1_Over', 'OU_Q1_Under')

# Recent Momentum vs Long-Term Form (L5 minus L15)
df['H_Momentum_Margin'] = df['H_FT_Margin_L5'] - df['H_FT_Margin_L15']
df['A_Momentum_Margin'] = df['A_FT_Margin_L5'] - df['A_FT_Margin_L15']
df['H_Pace_Shift'] = df['H_FT_Total_L5'] - df['H_FT_Total_L15']
df['A_Pace_Shift'] = df['A_FT_Total_L5'] - df['A_FT_Total_L15']

# Short-term vs Long-term Line Edges
df['Edge_FT_Total_vs_Line_L5'] = df['Proj_FT_Total_L5'] - df['OU_FT_Line']
df['Proj_1H_Total_L5'] = (df['H_1H_Pts_For_L5'] + df['A_1H_Pts_Against_L5'] + df['A_1H_Pts_For_L5'] + df['H_1H_Pts_Against_L5']) / 2.0
df['Edge_1H_Total_vs_Line_L5'] = df['Proj_1H_Total_L5'] - df['OU_1H_Line']
df['Proj_Q1_Total_L5'] = (df['H_Q1_Pts_For_L5'] + df['A_Q1_Pts_Against_L5'] + df['A_Q1_Pts_For_L5'] + df['H_Q1_Pts_Against_L5']) / 2.0
df['Edge_Q1_Total_vs_Line_L5'] = df['Proj_Q1_Total_L5'] - df['OU_Q1_Line']

# Combined Beat-the-Line history
df['Combined_Beat_OU_FT_L15'] = df['H_Beat_OU_FT_L15'] + df['A_Beat_OU_FT_L15']
df['Combined_Beat_OU_Q1_L15'] = df['H_Beat_OU_Q1_L15'] + df['A_Beat_OU_Q1_L15']

# -----------------------------------------------------------------
# 2. DEFINE STRICT PRE-GAME FEATURE SETS FOR EACH MARKET
# -----------------------------------------------------------------
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

pace_total_features = [
    'H_FT_Pts_For_L5', 'H_FT_Pts_Against_L5', 'H_FT_Pts_For_L15', 'H_FT_Pts_Against_L15',
    'A_FT_Pts_For_L5', 'A_FT_Pts_Against_L5', 'A_FT_Pts_For_L15', 'A_FT_Pts_Against_L15',
    'H_1H_Pts_For_L5', 'H_1H_Pts_Against_L5', 'H_1H_Pts_For_L15', 'H_1H_Pts_Against_L15',
    'A_1H_Pts_For_L5', 'A_1H_Pts_Against_L5', 'A_1H_Pts_For_L15', 'A_1H_Pts_Against_L15',
    'H_Q1_Pts_For_L5', 'H_Q1_Pts_Against_L5', 'H_Q1_Pts_For_L15', 'H_Q1_Pts_Against_L15',
    'A_Q1_Pts_For_L5', 'A_Q1_Pts_Against_L5', 'A_Q1_Pts_For_L15', 'A_Q1_Pts_Against_L15',
    'Proj_FT_Total_L5', 'Proj_FT_Total_L15', 'Proj_1H_Total_L5', 'Proj_1H_Total_L15',
    'Proj_Q1_Total_L5', 'Proj_Q1_Total_L15', 'H_Pace_Shift', 'A_Pace_Shift',
    'H_Beat_OU_FT_L5', 'H_Beat_OU_FT_L15', 'A_Beat_OU_FT_L5', 'A_Beat_OU_FT_L15',
    'H_Beat_OU_Q1_L5', 'H_Beat_OU_Q1_L15', 'A_Beat_OU_Q1_L5', 'A_Beat_OU_Q1_L15',
    'Combined_Beat_OU_FT_L15', 'Combined_Beat_OU_Q1_L15'
]

market_configs = {
    'FT_ML': {
        'name': 'Full-Time Moneyline (Home vs Away)',
        'target': 'Target_Home_Win',
        'odds_1': 'HA_FT_Home', 'odds_2': 'HA_FT_Away',
        'side_1_label': 'Home ML', 'side_2_label': 'Away ML',
        'push_cond': lambda d: pd.Series(False, index=d.index),
        'features': base_team_features + ['HA_FT_Home', 'HA_FT_Away', 'Market_Implied_Home_Prob', 'Elo_vs_Market_Prob_Diff', 'OU_FT_Line']
    },
    'FT_OU': {
        'name': 'Full-Time Over/Under',
        'target': 'Target_FT_Over',
        'odds_1': 'OU_FT_Over', 'odds_2': 'OU_FT_Under',
        'side_1_label': 'FT Over', 'side_2_label': 'FT Under',
        'push_cond': lambda d: d['Actual_FT_Total'] == d['OU_FT_Line'],
        'features': base_team_features + pace_total_features + ['OU_FT_Line', 'Mkt_Prob_FT_OU_1', 'Edge_FT_Total_vs_Line_L5', 'Edge_FT_Total_vs_Line_L15', 'Market_Implied_Home_Prob']
    },
    '1H_OU': {
        'name': '1st Half Over/Under',
        'target': 'Target_1H_Over',
        'odds_1': 'OU_1H_Over', 'odds_2': 'OU_1H_Under',
        'side_1_label': '1H Over', 'side_2_label': '1H Under',
        'push_cond': lambda d: d['Actual_1H_Total'] == d['OU_1H_Line'],
        'features': base_team_features + pace_total_features + ['OU_1H_Line', 'OU_FT_Line', 'Mkt_Prob_1H_OU_1', 'Edge_1H_Total_vs_Line_L5', 'Edge_1H_Total_vs_Line_L15', 'Market_Implied_Home_Prob']
    },
    'Q1_OU': {
        'name': '1st Quarter Over/Under',
        'target': 'Target_Q1_Over',
        'odds_1': 'OU_Q1_Over', 'odds_2': 'OU_Q1_Under',
        'side_1_label': 'Q1 Over', 'side_2_label': 'Q1 Under',
        'push_cond': lambda d: d['Actual_Q1_Total'] == d['OU_Q1_Line'],
        'features': base_team_features + pace_total_features + ['OU_Q1_Line', 'OU_FT_Line', 'Mkt_Prob_Q1_OU_1', 'Edge_Q1_Total_vs_Line_L5', 'Edge_Q1_Total_vs_Line_L15', 'Market_Implied_Home_Prob']
    }
}

# -----------------------------------------------------------------
# 3. WALK-FORWARD TRAINING & BACKTESTING LOOP
# -----------------------------------------------------------------
test_seasons = ['2022/2023', '2023/2024', '2024/2025', '2025/2026']

def build_model():
    if HAS_LGBM:
        base = LGBMClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=7,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
            reg_alpha=2.0, reg_lambda=5.0, random_state=42, verbose=-1
        )
    else:
        base = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.03, max_depth=3, max_leaf_nodes=7,
            min_samples_leaf=50, l2_regularization=5.0, random_state=42
        )
    # Calibrate probabilities using sigmoid so model probabilities are honest
    return CalibratedClassifierCV(base, method='sigmoid', cv=3)

all_backtest_logs = []

for m_key, cfg in market_configs.items():
    print(f"\n>>> Training & Backtesting Market: {cfg['name']}")
    feats = [f for f in cfg['features'] if f in df.columns]
    
    # Filter valid rows for this market (exclude pushes from training/eval)
    valid_mask = (
        df[cfg['target']].notnull() &
        df[cfg['odds_1']].notnull() & (df[cfg['odds_1']] > 1.01) &
        df[cfg['odds_2']].notnull() & (df[cfg['odds_2']] > 1.01) &
        (~cfg['push_cond'](df)) &
        df['H_FT_Margin_L15'].notnull() & df['A_FT_Margin_L15'].notnull()
    )
    m_df = df[valid_mask].copy().reset_index(drop=True)
    
    season_preds = []
    for s_idx, test_season in enumerate(test_seasons):
        # All games logically BEFORE this test season
        test_season_rows = m_df[m_df['Season'] == test_season]
        if len(test_season_rows) == 0:
            continue
        test_start_date = test_season_rows['Date_Clean'].min()
        train_slice = m_df[m_df['Date_Clean'] < test_start_date]
        test_slice = test_season_rows.copy()
        
        if len(train_slice) < 1000 or len(test_slice) == 0:
            continue
            
        med = train_slice[feats].median()
        X_train = train_slice[feats].fillna(med).fillna(0)
        y_train = train_slice[cfg['target']].astype(int)
        X_test = test_slice[feats].fillna(med).fillna(0)
        y_test = test_slice[cfg['target']].astype(int)
        
        model = build_model()
        model.fit(X_train, y_train)
        
        probs_1 = model.predict_proba(X_test)[:, 1]
        
        # Blend ML Model + Market Implied (Shrinkage prevents extreme overconfidence)
        p1_raw = 1.0 / test_slice[cfg['odds_1']]
        p2_raw = 1.0 / test_slice[cfg['odds_2']]
        mkt_p1 = p1_raw / (p1_raw + p2_raw)
        
        if m_key == 'FT_ML':
            blended_p1 = 0.65 * probs_1 + 0.35 * mkt_p1
        else:
            blended_p1 = 0.75 * probs_1 + 0.25 * mkt_p1
            
        test_slice['Model_Prob_1'] = blended_p1
        test_slice['Model_Prob_2'] = 1.0 - blended_p1
        test_slice['Mkt_Fair_Prob_1'] = mkt_p1
        test_slice['Mkt_Fair_Prob_2'] = 1.0 - mkt_p1
        test_slice['Market_Key'] = m_key
        
        # Record both potential sides as candidate bets
        for idx_row, r in test_slice.iterrows():
            actual_1 = int(r[cfg['target']])
            # Side 1 Candidate
            season_preds.append({
                'Game_ID': r['Game_ID'], 'Date': r['Date_Clean'], 'Season': r['Season'],
                'Matchup': f"{r['AwayTeam']} @ {r['HomeTeam']}",
                'Market': m_key, 'Bet_Side': cfg['side_1_label'],
                'Odds': r[cfg['odds_1']], 'Model_Prob': r['Model_Prob_1'],
                'Mkt_Prob': r['Mkt_Fair_Prob_1'], 'Prob_Edge': r['Model_Prob_1'] - r['Mkt_Fair_Prob_1'],
                'EV': (r['Model_Prob_1'] * r[cfg['odds_1']]) - 1.0,
                'Won': 1 if actual_1 == 1 else 0,
                'Profit_1U': (r[cfg['odds_1']] - 1.0) if actual_1 == 1 else -1.0
            })
            # Side 2 Candidate
            season_preds.append({
                'Game_ID': r['Game_ID'], 'Date': r['Date_Clean'], 'Season': r['Season'],
                'Matchup': f"{r['AwayTeam']} @ {r['HomeTeam']}",
                'Market': m_key, 'Bet_Side': cfg['side_2_label'],
                'Odds': r[cfg['odds_2']], 'Model_Prob': r['Model_Prob_2'],
                'Mkt_Prob': r['Mkt_Fair_Prob_2'], 'Prob_Edge': r['Model_Prob_2'] - r['Mkt_Fair_Prob_2'],
                'EV': (r['Model_Prob_2'] * r[cfg['odds_2']]) - 1.0,
                'Won': 1 if actual_1 == 0 else 0,
                'Profit_1U': (r[cfg['odds_2']] - 1.0) if actual_1 == 0 else -1.0
            })

    m_preds_df = pd.DataFrame(season_preds)
    all_backtest_logs.append(m_preds_df)

    # Evaluate Higher-Probability Pick Per Game (Standard Accuracy)
    top_prob_per_game = m_preds_df.sort_values('Model_Prob', ascending=False).drop_duplicates('Game_ID')
    overall_acc = top_prob_per_game['Won'].mean() * 100
    print(f"    Overall Out-of-Sample Accuracy (All {len(top_prob_per_game):,} games): {overall_acc:.2f}%")

    # Tiered Confidence Breakdown
    print("    --- Strategy A: High-Confidence Win-Rate Tiers ---")
    for thresh in [0.55, 0.58, 0.62, 0.68, 0.75]:
        tier = top_prob_per_game[top_prob_per_game['Model_Prob'] >= thresh]
        if len(tier) >= 20:
            w_rate = tier['Won'].mean() * 100
            roi = (tier['Profit_1U'].sum() / len(tier)) * 100
            print(f"      Prob >= {thresh*100:4.0f}% | Bets: {len(tier):4d} | Win Rate: {w_rate:5.2f}% | Flat ROI: {roi:+6.2f}%")

    print("    --- Strategy B: Positive Expected Value (+EV Edge vs Bookie) ---")
    best_ev_per_game = m_preds_df.sort_values('EV', ascending=False).drop_duplicates('Game_ID')
    for ev_thresh in [0.02, 0.04, 0.06, 0.08]:
        tier = best_ev_per_game[(best_ev_per_game['EV'] >= ev_thresh) & (best_ev_per_game['Odds'] <= 3.0)]
        if len(tier) >= 20:
            w_rate = tier['Won'].mean() * 100
            roi = (tier['Profit_1U'].sum() / len(tier)) * 100
            avg_odds = tier['Odds'].mean()
            print(f"      EV >= +{ev_thresh*100:3.0f}%  | Bets: {len(tier):4d} | Win Rate: {w_rate:5.2f}% | Avg Odds: {avg_odds:.2f} | Flat ROI: {roi:+6.2f}%")

full_log_df = pd.concat(all_backtest_logs, ignore_index=True)
full_log_df.to_csv(PREDICTIONS_OUT_PATH, index=False)

# -----------------------------------------------------------------
# 4. SIMULATE THE "WEEKLY FANTASY SNIPER" (TOP N PICKS PER WEEK)
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("WEEKLY FANTASY LEAGUE SIMULATION (2022/23 - 2025/26)")
print("=" * 75)

full_log_df['Year_Week'] = full_log_df['Date'].dt.strftime('%Y-W%U')

# Simulation 1: Leaderboard Climber (Highest Win-Probability Picks Each Week)
print("\n[MODE 1: PURE WIN-RATE SNIPER] (For Pick'em / Win-Streak Leaderboard):")
for top_k in [3, 5, 10]:
    weekly_picks = (
        full_log_df.sort_values('Model_Prob', ascending=False)
        .groupby('Year_Week')
        .head(top_k)
    )
    w_rate = weekly_picks['Won'].mean() * 100
    print(f"  Top {top_k:2d} Highest-Prob Bets/Week -> Total Bets: {len(weekly_picks):4d} | Win Rate: {w_rate:.2f}% | Avg Odds: {weekly_picks['Odds'].mean():.2f}")

# Simulation 2: Totals & Derivative Value Sniper (1H_OU, Q1_OU, FT_OU with highest Prob >= 55%)
print("\n[MODE 2: EVEN-MONEY TOTALS SNIPER (FT, 1H, Q1 Over/Unders ~1.90 Odds)]:")
totals_only = full_log_df[full_log_df['Market'].isin(['FT_OU', '1H_OU', 'Q1_OU'])].copy()
for top_k in [3, 5, 10]:
    weekly_totals = (
        totals_only.sort_values('Model_Prob', ascending=False)
        .groupby('Year_Week')
        .head(top_k)
    )
    w_rate = weekly_totals['Won'].mean() * 100
    roi = (weekly_totals['Profit_1U'].sum() / len(weekly_totals)) * 100
    print(f"  Top {top_k:2d} Totals Bets/Week       -> Total Bets: {len(weekly_totals):4d} | Win Rate: {w_rate:.2f}% | Flat ROI: {roi:+.2f}%")

print("=" * 75)
