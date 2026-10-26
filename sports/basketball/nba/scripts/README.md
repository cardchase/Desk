# 🏀 NBA Institutional Prediction Command Center & Technical Manual

> **High-Level Operations & Low-Level Quantitative Architecture**  
> *Single Source of Truth for NBA Match Analytics, Derivative Markets, and Mathematical Modeling.*

---

## 📑 Table of Contents
1. [Executive Overview & Philosophy](#-executive-overview--philosophy)
2. [Quick-Start: Step-by-Step Operations](#-quick-start-step-by-step-operations)
   - [Daily Betting Run (Interactive Command Center)](#1-daily-betting-run-interactive-command-center)
   - [Command-Line Shortcuts (Non-Interactive)](#2-command-line-shortcuts-non-interactive)
   - [Weekly Historical Synchronization](#3-weekly-historical-synchronization)
3. [Understanding the Deliverables](#-understanding-the-deliverables)
   - [Actionable Betting Checklist (Executive Card)](#1-actionable-betting-checklist-executive-card)
   - [The Four Bet Archetypes](#2-the-four-bet-archetypes)
   - [Critical Guardrail: Reverse Line Movement (Steam Alert)](#3-critical-guardrail-reverse-line-movement-steam-alert)
4. [Low-Level Mathematical Architecture](#-low-level-mathematical-architecture)
   - [Dynamic Elo Power Rating Engine](#1-dynamic-elo-power-rating-engine)
   - [Rolling Feature Engineering & Rest Modeling](#2-rolling-feature-engineering--rest-modeling)
   - [The Dual 50/50 Regressor for Totals](#3-the-dual-5050-regressor-for-totals)
   - [Classifier Training & Strict Derivative Tie-Filtering](#4-classifier-training--strict-derivative-tie-filtering)
   - [Ensemble Blending & Gaussian Continuous Mapping](#5-ensemble-blending--gaussian-continuous-mapping)
   - [Financial Sizing: Quarter-Kelly & Expected Value (+EV)](#6-financial-sizing-quarter-kelly--expected-value-ev)
   - [Line Sensitivity Stress-Testing & Play-To Cutoffs](#7-line-sensitivity-stress-testing--play-to-cutoffs)
5. [Directory Layout & File Map](#-directory-layout--file-map)

---

## 🎯 Executive Overview & Philosophy

The NBA betting market is one of the most efficient sports betting markets in the world. High-volume syndicates and bookmaker algorithms price closing point spreads and totals with extreme accuracy. Consequently, simple heuristic models (e.g. relying solely on raw scoring averages or uncalibrated regression) consistently lose to the sportsbook vig (margin).

This engine is engineered on three core principles:
1. **Line-Agnostic vs. Market-Relative Decomposition**: Instead of trying to reinvent the bookmaker's line, we split scoring predictions into a **Pure Pace Regressor** (which measures underlying possession tempo and scoring efficiency independent of the line) and a **Market Residual Regressor** (which directly isolates systematic inefficiencies, biases, and public overreactions in bookmaker lines).
2. **Derivative Cleanliness**: 1st Half (1H) and 1st Quarter (Q1) betting markets suffer from frequent push/tie outcomes. By strictly removing tie observations from historical training sets and using continuous Gaussian mapping, we eliminate classification skew.
3. **Institutional Execution Protocols**: Predicting a winner or total is meaningless without knowing **at what price** and **up to what line** the bet remains profitable. Every pick is accompanied by a mathematical **Play-To Cutoff**, a **Minimum Playable Odds** threshold, a **Safety Cushion**, and a **Reverse Line Movement (Steam) Guardrail**.

---

## 🚀 Quick-Start: Step-by-Step Operations

### 1. Daily Betting Run (Interactive Command Center)

Every morning or afternoon before placing bets, execute the master interactive runner:

```powershell
python sports/basketball/nba/scripts/run_nba_predictions.py
```

#### What happens step-by-step:
1. **Interactive Prompt 1: Odds & Fixture Data**
   - *Prompt*: `Do you want to scrape fresh upcoming games & odds from OddsPortal?`
   - **Option [1] (Yes)**: Launches the Playwright headless scraper to extract live fixtures, spreads, and multi-quarter odds across Bet365, Pinnacle, and consensus books (~60–90 seconds).
   - **Option [2] (No)**: Uses the cached `sports/basketball/nba/data/upcoming_games.csv` (Instant, recommended if you have already scraped today).
2. **Interactive Prompt 2: Slate Selection**
   - *Prompt*: `Which games do you want to run predictions for?`
   - **Option [1]**: Today's slate.
   - **Option [2]**: Tomorrow's slate.
   - **Option [3]**: All upcoming games currently in the database.
   - **Option [4]**: Custom date or team filter (e.g., `'24 Oct'` or `'Lakers'`).
3. **Model Execution**:
   - Trains the walk-forward LightGBM classifiers and dual regressors on 13,300+ historical games.
   - Runs sensitivity stress tests.
   - Synthesizes the prioritized **Top-8 Actionable Checklist** and prints the executive betting summary directly into your terminal.

---

### 2. Command-Line Shortcuts (Non-Interactive)

To bypass interactive prompts and run direct one-line commands (ideal for batch jobs or fast queries):

```powershell
# Predict Today's slate immediately (using existing scraped odds)
python sports/basketball/nba/scripts/run_nba_predictions.py --today --no-scrape

# Predict Tomorrow's slate immediately
python sports/basketball/nba/scripts/run_nba_predictions.py --tomorrow --no-scrape

# Scrape fresh live odds first, then predict today's slate
python sports/basketball/nba/scripts/run_nba_predictions.py --today --scrape

# Predict all games in upcoming database
python sports/basketball/nba/scripts/run_nba_predictions.py --all --no-scrape

# Target a specific calendar date
python sports/basketball/nba/scripts/run_nba_predictions.py --date "2026-10-25" --no-scrape
```

---

### 3. Weekly Historical Synchronization

As games conclude throughout the week, your historical training dataset must stay synchronized. Run this incremental updater:

```powershell
python sports/basketball/nba/scripts/update_nba_history.py
```

#### How it works:
1. Scans `sports/basketball/nba/data/nba_historical.csv` and identifies the date and matchup of the latest recorded game.
2. Navigates to OddsPortal's completed NBA results feed.
3. Automatically filters out games already in the database and **only extracts newly finished matches** (box scores, 1Q–4Q quarter splits, OT periods, and closing odds).
4. Safely appends new games to the historical CSV, recalibrates team Elo ratings, and updates rolling form windows.

---

## 📊 Understanding the Deliverables

Running the predictor produces two files in `sports/basketball/nba/data/`:

| File | Purpose | Audience |
|---|---|---|
| **`actionable_betting_checklist.csv`** | **Primary Executive Betting Card.** The curated Top-8 picks with exact unit sizing, cushions, and cutoff thresholds. | **The Bettor / Sportsbook App User** |
| **`weekly_betting_card.csv`** | **Full Multi-Market Database.** Contains raw quantitative metrics for every game across FT, 1H, and Q1–Q4. | **Data Analysts / Quants** |

---

### 1. Actionable Betting Checklist (Executive Card)

Open `sports/basketball/nba/data/actionable_betting_checklist.csv` in Excel, Google Sheets, or your IDE:

| Column Header | Definition & Plain-English Interpretation |
|---|---|
| **`Priority_Rank`** | Overall algorithmic rank (1 through 8) based on conviction score and category balance. |
| **`Category_Tag`** | The strategic archetype of the wager (see Archetypes section below). |
| **`Matchup`** | Away Team @ Home Team. |
| **`Market`** | Target market: `FT_ML` (Full Time Moneyline), `FT_OU` (Full Game Total), `1H_OU` (1st Half Total), `Q1_OU` (1st Quarter Total). |
| **`Target_Selection`** | The specific pick to place on your sportsbook (e.g. `Boston Celtics ML` or `Under 228.5`). |
| **`Current_Odds`** | Decimal odds scraped from the market (e.g., `1.91`). |
| **`Model_Win_Prob_%`** | Calibrated true win probability estimated by the ensemble model. |
| **`Bookie_Implied_%`** | Vig-free implied probability embedded in the bookmaker's current price. |
| **`Model_Edge_%`** | Percentage edge over the bookmaker: $\text{Edge} = P_{\text{model}} - P_{\text{bookie}}$. |
| **`Expected_Value_%`** | Expected return on investment per unit wagered ($+EV\%$). |
| **`Model_Projected`** | Projected score or point total for totals markets (e.g. `221.8 pts`). |
| **`Safety_Cushion`** | Buffer margin between model projection and bookmaker line (e.g. `+6.7 pts under` or `+8.2% win margin`). |
| **`Play_To_Cutoff`** | **CRITICAL EXECUTION GUARDRAIL**: The worst line you can accept. If the line drops below this threshold, **DO NOT BET**. |
| **`Min_Playable_Odds`**| The lowest acceptable decimal price allowing for a $+2.0\%$ EV buffer. |
| **`Rec_Units`** | Recommended capital allocation (0.75u to 2.0u) derived from fractional Kelly criterion. |
| **`Betting_Advice_Notes`**| Clear, plain-English summary of team pace, injuries, rest advantages, and execution instructions. |

---

### 2. The Four Bet Archetypes

The checklist enforces strict portfolio diversification across four distinct market angles:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXECUTIVE PORTFOLIO COMPOSITION                       │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Archetype Tag            │ Primary Metric Filter    │ Strategic Role        │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ [WIN-RATE ANCHOR]        │ Model Win Prob >= 65.0%  │ High-certainty play;  │
│                          │ Elo & rest advantage     │ Parlay / Pick'em base │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ [HIGH-CONVICTION TOTAL]  │ Line Cushion >= 3.5 pts  │ Exploits market pace  │
│                          │ EV >= 4.0%               │ mispricings (O/U)     │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ [VALUE UNDERDOG / SPREAD]│ Scraped Odds >= 1.85     │ High $+EV$ upside;    │
│                          │ Edge >= 2.0%             │ Alt spread protection │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ [DERIVATIVE PACE SNIPER] │ 1H / Q1 Cushion >= 2.0 pt│ First Half / Q1 pace  │
│                          │ EV >= 3.0%               │ mismatch exploitation │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

---

### 3. Critical Guardrail: Reverse Line Movement (Steam Alert)

Before locking in any bet on your mobile sportsbook app (e.g., Bet365, FanDuel, DraftKings):
> **The Steam Rule**: If the decimal odds on your sportsbook app have drifted **higher by $+0.15$ or more** compared to the scraped odds (e.g., scraped at `1.82`, but now sitting at `2.02`), **PAUSE AND WAIT**.
> 
> A drift of that magnitude indicates sudden institutional sharp money or late-breaking news (such as a star player ruled out in shootaround). Do not place the wager until the official starting lineup is confirmed 30 minutes before tip-off.

---

## 🔬 Low-Level Mathematical Architecture

### 1. Dynamic Elo Power Rating Engine

Every team's baseline quality is dynamically updated game-by-game across all 13,300+ matches using an adjusted Elo model featuring a **Margin-of-Victory (MOV) multiplier** and **home-court advantage**.

#### Expected Win Probability:
$$E_H = \frac{1}{1 + 10^{\frac{R_A - (R_H + H_{\text{adv}})}{400}}}$$

Where:
- $R_H, R_A$: Pre-game Elo ratings of Home and Away teams.
- $H_{\text{adv}} = +70.0$ Elo points ($\approx 2.5\text{ to }3.0$ points of spread advantage in the modern NBA).

#### Margin-of-Victory Multiplier:
To prevent garbage-time blowouts from distorting ratings while still rewarding dominant wins, the Elo adjustment incorporates non-linear scaling:
$$\text{MOV} = \frac{(|\text{Margin}| + 3.0)^{0.8}}{7.5 + 0.006 \cdot \Delta R_{\text{winner}}}$$

$$\Delta R_H = K \cdot \text{MOV} \cdot (S_H - E_H), \quad K = 20.0$$

Where $S_H \in \{1.0, 0.0\}$ is the actual game outcome.

#### Inter-Season Mean Reversion:
NBA rosters change significantly during the offseason (draft, free agency, trades). Across season boundaries, team ratings undergo a $25\%$ regression toward the league baseline ($1500.0$):
$$R_{\text{Season } t+1} = 0.75 \cdot R_{\text{Season } t} + 0.25 \cdot 1500.0$$

---

### 2. Rolling Feature Engineering & Rest Modeling

For every match in history, we calculate isolated, leakage-free rolling form metrics:
- **Form Windows**: Short-term momentum ($L5$, last 5 games) and structural baseline ($L15$, last 15 games).
- **Rest & Schedule Fatigue**:
  $$\text{Rest Days} = \text{clip}\left(\text{Date}_{\text{game}} - \text{Date}_{\text{prev}}, 1, 7\right)$$
  $$\text{Is\_B2B} = \mathbb{I}(\text{Rest Days} == 1)$$
- **Projected Total Match Pace**:
  $$\text{Proj\_Total}_{L15} = \frac{\text{PtsFor}_{H, L15} + \text{PtsAgainst}_{A, L15} + \text{PtsFor}_{A, L15} + \text{PtsAgainst}_{H, L15}}{2}$$
- **Pace Shift & Differential Momentum**:
  $$\text{Pace Shift} = \text{FT\_Total}_{L5} - \text{FT\_Total}_{L15}$$
  $$\text{Net Margin Diff} = \text{FT\_Margin}_{H, L15} - \text{FT\_Margin}_{A, L15}$$

---

### 3. The Dual 50/50 Regressor for Totals

Predicting NBA point totals is notoriously difficult when using a single model. If you only look at team pace, you ignore the bookmaker's sharp information. If you only look at market odds, you reproduce the bookmaker's biases.

We solve this with a **Dual 50/50 Blended Architecture**:

```text
                          ┌────────────────────────┐
                          │   Historical Match     │
                          │   Features & Stats     │
                          └───────────┬────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
 ┌───────────────────────────┐                 ┌───────────────────────────┐
 │   Model A: Pure Pace      │                 │  Model B: Market Residual │
 │   LGBM Regressor          │                 │  LGBM Regressor           │
 ├───────────────────────────┤                 ├───────────────────────────┤
 │ • Line-agnostic features  │                 │ • Market-relative features│
 │ • Offensive/Defensive L5  │                 │ • Distance from OU Line   │
 │ • Possession pace shifts  │                 │ • Historical line beats   │
 └─────────────┬─────────────┘                 └─────────────┬─────────────┘
               │                                             │
               │ Predicted Pure Points:                      │ Predicted Residual (Δ):
               │ Y_pure                                      │ Y_resid = Line + Δ
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   Blended 50/50 Projection  │
                       │   Y_proj = 0.5*A + 0.5*B    │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │ Continuous Gaussian Mapping │
                       │    P(Over) = Φ(Cushion / σ) │
                       └─────────────────────────────┘
```

1. **Model A (Pure Pace Regressor)**:
   - Trained on pure box score metrics (Points For/Against across FT, 1H, and Q1).
   - Generates $\hat{Y}_{\text{pure}}$ independent of where the bookmaker set the line.
2. **Model B (Market Residual Regressor)**:
   - Trained on the residual target: $Y_{\text{resid}} = \text{Actual Score} - \text{Market Line}$.
   - Learns systematic bookmaker line mispricings, producing $\hat{Y}_{\text{resid}} = \text{Line} + \hat{\Delta}$.
3. **Blended Projection**:
   $$\hat{Y}_{\text{proj}} = 0.50 \cdot \hat{Y}_{\text{pure}} + 0.50 \cdot \hat{Y}_{\text{resid}}$$

---

### 4. Classifier Training & Strict Derivative Tie-Filtering

Full Time basketball games cannot end in a tie due to overtime. However, **1st Half (1H)** and **1st Quarter (Q1)** games frequently end in exact ties.

- **The Problem**: Standard binary classification treats ties either as losses or ignores them improperly, severely distorting the decision boundary for derivative Moneylines.
- **The Solution**: The engine explicitly applies `tie_check`:
  ```python
  if cfg.get('tie_check'):
      c_h, c_a = cfg['tie_check']
      valid_train = valid_train[valid_train[c_h] != valid_train[c_a]]
  ```
  By removing tied historical periods from the training distribution, the classifier outputs clean two-way win probabilities.

---

### 5. Ensemble Blending & Gaussian Continuous Mapping

For Totals markets, raw classifier outputs are merged with continuous Gaussian probabilities derived from the projected point cushion.

#### Normal Cumulative Distribution Function (CDF):
$$P_{\text{Gauss}}(\text{Over}) = \Phi\left(\frac{\hat{Y}_{\text{proj}} - \text{Line}}{\sigma}\right) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{\hat{Y}_{\text{proj}} - \text{Line}}{\sigma \sqrt{2}}\right)\right]$$

Historical standard deviations calibrated across 13,300+ games:
- $\sigma_{\text{FT}} = 17.5\text{ points}$
- $\sigma_{\text{1H}} = 11.2\text{ points}$
- $\sigma_{\text{Q1}} = 7.8\text{ points}$

#### Master Tri-Blend Ensemble:
$$P_{\text{final}}(\text{Over}) = 0.55 \cdot P_{\text{Gauss}} + 0.25 \cdot P_{\text{LGBM\_Classifier}} + 0.20 \cdot P_{\text{Market\_Implied}}$$

For Moneylines (where no point line exists):
$$P_{\text{final}}(\text{Home Win}) = 0.65 \cdot P_{\text{LGBM\_Calibrated}} + 0.35 \cdot P_{\text{Market\_Implied}}$$

---

### 6. Financial Sizing: Quarter-Kelly & Expected Value (+EV)

#### Expected Value ($EV$):
$$\text{EV} = (P_{\text{model}} \cdot \text{Decimal Odds}) - 1.0$$
*A wager is strictly $+EV$ if and only if $\text{EV} > 0.0$.*

#### Fractional Kelly Criterion:
Full Kelly staking maximises the expected geometric growth rate of bankroll but introduces massive drawdown volatility. We implement a conservative **Quarter-Kelly ($0.25 \times K$)**:

$$f^* = \max\left(0.0, \frac{P_{\text{model}} \cdot \text{Odds} - 1.0}{\text{Odds} - 1.0}\right)$$

$$\text{Recommended Units} = \text{round}\left(\min\left(2.0, \max\left(0.75, f^* \cdot 0.25 \cdot 100\right)\right), 1\right)$$

- Units are bounded strictly between $0.75\text{u}$ and $2.0\text{u}$ to preserve capital through inevitable variance runs.

---

### 7. Line Sensitivity Stress-Testing & Play-To Cutoffs

Sportsbooks alter lines throughout the day as bets pour in. If our model recommends `Under 228.5`, at what point does line movement destroy our edge?

#### The Play-To Formula:
To maintain a minimum $53.0\%$ win floor (beating standard $-110$ / $1.91$ juice):
$$\text{Buffer} = 0.075 \cdot \sigma$$

$$\text{Play\_To\_Line}_{\text{Over}} = \text{round}\left(\hat{Y}_{\text{proj}} - \text{Buffer}, 1\right)$$
$$\text{Play\_To\_Line}_{\text{Under}} = \text{round}\left(\hat{Y}_{\text{proj}} + \text{Buffer}, 1\right)$$

#### Real-Time Line Shift Simulation:
The engine runs a real-time sensitivity loop evaluating shifts from $-2.0$ to $+2.0$ points:
```text
  * LINE SHIFT SIMULATION (FT Over/Under):
    Shift | Line  | Pick       | Model Prob | Odds  | EV%    | Cushion | Recommendation
    ----------------------------------------------------------------------------------------
    -2.0  | 226.5 | Under 226.5|      54.1% |  1.91 |  +3.3% |   +4.7  | [VALUE PLAY]
    -1.0  | 227.5 | Under 227.5|      56.3% |  1.91 |  +7.5% |   +5.7  | [STRONG BUY]
     0.0  | 228.5 | Under 228.5|      58.4% |  1.91 | +11.6% |   +6.7  | [STRONG BUY]
    +1.0  | 229.5 | Under 229.5|      60.5% |  1.91 | +15.6% |   +7.7  | [STRONG BUY]
    +2.0  | 230.5 | Under 230.5|      62.6% |  1.91 | +19.5% |   +8.7  | [STRONG BUY]
```

---

## 🗂️ Directory Layout & File Map

```text
sports/basketball/nba/
├── data/
│   ├── nba_historical.csv                 <- 13,300+ matches (training dataset)
│   ├── upcoming_games.csv                 <- Fresh live odds & fixture slate
│   ├── actionable_betting_checklist.csv   <- EXECUTIVE DELIVERABLE (Top-8 picks)
│   └── weekly_betting_card.csv            <- Complete multi-market quantitative card
│
├── scripts/
│   ├── run_nba_predictions.py             <- Interactive Command Center (daily entrypoint)
│   ├── nba_master_predictor.py            <- Machine learning engine & math core
│   ├── basketball_upcoming.py             <- Playwright live odds extractor
│   ├── update_nba_history.py              <- Incremental historical updater
│   ├── basketball_historical.py           <- OddsPortal full-season historical scraper
│   └── README.md                          <- This technical manual
│
└── archive/                               <- Development logs & experimental scripts
```

---

## 🛡️ Best Practices & Execution Checklist

1. **Morning Run**: Run `python sports/basketball/nba/scripts/run_nba_predictions.py` at 10:00 AM – 12:00 PM local time to capture opening line value before sharp syndicates move the lines.
2. **Review Checklist**: Open `sports/basketball/nba/data/actionable_betting_checklist.csv`.
3. **Compare With App**: Verify the odds and lines on your betting app:
   - Check that odds $\ge$ `Min_Playable_Odds`.
   - Check that the line has not moved past `Play_To_Cutoff`.
   - Ensure odds have not drifted $\ge +0.15$ (Steam Alert).
4. **Place Recommended Units**: Wager the exact `Rec_Units` stated.
5. **Post-Slate Sync**: Once or twice weekly, run `python sports/basketball/nba/scripts/update_nba_history.py` to ingest results and keep models freshly calibrated.
