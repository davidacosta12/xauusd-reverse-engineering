# CLAUDE.md — XAUUSD M15 Reverse Engineering Project

This file provides persistent context for all Claude Code sessions in this project.
Read it in full before taking any action.

---

## Project Goal

Reverse-engineer an unknown algorithmic trading strategy operating on **XAUUSD M15**
using 30 real closed trades from a live MT5 account (investor/read-only access).

**Success criteria:**
- Directional match ≥ 70% vs. original trades on out-of-sample data
- Equity curve correlation ≥ 0.85 vs. original equity curve
- If both criteria met → generate a functional MQL5 Expert Advisor

---

## Broker & Account

| Field       | Value                     |
|-------------|---------------------------|
| Broker      | MEX Atlantic              |
| Server      | MEXAtlantic-Real          |
| Account     | 921339                    |
| Access type | Investor (read-only)      |
| Instrument  | XAUUSD                    |
| Timeframe   | M15                       |

Credentials are stored in `.env` (never committed). See `.env.example` for keys.

---

## Known Trade Statistics (Ground Truth)

| Metric          | Value                          |
|-----------------|-------------------------------|
| Period          | 2026-03-19 → 2026-04-27       |
| Total trades    | 30                             |
| Win rate        | 73%                            |
| Total P&L       | +$421.40                       |
| Instrument      | XAUUSD                         |
| Timeframe       | M15                            |

---

## Project Phases

### Phase 0 — Setup (current)
- Professional folder structure, dependencies, CLAUDE.md, git init.

### Phase 1 — Data Extraction
- Connect to MT5 via `MetaTrader5` Python library using investor credentials.
- Extract full closed-trade history from account 921339.
- Extract OHLCV M15 bars for the full trade period + 200-bar buffer.
- Save raw data as Parquet in `data/raw/`.
- Key files: `src/extraction/mt5_client.py`, `trade_history.py`, `market_data.py`.

### Phase 2 — Exploratory Analysis
- Session/time-of-day distribution of entries.
- Hold-time distribution, SL/TP ratio patterns.
- Direction bias (long vs short).
- Clustering of entry conditions.
- Key notebook: `notebooks/02_eda.ipynb`.

### Phase 3 — Feature Engineering (45 features)
- **Technical (src/features/technical.py):** RSI, MACD, EMA crosses, ATR, BB width,
  Stochastic, ADX, CCI, Donchian, VWAP deviation.
- **SMC (src/features/smc.py):** BOS, CHoCH, FVG, Order Blocks, liquidity sweeps,
  premium/discount zones (using `smartmoneyconcepts` library).
- **Microstructure (src/features/microstructure.py):** spread proxy, volume delta,
  bar range percentile, wick ratios, body/range ratio.
- Key notebook: `notebooks/03_features.ipynb`.

### Phase 4 — Strategy Identification
- Manual hypothesis testing (rule-based filters).
- ML phase: XGBoost + SHAP for feature importance (max 5 features).
- All ML is restricted to classification of direction, not price prediction.
- Key notebook: `notebooks/04_strategy_id.ipynb`.

### Phase 5 — Backtesting
- Use `backtesting.py` for event-driven validation.
- Use `vectorbt` for parameter sweep / optimization.
- Metrics: Sharpe, Calmar, max DD, win rate, profit factor.
- Key files: `src/backtest/engine.py`, `matcher.py`, `metrics.py`.

### Phase 6 — EA Generation (conditional)
- Only if Phase 5 meets success criteria.
- Generate MQL5 EA skeleton from identified rules.
- Output in `ea/` directory.

---

## Technology Stack

```
MetaTrader5==5.0.5735
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
plotly>=5.20
pandas-ta>=0.3.14b
smartmoneyconcepts>=0.0.14
vectorbt>=0.26
backtesting>=0.3.3
scikit-learn>=1.4
xgboost>=2.0
shap>=0.45
quantstats>=0.0.62
jupyter>=1.0
ipykernel>=6.29
python-dotenv>=1.0
pyarrow>=15.0
```

---

## Coding Standards

- Python 3.11+, type hints on all functions, docstrings on all public functions/classes.
- No bare `except` clauses; always catch specific exceptions.
- All data persisted as Parquet (never CSV for raw data).
- Secrets only via `python-dotenv`; never hardcoded.
- Logging via `logging` stdlib (not `print`) in production modules.
- ML models: no data leakage — features computed only from bars available at entry time.

---

## Important Constraints

- The MT5 account is **investor-only**: no trades can be placed, only history read.
- Do not commit `.env`, `data/raw/`, `*.parquet`, or model artifacts.
- ML must be explainable: prefer interpretable models + SHAP over black-box ensembles.
- Backtest must use transaction costs (spread + commission estimate from real trades).

---

## Directory Map

```
xauusd-reverse-engineering/
├── .env                   # real credentials (gitignored)
├── .env.example           # template
├── CLAUDE.md              # this file
├── requirements.txt
├── pyproject.toml
├── setup.ps1              # one-shot environment setup
├── data/
│   ├── raw/               # Parquet from MT5 (gitignored)
│   ├── processed/         # feature-engineered datasets
│   └── external/          # macro calendars, session times, etc.
├── notebooks/             # numbered Jupyter notebooks per phase
├── src/
│   ├── extraction/        # MT5 data extraction modules
│   ├── features/          # indicator & SMC feature builders
│   ├── analysis/          # EDA helpers
│   ├── strategies/        # rule-based strategy objects
│   ├── backtest/          # backtesting engine wrappers & metrics
│   └── ea_generator/      # MQL5 code generation
├── ea/                    # generated MQL5 Expert Advisors
├── reports/               # HTML/PDF reports (quantstats, plotly)
└── tests/                 # unit tests
```
