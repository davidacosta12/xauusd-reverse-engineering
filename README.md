# XAUUSD M15 — Strategy Reverse Engineering

Quantitative project to reverse-engineer an unknown algorithmic trading strategy
from 30 real closed trades on **XAUUSD M15** (win rate 73%, +$421.40 P&L) using
data extracted directly from a live MT5 account.

---

## Objective

1. Extract real trade history + OHLCV data from MT5 via Python.
2. Reconstruct market context at each entry with 45 technical features.
3. Identify the entry/exit logic with manual hypotheses + explainable ML.
4. Validate with professional backtesting (directional match ≥70%, equity ρ ≥0.85).
5. Generate a functional MQL5 Expert Advisor if criteria are met.

---

## Stack

| Layer              | Library                              |
|--------------------|--------------------------------------|
| MT5 data           | MetaTrader5 5.0.5735                 |
| Data wrangling     | pandas, numpy, pyarrow               |
| Technical features | pandas-ta                            |
| SMC features       | smartmoneyconcepts                   |
| Backtesting        | backtesting.py, vectorbt             |
| ML                 | scikit-learn, xgboost, shap          |
| Reporting          | quantstats, plotly, seaborn          |
| Environment        | Python 3.11, python-dotenv           |

---

## Project Structure

```
xauusd-reverse-engineering/
├── data/
│   ├── raw/          # Parquet files from MT5 (gitignored)
│   ├── processed/    # Feature-engineered datasets
│   └── external/     # Macro calendars, session schedules
├── notebooks/        # Numbered Jupyter notebooks per phase
├── src/
│   ├── extraction/   # MT5 connection & data download
│   ├── features/     # Technical, SMC, microstructure features
│   ├── analysis/     # EDA utilities
│   ├── strategies/   # Rule-based strategy objects
│   ├── backtest/     # Backtesting engine wrappers & metrics
│   └── ea_generator/ # MQL5 code generation
├── ea/               # Generated Expert Advisors
├── reports/          # HTML/PDF output reports
└── tests/            # Unit tests
```

---

## Setup

### Prerequisites

- Windows 10/11
- Python 3.11.x installed and on PATH
- MetaTrader5 terminal installed and logged in (investor account)

### 1 — Clone / open the project

```powershell
cd "C:\Users\lenovo\OneDrive\Escritorio\xauusd-reverse-engineering"
```

### 2 — Run the automated setup script

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

This will:
- Create a `.venv` virtual environment
- Install all dependencies from `requirements.txt`
- Run a quick import verification

### 3 — Create your `.env` file

```powershell
Copy-Item .env.example .env
```

Open `.env` and fill in:
```
MT5_PASSWORD=<your investor password>
MT5_PATH=<full path to terminal64.exe>
```

### 4 — Verify MT5 connection

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from src.extraction.mt5_client import MT5Client; c = MT5Client(); print(c.connect())"
```

---

## Workflow (Phase by Phase)

| Phase | Notebook                     | Description                        |
|-------|------------------------------|------------------------------------|
| 1     | `01_data_extraction.ipynb`   | Pull trades + OHLCV from MT5       |
| 2     | `02_eda.ipynb`               | Exploratory analysis               |
| 3     | `03_features.ipynb`          | Build 45-feature dataset           |
| 4     | `04_strategy_id.ipynb`       | Identify entry/exit rules          |
| 5     | `05_backtest.ipynb`          | Validate strategy                  |
| 6     | `06_ea_generation.ipynb`     | Generate MQL5 EA (if criteria met) |

---

## Success Criteria

| Metric                     | Target |
|----------------------------|--------|
| Directional match          | ≥ 70%  |
| Equity curve correlation ρ | ≥ 0.85 |

---

## Security

- Credentials are stored in `.env` (gitignored).
- The MT5 account is **investor-only** — no orders can be placed.
- Raw trade data is excluded from git (`data/raw/` is gitignored).
