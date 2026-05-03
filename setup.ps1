<#
.SYNOPSIS
    One-shot environment setup for xauusd-reverse-engineering.

.DESCRIPTION
    Creates a Python virtual environment, installs all dependencies,
    and verifies the key imports work correctly.

.EXAMPLE
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$VENV_DIR = ".venv"
$PYTHON = "python"

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Write-OK([string]$msg) {
    Write-Host "    [OK] $msg" -ForegroundColor Green
}

function Write-Fail([string]$msg) {
    Write-Host "    [FAIL] $msg" -ForegroundColor Red
}

# ── 1. Python version check ────────────────────────────────────────────────────
Write-Step "Checking Python version"
$pyVersion = & $PYTHON --version 2>&1
Write-Host "    Found: $pyVersion"
if ($pyVersion -notmatch "3\.11") {
    Write-Warning "Python 3.11 is recommended. Found: $pyVersion"
}

# ── 2. Create virtual environment ─────────────────────────────────────────────
Write-Step "Creating virtual environment in '$VENV_DIR'"
if (Test-Path $VENV_DIR) {
    Write-Host "    Virtual environment already exists — skipping creation."
} else {
    & $PYTHON -m venv $VENV_DIR
    Write-OK "Virtual environment created"
}

# ── 3. Activate venv ──────────────────────────────────────────────────────────
Write-Step "Activating virtual environment"
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Fail "Activation script not found at: $activateScript"
    exit 1
}
& $activateScript
Write-OK "Virtual environment active"

# ── 4. Upgrade pip ────────────────────────────────────────────────────────────
Write-Step "Upgrading pip"
& python -m pip install --upgrade pip --quiet
Write-OK "pip upgraded"

# ── 5. Install dependencies ───────────────────────────────────────────────────
Write-Step "Installing dependencies from requirements.txt"
Write-Host "    This may take several minutes..." -ForegroundColor Yellow
& pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip install failed. Check the output above."
    exit 1
}
Write-OK "All dependencies installed"

# ── 6. Verify critical imports ────────────────────────────────────────────────
Write-Step "Verifying critical imports"

$checks = @(
    @{ module = "pandas";           label = "pandas" },
    @{ module = "numpy";            label = "numpy" },
    @{ module = "MetaTrader5";      label = "MetaTrader5" },
    @{ module = "pandas_ta";        label = "pandas-ta" },
    @{ module = "backtesting";      label = "backtesting" },
    @{ module = "vectorbt";         label = "vectorbt" },
    @{ module = "sklearn";          label = "scikit-learn" },
    @{ module = "xgboost";          label = "xgboost" },
    @{ module = "shap";             label = "shap" },
    @{ module = "quantstats";       label = "quantstats" },
    @{ module = "plotly";           label = "plotly" },
    @{ module = "dotenv";           label = "python-dotenv" },
    @{ module = "pyarrow";          label = "pyarrow" }
)

$failed = @()
foreach ($c in $checks) {
    $result = & python -c "import $($c.module)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK $c.label
    } else {
        Write-Fail "$($c.label) — $result"
        $failed += $c.label
    }
}

# ── 7. Register Jupyter kernel ─────────────────────────────────────────────────
Write-Step "Registering Jupyter kernel"
& python -m ipykernel install --user --name xauusd-re --display-name "XAUUSD RE (Python 3.11)"
Write-OK "Kernel registered as 'xauusd-re'"

# ── 8. Summary ────────────────────────────────────────────────────────────────
Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "  Setup complete. All imports verified." -ForegroundColor Green
} else {
    Write-Host "  Setup complete WITH WARNINGS." -ForegroundColor Yellow
    Write-Host "  Failed imports: $($failed -join ', ')" -ForegroundColor Red
}
Write-Host "=" * 60 -ForegroundColor Cyan

Write-Host @"

Next steps:
  1. Copy .env.example to .env and fill in your credentials:
         Copy-Item .env.example .env

  2. Open .env and set:
         MT5_PASSWORD=<your investor password>
         MT5_PATH=<full path to terminal64.exe>

  3. Start Jupyter:
         jupyter notebook

  4. Open notebooks/01_data_extraction.ipynb to begin Phase 1.
"@ -ForegroundColor White
