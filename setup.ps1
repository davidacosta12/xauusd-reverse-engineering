<#
.SYNOPSIS
    One-shot environment setup for xauusd-reverse-engineering.

.EXAMPLE
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "`n==> [1/5] Creating virtual environment (.venv)" -ForegroundColor Cyan
python -m venv .venv
Write-Host "    Done." -ForegroundColor Green

Write-Host "`n==> [2/5] Activating virtual environment" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1
Write-Host "    Done." -ForegroundColor Green

Write-Host "`n==> [3/5] Upgrading pip" -ForegroundColor Cyan
python -m pip install --upgrade pip
Write-Host "    Done." -ForegroundColor Green

Write-Host "`n==> [4/5] Installing dependencies from requirements.txt" -ForegroundColor Cyan
Write-Host "    This may take several minutes..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "    Done." -ForegroundColor Green

Write-Host "`n==> [5/5] Verifying critical imports" -ForegroundColor Cyan
python -c "import MetaTrader5, pandas, vectorbt, backtesting; print('All imports OK')"
Write-Host "    Done." -ForegroundColor Green

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy-Item .env.example .env"
Write-Host "  2. Edit .env  ->  set MT5_PASSWORD and MT5_PATH"
Write-Host "  3. jupyter notebook"
