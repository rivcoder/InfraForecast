# InfraForecast - Run Script
# Usage:
#   ./run.ps1          <- Full pipeline (first-time setup, downloads PDFs)
#   ./run.ps1 -App     <- Just launch the Streamlit app (DB already built)

param([switch]$App)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  InfraForecast - Indian Infrastructure Overrun Analytics" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# ── Install dependencies ───────────────────────────────────────────────────────
if (-not $App) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed." -ForegroundColor Green

    # ── Run pipeline ───────────────────────────────────────────────────────────
    Write-Host ""
    Write-Host "Running pipeline (this will download ~30MB of PDFs on first run)..." -ForegroundColor Yellow
    python src/pipeline.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Pipeline failed. Check data/pipeline.log for details." -ForegroundColor Red
        exit 1
    }
    Write-Host "Pipeline complete." -ForegroundColor Green

    # ── Run tests ─────────────────────────────────────────────────────────────
    Write-Host ""
    Write-Host "Running tests..." -ForegroundColor Yellow
    python -m pytest src/test_pipeline.py -v
    Write-Host ""
}

# ── Launch app ─────────────────────────────────────────────────────────────────
Write-Host "Launching Streamlit dashboard..." -ForegroundColor Cyan
Write-Host "Open: http://localhost:8501" -ForegroundColor Green
Write-Host ""
streamlit run src/app.py
