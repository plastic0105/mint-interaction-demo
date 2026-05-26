# ============================================================
#  MINT Demo — One-click setup (Windows PowerShell)
#  Prerequisites: conda (Anaconda / Miniconda) must be installed
# ============================================================

Set-StrictMode -Off
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot

Write-Host "`n=== [1/5] Cloning MINT repository ===" -ForegroundColor Cyan
if (-not (Test-Path "$ROOT\mint_repo")) {
    git clone https://github.com/VarunUllanat/mint.git "$ROOT\mint_repo"
} else {
    Write-Host "  mint_repo already exists, skipping clone."
}

# Patch missing 'import random' in extract.py (known bug in MINT repo)
$extractPath = "$ROOT\mint_repo\mint\helpers\extract.py"
$raw = [System.IO.File]::ReadAllText($extractPath, [System.Text.Encoding]::UTF8)
if ($raw -notmatch "(?m)^import random") {
    [System.IO.File]::WriteAllText($extractPath, "import random`n" + $raw, [System.Text.Encoding]::UTF8)
    Write-Host "  Patched extract.py (added import random)" -ForegroundColor Yellow
}

Write-Host "`n=== [2/5] Creating conda environment ===" -ForegroundColor Cyan
# Remove existing env if present (--force not supported in older conda)
$envList = conda env list 2>&1
if ($envList -match "mint-demo") {
    Write-Host "  Removing existing mint-demo environment..." -ForegroundColor DarkYellow
    conda env remove -n mint-demo -y
}
conda env create -f "$ROOT\env.yml"
if ($LASTEXITCODE -ne 0) { throw "conda env create failed" }

Write-Host "`n=== [3/5] Installing MINT package ===" -ForegroundColor Cyan
conda run -n mint-demo pip install -e "$ROOT\mint_repo" --no-deps -q
if ($LASTEXITCODE -ne 0) { throw "pip install mint failed" }

Write-Host "`n=== [4/5] Downloading model checkpoints ===" -ForegroundColor Cyan
Write-Host "  mint.ckpt is 3.25 GB — please be patient." -ForegroundColor DarkYellow
conda run -n mint-demo python "$ROOT\download_checkpoints.py" "$ROOT"
if ($LASTEXITCODE -ne 0) { throw "Checkpoint download failed" }

Write-Host "`n=== [5/5] Pre-computing embeddings for visualization ===" -ForegroundColor Cyan
Write-Host "  Runs MINT on 10 protein pairs (~3-5 min on CPU)" -ForegroundColor DarkYellow
conda run -n mint-demo python "$ROOT\precompute.py"
if ($LASTEXITCODE -ne 0) { throw "Precompute failed" }

Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
Write-Host "`nLaunch the demo with:" -ForegroundColor White
Write-Host "  conda activate mint-demo" -ForegroundColor Yellow
Write-Host "  streamlit run `"$ROOT\app.py`"" -ForegroundColor Yellow
