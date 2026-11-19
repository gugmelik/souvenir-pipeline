# One-time setup on Windows (PowerShell), run from C:\dev\occo\occo-souvenir-pipeline
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -U pip
# Core + VLM client + face + AWS + background removal + OCR + tests.
# Add ",local" only on a machine with a big GPU; add ",gemini" for the Nano Banana baseline.
& .\.venv\Scripts\python.exe -m pip install -e ".[vlm,face,aws,seg,ocr,dev]"

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env - fill in your endpoints." }

& .\.venv\Scripts\python.exe -m souvenir poses
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m souvenir demo
Write-Host "`nDone. Activate with: .\.venv\Scripts\Activate.ps1"
