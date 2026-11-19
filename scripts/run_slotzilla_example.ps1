# Run the Slotzilla product on one real photo, three ways, and compare.
#   .\scripts\run_slotzilla_example.ps1 [-Image <path or URL>]
param(
  [string]$Image = "https://assets.occo.io/imageProcessing/2554048889175593/2357516365682295/oc62_12_22_46_4_55_cap01-cm0-sid65-cap01_fcam01cstrt_vapv-lkqq-wjnq-9575_res_utshxml_ut_o.jpg"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

# 1) How long is the original brief, and where does Qwen's 512-token default cut it?
& $py -m souvenir tokens --file briefs\slotzilla_flight_brief.md

# 2) Analyze + plan only (needs the VLM + detector, no image generation).
& $py -m souvenir plan --image $Image --seed 7

# 3) Full pipeline.
& $py -m souvenir run --image $Image --seed 7

# 4) Baseline: the original long brief in a single Qwen call.
& $py -m souvenir run --image $Image --mode single:main --brief briefs\slotzilla_flight_brief.md --seed 7
