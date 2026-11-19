# Install the occo souvenir nodes into a ComfyUI on Windows.
#   powershell -ExecutionPolicy Bypass -File comfyui\install.ps1 -ComfyUI "D:\ComfyUI_windows_portable\ComfyUI"
param(
  [Parameter(Mandatory = $true)][string]$ComfyUI,   # the folder that contains custom_nodes\ and models\
  [switch]$WithFace,                                 # also install insightface (detector + identity check)
  [switch]$WithRembg                                 # also install rembg (better cutouts)
)
$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "ComfyUI-occo-souvenir"
$dst = Join-Path $ComfyUI "custom_nodes\ComfyUI-occo-souvenir"

# 1) Junction, so the nodes keep using this project's src\ (one copy of the code)
if (Test-Path $dst) { Write-Host "already linked: $dst" }
else { New-Item -ItemType Junction -Path $dst -Target $src | Out-Null; Write-Host "linked $dst -> $src" }

# 2) Python packages in ComfyUI's own Python (portable build uses python_embeded)
$py = Join-Path (Split-Path $ComfyUI -Parent) "python_embeded\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m pip install "pydantic>=2.5" pyyaml scipy requests openai
if ($WithFace)  { & $py -m pip install insightface onnxruntime }
if ($WithRembg) { & $py -m pip install rembg onnxruntime }

# 3) Model files the workflow expects (same names as comfyui-api / Modal)
$need = @(
  "models\diffusion_models\qwen_image_edit_2511_fp8mixed.safetensors",
  "models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors",
  "models\vae\qwen_image_vae.safetensors",
  "models\loras\Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
foreach ($m in $need) {
  $p = Join-Path $ComfyUI $m
  if (Test-Path $p) { Write-Host "ok       $m" } else { Write-Host "MISSING  $m  (see comfyui\README.md)" -ForegroundColor Yellow }
}
Write-Host "`nRestart ComfyUI, then open comfyui\workflows\occo_souvenir_slotzilla.json"
