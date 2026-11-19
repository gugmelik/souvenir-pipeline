#!/usr/bin/env bash
# Install the occo souvenir nodes into a ComfyUI on Linux (run after cloning this repo).
#
#   bash comfyui/install.sh ~/ComfyUI                 # nodes + Python packages
#   bash comfyui/install.sh ~/ComfyUI --models bf16   # ... and download the bf16 model files
#   bash comfyui/install.sh ~/ComfyUI --models fp8    # ... or the fp8 ones
#
# Use the Python that runs ComfyUI: activate ComfyUI's venv first, or set PYTHON=/path/to/python.
set -euo pipefail

COMFY="${1:?usage: install.sh <ComfyUI dir> [--models bf16|fp8|none] [--no-face] [--no-rembg]}"
shift
MODELS=none; FACE=1; REMBG=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --no-face) FACE=0; shift ;;
    --no-rembg) REMBG=0; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY="$(cd "$COMFY" && pwd)"
PY="${PYTHON:-python3}"
[[ -d "$COMFY/custom_nodes" ]] || { echo "$COMFY has no custom_nodes/ - is this the ComfyUI folder?"; exit 1; }

# 1) symlink the node pack (it imports ../../src from this repo, so keep the repo where it is)
LINK="$COMFY/custom_nodes/ComfyUI-occo-souvenir"
if [[ -L "$LINK" || -e "$LINK" ]]; then echo "already present: $LINK"; else ln -s "$HERE/ComfyUI-occo-souvenir" "$LINK"; echo "linked $LINK"; fi

# 2) Python packages into ComfyUI's environment
"$PY" -m pip install "pydantic>=2.5" pyyaml scipy requests openai
if [[ $FACE == 1 ]]; then "$PY" -m pip install insightface onnxruntime-gpu || "$PY" -m pip install insightface onnxruntime; fi
if [[ $REMBG == 1 ]]; then "$PY" -m pip install rembg; fi

# 3) model files (names must match the workflow)
dl() {  # repo, path in repo, target folder
  local dest="$COMFY/models/$3/$(basename "$2")"
  if [[ -f "$dest" ]]; then echo "ok        $dest"; return; fi
  "$PY" -m pip install -q -U huggingface_hub
  local tmp; tmp="$(mktemp -d)"
  hf download "$1" "$2" --local-dir "$tmp"
  mkdir -p "$COMFY/models/$3" && mv "$tmp/$2" "$dest" && rm -rf "$tmp"
  echo "downloaded $dest"
}
case "$MODELS" in
  bf16) dl Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors diffusion_models ;;
  fp8)  dl Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors diffusion_models ;;
  none) ;;
  *) echo "--models must be bf16, fp8 or none"; exit 1 ;;
esac
if [[ "$MODELS" != none ]]; then
  dl Comfy-Org/HunyuanVideo_1.5_repackaged split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors text_encoders
  dl Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors vae
  dl lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors loras
fi

echo
echo "Restart ComfyUI; the log should list custom_nodes/ComfyUI-occo-souvenir."
echo "Open: $HERE/workflows/occo_souvenir_slotzilla_bf16.json (bf16) or occo_souvenir_slotzilla.json (fp8)"
