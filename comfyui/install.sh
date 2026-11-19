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

REPO="$(dirname "$HERE")"

# 0) warn if a different ComfyUI is the one actually running
RUNNING="$(ps -eo args 2>/dev/null | grep -E "[m]ain.py" | head -1 || true)"
if [[ -n "$RUNNING" ]]; then
  echo "running ComfyUI: $RUNNING"
  RUN_DIR="$(readlink -f "/proc/$(pgrep -f "[m]ain.py" | head -1)/cwd" 2>/dev/null || true)"
  if [[ -n "$RUN_DIR" && "$RUN_DIR" != "$COMFY" ]]; then
    echo "WARNING: the running ComfyUI works in $RUN_DIR, but you are installing into $COMFY"
  fi
fi

# 1) the pipeline package, importable from ComfyUI's Python (editable: git pull updates it)
#    --no-deps so torch/numpy/pillow in the ComfyUI environment are never touched
"$PY" -m pip install --no-deps -e "$REPO"
"$PY" -m pip install "pydantic>=2.5" pyyaml scipy requests openai

# 2) copy the node pack as a real folder (some setups skip or mis-resolve symlinks)
DEST="$COMFY/custom_nodes/ComfyUI-occo-souvenir"
if [[ -L "$DEST" ]]; then rm "$DEST"; fi
rm -rf "$DEST"
cp -r "$HERE/ComfyUI-occo-souvenir" "$DEST"
echo "installed nodes -> $DEST"

if [[ $FACE == 1 ]]; then
  # insightface compiles a small C extension; RunPod images run as root, so install a compiler if missing
  if ! command -v g++ >/dev/null && command -v apt-get >/dev/null; then apt-get update -qq && apt-get install -y -qq build-essential python3-dev; fi
  "$PY" -m pip install insightface onnxruntime-gpu || "$PY" -m pip install insightface onnxruntime || echo "WARNING: insightface failed; use detector=manual"
fi
if [[ $REMBG == 1 ]]; then "$PY" -m pip install rembg || echo "WARNING: rembg failed; cutout falls back to white_key"; fi

# 2b) style references and approved plates -> ComfyUI/input, so the workflow's Load Image nodes find them
mkdir -p "$COMFY/input"
for f in "$REPO"/assets/style_refs/*.png "$REPO"/assets/plates/approved/*.png; do
  [[ -f "$f" ]] && cp "$f" "$COMFY/input/" && echo "input: $(basename "$f")"
done

# 3) model files (names must match the workflow)
dl() {  # repo, path in repo, target folder
  local dest="$COMFY/models/$3/$(basename "$2")"
  if [[ -f "$dest" ]]; then echo "ok        $dest"; return; fi
  "$PY" -m pip install -q -U huggingface_hub
  local tmp; tmp="$(mktemp -d -p "$COMFY/models")"   # same disk as the target, so mv is instant
  "$PY" -c "import sys; from huggingface_hub import hf_hub_download; hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3])" "$1" "$2" "$tmp"
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

# 4) self-test: load ComfyUI's nodes + ours exactly like ComfyUI does, check every node the workflow needs
echo; echo "=== self-test ==="
(cd "$COMFY" && "$PY" - <<'PYEOF'
import asyncio, importlib.util, inspect, logging, sys
logging.disable(logging.WARNING)  # ComfyUI's own server-only extras warn when loaded outside the server
sys.path.insert(0, ".")
sys.argv = [sys.argv[0], "--cpu"]
try:
    import comfy.options; comfy.options.enable_args_parsing()
except Exception:
    pass
import nodes
init = getattr(nodes, "init_extra_nodes", None)
if init:
    r = init(init_custom_nodes=False) if "init_custom_nodes" in inspect.signature(init).parameters else init()
    if inspect.iscoroutine(r):
        asyncio.run(r)
p = "custom_nodes/ComfyUI-occo-souvenir"
spec = importlib.util.spec_from_file_location("occo_selftest", p + "/__init__.py", submodule_search_locations=[p])
m = importlib.util.module_from_spec(spec); sys.modules["occo_selftest"] = m
try:
    spec.loader.exec_module(m)
except Exception as e:
    import traceback; traceback.print_exc()
    print("\nFAIL: occo nodes do not import (error above)"); sys.exit(1)
core = ["UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow", "CFGNorm", "LoraLoaderModelOnly",
        "TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod", "EmptyImage", "PreviewAny",
        "KSampler", "VAEEncode", "VAEDecode", "LoadImage", "SaveImage", "PreviewImage"]
missing = [c for c in core if c not in nodes.NODE_CLASS_MAPPINGS]
print("occo nodes:", ", ".join(m.NODE_CLASS_MAPPINGS))
if missing:
    print("FAIL: this ComfyUI is too old, missing core nodes:", missing, "-> update ComfyUI (git pull) and retry")
    sys.exit(1)
print("OK: all nodes the workflow needs are available")
PYEOF
) || { echo; echo "Self-test failed - send the output above."; exit 1; }

echo
echo "Now FULLY restart ComfyUI (kill the process, start it again); a browser refresh is not enough."
echo "The startup log should list custom_nodes/ComfyUI-occo-souvenir without IMPORT FAILED."
echo "Open: $HERE/workflows/occo_souvenir_slotzilla_bf16.json (bf16) or occo_souvenir_slotzilla.json (fp8)"
