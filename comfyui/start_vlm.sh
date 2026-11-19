#!/usr/bin/env bash
# Start a VLM (Qwen3-VL via vLLM) on the same GPU pod as ComfyUI, for the occo nodes
# (person cards, LLM planner, verify). Run it before ComfyUI so both fit in VRAM.
#
#   bash comfyui/start_vlm.sh              # install once (separate venv), start, wait until ready
#   VLM_GPU_FRACTION=0.2 bash comfyui/start_vlm.sh
#   bash comfyui/start_vlm.sh stop
#
# The workflows already point at http://127.0.0.1:8100/v1 (vlm_base_url on Detect people / Plan / Verify).
set -euo pipefail

MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${VLM_PORT:-8100}"
FRACTION="${VLM_GPU_FRACTION:-0.25}"        # share of GPU memory vLLM may take (0.25 of 96 GB = 24 GB)
VENV="${VLM_VENV:-/workspace/vllm-venv}"    # separate from ComfyUI's venv: vLLM brings its own torch
LOG="${VLM_LOG:-/workspace/vllm.log}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"   # on the volume, so the weights survive pod restarts

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "vllm serve" && echo "stopped" || echo "not running"
  exit 0
fi

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "VLM already running: http://127.0.0.1:$PORT/v1"; exit 0
fi

# The RunPod ComfyUI image pins torch for ComfyUI through pip settings (PIP_CONSTRAINT /
# extra index cu132). vLLM needs its own torch, so ignore those settings for this venv;
# otherwise pip backtracks to an ancient vLLM source release and fails building it.
clean_env() {
  env -u PIP_CONSTRAINT -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS \
      -u UV_CONSTRAINT -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INDEX \
      PIP_CONFIG_FILE=/dev/null UV_NO_CONFIG=1 "$@"
}

if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "installing vLLM into $VENV (first time only, a few minutes)"
  rm -rf "$VENV"                                   # drop a half-finished venv from a failed try
  if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    clean_env python3 -m pip install -q --user uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  clean_env "$UV" venv --python 3.12 --seed "$VENV"   # vLLM wheels exist for 3.10-3.13
  # prebuilt wheel only (never compile); torch-backend=auto picks the CUDA build that fits the driver
  clean_env "$UV" pip install --python "$VENV/bin/python" \
      --index-url https://pypi.org/simple --only-binary vllm --torch-backend=auto "${VLM_PIP_SPEC:-vllm}"
  "$VENV/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok:', torch.cuda.is_available())"
fi

echo "starting $MODEL on port $PORT (log: $LOG)"
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "$FRACTION" \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 2}' \
  > "$LOG" 2>&1 &

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready: http://127.0.0.1:$PORT/v1  (model: $MODEL)"
    exit 0
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "vLLM exited - last log lines:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
echo "still loading after 20 min - check $LOG"; exit 1
