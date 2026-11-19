"""Backend for occo's gpu-instances project (C:\\dev\\occo\\gpu-instances).

Qwen-Image-Edit-2511 + lightx2v Lightning LoRA (4-step), up to MAX_NUM_IMAGES (default 4) inputs.
Two ways to reach it:

mode: pod      one pod started with `scripts\\start.bat`
               POST {url}/v1/images/edit  (Bearer SERVER_API_KEY)
               url + key from gpu-instances/.state.json, or GPU_POD_URL / GPU_POD_KEY
               body {"prompt", "images": [data URLs], "seed", "num_inference_steps", "true_cfg_scale",
                     "width", "height", "num_images": 1, "output_format": "png"} -> {"images": [base64], ...}

mode: gateway  production job API (Lambda gateway + fleet), see gpu-instances/PRODUCTION.md
               POST /v1/uploads -> PUT bytes -> "upload:upl_..." refs
               POST /v1/jobs?wait=N  (Idempotency-Key) -> poll GET /v1/jobs/{id} -> output.images[].url
               url + key from PROD_URL / PROD_API_KEY (gpu-instances/.env or this project's .env)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from souvenir.imaging import from_b64, limit_size, to_data_uri, to_png_bytes

DEFAULT_DIR = "C:/dev/occo/gpu-instances"


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class GpuInstancesBackend:
    name = "gpu_instances"

    def __init__(
        self,
        mode: str = "pod",
        project_dir: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        model: str = "qwen-image-edit-2511",
        num_inference_steps: int = 4,
        true_cfg_scale: float = 1.0,
        max_images: int = 4,
        max_side: int = 2048,
        timeout_s: int = 600,
        sync_wait_s: int = 30,
        poll_s: float = 2.0,
    ):
        self.mode = mode
        self.model = model
        self.steps = num_inference_steps
        self.cfg = true_cfg_scale
        self.max_images = max_images
        self.max_side = max_side
        self.timeout_s = timeout_s
        self.sync_wait_s = sync_wait_s
        self.poll_s = poll_s
        root = Path(project_dir or DEFAULT_DIR)

        if mode == "pod":
            if not url:
                state_file = root / ".state.json"
                if not state_file.is_file():
                    raise RuntimeError(f"no running pod: {state_file} not found. Run scripts\\start.bat in gpu-instances, "
                                       "or set GPU_POD_URL and GPU_POD_KEY")
                st = json.loads(state_file.read_text(encoding="utf-8"))
                if not st.get("pod_id"):
                    raise RuntimeError("gpu-instances .state.json has no active pod; run scripts\\start.bat first")
                url, api_key = st["url"], api_key or st.get("server_api_key", "")
        elif mode == "gateway":
            env = _read_env_file(root / ".env")
            url = url or env.get("PROD_URL")
            api_key = api_key or env.get("PROD_API_KEY")
            if not url or not api_key:
                raise RuntimeError("gateway mode needs PROD_URL and PROD_API_KEY (gpu-instances/.env or this .env)")
        else:
            raise ValueError(f"gpu_instances mode must be 'pod' or 'gateway', got {mode!r}")

        self.url = str(url).rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------ public

    def edit(self, images: list[Image.Image], prompt: str, *, negative_prompt: str = " ", seed: int | None = None,
             width: int | None = None, height: int | None = None) -> Image.Image:
        imgs = [limit_size(im.convert("RGB"), self.max_side) for im in images[: self.max_images]]
        params: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": self.steps,
            "true_cfg_scale": self.cfg,
            "num_images": 1,
            "output_format": "png",
        }
        if seed is not None:
            params["seed"] = int(seed) % 2147483647
        if width and height:
            params["width"], params["height"] = int(width), int(height)
        out = self._pod(imgs, params) if self.mode == "pod" else self._gateway(imgs, params)
        if width and height and out.size != (width, height):
            out = out.resize((width, height), Image.LANCZOS)
        return out

    # ------------------------------------------------------------------ single pod

    def _pod(self, imgs: list[Image.Image], params: dict[str, Any]) -> Image.Image:
        body = {**params, "images": [to_data_uri(im) for im in imgs]}
        for attempt in range(6):
            r = self.session.post(f"{self.url}/v1/images/edit", json=body, timeout=self.timeout_s)
            if r.status_code in (429, 503):  # queue full / model still loading
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"gpu-instances pod returned {r.status_code}: {r.text[:500]}")
            data = r.json()
            if not data.get("images"):
                raise RuntimeError(f"pod returned no images: {str(data)[:300]}")
            return from_b64(data["images"][0])
        raise RuntimeError("gpu-instances pod kept answering 429/503 (busy or loading)")

    # ------------------------------------------------------------------ production gateway

    def _upload(self, img: Image.Image) -> str:
        r = self.session.post(f"{self.url}/v1/uploads", json={"content_type": "image/png", "filename": "input.png"}, timeout=60)
        r.raise_for_status()
        u = r.json()
        put = requests.put(u["url"], data=to_png_bytes(img), headers=u.get("headers") or {}, timeout=300)
        put.raise_for_status()
        return u["ref"]

    def _gateway(self, imgs: list[Image.Image], params: dict[str, Any]) -> Image.Image:
        refs = [self._upload(im) for im in imgs]
        body = {"model": self.model, "input": {**params, "images": refs}, "priority": "normal"}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]
        r = self.session.post(f"{self.url}/v1/jobs", params={"wait": self.sync_wait_s}, json=body,
                              headers={"Idempotency-Key": key}, timeout=self.sync_wait_s + 60)
        if r.status_code >= 400:
            raise RuntimeError(f"gateway rejected job ({r.status_code}): {r.text[:500]}")
        job = r.json()
        deadline = time.time() + self.timeout_s
        while job.get("status") not in ("done", "failed", "canceled"):
            if time.time() > deadline:
                raise TimeoutError(f"job {job.get('id')} not done after {self.timeout_s}s (status {job.get('status')})")
            time.sleep(self.poll_s)
            g = self.session.get(f"{self.url}/v1/jobs/{job['id']}", timeout=60)
            g.raise_for_status()
            job = g.json()
        if job["status"] != "done":
            raise RuntimeError(f"job {job.get('id')} {job['status']}: {job.get('error')}")
        files = (job.get("output") or {}).get("images") or []
        if not files:
            raise RuntimeError(f"job {job.get('id')} finished without images")
        d = requests.get(files[0]["url"], timeout=300)
        d.raise_for_status()
        return Image.open(io.BytesIO(d.content)).convert("RGB")


def build(section: dict[str, Any]) -> GpuInstancesBackend:
    return GpuInstancesBackend(
        mode=section.get("mode", "pod"),
        project_dir=section.get("project_dir") or os.environ.get("GPU_INSTANCES_DIR"),
        url=section.get("url"),
        api_key=section.get("api_key"),
        model=section.get("model", "qwen-image-edit-2511"),
        num_inference_steps=int(section.get("num_inference_steps", 4)),
        true_cfg_scale=float(section.get("true_cfg_scale", 1.0)),
        max_images=int(section.get("max_images", 4)),
        max_side=int(section.get("max_side", 2048)),
        timeout_s=int(section.get("timeout_s", 600)),
        sync_wait_s=int(section.get("sync_wait_s", 30)),
    )
